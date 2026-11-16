# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capload -> nullsse on localhost: counts reconcile exactly, latency fields present."""

import os

from capcommon import expect
from capcommon.schedule import per_worker_tier_counts


def _load_tiers(lz):
    return [t.seq for t in lz.table.tiers if t.is_load]


def test_keepalive_counts_reconcile(launch):
    lz = launch()
    rc_l, rc_s = lz.run_pair(sink_kw={"workers": 2}, load_kw={"workers": 2})
    assert rc_l == 0, lz.errtext("capload")
    assert rc_s == 0, lz.errtext("nullsse")
    per_tier = {int(k): v for k, v in lz.sched_meta["per_tier"].items()}
    assert sum(per_tier.values()) > 0
    checked = 0
    for seq in _load_tiers(lz):
        t = lz.table.tiers[seq]
        want = per_tier.get(seq, 0)
        assert want > 0  # the table is built so every load tier has arrivals
        sums = lz.summaries("capload", seq)
        assert len(sums) == 2
        pw = per_worker_tier_counts(lz.sched_meta, 2)
        for w, s in enumerate(sums):
            assert s["counts"]["planned"] == pw[(w, seq)]
            assert s["counts"]["sent"] == s["counts"]["planned"]
            assert s["counts"]["not_sent"] == 0
            assert s["n_workers"] == 2
            assert s["tier_table_sha256"] == lz.table.sha256
        assert lz.sum_count("capload", seq, "counts.planned") == want
        assert lz.sum_count("capload", seq, "counts.ok") == want
        assert lz.sum_count("capload", seq, "counts.failed") == 0
        m = expect.msgs_per_request(t.max_tokens, t.tokens_per_msg, t.stream_interval, "product")
        frames = expect.sse_frames_per_request(m, True)
        assert lz.sum_count("capload", seq, "frames.frames") == want * frames
        assert lz.sum_count("nullsse", seq, "counts.requests") == want
        assert lz.sum_count("nullsse", seq, "counts.frames_sent") == want * frames
        assert lz.sum_count("capload", seq, "frames.completion_tokens") == want * t.max_tokens
        # per-request rows: latency fields present and ordered
        rows = 0
        for hdr, cols in lz.capbins("capload", "requests", seq):
            assert hdr["identity"]["tier_seq"] == seq
            assert hdr["meta"]["overflow"] == 0
            for i in range(hdr["n_rows"]):
                pl, se, fi, do = (cols["planned_rel_ns"][i], cols["sent_rel_ns"][i],
                                  cols["first_rel_ns"][i], cols["done_rel_ns"][i])
                assert 0 <= pl <= se <= fi <= do
                assert cols["status"][i] == 0
                assert cols["frames"][i] == frames
                assert cols["tier_seq"][i] == seq
                rows += 1
        assert rows == want
        # chunk_sample_mod 1: one row per frame. Chunk rows are archived by event time
        # (INTERFACES 5.4), so join by req_id over every tier directory.
        rids = set()
        for _h, cols in lz.capbins("capload", "requests", seq):
            rids.update(cols["req_id"])
        nchunk = 0
        for s2 in range(len(lz.table.tiers)):
            for _h, cols in lz.capbins("capload", "chunks", s2):
                nchunk += sum(1 for r in cols["req_id"] if r in rids)
        assert nchunk == want * frames
        nstream = sum(h["n_rows"] for h, _c in lz.capbins("nullsse", "streams", seq))
        assert nstream == want
        checked += 1
    assert checked == len(_load_tiers(lz)) and checked > 0
    # every process published a snapshot: parent + 2 workers per role
    snaps = [f for f in os.listdir(lz.shm) if f.endswith(".bin")]
    assert len(snaps) == 6
