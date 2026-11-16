# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capload / nullsse modes on localhost (functional checks, short runs)."""

import time

from capcommon import expect
from capcommon.jsonio import write_json_stamped
from capcommon.identity import Identity
from capcommon.runtree import RunTree
from capcommon.schedule import BeatSchedule

from capload_support import MS, SHA, basic_spec, table_doc


def _all_tiers(lz):
    return range(len(lz.table.tiers))


def _total(lz, role, path):
    return sum(lz.sum_count(role, s, path) for s in _all_tiers(lz))


def test_new_per_request_accepts(launch):
    lz = launch()
    rc_l, rc_s = lz.run_pair(sink_kw={"workers": 2},
                             load_kw={"workers": 2, "conn_policy": "new-per-request"})
    assert (rc_l, rc_s) == (0, 0), lz.errtext("capload")
    planned = _total(lz, "capload", "counts.planned")
    assert planned == lz.sched_meta["n_requests"] > 0
    assert _total(lz, "capload", "counts.ok") == planned
    # every request opened its own connection, and the sink accepted exactly that many
    assert _total(lz, "capload", "conn_pool.new_conns") == planned
    assert _total(lz, "capload", "conn_pool.requests_on_new_conn") == planned
    assert _total(lz, "nullsse", "counts.accepts") == planned
    assert _total(lz, "capload", "failures_split.connect") == 0


def test_keepalive_pool_reuses(launch):
    lz = launch()
    rc_l, rc_s = lz.run_pair(load_kw={"workers": 1, "pool_reset": "never",
                                      "pool_pick": "random"})
    assert (rc_l, rc_s) == (0, 0), lz.errtext("capload")
    planned = _total(lz, "capload", "counts.planned")
    accepts = _total(lz, "nullsse", "counts.accepts")
    assert 0 < accepts < planned  # keep-alive: far fewer connections than requests
    assert _total(lz, "capload", "conn_pool.new_conns") == accepts


def test_beat_pacing_frames_not_before_planned_step(launch):
    doc = table_doc(basic_spec(n=6, up_kw={"k": 2, "s": 2}))
    lz = launch(doc)
    rc_l, rc_s = lz.run_pair(sink_kw={"workers": 1, "pacing": "beat"},
                             load_kw={"workers": 1, "sample_mod": 1})
    assert (rc_l, rc_s) == (0, 0), lz.errtext("capload")
    beat = BeatSchedule(lz.table)
    streams = {}
    for s in _all_tiers(lz):
        for _h, c in lz.capbins("nullsse", "streams", s):
            for i in range(len(c["req_id"])):
                streams[c["req_id"][i]] = (c["first_step"][i], c["tier_seq"][i])
    assert len(streams) == lz.sched_meta["n_requests"]
    checked = 0
    for s in _all_tiers(lz):
        for _h, c in lz.capbins("capload", "chunks", s):
            for i in range(len(c["req_id"])):
                fs, tseq = streams[c["req_id"][i]]
                t = lz.table.tiers[tseq]
                offs = expect.msg_step_offsets(t.max_tokens, t.tokens_per_msg,
                                               t.stream_interval, "product")
                idx = c["idx"][i]
                assert fs >= 0
                if idx < len(offs):
                    planned = beat.planned_rel_ns(fs + offs[idx])
                    assert c["recv_rel_ns"][i] >= planned
                    checked += 1
    assert checked > 0
    # the up tier has k = s = 2 -> its frame count follows expect (product gate)
    up = 2
    t = lz.table.tiers[up]
    m = expect.msgs_per_request(t.max_tokens, 2, 2, "product")
    want = int(lz.sched_meta["per_tier"][str(up)])
    assert lz.sum_count("capload", up, "frames.frames") == want * (m + 1)
    assert lz.sum_count("nullsse", up, "counts.frames_sent") == want * (m + 1)


def test_stop_early_reports_not_sent(launch):
    lz = launch()
    sink = lz.start_sink(workers=1)
    load = lz.start_load(workers=1)
    # stop during the first drain tier (rel 0.5 s .. 0.8 s): the up tier is never sent
    target = lz.origin + 650 * MS
    while time.monotonic_ns() < target:
        time.sleep(0.01)
    rt = RunTree(lz.run_dir)
    ident = Identity(experiment="E1", arm="unit-a", launch_seq=1, job_id="local-test",
                     restart_count=0, register_sha=SHA, role="launch")
    write_json_stamped(rt.stop_json, "capkit.stop/1", ident,
                       {"reason": "unit_test", "t_mono_ns": time.monotonic_ns(),
                        "after_tier_seq": 0})
    rc_l = lz.stop(load)
    rc_s = lz.stop(sink)
    assert rc_l == 5, lz.errtext("capload")
    assert rc_s == 0
    w = lz.summaries("capload", 0)[0]
    assert w["counts"]["sent"] == w["counts"]["planned"] > 0
    up = lz.summaries("capload", 2)[0]
    assert up["counts"]["planned"] > 0
    assert up["counts"]["sent"] != up["counts"]["planned"]  # -> the judge marks it NOT_RUN
    assert up["counts"]["not_sent"] == up["counts"]["planned"]
    assert up["stop"]["reason"] == "unit_test"
    assert up["stopped_early"] is True


def test_slow_reader_fionread(launch):
    spec = [("warmup", 300, 20.0, {"n": 20}), ("drain", 300, 0.0, {}),
            ("up", 300, 20.0, {"n": 20}), ("drain", 1500, 0.0, {})]
    results = {}
    for name, pause in (("control", 0), ("slow", 50000)):
        lz = launch(table_doc(spec))
        rc_l, rc_s = lz.run_pair(sink_kw={"workers": 1, "chunk_bytes": 20000},
                                 load_kw={"workers": 1,
                                          "extra": ["--slow-read-pause-us", str(pause)]})
        assert (rc_l, rc_s) == (0, 0), lz.errtext("capload")
        results[name] = (_total(lz, "capload", "fionread.samples"),
                         _total(lz, "capload", "fionread.nonzero"),
                         max(s["fionread"]["max_bytes"] for t in _all_tiers(lz)
                             for s in lz.summaries("capload", t)))
    assert results["control"][0] > 0 and results["slow"][0] > 0  # sampling happened
    assert results["slow"][1] > 0 and results["slow"][2] > 0
    assert results["slow"][1] > results["control"][1]
