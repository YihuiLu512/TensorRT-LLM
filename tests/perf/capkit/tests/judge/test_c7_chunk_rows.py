# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""C7 per-request chunk rows. capload files chunk rows by the tier of their receive time
(INTERFACES 5.4) while the sampled requests are those planned in the tier, so the rows of a request
planned just before a tier's end are split between that tier and the drain tier after it. They are
counted together; the count must still equal the frames per request, stray ids in the tier's own
files are still RED, and rows in any other tier are not counted."""

from types import SimpleNamespace

from conftest import judge_mod

from capcommon import expect

C = judge_mod("checks")

MAX_TOKENS = 16
CHAIN = {"name": "sse", "kind": "chunks", "gate": "product", "include_usage": True,
         "hops": [{"role": "capload", "source": "summary.frames.frames", "scope": "launch"}],
         "corrections": []}
PER = expect.expected_hop_count(CHAIN, 1, MAX_TOKENS, 1, 1, "capload")
# req_id -> planned tier: 0..3 in tier 2, 4..9 in tier 4 (tiers 3 and 5 are drains)
SCHEDULE_TIER = [2, 2, 2, 2, 4, 4, 4, 4, 4, 4]


def tier(seq):
    return SimpleNamespace(seq=seq, max_tokens=MAX_TOKENS, tokens_per_msg=1, stream_interval=1)


class TD:
    """One tier's merged capload rows: chunk rows {req_id: n}, requests {req_id: status}."""

    def __init__(self, seq, chunk_rows, requests=None, mod=1):
        self.tier = tier(seq)
        self.summaries = {}
        self._chunks = [rid for rid, n in sorted(chunk_rows.items()) for _ in range(n)]
        self._reqs = requests or {}
        self._mod = mod

    def roles_with(self, kind):
        return ["capload"]

    def rows(self, kind, role=None):
        meta = {"chunk_sample_mod": self._mod}
        if kind == "loadgen.chunks":
            return {"req_id": list(self._chunks)}, [("capload", 0, meta)]
        if kind == "loadgen.requests":
            return ({"req_id": list(self._reqs), "status": list(self._reqs.values())},
                    [("capload", 0, meta)])
        return {}, []


def ctx():
    return SimpleNamespace(launch={"reconcile": {"chains": [CHAIN]}}, roles={},
                           schedule_tier=list(SCHEDULE_TIER),
                           schedule_meta={"per_tier": {"2": 4, "4": 6}})


OK4 = {rid: 0 for rid in range(4, 10)}


def full(**override):
    rows = {rid: PER for rid in range(4, 10)}
    rows.update({int(k[1:]): v for k, v in override.items()})
    return rows


def c7(td, next_drain):
    r = C.Result("C7", td.tier.seq, "test")
    C._chunk_rows(ctx(), td, r, next_drain)
    return r.finish()


def test_rows_split_across_the_tier_boundary_add_up_to_pass():
    # req 9 was planned in tier 4 and received 8 frames before the tier end, the rest after
    td = TD(4, full(r9=8), OK4)
    r = c7(td, TD(5, {9: PER - 8}))
    assert (r.status, r.expected, r.checked, r.red) == (C.PASS, 6, 6, 0)
    assert any("counted from tier 5" in d for d in r.detail)


def test_split_rows_with_the_wrong_total_stay_red():
    for after in (PER - 9, PER - 7):
        r = c7(TD(4, full(r9=8), OK4), TD(5, {9: after}))
        assert (r.status, r.red) == (C.RED, 1), after
        assert "1 sampled requests with != %d rows, 0 stray ids" % PER in r.detail[-1]


def test_rows_outside_this_tier_and_the_next_drain_are_not_counted():
    # the missing 9 rows lie in some other tier's files: not passed here, so still missing
    r = c7(TD(4, full(r9=8), OK4), TD(5, {}))
    assert (r.status, r.red) == (C.RED, 1)
    r = c7(TD(4, full(r9=8), OK4), None)
    assert (r.status, r.red) == (C.RED, 1)


def test_stray_ids_in_the_tier_own_files_stay_red():
    # rows of a request planned in tier 2 inside tier 4's own files are stray, as before
    r = c7(TD(4, {**full(), 1: PER}, OK4), TD(5, {}))
    assert (r.status, r.red) == (C.RED, 1)
    assert "0 sampled requests with != %d rows, 1 stray ids" % PER in r.detail[-1]


def test_next_drain_rows_of_other_requests_are_neither_counted_nor_stray():
    # the drain's files may hold rows of requests planned elsewhere: they belong to no check here
    r = c7(TD(4, full(), OK4), TD(5, {1: 3}))
    assert (r.status, r.red) == (C.PASS, 0)
    assert not any("counted from tier" in d for d in r.detail)


def test_unsampled_rows_in_the_next_drain_are_not_counted():
    # mod 2: only even req_ids are sampled; req 9's rows in the drain do not count anywhere
    rows = {rid: PER for rid in range(4, 10) if rid % 2 == 0}
    r = c7(TD(4, rows, OK4, mod=2), TD(5, {9: PER, 8: 0}, mod=2))
    assert (r.status, r.expected, r.checked, r.red) == (C.PASS, 3, 3, 0)


def test_failed_split_request_is_left_to_c3():
    reqs = {**OK4, 9: 3}
    r = c7(TD(4, full(r9=2), reqs), TD(5, {9: 1}))
    assert (r.status, r.checked, r.red) == (C.PASS, 5, 0)


def test_check_chains_passes_the_next_drain_to_the_chunk_rows():
    c = ctx()
    td = TD(4, full(r9=8), OK4)
    r = C.check_chains(c, None, td, TD(5, {9: PER - 8}), "chunks")
    assert (r.status, r.red, r.checked) == (C.PASS, 0, 6)
    r = C.check_chains(c, None, td, TD(5, {}), "chunks")
    assert (r.status, r.red) == (C.RED, 1)
