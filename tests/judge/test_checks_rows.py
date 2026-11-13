# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-level rules of GPU-IDLE, C3 and C8 on hand-made tree data: the expected count comes from
the configuration (sidecar meta.gpu_every_ns, schedule per_tier), an unreadable reading is
never counted as checked, and histogram slots are counted by their bucket totals."""

from types import SimpleNamespace

from conftest import judge_mod

C = judge_mod("checks")
NS = 1_000_000_000
TIER = SimpleNamespace(seq=2, window_ns=10 * NS, window_start_ns=10 * NS,
                       window_end_ns=20 * NS)


class LD:
    def __init__(self, rows, every_ns, count=1, pc_ok=True, header_pc_ok=True):
        self.gpu_idle = {"positive_control_ok": pc_ok, "gpu_count": count}
        self.sidecar_meta = {} if every_ns is None else {"gpu_every_ns": every_ns}
        self._gpu = ({"gpu_count": count, "positive_control_ok": header_pc_ok}, rows)

    def sidecar(self, name):
        return self._gpu if name == "gpu" else None


def rows(every_ns, util=0, pids=(), rc=0, override=None):
    out = []
    t = 9 * NS
    while t < 21 * NS:
        row = {"t_rel_ns": t, "gpu_index": 0, "util_pct": util, "mem_used_mib": 0,
               "compute_pids": list(pids), "query_rc": rc}
        if override and t in override:
            row.update(override[t])
        out.append(row)
        t += every_ns
    return out


def gpu(ld):
    return C.check_gpu_idle(None, ld, [TIER])


def test_expected_rows_follow_the_sidecar_gpu_period():
    r, void = gpu(LD(rows(2 * NS), 2 * NS))
    assert (r.status, r.expected, r.checked, void) == (C.PASS, 5, 5, False)


def test_missing_gpu_period_is_not_run():
    r, void = gpu(LD(rows(NS), None))
    assert r.status == C.NOT_RUN and not void
    assert "gpu_every_ns" in r.not_run[0]


def test_unreadable_util_is_not_counted_and_not_run():
    r, void = gpu(LD(rows(NS, override={12 * NS: {"util_pct": None}}), NS))
    assert r.status == C.NOT_RUN and not void
    assert (r.expected, r.checked) == (10, 9)


def test_missing_compute_pids_is_not_run():
    r, _ = gpu(LD(rows(NS, override={15 * NS: {"compute_pids": None}}), NS))
    assert r.status == C.NOT_RUN and r.checked == 9


def test_failed_query_is_not_counted():
    r, _ = gpu(LD(rows(NS, override={11 * NS: {"query_rc": 2, "util_pct": None,
                                               "compute_pids": None}}), NS))
    assert r.status == C.NOT_RUN and r.checked == 9


def test_busy_evidence_is_red_even_without_util():
    r, void = gpu(LD(rows(NS, override={13 * NS: {"util_pct": None,
                                                  "compute_pids": [4242]}}), NS))
    assert r.status == C.RED and void and r.red == 1


def test_busy_util_is_red():
    r, void = gpu(LD(rows(NS, override={13 * NS: {"util_pct": 7}}), NS))
    assert r.status == C.RED and void


def test_sidecar_positive_control_failure_is_not_run():
    r, _ = gpu(LD(rows(NS), NS, header_pc_ok=False))
    assert r.status == C.NOT_RUN and "positive_control_ok" in r.not_run[0]


def test_all_idle_one_second_period_passes():
    r, void = gpu(LD(rows(NS), NS))
    assert (r.status, r.expected, r.checked, void) == (C.PASS, 10, 10, False)


# ---- C3: outcomes must cover every planned request -------------------------------------------

def c3_ctx(planned):
    return SimpleNamespace(schedule_meta={"per_tier": {"2": planned}},
                           roles={"capload": {"role": "capload", "kind": "client"}})


def c3_td(ok, failed):
    doc = {"counts": {"ok": ok, "failed": failed}, "failures": []}
    return SimpleNamespace(tier=TIER, summaries={"capload": {0: doc}})


def test_c3_passes_when_every_planned_request_has_an_outcome():
    assert C.check_c3(c3_ctx(10), c3_td(9, 1)).status == C.PASS
    assert C.check_c3(c3_ctx(10), c3_td(10, 0)).status == C.PASS


def test_c3_missing_outcomes_is_not_run():
    r = C.check_c3(c3_ctx(10), c3_td(6, 0))
    assert r.status == C.NOT_RUN and "6 of 10" in r.not_run[0]


# ---- C8: histogram slots of lag sites --------------------------------------------------------

def c8_ctx(*kinds_per_role):
    return SimpleNamespace(roles={"r%d" % i: {"role": "r%d" % i, "proc_kinds": k}
                                  for i, k in enumerate(kinds_per_role)})


def c8(values_a, values_b, slot, sites=None, ctx=None):
    sites = sites or {"S": {"proc_kind": "frontend", "slots": [slot]}}
    ld = SimpleNamespace(overlay={"sites": sites})
    snaps = {7: [SimpleNamespace(values=values_a), SimpleNamespace(values=values_b)]}
    return C.check_c8(ctx or c8_ctx(["frontend"]), ld, SimpleNamespace(tier=TIER, snaps=snaps))


def test_c8_counts_histogram_slots_by_bucket_total():
    r = c8({"lag.Lfe": [0, 1, 2]}, {"lag.Lfe": [0, 3, 5]}, "lag.Lfe")
    assert (r.status, r.checked, r.red) == (C.PASS, 1, 0)


def test_c8_histogram_without_new_samples_is_red():
    r = c8({"lag.Lfe": [0, 1, 2]}, {"lag.Lfe": [0, 1, 2]}, "lag.Lfe")
    assert r.status == C.RED


def test_c8_counter_slot_still_works():
    assert c8({"count.F2.msgs": 3}, {"count.F2.msgs": 9}, "count.F2.msgs").status == C.PASS
    assert c8({"count.F2.msgs": 3}, {"count.F2.msgs": 3}, "count.F2.msgs").status == C.RED


def test_c8_site_of_an_absent_process_kind_is_not_applicable():
    sites = {"F2": {"proc_kind": "frontend", "slots": ["count.F2.msgs"]},
             "Lmfl": {"proc_kind": "frontend_launcher", "slots": ["lag.Lmfl"]}}
    r = c8({"count.F2.msgs": 3}, {"count.F2.msgs": 9}, None, sites=sites)
    assert (r.status, r.expected, r.checked, r.red) == (C.PASS, 1, 1, 0)
    assert "not applicable in this topology: Lmfl(frontend_launcher)" in r.detail[0]


def test_c8_no_applicable_site_is_not_run():
    sites = {"Lmfl": {"proc_kind": "frontend_launcher", "slots": ["lag.Lmfl"]}}
    r = c8({}, {}, None, sites=sites)
    assert r.status == C.NOT_RUN and "Lmfl(frontend_launcher)" in r.detail[0]


def test_c8_without_declared_proc_kinds_is_not_run():
    no_decl = SimpleNamespace(roles={"capload": {"role": "capload"}})
    r = c8({"count.F2.msgs": 3}, {"count.F2.msgs": 9}, "count.F2.msgs", ctx=no_decl)
    assert r.status == C.NOT_RUN and "proc_kinds" in r.not_run[0]
    bad = c8_ctx("frontend")  # a string, not a list
    assert c8({"count.F2.msgs": 3}, {"count.F2.msgs": 9}, "count.F2.msgs",
              ctx=bad).status == C.NOT_RUN


def test_c8_union_of_role_kinds_and_red_on_applicable_site():
    sites = {"F2": {"proc_kind": "frontend", "slots": ["count.F2.msgs"]},
             "W1": {"proc_kind": "worker", "slots": ["count.W1.x"]}}
    r = c8({"count.F2.msgs": 3}, {"count.F2.msgs": 9}, None, sites=sites,
           ctx=c8_ctx(["frontend"], ["worker"], None))
    assert (r.status, r.expected, r.red) == (C.RED, 2, 1)  # W1 applies and never fired


# ---- status order RED > NOT_RUN > PASS; C13 launch level ----------------------------------

def test_red_evidence_wins_over_nothing_expected():
    r = C.Result("X", "launch", "n/a")
    r.red = 1
    assert r.finish().status == C.RED
    r = C.Result("X", "launch", "n/a")
    r.expected, r.checked = 3, 3
    r.not_run.append("something missing")
    assert r.finish().status == C.NOT_RUN


class C13LD:
    def __init__(self, procs, unattributed=(), aux=(), inventories=()):
        self.roles_json = {"procs": list(procs), "unattributed": list(unattributed),
                           "aux_helpers": list(aux)}
        self._inv = list(inventories)

    def procs_by_pid(self):
        return {p["pid"]: p for p in self.roles_json["procs"]}

    def thread_inventories(self):
        return self._inv


def c13_ctx(*roles):
    return SimpleNamespace(roles={r: {"role": r, "measure_class": mc, "expected_procs": 1}
                                  for r, mc in roles})


AUX = {"pid": 99, "role": "relay1", "category": "aux_helper", "rule": "mp_resource_tracker"}


def test_c13_unattributed_is_red_without_under_test_roles():
    ctx = c13_ctx(("nullsse", "stub_out_of_domain"))
    ld = C13LD([{"pid": 10, "role": "nullsse"}], unattributed=[{"pid": 77}])
    r = C.check_c13_launch(ctx, ld)
    assert (r.status, r.expected, r.red) == (C.RED, 0, 1)


def test_c13_aux_helpers_are_not_processes_under_test():
    ctx = c13_ctx(("relay1", "under_test"))
    inv = [({"pid": 10, "threads": [{"tid": 10, "identified_as": "MainThread"}]}, "p")]
    ld = C13LD([{"pid": 10, "role": "relay1"}], aux=[AUX], inventories=inv)
    r = C.check_c13_launch(ctx, ld)
    assert (r.status, r.expected, r.checked, r.red) == (C.PASS, 1, 1, 0)
    assert any("aux helper" in d and "relay1" in d for d in r.detail)
