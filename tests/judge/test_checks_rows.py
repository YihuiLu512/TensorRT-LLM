# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-level rules of GPU-IDLE, C3, C8, RECORDS and the C13 / D14 window scan on hand-made tree
data: the expected count comes from the configuration (sidecar meta.gpu_every_ns, schedule
per_tier, caplaunch's snapshot_copies), an unreadable reading is never counted as checked, and
histogram slots are counted by their bucket totals."""

import importlib
from types import SimpleNamespace

from conftest import PKG, judge_mod

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
    sites = {"F2": {"proc_kind": "frontend_openengine", "slots": ["count.F2.msgs"]},
             "Lmfl": {"proc_kind": "frontend_launcher", "slots": ["lag.Lmfl"]}}
    r = c8({"count.F2.msgs": 3}, {"count.F2.msgs": 9}, None, sites=sites,
           ctx=c8_ctx(["frontend_openengine"]))
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
    sites = {"F2": {"proc_kind": "frontend_openengine", "slots": ["count.F2.msgs"]},
             "W1": {"proc_kind": "worker", "slots": ["count.W1.x"]}}
    r = c8({"count.F2.msgs": 3}, {"count.F2.msgs": 9}, None, sites=sites,
           ctx=c8_ctx(["frontend_openengine"], ["worker"], None))
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


# ---- GPU-IDLE: only processes outside this launch are foreign occupancy ----------------------

OWN = {501: "registered process of role frontend", 777: "descendants in the tier 2 wstart scan"}


def test_own_process_holding_a_context_at_util_0_is_listed_not_red():
    ld = LD(rows(NS, pids=(501,), override={14 * NS: {"compute_pids": [501, 777]}}), NS)
    r, void = C.check_gpu_idle(None, ld, [TIER], own_pids=OWN)
    assert (r.status, void, r.red, r.checked) == (C.PASS, False, 0, 10)
    assert any("pid 501" in d and "10 window row(s)" in d for d in r.detail)
    assert any("pid 777" in d and "1 window row(s)" in d and "descendants" in d
               for d in r.detail)


def test_foreign_or_unattributable_compute_process_is_red():
    for pid in (999, None):
        ld = LD(rows(NS, override={13 * NS: {"compute_pids": [501, pid]}}), NS)
        r, void = C.check_gpu_idle(None, ld, [TIER], own_pids=OWN)
        assert (r.status, void) == (C.RED, True), pid
    # own process but the GPU is busy: CONSTRAINTS 2.4 (utilisation 0) still applies
    ld = LD(rows(NS, pids=(501,), override={13 * NS: {"util_pct": 3}}), NS)
    r, void = C.check_gpu_idle(None, ld, [TIER], own_pids=OWN)
    assert (r.status, void) == (C.RED, True)


def test_launch_pids_from_roles_json_and_window_scans():
    ld = SimpleNamespace(roles_json={"procs": [{"pid": 501, "role": "frontend"}],
                                     "aux_helpers": [{"pid": 502, "role": "capload"}]})
    td = SimpleNamespace(tier=TIER, tier_json={"scans": {
        "wstart": {"descendants": [[601, 1]], "monitor_descendants": [[602, 1, "sidecar"]],
                   "aux_helpers": []}, "wend": {"descendants": [[603, 1]]}}})
    assert sorted(C.launch_pids(ld, [td])) == [501, 502, 601, 602, 603]


# ---- RECORDS: a registered snapshot writer without its window snapshot ------------------------

def _snap():
    return SimpleNamespace(values={})


def records_td(snaps, copies):
    return SimpleNamespace(tier=TIER, exists=True, errors=[], snaps=snaps,
                           tier_json={"snapshot_copies": copies})


def test_records_red_on_snapshot_missing_even_when_another_process_has_its_copies():
    copies = [{"pid": 7, "role": "fe", "which": w, "file": "fe.7.%s.bin" % w}
              for w in ("wstart", "wend")]
    copies += [{"pid": 8, "role": "fe", "which": w, "snapshot_missing": True, "detail": "x"}
               for w in ("wstart", "wend")]
    r = C.check_records(records_td({7: [_snap(), _snap()]}, copies))
    assert (r.status, r.red) == (C.RED, 2)
    assert any("pid 8" in d and "no wstart window snapshot" in d for d in r.detail)


def test_records_red_when_a_listed_copy_was_not_read_or_the_list_is_absent():
    copies = [{"pid": 7, "role": "fe", "which": "wstart", "file": "fe.7.wstart.bin"}]
    assert C.check_records(records_td({}, copies)).status == C.RED
    td = records_td({}, [])
    td.tier_json = {}
    assert C.check_records(td).status == C.RED


def test_records_pass_when_every_listed_copy_is_there():
    copies = [{"pid": 7, "role": "fe", "which": w, "file": "fe.7.%s.bin" % w}
              for w in ("wstart", "wend")]
    assert C.check_records(records_td({7: [_snap(), _snap()]}, copies)).status == C.PASS


# ---- C8 with the process-kind vocabulary of the capprobe sites table ---------------------------

SITES = {"H1": {"proc_kind": "frontend", "slots": ["count.H1"]},
         "H3": {"proc_kind": "frontend", "slots": ["count.H3"]},
         "F2": {"proc_kind": "frontend_openengine", "slots": ["count.F2"]},
         "F4": {"proc_kind": "frontend_openengine", "slots": ["count.F4"]},
         "Lfe": {"proc_kind": "frontend_single", "slots": ["lag.Lfe"]},
         "Lmfl": {"proc_kind": "frontend_launcher", "slots": ["lag.Lmfl"]},
         "Lmfa": {"proc_kind": "frontend_attached", "slots": ["lag.Lmfa"]}}
COUNT = ["count.H1", "count.H3", "count.F2", "count.F4"]


def c8_multi(kinds, fired):
    """Snapshots of every process: all slots of the open groups present, fired ones grow."""
    a = {s: (0 if s.startswith("count") else [0, 0]) for s in COUNT + ["lag.Lfe", "lag.Lmfl",
                                                                     "lag.Lmfa"]}
    b = {s: ((5 if s.startswith("count") else [1, 4]) if s in fired else v)
         for s, v in a.items()}
    snaps = {p: [SimpleNamespace(values=a), SimpleNamespace(values=b)] for p in (1, 2, 3)}
    ld = SimpleNamespace(overlay={"sites": SITES})
    return C.check_c8(c8_ctx(kinds), ld, SimpleNamespace(tier=TIER, snaps=snaps))


def test_c8_single_frontend_checks_count_sites_and_lfe():
    r = c8_multi(["frontend", "frontend_openengine", "frontend_single"], COUNT + ["lag.Lfe"])
    assert (r.status, r.expected) == (C.PASS, 5)
    assert "Lmfa(frontend_attached), Lmfl(frontend_launcher)" in r.detail[0]


def test_c8_multi_frontend_checks_count_sites_without_lfe():
    kinds = ["frontend", "frontend_openengine", "frontend_launcher", "frontend_attached"]
    r = c8_multi(kinds, COUNT + ["lag.Lmfl", "lag.Lmfa"])
    assert (r.status, r.expected) == (C.PASS, 6)
    assert "Lfe(frontend_single)" in r.detail[0]
    # a count site that never fired is still checked (not "not applicable")
    r = c8_multi(kinds, ["count.H1", "count.H3", "count.F2", "lag.Lmfl", "lag.Lmfa"])
    assert r.status == C.RED and any("site F4" in d for d in r.detail)


def test_c8_old_architecture_frontend_checks_only_h1_h3():
    r = c8_multi(["frontend"], ["count.H1", "count.H3"])
    assert (r.status, r.expected) == (C.PASS, 2)


# ---- shared /proc/stat window scan (C13 second half and D14) ---------------------------------

T = importlib.import_module("%s.treedata" % PKG)
STAT = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


def cpu_row(t, cpu, **v):
    row = {"t_rel_ns": t, "cpu": cpu}
    row.update({k: v.get(k, 0) for k in STAT})
    return row


def test_offline_core_null_row_makes_the_reading_unusable_not_a_crash():
    rows = [cpu_row(11 * NS, 4, user=10, idle=90), cpu_row(19 * NS, 4, user=90, idle=710)]
    d, why = T.cpu_tick_deltas(rows, {4}, TIER.window_start_ns, TIER.window_end_ns)
    assert why is None and d[4]["user"] == 80 and d[4]["t_last_rel_ns"] == 19 * NS
    nulls = dict(cpu_row(15 * NS, 4), **{k: None for k in STAT})
    d, why = T.cpu_tick_deltas(rows + [nulls], {4}, TIER.window_start_ns, TIER.window_end_ns)
    assert d is None and "offline" in why
    D14 = judge_mod("d14")
    rd, why = D14.load_reading(rows + [nulls], {4}, TIER.window_start_ns, TIER.window_end_ns)
    assert rd is None and "offline" in why
    frac, why = C._foreign_frac(rows + [nulls], [], {}, "r", {4}, TIER.window_start_ns,
                                TIER.window_end_ns, 100, True)
    assert frac is None and "offline" in why


def test_c13_subtracts_softirq_only_never_hard_irq():
    rows = [cpu_row(10 * NS, 4), cpu_row(19 * NS, 4, irq=50, softirq=30, idle=820)]
    args = ([], {}, "r", {4}, TIER.window_start_ns, TIER.window_end_ns, 100)
    # 900 ticks in 9 s at 100 Hz; no registered thread
    assert C._foreign_frac(rows, *args, False)[0] == (50 + 30) / 900
    assert C._foreign_frac(rows, *args, True)[0] == 50 / 900
