# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal localisation I (design 2.6; INTERFACES errata B2-2E): I1 per core group (a role's
processes whose cores intersect, parent included; single core: core idle; several cores: per
process, GIL domain summed with run-queue wait, or one non-GIL thread), I3 by the side where
the backlog sits, and the combination into LOCALIZED / UNDETERMINED / ...; plus
reclassify_failure's load-generator connection failures."""

from types import SimpleNamespace as NS

from conftest import judge_mod

L = judge_mod("localize")
X = judge_mod("symptoms")
S = 1_000_000_000
BAND = {"min": -0.05, "max": 0.05, "n": 3}


def _tier(seq=4, ws=0, we=4 * S):
    return NS(seq=seq, window_start_ns=ws, window_end_ns=we)


def _ld(cpu_rows, th_rows, procs, inventory=(), clk=100):
    return NS(sidecar_meta={"clk_tck": clk},
              sidecar=lambda name: ({}, {"cpus": cpu_rows, "threads": th_rows}[name]),
              roles_json={"procs": procs}, thread_inventories=lambda: list(inventory),
              procs_of_role=lambda role: [p for p in procs if p.get("role") == role])


def _cpu_rows(core, busy_per_s, softirq_per_s=0, secs=5):
    rows = []
    for k in range(secs):
        b = busy_per_s * k
        si = softirq_per_s * k
        rows.append({"t_rel_ns": k * S, "cpu": core, "user": b, "nice": 0, "system": 0,
                     "idle": 100 * k - b - si, "iowait": 0, "irq": 0, "softirq": si,
                     "steal": 0})
    return rows


def _th_rows(pid, rates, run_delay=None, secs=5):
    """rates: {tid: ticks per second}; run_delay: {tid: ns of run-queue wait per second}."""
    rows = []
    for k in range(secs):
        for tid, r in rates.items():
            rd = None if run_delay is None else run_delay.get(tid, 0) * k
            rows.append({"t_rel_ns": k * S, "pid": pid, "tid": tid, "comm": "t%d" % tid,
                         "utime_ticks": r * k, "stime_ticks": 0, "run_delay_ns": rd})
    return rows


def _td(pid, classes, seq=4):
    th = [{"tid": t, "class": c} for t, c in classes.items()]
    return NS(tier=_tier(seq), tier_json={"threads": {"wstart": {str(pid): th}}}, summaries={})


def group(core_rates, classes, cores=(4,), others=(), softirq=0, run_delay=None):
    """group_readings of pid 7 (role r) alone on `cores`; core_rates {tid: ticks/s}; others:
    one process of role o on each listed core."""
    pid = 7
    cpu_rows = []
    total = sum(core_rates.values())
    for c in cores:
        cpu_rows += _cpu_rows(c, total // len(cores), softirq)
    procs = [{"pid": pid, "role": "r", "cpus": list(cores)}] + [
        {"pid": 100 + i, "role": "o", "cpus": [c]} for i, c in enumerate(others)]
    ld = _ld(cpu_rows, _th_rows(pid, core_rates, run_delay), procs)
    (g,), nocpu = L.core_groups(ld, "r")
    assert nocpu == []
    return L.group_readings(ld, _td(pid, classes), g)


def readings(*a, **kw):
    """What i1_process reads for pid 7: the single-core group's unit, or pid 7's unit."""
    g, why = group(*a, **kw)
    if g is None:
        return None, why
    return L.unit_view(g, None if g["n_cores"] == 1 else 7), None


# ---- I1 readings -----------------------------------------------------------------------------

def test_single_core_readings_and_gil_sum():
    g, why = group({7: 60, 8: 30, 9: 5}, {7: "python", 8: "python", 9: "grpc_core"},
                   softirq=2)
    assert why is None and g["n_cores"] == 1 and g["pids"] == [7]
    assert abs(g["core_busy_frac"] - 0.95) < 1e-9  # 95 user; softirq left out
    assert abs(g["core_busy_frac_softirq"] - 0.97) < 1e-9  # + 2 softirq
    assert abs(g["procs"][7]["gil_cores"] - 0.9) < 1e-9
    cur = L.unit_view(g)
    assert {t["tid"]: t["class"] for t in cur["threads"]} == {7: "python", 8: "python",
                                                            9: "grpc_core"}


def test_d14_picks_the_softirq_statistic():
    base, _ = readings({7: 30}, {7: "python"}, softirq=3)
    hot, _ = readings({7: 93}, {7: "python"}, softirq=3)  # 0.93 without, 0.96 with softirq
    b = {"T2:core_busy_frac:r": BAND, "T2:core_busy_frac_softirq:r": BAND}
    r = _i1(hot, base, b, d14=True)
    assert (r["status"], r["statistic"]) == ("TRUE", "T2:core_busy_frac_softirq:r")
    r = _i1(hot, base, b, d14=False)
    assert (r["status"], r["statistic"]) == ("FALSE", "T2:core_busy_frac:r")
    r = _i1(hot, base, b, d14=None)
    assert r["status"] == "NOT_RUN" and "D14" in r["reasons"][0]


def test_cores_shared_with_another_role_are_not_run():
    cur, why = readings({7: 50}, {7: "python"}, cores=(4,), others=(4,))
    assert cur is None and "shared" in why and "another role" in why
    cur, why = readings({7: 50}, {7: "python"}, cores=(4,), others=(5,))
    assert cur is not None


def _i1(cur, base, bands, near=0.95, d14=False):
    return L.i1_process(cur, base, bands, "r", near, d14, {}, False)


def test_single_core_rule_needs_near_one_and_an_offset_above_the_band():
    base, _ = readings({7: 30}, {7: "python"})
    hot, _ = readings({7: 98}, {7: "python"})
    b = {"T2:core_busy_frac:r": BAND}
    r = _i1(hot, base, b)
    assert (r["status"], r["via"], r["branch"]) == ("TRUE", "core_idle", "single_core")
    # a thread that already spins at the baseline: busy, but no offset -> not localised
    spin_base, _ = readings({7: 97}, {7: "python"})
    assert _i1(hot, spin_base, b)["status"] == "FALSE"
    # a rise that stays far from one core
    mid, _ = readings({7: 60}, {7: "python"})
    assert _i1(mid, base, b)["status"] == "FALSE"


def test_i1_not_run_without_threshold_band_or_baseline():
    base, _ = readings({7: 30}, {7: "python"})
    hot, _ = readings({7: 98}, {7: "python"})
    b = {"T2:core_busy_frac:r": BAND}
    assert _i1(hot, base, b, near=None)["status"] == "NOT_RUN"
    assert _i1(hot, base, {})["status"] == "NOT_RUN"
    assert _i1(hot, None, b)["status"] == "NOT_RUN"


def test_offsets_are_recorded_except_on_the_baseline_tier():
    base, _ = readings({7: 30}, {7: "python"})
    hot, _ = readings({7: 98}, {7: "python"})
    rec = {}
    # recorded even without a threshold or a D14 result: the pilot builds their bands from
    # trees judged before it existed
    r = L.i1_process(hot, base, {}, "r", None, None, rec, False)
    assert r["status"] == "NOT_RUN"
    assert abs(rec["T2:core_busy_frac:r"][0] - 0.68) < 1e-9
    assert abs(rec["T2:core_busy_frac_softirq:r"][0] - 0.68) < 1e-9
    rec = {}
    L.i1_process(base, base, {}, "r", 0.95, False, rec, True)
    assert rec == {}


MC = {"T2:gil_domain_cores:r": BAND, "T2:gil_run_delay_frac:r": BAND,
      "T2:nongil_thread_cores:r": BAND}


def test_multi_core_gil_domain_summed_over_python_threads():
    cls = {7: "python", 8: "python", 9: "zmq_io"}
    base, _ = readings({7: 30, 8: 30, 9: 10}, cls, cores=(4, 5), run_delay={})
    hot, _ = readings({7: 60, 8: 37, 9: 10}, cls, cores=(4, 5), run_delay={})
    assert hot["n_cores"] == 2 and abs(hot["gil_cores"] - 0.97) < 1e-9
    r = _i1(hot, base, MC)
    assert (r["status"], r["via"], r["branch"]) == ("TRUE", "gil_domain", "multi_core")
    # neither thread alone is near one core: a per-thread rule would miss it
    assert all(t["cores"] < 0.95 for t in hot["threads"])


def test_multi_core_rising_run_queue_wait_is_not_the_domain():
    cls = {7: "python", 8: "python"}
    base, _ = readings({7: 30, 8: 30}, cls, cores=(4, 5), run_delay={7: 0})
    hot, _ = readings({7: 60, 8: 37}, cls, cores=(4, 5), run_delay={7: S // 5})
    assert _i1(hot, base, MC)["status"] == "FALSE"
    # run-queue wait unreadable -> the GIL branch cannot be judged
    hot_nord, _ = readings({7: 60, 8: 37}, cls, cores=(4, 5), run_delay=None)
    assert _i1(hot_nord, base, MC)["status"] == "NOT_RUN"


def test_multi_core_non_gil_thread_alone():
    cls = {7: "python", 9: "zmq_io"}
    base, _ = readings({7: 20, 9: 30}, cls, cores=(4, 5), run_delay={})
    hot, _ = readings({7: 20, 9: 97}, cls, cores=(4, 5), run_delay={})
    r = _i1(hot, base, MC)
    assert (r["status"], r["via"], r["threads_hot"]) == ("TRUE", "non_gil_thread", [9])
    no_band = {k: v for k, v in MC.items() if "nongil" not in k}
    assert _i1(hot, base, no_band)["status"] == "NOT_RUN"


def test_multi_core_thread_of_unknown_class_is_not_run():
    # thread 8 is in neither the inventory nor tier.json: it may be a python thread, so the
    # GIL domain read without it may be short; NOT_RUN, not FALSE
    base, _ = readings({7: 30, 8: 30}, {7: "python"}, cores=(4, 5), run_delay={})
    hot, _ = readings({7: 60, 8: 37}, {7: "python"}, cores=(4, 5), run_delay={})
    rec = {}
    r = L.i1_process(hot, base, MC, "r", 0.95, False, rec, False)
    assert r["status"] == "NOT_RUN" and r["unknown_threads"] == [8]
    assert "thread readings incomplete" in r["reasons"][0]
    assert "T2:gil_domain_cores:r" not in rec and "T2:nongil_thread_cores:r" not in rec
    # nor is it a non-GIL candidate when it is the one near a core
    base, _ = readings({7: 20, 9: 30}, {7: "python"}, cores=(4, 5), run_delay={})
    hot, _ = readings({7: 20, 9: 97}, {7: "python"}, cores=(4, 5), run_delay={})
    r = _i1(hot, base, MC)
    assert r["status"] == "NOT_RUN" and "threads_hot" not in r
    # a python thread with a single sidecar row in the window: its CPU there is unknown
    ld = _ld(_cpu_rows(4, 50) + _cpu_rows(5, 47),
             _th_rows(7, {7: 60, 8: 37}) + [{"t_rel_ns": 2 * S, "pid": 7, "tid": 5, "comm": "t5",
                                             "utime_ticks": 80, "stime_ticks": 0,
                                             "run_delay_ns": 0}],
             [{"pid": 7, "role": "r", "cpus": [4, 5]}])
    (g,), _ = L.core_groups(ld, "r")
    gr, _ = L.group_readings(ld, _td(7, {7: "python", 8: "python", 5: "python"}), g)
    assert gr["procs"][7]["partial_threads"] == [(5, "python")]
    base, _ = readings({7: 30, 8: 30}, {7: "python", 8: "python"}, cores=(4, 5), run_delay={})
    r = _i1(L.unit_view(gr, 7), base, MC)
    assert r["status"] == "NOT_RUN" and r["unknown_threads"] == [5]
    # an unknown thread without CPU in either window changes nothing
    base, _ = readings({7: 30, 8: 30, 6: 0}, {7: "python", 8: "python"}, cores=(4, 5),
                       run_delay={})
    hot, _ = readings({7: 60, 8: 37, 6: 0}, {7: "python", 8: "python"}, cores=(4, 5),
                      run_delay={})
    assert _i1(hot, base, MC)["status"] == "TRUE"


# ---- I1 on the real topology: a parent registered with its workers on the role's core set ----

# roles.json as caplaunch writes it (E1-0): sink parent 10 + workers 11, 12 on cores 0-1; ld
# parent 20 + its worker 21 on core 2; canary 30 (observer) on core 3.
REAL = [{"pid": 10, "role": "sink", "cpus": [0, 1]}, {"pid": 11, "role": "sink", "cpus": [0, 1]},
        {"pid": 12, "role": "sink", "cpus": [0, 1]}, {"pid": 20, "role": "ld", "cpus": [2]},
        {"pid": 21, "role": "ld", "cpus": [2]}, {"pid": 30, "role": "canary", "cpus": [3]}]


def _real_ld(rates, procs=REAL):
    """rates: {pid: ticks/s of its one python thread (tid = pid)}; each core's busy ticks are
    the sum over the processes pinned to it, split evenly over a process's cores."""
    busy = {}
    for p in procs:
        for c in p["cpus"]:
            busy[c] = busy.get(c, 0) + rates.get(p["pid"], 0) // len(p["cpus"])
    cpu_rows = [r for c in sorted(busy) for r in _cpu_rows(c, busy[c])]
    th = [r for p in procs for r in _th_rows(p["pid"], {p["pid"]: rates.get(p["pid"], 0)},
                                             run_delay={})]
    return _ld(cpu_rows, th, procs)


def _real_td(seq):
    th = {str(p["pid"]): [{"tid": p["pid"], "class": "python"}] for p in REAL}
    return NS(tier=_tier(seq), tier_json={"threads": {"wstart": th}}, summaries={})


def test_core_groups_join_a_roles_parent_and_workers():
    ld = _real_ld({})
    (sink,), nocpu = L.core_groups(ld, "sink")
    assert (sink["pids"], sink["cores"], sink["shared_with"], nocpu) == ([10, 11, 12], {0, 1},
                                                                         [], [])
    (g,), _ = L.core_groups(ld, "ld")
    assert (g["pids"], g["cores"], g["shared_with"]) == ([20, 21], {2}, [])
    # chained overlap joins too; another role on one of the cores makes the group shared
    procs = [dict(p, cpus=[1]) if p["pid"] == 12 else p for p in REAL] + [
        {"pid": 13, "role": "sink", "cpus": [1, 4]}, {"pid": 40, "role": "o", "cpus": [4]},
        {"pid": 14, "role": "sink", "cpus": []}]
    (g,), nocpu = L.core_groups(_real_ld({}, procs), "sink")
    assert (g["pids"], g["cores"], g["shared_with"], nocpu) == ([10, 11, 12, 13], {0, 1, 4},
                                                                [40], [14])


def _i1_real(hot, base, bands, rec):
    lds, tds = {4: _real_ld(hot), 2: _real_ld(base)}, {4: _real_td(4), 2: _real_td(2)}

    def readings(tdx, g):
        return L.group_readings(lds[tdx.tier.seq], tdx, g)
    out = {}
    for role in ("sink", "ld"):
        out[role] = L.i1_role(lds[4], tds[4], tds[2], role, readings,
                              lambda cur, b, role=role: L.i1_process(cur, b, bands, role, 0.95,
                                                                     False, rec, False))
    return out


SINK_MC = {"T2:gil_domain_cores:sink": BAND, "T2:gil_run_delay_frac:sink": BAND,
           "T2:nongil_thread_cores:sink": BAND, "T2:core_busy_frac:ld": BAND}


def test_i1_on_a_parent_and_workers_sharing_the_roles_cores():
    rec = {}
    out = _i1_real({10: 1, 11: 97, 12: 40, 20: 2, 21: 96}, {10: 1, 11: 30, 12: 30, 20: 2,
                                                            21: 30}, SINK_MC, rec)
    sink = out["sink"]  # two cores: one unit per process (one GIL each), each lists the group
    assert [(u["pid"], u["unit"], u["branch"], u["status"]) for u in sink] == [
        (10, "process", "multi_core", "FALSE"), (11, "process", "multi_core", "TRUE"),
        (12, "process", "multi_core", "FALSE")]
    assert all(u["pids"] == [10, 11, 12] for u in sink) and sink[1]["via"] == "gil_domain"
    assert sink[1]["readings"]["cores"] == [0, 1] and sink[1]["readings"]["n_cores"] == 2
    (ld,) = out["ld"]  # one core shared by parent and worker: one unit, the core is the reading
    assert (ld["pid"], ld["pids"], ld["unit"], ld["branch"], ld["status"], ld["via"]) == (
        None, [20, 21], "core_group", "single_core", "TRUE", "core_idle")
    assert {t["pid"] for t in ld["readings"]["threads"]} == {20, 21}
    assert len(rec["T2:core_busy_frac:ld"]) == 1  # recorded once per group, not per process
    assert len(rec["T2:gil_domain_cores:sink"]) == 3  # once per process


def test_conclusion_lists_every_pid_of_the_localised_group():
    i1 = _i1_real({10: 1, 11: 30, 12: 30, 20: 2, 21: 96}, {10: 1, 11: 30, 12: 30, 20: 2,
                                                           21: 30}, SINK_MC, {})
    out = {"I1": {r: {"status": "TRUE" if any(u["status"] == "TRUE" for u in us) else "FALSE",
                      "processes": us} for r, us in i1.items()},
           "I3": {"readings": [], "roles": {}}, "reasons": []}
    ctx = NS(roles={"sink": {"measure_class": "stub_out_of_domain"},
                    "ld": {"measure_class": "under_test"}})
    r = L._conclude(out, {"holds": True}, ctx)
    assert (r["status"], r["role"], r["pids"], r["granularity"]) == ("LOCALIZED", "ld",
                                                                     [20, 21], "core_group")
    assert sorted(r["thread_shares"]) == ["20", "21"]


# ---- I3 --------------------------------------------------------------------------------------

READ = {"name": "inflight", "receiver": "rd", "sender": "sv",
        "terms": [{"role": "sv", "slot": "backlog.yielded", "sign": 1, "agg": "sum"},
                  {"role": "rd", "slot": "backlog.read", "sign": -1, "agg": "sum"}]}
I3B = {"T1:i3_backlog:inflight": {"min": -3, "max": 3, "n": 3},
       "T2:i3_backlog:inflight": {"min": -3, "max": 3, "n": 3}}


def _i3_ld(inflight_by_s, rd_pids=(2, 3)):
    """Snapshot rows for 8 s: sv (pid 1) yielded = 100 s + inflight(s), rd pids read 100 s
    split between them, so yielded - read = inflight(s)."""
    rows = []
    for k in range(8):
        y = 100 * k + inflight_by_s(k)
        rows.append({"pid": 1, "snap_t_rel_ns": k * S + 1, "counters": {"backlog.yielded": y}})
        for i, p in enumerate(rd_pids):
            rows.append({"pid": p, "snap_t_rel_ns": k * S + 2,
                         "counters": {"backlog.read": 100 * k if i == 0 else 0}})
    procs = [{"pid": 1, "role": "sv"}] + [{"pid": p, "role": "rd"} for p in rd_pids]
    return NS(sidecar=lambda name: ({}, rows),
              procs_of_role=lambda role: [p for p in procs if p["role"] == role])


def _state(inflight_by_s, bands=I3B):
    ld = _i3_ld(inflight_by_s)
    td = NS(tier=_tier(4, 4 * S, 8 * S))
    base = NS(tier=_tier(2, 0, 4 * S))
    return L.i3_reading(ld, td, base, READ, bands, {}, False)


def test_i3_reading_states():
    assert _state(lambda k: 2)["state"] == "flat"
    assert _state(lambda k: 2 + (10 * (k - 4) if k >= 4 else 0))["state"] == "growing"
    r = _state(lambda k: 64 if k >= 4 else 2)  # held at a plateau
    assert (r["state"], r["T1"], r["T2"]) == ("elevated", 0, 62)
    assert _state(lambda k: 64 if k >= 4 else 2, bands={})["state"] == "NOT_RUN"


def test_i3_reading_needs_every_process_of_a_term():
    ld = _i3_ld(lambda k: 2)
    ld.sidecar = (lambda rows: (lambda name: ({}, [r for r in rows if r["pid"] != 3])))(
        ld.sidecar("snapshots")[1])
    r = L.i3_reading(ld, NS(tier=_tier(4, 4 * S, 8 * S)), NS(tier=_tier(2, 0, 4 * S)),
                     READ, I3B, {}, False)
    assert r["state"] == "NOT_RUN"


def _rd(name, rec, snd, state):
    return {"name": name, "receiver": rec, "sender": snd, "state": state}


def test_i3_roles_backlog_side():
    slow_server = [_rd("pending", "sv", None, "growing"), _rd("inflight", "rd", "sv", "flat")]
    r = L.i3_roles(slow_server, ["sv", "rd"])
    assert r["sv"]["status"] == "TRUE" and r["rd"]["status"] == "FALSE"
    slow_reader = [_rd("pending", "sv", None, "growing"),
                   _rd("inflight", "rd", "sv", "elevated")]
    r = L.i3_roles(slow_reader, ["sv", "rd"])
    assert r["sv"]["status"] == "FALSE" and r["sv"]["exit_blocked"] == ["inflight"]
    assert r["rd"]["status"] == "TRUE"
    unknown_exit = [_rd("pending", "sv", None, "growing"), _rd("inflight", "rd", "sv",
                                                               "NOT_RUN")]
    r = L.i3_roles(unknown_exit, ["sv", "rd"])
    assert r["sv"]["status"] == "NOT_RUN" and r["rd"]["status"] == "NOT_RUN"
    assert "x" not in L.i3_roles(slow_server, ["sv", "rd", "x"])  # no entry reading


# ---- combination -----------------------------------------------------------------------------

def _out(i1=None, i3=None):
    return {"I1": {r: {"status": s, "processes": [{"pid": 1, "status": s, "via": "core_idle",
                                                   "readings": {"threads": []}}]}
                   for r, s in (i1 or {}).items()},
            "I3": {"readings": [], "roles": {r: {"status": s} for r, s in (i3 or {}).items()}},
            "reasons": []}


CTX = NS(roles={"sv": {"measure_class": "under_test"}, "rd": {"measure_class": "load_source"}})


def test_conclusion_localized_undetermined_and_not_localized():
    r = L._conclude(_out({"sv": "TRUE", "rd": "FALSE"}, {"sv": "TRUE"}), {"holds": True}, CTX)
    assert (r["status"], r["role"], r["via"], r["finding"]) == ("LOCALIZED", "sv",
                                                              ["I1", "I3"], "domain_saturated")
    r = L._conclude(_out({"sv": "FALSE"}, {"sv": "TRUE"}), {"holds": True}, CTX)
    assert (r["status"], r["finding"]) == ("LOCALIZED", "backlog_at_entry")
    r = L._conclude(_out({"sv": "TRUE"}, {"rd": "TRUE"}), {"holds": True}, CTX)
    assert r["status"] == "UNDETERMINED" and r["holding"] == {"rd": ["I3"], "sv": ["I1"]}
    assert L._conclude(_out({"sv": "FALSE"}, {"sv": "FALSE"}), {"holds": True},
                       CTX)["status"] == "NOT_LOCALIZED"
    assert L._conclude(_out({"sv": "NOT_RUN"}, {}), {"holds": True}, CTX)["status"] == "NOT_RUN"


def test_an_unevaluated_competing_role_blocks_the_conclusion():
    # I1 of sv holds; rd's I3 could not be evaluated: rd is not ruled out
    r = L._conclude(_out({"sv": "TRUE", "rd": "FALSE"}, {"rd": "NOT_RUN"}), {"holds": True},
                    CTX)
    assert r["status"] == "NOT_RUN" and r["holding"] == {"sv": ["I1"]}
    assert r["not_run"] == ["I3:rd"] and "competing" in r["reasons"][-1]
    # the localised role's own other signal NOT_RUN does not block it
    r = L._conclude(_out({"sv": "TRUE"}, {"sv": "NOT_RUN"}), {"holds": True}, CTX)
    assert (r["status"], r["role"]) == ("LOCALIZED", "sv") and r["not_run"] == ["I3:sv"]


def test_no_conclusion_without_x():
    for holds in (False, None):
        r = L._conclude(_out({"sv": "TRUE"}), {"holds": holds}, CTX)
        assert r["status"] == "NOT_APPLICABLE" and r["holding"] == {"sv": ["I1"]}


# ---- reclassify_failure ----------------------------------------------------------------------

def test_loadgen_connection_failures_are_never_overload():
    for reason in ("conn_refused", "conn_fail", "conn_reset"):
        for hint in ("overload", "unclassified", "non_overload"):
            f = {"reason": reason, "class_hint": hint}
            assert X.reclassify_failure("ld", f, "load_source") == ("non_overload",
                                                                    "loadgen_connection")
            assert X.reclassify_failure("ld", f, None)[1] == "loadgen_connection"
    f = {"reason": "timeout_registered", "class_hint": "overload"}
    assert X.reclassify_failure("ld", f, "load_source") == ("overload", "registered_overload")
    f = {"reason": "http_503", "class_hint": "unclassified"}
    assert X.reclassify_failure("ld", f, "load_source") == ("non_overload", "unclassified")
    # a stub's own connection reason is not the load generator's
    f = {"reason": "conn_reset", "class_hint": "overload"}
    assert X.reclassify_failure("sink", f, "stub_out_of_domain") == ("overload",
                                                                   "registered_overload")


def test_failures_by_class_counts_categories():
    td = NS(summaries={"ld": {0: {"failures": [
        {"reason": "conn_fail", "count": 2, "class_hint": "overload"},
        {"reason": "timeout_registered", "count": 3, "class_hint": "overload"}]}}})
    f = X.failures_by_class(td, {"ld": {"measure_class": "load_source"}})
    assert (f["overload"], f["non_overload"]) == (3, 2)
    assert f["by_category"] == {"loadgen_connection": 2, "registered_overload": 3}
