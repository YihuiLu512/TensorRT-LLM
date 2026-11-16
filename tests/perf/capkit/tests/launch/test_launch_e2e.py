# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end launches with fake tools (functional only; no performance meaning)."""

import hashlib
import json
import os
import stat

import pytest

import conftest as cf
from capcommon.runtree import read_state
from capcommon.snapshot import read_snapshot_file
from caplaunch import procfs

TIER_FILES = ("000_warmup-0", "001_drain-1", "002_up-2", "003_drain-3", "004_down-4",
              "005_drain-5")


def _gate():
    return cf.gate([("nullsse", "backlog.active_streams"), ("capload", "backlog.inflight")])


def _launch(tmp_path, roles, tdoc=None, lead_ms=2500, overlay=None, shm=None, job_env_dir=None,
            probe_tier="P0"):
    wd = cf.make_workdir(tmp_path, tdoc or cf.tier_doc(_gate()), roles, probe_tier=probe_tier)
    T = os.path.join(str(tmp_path), "T")
    r = cf.run_launch(cf.launch_argv(wd, T, shm or os.path.join(str(tmp_path), "shm"),
                                     lead_ms=lead_ms, overlay=overlay, job_env_dir=job_env_dir))
    return r, T


def _j(T, *p):
    with open(os.path.join(T, *p)) as f:
        return json.load(f)


def rj_unattributed(T):
    return _j(T, "raw", "procs", "roles.json")["unattributed"]


def _no_leftovers(T):
    """Independent re-check: no live process carries this launch's marker env entry."""
    mark = _j(T, "raw", "launch.json")["launch_mark"]
    assert mark.startswith("CAPKIT_LAUNCH_MARK=")
    assert procfs.marked_pids(mark) == []
    end = _j(T, "raw", "launch_end.json")
    assert end["leftovers"]["after"] == []


def test_full_launch_passes(tmp_path):
    roles = cf.default_roles(server_args=["--drain-delay-ms", "150"],
                             client_args=["--drain-delay-ms", "100"])
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 0, r.stderr
    st = read_state(T)
    assert [s[1] for s in st] == ["VALID"] and st[-1][2] == "caplaunch"
    for n in ("launch.json", "launch_end.json", "dims.json"):
        assert os.path.isfile(os.path.join(T, "raw", n))
    L = _j(T, "raw", "launch.json")
    assert L["identity"]["arm"] == "arm-a" and L["identity"]["role"] == "launch"
    assert L["tier_table"]["sha256"] and L["register"]["sha256"] == L["identity"]["register_sha"]
    assert L["code"]["capkit_sha"] == cf.ZERO_SHA and L["image"]["sha256"] is None
    assert {r["role"]: r["proc_kinds"] for r in L["roles"]} == {r["role"]: [] for r in roles}
    assert L["origin_ns"] > L["written_mono_ns"]  # origin chosen before any process started
    roles_json = _j(T, "raw", "procs", "roles.json")
    by = {(p["role"], p["worker"]): p for p in roles_json["procs"]}
    assert set(by) == {("nullsse", 0), ("nullsse", 1), ("product", 0), ("capload", 0),
                       ("sidecar", 0), ("canary", 0)}
    assert by[("nullsse", 1)]["discovered_via"] == "ready.children"
    assert by[("nullsse", 0)]["discovered_via"] == "launched"
    pins = _j(T, "raw", "env", "pin_table.json")["roles"]
    assert procfs.expand_cpu_list(by[("product", 0)]["cpus_readback"]) == pins["product"]["cpus"]
    end = _j(T, "raw", "launch_end.json")
    assert end["tiers_completed"] == 5 and end["stop"]["reason"] == "completed"
    assert all(v < 0 for v in end["ready_at_rel_ns"].values())
    assert [g["passed"] for g in end["gate_outcomes"]] == [True, True, True]
    assert set(end["exit_codes"].values()) == {0}
    for d in TIER_FILES:
        t = _j(T, "raw", "tiers", d, "tier.json")
        assert t["pids_stable"] is True and t["incomplete"] is None
        miss = [c for c in t["snapshot_copies"] if c.get("snapshot_missing")]
        assert miss == [] and len(t["snapshot_copies"]) == 2 * 3  # 3 snapshot procs x 2 ends
        assert all(c["offset_from_boundary_ns"] >= 0 for c in t["snapshot_copies"])
        assert t["identity"]["tier_label"] == d.split("_", 1)[1]
    g = _j(T, "raw", "tiers", "003_drain-3", "drain_gate.json")
    assert g["passed"] and g["n_polls"] == len(g["polls"]) <= g["n_planned_polls"]
    assert g["polls"][0]["values"]["nullsse_active_streams"] > 0  # the gate saw real backlog
    assert os.path.isfile(os.path.join(T, "raw", "procs", "threads",
                                       "product.%d.json" % by[("product", 0)]["pid"]))
    dims = _j(T, "raw", "dims.json")
    assert dims["hw_profile_sha"] is None and "env_vars" in dims
    # K2: the job-level overlay manifest is in the tree verbatim, with its sha in launch.json
    with open(os.path.join(str(tmp_path), "W", "overlay_manifest.json"), "rb") as f:
        src = f.read()
    with open(os.path.join(T, "raw", "env", "overlay_manifest.json"), "rb") as f:
        assert f.read() == src
    ov = L["overlay_manifest"]
    assert ov["present"] is True and ov["sha256"] == hashlib.sha256(src).hexdigest()
    assert ov["needed_by_roles"] == ["product"]
    assert _j(T, "raw", "env", "gpu_idle.json") == {"schema": "test.gpu_idle/0"}  # job env/
    assert rj_unattributed(T) == []
    # RS20: the orchestrator is the writer of CAPKIT_CODE_SHA (= --capkit-sha) for every role;
    # the outer environment here has none
    vdir = os.path.join(T, "raw", "env", "versions")
    recs = [_j(vdir, n)["versions"] for n in sorted(os.listdir(vdir))]
    assert recs and {v["capkit_code_sha"] for v in recs} == {cf.ZERO_SHA}
    mc = _j(T, "raw", "procs", "monitor_children.json")
    assert mc["monitor_classes"] == ["load_source", "observer"] and mc["n_proc_scans"] > 0
    assert os.path.isfile(os.path.join(T, "raw", "sidecar", "generations.json"))
    _no_leftovers(T)


def test_gate_timeout_stops_launch(tmp_path):
    roles = cf.default_roles(server_args=["--never-drain"])
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 5, r.stderr
    st = read_state(T)
    assert st[-1][1] == "VALID" and "drain_gate_timeout" in st[-1][3]
    stop = _j(T, "control", "stop.json")
    assert stop["reason"] == "drain_gate_timeout" and stop["after_tier_seq"] == 0
    g = _j(T, "raw", "tiers", "001_drain-1", "drain_gate.json")
    assert g["passed"] is False and g["action"] == "stopped"
    assert g["n_polls"] == g["n_planned_polls"] > 0
    assert not os.path.exists(os.path.join(T, "raw", "tiers", "002_up-2", "tier.json"))
    end = _j(T, "raw", "launch_end.json")
    assert end["gate_outcomes"] == [{"seq": 1, "passed": False, "t_pass_rel_ns": None,
                                     "action": "stopped"}]
    _no_leftovers(T)


def test_record_only_gate_continues(tmp_path):
    roles = cf.default_roles(server_args=["--never-drain"], with_observers=False)
    gcfg = _gate()
    gcfg["mode"] = "record_only"
    r, T = _launch(tmp_path, roles, tdoc=cf.tier_doc(gcfg))
    assert r.returncode == 0, r.stderr
    end = _j(T, "raw", "launch_end.json")
    assert [g["action"] for g in end["gate_outcomes"]] == ["recorded_only"] * 3
    assert end["tiers_completed"] == 5
    _no_leftovers(T)


def test_not_ready_before_origin(tmp_path):
    roles = cf.default_roles(product_args=["--delay-ready-s", "4"], with_observers=False)
    r, T = _launch(tmp_path, roles, lead_ms=1500)
    assert r.returncode == 5, r.stderr
    st = read_state(T)
    assert st[-1][1] == "NOT_RUN" and "not ready before origin" in st[-1][3]
    assert _j(T, "control", "stop.json")["reason"] == "not_ready_before_origin"
    end = _j(T, "raw", "launch_end.json")
    assert end["ready_at_rel_ns"]["product"] is None and end["tiers_completed"] is None
    assert not os.path.exists(os.path.join(T, "raw", "tiers", "000_warmup-0", "tier.json"))
    _no_leftovers(T)


def test_early_exit_is_void_and_children_are_reaped(tmp_path):
    roles = cf.default_roles(server_args=["--crash-at-rel-ms", "900"], with_observers=False)
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 5, r.stderr
    st = read_state(T)
    assert st[-1][1] == "VOID" and "nullsse" in st[-1][3]
    end = _j(T, "raw", "launch_end.json")
    assert end["early_exit"]["role"] == "nullsse" and end["exit_codes"]["nullsse"] == 9
    # the crashed parent's worker child outlived it; the orchestrator killed it
    assert any(x["why"] == "registered:nullsse" for x in end["leftovers"]["before"])
    _no_leftovers(T)


def test_orphan_found_by_marker_and_killed(tmp_path):
    roles = cf.default_roles(product_args=["--orphan"], with_observers=False)
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 0, r.stderr
    end = _j(T, "raw", "launch_end.json")
    assert [x["why"] for x in end["leftovers"]["before"]] == ["launch_mark"]
    _no_leftovers(T)


def test_sigterm_ignored_falls_back_to_sigkill(tmp_path):
    roles = cf.default_roles(product_args=["--ignore-term"], with_observers=False)
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 0, r.stderr
    end = _j(T, "raw", "launch_end.json")
    assert "product" in end["forced_kills"] and end["exit_codes"]["product"] == -9
    _no_leftovers(T)


def test_probe_children_registered_by_snapshot(tmp_path):
    roles = cf.default_roles(product_args=["--probe-child", "--spawn-child"],
                             with_observers=False)
    for rr in roles:
        if rr["role"] == "product":
            rr["expected_procs"] = 3
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 0, r.stderr
    rj = _j(T, "raw", "procs", "roles.json")
    prod = [p for p in rj["procs"] if p["role"] == "product"]
    assert sorted(p["discovered_via"] for p in prod) == ["launched", "probe.snapshot"]
    assert [u["seen_via"] for u in rj["unattributed"]] == ["proc.children"]
    ident = _j(T, "raw", "procs", "identity", "product.json")
    assert ident["schema"] == "capkit.identity/1" and ident["identity"]["pid"] is None
    _no_leftovers(T)


def test_product_role_gets_an_entry_rule_for_its_own_command_line(tmp_path, monkeypatch):
    """R1b (P3): the product role's CAPPROBE_ENTRY matches the process the orchestrator started
    and no spawn child of it; an inherited outer value is replaced; launch.json and roles.json
    record the rule."""
    rep = tmp_path / "entry"
    rep.mkdir()
    roles = cf.default_roles(product_args=["--entry-report", str(rep)], with_observers=False)
    monkeypatch.setenv("CAPPROBE_ENTRY", "outer-value-must-not-reach-any-role")
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 0, r.stderr
    main, spawn = (json.loads((rep / n).read_text()) for n in ("main.json", "spawn.json"))
    assert main["rule"] is not None and main["match"] is not None, main
    assert main["rule"]["program"] == os.path.realpath(cf.sys.executable)
    assert main["rule"]["args"][0] == cf.DUMMY and "--entry-report" in main["rule"]["args"]
    assert spawn["rule"] == main["rule"] and spawn["match"] is None, spawn  # inherited, no match
    rj = _j(T, "raw", "procs", "roles.json")
    launched = [p for p in rj["procs"] if p["role"] == "product"
                and p["discovered_via"] == "launched"]
    assert len(launched) == 1 and launched[0]["probe_entry"] == main["rule"]
    lj = _j(T, "raw", "launch.json")["probe_entry"]
    assert lj["env"] == "CAPPROBE_ENTRY" and lj["schema"] == "capkit.probe_entry/1"
    assert lj["roles"] == ["product"]  # names only: the rule is built once, at the role's start
    assert lj["exact_rule_in"] == "procs/roles.json procs[].probe_entry"
    _no_leftovers(T)


def test_entry_rule_recorded_is_the_injected_one_from_the_first_roles_write(tmp_path,
                                                                            monkeypatch):
    """The CAPPROBE_ENTRY rule is computed once per start: the rule the product process got in
    its environment is the one in the start event, and every roles.json write that lists the
    launched product process already carries it (no write without it). The product's argv has
    an {addr:...} placeholder, resolved only at start."""
    rep = tmp_path / "entry"
    rep.mkdir()
    writes = tmp_path / "roles_writes.jsonl"
    roles = cf.default_roles(product_args=["--entry-report", str(rep)], with_observers=False)
    for rr in roles:
        if rr["role"] == "product":
            rr["start_after"] = ["nullsse"]
            rr["args"] += ["--peer-label", "{addr:nullsse}"]
    wd = cf.make_workdir(tmp_path, cf.tier_doc(_gate()), roles)
    T = os.path.join(str(tmp_path), "T")
    argv = cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm"), lead_ms=2500)
    assert argv[1:3] == ["-m", "caplaunch"]
    argv[1:3] = [os.path.join(cf.HERE, "fakes", "hooked_launch.py")]
    monkeypatch.setenv("HOOKED_LAUNCH_ROLES_WRITES", str(writes))
    r = cf.run_launch(argv)
    assert r.returncode == 0, r.stderr
    main = json.loads((rep / "main.json").read_text())
    rule = main["rule"]
    assert rule is not None and main["match"] is not None, main
    assert "{addr:" not in " ".join(rule["args"]) and "--peer-label" in rule["args"]
    end = _j(T, "raw", "launch_end.json")
    starts = [e for e in end["events"] if e["event"] == "start" and e["role"] == "product"]
    assert len(starts) == 1 and starts[0]["probe_entry"] == rule
    seen = []
    for line in writes.read_text().splitlines():
        procs = json.loads(line)["procs"]
        seen += [p for p in procs if p["role"] == "product" and p["discovered_via"] == "launched"]
    assert seen and all(p.get("probe_entry") == rule for p in seen), seen
    _no_leftovers(T)


INJ_DELTA_NS = 200_000
LOAD_TIERS = ("000_warmup-0", "002_up-2", "004_down-4")


def _p1_overlay(tmp_path):
    p = os.path.join(str(tmp_path), "overlay_p1.json")
    with open(p, "w") as f:
        json.dump({"schema": "capkit.overlay/1", "tier": "P1", "groups": [], "files": [],
                   "capkit_packages": [], "capkit_code_sha": cf.ZERO_SHA,
                   "target_commit": "d" * 40, "sites": {}}, f)
    return p


@pytest.mark.parametrize("inject", [True, False], ids=["registered", "absent"])
def test_product_side_injection_reaches_capprobe_with_heartbeat_in_every_window(tmp_path,
                                                                                inject):
    """FT1-18 end to end with the real orchestrator and the real capprobe in a product-like
    process passing F2 and F4: the launch_inject entry naming the product role is the process's
    injection (launch.json probe_inject, snapshot meta), the process writes snapshots, the
    orchestrator copies them at every window edge, and the heartbeat slots show a busy-wait at
    F2 on every pass (applied == hits, applied_ns >= applied x delta) and passes without a spec
    at F4. Absent (negative control): same P1 launch without the entry, no snapshot at all."""
    roles = cf.default_roles(product_args=["--probe-sites", "F2,F4", "--probe-every-ms", "5"],
                             with_observers=False)
    for rr in roles:
        if rr["role"] == "product":
            rr.update(probe_tier="P1", snapshots="all", proc_kinds=["frontend_openengine"])
    tdoc = cf.tier_doc(_gate())
    spec = {"role": "product", "site": "F2", "mode": "busy", "delta_ns": INJ_DELTA_NS}
    tdoc["launch_inject"] = [spec] if inject else []
    shm = os.path.join(str(tmp_path), "shm")
    r, T = _launch(tmp_path, roles, tdoc=tdoc, overlay=_p1_overlay(tmp_path), shm=shm,
                   probe_tier="P1")
    assert r.returncode == 0, r.stderr[-3000:]
    assert [s[1] for s in read_state(T)] == ["VALID"]
    L = _j(T, "raw", "launch.json")
    want = [{"site": "F2", "mode": "busy", "every_n": 1, "phase": 0, "where": "inline",
             "delta_ns": INJ_DELTA_NS, "role": "product"}]
    assert L["probe_inject"] == ({"product": want} if inject else {})
    [prod] = [p for p in _j(T, "raw", "procs", "roles.json")["procs"] if p["role"] == "product"]
    pid = prod["pid"]
    shm_files = [n for _d, _s, fs in os.walk(shm) for n in fs if n == "%d.bin" % pid]
    for d in TIER_FILES:
        t = _j(T, "raw", "tiers", d, "tier.json")
        assert [c for c in t["snapshot_copies"] if c.get("snapshot_missing")] == []
        mine = sorted(c["which"] for c in t["snapshot_copies"] if c["pid"] == pid)
        assert mine == (["wend", "wstart"] if inject else []), (d, mine)
    if not inject:
        assert shm_files == []
        _no_leftovers(T)
        return
    assert len(shm_files) == 1
    for d in LOAD_TIERS:
        sd = os.path.join(T, "raw", "tiers", d, "snapshots")
        a, b = (read_snapshot_file(os.path.join(sd, "product.%d.%s.bin" % (pid, w)))
                for w in ("wstart", "wend"))
        assert b.meta["probe"]["inject"] == [{k: v for k, v in want[0].items() if k != "role"}]
        assert b.meta["probe"]["tier"] == "P1" and b.meta["probe"]["groups"] == []
        dv = {k: b.values[k] - a.values[k] for k in b.values if k.startswith("inject.")}
        assert dv["inject.F2.hits"] > 10, (d, dv)
        assert dv["inject.F2.applied"] == dv["inject.F2.hits"], (d, dv)
        assert dv["inject.F2.applied_ns"] >= dv["inject.F2.applied"] * INJ_DELTA_NS, (d, dv)
        assert dv["inject.F4.hits"] > 10 and dv["inject.F4.applied"] == 0, (d, dv)
        assert not any(k.startswith("count.") for k in b.values)  # P1: no probe group
    _no_leftovers(T)


def test_overlay_missing_with_probe_role_is_not_run(tmp_path):
    """K2: a probe-tier role and no overlay manifest -> NOT_RUN before any role starts."""
    roles = cf.default_roles(with_observers=False)
    r, T = _launch(tmp_path, roles, overlay=os.path.join(str(tmp_path), "absent.json"))
    assert r.returncode == 5, r.stderr
    st = read_state(T)
    assert st[-1][1] == "NOT_RUN" and "overlay manifest" in st[-1][3]
    assert _j(T, "control", "stop.json")["reason"] == "overlay_manifest_missing"
    end = _j(T, "raw", "launch_end.json")
    assert set(end["exit_codes"].values()) == {None}  # nothing was started
    assert not os.path.exists(os.path.join(T, "raw", "env", "overlay_manifest.json"))
    assert _j(T, "raw", "launch.json")["overlay_manifest"]["present"] is False
    _no_leftovers(T)


def test_overlay_missing_without_probe_role_runs(tmp_path):
    """K2: no role carries a probe tier -> a missing manifest is only recorded."""
    roles = [rr for rr in cf.default_roles(client_args=["--drain-delay-ms", "100"],
                                           with_observers=False) if rr["role"] != "product"]
    roles[-1]["args"] = [x for x in roles[-1]["args"] if x not in ("--peer", "{addr:product}")]
    roles[-1]["start_after"] = ["nullsse"]
    r, T = _launch(tmp_path, roles, overlay=os.path.join(str(tmp_path), "absent.json"))
    assert r.returncode == 0, r.stderr
    ov = _j(T, "raw", "launch.json")["overlay_manifest"]
    assert ov == {"source": os.path.join(str(tmp_path), "absent.json"), "present": False,
                  "sha256": None, "needed_by_roles": []}
    _no_leftovers(T)


def _sidecar_args(roles, extra):
    for rr in roles:
        if rr["role"] == "sidecar":
            rr["args"] = rr["args"] + extra


def test_monitor_children_do_not_break_pid_stability(tmp_path):
    """K3: an observer's short-lived children (nvidia-smi-like) are audited, not compared."""
    roles = cf.default_roles(server_args=["--drain-delay-ms", "150"],
                             client_args=["--drain-delay-ms", "100"])
    _sidecar_args(roles, ["--churn-every-ms", "30"])
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 0, r.stderr
    for d in TIER_FILES:
        t = _j(T, "raw", "tiers", d, "tier.json")
        assert t["pids_stable"] is True, (d, t["scans"])
        assert any(m[2] == "sidecar" for m in t["scans"]["wstart"]["monitor_descendants"]
                   + t["scans"]["wend"]["monitor_descendants"])
    assert rj_unattributed(T) == []
    kids = _j(T, "raw", "procs", "monitor_children.json")["children"]
    reg = {(p["pid"], p["starttime_ticks"]) for p in _j(T, "raw", "procs", "roles.json")["procs"]}
    assert not [k for k in kids if (k["pid"], k["starttime_ticks"]) in reg]
    sc = [k for k in kids if k["role"] == "sidecar"]
    assert len(sc) >= 10 and all(k["measure_class"] == "observer" for k in sc)
    nv = os.path.realpath(os.path.join(str(tmp_path), "W", "fake-nvidia-smi"))
    assert all(k["is_nvidia_smi"] in (True, None) for k in sc)  # None: exited before readlink
    assert sum(1 for k in sc if k["is_nvidia_smi"]) >= 5
    assert _j(T, "raw", "launch.json")["nvidia_smi"]["realpath"] == nv
    sidecar_pid = [p["pid"] for p in _j(T, "raw", "procs", "roles.json")["procs"]
                   if p["role"] == "sidecar"][0]
    assert all(k["ppid"] == sidecar_pid and k["first_seen_mono_ns"] <= k["last_seen_mono_ns"]
               and k["n_seen"] >= 1 for k in sc)
    _no_leftovers(T)


def test_under_test_children_still_break_pid_stability(tmp_path):
    """K3: the same churn below an under_test role keeps P-PIDS red and C13's unattributed."""
    roles = cf.default_roles(server_args=["--drain-delay-ms", "150"],
                             client_args=["--drain-delay-ms", "100"],
                             product_args=["--churn-every-ms", "30"], with_observers=False)
    for rr in roles:
        if rr["role"] == "product":
            rr["measure_class"] = "under_test"
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 0, r.stderr
    stable = [_j(T, "raw", "tiers", d, "tier.json")["pids_stable"] for d in TIER_FILES]
    assert stable.count(False) >= 4, stable
    un = rj_unattributed(T)
    assert len(un) >= 5 and {u["seen_via"] for u in un} == {"proc.children"}
    assert _j(T, "raw", "procs", "monitor_children.json")["children"] == []
    _no_leftovers(T)


def test_tool_role_snapshot_child_is_unattributed(tmp_path):
    """K27: an unreported snapshot writer below a capkit tool (here load_source) is listed as
    unattributed with its snapshot evidence, never dropped and never registered."""
    roles = cf.default_roles(server_args=["--drain-delay-ms", "150"],
                             client_args=["--drain-delay-ms", "100", "--rogue-snapshot-child"],
                             with_observers=False)
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 0, r.stderr
    rj = _j(T, "raw", "procs", "roles.json")
    capload = [p for p in rj["procs"] if p["role"] == "capload"]
    assert [p["discovered_via"] for p in capload] == ["launched"]
    un = rj["unattributed"]
    assert len(un) == 1, un
    assert un[0]["seen_via"] == "probe.snapshot" and un[0]["role_hint"] == "rogue"
    assert un[0]["registered_ancestor"] == {"pid": capload[0]["pid"], "role": "capload"}
    _no_leftovers(T)


def test_shm_dir_not_private_is_refused(tmp_path):
    """K28: a group/other-writable --shm-dir is refused before anything starts (exit 3)."""
    shm = os.path.join(str(tmp_path), "shm")
    os.mkdir(shm)
    os.chmod(shm, 0o777)
    r, T = _launch(tmp_path, cf.default_roles(with_observers=False), shm=shm)
    assert r.returncode == 3 and "must not be writable by group or others" in r.stderr, r.stderr
    assert not os.path.exists(os.path.join(T, "raw", "launch.json"))


def test_shm_subdir_symlink_is_refused(tmp_path):
    """K28: a pre-created r<restart>-L<seq> that is a symlink is never followed."""
    shm = os.path.join(str(tmp_path), "shm")
    os.mkdir(shm, 0o700)
    elsewhere = os.path.join(str(tmp_path), "elsewhere")
    os.mkdir(elsewhere, 0o700)
    os.symlink(elsewhere, os.path.join(shm, "r0-L0"))
    r, T = _launch(tmp_path, cf.default_roles(with_observers=False), shm=shm)
    assert r.returncode == 3 and "symlinks are refused" in r.stderr, r.stderr
    assert os.listdir(elsewhere) == []
    assert stat.S_ISLNK(os.lstat(os.path.join(shm, "r0-L0")).st_mode)


def test_job_env_dir_missing_is_not_run(tmp_path):
    """--job-env-dir must exist: without it gpu_idle.json and the gate results never reach the
    tree, so the launch stops before starting any process."""
    roles = cf.default_roles(with_observers=False)
    r, T = _launch(tmp_path, roles, job_env_dir=os.path.join(str(tmp_path), "no-env"))
    assert r.returncode == 5, r.stderr
    st = read_state(T)
    assert st[-1][1] == "NOT_RUN" and "--job-env-dir" in st[-1][3]
    assert _j(T, "control", "stop.json")["reason"] == "job_env_dir_missing"
    assert set(_j(T, "raw", "launch_end.json")["exit_codes"].values()) == {None}
    _no_leftovers(T)


@pytest.mark.parametrize("extra_env", [
    {},
    {"PYTHONDONTWRITEBYTECODE": "1"},  # what jobs/capkit_inner.sh exports
    {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONOPTIMIZE": "1", "PYTHONNOUSERSITE": "1"},
], ids=["plain", "job-env", "more-flags"])
def test_tool_runtime_helper_is_attributed_not_unattributed(tmp_path, extra_env):
    """multiprocessing's resource tracker started by a tool role (here a stub) is an aux_helper
    of that role, also when interpreter flags from the environment put options in front of its
    `-c` (errata R5-1); a non-whitelisted child of the same role stays unattributed."""
    roles = cf.default_roles(server_args=["--drain-delay-ms", "150", "--mp-helper",
                                          "--plain-child"],
                             client_args=["--drain-delay-ms", "100"], with_observers=False)
    wd = cf.make_workdir(tmp_path, cf.tier_doc(_gate()), roles)
    T = os.path.join(str(tmp_path), "T")
    env = cf._env()
    env.update(extra_env)
    r = cf.subprocess.run(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm")), env=env,
                          capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rj = _j(T, "raw", "procs", "roles.json")
    aux = rj["aux_helpers"]
    assert len(aux) == 1, aux
    nullsse0 = [p["pid"] for p in rj["procs"] if p["role"] == "nullsse" and p["worker"] == 0
                and p["discovered_via"] == "launched"][0]
    assert aux[0]["role"] == "nullsse" and aux[0]["rule"] == "mp_resource_tracker"
    assert aux[0]["category"] == "aux_helper" and aux[0]["ppid"] == nullsse0
    assert "multiprocessing.resource_tracker" in aux[0]["cmdline"][-1]
    if extra_env:
        assert "-B" in aux[0]["cmdline"][1:-2], aux[0]["cmdline"]
    un = rj["unattributed"]
    assert [u["argv"] for u in un] == [["sleep", "600"]], un
    hp = aux[0]["pid"]
    for d in TIER_FILES:
        t = _j(T, "raw", "tiers", d, "tier.json")
        assert t["pids_stable"] is True
        for w in ("wstart", "wend"):
            sc = t["scans"][w]
            assert [x[0] for x in sc["aux_helpers"]] == [hp]
            assert hp not in [x[0] for x in sc["descendants"]]
    _no_leftovers(T)


@pytest.mark.parametrize("key,val", [
    ("CAPPROBE_TIER", "P2"),
    ("CAPPROBE_INJECT", "site=F2,mode=busy,delta_ns=20000,every_n=1"),
    ("CAPKIT_CODE_SHA", "1" * 40),
])
def test_outer_capprobe_tier_conflict_fails_before_anything_starts(tmp_path, key, val):
    """An outer CAPPROBE_TIER that contradicts the scenario (P0 here), an outer CAPPROBE_INJECT
    (never passed to product roles) or an outer CAPKIT_CODE_SHA other than --capkit-sha is a
    config error found before the launch tree is written or any role starts, not halfway through
    the role starts (which left launch.json without STATE / launch_end.json)."""
    roles = cf.default_roles(with_observers=False)
    wd = cf.make_workdir(tmp_path, cf.tier_doc(_gate()), roles)
    T = os.path.join(str(tmp_path), "T")
    env = cf._env()
    env[key] = val
    r = cf.subprocess.run(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm")), env=env,
                          capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and key in r.stderr, (r.returncode, r.stderr)
    assert not os.path.exists(os.path.join(T, "raw", "launch.json"))


@pytest.mark.parametrize("lag", ["\u00b2", "\u0661", "0", "-1", "1e3", ""],
                         ids=["superscript-2", "arabic-1", "zero", "negative", "float", "empty"])
def test_lag_period_param_must_be_an_ascii_integer(tmp_path, lag):
    """U6: --param lag_period_us is a positive ASCII integer; anything else (including digits
    str.isdigit() accepts but int() rejects, like '\u00b2') is a config error, exit 3, before the
    tree is written."""
    roles = cf.default_roles(with_observers=False)
    wd = cf.make_workdir(tmp_path, cf.tier_doc(_gate()), roles, params_required=["lag_period_us"])
    T = os.path.join(str(tmp_path), "T")
    r = cf.run_launch(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm"),
                                     params=["lag_period_us=%s" % lag]))
    assert r.returncode == 3 and "lag_period_us must be an integer > 0" in r.stderr, (
        r.returncode, r.stderr[-2000:])
    assert not os.path.exists(os.path.join(T, "raw", "launch.json"))


def test_lag_period_param_ascii_integer_is_accepted(tmp_path):
    """Positive control of the test above: the same launch with lag_period_us=1000 runs."""
    roles = cf.default_roles(with_observers=False)
    wd = cf.make_workdir(tmp_path, cf.tier_doc(_gate()), roles, params_required=["lag_period_us"])
    T = os.path.join(str(tmp_path), "T")
    r = cf.run_launch(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm"),
                                     params=["lag_period_us=1000"]))
    assert r.returncode == 0, r.stderr[-2000:]
    assert read_state(T)[-1][1] == "VALID"


def test_tree_with_a_state_line_is_refused(tmp_path):
    """The orchestrator only starts in a fresh tree: an existing STATE (another writer's
    conclusion) is never followed by this launch's line."""
    roles = cf.default_roles(with_observers=False)
    wd = cf.make_workdir(tmp_path, cf.tier_doc(_gate()), roles)
    T = os.path.join(str(tmp_path), "T")
    os.makedirs(T, mode=0o700)
    with open(os.path.join(T, "STATE"), "w") as f:
        f.write("2026-11-13T00:00:00Z\tVOID\tcapjobs.launches\tearlier\n")
    r = cf.run_launch(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm")))
    assert r.returncode == 3 and "STATE" in r.stderr, (r.returncode, r.stderr)
    assert [s[1] for s in read_state(T)] == ["VOID"]
    assert not os.path.exists(os.path.join(T, "raw", "launch.json"))


def _spin_role(**over):
    r = cf.role("spin", "capkit", "aux", asyncio=False, mclass="observer",
                module="capenv.spinners", snapshots="none", start_after=["product"],
                args=["--target-role", "product"])
    r.update(over)
    return r


def _spin_launch(tmp_path, roles, job_pin=None):
    """_launch, with job_pin(pin table bytes) -> bytes written as the job-level
    env/pin_table.json first (None: no such file)."""
    wd = cf.make_workdir(tmp_path, cf.tier_doc(_gate()), roles)
    if job_pin is not None:
        with open(wd["pin"], "rb") as f:
            data = job_pin(f.read())
        with open(os.path.join(wd["job_env_dir"], "pin_table.json"), "wb") as f:
            f.write(data)
    T = os.path.join(str(tmp_path), "T")
    return cf.run_launch(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm"))), T


def test_spinners_aux_role_spins_on_the_target_roles_first_core(tmp_path):
    """SB-B1 / #26 end to end with the real capenv.spinners: the orchestrator passes the launch
    tree's pin table, one spinner runs on the role's own core and one on the product role's first
    core; the spinners are monitor children (not registered, not unattributed), the started
    process has a version record, nothing is left behind. A job-level env/pin_table.json
    identical to --pin-table is accepted (it is the tree's copy)."""
    roles = cf.default_roles(server_args=["--drain-delay-ms", "150"],
                             client_args=["--drain-delay-ms", "100"]) + [_spin_role()]
    r, T = _spin_launch(tmp_path, roles, job_pin=lambda b: b)
    assert r.returncode == 0, r.stderr
    assert [s[1] for s in read_state(T)] == ["VALID"]
    pin_path = os.path.join(T, "raw", "env", "pin_table.json")
    spin = [x for x in _j(T, "raw", "launch.json")["roles"] if x["role"] == "spin"][0]
    assert spin["args"][spin["args"].index("--pin-table") + 1] == pin_path
    pins = _j(T, "raw", "env", "pin_table.json")["roles"]
    out = _j(T, "raw", "spinners", "spin.json")
    assert out["problems"] == [] and out["ready"] is True and out["pin_table"]["path"] == pin_path
    assert out["target_role"] == "product" and out["target_cpu"] == pins["product"]["cpus"][0]
    assert [(x["cpu"], x["purpose"], x["affinity"], x["reported"]) for x in out["spinners"]] == [
        (pins["spin"]["cpus"][0], "own", pins["spin"]["cpus"], True),
        (pins["product"]["cpus"][0], "target", [pins["product"]["cpus"][0]], True)]
    assert out["stop"]["reason"] == "completed"
    rj = _j(T, "raw", "procs", "roles.json")
    me = [p for p in rj["procs"] if p["role"] == "spin"]
    assert [p["discovered_via"] for p in me] == ["launched"]
    assert rj["unattributed"] == []
    kids = {k["pid"] for k in _j(T, "raw", "procs", "monitor_children.json")["children"]
            if k["role"] == "spin" and k["ppid"] == me[0]["pid"]}
    assert kids == {x["pid"] for x in out["spinners"]}
    assert os.path.isfile(os.path.join(T, "raw", "env", "versions",
                                       "spin.w0.%d.json" % me[0]["pid"]))
    assert _j(T, "raw", "launch_end.json")["exit_codes"]["spin"] == 0
    for d in TIER_FILES:
        assert _j(T, "raw", "tiers", d, "tier.json")["pids_stable"] is True
    _no_leftovers(T)


@pytest.mark.parametrize("over,needle", [
    ({"measure_class": "stub_out_of_domain"}, "measure_class"),
    ({"start_after": []}, "needs product in start_after"),
])
def test_spinners_role_against_the_scenario_rules_is_refused_before_any_start(tmp_path, over,
                                                                              needle):
    """A spinners role outside the fixed role fields (errata SB-B1-3; its unregistered spin
    processes would be unattributed below a non-monitor class) or with a target role it does
    not start after is a config error: exit 3 before the launch tree is written."""
    roles = cf.default_roles(with_observers=False) + [_spin_role(**over)]
    r, T = _spin_launch(tmp_path, roles)
    assert r.returncode == 3 and needle in r.stderr, (r.returncode, r.stderr[-2000:])
    assert not os.path.exists(os.path.join(T, "raw", "launch.json"))
    assert not os.path.exists(os.path.join(T, "raw", "spinners"))


def test_job_level_pin_table_differing_from_pin_table_is_refused(tmp_path):
    """A job-level env/pin_table.json would become the tree's raw/env copy (read by
    capenv.spinners and the judge): one that differs from --pin-table is a config error, exit 3,
    before the launch tree is written or any role starts."""
    roles = cf.default_roles(with_observers=False) + [_spin_role()]
    r, T = _spin_launch(tmp_path, roles, job_pin=lambda b: b + b"\n")
    assert r.returncode == 3 and "differs from --pin-table" in r.stderr, (
        r.returncode, r.stderr[-2000:])
    assert not os.path.exists(os.path.join(T, "raw", "launch.json"))
