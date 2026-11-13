# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end launches with fake tools (functional only; no performance meaning)."""

import hashlib
import json
import os
import stat

import conftest as cf
from capcommon.runtree import read_state
from caplaunch import procfs

TIER_FILES = ("000_warmup-0", "001_drain-1", "002_up-2", "003_drain-3", "004_down-4",
              "005_drain-5")


def _gate():
    return cf.gate([("nullsse", "backlog.active_streams"), ("capload", "backlog.inflight")])


def _launch(tmp_path, roles, tdoc=None, lead_ms=2500, overlay=None, shm=None, job_env_dir=None):
    wd = cf.make_workdir(tmp_path, tdoc or cf.tier_doc(_gate()), roles)
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


def test_tool_runtime_helper_is_attributed_not_unattributed(tmp_path):
    """multiprocessing's resource tracker started by a tool role (here a stub) is an aux_helper
    of that role; a non-whitelisted child of the same role stays unattributed."""
    roles = cf.default_roles(server_args=["--drain-delay-ms", "150", "--mp-helper",
                                          "--plain-child"],
                             client_args=["--drain-delay-ms", "100"], with_observers=False)
    r, T = _launch(tmp_path, roles)
    assert r.returncode == 0, r.stderr
    rj = _j(T, "raw", "procs", "roles.json")
    aux = rj["aux_helpers"]
    assert len(aux) == 1, aux
    nullsse0 = [p["pid"] for p in rj["procs"] if p["role"] == "nullsse" and p["worker"] == 0
                and p["discovered_via"] == "launched"][0]
    assert aux[0]["role"] == "nullsse" and aux[0]["rule"] == "mp_resource_tracker"
    assert aux[0]["category"] == "aux_helper" and aux[0]["ppid"] == nullsse0
    assert "multiprocessing.resource_tracker" in aux[0]["cmdline"][-1]
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


def test_outer_capprobe_tier_conflict_fails_before_anything_starts(tmp_path):
    """An outer CAPPROBE_TIER that contradicts the scenario (P0 here) is a config error found
    before the launch tree is written or any role starts, not halfway through the role starts
    (which left launch.json without STATE / launch_end.json)."""
    roles = cf.default_roles(with_observers=False)
    wd = cf.make_workdir(tmp_path, cf.tier_doc(_gate()), roles)
    T = os.path.join(str(tmp_path), "T")
    env = cf._env()
    env["CAPPROBE_TIER"] = "P2"
    r = cf.subprocess.run(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm")), env=env,
                          capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "CAPPROBE_TIER" in r.stderr, (r.returncode, r.stderr)
    assert not os.path.exists(os.path.join(T, "raw", "launch.json"))


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
