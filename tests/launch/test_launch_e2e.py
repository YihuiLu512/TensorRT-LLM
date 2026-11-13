# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end launches with fake tools (functional only; no performance meaning)."""

import json
import os

import conftest as cf
from capcommon.runtree import read_state
from caplaunch import procfs

TIER_FILES = ("000_warmup-0", "001_drain-1", "002_up-2", "003_drain-3", "004_down-4",
              "005_drain-5")


def _gate():
    return cf.gate([("nullsse", "backlog.active_streams"), ("capload", "backlog.inflight")])


def _launch(tmp_path, roles, tdoc=None, lead_ms=2500):
    wd = cf.make_workdir(tmp_path, tdoc or cf.tier_doc(_gate()), roles)
    T = os.path.join(str(tmp_path), "T")
    r = cf.run_launch(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm"),
                                     lead_ms=lead_ms))
    return r, T


def _j(T, *p):
    with open(os.path.join(T, *p)) as f:
        return json.load(f)


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
