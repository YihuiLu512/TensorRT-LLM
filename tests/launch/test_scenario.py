# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario validation and argv building (no processes)."""

import copy
import json
import os

import pytest

import conftest as cf
from capcommon.exitcodes import ConfigError
from capcommon.identity import Identity
from capcommon.tiers import TierTable
from caplaunch import scenario as sc

CONFIGS = os.path.join(cf.C, "configs", "e0")


def _doc(roles=None):
    roles = roles or cf.default_roles()
    names = [r["role"] for r in roles]
    return {"schema": sc.SCHEMA, "experiment": "E0", "name": "t", "tier_table": "a/t.json",
            "schedule": "a/s.capbin", "prompts": "a/p.json", "probe_tier": "P0",
            "probe_groups": [], "stop_grace_s": 1.0, "stop_order": names,
            "reconcile": {"schema": "capkit.reconcile/1", "chains": [
                {"name": "r", "kind": "requests", "gate": None, "include_usage": None,
                 "hops": [{"role": "capload", "source": "summary.counts.ok", "scope": "tier"}],
                 "corrections": []}]},
            "roles": roles, "params_required": [],
            "dims": {k: None for k in sc.DIMS_KEYS}}


def test_valid_default_scenario():
    roles = sc.validate_scenario(_doc())
    order = sc.start_order(roles)
    assert order.index("nullsse") < order.index("capload")
    assert order.index("product") < order.index("capload")


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["roles"][2]["args"].append("--workers"), "orchestrator-managed"),
    (lambda d: d["roles"][2]["args"].append("{addr:sidecar}"), "start_after"),
    (lambda d: d["roles"][2]["args"].append("{bogus}"), "unknown placeholder"),
    (lambda d: d["roles"][2]["args"].append("{param:tok}"), "params_required"),
    (lambda d: d["stop_order"].pop(), "stop_order"),
    (lambda d: d["roles"][0].update(snapshots="none"), "snapshots"),
    (lambda d: d["roles"][1].update(probe_tier="P2"), "probe_tier"),
    (lambda d: d["roles"][1].update(readiness=[{"kind": "capkit_ready"}]), "capkit_ready"),
    (lambda d: d["roles"][0].update(start_after=["capload"]), "cycle"),
    (lambda d: d["roles"][0].update(expected_procs=1), "expected_procs"),
    (lambda d: d["roles"][0].pop("measure_class"), "missing"),
    (lambda d: d["reconcile"]["chains"][0]["hops"][0].update(role="nobody"), "unknown role"),
    (lambda d: d["roles"][1].pop("proc_kinds"), "missing"),
    (lambda d: d["roles"][1].update(proc_kinds="frontend"), "proc_kinds"),
    (lambda d: d["roles"][1].update(proc_kinds=["frontend", "frontend"]), "proc_kinds"),
    (lambda d: d["roles"][2].update(proc_kinds=["frontend"]), "proc_kinds must be []"),
    (lambda d: d["roles"][1].update(proc_kinds=["frontend", "frontend_singel"]),
     "unknown proc_kinds ['frontend_singel']"),
])
def test_invalid_scenarios_rejected(mutate, needle):
    d = _doc()
    mutate(d)
    with pytest.raises(ConfigError) as e:
        sc.validate_scenario(d)
    assert needle in str(e.value)


@pytest.mark.parametrize("role_i", [0, 1, 2], ids=["nullsse", "product", "capload"])
@pytest.mark.parametrize("key", ["CAPPROBE_INJECT", "CAPPROBE_TIER", "CAPPROBE_GROUPS",
                                 "CAPPROBE_LAG_PERIOD_US", "CAPPROBE_ENTRY", "CAPPROBE_X"])
def test_role_env_may_not_set_capprobe_switches(role_i, key):
    """U4 (errata R5-4): a role's env never carries a CAPPROBE_* switch (it would reach the
    process past the orchestrator's checks, e.g. INJECT into a P1 product role); other keys pass."""
    for tier in ("P0", "P1"):
        d = _doc()
        d["probe_tier"] = tier
        d["roles"][1].update(probe_tier=tier)
        d["roles"][role_i]["env"] = {"OTHER": "1", key: "site=F2,mode=busy,delta_ns=1,every_n=1"}
        with pytest.raises(ConfigError) as e:
            sc.validate_scenario(d)
        assert "capprobe switches ['%s']" % key in str(e.value)
        d["roles"][role_i]["env"] = {"OTHER": "1", "XCAPPROBE_INJECT": "1"}
        sc.validate_scenario(d)


@pytest.mark.parametrize("tier,scope,ok", [
    ("P0", "none", True), ("P0", "all", True),
    ("P1", "none", True), ("P1", "all", True), ("P1", "workers", True),
    ("P2", "none", False), ("P2", "workers", False), ("P2", "all", True),
])
def test_snapshot_writing_product_role_must_expect_all_snapshots(tier, scope, ok):
    """S1 / RS19: only a role that writes snapshots (product, P2) is constrained, and to scope
    all: with none or workers a probe that stayed off in its started process would leave nothing
    missing for the judge. P0 / P1 roles write none, so any scope is accepted."""
    d = _doc()
    d["probe_tier"] = tier
    d["probe_groups"] = ["count"] if tier == "P2" else []
    d["roles"][1].update(probe_tier=tier, snapshots=scope)
    if ok:
        sc.validate_scenario(d)
    else:
        with pytest.raises(ConfigError, match="probe_tier P2 writes snapshots; snapshots must "
                                              "be all"):
            sc.validate_scenario(d)


def test_lag_group_needs_the_lag_period_param():
    """S2 / RS20: the orchestrator sets CAPPROBE_LAG_PERIOD_US from the scenario param
    lag_period_us, so a lag group on product roles without that param is refused."""
    d = _doc()
    d["probe_tier"] = "P2"
    d["probe_groups"] = ["count", "lag"]
    d["roles"][1].update(probe_tier="P2", snapshots="all")
    with pytest.raises(ConfigError, match="lag_period_us"):
        sc.validate_scenario(d)
    d["params_required"] = ["lag_period_us"]
    sc.validate_scenario(d)


def _tt(gate_cfg):
    return TierTable.from_doc(cf.tier_doc(gate_cfg))


def _pins(roles):
    out, c = {}, 0
    for r in roles:
        out[r["pin_role"]] = {"cpus": list(range(c, c + r["cpus_needed"])), "numa_node": 0}
        c += r["cpus_needed"]
    return out


def test_cross_check_pass_and_failures():
    d = _doc()
    roles = sc.validate_scenario(d)
    g = cf.gate([("nullsse", "backlog.active_streams")])
    cpus = sc.cross_check(d, roles, _tt(g), _pins(d["roles"]), 40 * cf.MS)
    assert cpus["nullsse"] == [0, 1]
    with pytest.raises(ConfigError, match="poll_ns"):
        sc.cross_check(d, roles, _tt(g), _pins(d["roles"]), 60 * cf.MS)
    bad = cf.gate([("nullsse", "backlog.inflight")])
    with pytest.raises(ConfigError, match="does not declare backlog slot"):
        sc.cross_check(d, roles, _tt(bad), _pins(d["roles"]), 40 * cf.MS)
    pins = _pins(d["roles"])
    pins["product"]["cpus"] = [0]
    with pytest.raises(ConfigError, match="D1"):
        sc.cross_check(d, roles, _tt(g), pins, 40 * cf.MS)
    pins = _pins(d["roles"])
    pins["nullsse"]["cpus"] = [0]
    with pytest.raises(ConfigError, match="cpus_needed"):
        sc.cross_check(d, roles, _tt(g), pins, 40 * cf.MS)


def test_build_argv_capkit_and_product(tmp_path):
    d = _doc()
    roles = sc.validate_scenario(d)
    ident = Identity("E0", "a", 0, "local-x", 0, "a" * 64, "launch", pid=1)
    own = {n: {"ready_file": "/T/r/%s" % n, "port_file": "/T/p/%s.port" % n,
               "report_addr": "/T/p/%s.addr" % n} for n in roles}
    ctx = sc.ArgvContext(ident, "/T", "/T/raw", {}, 1.5, "/shm", 40, "/W/t.json", 123, own,
                         lambda r: "/T/ports/%s.json" % r, lambda r: "127.0.0.1:99",
                         sidecar={"roles_file": "R", "out_dir": "O", "logs_dir": "L",
                                  "cpus_watch": "0-3"}, canary={"out": "C"})
    a = sc.build_argv(roles["capload"], ctx, [3, 4], "py")
    assert a[:3] == ["py", "-m", "capload"]
    assert a[a.index("--role") + 1] == "capload" and a[a.index("--cpus") + 1] == "3,4"
    assert a[a.index("--origin-ns") + 1] == "123" and "--port-file" not in a
    assert a[a.index("--peer") + 1] == "127.0.0.1:99"
    assert sc.build_argv(roles["capload"], ctx, [3, 4], "py", runtime=False)[-1] \
        == "{addr:product}"
    n = sc.build_argv(roles["nullsse"], ctx, [0, 1], "py")
    assert n[n.index("--port-file") + 1] == "/T/ports/nullsse.json"
    assert n[n.index("--workers") + 1] == "2"
    p = sc.build_argv(roles["product"], ctx, [2], "py")
    assert p[-1] == "/T/p/product.port" and "--role" not in p
    s = sc.build_argv(roles["sidecar"], ctx, [5], "py")
    assert s[s.index("--cpus-watch") + 1] == "0-3"
    assert s[s.index("--ready-file") + 1] == "/T/r/sidecar"  # B4 tools take --ready-file
    spl = tmp_path / "fp.json"
    spl.write_text(json.dumps({"sfe_args": ["--vocab-size", "7"]}))
    r2 = copy.deepcopy(roles["product"])
    r2["args"] = [{"splice_json": str(spl), "key": "sfe_args"}]
    assert sc.build_argv(r2, ctx, [2], "py")[-2:] == ["--vocab-size", "7"]


def test_shipped_e0_scenarios_validate():
    names = sorted(n for n in os.listdir(CONFIGS) if n.endswith(".scenario.json"))
    assert len(names) >= 5  # (a) x2 + (b) relay scenarios 1-4
    for n in names:
        with open(os.path.join(CONFIGS, n)) as f:
            sc.validate_scenario(json.load(f))


def test_e0_proc_kinds_are_capprobe_process_kinds():
    """Every proc_kinds value in configs/e0 is a capprobe SITES proc_kind (the judge's C8 picks
    the expected sites by it), and every product role of a probed scenario declares its kinds.
    A new-architecture single frontend (--num_serve_frontends 1) passes the count sites of
    `frontend` and `frontend_openengine` and the lag site of `frontend_single`, and no other."""
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location(
        "_capprobe_sites", os.path.join(cf.TOOLS, "capprobe", "sites.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # sites.py alone: importing capprobe would parse CAPPROBE_*
    vocab = {v["proc_kind"] for v in mod.SITES.values()}
    assert vocab == set(mod.PROC_KINDS) == sc.probe_proc_kinds()  # every kind has a site
    seen = 0
    for n in sorted(os.listdir(CONFIGS)):
        if not n.endswith(".scenario.json"):
            continue
        with open(os.path.join(CONFIGS, n)) as f:
            doc = json.load(f)
        for r in doc["roles"]:
            assert set(r["proc_kinds"]) <= vocab, (n, r["role"], r["proc_kinds"])
            if r["measure_class"] == "under_test" and r["launcher"] == "product":
                assert r["proc_kinds"], (n, r["role"])
                seen += 1
            args = [a for a in r["args"] if isinstance(a, str)]
            if any(args[i:i + 2] == ["--num_serve_frontends", "1"] for i in range(len(args))):
                assert r["proc_kinds"] == ["frontend", "frontend_openengine",
                                           "frontend_single"], (n, r["role"])
    assert seen >= 2


# ---- capenv.spinners (INTERFACES errata SB-B1, design test #26) ------------------------------

def _spin_role(args=("--target-role", "product")):
    return cf.role("spin", "capkit", "aux", asyncio=False, mclass="observer",
                   module="capenv.spinners", snapshots="none", args=list(args))


def _spin_ctx():
    ident = Identity("E0", "a", 0, "local-x", 0, "a" * 64, "launch", pid=1)
    names = [r["role"] for r in cf.default_roles()] + ["spin"]
    own = {n: {"ready_file": "/T/r/%s" % n, "port_file": "/T/p/%s.port" % n,
               "report_addr": "/T/p/%s.addr" % n} for n in names}
    return sc.ArgvContext(ident, "/T", "/T/raw", {}, 1.5, "/shm", 40, "/W/t.json", 123, own,
                          lambda r: "/T/ports/%s.json" % r, lambda r: "127.0.0.1:99",
                          sidecar={"roles_file": "R", "out_dir": "O", "logs_dir": "L",
                                   "cpus_watch": "0-3"}, canary={"out": "C"})


def test_spinners_aux_role_is_accepted():
    roles = sc.validate_scenario(_doc(cf.default_roles() + [_spin_role()]))
    assert roles["spin"]["kind"] == "aux" and roles["spin"]["module"] == "capenv.spinners"
    sc.validate_scenario(_doc(cf.default_roles() + [_spin_role(args=())]))


@pytest.mark.parametrize("extra", [["--pin-table", "/W/pin_table.json"],
                                   ["--pin-table=/W/pin_table.json"]])
@pytest.mark.parametrize("who", ["spin", "capload", "sidecar"])
def test_scenario_args_may_not_give_pin_table(extra, who):
    """--pin-table is orchestrator-managed: a scenario that writes it (spinners or any other
    capkit tool) is refused."""
    d = _doc(cf.default_roles() + [_spin_role()])
    [r for r in d["roles"] if r["role"] == who][0]["args"] += extra
    with pytest.raises(ConfigError, match="orchestrator-managed flags"):
        sc.validate_scenario(d)


def test_build_argv_gives_spinners_the_launch_trees_pin_table():
    """build_argv appends --pin-table <T>/raw/env/pin_table.json (the path the orchestrator
    copies the pin table to) for capenv.spinners only, and the tool's own parser accepts the
    whole argv."""
    from capcommon.runtree import RunTree  # noqa: PLC0415
    from capenv.spinners import build_parser  # noqa: PLC0415
    roles = sc.validate_scenario(_doc(cf.default_roles() + [_spin_role()]))
    ctx = _spin_ctx()
    a = sc.build_argv(roles["spin"], ctx, [6], "py")
    assert a.count("--pin-table") == 1
    assert a[a.index("--pin-table") + 1] == RunTree("/T").env("pin_table.json") \
        == "/T/raw/env/pin_table.json"
    assert a[a.index("--origin-ns") + 1] == "123" and a[a.index("--ready-file") + 1] == "/T/r/spin"
    assert a[a.index("--cpus") + 1] == "6" and a[-2:] == ["--target-role", "product"]
    for flag in ("--out", "--shm-dir", "--tier-table", "--workers", "--port-file", "--loop"):
        assert flag not in a
    ns = build_parser().parse_args(a[3:])
    assert ns.pin_table == "/T/raw/env/pin_table.json" and ns.target_role == "product"
    assert sc.build_argv(roles["spin"], ctx, [6], "py", runtime=False) == a
    for n, r in roles.items():
        if n != "spin":
            assert "--pin-table" not in sc.build_argv(r, ctx, [0, 1], "py", runtime=False), n
