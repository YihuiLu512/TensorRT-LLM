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
])
def test_invalid_scenarios_rejected(mutate, needle):
    d = _doc()
    mutate(d)
    with pytest.raises(ConfigError) as e:
        sc.validate_scenario(d)
    assert needle in str(e.value)


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
    the expected sites by it), and every product role of a probed scenario declares its kinds."""
    import importlib.util  # noqa: PLC0415
    spec = importlib.util.spec_from_file_location(
        "_capprobe_sites", os.path.join(cf.TOOLS, "capprobe", "sites.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # sites.py alone: importing capprobe would parse CAPPROBE_*
    vocab = {v["proc_kind"] for v in mod.SITES.values()}
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
    assert seen >= 2
