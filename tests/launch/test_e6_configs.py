# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""configs/e6 (card E6 arm iii: n independent single-frontend chains): the generator reproduces
the shipped files; every arm validates, expands and cross-checks; each chain i is wired only to
itself (capload<i> -> frontend<i> -> spb<i>, one reconcile chain per pair); the core demand fits
one NUMA node of the target; and an orchestrator launch of two independent copies of a fake
chain completes with every role registered on its own cores (no processes of the real product)."""

import json
import os
import subprocess
import sys

import pytest

import conftest as cf
from capcommon import jobinputs as JI
from capcommon.runtree import read_state
from capcommon.templates import placeholders
from capcommon.tiers import TierTable
from caplaunch import scenario as sc
from caplaunch import tiergen

GEN = os.path.join(cf.C, "configs", "gen_e6_configs.py")
D = os.path.join(cf.C, "configs", "e6")
E4 = os.path.join(cf.C, "configs", "e4")
SNAP_MS = 50
NUMA_CORES = 56  # physical cores per NUMA node of the target (DGXB200 G0, pin tables 2026-11-16)


def _load(d, n):
    with open(os.path.join(d, n)) as f:
        return json.load(f)


ARMS = sorted(n[:-len(".scenario.json")] for n in os.listdir(D) if n.endswith(".scenario.json"))


def _n(arm):
    return int([t for t in arm.split("-") if t.startswith("n")][0][1:])


def _roles(doc):
    return {r["role"]: r for r in doc["roles"]}


def _strs(items):
    return [a for a in items if isinstance(a, str)]


def _opt(args, flag):
    a = _strs(args)
    return a[a.index(flag) + 1] if flag in a else None


def _values(tpl):
    v = {}
    for k in tiergen.tbd_names(tpl):
        if k.startswith("c_") and k[2:].isdigit():
            v[k] = 2 << int(k[2:])
        elif k.startswith("rate_") and k[5:].isdigit():
            v[k] = 4.0 * (2 ** int(k[5:]))
        else:
            v[k] = {"arrival_seed": 1, "c_warmup": 1, "rate_warmup": 2.0, "n_tokens": 8,
                    "beat_ns": 8 * cf.MS, "gate_poll_ns": 100 * cf.MS, "gate_settle_polls": 2,
                    "gate_deadline_guard_ns": 100 * cf.MS,
                    "gate_max_snapshot_age_ns": 200 * cf.MS, "ramp_ns": 0, "tail_ns": 0,
                    "window_ns": 1280 * cf.MS, "warmup_ns": 1280 * cf.MS,
                    "drain_max_wait_ns": 1280 * cf.MS, "request_timeout_ns": 5000 * cf.MS}[k]
    return v


def test_generator_reproduces_the_shipped_files(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-B", GEN, str(out)], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    made = sorted(os.listdir(str(out / "e6")))
    assert made == sorted(n for n in os.listdir(D) if n != "README.md")
    for n in made:
        with open(str(out / "e6" / n), "rb") as a, open(os.path.join(D, n), "rb") as b:
            assert a.read() == b.read(), n


def test_arm_set():
    assert ARMS == ["e6-iii-n2-mc", "e6-iii-n2-r", "e6-iii-n4-mc", "e6-iii-n4-r"]


@pytest.mark.parametrize("arm", ARMS)
def test_arm_validates_expands_and_cross_checks(arm):
    doc = _load(D, arm + ".scenario.json")
    assert doc["experiment"] == "E6" and doc["name"] == arm
    roles = sc.validate_scenario(doc)
    JI.check_scenario(doc)
    tpl = _load(D, arm + ".tiers.template.json")
    todo = _load(D, arm + ".values.todo.json")
    assert set(todo["values"]) == tiergen.tbd_names(tpl)
    tt = TierTable.from_doc(tiergen.expand(tpl, _values(tpl)))
    pin, nxt = {}, 0
    for r in roles.values():
        pin[r["pin_role"]] = {"cpus": list(range(nxt, nxt + r["cpus_needed"])), "numa_node": 0}
        nxt += r["cpus_needed"]
    assert sorted(sc.cross_check(doc, roles, tt, pin, SNAP_MS * cf.MS)) == sorted(roles)
    assert nxt <= NUMA_CORES, (arm, nxt)
    used = set()

    def walk(o):
        if isinstance(o, str):
            used.update(a for n, a in placeholders(o) if n == "param")
        elif isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]
    walk(doc["roles"])
    walk(doc["job_inputs"])
    assert sorted(used) == doc["params_required"]


@pytest.mark.parametrize("arm", ARMS)
def test_each_chain_is_wired_only_to_itself(arm):
    n = _n(arm)
    doc = _load(D, arm + ".scenario.json")
    rs = _roles(doc)
    assert set(rs) == ({"spb%d" % i for i in range(n)} | {"frontend%d" % i for i in range(n)}
                       | {"capload%d" % i for i in range(n)} | {"sidecar", "canary"})
    for i in range(n):
        fe, cl, st = rs["frontend%d" % i], rs["capload%d" % i], rs["spb%d" % i]
        assert _opt(fe["args"], "--openengine-endpoint") == "{addr:spb%d}" % i
        assert fe["start_after"] == ["spb%d" % i]
        assert _opt(cl["args"], "--target") == "{addr:frontend%d}" % i
        assert cl["start_after"] == ["frontend%d" % i]
        assert _opt(fe["args"], "--num_serve_frontends") == "1"
        assert _opt(fe["args"], "--num_postprocess_workers") == "0"
        assert st["module"] == "capstubs.prebaked_server" and st["cpus_needed"] == 1
        assert fe["launcher"] == "product" and fe["measure_class"] == "under_test"
        assert cl["workers"] == cl["cpus_needed"] and cl["expected_procs"] == cl["workers"] + 1
    chains = doc["reconcile"]["chains"]
    assert [[(h["role"], h["source"]) for h in c["hops"]] for c in chains] == [
        [("capload%d" % i, "summary.counts.ok"), ("spb%d" % i, "summary.counts.streams_accepted")]
        for i in range(n)]
    tpl = _load(D, arm + ".tiers.template.json")
    gate_roles = {t["role"] for rd in tpl["default_gate"]["readings"] for t in rd["terms"]}
    assert gate_roles == {r for r in rs if r.startswith(("capload", "spb"))}


@pytest.mark.parametrize("arm", ARMS)
def test_each_chain_matches_the_e4_main_tier(arm):
    """Each copy is the E4-4b main tier: the same frontend argv apart from the stub name, the
    same S-PB argv, the same capload argv apart from the target, the same tier template apart
    from names and the gate's role lists."""
    shape = arm.rsplit("-", 1)[-1]
    e4 = _roles(_load(E4, "e4b-%s.scenario.json" % shape))
    rs = _roles(_load(D, arm + ".scenario.json"))
    for i in range(_n(arm)):
        assert [a.replace("spb%d" % i, "spb") if isinstance(a, str) else a
                for a in rs["frontend%d" % i]["args"]] == e4["frontend"]["args"]
        assert rs["frontend%d" % i]["cpus_needed"] == e4["frontend"]["cpus_needed"]
        assert rs["spb%d" % i]["args"] == e4["spb"]["args"]
        assert [a.replace("frontend%d" % i, "frontend") for a in rs["capload%d" % i]["args"]] \
            == e4["capload"]["args"]
    t4 = _load(E4, "e4b-%s.tiers.template.json" % shape)
    t6 = _load(D, arm + ".tiers.template.json")
    for k in ("arrival", "axis", "request_timeout_ns", "common", "tiers", "launch_inject"):
        assert t6[k] == t4[k], k


def _chain(i):
    return [
        cf.role("nullsse%d" % i, "capkit", "server", cpus_needed=1, workers=1, expected=1,
                module="capload.nullsse", backlog=["backlog.active_streams"],
                args=["--slot", "backlog.active_streams", "--drain-delay-ms", "150"]),
        cf.role("product%d" % i, "product", "server", asyncio=False, probe_tier="P0",
                snapshots="none", command=[sys.executable, cf.DUMMY],
                port={"kind": "plain_port_file", "host": "127.0.0.1"},
                readiness=[{"kind": "plain_port_file"},
                           {"kind": "http_get", "path": "/health", "expect_status": 200,
                            "timeout_s": 1.0}],
                args=["--host", "127.0.0.1", "--port-file", "{own:port_file}"]),
        cf.role("capload%d" % i, "capkit", "client", cpus_needed=1, workers=1,
                mclass="load_source", module="capload", backlog=["backlog.inflight"],
                start_after=["nullsse%d" % i, "product%d" % i],
                args=["--slot", "backlog.inflight", "--exit-after-end", "--drain-delay-ms",
                      "100", "--target-port-file", "{port_file:nullsse%d}" % i,
                      "--peer", "{addr:product%d}" % i]),
    ]


def test_two_independent_chains_launch_and_register_separately(tmp_path):
    """Job path with fake tools: two copies of (stub, product, load) in one scenario, each pair
    with its own reconcile chain; the launch is VALID, every role is registered with its own
    pinned cores, and nothing is left unattributed."""
    roles = _chain(0) + _chain(1)
    g = cf.gate([("nullsse0", "backlog.active_streams"), ("nullsse1", "backlog.active_streams"),
                 ("capload0", "backlog.inflight"), ("capload1", "backlog.inflight")])
    rec = {"schema": "capkit.reconcile/1", "chains": [
        {"name": "requests%d" % i, "kind": "requests", "gate": None, "include_usage": None,
         "hops": [{"role": "capload%d" % i, "source": "summary.counts.ok", "scope": "tier"}],
         "corrections": []} for i in (0, 1)]}
    stop = ["capload0", "capload1", "product0", "product1", "nullsse0", "nullsse1"]
    wd = cf.make_workdir(tmp_path, cf.tier_doc(g), roles, stop_order=stop, reconcile=rec)
    T = os.path.join(str(tmp_path), "T")
    r = cf.run_launch(cf.launch_argv(wd, T, os.path.join(str(tmp_path), "shm"), lead_ms=3500))
    assert r.returncode == 0, r.stderr
    assert [s[1] for s in read_state(T)] == ["VALID"]
    with open(os.path.join(T, "raw", "procs", "roles.json")) as f:
        rj = json.load(f)
    got = {p["role"] for p in rj["procs"]}
    assert got == {x["role"] for x in roles}
    assert rj["unattributed"] == []
    with open(os.path.join(T, "raw", "env", "pin_table.json")) as f:
        pins = json.load(f)["roles"]
    cpus = [c for x in roles for c in pins[x["role"]]["cpus"]]
    assert len(cpus) == len(set(cpus))
    with open(os.path.join(T, "raw", "launch_end.json")) as f:
        end = json.load(f)
    assert end["stop"]["reason"] == "completed" and set(end["exit_codes"].values()) == {0}
