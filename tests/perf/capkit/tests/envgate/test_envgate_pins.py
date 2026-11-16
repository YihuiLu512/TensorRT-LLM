# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin table generator on synthetic and live lscpu text; read-back verifier with live processes."""

import json
import os
import subprocess

import pytest

from b4_envgate_util import PY, capenv_env, id_args, kill_all, make_ident, spawn_py, IDLE_SRC
from capcommon.exitcodes import ConfigError
from capcommon.jsonio import write_json_stamped
from capenv import pins, roles as roles_mod, topology


def lscpu_2s_ht(cores_per_socket=56):
    """2 sockets x N cores x 2 threads, Linux numbering: cpu i and i + 2N are siblings
    (DGX B200 / Xeon 8570 style: 224 logical CPUs for N = 56)."""
    lines = ["# The following is the parsable format", "# CPU,Core,Socket,Node"]
    total_cores = 2 * cores_per_socket
    for cpu in range(2 * total_cores):
        core = cpu % total_cores
        sock = core // cores_per_socket
        lines.append("%d,%d,%d,%d" % (cpu, core, sock, sock))
    return "\n".join(lines) + "\n"


def lscpu_1s(n=32):
    return "# CPU,Core,Socket,Node\n" + "".join("%d,%d,0,0\n" % (i, i) for i in range(n))


SRC_SHA = "c" * 64


def test_dgx_like_topology_summary():
    topo = topology.parse_lscpu_p(lscpu_2s_ht())
    s = topo.summary()
    assert s["logical_cpus"] == 224 and s["physical_cores"] == 112 and s["sockets"] == 2
    assert s["threads_per_core"] == 2 and s["nodes"] == [0, 1]
    assert s["sibling_table"][0] == [0, 112] and s["sibling_table"][56] == [56, 168]


def test_generate_2s_one_logical_per_core_disjoint_same_node():
    topo = topology.parse_lscpu_p(lscpu_2s_ht())
    demand = [{"role": "capload", "cpus_needed": 8}, {"role": "nullsse", "cpus_needed": 4},
              {"role": "sidecar", "cpus_needed": 1}, {"role": "canary", "cpus_needed": 1}]
    t = pins.generate(topo, demand, avoid_cpus=[0], source_sha=SRC_SHA)
    all_cpus = [c for e in t["roles"].values() for c in e["cpus"]]
    assert len(all_cpus) == len(set(all_cpus)) == 14
    assert 0 not in all_cpus and 112 not in all_cpus  # whole core of cpu 0 avoided
    assert all(c < 112 for c in all_cpus)  # representatives only, never a sibling
    assert {e["numa_node"] for e in t["roles"].values()} == {0}
    assert all(topo.cpu_node(c) == 0 for c in all_cpus)
    sibs = {g[0]: g[1:] for g in t["siblings_left_idle"]}
    assert set(sibs) == set(all_cpus) and all(sibs[c] == [c + 112] for c in all_cpus)
    assert t["roles"]["capload"]["cpus"] == list(range(1, 9))
    assert len(t["table_sha256"]) == 64


def test_generate_explicit_numa_and_gap():
    topo = topology.parse_lscpu_p(lscpu_2s_ht())
    t = pins.generate(topo, [{"role": "a", "cpus_needed": 2, "numa": 1},
                             {"role": "b", "cpus_needed": 2}], source_sha=SRC_SHA)
    assert t["primary_numa"] == 1
    assert all(topo.cpu_node(c) == 1 for e in t["roles"].values() for c in e["cpus"])
    with pytest.raises(pins.PinGap) as ei:
        pins.generate(topo, [{"role": "big", "cpus_needed": 57}], source_sha=SRC_SHA)
    assert ei.value.gap[0]["short_by"] == 1  # 56 cores per node; never spills to node 1


def test_generate_1s_32_and_gap():
    topo = topology.parse_lscpu_p(lscpu_1s(32))
    t = pins.generate(topo, [{"role": "x", "cpus_needed": 31}], avoid_cpus=[0],
                      source_sha=SRC_SHA)
    assert t["roles"]["x"]["cpus"] == list(range(1, 32)) and t["siblings_left_idle"] == []
    with pytest.raises(pins.PinGap):
        pins.generate(topo, [{"role": "x", "cpus_needed": 32}], avoid_cpus=[0],
                      source_sha=SRC_SHA)


def test_bad_inputs_rejected():
    with pytest.raises(ConfigError):
        topology.parse_lscpu_p("# only comments\n")
    with pytest.raises(ConfigError):
        topology.parse_lscpu_p("0,0,0\n")
    topo = topology.parse_lscpu_p(lscpu_1s(4))
    with pytest.raises(ConfigError):
        pins.generate(topo, [{"role": "a", "cpus_needed": 1}, {"role": "a", "cpus_needed": 1}])
    with pytest.raises(ConfigError):
        pins.generate(topo, [{"role": "a", "cpus_needed": 1}], avoid_cpus=[99])
    bad = {"roles": {"a": {"cpus": [1]}, "b": {"cpus": [1]}}, "siblings_left_idle": []}
    with pytest.raises(ConfigError):
        pins.check_table(bad)


def test_generate_on_this_vm_lscpu():
    topo, rec = topology.probe()
    from capenv import procfs
    assert rec["rc"] == 0
    assert topo.summary()["logical_cpus"] == len(procfs.online_cpus())
    t = pins.generate(topo, [{"role": "a", "cpus_needed": 1}, {"role": "b", "cpus_needed": 1}],
                      source_sha=SRC_SHA)
    assert set(t["roles"]["a"]["cpus"]).isdisjoint(t["roles"]["b"]["cpus"])


def _roles_file(path, procs):
    payload = {"generation": 1, "procs": procs, "unattributed": []}
    write_json_stamped(path, roles_mod.ROLES_SCHEMA, make_ident("launch"), payload)


def _proc_entry(p, role):
    from capenv import procfs
    return {"pid": p.pid, "starttime_ticks": procfs.starttime_ticks(p.pid), "role": role,
            "worker": 0}


def test_verify_pass_red_and_not_run(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 3:
        pytest.skip("needs 3 cpus")
    a_cpu, b_cpu = cpus[1], cpus[2]
    table = {"roles": {"ra": {"cpus": [a_cpu], "numa_node": 0},
                       "rb": {"cpus": [b_cpu], "numa_node": 0}},
             "siblings_left_idle": [], "table_sha256": "d" * 64}
    pa = spawn_py(IDLE_SRC, [a_cpu])
    pb = spawn_py(IDLE_SRC, [b_cpu])
    procs = [pa, pb]
    try:
        rf = str(tmp_path / "roles.json")
        _roles_file(rf, [_proc_entry(pa, "ra"), _proc_entry(pb, "rb")])
        import time
        time.sleep(0.6)  # let both run once on their pinned cpu
        res = pins.verify(table, roles_mod.load(rf))
        assert res["verdict"] == "PASS", res["red"]
        assert res["expected_procs"] == 2 and res["checked_threads"] >= 2
        # positive control: pb's affinity now overlaps ra's cpu -> RED
        os.sched_setaffinity(pb.pid, {a_cpu, b_cpu})
        res2 = pins.verify(table, roles_mod.load(rf))
        assert res2["verdict"] == "RED"
        assert any(r.get("pid") == pb.pid for r in res2["red"])
        # a registered role missing from the table is red too
        _roles_file(rf, [_proc_entry(pa, "ra"), _proc_entry(pb, "rc")])
        res3 = pins.verify(table, roles_mod.load(rf))
        assert res3["verdict"] == "RED"
        assert any(r.get("why", "").startswith("registered role has no pin") for r in res3["red"])
        # nothing registered -> NOT_RUN, never PASS
        _roles_file(rf, [])
        assert pins.verify(table, roles_mod.load(rf))["verdict"] == "NOT_RUN"
    finally:
        kill_all(procs)


def test_cli_generate_and_gap_exit(tmp_path):
    lf = tmp_path / "lscpu.txt"
    lf.write_text(lscpu_2s_ht())
    dem = tmp_path / "demand.json"
    dem.write_text(json.dumps([{"role": "capload", "cpus_needed": 4},
                               {"role": "sidecar", "cpus_needed": 1}]))
    out = tmp_path / "pin_table.json"
    cmd = [PY, "-m", "capenv.pins", "generate"] + id_args("pins") + [
        "--demand", str(dem), "--out", str(out), "--source-sha", SRC_SHA,
        "--lscpu-file", str(lf), "--avoid-cpus", "0"]
    r = subprocess.run(cmd, env=capenv_env(), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    doc = json.loads(out.read_text())
    assert doc["schema"] == "capkit.pin_table/1" and doc["identity"]["role"] == "pins"
    assert doc["roles"]["capload"]["cpus"] == [1, 2, 3, 4]
    dem.write_text(json.dumps([{"role": "capload", "cpus_needed": 500}]))
    r = subprocess.run(cmd, env=capenv_env(), capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "DOES_NOT_FIT" in r.stderr
