# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The case-03 perf arms (perf/openengine_arms.py, TH-DSG-6, CP-06/07/08/11) on CPU."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

DS_ENV = {"TRTLLM_TEST_DS_SERVE_CMD": "python3 -m dsv4_serve"}
E3_ARMS = ["E3-native", "E3-native-al", "E3-mid", "E3-chain", "E3-chain-lean"]
WORKER_CONFIG = {
    "max_batch_size": 32,
    "max_num_tokens": 8192,
    "max_seq_len": 32768,
    "tensor_parallel_size": 2,
    "enable_attention_dp": True,
    "speculative_config": {"decoding_type": "DSpark", "max_draft_len": 5},
    "kv_cache_config": {"enable_block_reuse": True, "free_gpu_memory_fraction": 0.5},
    "cache_transceiver_config": {"backend": "NIXL"},
    "enable_tokenization_cache": True,
}


def _fields(arm):
    return {f.name: getattr(arm, f.name) for f in dataclasses.fields(arm) if f.name != "name"}


def test_case03_arms(arms):
    table = arms.DISAGG_ARMS
    assert [n for n in table if n.startswith("E3-")] == [*E3_ARMS, "E3-direct"]
    for name in E3_ARMS:
        arm = table[name]
        assert (arm.case, arm.topology, arm.glm_http) == ("case-03", "disagg", False)
    assert table["E3-direct"].topology == "direct"
    native = _fields(table["E3-native"])
    # E3-native-al changes only the alignment; E3-mid only the disaggregated server;
    # E3-chain only the workers (chain ds instead of dsv4-serve's own HTTP).
    assert {k for k, v in _fields(table["E3-native-al"]).items() if native[k] != v} == {"aligned"}
    aligned = _fields(table["E3-native-al"])
    assert {k for k, v in _fields(table["E3-mid"]).items() if aligned[k] != v} == {"orchestrator"}
    mid = _fields(table["E3-mid"])
    assert {k for k, v in _fields(table["E3-chain"]).items() if mid[k] != v} == {
        "chain",
        "ctx",
        "gen",
    }
    for name in ("E3-native", "E3-native-al", "E3-mid"):
        for role in ("ctx", "gen"):
            settings = arms.role_arm(table[name], role)
            assert (settings.chain, settings.ds_http, settings.frontends) == ("none", True, 1)
    for role in ("ctx", "gen"):
        assert arms.role_arm(table["E3-chain"], role) == arms.RoleArm("ds", 1, None)
        assert arms.role_arm(table["E3-chain-lean"], role) == arms.RoleArm("ds", 1, 0)
    assert [table[n].orchestrator for n in E3_ARMS] == ["ds", "ds", "trtllm", "trtllm", "trtllm"]
    assert [table[n].aligned for n in E3_ARMS] == [False, True, True, True, True]
    # The case-04 arms keep their defaults.
    assert all(
        (a.orchestrator, a.aligned) == ("trtllm", False)
        for n, a in table.items()
        if n.startswith("E4-")
    )


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"TRTLLM_PERF_ARM": "E3-native"}, "E3-native"),
        ({"TRTLLM_PERF_ARM": "E3-mid"}, "E3-mid"),
        ({"TRTLLM_PERF_ARM": "E3-chain", "TRTLLM_TEST_OPENENGINE_CHAIN": "ds"}, "E3-chain"),
    ],
)
def test_selected_arm(arms, env, expected):
    assert arms.selected_arm(env).name == expected


@pytest.mark.parametrize(
    "env, message",
    [
        ({"TRTLLM_PERF_ARM": "E3-chain"}, "set to 'ds'"),
        ({"TRTLLM_PERF_ARM": "E3-native", "TRTLLM_TEST_OPENENGINE_CHAIN": "ds"}, "unset"),
    ],
)
def test_selected_arm_refuses(arms, env, message):
    with pytest.raises(arms.ArmError, match=message):
        arms.selected_arm(env)


def _worker_cmd(tmp_path):
    config = tmp_path / "worker.yaml"
    config.write_text(yaml.safe_dump(WORKER_CONFIG))
    return ["trtllm-serve", "/models/ds", "--backend", "pytorch", "--config", str(config)]


@pytest.mark.parametrize("name", ["E3-native", "E3-native-al", "E3-mid"])
@pytest.mark.parametrize("role, server_role", [("ctx", "context"), ("gen", "generation")])
def test_native_worker_runs_dsv4_serve(arms, tmp_path, monkeypatch, name, role, server_role):
    monkeypatch.setenv("TRTLLM_TEST_DS_SERVE_CMD", "python3 -m dsv4_serve")
    cmd = _worker_cmd(tmp_path)
    plan = arms.prepare_worker(cmd, arms.DISAGG_ARMS[name], role, str(tmp_path / "out"))
    tokens = list(plan.cmd)
    assert tokens[:4] == ["python3", "-m", "dsv4_serve", "/models/ds"]
    assert tokens[-2:] == ["--server_role", server_role]
    written = yaml.safe_load(Path(tokens[tokens.index("--config") + 1]).read_text())
    assert written["num_serve_frontends"] == 1
    assert {k: written[k] for k in WORKER_CONFIG} == WORKER_CONFIG
    view = arms.role_view(arms.DISAGG_ARMS[name], role)
    # Port 0 and --report_addr, which dsv4-serve's worker honours with one front end.
    assert not arms.worker_needs_port(view, plan)


def test_native_worker_needs_the_ds_command(arms, tmp_path, monkeypatch):
    monkeypatch.delenv("TRTLLM_TEST_DS_SERVE_CMD", raising=False)
    with pytest.raises(arms.ArmError, match="TRTLLM_TEST_DS_SERVE_CMD"):
        arms.prepare_worker(
            _worker_cmd(tmp_path), arms.DISAGG_ARMS["E3-native"], "ctx", str(tmp_path / "o")
        )


def test_chain_worker_keeps_trtllm_serve(arms, tmp_path, monkeypatch):
    monkeypatch.setattr(arms, "_frontend_check", lambda: lambda namespace: None)
    cmd = _worker_cmd(tmp_path)
    plan = arms.prepare_worker(cmd, arms.DISAGG_ARMS["E3-chain"], "gen", str(tmp_path / "o"))
    assert list(plan.cmd)[:2] == ["trtllm-serve", "/models/ds"]
    assert list(plan.cmd)[-2:] == ["--server_role", "generation"]
    assert arms.is_ds_engine_role(arms.DISAGG_ARMS["E3-chain"], "gen")
    assert not arms.is_ds_engine_role(arms.DISAGG_ARMS["E4-chain"], "gen")


ORCHESTRATOR = {
    "hostname": "node-a",
    "port": 0,
    "backend": "pytorch",
    "context_servers": {
        "num_instances": 1,
        "urls": ["node-a:1"],
        "router": {"type": "conversation"},
    },
    "generation_servers": {"num_instances": 1, "urls": ["node-a:2"]},
    "gen_strip_message_history": True,
    "gen_tokids_ctxbytes": True,
}
DISAGG_CMD = ["trtllm-serve", "disaggregated", "-c", "<config>", "-t", "600", "-r", "600"]


def _orchestrator(arms, tmp_path, name, env, monkeypatch):
    monkeypatch.setattr(arms, "free_port", lambda host: 43210)
    config = tmp_path / "server_config.0.yaml"
    config.write_text(yaml.safe_dump(ORCHESTRATOR))
    cmd = [*DISAGG_CMD, "--report_addr", str(tmp_path / "addr")]
    plan = arms.prepare_orchestrator(
        arms.DISAGG_ARMS[name], cmd, str(config), "node-a", env, environ=DS_ENV
    )
    return plan, cmd, yaml.safe_load(config.read_text())


@pytest.mark.parametrize("name", ["E3-native", "E3-native-al"])
def test_native_orchestrator_is_dsv4_serve_on_a_chosen_port(arms, tmp_path, monkeypatch, name):
    env = {"DSV4_PROXY_LOAD_ROUTE": "1", "DSV4_PROXY_WORKERS": "4", "OTHER": "x"}
    plan, _, written = _orchestrator(arms, tmp_path, name, env, monkeypatch)
    assert list(plan.cmd) == ["python3", "-m", "dsv4_serve", "disaggregated", *DISAGG_CMD[2:]]
    assert plan.port == 43210 and written["port"] == 43210
    aligned = name == "E3-native-al"
    for key, value in arms.DS_ALIGNED_ORCHESTRATOR.items():
        assert written.get(key) == (value if aligned else None)
    expected_env = {"DSV4_PROXY_WORKERS": "1", "OTHER": "x"} if aligned else env
    assert plan.env == expected_env
    assert {k: written[k] for k in ORCHESTRATOR if k != "port"} == {
        k: v for k, v in ORCHESTRATOR.items() if k != "port"
    }


@pytest.mark.parametrize("name", ["E3-mid", "E3-chain"])
def test_trtllm_orchestrator_of_case03_writes_the_aligned_values(arms, tmp_path, monkeypatch, name):
    plan, cmd, written = _orchestrator(arms, tmp_path, name, {"A": "1"}, monkeypatch)
    assert list(plan.cmd) == cmd and plan.port is None and plan.env == {"A": "1"}
    assert written == {**ORCHESTRATOR, **arms.DS_ALIGNED_ORCHESTRATOR}


def test_case04_orchestrator_is_untouched(arms, tmp_path, monkeypatch):
    plan, cmd, written = _orchestrator(arms, tmp_path, "E4-chain", {"A": "1"}, monkeypatch)
    assert (list(plan.cmd), plan.port, plan.env, written) == (cmd, None, {"A": "1"}, ORCHESTRATOR)


def test_ds_orchestrator_command_refuses_another_command(arms):
    with pytest.raises(arms.ArmError, match="disaggregated"):
        arms.ds_orchestrator_command(["trtllm-serve", "/m"], DS_ENV)
    assert arms.ds_orchestrator_command(
        ["trtllm-serve", "disaggregated", "-c", "x", "--report_addr=/a"], DS_ENV
    )[-2:] == ["-c", "x"]


def _case(**changes):
    values = dict(
        runtime="multi_node_disagg_server",
        benchmark_mode="e2e",
        time_breakdown=False,
        numa_bind=False,
        dwdp_size=0,
        worker_gpus={"ctx": 2, "gen": 16},
        gpus_per_node=4,
    )
    values.update(changes)
    return values


def test_case03_worker_may_span_whole_nodes(arms):
    arms.check_disagg_case(arms.DISAGG_ARMS["E3-chain"], **_case())
    arms.check_disagg_case(arms.DISAGG_ARMS["E3-native"], **_case())
    with pytest.raises(arms.ArmError, match="more than one node"):
        arms.check_disagg_case(arms.DISAGG_ARMS["E4-chain"], **_case())
    with pytest.raises(arms.ArmError, match="more than one node"):
        arms.check_disagg_case(arms.DISAGG_ARMS["E3-chain"], **_case(worker_gpus={"gen": 6}))


@pytest.mark.parametrize(
    "tp, env, rank",
    [
        (4, {"SLURM_NODEID": "1"}, None),
        (16, {}, 0),
        (16, {"SLURM_PROCID": "3"}, 3),
        (16, {"SLURM_NODEID": "2", "SLURM_PROCID": "3"}, 2),
        (16, {"DSV4_NODE_RANK": "1", "SLURM_NODEID": "2"}, 1),
    ],
)
def test_ds_node_rank(arms, tp, env, rank):
    assert arms.ds_node_rank({"tensor_parallel_size": tp}, 4, env) == rank


def test_ds_node_rank_refuses(arms):
    with pytest.raises(arms.ArmError, match="does not fill"):
        arms.ds_node_rank({"tensor_parallel_size": 6}, 4, {})
    with pytest.raises(arms.ArmError, match="not a node rank"):
        arms.ds_node_rank({"tensor_parallel_size": 8}, 4, {"SLURM_NODEID": "x"})


def test_ds_follower_command(arms):
    trtllm = arms.ServerPlan(
        ("trtllm-serve", "/m", "--config", "c", "--server_role", "generation"), {}
    )
    assert arms.ds_follower_command(trtllm, DS_ENV) == [
        "python3", "-m", "dsv4_serve", "/m", "--config", "c", "--server_role", "generation",
    ]  # fmt: skip
    native = arms.ServerPlan(("python3", "-m", "dsv4_serve", "/m"), {})
    assert arms.ds_follower_command(native, DS_ENV) == list(native.cmd)


def test_run_ds_follower_stops_it_after_the_wait(arms, tmp_path):
    seen = []
    code = arms.run_ds_follower(
        [sys.executable, "-c", "import time; print('up', flush=True); time.sleep(60)"],
        env=dict(os.environ),
        log_path=str(tmp_path / "f.log"),
        wait=lambda proc: seen.append(proc.poll()),
    )
    assert seen == [None] and code is not None and code != 0


def _layout(layout):
    return [(i.worker, i.nodes, i.gpus, i.tp) for i in layout.instances], layout.orchestrator_node


def test_layouts(arms):
    nodes = [f"n{i}" for i in range(6)]
    assert _layout(arms.ds_layout("n0", nodes[:1])) == (
        [("CTX_0", ("n0",), "0,1", 2), ("GEN_0", ("n0",), "2,3", 2)],
        "n0",
    )
    assert _layout(arms.ds_layout("mini", nodes[:2])) == (
        [("CTX_0", ("n0",), "0,1", 2), ("GEN_0", ("n1",), "0,1,2,3", 4)],
        "n0",
    )
    # PU-08: 2 x DEP2 context on one node, DEP16 generation on four, the last node for the
    # disaggregated server and the client.
    assert _layout(arms.ds_layout("mid", nodes, client_node="last")) == (
        [
            ("CTX_0", ("n0",), "0,1", 2),
            ("CTX_1", ("n0",), "2,3", 2),
            ("GEN_0", ("n1", "n2", "n3", "n4"), "0,1,2,3", 16),
        ],
        "n5",
    )
    placed, _ = _layout(arms.ds_layout("mid", nodes, ctx_instances=3, client_node=1))
    assert [p[1] for p in placed] == [("n0",), ("n0",), ("n1",), ("n2", "n3", "n4", "n5")]
    assert arms.ds_layout("mini", nodes[:2], client_node=1).orchestrator_node == "n1"


@pytest.mark.parametrize(
    "name, count, kwargs, message",
    [
        ("mini", 1, {}, "needs 2"),
        ("mid", 5, {"client_node": "last"}, "needs 5"),
        ("n0", 1, {"client_node": "last"}, "besides"),
        ("n0", 1, {"client_node": 3}, "not one of"),
        ("big", 4, {}, "unknown"),
        ("mid", 6, {"ctx_instances": 0}, "at least one"),
    ],
)
def test_layout_refuses(arms, name, count, kwargs, message):
    with pytest.raises(arms.ArmError, match=message):
        arms.ds_layout(name, [f"n{i}" for i in range(count)], **kwargs)


def test_ds_instance_reading_reads_the_running_process(arms, tmp_path):
    config = tmp_path / "engine.yaml"
    config.write_text(yaml.safe_dump(WORKER_CONFIG))
    env = {
        **os.environ,
        "DSV4_TOKCACHE_ENTRIES": "1024",
        "DSV4_STEP_LOG": "/x.jsonl",
        "DSV4_XFER_PROF": "1",
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "--config",
            str(config),
            "--max_batch_size",
            "16",
        ],
        env=env,
    )
    cmdline = Path(f"/proc/{proc.pid}/cmdline")
    for _ in range(500):  # until the child has exec'd its command
        if b"--config" in cmdline.read_bytes():
            break
        time.sleep(0.01)
    try:
        plan = arms.ServerPlan(("trtllm-serve", "m"), {**WORKER_CONFIG, "num_serve_frontends": 1})
        notes = []
        reading = arms.ds_instance_reading(SimpleNamespace(handle=None, proc=proc), plan, {}, notes)
    finally:
        proc.kill()
        proc.wait()
    assert notes == []
    # The command line wins over the configuration file, as dsv4-serve reads them.
    assert reading["params"]["max_batch_size"] == 16
    assert reading["params"]["tensor_parallel_size"] == 2
    assert reading["params"]["speculative_config"] == WORKER_CONFIG["speculative_config"]
    assert reading["params"]["moe_expert_parallel_size"] is None
    assert reading["env"] == {"DSV4_TOKCACHE_ENTRIES": "1024"}
    assert reading["log_only_env"] == {"DSV4_STEP_LOG": "/x.jsonl", "DSV4_XFER_PROF": "1"}
    assert reading["gen_overlap_ignored_warning"] is False
    assert reading["frontend"] == {
        "num_serve_frontends": 1,
        "enable_tokenization_cache": True,
        "serve_frontend_affinity_port_stride": None,
    }


def test_ds_instance_reading_falls_back_to_the_plan(arms):
    plan = arms.ServerPlan(("python3", "-m", "dsv4_serve", "m"), dict(WORKER_CONFIG))
    notes = []
    gone = SimpleNamespace(pid=2**22 + 12345)
    reading = arms.ds_instance_reading(
        SimpleNamespace(handle=None, proc=gone), plan, {"DSV4_GEN_OVERLAP": "1"}, notes
    )
    assert reading["params"]["max_batch_size"] == 32
    assert reading["env"] == {"DSV4_GEN_OVERLAP": "1"}
    assert reading["gen_overlap_ignored_warning"] is None
    assert len(notes) == 4 and all(n.startswith("ds instance:") for n in notes)


BENCHMARK_OUTPUT = """\
Successful requests:                     80
Total generated tokens:                  81920
Output token throughput (tok/s):         1500.00
Mean TTFT (ms):                          300.0
Median TTFT (ms):                        290.0
P99 TTFT (ms):                           600.0
Mean TPOT (ms):                          8.0
Median TPOT (ms):                        7.9
P99 TPOT (ms):                           9.5
Mean ITL (ms):                           20.0
Median ITL (ms):                         19.0
P99 ITL (ms):                            30.0
Mean E2EL (ms):                          8500.0
Median E2EL (ms):                        8400.0
P99 E2EL (ms):                           9900.0
"""
CLIENT = [
    "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving", "--model", "m",
    "--max-concurrency", "8", "--dataset-name", "random", "--random-ids", "--tokenize-on-client",
]  # fmt: skip
INSTANCE = {
    "params": {"max_batch_size": 32, "tensor_parallel_size": 2},
    "env": {"DSV4_TOKCACHE_ENTRIES": "1024"},
    "log_only_env": {},
    "gen_overlap_ignored_warning": False,
    "frontend": {"num_serve_frontends": 1, "enable_tokenization_cache": True},
}


def _reading(worker, instance):
    return {
        "worker": worker,
        "role": "ctx" if worker.startswith("CTX") else "gen",
        "node": "node-a",
        "startup_s": 200.0,
        "gpus": 2,
        "frontends": {"configured": 1, "started": 1},
        "postproc_workers": {"configured": None, "started": 0},
        "stream_interval": None,
        "kv_transfer": {"backend": "NIXL", "runtime": None, "env": {}},
        "alignment": {"enable_block_reuse": True, "spec_decode_force_accepted": 2.51},
        "frontend_cpu_s": 4.0,
        "leg": None,
        "notes": [],
        "ds_instance": instance,
    }


def _result(arms, tmp_path, name, *, instance=INSTANCE, orchestrator=None, env=None, output=None):
    probe = arms.DisaggRunProbe(
        arms.DISAGG_ARMS[name], str(tmp_path), ("CTX_0", "GEN_0"), "http://127.0.0.1:1", "s1", 0
    )
    for phase in ("before", "after"):
        probe.readings[phase] = {w: _reading(w, instance) for w in probe.workers}
        probe.orchestrator[phase] = {"ctx.total_requests": 0 if phase == "before" else 80}
        probe.client_cpu[phase] = 1.0 if phase == "before" else 3.0
    probe.window = {"before": 1_000, "after": 2_000_000_000_000}
    config = tmp_path / "server_config.0.yaml"
    written = dict(ORCHESTRATOR)
    if orchestrator is None and arms.DISAGG_ARMS[name].aligned:
        written.update(arms.DS_ALIGNED_ORCHESTRATOR)
    written.update(orchestrator or {})
    config.write_text(yaml.safe_dump(written))
    return arms.build_disagg_result(
        probe,
        output=output or BENCHMARK_OUTPUT,
        client_cmd=CLIENT,
        test_id="disagg-e2e-gb300_ds",
        benchmark_mode="e2e",
        orchestrator_config=str(config),
        orchestrator_env=env if env is not None else {"DSV4_PROXY_WORKERS": "1"},
        environ={"TRTLLM_PERF_RUN": "1", "TRTLLM_PERF_DS_SHA": "e5e5da9"},
    )


@pytest.mark.parametrize("name", E3_ARMS)
def test_case03_result_matches_schema(arms, tmp_path, name):
    doc = _result(arms, tmp_path, name)
    assert arms.validate_disagg_result(doc) == []
    arm = arms.DISAGG_ARMS[name]
    assert (doc["case"], doc["orch"]["kind"], doc["versions"]["ds"]) == (
        "case-03",
        arm.orchestrator,
        "e5e5da9",
    )
    assert doc["ds_instances"] == {"ctx": INSTANCE, "gen": INSTANCE}
    assert doc["ds_alignment"]["frontend"] == {
        "ctx": INSTANCE["frontend"],
        "gen": INSTANCE["frontend"],
    }
    orch = doc["ds_alignment"]["orchestrator"]
    assert orch["gen_strip_message_history"] is True and orch["gen_tokids_ctxbytes"] is True
    if arm.aligned:
        assert {k: orch[k] for k in arms.DS_ALIGNED_ORCHESTRATOR} == arms.DS_ALIGNED_ORCHESTRATOR
    assert doc["ds_client"] is None
    assert any("accepts cache_transceiver_config.backend but ignores it" in n for n in doc["notes"])
    # dsv4-serve's workers serve their own HTTP: no chain proof.
    assert (doc["chain_proof"] is None) == arm.ctx.in_process


@pytest.mark.parametrize(
    "name, kwargs, message",
    [
        ("E3-native-al", {"orchestrator": {"max_retries": 1}}, "max_retries must be 3"),
        ("E3-chain", {"orchestrator": {"server_keep_alive_timeout": 10}}, "keep_alive_timeout"),
        (
            "E3-native-al",
            {"env": {"DSV4_PROXY_LOAD_ROUTE": "1", "DSV4_PROXY_WORKERS": "1"}},
            "load",
        ),
        ("E3-native-al", {"env": {}}, "one ds proxy"),
    ],
)
def test_case03_result_refuses_unaligned_values(arms, tmp_path, name, kwargs, message):
    doc = _result(arms, tmp_path, name, **kwargs)
    assert any(message in e for e in arms.validate_disagg_result(doc)), doc["ds_alignment"]


def test_case03_result_refuses_missing_blocks(arms, tmp_path):
    doc = _result(arms, tmp_path, "E3-chain")
    del doc["ds_instances"]
    assert "missing ['ds_instances']" in arms.validate_disagg_result(doc)[0]
    doc = _result(arms, tmp_path, "E3-chain")
    doc["orch"]["kind"] = "ds"
    assert any("orch.kind" in e for e in arms.validate_disagg_result(doc))
    case04 = _result(arms, tmp_path, "E3-chain")
    case04["case"] = "case-04"
    assert "unexpected" in arms.validate_disagg_result(case04)[0]


def test_compare_ds_instances(arms, tmp_path):
    al = _result(arms, tmp_path, "E3-native-al")
    chain = _result(arms, tmp_path, "E3-chain")
    assert arms.compare_ds_instances(al, chain) == []
    other = {
        **INSTANCE,
        "env": {"DSV4_TOKCACHE_ENTRIES": "64"},
        "log_only_env": {"DSV4_STEP_LOG": "y"},
    }
    differs = _result(arms, tmp_path, "E3-chain", instance=other)
    assert arms.compare_ds_instances(al, differs) == [
        "ctx.env.DSV4_TOKCACHE_ENTRIES: '1024' != '64'",
        "gen.env.DSV4_TOKCACHE_ENTRIES: '1024' != '64'",
    ]
    missing = _result(arms, tmp_path, "E3-chain", instance=None)
    assert arms.compare_ds_instances(al, missing) == [
        "ctx: no single ds instance reading on both sides",
        "gen: no single ds instance reading on both sides",
    ]


AGENTX_CMD = [
    "python", "/x/perf/agentx_client.py", "--model", "/models/m", "--concurrency", "8",
    "--dataset", "semianalysis_cc_traces_weka_062126",
]  # fmt: skip


def test_takes_client(arms):
    table = arms.DISAGG_ARMS
    for name in E3_ARMS:
        assert arms.takes_client(table[name], "agentx")
        assert arms.takes_client(table[name], "")
    assert not arms.takes_client(table["E3-direct"], "agentx")
    assert not arms.takes_client(table["E4-chain"], "agentx")
    assert not arms.takes_client(None, "agentx")
    assert not arms.takes_client(table["E3-chain"], "sa")


def test_arm_client_cmd(arms):
    out = arms.arm_client_cmd(arms.DISAGG_ARMS["E3-chain"], "agentx", AGENTX_CMD)
    assert out[:3] == [sys.executable, str(Path(arms.__file__).resolve()), "ds-agentx"]
    assert out[3:] == AGENTX_CMD[2:]
    # Other clients and arms keep their command.
    assert arms.arm_client_cmd(arms.DISAGG_ARMS["E3-chain"], "", CLIENT) == CLIENT
    assert arms.arm_client_cmd(arms.DISAGG_ARMS["E4-chain"], "agentx", AGENTX_CMD) == AGENTX_CMD
    assert arms.arm_client_cmd(None, "agentx", AGENTX_CMD) == AGENTX_CMD


def test_arm_client_cmd_refuses_options_ds_does_not_take(arms):
    with pytest.raises(arms.ArmError, match="not --max-output-tokens"):
        arms.arm_client_cmd(
            arms.DISAGG_ARMS["E3-native"], "agentx", AGENTX_CMD + ["--max-output-tokens", "1"]
        )


def _stat(unit, avg, p50=None, p99=None):
    entry = {"unit": unit, "avg": avg}
    if p50 is not None:
        entry.update(p50=p50, p99=p99)
    return entry


EXPORT = {
    "schema_version": "1",
    "metadata": {"submission_valid": True, "metric_duration_coverage": []},
    "was_cancelled": False,
    "error_summary": [],
    "request_count": _stat("requests", 120),
    "benchmark_duration": _stat("sec", 60.0),
    "request_throughput": _stat("requests/sec", 2.0),
    "output_token_throughput": _stat("tokens/sec", 500.0),
    "total_token_throughput": _stat("tokens/sec", 9000.0),
    "e2e_output_token_throughput": _stat("tokens/sec/user", 62.5),
    "time_to_first_token": _stat("ms", 300.0, 290.0, 600.0),
    "inter_chunk_latency": _stat("ms", 20.0, 19.0, 30.0),
    "inter_token_latency": _stat("ms", 8.0, 7.9, 9.5),
    "request_latency": _stat("ms", 8500.0, 8400.0, 9900.0),
}


def _ds_root(tmp_path, export):
    """A ds checkout whose dev/run_agentx.sh records its arguments and writes ``export``."""
    root = tmp_path / "ds"
    (root / "dev").mkdir(parents=True)
    source = tmp_path / "export.json"
    if export is not None:
        source.write_text(json.dumps(export))
    (root / "dev" / "run_agentx.sh").write_text(
        "#!/bin/bash\n"
        'mkdir -p "$4"\n'
        'echo "$@" > "$4/args.txt"\n'
        'echo "MODEL=$MODEL WARM=$AGENTX_WARMUP_PER_LANE" >> "$4/args.txt"\n'
        f'[ -f {source} ] && cp {source} "$4/profile_export_aiperf.json"\n'
        "exit 3\n"
    )
    return root


def _agentx_env(tmp_path, root, **extra):
    return {
        "PATH": os.environ["PATH"],
        "TRTLLM_TEST_DS_ROOT": str(root),
        "TRTLLM_PERF_DS_SHA": "e5e5da9",
        "TRTLLM_AGENTX_ARTIFACT_DIR": str(tmp_path / "agentx.0.0"),
        "AGENTX_DURATION": "60",
        **extra,
    }


AGENTX_ARGV = ["--model", "/models/m", "--concurrency", "8", "--host", "h", "--port", "9"]


def test_run_ds_agentx_reports_in_benchmark_serving_format(arms, tmp_path, capfd):
    root = _ds_root(tmp_path, EXPORT)
    env = _agentx_env(tmp_path, root, AIPERF_AGENTX_WARMUP_LEARN_REQUESTS="64")
    assert arms.run_ds_agentx(AGENTX_ARGV, env) == 0
    out = capfd.readouterr().out
    args = (tmp_path / "agentx.0.0" / "args.txt").read_text().splitlines()
    assert args == [f"http://h:9 8 60 {tmp_path / 'agentx.0.0'}", "MODEL=/models/m WARM=3"]
    arms.check_benchmark_output(out)
    assert "Total generated tokens: 30000" in out
    assert "Output token throughput (tok/s):         500.00" in out
    assert "P99 E2EL (ms):" in out
    settings = arms._client_settings(out)
    assert settings == {
        "client": "ds dev/run_agentx.sh",
        "ds_sha": "e5e5da9",
        "window_s": 60,
        "warmup_per_lane": 3,
        "extra_args": "",
        "dataset_requested": "semianalysis_cc_traces_weka_062126",
        "dataset_used": "semianalysis_cc_traces_weka_062126",
        "warmup_learning": {
            "AIPERF_AGENTX_WARMUP_LEARN_REQUESTS": "64",
            "AIPERF_AGENTX_WARMUP_LEARN_MAX_TOKENS": None,
            "AIPERF_AGENTX_WARMUP_LEARN_FREEZE_FILE": None,
        },
    }
    # The result carries the settings line (card ②).
    doc = _result(arms, tmp_path, "E3-chain", output=BENCHMARK_OUTPUT + out)
    assert doc["ds_client"] == settings
    assert arms.validate_disagg_result(doc) == []


def test_total_output_tokens_prefers_the_count(arms):
    export = {**EXPORT, "output_token_count": {"unit": "tokens", "sum": 29871}}
    assert arms._total_output_tokens(export) == (29871, "output_token_count.sum")
    assert arms._total_output_tokens({}) == (None, None)


@pytest.mark.parametrize(
    "export, setup, error, message",
    [
        (EXPORT, "no_root", "ArmError", "needs TRTLLM_TEST_DS_ROOT"),
        (EXPORT, "stale", "ArmError", "exists from an earlier run"),
        (None, None, "ArmError", "exited 3 and wrote no"),
        ({**EXPORT, "metadata": {"submission_valid": False}}, None, "RuntimeError", "submission_valid=false"),
    ],
)  # fmt: skip
def test_run_ds_agentx_refuses(arms, tmp_path, export, setup, error, message):
    root = _ds_root(tmp_path, export)
    env = _agentx_env(tmp_path, root)
    if setup == "no_root":
        env["TRTLLM_TEST_DS_ROOT"] = str(tmp_path / "missing")
    if setup == "stale":
        (tmp_path / "agentx.0.0").mkdir()
        (tmp_path / "agentx.0.0" / "profile_export_aiperf.json").write_text("{}")
    exc = arms.ArmError if error == "ArmError" else RuntimeError
    with pytest.raises(exc, match=message):
        arms.run_ds_agentx(AGENTX_ARGV, env)


def test_ds_agentx_command_line(arms, tmp_path):
    root = _ds_root(tmp_path, EXPORT)
    cmd = arms.arm_client_cmd(arms.DISAGG_ARMS["E3-native"], "agentx", AGENTX_CMD)
    env = _agentx_env(tmp_path, root)
    done = subprocess.run(
        cmd + ["--host", "h", "--port", "9"], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert arms.DS_CLIENT_LINE in done.stdout and "Total generated tokens: 30000" in done.stdout
    usage = subprocess.run(cmd[:2], env=env, capture_output=True, text=True, timeout=120)
    assert usage.returncode == 2 and "usage" in usage.stderr
