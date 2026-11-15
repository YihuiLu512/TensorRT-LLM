# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The disaggregated perf arms (perf/openengine_arms.py, TH-DSG-6) on CPU: table, selection, workers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

# name -> (topology, ctx (chain, front ends, postproc), gen (chain, front ends, postproc)).
EXPECTED_DISAGG_ARMS = {
    "E4-base": ("disagg", ("none", 1, None), ("none", 1, None)),
    "E4-chain": ("disagg", ("trtllm", 1, None), ("trtllm", 1, None)),
    "E4-ctx": ("disagg", ("trtllm", 1, None), ("none", 1, None)),
    "E4-gen": ("disagg", ("none", 1, None), ("trtllm", 1, None)),
    "E4-base-lean": ("disagg", ("none", 1, 0), ("none", 1, 0)),
    "E4-chain-lean": ("disagg", ("trtllm", 1, 0), ("trtllm", 1, 0)),
    "E4-direct": ("direct", None, None),
}
POC2_ARMS = ["A-def", "A-1f", "A-1f0", "B-s1", "B-full", "C-http", "C-s1", "C-full"]
WORKER_CONFIG = {
    "max_batch_size": 16,
    "kv_cache_config": {"enable_block_reuse": False},
    "cache_transceiver_config": {"backend": "DEFAULT", "transceiver_runtime": "PYTHON"},
    "num_postprocess_workers": 4,
    "stream_interval": 20,
}


def _role(settings):
    return (
        None
        if settings is None
        else (settings.chain, settings.frontends, settings.postprocess_workers)
    )


def test_disagg_arm_table(arms):
    assert list(arms.DISAGG_ARMS) == list(EXPECTED_DISAGG_ARMS)
    for name, (topology, ctx, gen) in EXPECTED_DISAGG_ARMS.items():
        arm = arms.DISAGG_ARMS[name]
        assert (arm.case, arm.topology, _role(arm.ctx), _role(arm.gen)) == (
            "case-04",
            topology,
            ctx,
            gen,
        )
        # The single-server settings are not used by a disaggregated arm.
        assert (arm.glm_http, arm.frontends, arm.postprocess_workers) == (False, None, None)


def test_poc2_arms_are_unchanged(arms):
    """ARMS still holds PoC-S2's eight single-server arms, each with the defaults of the new fields."""
    assert list(arms.ARMS) == POC2_ARMS
    for arm in arms.ARMS.values():
        assert (arm.topology, arm.ctx, arm.gen) == ("agg", None, None)
    assert set(arms.all_arms()) == set(POC2_ARMS) | set(EXPECTED_DISAGG_ARMS)


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, None),
        ({"TRTLLM_PERF_ARM": "E4-base"}, "E4-base"),
        ({"TRTLLM_PERF_ARM": "E4-base-lean"}, "E4-base-lean"),
        ({"TRTLLM_PERF_ARM": "E4-chain", "TRTLLM_TEST_OPENENGINE_CHAIN": "trtllm"}, "E4-chain"),
        ({"TRTLLM_PERF_ARM": "E4-ctx", "TRTLLM_TEST_OPENENGINE_CHAIN": "trtllm"}, "E4-ctx"),
        ({"TRTLLM_PERF_ARM": "A-def"}, "A-def"),
    ],
)
def test_selected_arm(arms, env, expected):
    arm = arms.selected_arm(env)
    assert (arm.name if arm else None) == expected


@pytest.mark.parametrize(
    "env, message",
    [
        ({"TRTLLM_PERF_ARM": "E4-chain"}, "needs TRTLLM_TEST_OPENENGINE_CHAIN set to 'trtllm'"),
        ({"TRTLLM_PERF_ARM": "E4-base", "TRTLLM_TEST_OPENENGINE_CHAIN": "trtllm"}, "unset"),
        ({"TRTLLM_PERF_ARM": "E5"}, "E4-chain"),
    ],
)
def test_selected_arm_refuses(arms, env, message):
    with pytest.raises(arms.ArmError, match=message):
        arms.selected_arm(env)


@pytest.mark.parametrize("name", [n for n, e in EXPECTED_DISAGG_ARMS.items() if e[0] == "disagg"])
def test_role_view_takes_only_the_role_settings(arms, name):
    arm = arms.DISAGG_ARMS[name]
    for role in ("ctx", "gen"):
        view = arms.role_view(arm, role)
        settings = arms.role_arm(arm, role)
        assert (view.chain, view.frontends, view.postprocess_workers) == _role(settings)
        assert view.topology == "agg" and view.in_process == settings.in_process
        changed = arms.apply_arm(WORKER_CONFIG, view)
        assert {k for k in changed if changed[k] != WORKER_CONFIG.get(k)} <= set(arms.ARM_SETTINGS)


def test_role_arm_of_the_direct_arm_is_refused(arms):
    with pytest.raises(arms.ArmError, match="no 'ctx' workers"):
        arms.role_arm(arms.DISAGG_ARMS["E4-direct"], "ctx")


@pytest.mark.parametrize("serving_type, expected", [("CTX_0", ("ctx", 0)), ("GEN_12", ("gen", 12))])
def test_worker_role(arms, serving_type, expected):
    assert arms.worker_role(serving_type) == expected


@pytest.mark.parametrize("serving_type", ["BENCHMARK", "DISAGG_SERVER", "CTX_MPI", "GEN"])
def test_worker_role_refuses_other_roles(arms, serving_type):
    with pytest.raises(arms.ArmError):
        arms.worker_role(serving_type)


def _case(arms, **changes):
    values = dict(
        runtime="multi_node_disagg_server",
        benchmark_mode="e2e",
        time_breakdown=False,
        numa_bind=False,
        dwdp_size=0,
        worker_gpus={"ctx": 1, "gen": 1},
        gpus_per_node=8,
    )
    values.update(changes)
    return values


@pytest.mark.parametrize("mode", ["e2e", "gen_only"])
def test_check_disagg_case_accepts(arms, mode):
    arms.check_disagg_case(arms.DISAGG_ARMS["E4-chain"], **_case(arms, benchmark_mode=mode))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"runtime": "aggr_server"}, "disaggregated cases only"),
        ({"benchmark_mode": "ctx_only"}, "benchmark modes"),
        ({"benchmark_mode": "gen_only_no_context"}, "benchmark modes"),
        ({"benchmark_mode": None}, "benchmark modes"),
        ({"time_breakdown": True}, "time_breakdown"),
        ({"dwdp_size": 4}, "DWDP"),
        ({"numa_bind": True}, "numa_bind"),
        ({"worker_gpus": {"ctx": 1, "gen": 16}}, "more than one node"),
    ],
)
def test_check_disagg_case_refuses(arms, changes, message):
    with pytest.raises(arms.ArmError, match=message):
        arms.check_disagg_case(arms.DISAGG_ARMS["E4-chain"], **_case(arms, **changes))


def test_check_disagg_case_refuses_the_direct_arm(arms):
    with pytest.raises(arms.ArmError, match="direct replay"):
        arms.check_disagg_case(arms.DISAGG_ARMS["E4-direct"], **_case(arms))


def _worker_cmd(tmp_path, *extra):
    config = tmp_path / "worker.yaml"
    config.write_text(yaml.safe_dump(WORKER_CONFIG))
    return ["trtllm-serve", "/models/m", "--backend", "pytorch", "--config", str(config), *extra]


@pytest.mark.parametrize(
    "name, role, server_role",
    [
        ("E4-chain", "ctx", "context"),
        ("E4-chain", "gen", "generation"),
        ("E4-ctx", "ctx", "context"),
        ("E4-ctx", "gen", None),
        ("E4-base", "ctx", None),
        ("E4-chain-lean", "gen", "generation"),
    ],
)
def test_prepare_worker(arms, tmp_path, name, role, server_role):
    arm = arms.DISAGG_ARMS[name]
    cmd = _worker_cmd(tmp_path)
    plan = arms.prepare_worker(cmd, arm, role, str(tmp_path / "out"))
    tokens = list(plan.cmd)
    if server_role is None:
        assert "--server_role" not in tokens
    else:
        assert tokens[-2:] == ["--server_role", server_role]
        tokens = tokens[:-2]
    index = cmd.index("--config") + 1
    assert tokens[:index] + tokens[index + 1 :] == cmd[:index] + cmd[index + 1 :]
    written = yaml.safe_load(Path(tokens[index]).read_text())
    settings = arms.role_arm(arm, role)
    assert written["num_serve_frontends"] == settings.frontends
    expected_workers = 4 if settings.postprocess_workers is None else settings.postprocess_workers
    assert written["num_postprocess_workers"] == expected_workers
    assert written["cache_transceiver_config"] == WORKER_CONFIG["cache_transceiver_config"]
    assert arms.worker_needs_port(arms.role_view(arm, role), plan) == (server_role is not None)


def test_prepare_worker_refuses_a_command_line_role(arms, tmp_path):
    cmd = _worker_cmd(tmp_path, "--server_role", "context")
    with pytest.raises(arms.ArmError, match="--server_role"):
        arms.prepare_worker(cmd, arms.DISAGG_ARMS["E4-base"], "ctx", str(tmp_path / "out"))


def test_parallel_gpus(arms):
    assert arms.parallel_gpus({}) == 1
    assert arms.parallel_gpus({"tensor_parallel_size": 4, "pipeline_parallel_size": 2}) == 8
    assert arms.parallel_gpus({"tensor_parallel_size": "2", "context_parallel_size": 2}) == 4


def test_kv_transfer_env_keeps_the_transfer_and_fill_gate_variables(arms):
    env = {
        "UCX_TLS": "cuda_ipc,tcp",
        "NIXL_LOG_LEVEL": "info",
        "TRTLLM_USE_NIXL_KVCACHE": "1",
        "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP": "1",
        "TLLM_BENCHMARK_REQ_QUEUES_SIZE": "16",
        "CUDA_VISIBLE_DEVICES": "0",
        "PATH": "/usr/bin",
    }
    assert arms.kv_transfer_env(env) == {
        k: env[k] for k in sorted(env) if k not in ("CUDA_VISIBLE_DEVICES", "PATH")
    }


# ----------------------------------------------------------------------------- gen_server_<i>.log


def test_log_mirror_copies_what_the_source_gains(arms, tmp_path):
    source, dest = tmp_path / "engine.log", tmp_path / "gen_server_0.log"
    current = {"path": None}
    mirror = arms.LogMirror(lambda: current["path"], str(dest))
    assert mirror.pump() == 0 and not dest.exists()
    current["path"] = str(source)
    assert mirror.pump() == 0, "a source that does not exist yet copies nothing"
    source.write_bytes(b"iter 1\n")
    assert mirror.pump() == 7
    with source.open("ab") as f:
        f.write(b"prev_device_step_time 3.5\n")
    mirror.pump()
    assert dest.read_bytes() == b"iter 1\nprev_device_step_time 3.5\n"
    # A new source (a restarted engine's log) is copied from its start.
    other = tmp_path / "engine-2.log"
    other.write_bytes(b"restart\n")
    current["path"] = str(other)
    mirror.pump()
    assert dest.read_bytes().endswith(b"restart\n") and mirror.copied == len(dest.read_bytes())


@pytest.mark.parametrize(
    "name, env, expected",
    [
        ("E4-chain", {}, True),
        ("E4-base", {}, False),
        ("E4-base", {"TRTLLM_PERF_SINGLE_NODE": "1"}, True),
        ("E4-ctx", {}, False),
    ],
)
def test_mirrors_gen_log(arms, name, env, expected):
    view = arms.role_view(arms.DISAGG_ARMS[name], "gen")
    assert arms.mirrors_gen_log(view, env) is expected


def test_gen_log_source(arms):
    chained = SimpleNamespace(handle=SimpleNamespace(engine_log="/e.log"))
    assert arms.gen_log_source(chained, "/server.log")() == "/e.log"
    in_process = SimpleNamespace(handle=None)
    assert arms.gen_log_source(in_process, "/server.log")() == "/server.log"


def test_count_spawned(arms):
    lines = {
        1: "python -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5)",
        2: "python -c from multiprocessing.resource_tracker import main;main(4)",
        3: "trtllm-serve m",
        4: None,
    }
    assert arms.count_spawned(lines, cmdline=lines.get) == 1


# ----------------------------------------------------------------------------- per-role readings


class _FakeTicks:
    def __init__(self):
        self.ticks = {100: 1000, 101: 500}

    def __call__(self, pid):
        return None if pid not in self.ticks else (7, self.ticks[pid])


def _reader(arms, view, tmp_path, *, env=None, config=None):
    ticks = _FakeTicks()
    cpu = arms.CpuReader(
        tree=lambda root: {root, root + 1} if root == 100 else {root},
        gpu=lambda: {101},
        ticks=ticks,
        now=lambda: 1,
    )
    plan = arms.ServerPlan(("trtllm-serve", "m"), dict(config or WORKER_CONFIG))
    server = SimpleNamespace(handle=None, frontend_pid=100, proc=None, arm=view)
    reader = arms.RoleReader(
        server=server,
        view=view,
        role="gen",
        worker="GEN_0",
        directory=str(tmp_path),
        address=lambda: "127.0.0.1:9",
        plan=plan,
        env=env or {},
        launched_at=0.0,
        reader=cpu,
    )
    reader._frontends_started = lambda notes: 1
    return reader, ticks


def test_role_reader_answers_each_request_once(arms, tmp_path):
    view = arms.role_view(arms.DISAGG_ARMS["E4-base"], "gen")
    env = {"UCX_TLS": "tcp", "TLLM_BENCHMARK_REQ_QUEUES_SIZE": "16", "HOME": "/h"}
    reader, ticks = _reader(arms, view, tmp_path, env=env)
    arms.write_json_atomic(arms.request_path(str(tmp_path), "ab12", 0, "before"), {})
    reader.poll()
    before = arms.reading_path(str(tmp_path), "ab12", 0, "before", "GEN_0")
    doc = yaml.safe_load(Path(before).read_text())
    assert (doc["worker"], doc["role"], doc["chain"], doc["phase"]) == (
        "GEN_0",
        "gen",
        "none",
        "before",
    )
    assert doc["frontends"] == {"configured": 1, "started": 1}
    assert doc["postproc_workers"] == {"configured": 4, "started": 0}
    assert doc["kv_transfer"] == {
        "backend": "DEFAULT",
        "runtime": "PYTHON",
        "env": {"TLLM_BENCHMARK_REQ_QUEUES_SIZE": "16", "UCX_TLS": "tcp"},
    }
    assert doc["alignment"]["enable_block_reuse"] is False and doc["stream_interval"] == 20
    Path(before).unlink()
    reader.poll()
    assert not Path(before).exists(), "a request is answered once"
    ticks.ticks[100] += 250
    arms.write_json_atomic(arms.request_path(str(tmp_path), "ab12", 0, "after"), {})
    reader.poll()
    after = yaml.safe_load(
        Path(arms.reading_path(str(tmp_path), "ab12", 0, "after", "GEN_0")).read_text()
    )
    assert after["frontend_cpu_s"] == pytest.approx(250 / reader.reader.ticks_per_s)
    assert after["leg"] is None, "an in-process worker has no leg of the chain"


def test_role_reader_after_without_before_is_noted(arms, tmp_path):
    view = arms.role_view(arms.DISAGG_ARMS["E4-base"], "gen")
    reader, _ = _reader(arms, view, tmp_path)
    doc = reader.read("cd34", 3, "after")
    assert doc["frontend_cpu_s"] is None
    assert any("no reading of this worker before" in n for n in doc["notes"])


def test_probe_waits_for_every_worker_and_notes_the_missing(arms, tmp_path):
    directory = str(tmp_path)
    for worker in ("CTX_0", "GEN_0"):
        arms.write_json_atomic(
            arms.reading_path(directory, "s1", 0, "before", worker), {"worker": worker}
        )
    probe = arms.DisaggRunProbe.start(
        arms.DISAGG_ARMS["E4-base"],
        directory=directory,
        workers=["CTX_0", "GEN_0", "GEN_1"],
        orchestrator_url="http://127.0.0.1:1",
        seq=0,
        session="s1",
        timeout=0.5,
    )
    assert set(probe.readings["before"]) == {"CTX_0", "GEN_0"}
    assert any("no before reading of GEN_1" in n for n in probe.notes)
    assert Path(arms.request_path(directory, "s1", 0, "before")).exists()
    assert probe.orchestrator["before"] is None, "an unreachable server is noted, not raised"
    assert "before" in probe.window


BENCHMARK_OUTPUT = """\
Successful requests:                     16
Total generated tokens:                  512
Output token throughput (tok/s):         159.50
Mean TTFT (ms):                          21.50
Median TTFT (ms):                        20.10
P99 TTFT (ms):                           40.25
Mean TPOT (ms):                          5.10
Median TPOT (ms):                        5.00
P99 TPOT (ms):                           6.75
Mean ITL (ms):                           5.20
Median ITL (ms):                         5.05
P99 ITL (ms):                            7.00
Mean E2EL (ms):                          180.00
Median E2EL (ms):                        175.50
P99 E2EL (ms):                           220.10
Average Per Iter Device Step Time (ms):  4.25
Median Per Iter Device Step Time (ms):   4.00
"""
CLIENT = [
    "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving", "--model", "m",
    "--max-concurrency", "4", "--dataset-name", "random", "--random-ids",
    "--tokenize-on-client", "--random-input-len", "128",
]  # fmt: skip
ORCHESTRATOR = {
    "hostname": "localhost",
    "port": 8000,
    "schedule_style": "context_first",
    "num_workers": 1,
    "gen_strip_message_history": False,
    "context_servers": {
        "num_instances": 1,
        "urls": ["10.0.0.1:8001"],
        "router": {"type": "round_robin"},
    },
    "generation_servers": {
        "num_instances": 1,
        "urls": ["10.0.0.2:8002"],
        "router": {"type": "load_balancing"},
    },
}


def _reading(arms, worker, *, ucx="tcp", cpu=1.5):
    role = arms.worker_role(worker)[0]
    return {
        "worker": worker,
        "role": role,
        "node": "node-a",
        "startup_s": 30.0,
        "gpus": 1,
        "frontends": {"configured": 1, "started": 1},
        "postproc_workers": {"configured": 4, "started": 4},
        "stream_interval": 20,
        "kv_transfer": {"backend": "NIXL", "runtime": "PYTHON", "env": {"UCX_TLS": ucx}},
        "alignment": {"enable_block_reuse": False, "return_perf_metrics": None},
        "frontend_cpu_s": cpu,
        "leg": None,
        "notes": [],
    }


def _probe(arms, name, tmp_path, **reading_changes):
    probe = arms.DisaggRunProbe(
        arms.DISAGG_ARMS[name], str(tmp_path), ("CTX_0", "GEN_0"), "http://127.0.0.1:1", "s1", 0
    )
    for phase in ("before", "after"):
        probe.readings[phase] = {w: _reading(arms, w, **reading_changes) for w in probe.workers}
        probe.orchestrator[phase] = {"ctx.total_requests": 0 if phase == "before" else 16}
        probe.client_cpu[phase] = 1.0 if phase == "before" else 3.5
    probe.window = {"before": 1_000, "after": 2_000_000_000_000}
    return probe


def _result(arms, probe, tmp_path, mode="e2e"):
    config = tmp_path / "disagg.yaml"
    config.write_text(yaml.safe_dump(ORCHESTRATOR))
    return arms.build_disagg_result(
        probe,
        output=BENCHMARK_OUTPUT,
        client_cmd=CLIENT,
        test_id="disagg-e2e-b200_x",
        benchmark_mode=mode,
        orchestrator_config=str(config),
        orchestrator_env={"TRTLLM_DISAGG_NO_RETRY": "1"},
        environ={"TRTLLM_PERF_RUN": "2", "SLURM_JOB_ID": "123"},
    )


@pytest.mark.parametrize(
    "name", ["E4-base", "E4-base-lean", "E4-chain", "E4-ctx", "E4-gen", "E4-chain-lean"]
)
def test_result_matches_schema(arms, tmp_path, name):
    doc = _result(arms, _probe(arms, name, tmp_path), tmp_path)
    assert arms.validate_disagg_result(doc) == []
    assert (doc["case"], doc["arm"], doc["topology"]) == ("case-04", name, "disagg")
    assert (doc["run"], doc["job_id"], doc["concurrency"], doc["dataset"]) == (
        2,
        "123",
        4,
        "token_ids",
    )
    assert doc["metrics"]["gen_device_step_ms"] == {"mean": 4.25, "median": 4.0}
    assert doc["metrics"]["ttft_ms"] == {"mean": 21.5, "median": 20.1, "p99": 40.25}
    for role in ("ctx", "gen"):
        block = doc["roles"][role]
        assert block["chain"] == arms.role_arm(arms.DISAGG_ARMS[name], role).chain
        assert block["instances"] == 1 and block["gpus_per_instance"] == 1
        assert block["frontends"] == {"configured": 1, "started": 1}
        assert block["postproc_workers"] == {"configured": 4, "started": 4}
        assert block["frontend_cpu_s"] == 1.5 and block["startup_s"] == 30.0
        assert block["frontend_cpu_us_per_token"] == pytest.approx(1.5e6 / 512)
    assert doc["orch"]["router"] == {"ctx": "round_robin", "gen": "load_balancing"}
    assert doc["orch"]["schedule_style"] == "context_first"
    assert doc["client"]["cpu_s"] == 2.5
    assert doc["kv_transfer"] == {
        "backend": {"ctx": "NIXL", "gen": "NIXL"},
        "runtime": {"ctx": "PYTHON", "gen": "PYTHON"},
        "env": {"ctx": {"UCX_TLS": "tcp"}, "gen": {"UCX_TLS": "tcp"}},
    }
    assert doc["alignment"]["disagg_no_retry"] is True
    assert doc["alignment"]["enable_block_reuse"] == {"ctx": False, "gen": False}
    chained = [r for r in ("ctx", "gen") if not arms.role_arm(arms.DISAGG_ARMS[name], r).in_process]
    if not chained:
        assert doc["chain_proof"] is None and doc["first_output"] is None
        return
    proof = doc["chain_proof"]
    for role in ("ctx", "gen"):
        assert (proof[role] is None) == (role not in chained)
        assert (doc["first_output"][role] is None) == (role not in chained)
    # No worker judged its leg: the proof is incomplete, never ok.
    assert proof["status"]["overall"] == "incomplete"
    assert proof["leg_expectation"] == (
        "cross_node" if len(chained) == 2 else f"one_leg:{chained[0]}"
    )
    assert proof["orch_requests"]["ctx"] == 16


def test_result_shows_differing_transfer_settings_per_role(arms, tmp_path):
    probe = _probe(arms, "E4-chain", tmp_path)
    probe.readings["after"]["GEN_0"]["kv_transfer"]["env"] = {"UCX_TLS": "cuda_ipc"}
    doc = _result(arms, probe, tmp_path)
    assert doc["kv_transfer"]["env"] == {"ctx": {"UCX_TLS": "tcp"}, "gen": {"UCX_TLS": "cuda_ipc"}}


def test_result_written_and_not_overwritten(arms, tmp_path):
    doc = _result(arms, _probe(arms, "E4-chain", tmp_path), tmp_path)
    path = arms.write_disagg_result(doc, str(tmp_path / "out"))
    assert yaml.safe_load(Path(path).read_text()) == doc
    assert arms.write_disagg_result(doc, str(tmp_path / "out")) != path


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d.pop("first_output"), "keys"),
        (lambda d: d.update(arm="A-def"), "not a disaggregated or direct arm"),
        (lambda d: d["roles"]["ctx"].update(chain="none"), "roles.ctx.chain"),
        (lambda d: d["roles"]["gen"]["frontends"].pop("started"), "configured and started"),
        (lambda d: d["chain_proof"]["ctx"].update(generate_sent=-1), "non-negative"),
        (lambda d: d["chain_proof"]["status"].update(overall="good"), "status values"),
        (lambda d: d["first_output"]["gen"].update(calls=-1), "non-negative"),
        (lambda d: d["first_output"]["gen"]["rest"].pop("p99_ms"), "first_output.gen.rest"),
        (lambda d: d["alignment"].pop("input_form"), "alignment"),
        (lambda d: d["metrics"].pop("gen_device_step_ms"), "metrics"),
        (lambda d: d.update(nodes=[]), "nodes"),
    ],
)
def test_invalid_results_are_rejected(arms, tmp_path, change, message):
    doc = _result(arms, _probe(arms, "E4-chain", tmp_path), tmp_path)
    change(doc)
    errors = arms.validate_disagg_result(doc)
    assert any(message in e for e in errors), errors
    with pytest.raises(arms.ArmError):
        arms.write_disagg_result(doc, str(tmp_path / "out"))


def test_in_process_arm_must_have_null_proof(arms, tmp_path):
    doc = _result(arms, _probe(arms, "E4-base", tmp_path), tmp_path)
    doc["first_output"] = {"directory": "x", "ctx": None, "gen": None}
    assert any("first_output must be null" in e for e in arms.validate_disagg_result(doc))


def test_static_members(arms, tmp_path):
    config = tmp_path / "disagg.yaml"
    config.write_text(yaml.safe_dump(ORCHESTRATOR))
    assert arms.static_members(str(config)) == {"ctx": ["10.0.0.1:8001"], "gen": ["10.0.0.2:8002"]}
    assert arms.static_members(str(tmp_path / "missing.yaml")) is None


# ----------------------------------------------------------------------------- first-output timestamps


def _write_probe(directory: Path, probe: str, records):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{probe}-host-1.jsonl").open("a") as f:
        for request_id, t_ns in records:
            f.write(f'{{"probe": "{probe}", "request_id": "{request_id}", "t_ns": {t_ns}}}\n')
        f.write("not json\n")


def test_first_output_gaps(arms):
    ms = 1_000_000
    records = {
        arms.ENGINE_FIRST_OUTPUT_PROBE: {
            "a": [10 * ms], "b": [11 * ms], "c": [12 * ms], "d": [13 * ms],
            "early": [1 * ms], "twice": [14 * ms, 15 * ms], "lost": [16 * ms],
        },
        arms.FRONTEND_FIRST_OUTPUT_PROBE: {
            "a": [11 * ms], "b": [13 * ms], "c": [15 * ms], "d": [17 * ms], "early": [2 * ms],
            "twice": [20 * ms],
        },
    }  # fmt: skip
    gaps = arms.first_output_gaps(records, (5 * ms, 20 * ms), first_wave=2)
    assert (gaps["calls"], gaps["unmatched"]) == (4, 1)
    assert gaps["first_wave"] == {"count": 2, "median_ms": 2.0, "p99_ms": 2.0, "max_ms": 2.0}
    assert gaps["rest"] == {"count": 2, "median_ms": 4.0, "p99_ms": 4.0, "max_ms": 4.0}
    assert arms.first_output_gaps(records, None, 2)["calls"] == 0


def test_first_output_block_reads_each_chained_worker(arms, tmp_path):
    probe = _probe(arms, "E4-gen", tmp_path)
    probe.window = {"before": 0, "after": 10**15}
    directory = Path(arms.first_output_dir(str(tmp_path), "GEN_0"))
    _write_probe(directory, arms.ENGINE_FIRST_OUTPUT_PROBE, [("r1", 1_000_000), ("r2", 2_000_000)])
    _write_probe(
        directory, arms.FRONTEND_FIRST_OUTPUT_PROBE, [("r1", 1_500_000), ("r2", 4_000_000)]
    )
    block = arms.first_output_block(probe, first_wave=1)
    assert block["ctx"] is None
    assert block["gen"]["calls"] == 2
    assert block["gen"]["first_wave"]["median_ms"] == 0.5 and block["gen"]["rest"]["max_ms"] == 2.0
    assert arms.first_output_env(
        arms.role_view(arms.DISAGG_ARMS["E4-gen"], "gen"), "/d", "GEN_0"
    ) == {arms.FIRST_OUTPUT_DIR_ENV: arms.first_output_dir("/d", "GEN_0")}
    assert (
        arms.first_output_env(arms.role_view(arms.DISAGG_ARMS["E4-gen"], "ctx"), "/d", "CTX_0")
        == {}
    )


def test_first_output_names_match_the_server(arms):
    """The names repeat tensorrt_llm/grpc/openengine/servicer.py's, which needs TensorRT-LLM to import."""
    servicer = (
        Path(arms.__file__).resolve().parents[4] / "tensorrt_llm/grpc/openengine/servicer.py"
    ).read_text()
    for name in (
        arms.FIRST_OUTPUT_DIR_ENV,
        arms.ENGINE_FIRST_OUTPUT_PROBE,
        arms.FRONTEND_FIRST_OUTPUT_PROBE,
    ):
        assert f'"{name}"' in servicer


# ----------------------------------------------------------------------------- UV-44 on the harness


def test_chain_engine_gets_the_fill_gate_and_timestamp_variables(harness_path):
    """A generation worker in the chain: its engine gets the gen_only fill gate, both get the timestamps."""
    from openengine_chain import split

    env = {
        "TLLM_BENCHMARK_REQ_QUEUES_SIZE": "16",
        "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP": "1",
        "TRTLLM_OPENENGINE_FIRST_OUTPUT_DIR": "/d/GEN_0",
    }
    args = ["trtllm-serve", "m", "--server_role", "generation"]
    engine = split.disagg_engine_env(env, args)
    assert {k: engine[k] for k in env} == env
    frontend = split.disagg_frontend_env(env)
    assert frontend["TRTLLM_OPENENGINE_FIRST_OUTPUT_DIR"] == "/d/GEN_0"
