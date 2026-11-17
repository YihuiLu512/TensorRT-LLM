# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gen-only modes' warmup pass under the disaggregated arms, and the readings it touches, on CPU.

perf/openengine_arms.py gives gen_only and gen_only_no_context a warmup pass
under a disaggregated arm (perf sanity sends none in them otherwise), counts the
post-processing workers an in-process server's engine forks, and splits the
front end's CPU time of a run over its processes. The single-node driver gives
each worker role a torch profiler trace path of its own.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[5]
DEFS = REPO / "tests" / "integration" / "defs"
PERF_SANITY = DEFS / "perf" / "test_perf_sanity.py"
GEN_ONLY_MODES = ("gen_only", "gen_only_no_context")
CHAIN_ENV = "TRTLLM_TEST_OPENENGINE_CHAIN"


def _arm_env(name: Optional[str]) -> Dict[str, str]:
    """The environment that selects arm ``name`` (none for None), with its chain."""
    if name is None:
        return {}
    env = {"TRTLLM_PERF_ARM": name}
    if name.startswith("E4-chain"):
        env[CHAIN_ENV] = "trtllm"
    return env


@pytest.fixture
def arm_env(monkeypatch):
    """Set the arm (and chain) of the run in os.environ; returns the setter."""

    def select(name: Optional[str], **extra: str) -> None:
        for key in ("TRTLLM_PERF_ARM", CHAIN_ENV, "TRTLLM_PERF_RUN", "TRTLLM_PERF_WARMUP_PASS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in {**_arm_env(name), **extra}.items():
            monkeypatch.setenv(key, value)

    return select


@pytest.fixture(scope="module")
def warmup_pass():
    name = "wt70_fix2_warmup_pass_under_test"
    spec = importlib.util.spec_from_file_location(name, DEFS / "perf" / "warmup_pass.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def _perf_sanity_functions(namespace, *names, owner=None):
    """Functions of test_perf_sanity.py (methods of class ``owner``), compiled from its source.

    Importing test_perf_sanity needs TensorRT-LLM and the integration conftest;
    ``namespace`` stands in for the module globals the functions use.
    """
    body = ast.parse(PERF_SANITY.read_text()).body
    if owner is not None:
        (cls,) = [n for n in body if isinstance(n, ast.ClassDef) and n.name == owner]
        body = cls.body
    nodes = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(n.name for n in nodes) == sorted(names)
    namespace.setdefault("Optional", Optional)
    namespace.setdefault("List", List)
    namespace.setdefault("Tuple", Tuple)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PERF_SANITY), "exec"), namespace)
    return [namespace[name] for name in names]


def _plan_functions(arms, warmup_pass):
    """test_perf_sanity's warmup_pass_requests_for and warmup_pass_plan_for, on the real modules."""
    namespace = {
        "openengine_arms": arms,
        "warmup_pass_enabled": warmup_pass.warmup_pass_enabled,
        "warmup_pass_applies": warmup_pass.warmup_pass_applies,
        "warmup_pass_plan": warmup_pass.warmup_pass_plan,
    }
    return _perf_sanity_functions(namespace, "warmup_pass_requests_for", "warmup_pass_plan_for")


def _client(concurrency=16, **changes):
    values = dict(
        concurrency=concurrency,
        warmup_pass_requests=concurrency,
        only_run_accuracy=False,
        warmup=False,
        num_requests=concurrency,
        to_env=lambda: {},
        accuracy_config=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


# ----------------------------------------------------------------------------- which runs get a pass


@pytest.mark.parametrize("name", ["E4-base", "E4-chain", "E4-base-lean", "E4-chain-lean"])
@pytest.mark.parametrize("mode", GEN_ONLY_MODES)
def test_the_gen_only_modes_get_a_pass_under_a_disaggregated_arm(arms, name, mode):
    assert arms.ARM_WARMUP_MODES == GEN_ONLY_MODES
    assert arms.arm_warmup_applies(mode, _arm_env(name))


@pytest.mark.parametrize("mode", GEN_ONLY_MODES)
def test_the_gen_only_modes_get_no_pass_without_a_disaggregated_arm(arms, mode):
    assert not arms.arm_warmup_applies(mode, {})
    # PoC-S2's arms are aggregated: no pass from this rule either.
    for name, arm in arms.ARMS.items():
        env = {"TRTLLM_PERF_ARM": name, **({CHAIN_ENV: arm.chain} if not arm.in_process else {})}
        assert not arms.arm_warmup_applies(mode, env), name


@pytest.mark.parametrize("mode", ["", None, "e2e", "ctx_only", "gen_first"])
def test_other_modes_are_left_to_perf_sanity(arms, mode):
    assert not arms.arm_warmup_applies(mode, _arm_env("E4-chain"))


def test_an_inconsistent_arm_is_refused_only_where_it_matters(arms):
    env = {CHAIN_ENV: "trtllm"}
    assert not arms.arm_warmup_applies("e2e", env)
    with pytest.raises(arms.ArmError):
        arms.arm_warmup_applies("gen_only", env)


@pytest.mark.parametrize(
    "mode, arm, extra, expected",
    [
        # perf sanity's own modes: unchanged, arm or not.
        ("e2e", None, {}, 16),
        ("e2e", "E4-chain", {}, 16),
        ("ctx_only", "E4-base", {}, 16),
        ("", None, {}, 16),
        # The gen-only modes: a pass only under a disaggregated arm.
        ("gen_only", None, {}, 0),
        ("gen_only_no_context", None, {}, 0),
        ("gen_only", "E4-base", {}, 16),
        ("gen_only", "E4-chain", {}, 16),
        ("gen_only_no_context", "E4-base", {}, 16),
        ("gen_only_no_context", "E4-chain", {}, 16),
        # TRTLLM_PERF_WARMUP_PASS=0 still turns every pass off.
        ("gen_only", "E4-chain", {"TRTLLM_PERF_WARMUP_PASS": "0"}, 0),
        ("gen_only_no_context", "E4-base", {"TRTLLM_PERF_WARMUP_PASS": "off"}, 0),
        ("e2e", None, {"TRTLLM_PERF_WARMUP_PASS": "0"}, 0),
    ],
)
def test_warmup_pass_requests_for(arms, warmup_pass, arm_env, mode, arm, extra, expected):
    requests_for, _ = _plan_functions(arms, warmup_pass)
    arm_env(arm, **extra)
    assert requests_for(_client(), mode) == expected
    assert requests_for(None, mode) == 0
    # An accuracy-only client and a client of another load generator get none.
    assert requests_for(_client(warmup_pass_requests=0), mode) == 0


def test_the_plan_gives_each_measured_client_one_wave(arms, warmup_pass, arm_env):
    _, plan_for = _plan_functions(arms, warmup_pass)
    arm_env("E4-chain")
    clients = [_client(16), _client(8), None]
    assert plan_for(clients, "gen_only", False) == [16, 8, 0]
    arm_env(None)
    assert plan_for(clients, "gen_only", False) == [0, 0, 0]


# ----------------------------------------------------------------------------- gen_only_no_context's run


class _GenOnlyNoContextRun:
    """AggrGenOnlyNoContextCmds.run_cmd compiled from source, every collaborator a recording fake.

    ``events`` lists what happened in order. The gen worker's log grows by one
    line per request the gen worker served, so a log offset tells how many
    requests had been served when it was taken.
    """

    def __init__(self, arms, warmup_pass, tmp_path, *, arm_in_process=False, clients=1):
        self.events: List[str] = []
        self.served = 0
        self.pending: List[dict] = []
        events = self.events
        run = self

        class ArmRun:
            def finish(self, **kwargs):
                events.append(f"finish(served={run.served})")
                return "result.json"

        class GenProc:
            pid = 7

            def start_run(self):
                events.append(f"start_run(served={run.served})")
                return ArmRun()

            def terminate(self):
                events.append("terminate gen")

            def wait(self):
                return 0

        class Pump:
            def pump(self):
                events.append("pump")

            def stop(self):
                events.append("pump stop")

        view = SimpleNamespace(name="E4-chain:gen", in_process=arm_in_process)
        fake_arms = SimpleNamespace(
            disagg_arm=arms.disagg_arm,
            selected_arm=arms.selected_arm,
            arm_warmup_applies=arms.arm_warmup_applies,
            ROLE_GEN=arms.ROLE_GEN,
            role_view=lambda arm, role: view,
            prepare_worker=lambda cmd, arm, role, out: SimpleNamespace(cmd=tuple(cmd), config={}),
            worker_frontends=lambda plan: 1,
            worker_needs_port=lambda view, plan: True,
            free_port=lambda host: 4321,
            current_test_id=lambda: "perf/test_perf_sanity.py::test_e2e[aggr-gen_only_no_context-x]",
        )

        def benchmark(cmd, env, log_path, timeout=None):
            events.append(f"benchmark({Path(log_path).name})")
            run.served += 16
            return f"measured output of {Path(log_path).name}"

        def warmup(client_cmd, client_config, num_requests, host, port, log_path, timeout):
            events.append(f"warmup({num_requests}, {host}:{port}, {Path(log_path).name})")
            run.served += num_requests
            return num_requests

        def append_steps(test_output_dir, num_gen_servers, pending, outputs):
            run.pending = pending

        namespace = {
            "openengine_arms": fake_arms,
            "os": __import__("os"),
            "copy": __import__("copy"),
            "subprocess": None,
            "GEN_ONLY_NO_CONTEXT_ENV": "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY",
            "GEN_ONLY_NO_CONTEXT_MODE": "gen_only_no_context",
            "DEFAULT_TIMEOUT": 10800,
            "DISAGG_SERVER_READY_TIMEOUT": 3600,
            "add_host_port_to_cmd": lambda cmd, host, port: list(cmd) + ["--host", host],
            "publish_addr_file": lambda path, host, port: None,
            "reserve_multi_frontend_port": lambda host: 1,
            "print_info": lambda *a, **k: None,
            "wait_for_reported_addr": lambda path, timeout, proc=None: ("localhost", 8000),
            "to_env_dict": lambda value: {},
            "wait_for_endpoint_ready": lambda *a, **k: events.append("ready"),
            "server_ready_timeout": lambda default, mode: default,
            "collect_startup_observation": lambda *a, **k: {},
            "write_startup_observations": lambda *a, **k: None,
            "gen_worker_log_sizes": lambda out, n: [run.served],
            "_run_benchmark_with_log": benchmark,
            "run_warmup_pass": warmup,
            "warmup_pass_log_path": lambda out,
            s,
            c: f"{out}/trtllm-benchmark-warmup-pass.{s}.{c}.log",
            "expected_gen_only_ngen": lambda cfg, concurrency, n: concurrency,
            "AccuracyConfig": SimpleNamespace(from_dict=lambda d: None),
            "append_gen_worker_device_step_time": append_steps,
            "append_time_breakdown_metrics": lambda *a, **k: None,
        }
        namespace["warmup_pass_requests_for"], namespace["warmup_pass_plan_for"] = _plan_functions(
            arms, warmup_pass
        )
        (self.run_cmd,) = _perf_sanity_functions(
            namespace, "run_cmd", owner="AggrGenOnlyNoContextCmds"
        )
        self.popen = []

        def launch(server_idx, cmd, env, gen_log, view, plan):
            events.append("launch gen")
            return GenProc(), Pump()

        class Proxy:
            def terminate(self):
                events.append("terminate proxy")

            def wait(self):
                return 0

        def popen(cmd, **kwargs):
            events.append("start proxy" if "disaggregated" in cmd else "start gen")
            return Proxy()

        namespace["subprocess"] = SimpleNamespace(Popen=popen, STDOUT=-2)
        self.cmds = SimpleNamespace(
            server_cmds=[(["trtllm-serve", "m"], ["trtllm-serve", "disaggregated", "-c", "x"])],
            client_cmds={
                0: [["python", "-m", "benchmark_serving", "--num-prompts", "16"]] * clients
            },
            client_configs={0: [_client(16) for _ in range(clients)]},
            server_configs=[(None, None, None)],
            timeout=3600,
            num_gen_servers=1,
            test_output_dir=str(tmp_path),
            perf_metrics_output_dir="",
            _gen_worker_log=lambda server_idx, gen_idx=0: str(tmp_path / "gen_server_0.log"),
            get_server_logs=lambda server_idx: [],
            _write_proxy_config=lambda server_idx, urls: str(tmp_path / "server_config.0.yaml"),
            _launch_arm_gen_worker=launch,
        )

    def __call__(self):
        return self.run_cmd(self.cmds, 0)


@pytest.mark.parametrize("arm", ["E4-base", "E4-chain"])
def test_gen_only_no_context_sends_the_pass_before_the_measured_window(
    arms, warmup_pass, arm_env, tmp_path, arm
):
    arm_env(arm)
    run = _GenOnlyNoContextRun(arms, warmup_pass, tmp_path)
    outputs = run()
    measured = run.events.index("benchmark(trtllm-benchmark.0.0.log)")
    assert run.events[measured - 3 : measured + 2] == [
        "warmup(16, localhost:8000, trtllm-benchmark-warmup-pass.0.0.log)",
        "pump",
        "start_run(served=16)",
        "benchmark(trtllm-benchmark.0.0.log)",
        "finish(served=32)",
    ]
    assert run.events.index("ready") < run.events.index(
        "warmup(16, localhost:8000, trtllm-benchmark-warmup-pass.0.0.log)"
    )
    # Not counted: the client's output is the measured run's alone, and its device step
    # window starts after the 16 warmup requests were served.
    assert outputs == ["measured output of trtllm-benchmark.0.0.log"]
    assert [p["start_offsets"] for p in run.pending] == [[16]]


@pytest.mark.parametrize("arm", ["E4-base", "E4-chain"])
def test_gen_only_no_context_keeps_the_pass_out_of_every_clients_window(
    arms, warmup_pass, arm_env, tmp_path, arm
):
    arm_env(arm)
    run = _GenOnlyNoContextRun(arms, warmup_pass, tmp_path, clients=2)
    outputs = run()
    assert len([e for e in run.events if e.startswith("warmup")]) == 2
    assert len(outputs) == 2
    # Client 0 measures requests 16..32; client 1's pass (32..48) is in neither window.
    windows = [(p["start_offsets"], p["end_offsets"]) for p in run.pending]
    assert windows == [([16], [32]), ([48], None)]


@pytest.mark.parametrize("arm, extra", [(None, {}), ("E4-chain", {"TRTLLM_PERF_WARMUP_PASS": "0"})])
def test_gen_only_no_context_sends_no_pass_otherwise(
    arms, warmup_pass, arm_env, tmp_path, arm, extra
):
    arm_env(arm, **extra)
    run = _GenOnlyNoContextRun(arms, warmup_pass, tmp_path)
    outputs = run()
    assert not [e for e in run.events if e.startswith("warmup")]
    # Without an arm the gen worker starts as perf sanity starts it, with no arm readings.
    assert ("start gen" in run.events) == (arm is None)
    assert outputs == ["measured output of trtllm-benchmark.0.0.log"]
    assert [p["start_offsets"] for p in run.pending] == [[0]]


# ----------------------------------------------------------------------------- post-processing workers

# An in-process server's tree: front end 100 (with a helper 101 it spawned), engine ranks 200
# (the leader) and 300, the leader's four forked post-processing workers 201-204, a compile
# worker 205 it started with a command line of its own, and a child 102 of the front end that
# happens to run the engine's command line.
_TREE = {100, 101, 102, 200, 201, 202, 203, 204, 205, 300}
_PPID = {101: 100, 102: 100, 200: 100, 300: 200, 201: 200, 202: 200, 203: 200, 204: 200, 205: 200}
_ENGINE_CMD = "python3 -m mpi4py.futures.server"
_CMDLINE = {
    100: "trtllm-serve m --server_role generation",
    101: "python3 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5)",
    102: _ENGINE_CMD,
    200: _ENGINE_CMD,
    300: _ENGINE_CMD,
    **{pid: _ENGINE_CMD for pid in (201, 202, 203, 204)},
    205: "python3 torch/_inductor/compile_worker/__main__.py",
}


def _cpu_reader(arms, gpu=frozenset({200, 300})):
    return arms.CpuReader(
        tree=lambda root: set(_TREE) if root == 100 else {root},
        gpu=lambda: None if gpu is None else set(gpu),
        ticks=lambda pid: (1, 0),
        now=lambda: 1,
    )


def test_count_engine_postproc(arms):
    count = arms.count_engine_postproc(100, _cpu_reader(arms), ppid=_PPID.get, cmdline=_CMDLINE.get)
    assert count == 4
    # Without nvidia-smi the engine's processes are unknown: none counted.
    assert (
        arms.count_engine_postproc(
            100, _cpu_reader(arms, None), ppid=_PPID.get, cmdline=_CMDLINE.get
        )
        == 0
    )
    # A child whose command line cannot be read is not counted.
    unreadable = {**_CMDLINE, 201: None}
    assert (
        arms.count_engine_postproc(100, _cpu_reader(arms), ppid=_PPID.get, cmdline=unreadable.get)
        == 3
    )


def test_read_ppid_of_this_process(arms):
    import os

    assert arms._read_ppid(os.getpid()) == os.getppid()
    assert arms._read_ppid(2**31 - 1) is None


def _role_reader(arms, view, tmp_path):
    plan = arms.ServerPlan(("trtllm-serve", "m"), {"num_postprocess_workers": 4})
    server = SimpleNamespace(handle=None, frontend_pid=100, proc=None, arm=view)
    reader = arms.RoleReader(
        server=server,
        view=view,
        role="gen",
        worker="GEN_0",
        directory=str(tmp_path),
        address=lambda: "127.0.0.1:9",
        plan=plan,
        env={},
        launched_at=0.0,
        reader=_cpu_reader(arms),
    )
    reader._frontends_started = lambda notes: 1
    return reader


@pytest.mark.parametrize("name, started", [("E4-base", 5), ("E4-chain", 1)])
def test_reading_counts_the_engines_postproc_workers_in_process_only(
    arms, tmp_path, monkeypatch, name, started
):
    monkeypatch.setattr(arms, "_read_ppid", _PPID.get)
    monkeypatch.setattr(arms, "_read_cmdline", _CMDLINE.get)
    spawned = arms.count_spawned
    monkeypatch.setattr(arms, "count_spawned", lambda pids: spawned(pids, _CMDLINE.get))
    view = arms.role_view(arms.DISAGG_ARMS[name], "gen")
    doc = _role_reader(arms, view, tmp_path).read("ab12", 0, "before")
    # The front end's own spawned helper (101) counts as before, in both arms; the engine's
    # four forked workers only for an in-process server, whose engine is in its tree.
    assert doc["postproc_workers"] == {"configured": 4, "started": started}
    forked = [n for n in doc["notes"] if "forked by the engine" in n]
    in_process = ["4 post-processing workers forked by the engine's processes"]
    assert forked == (in_process if name == "E4-base" else [])


def test_reading_leaves_case_03s_in_process_servers_alone(arms, tmp_path, monkeypatch):
    monkeypatch.setattr(arms, "_read_ppid", _PPID.get)
    monkeypatch.setattr(arms, "_read_cmdline", _CMDLINE.get)
    spawned = arms.count_spawned
    monkeypatch.setattr(arms, "count_spawned", lambda pids: spawned(pids, _CMDLINE.get))
    # dsv4-serve serving HTTP itself: in-process, but not TRT-LLM's executor.
    view = arms.role_view(arms.DISAGG_ARMS["E3-native"], "gen")
    assert view.in_process
    doc = _role_reader(arms, view, tmp_path).read("ab12", 0, "before")
    assert doc["postproc_workers"]["started"] == 1
    assert not [n for n in doc["notes"] if "forked by the engine" in n]


# ----------------------------------------------------------------------------- the front end's CPU time


def test_cpu_by_process_note(arms):
    cmd = {1: "python3 front end " + "x" * 300, 2: "python3 spawn_main", 3: None}
    note = arms.cpu_by_process_note({1: 150, 2: 7300, 3: 0}, 100.0, cmd.get)
    assert note.startswith("front-end CPU by process: 2 73.00s python3 spawn_main; 1 1.50s ")
    assert note.endswith("; 3 0.00s ?")
    # Each command line is cut to its start.
    assert f"1 1.50s {cmd[1][: arms.CPU_NOTE_CMDLINE_CHARS]}" in note
    many = {pid: pid for pid in range(1, arms.CPU_NOTE_PROCESSES + 4)}
    note = arms.cpu_by_process_note(many, 1.0, lambda pid: "p")
    assert note.endswith("; 3 more 6.00s")
    assert arms.cpu_by_process_note({}, 1.0) == "front-end CPU by process: none"


def test_run_probe_notes_the_cpu_time_by_process(arms, monkeypatch):
    # Front end 100 with post-processing workers 101 and 102; 102 starts during the run.
    ticks = {100: (1, 10), 101: (1, 5)}
    reader = arms.CpuReader(
        tree=lambda root: set(ticks),
        gpu=lambda: set(),
        ticks=lambda pid: ticks.get(pid),
        now=lambda: 1,
        ticks_per_s=10.0,
    )
    monkeypatch.setattr(arms, "_read_cmdline", {100: "fe", 101: "pp1", 102: "pp2"}.get)
    server = SimpleNamespace(frontend_pid=100, handle=None, arm=SimpleNamespace(in_process=False))
    probe = arms.RunProbe.start(server, reader)
    ticks.update({100: (1, 30), 101: (1, 705), 102: (5, 40)})
    assert probe.frontend_cpu_s() == pytest.approx((20 + 700 + 40) / 10.0)
    (note,) = [n for n in probe.notes if n.startswith("front-end CPU by process")]
    assert note == "front-end CPU by process: 101 70.00s pp1; 102 4.00s pp2; 100 2.00s fe"


# ----------------------------------------------------------------------------- the driver's trace paths


def test_each_worker_role_gets_a_trace_directory_of_its_own(driver, tmp_path):
    import os

    plan = driver.load_plan(
        "disagg-gen_only-b200_qwen3-0.6b-bf16_1k256_con16_ctx1_tp1_gen1_tp1_eplb0_mtp0_ccb-DEFAULT",
        str(REPO / "jenkins" / "scripts" / "perf" / "local" / "single_node_disagg" / "configs"),
    )
    trace = tmp_path / "traces" / "trace.json"
    base = {"TLLM_TORCH_PROFILE_TRACE": str(trace), "TLLM_PROFILE_START_STOP": "1-3"}
    for role in ("CTX_0", "GEN_0"):
        env = driver.role_env(base, plan, role, gpus=["0"], tag="t", ucx={})
        assert env["TLLM_TORCH_PROFILE_TRACE"] == str(tmp_path / "traces" / role / "trace.json")
        assert os.path.isdir(tmp_path / "traces" / role)
        assert env["TLLM_PROFILE_START_STOP"] == "1-3"
    for role in ("DISAGG_SERVER", "BENCHMARK"):
        env = driver.role_env(base, plan, role, gpus=None, tag="t", ucx={})
        assert "TLLM_TORCH_PROFILE_TRACE" not in env
    assert sorted(p.name for p in (tmp_path / "traces").iterdir()) == ["CTX_0", "GEN_0"]
    # Without a trace path no role gets one.
    env = driver.role_env({}, plan, "CTX_0", gpus=["0"], tag="t", ucx={})
    assert "TLLM_TORCH_PROFILE_TRACE" not in env
