# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start order, arguments, logs, shutdown order and timeouts, against fake processes."""

import glob
import json
import os
import pathlib
import re
import signal
import socket
import stat
import subprocess
import sys
import time

import pytest
import requests
import yaml
from openengine_chain import conventions as cv
from openengine_chain import launch, proof

pytestmark = pytest.mark.cpu_only

_FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_serve.py")
_SERVE_PY = pathlib.Path(__file__).resolve().parents[5] / "tensorrt_llm" / "commands" / "serve.py"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """An executable named trtllm-serve, the record directory and a base environment."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "trtllm-serve"
    exe.write_text(f"#!{sys.executable}\n" + open(_FAKE).read())
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(launch, "require_plugin", lambda: None)
    monkeypatch.setattr(launch, "frontend_supports_ready_timeout", lambda: True)
    monkeypatch.setattr(launch, "engine_supports_metrics_port", lambda: True)
    monkeypatch.setenv(cv.ENV_RESULTS_DIR, str(tmp_path / "results"))
    monkeypatch.delenv(cv.ENV_GLM_SERVE_CMD, raising=False)
    env = dict(os.environ, FAKE_RECORD_DIR=str(records))
    env.pop(cv.ENV_COUNTER_DIR, None)
    handles = []
    yield type("Fake", (), {"exe": str(exe), "records": records, "env": env, "handles": handles})
    for handle in handles:
        handle.terminate(grace=1)


def _record(records, name, timeout=10.0):
    path = records / f"{name}.json"
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{name} was not recorded"
        time.sleep(0.05)
    return json.loads(path.read_text())


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gone(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def _wait_healthy(handle, timeout=10.0):
    # The fake front end records its start before it listens.
    deadline = time.monotonic() + timeout
    while True:
        try:
            if requests.get(handle.url_for("health"), timeout=1).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        assert time.monotonic() < deadline, "front end not healthy"
        time.sleep(0.05)


def _start(fake, args, **kwargs):
    handle = launch.launch_serve_command([fake.exe, "/model"] + args, env=fake.env, **kwargs)
    fake.handles.append(handle)
    return handle


def test_engine_is_ready_before_frontend_starts_and_args_are_split(fake, tmp_path):
    port = _port()
    fake.env["TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"] = "2"
    sink = open(tmp_path / "sink.log", "w")
    handle = _start(
        fake,
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tp_size",
            "1",
            "--tool_parser",
            "qwen3",
            "--set",
            "num_postprocess_workers=2",
        ],
        chain="trtllm",
        sink=sink,
    )
    engine = _record(fake.records, "engine-start")
    frontend = _record(fake.records, "frontend-start")
    engine_port = int(handle.engine_endpoint.rsplit(":", 1)[1])
    assert engine["argv"] == [
        "/model",
        "--grpc",
        "--grpc-protocol",
        "openengine",
        "--host",
        "127.0.0.1",
        "--port",
        str(engine_port),
        "--tp_size",
        "1",
        launch.ENGINE_METRICS_PORT_OPTION,
        "0",
    ]
    assert frontend["argv"] == [
        "/model",
        "--openengine-endpoint",
        handle.engine_endpoint,
        "--openengine-ready-timeout",
        "60",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tool_parser",
        "qwen3",
        "--set",
        "num_postprocess_workers=2",
    ]
    # The engine needs FAKE_ENGINE_DELAY to turn SERVING; the front end starts after that.
    assert frontend["time"] - engine["time"] >= 0.4
    assert frontend["env"][cv.ENV_COUNTER_DIR] == handle.counter_dir
    assert cv.ENV_COUNTER_DIR not in engine["env"]
    assert engine["env"]["TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"] == "2"
    assert "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS" not in frontend["env"]
    assert handle.workdir.startswith(os.environ[cv.ENV_RESULTS_DIR])
    _wait_healthy(handle)
    assert (handle.host, handle.port) == ("127.0.0.1", port)

    handle.terminate()
    assert os.path.basename(handle.engine_log) == f"engine-{engine['pid']}.log"
    assert os.path.basename(handle.frontend_log) == f"frontend-{frontend['pid']}.log"
    assert "engine started" in open(handle.engine_log).read()
    assert "frontend started" in open(handle.frontend_log).read()
    sink.close()
    mirrored = open(tmp_path / "sink.log").read()
    assert "engine started" in mirrored and "frontend started" in mirrored
    assert (
        _record(fake.records, "frontend-term")["time"]
        < _record(fake.records, "engine-term")["time"]
    )
    assert handle.frontend.poll() is not None and handle.engine.poll() is not None
    assert not handle.alive


def test_frontend_ignoring_sigterm_is_killed_after_the_grace_period(fake):
    fake.env["FAKE_FRONTEND_IGNORE_TERM"] = "1"
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    _record(fake.records, "frontend-start")
    started = time.monotonic()
    handle.terminate(grace=1)
    assert time.monotonic() - started >= 1
    assert handle.frontend.returncode == -signal.SIGKILL
    assert handle.engine.poll() is not None


def test_processes_left_in_a_session_are_killed(fake):
    fake.env["FAKE_ENGINE_CHILD"] = "1"
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    child = _record(fake.records, "engine-start")["child"]
    handle.terminate(grace=1)
    deadline = time.monotonic() + 10
    while not _gone(child):
        assert time.monotonic() < deadline, "engine child survived shutdown"
        time.sleep(0.05)
    assert handle.teardown_errors == []


def test_engine_exiting_before_ready_fails_without_starting_the_frontend(fake, monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    fake.env["FAKE_ENGINE_MODE"] = "exit"
    with pytest.raises(launch.ChainStartupError, match="exited with code 3"):
        _start(fake, ["--port", str(_port())], chain="trtllm")
    assert not (fake.records / "frontend-start.json").exists()
    # The failed server stays registered, stopped, with the engine log for the report.
    (handle,) = launch._HANDLES
    assert not handle.alive and handle.frontend is None
    assert "engine started" in open(handle.engine_log).read()


def test_engine_that_lost_its_port_is_restarted_on_another(fake, monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    fake.env["FAKE_ENGINE_MODE"] = "busy_once"
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    _record(fake.records, "frontend-start")
    # The replaced attempt is no test's server: only the one that started is registered.
    assert launch._HANDLES == [handle]
    logs = sorted(glob.glob(os.path.join(handle.workdir, "engine-*.log")))
    assert len(logs) == 2 and handle.engine_log in logs
    (first_log,) = set(logs) - {handle.engine_log}
    assert "address already in use" in open(first_log).read()
    (note,) = [n for n in handle.teardown_notes if "lost its gRPC port" in n]
    assert first_log in note
    handle.terminate(grace=1)


def test_other_engine_failures_are_not_retried(fake, monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    fake.env["FAKE_ENGINE_MODE"] = "exit"
    with pytest.raises(launch.ChainStartupError):
        _start(fake, ["--port", str(_port())], chain="trtllm")
    assert len(launch._HANDLES) == 1


def test_engine_not_ready_in_time_fails_and_is_stopped(fake):
    fake.env["FAKE_ENGINE_MODE"] = "never_ready"
    with pytest.raises(launch.ChainStartupError, match="not SERVING within"):
        _start(fake, ["--port", str(_port())], chain="trtllm", engine_ready_timeout=2)
    pid = _record(fake.records, "engine-start")["pid"]
    assert _gone(pid)
    assert not (fake.records / "frontend-start.json").exists()


def _wait_gone(pid, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not _gone(pid):
        assert time.monotonic() < deadline, f"pid {pid} still running"
        time.sleep(0.05)


def test_front_end_refusal_stops_the_engine_while_it_loads(fake, tmp_path):
    # The fake engine would take 600 s to turn SERVING.
    fake.env["FAKE_ENGINE_DELAY"] = "600"
    sink = open(tmp_path / "sink.log", "w")
    started = time.monotonic()
    handle = _start(
        fake, ["--port", str(_port()), "--max_beam_width", "2"], chain="trtllm", sink=sink
    )
    assert time.monotonic() - started < 30
    check = _record(fake.records, "frontend-check-start")
    assert check["argv"][check["argv"].index("--max_beam_width") + 1] == "2"
    # The engine was stopped; the front end proper never started.
    engine = _record(fake.records, "engine-start")
    _record(fake.records, "engine-term")
    _wait_gone(engine["pid"])
    assert not (fake.records / "frontend-start.json").exists()
    # The check's exited process stands for the front end, as an exited front end would.
    assert handle.frontend.pid == check["pid"]
    assert handle.poll() == 2
    assert os.path.basename(handle.frontend_log) == f"frontend-check-{check['pid']}.log"
    assert "max_beam_width > 1 is not supported" in open(handle.frontend_log).read()
    assert handle.alive
    handle.terminate(grace=1)
    assert handle.teardown_errors == []
    sink.close()
    mirrored = open(tmp_path / "sink.log").read()
    assert "max_beam_width > 1 is not supported" in mirrored
    assert "engine started" not in mirrored


@pytest.mark.parametrize("beam", [["--max_beam_width", "4"], ["--max_beam_width=4"]])
def test_chain_process_of_a_refused_front_end_exits_with_its_code(
    fake, monkeypatch, tmp_path, beam
):
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    fake.env["FAKE_ENGINE_DELAY"] = "600"
    out = open(tmp_path / "out.log", "w")
    cmd = [fake.exe, "/model", "--port", str(_port())] + beam
    with launch.chain_popen(cmd, env=fake.env, stdout=out) as proc:
        assert proc.wait(timeout=30) == 2
        engine_pid = _record(fake.records, "engine-start")["pid"]
    _wait_gone(engine_pid)
    out.close()
    assert "max_beam_width > 1 is not supported" in open(tmp_path / "out.log").read()


def test_front_end_check_that_passes_is_stopped_before_the_front_end_starts(fake):
    port = _port()
    handle = _start(
        fake,
        ["--host", "127.0.0.1", "--port", str(port), "--max_beam_width", "1", "--tool_parser", "x"],
        chain="trtllm",
    )
    check = _record(fake.records, "frontend-check-start")
    frontend = _record(fake.records, "frontend-start")
    assert _record(fake.records, "frontend-check-term")["time"] < frontend["time"]
    _wait_gone(check["pid"])
    # The same front-end command line, with an engine address and a port of its own.
    endpoint = check["argv"][check["argv"].index("--openengine-endpoint") + 1]
    check_port = check["argv"][check["argv"].index("--port") + 1]
    assert endpoint != handle.engine_endpoint and check_port != str(port)
    assert [a for a in check["argv"] if a not in (endpoint, check_port)] == [
        a for a in frontend["argv"] if a not in (handle.engine_endpoint, str(port))
    ]
    assert check["env"][launch.FRONTEND_CHECK_ENV] == "1"
    assert cv.ENV_COUNTER_DIR not in check["env"]
    assert launch.FRONTEND_CHECK_ENV not in frontend["env"]
    _wait_healthy(handle)
    assert os.path.basename(handle.frontend_log) == f"frontend-{frontend['pid']}.log"
    handle.terminate(grace=1)
    assert handle.teardown_errors == []


@pytest.mark.parametrize("check_mode", ["never_ready", "busy_once", "killed", "done"])
def test_front_end_check_without_an_answer_leaves_the_start_as_it_was(fake, check_mode):
    # never_ready: still checking when the engine is ready; busy_once: lost its own port;
    # killed, done: exited by a signal or with code 0, which is no refusal.
    fake.env["FAKE_FRONTEND_CHECK_MODE"] = check_mode
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    check = _record(fake.records, "frontend-check-start")
    frontend = _record(fake.records, "frontend-start")
    _wait_gone(check["pid"])
    assert frontend["time"] - _record(fake.records, "engine-start")["time"] >= 0.4
    _wait_healthy(handle)
    assert handle.poll() is None
    handle.terminate(grace=1)
    assert handle.teardown_errors == []


def test_front_end_check_takes_a_descriptor_above_fd_setsize(fake):
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    need = 1200
    if hard != resource.RLIM_INFINITY and hard < need:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} is below {need}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, need), hard))
    fds = [os.open(os.devnull, os.O_RDONLY)]
    try:
        # Every descriptor below 1024 is taken, so the check's socket gets a higher one.
        while fds[-1] < 1024:
            fds.append(os.dup(fds[0]))
        handle = _start(fake, ["--port", str(_port())], chain="trtllm")
        frontend = _record(fake.records, "frontend-start")
        assert _record(fake.records, "frontend-check-term")["time"] < frontend["time"]
        _wait_healthy(handle)
        handle.terminate(grace=1)
        assert handle.teardown_errors == []
    finally:
        for fd in fds:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_front_end_check_is_stopped_when_the_engine_fails(fake, monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    fake.env["FAKE_ENGINE_MODE"] = "exit"
    fake.env["FAKE_FRONTEND_CHECK_MODE"] = "never_ready"
    with pytest.raises(launch.ChainStartupError, match="exited with code 3"):
        _start(fake, ["--port", str(_port())], chain="trtllm")
    _wait_gone(_record(fake.records, "frontend-check-start")["pid"])
    (handle,) = launch._HANDLES
    assert handle.frontend is None and handle.teardown_errors == []


def test_front_end_check_problems_of_a_start_without_a_server_are_printed(
    fake, monkeypatch, capsys
):
    monkeypatch.setattr(launch, "_HANDLES", [])
    monkeypatch.setenv(cv.ENV_GLM_SERVE_CMD, str(fake.records / "no-such-engine"))
    real_stop = launch.LoggedProcess.stop

    def stop(self, grace):
        problems = real_stop(self, grace)
        return problems + ([f"{self.role} left pid 1"] if self.role == "frontend-check" else [])

    monkeypatch.setattr(launch.LoggedProcess, "stop", stop)
    with pytest.raises(FileNotFoundError):
        _start(fake, ["--port", str(_port())], chain="glm")
    assert launch._HANDLES == []
    assert "front-end check of a failed start: frontend-check left pid 1" in capsys.readouterr().err


def test_chain_process_reports_an_engine_that_died(fake, monkeypatch):
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    with launch.chain_popen([fake.exe, "/model", "--port", str(_port())], env=fake.env) as proc:
        _record(fake.records, "frontend-start")
        assert proc.poll() is None
        os.kill(_record(fake.records, "engine-start")["pid"], signal.SIGKILL)
        deadline = time.monotonic() + 10
        while proc.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert proc.poll() != 0
    assert not proc.handle.alive


def test_launch_serve_writes_extra_config_and_strips_frontend_keys_for_the_engine(fake):
    handle = launch.launch_serve(
        "/model",
        ["--port", str(_port())],
        {"num_postprocess_workers": 2, "max_batch_size": 4},
        chain="trtllm",
        prefix=[fake.exe],
        env=fake.env,
    )
    fake.handles.append(handle)
    engine = _record(fake.records, "engine-start")["argv"]
    frontend = _record(fake.records, "frontend-start")["argv"]
    engine_yaml = engine[engine.index("--extra_llm_api_options") + 1]
    frontend_yaml = frontend[frontend.index("--extra_llm_api_options") + 1]
    assert yaml.safe_load(open(engine_yaml)) == {"max_batch_size": 4}
    assert yaml.safe_load(open(frontend_yaml)) == {
        "num_postprocess_workers": 2,
        "max_batch_size": 4,
    }
    handle.terminate(grace=1)
    assert not os.path.exists(frontend_yaml)


def test_no_chain_runs_the_command_unchanged(fake):
    handle = _start(fake, ["--port", str(_port()), "--log_level", "info"], chain="none")
    assert _record(fake.records, "frontend-start")["argv"] == [
        "/model",
        "--port",
        str(handle.port),
        "--log_level",
        "info",
    ]
    assert handle.engine is None
    handle.terminate(grace=1)


def test_chain_without_the_plugin_is_refused(monkeypatch):
    monkeypatch.delitem(sys.modules, cv.PLUGIN_MODULE, raising=False)
    with pytest.raises(launch.ChainError, match="plugin"):
        launch.require_plugin()


@pytest.mark.parametrize("value, chain", [("", "none"), ("trtllm", "trtllm"), ("glm", "glm")])
def test_resolve_chain(monkeypatch, value, chain):
    monkeypatch.setenv(cv.ENV_CHAIN, value)
    assert launch.resolve_chain() == chain


def test_resolve_chain_rejects_other_values(monkeypatch):
    monkeypatch.setenv(cv.ENV_CHAIN, "1")
    with pytest.raises(launch.ChainError):
        launch.resolve_chain()


def test_glm_chain_uses_the_glm_command(fake, monkeypatch):
    monkeypatch.setenv(cv.ENV_GLM_SERVE_CMD, f"{fake.exe} --glm-flag")
    handle = _start(fake, ["--port", str(_port())], chain="glm")
    assert _record(fake.records, "engine-start")["argv"][:3] == ["--glm-flag", "/model", "--grpc"]
    handle.terminate(grace=1)


def test_glm_chain_needs_the_glm_command(fake):
    with pytest.raises(launch.ChainError, match=cv.ENV_GLM_SERVE_CMD):
        _start(fake, ["--port", str(_port())], chain="glm")


def test_shutdown_keeps_the_engine_name_and_a_proven_topology(monkeypatch):
    from openengine_chain import proof

    monkeypatch.setattr(launch, "_HANDLES", [])
    monkeypatch.setattr(proof, "read_counters", lambda h: proof.Counters(1, 1))
    monkeypatch.setattr(proof, "check_topology", lambda h: pytest.fail("topology checked again"))

    def read_engine_name(h):
        h._engine_name = "tensorrt_llm"
        return h._engine_name

    monkeypatch.setattr(proof, "read_engine_name", read_engine_name)
    handle = launch.ServeHandle(
        chain="trtllm",
        model="m",
        host="localhost",
        port=None,
        frontend=subprocess.Popen(["sleep", "60"]),
        engine=None,
        engine_endpoint=None,
        workdir=None,
        counter_dir=None,
    )
    handle.proven_topology = proof.Check(proof.OK, "proven")
    handle.terminate(grace=1)
    assert handle._engine_name == "tensorrt_llm"
    assert handle.final_topology is handle.proven_topology
    assert handle.final_counters == proof.Counters(1, 1)
    assert handle.teardown_errors == [] and handle.teardown_notes == []


@pytest.mark.parametrize(
    "stdout, stderr", [(subprocess.PIPE, None), (None, subprocess.PIPE)], ids=["pipe", "split"]
)
def test_chain_popen_refuses_outputs_it_cannot_serve(stdout, stderr):
    with pytest.raises(launch.ChainError):
        with launch.chain_popen(["trtllm-serve", "/model"], stdout=stdout, stderr=stderr):
            pass


def _start_with_detached_worker(tmp_path, role):
    """A process whose child starts a session of its own; returns (process, worker pid)."""
    pid_file = tmp_path / "worker.pid"
    worker = (
        "import os, sys, time; os.setsid(); "
        "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(60)"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {worker!r}, {str(pid_file)!r}]); time.sleep(60)"
    )
    proc = launch.LoggedProcess(
        role, [sys.executable, "-c", parent], os.environ, str(tmp_path), None
    )
    deadline = time.monotonic() + 10
    while not (pid_file.exists() and pid_file.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    return proc, int(pid_file.read_text())


def test_stop_kills_workers_that_left_the_session(tmp_path):
    proc, worker_pid = _start_with_detached_worker(tmp_path, "engine")
    assert worker_pid not in launch.live_session_members(proc.pid)
    assert proc.stop(grace=1) == []
    assert _gone(worker_pid)


def test_shutdown_problems_of_a_replaced_attempt_stay_reported(fake, monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    fake.env["FAKE_ENGINE_MODE"] = "busy_once"
    real_stop = launch.LoggedProcess.stop
    stops = []

    def stop(self, grace):
        stops.append(self.role)
        problems = real_stop(self, grace)
        return problems + (["engine attempt 1 left pid 1"] if len(stops) == 1 else [])

    monkeypatch.setattr(launch.LoggedProcess, "stop", stop)
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    assert launch._HANDLES == [handle]
    assert handle.teardown_errors == ["engine attempt 1 left pid 1"]
    handle.terminate(grace=1)
    assert handle.teardown_errors == ["engine attempt 1 left pid 1"]


def test_process_tree_has_the_session_and_the_workers_that_left_it(tmp_path):
    proc, worker_pid = _start_with_detached_worker(tmp_path, "frontend")
    try:
        assert {proc.pid, worker_pid} <= launch.live_process_tree(proc.pid)
    finally:
        assert proc.stop(grace=1) == []


def test_stop_leaves_a_process_that_reused_a_recorded_pid(tmp_path):
    bystander = subprocess.Popen(["sleep", "60"])
    proc = launch.LoggedProcess("engine", ["sleep", "60"], os.environ, str(tmp_path), None)
    try:
        start = launch._live_processes()[bystander.pid][2]
        # As if a recorded worker exited and the bystander then got its PID.
        proc._known[bystander.pid] = start - 1
        assert proc.stop(grace=1) == []
        assert bystander.poll() is None
    finally:
        bystander.kill()
        bystander.wait()


def test_stop_kills_recorded_workers_after_their_parent_died(tmp_path):
    proc, worker_pid = _start_with_detached_worker(tmp_path, "engine")
    proc.remember_tree()
    os.kill(proc.pid, signal.SIGKILL)
    proc.proc.wait()
    assert worker_pid not in launch.live_descendants(proc.pid)
    assert proc.stop(grace=1) == []
    assert _gone(worker_pid)


def test_only_the_attempt_that_becomes_ready_reaches_the_sink(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(launch, "_HANDLES", [])
    fake.env["FAKE_ENGINE_MODE"] = "busy_once"
    sink = open(tmp_path / "sink.log", "w")
    handle = _start(fake, ["--port", str(_port())], chain="trtllm", sink=sink)
    _record(fake.records, "frontend-start")
    handle.terminate(grace=1)
    sink.close()
    mirrored = (tmp_path / "sink.log").read_text()
    # Tests scan the server's output for errors; the replaced attempt's are only in its log.
    assert "address already in use" not in mirrored
    assert mirrored.count("engine started") == 1 and "frontend started" in mirrored


def test_engine_output_of_a_failed_start_reaches_the_sink(fake, tmp_path):
    fake.env["FAKE_ENGINE_MODE"] = "exit"
    sink = open(tmp_path / "sink.log", "w")
    with pytest.raises(launch.ChainStartupError):
        _start(fake, ["--port", str(_port())], chain="trtllm", sink=sink)
    sink.close()
    assert "engine started" in (tmp_path / "sink.log").read_text()


def test_bind_error_far_before_the_end_of_the_log_is_found(tmp_path):
    log = tmp_path / "engine.log"
    filler = "shutting down\n" * (2 * launch.LOG_TAIL_BYTES // 14)
    log.write_text("RuntimeError: Failed to bind 127.0.0.1:1: address already in use\n" + filler)
    assert "bind" not in launch.tail(str(log))
    assert launch._lost_its_port(str(log), 1)
    log.write_text(filler)
    assert not launch._lost_its_port(str(log), 1)
    # Both words, but not on one line: another failure, not a lost port.
    log.write_text("loading nanobind module\n" + filler + "mpi: address already in use, retrying\n")
    assert not launch._lost_its_port(str(log), 1)
    grpc_line = (
        "E1107 add_port.cc:83] Failed to add port to server: No address added out of total 1 "
        "resolved for '127.0.0.1:{0}' (Error in bind for address '[::ffff:127.0.0.1]:{0}': "
        "Address already in use)\n"
    )
    log.write_text(grpc_line.format(1) + filler)
    assert launch._lost_its_port(str(log), 1)
    # Another port: not the engine's, so not retried on a new engine port.
    log.write_text(grpc_line.format(18) + filler)
    assert not launch._lost_its_port(str(log), 1)
    assert not launch._lost_its_port(str(tmp_path / "missing.log"), 1)
    assert not launch._lost_its_port(None, 1)


def test_foreign_listeners_are_the_sockets_other_processes_hold():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert launch.foreign_listeners(port, {os.getpid()}) == set()
        assert len(launch.foreign_listeners(port, set())) == 1
    assert launch.foreign_listeners(port, {os.getpid()}) is None


def test_engine_port_served_by_another_server_too_is_retried_on_another(fake, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import grpc
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    monkeypatch.setattr(launch, "_HANDLES", [])
    # A gRPC server binds with SO_REUSEPORT by default: the fake engine binds this port too.
    other = grpc.server(ThreadPoolExecutor(1))
    servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(servicer, other)
    taken = other.add_insecure_port("127.0.0.1:0")
    other.start()
    servicer.set("inference", health_pb2.HealthCheckResponse.SERVING)
    real_free_port = launch.free_port
    # The front-end check picks its port first; the engine's first attempt gets ``taken``.
    ports = iter([real_free_port(), taken])
    monkeypatch.setattr(launch, "free_port", lambda: next(ports, None) or real_free_port())
    try:
        handle = _start(fake, ["--port", str(_port())], chain="trtllm")
        assert handle.engine_endpoint != f"127.0.0.1:{taken}"
        assert launch._HANDLES == [handle]
        assert len(glob.glob(os.path.join(handle.workdir, "engine-*.log"))) == 2
        handle.terminate(grace=1)
    finally:
        other.stop(0)


def test_stop_kills_later_children_of_a_recorded_worker_after_their_parent_died(tmp_path):
    worker_file, child_file = tmp_path / "worker.pid", tmp_path / "child.pid"
    child = "import time; time.sleep(60)"
    worker = (
        "import os, subprocess, sys, time; os.setsid(); "
        f"open({str(worker_file)!r}, 'w').write(str(os.getpid())); "
        f"time.sleep(1); p = subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"open({str(child_file)!r}, 'w').write(str(p.pid)); time.sleep(60)"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {worker!r}]); time.sleep(60)"
    )
    proc = launch.LoggedProcess(
        "engine", [sys.executable, "-c", parent], os.environ, str(tmp_path), None
    )
    deadline = time.monotonic() + 10
    while not (worker_file.exists() and worker_file.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    worker_pid = int(worker_file.read_text())
    proc.remember_tree()
    os.kill(proc.pid, signal.SIGKILL)
    proc.proc.wait()
    while not (child_file.exists() and child_file.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    child_pid = int(child_file.read_text())
    # Started after the record and outside the engine's session.
    assert child_pid not in proc._known and child_pid not in launch.live_session_members(proc.pid)
    assert proc.stop(grace=1) == []
    assert _gone(worker_pid) and _gone(child_pid)


class _Unkillable:
    """A process that neither SIGTERM nor SIGKILL ends (as one stuck in the kernel)."""

    pid = 999999
    args = ["unkillable"]

    def poll(self):
        return None

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout=None):
        raise subprocess.TimeoutExpired(self.args, timeout)


def test_shutdown_does_not_wait_forever_for_a_process_sigkill_cannot_end():
    handle = launch.ServeHandle(
        chain=cv.CHAIN_NONE,
        model="",
        host="localhost",
        port=None,
        frontend=_Unkillable(),
        engine=None,
        engine_endpoint=None,
        workdir=None,
        counter_dir=None,
    )
    handle.terminate(grace=0.1)
    assert handle.teardown_errors == ["pid 999999 still running after SIGKILL"]


def test_launch_serve_passes_check_plugin_on(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        launch, "launch_serve_command", lambda cmd, **kwargs: seen.update(kwargs) or object()
    )
    launch.launch_serve("/m", [], chain="trtllm", check_plugin=False)
    assert seen["check_plugin"] is False


def test_stop_leaves_the_session_of_a_recorded_process_that_exited(tmp_path):
    # A session whose leader exited; its id may since have been reused by an unrelated one.
    leader = subprocess.Popen(
        ["setsid", "sh", "-c", "sleep 60 & echo $! > " + str(tmp_path / "left.pid")]
    )
    leader.wait()
    deadline = time.monotonic() + 10
    while not (tmp_path / "left.pid").read_text().strip() and time.monotonic() < deadline:
        time.sleep(0.05)
    orphan = int((tmp_path / "left.pid").read_text())
    session = launch._live_processes()[orphan][1]
    proc = launch.LoggedProcess("engine", ["sleep", "60"], os.environ, str(tmp_path), None)
    try:
        proc._known[session] = 1
        assert proc.stop(grace=1) == []
        assert not _gone(orphan)
        (note,) = proc.notes
        assert str(orphan) in note and "not killed" in note
        # A second stop, as terminate() makes after a refusal stopped the engine, adds nothing.
        assert proc.stop(grace=1) == []
        assert len(proc.notes) == 1
    finally:
        os.kill(orphan, signal.SIGKILL)


def test_processes_started_after_the_sigkill_are_killed_too(tmp_path, monkeypatch):
    late = subprocess.Popen(["sleep", "60"])
    proc = launch.LoggedProcess("engine", ["sleep", "60"], os.environ, str(tmp_path), None)
    real_table = launch._live_processes
    calls = []

    def table():
        calls.append(1)
        found = real_table()
        # Looks 1 and 2 record the tree and pick what to SIGKILL; from look 3 on,
        # ``late`` shows up in the engine's session.
        if len(calls) > 2 and late.pid in found:
            ppid, _, start = found[late.pid]
            found[late.pid] = (ppid, proc.pid, start)
        return found

    monkeypatch.setattr(launch, "_live_processes", table)
    try:
        assert proc.stop(grace=1) == []
        late.wait(timeout=5)
        assert late.returncode == -signal.SIGKILL
    finally:
        if late.poll() is None:
            late.kill()
            late.wait()


def test_interrupt_while_stopping_the_front_end_still_stops_the_engine(fake, monkeypatch):
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    engine_pid = _record(fake.records, "engine-start")["pid"]
    real_stop = handle._frontend.stop

    def interrupted(grace):
        real_stop(grace)
        raise KeyboardInterrupt

    monkeypatch.setattr(handle._frontend, "stop", interrupted)
    with pytest.raises(KeyboardInterrupt):
        handle.terminate(grace=1)
    assert _gone(engine_pid)
    assert any("KeyboardInterrupt" in error for error in handle.teardown_errors)


def test_interrupt_while_reading_counters_still_stops_both_processes(fake, monkeypatch):
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    engine_pid = _record(fake.records, "engine-start")["pid"]
    frontend_pid = handle.frontend.pid

    def interrupted(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(proof, "read_counters", interrupted)
    with pytest.raises(KeyboardInterrupt):
        handle.terminate(grace=1)
    assert _gone(engine_pid) and _gone(frontend_pid)


def test_uvicorn_bind_error_of_the_front_end_is_found(tmp_path):
    log = tmp_path / "frontend.log"
    line = (
        "ERROR:    [Errno 98] error while attempting to bind on address "
        "('127.0.0.1', {}): address already in use\n"
    )
    log.write_text("INFO:     Application startup complete.\n" + line.format(42213))
    assert launch._lost_its_port(str(log), 42213)
    assert not launch._lost_its_port(str(log), 4221)
    log.write_text(line.format(18))
    assert not launch._lost_its_port(str(log), 1)


def test_port_reserved_by_another_multi_frontend_server_is_found(tmp_path):
    log = tmp_path / "frontend.log"
    line = (
        "RuntimeError: Port {} is already reserved by another multi-frontend trtllm-serve: "
        "[Errno 98] Address already in use\n"
    )
    log.write_text(line.format(42213))
    assert launch._lost_its_port(str(log), 42213)
    assert not launch._lost_its_port(str(log), 4221)
    # The probe bind of a multi-frontend server names the port after a colon.
    log.write_text(
        "RuntimeError: Failed to bind socket to 127.0.0.1:42213: [Errno 98] Address already "
        "in use. Port holder(s): 127.0.0.1:42213 status=LISTEN pid=7\n"
    )
    assert launch._lost_its_port(str(log), 42213)


def _write_config(tmp_path, name, config):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return str(path)


@pytest.mark.parametrize(
    "chain, config, sets, flagged",
    [
        ("trtllm", {"enable_iter_perf_stats": True}, [], True),
        ("trtllm", {"return_perf_metrics": True}, [], False),
        ("trtllm", {"enable_iter_perf_stats": True}, ["enable_iter_perf_stats=false"], False),
        ("trtllm", {}, ["enable_iter_perf_stats=true"], True),
        ("glm", {"enable_iter_perf_stats": True}, [], False),
    ],
)
def test_engine_gets_load_metrics_when_the_configuration_has_iteration_stats(
    fake, monkeypatch, tmp_path, chain, config, sets, flagged
):
    monkeypatch.setenv(cv.ENV_GLM_SERVE_CMD, fake.exe)
    args = [
        "--port",
        str(_port()),
        "--extra_llm_api_options",
        _write_config(tmp_path, "c.yaml", config),
    ]
    for assignment in sets:
        args += ["--set", assignment]
    handle = _start(fake, args, chain=chain)
    engine = _record(fake.records, "engine-start")["argv"]
    frontend = _record(fake.records, "frontend-start")["argv"]
    assert (launch.ENGINE_LOAD_METRICS_FLAG in engine) is flagged
    assert launch.ENGINE_LOAD_METRICS_FLAG not in frontend
    handle.terminate(grace=1)


def _blocked_port():
    """A port another process listens on without ever answering, as in a TOCTOU collision."""
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(16)
    return blocker, blocker.getsockname()[1]


def test_front_end_that_lost_its_port_is_restarted_on_another(fake):
    blocker, taken = _blocked_port()
    with blocker:
        handle = _start(fake, ["--host", "127.0.0.1", "--port", str(taken)], chain="trtllm")
        deadline = time.monotonic() + 10
        while handle.poll() is None:
            assert time.monotonic() < deadline, "the front end did not exit"
            time.sleep(0.05)
        problem = handle.frontend_port_problem()
        assert problem is not None and f"port {taken} is in use" in problem
        engine_pid = handle.engine.pid
        lost = handle.frontend.pid
        new_port = _port()
        handle.restart_frontend(new_port)
        assert handle.port == new_port and handle.frontend.pid != lost
        assert handle.engine.pid == engine_pid
        _wait_healthy(handle)
        assert handle.frontend_port_problem() is None
        assert any(f"restarted on port {new_port}" in note for note in handle.teardown_notes)
        assert _record(fake.records, "frontend-start")["argv"][-2:] == ["--port", str(new_port)]
        handle.terminate(grace=1)
    assert handle.teardown_errors == []


def test_front_end_sharing_its_port_with_another_listener_has_a_port_problem(fake):
    handle = _start(fake, ["--host", "127.0.0.1", "--port", str(_port())], chain="trtllm")
    _wait_healthy(handle)
    assert handle.frontend_port_problem() is None
    # Another address of the same port binds without an error, as SO_REUSEPORT does.
    with socket.socket() as other:
        other.bind(("127.0.0.2", handle.port))
        other.listen(1)
        assert "another process also listens" in handle.frontend_port_problem()
    assert handle.frontend_port_problem() is None
    handle.terminate(grace=1)


def test_front_end_exiting_for_another_reason_has_no_port_problem(fake):
    fake.env["FAKE_FRONTEND_MODE"] = "exit"
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    deadline = time.monotonic() + 10
    while handle.poll() is None:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert handle.frontend_port_problem() is None
    handle.terminate(grace=1)


def test_restart_refuses_a_server_whose_engine_exited(fake):
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    _record(fake.records, "frontend-start")
    handle._engine.stop(1)
    with pytest.raises(launch.ChainStartupError, match="engine has exited"):
        handle.restart_frontend(_port())
    handle.terminate(grace=1)


def _engine_logs(handle):
    return sorted(glob.glob(os.path.join(handle.workdir, "engine-*.log")))


def test_trtllm_engine_gets_an_ephemeral_metrics_port(fake):
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    engine = _record(fake.records, "engine-start")["argv"]
    frontend = _record(fake.records, "frontend-start")["argv"]
    assert [a for a in engine if a.startswith(launch.ENGINE_METRICS_PORT_OPTION)] == [
        launch.ENGINE_METRICS_PORT_OPTION
    ]
    assert engine[engine.index(launch.ENGINE_METRICS_PORT_OPTION) + 1] == "0"
    assert not any(a.startswith(launch.ENGINE_METRICS_PORT_OPTION) for a in frontend)
    handle.terminate(grace=1)


@pytest.mark.parametrize(
    "given",
    [
        [launch.ENGINE_METRICS_PORT_OPTION, "18001"],
        [f"{launch.ENGINE_METRICS_PORT_OPTION}=18001"],
    ],
)
def test_metrics_port_the_command_gives_is_kept(fake, given):
    handle = _start(fake, ["--port", str(_port())] + given, chain="trtllm")
    engine = _record(fake.records, "engine-start")["argv"]
    frontend = _record(fake.records, "frontend-start")["argv"]
    assert [a for a in engine if a.startswith(launch.ENGINE_METRICS_PORT_OPTION)] == given[:1]
    assert engine[engine.index(given[0]) : engine.index(given[0]) + len(given)] == given
    assert not any(a.startswith(launch.ENGINE_METRICS_PORT_OPTION) for a in frontend)
    handle.terminate(grace=1)


def test_glm_engine_gets_no_metrics_port(fake, monkeypatch):
    monkeypatch.setenv(cv.ENV_GLM_SERVE_CMD, fake.exe)
    handle = _start(fake, ["--port", str(_port())], chain="glm")
    engine = _record(fake.records, "engine-start")["argv"]
    assert not any(a.startswith(launch.ENGINE_METRICS_PORT_OPTION) for a in engine)
    handle.terminate(grace=1)


def test_glm_chain_refuses_a_metrics_port_the_command_gives(fake, monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    monkeypatch.setenv(cv.ENV_GLM_SERVE_CMD, fake.exe)
    args = ["--port", str(_port()), launch.ENGINE_METRICS_PORT_OPTION, "18001"]
    with pytest.raises(launch.ChainError, match="not pass it to the glm chain's engine"):
        _start(fake, args, chain="glm")
    assert launch._HANDLES == []
    assert not os.listdir(fake.records)
    assert not glob.glob(os.path.join(os.environ[cv.ENV_RESULTS_DIR], "chain-*"))


@pytest.mark.parametrize("joined", [False, True])
def test_metrics_port_equal_to_the_front_end_port_is_refused(fake, monkeypatch, joined):
    monkeypatch.setattr(launch, "_HANDLES", [])
    port = str(_port())
    option = launch.ENGINE_METRICS_PORT_OPTION
    given = [f"{option}={port}"] if joined else [option, port]
    with pytest.raises(launch.ChainError, match="must differ from --port"):
        _start(fake, ["--port", port] + given, chain="trtllm")
    assert launch._HANDLES == []
    assert not os.listdir(fake.records)
    assert not glob.glob(os.path.join(os.environ[cv.ENV_RESULTS_DIR], "chain-*"))


def test_metrics_port_0_may_equal_the_front_end_port(fake, monkeypatch):
    class PastTheChecks(Exception):
        pass

    def results_dir():
        raise PastTheChecks

    monkeypatch.setattr(launch, "_results_dir", results_dir)
    with pytest.raises(PastTheChecks):
        _start(fake, ["--port", "0", launch.ENGINE_METRICS_PORT_OPTION, "0"], chain="trtllm")


@pytest.mark.parametrize(
    ("source", "accepts"),
    [
        ('option(\n    "--openengine-metrics-port",\n    type=click.IntRange(0, 65535))', True),
        ('option("--openengine-metrics-port", default=None, type=IntRange(min=0, max=9))', True),
        ('option("--openengine-metrics-port", type=click.IntRange(1, 65535))', False),
        ('option("--openengine-metrics-port", type=int)', False),
        ('option("--openengine-metrics-port", type=PORTS)\nPORTS = IntRange(0, 65535)', False),
        (
            '# "--openengine-metrics-port", type=click.IntRange(0,\n'
            'option("--openengine-metrics-port", type=click.IntRange(1, 65535))',
            False,
        ),
        ('option("--openengine-metrics-port", type=click.IntRange(0,', False),
        ("", False),
    ],
)
def test_metrics_port_probe_needs_the_option_to_take_0(monkeypatch, source, accepts):
    monkeypatch.setattr(launch, "_serve_source", lambda: source)
    assert launch.engine_supports_metrics_port() is accepts


def test_metrics_port_probe_accepts_this_checkouts_trtllm_serve(monkeypatch):
    monkeypatch.setattr(launch, "_serve_source", lambda: _SERVE_PY.read_text(encoding="utf-8"))
    assert launch.engine_supports_metrics_port()


def test_no_metrics_port_for_a_trtllm_serve_without_the_option(fake, monkeypatch):
    monkeypatch.setattr(launch, "engine_supports_metrics_port", lambda: False)
    handle = _start(fake, ["--port", str(_port())], chain="trtllm")
    engine = _record(fake.records, "engine-start")["argv"]
    assert launch.ENGINE_METRICS_PORT_OPTION not in engine
    handle.terminate(grace=1)


def test_engines_with_prometheus_do_not_take_its_default_port(fake, monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    blocker, default = _blocked_port()
    # The fake engine's stand-in for 8001, which another listener holds.
    fake.env["FAKE_ENGINE_METRICS"] = str(default)
    with blocker:
        handles = [_start(fake, ["--port", str(_port())], chain="trtllm") for _ in range(2)]
        bound = []
        for handle in handles:
            _wait_healthy(handle)
            log = open(handle.engine_log).read()
            (port,) = re.findall(r"OpenEngine Prometheus endpoint: 127\.0\.0\.1:(\d+)/", log)
            bound.append(int(port))
        assert launch._HANDLES == handles
        assert all(len(_engine_logs(h)) == 1 for h in handles)
        assert default not in bound and 0 not in bound and bound[0] != bound[1]
        for handle in handles:
            handle.terminate(grace=1)


def test_metrics_port_the_command_gives_is_not_changed_when_taken(fake, monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    blocker, taken = _blocked_port()
    fake.env["FAKE_ENGINE_METRICS"] = str(taken)
    args = ["--port", str(_port()), launch.ENGINE_METRICS_PORT_OPTION, str(taken)]
    with blocker, pytest.raises(launch.ChainStartupError, match="exited with code 1"):
        _start(fake, args, chain="trtllm")
    (handle,) = launch._HANDLES
    assert len(_engine_logs(handle)) == 1
    assert f"('127.0.0.1', {taken})" in open(handle.engine_log).read()


def test_bind_error_on_another_port_is_not_retried(fake, monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    fake.env["FAKE_ENGINE_MODE"] = "exit"
    fake.env["FAKE_ENGINE_SAY"] = (
        "OSError: [Errno 98] Address already in use (while attempting to bind on address "
        "('127.0.0.1', 1))"
    )
    with pytest.raises(launch.ChainStartupError, match="exited with code 3"):
        _start(fake, ["--port", str(_port())], chain="trtllm")
    (handle,) = launch._HANDLES
    assert len(_engine_logs(handle)) == 1


def _distinct_ports(count):
    ports = set()
    while len(ports) < count:
        ports.add(_port())
    return list(ports)


def test_free_port_other_than_skips_the_ports_given(monkeypatch):
    ports = iter([5, 6, 7])
    monkeypatch.setattr(launch, "free_port", lambda: next(ports))
    assert launch._free_port_other_than(5, None, 6) == 7
    monkeypatch.setattr(launch, "free_port", lambda: 5)
    with pytest.raises(launch.ChainError, match="no free port"):
        launch._free_port_other_than(5)


def test_engine_port_avoids_the_ports_of_the_server(fake, monkeypatch):
    check_port, frontend_port, given = _distinct_ports(3)
    real_free_port = launch.free_port
    # The front-end check's port, then the ports the engine port must avoid.
    ports = iter([check_port, frontend_port, given, check_port])
    monkeypatch.setattr(launch, "free_port", lambda: next(ports, None) or real_free_port())
    args = ["--port", str(frontend_port), launch.ENGINE_METRICS_PORT_OPTION, str(given)]
    handle = _start(fake, args, chain="trtllm")
    engine_port = int(handle.engine_endpoint.rsplit(":", 1)[1])
    assert engine_port not in (check_port, frontend_port, given)
    assert len(_engine_logs(handle)) == 1
    handle.terminate(grace=1)
