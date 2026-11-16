# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How long the harness waits: time limits of tests, processes after SIGKILL, servers outside the chain."""

import importlib.util
import os
import pathlib
import signal
import threading
import time
import types

import plugin_support
import pytest
from openengine_chain import conventions as cv
from openengine_chain import expectations as ex
from openengine_chain import launch, remote

pytestmark = pytest.mark.cpu_only

E = ex.Expectation


# Time limits in the expectation tables.


def test_time_limit_needs_seconds_and_only_it_takes_them():
    with pytest.raises(ValueError, match="seconds is required"):
        E("WT-60-fix1", "t.py::t", ex.TIME_LIMIT, "r")
    with pytest.raises(ValueError, match="seconds is required"):
        E("WT-60-fix1", "t.py::t", ex.SKIP, "r", seconds=900)
    for bad in (0, -1, 1.5, "900", True):
        with pytest.raises(ValueError, match="positive int"):
            E("WT-60-fix1", "t.py::t", ex.TIME_LIMIT, "r", seconds=bad)
    assert E("WT-60-fix1", "t.py::t", ex.TIME_LIMIT, "r", seconds=900).seconds == 900


def test_a_test_has_at_most_one_time_limit_besides_its_other_expectations():
    table = [
        E("WT-60-fix1", "t.py::test_*", ex.TIME_LIMIT, "r", seconds=900),
        E("WT-11", "t.py::test_a", ex.SKIP, "r"),
        E("WT-38", "t.py::test_a", ex.LEG_EXPECT, "r", leg=ex.GEN_ZERO_OK),
    ]
    found = ex.applicable(table, "dir/t.py::test_a", "trtllm")
    assert ex.time_limit(found) is table[0]
    assert ex.outcome_expectation(found) is table[1]
    assert ex.leg_expectation(found) is table[2]
    assert ex.time_limit(ex.applicable(table[1:], "dir/t.py::test_a", "trtllm")) is None
    clash = table + [E("WT-61", "t.py::test_a", ex.TIME_LIMIT, "r", seconds=600)]
    with pytest.raises(ValueError, match="conflicting expectations"):
        ex.applicable(clash, "dir/t.py::test_a", "trtllm")


@pytest.mark.parametrize("chain, limited", [("trtllm", True), ("ds", False), ("glm", False)])
def test_shipped_table_gives_xd_313_two_cold_engine_starts_on_the_trtllm_chain(chain, limited):
    nodeid = "unittest/llmapi/apps/test_disagg_serving_perf_metrics.py::test_completion_metrics"
    limit = ex.time_limit(ex.applicable(ex.load(), nodeid, chain))
    if not limited:
        assert limit is None
        return
    assert limit.wt == "WT-60-fix1"
    # Two engines that start one after the other, each in up to about 240 s cold.
    assert limit.seconds >= 2 * 240 + 300
    other = "unittest/llmapi/apps/test_disagg_serving_perf_metrics.py::test_other"
    assert ex.time_limit(ex.applicable(ex.load(), other, chain)) is None


LIMITED = """
import time

import pytest

@pytest.mark.timeout(300)
def test_limited(request):
    assert request.node.get_closest_marker("timeout").args == (900,)

@pytest.mark.timeout(300)
def test_own_limit(request):
    assert request.node.get_closest_marker("timeout").args == (300,)

@pytest.mark.timeout(1)
def test_slow_limited():
    time.sleep(2)

@pytest.mark.timeout(1)
def test_slow():
    time.sleep(2)
"""

TIME_LIMITS = [
    E("WT-60-fix1", "test_limits.py::test_limited", ex.TIME_LIMIT, "r", seconds=900),
    E("WT-60-fix1", "test_limits.py::test_slow_limited", ex.TIME_LIMIT, "r", seconds=30),
]


@pytest.mark.parametrize("chain", ["trtllm", None])
def test_plugin_puts_a_time_limit_before_the_tests_own(pytester, monkeypatch, tmp_path, chain):
    # pytest-timeout reads the item's closest timeout mark; with the time_limit entry
    # in place a test that outlives its own limit is not stopped.
    pytest.importorskip("pytest_timeout")
    plugin_support.install(monkeypatch)
    monkeypatch.setattr(ex, "load", lambda directory=None: list(TIME_LIMITS))
    if chain is None:
        monkeypatch.delenv(cv.ENV_CHAIN, raising=False)
    else:
        monkeypatch.setenv(cv.ENV_CHAIN, chain)
    monkeypatch.delenv(cv.ENV_RESULTS_DIR, raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    pytester.makepyfile(test_limits=LIMITED)
    # The thread method ends the whole process on a timeout; the signal one fails the test.
    result = pytester.runpytest_inprocess("-p", cv.PLUGIN_MODULE, "--timeout-method=signal")
    if chain is None:
        # No chain, no expectation tables: every test keeps its own limit.
        result.assert_outcomes(passed=1, failed=3)
        return
    result.assert_outcomes(passed=3, failed=1)
    result.stdout.fnmatch_lines(["*FAILED*test_limits.py::test_slow*Timeout*"])


# Processes after SIGKILL.


def _gone(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def _late_sigkill(monkeypatch, pid, delay):
    """SIGKILL to ``pid`` arrives ``delay`` seconds late, once; ``None`` drops it every time.

    It stands for a process that takes time to exit after SIGKILL, as one that
    holds GPUs does, or for one that SIGKILL cannot end.
    """
    real_kill = os.kill
    timers = []

    def kill(target, sig):
        if target != pid() or sig != signal.SIGKILL:
            return real_kill(target, sig)
        if delay is not None and not timers:
            timers.append(threading.Timer(delay, real_kill, (target, sig)))
            timers[0].start()

    monkeypatch.setattr(os, "kill", kill)
    return timers


def _engine_with_a_child(tmp_path):
    """An engine whose child stays in its session when the engine exits; returns both."""
    pid_file = tmp_path / "child.pid"
    proc = launch.LoggedProcess(
        "engine",
        # The child does not hold the log's pipe, so the log ends when the engine exits.
        ["sh", "-c", f"sleep 60 > /dev/null 2>&1 & echo $! > {pid_file}; wait"],
        os.environ,
        str(tmp_path),
        None,
    )
    deadline = time.monotonic() + 10
    while not (pid_file.exists() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline, "the child did not start"
        time.sleep(0.05)
    return proc, int(pid_file.read_text())


def test_default_wait_covers_a_large_engine_freeing_its_gpus():
    assert launch.SURVIVOR_WAIT_S >= 60


def test_child_slow_to_exit_after_sigkill_is_not_a_survivor(tmp_path, monkeypatch):
    proc, child = _engine_with_a_child(tmp_path)
    timers = _late_sigkill(monkeypatch, lambda: child, 1.5)
    try:
        assert proc.stop(grace=5) == []
        assert timers and _gone(child)
        (note,) = proc.notes
        assert f"processes [{child}] took" in note and "to exit after SIGKILL" in note
    finally:
        monkeypatch.undo()
        if not _gone(child):
            os.kill(child, signal.SIGKILL)


def test_engine_slow_to_exit_after_sigkill_is_not_a_survivor(tmp_path, monkeypatch):
    # The engine ignores SIGTERM, so it gets SIGKILL after the grace period.
    ready = tmp_path / "ready"
    proc = launch.LoggedProcess(
        "engine",
        ["sh", "-c", f"trap '' TERM; touch {ready}; while :; do sleep 0.1; done"],
        os.environ,
        str(tmp_path),
        None,
    )
    deadline = time.monotonic() + 10
    while not ready.exists():
        assert time.monotonic() < deadline, "the engine did not start"
        time.sleep(0.05)
    timers = _late_sigkill(monkeypatch, lambda: proc.pid, 1.5)
    try:
        assert proc.stop(grace=0.5) == []
        assert timers and proc.poll() == -signal.SIGKILL
    finally:
        monkeypatch.undo()
        if proc.poll() is None:
            proc.proc.kill()
            proc.proc.wait()


@pytest.mark.parametrize("delay", [1.5, None], ids=["slow", "survivor"])
def test_process_still_there_when_the_wait_ends_is_a_survivor(tmp_path, monkeypatch, delay):
    # A slow exit that outlasts the wait is reported, as the old 5 s wait reported the
    # engines of R11; a process that SIGKILL cannot end is reported once the wait ends.
    monkeypatch.setattr(launch, "SURVIVOR_WAIT_S", 0.5)
    proc, child = _engine_with_a_child(tmp_path)
    timers = _late_sigkill(monkeypatch, lambda: child, delay)
    try:
        (problem,) = proc.stop(grace=5)
        assert problem == f"engine pid {proc.pid}: processes [{child}] survived SIGKILL"
        assert proc.notes == []
    finally:
        for timer in timers:
            timer.cancel()
        monkeypatch.undo()
        if not _gone(child):
            os.kill(child, signal.SIGKILL)


# Servers outside the chain, such as the disaggregated server of chain workers.

_SERVER = (
    pathlib.Path(__file__).resolve().parents[4]
    / "unittest"
    / "llmapi"
    / "apps"
    / "openai_server.py"
)


@pytest.fixture
def server_module(monkeypatch):
    pytest.importorskip("openai")
    pytest.importorskip("tensorrt_llm")
    spec = importlib.util.spec_from_file_location("openai_server_under_test", _SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    made = []

    class FakeProc:
        def __init__(self, *args, **kwargs):
            self.calls = []
            made.append(self)

        def terminate(self):
            self.calls.append("terminate")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return None

    monkeypatch.setattr(module.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(module, "time", types.SimpleNamespace(time=time.time, sleep=lambda s: None))
    module.made = made
    return module


class Worker:
    """A chain worker's handle: ``exit_code`` set means its front end or engine exited."""

    def __init__(self, role="ctx", exit_code=None, port_problem=None, ended=False):
        self.role = role
        self.port = 4100
        self.exit_code = exit_code
        self.port_problem = port_problem
        self.frontend_log = "/logs/frontend-1.log"
        self.engine_log = "/logs/engine-2.log"
        self.started_seq = launch.next_seq()
        self.ended_seq = launch.next_seq() if ended else None
        launch._HANDLES.append(self)

    @property
    def alive(self):
        return self.ended_seq is None

    def poll(self):
        return self.exit_code

    def frontend_port_problem(self):
        return self.port_problem


class Unavailable:
    status_code = 503


class Healthy:
    status_code = 200


@pytest.fixture
def workers(monkeypatch):
    monkeypatch.setattr(launch, "_HANDLES", [])
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    monkeypatch.delenv(cv.ENV_TOPOLOGY, raising=False)


def test_exited_workers_names_each_worker_that_exited_and_why(workers):
    Worker("gen")
    Worker("ctx", exit_code=1, port_problem="the front end exited: port 4100 is in use; log x")
    Worker("gen", exit_code=-9)
    # Stopped by the test, not a worker, or still running: none of them is waited for.
    Worker("ctx", exit_code=0, ended=True)
    Worker(None, exit_code=1)
    assert remote.exited_workers() == [
        "the ctx worker on port 4100 exited (1): the front end exited: port 4100 is in use; log x",
        "the gen worker on port 4100 exited (-9): logs /logs/frontend-1.log and /logs/engine-2.log",
    ]


def test_disaggregated_server_stops_waiting_once_a_chain_worker_has_exited(
    workers, server_module, monkeypatch
):
    Worker("ctx", exit_code=1, port_problem="the front end exited: port 4100 is in use; log x")
    monkeypatch.setattr(server_module.requests, "get", lambda url, **kwargs: Unavailable())
    with pytest.raises(RuntimeError, match="the ctx worker on port 4100 exited .1.: the front"):
        server_module.RemoteDisaggOpenAIServer(ctx_servers=[], gen_servers=[], port=4321)
    (proc,) = server_module.made
    assert proc.calls == ["terminate"]
    config = os.path.join(server_module.tempfile.gettempdir(), "disagg_config_4321.yaml")
    assert not os.path.exists(config)


def test_disaggregated_server_answering_503_stops_waiting_at_its_timeout(
    workers, server_module, monkeypatch
):
    Worker("ctx")
    monkeypatch.setattr(server_module.requests, "get", lambda url, **kwargs: Unavailable())
    clock = iter(range(0, 100, 10))
    monkeypatch.setattr(server_module.time, "time", lambda: next(clock))
    server = server_module.RemoteDisaggOpenAIServer(
        ctx_servers=[], gen_servers=[], port=4322, wait_ready=False
    )
    with pytest.raises(RuntimeError, match="failed to start in time"):
        server.wait_for_server(timeout=25)
    assert server_module.made[-1].calls == ["terminate"]


def test_disaggregated_server_with_running_chain_workers_waits_for_its_health(
    workers, server_module, monkeypatch
):
    Worker("ctx")
    Worker("gen")
    answers = iter([Unavailable(), Unavailable(), Healthy()])
    monkeypatch.setattr(server_module.requests, "get", lambda url, **kwargs: next(answers))
    server_module.RemoteDisaggOpenAIServer(ctx_servers=[], gen_servers=[], port=4323)
    assert server_module.made[-1].calls == []


def test_without_a_chain_the_disaggregated_server_does_not_look_at_workers(
    server_module, monkeypatch
):
    monkeypatch.delenv(cv.ENV_CHAIN, raising=False)
    monkeypatch.delenv(cv.ENV_TOPOLOGY, raising=False)
    monkeypatch.setattr(remote, "exited_workers", lambda: pytest.fail("must not be asked"))
    answers = iter([Unavailable(), Healthy()])
    monkeypatch.setattr(server_module.requests, "get", lambda url, **kwargs: next(answers))
    server_module.RemoteDisaggOpenAIServer(ctx_servers=[], gen_servers=[], port=4324)
