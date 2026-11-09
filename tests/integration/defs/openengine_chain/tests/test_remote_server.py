# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RemoteOpenAIServer: unchanged with the switch off, the harness with it on."""

import contextlib
import importlib.util
import os
import pathlib
import sys
import types

import pytest
from openengine_chain import conventions as cv
from openengine_chain import remote

pytestmark = pytest.mark.cpu_only

_SERVER = (
    pathlib.Path(__file__).resolve().parents[4]
    / "unittest"
    / "llmapi"
    / "apps"
    / "openai_server.py"
)


@pytest.fixture
def server_module():
    pytest.importorskip("openai")
    pytest.importorskip("tensorrt_llm")
    spec = importlib.util.spec_from_file_location("openai_server_under_test", _SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeProc:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.calls = []

    def terminate(self):
        self.calls.append("terminate")

    def wait(self, timeout=None):
        self.calls.append("wait")
        return 0

    def kill(self):
        self.calls.append("kill")

    def poll(self):
        return None


@pytest.fixture
def popen(monkeypatch, server_module):
    made = []

    def fake_popen(*args, **kwargs):
        made.append(FakeProc(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(server_module.subprocess, "Popen", fake_popen)
    return made


def test_switch_off_starts_one_trtllm_serve_as_before(monkeypatch, server_module, popen):
    monkeypatch.delenv(cv.ENV_CHAIN, raising=False)
    monkeypatch.setattr(
        remote, "launch_for_remote_server", lambda *a, **k: pytest.fail("the harness must not run")
    )
    server = server_module.RemoteOpenAIServer(
        "/m", ["--max_batch_size", "4"], port=1234, wait=False
    )
    (proc,) = popen
    assert proc.args == (
        [
            "trtllm-serve",
            "/m",
            "--host",
            "localhost",
            "--port",
            "1234",
            "--max_batch_size",
            "4",
            "--num_serve_frontends",
            "1",
        ],
    )
    assert proc.kwargs == {"env": os.environ.copy(), "stdout": sys.stdout, "stderr": sys.stdout}
    assert server._chain_handle is None
    server.terminate()
    assert proc.calls == ["terminate", "wait"]
    assert server.proc is None


class FakeHandle:
    def __init__(self):
        self.frontend = FakeProc()
        self.terminated = 0
        self.exit_code = None
        self.port_problem = None

    def poll(self):
        return self.exit_code

    def frontend_port_problem(self):
        return self.port_problem

    def terminate(self):
        self.terminated += 1

    @contextlib.contextmanager
    def stop_on_error(self):
        try:
            yield
        except BaseException:
            self.terminate()
            raise


@pytest.fixture
def chain(monkeypatch):
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    launched = []

    def launch(cmd, **kwargs):
        launched.append((cmd, kwargs, FakeHandle()))
        return launched[-1][2]

    monkeypatch.setattr(remote, "launch_for_remote_server", launch)
    return launched


def test_switch_on_hands_the_command_to_the_harness(server_module, popen, chain):
    server = server_module.RemoteOpenAIServer(
        "/m", ["--max_batch_size", "4"], port=1234, wait=False
    )
    assert popen == []
    ((cmd, kwargs, handle),) = chain
    assert cmd == [
        "trtllm-serve",
        "/m",
        "--host",
        "localhost",
        "--port",
        "1234",
        "--max_batch_size",
        "4",
        "--num_serve_frontends",
        "1",
    ]
    assert kwargs["llmapi_launch"] is False and kwargs["role"] is None
    assert server.proc is handle.frontend
    server.terminate()
    assert handle.terminated == 1
    assert handle.frontend.calls == []


def test_switch_on_stops_both_processes_when_the_server_never_gets_ready(
    monkeypatch, server_module, popen, chain
):
    def fail(self, timeout):
        raise RuntimeError("Server exited unexpectedly.")

    monkeypatch.setattr(server_module.RemoteOpenAIServer, "wait_for_server", fail)
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        server_module.RemoteOpenAIServer("/m", [], port=1234)
    assert chain[0][2].terminated == 1


def test_switch_on_failed_start_releases_the_extra_config(monkeypatch, server_module, popen):
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    seen = []

    def launch(cmd, **kwargs):
        seen.append(cmd[cmd.index("--extra_llm_api_options") + 1])
        raise remote.ChainError("engine exited")

    monkeypatch.setattr(remote, "launch_for_remote_server", launch)
    with pytest.raises(remote.ChainError):
        server_module.RemoteOpenAIServer("/m", [], port=1234, extra_config={"a": 1})
    assert seen and not os.path.exists(seen[0])


def test_switch_on_start_that_times_out_releases_the_extra_config_once(
    monkeypatch, server_module, popen, chain, capsys
):
    def refused(url, **kwargs):
        raise server_module.requests.ConnectionError("refused")

    # The clock passes the timeout only after the first refused /health, so the wait
    # loop's own timeout path, which calls terminate(), ends the start.
    clock = iter([0, 0, 1])
    fake_time = types.SimpleNamespace(time=lambda: next(clock, 1), sleep=lambda s: None)
    monkeypatch.setattr(server_module.requests, "get", refused)
    monkeypatch.setattr(server_module, "time", fake_time)
    monkeypatch.setattr(server_module.RemoteOpenAIServer, "MAX_SERVER_START_WAIT_S", 0)
    with pytest.raises(RuntimeError, match="failed to start in time"):
        server_module.RemoteOpenAIServer("/m", [], port=1234, extra_config={"a": 1})
    assert "Error removing" not in capsys.readouterr().out


class Unavailable:
    status_code = 503


@pytest.mark.parametrize(
    "exit_code, timeout, error",
    [(1, 600, "exited unexpectedly"), (None, 0, "failed to start in time")],
)
def test_switch_on_wait_ends_while_the_front_end_answers_503(
    monkeypatch, server_module, popen, chain, exit_code, timeout, error
):
    server = server_module.RemoteOpenAIServer("/m", [], port=1234, wait=False)
    handle = chain[0][2]
    handle.exit_code = exit_code
    monkeypatch.setattr(server_module.requests, "get", lambda url, **kwargs: Unavailable())
    monkeypatch.setattr(server_module.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match=error):
        server.wait_for_server(timeout=timeout)


def test_remote_entry_rejects_unsupported_modes():
    with pytest.raises(remote.ChainError, match="llmapi-launch"):
        remote.launch_for_remote_server(
            ["trtllm-serve", "/m"], env={}, llmapi_launch=True, role=None, sink=None
        )
    with pytest.raises(remote.ChainError, match="server_role"):
        remote.launch_for_remote_server(
            ["trtllm-serve", "/m"], env={}, llmapi_launch=False, role="CONTEXT", sink=None
        )


def test_chain_without_the_harness_on_the_path_releases_its_files(
    monkeypatch, server_module, tmp_path
):
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    # As if PYTHONPATH lacked the harness package.
    monkeypatch.setitem(sys.modules, "openengine_chain.remote", None)
    made = []
    real = server_module.tempfile.NamedTemporaryFile

    def named(*args, **kwargs):
        made.append(real(*args, dir=tmp_path, **kwargs))
        return made[-1]

    monkeypatch.setattr(server_module.tempfile, "NamedTemporaryFile", named)
    log = tmp_path / "server.log"
    with pytest.raises(ImportError):
        server_module.RemoteOpenAIServer(
            "/m", [], port=1234, extra_config={"a": 1}, log_path=str(log)
        )
    assert made and not os.path.exists(made[0].name)


class Healthy:
    status_code = 200


def test_switch_off_health_check_has_no_timeout_as_before(monkeypatch, server_module, popen):
    monkeypatch.delenv(cv.ENV_CHAIN, raising=False)
    seen = []

    def get(url, **kwargs):
        seen.append(kwargs)
        return Healthy()

    monkeypatch.setattr(server_module.requests, "get", get)
    server_module.RemoteOpenAIServer("/m", [], port=1234)
    assert seen == [{}]


def test_switch_on_health_check_has_a_timeout(monkeypatch, server_module, popen, chain):
    seen = []

    def get(url, **kwargs):
        seen.append(kwargs)
        return Healthy()

    monkeypatch.setattr(server_module.requests, "get", get)
    server_module.RemoteOpenAIServer("/m", [], port=1234)
    assert seen == [{"timeout": server_module.RemoteOpenAIServer.CHAIN_HEALTH_TIMEOUT_S}]


class RestartableHandle(FakeHandle):
    """Its front end loses the port ``lost`` times, then serves."""

    def __init__(self, lost):
        super().__init__()
        self.lost = lost
        self.exit_code = 1 if lost else None
        self.port_problem = "port in use" if lost else None
        self.restarts = []

    def restart_frontend(self, port):
        self.restarts.append(port)
        self.frontend = FakeProc()
        self.lost -= 1
        self.exit_code = 1 if self.lost else None
        self.port_problem = "port in use" if self.lost else None


@pytest.mark.parametrize("lost, ok", [(1, True), (2, True), (3, False)])
def test_switch_on_front_end_that_lost_a_picked_port_moves_to_another(
    monkeypatch, server_module, popen, lost, ok
):
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    handle = RestartableHandle(lost)
    monkeypatch.setattr(remote, "launch_for_remote_server", lambda cmd, **kwargs: handle)
    ports = iter(range(2000, 2010))
    monkeypatch.setattr(server_module, "get_free_port", lambda: next(ports))
    urls = []

    def get(url, **kwargs):
        urls.append(url)
        return Healthy()

    monkeypatch.setattr(server_module.requests, "get", get)
    monkeypatch.setattr(server_module.time, "sleep", lambda s: None)
    if ok:
        server = server_module.RemoteOpenAIServer("/m", [])
        assert handle.restarts == list(range(2001, 2001 + lost))
        assert server.port == 2000 + lost and server.proc is handle.frontend
        assert urls[-1] == f"http://localhost:{server.port}/health"
    else:
        with pytest.raises(RuntimeError, match="cannot serve on its port: port in use"):
            server_module.RemoteOpenAIServer("/m", [])
        attempts = server_module.RemoteOpenAIServer.CHAIN_FRONTEND_PORT_ATTEMPTS
        assert handle.restarts == list(range(2001, 2000 + attempts))
        assert handle.terminated == 1


def test_switch_on_front_end_that_lost_a_given_port_fails_at_once(
    monkeypatch, server_module, popen
):
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    handle = RestartableHandle(1)
    monkeypatch.setattr(remote, "launch_for_remote_server", lambda cmd, **kwargs: handle)
    monkeypatch.setattr(server_module.requests, "get", lambda url, **kwargs: Healthy())
    monkeypatch.setattr(server_module.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="cannot serve on its port"):
        server_module.RemoteOpenAIServer("/m", [], port=1234)
    assert handle.restarts == [] and handle.terminated == 1


def test_switch_on_answer_from_another_listener_moves_the_front_end(
    monkeypatch, server_module, popen
):
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    handle = RestartableHandle(0)
    # The front end runs, but another process answers /health on its port.
    handle.port_problem = "another process also listens"
    handle.lost = 1
    monkeypatch.setattr(remote, "launch_for_remote_server", lambda cmd, **kwargs: handle)
    ports = iter(range(2000, 2010))
    monkeypatch.setattr(server_module, "get_free_port", lambda: next(ports))
    monkeypatch.setattr(server_module.requests, "get", lambda url, **kwargs: Healthy())
    server = server_module.RemoteOpenAIServer("/m", [])
    assert handle.restarts == [2001] and server.port == 2001


def test_switch_on_answer_after_the_chain_exited_is_not_ready(monkeypatch, server_module, popen):
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    handle = RestartableHandle(0)
    monkeypatch.setattr(remote, "launch_for_remote_server", lambda cmd, **kwargs: handle)

    def get(url, **kwargs):
        # The front end exits, for no port problem, while another process answers.
        handle.exit_code = 1
        return Healthy()

    monkeypatch.setattr(server_module.requests, "get", get)
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        server_module.RemoteOpenAIServer("/m", [])
    assert handle.restarts == [] and handle.terminated == 1


def test_switch_on_engine_exit_is_not_a_port_problem(monkeypatch, server_module, popen, chain):
    server = server_module.RemoteOpenAIServer("/m", [], wait=False)
    handle = chain[0][2]
    handle.exit_code = 1
    monkeypatch.setattr(server_module.requests, "get", lambda url, **kwargs: Unavailable())
    monkeypatch.setattr(server_module.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        server.wait_for_server(timeout=600)


# ---------------------------------------------------------------- real processes


@pytest.fixture
def fake_chain(tmp_path, monkeypatch, server_module):
    """``trtllm-serve`` on PATH is the fake of fake_serve.py; the chain harness starts it."""
    from openengine_chain import launch

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "trtllm-serve"
    exe.write_text(
        f"#!{sys.executable}\n" + (pathlib.Path(__file__).parent / "fake_serve.py").read_text()
    )
    exe.chmod(0o755)
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(launch, "require_plugin", lambda: None)
    monkeypatch.setattr(launch, "frontend_supports_ready_timeout", lambda: True)
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    monkeypatch.setenv(cv.ENV_RESULTS_DIR, str(tmp_path / "results"))
    monkeypatch.setattr(server_module.RemoteOpenAIServer, "CHAIN_HEALTH_TIMEOUT_S", 1)
    env = dict(os.environ, FAKE_RECORD_DIR=str(records))
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env.pop(cv.ENV_COUNTER_DIR, None)
    return env


def _silent_listener():
    """A socket that accepts connections and never answers, on a port it took first."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    return listener


def test_chain_front_end_whose_picked_port_is_taken_moves_to_another(
    monkeypatch, server_module, fake_chain, tmp_path
):
    import time

    with _silent_listener() as listener:
        taken = listener.getsockname()[1]
        real = server_module.get_free_port
        ports = iter([taken])
        monkeypatch.setattr(server_module, "get_free_port", lambda: next(ports, None) or real())
        start = time.monotonic()
        server = server_module.RemoteOpenAIServer(
            "/m", [], host="127.0.0.1", env=fake_chain, log_path=str(tmp_path / "server.log")
        )
        try:
            assert server.port != taken
            assert time.monotonic() - start < 60
            assert (
                server_module.requests.get(server.url_for("health"), timeout=5).status_code == 200
            )
            notes = server._chain_handle.teardown_notes
            assert any(f"could not serve on port {taken}" in note for note in notes)
        finally:
            server.terminate()


def test_chain_front_end_whose_given_port_is_taken_fails_fast(server_module, fake_chain, tmp_path):
    import time

    with _silent_listener() as listener:
        taken = listener.getsockname()[1]
        start = time.monotonic()
        with pytest.raises(RuntimeError, match=f"port {taken} is in use"):
            server_module.RemoteOpenAIServer(
                "/m",
                [],
                host="127.0.0.1",
                port=taken,
                env=fake_chain,
                log_path=str(tmp_path / "server.log"),
            )
        # Without a timeout, /health would wait for the listener's answer for good.
        assert time.monotonic() - start < 60


def test_chain_server_whose_front_end_refuses_fails_without_waiting_for_the_engine(
    monkeypatch, server_module, fake_chain, tmp_path
):
    import time

    from openengine_chain import expectations as ex
    from openengine_chain import launch, pytest_plugin

    monkeypatch.setattr(launch, "_HANDLES", [])
    # The fake engine would take 600 s to turn SERVING.
    fake_chain["FAKE_ENGINE_DELAY"] = "600"
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="exited unexpectedly") as info:
        server_module.RemoteOpenAIServer(
            "/m",
            ["--max_beam_width", "2"],
            host="127.0.0.1",
            env=fake_chain,
            log_path=str(tmp_path / "server.log"),
        )
    assert time.monotonic() - start < 60
    (handle,) = launch._HANDLES
    assert not handle.alive and handle.engine.poll() is not None
    assert os.path.basename(handle.frontend_log).startswith("frontend-check-")
    # The plugin's STARTUP_REJECTED check finds the front end's own refusal.
    refused = ex.Expectation(
        "WT-11",
        "_test_x.py::test_x",
        ex.STARTUP_REJECTED,
        "beam",
        log_contains="max_beam_width > 1 is not supported",
    )
    assert pytest_plugin._startup_rejected(refused, info.value, [handle]) == (True, "")
