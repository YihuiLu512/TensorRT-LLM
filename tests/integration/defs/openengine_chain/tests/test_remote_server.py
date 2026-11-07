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

    def poll(self):
        return self.exit_code

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
    def refused(url):
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
    monkeypatch.setattr(server_module.requests, "get", lambda url: Unavailable())
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
