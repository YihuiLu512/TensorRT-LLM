# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Several OpenEngine serving frontends (SV-MF), on CPU against the gRPC fake engine.

The process tests start the real ``trtllm-serve`` command line, which spawns
the attached frontends as it would in production.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess  # nosec B404
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import click
import psutil
import pytest
import requests

from tensorrt_llm.commands import serve as serve_cmd
from tensorrt_llm.grpc.openengine_conventions import (
    COUNTER_GENERATE_RECEIVED,
    SERVE_COUNTER_DIR_ENV,
    SERVE_COUNTER_GENERATE_SENT,
    compute_tokenizer_fingerprint,
)
from tensorrt_llm.serve.multi_frontend import AttachedFrontendWatchdog
from tensorrt_llm.serve.openengine_poc import contracts, multi_frontend
from tensorrt_llm.serve.openengine_poc import launch as launch_module
from tensorrt_llm.serve.openengine_poc.facade import _FacadeExecutor
from tensorrt_llm.usage import usage_lib

_CLIENT_TESTS = Path(__file__).resolve().parents[3] / "grpc" / "openengine_client"
if str(_CLIENT_TESTS) not in sys.path:
    sys.path.insert(0, str(_CLIENT_TESTS))

from fake_openengine import PROFILES, FakeEngine, running_in_thread  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

MODEL = "stub-model"
NUM_FRONTENDS = 4
STARTUP_SECONDS = 300
VOCAB = {"<unk>": 0, "</s>": 1, "He": 2, "llo": 3, " world": 4, "!": 5, " again": 6}
EOS = 1


@pytest.fixture(autouse=True)
def _fresh_slots():
    contracts.reset_slot_cache()
    yield
    contracts.reset_slot_cache()


# -- Postprocessing workers per frontend ----------------------------------------


@pytest.mark.parametrize(
    "total, shares",
    [
        (0, [0, 0, 0, 0]),
        (8, [2, 2, 2, 2]),
        (6, [2, 2, 1, 1]),
        (5, [2, 1, 1, 1]),
        # Every frontend gets one once there are any.
        (2, [1, 1, 1, 1]),
    ],
)
def test_postproc_workers_are_split_between_the_frontends(total, shares):
    assert [multi_frontend.postproc_share(total, 4, i) for i in range(4)] == shares


def test_one_frontend_postprocesses_with_every_worker():
    assert multi_frontend.postproc_share(3, 1, 0) == 3


# -- Choosing how to launch -------------------------------------------------------


def _config(**changes: Any) -> contracts.FrontendLaunchConfig:
    fields: dict[str, Any] = dict(
        host="127.0.0.1", port=8000, llm_args={"model": "m"}, endpoint="e:1"
    )
    fields.update(changes)
    return contracts.FrontendLaunchConfig(**fields)


def test_the_slot_resolves_to_this_launcher():
    impl, source = contracts.resolve_slot("launch_frontends")
    assert impl is multi_frontend.launch_frontends
    assert source == f"{multi_frontend.__name__}:launch_frontends"


@pytest.mark.parametrize("llm_args", [{"model": "m"}, {"model": "m", "num_serve_frontends": 1}])
def test_one_frontend_runs_as_the_default_launcher(llm_args):
    config = _config(llm_args=llm_args)
    with (
        mock.patch.object(contracts, "default_launch_frontends") as default,
        mock.patch.object(multi_frontend, "_serve_launcher") as launcher,
    ):
        contracts.launch_frontends(config)
    default.assert_called_once_with(config)
    launcher.assert_not_called()


def test_several_frontends_run_the_launcher():
    config = _config(llm_args={"model": "m", "num_serve_frontends": 3})
    with (
        mock.patch.object(contracts, "default_launch_frontends") as default,
        mock.patch.object(multi_frontend, "_serve_launcher") as launcher,
    ):
        contracts.launch_frontends(config)
    launcher.assert_called_once_with(config, 3)
    default.assert_not_called()


@pytest.mark.parametrize("value", [0, -1, "2", 2.0, True])
def test_a_bad_frontend_count_is_a_usage_error(value):
    with pytest.raises(click.UsageError, match="num_serve_frontends"):
        contracts.launch_frontends(_config(llm_args={"model": "m", "num_serve_frontends": value}))


@pytest.mark.parametrize(
    "changes",
    [
        {"port": 0, "report_addr": "/tmp/addr"},
        {"report_addr": "/tmp/addr"},
        {"port": 0},
    ],
)
def test_several_frontends_refuse_port_zero_and_report_addr(changes):
    """As the in-process server does; nothing is bound or started before."""
    config = _config(llm_args={"model": "m", "num_serve_frontends": 2}, **changes)
    with (
        mock.patch.object(multi_frontend, "_bind_shared_port") as bind,
        mock.patch.object(multi_frontend, "_build_frontend") as build,
        pytest.raises(click.UsageError, match="single serving frontend"),
    ):
        contracts.launch_frontends(config)
    bind.assert_not_called()
    build.assert_not_called()


def test_an_attached_frontend_needs_several_frontends(monkeypatch):
    monkeypatch.setenv(multi_frontend.LAUNCHER_UDS_ENV, "/tmp/launcher.sock")
    with pytest.raises(click.UsageError, match="num_serve_frontends is 1"):
        contracts.launch_frontends(_config())
    # Consumed, so the processes it starts do not inherit it.
    assert multi_frontend.LAUNCHER_UDS_ENV not in os.environ


@pytest.mark.parametrize("value", [None, "x", "0", "2"])
def test_an_attached_frontend_needs_its_frontend_id(monkeypatch, value):
    monkeypatch.setenv(multi_frontend.LAUNCHER_UDS_ENV, "/tmp/launcher.sock")
    if value is None:
        monkeypatch.delenv("TLLM_EXECUTOR_FRONTEND_ID", raising=False)
    else:
        monkeypatch.setenv("TLLM_EXECUTOR_FRONTEND_ID", value)
    config = _config(llm_args={"model": "m", "num_serve_frontends": 2})
    with (
        mock.patch.object(multi_frontend, "_bind_shared_port") as bind,
        pytest.raises(click.UsageError, match="TLLM_EXECUTOR_FRONTEND_ID"),
    ):
        contracts.launch_frontends(config)
    bind.assert_not_called()


@pytest.mark.parametrize("value", [2.9, -1, True, "2"])
def test_a_bad_postprocessing_worker_count_is_a_usage_error(value):
    config = _config(
        llm_args={"model": "m", "num_serve_frontends": 2, "num_postprocess_workers": value}
    )
    with (
        mock.patch.object(multi_frontend, "_bind_shared_port") as bind,
        pytest.raises(click.UsageError, match="num_postprocess_workers"),
    ):
        contracts.launch_frontends(config)
    bind.assert_not_called()


# -- How an attached frontend follows the launcher ---------------------------------


def _watchdog_verdict(statuses: list[int | None]) -> list[BaseException]:
    """What the launcher watchdog of an attached frontend decides on these ``/health`` answers.

    The last answer repeats.
    """

    class Forwarder:
        def __init__(self):
            self.answers = iter(statuses)
            self.last = None

        async def get_status(self, path, timeout_s=5.0):
            self.last = next(self.answers, self.last)
            return self.last

    server = SimpleNamespace(_launcher_forwarder=Forwarder())
    multi_frontend._follow_launcher(SimpleNamespace(_executor=_FacadeExecutor()), server)
    dead: list[BaseException] = []
    watchdog = AttachedFrontendWatchdog(
        server._launcher_forwarder,
        os.getppid(),
        dead.append,
        ppid_interval_s=0.01,
        health_interval_s=0.01,
        health_failures=3,
    )

    async def run() -> None:
        task = asyncio.create_task(watchdog.run())
        await asyncio.sleep(0.5)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run())
    return dead


def test_an_attached_frontend_keeps_serving_while_the_engine_is_unhealthy():
    # The launcher answers /health with 503 while its engine connection is down;
    # each frontend watches the engine itself.
    assert _watchdog_verdict([200, 503]) == []


def test_an_attached_frontend_stops_when_the_launcher_no_longer_answers():
    dead = _watchdog_verdict([200, 503, None])
    assert len(dead) == 1
    assert "failed 3 consecutive times" in str(dead[0])


def test_a_lost_launcher_stops_an_attached_frontend_once():
    facade = SimpleNamespace(_executor=_FacadeExecutor())
    server = SimpleNamespace(_launcher_forwarder=SimpleNamespace(get_status=None))
    multi_frontend._follow_launcher(facade, server)
    with mock.patch("signal.raise_signal") as raise_signal:
        server._on_launcher_lost(RuntimeError("launcher gone"))
        server._on_launcher_lost(RuntimeError("launcher gone again"))
    raise_signal.assert_called_once_with(signal.SIGINT)
    # /health answers 503 and, with doing_shutdown set, sends no second SIGINT.
    assert facade._executor.doing_shutdown
    assert "launcher gone" == str(facade._executor._fatal_error)


def test_the_cleanup_runs_to_its_end_when_cancelled():
    steps = []

    async def cleanup() -> None:
        await asyncio.sleep(0.05)
        steps.append("first")
        await asyncio.sleep(0.05)
        steps.append("second")

    async def scenario() -> None:
        task = asyncio.create_task(multi_frontend._finish(cleanup()))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert steps == ["first", "second"]


def test_an_attached_frontend_listens_only_once_the_launcher_listens():
    async def scenario(path: str) -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(path)
            waiting = asyncio.create_task(
                multi_frontend._wait_for_launcher(path, os.getppid(), interval_s=0.01)
            )
            await asyncio.sleep(0.3)
            assert not waiting.done()
            sock.listen()
            await asyncio.wait_for(waiting, 10)

    with tempfile.TemporaryDirectory(prefix="wt14-") as directory:
        asyncio.run(scenario(os.path.join(directory, "launcher.sock")))


def test_an_attached_frontend_stops_waiting_when_the_launcher_is_gone(tmp_path):
    # -1 is never the parent pid, as after the launcher has exited.
    with pytest.raises(RuntimeError, match="exited before it started serving"):
        asyncio.run(multi_frontend._wait_for_launcher(str(tmp_path / "launcher.sock"), -1))


_SIGTERM_SCRIPT = """
import asyncio, os, signal, sys
from types import SimpleNamespace
import uvloop
from tensorrt_llm.serve.openengine_poc import multi_frontend

async def server(host, port, sockets):
    # A server that has not taken SIGTERM over yet.
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.sleep(60)

async def main(sigterm):
    sigterm.cancel_on_signal(asyncio.current_task())
    try:
        if sys.argv[1] == "while-starting":
            os.kill(os.getpid(), signal.SIGTERM)
        if sys.argv[1] == "before-the-server-takes-it-over":
            config = SimpleNamespace(host="127.0.0.1", port=0)
            await multi_frontend._serve(server, config, [], sigterm)
        await asyncio.sleep(60)
    finally:
        await asyncio.sleep(0)
        print("cleaned up", flush=True)

with multi_frontend._stopped_by_sigterm() as sigterm:
    if sys.argv[1] == "before-the-loop":
        os.kill(os.getpid(), signal.SIGTERM)
    uvloop.run(main(sigterm))
print("not ended", flush=True)
"""


@pytest.mark.parametrize(
    "when", ["before-the-loop", "while-starting", "before-the-server-takes-it-over"]
)
def test_sigterm_before_serving_cleans_up_then_ends_the_process(when):
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", _SIGTERM_SCRIPT, when],
        capture_output=True,
        text=True,
        timeout=STARTUP_SECONDS,
    )
    assert result.returncode == -signal.SIGTERM, result.stderr
    assert result.stdout.splitlines() == ["cleaned up"], result.stderr


# -- The command line --------------------------------------------------------------


def _run_cli(model_dir: Path, *args: str) -> tuple[mock.Mock, mock.Mock]:
    """Run ``trtllm-serve --openengine-endpoint`` with both launchers mocked."""
    with (
        mock.patch.object(serve_cmd, "get_is_diffusion_only_model", return_value=False),
        mock.patch.object(serve_cmd, "device_count", side_effect=AssertionError("no GPU here")),
        mock.patch.object(serve_cmd, "launch_server") as launch_server,
        mock.patch.object(launch_module, "launch_openengine_server") as single,
        mock.patch.object(multi_frontend, "_serve_launcher") as launcher,
        mock.patch.object(usage_lib, "_mark_llm_startup"),
        mock.patch.object(usage_lib, "_capture_startup_context"),
    ):
        serve_cmd.main(
            args=[str(model_dir), "--openengine-endpoint", "localhost:1", *args],
            standalone_mode=False,
        )
    launch_server.assert_not_called()
    return single, launcher


def test_cli_defaults_to_eight_frontends(tmp_path):
    single, launcher = _run_cli(tmp_path)
    single.assert_not_called()
    (config, count), _ = launcher.call_args
    assert count == 8
    assert config.llm_args["num_serve_frontends"] == 8


def test_cli_runs_the_frontend_count_it_is_given(tmp_path):
    _, launcher = _run_cli(tmp_path, "--num_serve_frontends", "3")
    assert launcher.call_args.args[1] == 3


@pytest.mark.parametrize(
    "args", [["--port", "0", "--report_addr", "ADDR"], ["--report_addr", "ADDR"]]
)
def test_cli_falls_back_to_one_frontend_for_port_zero_and_report_addr(tmp_path, args):
    args = [str(tmp_path / "addr") if a == "ADDR" else a for a in args]
    single, launcher = _run_cli(tmp_path, *args)
    launcher.assert_not_called()
    single.assert_called_once()
    assert single.call_args.args[2]["num_serve_frontends"] == 1
    assert single.call_args.kwargs["report_addr"] == str(tmp_path / "addr")


def test_cli_refuses_an_explicit_count_with_report_addr(tmp_path):
    """An explicit count is left alone, so the launcher refuses the conflict."""
    with (
        mock.patch.object(serve_cmd, "get_is_diffusion_only_model", return_value=False),
        mock.patch.object(serve_cmd, "launch_server"),
        mock.patch.object(usage_lib, "_mark_llm_startup"),
        mock.patch.object(usage_lib, "_capture_startup_context"),
        pytest.raises(click.UsageError, match="single serving frontend"),
    ):
        serve_cmd.main(
            args=[
                str(tmp_path),
                "--openengine-endpoint",
                "localhost:1",
                "--num_serve_frontends",
                "2",
                "--report_addr",
                str(tmp_path / "addr"),
            ],
            standalone_mode=False,
        )


# -- Processes: the real command line against the gRPC fake ------------------------


def _model_dir(path: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteFallback, Fuse
    from tokenizers.decoders import Sequence as DecoderSequence
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
    backend.decoder = DecoderSequence([ByteFallback(), Fuse()])
    backend.add_special_tokens(["</s>"])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    hf.save_pretrained(path)
    return path


def _engine(model_dir: Path) -> FakeEngine:
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    # The fingerprint of the tokenizer the frontends load from the directory.
    tokenizer = ModelLoader.load_hf_tokenizer(str(model_dir))
    fingerprint = compute_tokenizer_fingerprint(tokenizer, model_dir)
    profile = PROFILES["trtllm-like"].replace(
        model=MODEL,
        max_context_length=64,
        vocab_size=len(VOCAB),
        eos_token_ids=(EOS,),
        bos_token_id=None,
        fingerprint=fingerprint.to_extra(),
    )
    return FakeEngine(profile, record=True)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _connections(process: psutil.Process) -> list[Any]:
    method = getattr(process, "net_connections", None) or process.connections
    return method(kind="tcp")


def _listens(process: psutil.Process, port: int) -> bool:
    try:
        return any(
            c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port
            for c in _connections(process)
        )
    except psutil.NoSuchProcess:
        return False


def _model_info_calls(engine: FakeEngine) -> int:
    return sum(1 for rpc, _ in list(engine.control_calls) if rpc == "GetModelInfo")


def _counts(counter_dir: Path) -> dict[int, int]:
    """Each frontend's ``generate_sent``, by pid."""
    counts = {}
    for path in counter_dir.glob("*.json"):
        data = json.loads(path.read_text())
        counts[data["pid"]] = data[SERVE_COUNTER_GENERATE_SENT]
    return counts


def _gone(process: psutil.Process) -> bool:
    """Exited; a zombie counts, since no one may reap an orphan in a container."""
    try:
        return process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def _still_running(processes: list[psutil.Process], timeout: float) -> list[psutil.Process]:
    """The ``processes`` that have not exited after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        alive = [process for process in processes if not _gone(process)]
        if not alive or time.monotonic() > deadline:
            return alive
        time.sleep(0.1)


def _wait(predicate, what: str, timeout: float = 30.0, process: subprocess.Popen | None = None):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if process is not None and process.poll() is not None:
            raise AssertionError(f"the launcher exited with {process.returncode} before {what}")
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


class _Serve:
    """``trtllm-serve --openengine-endpoint`` with several frontends, as a process."""

    def __init__(
        self,
        tmp_path: Path,
        model_dir: Path,
        engine: FakeEngine,
        frontends: int,
        env: dict[str, str] | None = None,
    ):
        self.port = _free_port()
        self.frontends = frontends
        self.counter_dir = tmp_path / "counters"
        self.log_path = tmp_path / "serve.log"
        env = {**os.environ, **(env or {})}
        env[SERVE_COUNTER_DIR_ENV] = str(self.counter_dir)
        env.pop(multi_frontend.LAUNCHER_UDS_ENV, None)
        # The attached frontends re-exec the command line, so start it as users do,
        # with the entry point of the interpreter running the tests.
        entry = Path(sys.executable).parent / "trtllm-serve"
        prefix = (
            [str(entry)]
            if entry.is_file()
            else [sys.executable, "-m", "tensorrt_llm.commands.serve"]
        )
        cmd = [
            *prefix,
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--served_model_name",
            MODEL,
            "--openengine-endpoint",
            engine.target,
            "--openengine-ready-timeout",
            "60",
            "--num_serve_frontends",
            str(frontends),
        ]
        self._log = self.log_path.open("w")
        self.proc = subprocess.Popen(  # nosec B603
            cmd, env=env, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True
        )
        self.launcher = psutil.Process(self.proc.pid)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def attached(self) -> list[psutil.Process]:
        """The frontends the launcher spawned."""
        try:
            children = self.launcher.children()
        except psutil.NoSuchProcess:
            return []
        attached = []
        for child in children:
            with contextlib.suppress(psutil.NoSuchProcess):
                if "--openengine-endpoint" in child.cmdline():
                    attached.append(child)
        return attached

    def listeners(self) -> int:
        return sum(_listens(p, self.port) for p in [self.launcher, *self.attached()])

    def log(self) -> str:
        self._log.flush()
        return self.log_path.read_text(errors="replace")

    def stop(self) -> None:
        processes = [self.launcher, *self.attached()]
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        alive = _still_running(processes, 30)
        for process in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                process.kill()
        self._log.close()
        assert not alive, f"frontends still running after the launcher stopped: {alive}"


@contextlib.contextmanager
def _short_tmpdir():
    """A private TMPDIR, short enough for the launcher's Unix socket path."""
    path = Path(tempfile.mkdtemp(prefix="wt14-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _complete(url: str, index: int) -> requests.Response:
    # A new connection each time, so the kernel picks a frontend for each request.
    return requests.post(
        f"{url}/v1/completions",
        json={"model": MODEL, "prompt": "He", "max_tokens": 2, "user": str(index)},
        headers={"Connection": "close"},
        timeout=30,
    )


@pytest.fixture(scope="module")
def serving(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("wt14_serve")
    model_dir = _model_dir(tmp_path / "model")
    engine = _engine(model_dir)
    with running_in_thread(engine):
        serve = _Serve(tmp_path, model_dir, engine, NUM_FRONTENDS)
        try:
            # Read together with the first listening socket of any frontend.
            model_info_calls = _wait(
                lambda: serve.listeners() > 0 and [_model_info_calls(engine)],
                "a frontend to listen",
                STARTUP_SECONDS,
                serve.proc,
            )[0]
            yield serve, engine, model_info_calls
        finally:
            serve.stop()


def test_no_frontend_listens_before_every_frontend_connected(serving):
    serve, engine, model_info_calls = serving
    # Each frontend reads ModelInfo once, when it connects (SV-EC-7); all of
    # them had connected by the time the first frontend started listening.
    assert model_info_calls == NUM_FRONTENDS, serve.log()
    assert len(serve.attached()) == NUM_FRONTENDS - 1
    assert _wait(lambda: serve.listeners() == NUM_FRONTENDS, "every frontend to listen")
    pids = {serve.launcher.pid, *(p.pid for p in serve.attached())}
    # Each frontend writes its count from the start.
    assert _wait(lambda: set(_counts(serve.counter_dir)) == pids, "every frontend's count file")
    assert f"All {NUM_FRONTENDS} serving frontends are connected" in serve.log()


def test_requests_spread_over_the_frontends_and_the_counts_add_up(serving):
    serve, engine, _ = serving
    _wait(lambda: serve.listeners() == NUM_FRONTENDS, "every frontend to listen")
    received_before = engine.counters[COUNTER_GENERATE_RECEIVED]
    sent_before = _counts(serve.counter_dir)
    num_requests = 48
    for index in range(num_requests):
        response = _complete(serve.url, index)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["text"]
    received = engine.counters[COUNTER_GENERATE_RECEIVED] - received_before
    assert received == num_requests

    def sent_by_pid() -> dict[int, int] | None:
        counts = _counts(serve.counter_dir)
        sent = {pid: counts[pid] - sent_before.get(pid, 0) for pid in counts}
        return sent if sum(sent.values()) == received else None

    sent = _wait(sent_by_pid, "the frontends' counts to add up to what the engine received")
    pids = {serve.launcher.pid, *(p.pid for p in serve.attached())}
    assert set(sent) == pids
    assert sum(1 for count in sent.values() if count > 0) >= 2, sent
    # /server_info answers with the count of whichever frontend served it.
    info = requests.get(f"{serve.url}/server_info", timeout=30).json()
    assert info["engine"]["generate_sent"] in {_counts(serve.counter_dir)[pid] for pid in pids}


def test_the_responses_store_is_off_with_several_frontends(serving):
    serve, _, _ = serving
    assert "stateful Responses API storage" in serve.log()
    # The launcher turns it off before it spawns the attached frontends.
    attached = serve.attached()
    assert attached
    for process in attached:
        assert process.environ().get("TRTLLM_RESPONSES_API_DISABLE_STORE") == "1"


def test_a_killed_frontend_is_handled_as_in_process(tmp_path):
    """An attached frontend that dies is logged and the others serve; the launcher's death stops them."""
    model_dir = _model_dir(tmp_path / "model")
    engine = _engine(model_dir)
    # The launcher killed by SIGKILL leaves its temporary directory behind.
    with _short_tmpdir() as tmpdir, running_in_thread(engine):
        serve = _Serve(tmp_path, model_dir, engine, 3, env={"TMPDIR": str(tmpdir)})
        try:
            _wait(
                lambda: serve.listeners() == 3,
                "every frontend to listen",
                STARTUP_SECONDS,
                serve.proc,
            )
            victim, survivor = serve.attached()
            victim.send_signal(signal.SIGKILL)
            _wait(lambda: "exited with signal 9" in serve.log(), "the launcher to log the exit")
            assert "2 of 3 frontends keep serving" in serve.log()
            assert serve.proc.poll() is None
            for index in range(8):
                assert _complete(serve.url, index).status_code == 200
            # Without the launcher, an attached frontend stops itself.
            serve.launcher.send_signal(signal.SIGKILL)
            serve.proc.wait(timeout=30)
            assert not _still_running([survivor], 60), serve.log()
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(serve.proc.pid, signal.SIGKILL)
            serve.stop()


def test_sigterm_stops_every_frontend_and_removes_the_launchers_directory(tmp_path):
    model_dir = _model_dir(tmp_path / "model")
    engine = _engine(model_dir)
    with _short_tmpdir() as tmpdir, running_in_thread(engine):
        serve = _Serve(tmp_path, model_dir, engine, 2, env={"TMPDIR": str(tmpdir)})
        try:
            _wait(
                lambda: serve.listeners() == 2,
                "every frontend to listen",
                STARTUP_SECONDS,
                serve.proc,
            )
            assert len(list(tmpdir.glob("tllm-oe-frontends-*"))) == 1
            attached = serve.attached()
            serve.proc.send_signal(signal.SIGTERM)
            assert serve.proc.wait(timeout=60) == -signal.SIGTERM, serve.log()
            assert not _still_running(attached, 30), serve.log()
            assert not list(tmpdir.glob("tllm-oe-frontends-*")), serve.log()
        finally:
            serve.stop()
