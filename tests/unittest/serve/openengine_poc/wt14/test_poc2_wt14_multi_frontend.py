# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Several OpenEngine serving frontends (SV-MF), on CPU against the gRPC fake engine.

The process tests start the real ``trtllm-serve`` command line, which spawns
the attached frontends as it would in production.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
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
from tensorrt_llm.serve.openengine_poc import contracts, multi_frontend
from tensorrt_llm.serve.openengine_poc import launch as launch_module
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

    def __init__(self, tmp_path: Path, model_dir: Path, engine: FakeEngine, frontends: int):
        self.port = _free_port()
        self.frontends = frontends
        self.counter_dir = tmp_path / "counters"
        self.log_path = tmp_path / "serve.log"
        env = dict(os.environ)
        env[SERVE_COUNTER_DIR_ENV] = str(self.counter_dir)
        env.pop(multi_frontend.LAUNCHER_UDS_ENV, None)
        # The attached frontends re-exec the command line, so start it as users do.
        entry = shutil.which("trtllm-serve")
        prefix = [entry] if entry else [sys.executable, "-m", "tensorrt_llm.commands.serve"]
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
        _, alive = psutil.wait_procs(processes, timeout=30)
        for process in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                process.kill()
        self._log.close()
        assert not alive, f"frontends still running after the launcher stopped: {alive}"


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
            # Read together with the launcher's first listening socket.
            model_info_calls = _wait(
                lambda: _listens(serve.launcher, serve.port) and [_model_info_calls(engine)],
                "the launcher to listen",
                STARTUP_SECONDS,
                serve.proc,
            )[0]
            yield serve, engine, model_info_calls
        finally:
            serve.stop()


def test_the_launcher_listens_only_after_every_frontend_connected(serving):
    serve, engine, model_info_calls = serving
    # Each frontend reads ModelInfo once, when it connects (SV-EC-7); all of
    # them had connected by the time the launcher started listening.
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


def test_a_killed_frontend_is_handled_as_in_process(tmp_path):
    """An attached frontend that dies is logged and the others serve; the launcher's death stops them."""
    model_dir = _model_dir(tmp_path / "model")
    engine = _engine(model_dir)
    with running_in_thread(engine):
        serve = _Serve(tmp_path, model_dir, engine, 3)
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
            gone, alive = psutil.wait_procs([survivor], timeout=60)
            assert not alive, serve.log()
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(serve.proc.pid, signal.SIGKILL)
            serve.stop()
