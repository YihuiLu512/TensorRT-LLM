# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker frontends on the gRPC fake engine, behind a real orchestrator, as processes.

Each worker frontend is the real ``trtllm-serve <model dir> --openengine-endpoint``
command line, talking to a :class:`FakeEngine` served from a thread of the test
process. The orchestrator is the real ``trtllm-serve disaggregated`` command
line. With HTTP service discovery the orchestrator holds the cluster storage
itself, so ``cluster_uri`` is the orchestrator's own address.

The module also holds two HTTP middlewares the worker frontends load with
``--middleware``: :func:`record_requests` appends ``<method> <path>`` of every
request to the file named by ``WT50_REQUEST_LOG``, and
:func:`fail_server_info` answers ``GET /server_info`` as an unhandled error does.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess  # nosec B404
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import requests
import yaml

from tensorrt_llm.serve.multi_frontend import LAUNCHER_UDS_ENV

_CLIENT_TESTS = Path(__file__).resolve().parents[3] / "grpc" / "openengine_client"
if str(_CLIENT_TESTS) not in sys.path:
    sys.path.insert(0, str(_CLIENT_TESTS))

from fake_openengine import PROFILES, FakeEngine, running_in_thread  # noqa: E402

MODEL = "stub-model"
VOCAB = {"<unk>": 0, "</s>": 1, "He": 2, "llo": 3, " world": 4, "!": 5, " again": 6}
EOS = 1
STARTUP_SECONDS = 300
HEARTBEAT_SECONDS = 1
INACTIVE_SECONDS = 4
REQUEST_LOG_ENV = "WT50_REQUEST_LOG"
HERE = Path(__file__).resolve().parent

__all__ = ["running_in_thread"]


# -- Middlewares the worker frontends load ------------------------------------------


async def record_requests(request: Any, call_next: Callable[[Any], Any]) -> Any:
    """Append ``<method> <path>`` to the file ``WT50_REQUEST_LOG`` names, then serve the request."""
    path = os.environ.get(REQUEST_LOG_ENV)
    if path:
        with open(path, "a", encoding="utf-8") as log:
            log.write(f"{request.method} {request.url.path}\n")
    return await call_next(request)


async def fail_server_info(request: Any, call_next: Callable[[Any], Any]) -> Any:
    """Answer ``GET /server_info`` as an unhandled error does; serve every other request.

    The answer is a plain-text 500: the orchestrator reads the body of
    ``/server_info`` as JSON without checking the status, so a JSON body would
    still be taken as the server's information.
    """
    from fastapi.responses import PlainTextResponse

    if request.method == "GET" and request.url.path == "/server_info":
        return PlainTextResponse("Internal Server Error", status_code=500)
    return await call_next(request)


# -- Model directory and engines ------------------------------------------------------


def model_dir(path: Path) -> Path:
    """A model directory holding only a word-level tokenizer over :data:`VOCAB`."""
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


def engine(model_path: Path, **profile_changes: Any) -> FakeEngine:
    """A ``trtllm-like`` fake engine serving the tokenizer in ``model_path``."""
    from tensorrt_llm.grpc.openengine_conventions import compute_tokenizer_fingerprint
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    tokenizer = ModelLoader.load_hf_tokenizer(str(model_path))
    fingerprint = compute_tokenizer_fingerprint(tokenizer, model_path)
    profile = PROFILES["trtllm-like"].replace(
        model=MODEL,
        max_context_length=64,
        vocab_size=len(VOCAB),
        eos_token_ids=(EOS,),
        bos_token_id=None,
        fingerprint=fingerprint.to_extra(),
    )
    if profile_changes:
        profile = profile.replace(**profile_changes)
    return FakeEngine(profile, record=True)


def generate_calls(fake: FakeEngine) -> int:
    """``Generate`` calls the fake engine has received."""
    return len(fake.requests)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# -- Processes --------------------------------------------------------------------------


def _serve_prefix() -> list[str]:
    # As users start it: with the entry point of the interpreter running the tests.
    entry = Path(sys.executable).parent / "trtllm-serve"
    return (
        [str(entry)] if entry.is_file() else [sys.executable, "-m", "tensorrt_llm.commands.serve"]
    )


class Process:
    """A ``trtllm-serve`` process with its output in ``log_path``."""

    def __init__(self, cmd: Sequence[str], log_path: Path, env: dict[str, str] | None = None):
        self.cmd = list(cmd)
        self.log_path = log_path
        full_env = {**os.environ, **(env or {})}
        # A launcher's socket in the test's environment would make this an attached frontend.
        full_env.pop(LAUNCHER_UDS_ENV, None)
        self._log = log_path.open("w")
        try:
            self.proc = subprocess.Popen(  # nosec B603
                self.cmd,
                env=full_env,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self._log.close()
            raise

    def alive(self) -> bool:
        return self.proc.poll() is None

    def log(self) -> str:
        self._log.flush()
        return self.log_path.read_text(errors="replace")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        else:
            # The process group may hold attached frontends of an exited launcher.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.proc.pid, signal.SIGKILL)
        self._log.close()


def cluster_config(cluster_uri: str) -> dict[str, Any]:
    """A ``disagg_cluster`` section; for HTTP storage, ``cluster_uri`` is the orchestrator's URL."""
    return {
        "cluster_uri": cluster_uri,
        "cluster_name": "wt50",
        "heartbeat_interval_sec": HEARTBEAT_SECONDS,
        "inactive_timeout_sec": INACTIVE_SECONDS,
    }


def start_orchestrator(
    tmp_path: Path,
    port: int,
    *,
    cluster: dict[str, Any] | None = None,
    urls: dict[str, list[str]] | None = None,
    auth_key: str | None = None,
) -> Process:
    """``trtllm-serve disaggregated`` on ``port``: service discovery with ``cluster``, else ``urls``."""
    config: dict[str, Any] = {"hostname": "127.0.0.1", "port": port}
    if cluster is not None:
        config["disagg_cluster"] = cluster
        config["context_servers"] = {"router": {"type": "round_robin"}}
        config["generation_servers"] = {"router": {"type": "round_robin"}}
    else:
        assert urls is not None
        config["context_servers"] = {"num_instances": len(urls["ctx"]), "urls": urls["ctx"]}
        config["generation_servers"] = {"num_instances": len(urls["gen"]), "urls": urls["gen"]}
    if auth_key is not None:
        config["internal_request_auth_key"] = auth_key
    path = tmp_path / "orchestrator.yaml"
    path.write_text(yaml.safe_dump(config))
    return Process(
        [*_serve_prefix(), "disaggregated", "-c", str(path)], tmp_path / "orchestrator.log"
    )


def start_worker(
    tmp_path: Path,
    name: str,
    model_path: Path,
    fake: FakeEngine,
    *,
    role: str | None,
    config: dict[str, Any] | None = None,
    port: int = 0,
    frontends: int = 1,
    args: Sequence[str] = (),
    middleware: Sequence[str] = ("wt50_chain.record_requests",),
    env: dict[str, str] | None = None,
) -> Process:
    """A worker frontend for ``fake``; its requests are logged to ``<name>.requests``."""
    cmd = [
        *_serve_prefix(),
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served_model_name",
        MODEL,
        "--openengine-endpoint",
        fake.target,
        "--openengine-ready-timeout",
        "60",
        "--num_serve_frontends",
        str(frontends),
    ]
    if role is not None:
        cmd += ["--server_role", role]
    if config is not None:
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config))
        cmd += ["--config", str(path)]
    for item in middleware:
        cmd += ["--middleware", item]
    cmd += list(args)
    pythonpath = os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))
    env = {
        "PYTHONPATH": pythonpath,
        REQUEST_LOG_ENV: str(tmp_path / f"{name}.requests"),
        **(env or {}),
    }
    return Process(cmd, tmp_path / f"{name}.log", env)


def requests_seen(tmp_path: Path, name: str) -> list[str]:
    """``<method> <path>`` of each request worker ``name`` received, in order."""
    path = tmp_path / f"{name}.requests"
    return path.read_text().splitlines() if path.exists() else []


def wait(
    predicate: Callable[[], Any],
    what: str,
    timeout: float = STARTUP_SECONDS,
    processes: Sequence[Process] = (),
) -> Any:
    """Poll ``predicate`` until it returns a true value; fail if a process exits first."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        for process in processes:
            if not process.alive():
                raise AssertionError(
                    f"{process.log_path.name} exited with {process.proc.returncode} before "
                    f"{what}:\n{process.log()[-4000:]}"
                )
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.2)


def get(url: str, timeout: float = 5.0) -> requests.Response | None:
    """``GET url``; None when nothing answers."""
    try:
        return requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None


def status(url: str) -> int | None:
    response = get(url)
    return None if response is None else response.status_code


def cluster_info(orchestrator: str) -> dict[str, Any]:
    response = get(f"{orchestrator}/cluster_info")
    return {} if response is None or response.status_code != 200 else response.json()


def registered(orchestrator: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The context and generation workers registered with the orchestrator's cluster manager."""
    current = cluster_info(orchestrator).get("current_workers", {})
    return list(current.get("context_servers", [])), list(current.get("generation_servers", []))


def listed(orchestrator: str) -> tuple[list[str], list[str]]:
    """The context and generation servers in the orchestrator's routers, which it sends requests to."""
    lists = cluster_info(orchestrator).get("server_lists", {})
    return list(lists.get("context", [])), list(lists.get("generation", []))


def storage_keys(orchestrator: str) -> list[str]:
    """Every key in the orchestrator's HTTP cluster storage."""
    response = get(f"{orchestrator}/get_prefix?key_prefix=&keys_only=true")
    if response is None or response.status_code != 200:
        return []
    return sorted(response.json()["result"])


@contextlib.contextmanager
def stopping(*processes: Process) -> Iterator[list[Process]]:
    """Stop ``processes`` on the way out, printing their logs when the block failed.

    Yields the list of processes to stop: append each process started inside
    the block, so that it is stopped on every path too.
    """
    running = list(processes)
    try:
        yield running
    except BaseException:
        for process in running:
            print(f"----- {process.log_path.name} -----\n{process.log()[-6000:]}")
        raise
    finally:
        for process in reversed(running):
            process.stop()
