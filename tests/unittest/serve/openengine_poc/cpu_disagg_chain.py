# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A disaggregated serving chain on CPU: orchestrator, two worker frontends, two fake engines.

:class:`CpuDisaggChain` starts, on the local host:

* a context and a generation :class:`fake_openengine.FakeEngine`, served from
  threads of the calling process, with :func:`fake_openengine.disagg_profile`
  of the chosen handoff shape and the fingerprint of the chain's tokenizer;
* for each engine, a real ``trtllm-serve --openengine-endpoint`` worker
  frontend process with one serving frontend, its own serving-side counter
  directory and request dump directory, and the HTTP recorder of
  :mod:`cpu_disagg_chain_recorder`;
* the orchestrator, ``trtllm-serve disaggregated``, with the two worker
  frontends as its static context and generation servers.

Every server binds port 0 and reports the port it got. Test files use it
through a fixture, for example::

    __extra_import_path__ = [".."]
    from cpu_disagg_chain import CpuDisaggChain


    @pytest.fixture(scope="module")
    def chain(tmp_path_factory):
        with CpuDisaggChain(tmp_path_factory.mktemp("chain"), shape="trtllm") as chain:
            yield chain

The engines can be scripted, given faults (``chain.ctx.engine.disagg_faults``)
and read directly; the frontends' counters, request dumps and recorded HTTP
requests are read from their directories.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import signal
import subprocess  # nosec B404
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
import yaml

from tensorrt_llm.grpc.openengine_conventions import (
    SERVE_COUNTER_DIR_ENV,
    SERVE_REQUEST_DUMP_DIR_ENV,
    compute_tokenizer_fingerprint,
)
from tensorrt_llm.serve.openengine_poc import contracts

__extra_import_path__ = [".", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain_recorder import HTTP_LOG_ENV, read_http_log  # noqa: E402
from fake_openengine import (  # noqa: E402
    HANDOFF_SHAPES,
    INSTANCE_ROLES,
    DisaggFaults,
    FakeEngine,
    FakeProfile,
    Hang,
    disagg_profile,
    running_in_thread,
)
from fakes import TINY_EOS, TINY_VOCAB, tiny_hf_tokenizer  # noqa: E402

# DisaggFaults and Hang are re-exported so that tests use the classes of the
# module the engines come from.
__all__ = [
    "DisaggFaults",
    "EOS",
    "Hang",
    "MODEL",
    "SHAPES",
    "STARTUP_SECONDS",
    "VOCAB",
    "ChainWorker",
    "CpuDisaggChain",
    "make_model_dir",
]

MODEL = "cpu-chain-model"
# The vocabulary of the chain's word-level tokenizer: each word is one token.
VOCAB = TINY_VOCAB
EOS = TINY_EOS
STARTUP_SECONDS = 300
SHAPES = HANDOFF_SHAPES
ROLES = INSTANCE_ROLES
# Launcher variables a plain HTTP process must not inherit.
_LAUNCHER_ENV_PREFIXES = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "UCX_", "I_MPI_", "HYDRA_", "MPI_")
_RECORDER = "cpu_disagg_chain_recorder.RecordHttpRequests"


def make_model_dir(path: Path) -> Path:
    """Write a tokenizer for :data:`VOCAB` to ``path``, as a model directory the frontends load."""
    tiny_hf_tokenizer().save_pretrained(path)
    return path


def _fingerprint(model_dir: Path) -> dict[str, Any]:
    """The tokenizer fingerprint the frontends compute for ``model_dir``."""
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    tokenizer = ModelLoader.load_hf_tokenizer(str(model_dir))
    return compute_tokenizer_fingerprint(tokenizer, model_dir).to_extra()


def _chain_profile(profile: FakeProfile, fingerprint: Mapping[str, Any]) -> FakeProfile:
    """``profile`` serving the chain's model, with the tokenizer ``fingerprint``."""
    return profile.replace(
        model=MODEL,
        max_context_length=64,
        vocab_size=len(VOCAB),
        eos_token_ids=(EOS,),
        bos_token_id=None,
        fingerprint=fingerprint,
    )


def _process_env(extra: Mapping[str, str]) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(_LAUNCHER_ENV_PREFIXES)
    }
    # Each process starts its own metrics directory.
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    env["PYTHONUNBUFFERED"] = "1"
    # The recorder loads by module name in the frontend process.
    here = str(Path(__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [here, env.get("PYTHONPATH")]))
    env.update(extra)
    return env


def _serve_command() -> list[str]:
    entry = Path(sys.executable).parent / "trtllm-serve"
    if entry.is_file():
        return [str(entry)]
    return [sys.executable, "-m", "tensorrt_llm.commands.serve"]


def _wait_for(
    predicate: Callable[[], Any], what: str, timeout: float, process: subprocess.Popen
) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if process.poll() is not None:
            raise RuntimeError(f"the process exited with {process.returncode} before {what}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out after {timeout} s waiting for {what}")
        time.sleep(0.1)


def _healthy(url: str) -> bool:
    try:
        return requests.get(f"{url}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def _reported_url(report_addr: Path) -> str | None:
    try:
        address = report_addr.read_text().strip()
    except FileNotFoundError:
        return None
    return f"http://{address}" if address else None


class _Process:
    """A server process in its own session, with its output in a log file."""

    def __init__(self, command: Sequence[str], env: Mapping[str, str], log_path: Path) -> None:
        self.command = list(command)
        self.log_path = log_path
        self._log = log_path.open("w")
        self._stopped = False
        try:
            self.popen = subprocess.Popen(  # nosec B603
                self.command,
                env=dict(env),
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self._log.close()
            raise

    def log(self) -> str:
        if not self._log.closed:
            self._log.flush()
        return self.log_path.read_text(errors="replace")

    def stop(self, timeout: float = 30.0) -> int | None:
        """Stop the process group with SIGTERM, then SIGKILL; return the exit code.

        Only the first call signals the group: once its leader is reaped, the
        group id may be reused.
        """
        if self._stopped:
            return self.popen.returncode
        self._stopped = True
        if self.popen.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.popen.pid, signal.SIGTERM)
            try:
                self.popen.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.popen.pid, signal.SIGKILL)
                self.popen.wait()
        # Children left in the group by the leader.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.popen.pid, signal.SIGKILL)
        self._log.close()
        return self.popen.returncode


@dataclasses.dataclass
class ChainWorker:
    """One worker of the chain: a fake engine and the worker frontend in front of it.

    Attributes:
        role: ``"context"`` or ``"generation"``.
        engine: The fake engine; read and change it from the test's thread.
        directory: Where the worker keeps its files.
        url: The frontend's HTTP address, once it has started.
    """

    role: str
    engine: FakeEngine
    directory: Path
    url: str = ""
    process: _Process | None = None

    @property
    def counter_dir(self) -> Path:
        """The frontend's serving-side counters (``TRTLLM_OPENENGINE_COUNTER_DIR``)."""
        return self.directory / "counters"

    @property
    def dump_dir(self) -> Path:
        """Where the frontend writes the ``GenerateRequest``s it sends, when it does."""
        return self.directory / "requests"

    @property
    def http_log(self) -> Path:
        return self.directory / "http.jsonl"

    @property
    def host_port(self) -> str:
        return self.url.removeprefix("http://")

    def http_requests(self) -> list[dict[str, Any]]:
        """The HTTP requests the frontend has answered, oldest first (see the recorder)."""
        return read_http_log(self.http_log)

    def serve_counters(self) -> dict[str, int]:
        """The frontend's serving-side counters (``SERVE_COUNTERS``)."""
        return contracts.read_serve_counters(self.counter_dir)

    def log(self) -> str:
        return self.process.log() if self.process is not None else ""


class CpuDisaggChain:
    """The chain described in the module docstring; a context manager that starts and stops it.

    Args:
        directory: An empty directory for the model, the configuration and
            every process's files and log.
        shape: The handoff shape of both engines (:data:`SHAPES`).
        profiles: ``{role: FakeProfile}`` replacing the engine profile of a
            role; the chain's model, vocabulary and fingerprint are applied
            over it. By default :func:`disagg_profile` of the shape, with the
            ds role.
        engine_options: Keyword arguments of :class:`FakeEngine`, for both engines.
        frontend_args: ``{role: [argument, ...]}`` added to that worker
            frontend's command line.
        orchestrator_config: Entries merged over the orchestrator's
            configuration (top level only).
        env: Variables set in every process the chain starts.
        startup_timeout: Seconds to wait for each server to answer ``/health``.
    """

    def __init__(
        self,
        directory: Path,
        shape: str = "trtllm",
        *,
        profiles: Mapping[str, FakeProfile] | None = None,
        engine_options: Mapping[str, Any] | None = None,
        frontend_args: Mapping[str, Sequence[str]] | None = None,
        orchestrator_config: Mapping[str, Any] | None = None,
        env: Mapping[str, str] | None = None,
        startup_timeout: float = STARTUP_SECONDS,
    ) -> None:
        if shape not in SHAPES:
            raise ValueError(f"shape must be one of {SHAPES}")
        self.directory = Path(directory)
        self.shape = shape
        self.model_dir = make_model_dir(self.directory / "model")
        self.frontend_args = {role: list((frontend_args or {}).get(role, ())) for role in ROLES}
        self.orchestrator_config = dict(orchestrator_config or {})
        self.env = dict(env or {})
        self.startup_timeout = startup_timeout
        base = "trtllm-like" if shape == "trtllm" else "ds-like"
        fingerprint = _fingerprint(self.model_dir)
        workers = {}
        for role in ROLES:
            profile = (profiles or {}).get(role) or disagg_profile(
                base, role if shape == "ds" else None
            )
            engine = FakeEngine(_chain_profile(profile, fingerprint), **(engine_options or {}))
            workers[role] = ChainWorker(role, engine, self.directory / role)
        self.ctx: ChainWorker = workers["context"]
        self.gen: ChainWorker = workers["generation"]
        self.url = ""
        self._orchestrator: _Process | None = None
        self._stack = contextlib.ExitStack()
        self._started = False

    @property
    def workers(self) -> tuple[ChainWorker, ChainWorker]:
        return self.ctx, self.gen

    def orchestrator_log(self) -> str:
        return self._orchestrator.log() if self._orchestrator is not None else ""

    # -- Lifecycle -------------------------------------------------------------

    def start(self) -> CpuDisaggChain:
        """Start the engines, the frontends, then the orchestrator; wait until each is healthy.

        A chain starts once: make a new one to start again.
        """
        if self._started:
            raise RuntimeError("a CpuDisaggChain starts only once")
        self._started = True
        try:
            for worker in self.workers:
                self._stack.enter_context(running_in_thread(worker.engine))
            for worker in self.workers:
                self._start_frontend(worker)
            for worker in self.workers:
                worker.url = self._wait_healthy(
                    worker.process, worker.directory / "addr", worker.role
                )
            self._start_orchestrator()
        except BaseException:
            self.stop()
            raise
        return self

    def _start_frontend(self, worker: ChainWorker) -> None:
        for path in (worker.directory, worker.counter_dir, worker.dump_dir):
            path.mkdir(parents=True, exist_ok=True)
        command = [
            *_serve_command(),
            str(self.model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--report_addr",
            str(worker.directory / "addr"),
            "--served_model_name",
            MODEL,
            "--openengine-endpoint",
            worker.engine.target,
            "--openengine-ready-timeout",
            "60",
            "--middleware",
            _RECORDER,
            *self.frontend_args[worker.role],
        ]
        env = _process_env(
            {
                SERVE_COUNTER_DIR_ENV: str(worker.counter_dir),
                SERVE_REQUEST_DUMP_DIR_ENV: str(worker.dump_dir),
                HTTP_LOG_ENV: str(worker.http_log),
                **self.env,
            }
        )
        worker.process = _Process(command, env, worker.directory / "frontend.log")

    def _start_orchestrator(self) -> None:
        directory = self.directory / "orchestrator"
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "hostname": "127.0.0.1",
            "port": 0,
            "context_servers": {
                "num_instances": 1,
                "urls": [self.ctx.host_port],
                "router": {"type": "round_robin"},
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": [self.gen.host_port],
                "router": {"type": "round_robin"},
            },
            **self.orchestrator_config,
        }
        config_path = directory / "disagg.yaml"
        config_path.write_text(yaml.safe_dump(config))
        command = [
            *_serve_command(),
            "disaggregated",
            "-c",
            str(config_path),
            "--report_addr",
            str(directory / "addr"),
        ]
        self._orchestrator = _Process(
            command, _process_env(self.env), directory / "orchestrator.log"
        )
        self.url = self._wait_healthy(self._orchestrator, directory / "addr", "orchestrator")

    def _wait_healthy(self, process: _Process, report_addr: Path, what: str) -> str:
        try:
            url = _wait_for(
                lambda: _reported_url(report_addr),
                f"the {what} to bind",
                self.startup_timeout,
                process.popen,
            )
            _wait_for(
                lambda: _healthy(url),
                f"the {what} to be healthy",
                self.startup_timeout,
                process.popen,
            )
        except (RuntimeError, TimeoutError) as error:
            raise RuntimeError(f"{error}; its log:\n{process.log()[-4000:]}") from None
        return url

    def stop(self) -> None:
        """Stop the orchestrator, then the frontends, then the engines; safe to call twice."""
        with contextlib.ExitStack() as stack:
            # Callbacks run last-in first-out, each even if an earlier one raised.
            stack.callback(self._stack.close)
            for process in (self.gen.process, self.ctx.process, self._orchestrator):
                if process is not None:
                    stack.callback(process.stop)

    def __enter__(self) -> CpuDisaggChain:
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    # -- Requests --------------------------------------------------------------

    def post(
        self, path: str, body: Mapping[str, Any], url: str | None = None, timeout: float = 60.0
    ) -> requests.Response:
        """POST ``body`` to ``path`` of the orchestrator (or of ``url``), on a new connection."""
        return requests.post(
            f"{url or self.url}{path}",
            json={"model": MODEL, **body},
            headers={"Connection": "close"},
            timeout=timeout,
        )

    def complete(
        self, prompt: str = "He", url: str | None = None, timeout: float = 60.0, **fields: Any
    ) -> requests.Response:
        """A ``/v1/completions`` request to the orchestrator (or to ``url``)."""
        body = {"prompt": prompt, "max_tokens": 2, **fields}
        return self.post("/v1/completions", body, url, timeout)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Each worker's engine counters and frontend counters, by role, for before/after deltas."""
        return {
            worker.role: {"engine": worker.engine.counters, "frontend": worker.serve_counters()}
            for worker in self.workers
        }
