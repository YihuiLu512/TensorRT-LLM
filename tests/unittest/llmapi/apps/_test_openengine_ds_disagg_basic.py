# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: the module's forms, and basic serving.

The DeepSeek-V4.1 module (``_test_openengine_ds_disagg_*.py``) serves one model in two forms,
each on layout N-0 of one node with 4 GB300 GPUs (context DEP2 on GPUs 0-1, generation TP2 on
GPUs 2-3):

* ``K``, the ds-engine's own disaggregated deployment: a ds proxy in front of a ds context
  worker and a ds generation worker. :class:`DsNative` starts the three processes with the
  commands of the ds-engine's ``dev/test_disagg.sh`` on ``127.0.0.1`` and free ports, not
  through the chain harness; no chain proof is taken on it.
* the chain, ``TRTLLM_TEST_OPENENGINE_CHAIN=ds``: a ``trtllm-serve disaggregated``
  orchestrator in front of a context and a generation ``trtllm-serve`` worker (started with
  ``--server_role``, registered through the orchestrator's HTTP cluster), each a worker front
  end with the DeepSeek-V4.1 tokenizer and parsers in front of a ds-engine OpenEngine server.
  :func:`ds_disagg_chain` starts it through the chain harness.

A pytest session serves one form: the chain when ``TRTLLM_TEST_OPENENGINE_CHAIN=ds``, form K
when the variable is unset (:func:`session_form`). A test shared by the two forms (``K+链`` in
the plan) runs once in each session. Both forms take the ds-engine command from
``TRTLLM_TEST_DS_SERVE_CMD`` (for example ``python3 -m dsv4_serve``, with the ds-engine's
``python`` directory on ``PYTHONPATH``), the engine options :data:`ENGINE_ARGS` and the
per-role configurations :data:`CONTEXT_CONFIG` and :data:`GENERATION_CONFIG` (DSpark speculative
decoding, block reuse on the context worker). Only the chain's worker front ends take the
tokenizer and parser options. A test changes the engine options and configurations through a
:class:`Group` of its own. An engine group holds all four GPUs, so :class:`Groups` keeps one
group up at a time: a file's tests share the default group (:func:`served`), and a fixture that
needs another group asks :class:`Groups` for it, which stops the group that is up first. Groups
change in fixtures only: the chain proof reads the orchestrator from the start of a test's call.

Needs ``${LLM_MODELS_ROOT}/DeepSeek-V4.1-Flash-NVFP4``.
"""

from __future__ import annotations

import collections
import contextlib
import inspect
import json
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import requests
import yaml

# The chain harness, imported where a chain starts.
__extra_import_path__ = ["~/tests/integration/defs"]

pytestmark = [pytest.mark.threadleak(enabled=False)]

MODEL_DIR = "DeepSeek-V4.1-Flash-NVFP4"
MODEL_NAME = "DeepSeek-V4.1-Flash"
ENV_CHAIN = "TRTLLM_TEST_OPENENGINE_CHAIN"
ENV_DS_SERVE_CMD = "TRTLLM_TEST_DS_SERVE_CMD"
DUMMY_API_KEY = "tensorrt_llm"

# Engine options of both forms (dev/test_disagg.sh COMMON, without the tokenizer).
ENGINE_ARGS = (
    "--max_batch_size",
    "32",
    "--max_num_tokens",
    "8192",
    "--max_seq_len",
    "32768",
    "--kv_cache_free_gpu_memory_fraction",
    "0.5",
)
TOKENIZER_ARGS = ("--custom_tokenizer", "deepseek_v41")
PARSER_ARGS = ("--reasoning_parser", "deepseek_v41", "--tool_parser", "deepseek_v41")
# dev/test_disagg.sh's per-role configurations (tests/configs/disagg_{ctx_dep2,gen_tp2}.yaml).
CONTEXT_CONFIG = {
    "enable_attention_dp": True,
    "attention_dp_config": {"kv_cache_routing_conversation_affinity": True},
    "speculative_config": {"decoding_type": "DSpark", "max_draft_len": 5},
    "kv_cache_config": {"enable_block_reuse": True},
    "cache_transceiver_config": {"backend": "NIXL"},
}
GENERATION_CONFIG = {
    "speculative_config": {"decoding_type": "DSpark", "max_draft_len": 5},
    "kv_cache_config": {"enable_block_reuse": False},
    "cache_transceiver_config": {"backend": "NIXL"},
}
NATIVE_READY_TIMEOUT_S = 1200
NATIVE_STOP_TIMEOUT_S = 60
# The GPUs of each worker on layout N-0.
ROLE_GPUS = {"context": "0,1", "generation": "2,3"}
ROLES = ("context", "generation")
FORM_K = "K"
FORM_CHAIN = "chain"


@dataclass(frozen=True)
class Group:
    """What an engine group changes from the module's default one; the same in both forms.

    ``context`` and ``generation`` are merged over :data:`CONTEXT_CONFIG` and
    :data:`GENERATION_CONFIG` key by key (a ``None`` value drops the key); ``env`` adds
    environment variables to each role's ds engine (form K: the ds worker); ``front`` adds keys to
    the disaggregated server's configuration (form K: the ds proxy's); ``engine_args`` replaces
    :data:`ENGINE_ARGS`.
    """

    context: Mapping[str, object] = field(default_factory=dict)
    generation: Mapping[str, object] = field(default_factory=dict)
    env: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    front: Mapping[str, object] = field(default_factory=dict)
    engine_args: Sequence[str] = ENGINE_ARGS

    def config(self, role: str) -> dict:
        """The role's worker configuration."""
        base = CONTEXT_CONFIG if role == "context" else GENERATION_CONFIG
        merged = {**base, **(self.context if role == "context" else self.generation)}
        return {key: value for key, value in merged.items() if value is not None}


DEFAULT_GROUP = Group()


def ds_command(env: Mapping[str, str] | None = None) -> list[str]:
    """The ds-engine command prefix from ``TRTLLM_TEST_DS_SERVE_CMD``; empty when unset."""
    return shlex.split((os.environ if env is None else env).get(ENV_DS_SERVE_CMD, ""))


def free_ports(count: int) -> list[int]:
    """``count`` distinct free ports, each held until all are chosen."""
    with contextlib.ExitStack() as stack:
        sockets = [stack.enter_context(socket.socket()) for _ in range(count)]
        for s in sockets:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in sockets]


@dataclass(frozen=True)
class NativeProcess:
    """One process of form K: its name, command line and the environment it adds."""

    name: str
    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class NativePorts:
    proxy: int
    context: int
    generation: int


def proxy_config(ports: NativePorts) -> dict:
    """dev/test_disagg.sh's ds proxy configuration (tests/configs/disagg_local.yaml) on ``ports``.

    The workers are addressed as ``127.0.0.1``, where they listen.
    """
    return {
        "hostname": "127.0.0.1",
        "port": ports.proxy,
        "context_servers": {
            "num_instances": 1,
            "urls": [f"127.0.0.1:{ports.context}"],
            "router": {"type": "conversation"},
        },
        "generation_servers": {"num_instances": 1, "urls": [f"127.0.0.1:{ports.generation}"]},
        "gen_strip_message_history": True,
        "gen_tokids_ctxbytes": True,
    }


def native_processes(
    ds: Sequence[str],
    model: str,
    ports: NativePorts,
    configs: Mapping[str, str],
    group: Group = DEFAULT_GROUP,
) -> list[NativeProcess]:
    """Form K's processes: dev/test_disagg.sh's commands, on 127.0.0.1 and ``ports``.

    ``configs`` maps ``context``, ``generation`` and ``proxy`` to their YAML files; ``group``
    gives the engine options and each worker's added environment.
    """
    common = list(group.engine_args) + list(TOKENIZER_ARGS)

    def worker(role: str, port: int, gpus: str) -> NativeProcess:
        argv = [*ds, model, "--host", "127.0.0.1", "--port", str(port), "--tp_size", "2"]
        argv += ["--server_role", role, *common, "--extra_llm_api_options", configs[role]]
        return NativeProcess(
            role, tuple(argv), {"CUDA_VISIBLE_DEVICES": gpus, **group.env.get(role, {})}
        )

    proxy = (*ds, "disaggregated", "-c", configs["proxy"], "-t", str(NATIVE_READY_TIMEOUT_S))
    return [
        worker("context", ports.context, "0,1"),
        worker("generation", ports.generation, "2,3"),
        NativeProcess("proxy", proxy),
    ]


def _tail(path: Path, lines: int = 30) -> str:
    with contextlib.suppress(OSError), open(path, errors="replace") as f:
        return "".join(collections.deque(f, maxlen=lines)).rstrip("\n")
    return ""


class DsNative:
    """Form K on this node: the ds proxy, context worker and generation worker.

    A context manager: entering starts the three processes and waits until the proxy's
    ``/health`` answers 200; leaving stops them. Logs and configuration files stay in
    :attr:`workdir`.
    """

    def __init__(
        self,
        ds: Sequence[str],
        model: str,
        *,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        ready_timeout: float = NATIVE_READY_TIMEOUT_S,
        group: Group = DEFAULT_GROUP,
    ) -> None:
        if not ds:
            raise ValueError(f"form K needs the ds-engine command ({ENV_DS_SERVE_CMD})")
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="ds-native-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ports = NativePorts(*free_ports(3))
        configs = {}
        for name, content in (
            ("context", group.config("context")),
            ("generation", group.config("generation")),
            ("proxy", {**proxy_config(self.ports), **group.front}),
        ):
            path = self.workdir / f"{name}.yaml"
            path.write_text(yaml.safe_dump(content))
            configs[name] = str(path)
        self.processes = native_processes(ds, model, self.ports, configs, group)
        self._env = dict(os.environ if env is None else env)
        self._ready_timeout = ready_timeout
        self._running: dict[str, subprocess.Popen] = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.ports.proxy}"

    def worker_url(self, role: str) -> str:
        """The HTTP root of the ``role`` ds worker."""
        return f"http://127.0.0.1:{getattr(self.ports, role)}"

    def log(self, name: str) -> Path:
        """The log file of process ``name`` (``context``, ``generation`` or ``proxy``)."""
        return self.workdir / f"{name}.log"

    def __enter__(self) -> DsNative:
        try:
            for process in self.processes:
                log = open(self.workdir / f"{process.name}.log", "w")
                with log:
                    self._running[process.name] = subprocess.Popen(
                        list(process.argv),
                        env={**self._env, **process.env},
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
            self._wait_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self._ready_timeout
        while True:
            with contextlib.suppress(requests.RequestException):
                if requests.get(f"{self.url}/health", timeout=5).status_code == 200:
                    return
            for name, proc in self._running.items():
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"form K: the {name} process exited with {proc.returncode}; log "
                        f"{self.workdir / (name + '.log')}:\n{_tail(self.workdir / (name + '.log'))}"
                    )
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"form K: the proxy is not healthy after {self._ready_timeout:.0f} s; "
                    f"logs in {self.workdir}"
                )
            time.sleep(2)

    def stop(self) -> None:
        """Stop the processes' process groups: SIGTERM, then SIGKILL after a grace period.

        Every group gets the SIGKILL, so that members still alive after their leader exited
        (rank processes of a worker) are stopped too.
        """
        for proc in self._running.values():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
        deadline = time.monotonic() + NATIVE_STOP_TIMEOUT_S
        for proc in self._running.values():
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        survivors = []
        for name, proc in self._running.items():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            try:
                proc.wait(timeout=NATIVE_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                survivors.append(f"{name} (pid {proc.pid})")
        self._running.clear()
        if survivors:
            warnings.warn(f"form K: still running after SIGKILL: {', '.join(survivors)}")


@dataclass
class Served:
    """A served form as the tests use it.

    ``url`` is the HTTP root a client sends to: the disaggregated server (form K: the ds proxy).
    ``workers`` maps each role to its worker's HTTP root (the chain: the worker front end; form
    K: the ds worker), ``engines`` maps each role to its ds engine's gRPC address (the chain
    only), and ``servers`` holds the chain's ``RemoteOpenAIServer`` objects by role and
    ``"orchestrator"``. ``engine_endpoint`` is the first engine's address.
    """

    url: str
    engine_endpoint: str | None = None
    form: str = FORM_CHAIN
    workers: Mapping[str, str] = field(default_factory=dict)
    engines: Mapping[str, str] = field(default_factory=dict)
    servers: Mapping[str, object] = field(default_factory=dict)
    native: DsNative | None = None

    def client(self, url: str | None = None):
        import openai

        return openai.OpenAI(api_key=DUMMY_API_KEY, base_url=f"{url or self.url}/v1", max_retries=0)

    @property
    def chain(self) -> bool:
        return self.form == FORM_CHAIN


def session_form(env: Mapping[str, str] | None = None) -> str:
    """The form this pytest session serves: the chain with ``CHAIN=ds``, form K without CHAIN."""
    chain = (os.environ if env is None else env).get(ENV_CHAIN) or "none"
    if chain == "ds":
        return FORM_CHAIN
    if chain == "none":
        return FORM_K
    raise ValueError(
        f"{ENV_CHAIN}={chain!r}: the DeepSeek-V4.1 module serves the ds chain or form K"
    )


@pytest.fixture(scope="module")
def model_path() -> str:
    from ..test_llm import get_model_path

    try:
        path = get_model_path(MODEL_DIR)
    except TypeError:  # llm_models_root() found no models root
        pytest.skip("no models root: set LLM_MODELS_ROOT")
    assert os.path.isdir(path), f"{path} is not a directory"
    return path


def chain_serve_args() -> list[str]:
    """The ``trtllm-serve`` options of one aggregated worker over all four GPUs, besides the model."""
    return [
        "--tp_size",
        "4",
        *ENGINE_ARGS,
        *TOKENIZER_ARGS,
        *PARSER_ARGS,
        "--served_model_name",
        MODEL_NAME,
    ]


def worker_serve_args(role: str, group: Group = DEFAULT_GROUP) -> list[str]:
    """The ``trtllm-serve`` options of the chain's ``role`` worker, besides model, port and config."""
    return [
        "--server_role",
        role,
        "--tp_size",
        "2",
        *group.engine_args,
        *TOKENIZER_ARGS,
        *PARSER_ARGS,
        "--served_model_name",
        MODEL_NAME,
    ]


REGISTER_TIMEOUT_S = 300


def wait_registered(
    url: str, counts: Mapping[str, int], timeout: float = REGISTER_TIMEOUT_S
) -> None:
    """Wait until the disaggregated server at ``url`` lists ``counts`` workers per role."""
    deadline = time.monotonic() + timeout
    while True:
        with contextlib.suppress(requests.RequestException, ValueError, KeyError):
            lists = requests.get(f"{url}/cluster_info", timeout=10).json()["server_lists"]
            if all(len(lists[role]) >= n for role, n in counts.items()):
                return
        assert time.monotonic() < deadline, f"the workers did not register at {url}: {counts}"
        time.sleep(1)


@contextlib.contextmanager
def ds_disagg_chain(model: str, workdir: str | os.PathLike, group: Group = DEFAULT_GROUP):
    """The chain on layout N-0: the orchestrator and the two workers, registered, as :class:`Served`.

    Each worker is a ``trtllm-serve`` command the chain harness splits into a worker front end
    and a ds engine on the role's GPUs; the workers register through the orchestrator's HTTP
    cluster. Logs go to ``workdir``.
    """
    from openengine_chain import launch

    from tensorrt_llm._utils import get_free_ports

    from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    disagg_port, ctx_port, gen_port = get_free_ports(3)
    # The harness picks the engines' ports as the workers start: not these three.
    launch.reserve_ports(disagg_port, ctx_port, gen_port)
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}

    def worker(role: str, port: int) -> RemoteOpenAIServer:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ROLE_GPUS[role], **group.env.get(role, {})}
        return RemoteOpenAIServer(
            model,
            port=port,
            env=env,
            cli_args=worker_serve_args(role, group),
            extra_config={**group.config(role), "disagg_cluster": cluster},
            log_path=str(workdir / f"output_{role}.log"),
            wait=False,
        )

    front = {"hostname": "localhost", "port": disagg_port, "disagg_cluster": cluster, **group.front}
    with worker("context", ctx_port) as ctx, worker("generation", gen_port) as gen:
        with RemoteDisaggOpenAIServer(
            ctx_servers=[],
            gen_servers=[],
            port=disagg_port,
            disagg_config=front,
            log_path=str(workdir / "output_orchestrator.log"),
        ) as orchestrator:
            for server in (ctx, gen):
                server.wait_for_server(timeout=NATIVE_READY_TIMEOUT_S)
            url = f"http://localhost:{disagg_port}"
            wait_registered(url, {"context": 1, "generation": 1})
            engines = {
                role: server._chain_handle.engine_endpoint
                for role, server in (("context", ctx), ("generation", gen))
                if server._chain_handle is not None
            }
            yield Served(
                url,
                engines.get("context"),
                FORM_CHAIN,
                workers={"context": ctx.url_root, "generation": gen.url_root},
                engines=engines,
                servers={"context": ctx, "generation": gen, "orchestrator": orchestrator},
            )


@contextlib.contextmanager
def start_form(form: str, model: str, workdir: str | os.PathLike, group: Group = DEFAULT_GROUP):
    """Form ``form`` with engine group ``group``, as :class:`Served`, until the block ends."""
    if form == FORM_CHAIN:
        with ds_disagg_chain(model, workdir, group) as served:
            yield served
        return
    ds = ds_command()
    if not ds:
        pytest.skip(f"form K needs {ENV_DS_SERVE_CMD}")
    with DsNative(ds, model, workdir=str(Path(workdir) / "k"), group=group) as native:
        yield Served(
            native.url,
            form=FORM_K,
            workers={role: native.worker_url(role) for role in ROLES},
            native=native,
        )


class Groups:
    """The engine groups of one test file: at most one is up at a time (each holds all GPUs).

    A group that failed to start is not started again: later tests on it get the same error.
    """

    def __init__(self, form: str, model: str, workdir: Path) -> None:
        self.form, self.model, self.workdir = form, model, workdir
        self._stack: contextlib.ExitStack | None = None
        self._group: Group | None = None
        self._served: Served | None = None
        self._failed: list[tuple[Group, BaseException]] = []
        self._count = 0

    def get(self, group: Group = DEFAULT_GROUP) -> Served:
        """``group`` up, stopping the one that is up first if it is another group."""
        if self._served is not None and self._group == group:
            return self._served
        for failed, error in self._failed:
            if failed == group:
                raise RuntimeError(f"engine group {group} did not start: {error!r}") from error
        self.close()
        self._count += 1
        stack = contextlib.ExitStack()
        try:
            served = stack.enter_context(
                start_form(self.form, self.model, self.workdir / f"group{self._count}", group)
            )
        except pytest.skip.Exception:
            stack.close()
            raise
        except Exception as error:
            stack.close()
            self._failed.append((group, error))
            raise
        self._stack, self._group, self._served = stack, group, served
        return served

    def close(self) -> None:
        """Stop the group that is up, if any; the next :meth:`get` starts a new one."""
        stack, self._stack, self._group, self._served = self._stack, None, None, None
        if stack is not None:
            stack.close()


@pytest.fixture(scope="module")
def form() -> str:
    """The form this session serves (:func:`session_form`)."""
    return session_form()


@pytest.fixture(scope="module")
def groups(form, model_path, tmp_path_factory):
    """The test file's engine groups; tests ask it for the group they run on."""
    g = Groups(form, model_path, tmp_path_factory.mktemp(f"ds-{form}"))
    yield g
    g.close()


@pytest.fixture
def served(groups) -> Served:
    """The module's default engine group of this session's form."""
    return groups.get()


def engine_server_info(endpoint: str):
    """The engine's ``ServerInfo`` message."""
    import grpc

    from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc, server_pb2

    with grpc.insecure_channel(endpoint) as channel:
        control = openengine_pb2_grpc.ControlStub(channel)
        return control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=60)


def engine_extras(endpoint: str) -> tuple[dict, dict]:
    """The engine's ``ServerInfo.extra`` and ``ModelInfo.extra``."""
    import grpc

    from tensorrt_llm.grpc.openengine.bindings import model_pb2, openengine_pb2_grpc
    from tensorrt_llm.grpc.openengine_conventions import struct_to_dict

    server = engine_server_info(endpoint)
    with grpc.insecure_channel(endpoint) as channel:
        control = openengine_pb2_grpc.ControlStub(channel)
        model = control.GetModelInfo(model_pb2.GetModelInfoRequest(), timeout=60)
    return struct_to_dict(server.extra), struct_to_dict(model.extra)


def engine_counter(endpoint: str, key: str) -> int:
    """Counter ``key`` the ds engine at ``endpoint`` reports in ``Control.GetLoad``.

    A counter the engine does not report fails the test instead of reading as 0.
    """
    from openengine_chain import direct_replay

    attributes = direct_replay.get_load(endpoint).get("attributes", {})
    assert key in attributes, f"the engine at {endpoint} reports no {key}: {sorted(attributes)}"
    return int(float(attributes[key]))


def only_chain(served: Served, why: str) -> None:
    """Skip a test that runs on the chain only; ``why`` says what form K lacks."""
    if not served.chain:
        pytest.skip(f"form K: {why}")


COMPLETION_PROMPT = "The capital of France is"
CHAT_MESSAGES = [
    {"role": "user", "content": "What is the capital of France? Answer with one word."}
]
NO_THINKING = {"chat_template_kwargs": {"thinking": False}}


def generate(served: Served, api: str, stream: bool, url: str | None = None, **request):
    """One completions or chat request: ``(text, finish_reason, usage)``; streamed or not."""
    c = served.client(url)
    request.setdefault("model", MODEL_NAME)
    request.setdefault("temperature", 0.0)
    if api == "completions":
        request.setdefault("prompt", COMPLETION_PROMPT)
        call = c.completions.create
    else:
        request.setdefault("messages", CHAT_MESSAGES)
        request.setdefault("extra_body", NO_THINKING)
        call = c.chat.completions.create
    # The OpenAI client takes only the API's own parameters; the server's extensions go in the body.
    known = inspect.signature(call).parameters
    extra = {key: request.pop(key) for key in list(request) if key not in known}
    if extra:
        request["extra_body"] = {**request.get("extra_body", {}), **extra}
    if not stream:
        out = call(**request)
        choice = out.choices[0]
        text = choice.text if api == "completions" else (choice.message.content or "")
        return text, choice.finish_reason, out.usage
    text, finish, usage = "", None, None
    for chunk in call(stream=True, stream_options={"include_usage": True}, **request):
        usage = chunk.usage or usage
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        text += (choice.text if api == "completions" else choice.delta.content) or ""
        finish = choice.finish_reason or finish
    return text, finish, usage


def test_startup_tokenizer_and_fingerprint(served, model_path):
    """NT-100 (F-07): both workers started with the DeepSeek-V4.1 tokenizer options serve the model.

    Each ds engine reports the front end's conventions version, and the DeepSeek-V4.1
    tokenizer's fingerprint matches each engine's over the same files.
    """
    only_chain(served, "the ds proxy has no worker front end")
    from tensorrt_llm.grpc.openengine_conventions import (
        CONVENTIONS_VERSION,
        TokenizerFingerprint,
        compare_tokenizer_fingerprints,
        compute_tokenizer_fingerprint,
        conventions_version,
    )
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    local = compute_tokenizer_fingerprint(
        DeepseekV41Tokenizer.from_pretrained(model_path), model_path
    )
    assert set(served.engines) == set(ROLES), served.engines
    for role, endpoint in served.engines.items():
        server_extra, model_extra = engine_extras(endpoint)
        assert conventions_version(server_extra) == CONVENTIONS_VERSION, (role, server_extra)
        reported = {k: v for k, v in model_extra.items() if k.startswith("trtllm.tokenizer.")}
        engine = TokenizerFingerprint.from_extra(reported)
        assert engine is not None, f"the {role} engine reports no usable fingerprint: {reported}"
        mismatches = compare_tokenizer_fingerprints(local, engine)
        assert not mismatches, json.dumps(
            {"role": role, "mismatches": [str(m) for m in mismatches], "engine": reported}
        )
        # Both sides hashed the same files, so the comparison covered every one of them.
        assert local.files and set(local.files) == set(engine.files), (local.files, engine.files)
    models = served.client().models.list()
    assert [m.id for m in models.data] == [MODEL_NAME]


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.parametrize("api", ["completions", "chat"])
def test_chat_and_completions_basic(served, api, stream):
    """NT-101 (F-01, F-02, DG-12, DG-13, DG-83, DG-86, DS-S01, DS-S02): both forms answer.

    The first token, decoded by the context leg and delivered by the generation leg, is in the
    output once: a request that stops at ``max_tokens`` counts exactly ``max_tokens`` tokens.
    On the chain the context engine sends a PrefillReady per request.
    """
    from tensorrt_llm.grpc.openengine_conventions import COUNTER_PREFILL_READY_SENT

    before = (
        engine_counter(served.engines["context"], COUNTER_PREFILL_READY_SENT) if served.chain else 0
    )
    text, finish, usage = generate(served, api, stream, max_tokens=32)
    assert "Paris" in text, text
    assert finish in ("stop", "length"), finish
    # Stream and non-stream give the same tokens (greedy).
    other, _, _ = generate(served, api, not stream, max_tokens=32)
    assert other == text, (other, text)
    _, finish, usage = generate(served, api, stream, max_tokens=5, ignore_eos=True)
    assert finish == "length" and usage.completion_tokens == 5, (finish, usage)
    if served.chain:
        sent = engine_counter(served.engines["context"], COUNTER_PREFILL_READY_SENT) - before
        assert sent >= 3, sent


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_completions_echo(served, stream):
    """NT-102 (F-02): with ``echo`` the completion starts with the prompt (UV-21)."""
    text, _, _ = generate(served, "completions", stream, max_tokens=8, echo=True)
    assert text.startswith(COMPLETION_PROMPT), text
    assert len(text) > len(COMPLETION_PROMPT), text


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_responses_and_messages(served, stream):
    """NT-103 (F-03, F-04): ``/v1/responses`` and ``/v1/messages`` answer through the chain."""
    only_chain(served, "the ds proxy serves neither /v1/responses nor /v1/messages")
    question = "What is the capital of France? Answer with one word."
    body = {"model": MODEL_NAME, "input": question, "max_output_tokens": 64, "stream": stream}
    body["temperature"] = 0.0
    body["reasoning"] = {"effort": "none"}
    response = requests.post(f"{served.url}/v1/responses", json=body, timeout=300)
    assert response.status_code == 200, response.text
    assert "Paris" in response.text, response.text[-2000:]
    body = {
        "model": MODEL_NAME,
        "max_tokens": 64,
        "stream": stream,
        "messages": [{"role": "user", "content": question}],
    }
    response = requests.post(f"{served.url}/v1/messages", json=body, timeout=300)
    assert response.status_code == 200, response.text
    assert "Paris" in response.text, response.text[-2000:]


def test_messages_count_tokens(served, model_path):
    """NT-104 (DG-05): the token count is the DeepSeek-V4.1 renderer's, without a Generate call."""
    only_chain(served, "the ds proxy does not serve /v1/messages/count_tokens")
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    messages = [{"role": "user", "content": "How many tokens is this sentence?"}]
    body = {"model": MODEL_NAME, "messages": messages}
    response = requests.post(f"{served.url}/v1/messages/count_tokens", json=body, timeout=60)
    assert response.status_code == 200, response.text
    tokenizer = DeepseekV41Tokenizer.from_pretrained(model_path)
    rendered = {
        thinking: len(tokenizer.apply_chat_template(messages, tokenize=True, thinking=thinking))
        for thinking in (False, True)
    }
    assert response.json()["input_tokens"] in rendered.values(), (response.json(), rendered)


def test_models_and_model_name(served):
    """NT-105 (F-05a, DG-80): ``/v1/models`` through the orchestrator; responses name the model."""
    only_chain(served, "the ds proxy's model list is not under test")
    models = served.client().models.list()
    assert [m.id for m in models.data] == [MODEL_NAME]
    out = served.client().completions.create(
        model=MODEL_NAME, prompt=COMPLETION_PROMPT, max_tokens=4, temperature=0.0
    )
    assert out.model == MODEL_NAME, out


@pytest.mark.parametrize("role", ROLES)
def test_health_generate_and_aggregated_direct(served, role):
    """NT-107 (F-05d, DS-S03): each worker serves an aggregated request sent to it directly."""
    url = served.workers[role]
    text, finish, _ = generate(served, "completions", False, url=url, max_tokens=16)
    assert "Paris" in text and finish in ("stop", "length"), (text, finish)
    if served.chain:
        assert requests.get(f"{url}/health_generate", timeout=300).status_code == 200


METRICS_GROUP = Group(
    context={"return_perf_metrics": True}, generation={"return_perf_metrics": True}
)


@pytest.fixture
def metrics_served(groups) -> Served:
    """The chain with per-request perf metrics on; skipped in form K (no worker front ends)."""
    if groups.form != FORM_CHAIN:
        pytest.skip("form K: the ds workers are not worker front ends")
    return groups.get(METRICS_GROUP)


@pytest.mark.parametrize("endpoint", ["version", "health", "server_info", "metrics", "prometheus"])
def test_worker_frontend_endpoints_direct(metrics_served, endpoint, record_property):
    """NT-106 (F-05b, F-05c, F-06, F-42a, F-42b, F-43, DG-89): each worker front end's endpoints.

    The workers return per-request perf metrics, which mounts ``/prometheus/metrics``. In
    ``/server_info`` each field the ds engine's ``ServerInfo`` reports (the batch size, the KV
    block size, the context engine's KV endpoint) must equal it; the fields it does not report
    are recorded in the ``server_info_from_frontend`` property. ``/metrics`` holds snapshots
    derived from ``GetLoad`` only (DV-18); ``/kv_cache_events`` is read once and recorded (the
    ds engine has no KV events).
    """
    served = metrics_served
    path = {
        "version": "/version",
        "health": "/health",
        "server_info": "/server_info",
        "metrics": "/metrics",
        "prometheus": "/prometheus/metrics",
    }[endpoint]
    for role, url in served.workers.items():
        response = requests.get(f"{url}{path}", timeout=60)
        assert response.status_code == 200, (role, response.status_code, response.text[:2000])
        if endpoint == "version":
            assert response.json().get("version"), response.text
        elif endpoint == "server_info":
            _check_server_info(served, role, response.json(), record_property)
        elif endpoint == "metrics":
            snapshots = response.json()
            assert isinstance(snapshots, list), response.text[:2000]
            record_property(
                f"metrics_keys_{role}", json.dumps(sorted(snapshots[-1]) if snapshots else [])
            )
            events = requests.get(f"{url}/kv_cache_events", timeout=60)
            record_property(f"kv_cache_events_{role}", f"{events.status_code} {events.text[:200]}")


def _check_server_info(served: Served, role: str, info: dict, record_property) -> None:
    """DG-89: ``/server_info`` against the ``role`` ds engine's ``ServerInfo``."""
    engine = engine_server_info(served.engines[role])
    assert int(info["tokens_per_block"]) > 0, info
    from_frontend = []
    if engine.capacity.max_running_requests:
        assert info["max_batch_size"] == engine.capacity.max_running_requests, (info, engine)
    else:
        from_frontend.append("max_batch_size")
    if engine.capacity.kv_block_size:
        assert info["tokens_per_block"] == engine.capacity.kv_block_size, (info, engine)
    else:
        from_frontend.append("tokens_per_block")
    if role == "context":
        (text,) = info["disaggregated_params"]["ctx_info_endpoint"]
        endpoints = engine.kv_connector.local_endpoints
        if endpoints:
            # The engine's first endpoint, in whichever notation the front end writes it.
            first = endpoints[0]
            assert first.host in text and text.endswith(f":{first.port}"), (text, first)
        else:
            from_frontend.append("ctx_info_endpoint")
    record_property(f"server_info_from_frontend_{role}", ",".join(from_frontend) or "none")
