# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: basic serving.

The DeepSeek-V4.1 module (``_test_openengine_ds_disagg_*.py``) serves one model in two forms:

* ``K``, the ds-engine's own disaggregated deployment: a ds proxy in front of a ds context
  worker and a ds generation worker. :class:`DsNative` starts the three processes with the
  commands of the ds-engine's ``dev/test_disagg.sh`` on ``127.0.0.1`` and free ports (layout
  N-0: context DEP2 on GPUs 0-1, generation TP2 on GPUs 2-3), not through the chain harness;
  no chain proof is taken on it.
* the chain, ``TRTLLM_TEST_OPENENGINE_CHAIN=ds``: ``trtllm-serve`` worker front ends with the
  DeepSeek-V4.1 tokenizer and parsers, each in front of a ds-engine OpenEngine server.
  ``ds_chain`` is the aggregated form: one front end on one ds engine with all four GPUs.

Both forms take the ds-engine command from ``TRTLLM_TEST_DS_SERVE_CMD`` (for example
``python3 -m dsv4_serve``, with the ds-engine's ``python`` directory on ``PYTHONPATH``) and
the engine options :data:`ENGINE_ARGS`. Only form K's workers also get the per-role
configurations :data:`CONTEXT_CONFIG` and :data:`GENERATION_CONFIG` (DSpark speculative
decoding, block reuse), so the aggregated chain form is not a like-for-like comparison with
form K. Each form holds all four GPUs and its fixture stays up until the end of the test file
that requests it, so a test file uses only one of the two forms; each such file starts its own.
Needs one node with 4 GB300 GPUs (the ds-engine moves KV over the NVLink fabric) and
``${LLM_MODELS_ROOT}/DeepSeek-V4.1-Flash-NVFP4``.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import shlex
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import requests
import yaml

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
    ds: Sequence[str], model: str, ports: NativePorts, configs: Mapping[str, str]
) -> list[NativeProcess]:
    """Form K's processes: dev/test_disagg.sh's commands, on 127.0.0.1 and ``ports``.

    ``configs`` maps ``context``, ``generation`` and ``proxy`` to their YAML files.
    """
    common = list(ENGINE_ARGS) + list(TOKENIZER_ARGS)

    def worker(role: str, port: int, gpus: str) -> NativeProcess:
        argv = [*ds, model, "--host", "127.0.0.1", "--port", str(port), "--tp_size", "2"]
        argv += ["--server_role", role, *common, "--extra_llm_api_options", configs[role]]
        return NativeProcess(role, tuple(argv), {"CUDA_VISIBLE_DEVICES": gpus})

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
    ) -> None:
        if not ds:
            raise ValueError(f"form K needs the ds-engine command ({ENV_DS_SERVE_CMD})")
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="ds-native-"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.ports = NativePorts(*free_ports(3))
        configs = {}
        for name, content in (
            ("context", CONTEXT_CONFIG),
            ("generation", GENERATION_CONFIG),
            ("proxy", proxy_config(self.ports)),
        ):
            path = self.workdir / f"{name}.yaml"
            path.write_text(yaml.safe_dump(content))
            configs[name] = str(path)
        self.processes = native_processes(ds, model, self.ports, configs)
        self._env = dict(os.environ if env is None else env)
        self._ready_timeout = ready_timeout
        self._running: dict[str, subprocess.Popen] = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.ports.proxy}"

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
        for proc in self._running.values():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        self._running.clear()


@dataclass(frozen=True)
class Served:
    """A served form as the tests use it: its HTTP root, and the ds engine behind it on the chain."""

    url: str
    engine_endpoint: str | None = None

    def client(self):
        import openai

        return openai.OpenAI(api_key=DUMMY_API_KEY, base_url=f"{self.url}/v1")


@pytest.fixture(scope="module")
def model_path() -> str:
    from ..test_llm import get_model_path

    return get_model_path(MODEL_DIR)


@pytest.fixture(scope="module")
def ds_native(model_path):
    """Form K, per test file."""
    ds = ds_command()
    if not ds:
        pytest.skip(f"form K needs {ENV_DS_SERVE_CMD}")
    with DsNative(ds, model_path) as native:
        yield Served(native.url)


def chain_serve_args() -> list[str]:
    """The ``trtllm-serve`` options of the aggregated chain form, besides the model."""
    return [
        "--tp_size",
        "4",
        *ENGINE_ARGS,
        *TOKENIZER_ARGS,
        *PARSER_ARGS,
        "--served_model_name",
        MODEL_NAME,
    ]


@pytest.fixture(scope="module")
def ds_chain(model_path):
    """The aggregated chain form, per test file: one worker front end on one ds engine."""
    if os.environ.get(ENV_CHAIN) != "ds":
        pytest.skip(f"the chain form runs with {ENV_CHAIN}=ds")
    from .openai_server import RemoteOpenAIServer

    with RemoteOpenAIServer(model_path, chain_serve_args()) as server:
        yield Served(server.url_root, server._chain_handle.engine_endpoint)


def engine_extras(endpoint: str) -> tuple[dict, dict]:
    """The engine's ``ServerInfo.extra`` and ``ModelInfo.extra``."""
    import grpc

    from tensorrt_llm.grpc.openengine.bindings import model_pb2, openengine_pb2_grpc, server_pb2
    from tensorrt_llm.grpc.openengine_conventions import struct_to_dict

    with grpc.insecure_channel(endpoint) as channel:
        control = openengine_pb2_grpc.ControlStub(channel)
        server = control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=60)
        model = control.GetModelInfo(model_pb2.GetModelInfoRequest(), timeout=60)
    return struct_to_dict(server.extra), struct_to_dict(model.extra)


def test_startup_tokenizer_and_fingerprint(ds_chain, model_path):
    """NT-100: the front end started with the DeepSeek-V4.1 tokenizer options serves the model.

    The engine reports the front end's conventions version, and the DeepSeek-V4.1
    tokenizer's fingerprint matches the engine's over the same files.
    """
    from tensorrt_llm.grpc.openengine_conventions import (
        CONVENTIONS_VERSION,
        TokenizerFingerprint,
        compare_tokenizer_fingerprints,
        compute_tokenizer_fingerprint,
        conventions_version,
    )
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    server_extra, model_extra = engine_extras(ds_chain.engine_endpoint)
    assert conventions_version(server_extra) == CONVENTIONS_VERSION, server_extra
    reported = {k: v for k, v in model_extra.items() if k.startswith("trtllm.tokenizer.")}
    engine = TokenizerFingerprint.from_extra(reported)
    assert engine is not None, f"the engine reports no usable fingerprint: {reported}"
    local = compute_tokenizer_fingerprint(
        DeepseekV41Tokenizer.from_pretrained(model_path), model_path
    )
    mismatches = compare_tokenizer_fingerprints(local, engine)
    assert not mismatches, json.dumps(
        {"mismatches": [str(m) for m in mismatches], "engine": reported, "serve": str(local)}
    )
    # Both sides hashed the same files, so the comparison covered every one of them.
    assert local.files and set(local.files) == set(engine.files), (local.files, engine.files)
    models = ds_chain.client().models.list()
    assert [m.id for m in models.data] == [MODEL_NAME]
