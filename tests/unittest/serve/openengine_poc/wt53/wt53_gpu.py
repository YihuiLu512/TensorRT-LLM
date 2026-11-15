# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A disaggregated cluster on one GPU and its readings, for GPU tests through the orchestrator.

The tests run on the in-process workers and, when
``TRTLLM_TEST_OPENENGINE_CHAIN`` selects a chain
(``tests/integration/defs/openengine_chain``), on the OpenEngine chain, where
each worker is a frontend and an OpenEngine engine; the readings
of the engine (its ``GetServerInfo``, its Generate count, the requests its
frontend sent it) exist only there.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import secrets
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import requests
import torch
from test_common.http_utils import wait_for_endpoint_ready
from utils.llm_data import llm_models_root

from tensorrt_llm.grpc.openengine_conventions import SERVE_REQUEST_DUMP_DIR_ENV
from tensorrt_llm.serve.disagg_auto_scaling import get_worker_key_prefix

# For openengine_chain, imported where it is used.
__extra_import_path__ = ["~/tests/integration/defs"]

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
chain_only = pytest.mark.skipif(
    not CHAIN, reason="reads the OpenEngine engine; runs on the chain only"
)
MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
MODEL_NAME = "Qwen3/Qwen3-0.6B"
REQUEST_TIMEOUT_S = 120
ENGINE_READY_TIMEOUT_S = 1200
REGISTER_TIMEOUT_S = 300
ROLES = ("context", "generation")


def model_dir() -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), MODEL_SUBDIR)
    assert path is not None and os.path.isdir(path), f"{MODEL_SUBDIR} is not under LLM_MODELS_ROOT"
    return path


def needs_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")


def worker_config(**overrides: Any) -> dict[str, Any]:
    """A worker's configuration: KV transfer on, no CUDA graphs, small KV cache; ``overrides`` win."""
    config: dict[str, Any] = {
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "kv_cache_config": {"free_gpu_memory_fraction": 0.1, "enable_block_reuse": False},
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
    }
    config.update(overrides)
    return config


@dataclass
class Cluster:
    """An orchestrator and its context and generation workers (``RemoteOpenAIServer``)."""

    orchestrator: Any
    workers: dict[str, list[Any]]
    # The key and the environment are left out of this object's repr.
    auth_key: str = field(repr=False)
    dump_dir: Path
    logs: dict[str, str] = field(default_factory=dict)
    # The orchestrator's configuration and every process's environment, for another orchestrator.
    orchestrator_config: dict[str, Any] = field(default_factory=dict, repr=False)
    env: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def ctx(self) -> Any:
        return self.workers["context"][0]

    @property
    def gen(self) -> Any:
        return self.workers["generation"][0]

    def url(self, path: str) -> str:
        return f"{self.orchestrator.url_root}/{path.lstrip('/')}"

    def post(self, path: str, body: Mapping[str, Any], headers: Mapping[str, str] | None = None):
        return requests.post(
            self.url(path),
            json={"model": MODEL_NAME, **body},
            headers=dict(headers or {}),
            timeout=REQUEST_TIMEOUT_S,
        )


def _merged(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of ``base`` updated with ``extra``, merging the values that are mappings on both sides.

    Nothing in the result is shared with the arguments, so a consumer that
    changes one section of a configuration (the orchestrator pops a router's
    ``type``) does not change another.
    """
    merged = copy.deepcopy(dict(base))
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merged(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@contextlib.contextmanager
def cluster(
    workdir: Path,
    *,
    counts: Mapping[str, int] | None = None,
    config: Mapping[str, Any] | None = None,
    role_config: Mapping[str, Mapping[str, Any]] | None = None,
    args: Sequence[str] = (),
    orchestrator: Mapping[str, Any] | None = None,
    servers: Mapping[str, Mapping[str, Any]] | None = None,
    env: Mapping[str, str] | None = None,
    register: bool = False,
    ports: Mapping[str, int] | None = None,
) -> Iterator[Cluster]:
    """Workers of both roles on this host and an orchestrator in front of them.

    ``counts`` is the number of workers of each role (default one each),
    ``config`` every worker's configuration (:func:`worker_config` by
    default), ``role_config`` what one role's workers add to it and ``args``
    their command line. ``orchestrator`` is added to the orchestrator's
    configuration and ``servers`` to its ``context_servers`` /
    ``generation_servers`` sections, by role. With ``register`` the workers
    register through the orchestrator's cluster; otherwise the orchestrator is
    given their URLs once they serve. ``env`` is added to every process's
    environment. ``ports`` gives the first worker of a role its port. Each
    worker frontend writes the GenerateRequests it sends to
    ``dump_dir/<role>-<i>`` (on the chain).
    """
    # The calling test's package (tests/unittest/llmapi/apps), imported under this name by pytest.
    from llmapi.apps.openai_server import RemoteOpenAIServer

    needs_gpu()
    model = model_dir()
    workdir.mkdir(parents=True, exist_ok=True)
    counts = {"context": 1, "generation": 1, **(counts or {})}
    auth_key = secrets.token_hex(32)
    dump_dir = workdir / "requests"
    process_env = {**os.environ, "UCX_NET_DEVICES": "lo", **(env or {})}
    orchestrator_port = pick_port()
    disagg_cluster = {"cluster_uri": f"http://localhost:{orchestrator_port}", "cluster_name": ""}
    base = _merged(
        {"internal_request_auth_key": auth_key},
        worker_config() if config is None else config,
    )
    if register:
        base["disagg_cluster"] = dict(disagg_cluster)
    logs: dict[str, str] = {}
    workers: dict[str, list[Any]] = {role: [] for role in ROLES}
    with contextlib.ExitStack() as stack:
        for role in ROLES:
            for index in range(counts[role]):
                name = f"{role}-{index}"
                logs[name] = str(workdir / f"output_{name}.log")
                workers[role].append(
                    stack.enter_context(
                        RemoteOpenAIServer(
                            model,
                            port=(ports or {}).get(role) if index == 0 else None,
                            env={**process_env, SERVE_REQUEST_DUMP_DIR_ENV: str(dump_dir / name)},
                            cli_args=["--server_role", role, *args],
                            llmapi_launch=False,
                            extra_config=_merged(base, (role_config or {}).get(role, {})),
                            log_path=logs[name],
                            wait=False,
                        )
                    )
                )
        if not register:
            # The orchestrator is given the URLs, which the chain settles once a worker serves.
            for worker in (w for role in ROLES for w in workers[role]):
                worker.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
        sections = {}
        for role, key in zip(ROLES, ("context_servers", "generation_servers")):
            urls = [] if register else [f"{s.host}:{s.port}" for s in workers[role]]
            sections[key] = _merged(
                {"num_instances": len(urls), "urls": urls}, (servers or {}).get(role, {})
            )
        disagg_config = {
            "hostname": "localhost",
            "port": orchestrator_port,
            "internal_request_auth_key": auth_key,
            **sections,
            **(orchestrator or {}),
        }
        if register:
            disagg_config["disagg_cluster"] = disagg_cluster
        logs["orchestrator"] = str(workdir / "output_orchestrator.log")
        server = stack.enter_context(
            _orchestrator(orchestrator_port, process_env, disagg_config, logs["orchestrator"])
        )
        if register:
            for worker in (w for role in ROLES for w in workers[role]):
                worker.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
        _wait_ready(server)
        result = Cluster(server, workers, auth_key, dump_dir, logs, disagg_config, process_env)
        wait_for_members(result, counts)
        yield result


def members(orchestrator: Any) -> dict[str, list[str]]:
    """The workers the orchestrator routes to, by role (its ``/cluster_info``)."""
    response = requests.get(orchestrator.url_for("cluster_info"), timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    lists = response.json()["server_lists"]
    return {role: [str(address) for address in lists[role]] for role in ROLES}


def wait_for_members(served: Cluster, counts: Mapping[str, int]) -> None:
    """Wait until the orchestrator lists ``counts`` workers of each role."""
    deadline = time.monotonic() + REGISTER_TIMEOUT_S
    while True:
        listed = members(served.orchestrator)
        if all(len(listed[role]) >= counts[role] for role in ROLES):
            return
        assert time.monotonic() < deadline, f"the orchestrator lists {listed}, expected {counts}"
        time.sleep(1)


def server_info(server: Any) -> dict[str, Any]:
    """A worker's ``/server_info``."""
    response = requests.get(server.url_for("server_info"), timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    return response.json()


def engine_server_info(server: Any) -> Any:
    """The ``GetServerInfo`` of a chain worker's engine."""
    import grpc

    from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc, server_pb2

    with grpc.insecure_channel(server._chain_handle.engine_endpoint) as channel:
        control = openengine_pb2_grpc.ControlStub(channel)
        return control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=30)


def engine_received(server: Any) -> int:
    """The Generate calls a chain worker's engine has received."""
    from openengine_chain import proof

    return proof.engine_received(server._chain_handle)


def sent_names(served: Cluster, role: str, index: int = 0) -> set[str]:
    """The names of the GenerateRequests a chain worker's frontend has written so far."""
    directory = served.dump_dir / f"{role}-{index}"
    return {path.name for path in directory.glob("*.json")}


def sent_since(
    served: Cluster, role: str, before: set[str], index: int = 0
) -> list[dict[str, Any]]:
    """The GenerateRequests a chain worker's frontend sent its engine since ``before``, as JSON.

    A file is named after the call's request id, so ``before`` is
    :func:`sent_names` taken before the requests were sent.
    """
    directory = served.dump_dir / f"{role}-{index}"
    names = sorted(sent_names(served, role, index) - before)
    return [json.loads((directory / name).read_text()) for name in names]


def _orchestrator(port: int, env: Mapping[str, str], config: Mapping[str, Any], log_path: str):
    from llmapi.apps.openai_server import RemoteDisaggOpenAIServer

    return RemoteDisaggOpenAIServer(
        ctx_servers=[],
        gen_servers=[],
        port=port,
        env=dict(env),
        llmapi_launch=False,
        disagg_config=dict(config),
        log_path=log_path,
        wait_ready=False,
    )


def _wait_ready(server: Any) -> None:
    """Wait for an orchestrator's ``/health``, failing at once if its process exits."""
    wait_for_endpoint_ready(
        server.url_for("health"), timeout=ENGINE_READY_TIMEOUT_S, server_proc=server.proc
    )


@contextlib.contextmanager
def another_orchestrator(
    served: Cluster,
    log_path: Path,
    *,
    orchestrator: Mapping[str, Any] | None = None,
    servers: Mapping[str, Mapping[str, Any]] | None = None,
) -> Iterator[Any]:
    """A second orchestrator in front of ``served``'s workers, given their URLs.

    Its configuration is ``served``'s with ``orchestrator`` added and
    ``servers`` added to the server sections by role; it is not a cluster.
    """
    port = pick_port()
    config = {k: v for k, v in served.orchestrator_config.items() if k != "disagg_cluster"}
    config["port"] = port
    for role, key in zip(ROLES, ("context_servers", "generation_servers")):
        urls = [f"{s.host}:{s.port}" for s in served.workers[role]]
        section = _merged(config[key], {"num_instances": len(urls), "urls": urls})
        config[key] = _merged(section, (servers or {}).get(role, {}))
    config = _merged(config, orchestrator or {})
    with _orchestrator(port, served.env, config, str(log_path)) as server:
        _wait_ready(server)
        yield server


def sent_by_frontend(server: Any) -> dict[int, int]:
    """The Generate calls each frontend process of a chain worker has sent, by pid."""
    from tensorrt_llm.serve.openengine_poc.contracts import read_generate_sent_by_pid

    return read_generate_sent_by_pid(server._chain_handle.counter_dir)


def engine_counts(served: Cluster, role: str) -> list[int]:
    """The Generate calls each chain worker of ``role`` has had its engine receive."""
    return [engine_received(server) for server in served.workers[role]]


def chain_log(server: Any, side: str, start: int = 0) -> str:
    """The ``frontend`` or ``engine`` log of a chain worker, from byte ``start`` on."""
    path = getattr(server._chain_handle, f"{side}_log")
    assert path, f"the chain worker has no {side} log"
    with open(path, "rb") as f:
        f.seek(start)
        return f.read().decode(errors="replace")


def chain_log_size(server: Any, side: str) -> int:
    """The size of a chain worker's ``frontend`` or ``engine`` log, to read on from with :func:`chain_log`."""
    path = getattr(server._chain_handle, f"{side}_log")
    assert path, f"the chain worker has no {side} log"
    return os.path.getsize(path)


def registered_workers(served: Cluster) -> list[str]:
    """The keys the workers registered under in the orchestrator's cluster storage.

    A key holds the registering process's id, so two frontends of one worker
    that both registered would have two keys.
    """
    prefix = get_worker_key_prefix(served.orchestrator_config["disagg_cluster"]["cluster_name"])
    response = requests.get(
        served.orchestrator.url_for("get_prefix"),
        params={"key_prefix": prefix, "keys_only": True},
        timeout=REQUEST_TIMEOUT_S,
    )
    # The storage answers {"result": {key: ""}}, with 400 when nothing matched.
    assert response.status_code == 200, response.text
    return sorted(response.json()["result"])


def pick_port() -> int:
    """A port for a process a test starts, picked as the chain harness picks its own and kept off them."""
    from openengine_chain import launch

    return launch.pick_port()


def port_block(count: int) -> int:
    """The first of ``count`` consecutive free ports, picked and kept as :func:`pick_port`'s."""
    from openengine_chain import launch

    return launch._free_port_block(count)
