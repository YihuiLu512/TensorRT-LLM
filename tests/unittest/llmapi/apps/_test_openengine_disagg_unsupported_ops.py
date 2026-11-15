# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operations options and endpoints a worker has in-process and a worker frontend refuses (F-47a to F-47c).

The deployments are those of ``_test_openengine_disagg_common`` (Qwen3-0.6B, one
context and one generation worker on one GPU), in-process or on the
OpenEngine chain. The in-process worker has the resource governor, sleep mode
and the profiling endpoints; a worker frontend has no engine in its process to
run them on.

- NT-078 (F-47a, F-47b): workers started with ``enable_resource_governor`` in
  their configuration, or with ``--enable_sleep_mode``. In-process they start
  and the deployment serves. On the chain each worker frontend refuses to
  start and its log says why; the test leaves the startup error to propagate,
  for the chain's ``startup_rejected`` expectation to judge.
- NT-079 (F-47c): ``/start_profile`` and ``/stop_profile`` sent to each worker
  itself. In-process the worker starts and stops profiling; on the chain the
  worker frontend answers 501 for its OpenEngine engine.
"""

import contextlib
import os
import secrets
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import requests
from test_common.http_utils import wait_for_endpoint_ready

from tensorrt_llm._utils import get_free_ports

from ._test_openengine_disagg_common import (
    BASE_WORKER_CONFIG,
    CHAIN,
    ENGINE_READY_TIMEOUT_S,
    MODEL_NAME,
    REQUEST_TIMEOUT_S,
    ROLES,
    Deployment,
    disagg_deployment,
    model_path,
    needs_gpu,
)
from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"
# Per option: the workers' extra command line, configuration and environment, and
# what the worker frontend's log says when it refuses to start.
OPS_OPTIONS: Mapping[str, Mapping[str, Any]] = {
    "resource_governor": {
        "worker_config": {"enable_resource_governor": True},
        "refusal": "enable_resource_governor is not supported",
    },
    "sleep_mode": {
        "worker_args": ("--enable_sleep_mode",),
        # Sleep mode refuses to start without the key that guards its endpoints.
        "env": {"TRTLLM_RUNTIME_CONTROL_API_KEY": secrets.token_hex(16)},
        "refusal": "--openengine-endpoint cannot be combined with --enable_sleep_mode",
    },
}


@contextlib.contextmanager
def _deployment_whose_workers_may_exit(
    directory: Path,
    *,
    worker_args: Sequence[str] = (),
    worker_config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> Iterator[Deployment]:
    """The deployment ``disagg_deployment`` starts, waiting for the workers before the orchestrator.

    ``disagg_deployment`` waits for the orchestrator's ``/health``, which turns 200 only once
    workers have registered, so a worker that exits at startup would hold it until its
    two-hour start limit. Here the orchestrator starts without that wait, and a worker that
    exits raises ``RuntimeError`` as soon as it is waited for.
    """
    needs_gpu()
    path = model_path(MODEL_NAME)
    ports = get_free_ports(1 + len(ROLES))
    disagg_port, worker_ports = ports[0], iter(ports[1:])
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}
    auth_key = secrets.token_hex(32)
    config = {
        **BASE_WORKER_CONFIG,
        "model": MODEL_NAME,
        "disagg_cluster": cluster,
        "internal_request_auth_key": auth_key,
        **(worker_config or {}),
    }
    process_env = {**os.environ, "UCX_NET_DEVICES": "lo", **(env or {})}
    logs = {role: [str(directory / f"{role}_0.log")] for role in ROLES}
    with contextlib.ExitStack() as stack:
        orchestrator = stack.enter_context(
            RemoteDisaggOpenAIServer(
                ctx_servers=[],
                gen_servers=[],
                port=disagg_port,
                env=process_env,
                disagg_config={
                    "hostname": "localhost",
                    "port": disagg_port,
                    "disagg_cluster": cluster,
                    "internal_request_auth_key": auth_key,
                },
                log_path=str(directory / "orchestrator.log"),
                wait_ready=False,
            )
        )
        workers = {
            role: [
                stack.enter_context(
                    RemoteOpenAIServer(
                        path,
                        port=next(worker_ports),
                        env=process_env,
                        cli_args=["--server_role", role, *worker_args],
                        extra_config=config,
                        log_path=logs[role][0],
                        wait=False,
                    )
                )
            ]
            for role in ROLES
        }
        for role in ROLES:
            workers[role][0].wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
        wait_for_endpoint_ready(orchestrator.url_root + "/health")
        yield Deployment(orchestrator, workers, logs, MODEL_NAME)


def _logs(directory: Path) -> str:
    return "".join(path.read_text(errors="replace") for path in sorted(directory.glob("*_0.log")))


@pytest.mark.parametrize("option", sorted(OPS_OPTIONS))
def test_ops_options_startup_rejected(tmp_path, option):
    settings = OPS_OPTIONS[option]
    kwargs = {
        key: settings[key] for key in ("worker_args", "worker_config", "env") if key in settings
    }
    if not CHAIN:
        with _deployment_whose_workers_may_exit(tmp_path, **kwargs) as deployment:
            response = deployment.client().completions.create(
                model=deployment.model, prompt=PROMPT, max_tokens=4
            )
            assert response.choices[0].text
        return
    try:
        with _deployment_whose_workers_may_exit(tmp_path, **kwargs):
            pytest.fail(f"the worker frontends started with {option}")
    except RuntimeError:
        assert settings["refusal"] in _logs(tmp_path), f"no worker log names {option}"
        # The chain's startup_rejected expectation passes the test on this error.
        raise


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(tmp_path_factory.mktemp("disagg_unsupported_ops")) as deployment:
        yield deployment


def test_profile_endpoints_refused_direct(deployment, tmp_path):
    for role in ROLES:
        worker = deployment.workers[role][0]
        start = requests.post(
            worker.url_for("start_profile"),
            json={"output_dir": str(tmp_path / role)},
            timeout=REQUEST_TIMEOUT_S,
        )
        stop = requests.post(worker.url_for("stop_profile"), timeout=REQUEST_TIMEOUT_S)
        if CHAIN:
            for response in (start, stop):
                assert response.status_code == 501, (role, response.text)
                assert "OpenEngine" in response.json()["message"], (role, response.text)
        else:
            assert start.status_code == 200, (role, start.text)
            assert start.json() == {"message": "Profiling started"}, (role, start.text)
            assert stop.status_code == 200, (role, stop.text)
            assert stop.json() == {"message": "Profiling stopped"}, (role, stop.text)
