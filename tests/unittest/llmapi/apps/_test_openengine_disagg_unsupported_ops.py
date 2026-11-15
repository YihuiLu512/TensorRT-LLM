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
  start; the test leaves the startup error to propagate, for the chain's
  ``startup_rejected`` expectation to judge, which also looks for the reason
  in the worker frontend's log. The chain harness has no side to put
  ``--enable_sleep_mode`` on, so the chain's expectations skip that case.
- NT-079 (F-47c): ``/start_profile`` and ``/stop_profile`` sent to each worker
  itself. In-process the worker starts and stops profiling; on the chain the
  worker frontend answers 501 for its OpenEngine engine.
"""

import secrets
from collections.abc import Mapping
from typing import Any

import pytest
import requests

from ._test_openengine_disagg_common import CHAIN, REQUEST_TIMEOUT_S, ROLES, disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"
# Per option: the workers' extra command line, configuration and environment.
OPS_OPTIONS: Mapping[str, Mapping[str, Any]] = {
    "resource_governor": {"worker_config": {"enable_resource_governor": True}},
    "sleep_mode": {
        "worker_args": ("--enable_sleep_mode",),
        # Sleep mode refuses to start without the key that guards its endpoints.
        "env": {"TRTLLM_RUNTIME_CONTROL_API_KEY": secrets.token_hex(16)},
    },
}


@pytest.mark.parametrize("option", sorted(OPS_OPTIONS))
def test_ops_options_startup_rejected(tmp_path, option):
    kwargs = OPS_OPTIONS[option]
    if not CHAIN:
        with disagg_deployment(tmp_path, **kwargs) as deployment:
            response = deployment.client().completions.create(
                model=deployment.model, prompt=PROMPT, max_tokens=4
            )
            assert response.choices[0].text
        return
    # A worker frontend that exits makes the deployment raise RuntimeError, which the
    # chain's startup_rejected expectation turns into a pass.
    with disagg_deployment(tmp_path, **kwargs):
        pytest.fail(f"the worker frontends started with {option}")


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
