# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A generation-first decode call to an engine that does not declare it (SV-DSG-8).

The generation worker frontend refuses the request with 400 and sends its
engine nothing. The engine here has KV transfer but does not declare
``trtllm.disagg.generation_first``.
"""

from __future__ import annotations

import pytest
import requests
import yaml

from tensorrt_llm.grpc.openengine_conventions import (
    CAP_DISAGG_GENERATION_FIRST,
    DISAGG_REQUEST_ID_MIN,
    SCHEDULE_STYLE_GENERATION_FIRST,
)

__extra_import_path__ = [".."]
from cpu_disagg_chain import MODEL, CpuDisaggChain, disagg_profile  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("chain-no-gen-first")
    config = directory / "frontend.yaml"
    config.write_text(yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}}))
    profile = disagg_profile("trtllm-like", capabilities={CAP_DISAGG_GENERATION_FIRST: False})
    args = ["--config", str(config)]
    with CpuDisaggChain(
        directory,
        "trtllm",
        profiles={"context": profile, "generation": profile},
        frontend_args={"context": args, "generation": args},
    ) as chain:
        yield chain


def test_a_generation_first_decode_to_an_engine_without_it_is_refused(chain):
    worker = chain.gen
    before = len(worker.engine.requests)
    response = requests.post(
        f"{worker.url}/v1/completions",
        json={
            "model": MODEL,
            "prompt": "He llo",
            "max_tokens": 2,
            "disaggregated_params": {
                "request_type": "generation_only",
                "ctx_request_id": DISAGG_REQUEST_ID_MIN,
                "disagg_request_id": DISAGG_REQUEST_ID_MIN,
                "schedule_style": SCHEDULE_STYLE_GENERATION_FIRST,
            },
        },
        timeout=60,
    )
    assert response.status_code == 400, response.text
    assert CAP_DISAGG_GENERATION_FIRST in response.text
    assert len(worker.engine.requests) == before
