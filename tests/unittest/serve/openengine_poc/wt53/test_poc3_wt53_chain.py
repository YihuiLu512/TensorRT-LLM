# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The second batch's serving side on the CPU chain (``cpu_disagg_chain``).

- ``/server_info`` of a worker frontend takes ``max_batch_size`` and
  ``kv_cache_hash_algo`` from the engine's ``ServerInfo``, and the context
  endpoint from its KV connector (SV-DSG-11, CS-10): the frontend YAML says
  otherwise, so the values prove where they come from.
- ``orig_elapsed_s`` that the orchestrator stamps reaches the generation
  engine of a TensorRT-LLM-shaped chain in the decode call's ``extra``, never
  the prefill call's; a ds-shaped engine, which does not declare it, is not
  sent it and serves the request (EX-DSG-3, CS-15).
"""

from __future__ import annotations

import pytest
import requests
import yaml

from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_CONTEXT_ONLY,
    BUCKET_GENERATION_ONLY,
    REQ_ORIG_ELAPSED_S,
)

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain import CpuDisaggChain  # noqa: E402
from fake_openengine import FakeKvConnector, disagg_profile  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

HASH_ALGO = "v2_sha256_64"
ENGINE_MAX_RUNNING = 96
YAML_MAX_BATCH_SIZE = 8
CTX_ENDPOINT = ("10.1.2.3", 4567, "tcp")


def _profiles(shape: str) -> dict:
    """Each role's engine, the context one naming where generation-first requests find it."""
    base = "trtllm-like" if shape == "trtllm" else "ds-like"
    profiles = {}
    for role in ("context", "generation"):
        endpoints = (CTX_ENDPOINT,) if role == "context" else ()
        profiles[role] = disagg_profile(
            base,
            role if shape == "ds" else None,
            kv_connector=FakeKvConnector(local_endpoints=endpoints),
            max_running_requests=ENGINE_MAX_RUNNING,
            server_info_extra={"trtllm_kv_event_hash_algo": HASH_ALGO},
        )
    return profiles


@pytest.fixture(scope="module", params=["trtllm", "ds"])
def chain(request, tmp_path_factory):
    shape = request.param
    directory = tmp_path_factory.mktemp(f"chain-{shape}")
    config = directory / "frontend.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "cache_transceiver_config": {"backend": "DEFAULT"},
                "max_batch_size": YAML_MAX_BATCH_SIZE,
            }
        )
    )
    args = ["--config", str(config)]
    with CpuDisaggChain(
        directory,
        shape,
        profiles=_profiles(shape),
        frontend_args={"context": args, "generation": args},
    ) as chain:
        yield chain


def _server_info(url: str) -> dict:
    response = requests.get(f"{url}/server_info", timeout=60)
    assert response.status_code == 200, response.text
    return response.json()


def test_server_info_takes_its_batch_size_and_hash_algo_from_the_engine(chain):
    """DG-89 on CPU: max_batch_size, kv_cache_hash_algo and the context endpoint."""
    for worker in chain.workers:
        info = _server_info(worker.url)
        assert info["max_batch_size"] == ENGINE_MAX_RUNNING != YAML_MAX_BATCH_SIZE
        assert info["kv_cache_hash_algo"] == HASH_ALGO
        assert info["engine"]["kind"] == "openengine"
    # One string, as an in-process context worker gives it; the engine reports no DP rank.
    assert _server_info(chain.ctx.url)["disaggregated_params"] == {
        "ctx_dp_rank": None,
        "ctx_info_endpoint": ["tcp://10.1.2.3:4567"],
    }
    assert _server_info(chain.gen.url)["disaggregated_params"] is None


def _calls(engine, bucket: str) -> list:
    return [r for r in engine.received if r.admitted is not None and r.admitted.bucket == bucket]


def test_orig_elapsed_s_reaches_the_generation_engine_only(chain):
    """EG-21 on CPU: the decode call carries the orchestrator's value; the prefill call does not."""
    ctx_before = len(_calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY))
    gen_before = len(_calls(chain.gen.engine, BUCKET_GENERATION_ONLY))
    response = chain.complete("He llo", max_tokens=3)
    assert response.status_code == 200, chain.orchestrator_log()[-4000:]
    (prefill,) = _calls(chain.ctx.engine, BUCKET_CONTEXT_ONLY)[ctx_before:]
    (decode,) = _calls(chain.gen.engine, BUCKET_GENERATION_ONLY)[gen_before:]
    assert REQ_ORIG_ELAPSED_S not in prefill.request.extra.fields
    gen_http = [r for r in chain.gen.http_requests() if r["path"] == "/v1/completions"][-1]
    stamped = gen_http["body"]["disaggregated_params"]["orig_elapsed_s"]
    assert stamped is not None and stamped >= 0
    if chain.shape == "trtllm":
        sent = decode.request.extra.fields[REQ_ORIG_ELAPSED_S].number_value
        assert sent == pytest.approx(stamped)
    else:
        # ds-engine does not declare trtllm.disagg.orig_elapsed_s.
        assert REQ_ORIG_ELAPSED_S not in decode.request.extra.fields
