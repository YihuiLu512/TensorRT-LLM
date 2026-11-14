# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A context leg's output carries the handoff with the postprocessing pool on, as with it off.

With the pool, the facade hands a request's events to the pool session, and
the pool builds the HTTP response; the server still builds the request's
perf record from the first of the facade's own outputs. Each chain
(``cpu_disagg_chain``) runs its worker frontends with perf records written
to one directory per role, with the pool off or on: a context leg's record
carries the ``ctx_request_id`` of the handoff, as its HTTP response does
(SV-DSG-4, SV-DSG-12).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml

from tensorrt_llm.grpc.openengine_conventions import DISAGG_REQUEST_ID_MIN

__extra_import_path__ = [".."]
from cpu_disagg_chain import CpuDisaggChain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

ROLES = ("context", "generation")
# Postprocessing processes per frontend.
POOL_SIZES = [0, 2]
PROMPT = "He llo"


@pytest.fixture(scope="module", params=POOL_SIZES, ids=[f"pool{n}" for n in POOL_SIZES])
def chain(request, tmp_path_factory):
    workers = request.param
    directory = tmp_path_factory.mktemp(f"chain-pool{workers}")
    args = {}
    for role in ROLES:
        config = directory / f"{role}.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    # SV-DSG-1: without it OpenAIServer refuses a generation leg's handoff fields.
                    "cache_transceiver_config": {"backend": "DEFAULT"},
                    "return_perf_metrics": True,
                    "perf_metrics_output_dir": str(directory / "perf" / role),
                }
            )
        )
        args[role] = ["--config", str(config)]
        if workers:
            args[role] += ["--num_postprocess_workers", str(workers)]
    with CpuDisaggChain(directory, "trtllm", frontend_args=args) as chain:
        chain.perf_dirs = {role: directory / "perf" / role for role in ROLES}
        yield chain


def _record(directory: Path, disagg_request_id: int, timeout: float = 10.0) -> dict | None:
    """The perf record of ``disagg_request_id`` in ``directory``, once written; None if none is."""
    deadline = time.monotonic() + timeout
    while True:
        for path in sorted(directory.glob("perf_metrics-*.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A line the writer has not finished yet.
                    continue
                if record.get("disagg_request_id") == disagg_request_id:
                    return record
        if time.monotonic() > deadline:
            return None
        time.sleep(0.1)


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_the_context_legs_perf_record_carries_the_handoff(chain, path):
    disagg_request_id = DISAGG_REQUEST_ID_MIN + (401 if path == "/v1/completions" else 402)
    if path == "/v1/completions":
        body = {"prompt": PROMPT, "max_tokens": 4}
    else:
        body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4}
    body["disaggregated_params"] = {
        "request_type": "context_only",
        "disagg_request_id": disagg_request_id,
    }
    response = chain.post(path, body, url=chain.ctx.url)
    assert response.status_code == 200, response.text
    _, sent = chain.ctx.engine.handoffs[-1]
    ctx_request_id = int(sent.session_id)
    assert response.json()["choices"][0]["disaggregated_params"]["ctx_request_id"] == ctx_request_id
    record = _record(chain.perf_dirs["context"], disagg_request_id)
    assert record is not None, chain.ctx.log()[-4000:]
    assert record.get("ctx_request_id") == ctx_request_id, record


def test_both_legs_of_a_request_record_the_same_handoff(chain):
    seen = len(chain.ctx.http_requests())
    response = chain.complete(prompt=PROMPT, max_tokens=4)
    assert response.status_code == 200, chain.orchestrator_log()[-4000:]
    [leg] = [r for r in chain.ctx.http_requests()[seen:] if r["path"] == "/v1/completions"]
    disagg_request_id = leg["body"]["disaggregated_params"]["disagg_request_id"]
    _, sent = chain.ctx.engine.handoffs[-1]
    for worker in chain.workers:
        record = _record(chain.perf_dirs[worker.role], disagg_request_id)
        assert record is not None, (worker.role, worker.log()[-4000:])
        assert record.get("ctx_request_id") == int(sent.session_id), (worker.role, record)
