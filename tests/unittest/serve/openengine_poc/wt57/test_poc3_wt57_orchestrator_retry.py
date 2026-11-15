# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The orchestrator's retries and request timeout with worker frontends (DG-45 to DG-50), on CPU.

Each test runs the CPU chain (``cpu_disagg_chain``): the real orchestrator,
two real worker frontends and their fake engines. The orchestrator keeps its
default of one retry unless a test says otherwise.

* NT-012 (DG-45): a context leg that fails is sent again with a new
  ``disagg_request_id``, which is the id the context engine is given and
  the generation leg then names; the failed attempt leaves no handoff behind.
* NT-013 (DG-46, DG-47): a generation leg is sent again, to the same
  address, only while its worker frontend cannot be reached; once the
  frontend has answered with an error, or has streamed bytes and then broken
  off, the leg is not sent again, and its ``kv.session`` reaches the engine
  once. The id of a leg sent again cannot be seen, as no server receives it.
* NT-014 (DG-49): ``TRTLLM_DISAGG_NO_RETRY=1`` turns the context leg's retry off.
* NT-015 (DG-50): the orchestrator's request timeout (``-r``) ends a leg the
  engine never answers, and the worker frontend then ends the engine call.
"""

from __future__ import annotations

import json
import re
import time

import grpc
import pytest
import requests

from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_GENERATION_ONLY,
    COUNTER_PREFILL_READY_SENT,
    REQ_DISAGG_REQUEST_ID,
)

__extra_import_path__ = [".", "..", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain import MODEL  # noqa: E402
from fake_openengine import DisaggFaults  # noqa: E402
from wt57_chain import FAILURE, PROMPT, Fail, Hang, calls, start_chain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REQUEST_TIMEOUT_S = 3
_GEN_RETRIES = re.compile(r"^gen_retry_requests_total(?:\{[^}]*\})?\s+(\S+)\s*$", re.MULTILINE)


def _gen_retries(chain) -> int:
    """The orchestrator's count of generation legs it sent again."""
    metrics = requests.get(f"{chain.url}/prometheus/metrics", timeout=30).text
    found = _GEN_RETRIES.search(metrics)
    return int(float(found.group(1))) if found else 0


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    with start_chain(tmp_path_factory.mktemp("wt57-retry")) as chain:
        yield chain


@pytest.fixture(scope="module")
def no_retry_chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wt57-no-retry")
    with start_chain(directory, env={"TRTLLM_DISAGG_NO_RETRY": "1"}) as chain:
        yield chain


@pytest.fixture(scope="module")
def timed_chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wt57-timeout")
    with start_chain(
        directory, orchestrator_config={"max_retries": 0}, request_timeout=REQUEST_TIMEOUT_S
    ) as chain:
        yield chain


def test_ctx_retry_sends_a_new_disagg_request_id_to_the_engine(chain):
    before = calls(chain)
    handoffs = chain.ctx.engine.counters[COUNTER_PREFILL_READY_SENT]
    chain.ctx.engine.push_script(Fail(grpc.StatusCode.INTERNAL, FAILURE))
    response = chain.complete(prompt=PROMPT, max_tokens=4)
    assert response.status_code == 200, response.text
    # Only the attempt that succeeded hands its KV cache off, to the generation leg.
    assert chain.ctx.engine.counters[COUNTER_PREFILL_READY_SENT] - handoffs == 1
    assert calls(chain)["context"] - before["context"] == 2
    assert calls(chain)["generation"] - before["generation"] == 1
    failed, retried = chain.ctx.engine.requests[-2:]
    first = failed.extra[REQ_DISAGG_REQUEST_ID]
    second = retried.extra[REQ_DISAGG_REQUEST_ID]
    assert first != second
    # The generation leg pairs with the attempt that succeeded.
    session = chain.gen.engine.requests[-1].kv.session
    assert session.attributes_struct["disagg_request_id"] == second


def _refused_by_the_frontend(chain, code, status) -> None:
    before = calls(chain)
    chain.gen.engine.push_script(Fail(code, FAILURE))
    response = chain.complete(prompt=PROMPT, max_tokens=4)
    assert response.status_code == status, response.text
    assert FAILURE in response.text, response.text
    # The generation worker frontend answered, so the leg may own the context
    # KV cache: it is not sent again, and the context leg is not repeated either.
    assert calls(chain)["generation"] - before["generation"] == 1
    assert calls(chain)["context"] - before["context"] == 1


def _frontend_unreachable(directory) -> None:
    with start_chain(directory) as chain:
        before = _gen_retries(chain)
        chain.gen.process.stop()
        response = chain.complete(prompt=PROMPT, max_tokens=4)
        assert response.status_code >= 500, response.text
        # One retry (the default max_retries), to the same unreachable frontend.
        assert _gen_retries(chain) - before == 1
        assert calls(chain) == {"context": 1, "generation": 0}


def _streamed_chunks(response: requests.Response) -> list[dict]:
    """The JSON chunks of a streamed completion, up to where the stream broke off."""
    chunks = []
    try:
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line.removeprefix("data: ")))
    except requests.exceptions.ChunkedEncodingError:
        pass
    return chunks


def _stream_broken_after_bytes(chain) -> None:
    before = calls(chain)
    retries = _gen_retries(chain)
    # The generation engine sends its first token, then ends the call without a terminal event.
    chain.gen.engine.disagg_faults = DisaggFaults(
        close_before_terminal=frozenset({BUCKET_GENERATION_ONLY})
    )
    try:
        with requests.post(
            f"{chain.url}/v1/completions",
            json={"model": MODEL, "prompt": PROMPT, "max_tokens": 4, "stream": True},
            stream=True,
            timeout=60,
        ) as response:
            chunks = _streamed_chunks(response)
    finally:
        chain.gen.engine.disagg_faults = DisaggFaults()
    assert chunks, "no bytes reached the client before the break"
    finished = [
        choice.get("finish_reason")
        for chunk in chunks
        for choice in chunk.get("choices", [])
        if choice.get("finish_reason")
    ]
    assert not set(finished) & {"stop", "length"}, chunks
    assert _gen_retries(chain) == retries
    assert calls(chain)["generation"] - before["generation"] == 1
    assert calls(chain)["context"] - before["context"] == 1


@pytest.mark.parametrize("case", ["refused", "failed", "unreachable", "stream_broken"])
def test_gen_leg_retry_rules_hold_with_worker_frontends(request, tmp_path, case):
    if case == "unreachable":
        _frontend_unreachable(tmp_path)
        return
    chain = request.getfixturevalue("chain")
    if case == "refused":
        _refused_by_the_frontend(chain, grpc.StatusCode.INVALID_ARGUMENT, 400)
    elif case == "failed":
        _refused_by_the_frontend(chain, grpc.StatusCode.INTERNAL, 500)
    else:
        _stream_broken_after_bytes(chain)


def test_no_retry_env_disables_retries(no_retry_chain):
    chain = no_retry_chain
    before = calls(chain)
    chain.ctx.engine.push_script(Fail(grpc.StatusCode.INTERNAL, FAILURE))
    response = chain.complete(prompt=PROMPT, max_tokens=4)
    assert response.status_code == 500, response.text
    assert calls(chain)["context"] - before["context"] == 1
    assert calls(chain)["generation"] - before["generation"] == 0
    # The same chain still serves a request that does not fail.
    assert chain.complete(prompt=PROMPT, max_tokens=4).status_code == 200


@pytest.mark.parametrize("role", ["context", "generation"])
def test_request_timeout_cuts_the_worker_request(timed_chain, role):
    chain = timed_chain
    worker = chain.ctx if role == "context" else chain.gen
    before = calls(chain)
    worker.engine.push_script(Hang())
    started = time.monotonic()
    response = chain.complete(prompt=PROMPT, max_tokens=4, timeout=60)
    elapsed = time.monotonic() - started
    assert response.status_code >= 500, response.text
    assert REQUEST_TIMEOUT_S <= elapsed < REQUEST_TIMEOUT_S + 20, elapsed
    assert calls(chain)["context"] - before["context"] == 1
    assert calls(chain)["generation"] - before["generation"] == (role == "generation")
    # The orchestrator dropped the connection; the worker frontend ends the engine call.
    hung = worker.engine.requests[-1].request_id
    deadline = time.monotonic() + 30
    while hung in worker.engine.running_request_ids and time.monotonic() < deadline:
        time.sleep(0.2)
    assert hung not in worker.engine.running_request_ids
    assert hung in worker.engine.cancelled + worker.engine.aborted
