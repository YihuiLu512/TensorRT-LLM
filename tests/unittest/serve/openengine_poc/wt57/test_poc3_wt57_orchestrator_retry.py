# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The orchestrator's retries and request timeout with worker frontends (DG-45 to DG-50), on CPU.

Each test runs the CPU chain (``cpu_disagg_chain``): the real orchestrator,
two real worker frontends and their fake engines. The orchestrator keeps its
default of one retry unless a test says otherwise.

* NT-012 (DG-45): a context leg that fails is sent again with a new
  ``disagg_request_id``, which is the id the context engine is given and
  the generation leg then names.
* NT-013 (DG-46, DG-47): a generation leg that the generation worker
  frontend answered with an error is not sent again.
* NT-014 (DG-49): ``TRTLLM_DISAGG_NO_RETRY=1`` turns the context leg's retry off.
* NT-015 (DG-50): the orchestrator's request timeout (``-r``) ends a leg the
  engine never answers, and the worker frontend then ends the engine call.
"""

from __future__ import annotations

import time

import grpc
import pytest

from tensorrt_llm.grpc.openengine_conventions import REQ_DISAGG_REQUEST_ID

__extra_import_path__ = [".", ".."]
from wt57_chain import FAILURE, PROMPT, Fail, Hang, calls, start_chain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

REQUEST_TIMEOUT_S = 3


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
    chain.ctx.engine.push_script(Fail(grpc.StatusCode.INTERNAL, FAILURE))
    response = chain.complete(prompt=PROMPT, max_tokens=4)
    assert response.status_code == 200, response.text
    assert calls(chain)["context"] - before["context"] == 2
    assert calls(chain)["generation"] - before["generation"] == 1
    failed, retried = chain.ctx.engine.requests[-2:]
    first = failed.extra[REQ_DISAGG_REQUEST_ID]
    second = retried.extra[REQ_DISAGG_REQUEST_ID]
    assert first != second
    # The generation leg pairs with the attempt that succeeded.
    session = chain.gen.engine.requests[-1].kv.session
    assert session.attributes_struct["disagg_request_id"] == second


@pytest.mark.parametrize(
    "code, status",
    [(grpc.StatusCode.INVALID_ARGUMENT, 400), (grpc.StatusCode.INTERNAL, 500)],
    ids=["refused", "failed"],
)
def test_gen_leg_retry_rules_hold_with_worker_frontends(chain, code, status):
    before = calls(chain)
    chain.gen.engine.push_script(Fail(code, FAILURE))
    response = chain.complete(prompt=PROMPT, max_tokens=4)
    assert response.status_code == status, response.text
    assert FAILURE in response.text, response.text
    # The generation worker frontend answered, so the leg may own the context
    # KV cache: it is not sent again, and the context leg is not repeated either.
    assert calls(chain)["generation"] - before["generation"] == 1
    assert calls(chain)["context"] - before["context"] == 1


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


def test_request_timeout_cuts_the_worker_request(timed_chain):
    chain = timed_chain
    before = calls(chain)
    chain.ctx.engine.push_script(Hang())
    started = time.monotonic()
    response = chain.complete(prompt=PROMPT, max_tokens=4, timeout=60)
    elapsed = time.monotonic() - started
    assert response.status_code >= 500, response.text
    assert REQUEST_TIMEOUT_S <= elapsed < REQUEST_TIMEOUT_S + 20, elapsed
    assert calls(chain)["context"] - before["context"] == 1
    assert calls(chain)["generation"] - before["generation"] == 0
    # The orchestrator dropped the connection; the worker frontend ends the engine call.
    hung = chain.ctx.engine.requests[-1].request_id
    deadline = time.monotonic() + 30
    while hung in chain.ctx.engine.running_request_ids and time.monotonic() < deadline:
        time.sleep(0.2)
    assert hung not in chain.ctx.engine.running_request_ids
    assert hung in chain.ctx.engine.cancelled + chain.ctx.engine.aborted
