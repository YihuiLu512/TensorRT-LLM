# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cancellation by leg and a lost engine (SV-DSG-10), on CPU.

NT-009 (DG-38, DG-39c): a client that disconnects while its leg runs gets
that leg's Generate call cancelled on that leg's engine: the context leg
after its first token and before its handoff, the generation leg while it
decodes, and, through the orchestrator, a streamed request's generation
leg. A context leg that has handed off is not cancelled. A non-streamed request whose client leaves is
not cancelled: the orchestrator does not notice (DG-39). A facade whose
engine stays unreachable reports itself lost and unhealthy, and fails a
request with 503 (DG-87). The fake engine has no queue, so a call cancelled
while queued is left to the GPU tests.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable

import pytest
import requests
import yaml

from tensorrt_llm.grpc.openengine_conventions import DISAGG_REQUEST_ID_MIN
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc.facade import OpenEngineLLM, OpenEngineServeError

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain import (  # noqa: E402
    MODEL,
    CpuDisaggChain,
    _chain_profile,
    _fingerprint,
    make_model_dir,
)
from fake_openengine import FakeEngine, disagg_profile  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

PROMPT = "He llo world"
# The fake engine waits this long between steps in the tests that disconnect.
SLOW_STEP = 1.0


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wt52-cancel")
    config = directory / "frontend.yaml"
    config.write_text(yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}}))
    args = ["--config", str(config)]
    with CpuDisaggChain(
        directory, "trtllm", frontend_args={"context": args, "generation": args}
    ) as chain:
        yield chain


@pytest.fixture
def slow(chain):
    """Make ``chain.<role>.engine`` slow for one test: ``slow("context")``."""
    engines = []

    def make_slow(role: str) -> None:
        engine = chain.ctx.engine if role == "context" else chain.gen.engine
        engine.step_delay = SLOW_STEP
        engines.append(engine)

    yield make_slow
    for engine in engines:
        engine.step_delay = 0.0


def _wait(condition: Callable[[], bool], what: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.1)


def _context_leg_body(disagg_request_id: int, **fields) -> dict:
    return {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": 4,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": disagg_request_id,
        },
        **fields,
    }


def _leave_after(url: str, body: dict, seconds: float) -> None:
    """Send ``body`` to ``url`` and close the connection after ``seconds`` without an answer."""
    with pytest.raises(requests.exceptions.ReadTimeout):
        requests.post(url, json=body, headers={"Connection": "close"}, timeout=(5, seconds))


def _leave_after_first_chunk(url: str, body: dict) -> None:
    """Stream ``body`` from ``url`` and close the connection once one piece of text arrived."""
    with requests.post(
        url, json={**body, "stream": True}, headers={"Connection": "close"}, timeout=60, stream=True
    ) as response:
        assert response.status_code == 200, response.text
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: ") and line != "data: [DONE]":
                choice = json.loads(line[len("data: ") :])["choices"][0]
                if choice.get("text"):
                    return
    raise AssertionError("the stream ended before any text")


def _new_call(engine, seen: int, what: str) -> str:
    _wait(lambda: len(engine.requests) > seen, what)
    return engine.requests[seen].request_id


def test_disconnect_aborts_the_leg_in_each_phase(chain, slow):
    """NT-009: each leg's call is cancelled on its own engine when its client leaves."""
    ctx, gen = chain.ctx.engine, chain.gen.engine

    # The context leg before its handoff: the engine has sent the first token, not prefill_ready.
    slow("context")
    seen, gen_seen = len(ctx.requests), len(gen.requests)
    _leave_after(
        f"{chain.ctx.url}/v1/completions", _context_leg_body(DISAGG_REQUEST_ID_MIN + 1), 0.3
    )
    prefill = _new_call(ctx, seen, "the prefill call")
    _wait(lambda: prefill in ctx.cancelled, "the prefill call to be cancelled")
    assert len(gen.requests) == gen_seen
    ctx.step_delay = 0.0

    # The context leg after its handoff: the call has ended; nothing is cancelled.
    seen = len(ctx.requests)
    response = requests.post(
        f"{chain.ctx.url}/v1/completions",
        json=_context_leg_body(DISAGG_REQUEST_ID_MIN + 2),
        timeout=60,
    )
    assert response.status_code == 200, response.text
    handed_off = ctx.requests[seen].request_id
    time.sleep(2 * SLOW_STEP)
    assert handed_off not in ctx.cancelled and handed_off not in ctx.aborted

    # The generation leg while it decodes, streamed.
    slow("generation")
    ctx_response = response.json()
    handoff = {
        **ctx_response["choices"][0]["disaggregated_params"],
        "request_type": "generation_only",
    }
    body = {
        "model": MODEL,
        "prompt": ctx_response["prompt_token_ids"],
        "max_tokens": 40,
        "disaggregated_params": handoff,
    }
    seen = len(gen.requests)
    _leave_after_first_chunk(f"{chain.gen.url}/v1/completions", body)
    decode = _new_call(gen, seen, "the decode call")
    _wait(lambda: decode in gen.cancelled, "the decode call to be cancelled")

    # The generation leg while it decodes, not streamed.
    seen = len(gen.requests)
    _leave_after(f"{chain.gen.url}/v1/completions", body, 2 * SLOW_STEP)
    decode = _new_call(gen, seen, "the second decode call")
    _wait(lambda: decode in gen.cancelled, "the second decode call to be cancelled")


def test_a_streamed_request_left_at_the_orchestrator_cancels_its_generation_leg(chain, slow):
    """DG-38: the orchestrator closes its connection to the generation worker, which cancels the call."""
    slow("generation")
    ctx, gen = chain.ctx.engine, chain.gen.engine
    ctx_seen, seen = len(ctx.requests), len(gen.requests)
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": 40}
    _leave_after_first_chunk(f"{chain.url}/v1/completions", body)
    decode = _new_call(gen, seen, "the decode call")
    _wait(lambda: decode in gen.cancelled, "the decode call to be cancelled")
    prefill = ctx.requests[ctx_seen].request_id
    assert prefill not in ctx.cancelled and prefill not in ctx.aborted


def test_a_request_left_at_the_orchestrator_unstreamed_runs_to_its_end(chain, slow):
    """DG-39: the orchestrator does not notice; both legs run to their end, as in-process."""
    slow("generation")
    ctx, gen = chain.ctx.engine, chain.gen.engine
    seen = len(gen.requests)
    max_tokens = 4
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": max_tokens}
    _leave_after(f"{chain.url}/v1/completions", body, SLOW_STEP)
    decode = _new_call(gen, seen, "the decode call")
    _wait(
        lambda: decode not in gen._active,
        "the decode call to end",
        timeout=(max_tokens + 4) * SLOW_STEP,
    )
    assert decode not in gen.cancelled and decode not in gen.aborted
    assert not set(ctx.cancelled) & {r.request_id for r in ctx.requests[-1:]}


def test_a_facade_whose_engine_is_lost_reports_it(tmp_path):
    """DG-87 (SV-DSG-10): a lost engine makes the facade call ``on_engine_lost`` and fail requests.

    The engine stops answering; once ``engine_lost_timeout`` has passed the
    facade calls ``on_engine_lost`` (which stops the server process when
    ``trtllm-serve`` starts it), ``check_health`` is false, and a request it is
    given fails with 503.
    """
    model = make_model_dir(tmp_path / "model")
    profile = _chain_profile(disagg_profile("trtllm-like"), _fingerprint(model))
    lost: list[OpenEngineLLM] = []

    async def main() -> None:
        fake = FakeEngine(profile)
        await fake.start()
        facade = OpenEngineLLM.from_llm_args(
            {"model": str(model)},
            fake.target,
            ready_timeout=30,
            health_poll_interval=0.1,
            engine_lost_timeout=0.5,
            on_engine_lost=lost.append,
        )
        await facade.start()
        try:
            assert facade.check_health()
            await fake.stop()
            for _ in range(100):
                if lost:
                    break
                await asyncio.sleep(0.1)
            assert lost == [facade]
            assert not facade.check_health()
            with pytest.raises(OpenEngineServeError) as error:
                await facade.generate_async(
                    {"prompt_token_ids": [2, 3]}, SamplingParams(max_tokens=2)
                )
            assert error.value.status_code == 503
        finally:
            await facade.aclose()

    asyncio.run(main())
