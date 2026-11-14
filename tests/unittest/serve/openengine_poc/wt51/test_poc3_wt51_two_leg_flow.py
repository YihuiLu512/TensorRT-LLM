# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated requests through real worker frontends on the CPU chain (SV-DSG-2 to SV-DSG-5, SV-DSG-13).

Each chain (``cpu_disagg_chain``) runs an orchestrator, a context and a
generation worker frontend, and their fake engines, for one handoff shape,
with the postprocessing pool off or on. NT-003: a context leg that finishes on
the context engine is the whole answer. NT-017: the caller's
``disagg_request_id`` reaches both engines. The handoff crosses from one
engine to the other unchanged, and the counters of every worker frontend
add up (SV-DSG-13).
"""

from __future__ import annotations

import json

import pytest
import requests
from google.protobuf import json_format

from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_AGGREGATED,
    BUCKET_CONTEXT_ONLY,
    BUCKET_GENERATION_ONLY,
    COUNTER_PREFILL_READY_SENT,
    DISAGG_REQUEST_ID_MIN,
    REQ_DISAGG_REQUEST_ID,
    REQUEST_BUCKETS,
    SERVE_COUNTER_GENERATE_SENT,
    SERVE_COUNTER_GENERATE_SENT_BUCKETS,
    SERVE_COUNTER_HANDOFF_UNREGISTERED_ATTRIBUTES,
    SERVE_COUNTER_HTTP_RECEIVED_BUCKETS,
    SERVE_COUNTER_HTTP_REJECTED,
    SERVE_COUNTER_PREFILL_READY_RECEIVED,
)

__extra_import_path__ = [".."]
from cpu_disagg_chain import CpuDisaggChain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

# (handoff shape, postprocessing processes per frontend)
CHAINS = [("trtllm", 0), ("trtllm", 2), ("ds", 0), ("ds", 2)]
PROMPT = "He llo"
# The outputs each chain gave, by test and chain, compared across pool sizes.
_OUTPUTS: dict[tuple[str, str], dict[tuple[str, int], object]] = {}


@pytest.fixture(scope="module", params=CHAINS, ids=[f"{s}-pool{n}" for s, n in CHAINS])
def chain(request, tmp_path_factory):
    shape, workers = request.param
    args = ["--num_postprocess_workers", str(workers)] if workers else []
    directory = tmp_path_factory.mktemp(f"chain-{shape}-{workers}")
    with CpuDisaggChain(
        directory, shape, frontend_args={"context": args, "generation": args}
    ) as chain:
        chain.workers_per_frontend = workers
        yield chain


def _delta(before: dict, after: dict, role: str, side: str, key: str) -> int:
    return after[role][side].get(key, 0) - before[role][side].get(key, 0)


def _check_counters(chain, before: dict, after: dict) -> None:
    """SV-DSG-13 on every worker frontend, and prefill_ready against the context engine."""
    for worker in chain.workers:
        role = worker.role
        received = sum(
            _delta(before, after, role, "frontend", SERVE_COUNTER_HTTP_RECEIVED_BUCKETS[bucket])
            for bucket in REQUEST_BUCKETS
        )
        sent = _delta(before, after, role, "frontend", SERVE_COUNTER_GENERATE_SENT)
        rejected = _delta(before, after, role, "frontend", SERVE_COUNTER_HTTP_REJECTED)
        assert received == sent + rejected, (role, before[role], after[role])
        buckets = sum(
            _delta(before, after, role, "frontend", key)
            for key in SERVE_COUNTER_GENERATE_SENT_BUCKETS.values()
        )
        assert buckets == sent, (role, after[role])
    assert _delta(before, after, "context", "frontend", SERVE_COUNTER_PREFILL_READY_RECEIVED) == (
        _delta(before, after, "context", "engine", COUNTER_PREFILL_READY_SENT)
    )


def _frontend(before: dict, after: dict, role: str, key: str) -> int:
    return _delta(before, after, role, "frontend", key)


def _post_stream(chain, path: str, body: dict) -> tuple[int, str]:
    with requests.post(
        f"{chain.url}{path}",
        json={"model": "cpu-chain-model", **body, "stream": True},
        headers={"Connection": "close"},
        timeout=60,
        stream=True,
    ) as response:
        text = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choice = json.loads(line[len("data: ") :])["choices"][0]
            text += choice.get("text") or (choice.get("delta") or {}).get("content") or ""
        return response.status_code, text


def _body(path: str) -> dict:
    if path == "/v1/completions":
        return {"prompt": PROMPT, "max_tokens": 4}
    return {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4}


def _text(path: str, response: dict) -> str:
    choice = response["choices"][0]
    return choice["text"] if path == "/v1/completions" else choice["message"]["content"]


@pytest.mark.parametrize("stream", [False, True], ids=["whole", "streamed"])
@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
def test_a_request_goes_through_both_legs(chain, path, stream):
    before = chain.snapshot()
    if stream:
        status, text = _post_stream(chain, path, _body(path))
    else:
        response = chain.post(path, _body(path))
        status, text = response.status_code, _text(path, response.json())
    assert status == 200, chain.orchestrator_log()[-4000:]
    assert text
    after = chain.snapshot()
    assert (
        _frontend(
            before, after, "context", SERVE_COUNTER_HTTP_RECEIVED_BUCKETS[BUCKET_CONTEXT_ONLY]
        )
        == 1
    )
    assert (
        _frontend(
            before, after, "generation", SERVE_COUNTER_HTTP_RECEIVED_BUCKETS[BUCKET_GENERATION_ONLY]
        )
        == 1
    )
    assert (
        _frontend(
            before, after, "context", SERVE_COUNTER_GENERATE_SENT_BUCKETS[BUCKET_CONTEXT_ONLY]
        )
        == 1
    )
    # The decode call is the one with kv.session (EX-DSG-5).
    assert (
        _frontend(
            before, after, "generation", SERVE_COUNTER_GENERATE_SENT_BUCKETS[BUCKET_GENERATION_ONLY]
        )
        == 1
    )
    assert _frontend(before, after, "context", SERVE_COUNTER_PREFILL_READY_RECEIVED) == 1
    _check_counters(chain, before, after)
    _OUTPUTS.setdefault((path, str(stream)), {})[(chain.shape, chain.workers_per_frontend)] = (
        chain.shape,
        text,
    )


def _context_leg(chain, disagg_request_id: int, path: str = "/v1/completions") -> dict:
    body = {
        **_body(path),
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": disagg_request_id,
        },
    }
    response = chain.post(path, body, url=chain.ctx.url)
    assert response.status_code == 200, response.text
    return response.json()


def _generation_leg(chain, ctx: dict, **changes) -> requests.Response:
    params = {**ctx["choices"][0]["disaggregated_params"], "request_type": "generation_only"}
    params.update(changes)
    body = {"prompt": ctx["prompt_token_ids"], "max_tokens": 4, "disaggregated_params": params}
    return chain.post("/v1/completions", body, url=chain.gen.url)


def test_the_context_response_carries_the_handoff_and_no_text(chain):
    disagg_request_id = DISAGG_REQUEST_ID_MIN + 101
    ctx = _context_leg(chain, disagg_request_id)
    choice = ctx["choices"][0]
    # SV-DSG-4: ends as length, and the first token is the generation leg's to send.
    assert choice["finish_reason"] == "length"
    assert choice["text"] == ""
    assert ctx["prompt_token_ids"]
    params = choice["disaggregated_params"]
    _, sent = chain.ctx.engine.handoffs[-1]
    attributes = json_format.MessageToDict(sent.attributes_struct)
    assert params["request_type"] == "context_only"
    assert params["disagg_request_id"] == disagg_request_id
    assert params["ctx_request_id"] == int(sent.session_id)
    assert params["ctx_info_endpoint"] == attributes["ctx_info_endpoint"]
    assert params["first_gen_tokens"] == [int(token) for token in attributes["first_gen_tokens"]]
    assert "openengine_handoff_attributes" not in params
    # The endpoint names the chain's own engine; the rest must not depend on the pool.
    compared = {key: value for key, value in params.items() if key != "ctx_info_endpoint"}
    _OUTPUTS.setdefault(("ctx", ""), {})[(chain.shape, chain.workers_per_frontend)] = (
        chain.shape,
        (choice["text"], choice["finish_reason"], compared, ctx["prompt_token_ids"]),
    )


def test_the_disagg_request_id_and_the_handoff_reach_both_engines(chain):
    # NT-017, and NT-018 end to end: the decode call carries the session the prefill call sent.
    disagg_request_id = DISAGG_REQUEST_ID_MIN + 202
    ctx = _context_leg(chain, disagg_request_id)
    prefill = chain.ctx.engine.requests[-1]
    assert prefill.extra[REQ_DISAGG_REQUEST_ID] == str(disagg_request_id)
    _, sent = chain.ctx.engine.handoffs[-1]
    response = _generation_leg(chain, ctx)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["text"]
    decode = chain.gen.engine.requests[-1]
    assert decode.kv.HasField("session")
    assert decode.kv.session.session_id == sent.session_id
    assert json_format.MessageToDict(decode.kv.session.attributes_struct) == (
        json_format.MessageToDict(sent.attributes_struct)
    )
    assert decode.kv.session.attributes_struct["disagg_request_id"] == str(disagg_request_id)


def test_a_request_that_ends_on_the_context_engine_is_the_whole_answer(chain):
    # NT-003: no generation leg, and the tokens held back are the answer.
    engine = chain.ctx.engine
    before = chain.snapshot()
    engine.disagg_faults.finish_prefill_early = True
    try:
        response = chain.complete(prompt=PROMPT, max_tokens=4)
    finally:
        engine.disagg_faults.finish_prefill_early = False
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["text"] and choice["finish_reason"] == "stop"
    after = chain.snapshot()
    for bucket in REQUEST_BUCKETS:
        key = SERVE_COUNTER_HTTP_RECEIVED_BUCKETS[bucket]
        assert _frontend(before, after, "generation", key) == 0
    assert _frontend(before, after, "context", SERVE_COUNTER_PREFILL_READY_RECEIVED) == 0
    _check_counters(chain, before, after)


def test_unregistered_attributes_are_passed_on_and_counted(chain):
    if chain.shape == "ds":
        pytest.skip("the ds engine refuses attributes the conventions do not name")
    engine = chain.ctx.engine
    before = chain.snapshot()
    engine.disagg_faults.extra_attributes = {"vendor_hint": "x"}
    try:
        ctx = _context_leg(chain, DISAGG_REQUEST_ID_MIN + 303)
    finally:
        engine.disagg_faults.extra_attributes = {}
    params = ctx["choices"][0]["disaggregated_params"]
    assert json.loads(params["openengine_handoff_attributes"]) == {"vendor_hint": "x"}
    after = chain.snapshot()
    assert _frontend(before, after, "context", SERVE_COUNTER_HANDOFF_UNREGISTERED_ATTRIBUTES) == 1
    assert "vendor_hint" in chain.ctx.log()
    response = _generation_leg(chain, ctx)
    assert response.status_code == 200, response.text
    decode = chain.gen.engine.requests[-1]
    assert decode.kv.session.attributes_struct["vendor_hint"] == "x"


def test_a_new_field_naming_a_mapped_attribute_is_refused(chain):
    # NT-019: 400 from the generation worker, and no Generate call.
    ctx = _context_leg(chain, DISAGG_REQUEST_ID_MIN + 404)
    before = chain.snapshot()
    response = _generation_leg(
        chain, ctx, openengine_handoff_attributes=json.dumps({"first_gen_tokens": [1]})
    )
    assert response.status_code == 400, response.text
    assert "first_gen_tokens" in response.text
    after = chain.snapshot()
    assert _frontend(before, after, "generation", SERVE_COUNTER_HTTP_REJECTED) == 1
    assert _frontend(before, after, "generation", SERVE_COUNTER_GENERATE_SENT) == 0
    _check_counters(chain, before, after)


def test_an_aggregated_request_to_a_worker_counts_as_aggregated(chain):
    for worker in chain.workers:
        before = chain.snapshot()
        response = chain.complete(url=worker.url)
        assert response.status_code == 200, response.text
        after = chain.snapshot()
        key = SERVE_COUNTER_HTTP_RECEIVED_BUCKETS[BUCKET_AGGREGATED]
        assert _frontend(before, after, worker.role, key) == 1
        _check_counters(chain, before, after)


def test_the_pool_does_not_change_the_outputs(chain):
    # Once both chains of a shape have run, their outputs must be the same.
    compared = 0
    for outputs in _OUTPUTS.values():
        by_shape: dict[str, list] = {}
        for shape, output in outputs.values():
            by_shape.setdefault(shape, []).append(output)
        for shape_outputs in by_shape.values():
            if len(shape_outputs) == 2:
                assert shape_outputs[0] == shape_outputs[1]
                compared += 1
    if not compared:
        pytest.skip("the chain of the other pool size has not run yet")
