# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-level features of the OpenEngine Generate adapter.

Covers the ``extra`` request keys (thinking budget, logit_bias, kv_hints,
scheduling), ``openengine-priority``, the ``num_sequences`` limit, admission of
values that would break sampling, and how the engine decides what it accepts.
"""

import dataclasses
import json
import math
import re
from types import SimpleNamespace
from typing import Any

import pytest

grpc = pytest.importorskip(  # noqa: E402
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.73.1,<2")'
)
pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")
import torch  # noqa: E402
from conftest import FakeServicerContext  # noqa: E402

from tensorrt_llm.grpc.openengine.bindings import generation_pb2  # noqa: E402
from tensorrt_llm.grpc.openengine.errors import UnsupportedFeatureError  # noqa: E402
from tensorrt_llm.grpc.openengine.request_mapping import (  # noqa: E402
    PORTABLE_REQUEST_CAPABILITIES,
    RequestCapabilities,
    _metadata_from_context,
    accepted_sampling_values,
    extra_options_from_request,
    max_num_sequences_for,
    request_capabilities_for_model,
    sampling_params_from_request,
)
from tensorrt_llm.kv_hints import KvHintsEnvelope  # noqa: E402
from tensorrt_llm.kv_hints.protocol import admit_kv_hints  # noqa: E402
from tensorrt_llm.llmapi.llm_args import (  # noqa: E402
    AdvancedSamplingMode,
    AutoDecodingConfig,
    DecodingBaseConfig,
    Eagle3DecodingConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
    WaitingQueuePolicy,
)
from tensorrt_llm.llmapi.thinking_budget import ThinkingBudgetLogitsProcessor  # noqa: E402

pytestmark = pytest.mark.cpu_only

VOCAB = 1000
FULL = RequestCapabilities(
    max_num_sequences=4,
    supports_priority=True,
    vocab_size=VOCAB,
    supports_logit_bias=True,
    supports_thinking_token_budget=True,
    supports_kv_hints=True,
    supports_scheduling=True,
    lora_ckpt_source="hf",
    max_loras=2,
    supports_logprobs=True,
)
EXTRA_KEYS = (
    "trtllm.thinking_token_budget",
    "trtllm.logit_bias",
    "trtllm.kv_hints",
    "trtllm.scheduling",
)
VALID_EXTRA = {
    "trtllm.thinking_token_budget": {
        "budget": 8,
        "start_token_ids": [10],
        "end_token_ids": [11, 12],
    },
    "trtllm.logit_bias": {"5": 100, "7": -100},
    "trtllm.kv_hints": {
        "protocol_version": "0.1",
        "message_id": "m-1",
        "actions": [
            {
                "action_id": "a-1",
                "action_type": "kv.fetch",
                "action_version": "1",
                "payload": {"blocks": 3},
            }
        ],
    },
    "trtllm.scheduling": {"attention_dp_relax": False, "agent_hierarchy": [["planner", 1]]},
}


def _request(extra: dict[str, Any] | None = None, **fields: Any) -> generation_pb2.GenerateRequest:
    request = generation_pb2.GenerateRequest(request_id="r", model="m", prompt="hi", **fields)
    if extra is not None:
        request.extra.update(extra)
    return request


def _map(request, capabilities=FULL):
    """Map the whole request the way Generate does."""
    return (
        sampling_params_from_request(request, None, capabilities),
        extra_options_from_request(request, capabilities),
    )


# -- Valid values take effect ---------------------------------------------------


def test_thinking_budget_attaches_the_logits_processor():
    params, _ = _map(_request({"trtllm.thinking_token_budget": VALID_EXTRA[EXTRA_KEYS[0]]}))

    processor = params.logits_processor
    assert isinstance(processor, ThinkingBudgetLogitsProcessor)
    assert processor.thinking_token_budget == 8
    assert processor.reasoning_start_token_ids == [10]
    assert processor.reasoning_end_token_ids == [11, 12]


def test_zero_thinking_budget_is_accepted():
    params, _ = _map(
        _request(
            {
                "trtllm.thinking_token_budget": {
                    "budget": 0,
                    "start_token_ids": [1],
                    "end_token_ids": [2],
                }
            }
        )
    )

    assert params.logits_processor.thinking_token_budget == 0


def test_logit_bias_becomes_a_vocabulary_wide_embedding_bias():
    params, _ = _map(_request({"trtllm.logit_bias": {"5": 100, "7": -2.5}}))

    bias = params.embedding_bias
    assert bias.dtype == torch.float32 and bias.shape == (VOCAB,)
    assert bias[5].item() == 100 and bias[7].item() == -2.5
    assert torch.count_nonzero(bias).item() == 2


def test_empty_logit_bias_sets_no_embedding_bias():
    params, _ = _map(_request({"trtllm.logit_bias": {}}))

    assert params.embedding_bias is None


def test_kv_hints_are_admitted_with_integral_numbers_as_ints():
    _, options = _map(_request({"trtllm.kv_hints": VALID_EXTRA["trtllm.kv_hints"]}))

    assert isinstance(options.kv_hints, KvHintsEnvelope)
    assert options.kv_hints.message_id == "m-1"
    payload = options.kv_hints.actions[0].payload
    assert payload == {"blocks": 3} and type(payload["blocks"]) is int


def _envelope(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": "0.1",
        "message_id": "m-1",
        "actions": [
            {
                "action_id": "a-1",
                "action_type": "kv.fetch",
                "action_version": "1",
                "payload": payload,
            }
        ],
    }


def test_kv_hints_json_text_is_admitted_exactly_as_over_http():
    """The JSON text keeps integers beyond 2**53 and the type of every number."""
    envelope = _envelope(
        {
            "block_hashes": [2**64 - 1, 2**53 + 1, -(2**63), 11],
            "ratio": 2.0,
            "label": "123",
        }
    )
    text = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    _, options = _map(_request({"trtllm.kv_hints": text}))

    payload = options.kv_hints.actions[0].payload
    assert payload["block_hashes"] == [2**64 - 1, 2**53 + 1, -(2**63), 11]
    assert all(type(value) is int for value in payload["block_hashes"])
    assert type(payload["ratio"]) is float and payload["label"] == "123"
    over_http = admit_kv_hints(json.loads(text))
    assert options.kv_hints.model_dump_json() == over_http.model_dump_json()


def test_kv_hints_json_text_may_contain_insignificant_whitespace():
    text = json.dumps(_envelope({"blocks": 3}), indent=4)

    _, options = _map(_request({"trtllm.kv_hints": text}))

    assert options.kv_hints.actions[0].payload == {"blocks": 3}


def test_kv_hints_object_holds_numbers_as_doubles():
    """The object form is still accepted; a Struct cannot tell 2.0 from 2."""
    _, options = _map(
        _request({"trtllm.kv_hints": _envelope({"ratio": 2.0, "largest": float(2**53 - 1)})})
    )

    payload = options.kv_hints.actions[0].payload
    assert payload["ratio"] == 2 and type(payload["ratio"]) is int
    assert payload["largest"] == 2**53 - 1


@pytest.mark.parametrize("number", [float(2**53), float(-(2**63)), 1.5e19])
def test_kv_hints_object_rejects_numbers_a_double_may_have_rounded(number):
    """2**53 + 1 arrives as the double 2**53, so the object form cannot carry it."""
    with pytest.raises(ValueError, match="JSON text"):
        _map(_request({"trtllm.kv_hints": _envelope({"block_hashes": [number]})}))


def test_scheduling_carries_relax_and_agent_hierarchy():
    _, options = _map(_request({"trtllm.scheduling": VALID_EXTRA["trtllm.scheduling"]}))

    assert options.attention_dp_relax is False
    assert options.agent_hierarchy == [("planner", 1)]


@pytest.mark.parametrize("value", [{}, {"agent_hierarchy": []}])
def test_scheduling_fields_may_be_omitted(value):
    _, options = _map(_request({"trtllm.scheduling": value}))

    assert options.attention_dp_relax is None and options.agent_hierarchy is None


def test_requests_without_extra_keys_are_unchanged():
    params, options = _map(_request())

    assert params.logits_processor is None and params.embedding_bias is None
    assert options.kv_hints is None and options.agent_hierarchy is None


def test_keys_without_the_trtllm_prefix_are_left_to_their_own_readers():
    """Only `trtllm.*` keys are this convention's; others, like request_type, are read elsewhere."""
    _map(_request({"request_type": "context_only"}), PORTABLE_REQUEST_CAPABILITIES)


# -- Wrong types and values are INVALID_ARGUMENT (ValueError, not UNIMPLEMENTED) --

INVALID_VALUES = [
    ("trtllm.thinking_token_budget", 8),
    ("trtllm.thinking_token_budget", {"start_token_ids": [1], "end_token_ids": [2]}),
    ("trtllm.thinking_token_budget", {"budget": -1, "start_token_ids": [1], "end_token_ids": [2]}),
    ("trtllm.thinking_token_budget", {"budget": 1.5, "start_token_ids": [1], "end_token_ids": [2]}),
    ("trtllm.thinking_token_budget", {"budget": "8", "start_token_ids": [1], "end_token_ids": [2]}),
    ("trtllm.thinking_token_budget", {"budget": 8, "start_token_ids": [], "end_token_ids": [2]}),
    ("trtllm.thinking_token_budget", {"budget": 8, "start_token_ids": [1]}),
    ("trtllm.thinking_token_budget", {"budget": 8, "start_token_ids": 1, "end_token_ids": [2]}),
    (
        "trtllm.thinking_token_budget",
        {"budget": 8, "start_token_ids": [1], "end_token_ids": [VOCAB]},
    ),
    ("trtllm.thinking_token_budget", {"budget": 8, "start_token_ids": [-1], "end_token_ids": [2]}),
    (
        "trtllm.thinking_token_budget",
        {"budget": 8, "start_token_ids": [1] * 33, "end_token_ids": [2]},
    ),
    (
        "trtllm.thinking_token_budget",
        {"budget": 8, "start_token_ids": [1], "end_token_ids": [2], "extra": 1},
    ),
    ("trtllm.logit_bias", [1, 2]),
    ("trtllm.logit_bias", {"abc": 1}),
    ("trtllm.logit_bias", {"-1": 1}),
    ("trtllm.logit_bias", {"1.0": 1}),
    # "007" and "7" would name the same token.
    ("trtllm.logit_bias", {"007": 1}),
    ("trtllm.logit_bias", {str(VOCAB): 1}),
    ("trtllm.logit_bias", {"5": "1"}),
    ("trtllm.logit_bias", {"5": 100.5}),
    ("trtllm.logit_bias", {"5": -101}),
    ("trtllm.logit_bias", {"5": math.inf}),
    ("trtllm.logit_bias", {str(i): 1 for i in range(301)}),
    ("trtllm.kv_hints", "hints"),
    ("trtllm.kv_hints", 3),
    ("trtllm.kv_hints", "[]"),
    ("trtllm.kv_hints", '{"protocol_version": "0.1"'),
    ("trtllm.kv_hints", json.dumps(_envelope({"x": 1})).replace("1}", "NaN}")),
    ("trtllm.kv_hints", json.dumps(_envelope({"x": 1})).replace("1}", "1e400}")),
    # Outside the -2**63..2**64-1 range that HTTP admission accepts.
    ("trtllm.kv_hints", json.dumps(_envelope({"block_hashes": [2**64]}))),
    # More digits than the interpreter converts to an int.
    pytest.param(
        "trtllm.kv_hints",
        json.dumps(_envelope({"x": 1})).replace("1}", "9" * 5000 + "}"),
        id="trtllm.kv_hints-long-integer",
    ),
    pytest.param(
        "trtllm.kv_hints",
        json.dumps(_envelope({"x": "y" * (64 * 1024)})),
        id="trtllm.kv_hints-oversized",
    ),
    pytest.param(
        "trtllm.kv_hints",
        " " * (4 * 64 * 1024) + json.dumps(_envelope({})),
        id="trtllm.kv_hints-long-text",
    ),
    pytest.param(
        "trtllm.kv_hints",
        json.dumps(_envelope({"x": [[]]})).replace("[[]]", "[" * 100000 + "]" * 100000),
        id="trtllm.kv_hints-deep",
    ),
    ("trtllm.kv_hints", {"protocol_version": "9.9", "message_id": "m", "actions": []}),
    ("trtllm.kv_hints", {"protocol_version": "0.1", "actions": []}),
    ("trtllm.scheduling", True),
    ("trtllm.scheduling", {"attention_dp_relax": "no"}),
    ("trtllm.scheduling", {"agent_hierarchy": "planner"}),
    ("trtllm.scheduling", {"agent_hierarchy": [["planner"]]}),
    ("trtllm.scheduling", {"agent_hierarchy": [[1, 1]]}),
    ("trtllm.scheduling", {"agent_hierarchy": [["planner", 1.5]]}),
    ("trtllm.scheduling", {"agent_hierarchy": [["planner", 2**31]]}),
    # The agent-tree reorder policy has one level below its root.
    ("trtllm.scheduling", {"agent_hierarchy": [["planner", 1], ["worker", 2]]}),
    ("trtllm.scheduling", {"attention_dp_rank": 1}),
]


@pytest.mark.parametrize(("key", "value"), INVALID_VALUES)
def test_invalid_values_of_declared_keys_are_invalid_arguments(key, value):
    with pytest.raises(ValueError) as error:
        _map(_request({key: value}))

    assert not isinstance(error.value, UnsupportedFeatureError)
    assert key in str(error.value)


# -- Undeclared `trtllm.*` keys are UNIMPLEMENTED ---------------------------------


@pytest.mark.parametrize("key", EXTRA_KEYS)
def test_keys_the_engine_does_not_declare_are_unimplemented(key):
    with pytest.raises(UnsupportedFeatureError, match=key):
        _map(_request({key: VALID_EXTRA[key]}), PORTABLE_REQUEST_CAPABILITIES)


@pytest.mark.parametrize("key", ["trtllm.best_of", "trtllm.unknown"])
def test_unknown_trtllm_keys_are_unimplemented(key):
    with pytest.raises(UnsupportedFeatureError, match=key):
        _map(_request({key: 2}))


def test_logit_bias_is_unimplemented_when_the_engine_cannot_apply_it():
    capabilities = dataclasses.replace(FULL, supports_logit_bias=False)

    with pytest.raises(UnsupportedFeatureError, match="trtllm.logit_bias"):
        _map(_request({"trtllm.logit_bias": {"5": 1}}), capabilities)


# -- num_sequences limit and admission of values that break sampling -------------


def test_num_sequences_up_to_the_declared_limit_is_accepted():
    params, _ = _map(
        _request(sampling=generation_pb2.SamplingParams(temperature=1.0, num_sequences=4))
    )

    assert params.n == 4


def test_num_sequences_above_the_declared_limit_is_invalid():
    request = _request(sampling=generation_pb2.SamplingParams(temperature=1.0, num_sequences=5))

    with pytest.raises(ValueError, match="at most 4"):
        _map(request)


def test_empty_stop_string_is_invalid():
    request = _request(
        stopping=generation_pb2.StoppingOptions(
            conditions=[generation_pb2.StopCondition(stop_text="")]
        )
    )

    with pytest.raises(ValueError, match="stop_text must be non-empty"):
        _map(request)


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "top_p",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_sampling_values_are_invalid(field, value):
    request = _request(sampling=generation_pb2.SamplingParams(**{field: value}))

    with pytest.raises(ValueError, match=f"sampling.{field} must be finite"):
        _map(request)


# -- openengine-priority (OE-META-3) ----------------------------------------------


@pytest.mark.parametrize(("value", "priority"), [("0", 0.0), ("250", 0.25), ("1000", 1.0)])
def test_priority_metadata_is_scaled_to_the_engine_range(value, priority):
    context = FakeServicerContext(metadata=(("openengine-priority", value),))

    assert _metadata_from_context(context, FULL).priority == priority


@pytest.mark.parametrize("value", ["+1", " 1", "1_0", "\u0661", "-1", ""])
def test_malformed_target_dp_rank_metadata_is_invalid(value):
    context = FakeServicerContext(metadata=(("openengine-target-dp-rank", value),))

    with pytest.raises(ValueError) as error:
        _metadata_from_context(context, FULL)
    assert not isinstance(error.value, UnsupportedFeatureError)


@pytest.mark.parametrize("value", ["1001", "-1", "0.5", "high", "+5", " 5", "2_5", "\u0665", ""])
def test_malformed_priority_metadata_is_invalid(value):
    context = FakeServicerContext(metadata=(("openengine-priority", value),))

    with pytest.raises(ValueError) as error:
        _metadata_from_context(context, FULL)
    assert not isinstance(error.value, UnsupportedFeatureError)


def test_priority_metadata_is_unimplemented_without_priority_scheduling():
    context = FakeServicerContext(metadata=(("openengine-priority", "250"),))

    with pytest.raises(UnsupportedFeatureError, match="openengine-priority"):
        _metadata_from_context(context, PORTABLE_REQUEST_CAPABILITIES)


def test_missing_priority_metadata_leaves_the_default():
    assert _metadata_from_context(FakeServicerContext(), FULL).priority is None


# -- What the engine declares, decided once per engine ----------------------------


def _llm(*, vocab_size: int | None = VOCAB, **args: Any) -> SimpleNamespace:
    """The LLM attributes Control and Generate read, with engine-like defaults."""
    engine_args = SimpleNamespace(
        max_seq_len=4096,
        max_batch_size=16,
        max_num_tokens=8192,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        guided_decoding_backend=None,
        reasoning_parser=None,
        kv_cache_config=SimpleNamespace(tokens_per_block=32),
        _enable_routing_load=False,
    )
    for key, value in args.items():
        setattr(engine_args, key, value)
    llm = SimpleNamespace(
        args=engine_args,
        llm_id="instance-1",
        _hf_model_config=None if vocab_size is None else SimpleNamespace(vocab_size=vocab_size),
        tokenizer=SimpleNamespace(),
        input_processor=SimpleNamespace(),
        _executor=SimpleNamespace(is_shutdown=lambda: False),
        _check_health=lambda: True,
        generate_kwargs=None,
    )

    def generate_async(**kwargs: Any) -> "_EmptyResult":
        llm.generate_kwargs = kwargs
        kwargs["sampling_params"]._validate()
        return _EmptyResult()

    llm.generate_async = generate_async
    return llm


class _EmptyResult:
    """A result stream that ends at once; Generate reports it as an error event."""

    finished = False
    prompt_token_ids: list[int] = []
    outputs: list[Any] = []
    cached_tokens = 0
    error = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    def abort(self) -> None:
        pass


PRIORITY_SCHEDULER = SimpleNamespace(waiting_queue_policy=WaitingQueuePolicy.PRIORITY)
LORA_CONFIG = SimpleNamespace(lora_ckpt_source="hf", max_loras=3)


@pytest.mark.parametrize(
    ("scheduler_config", "gen_admission", "expected"),
    [
        (None, None, False),
        (SimpleNamespace(waiting_queue_policy=WaitingQueuePolicy.FCFS), None, False),
        (PRIORITY_SCHEDULER, None, True),
        # GEN admission ordering replaces the priority queue.
        (PRIORITY_SCHEDULER, "50", False),
        (PRIORITY_SCHEDULER, "0", True),
    ],
)
def test_priority_is_declared_only_when_the_scheduler_queues_by_priority(
    monkeypatch, scheduler_config, gen_admission, expected
):
    if gen_admission is None:
        monkeypatch.delenv("TLLM_GEN_ADMISSION_TTFT_PERCENTILE", raising=False)
    else:
        monkeypatch.setenv("TLLM_GEN_ADMISSION_TTFT_PERCENTILE", gen_admission)

    capabilities = request_capabilities_for_model(_llm(scheduler_config=scheduler_config))

    assert capabilities.supports_priority is expected


DRAFT = "/models/draft"
SPECULATIVE_CONFIGS = [
    (lambda: None, True),
    (lambda: NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2), True),
    (lambda: MTPDecodingConfig(max_draft_len=3), False),
    (lambda: Eagle3DecodingConfig(max_draft_len=3, speculative_model=DRAFT), False),
    (lambda: AutoDecodingConfig(), False),
]


@pytest.mark.parametrize(("make_config", "supported"), SPECULATIVE_CONFIGS)
def test_speculation_limits_the_request_keys(make_config, supported):
    """The one-model speculative sampler rejects every request with an embedding_bias or logprobs."""
    speculative_config = make_config()
    args = _llm(speculative_config=speculative_config).args
    capabilities = request_capabilities_for_model(
        SimpleNamespace(args=args, _hf_model_config=SimpleNamespace(vocab_size=VOCAB))
    )

    assert capabilities.supports_logit_bias is supported
    assert capabilities.supports_logprobs is supported
    # The thinking budget is declared only without speculative decoding.
    assert capabilities.supports_thinking_token_budget is (speculative_config is None)
    # The same one-model test that limits min_p and the penalties to their no-effect values.
    assert (capabilities.sampling["min_p"].accept == "any") is supported


NO_EFFECT_VALUES = {
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
# A value of each sampling field that changes sampling.
EFFECTIVE_VALUES = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 5,
    "min_p": 0.1,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "repetition_penalty": 1.2,
    "seed": 7,
}


def _speculative_capabilities():
    return request_capabilities_for_model(
        _llm(speculative_config=MTPDecodingConfig(max_draft_len=3))
    )


def test_one_model_speculation_accepts_only_the_no_effect_values():
    capabilities = _speculative_capabilities()
    for name, value in EFFECTIVE_VALUES.items():
        if name in NO_EFFECT_VALUES:
            accepted = capabilities.sampling[name]
            assert (accepted.accept, accepted.values) == ("values", (NO_EFFECT_VALUES[name],))
        else:
            assert capabilities.sampling[name].accept == "any", name


@pytest.mark.parametrize("name", sorted(EFFECTIVE_VALUES))
def test_generate_rejects_a_sampling_value_outside_the_declared_set(name):
    """Declaration and check come from one table (CAP-GEN-3): an undeclared value is UNIMPLEMENTED."""
    capabilities = _speculative_capabilities()
    request = _request(sampling=generation_pb2.SamplingParams(**{name: EFFECTIVE_VALUES[name]}))
    if name in NO_EFFECT_VALUES:
        with pytest.raises(
            UnsupportedFeatureError, match=re.escape(f"sampling.{name}=") + ".*accepts only"
        ):
            sampling_params_from_request(request, None, capabilities)
        no_effect = _request(
            sampling=generation_pb2.SamplingParams(**{name: NO_EFFECT_VALUES[name]})
        )
        params = sampling_params_from_request(no_effect, None, capabilities)
        assert getattr(params, name) == NO_EFFECT_VALUES[name]
    else:
        params = sampling_params_from_request(request, None, capabilities)
        assert getattr(params, name) == pytest.approx(EFFECTIVE_VALUES[name])
    # Without speculation every value is accepted.
    sampling_params_from_request(request, None, request_capabilities_for_model(_llm()))


# A request for each kind of log probabilities, keyed by its ResponseOptions flag.
LOGPROB_REQUESTS = {
    "return_prompt_logprobs": generation_pb2.ResponseOptions(
        return_prompt_logprobs=True,
        prompt_candidates=generation_pb2.CandidateTokenSelection(top_n=2),
    ),
    "return_output_logprobs": generation_pb2.ResponseOptions(
        return_output_logprobs=True,
        output_candidates=generation_pb2.CandidateTokenSelection(top_n=2),
    ),
}
LOGPROB_PARAMS = {
    "return_prompt_logprobs": "prompt_logprobs",
    "return_output_logprobs": "logprobs",
}


@pytest.mark.parametrize("name", sorted(LOGPROB_REQUESTS))
def test_logprobs_are_unimplemented_under_one_model_speculation(name):
    request = _request(response=LOGPROB_REQUESTS[name])

    with pytest.raises(UnsupportedFeatureError, match=re.escape(f"response.{name} is not")):
        sampling_params_from_request(request, None, _speculative_capabilities())
    # Asking for none is accepted.
    not_requested = _request(response=generation_pb2.ResponseOptions(**{name: False}))
    sampling_params_from_request(not_requested, None, _speculative_capabilities())
    # Without speculation the request is accepted.
    params = sampling_params_from_request(request, None, request_capabilities_for_model(_llm()))
    assert getattr(params, LOGPROB_PARAMS[name]) == 2


def test_a_sampling_field_missing_from_the_declaration_accepts_nothing():
    capabilities = dataclasses.replace(FULL, sampling={"temperature": FULL.sampling["temperature"]})
    sampling_params_from_request(
        _request(sampling=generation_pb2.SamplingParams(temperature=0.5)), None, capabilities
    )
    with pytest.raises(UnsupportedFeatureError, match="sampling.top_k=5 .*accepts no value"):
        sampling_params_from_request(
            _request(sampling=generation_pb2.SamplingParams(top_k=5)), None, capabilities
        )


def test_the_declaration_reports_the_version_2_keys_from_the_same_table():
    declaration = _speculative_capabilities().sampling_declaration()
    assert declaration["trtllm.params"]["repetition_penalty"]["values"] == [1.0]
    assert declaration["trtllm.params"]["temperature"]["accept"] == "any"
    assert {key: value for key, value in declaration.items() if key != "trtllm.params"} == {
        "trtllm.sampling.min_p": False,
        "trtllm.sampling.frequency_penalty": False,
        "trtllm.sampling.presence_penalty": False,
        "trtllm.sampling.repetition_penalty": False,
    }
    plain = request_capabilities_for_model(_llm()).sampling_declaration()
    assert all(plain[key] is True for key in plain if key != "trtllm.params")


@pytest.mark.parametrize(
    ("mode", "skipped"),
    [
        (AdvancedSamplingMode.FULL, set()),
        (AdvancedSamplingMode.NO_TOPK, {"top_k"}),
        (AdvancedSamplingMode.NO_TOPP, {"top_p"}),
        (AdvancedSamplingMode.NO_TOPK_NO_TOPP, {"top_k", "top_p"}),
    ],
)
def test_a_filter_the_speculative_sampler_skips_accepts_only_its_no_effect_value(mode, skipped):
    capabilities = request_capabilities_for_model(
        _llm(speculative_config=MTPDecodingConfig(max_draft_len=3, advanced_sampling_mode=mode))
    )
    no_effect = {"top_k": 0, "top_p": 1.0}
    for name, value in no_effect.items():
        accepted = capabilities.sampling[name]
        if name in skipped:
            assert (accepted.accept, accepted.values) == ("values", (value,)), name
            with pytest.raises(UnsupportedFeatureError, match=re.escape(f"sampling.{name}=")):
                capabilities.check_sampling_value(name, 5 if name == "top_k" else 0.5)
        else:
            assert accepted.accept == "any", name
        capabilities.check_sampling_value(name, value)
    assert capabilities.sampling["temperature"].accept == "any"


def test_the_engine_arguments_the_sampling_declaration_reads_exist():
    """The declaration reads these with getattr, so pin them against the real argument classes."""
    assert {"backend", "generation_config", "speculative_config"} <= set(TorchLlmArgs.model_fields)
    assert "advanced_sampling_mode" in DecodingBaseConfig.model_fields


@pytest.mark.parametrize(
    ("backend", "generation_config"),
    [("pytorch", "auto"), ("pytorch", None), ("_autodeploy", "auto")],
)
def test_the_declared_defaults_name_the_generation_config_when_the_llm_applies_it(
    backend, generation_config
):
    accepted = accepted_sampling_values(
        SimpleNamespace(backend=backend, generation_config=generation_config)
    )
    from_config = {"temperature", "top_p", "top_k", "min_p", "repetition_penalty"}
    for name, values in accepted.items():
        if (backend, generation_config) == ("pytorch", "auto") and name in from_config:
            assert values.default == "generation_config", name
        else:
            assert values.default != "generation_config", name


def test_vocab_size_comes_from_the_model_config():
    llm = _llm(vocab_size=VOCAB + 24)
    # The model's logits width, not the tokenizer's vocabulary.
    llm.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=50))
    assert request_capabilities_for_model(llm).vocab_size == VOCAB + 24

    llm._hf_model_config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=VOCAB))
    assert request_capabilities_for_model(llm).vocab_size == VOCAB

    llm._hf_model_config = None
    assert request_capabilities_for_model(llm).vocab_size is None


def test_logit_bias_support_needs_a_vocabulary_size():
    with pytest.raises(ValueError, match="vocab_size"):
        dataclasses.replace(PORTABLE_REQUEST_CAPABILITIES, supports_logit_bias=True)


def test_token_id_keys_need_a_known_vocabulary():
    capabilities = request_capabilities_for_model(_llm(vocab_size=None))

    assert capabilities.vocab_size is None
    assert capabilities.supports_logit_bias is False
    assert capabilities.supports_thinking_token_budget is False


def test_lora_follows_the_lora_config():
    assert request_capabilities_for_model(_llm()).supports_lora is False
    capabilities = request_capabilities_for_model(
        _llm(lora_config=SimpleNamespace(lora_ckpt_source="nemo", max_loras=None))
    )
    assert capabilities.lora_ckpt_source == "nemo" and capabilities.max_loras is None
    assert request_capabilities_for_model(_llm(lora_config=LORA_CONFIG)).max_loras == 3


AGENT_TREE = SimpleNamespace(
    policy_name="AgentTree", policy_args=SimpleNamespace(agent_types=["AgentDeepResearch"])
)
EMPTY_AGENT_TREE = SimpleNamespace(
    policy_name="AgentTree", policy_args=SimpleNamespace(agent_types=[])
)
V1_KV_CACHE = SimpleNamespace(tokens_per_block=32, use_kv_cache_manager_v2=False)


AGENT_TREE_ARGS = {"reorder_policy_config": AGENT_TREE, "kv_cache_config": V1_KV_CACHE}


@pytest.mark.parametrize(
    ("args", "supported"),
    [
        ({}, False),
        # One field read is not enough: a client may send both.
        ({"enable_attention_dp": True}, False),
        (AGENT_TREE_ARGS, False),
        ({"enable_attention_dp": True, **AGENT_TREE_ARGS}, True),
        # Without agent types the policy builds no tree and reorders nothing.
        (
            {
                "enable_attention_dp": True,
                **AGENT_TREE_ARGS,
                "reorder_policy_config": EMPTY_AGENT_TREE,
            },
            False,
        ),
        # Only the C++ scheduler of a V1 KV cache manager reorders by hierarchy.
        ({"enable_attention_dp": True, "reorder_policy_config": AGENT_TREE}, False),
        (
            {
                "enable_attention_dp": True,
                **AGENT_TREE_ARGS,
                "scheduler_config": SimpleNamespace(use_python_scheduler=True),
            },
            False,
        ),
    ],
)
def test_scheduling_is_declared_only_where_both_fields_are_read(args, supported):
    capabilities = request_capabilities_for_model(_llm(**args))

    assert capabilities.supports_scheduling is supported
    assert capabilities.extra_capabilities()["trtllm.request.scheduling"] is supported


def test_the_llm_args_fields_capabilities_read_exist():
    """The capabilities read these by name; a rename would silently change what is declared."""
    from tensorrt_llm.llmapi.llm_args import (
        AgentTreeConfig,
        KvCacheConfig,
        LoraConfig,
        ReorderRequestPolicyConfig,
        SchedulerConfig,
        TorchLlmArgs,
    )

    for model, fields in (
        (
            TorchLlmArgs,
            (
                "max_batch_size",
                "pipeline_parallel_size",
                "scheduler_config",
                "speculative_config",
                "lora_config",
                "num_serve_frontends",
                "enable_attention_dp",
                "reorder_policy_config",
                "kv_cache_config",
            ),
        ),
        (SchedulerConfig, ("waiting_queue_policy", "use_python_scheduler")),
        (KvCacheConfig, ("use_kv_cache_manager_v2",)),
        (LoraConfig, ("lora_ckpt_source", "max_loras")),
        (ReorderRequestPolicyConfig, ("policy_args",)),
        (AgentTreeConfig, ("agent_types",)),
    ):
        assert set(fields) <= set(model.model_fields), (
            model.__name__,
            set(fields) - set(model.model_fields),
        )


def test_lora_is_unsupported_with_several_frontends():
    """Each frontend would keep its own registry, so adapter ids could collide."""
    capabilities = request_capabilities_for_model(
        _llm(lora_config=LORA_CONFIG, num_serve_frontends=2)
    )

    assert capabilities.supports_lora is False
    assert capabilities.lora_ckpt_source is None and capabilities.max_loras is None


# -- Control reports what Generate enforces ----------------------------------------


def _control(llm):
    from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer
    from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer

    return OpenEngineControlServicer(llm, "m", OpenEngineInferenceServicer(llm, "m"))


def _run(coroutine):
    import asyncio

    return asyncio.run(coroutine)


@pytest.mark.parametrize(
    ("batch_size", "stages", "expected"),
    [(16, 1, 16), (4, 1, 4), (4, 2, 8), (32, 1, 16), (None, 1, 16)],
)
def test_num_sequences_limit_matches_model_info(batch_size, stages, expected):
    from tensorrt_llm.grpc.openengine.bindings import model_pb2

    llm = _llm(max_batch_size=batch_size, pipeline_parallel_size=stages)
    info = _run(_control(llm).GetModelInfo(model_pb2.GetModelInfoRequest(), FakeServicerContext()))

    assert max_num_sequences_for(llm.args) == expected
    assert info.generation.max_num_sequences == expected


@pytest.mark.parametrize("enabled", [False, True])
def test_model_info_and_server_info_report_the_request_capabilities(monkeypatch, enabled):
    from tensorrt_llm.grpc.openengine.bindings import model_pb2, server_pb2

    monkeypatch.delenv("TLLM_GEN_ADMISSION_TTFT_PERCENTILE", raising=False)
    llm = (
        _llm(
            scheduler_config=PRIORITY_SCHEDULER,
            lora_config=LORA_CONFIG,
            enable_attention_dp=True,
            **AGENT_TREE_ARGS,
        )
        if enabled
        else _llm(vocab_size=None)
    )
    control = _control(llm)
    info = _run(control.GetModelInfo(model_pb2.GetModelInfoRequest(), FakeServicerContext()))
    server = _run(control.GetServerInfo(server_pb2.GetServerInfoRequest(), FakeServicerContext()))

    assert info.generation.supports_priority is enabled
    assert info.supports_lora is enabled
    assert info.extra["trtllm.request.logit_bias"] is enabled
    assert info.extra["trtllm.request.thinking_token_budget"] is enabled
    assert info.extra["trtllm.request.scheduling"] is enabled
    # Generate reads kv_hints on every engine.
    assert info.extra["trtllm.request.kv_hints"] is True
    assert info.extra["trtllm.request.best_of"] is False
    assert server.capacity.HasField("max_loras") is enabled
    if enabled:
        assert server.capacity.max_loras == 3


# -- Generate applies the options --------------------------------------------------


async def _drain(servicer, request, context):
    async for _ in servicer.Generate(request, context):
        pass


def _generate(llm, request, metadata=()):
    """Run Generate; returns the status it aborted with, or None."""
    from conftest import AbortError

    from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer

    servicer = OpenEngineInferenceServicer(llm, "m")
    context = FakeServicerContext(metadata=metadata)
    try:
        _run(_drain(servicer, request, context))
    except AbortError:
        pass
    return servicer, context.abort_code


def test_generate_forwards_every_request_option(monkeypatch, tmp_path):
    from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer

    monkeypatch.delenv("TLLM_GEN_ADMISSION_TTFT_PERCENTILE", raising=False)
    llm = _llm(
        scheduler_config=PRIORITY_SCHEDULER,
        lora_config=LORA_CONFIG,
        enable_attention_dp=True,
        reorder_policy_config=AGENT_TREE,
        kv_cache_config=V1_KV_CACHE,
    )
    servicer = OpenEngineInferenceServicer(llm, "m")
    servicer.lora_registry.register("adapter-a", 7, str(tmp_path))
    request = _request(VALID_EXTRA, lora_name="adapter-a")
    context = FakeServicerContext(metadata=(("openengine-priority", "250"),))

    _run(_drain(servicer, request, context))

    kwargs = llm.generate_kwargs
    assert kwargs["priority"] == 0.25
    assert kwargs["kv_hints"].message_id == "m-1"
    lora = kwargs["lora_request"]
    assert (lora.lora_name, lora.lora_int_id, lora.lora_path) == ("adapter-a", 7, str(tmp_path))
    assert lora.lora_ckpt_source == "hf"
    scheduling = kwargs["scheduling_params"]
    assert scheduling.attention_dp_relax is False
    assert scheduling.agent_hierarchy == [("planner", 1)]
    params = kwargs["sampling_params"]
    assert isinstance(params.logits_processor, ThinkingBudgetLogitsProcessor)
    assert params.embedding_bias[5].item() == 100


@pytest.mark.parametrize("name", sorted(LOGPROB_REQUESTS))
def test_generate_rejects_logprobs_it_does_not_declare(name):
    """Under one-model speculation GetModelInfo declares no logprobs and Generate refuses them."""
    from tensorrt_llm.grpc.openengine.bindings import model_pb2

    llm = _llm(speculative_config=MTPDecodingConfig(max_draft_len=3))
    info = _run(_control(llm).GetModelInfo(model_pb2.GetModelInfoRequest(), FakeServicerContext()))

    assert info.generation.output_logprobs.supported is False
    assert info.generation.prompt_logprobs.supported is False
    servicer, code = _generate(llm, _request(response=LOGPROB_REQUESTS[name]))
    assert code == grpc.StatusCode.UNIMPLEMENTED
    assert llm.generate_kwargs is None
    assert servicer.generate_counters.accepted == 0


def test_generate_defaults_without_options():
    from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY

    llm = _llm()
    _, code = _generate(llm, _request())

    assert code is None
    kwargs = llm.generate_kwargs
    assert kwargs["priority"] == DEFAULT_REQUEST_PRIORITY
    assert kwargs["kv_hints"] is None and kwargs["lora_request"] is None
    assert kwargs["scheduling_params"] is None


@pytest.mark.parametrize(
    ("llm_args", "request_fields", "metadata", "expected"),
    [
        ({}, {"extra": {"trtllm.kv_hints": "x"}}, (), grpc.StatusCode.INVALID_ARGUMENT),
        ({}, {"extra": {"trtllm.best_of": 2}}, (), grpc.StatusCode.UNIMPLEMENTED),
        ({}, {"lora_name": "adapter-a"}, (), grpc.StatusCode.UNIMPLEMENTED),
        (
            {"lora_config": LORA_CONFIG},
            {"lora_name": "adapter-a"},
            (),
            grpc.StatusCode.FAILED_PRECONDITION,
        ),
        ({}, {}, (("openengine-priority", "250"),), grpc.StatusCode.UNIMPLEMENTED),
        (
            {"scheduler_config": PRIORITY_SCHEDULER},
            {},
            (("openengine-priority", "2000"),),
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
        (
            {"max_batch_size": 2},
            {"sampling": generation_pb2.SamplingParams(temperature=1.0, num_sequences=3)},
            (),
            grpc.StatusCode.INVALID_ARGUMENT,
        ),
    ],
)
def test_generate_rejects_before_acceptance(
    monkeypatch, llm_args, request_fields, metadata, expected
):
    monkeypatch.delenv("TLLM_GEN_ADMISSION_TTFT_PERCENTILE", raising=False)
    llm = _llm(**llm_args)
    extra = request_fields.pop("extra", None)

    servicer, code = _generate(llm, _request(extra, **request_fields), metadata)

    assert code == expected
    assert llm.generate_kwargs is None
    assert servicer.generate_counters.accepted == 0
