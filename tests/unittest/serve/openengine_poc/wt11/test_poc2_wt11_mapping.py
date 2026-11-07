# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-by-row tests of the OpenEngine request mapping (SV-MAP)."""

import dataclasses
import json
import math
from types import SimpleNamespace

import pytest
import torch
from pydantic import BaseModel

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.kv_hints import KvHintsEnvelope
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.llmapi.thinking_budget import (
    ThinkingBudgetLogitsProcessor,
    add_thinking_budget_logits_processor,
)
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, CompletionRequest
from tensorrt_llm.serve.openengine_poc import contracts, mapping
from tensorrt_llm.serve.openengine_poc.engine_client import GuidedSpec, MediaSpec

__extra_import_path__ = [".."]
from fakes import glm_like, trtllm_like  # noqa: E402

pytestmark = pytest.mark.cpu_only

PROMPT = [2, 3, 4]


class CharTokenizer:
    """Encodes text as its code points; enough for the reasoning boundary lookup."""

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def _caps(preset="trtllm", **changes):
    caps = {"trtllm": trtllm_like, "glm": glm_like}[preset]()[1]
    return dataclasses.replace(caps, **changes)


def _declare(caps, *keys, drop=()):
    declared = (caps.extra.declared | set(keys)) - set(drop)
    return dataclasses.replace(caps, extra=dataclasses.replace(caps.extra, declared=declared))


def _map(params=None, caps=None, options=None, prompt=PROMPT):
    return mapping.map_request(
        list(prompt),
        params or SamplingParams(),
        caps or _caps(),
        options or contracts.GenerateOptions(),
        engine_name="eng",
    )


def _one(*args, **kwargs):
    (request,) = _map(*args, **kwargs)
    return request


def _sampled(**kwargs):
    """SamplingParams that sample: greedy decoding refuses n > 1 and best_of > 1."""
    return SamplingParams(temperature=1.0, **kwargs)


def _explicit(params, *names):
    params._set_request_provided_fields(names)
    return params


# --- the slot -------------------------------------------------------------------


def test_the_slot_resolves_to_this_module():
    contracts.reset_slot_cache()
    assert contracts.resolve_slot("map_request")[1] == f"{mapping.__name__}:map_request"
    request = _one()
    assert (
        contracts.map_request(
            PROMPT, SamplingParams(), _caps(), contracts.GenerateOptions(), engine_name="eng"
        )[0].sampling
        == request.sampling
    )


def test_every_call_has_a_fresh_request_id_the_model_and_detokenize_false():
    a, b = _one(), _one()
    assert a.request_id.startswith("trtllm-") and a.request_id != b.request_id
    assert a.model == "fake-model"
    assert a.extra == {conv.REQ_DETOKENIZE: False}
    assert a.prompt_token_ids == PROMPT


# --- sampling (SV-MAP-2, SV-MAP-3) ------------------------------------------------


def test_set_sampling_fields_are_sent():
    params = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        top_k=5,
        seed=7,
        min_p=0.1,
        repetition_penalty=1.2,
        presence_penalty=0.3,
        frequency_penalty=0.4,
    )
    assert _one(params).sampling == {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 5,
        "seed": 7,
        "min_p": 0.1,
        "repetition_penalty": 1.2,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.4,
    }


def test_neutral_values_are_not_sent_unless_explicit():
    params = SamplingParams(
        top_k=0, min_p=0.0, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0
    )
    assert _one(params).sampling == {}
    explicit = _explicit(
        SamplingParams(top_k=0, min_p=0.0, repetition_penalty=1.0),
        "top_k",
        "min_p",
        "repetition_penalty",
    )
    assert _one(explicit).sampling == {"top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0}
    # SamplingParams does not record whether the request set these two.
    penalties = _explicit(
        SamplingParams(frequency_penalty=0.0, presence_penalty=0.0),
        "frequency_penalty",
        "presence_penalty",
    )
    assert _one(penalties, _caps("glm")).sampling == {}


def test_values_the_server_filled_in_are_not_sent():
    # OpenAIServer fills temperature and top_p with 1.0 when the request has none.
    params = _explicit(SamplingParams(temperature=1.0, top_p=1.0, top_k=3), "top_k")
    assert _one(params).sampling == {"top_k": 3}


def test_an_explicit_neutral_gated_field_is_rejected_by_an_engine_without_it():
    params = _explicit(SamplingParams(repetition_penalty=1.0), "repetition_penalty")
    with pytest.raises(ValueError, match=r"repetition_penalty=1.0 .*'eng'"):
        _one(params, _caps("glm"))


@pytest.mark.parametrize("name", sorted(conv.SAMPLING_FIELD_CAPABILITY))
def test_a_gated_field_needs_its_capability(name):
    params = SamplingParams(**{name: 0.5})
    caps = _declare(_caps(), drop=[conv.SAMPLING_FIELD_CAPABILITY[name]])
    with pytest.raises(ValueError, match=name):
        _one(params, caps)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_non_finite_sampling_values_are_rejected(value):
    params = SamplingParams()
    params.temperature = value
    with pytest.raises(ValueError, match="finite"):
        _one(params)


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_a_seed_outside_uint64_is_rejected(seed):
    params = SamplingParams()
    params.seed = seed
    with pytest.raises(ValueError, match="seed"):
        _one(params)


def test_seed_scope_does_not_stop_a_seed():
    caps = dataclasses.replace(_caps(), extra=dataclasses.replace(_caps().extra, seed_scope="none"))
    assert _one(SamplingParams(seed=3), caps).sampling == {"seed": 3}


# --- n (SV-MAP-6) -----------------------------------------------------------------


def test_n_within_the_limit_is_one_call():
    (request,) = _map(_sampled(n=3, seed=5))
    assert request.num_sequences == 3 and request.sampling["seed"] == 5


@pytest.mark.parametrize(
    "n, limit, sizes", [(17, 16, [16, 1]), (32, 16, [16, 16]), (3, 1, [1, 1, 1]), (5, 2, [2, 2, 1])]
)
def test_n_beyond_the_limit_is_split(n, limit, sizes):
    caps = _caps(max_num_sequences=limit)
    requests = _map(_sampled(n=n, seed=9, prompt_logprobs=1), caps)
    assert [r.num_sequences for r in requests] == sizes
    assert len(requests) == math.ceil(n / limit)
    starts = [sum(sizes[:k]) for k in range(len(sizes))]
    assert [r.sampling["seed"] for r in requests] == [9 + start for start in starts]
    assert [r.prompt_logprobs_top_n for r in requests] == [1] + [None] * (len(sizes) - 1)
    assert len({r.request_id for r in requests}) == len(sizes)


@pytest.mark.parametrize("limit, seeds", [(3, [0, 3, 6]), (1, [0, 1, 2, 3, 4, 5, 6])])
def test_a_split_request_without_a_seed_gets_the_in_process_seeds(limit, seeds):
    # In-process, sequence i of an unseeded request gets seed 0 + i.
    requests = _map(_sampled(n=7), _caps(max_num_sequences=limit))
    assert [r.sampling["seed"] for r in requests] == seeds
    # One call is left to the engine, which seeds it as in-process.
    assert "seed" not in _one(_sampled(n=3), _caps(max_num_sequences=3)).sampling


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("top_k", 1 << 31, "top_k must be in"),
        ("top_k", 1 << 1100, "top_k must be in"),
        ("max_tokens", 1 << 32, "max_tokens must be less than 2"),
        ("min_tokens", 1 << 32, "min_tokens must be in"),
        ("stop_token_ids", [1 << 32], "stop token id"),
        ("stop_token_ids", [-1], "stop token id"),
    ],
)
def test_integers_outside_the_request_field_ranges_are_rejected(name, value, message):
    params = SamplingParams()
    setattr(params, name, value)
    params = _explicit(params, name)
    caps = _caps(tokenizer_fingerprint=None, max_context_length=None, max_output_tokens=None)
    with pytest.raises(ValueError, match=message):
        _map(params, caps)


def test_max_tokens_beyond_the_field_range_is_capped_when_the_engine_has_a_limit():
    caps = _caps(max_context_length=None, max_output_tokens=100)
    assert _one(SamplingParams(max_tokens=1 << 32), caps).max_tokens == 100


def test_stop_token_ids_are_not_checked_against_the_vocabulary():
    # As in-process; the vocabulary the tokenizer fingerprint reports is not the model's.
    caps = _caps(tokenizer_fingerprint=conv.TokenizerFingerprint(10, (1,), None))
    assert _one(SamplingParams(stop_token_ids=[10]), caps).stop_token_ids == [10]


def test_prompt_token_ids_beyond_the_field_range_are_rejected():
    with pytest.raises(ValueError, match=r"prompt token id 4294967296 .*\[0, 4294967296\)"):
        _one(caps=_caps(tokenizer_fingerprint=None), prompt=[1, 1 << 32])


@pytest.mark.parametrize("n", [0, -1])
def test_n_below_one_is_rejected(n):
    params = _sampled()
    params.n = n
    with pytest.raises(ValueError, match="n must be at least 1"):
        _map(params)


def test_split_seeds_wrap_around_uint64():
    requests = _map(_sampled(n=2, seed=(1 << 64) - 1), _caps("glm"))
    assert [r.sampling["seed"] for r in requests] == [(1 << 64) - 1, 0]


def test_an_engine_without_a_limit_counts_as_one():
    assert len(_map(_sampled(n=2), _caps(max_num_sequences=0))) == 2


def test_the_sequence_limit_is_logged_once_per_engine(monkeypatch):
    warnings = []
    monkeypatch.setattr(mapping.logger, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(mapping, "_logged", set())
    for _ in range(3):
        _map(_sampled(n=2), _caps("glm"))
    mapping.log_sequence_capability("eng", _caps("glm"))
    assert len(warnings) == 1 and "at most 1 sequence" in warnings[0]
    mapping.log_sequence_capability("eng", _caps("glm", max_num_sequences=4))
    assert len(warnings) == 2


def test_best_of_needs_the_capability_and_a_single_call():
    params = _sampled(n=1, best_of=2)
    with pytest.raises(ValueError, match="best_of"):
        _map(params)
    caps = _declare(_caps(), conv.CAP_REQ_BEST_OF)
    assert _one(params, caps).extra[conv.REQ_BEST_OF] == 2
    with pytest.raises(ValueError, match="split"):
        _map(_sampled(n=2, best_of=3), dataclasses.replace(caps, max_num_sequences=1))
    with pytest.raises(ValueError, match="at most 4"):
        _map(_sampled(n=2, best_of=5), dataclasses.replace(caps, max_num_sequences=4))


# --- prompt and length ------------------------------------------------------------


def test_prompt_checks():
    caps = _caps(tokenizer_fingerprint=conv.TokenizerFingerprint(10, (1,), None))
    with pytest.raises(ValueError, match="at least one token"):
        _one(caps=caps, prompt=[])
    with pytest.raises(ValueError, match="token id 10"):
        _one(caps=caps, prompt=[1, 10])
    with pytest.raises(ValueError, match="token id -1"):
        _one(caps=caps, prompt=[-1])
    with pytest.raises(ValueError, match="maximum context length"):
        _one(caps=_caps(max_context_length=3), prompt=PROMPT)


def test_max_tokens():
    caps = _caps(max_context_length=10, max_output_tokens=4)
    assert _one(SamplingParams(max_tokens=None), caps).max_tokens == 4
    assert _one(SamplingParams(max_tokens=2), caps).max_tokens == 2
    assert _one(SamplingParams(max_tokens=50), caps).max_tokens == 4
    params = SamplingParams()
    params.max_tokens = 0
    with pytest.raises(ValueError, match="max_tokens"):
        _one(params, caps)


# --- stopping ---------------------------------------------------------------------


def test_stop_conditions_go_to_an_engine_that_matches_strings():
    request = _one(
        SamplingParams(
            stop=["a", "bc", "a"], stop_token_ids=[7, 7, 8], min_tokens=2, ignore_eos=True
        )
    )
    assert request.stop_texts == ["a", "bc"] and request.stop_token_ids == [7, 8]
    assert request.min_tokens == 2 and request.ignore_eos is True
    assert request.include_stop_in_output is None


def test_stop_strings_stay_on_the_serving_side_without_the_capability():
    caps = _declare(_caps(), drop=[conv.CAP_STOP_TEXT])
    assert _one(SamplingParams(stop="x"), caps).stop_texts == []
    params = SamplingParams(stop="x", detokenize=False)
    with pytest.raises(ValueError, match="detokeniz"):
        _one(params, caps)
    assert _one(params).stop_texts == ["x"]  # The engine matches them.


def test_an_empty_stop_string_is_rejected():
    params = SamplingParams()
    params.stop = ["a", ""]
    with pytest.raises(ValueError, match="non-empty"):
        _one(params)


def test_include_stop_str_in_output():
    params = SamplingParams(stop="x", stop_token_ids=[5], include_stop_str_in_output=True)
    assert _one(params).include_stop_in_output is True
    caps = _caps(supports_stop_in_output=False)
    with pytest.raises(ValueError, match="include_stop_str_in_output"):
        _one(params, caps)
    only_strings = _one(SamplingParams(stop="x", include_stop_str_in_output=True), caps)
    assert only_strings.stop_texts == [] and only_strings.include_stop_in_output is None
    assert _one(SamplingParams(include_stop_str_in_output=True)).include_stop_in_output is None


# --- logprobs and guided decoding ---------------------------------------------------


def test_logprobs():
    request = _one(SamplingParams(logprobs=3, prompt_logprobs=0))
    assert (request.output_logprobs_top_n, request.prompt_logprobs_top_n) == (3, 0)
    with pytest.raises(ValueError, match="more than the 20"):
        _one(SamplingParams(logprobs=21), _caps("glm"))
    with pytest.raises(ValueError, match="prompt_logprobs"):
        _one(SamplingParams(prompt_logprobs=1), _caps("glm"))
    negative = SamplingParams()
    negative.logprobs = -1
    with pytest.raises(ValueError, match="logprobs must be non-negative"):
        _one(negative)


class Answer(BaseModel):
    text: str


@pytest.mark.parametrize(
    "params, spec",
    [
        (GuidedDecodingParams(regex="a+"), GuidedSpec("regex", "a+")),
        (GuidedDecodingParams(grammar="root ::= x"), GuidedSpec("ebnf_grammar", "root ::= x")),
        (GuidedDecodingParams(json_object=True), GuidedSpec("json_object")),
        (GuidedDecodingParams(structural_tag="{}"), GuidedSpec("structural_tag", "{}")),
        (
            GuidedDecodingParams(json='{"type": "object"}'),
            GuidedSpec("json_schema", '{"type": "object"}'),
        ),
        (
            GuidedDecodingParams(json={"type": "object"}),
            GuidedSpec("json_schema", '{"type": "object"}'),
        ),
        (
            GuidedDecodingParams(json=Answer),
            GuidedSpec("json_schema", json.dumps(Answer.model_json_schema())),
        ),
        # Empty values are still a choice, as in-process.
        (GuidedDecodingParams(json={}), GuidedSpec("json_schema", "{}")),
        (GuidedDecodingParams(regex=""), GuidedSpec("regex", "")),
    ],
)
def test_guided_decoding(params, spec):
    assert _one(SamplingParams(guided_decoding=params)).guided == spec
    with pytest.raises(ValueError, match="guided_decoding"):
        _one(SamplingParams(guided_decoding=params), _caps("glm"))


# --- thinking budget (SV-MAP-5) -------------------------------------------------------


def _budget_options(parser="qwen3"):
    return contracts.GenerateOptions(reasoning_parser=parser, tokenizer=CharTokenizer())


@pytest.mark.parametrize("parser", sorted(ReasoningParserFactory._parsers))
def test_thinking_budget_ids_match_the_in_process_algorithm(parser):
    in_process = SamplingParams(thinking_token_budget=8)
    try:
        add_thinking_budget_logits_processor(
            in_process, reasoning_parser=parser, tokenizer=CharTokenizer()
        )
    except ValueError:
        with pytest.raises(ValueError):
            _one(SamplingParams(thinking_token_budget=8), options=_budget_options(parser))
        return
    processor = in_process.logits_processor
    assert isinstance(processor, ThinkingBudgetLogitsProcessor)
    request = _one(SamplingParams(thinking_token_budget=8), options=_budget_options(parser))
    assert request.extra[conv.REQ_THINKING_TOKEN_BUDGET] == {
        "budget": 8,
        "start_token_ids": processor.reasoning_start_token_ids,
        "end_token_ids": processor.reasoning_end_token_ids,
    }


@pytest.mark.parametrize("parser", sorted(ReasoningParserFactory._parsers))
@pytest.mark.parametrize(
    "kwargs",
    [{"enable_thinking": False}, {"enable_thinking": True}, {"thinking": True}],
    ids=lambda k: "-".join(f"{key}={value}" for key, value in k.items()),
)
def test_thinking_budget_ids_do_not_depend_on_chat_template_kwargs(parser, kwargs):
    # The chat path passes the request's chat_template_kwargs; map_request has none.
    in_process = SamplingParams(thinking_token_budget=8)
    try:
        add_thinking_budget_logits_processor(
            in_process,
            reasoning_parser=parser,
            tokenizer=CharTokenizer(),
            chat_template_kwargs=kwargs,
        )
    except ValueError:
        return
    processor = in_process.logits_processor
    request = _one(SamplingParams(thinking_token_budget=8), options=_budget_options(parser))
    budget = request.extra[conv.REQ_THINKING_TOKEN_BUDGET]
    assert budget["start_token_ids"] == processor.reasoning_start_token_ids
    assert budget["end_token_ids"] == processor.reasoning_end_token_ids


def test_thinking_budget_rejections():
    params = SamplingParams(thinking_token_budget=4)
    with pytest.raises(ValueError, match="thinking_token_budget"):
        _one(params, _caps("glm"), _budget_options())
    with pytest.raises(ValueError, match="requires a configured reasoning_parser"):
        _one(params, options=contracts.GenerateOptions(tokenizer=CharTokenizer()))
    with pytest.raises(ValueError, match="requires a tokenizer"):
        _one(params, options=contracts.GenerateOptions(reasoning_parser="qwen3"))
    assert (
        conv.REQ_THINKING_TOKEN_BUDGET not in _one(SamplingParams(thinking_token_budget=-1)).extra
    )


# --- request options ------------------------------------------------------------------


def test_logit_bias_as_data():
    caps = _caps(tokenizer_fingerprint=conv.TokenizerFingerprint(100, (1,), None))
    options = contracts.GenerateOptions(logit_bias={"007": 5, "3": -100.0})
    assert _one(caps=caps, options=options).extra[conv.REQ_LOGIT_BIAS] == {"7": 5.0, "3": -100.0}
    # OpenAIServer checks the ids against the model's vocabulary, as in-process.
    assert _one(caps=caps, options=contracts.GenerateOptions(logit_bias={"100": 1})).extra[
        conv.REQ_LOGIT_BIAS
    ] == {"100": 1.0}
    for bias, match in [
        ({str(1 << 32): 1}, r"out of range \[0, 2\*\*32\)"),
        ({"x": 1}, "Invalid logit_bias key"),
        ({"1": math.inf}, r"\[-100"),
        ({"1": 101}, r"\[-100"),
    ]:
        with pytest.raises(ValueError, match=match):
            _one(caps=caps, options=contracts.GenerateOptions(logit_bias=bias))
    too_many = {str(i): 1 for i in range(conv.LOGIT_BIAS_MAX_ENTRIES + 1)}
    with pytest.raises(ValueError, match="entries"):
        _one(options=contracts.GenerateOptions(logit_bias=too_many))
    # Spellings of one id count once.
    aliases = {str(i): 1 for i in range(conv.LOGIT_BIAS_MAX_ENTRIES)} | {"07": 1}
    assert len(
        _one(options=contracts.GenerateOptions(logit_bias=aliases)).extra[conv.REQ_LOGIT_BIAS]
    ) == (conv.LOGIT_BIAS_MAX_ENTRIES)
    with pytest.raises(ValueError, match="logit_bias"):
        _one(caps=_caps("glm"), options=options)


def test_kv_hints_conversation_and_scheduling():
    hints = KvHintsEnvelope(protocol_version="1", message_id="m", actions=[], extra_field=1)
    scheduling = SimpleNamespace(
        attention_dp_rank=1,
        attention_dp_relax=False,
        agent_hierarchy=[("planner", 2)],
        subagent_affinity_id=None,
    )
    conversation = contracts.GenerateOptions(
        conversation_params=SimpleNamespace(conversation_id="c1")
    )
    each = {
        "kv_hints": contracts.GenerateOptions(kv_hints=hints),
        "scheduling_params": contracts.GenerateOptions(scheduling_params=scheduling),
    }
    assert _one(options=each["kv_hints"]).extra[conv.REQ_KV_HINTS] == hints.model_dump(mode="json")
    assert _one(options=conversation).extra[conv.REQ_CONVERSATION_ID] == "c1"
    # A routing hint (OpenAIServer also takes it from gateway headers): not sent
    # to an engine without it, as in-process ignores it without conversation routing.
    assert conv.REQ_CONVERSATION_ID not in _one(caps=_caps("glm"), options=conversation).extra
    assert _one(options=each["scheduling_params"]).extra[conv.REQ_SCHEDULING] == {
        "attention_dp_relax": False,
        "agent_hierarchy": [["planner", 2]],
    }
    for name, options in each.items():
        with pytest.raises(ValueError, match=name):
            _one(caps=_caps("glm"), options=options)


@pytest.mark.parametrize("value", [2**53 + 1, -(2**53) - 1, 2**64 - 1])
def test_kv_hints_integers_a_struct_would_round_are_rejected(value):
    exact = KvHintsEnvelope(
        protocol_version="1", message_id="m", actions=[], block_hashes=[2**53, -(2**53)]
    )
    assert _one(options=contracts.GenerateOptions(kv_hints=exact)).extra[conv.REQ_KV_HINTS][
        "block_hashes"
    ] == [2**53, -(2**53)]
    hints = KvHintsEnvelope(
        protocol_version="1",
        message_id="m",
        actions=[
            {
                "action_id": "a",
                "action_type": "kv.fetch",
                "action_version": "1",
                "payload": {"block_hashes": [1, value]},
            }
        ],
    )
    with pytest.raises(ValueError, match=r"kv_hints.*2\*\*53"):
        _one(options=contracts.GenerateOptions(kv_hints=hints))
    # An engine without kv_hints is told so, not about the integer.
    with pytest.raises(ValueError, match=r"kv_hints=<dict> is not supported [^(]*'eng'$"):
        _one(caps=_caps("glm"), options=contracts.GenerateOptions(kv_hints=hints))


def test_scheduling_and_best_of_integers_a_struct_would_round_are_rejected():
    scheduling = SimpleNamespace(
        attention_dp_rank=None,
        attention_dp_relax=True,
        agent_hierarchy=[("planner", 2**53 + 1)],
        subagent_affinity_id=None,
    )
    caps = _declare(_caps(), conv.CAP_REQ_SCHEDULING)
    with pytest.raises(ValueError, match=r"scheduling_params.*2\*\*53"):
        _one(caps=caps, options=contracts.GenerateOptions(scheduling_params=scheduling))


def test_default_scheduling_params_send_nothing():
    default = SimpleNamespace(
        attention_dp_rank=3,
        attention_dp_relax=True,
        agent_hierarchy=None,
        subagent_affinity_id=None,
    )
    request = _one(caps=_caps("glm"), options=contracts.GenerateOptions(scheduling_params=default))
    assert conv.REQ_SCHEDULING not in request.extra
    affinity = SimpleNamespace(**{**vars(default), "subagent_affinity_id": "p"})
    with pytest.raises(ValueError, match="subagent_affinity_id"):
        _one(options=contracts.GenerateOptions(scheduling_params=affinity))


def test_cache_salt_lora_media_and_disaggregation():
    media = [MediaSpec("image", url="http://x")]
    request = _one(
        caps=_caps(supports_lora=True),
        options=contracts.GenerateOptions(
            cache_salt="s", lora_name="a", lora_request=object(), media=media
        ),
    )
    assert (request.cache_salt, request.lora_name, request.media) == ("s", "a", media)
    for options, match in [
        (contracts.GenerateOptions(cache_salt="s"), "cache_salt"),
        (contracts.GenerateOptions(lora_name="a", lora_request=object()), "lora_request"),
        (contracts.GenerateOptions(lora_request=object()), "not registered"),
        (contracts.GenerateOptions(disaggregated_params=object()), "disaggregated_params"),
    ]:
        with pytest.raises(ValueError, match=match):
            _one(caps=_caps("glm"), options=options)


# --- the request-data contract --------------------------------------------------------------


def test_takes_request_data_reads_the_class():
    class Facade:
        takes_request_data = True

    class Plain:
        pass

    plain = Plain()
    plain.takes_request_data = True
    assert contracts.takes_request_data(Facade())
    assert not contracts.takes_request_data(plain)
    assert not contracts.takes_request_data(SimpleNamespace())


@pytest.mark.parametrize(
    "request_cls, kwargs",
    [
        (CompletionRequest, {"prompt": "x"}),
        (ChatCompletionRequest, {"messages": [{"role": "user", "content": "x"}]}),
    ],
)
def test_logit_bias_is_validated_but_not_materialized(request_cls, kwargs):
    request = request_cls(model="m", logit_bias={"1": 2.0}, **kwargs)
    assert isinstance(request.to_sampling_params(vocab_size=4).embedding_bias, torch.Tensor)
    assert (
        request.to_sampling_params(vocab_size=4, materialize_logit_bias=False).embedding_bias
        is None
    )
    bad = request_cls(model="m", logit_bias={"9": 2.0}, **kwargs)
    with pytest.raises(ValueError, match="vocabulary"):
        bad.to_sampling_params(vocab_size=4, materialize_logit_bias=False)


def test_the_server_passes_request_data_only_to_a_generator_that_takes_it():
    from tensorrt_llm.serve.openai_server import OpenAIServer

    class Facade:
        takes_request_data = True
        args = SimpleNamespace(reasoning_parser="qwen3")

    server = OpenAIServer.__new__(OpenAIServer)
    server.tokenizer = tokenizer = CharTokenizer()
    request = SimpleNamespace(logit_bias={"1": 2.0})
    server.generator = Facade()
    assert server._request_data_kwargs(request) == {
        "logit_bias": {"1": 2.0},
        "reasoning_parser": "qwen3",
        "tokenizer": tokenizer,
    }
    server.use_harmony = False
    assert server._responses_request_data_kwargs(request)["reasoning_parser"] == "qwen3"
    # As in-process: a harmony Responses request uses gpt_oss and has no tokenizer.
    server.use_harmony = True
    harmony = server._responses_request_data_kwargs(request)
    assert (harmony["reasoning_parser"], harmony["tokenizer"]) == ("gpt_oss", None)
    server.generator = SimpleNamespace()
    assert server._request_data_kwargs(request) == {}
    assert server._responses_request_data_kwargs(request) == {}
