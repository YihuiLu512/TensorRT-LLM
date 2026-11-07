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
    (request,) = _map(SamplingParams(n=3, seed=5))
    assert request.num_sequences == 3 and request.sampling["seed"] == 5


@pytest.mark.parametrize(
    "n, limit, sizes", [(17, 16, [16, 1]), (32, 16, [16, 16]), (3, 1, [1, 1, 1]), (5, 2, [2, 2, 1])]
)
def test_n_beyond_the_limit_is_split(n, limit, sizes):
    caps = _caps(max_num_sequences=limit)
    requests = _map(SamplingParams(n=n, seed=9, prompt_logprobs=1), caps)
    assert [r.num_sequences for r in requests] == sizes
    assert len(requests) == math.ceil(n / limit)
    assert [r.sampling["seed"] for r in requests] == [9 + k for k in range(len(sizes))]
    assert [r.prompt_logprobs_top_n for r in requests] == [1] + [None] * (len(sizes) - 1)
    assert len({r.request_id for r in requests}) == len(sizes)


def test_split_seeds_wrap_around_uint64():
    requests = _map(SamplingParams(n=2, seed=(1 << 64) - 1), _caps("glm"))
    assert [r.sampling["seed"] for r in requests] == [(1 << 64) - 1, 0]


def test_an_engine_without_a_limit_counts_as_one():
    assert len(_map(SamplingParams(n=2), _caps(max_num_sequences=0))) == 2


def test_the_sequence_limit_is_logged_once_per_engine(monkeypatch):
    warnings = []
    monkeypatch.setattr(mapping.logger, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(mapping, "_logged", set())
    for _ in range(3):
        _map(SamplingParams(n=2), _caps("glm"))
    mapping.log_sequence_capability("eng", _caps("glm"))
    assert len(warnings) == 1 and "at most 1 sequence" in warnings[0]
    mapping.log_sequence_capability("eng", _caps("glm", max_num_sequences=4))
    assert len(warnings) == 2


def test_best_of_needs_the_capability_and_a_single_call():
    params = SamplingParams(n=1, best_of=2)
    with pytest.raises(ValueError, match="best_of"):
        _map(params)
    caps = _declare(_caps(), conv.CAP_REQ_BEST_OF)
    assert _one(params, caps).extra[conv.REQ_BEST_OF] == 2
    with pytest.raises(ValueError, match="split"):
        _map(SamplingParams(n=2, best_of=3), dataclasses.replace(caps, max_num_sequences=1))


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
    for bias, match in [
        ({"100": 1}, "vocabulary"),
        ({"x": 1}, "Invalid logit_bias key"),
        ({"1": math.inf}, r"\[-100"),
        ({"1": 101}, r"\[-100"),
    ]:
        with pytest.raises(ValueError, match=match):
            _one(caps=caps, options=contracts.GenerateOptions(logit_bias=bias))
    too_many = {str(i): 1 for i in range(conv.LOGIT_BIAS_MAX_ENTRIES + 1)}
    with pytest.raises(ValueError, match="entries"):
        _one(options=contracts.GenerateOptions(logit_bias=too_many))
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
    each = {
        "kv_hints": contracts.GenerateOptions(kv_hints=hints),
        "conversation_id": contracts.GenerateOptions(
            conversation_params=SimpleNamespace(conversation_id="c1")
        ),
        "scheduling_params": contracts.GenerateOptions(scheduling_params=scheduling),
    }
    assert _one(options=each["kv_hints"]).extra[conv.REQ_KV_HINTS] == hints.model_dump(mode="json")
    assert _one(options=each["conversation_id"]).extra[conv.REQ_CONVERSATION_ID] == "c1"
    assert _one(options=each["scheduling_params"]).extra[conv.REQ_SCHEDULING] == {
        "attention_dp_relax": False,
        "agent_hierarchy": [["planner", 2]],
    }
    for name, options in each.items():
        with pytest.raises(ValueError, match=name):
            _one(caps=_caps("glm"), options=options)


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


# --- contract v4: request data -----------------------------------------------------------


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


def test_the_server_passes_logit_bias_only_to_a_generator_that_takes_it():
    from tensorrt_llm.serve.openai_server import OpenAIServer

    class Facade:
        takes_request_data = True

    server = OpenAIServer.__new__(OpenAIServer)
    request = SimpleNamespace(logit_bias={"1": 2.0})
    server.generator = Facade()
    assert server._request_data_kwargs(request) == {"logit_bias": {"1": 2.0}}
    server.generator = SimpleNamespace()
    assert server._request_data_kwargs(request) == {}
