# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""X-1, the no-silent-drop scan (TH-PRF-4), on the request mapping.

Every field of ``SamplingParams``, ``GenerateOptions`` and the OpenAI request
models must be classified; a mapped field set to a non-default value must reach
the engine's request when the engine declares what it needs and be rejected (or
handled as the table's fallback says) when it does not; a rejected field must
be rejected; a field the serving side handles must not be rejected and must
name the feature row whose tests verify it, or say why none is needed.

The last section sends the probes through the ``OpenEngineLLM`` facade to the
fake OpenEngine engine over gRPC (row F-44-cpu).
"""

import ast
import asyncio
import dataclasses
import inspect
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.sampling_params import GuidedDecodingParams, LogprobMode, SamplingParams
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponsesRequest,
)
from tensorrt_llm.serve.openengine_poc import OpenEngineLLM, contracts, mapping
from tensorrt_llm.serve.openengine_poc import field_classes as fc
from tensorrt_llm.serve.openengine_poc.engine_client import EngineRequest, GuidedSpec, MediaSpec
from tensorrt_llm.serve.openengine_poc.engine_client_impl import to_generate_request
from tensorrt_llm.tokenizer import TransformersTokenizer

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from fake_openengine import PROFILES as FAKE_PROFILES  # noqa: E402
from fake_openengine import FakeEngine  # noqa: E402
from fakes import glm_like, trtllm_like  # noqa: E402
from openengine_client_helpers import WAIT_SECONDS  # noqa: E402

pytestmark = pytest.mark.cpu_only

PROMPT = [2, 3, 4]
REQUEST_MODELS = (CompletionRequest, ChatCompletionRequest, ResponsesRequest)


class CharTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


def _all_declared():
    caps = trtllm_like()[1]
    return dataclasses.replace(
        caps,
        supports_lora=True,
        extra=dataclasses.replace(caps.extra, declared=frozenset(conv.BOOL_CAPABILITY_KEYS)),
    )


def _none_declared():
    caps = glm_like()[1]
    return dataclasses.replace(
        caps,
        output_logprobs_max_top_n=None,
        supports_stop_in_output=False,
        extra=dataclasses.replace(caps.extra, declared=frozenset()),
    )


PROFILES = {
    "trtllm-like": trtllm_like()[1],
    "glm-like": glm_like()[1],
    "all-declared": _all_declared(),
    "none-declared": _none_declared(),
}


# --- coverage ---------------------------------------------------------------------


def _difference(table: dict, fields: set) -> str:
    return f"unclassified: {sorted(fields - set(table))}; stale: {sorted(set(table) - fields)}"


def test_every_sampling_params_field_is_classified():
    fields = {f.name for f in dataclasses.fields(SamplingParams)}
    assert set(fc.SAMPLING_PARAMS) == fields, _difference(fc.SAMPLING_PARAMS, fields)


def test_every_generate_option_is_classified():
    fields = {f.name for f in dataclasses.fields(contracts.GenerateOptions)}
    assert set(fc.GENERATE_OPTIONS) == fields, _difference(fc.GENERATE_OPTIONS, fields)


@pytest.mark.parametrize("model", REQUEST_MODELS, ids=lambda m: m.__name__)
def test_every_request_model_field_is_classified(model):
    table = fc.REQUEST_MODELS[model.__name__]
    fields = set(model.model_fields)
    assert set(table) == fields, _difference(table, fields)
    for name, route in table.items():
        if isinstance(route, fc.FieldClass):
            assert route.kind is fc.Kind.SERVE, name
            continue
        kind, _, target = route.partition(".")
        target_table = {"sampling_params": fc.SAMPLING_PARAMS, "options": fc.GENERATE_OPTIONS}[kind]
        assert target in target_table, f"{name} -> {route}"
        assert target_table[target].kind is not fc.Kind.INTERNAL, f"{name} -> {route}"


def test_every_serve_field_names_a_feature_row():
    for table in (
        fc.SAMPLING_PARAMS,
        fc.GENERATE_OPTIONS,
        *[
            {k: v for k, v in t.items() if isinstance(v, fc.FieldClass)}
            for t in fc.REQUEST_MODELS.values()
        ],
    ):
        for name, field_class in table.items():
            if field_class.kind is not fc.Kind.SERVE:
                continue
            verified_by = field_class.verified_by
            if verified_by.startswith("none needed: "):
                continue
            rows = re.findall(r"\brow (F-[\w+-]+)", verified_by)
            assert rows and verified_by.startswith("row "), (name, verified_by)
            assert set(rows) <= _feature_rows(), (name, verified_by)


def _feature_rows() -> set[str]:
    path = Path(__file__).resolve().parents[5] / "scripts" / "poc2" / "feature_deps.yaml"
    with open(path, encoding="utf-8") as f:
        return {row["id"] for row in yaml.safe_load(f)["rows"]}


# --- probes -------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Probe:
    """One field set to a non-default value.

    Attributes:
        params: SamplingParams attributes to set (assigned after construction).
        options: GenerateOptions fields to set.
        supported: Whether an engine with these capabilities takes the field.
        landed: Whether the field is in the engine requests.
        fallback: Whether an unsupported field is handled as the table's fallback
            says instead of rejected.
    """

    params: dict = dataclasses.field(default_factory=dict)
    options: dict = dataclasses.field(default_factory=dict)
    supported: Callable[[Any], bool] = lambda caps: True
    landed: Callable[[list[EngineRequest]], bool] = lambda requests: True
    fallback: bool = False


def _cap(key):
    return lambda caps: caps.supports(key)


def _first(check):
    return lambda requests: check(requests[0])


_BUDGET_OPTIONS = {"reasoning_parser": "qwen3", "tokenizer": CharTokenizer()}

MAPPED_PROBES: dict[str, Probe] = {
    "max_tokens": Probe({"max_tokens": 5}, landed=_first(lambda r: r.max_tokens == 5)),
    "stop": Probe(
        {"stop": ["zz"]},
        supported=_cap(conv.CAP_STOP_TEXT),
        landed=_first(lambda r: r.stop_texts == ["zz"]),
        fallback=True,
    ),
    "stop_token_ids": Probe(
        {"stop_token_ids": [9]}, landed=_first(lambda r: r.stop_token_ids == [9])
    ),
    "include_stop_str_in_output": Probe(
        {"include_stop_str_in_output": True, "stop_token_ids": [9]},
        supported=lambda caps: caps.supports_stop_in_output,
        landed=_first(lambda r: r.include_stop_in_output is True),
    ),
    # SamplingParams sets best_of to n when it is not given.
    "n": Probe({"n": 2, "best_of": 2}, landed=lambda rs: sum(r.num_sequences for r in rs) == 2),
    "best_of": Probe(
        {"n": 1, "best_of": 2},
        supported=_cap(conv.CAP_REQ_BEST_OF),
        landed=_first(lambda r: r.extra[conv.REQ_BEST_OF] == 2),
    ),
    **{
        name: Probe({name: value}, landed=_first(lambda r, n=name, v=value: r.sampling[n] == v))
        for name, value in (("top_k", 5), ("top_p", 0.5), ("seed", 3), ("temperature", 0.5))
    },
    **{
        name: Probe(
            {name: 0.5},
            supported=_cap(conv.SAMPLING_FIELD_CAPABILITY[name]),
            landed=_first(lambda r, n=name: r.sampling[n] == 0.5),
        )
        for name in conv.SAMPLING_FIELD_CAPABILITY
    },
    "min_tokens": Probe({"min_tokens": 2}, landed=_first(lambda r: r.min_tokens == 2)),
    "logprobs": Probe(
        {"logprobs": 2},
        supported=lambda caps: (caps.output_logprobs_max_top_n or -1) >= 2,
        landed=_first(lambda r: r.output_logprobs_top_n == 2),
    ),
    "prompt_logprobs": Probe(
        {"prompt_logprobs": 1},
        supported=lambda caps: (caps.prompt_logprobs_max_top_n or -1) >= 1,
        landed=_first(lambda r: r.prompt_logprobs_top_n == 1),
    ),
    "guided_decoding": Probe(
        {"guided_decoding": GuidedDecodingParams(regex="a+")},
        supported=lambda caps: "regex" in caps.guided_modes,
        landed=_first(lambda r: r.guided == GuidedSpec("regex", "a+")),
    ),
    "thinking_token_budget": Probe(
        {"thinking_token_budget": 4},
        _BUDGET_OPTIONS,
        supported=_cap(conv.CAP_REQ_THINKING_TOKEN_BUDGET),
        landed=_first(lambda r: r.extra[conv.REQ_THINKING_TOKEN_BUDGET]["budget"] == 4),
    ),
    "ignore_eos": Probe({"ignore_eos": True}, landed=_first(lambda r: r.ignore_eos is True)),
}

OPTION_PROBES: dict[str, Probe] = {
    "lora_request": Probe(
        options={"lora_request": object(), "lora_name": "a"},
        supported=lambda caps: caps.supports_lora,
        landed=_first(lambda r: r.lora_name == "a"),
    ),
    "lora_name": Probe(
        options={"lora_name": "a"},
        supported=lambda caps: caps.supports_lora,
        landed=_first(lambda r: r.lora_name == "a"),
    ),
    "cache_salt": Probe(
        options={"cache_salt": "s"},
        supported=lambda caps: caps.supports_cache_salt,
        landed=_first(lambda r: r.cache_salt == "s"),
    ),
    "kv_hints": Probe(
        options={"kv_hints": {"protocol_version": "1"}},
        supported=_cap(conv.CAP_REQ_KV_HINTS),
        landed=_first(lambda r: r.extra[conv.REQ_KV_HINTS] == {"protocol_version": "1"}),
    ),
    "conversation_params": Probe(
        options={"conversation_params": type("C", (), {"conversation_id": "c"})()},
        supported=_cap(conv.CAP_REQ_CONVERSATION_ID),
        landed=_first(lambda r: r.extra.get(conv.REQ_CONVERSATION_ID) == "c"),
        fallback=True,
    ),
    "scheduling_params": Probe(
        options={"scheduling_params": type("S", (), {"attention_dp_relax": False})()},
        supported=_cap(conv.CAP_REQ_SCHEDULING),
        landed=_first(lambda r: r.extra[conv.REQ_SCHEDULING] == {"attention_dp_relax": False}),
    ),
    "media": Probe(
        options={"media": [MediaSpec("image", url="http://x")]},
        landed=_first(lambda r: r.media == [MediaSpec("image", url="http://x")]),
    ),
    "logit_bias": Probe(
        options={"logit_bias": {"1": 2.0}},
        supported=_cap(conv.CAP_REQ_LOGIT_BIAS),
        landed=_first(lambda r: r.extra[conv.REQ_LOGIT_BIAS] == {"1": 2.0}),
    ),
}
# The facade turns these into metadata after map_request; the gRPC scan checks them.
FACADE_MAPPED = {"priority", "trace_headers"}

REJECTED_VALUES = {
    "end_id": 1,
    "pad_id": 1,
    "bad": "x",
    "bad_token_ids": [1],
    "embedding_bias": torch.ones(4),
    "logits_processor": lambda *args: None,
    "apply_batched_logits_processor": True,
    "use_beam_search": True,
    "logprobs_mode": LogprobMode.PROCESSED,
    "top_p_min": 0.1,
    "top_p_reset_ids": 1,
    "top_p_decay": 0.9,
    "beam_search_diversity_rate": 0.5,
    "prompt_ignore_length": 2,
    "length_penalty": 0.5,
    "early_stopping": 1,
    "no_repeat_ngram_size": 2,
    "beam_width_array": [2],
    "logprobs_simple_format": True,
    "prompt_logprobs_simple_format": True,
    "return_context_logits": True,
    "return_generation_logits": True,
    "exclude_input_from_output": False,
    "return_encoder_output": True,
    "return_routed_experts": True,
    "additional_model_outputs": ["x"],
    "_decoder_output_token_prefix": (1,),
}
REJECTED_OPTION_VALUES = {"disaggregated_params": object()}
SERVE_VALUES = {
    "detokenize": False,
    "add_special_tokens": False,
    "truncate_prompt_tokens": 2,
    "skip_special_tokens": False,
    "spaces_between_special_tokens": False,
    "return_perf_metrics": True,
}
SERVE_OPTION_VALUES = {"streaming": True}


def _kinds(table, kind):
    return {name for name, field_class in table.items() if field_class.kind is kind}


def test_every_classified_field_has_a_probe():
    assert set(MAPPED_PROBES) == _kinds(fc.SAMPLING_PARAMS, fc.Kind.MAPPED)
    assert set(OPTION_PROBES) | FACADE_MAPPED == _kinds(fc.GENERATE_OPTIONS, fc.Kind.MAPPED)
    assert set(REJECTED_VALUES) == _kinds(fc.SAMPLING_PARAMS, fc.Kind.REJECTED)
    assert set(REJECTED_OPTION_VALUES) == _kinds(fc.GENERATE_OPTIONS, fc.Kind.REJECTED)
    assert set(SERVE_VALUES) == _kinds(fc.SAMPLING_PARAMS, fc.Kind.SERVE)
    assert set(SERVE_OPTION_VALUES) == _kinds(fc.GENERATE_OPTIONS, fc.Kind.SERVE)


def _run(caps, params: dict = (), options: dict = ()):
    sampling_params = SamplingParams()
    for name, value in dict(params).items():
        setattr(sampling_params, name, value)
    return mapping.map_request(
        PROMPT, sampling_params, caps, contracts.GenerateOptions(**dict(options)), engine_name="eng"
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("name", sorted({**MAPPED_PROBES, **OPTION_PROBES}))
def test_a_mapped_field_reaches_the_engine_or_is_rejected(profile, name):
    caps = PROFILES[profile]
    probe = {**MAPPED_PROBES, **OPTION_PROBES}[name]
    if probe.supported(caps):
        assert probe.landed(_run(caps, probe.params, probe.options)), name
    elif probe.fallback:
        requests = _run(caps, probe.params, probe.options)
        assert not probe.landed(requests), f"{name} was sent although not declared"
    else:
        with pytest.raises(ValueError):
            _run(caps, probe.params, probe.options)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("name", sorted({**REJECTED_VALUES, **REJECTED_OPTION_VALUES}))
def test_a_rejected_field_is_rejected(profile, name):
    params = {name: REJECTED_VALUES[name]} if name in REJECTED_VALUES else {}
    options = {name: REJECTED_OPTION_VALUES[name]} if name in REJECTED_OPTION_VALUES else {}
    with pytest.raises(ValueError, match=name):
        _run(PROFILES[profile], params, options)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("name", sorted({**SERVE_VALUES, **SERVE_OPTION_VALUES}))
def test_a_serve_field_is_not_rejected(profile, name):
    params = {name: SERVE_VALUES[name]} if name in SERVE_VALUES else {}
    options = {name: SERVE_OPTION_VALUES[name]} if name in SERVE_OPTION_VALUES else {}
    (request,) = _run(PROFILES[profile], params, options)
    assert request.extra[conv.REQ_DETOKENIZE] is False


def test_an_unclassified_field_is_rejected(monkeypatch):
    table = dict(fc.SAMPLING_PARAMS)
    del table["top_k"]
    monkeypatch.setattr(mapping, "SAMPLING_PARAMS", table)
    with pytest.raises(ValueError, match="unclassified"):
        _run(PROFILES["trtllm-like"], {"top_k": 5})


# --- the same probes through the facade, against the gRPC fake engine ----------------

_GRPC_MODEL = "x1-model"
_GRPC_VOCAB = {"<unk>": 0, "</s>": 1, **{f"t{i}": i for i in range(2, 16)}}
_GRPC_EOS = 1
GRPC_PROMPT = [2, 3, 5]
_TRACEPARENT = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"


class SmallTokenizer:
    """Encodes any text as one id in the fake engine's vocabulary."""

    def encode(self, text, add_special_tokens=True):
        return [2 + len(text) % 14]


# generate_async takes neither: the facade resolves lora_name from lora_request
# through the LoRA registry and media from the prompt through the map_media slot.
GRPC_DERIVED = {"lora_name", "media"}
# What the facade sends as gRPC metadata: (options, supported, check on the call).
FACADE_PROBES = {
    "priority": (
        {"priority": 0.9},
        lambda caps: caps.supports_priority,
        lambda call: call.metadata_value(conv.META_PRIORITY) == str(conv.encode_priority(0.9)),
    ),
    "trace_headers": (
        {"trace_headers": {conv.META_TRACEPARENT: _TRACEPARENT}},
        lambda caps: True,
        lambda call: call.metadata_value(conv.META_TRACEPARENT) == _TRACEPARENT,
    ),
}
GRPC_PROFILES = {
    "trtllm-like": {"supports_priority": True},
    "glm-like": {},
}


def _grpc_profile(name):
    return FAKE_PROFILES[name].replace(
        model=_GRPC_MODEL,
        max_context_length=64,
        vocab_size=len(_GRPC_VOCAB),
        eos_token_ids=(_GRPC_EOS,),
        bos_token_id=None,
        **GRPC_PROFILES[name],
    )


def _grpc_facade(engine, tmp_path):
    backend = Tokenizer(WordLevel(_GRPC_VOCAB, unk_token="<unk>"))
    backend.add_special_tokens(["</s>"])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    args = SimpleNamespace(
        backend="pytorch",
        reasoning_parser=None,
        gather_generation_logits=False,
        guided_decoding_backend=None,
        num_postprocess_workers=0,
        cache_transceiver_config=None,
        trust_remote_code=False,
        checkpoint_format=None,
        return_perf_metrics=False,
        perf_metrics_output_dir=None,
        speculative_config=None,
        post_processor_hook=None,
        sleep_config=None,
        num_serve_frontends=1,
        enable_iter_perf_stats=False,
        max_beam_width=1,
        custom_tokenizer=None,
        max_batch_size=None,
        kv_cache_config=None,
    )
    return OpenEngineLLM(
        args=args,
        tokenizer=TransformersTokenizer(hf),
        input_processor=lambda prompt, sampling_params: (list(GRPC_PROMPT), None),
        hf_model_dir=str(tmp_path),
        endpoint=engine.target,
        ready_timeout=WAIT_SECONDS,
    )


def _sampling(params) -> SamplingParams:
    sampling_params = SamplingParams()
    for name, value in dict(params).items():
        setattr(sampling_params, name, value)
    return sampling_params


def _without_id(message):
    message.ClearField("request_id")
    return message.SerializeToString(deterministic=True)


def _over_grpc(profile, tmp_path, check):
    """Run ``check(engine, facade)`` with a started facade on a fake ``profile`` engine."""

    async def main():
        async with FakeEngine(_grpc_profile(profile)) as engine:
            facade = _grpc_facade(engine, tmp_path)
            await facade.start()
            try:
                await asyncio.wait_for(check(engine, facade), WAIT_SECONDS * 2)
            finally:
                await facade.aclose()

    asyncio.run(main())


@pytest.mark.parametrize("profile", GRPC_PROFILES)
@pytest.mark.parametrize("name", sorted({**MAPPED_PROBES, **OPTION_PROBES}.keys() - GRPC_DERIVED))
def test_over_grpc_the_engine_receives_what_the_mapping_sends(profile, name, tmp_path):
    probe = {**MAPPED_PROBES, **OPTION_PROBES}[name]
    options = {key: value for key, value in probe.options.items() if key not in GRPC_DERIVED}
    if name == "thinking_token_budget":
        options = {**options, "tokenizer": SmallTokenizer()}

    async def check(engine, facade):
        caps = facade.client.model_info()
        engine_name = facade.client.server_info().engine_name
        if not probe.supported(caps) and not probe.fallback:
            with pytest.raises(ValueError):
                await facade.generate_async(GRPC_PROMPT, _sampling(probe.params), **options)
            assert engine.requests == []
            return
        await facade.generate_async(GRPC_PROMPT, _sampling(probe.params), **options)
        expected = mapping.map_request(
            GRPC_PROMPT,
            _sampling(probe.params),
            caps,
            contracts.GenerateOptions(**options),
            engine_name=engine_name,
        )
        assert bool(probe.landed(expected)) == bool(probe.supported(caps)), name
        received = sorted(_without_id(message) for message in engine.requests)
        assert received == sorted(_without_id(to_generate_request(r)) for r in expected), name

    _over_grpc(profile, tmp_path, check)


@pytest.mark.parametrize("profile", GRPC_PROFILES)
@pytest.mark.parametrize("name", sorted(FACADE_PROBES))
def test_over_grpc_a_facade_mapped_field_reaches_the_engine_or_is_rejected(profile, name, tmp_path):
    options, supported, landed = FACADE_PROBES[name]

    async def check(engine, facade):
        if not supported(facade.client.model_info()):
            with pytest.raises(ValueError, match=name):
                facade.generate_async(GRPC_PROMPT, SamplingParams(max_tokens=1), **options)
            assert engine.requests == []
            return
        await facade.generate_async(GRPC_PROMPT, SamplingParams(max_tokens=1), **options)
        assert landed(engine.accepted[0]), name

    _over_grpc(profile, tmp_path, check)


def _server_generate_async_keywords() -> set[str]:
    """Every keyword ``OpenAIServer`` passes to ``generate_async``, with the request data."""
    source = Path(inspect.getfile(contracts)).parents[1] / "openai_server.py"
    keywords = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "generate_async":
            for keyword in node.keywords:
                if keyword.arg is not None:
                    keywords.add(keyword.arg)
                else:
                    # Only the request-data helpers are spread into the call.
                    assert ast.unparse(keyword.value).startswith(
                        ("self._request_data_kwargs(", "self._responses_request_data_kwargs(")
                    ), ast.unparse(keyword.value)
    return keywords | {"logit_bias", "reasoning_parser", "tokenizer"}


def test_the_facade_names_every_keyword_the_server_passes():
    """Nothing the server passes falls into the facade's ``**_ignored``."""
    parameters = inspect.signature(OpenEngineLLM.generate_async).parameters
    named = {name for name, p in parameters.items() if p.kind is not p.VAR_KEYWORD}
    assert _server_generate_async_keywords() - named == set()
    assert contracts.takes_request_data(object.__new__(OpenEngineLLM))
    # Each keyword other than the prompt and the pool's own names a GenerateOptions field.
    options = {f.name for f in dataclasses.fields(contracts.GenerateOptions)}
    assert named - {"self", "inputs", "sampling_params", "_postproc_params"} <= options
