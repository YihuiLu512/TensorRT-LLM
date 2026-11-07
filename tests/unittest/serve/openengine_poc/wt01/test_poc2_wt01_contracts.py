# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the contract slots: lookup, and that each default matches the adapter and mapping code."""

import asyncio
import collections
import dataclasses
import enum
import json
import os
import pickle
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest import mock

import click
import pydantic
import pytest
import torch

from tensorrt_llm.executor.postproc_worker import PostprocArgs, PostprocParams
from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.grpc.openengine.bindings import generation_pb2, model_pb2, server_pb2
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc import contracts
from tensorrt_llm.serve.openengine_poc.engine_client import (
    OutputFinished,
    PromptInfo,
    TokenDelta,
    Usage,
)
from tensorrt_llm.serve.postprocess_handlers import chat_stream_post_processor

__extra_import_path__ = [".."]
from fakes import TINY_EOS, FakeEngineClient, tiny_tokenizer  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

PACKAGE = "tensorrt_llm.serve.openengine_poc"


@pytest.fixture(autouse=True)
def _fresh_slots():
    contracts.reset_slot_cache()
    yield
    contracts.reset_slot_cache()


def test_contract_version():
    assert contracts.POC2_CONTRACT_VERSION == 1


def test_slot_table():
    table = {
        name: (slot.module, slot.attribute, slot.owner) for name, slot in contracts.slots().items()
    }
    assert table == {
        "map_request": (f"{PACKAGE}.mapping", "map_request", "WT-11"),
        "create_response_assembler": (
            f"{PACKAGE}.response",
            "create_response_assembler",
            "WT-12",
        ),
        "create_postproc_dispatch": (
            f"{PACKAGE}.postproc_pool",
            "create_postproc_dispatch",
            "WT-13",
        ),
        "launch_frontends": (f"{PACKAGE}.multi_frontend", "launch_frontends", "WT-14"),
        "create_metrics_recorder": (f"{PACKAGE}.metrics", "create_metrics_recorder", "WT-15"),
        "map_media": (f"{PACKAGE}.media", "map_media", "WT-16"),
        "create_lora_registry": (f"{PACKAGE}.lora", "create_lora_registry", "WT-16"),
    }


def test_every_slot_uses_its_default_until_its_module_defines_it():
    for name, slot in contracts.slots().items():
        impl, source = contracts.resolve_slot(name)
        assert (impl, source) == (slot.default, "default"), name


def test_slot_uses_the_module_once_it_defines_the_function(monkeypatch):
    module = types.ModuleType(f"{PACKAGE}.media")
    calls = []
    module.map_media = lambda inputs, caps, *, engine_name: calls.append(engine_name) or ["x"]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    caps = conv.ModelCapabilities(served_model_name="m")
    assert contracts.map_media({}, caps, engine_name="e") == ["x"]
    assert calls == ["e"]
    assert contracts.resolve_slot("map_media")[1] == f"{PACKAGE}.media:map_media"


def test_slot_lookup_is_logged_once(monkeypatch):
    lines = []
    monkeypatch.setattr(contracts.logger, "info", lines.append)
    contracts.resolve_slot("map_media")
    contracts.resolve_slot("map_media")
    assert lines == ["OpenEngine slot map_media: using default"]


def test_a_broken_implementation_module_is_not_hidden(monkeypatch):
    def missing_dependency(name):
        raise ModuleNotFoundError("No module named 'some_dependency'", name="some_dependency")

    monkeypatch.setattr(contracts.importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="some_dependency"):
        contracts.resolve_slot("map_media")

    def broken(name):
        raise SyntaxError("bad module")

    monkeypatch.setattr(contracts.importlib, "import_module", broken)
    with pytest.raises(SyntaxError):
        contracts.resolve_slot("map_media")


def test_a_slot_resolved_while_its_module_is_imported_is_looked_up_again(monkeypatch):
    real_import = contracts.importlib.import_module
    name = f"{PACKAGE}.media"
    loaded = types.ModuleType(name)
    loaded.__spec__ = SimpleNamespace(_initializing=True)

    def import_module(module_name):
        return loaded if module_name == name else real_import(module_name)

    monkeypatch.setattr(contracts.importlib, "import_module", import_module)
    # Still being imported, without the function yet: the default, not cached.
    assert contracts.resolve_slot("map_media")[1] == "default"
    loaded.__spec__._initializing = False
    loaded.map_media = lambda *args, **kwargs: []
    assert contracts.resolve_slot("map_media")[1] == f"{name}:map_media"


def test_a_slot_module_may_resolve_other_slots_while_it_is_imported(monkeypatch):
    real_import = contracts.importlib.import_module
    nested = []

    def import_module(name):
        if name == f"{PACKAGE}.media":
            # What a slot module that calls another slot at import time does.
            nested.append(contracts.resolve_slot("create_lora_registry")[1])
            module = types.ModuleType(name)
            module.map_media = lambda *args, **kwargs: []
            return module
        return real_import(name)

    monkeypatch.setattr(contracts.importlib, "import_module", import_module)
    assert contracts.resolve_slot("map_media")[1] == f"{PACKAGE}.media:map_media"
    assert nested == ["default"]


# --- map_request ----------------------------------------------------------------


def _engine(engine_name="tensorrt_llm", conventions=False, extra=None, **model_fields):
    server = server_pb2.ServerInfo(engine_name=engine_name, schema_revision=1)
    if conventions:
        server.extra.update({conv.CONVENTIONS_VERSION_KEY: 2})
    model = model_pb2.ModelInfo(
        served_model_name="stub-model",
        max_context_length=model_fields.pop("max_context_length", 64),
        supports_token_ids_input=True,
        **model_fields,
    )
    if extra:
        model.extra.update(extra)
    return server, model


def _proto_view(request: generation_pb2.GenerateRequest) -> dict:
    sampling = {f.name: v for f, v in request.sampling.ListFields()}
    num_sequences = sampling.pop("num_sequences", 1)
    stopping = request.stopping
    return {
        "model": request.model,
        "prompt": list(request.token_ids.ids),
        "sampling": sampling,
        "num_sequences": num_sequences,
        "max_tokens": stopping.max_tokens if stopping.HasField("max_tokens") else None,
        "min_tokens": stopping.min_tokens if stopping.HasField("min_tokens") else None,
        "ignore_eos": stopping.ignore_eos if stopping.HasField("ignore_eos") else None,
        "stop_token_ids": [c.stop_token_id for c in stopping.conditions],
        "extra": conv.struct_to_dict(request.extra),
    }


def _engine_request_view(request) -> dict:
    return {
        "model": request.model,
        "prompt": request.prompt_token_ids,
        "sampling": request.sampling,
        "num_sequences": request.num_sequences,
        "max_tokens": request.max_tokens,
        "min_tokens": request.min_tokens,
        "ignore_eos": request.ignore_eos,
        "stop_token_ids": request.stop_token_ids,
        "extra": request.extra,
    }


def _both(prompt, params, server, model):
    from tensorrt_llm.serve.openengine_poc.mapping import (
        build_generate_request,
        builtin_capabilities,
    )

    def poc_s1():
        return _proto_view(
            build_generate_request(
                prompt,
                params,
                model,
                engine_name=server.engine_name,
                builtin=builtin_capabilities(server),
            )
        )

    def default():
        caps = conv.model_capabilities_from_proto(server, model)
        (request,) = contracts.default_map_request(
            prompt, params, caps, contracts.GenerateOptions(), engine_name=server.engine_name
        )
        assert request.request_id.startswith("trtllm-") and len(request.request_id) == 39
        return _engine_request_view(request)

    return poc_s1, default


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(),
        SamplingParams(temperature=0.0, top_p=0.9, top_k=5, seed=7),
        SamplingParams(min_p=0.1, frequency_penalty=0.5, presence_penalty=0.2),
        SamplingParams(repetition_penalty=1.1, max_tokens=8, min_tokens=2, ignore_eos=True),
        SamplingParams(stop_token_ids=[5, 5, 6], max_tokens=1000),
        # Several outputs need sampling: greedy decoding rejects best_of > 1.
        SamplingParams(n=2, best_of=2, temperature=1.0),
        SamplingParams(top_k=0, min_p=0.0, repetition_penalty=1.0),
    ],
    ids=lambda p: "params",
)
@pytest.mark.parametrize("engine", ["builtin", "declared", "glm"])
def test_default_map_request_matches_poc_s1(params, engine):
    if engine == "builtin":
        server, model = _engine()
    elif engine == "declared":
        server, model = _engine(
            conventions=True,
            extra={key: True for key in conv.SAMPLING_CAPABILITY_KEYS},
            max_output_tokens=16,
        )
    else:
        server, model = _engine("glm5", extra={conv.CAP_MIN_P: True})
    model.generation.max_num_sequences = 4
    poc_s1, default = _both([2, 3, 5], params, server, model)
    try:
        expected = poc_s1()
    except ValueError as error:
        with pytest.raises(ValueError) as caught:
            default()
        assert str(caught.value) == str(error)
    else:
        assert default() == expected


@pytest.mark.parametrize(
    "params, prompt",
    [
        (SamplingParams(use_beam_search=True, n=2, best_of=2), [1]),
        (SamplingParams(logprobs=2), [1]),
        (SamplingParams(n=3, best_of=4, temperature=1.0), [1]),
        (SamplingParams(n=9, temperature=1.0), [1]),
        (SamplingParams(seed=-1), [1]),
        (SamplingParams(max_tokens=1), list(range(64))),
        (SamplingParams(truncate_prompt_tokens=2), [1]),
        (SamplingParams(prompt_ignore_length=1), [1]),
        (SamplingParams(embedding_bias=torch.zeros(2)), [1]),
    ],
)
def test_default_map_request_rejects_like_poc_s1(params, prompt):
    server, model = _engine()
    model.generation.max_num_sequences = 4
    poc_s1, default = _both(prompt, params, server, model)
    with pytest.raises(ValueError) as expected:
        poc_s1()
    with pytest.raises(ValueError) as got:
        default()
    assert str(got.value) == str(expected.value)


def test_default_map_request_rejects_gated_fields_the_engine_lacks():
    server, model = _engine("glm5")
    caps = conv.model_capabilities_from_proto(server, model)
    with pytest.raises(ValueError, match="frequency_penalty"):
        contracts.default_map_request(
            [1],
            SamplingParams(frequency_penalty=0.5),
            caps,
            contracts.GenerateOptions(),
            engine_name="glm5",
        )


@pytest.mark.parametrize(
    "options, match",
    [
        (contracts.GenerateOptions(lora_request=object()), "lora_request"),
        (contracts.GenerateOptions(lora_name="adapter"), "lora_name"),
        (contracts.GenerateOptions(disaggregated_params=object()), "disaggregated_params"),
        (contracts.GenerateOptions(cache_salt="s"), "cache_salt"),
        (contracts.GenerateOptions(kv_hints=object()), "kv_hints"),
        (
            contracts.GenerateOptions(media=[contracts.MediaSpec("image", url="http://x")]),
            "multimodal",
        ),
    ],
)
def test_default_map_request_rejects_options_like_the_adapter(options, match):
    caps = conv.ModelCapabilities(served_model_name="m", max_context_length=64)
    with pytest.raises(ValueError, match=match):
        contracts.default_map_request([1], SamplingParams(), caps, options, engine_name="e")


def test_default_map_request_ignores_the_options_the_adapter_ignored():
    caps = conv.ModelCapabilities(served_model_name="m", max_context_length=64)
    options = contracts.GenerateOptions(
        priority=0.9,
        scheduling_params=object(),
        conversation_params=object(),
        trace_headers={"traceparent": "x"},
    )
    (request,) = contracts.default_map_request(
        [1], SamplingParams(), caps, options, engine_name="e"
    )
    assert request.metadata == {} and request.extra == {"detokenize": False}


def test_default_map_request_needs_detokenize_for_stop_strings():
    caps = conv.ModelCapabilities(served_model_name="m", max_context_length=64)
    with pytest.raises(ValueError, match="without detokenizing"):
        contracts.default_map_request(
            [1],
            SamplingParams(stop=["x"], detokenize=False),
            caps,
            contracts.GenerateOptions(),
            engine_name="e",
        )


# --- create_response_assembler ---------------------------------------------------


def _to_proto(event) -> generation_pb2.GenerateResponse:
    response = generation_pb2.GenerateResponse(request_id="r")
    if isinstance(event, TokenDelta):
        response.token.output_index = event.output_index
        response.token.tokens.extend(generation_pb2.TokenInfo(token_id=t) for t in event.token_ids)
    elif isinstance(event, OutputFinished):
        finished = response.finished
        finished.output_index = event.output_index
        finished.reason = {
            "stop": generation_pb2.FINISH_REASON_STOP,
            "length": generation_pb2.FINISH_REASON_LENGTH,
            "cancelled": generation_pb2.FINISH_REASON_CANCELLED,
        }[event.reason]
        if event.eos_token_id is not None:
            finished.stop_match.eos_token_id = event.eos_token_id
        elif event.stop_token_id is not None:
            finished.stop_match.stop_token_id = event.stop_token_id
        elif event.stop_text is not None:
            finished.stop_match.stop_text = event.stop_text
        if event.usage is not None:
            response.usage.prompt_tokens = event.usage.prompt_tokens
            response.usage.completion_tokens = event.usage.completion_tokens
            if event.usage.cached_prompt_tokens is not None:
                response.usage.cached_prompt_tokens = event.usage.cached_prompt_tokens
    else:
        response.prompt.SetInParent()
    return response


class _ListStream:
    """Stands in for a GenerateStream; yields scripted responses."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.request_id = "r"
        self.cancelled = False

    async def __anext__(self):
        if self.cancelled:
            raise StopAsyncIteration
        try:
            return next(self._responses)
        except StopIteration:
            raise StopAsyncIteration from None

    def cancel(self):
        self.cancelled = True

    async def aclose(self):
        pass


def _snapshot(outputs):
    return [
        (
            o.text,
            list(o.token_ids),
            o.text_diff,
            list(o.token_ids_diff),
            o.finish_reason,
            o.stop_reason,
        )
        for o in outputs
    ]


def _poc_s1_steps(events, params, cancel_after=None):
    from tensorrt_llm.serve.detokenize import IncrementalDetokenizer
    from tensorrt_llm.serve.openengine_poc.adapter import OpenEngineRequestOutput

    tokenizer = tiny_tokenizer()
    detokenizers = [
        IncrementalDetokenizer(
            tokenizer,
            stop=params.stop,
            include_stop_str_in_output=params.include_stop_str_in_output,
            skip_special_tokens=params.skip_special_tokens,
            spaces_between_special_tokens=params.spaces_between_special_tokens,
        )
        if params.detokenize
        else None
        for _ in range(params.n)
    ]
    output = OpenEngineRequestOutput(_ListStream([_to_proto(e) for e in events]), [2], detokenizers)

    async def run():
        steps = []
        async for step in output:
            steps.append(_snapshot(step.outputs))
            if cancel_after is not None and len(steps) == cancel_after:
                output.abort()
        return steps, output.cached_tokens

    return asyncio.run(run())


def _default_steps(events, params, cancel_after=None):
    assembler = contracts.default_create_response_assembler(
        sampling_params=params, tokenizer=tiny_tokenizer(), num_outputs=params.n
    )
    steps = []
    for event in events:
        if assembler.apply(event):
            steps.append(_snapshot(assembler.outputs))
            if cancel_after is not None and len(steps) == cancel_after:
                assembler.finish_cancelled()
                steps.append(_snapshot(assembler.outputs))
                break
        if assembler.finished:
            break
    return steps, assembler.cached_tokens


FINAL_USAGE = Usage(prompt_tokens=1, completion_tokens=3, cached_prompt_tokens=4)


@pytest.mark.parametrize(
    "events, params",
    [
        pytest.param(
            [
                PromptInfo(),
                TokenDelta(0, (2, 3)),
                TokenDelta(0, (4,)),
                OutputFinished(0, "length", usage=FINAL_USAGE),
            ],
            SamplingParams(),
            id="length",
        ),
        pytest.param(
            [TokenDelta(0, (2, 3)), TokenDelta(0, (4, 5)), OutputFinished(0, "length")],
            SamplingParams(stop=[" world"]),
            id="serve-side-stop-string",
        ),
        pytest.param(
            [
                TokenDelta(0, (2,)),
                OutputFinished(0, "stop", eos_token_id=TINY_EOS, usage=FINAL_USAGE),
            ],
            SamplingParams(),
            id="eos",
        ),
        pytest.param(
            [TokenDelta(0, (2,)), OutputFinished(0, "stop", stop_token_id=5)],
            SamplingParams(),
            id="stop-token",
        ),
        pytest.param(
            [TokenDelta(0, (2,)), OutputFinished(0, "stop", stop_text="llo")],
            SamplingParams(),
            id="engine-stop-text",
        ),
        pytest.param(
            [
                TokenDelta(0, (2,)),
                TokenDelta(1, (4,)),
                TokenDelta(0, (3,)),
                OutputFinished(1, "length"),
                OutputFinished(0, "length", usage=FINAL_USAGE),
            ],
            SamplingParams(n=2, best_of=2, temperature=1.0),
            id="two-outputs",
        ),
        pytest.param(
            [TokenDelta(0, (2, 3)), OutputFinished(0, "length")],
            SamplingParams(detokenize=False),
            id="no-detokenize",
        ),
    ],
)
def test_default_assembler_matches_poc_s1(events, params):
    assert _default_steps(events, params) == _poc_s1_steps(events, params)


def test_default_assembler_cancel_matches_poc_s1():
    events = [TokenDelta(0, (2,)), TokenDelta(0, (3,)), OutputFinished(0, "length")]
    params = SamplingParams(stop=["llo!"])
    expected = _poc_s1_steps(events, params, cancel_after=1)
    got = _default_steps(events, params, cancel_after=1)
    assert got == expected
    assert got[0][-1][0][4] == "cancelled"


# --- create_postproc_dispatch ---------------------------------------------------


@pytest.mark.parametrize(
    "workers, visual_gen, enabled", [(0, False, False), (2, False, True), (2, True, False)]
)
def test_default_postproc_dispatch(workers, visual_gen, enabled):
    generator = SimpleNamespace(args=SimpleNamespace(num_postprocess_workers=workers))
    dispatch = contracts.create_postproc_dispatch(generator, is_visual_gen=visual_gen)
    assert dispatch.enabled is enabled
    params = PostprocParams(
        post_processor=lambda out, args: ("local", out), postproc_args=PostprocArgs()
    )
    assert dispatch.generate_params(params) is (params if enabled else None)

    promise = SimpleNamespace(prompt_token_ids=[1, 2, 3])
    args = PostprocArgs()
    dispatch.prepare(args, promise, tokenizer="tok")
    assert (args.tokenizer, args.num_prompt_tokens) == ((None, None) if enabled else ("tok", 3))
    args = PostprocArgs()
    dispatch.prepare(args, promise)
    assert args.tokenizer is None

    output = SimpleNamespace(outputs=[SimpleNamespace(_postprocess_result="worker")])
    assert dispatch.result(output, params) == ("worker" if enabled else ("local", output))


def test_visual_gen_needs_no_args():
    dispatch = contracts.create_postproc_dispatch(SimpleNamespace(), is_visual_gen=True)
    assert not dispatch.enabled


# --- launch_frontends -----------------------------------------------------------


def test_default_launch_calls_the_single_frontend_launcher():
    from tensorrt_llm.serve.openengine_poc import launch

    config = contracts.FrontendLaunchConfig(
        host="h", port=1, llm_args={"model": "m"}, endpoint="e:2", tool_parser="t"
    )
    with mock.patch.object(launch, "launch_openengine_server") as launcher:
        contracts.launch_frontends(config)
    launcher.assert_called_once_with(
        "h",
        1,
        {"model": "m"},
        "e:2",
        tool_parser="t",
        middleware=(),
        chat_template=None,
        served_model_name=None,
        allow_request_chat_template=False,
        multimodal_server_config=None,
        num_input_processor_workers=8,
        num_media_load_workers=8,
    )
    with mock.patch.object(launch, "launch_openengine_server") as launcher:
        contracts.launch_frontends(dataclasses.replace(config, ready_timeout=5.0))
    assert launcher.call_args.kwargs["ready_timeout"] == 5.0


def test_default_launch_rejects_several_frontends():
    config = contracts.FrontendLaunchConfig(
        host="h", port=1, llm_args={"num_serve_frontends": 2}, endpoint="e:2"
    )
    with pytest.raises(click.UsageError, match="single serving frontend"):
        contracts.launch_frontends(config)


def test_default_launch_rejects_a_report_address():
    config = contracts.FrontendLaunchConfig(
        host="h", port=1, llm_args={}, endpoint="e:2", report_addr="/tmp/addr"
    )
    with pytest.raises(click.UsageError, match="--report_addr"):
        contracts.launch_frontends(config)


# --- metrics, media, LoRA -------------------------------------------------------


def test_default_metrics_record_nothing():
    recorder = contracts.create_metrics_recorder(SimpleNamespace())
    request = recorder.start_request("r", 0.0)
    request.on_event(TokenDelta(0, (1,)), 1.0)
    assert request.metrics_dict() == {}
    assert recorder.iteration_stats(None) is None
    with pytest.raises(contracts.OpenEngineUnsupportedError):
        asyncio.run(recorder.kv_cache_events(FakeEngineClient(), None))


@pytest.mark.parametrize(
    "key", ["multi_modal_data", "multi_modal_embeddings", "mm_processor_kwargs"]
)
def test_default_media_rejects_multimodal_inputs(key):
    caps = conv.ModelCapabilities(served_model_name="m")
    assert contracts.map_media({"prompt": "x"}, caps, engine_name="e") == []
    with pytest.raises(ValueError, match="multimodal"):
        contracts.map_media({key: {}}, caps, engine_name="e")


def test_default_lora_registry_rejects_adapters():
    client = FakeEngineClient()
    registry = contracts.create_lora_registry(client, client.model_info())
    assert asyncio.run(registry.resolve(None)) is None
    with pytest.raises(ValueError, match="lora_request"):
        asyncio.run(registry.resolve(object()))


# --- postprocessing messages ----------------------------------------------------


def test_post_processor_path_round_trip():
    path = contracts.post_processor_path(chat_stream_post_processor)
    assert path == "tensorrt_llm.serve.postprocess_handlers.chat_stream_post_processor"
    assert contracts.resolve_post_processor(path) is chat_stream_post_processor


def test_post_processor_path_rejects_what_cannot_be_imported_back():
    def local(rsp, args):
        return rsp

    for function in (local, lambda rsp, args: rsp, json.dumps):
        with pytest.raises(contracts.PostprocDataError):
            contracts.post_processor_path(function)
    with pytest.raises(contracts.PostprocDataError, match="must come from"):
        contracts.resolve_post_processor("os.system")
    with pytest.raises(contracts.PostprocDataError, match="not a function"):
        contracts.resolve_post_processor("tensorrt_llm.serve.postprocess_handlers.nope")
    # A class is callable but not a postprocessor.
    with pytest.raises(contracts.PostprocDataError, match="not a function"):
        contracts.resolve_post_processor("tensorrt_llm.serve.openengine_poc.contracts.DetokOptions")
    with pytest.raises(contracts.PostprocDataError, match="cannot import"):
        contracts.resolve_post_processor("tensorrt_llm.serve.no_such_module.post")
    # Imported into the module, not defined there.
    with pytest.raises(contracts.PostprocDataError, match="not defined in"):
        contracts.resolve_post_processor(
            "tensorrt_llm.serve.postprocess_handlers.make_tool_call_id"
        )


def test_check_plain_data_accepts_plain_postproc_args():
    args = PostprocArgs(num_prompt_tokens=3)
    contracts.check_plain_data(args)
    contracts.check_plain_data({"a": [1, 2.0, "x", None, (True,)], "b": {1, 2}})
    model = conv.ModelCapabilities(served_model_name="m")
    contracts.check_plain_data(model)


@pytest.mark.parametrize(
    "value, where",
    [
        (PostprocArgs(tokenizer=object()), "postproc_args.tokenizer"),
        ({"cb": lambda: None}, "postproc_args['cb']"),
        ([1, [os]], "postproc_args[1][0]"),
        (SamplingParams(logits_processor=lambda *a: None), "postproc_args.logits_processor"),
    ],
)
def test_check_plain_data_names_the_offending_value(value, where):
    with pytest.raises(contracts.PostprocDataError) as error:
        contracts.check_plain_data(value)
    assert str(error.value).startswith(where + " is a ")


class _ExtraModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    name: str = "x"


def test_check_plain_data_checks_attributes_beyond_the_declared_fields():
    args = PostprocArgs()
    args.parser = object()
    with pytest.raises(contracts.PostprocDataError, match=r"^postproc_args\.parser is a "):
        contracts.check_plain_data(args)
    with pytest.raises(contracts.PostprocDataError, match=r"^postproc_args\.hook is a "):
        contracts.check_plain_data(_ExtraModel(hook=object()))
    contracts.check_plain_data(_ExtraModel(note="plain"))


_Point = collections.namedtuple("_Point", "x y")


class _TaggedStr(str):
    pass


def test_check_plain_data_rejects_container_subclasses():
    with pytest.raises(
        contracts.PostprocDataError, match=r"^postproc_args\['d'\] is a defaultdict"
    ):
        contracts.check_plain_data({"d": collections.defaultdict(lambda: 0)})
    contracts.check_plain_data([_Point(1, 2)])
    # Subclasses that pickle extra state, or a class pickle cannot find by name.
    local_point = collections.namedtuple("LocalPoint", "x y")
    with pytest.raises(contracts.PostprocDataError, match="LocalPoint"):
        contracts.check_plain_data([local_point(1, 2)])
    with pytest.raises(contracts.PostprocDataError, match="_TaggedStr"):
        contracts.check_plain_data({"s": _TaggedStr("x")})


def test_check_plain_data_needs_classes_pickle_can_import():
    class LocalColor(enum.Enum):
        RED = 1

    @dataclasses.dataclass
    class LocalRecord:
        x: int = 1

    for value in (LocalColor.RED, LocalRecord()):
        with pytest.raises(contracts.PostprocDataError, match="cannot import by name"):
            contracts.check_plain_data([value])


@dataclasses.dataclass
class _Lazy:
    x: int = 1
    later: Any = dataclasses.field(init=False)


@dataclasses.dataclass(frozen=True, slots=True)
class _FrozenLazy:
    x: int = 1
    later: Any = dataclasses.field(init=False)


class _Hooks(enum.Enum):
    PLAIN = 1
    CALLABLE = (print,)


@dataclasses.dataclass(frozen=True)
class _FrozenLazyChild(_FrozenLazy):
    pass


@dataclasses.dataclass(slots=True)
class _SlottedLazy:
    x: int = 1
    later: Any = dataclasses.field(init=False)


def test_check_plain_data_and_an_unset_init_false_field():
    # The default pickling skips an unset field.
    for value in (_Lazy(), _SlottedLazy()):
        contracts.check_plain_data(value)
        pickle.loads(pickle.dumps(value))
    # A frozen dataclass with __slots__, and its subclasses, read every field.
    for value in (_FrozenLazy(), _FrozenLazyChild()):
        with pytest.raises(contracts.PostprocDataError, match=r"\.later is not set"):
            contracts.check_plain_data(value)
        with pytest.raises(AttributeError):
            pickle.dumps(value)


def test_check_plain_data_checks_enum_values():
    contracts.check_plain_data([_Hooks.PLAIN])
    with pytest.raises(contracts.PostprocDataError, match=r"\[0\]\.value\[0\]"):
        contracts.check_plain_data([_Hooks.CALLABLE])


def test_check_plain_data_rejects_a_tensor():
    with pytest.raises(contracts.PostprocDataError, match="Tensor"):
        contracts.check_plain_data({"bias": torch.zeros(2)})


def test_pp_messages_hold_plain_data():
    detok = contracts.DetokOptions(stop=("x",), min_tokens=2)
    messages = [
        contracts.PPOpen(
            "r",
            True,
            1,
            contracts.post_processor_path(chat_stream_post_processor),
            PostprocArgs(),
            detok,
        ),
        contracts.PPEvent("r", TokenDelta(0, (1, 2))),
        contracts.PPClose("r", "done"),
        contracts.PPResult("r", ["data: x\n\n"], True, stop_hit=(0,), usage_fix=FINAL_USAGE),
    ]
    for message in messages:
        contracts.check_plain_data(message, path="message")


# --- GenerateCounter ------------------------------------------------------------


def test_counter_writes_one_file_per_process(tmp_path):
    counter = contracts.GenerateCounter(tmp_path)
    assert counter.value == 0 and list(tmp_path.iterdir()) == []
    assert counter.increment() == 1
    assert counter.increment(2) == 3
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == [f"{os.getpid()}.json"]
    assert json.loads((tmp_path / files[0]).read_text()) == {
        "pid": os.getpid(),
        "generate_sent": 3,
    }
    (tmp_path / "99999999.json").write_text(json.dumps({"pid": 99999999, "generate_sent": 4}))
    assert contracts.read_generate_sent(tmp_path) == 7
    with pytest.raises(ValueError):
        counter.increment(-1)


def test_counter_creates_its_directory_and_leaves_no_temporary_file(tmp_path):
    counter_dir = tmp_path / "new" / "counters"
    contracts.GenerateCounter(counter_dir).increment()
    assert [p.name for p in counter_dir.iterdir()] == [f"{os.getpid()}.json"]
    assert contracts.read_generate_sent(counter_dir) == 1


def test_counter_recreates_a_removed_directory(tmp_path):
    counter_dir = tmp_path / "counters"
    counter = contracts.GenerateCounter(counter_dir)
    counter.increment()
    (counter_dir / f"{os.getpid()}.json").unlink()
    counter_dir.rmdir()
    counter.increment()
    assert contracts.read_generate_sent(counter_dir) == 2


def test_counter_counts_a_call_whose_write_failed(tmp_path, monkeypatch):
    counter = contracts.GenerateCounter(tmp_path)
    counter.increment()

    def fail(*args, **kwargs):
        raise PermissionError("read-only")

    with monkeypatch.context() as patch:
        patch.setattr(contracts.tempfile, "mkstemp", fail)
        with pytest.raises(PermissionError):
            counter.increment()
    assert counter.value == 2 and contracts.read_generate_sent(tmp_path) == 1
    # The next write brings the file up to date.
    assert counter.increment() == 3
    assert contracts.read_generate_sent(tmp_path) == 3


def test_counter_directory_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(conv.SERVE_COUNTER_DIR_ENV, str(tmp_path))
    contracts.GenerateCounter().increment()
    assert contracts.read_generate_sent(tmp_path) == 1
    monkeypatch.delenv(conv.SERVE_COUNTER_DIR_ENV)
    counter = contracts.GenerateCounter()
    counter.increment()
    assert counter.value == 1 and contracts.read_generate_sent(tmp_path) == 1
