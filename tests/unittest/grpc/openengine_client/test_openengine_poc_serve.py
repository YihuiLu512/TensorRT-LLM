# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``OpenAIServer`` on the ``OpenEngineLLM`` facade, against the fake OpenEngine engine on CPU."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from types import SimpleNamespace
from typing import Any
from unittest import mock

import click
import grpc
import httpx
import pytest
import torch
from fake_openengine import PROFILES, Delay, Fail, FakeEngine, FakeProfile, Hang
from google.protobuf import struct_pb2
from openengine_client_helpers import WAIT_SECONDS, engine_error, token
from tokenizers import Tokenizer
from tokenizers.decoders import ByteFallback, Fuse
from tokenizers.decoders import Sequence as DecoderSequence
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.commands import serve as serve_cmd
from tensorrt_llm.grpc.openengine.bindings import (
    error_pb2,
    generation_pb2,
    lifecycle_pb2,
    model_pb2,
)
from tensorrt_llm.grpc.openengine_client import (
    OpenEngineIncompatibleServerError,
    OpenEngineRpcError,
)
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_STOP_TEXT,
    CONVENTIONS_VERSION,
    CONVENTIONS_VERSION_KEY,
    META_PRIORITY,
    META_TARGET_DP_RANK,
    META_TRACEPARENT,
    SERVE_COUNTER_DIR_ENV,
    encode_priority,
)
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams
from tensorrt_llm.scheduling_params import SchedulingParams
from tensorrt_llm.serve.detokenize import IncrementalDetokenizer
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.openengine_poc import OpenEngineLLM, build_generate_request, contracts
from tensorrt_llm.serve.openengine_poc import facade as facade_module
from tensorrt_llm.serve.openengine_poc import launch as launch_module
from tensorrt_llm.serve.openengine_poc import mapping as mapping_module
from tensorrt_llm.serve.openengine_poc.engine_client import OutputFinished
from tensorrt_llm.serve.openengine_poc.facade import (
    UNSUPPORTED_ENDPOINTS,
    OpenEngineServeError,
    OpenEngineStartupError,
    adapt_server,
    check_frontend_options,
)
from tensorrt_llm.tokenizer import TransformersTokenizer
from tensorrt_llm.usage import usage_lib

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

MODEL = "stub-model"
MAX_CONTEXT = 64
_VOCAB = {"<unk>": 0, "</s>": 1, "He": 2, "llo": 3, " world": 4, "!": 5, " again": 6}
UNK, EOS, HE, LLO, WORLD, BANG, AGAIN = range(7)
PROMPT_IDS = [2, 3, 5]


def _hf_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel(_VOCAB, unk_token="<unk>"))
    backend.decoder = DecoderSequence([ByteFallback(), Fuse()])
    backend.add_special_tokens(["</s>"])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    hf.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return hf


def _tokenizer() -> TransformersTokenizer:
    return TransformersTokenizer(_hf_tokenizer())


def _model_info(**extra_caps: bool) -> model_pb2.ModelInfo:
    info = model_pb2.ModelInfo(
        model_id=MODEL,
        served_model_name=MODEL,
        max_context_length=MAX_CONTEXT,
        supports_token_ids_input=True,
    )
    if extra_caps:
        extra = struct_pb2.Struct()
        extra.update({f"trtllm.sampling.{k}": v for k, v in extra_caps.items()})
        info.extra.CopyFrom(extra)
    return info


def _finished(
    reason: int = generation_pb2.FINISH_REASON_STOP,
    eos: bool = True,
    index: int = 0,
    terminal: bool = True,
) -> generation_pb2.GenerateResponse:
    event = generation_pb2.GenerationFinished(output_index=index, reason=reason)
    if eos and reason == generation_pb2.FINISH_REASON_STOP:
        event.stop_match.eos_token_id = EOS
    response = generation_pb2.GenerateResponse(finished=event)
    if terminal:
        # Only the response that ends the call carries usage.
        response.usage.CopyFrom(
            generation_pb2.Usage(prompt_tokens=3, completion_tokens=3, total_tokens=6)
        )
    return response


class _InputProcessor:
    """Records the rendered prompt text and returns fixed prompt ids."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: dict[str, Any], sampling_params: SamplingParams):
        self.prompts.append(prompt["prompt"])
        return list(PROMPT_IDS), None


def _args() -> SimpleNamespace:
    return SimpleNamespace(
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


def _facade(engine: FakeEngine, tmp_path, **kwargs: Any) -> OpenEngineLLM:
    return OpenEngineLLM(
        args=_args(),
        tokenizer=_tokenizer(),
        input_processor=_InputProcessor(),
        hf_model_dir=str(tmp_path),
        endpoint=engine.target,
        ready_timeout=WAIT_SECONDS,
        **kwargs,
    )


def _serve(engine: FakeEngine, tmp_path, body, **facade_kwargs: Any) -> Any:
    """Run ``body(engine, facade, http)`` with the server wired to ``engine``.

    ``facade_kwargs`` go to :class:`OpenEngineLLM`.
    """

    async def main() -> Any:
        async with engine:
            facade = _facade(engine, tmp_path, **facade_kwargs)
            await facade.start()
            previous_media_executor = BaseMediaIO._executor
            server = OpenAIServer(
                generator=facade,
                model=MODEL,
                tool_parser=None,
                server_role=None,
                metadata_server_cfg=None,
            )
            adapt_server(server, facade)
            BaseMediaIO.set_executor(previous_media_executor)
            server._media_load_executor.shutdown(wait=False)
            try:
                transport = httpx.ASGITransport(app=server.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                    return await asyncio.wait_for(body(engine, facade, http), WAIT_SECONDS * 2)
            finally:
                await facade.aclose()
                server._input_proc_executor.shutdown(wait=False)
                server._metrics_stats_executor.shutdown(wait=False)

    return asyncio.run(main())


def _profile() -> FakeProfile:
    """``trtllm-like`` serving this file's model, declaring none of the gated sampling fields.

    Nor stop strings: the frontend matches them, as these tests expect.
    """
    return PROFILES["trtllm-like"].replace(
        model=MODEL,
        max_context_length=MAX_CONTEXT,
        vocab_size=len(_VOCAB),
        eos_token_ids=(EOS,),
        bos_token_id=None,
        capabilities={
            **{
                f"trtllm.sampling.{name}": False
                for name in ("min_p", "frequency_penalty", "presence_penalty", "repetition_penalty")
            },
            CAP_STOP_TEXT: False,
        },
    )


def _engine(*steps: Any, profile: FakeProfile | None = None) -> FakeEngine:
    """A fake engine that answers every accepted request with ``steps``.

    Without ``steps`` it answers with its built-in generator.
    """
    return FakeEngine(profile or _profile(), script=list(steps) or None)


def _sse_chunks(text: str) -> list[dict[str, Any]]:
    chunks = []
    for line in text.splitlines():
        if line.startswith("data: ") and line.strip() != "data: [DONE]":
            chunks.append(json.loads(line[len("data: ") :]))
    return chunks


def _chat(stream: bool, **fields: Any) -> dict[str, Any]:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Hi"}], "stream": stream}
    body.update(fields)
    return body


# -- Request mapping ------------------------------------------------------------


def _chat_sampling(**fields: Any) -> SamplingParams:
    request = ChatCompletionRequest(**_chat(False, **fields))
    return request.to_sampling_params(backend="pytorch")


def test_mapping_leaves_server_filled_defaults_unset() -> None:
    request = build_generate_request(
        PROMPT_IDS, _chat_sampling(), _model_info(), engine_name="stub"
    )
    assert list(request.token_ids.ids) == PROMPT_IDS
    assert request.model == MODEL
    assert request.sampling.ListFields() == []
    assert request.stopping.max_tokens == MAX_CONTEXT - len(PROMPT_IDS)
    assert request.extra.fields["detokenize"].bool_value is False
    assert not request.stopping.conditions


def test_mapping_sends_explicit_values_and_keeps_stop_strings_out() -> None:
    sampling = _chat_sampling(
        temperature=0.0,
        top_p=1.0,
        seed=7,
        max_completion_tokens=5,
        stop=[" world"],
        stop_token_ids=[EOS, BANG, EOS],
    )
    request = build_generate_request(PROMPT_IDS, sampling, _model_info(), engine_name="stub")
    assert request.sampling.HasField("temperature") and request.sampling.temperature == 0.0
    assert request.sampling.HasField("top_p")
    assert request.sampling.seed == 7
    assert request.stopping.max_tokens == 5
    assert [c.stop_token_id for c in request.stopping.conditions] == [EOS, BANG]
    assert all(c.WhichOneof("condition") == "stop_token_id" for c in request.stopping.conditions)


@pytest.mark.parametrize(
    "sampling, message",
    [
        (lambda: _chat_sampling(presence_penalty=0.5), "presence_penalty"),
        (lambda: _chat_sampling(n=2), "n=2"),
        (lambda: _chat_sampling(logprobs=True), "logprobs"),
        (
            lambda: SamplingParams(guided_decoding=GuidedDecodingParams(regex="a+")),
            "guided_decoding",
        ),
        (lambda: SamplingParams(max_tokens=4, length_penalty=1.2), "length_penalty"),
        (lambda: SamplingParams(embedding_bias=torch.zeros(8)), "logit_bias"),
    ],
)
def test_mapping_rejects_what_the_path_or_the_engine_does_not_support(
    sampling, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_generate_request(PROMPT_IDS, sampling(), _model_info(), engine_name="stub")


def test_mapping_sends_a_declared_penalty() -> None:
    sampling = _chat_sampling(presence_penalty=0.5)
    request = build_generate_request(
        PROMPT_IDS, sampling, _model_info(presence_penalty=True), engine_name="stub"
    )
    assert request.sampling.presence_penalty == 0.5


@pytest.mark.parametrize("value", [0.0, 0.5])
def test_mapping_checks_an_explicit_penalty_against_the_declared_values(value) -> None:
    """An explicit value is sent as given when the engine accepts it, even a neutral one."""
    info = _model_info()
    info.extra.update(
        {"trtllm.params": {"presence_penalty": {"accept": "values", "values": [0.0]}}}
    )
    sampling = _chat_sampling(presence_penalty=value)
    if value == 0.0:
        request = build_generate_request(PROMPT_IDS, sampling, info, engine_name="stub")
        assert request.sampling.HasField("presence_penalty")
        assert request.sampling.presence_penalty == 0.0
    else:
        with pytest.raises(ValueError, match=r"presence_penalty=0\.5 .*it accepts only 0\.0"):
            build_generate_request(PROMPT_IDS, sampling, info, engine_name="stub")
    # Without the declaration (as from an engine that reports no conventions), nothing is accepted.
    with pytest.raises(ValueError, match="presence_penalty.*it accepts no value"):
        build_generate_request(PROMPT_IDS, sampling, _model_info(), engine_name="stub")


def test_mapping_caps_max_tokens_at_the_room_left_in_the_context(monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr(mapping_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    request = build_generate_request(
        PROMPT_IDS, SamplingParams(max_tokens=1000), _model_info(), engine_name="stub"
    )
    assert request.stopping.max_tokens == MAX_CONTEXT - len(PROMPT_IDS)
    assert any("max_tokens=1000" in w for w in warnings)
    with pytest.raises(ValueError, match="max_tokens"):
        build_generate_request(
            PROMPT_IDS, SamplingParams(max_tokens=0), _model_info(), engine_name="stub"
        )


def test_mapping_rejects_a_prompt_that_fills_the_context() -> None:
    with pytest.raises(ValueError, match=f"maximum context length is {MAX_CONTEXT} tokens"):
        build_generate_request(
            list(range(MAX_CONTEXT)), SamplingParams(), _model_info(), engine_name="stub"
        )


# -- OpenAIServer end to end ----------------------------------------------------


def test_streaming_chat_end_to_end(tmp_path) -> None:
    engine = _engine(token(0, HE), token(0, LLO, WORLD), _finished())

    async def body(engine, facade, http):
        response = await http.post("/v1/chat/completions", json=_chat(True))
        assert response.status_code == 200
        chunks = _sse_chunks(response.text)
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        assert content == "Hello world"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert response.text.rstrip().endswith("data: [DONE]")
        assert facade.input_processor.prompts == ["Hi"]
        assert list(engine.requests[0].token_ids.ids) == PROMPT_IDS
        assert facade._check_health()

    _serve(engine, tmp_path, body)


def test_stop_string_truncates_and_cancels_the_engine_request(tmp_path) -> None:
    engine = _engine(token(0, HE), token(0, LLO), token(0, WORLD, AGAIN), Hang())

    async def body(engine, facade, http):
        response = await http.post("/v1/chat/completions", json=_chat(False, stop=[" world"]))
        assert response.status_code == 200, response.text
        choice = response.json()["choices"][0]
        assert choice["message"]["content"] == "Hello"
        assert choice["finish_reason"] == "stop"
        assert choice["stop_reason"] == " world"
        # As with LLM, the token matched as the stop string is not counted.
        assert response.json()["usage"]["completion_tokens"] == 2
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        assert not engine.requests[0].stopping.conditions

    _serve(engine, tmp_path, body)


def test_completion_from_token_ids_reports_length(tmp_path) -> None:
    engine = _engine(
        token(0, WORLD), token(0, BANG), _finished(generation_pb2.FINISH_REASON_LENGTH)
    )

    async def body(engine, facade, http):
        response = await http.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": PROMPT_IDS, "max_tokens": 2, "temperature": 0},
        )
        assert response.status_code == 200, response.text
        choice = response.json()["choices"][0]
        assert choice["text"] == " world!"
        assert choice["finish_reason"] == "length"
        assert response.json()["usage"]["prompt_tokens"] == len(PROMPT_IDS)
        sent = engine.requests[0]
        assert list(sent.token_ids.ids) == PROMPT_IDS
        assert sent.stopping.max_tokens == 2
        assert sent.sampling.temperature == 0.0

    _serve(engine, tmp_path, body)


def _expected_request(
    max_tokens: int, ignore_eos: bool | None = None
) -> generation_pb2.GenerateRequest:
    """The whole request a greedy call with ``max_tokens`` must reach the engine as."""
    stopping = generation_pb2.StoppingOptions(max_tokens=max_tokens)
    if ignore_eos is not None:
        stopping.ignore_eos = ignore_eos
    extra = struct_pb2.Struct()
    extra.update({"detokenize": False})
    return generation_pb2.GenerateRequest(
        model=MODEL,
        token_ids=generation_pb2.TokenIds(ids=PROMPT_IDS),
        sampling=generation_pb2.SamplingParams(temperature=0.0),
        stopping=stopping,
        extra=extra,
    )


@pytest.mark.parametrize(
    "path,payload,expected",
    [
        (
            "/v1/completions",
            {"prompt": PROMPT_IDS, "max_tokens": 16, "ignore_eos": True, "detokenize": False},
            _expected_request(16, ignore_eos=True),
        ),
        ("/v1/completions", {"prompt": PROMPT_IDS, "max_tokens": 16}, _expected_request(16)),
        (
            "/v1/completions",
            {"prompt": "Hello", "max_tokens": 16, "stop": ["Question:", "</s>"]},
            _expected_request(16),
        ),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 16, "ignore_eos": True},
            _expected_request(16, ignore_eos=True),
        ),
    ],
    ids=["token-ids-ignore-eos", "token-ids-eos", "text-with-stop-strings", "chat"],
)
def test_engine_receives_exactly_the_expected_greedy_request(
    tmp_path, path, payload, expected
) -> None:
    """The ``GenerateRequest`` the engine receives equals ``expected`` field for field.

    Only the client-assigned ``request_id`` is left out of the comparison. Stop
    strings are matched in the server and are not sent, and every request asks
    for token ids only (``extra.detokenize = false``).
    """
    engine = _engine(token(0, WORLD), _finished(generation_pb2.FINISH_REASON_LENGTH))

    async def body(engine, facade, http):
        response = await http.post(path, json={"model": MODEL, "temperature": 0, **payload})
        assert response.status_code == 200, response.text
        sent = generation_pb2.GenerateRequest()
        sent.CopyFrom(engine.requests[0])
        assert sent.request_id
        sent.ClearField("request_id")
        assert sent == expected

    _serve(engine, tmp_path, body)


def test_unsupported_parameter_is_rejected_before_the_engine_sees_it(tmp_path) -> None:
    engine = _engine(_finished())

    async def body(engine, facade, http):
        response = await http.post("/v1/chat/completions", json=_chat(True, presence_penalty=0.5))
        assert response.status_code == 400
        assert "presence_penalty" in response.text
        assert engine.requests == []

    _serve(engine, tmp_path, body)


@pytest.mark.parametrize(
    "code,error,status",
    [
        (grpc.StatusCode.INVALID_ARGUMENT, ValueError, 400),
        (grpc.StatusCode.UNIMPLEMENTED, ValueError, 400),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, OpenEngineServeError, 503),
        (grpc.StatusCode.UNAVAILABLE, OpenEngineServeError, 503),
        (grpc.StatusCode.DEADLINE_EXCEEDED, OpenEngineServeError, 504),
        (grpc.StatusCode.INTERNAL, OpenEngineServeError, 500),
    ],
)
def test_engine_refusal_fails_the_request(tmp_path, code, error, status) -> None:
    """A request the engine refuses before accepting it fails with the engine's message.

    The request was at fault for INVALID_ARGUMENT and UNIMPLEMENTED, so those
    become ``ValueError`` and HTTP 400; the other statuses keep their own HTTP
    status.
    """
    engine = _engine(Fail(code, "engine refused it"))

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=2))
        with pytest.raises(error, match="engine refused it"):
            await output.aresult()
        assert engine.counters["trtllm.counters.generate_accepted"] == 0
        response = await http.post("/v1/completions", json={"model": MODEL, "prompt": PROMPT_IDS})
        assert response.status_code == status
        assert "engine refused it" in response.text
        # The engine is named, its address is not.
        assert engine.target not in response.text

    _serve(engine, tmp_path, body)


@pytest.mark.parametrize(
    "code,status",
    [
        (error_pb2.ERROR_CODE_INVALID_ARGUMENT, 400),
        (error_pb2.ERROR_CODE_OVERLOADED, 503),
        (error_pb2.ERROR_CODE_INTERNAL, 500),
    ],
)
def test_engine_error_fails_the_request(tmp_path, code: int, status: int) -> None:
    engine = _engine(engine_error(code, retryable=False))

    async def body(engine, facade, http):
        response = await http.post("/v1/chat/completions", json=_chat(False))
        assert response.status_code == status
        assert "engine gave up" in response.text
        expected_type = {
            400: "BadRequestError",
            503: "ServiceUnavailableError",
            500: "InternalServerError",
        }[status]
        assert response.json()["type"] == expected_type

    _serve(engine, tmp_path, body)


def test_a_grpc_peer_address_is_not_reported_to_the_client() -> None:
    details = "failed to connect to all addresses; last error: UNKNOWN: ipv4:10.1.2.3:4567: refused"
    error = facade_module.request_error(
        OpenEngineRpcError(
            grpc.StatusCode.UNAVAILABLE, details, rpc="Generate", target="engine-host:4567"
        ),
        "tensorrt_llm",
    )
    assert isinstance(error, OpenEngineServeError) and error.status_code == 503
    assert "10.1.2.3" not in str(error) and "<engine>" in str(error)
    error = facade_module.request_error(
        OpenEngineRpcError(
            grpc.StatusCode.INTERNAL,
            "lost engine-host:4567",
            rpc="Generate",
            target="engine-host:4567",
        ),
        "tensorrt_llm",
    )
    assert "engine-host" not in str(error)


def test_a_cancellation_by_the_engine_is_an_error(tmp_path) -> None:
    engine = _engine(token(0, HE), _finished(generation_pb2.FINISH_REASON_CANCELLED))

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=4))
        with pytest.raises(OpenEngineServeError, match="cancelled the request") as error:
            await output.aresult()
        assert error.value.status_code == 503
        engine.push_script(token(0, HE), _finished(generation_pb2.FINISH_REASON_CANCELLED))
        response = await http.post("/v1/chat/completions", json=_chat(False))
        assert response.status_code == 503, response.text

    _serve(engine, tmp_path, body)


def test_abort_cancels_the_engine_request(tmp_path) -> None:
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        first = await output.__anext__()
        assert first.outputs[0].text_diff == "He"
        output.abort()
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        # As with LLM, an aborted request ends normally with finish reason cancelled.
        result = await output.aresult()
        assert result.finished and result.error is None
        assert result.outputs[0].finish_reason == "cancelled"
        assert result.outputs[0].text == "He"

    _serve(engine, tmp_path, body)


def test_abort_while_waiting_for_the_engine_ends_the_request(tmp_path) -> None:
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        await output.__anext__()
        waiting = asyncio.create_task(output.aresult())
        await asyncio.sleep(0.05)
        assert not waiting.done()
        output.abort()
        result = await asyncio.wait_for(waiting, WAIT_SECONDS)
        assert result.outputs[0].finish_reason == "cancelled"

    _serve(engine, tmp_path, body)


def test_abort_releases_text_held_back_for_a_stop_string(tmp_path) -> None:
    engine = _engine(token(0, HE), token(0, BANG), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8, stop=["! again"]))
        await output.__anext__()
        held = await output.__anext__()
        # "!" could start the stop string, so it is held back.
        assert held.outputs[0].text == "He"
        output.abort()
        result = await output.aresult()
        assert result.outputs[0].finish_reason == "cancelled"
        assert result.outputs[0].text == "He!"
        assert result.outputs[0].token_ids == [HE, BANG]

    _serve(engine, tmp_path, body)


def test_engine_error_in_a_streaming_completion_is_an_error_event(tmp_path) -> None:
    engine = _engine(engine_error(error_pb2.ERROR_CODE_INTERNAL, retryable=False))

    async def body(engine, facade, http):
        response = await http.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": PROMPT_IDS, "max_tokens": 2, "stream": True},
        )
        # The stream has started before the engine is read, so the status stays 200.
        assert response.status_code == 200
        errors = [c for c in _sse_chunks(response.text) if "error" in c]
        assert len(errors) == 1 and "engine gave up" in errors[0]["error"]["message"]

    _serve(engine, tmp_path, body)


async def _steps(output) -> list[tuple[str, list[int], str | None]]:
    """The text and token ids each yield adds, and the finish reason it reports, of output 0."""
    steps = []
    async for step in output:
        out = step.outputs[0]
        steps.append((out.text_diff, list(out.token_ids_diff), out.finish_reason))
    return steps


def test_the_last_tokens_and_the_finish_reason_are_one_chunk(tmp_path) -> None:
    # The engine sends the finish in a message of its own, here well after the last token.
    engine = _engine(
        token(0, HE), token(0, LLO), Delay(0.3), _finished(generation_pb2.FINISH_REASON_LENGTH)
    )

    async def body(engine, facade, http):
        response = await http.post("/v1/chat/completions", json=_chat(True, max_tokens=2))
        assert response.status_code == 200, response.text
        choices = [c["choices"][0] for c in _sse_chunks(response.text) if c["choices"]]
        finishing = [c for c in choices if c.get("finish_reason") is not None]
        assert len(finishing) == 1
        assert finishing[0]["delta"].get("content") == "llo"
        assert finishing[0]["finish_reason"] == "length"
        assert "".join(c["delta"].get("content") or "" for c in choices) == "Hello"

    _serve(engine, tmp_path, body)


def test_a_step_ending_at_max_tokens_waits_for_the_finish(tmp_path) -> None:
    engine = _engine(
        token(0, HE), token(0, LLO), Delay(0.3), _finished(generation_pb2.FINISH_REASON_LENGTH)
    )

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=2), streaming=True)
        assert await _steps(output) == [("He", [HE], None), ("llo", [LLO], "length")]

    _serve(engine, tmp_path, body)


def test_a_finish_before_max_tokens_is_a_step_of_its_own(tmp_path) -> None:
    # Nothing tells the facade that the engine will end the output, so it does not wait.
    engine = _engine(token(0, HE), token(0, LLO), _finished())

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8), streaming=True)
        assert await _steps(output) == [
            ("He", [HE], None),
            ("llo", [LLO], None),
            ("", [], "stop"),
        ]

    _serve(engine, tmp_path, body)


def test_a_step_of_several_outputs_covers_each_one_from_its_start(tmp_path) -> None:
    engine = _engine(
        token(0, HE),
        token(1, WORLD),
        token(0, LLO),
        Delay(0.2),
        token(1, BANG),
        _finished(generation_pb2.FINISH_REASON_LENGTH, index=0, terminal=False),
        _finished(generation_pb2.FINISH_REASON_LENGTH, index=1),
        profile=_profile().replace(max_num_sequences=2),
    )

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(n=2, temperature=1.0, max_tokens=2), streaming=True
        )
        steps = []
        async for step in output:
            steps.append([(o.text_diff, o.finish_reason) for o in step.outputs])
        # Output 0 has all its tokens with "llo": that step goes on until both outputs finished.
        assert steps == [
            [("He", None), ("", None)],
            [("", None), (" world", None)],
            [("llo", "length"), ("!", "length")],
        ]
        assert [o.text for o in output.outputs] == ["Hello", " world!"]

    _serve(engine, tmp_path, body)


def test_another_call_that_ends_does_not_end_a_step_that_waits_for_the_finish(tmp_path) -> None:
    # One sequence per call, so n=2 is two calls; one of them ends while the
    # other's last tokens wait for their finish.
    engine = _engine(profile=_profile().replace(max_num_sequences=1))
    length = generation_pb2.FINISH_REASON_LENGTH
    engine.push_script(token(0, HE), token(0, LLO), Delay(0.4), _finished(length))
    engine.push_script(token(0, WORLD), token(0, BANG), Delay(0.2), _finished(length))

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(n=2, temperature=1.0, max_tokens=2), streaming=True
        )
        steps = []
        async for step in output:
            steps.append([(o.text_diff, o.finish_reason) for o in step.outputs])
        assert len(engine.requests) == 2
        assert sorted(o.text for o in output.outputs) == [" world!", "Hello"]
        for index in range(2):
            # The step that finishes an output also carries its last tokens.
            text, reason = next(s[index] for s in steps if s[index][1] is not None)
            assert text.endswith(("llo", "!")) and reason == "length"

    _serve(engine, tmp_path, body)


def test_a_call_that_fails_while_a_step_waits_for_the_finish_fails_after_that_step(
    tmp_path,
) -> None:
    engine = _engine(token(0, HE), token(0, LLO), Delay(0.2), Fail(grpc.StatusCode.INTERNAL, "x"))

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=2), streaming=True)
        assert (await output.__anext__()).outputs[0].text_diff == "He"
        step = await output.__anext__()
        assert (step.outputs[0].text_diff, step.outputs[0].finish_reason) == ("llo", None)
        with pytest.raises(OpenEngineServeError):
            await output.__anext__()
        assert output.finished and output.error is not None

    _serve(engine, tmp_path, body)


def test_health_is_lost_after_close(tmp_path) -> None:
    engine = _engine(_finished())

    async def body(engine, facade, http):
        assert (await http.get("/health")).status_code == 200
        await facade.aclose()
        assert not facade._check_health()
        assert (await http.get("/health")).status_code != 200

    _serve(engine, tmp_path, body)


def test_health_follows_the_engine(tmp_path) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        engine.set_health(lifecycle_pb2.HEALTH_STATE_NOT_READY)
        while facade._check_health():
            await asyncio.sleep(0.01)
        assert (await http.get("/health")).status_code != 200
        engine.set_health(lifecycle_pb2.HEALTH_STATE_READY)
        while not facade._check_health():
            await asyncio.sleep(0.01)
        assert (await http.get("/health")).status_code == 200

    _serve(engine, tmp_path, body, health_poll_interval=0.01)


def _lost_recorder(lost: list) -> Any:
    def on_engine_lost(facade) -> None:
        lost.append(facade)
        # As the default handler does, so /health does not send SIGINT itself.
        facade._executor.doing_shutdown = True

    return on_engine_lost


def test_a_failed_health_watcher_stops_the_frontend(tmp_path, monkeypatch) -> None:
    lost = []
    engine = _engine()

    async def body(engine, facade, http):
        async def broken_health():
            raise TypeError("unexpected")

        monkeypatch.setattr(facade.client, "health", broken_health)
        while not lost:
            await asyncio.sleep(0.01)
        assert not facade._check_health()
        assert "TypeError" in str(facade._executor._fatal_error)
        assert (await http.get("/health")).status_code == 503

    _serve(
        engine,
        tmp_path,
        body,
        health_poll_interval=0.01,
        on_engine_lost=_lost_recorder(lost),
    )


def test_two_sequences_are_detokenized_separately(tmp_path) -> None:
    engine = _engine(
        token(0, HE),
        token(1, WORLD),
        token(0, LLO),
        _finished(index=0, terminal=False),
        token(1, BANG),
        _finished(generation_pb2.FINISH_REASON_LENGTH, index=1),
        profile=_profile().replace(max_num_sequences=2),
    )

    async def body(engine, facade, http):
        # SamplingParams refuses n > 1 with greedy decoding, so sample.
        output = await facade.generate_async(
            PROMPT_IDS, SamplingParams(n=2, temperature=1.0, max_tokens=2)
        )
        assert [o.text for o in output.outputs] == ["Hello", " world!"]
        assert [o.finish_reason for o in output.outputs] == ["stop", "length"]
        assert engine.requests[0].sampling.num_sequences == 2

    _serve(engine, tmp_path, body)


def test_prompt_over_the_context_reports_context_length_exceeded(tmp_path) -> None:
    engine = _engine(_finished())

    async def body(engine, facade, http):
        response = await http.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": [HE] * MAX_CONTEXT, "max_tokens": 2},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "context_length_exceeded"
        assert engine.requests == []

    _serve(engine, tmp_path, body)


def test_a_failure_while_applying_a_response_cancels_the_engine_request(
    tmp_path, monkeypatch
) -> None:
    engine = _engine(token(0, HE), Hang())

    def fail(self, token_ids):
        raise RuntimeError("detokenizer broke")

    monkeypatch.setattr(IncrementalDetokenizer, "step", fail)

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        with pytest.raises(RuntimeError, match="detokenizer broke"):
            await output.__anext__()
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        assert output.finished and output.error is not None

    _serve(engine, tmp_path, body)


def test_chat_tokenizes_through_preprocess(tmp_path) -> None:
    engine = _engine(token(0, HE), _finished())

    async def body(engine, facade, http):
        calls = []
        original = facade.preprocess

        def preprocess(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        facade.preprocess = preprocess
        response = await http.post("/v1/chat/completions", json=_chat(False))
        assert response.status_code == 200, response.text
        assert len(calls) == 1
        assert list(engine.requests[0].token_ids.ids) == PROMPT_IDS

    _serve(engine, tmp_path, body)


def test_detokenize_false_returns_token_ids_only(tmp_path) -> None:
    engine = _engine(
        token(0, WORLD), token(0, BANG), _finished(generation_pb2.FINISH_REASON_LENGTH)
    )

    async def body(engine, facade, http):
        output = await facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=2, detokenize=False)
        )
        assert output.outputs[0].text == ""
        assert output.outputs[0].token_ids == [WORLD, BANG]
        assert output.outputs[0].finish_reason == "length"
        with pytest.raises(ValueError, match="detokenize"):
            facade.generate_async(
                PROMPT_IDS, SamplingParams(max_tokens=2, detokenize=False, stop=["!"])
            )

    _serve(engine, tmp_path, body)


@pytest.mark.parametrize("declared", [False, None])
def test_engine_without_token_id_input_is_refused(tmp_path, declared) -> None:
    # None leaves the capability undeclared, which counts as unsupported.
    engine = _engine(profile=_profile().replace(supports_token_ids_input=declared))

    async def main() -> None:
        async with engine:
            facade = _facade(engine, tmp_path)
            with pytest.raises(OpenEngineStartupError, match="supports_token_ids_input"):
                await facade.start()

    asyncio.run(main())


def test_facade_needs_a_hugging_face_tokenizer(tmp_path) -> None:
    with pytest.raises(TypeError, match="Hugging Face tokenizer"):
        OpenEngineLLM(
            args=_args(),
            tokenizer=object(),
            input_processor=_InputProcessor(),
            hf_model_dir=str(tmp_path),
            endpoint="localhost:1",
        )


# -- Building from trtllm-serve arguments ---------------------------------------


def test_from_llm_args_loads_the_model_tokenizer_and_keeps_serving_options(tmp_path) -> None:
    _hf_tokenizer().save_pretrained(tmp_path)

    facade = OpenEngineLLM.from_llm_args(
        {"model": str(tmp_path), "num_postprocess_workers": 0, "max_seq_len": 32}, "localhost:1"
    )

    assert isinstance(facade.tokenizer, TransformersTokenizer)
    assert facade.tokenizer.decode([HE, LLO]) == "Hello"
    # The options reach OpenAIServer as given; nothing is overridden.
    assert facade.args.max_seq_len == 32


def test_from_llm_args_wraps_a_bare_tokenizer_from_the_input_processor(
    tmp_path, monkeypatch
) -> None:
    _hf_tokenizer().save_pretrained(tmp_path)
    processor = SimpleNamespace(tokenizer=_hf_tokenizer())
    monkeypatch.setattr(
        "tensorrt_llm.inputs.registry.create_input_processor", lambda *a, **k: processor
    )

    facade = OpenEngineLLM.from_llm_args({"model": str(tmp_path)}, "localhost:1")

    assert isinstance(facade.tokenizer, TransformersTokenizer)
    assert facade.tokenizer.tokenizer is processor.tokenizer


def test_from_llm_args_needs_a_local_model_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="local model directory"):
        OpenEngineLLM.from_llm_args({"model": str(tmp_path / "missing")}, "localhost:1")


def _run_cli(tmp_path, *args: str, diffusion_only: bool = False, endpoint: str = "localhost:1"):
    """Run the CLI with one frontend (unless ``args`` set another count); returns the launch mock."""
    with (
        mock.patch.object(serve_cmd, "get_is_diffusion_only_model", return_value=diffusion_only),
        mock.patch.object(serve_cmd, "device_count", side_effect=AssertionError("no GPU here")),
        mock.patch.object(serve_cmd, "launch_server") as launch_server,
        mock.patch.object(launch_module, "launch_openengine_server") as launch,
        mock.patch.object(usage_lib, "_mark_llm_startup") as mark_startup,
        mock.patch.object(usage_lib, "_capture_startup_context") as capture_startup,
    ):
        serve_cmd.main(
            args=[
                str(tmp_path),
                "--openengine-endpoint",
                endpoint,
                "--num_serve_frontends",
                "1",
                *args,
            ],
            standalone_mode=False,
        )
    launch_server.assert_not_called()
    # No LLM is built, so no LLM startup is recorded.
    mark_startup.assert_not_called()
    capture_startup.assert_not_called()
    return launch


def test_cli_launches_one_openengine_frontend_without_a_gpu(tmp_path) -> None:
    launch = _run_cli(tmp_path)
    launch.assert_called_once()
    llm_args = launch.call_args.args[2]
    assert launch.call_args.args[3] == "localhost:1"
    assert llm_args.get("num_serve_frontends", 1) == 1


@pytest.mark.parametrize(
    "args, name",
    [
        (["--grpc"], "--grpc"),
        (["--server_role", "context"], "--server_role"),
        (["--disagg_cluster_uri", "http://localhost:1"], "--disagg_cluster_uri"),
        (["--enable_visual_gen"], "--enable_visual_gen"),
    ],
)
def test_cli_rejects_options_the_openengine_path_cannot_serve(tmp_path, args, name) -> None:
    with pytest.raises(click.UsageError, match=name):
        _run_cli(tmp_path, *args)


def test_cli_reports_the_sleep_mode_conflict_before_sleep_mode_checks(
    tmp_path, monkeypatch
) -> None:
    # Without the key, sleep mode's own check would fail first.
    monkeypatch.delenv("TRTLLM_RUNTIME_CONTROL_API_KEY", raising=False)
    with pytest.raises(click.UsageError, match="cannot be combined with --enable_sleep_mode"):
        _run_cli(tmp_path, "--enable_sleep_mode")


@pytest.mark.parametrize(
    "args, name, value",
    [
        (
            ["--post_processor_hook", "guardrails:redact"],
            "post_processor_hook",
            "guardrails:redact",
        ),
        (
            ["--otlp_traces_endpoint", "http://localhost:4318"],
            "otlp_traces_endpoint",
            "http://localhost:4318",
        ),
    ],
)
def test_cli_hands_frontend_options_to_the_frontend(tmp_path, args, name, value) -> None:
    """The CLI passes these on; the frontend accepts or refuses them when it starts."""
    launch = _run_cli(tmp_path, *args)
    assert launch.call_args.args[2][name] == value


def test_cli_passes_the_ready_timeout(tmp_path) -> None:
    launch = _run_cli(tmp_path, "--openengine-ready-timeout", "12.5")
    assert launch.call_args.kwargs["ready_timeout"] == 12.5
    assert "ready_timeout" not in _run_cli(tmp_path).call_args.kwargs


def test_cli_passes_the_engine_options_and_the_report_address(tmp_path) -> None:
    launch = _run_cli(
        tmp_path,
        "--openengine-control-timeout",
        "2.5",
        "--openengine-allow-newer-schema",
        "--port",
        "0",
        "--report_addr",
        str(tmp_path / "addr"),
    )
    assert launch.call_args.args[1] == 0
    kwargs = launch.call_args.kwargs
    assert kwargs["control_timeout"] == 2.5
    assert kwargs["allow_newer_schema"] is True
    assert kwargs["report_addr"] == str(tmp_path / "addr")
    defaults = _run_cli(tmp_path).call_args.kwargs
    assert not {"control_timeout", "allow_newer_schema", "report_addr"} & set(defaults)


@pytest.mark.parametrize(
    "args, message",
    [
        (["--openengine-ready-timeout", "0"], "greater than 0"),
        (["--openengine-ready-timeout", "-1"], "greater than 0"),
        (["--openengine-control-timeout", "0"], "greater than 0"),
        (["--openengine-control-timeout", "inf"], "finite number"),
        (["--openengine-ready-timeout", "nan"], "finite number"),
    ],
)
def test_cli_rejects_a_bad_ready_timeout(tmp_path, args, message) -> None:
    with pytest.raises(click.UsageError, match=message):
        _run_cli(tmp_path, *args)


@pytest.mark.parametrize(
    "args",
    [
        ["--openengine-ready-timeout", "5"],
        ["--openengine-control-timeout", "5"],
        ["--openengine-allow-newer-schema"],
    ],
)
def test_cli_rejects_engine_options_without_an_endpoint(tmp_path, args) -> None:
    with pytest.raises(click.UsageError, match="requires --openengine-endpoint"):
        serve_cmd.main(args=[str(tmp_path), *args], standalone_mode=False)


@pytest.mark.parametrize(
    "endpoint",
    [
        "localhost",
        ":8000",
        " :8000",
        "host :8000",
        "localhost:port",
        "localhost:99999",
        "::1",
        "::1:8000",
        "[:8000",
        "[]:8000",
        "[zz:zz]:8000",
        "[[::1]]:8000",
        "host/x:8000",
        "host,y:8000",
    ],
)
def test_cli_rejects_an_endpoint_that_is_not_host_port(tmp_path, endpoint) -> None:
    with pytest.raises(click.UsageError, match="host:port"):
        serve_cmd.main(
            args=[str(tmp_path), "--openengine-endpoint", endpoint], standalone_mode=False
        )


def test_cli_accepts_a_bracketed_ipv6_endpoint(tmp_path) -> None:
    launch = _run_cli(tmp_path, endpoint="[::1]:8000")
    assert launch.call_args.args[3] == "[::1]:8000"


def test_cli_rejects_a_diffusion_only_model(tmp_path) -> None:
    with pytest.raises(click.UsageError, match="VisualGen"):
        _run_cli(tmp_path, diffusion_only=True)


def test_cli_rejects_a_disaggregated_cluster_in_the_config(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("disagg_cluster:\n  cluster_uri: http://localhost:1\n")
    with pytest.raises(click.UsageError, match="disagg_cluster"):
        _run_cli(tmp_path, "--config", str(config))


# -- Facade: startup, engine loss, counting, metadata, endpoints ------------------


def _start_fails(engine: FakeEngine, tmp_path) -> OpenEngineStartupError:
    async def main() -> OpenEngineStartupError:
        async with engine:
            facade = _facade(engine, tmp_path)
            with pytest.raises(OpenEngineStartupError) as error:
                await facade.start()
            return error.value

    return asyncio.run(main())


def test_startup_fails_on_another_conventions_version(tmp_path) -> None:
    other = CONVENTIONS_VERSION + 1
    profile = _profile().replace(server_info_extra={CONVENTIONS_VERSION_KEY: other})
    error = _start_fails(_engine(profile=profile), tmp_path)
    assert "conventions version" in str(error)
    assert f"serve={CONVENTIONS_VERSION} engine={other}" in str(error)


def test_startup_fails_on_a_conventions_version_that_is_not_an_integer(tmp_path) -> None:
    profile = _profile().replace(server_info_extra={CONVENTIONS_VERSION_KEY: "two"})
    error = _start_fails(_engine(profile=profile), tmp_path)
    assert "conventions version" in str(error)
    assert f"serve={CONVENTIONS_VERSION} engine='two'" in str(error)


def test_startup_fails_on_another_tokenizer_and_names_both_values(tmp_path) -> None:
    error = _start_fails(_engine(profile=_profile().replace(vocab_size=99)), tmp_path)
    assert "tokenizer fingerprint" in str(error)
    assert f"vocab_size: serve={len(_VOCAB)} engine=99" in str(error)


def test_startup_continues_with_a_warning_without_the_conventions(tmp_path, monkeypatch) -> None:
    warnings = []
    monkeypatch.setattr(facade_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    engine = _engine(profile=_profile().replace(follows_conventions=False))

    async def body(engine, facade, http):
        assert facade.check_health()

    _serve(engine, tmp_path, body)
    assert any(CONVENTIONS_VERSION_KEY in w for w in warnings)
    assert any("no tokenizer fingerprint" in w for w in warnings)


def test_a_lost_engine_stops_the_frontend_after_the_timeout(tmp_path) -> None:
    lost = []
    engine = _engine()

    async def body(engine, facade, http):
        # Shorter than the timeout: the frontend stays up.
        engine.set_health(lifecycle_pb2.HEALTH_STATE_NOT_READY)
        while facade.check_health():
            await asyncio.sleep(0.01)
        engine.set_health(lifecycle_pb2.HEALTH_STATE_READY)
        while not facade.check_health():
            await asyncio.sleep(0.01)
        assert lost == []
        # Longer than the timeout: the frontend stops with a fatal error.
        await engine.stop()
        while not lost:
            await asyncio.sleep(0.01)
        assert lost == [facade]
        assert "unavailable" in str(facade._executor._fatal_error)
        assert (await http.get("/health")).status_code == 503

    _serve(
        engine,
        tmp_path,
        body,
        health_poll_interval=0.01,
        engine_lost_timeout=2.0,
        on_engine_lost=_lost_recorder(lost),
    )


def test_the_default_engine_lost_handler_sends_one_sigint(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(facade_module.signal, "raise_signal", sent.append)
    facade = SimpleNamespace(_executor=facade_module._FacadeExecutor())
    facade_module._stop_frontend(facade)
    facade_module._stop_frontend(facade)
    assert sent == [facade_module.signal.SIGINT]
    assert facade._executor.doing_shutdown


def test_a_restarted_engine_is_followed(tmp_path, monkeypatch) -> None:
    fingerprints = []
    compute = facade_module.compute_tokenizer_fingerprint
    monkeypatch.setattr(
        facade_module,
        "compute_tokenizer_fingerprint",
        lambda *args: fingerprints.append(args) or compute(*args),
    )
    engine = _engine()

    async def body(engine, facade, http):
        before = facade.client.server_info().instance_id
        lora_before = facade._lora
        engine.instance_id = "fake-restarted"
        while facade.client.server_info().instance_id == before:
            await asyncio.sleep(0.01)
        assert facade.client.server_info().instance_id == "fake-restarted"
        assert facade.check_health()
        # The restarted engine lost its adapters; the local fingerprint is reused.
        assert facade._lora is not lora_before
        assert len(fingerprints) == 1
        response = await http.post("/v1/completions", json={"model": MODEL, "prompt": PROMPT_IDS})
        assert response.status_code == 200, response.text

    _serve(engine, tmp_path, body, health_poll_interval=0.01)


def test_an_unusable_restarted_engine_is_lost(tmp_path) -> None:
    lost = []
    engine = _engine()

    async def body(engine, facade, http):
        engine.profile = engine.profile.replace(vocab_size=99)
        engine.instance_id = "fake-restarted"
        while not lost:
            await asyncio.sleep(0.01)
        assert "vocab_size" in str(facade._executor._fatal_error)
        await asyncio.wait_for(facade._health_task, WAIT_SECONDS)
        assert not facade._check_health()

    _serve(
        engine,
        tmp_path,
        body,
        health_poll_interval=0.01,
        on_engine_lost=_lost_recorder(lost),
    )


def test_a_restarted_engine_with_an_unusable_schema_is_lost_at_once(tmp_path, monkeypatch) -> None:
    lost = []
    engine = _engine()

    async def body(engine, facade, http):
        async def refresh():
            raise OpenEngineIncompatibleServerError(
                "schema revision 99 is newer", target=engine.target
            )

        monkeypatch.setattr(facade.client, "refresh", refresh)
        engine.instance_id = "fake-restarted"
        while not lost:
            await asyncio.sleep(0.01)
        assert "schema revision 99" in str(facade._executor._fatal_error)
        await asyncio.wait_for(facade._health_task, WAIT_SECONDS)
        assert not facade._check_health()

    _serve(
        engine,
        tmp_path,
        body,
        health_poll_interval=0.01,
        # Far longer than the test: only the restart check can declare the engine lost.
        engine_lost_timeout=3600.0,
        on_engine_lost=_lost_recorder(lost),
    )


def test_counting_after_close_writes_the_file_directly(tmp_path, monkeypatch) -> None:
    counter_dir = tmp_path / "counters"
    monkeypatch.setenv(SERVE_COUNTER_DIR_ENV, str(counter_dir))
    facade = _facade(SimpleNamespace(target="127.0.0.1:1"), tmp_path)
    facade._record_sent(1)
    facade.shutdown()
    facade._counter_writer.shutdown(wait=True)
    facade._record_sent(1)
    assert facade.generate_sent == 2
    assert contracts.read_generate_sent(counter_dir) == 2


def test_media_are_refused_by_the_media_slot_before_tokenizing(tmp_path) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        prompt = {"prompt": "Hi", "multi_modal_data": {"image": [object()]}}
        with pytest.raises(ValueError, match="multimodal"):
            facade.preprocess(prompt)
        with pytest.raises(ValueError, match="multimodal"):
            facade.generate_async(prompt, SamplingParams(max_tokens=1))
        assert facade.input_processor.prompts == []
        assert engine.requests == []

    _serve(engine, tmp_path, body)


def test_media_the_media_slot_maps_survive_preprocess(tmp_path, monkeypatch) -> None:
    engine = _engine()
    mapped = [object()]
    calls = []

    def map_media(inputs, caps, *, engine_name):
        calls.append(inputs)
        return mapped

    monkeypatch.setattr(facade_module.contracts, "map_media", map_media)

    async def body(engine, facade, http):
        prompt = {"prompt": "He", "multi_modal_data": {"image": ["http://h/i.png"]}}
        tokenized = facade.preprocess(prompt)
        # Tokenized only, placeholders unexpanded: the input processor is not used.
        assert tokenized["prompt_token_ids"] == [HE]
        assert facade.input_processor.prompts == []
        assert "multi_modal_data" not in tokenized
        # generate_async takes the media as preprocess mapped them, without mapping again.
        assert facade._media(tokenized) == mapped
        assert calls == [prompt]

    _serve(engine, tmp_path, body)


def test_a_lora_request_is_known_by_its_first_call_id(tmp_path, monkeypatch) -> None:
    engine = _engine()

    class Registry:
        async def resolve(self, lora_request):
            return "adapter-a"

    def map_without_lora(prompt_token_ids, sampling_params, caps, options, *, engine_name):
        options = dataclasses.replace(options, lora_request=None, lora_name=None)
        return contracts.default_map_request(
            prompt_token_ids, sampling_params, caps, options, engine_name=engine_name
        )

    monkeypatch.setattr(facade_module.contracts, "map_request", map_without_lora)

    async def body(engine, facade, http):
        facade._lora = Registry()
        facade._lora_supported = True
        output = await facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=1), lora_request=object()
        )
        assert engine.requests[0].request_id == output.request_id == output.id

    _serve(engine, tmp_path, body)


def test_unread_events_past_the_limit_cancel_the_request_with_503(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(facade_module, "MAX_BUFFERED_EVENTS", 1)
    engine = _engine(token(0, HE), token(0, LLO), token(0, WORLD), token(0, BANG), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        await output.__anext__()
        # The reader does not keep up; the events after the first wait in the buffer.
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        with pytest.raises(OpenEngineServeError, match="not read fast enough") as error:
            await output.aresult()
        assert error.value.status_code == 503

    _serve(engine, tmp_path, body)


def test_a_failure_while_starting_the_calls_cancels_those_sent(tmp_path, monkeypatch) -> None:
    engine = _engine(Hang())

    def two_calls(prompt_token_ids, sampling_params, caps, options, *, engine_name):
        (request,) = contracts.default_map_request(
            prompt_token_ids, sampling_params, caps, options, engine_name=engine_name
        )
        return [request, dataclasses.replace(request, request_id=request.request_id + "-2")]

    monkeypatch.setattr(facade_module.contracts, "map_request", two_calls)

    async def body(engine, facade, http):
        generate = facade.client.generate
        calls = []

        def generate_once(request):
            calls.append(request)
            if len(calls) > 1:
                raise TypeError("unexpected")
            return generate(request)

        monkeypatch.setattr(facade.client, "generate", generate_once)
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        with pytest.raises(OpenEngineServeError, match="unexpected") as error:
            await output.aresult()
        assert error.value.status_code == 500
        assert output.finished
        # The call created before the failure is cancelled, not left running.
        assert len(output._streams) == 1 and output._streams[0].cancelled

    _serve(engine, tmp_path, body)


def test_a_request_mapped_to_no_call_fails(tmp_path, monkeypatch) -> None:
    engine = _engine()
    monkeypatch.setattr(facade_module.contracts, "map_request", lambda *a, **k: [])

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=1))
        with pytest.raises(OpenEngineServeError, match="no Generate call") as error:
            await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert error.value.status_code == 500
        assert engine.requests == []

    _serve(engine, tmp_path, body)


def test_a_request_cut_off_by_closing_the_frontend_is_503(tmp_path) -> None:
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        await output.__anext__()
        await facade.aclose()
        with pytest.raises(OpenEngineServeError, match="shutting down") as error:
            await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert error.value.status_code == 503

    _serve(engine, tmp_path, body)


def test_a_lost_engine_ends_the_requests_still_running_with_503(tmp_path) -> None:
    lost = []
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        await output.__anext__()
        facade._engine_lost("test")
        with pytest.raises(OpenEngineServeError, match="is lost") as error:
            await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert error.value.status_code == 503
        assert lost == [facade]

    _serve(engine, tmp_path, body, on_engine_lost=_lost_recorder(lost))


def test_a_call_failing_after_abort_ends_the_request_as_cancelled(tmp_path) -> None:
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        await output.__anext__()
        output._aborted = True
        output._queue.put_nowait((0, RuntimeError("late failure")))
        result = await asyncio.wait_for(output.__anext__(), WAIT_SECONDS)
        assert result.finished and result.error is None
        assert result.outputs[0].finish_reason == "cancelled"
        output.abort()

    _serve(engine, tmp_path, body)


def test_stats_when_the_engine_load_cannot_be_read_are_empty(tmp_path, monkeypatch) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        async def load():
            raise OpenEngineRpcError(
                grpc.StatusCode.UNAVAILABLE, "down", rpc="GetLoad", target=engine.target
            )

        facade._metrics_use_load = True
        monkeypatch.setattr(facade.client, "load", load)
        assert await facade._iteration_stats() == []

    _serve(engine, tmp_path, body)


def test_load_reads_for_the_stats_are_spaced_out(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(facade_module, "LOAD_READ_INTERVAL_S", 0.3)
    engine = _engine()

    async def body(engine, facade, http):
        reads: list[float] = []

        async def load():
            reads.append(asyncio.get_running_loop().time())
            return None

        facade._metrics_use_load = True
        monkeypatch.setattr(facade.client, "load", load)
        assert await facade._iteration_stats() == []
        # The next turn is further away than the read may wait: no read, no stats.
        assert await facade._iteration_stats(0) == []
        assert len(reads) == 1
        # A read that may wait takes its turn.
        assert await facade._iteration_stats(1.0) == []
        assert len(reads) == 2 and reads[1] - reads[0] >= 0.29
        # Concurrent reads take turns too.
        await asyncio.gather(facade._iteration_stats(), facade._iteration_stats())
        assert len(reads) == 4 and reads[3] - reads[2] >= 0.29

    _serve(engine, tmp_path, body)


def test_generate_sent_is_written_to_the_counter_dir(tmp_path, monkeypatch) -> None:
    counter_dir = tmp_path / "counters"
    monkeypatch.setenv(SERVE_COUNTER_DIR_ENV, str(counter_dir))
    engine = _engine()

    def written(facade) -> int:
        # The file is written off the event loop, in order; wait for the queued writes.
        facade._counter_writer.submit(lambda: None).result(WAIT_SECONDS)
        return contracts.read_generate_sent(counter_dir)

    async def body(engine, facade, http):
        assert written(facade) == 0
        response = await http.post(
            "/v1/completions", json={"model": MODEL, "prompt": PROMPT_IDS, "max_tokens": 2}
        )
        assert response.status_code == 200, response.text
        # Refused by the engine: sent all the same.
        engine.push_script(Fail(grpc.StatusCode.INVALID_ARGUMENT, "refused"))
        response = await http.post("/v1/completions", json={"model": MODEL, "prompt": PROMPT_IDS})
        assert response.status_code == 400 and "refused" in response.text
        # Refused by the facade: never sent.
        response = await http.post("/v1/chat/completions", json=_chat(False, presence_penalty=0.5))
        assert response.status_code == 400
        assert facade.generate_sent == 2 == engine.counters["trtllm.counters.generate_received"]
        assert written(facade) == 2
        info = (await http.get("/server_info")).json()
        assert info["engine"]["generate_sent"] == 2

    _serve(engine, tmp_path, body)
    assert contracts.read_generate_sent(counter_dir) == 2


def test_a_lora_request_without_a_lora_registry_is_refused_at_once(tmp_path) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        # As with the default registry slot, which refuses every adapter.
        facade._lora_supported = False
        with pytest.raises(ValueError, match="lora_request"):
            facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=1), lora_request=object())
        assert engine.requests == []

    _serve(engine, tmp_path, body)


def test_trace_headers_that_are_not_ascii_are_not_sent(tmp_path) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        await facade.generate_async(
            PROMPT_IDS,
            SamplingParams(max_tokens=1),
            trace_headers={META_TRACEPARENT: "00-\u00e9-01"},
        )
        assert engine.accepted[0].metadata_value(META_TRACEPARENT) is None

    _serve(engine, tmp_path, body)


def test_request_metadata_reaches_the_engine(tmp_path) -> None:
    engine = _engine(profile=_profile().replace(supports_priority=True))
    traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

    async def body(engine, facade, http):
        await facade.generate_async(
            PROMPT_IDS,
            SamplingParams(max_tokens=1),
            priority=0.9,
            trace_headers={META_TRACEPARENT: traceparent},
            scheduling_params=SchedulingParams(attention_dp_rank=0),
        )
        received = engine.accepted[0]
        assert received.metadata_value(META_TRACEPARENT) == traceparent
        assert received.metadata_value(META_PRIORITY) == str(encode_priority(0.9))
        assert received.admitted.priority == encode_priority(0.9)
        assert received.metadata_value(META_TARGET_DP_RANK) == "0"
        # An explicit priority is sent whatever its value; none is sent when it is not given.
        await facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=1), priority=0.5)
        assert engine.accepted[1].metadata_value(META_PRIORITY) == str(encode_priority(0.5))
        await facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=1))
        assert engine.accepted[2].metadata_value(META_PRIORITY) is None

    _serve(engine, tmp_path, body)


@pytest.mark.parametrize(
    "profile_changes, options, message",
    [
        ({"supports_priority": False}, {"priority": 0.9}, "priority"),
        ({"supports_priority": False}, {"priority": 0.5}, "priority=0.5"),
        (
            {"dp_rank_targeting": False},
            {"scheduling_params": SchedulingParams(attention_dp_rank=0)},
            "attention_dp_rank",
        ),
    ],
)
def test_request_options_the_engine_does_not_declare_are_refused(
    tmp_path, profile_changes, options, message
) -> None:
    engine = _engine(profile=_profile().replace(**profile_changes))

    async def body(engine, facade, http):
        with pytest.raises(ValueError, match=message):
            facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=1), **options)
        assert engine.requests == []

    _serve(engine, tmp_path, body)


def test_server_info_has_the_engine_block(tmp_path) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        response = await http.get("/server_info")
        assert response.status_code == 200
        block = response.json()["engine"]
        assert block == {
            "kind": "openengine",
            "endpoint": engine.target,
            "engine_name": engine.profile.engine_name,
            "engine_version": engine.profile.engine_version,
            "instance_id": engine.instance_id,
            "generate_sent": 0,
        }
        assert "/server_info" in (await http.get("/openapi.json")).json()["paths"]

    _serve(engine, tmp_path, body)


def test_an_unsupported_error_from_a_handler_answers_501(tmp_path) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        response = await http.post("/kv_cache_events")
        assert response.status_code == 501, response.text
        assert "/kv_cache_events" in response.text

    _serve(engine, tmp_path, body)


def _without_components(monkeypatch) -> None:
    """Resolve every slot to its default, as when no component is installed."""
    monkeypatch.setattr(
        facade_module.contracts,
        "resolve_slot",
        lambda name: (contracts.slots()[name].default, "default"),
    )


def test_stats_without_a_metrics_recorder_do_not_query_the_engine(tmp_path, monkeypatch) -> None:
    _without_components(monkeypatch)
    engine = _engine()

    async def body(engine, facade, http):
        assert await facade._iteration_stats() == []
        assert "GetLoad" not in [rpc for rpc, _ in engine.control_calls]

    _serve(engine, tmp_path, body)


@pytest.mark.parametrize("method, path", UNSUPPORTED_ENDPOINTS)
def test_endpoints_that_need_an_in_process_engine_answer_501(tmp_path, method, path) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        response = await http.request(method, path)
        assert response.status_code == 501, response.text
        assert "OpenEngine" in response.text

    _serve(engine, tmp_path, body)


def _options(**changes: Any) -> SimpleNamespace:
    return SimpleNamespace(**{**vars(_args()), **changes})


@pytest.mark.parametrize(
    "changes, name",
    [
        ({"enable_energy_metrics": True}, "enable_energy_metrics"),
        ({"enable_resource_governor": True}, "enable_resource_governor"),
        ({"max_beam_width": 2}, "max_beam_width"),
        ({"return_perf_metrics": True}, "return_perf_metrics"),
        ({"perf_metrics_output_dir": "/tmp/perf"}, "perf_metrics_output_dir"),
        ({"enable_iter_perf_stats": True}, "enable_iter_perf_stats"),
        ({"otlp_traces_endpoint": "http://localhost:4318"}, "otlp_traces_endpoint"),
    ],
)
def test_options_the_frontend_cannot_honour_are_refused(changes, name, monkeypatch) -> None:
    _without_components(monkeypatch)
    with pytest.raises(ValueError, match=name):
        check_frontend_options(_options(**changes))


def test_options_are_accepted_once_their_component_is_present(monkeypatch) -> None:
    resolve_slot = contracts.resolve_slot

    def implemented(name: str):
        impl, _ = resolve_slot(name)
        return impl, f"tests:{name}"

    monkeypatch.setattr(facade_module.contracts, "resolve_slot", implemented)
    check_frontend_options(
        _options(
            post_processor_hook="guardrails:redact",
            num_postprocess_workers=2,
            return_perf_metrics=True,
            enable_iter_perf_stats=True,
            otlp_traces_endpoint="http://localhost:4318",
        )
    )
    check_frontend_options(_options())


def test_from_llm_args_refuses_an_option_it_cannot_honour(tmp_path) -> None:
    _hf_tokenizer().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="enable_energy_metrics"):
        OpenEngineLLM.from_llm_args(
            {"model": str(tmp_path), "enable_energy_metrics": True}, "localhost:1"
        )


def test_port_zero_needs_a_report_addr(tmp_path) -> None:
    with pytest.raises(click.UsageError, match="report_addr"):
        launch_module.launch_openengine_server("127.0.0.1", 0, {"model": str(tmp_path)}, "x:1")


def test_launch_reports_a_configuration_error_as_a_usage_error(tmp_path) -> None:
    _hf_tokenizer().save_pretrained(tmp_path)
    with pytest.raises(click.UsageError, match="enable_energy_metrics"):
        launch_module.launch_openengine_server(
            "127.0.0.1", 8000, {"model": str(tmp_path), "enable_energy_metrics": True}, "x:1"
        )


def test_launch_binds_before_the_engine_and_reports_the_address(tmp_path, monkeypatch) -> None:
    """With report_addr, the port the kernel picked is published before the engine is awaited."""
    report = tmp_path / "addr"
    events = []

    class Facade:
        async def start(self):
            events.append(("start", report.read_text().strip()))

        async def aclose(self):
            events.append(("aclose",))

    class Server:
        def __init__(self, **kwargs):
            self.app = SimpleNamespace()

        async def __call__(self, host, port, sockets=None):
            events.append(("serve", host, port, [s.getsockname()[1] for s in sockets]))

    monkeypatch.setattr(launch_module.OpenEngineLLM, "from_llm_args", lambda *a, **k: Facade())
    monkeypatch.setattr(launch_module, "adapt_server", lambda server, facade: None)
    monkeypatch.setattr("tensorrt_llm.serve.openai_server.OpenAIServer", Server)
    monkeypatch.setattr(launch_module.uvloop, "run", asyncio.run)

    launch_module.launch_openengine_server(
        "127.0.0.1", 0, {"model": str(tmp_path)}, "x:1", report_addr=str(report)
    )

    (start, published), serve, close = events
    host, _, port = published.rpartition(":")
    assert (start, host) == ("start", "127.0.0.1") and int(port) > 0
    assert serve == ("serve", "127.0.0.1", int(port), [int(port)])
    assert close == ("aclose",)


def test_launch_closes_the_socket_when_publishing_the_address_fails(tmp_path, monkeypatch) -> None:
    bound = []

    def bind(host, port):
        sock = launch_module.socket.socket()
        bound.append(sock)
        return sock

    def publish(report_addr, host, port):
        raise OSError("read-only file system")

    closed = []

    class Facade:
        async def aclose(self):
            closed.append(True)

    monkeypatch.setattr(launch_module.OpenEngineLLM, "from_llm_args", lambda *a, **k: Facade())
    monkeypatch.setattr(launch_module, "adapt_server", lambda server, facade: None)
    monkeypatch.setattr(
        "tensorrt_llm.serve.openai_server.OpenAIServer",
        lambda **kwargs: SimpleNamespace(app=SimpleNamespace()),
    )
    monkeypatch.setattr(launch_module, "_bind", bind)
    monkeypatch.setattr(serve_cmd, "_publish_bound_address", publish)

    with pytest.raises(OSError, match="read-only"):
        launch_module.launch_openengine_server(
            "127.0.0.1", 0, {"model": str(tmp_path)}, "x:1", report_addr=str(tmp_path / "addr")
        )
    assert bound[0].fileno() == -1
    # The facade, which may hold a postprocessing pool, is closed too.
    assert closed == [True]


# -- Postprocessing pool ----------------------------------------------------------


class _FakeSession:
    """Answers every event with one result; the result for a finished output is final."""

    def __init__(
        self,
        request_id: str,
        *,
        overflow_after: int | None,
        fail_iteration: bool,
        final_after: int | None = None,
        final_on_cancel: bool = True,
        fail_on_cancel: bool = False,
    ):
        self.request_id = request_id
        self._final_after = final_after
        self._final_on_cancel = final_on_cancel
        self._fail_on_cancel = fail_on_cancel
        self.events: list[Any] = []
        self.closed: list[str] = []
        self._overflow_after = overflow_after
        self._results: asyncio.Queue = asyncio.Queue()
        self._over = False
        self._fail_iteration = fail_iteration

    def _result(self, chunks: Any, final: bool) -> None:
        if not self._over:
            self._over = final
            self._results.put_nowait(contracts.PPResult(self.request_id, chunks, final))

    def send(self, event: Any) -> None:
        if self._over:
            return
        if self._overflow_after is not None and len(self.events) >= self._overflow_after:
            # As the contract says: the session is over and its iteration raises the same error.
            self._over = True
            error = contracts.PostprocOverflowError("too many events")
            self._results.put_nowait(error)
            raise error
        self.events.append(event)
        if self._fail_iteration:
            # Failing on the first event: the call has reached the engine by then.
            self._over = True
            self._results.put_nowait(contracts.PostprocError("the postprocessor raised"))
            return
        final = isinstance(event, OutputFinished) or len(self.events) == self._final_after
        self._result([f"chunk-{len(self.events)}"], final)

    def close(self, reason: str) -> None:
        self.closed.append(reason)
        if reason == "cancelled" and self._fail_on_cancel and not self._over:
            self._over = True
            self._results.put_nowait(contracts.PostprocError("failed while cancelling"))
            return
        if reason == "cancelled" and not self._final_on_cancel:
            # Nobody has to read a cancelled session: it may end without a final result.
            self._over = True
            self._results.put_nowait(None)
            return
        self._result([f"closed-{reason}"], True)

    async def _iterate(self):
        while True:
            result = await self._results.get()
            if result is None:
                return
            if isinstance(result, Exception):
                raise result
            yield result
            if result.is_final:
                return

    def __aiter__(self):
        return self._iterate()


class _FakePool:
    def __init__(self, num_workers: int, tokenizer_dir: str, **behaviour: Any) -> None:
        self.size = num_workers
        self.tokenizer_dir = tokenizer_dir
        self.sessions: list[_FakeSession] = []
        self.opened: list[dict[str, Any]] = []
        self.closed = False
        self._behaviour = {
            "overflow_after": None,
            "fail_iteration": False,
            "open_error": None,
            "final_after": None,
            "final_on_cancel": True,
            "fail_on_cancel": False,
        }
        self._behaviour.update(behaviour)

    def open(self, request_id, postproc_params, sampling_params, **kwargs):
        if self._behaviour["open_error"] is not None:
            raise self._behaviour["open_error"]
        self.opened.append({"request_id": request_id, "params": postproc_params, **kwargs})
        session = _FakeSession(
            request_id,
            overflow_after=self._behaviour["overflow_after"],
            fail_iteration=self._behaviour["fail_iteration"],
            final_after=self._behaviour["final_after"],
            final_on_cancel=self._behaviour["final_on_cancel"],
            fail_on_cancel=self._behaviour["fail_on_cancel"],
        )
        self.sessions.append(session)
        return session

    async def aclose(self) -> None:
        self.closed = True


def _with_pool(monkeypatch, **behaviour: Any) -> list[_FakePool]:
    pools: list[_FakePool] = []

    def create_postproc_pool(args, *, num_workers, tokenizer_dir):
        pool = _FakePool(num_workers, tokenizer_dir, **behaviour)
        pools.append(pool)
        return pool

    monkeypatch.setattr(facade_module.contracts, "create_postproc_pool", create_postproc_pool)
    return pools


def test_the_facade_hands_its_pool_to_the_server(tmp_path, monkeypatch) -> None:
    pools = _with_pool(monkeypatch)
    engine = _engine()

    async def body(engine, facade, http):
        assert facade.postproc_pool is pools[0]
        assert pools[0].size == 3 and pools[0].tokenizer_dir == str(tmp_path)
        # The server recognizes the facade by its class and postprocesses in its pool.
        assert contracts.create_postproc_dispatch(facade, is_visual_gen=False).enabled
        assert not contracts.create_postproc_dispatch(facade, is_visual_gen=True).enabled

    _serve(engine, tmp_path, body, postproc_workers=3)
    assert pools[0].closed and facade_module.OpenEngineLLM.postproc_host is True


def test_without_a_pool_postprocessing_stays_in_the_handler(tmp_path) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        assert facade.postproc_pool is None
        output = await facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=1), _postproc_params=object()
        )
        assert output.outputs[0]._postprocess_result is None

    _serve(engine, tmp_path, body)


def test_a_request_is_postprocessed_in_the_pool(tmp_path, monkeypatch) -> None:
    pools = _with_pool(monkeypatch)
    engine = _engine(token(0, HE), _finished())
    params = object()

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=4), streaming=True, _postproc_params=params
        )
        seen = [list(step.outputs[0]._postprocess_result) async for step in output]
        assert seen == [["chunk-1"], ["chunk-2"]]
        assert output.finished and output.error is None
        (opened,) = pools[0].opened
        assert opened == {
            "request_id": output.id,
            "params": params,
            "streaming": True,
            "num_outputs": 1,
            "num_prompt_tokens": len(PROMPT_IDS),
            "engine_stop_texts": False,
        }
        session = pools[0].sessions[0]
        assert [type(event).__name__ for event in session.events][-1] == "OutputFinished"
        assert len(session.closed) == 1
        # Without _postproc_params the request is assembled in the facade.
        plain = await facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=1))
        assert plain.outputs[0]._postprocess_result is None and len(pools[0].opened) == 1

    _serve(engine, tmp_path, body, postproc_workers=1)


@pytest.mark.parametrize(
    "behaviour, status",
    [
        ({"open_error": contracts.PostprocDataError("not plain data")}, 500),
        ({"open_error": contracts.PostprocWorkerError("the pool is closed")}, 500),
    ],
)
def test_a_request_the_pool_cannot_open_fails_before_it_starts(
    tmp_path, monkeypatch, behaviour, status
) -> None:
    _with_pool(monkeypatch, **behaviour)
    engine = _engine()

    async def body(engine, facade, http):
        with pytest.raises(OpenEngineServeError) as error:
            facade.generate_async(PROMPT_IDS, SamplingParams(max_tokens=1), _postproc_params=1)
        assert error.value.status_code == status
        assert engine.requests == []

    _serve(engine, tmp_path, body, postproc_workers=1)


@pytest.mark.parametrize(
    "behaviour, status, message",
    [
        ({"overflow_after": 1}, 503, "too many events"),
        ({"fail_iteration": True}, 500, "the postprocessor raised"),
    ],
)
def test_a_postprocessing_failure_ends_the_request_and_its_calls(
    tmp_path, monkeypatch, behaviour, status, message
) -> None:
    pools = _with_pool(monkeypatch, **behaviour)
    engine = _engine(token(0, HE), token(0, LLO), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=8), _postproc_params=object()
        )
        with pytest.raises(OpenEngineServeError, match=message) as error:
            await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert error.value.status_code == status
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        assert pools[0].sessions[0].closed == ["cancelled"]

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_an_aborted_postprocessed_request_closes_its_session(tmp_path, monkeypatch) -> None:
    pools = _with_pool(monkeypatch)
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=8), _postproc_params=object()
        )
        first = await output.__anext__()
        assert first.outputs[0]._postprocess_result == ["chunk-1"]
        output.abort()
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        result = await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert result.finished and result.error is None
        assert result.outputs[0]._postprocess_result == ["closed-cancelled"]
        assert pools[0].sessions[0].closed == ["cancelled"]

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_an_aborted_session_without_a_final_result_ends_the_request(tmp_path, monkeypatch) -> None:
    _with_pool(monkeypatch, final_on_cancel=False)
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=8), _postproc_params=object()
        )
        await output.__anext__()
        output.abort()
        result = await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert result.finished and result.error is None
        assert result.outputs[0]._postprocess_result == []

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_a_final_result_cancels_the_calls_before_it_is_read(tmp_path, monkeypatch) -> None:
    pools = _with_pool(monkeypatch, final_after=2)
    engine = _engine(token(0, HE), token(0, LLO), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=8), _postproc_params=object()
        )
        first = await asyncio.wait_for(output.__anext__(), WAIT_SECONDS)
        assert not first.finished
        # Nobody reads now: the final result alone must cancel the calls.
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        result = await asyncio.wait_for(output.__anext__(), WAIT_SECONDS)
        assert result.finished and result.outputs[0]._postprocess_result == ["chunk-2"]
        assert pools[0].sessions[0].closed == ["cancelled"]

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_unread_results_past_the_limit_end_the_request_with_503(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(facade_module, "MAX_BUFFERED_EVENTS", 2)
    _with_pool(monkeypatch)
    engine = _engine(*(token(0, HE) for _ in range(6)), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=8), _postproc_params=object()
        )
        await output.__anext__()
        # The reader does not keep up; results wait until the limit is reached.
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        with pytest.raises(OpenEngineServeError, match="not read fast enough") as error:
            await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert error.value.status_code == 503

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_a_failed_call_ends_a_postprocessed_request(tmp_path, monkeypatch) -> None:
    pools = _with_pool(monkeypatch)
    engine = _engine(engine_error(error_pb2.ERROR_CODE_INTERNAL, retryable=False))

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=8), _postproc_params=object()
        )
        with pytest.raises(OpenEngineServeError, match="engine gave up") as error:
            await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert error.value.status_code == 500
        assert pools[0].sessions[0].closed == ["cancelled"]

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_the_stats_are_empty_when_reading_the_load_times_out(tmp_path) -> None:
    engine = _engine()

    async def body(engine, facade, http):
        async def slow_stats(timeout=None):
            await asyncio.sleep(WAIT_SECONDS)
            return [{"never": True}]

        facade._iteration_stats = slow_stats
        facade._control_timeout = 0.05
        assert await asyncio.to_thread(facade.get_stats, 0) == []

    _serve(engine, tmp_path, body)


def test_the_tokenizer_directory_is_the_tokenizer_path_when_it_is_one(tmp_path) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    assert facade_module._tokenizer_dir(str(tokenizer_dir), tmp_path) == str(tokenizer_dir)
    assert facade_module._tokenizer_dir("org/hub-name", tmp_path) == str(tmp_path)
    assert facade_module._tokenizer_dir(None, tmp_path) == str(tmp_path)


@pytest.mark.parametrize("stop_texts", [[], ["!"]])
@pytest.mark.parametrize("lora", [False, True])
def test_the_assembler_and_the_pool_learn_whether_the_engine_matches_stop_strings(
    tmp_path, monkeypatch, stop_texts, lora
) -> None:
    pools = _with_pool(monkeypatch)
    engine = _engine(token(0, HE), _finished())
    told = []
    create = facade_module.contracts.create_response_assembler
    monkeypatch.setattr(
        facade_module.contracts,
        "create_response_assembler",
        lambda **kwargs: told.append(kwargs["engine_stop_texts"]) or create(**kwargs),
    )

    def map_with_stop_texts(prompt_token_ids, sampling_params, caps, options, *, engine_name):
        options = dataclasses.replace(options, lora_request=None, lora_name=None)
        (request,) = contracts.default_map_request(
            prompt_token_ids, sampling_params, caps, options, engine_name=engine_name
        )
        return [dataclasses.replace(request, stop_texts=list(stop_texts))]

    monkeypatch.setattr(facade_module.contracts, "map_request", map_with_stop_texts)

    class Registry:
        async def resolve(self, lora_request):
            return "adapter-a"

    async def body(engine, facade, http):
        facade._lora = Registry()
        facade._lora_supported = True
        generate = facade.client.generate
        # The fake engine declares no stop strings; what the facade decided is checked above it.
        monkeypatch.setattr(
            facade.client,
            "generate",
            lambda request: generate(dataclasses.replace(request, stop_texts=[])),
        )
        output = facade.generate_async(
            PROMPT_IDS,
            SamplingParams(max_tokens=4),
            lora_request=object() if lora else None,
            _postproc_params=object(),
        )
        # A request with an adapter is assembled at its first read, once the adapter is registered.
        assert told == ([] if lora else [bool(stop_texts)])
        if lora:
            assert output.outputs == []
        await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert told == [bool(stop_texts)]
        assert pools[0].opened[0]["engine_stop_texts"] is bool(stop_texts)

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_the_fingerprint_covers_the_model_directory_whatever_the_tokenizer_directory(
    tmp_path, monkeypatch
) -> None:
    pools = _with_pool(monkeypatch)
    directories = []
    compute = facade_module.compute_tokenizer_fingerprint
    monkeypatch.setattr(
        facade_module,
        "compute_tokenizer_fingerprint",
        lambda tokenizer, directory: directories.append(directory) or compute(tokenizer, directory),
    )
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    engine = _engine()

    async def body(engine, facade, http):
        # The engine hashes its model directory, so the two sides compare like with like.
        assert directories == [str(tmp_path)]
        assert pools[0].tokenizer_dir == str(tokenizer_dir)

    _serve(engine, tmp_path, body, tokenizer_dir=str(tokenizer_dir), postproc_workers=1)


def _facade_with_postproc_dir(tmp_path, postproc_dir: str) -> OpenEngineLLM:
    return OpenEngineLLM(
        args=SimpleNamespace(**vars(_args()), postprocess_tokenizer_dir=postproc_dir),
        tokenizer=_tokenizer(),
        input_processor=_InputProcessor(),
        hf_model_dir=str(tmp_path),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        endpoint="127.0.0.1:1",
        postproc_workers=1,
    )


def test_the_pool_takes_the_postprocess_tokenizer_directory(tmp_path, monkeypatch) -> None:
    pools = _with_pool(monkeypatch)
    postproc_dir = tmp_path / "postproc"
    postproc_dir.mkdir()
    _facade_with_postproc_dir(tmp_path, str(postproc_dir))
    assert pools[0].tokenizer_dir == str(postproc_dir)


def test_a_pool_is_refused_when_the_tokenizer_is_not_in_a_local_directory(
    tmp_path, monkeypatch
) -> None:
    pools = _with_pool(monkeypatch)
    # trtllm-serve sets postprocess_tokenizer_dir to a Hub --tokenizer as given.
    with pytest.raises(ValueError, match="local directory"):
        _facade_with_postproc_dir(tmp_path, "org/hub-tokenizer")
    assert pools == []


def test_a_postprocessed_request_cut_off_by_closing_the_frontend_is_503(
    tmp_path, monkeypatch
) -> None:
    pools = _with_pool(monkeypatch)
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=8), _postproc_params=object()
        )
        await asyncio.wait_for(output.__anext__(), WAIT_SECONDS)
        await facade.aclose()
        with pytest.raises(OpenEngineServeError, match="shutting down") as error:
            await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert error.value.status_code == 503
        # The session learns that the request was stopped, not that it finished.
        assert pools[0].sessions[0].closed == ["cancelled"]

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_a_postprocessing_failure_after_an_abort_ends_the_request_as_cancelled(
    tmp_path, monkeypatch
) -> None:
    _with_pool(monkeypatch, fail_on_cancel=True)
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=8), _postproc_params=object()
        )
        await asyncio.wait_for(output.__anext__(), WAIT_SECONDS)
        output.abort()
        result = await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert result.finished and result.error is None
        assert result.outputs[0]._postprocess_result == []

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_a_request_cancelled_while_starting_is_not_sent_again(tmp_path, monkeypatch) -> None:
    engine = _engine()
    resolving = asyncio.Event()

    class Registry:
        async def resolve(self, lora_request):
            resolving.set()
            await asyncio.Event().wait()

    def map_without_lora(prompt_token_ids, sampling_params, caps, options, *, engine_name):
        options = dataclasses.replace(options, lora_request=None, lora_name=None)
        return contracts.default_map_request(
            prompt_token_ids, sampling_params, caps, options, engine_name=engine_name
        )

    monkeypatch.setattr(facade_module.contracts, "map_request", map_without_lora)

    async def body(engine, facade, http):
        facade._lora = Registry()
        facade._lora_supported = True
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=1), lora_request=object()
        )
        reading = asyncio.create_task(output.__anext__())
        await asyncio.wait_for(resolving.wait(), WAIT_SECONDS)
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading
        with pytest.raises(StopAsyncIteration):
            await output.__anext__()
        assert engine.requests == []

    _serve(engine, tmp_path, body)


def test_startup_fails_when_the_local_tokenizer_cannot_be_fingerprinted(
    tmp_path, monkeypatch
) -> None:
    def unreadable(tokenizer, directory):
        raise OSError("tokenizer.json cannot be read")

    monkeypatch.setattr(facade_module, "compute_tokenizer_fingerprint", unreadable)
    error = _start_fails(_engine(), tmp_path)
    assert "tokenizer fingerprint" in str(error) and "cannot be read" in str(error)


def test_postprocessed_calls_that_end_before_every_output_are_500(tmp_path, monkeypatch) -> None:
    pools = _with_pool(monkeypatch)
    engine = _engine(token(0, HE), Hang())

    async def body(engine, facade, http):
        output = facade.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=8), _postproc_params=object()
        )
        await asyncio.wait_for(output.__anext__(), WAIT_SECONDS)
        # The call ends quietly before its output finished, as when its connection closes.
        output._queue.put_nowait((0, facade_module._END))
        with pytest.raises(OpenEngineServeError, match="ended a call before") as error:
            await asyncio.wait_for(output.aresult(), WAIT_SECONDS)
        assert error.value.status_code == 500
        assert pools[0].sessions[0].closed == ["cancelled"]

    _serve(engine, tmp_path, body, postproc_workers=1)


def test_closing_closes_the_pool_even_when_closing_the_connection_fails(
    tmp_path, monkeypatch
) -> None:
    pools = _with_pool(monkeypatch)
    engine = _engine()

    async def main() -> None:
        async with engine:
            facade = _facade(engine, tmp_path, postproc_workers=1)
            await facade.start()
            close = facade.client.close

            async def failing_close() -> None:
                await close()
                raise RuntimeError("close failed")

            monkeypatch.setattr(facade.client, "close", failing_close)
            with pytest.raises(RuntimeError, match="close failed"):
                await facade.aclose()
            assert pools[0].closed

    asyncio.run(main())


def test_a_restarted_engine_whose_tokenizer_cannot_be_checked_is_lost(
    tmp_path, monkeypatch
) -> None:
    lost = []
    engine = _engine()

    def unreadable(tokenizer, directory):
        raise OSError("tokenizer.json cannot be read")

    async def body(engine, facade, http):
        monkeypatch.setattr(facade_module, "compute_tokenizer_fingerprint", unreadable)
        # As for an engine that reported no fingerprint at startup.
        facade._local_fingerprint = None
        engine.instance_id = "fake-restarted"
        while not lost:
            await asyncio.sleep(0.01)
        assert "cannot be read" in str(facade._executor._fatal_error)

    _serve(engine, tmp_path, body, health_poll_interval=0.01, on_engine_lost=lost.append)
