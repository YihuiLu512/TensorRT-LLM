# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``OpenAIServer`` on ``OpenEngineLLMAdapter``, against a scripted OpenEngine engine on CPU."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest import mock

import click
import httpx
import pytest
import torch
from google.protobuf import struct_pb2
from openengine_client_helpers import WAIT_SECONDS, Hang, StubEngine, engine_error, script, token
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
    server_pb2,
)
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.sampling_params import GuidedDecodingParams, SamplingParams
from tensorrt_llm.serve.detokenize import IncrementalDetokenizer
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.openengine_poc import OpenEngineLLMAdapter, build_generate_request
from tensorrt_llm.serve.openengine_poc import adapter as adapter_module
from tensorrt_llm.serve.openengine_poc import launch as launch_module
from tensorrt_llm.serve.openengine_poc import mapping as mapping_module
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
    )


def _adapter(engine: StubEngine, tmp_path) -> OpenEngineLLMAdapter:
    return OpenEngineLLMAdapter(
        args=_args(),
        tokenizer=_tokenizer(),
        input_processor=_InputProcessor(),
        hf_model_dir=str(tmp_path),
        endpoint=engine.target,
    )


def _serve(engine: StubEngine, tmp_path, body) -> Any:
    """Run ``body(engine, adapter, http)`` with the server wired to ``engine``."""

    async def main() -> Any:
        async with engine:
            adapter = _adapter(engine, tmp_path)
            await adapter.start(ready_timeout=WAIT_SECONDS)
            previous_media_executor = BaseMediaIO._executor
            server = OpenAIServer(
                generator=adapter,
                model=MODEL,
                tool_parser=None,
                server_role=None,
                metadata_server_cfg=None,
            )
            BaseMediaIO.set_executor(previous_media_executor)
            server._media_load_executor.shutdown(wait=False)
            try:
                transport = httpx.ASGITransport(app=server.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                    return await asyncio.wait_for(body(engine, adapter, http), WAIT_SECONDS * 2)
            finally:
                await adapter.aclose()
                server._input_proc_executor.shutdown(wait=False)
                server._metrics_stats_executor.shutdown(wait=False)

    return asyncio.run(main())


def _engine(
    *steps: Any,
    info: model_pb2.ModelInfo | None = None,
    health_states: tuple[int, ...] = (lifecycle_pb2.HEALTH_STATE_READY,),
    **caps: bool,
) -> StubEngine:
    async def get_model_info(request, context):
        return info if info is not None else _model_info(**caps)

    return StubEngine(
        script(*steps), control={"GetModelInfo": get_model_info}, health_states=health_states
    )


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


def test_mapping_sends_explicit_non_neutral_values_and_keeps_stop_strings_out() -> None:
    sampling = _chat_sampling(
        temperature=0.0,
        top_p=1.0,
        presence_penalty=0.0,
        seed=7,
        max_completion_tokens=5,
        stop=[" world"],
        stop_token_ids=[EOS, BANG, EOS],
    )
    request = build_generate_request(PROMPT_IDS, sampling, _model_info(), engine_name="stub")
    assert request.sampling.HasField("temperature") and request.sampling.temperature == 0.0
    assert request.sampling.HasField("top_p")
    assert not request.sampling.HasField("presence_penalty")
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


@pytest.mark.parametrize(
    "engine_name, conventions_version, builtin",
    [
        ("tensorrt_llm", None, True),
        ("tensorrt_llm", 1, False),
        ("other_engine", None, False),
    ],
)
def test_only_trtllm_without_conventions_gets_the_builtin_capabilities(
    engine_name, conventions_version, builtin
) -> None:
    server_info = server_pb2.ServerInfo(engine_name=engine_name)
    if conventions_version is not None:
        server_info.extra.update({mapping_module.CONVENTIONS_VERSION_KEY: conventions_version})
    sampling = _chat_sampling(presence_penalty=0.5)

    def build():
        return build_generate_request(
            PROMPT_IDS,
            sampling,
            _model_info(),
            engine_name=engine_name,
            builtin=mapping_module.builtin_capabilities(server_info),
        )

    if builtin:
        assert build().sampling.presence_penalty == 0.5
    else:
        with pytest.raises(ValueError, match="presence_penalty"):
            build()


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

    async def body(engine, adapter, http):
        response = await http.post("/v1/chat/completions", json=_chat(True))
        assert response.status_code == 200
        chunks = _sse_chunks(response.text)
        content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        assert content == "Hello world"
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert response.text.rstrip().endswith("data: [DONE]")
        assert adapter.input_processor.prompts == ["Hi"]
        assert list(engine.requests[0].token_ids.ids) == PROMPT_IDS
        assert adapter._check_health()

    _serve(engine, tmp_path, body)


def test_stop_string_truncates_and_cancels_the_engine_request(tmp_path) -> None:
    engine = _engine(token(0, HE), token(0, LLO), token(0, WORLD, AGAIN), Hang())

    async def body(engine, adapter, http):
        response = await http.post("/v1/chat/completions", json=_chat(False, stop=[" world"]))
        assert response.status_code == 200, response.text
        choice = response.json()["choices"][0]
        assert choice["message"]["content"] == "Hello"
        assert choice["finish_reason"] == "stop"
        assert choice["stop_reason"] == " world"
        assert response.json()["usage"]["completion_tokens"] == 3
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        assert not engine.requests[0].stopping.conditions

    _serve(engine, tmp_path, body)


def test_completion_from_token_ids_reports_length(tmp_path) -> None:
    engine = _engine(
        token(0, WORLD), token(0, BANG), _finished(generation_pb2.FINISH_REASON_LENGTH)
    )

    async def body(engine, adapter, http):
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


def test_unsupported_parameter_is_rejected_before_the_engine_sees_it(tmp_path) -> None:
    engine = _engine(_finished())

    async def body(engine, adapter, http):
        response = await http.post("/v1/chat/completions", json=_chat(True, presence_penalty=0.5))
        assert response.status_code == 400
        assert "presence_penalty" in response.text
        assert engine.requests == []

    _serve(engine, tmp_path, body)


@pytest.mark.parametrize(
    "code", [error_pb2.ERROR_CODE_INVALID_ARGUMENT, error_pb2.ERROR_CODE_INTERNAL]
)
def test_engine_error_fails_the_request(tmp_path, code: int) -> None:
    engine = _engine(engine_error(code, retryable=False))

    async def body(engine, adapter, http):
        response = await http.post("/v1/chat/completions", json=_chat(False))
        # The chat handler answers 400 for any generator error, so an engine
        # INTERNAL error is a 400 here as well.
        assert response.status_code == 400
        assert "engine gave up" in response.text

    _serve(engine, tmp_path, body)


def test_abort_cancels_the_engine_request(tmp_path) -> None:
    engine = _engine(token(0, HE), Hang())

    async def body(engine, adapter, http):
        output = adapter.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
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

    async def body(engine, adapter, http):
        output = adapter.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
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

    async def body(engine, adapter, http):
        output = adapter.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8, stop=["! again"]))
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

    async def body(engine, adapter, http):
        response = await http.post(
            "/v1/completions",
            json={"model": MODEL, "prompt": PROMPT_IDS, "max_tokens": 2, "stream": True},
        )
        # The stream has started before the engine is read, so the status stays 200.
        assert response.status_code == 200
        errors = [c for c in _sse_chunks(response.text) if "error" in c]
        assert len(errors) == 1 and "engine gave up" in errors[0]["error"]["message"]

    _serve(engine, tmp_path, body)


def test_health_is_lost_after_close(tmp_path) -> None:
    engine = _engine(_finished())

    async def body(engine, adapter, http):
        assert (await http.get("/health")).status_code == 200
        await adapter.aclose()
        assert not adapter._check_health()
        assert (await http.get("/health")).status_code != 200

    _serve(engine, tmp_path, body)


def test_health_follows_the_engine(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(adapter_module, "HEALTH_POLL_SECONDS", 0.01)
    engine = _engine(
        health_states=(lifecycle_pb2.HEALTH_STATE_READY, lifecycle_pb2.HEALTH_STATE_NOT_READY)
    )

    async def body(engine, adapter, http):
        while adapter._check_health():
            await asyncio.sleep(0.01)
        assert (await http.get("/health")).status_code != 200

    _serve(engine, tmp_path, body)


def test_health_is_lost_when_the_watcher_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(adapter_module, "HEALTH_POLL_SECONDS", 0.01)
    engine = _engine()

    async def body(engine, adapter, http):
        async def broken_health():
            raise TypeError("unexpected")

        monkeypatch.setattr(adapter.client, "health", broken_health)
        while not adapter._health_task.done():
            await asyncio.sleep(0.01)
        assert not adapter._check_health()
        assert (await http.get("/health")).status_code != 200

    _serve(engine, tmp_path, body)


def test_two_sequences_are_detokenized_separately(tmp_path) -> None:
    info = _model_info()
    info.generation.max_num_sequences = 2
    engine = _engine(
        token(0, HE),
        token(1, WORLD),
        token(0, LLO),
        _finished(index=0, terminal=False),
        token(1, BANG),
        _finished(generation_pb2.FINISH_REASON_LENGTH, index=1),
        info=info,
    )

    async def body(engine, adapter, http):
        # SamplingParams refuses n > 1 with greedy decoding, so sample.
        output = await adapter.generate_async(
            PROMPT_IDS, SamplingParams(n=2, temperature=1.0, max_tokens=2)
        )
        assert [o.text for o in output.outputs] == ["Hello", " world!"]
        assert [o.finish_reason for o in output.outputs] == ["stop", "length"]
        assert engine.requests[0].sampling.num_sequences == 2

    _serve(engine, tmp_path, body)


def test_prompt_over_the_context_reports_context_length_exceeded(tmp_path) -> None:
    engine = _engine(_finished())

    async def body(engine, adapter, http):
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

    async def body(engine, adapter, http):
        output = adapter.generate_async(PROMPT_IDS, SamplingParams(max_tokens=8))
        with pytest.raises(RuntimeError, match="detokenizer broke"):
            await output.__anext__()
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        assert output.finished and output.error is not None

    _serve(engine, tmp_path, body)


def test_chat_tokenizes_through_preprocess(tmp_path) -> None:
    engine = _engine(token(0, HE), _finished())

    async def body(engine, adapter, http):
        calls = []
        original = adapter.preprocess

        def preprocess(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        adapter.preprocess = preprocess
        response = await http.post("/v1/chat/completions", json=_chat(False))
        assert response.status_code == 200, response.text
        assert len(calls) == 1
        assert list(engine.requests[0].token_ids.ids) == PROMPT_IDS

    _serve(engine, tmp_path, body)


def test_detokenize_false_returns_token_ids_only(tmp_path) -> None:
    engine = _engine(
        token(0, WORLD), token(0, BANG), _finished(generation_pb2.FINISH_REASON_LENGTH)
    )

    async def body(engine, adapter, http):
        output = await adapter.generate_async(
            PROMPT_IDS, SamplingParams(max_tokens=2, detokenize=False)
        )
        assert output.outputs[0].text == ""
        assert output.outputs[0].token_ids == [WORLD, BANG]
        assert output.outputs[0].finish_reason == "length"
        with pytest.raises(ValueError, match="detokenize"):
            adapter.generate_async(
                PROMPT_IDS, SamplingParams(max_tokens=2, detokenize=False, stop=["!"])
            )

    _serve(engine, tmp_path, body)


@pytest.mark.parametrize("declared", [False, None])
def test_engine_without_token_id_input_is_refused(tmp_path, declared) -> None:
    info = _model_info()
    if declared is None:
        # A capability the engine does not declare counts as unsupported.
        info.ClearField("supports_token_ids_input")
    else:
        info.supports_token_ids_input = declared
    engine = _engine(info=info)

    async def main() -> None:
        async with engine:
            adapter = _adapter(engine, tmp_path)
            with pytest.raises(ValueError, match="token id input"):
                await adapter.start(ready_timeout=WAIT_SECONDS)

    asyncio.run(main())


def test_adapter_needs_a_hugging_face_tokenizer(tmp_path) -> None:
    with pytest.raises(TypeError, match="Hugging Face tokenizer"):
        OpenEngineLLMAdapter(
            args=_args(),
            tokenizer=object(),
            input_processor=_InputProcessor(),
            hf_model_dir=str(tmp_path),
            endpoint="localhost:1",
        )


# -- Building from trtllm-serve arguments ---------------------------------------


def test_from_llm_args_loads_the_model_tokenizer_and_forces_serving_options(
    tmp_path, monkeypatch
) -> None:
    _hf_tokenizer().save_pretrained(tmp_path)
    warnings = []
    monkeypatch.setattr(adapter_module.logger, "warning", lambda msg, *a, **k: warnings.append(msg))

    adapter = OpenEngineLLMAdapter.from_llm_args(
        {"model": str(tmp_path), "num_postprocess_workers": 2}, "localhost:1"
    )

    assert isinstance(adapter.tokenizer, TransformersTokenizer)
    assert adapter.tokenizer.decode([HE, LLO]) == "Hello"
    assert adapter.args.num_postprocess_workers == 0
    # The logger is shared, so keep only the adapter's own warnings.
    overrides = [w for w in warnings if "OpenEngine" in w]
    assert len(overrides) == 1 and "num_postprocess_workers=2" in overrides[0]


def test_from_llm_args_wraps_a_bare_tokenizer_from_the_input_processor(
    tmp_path, monkeypatch
) -> None:
    _hf_tokenizer().save_pretrained(tmp_path)
    processor = SimpleNamespace(tokenizer=_hf_tokenizer())
    monkeypatch.setattr(
        "tensorrt_llm.inputs.registry.create_input_processor", lambda *a, **k: processor
    )

    adapter = OpenEngineLLMAdapter.from_llm_args({"model": str(tmp_path)}, "localhost:1")

    assert isinstance(adapter.tokenizer, TransformersTokenizer)
    assert adapter.tokenizer.tokenizer is processor.tokenizer


def test_from_llm_args_needs_a_local_model_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="local model directory"):
        OpenEngineLLMAdapter.from_llm_args({"model": str(tmp_path / "missing")}, "localhost:1")


def _run_cli(tmp_path, *args: str, diffusion_only: bool = False, endpoint: str = "localhost:1"):
    with (
        mock.patch.object(serve_cmd, "get_is_diffusion_only_model", return_value=diffusion_only),
        mock.patch.object(serve_cmd, "device_count", side_effect=AssertionError("no GPU here")),
        mock.patch.object(serve_cmd, "launch_server") as launch_server,
        mock.patch.object(launch_module, "launch_openengine_server") as launch,
        mock.patch.object(usage_lib, "_mark_llm_startup") as mark_startup,
        mock.patch.object(usage_lib, "_capture_startup_context") as capture_startup,
    ):
        serve_cmd.main(
            args=[str(tmp_path), "--openengine-endpoint", endpoint, *args],
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
        (["--port", "0"], "--port 0"),
        (["--report_addr", "/tmp/addr"], "--report_addr"),
        (["--grpc"], "--grpc"),
        (["--server_role", "context"], "--server_role"),
        (["--disagg_cluster_uri", "http://localhost:1"], "--disagg_cluster_uri"),
        (["--enable_visual_gen"], "--enable_visual_gen"),
        (["--num_serve_frontends", "2"], "single serving frontend"),
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
    "args, name",
    [
        (["--post_processor_hook", "guardrails:redact"], "post_processor_hook"),
        (["--otlp_traces_endpoint", "http://localhost:4318"], "otlp_traces_endpoint"),
    ],
)
def test_cli_rejects_llm_side_features_it_would_skip(tmp_path, args, name) -> None:
    with pytest.raises(click.UsageError, match=name):
        _run_cli(tmp_path, *args)


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
