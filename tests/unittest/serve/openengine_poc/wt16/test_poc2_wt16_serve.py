# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media and LoRA through OpenAIServer on the OpenEngine facade, against the gRPC fake on CPU."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
from types import SimpleNamespace

import click
import httpx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.commands import serve as serve_cmd
from tensorrt_llm.grpc.openengine.bindings import generation_pb2
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    MultimodalPlaceholderMetadata,
)
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.openengine_poc import OpenEngineLLM, contracts
from tensorrt_llm.serve.openengine_poc import facade as facade_module
from tensorrt_llm.serve.openengine_poc.facade import adapt_server
from tensorrt_llm.tokenizer import TransformersTokenizer

__extra_import_path__ = ["~/tests/unittest/grpc/openengine_client"]
from fake_openengine import PROFILES, FakeEngine  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

MODEL = "stub-vlm"
MODEL_TYPE = "poc2_wt16_vlm"
WAIT_SECONDS = 30
_VOCAB = {"<unk>": 0, "</s>": 1, "Describe": 2, "this": 3, "<image>": 4, "<video>": 5, "<s>": 6}
UNK, EOS, DESCRIBE, THIS, IMAGE, VIDEO, BOS = range(7)
PNG = b"\x89PNG\r\n\x1a\nfake-image"


def _tokenizer() -> TransformersTokenizer:
    backend = Tokenizer(WordLevel(_VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.add_special_tokens(["<s>", "</s>", "<image>", "<video>"])
    # Adds a BOS only when asked to add special tokens; the chat template adds none.
    backend.post_processor = TemplateProcessing(single="<s> $A", special_tokens=[("<s>", BOS)])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>")
    hf.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    return TransformersTokenizer(hf)


def _profile(**changes):
    fields = dict(
        model=MODEL,
        max_context_length=64,
        vocab_size=len(_VOCAB),
        eos_token_ids=(EOS,),
        bos_token_id=None,
        modalities=frozenset({"image", "video"}),
        supports_lora=True,
        capabilities={
            f"trtllm.sampling.{name}": False
            for name in ("min_p", "frequency_penalty", "presence_penalty", "repetition_penalty")
        },
    )
    fields.update(changes)
    return PROFILES["trtllm-like"].replace(**fields)


def _args():
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


def _passthrough_map_request(prompt_token_ids, sampling_params, caps, options, *, engine_name):
    """The default mapping, plus the media and the adapter name it does not carry."""
    plain = dataclasses.replace(options, media=(), lora_request=None, lora_name=None)
    requests = contracts.default_map_request(
        prompt_token_ids, sampling_params, caps, plain, engine_name=engine_name
    )
    for request in requests:
        request.media = list(options.media)
        request.lora_name = options.lora_name
    return requests


@pytest.fixture(autouse=True)
def _setup(monkeypatch, tmp_path):
    # The request mapping that carries media and adapters is another slot's.
    monkeypatch.setattr(facade_module.contracts, "map_request", _passthrough_map_request)
    monkeypatch.setitem(
        MULTIMODAL_PLACEHOLDER_REGISTRY._multimodal_placeholder_by_model_type,
        MODEL_TYPE,
        MultimodalPlaceholderMetadata(
            placeholder_map={"image": "<image>", "video": "<video>", "audio": "<audio>"}
        ),
    )
    root = tmp_path / "media"
    root.mkdir()
    (root / "seashore.png").write_bytes(PNG)
    monkeypatch.setenv("TRTLLM_MEDIA_ALLOWED_LOCAL_PATH", str(root))


def _serve(tmp_path, body, profile=None):
    """Run ``body(engine, http)`` against OpenAIServer on the facade and the gRPC fake."""

    async def main():
        async with FakeEngine(profile or _profile()) as engine:
            tokenizer = _tokenizer()

            def input_processor(prompt, params):
                # Text prompts only: a prompt with media is tokenized without it.
                assert "multi_modal_data" not in prompt
                return tokenizer.encode(prompt["prompt"], add_special_tokens=False), None

            facade = OpenEngineLLM(
                args=_args(),
                tokenizer=tokenizer,
                input_processor=input_processor,
                hf_model_dir=str(tmp_path),
                endpoint=engine.target,
                ready_timeout=WAIT_SECONDS,
            )
            await facade.start()
            previous_media_executor = BaseMediaIO._executor
            server = OpenAIServer(
                generator=facade,
                model=MODEL,
                tool_parser=None,
                server_role=None,
                metadata_server_cfg=None,
            )
            server.model_config = SimpleNamespace(model_type=MODEL_TYPE)
            adapt_server(server, facade)
            BaseMediaIO.set_executor(previous_media_executor)
            server._media_load_executor.shutdown(wait=False)
            try:
                transport = httpx.ASGITransport(app=server.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                    return await asyncio.wait_for(body(engine, http), WAIT_SECONDS * 2)
            finally:
                await facade.aclose()
                server._input_proc_executor.shutdown(wait=False)
                server._metrics_stats_executor.shutdown(wait=False)

    return asyncio.run(main())


def _chat(content, **fields):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": 1,
    }
    body.update(fields)
    return body


def _image(url):
    return {"type": "image_url", "image_url": {"url": url}}


def test_chat_media_go_to_the_engine_as_references(tmp_path):
    local = str(tmp_path / "media" / "seashore.png")
    # Unresolvable: fetching it here would fail the request.
    remote = "http://media.invalid/a.png"

    async def body(engine, http):
        response = await http.post(
            "/v1/chat/completions",
            json=_chat(
                [
                    {"type": "text", "text": "Describe this"},
                    _image(remote),
                    {"type": "video_url", "video_url": {"url": "https://media.invalid/v.mp4"}},
                    _image(local),
                ]
            ),
        )
        assert response.status_code == 200, response.text
        return engine.received

    [received] = _serve(tmp_path, body)
    items = received.request.media
    assert [(item.modality, item.WhichOneof("source")) for item in items] == [
        (generation_pb2.MODALITY_IMAGE, "url"),
        (generation_pb2.MODALITY_VIDEO, "url"),
        (generation_pb2.MODALITY_IMAGE, "data_uri"),
    ]
    assert items[0].url == "http://media.invalid/a.png"
    assert items[2].data_uri == "data:image/png;base64," + base64.b64encode(PNG).decode()
    # OE-MED-2: one placeholder token per item, unexpanded.
    ids = list(received.request.token_ids.ids)
    assert ids.count(IMAGE) == 2 and ids.count(VIDEO) == 1
    assert DESCRIBE in ids and THIS in ids and BOS not in ids


def test_chat_media_parsed_from_the_raw_body_go_as_references(tmp_path):
    # A tool call with an extra field fails the strict parse; the server parses the raw body.
    tool_call = {
        "id": "call_1",
        "type": "function",
        "name": "describe",
        "function": {"name": "describe", "arguments": "{}"},
    }
    messages = [
        {"role": "user", "content": [_image("http://media.invalid/a.png")]},
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "content": "done", "tool_call_id": "call_1"},
    ]

    async def body(engine, http):
        response = await http.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": messages, "max_completion_tokens": 1},
        )
        assert response.status_code == 200, response.text
        return engine.received

    [received] = _serve(tmp_path, body)
    [item] = received.request.media
    assert item.url == "http://media.invalid/a.png"
    assert list(received.request.token_ids.ids).count(IMAGE) == 1


def test_only_media_preprocess_mapped_skip_the_mapping():
    facade = object.__new__(OpenEngineLLM)
    forged = {"prompt_token_ids": [DESCRIBE], "openengine_media": ["anything"]}
    with pytest.raises(ValueError, match="set by preprocess only"):
        facade._media(forged)


def test_responses_media_go_to_the_engine_as_references(tmp_path):
    async def body(engine, http):
        response = await http.post(
            "/v1/responses",
            json={
                "model": MODEL,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Describe this"},
                            {"type": "input_image", "image_url": "http://media.invalid/a.png"},
                        ],
                    }
                ],
                "max_output_tokens": 1,
                "store": False,
            },
        )
        assert response.status_code == 200, response.text
        return engine.received

    [received] = _serve(tmp_path, body)
    [item] = received.request.media
    assert item.url == "http://media.invalid/a.png"
    ids = list(received.request.token_ids.ids)
    # The rendered template is not given a second BOS.
    assert ids.count(IMAGE) == 1 and BOS not in ids


@pytest.mark.parametrize(
    "content, fields, match",
    [
        (
            [{"type": "image_embeds", "image_embeds": {"data": "AAAA"}}],
            {},
            "image embeddings",
        ),
        (
            [_image("http://media.invalid/a.png")],
            {"media_io_kwargs": {"image": {}}},
            "media_io_kwargs",
        ),
        (
            [{"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}],
            {},
            "multimodal audio inputs are not supported",
        ),
        ([_image("/etc/passwd")], {}, "Local media paths are not accepted"),
        (
            [_image("http://media.invalid/a.png")],
            {"truncate_prompt_tokens": 1},
            "truncate_prompt_tokens is not supported",
        ),
        (
            [_image("http://media.invalid/a.png")],
            {"prompt_token_ids": [DESCRIBE, IMAGE], "truncate_prompt_tokens": 1},
            "truncate_prompt_tokens is not supported",
        ),
    ],
    ids=[
        "embeds",
        "media_io_kwargs",
        "undeclared-modality",
        "outside-allowed-dir",
        "truncate",
        "truncate-token-prompt",
    ],
)
def test_chat_media_that_cannot_be_sent_are_bad_requests(tmp_path, content, fields, match):
    async def body(engine, http):
        response = await http.post("/v1/chat/completions", json=_chat(content, **fields))
        assert response.status_code == 400, response.text
        assert match in response.json()["message"]
        return engine.received

    assert _serve(tmp_path, body) == []


def test_chat_lora_is_registered_then_named(tmp_path):
    adapter = tmp_path / "adapter"
    lora_request = {"lora_name": "adapter-a", "lora_int_id": 3, "lora_path": str(adapter)}

    async def body(engine, http):
        for _ in range(2):
            response = await http.post(
                "/v1/chat/completions", json=_chat("Describe this", lora_request=lora_request)
            )
            assert response.status_code == 200, response.text
        conflict = await http.post(
            "/v1/chat/completions",
            json=_chat("Describe this", lora_request={**lora_request, "lora_path": "/other"}),
        )
        assert conflict.status_code == 400, conflict.text
        loads = [request.adapter for rpc, request in engine.control_calls if rpc == "LoadLora"]
        return loads, engine.received

    loads, received = _serve(tmp_path, body)
    assert [(a.lora_name, a.lora_id, a.source_path) for a in loads] == [
        ("adapter-a", 3, str(adapter))
    ]
    assert [r.request.lora_name for r in received] == ["adapter-a", "adapter-a"]


@pytest.mark.parametrize("stream", [False, True])
def test_chat_lora_on_an_engine_without_lora_is_a_bad_request(tmp_path, stream):
    # Refused before a streamed response starts, too.
    lora_request = {"lora_name": "adapter-a", "lora_int_id": 3, "lora_path": "/loras/a"}

    async def body(engine, http):
        response = await http.post(
            "/v1/chat/completions",
            json=_chat("Describe this", lora_request=lora_request, stream=stream),
        )
        assert response.status_code == 400, response.text
        assert "does not declare LoRA support" in response.json()["message"]
        return engine.control_calls, engine.received

    calls, received = _serve(tmp_path, body, profile=_profile(supports_lora=False))
    assert [rpc for rpc, _ in calls if rpc == "LoadLora"] == [] and received == []


def test_server_media_io_kwargs_are_refused_at_startup(tmp_path):
    # The engine decodes the media and has no carrier for them.
    with pytest.raises(click.UsageError, match="--media_io_kwargs"):
        serve_cmd.main(
            args=[
                str(tmp_path),
                "--openengine-endpoint",
                "localhost:1",
                "--media_io_kwargs",
                '{"video": {"num_frames": 8}}',
            ],
            standalone_mode=False,
        )


@pytest.mark.parametrize("stream", [False, True])
def test_a_lora_request_the_mapping_refuses_registers_nothing(tmp_path, monkeypatch, stream):
    # The default mapping refuses LoRA; the request is mapped before the adapter is registered.
    monkeypatch.setattr(facade_module.contracts, "map_request", contracts.default_map_request)
    lora_request = {"lora_name": "adapter-a", "lora_int_id": 3, "lora_path": "/loras/a"}

    async def body(engine, http):
        response = await http.post(
            "/v1/chat/completions",
            json=_chat("Describe this", lora_request=lora_request, stream=stream),
        )
        assert response.status_code == 400, response.text
        assert "lora_request is not supported" in response.json()["message"]
        return engine.control_calls, engine.received

    calls, received = _serve(tmp_path, body)
    assert [rpc for rpc, _ in calls if rpc == "LoadLora"] == [] and received == []
