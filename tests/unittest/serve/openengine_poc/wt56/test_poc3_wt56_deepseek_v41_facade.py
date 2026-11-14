# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4.1 custom tokenizer through the OpenEngine facade, against the gRPC fake on CPU.

The worker front end is built as ``launch_openengine_server`` builds it, from the
``trtllm-serve`` arguments with ``--custom_tokenizer deepseek_v41``. The fake engine
takes the glm-like profile and declares, as the ds-engine OpenEngine server does for
the checkpoint, its model, token ids and tokenizer fingerprint, the fingerprint
computed the way that server computes it, so the facade's startup check compares
the two implementations of the fingerprint on the real tokenizer.

Needs the tokenizer files of ``${LLM_MODELS_ROOT}/DeepSeek-V4.1-Flash-NVFP4``; no weights are read.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from tensorrt_llm.commands import serve as serve_cmd
from tensorrt_llm.grpc.openengine_conventions import (
    FINGERPRINT_FILES,
    FP_BOS_TOKEN_ID,
    FP_EOS_TOKEN_IDS,
    FP_FILES_SHA256,
    FP_VOCAB_SIZE,
)
from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.openengine_poc import OpenEngineLLM
from tensorrt_llm.serve.openengine_poc.facade import adapt_server
from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer, encode_messages

__extra_import_path__ = ["~/tests/unittest/grpc/openengine_client"]
from fake_openengine import PROFILES, FakeEngine  # noqa: E402
from utils.llm_data import llm_models_root  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

MODEL_DIR = "DeepSeek-V4.1-Flash-NVFP4"
TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
WAIT_SECONDS = 60
# The files the ds-engine server hashes into the fingerprint (TOKENIZER_HASHED_FILES).
DS_HASHED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
)
MESSAGES = [{"role": "user", "content": "What is 2+2?"}]


@pytest.fixture(scope="module")
def model_dir() -> Path:
    models = llm_models_root()
    if models is None:
        pytest.skip("no LLM models root; set LLM_MODELS_ROOT")
    root = models / MODEL_DIR
    missing = [name for name in TOKENIZER_FILES if not (root / name).is_file()]
    if missing:
        pytest.skip(f"{root} lacks {', '.join(missing)}; set LLM_MODELS_ROOT")
    return root


def _special_token(value):
    """A special token of tokenizer_config.json, stored as a string or an AddedToken dict."""
    if isinstance(value, dict):
        return value.get("content")
    return value


def _token_ids(value) -> list[int]:
    values = value if isinstance(value, list) else [value]
    return [v for v in values if isinstance(v, int) and not isinstance(v, bool)]


def ds_fingerprint(model_dir: Path) -> dict:
    """The tokenizer fingerprint the ds-engine OpenEngine server reports in ``ModelInfo.extra``.

    Computed as ds-engine computes it, independently of TensorRT-LLM's
    ``compute_tokenizer_fingerprint``: ds-engine reads ``tokenizer.json`` with the
    ``tokenizers`` library, counts added tokens in the vocabulary size, looks the EOS and
    BOS ids up from ``tokenizer_config.json`` and adds the ``eos_token_id`` of
    ``generation_config.json`` (``python/dsv4_serve/tokenizer.py``,
    ``python/dsv4_serve/openengine/capabilities.py::tokenizer_fingerprint``).
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))

    def token_id(key):
        token = _special_token(config.get(key))
        return tokenizer.token_to_id(token) if token else None

    eos = set(_token_ids(token_id("eos_token")))
    generation_config = model_dir / "generation_config.json"
    if generation_config.is_file():
        loaded = json.loads(generation_config.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            eos.update(_token_ids(loaded.get("eos_token_id")))
    hashes = {
        name: hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
        for name in DS_HASHED_FILES
        if (model_dir / name).is_file()
    }
    bos = token_id("bos_token")
    return {
        FP_VOCAB_SIZE: tokenizer.get_vocab_size(with_added_tokens=True),
        FP_EOS_TOKEN_IDS: sorted(eos),
        FP_BOS_TOKEN_ID: bos if isinstance(bos, int) else None,
        "trtllm.tokenizer.source": str(model_dir),
        FP_FILES_SHA256: hashes,
    }


def _ds_profile(model_dir: Path):
    """The glm-like profile with the ds-engine server's model, token ids and fingerprint."""
    fingerprint = ds_fingerprint(model_dir)
    return PROFILES["glm-like"].replace(
        name="ds-like",
        engine_name="dsv4_serve",
        model=str(model_dir),
        max_context_length=32768,
        vocab_size=fingerprint[FP_VOCAB_SIZE],
        eos_token_ids=tuple(fingerprint[FP_EOS_TOKEN_IDS]),
        bos_token_id=fingerprint[FP_BOS_TOKEN_ID],
        output_logprobs_max_top_n=None,
        prompt_logprobs_max_top_n=None,
        fingerprint=fingerprint,
    )


def _llm_args(model_dir: Path) -> dict:
    llm_args, _ = serve_cmd.get_llm_args(
        model=str(model_dir),
        custom_tokenizer="deepseek_v41",
        reasoning_parser="deepseek_v41",
        gpus_per_node=1,
    )
    return llm_args


def _serve(model_dir: Path, body):
    """Run ``body(facade, engine, http)`` against OpenAIServer built as launch_openengine_server builds it."""

    async def main():
        async with FakeEngine(_ds_profile(model_dir)) as engine:
            facade = OpenEngineLLM.from_llm_args(
                _llm_args(model_dir), engine.target, ready_timeout=WAIT_SECONDS
            )
            # OpenAIServer installs its media executor as the process-wide one.
            previous_media_executor = BaseMediaIO._executor
            server = None
            try:
                server = OpenAIServer(
                    generator=facade,
                    model=str(model_dir),
                    tool_parser="deepseek_v41",
                    server_role=None,
                    metadata_server_cfg=None,
                )
                adapt_server(server, facade)
                await facade.start()
                transport = httpx.ASGITransport(app=server.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                    return await asyncio.wait_for(body(facade, engine, http), WAIT_SECONDS)
            finally:
                BaseMediaIO.set_executor(previous_media_executor)
                await facade.aclose()
                if server is not None:
                    server._input_proc_executor.shutdown(wait=False)
                    server._metrics_stats_executor.shutdown(wait=False)
                    server._media_load_executor.shutdown(wait=False)

    return asyncio.run(main())


def test_ds_fingerprint_matches_the_serving_side(model_dir):
    """UV-15 on the real tokenizer: the two implementations of the fingerprint agree."""
    from tensorrt_llm.grpc.openengine_conventions import (
        TokenizerFingerprint,
        compare_tokenizer_fingerprints,
        compute_tokenizer_fingerprint,
    )

    tokenizer = DeepseekV41Tokenizer.from_pretrained(str(model_dir))
    local = compute_tokenizer_fingerprint(tokenizer, model_dir)
    engine = TokenizerFingerprint.from_extra(ds_fingerprint(model_dir))
    assert engine is not None
    assert compare_tokenizer_fingerprints(local, engine) == []
    # Both sides hash the same files, so the comparison covers every one of them.
    assert set(FINGERPRINT_FILES) == set(DS_HASHED_FILES)
    assert local.files and set(local.files) == set(engine.files)


def test_facade_loads_custom_tokenizer(model_dir):
    """NT-176: the facade starts with ``custom_tokenizer="deepseek_v41"`` and renders chat natively."""

    async def body(facade, engine, http):
        response = await http.post(
            "/v1/chat/completions",
            json={
                "model": str(model_dir),
                "messages": MESSAGES,
                "max_completion_tokens": 1,
                "chat_template_kwargs": {"thinking": True},
            },
        )
        assert response.status_code == 200, response.text
        return facade, list(engine.requests[-1].token_ids.ids)

    facade, prompt_ids = _serve(model_dir, body)
    tokenizer = facade.tokenizer
    assert isinstance(tokenizer, DeepseekV41Tokenizer)
    # The prompt is the checkpoint renderer's, not a Hugging Face chat template's.
    rendered = encode_messages(MESSAGES, thinking_mode="thinking")
    assert prompt_ids == tokenizer.encode(rendered, add_special_tokens=False)
    assert tokenizer.decode(prompt_ids).endswith("<｜Assistant｜><think>")
