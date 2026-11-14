# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The first token and the usage of a disaggregated request (SV-DSG-5, SV-DSG-6), on the CPU chain.

NT-022 (DG-86): through the orchestrator, streamed or not, the first token
reaches the client exactly once: the tokens equal those of the same request
served aggregated by one worker, and ``completion_tokens`` counts the first
token once. Usage (CS-07, F-30): the generation worker frontend reports the
context worker's ``prompt_tokens`` and ``cached_tokens`` (``ctx_usage``), not
the ones its own engine reports.
"""

from __future__ import annotations

import json

import pytest
import requests
import yaml

__extra_import_path__ = [".."]
from cpu_disagg_chain import MODEL, CpuDisaggChain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

PROMPT = "He llo"
CHAINS = [("trtllm", 0), ("trtllm", 2), ("ds", 0)]


@pytest.fixture(scope="module", params=CHAINS, ids=[f"{s}-pool{n}" for s, n in CHAINS])
def chain(request, tmp_path_factory):
    shape, workers = request.param
    directory = tmp_path_factory.mktemp(f"wt52-first-token-{shape}-{workers}")
    config = directory / "frontend.yaml"
    config.write_text(yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}}))
    args = ["--config", str(config)]
    if workers:
        args += ["--num_postprocess_workers", str(workers)]
    with CpuDisaggChain(
        directory, shape, frontend_args={"context": args, "generation": args}
    ) as chain:
        yield chain


def _body(path: str, max_tokens: int) -> dict:
    tokens = {"max_tokens": max_tokens}
    if path == "/v1/completions":
        return {"prompt": PROMPT, **tokens}
    return {"messages": [{"role": "user", "content": PROMPT}], **tokens}


def _whole(chain, url: str, path: str, max_tokens: int) -> tuple[str, dict]:
    response = chain.post(path, _body(path, max_tokens), url=url)
    assert response.status_code == 200, response.text
    data = response.json()
    choice = data["choices"][0]
    text = choice["text"] if path == "/v1/completions" else choice["message"]["content"]
    return text, data["usage"]


def _streamed(chain, url: str, path: str, max_tokens: int) -> tuple[list[str], dict]:
    body = {
        "model": MODEL,
        **_body(path, max_tokens),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    pieces: list[str] = []
    usage: dict = {}
    with requests.post(
        f"{url}{path}", json=body, headers={"Connection": "close"}, timeout=60, stream=True
    ) as response:
        assert response.status_code == 200, response.text
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: ") :])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or ():
                piece = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if piece:
                    pieces.append(piece)
    return pieces, usage


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("max_tokens", [1, 4])
def test_first_token_is_emitted_exactly_once(chain, path, max_tokens):
    """NT-022: the disaggregated answer is the aggregated one, with the first token once."""
    expected, expected_usage = _whole(chain, chain.gen.url, path, max_tokens)
    assert expected
    text, usage = _whole(chain, chain.url, path, max_tokens)
    assert text == expected
    assert usage["completion_tokens"] == expected_usage["completion_tokens"]
    pieces, streamed_usage = _streamed(chain, chain.url, path, max_tokens)
    assert "".join(pieces) == expected
    assert streamed_usage["completion_tokens"] == expected_usage["completion_tokens"]


def test_the_context_leg_holds_back_its_tokens(chain):
    """SV-DSG-5: the context leg's (non-streamed) response carries the handoff and no text."""
    body = {
        "prompt": PROMPT,
        "max_tokens": 4,
        "disaggregated_params": {"request_type": "context_only", "disagg_request_id": 1 << 41},
    }
    response = chain.post("/v1/completions", body, url=chain.ctx.url)
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["text"] == "" and choice["disaggregated_params"]["first_gen_tokens"]


def test_usage_comes_from_the_context_leg(chain):
    """F-30 (SV-DSG-6): prompt_tokens and cached_tokens are the context engine's, not the generation engine's."""
    ctx_engine, gen_engine = chain.ctx.engine, chain.gen.engine
    ctx_engine.cached_prompt_tokens, gen_engine.cached_prompt_tokens = 1, 2
    try:
        aggregated = chain.complete(prompt=PROMPT, url=chain.gen.url, max_tokens=4).json()["usage"]
        whole = chain.complete(prompt=PROMPT, max_tokens=4).json()["usage"]
        _, streamed = _streamed(chain, chain.url, "/v1/completions", 4)
    finally:
        ctx_engine.cached_prompt_tokens = gen_engine.cached_prompt_tokens = 0
    assert aggregated["prompt_tokens_details"]["cached_tokens"] == 2
    for usage in (whole, streamed):
        assert usage["prompt_tokens"] == aggregated["prompt_tokens"]
        assert usage["prompt_tokens_details"]["cached_tokens"] == 1
        assert usage["completion_tokens"] == aggregated["completion_tokens"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
