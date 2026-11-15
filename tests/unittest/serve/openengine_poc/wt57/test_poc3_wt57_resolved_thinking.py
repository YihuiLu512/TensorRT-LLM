# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The reasoning mode resolved on the context worker reaches the generation worker (DG-65), on CPU.

NT-016: both worker frontends of the CPU chain (``cpu_disagg_chain``) run a
reasoning parser that takes its mode from the rendered prompt
(``poolside_v1``) and a chat template that ends the prompt with ``<think>``
or ``</think>``. Only the context worker renders the prompt; it sends the
mode it read in ``disaggregated_params.resolved_thinking``, the orchestrator
puts it in the generation leg, and the generation worker parses the output
in that mode. The mode travels in that HTTP field only, not in the KV
session the engines get or the handoff attributes relayed for them. The unit
side of the contract is
``unittest/llmapi/apps/test_reasoning_prompt_resolution.py``.
"""

from __future__ import annotations

import pytest

__extra_import_path__ = [".", ".."]
from wt57_chain import PROMPT, start_chain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

# Like the templates that prefill the marker: the mode is only in the prompt's tail.
TEMPLATE = (
    "{%- for m in messages -%}{{- m['content'] -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{%- if enable_thinking | default(true) -%}{{- '<think>' -}}"
    "{%- else -%}{{- '</think>' -}}{%- endif -%}"
    "{%- endif -%}"
)


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wt57-thinking")
    args = ["--reasoning_parser", "poolside_v1", "--chat_template", TEMPLATE]
    with start_chain(directory, frontend_args=args) as chain:
        yield chain


@pytest.mark.parametrize(
    "kwargs, thinking",
    [(None, True), ({"enable_thinking": True}, True), ({"enable_thinking": False}, False)],
    ids=["default", "on", "off"],
)
def test_resolved_thinking_travels_from_ctx_to_gen(chain, kwargs, thinking):
    body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4}
    if kwargs is not None:
        body["chat_template_kwargs"] = kwargs
    response = chain.post("/v1/chat/completions", body)
    assert response.status_code == 200, response.text

    gen_legs = [
        record["body"]
        for record in chain.gen.http_requests()
        if record["path"] == "/v1/chat/completions"
    ]
    assert gen_legs, "the generation worker frontend got no chat request"
    params = gen_legs[-1]["disaggregated_params"]
    assert params["request_type"] == "generation_only"
    assert params["resolved_thinking"] is thinking
    # An HTTP field between the worker frontends: it does not ride in the KV session
    # the engines exchange, nor in the handoff attributes relayed for them.
    assert "resolved_thinking" not in chain.gen.engine.requests[-1].kv.session.attributes_struct
    assert "resolved_thinking" not in (params.get("openengine_handoff_attributes") or {})

    # The generation worker parsed the output in the relayed mode: with no
    # closing marker in the output, thinking on makes all of it reasoning.
    message = response.json()["choices"][0]["message"]
    if thinking:
        assert message.get("reasoning_content") or message.get("reasoning"), message
        assert not message.get("content"), message
    else:
        assert message.get("content"), message
        assert not (message.get("reasoning_content") or message.get("reasoning")), message
