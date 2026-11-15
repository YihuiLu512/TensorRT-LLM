# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a disaggregated deployment refuses in the baseline form too, refused alike on the chain (DG-63).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain; the same test runs in both forms and expects the same answer.

- NT-097 (DG-63, LIM-04): a chat request with ``tool_choice="required"`` is
  refused by the context worker while it prepares the chat request, and the
  orchestrator answers 400 with the worker's message. Qwen3 has no serving
  extension that enforces required tool calls, so the message is the one for a
  model that cannot honour it.
"""

import pytest
import requests

from ._test_openengine_disagg_common import REQUEST_TIMEOUT_S, disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

QUESTION = [{"role": "user", "content": "What is the weather like in Paris?"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]
REFUSAL = "tool_choice='required' is not supported for this model."


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(tmp_path_factory.mktemp("disagg_baseline_limits")) as deployment:
        yield deployment


def test_tool_choice_required_is_rejected_like_the_baseline(deployment):
    body = {
        "model": deployment.model,
        "messages": QUESTION,
        "max_tokens": 16,
        "tools": TOOLS,
        "tool_choice": "required",
    }
    response = requests.post(
        f"{deployment.url}/v1/chat/completions", json=body, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": REFUSAL}, response.text
