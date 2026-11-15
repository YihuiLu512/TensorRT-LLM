# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The orchestrator passes a request's model name through to the workers and back (DG-80).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain. The two workers serve the model under different
``--served_model_name`` values, and the client names it differently again.

- NT-099 (DG-80): a chat answer, whole or streamed, names the model the client
  asked for, not either worker's served name; a whole completion names the
  generation worker's served name instead, in-process as on the chain;
  ``/v1/models`` through the orchestrator lists the context worker's served
  name.
"""

import pytest

from ._test_openengine_disagg_common import disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

QUESTION = [{"role": "user", "content": "What is 1 + 1?"}]
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
SERVED_NAMES = {"context": "nt099-context-name", "generation": "nt099-generation-name"}


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_orchestrator"),
        role_args={role: ["--served_model_name", name] for role, name in SERVED_NAMES.items()},
    ) as deployment:
        yield deployment


def test_request_model_name_is_passed_through(deployment):
    client = deployment.client()
    asked = deployment.model
    assert asked not in SERVED_NAMES.values()
    kwargs = dict(
        model=asked, messages=QUESTION, max_tokens=8, temperature=0.0, extra_body=NO_THINKING
    )

    whole = client.chat.completions.create(**kwargs)
    assert whole.model == asked
    assert whole.choices[0].message.content

    chunks = list(client.chat.completions.create(stream=True, **kwargs))
    assert chunks
    assert {chunk.model for chunk in chunks} == {asked}
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)

    completion = client.completions.create(
        model=asked, prompt="1 + 1 =", max_tokens=4, temperature=0.0
    )
    assert completion.model == SERVED_NAMES["generation"]
    assert completion.choices[0].text

    assert [model.id for model in client.models.list().data] == [SERVED_NAMES["context"]]
