# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming a disaggregated request whose engines stream every few iterations (EG-15).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` (Qwen3-0.6B,
one context and one generation worker on one GPU), in-process or on the
OpenEngine chain, with ``stream_interval`` above 1 on both workers: an engine
then sends a streamed request's tokens every ``stream_interval`` decoding
iterations. Each chunk of a completion streamed with
``continuous_usage_stats`` gives the number of tokens generated so far.

- NT-196 (EG-15): the first chunk with text arrives at the generation leg's
  first iteration that has tokens to send, so it holds fewer than
  ``stream_interval`` tokens. The later chunks come at the interval, which
  shows that the engines use it.
"""

import pytest

from ._test_openengine_disagg_common import disagg_deployment

pytestmark = pytest.mark.threadleak(enabled=False)

PROMPT = "The capital of France is"
STREAM_INTERVAL = 16
MAX_TOKENS = 2 * STREAM_INTERVAL + 8


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    with disagg_deployment(
        tmp_path_factory.mktemp("disagg_streaming"),
        worker_config={"stream_interval": STREAM_INTERVAL},
    ) as deployment:
        yield deployment


def test_first_chunk_does_not_wait_for_stream_interval(deployment):
    stream = deployment.client().completions.create(
        model=deployment.model,
        prompt=PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True, "continuous_usage_stats": True},
        extra_body={"ignore_eos": True},
    )
    chunks = list(stream)
    # Tokens generated so far, at each chunk that has text.
    counts = [
        chunk.usage.completion_tokens
        for chunk in chunks
        if chunk.choices and chunk.choices[0].text and chunk.usage is not None
    ]
    assert counts, "no chunk with text"
    assert counts[0] < STREAM_INTERVAL, counts
    # The last token may have no text, so the total is read from the last usage.
    usage = [chunk.usage.completion_tokens for chunk in chunks if chunk.usage is not None]
    assert usage[-1] == MAX_TOKENS, usage
    # stream_interval is in effect: a chunk per iteration would give about MAX_TOKENS chunks.
    assert len(counts) <= 3 + MAX_TOKENS // STREAM_INTERVAL, counts
