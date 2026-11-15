# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: a smoke subset per placement of the workers.

``one_node`` is layout N-0, the module's default group on one node with 4 GPUs. ``two_node``
puts the context and the generation worker on two nodes, and ``gen_multinode`` runs one
generation instance across nodes; neither fits on one node, so both are skipped there. The
forms, fixtures and environment are those of ``_test_openengine_ds_disagg_basic.py``.
"""

from __future__ import annotations

import pytest

from . import _test_openengine_ds_disagg_basic as basic

# The module's fixtures.
model_path = basic.model_path
form = basic.form
groups = basic.groups

pytestmark = [pytest.mark.threadleak(enabled=False)]

# Layouts this file cannot start on one node, with what they need.
MULTI_NODE = {
    "two_node": "needs 2 GB300 nodes x 4 GPUs: the context worker on one node and the generation "
    "worker on the other",
    "gen_multinode": "needs at least 3 GB300 nodes: a generation instance across two nodes and "
    "the context worker on a third",
}


@pytest.fixture
def layout_served(request, groups) -> basic.Served:
    """The served form on layout ``request.param``; layouts this node cannot hold are skipped."""
    if request.param in MULTI_NODE:
        pytest.skip(MULTI_NODE[request.param])
    return groups.get()


@pytest.mark.parametrize("layout_served", ["one_node", *MULTI_NODE], indirect=True)
def test_layout_smoke(layout_served):
    """NT-147 (DS-S19a, DS-S19b): the smoke subset passes on each layout.

    Completions and chat, streamed and not, answer; greedy completions give the same text
    streamed and not.
    """
    served = layout_served
    for api in ("completions", "chat"):
        texts = {}
        for stream in (False, True):
            text, finish, usage = basic.generate(served, api, stream, max_tokens=32)
            assert "Paris" in text, (api, stream, text)
            assert finish in ("stop", "length"), (api, stream, finish)
            assert usage is not None and usage.completion_tokens > 0, (api, stream, usage)
            texts[stream] = text
        if api == "completions":
            assert texts[False] == texts[True], texts
