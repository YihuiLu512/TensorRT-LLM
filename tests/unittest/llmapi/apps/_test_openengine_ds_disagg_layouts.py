# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: a smoke subset per placement of the workers.

``one_node`` is layout N-0, the module's default group on one node with 4 GPUs. ``two_node``
(the context and the generation worker on two nodes) and ``gen_multinode`` (one generation
instance across nodes) are always skipped in this module: it starts its servers on one node.
Evidence for those two layouts has to come from a multi-node job outside it. The forms,
fixtures and environment are those of ``_test_openengine_ds_disagg_basic.py``.
"""

from __future__ import annotations

import pytest

from . import _test_openengine_ds_disagg_basic as basic

# The module's fixtures.
model_path = basic.model_path
form = basic.form
groups = basic.groups

pytestmark = [pytest.mark.threadleak(enabled=False)]

# The layouts this module never runs, with what they need.
MULTI_NODE = {
    "two_node": "not run in this module: needs 2 GB300 nodes x 4 GPUs, the context worker on one "
    "node and the generation worker on the other",
    "gen_multinode": "not run in this module: needs at least 3 GB300 nodes, a generation instance "
    "across two nodes and the context worker on a third",
}


@pytest.fixture
def layout_served(request, groups) -> basic.Served:
    """The served form on layout ``request.param``; the layouts this module never runs skip."""
    if request.param in MULTI_NODE:
        pytest.skip(MULTI_NODE[request.param])
    return groups.get()


@pytest.mark.parametrize("layout_served", ["one_node", *MULTI_NODE], indirect=True)
def test_layout_smoke(layout_served):
    """NT-147 (DS-S19a, DS-S19b): the smoke subset on ``one_node``; the other two are skipped.

    Completions and chat, streamed and not, answer; greedy completions give the same text
    streamed and not. ``two_node`` and ``gen_multinode`` never run here (see the module's
    docstring), so this test is no evidence for DS-S19a or DS-S19b.
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
