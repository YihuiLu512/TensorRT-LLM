# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-09's chain expectations: only the Eagle3 logprobs case sends no Generate call."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_DEFS = Path(__file__).resolve().parents[4] / "integration" / "defs"
_SPEC = "llmapi/apps/_test_openengine_wt09_spec.py"


@pytest.fixture(scope="module")
def ex(tmp_path_factory):
    """The harness's expectations module, imported under its top-level name as runner.py does."""
    directory = tmp_path_factory.mktemp("pythonpath")
    (directory / "openengine_chain").symlink_to(_DEFS / "openengine_chain")
    sys.path.insert(0, str(directory))
    try:
        yield importlib.import_module("openengine_chain.expectations")
    finally:
        sys.path.remove(str(directory))


@pytest.mark.parametrize(
    "test, listed",
    [
        ("test_output_logprobs[eagle3]", True),
        ("test_output_logprobs[plain]", False),
        ("test_penalties[eagle3-presence_penalty]", False),
        ("test_penalties[plain-presence_penalty]", False),
    ],
)
def test_only_the_refused_logprobs_case_expects_no_engine_traffic(ex, test, listed):
    found = ex.applicable(ex.load(), f"{_SPEC}::{test}", "trtllm")
    assert [(e.wt, e.kind) for e in found] == ([("WT-09", ex.NO_ENGINE_TRAFFIC)] if listed else [])
