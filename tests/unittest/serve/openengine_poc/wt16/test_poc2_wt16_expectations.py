# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-16's chain expectations: every test that starts serve with --media_io_kwargs is covered."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_DEFS = Path(__file__).resolve().parents[4] / "integration" / "defs"


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
    "nodeid",
    [
        "accuracy/test_glm53_flash.py::TestGLM53FlashFP8::test_video_url",
        "llmapi/apps/_test_openengine_glm53.py::test_video_url[glm]",
        "llmapi/apps/_test_openengine_glm53.py::test_video_url[trtllm]",
    ],
)
def test_tests_passing_media_io_kwargs_expect_the_frontend_to_refuse_to_start(ex, nodeid):
    from openengine_chain import conventions as cv

    for chain in cv.CHAINS:
        decided = ex.outcome_expectation(ex.applicable(ex.load(), nodeid, chain))
        assert decided is not None and decided.kind == ex.STARTUP_REJECTED, chain
        assert decided.log_contains == "cannot be combined with --media_io_kwargs"
