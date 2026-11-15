# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of these tests: the perf arms module with the harness on the path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[4]
DEFS = TESTS / "integration" / "defs"


@pytest.fixture(scope="module")
def harness_path(tmp_path_factory):
    """Put the harness package on ``sys.path`` under its top-level name, as runner.py does."""
    directory = tmp_path_factory.mktemp("pythonpath")
    (directory / "openengine_chain").symlink_to(DEFS / "openengine_chain")
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path.remove(str(directory))
        for name, module in list(sys.modules.items()):
            origin = getattr(module, "__dict__", {}).get("__file__")
            if origin and origin.startswith(str(directory)):
                del sys.modules[name]


@pytest.fixture(scope="module")
def arms(harness_path):
    name = "wt71_openengine_arms_under_test"
    spec = importlib.util.spec_from_file_location(name, DEFS / "perf" / "openengine_arms.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up while the class bodies run.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)
