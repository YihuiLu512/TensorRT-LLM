# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of these tests: the perf arms module, the single-node driver, the harness on the path."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[4]
REPO = TESTS.parent
DEFS = TESTS / "integration" / "defs"
DRIVER = REPO / "jenkins" / "scripts" / "perf" / "local" / "single_node_disagg"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up while the class bodies run.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


@pytest.fixture(scope="module")
def arms(harness_path):
    name = "wt70_openengine_arms_under_test"
    try:
        yield _load(name, DEFS / "perf" / "openengine_arms.py")
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def driver():
    name = "wt70_single_node_driver_under_test"
    try:
        yield _load(name, DRIVER / "run_single_node_disagg.py")
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _no_frontend_option_check(request, monkeypatch):
    """The OpenEngine front end's option check needs TensorRT-LLM; the arms tests do not need it."""
    if "arms" in request.fixturenames:
        module = request.getfixturevalue("arms")
        monkeypatch.setattr(module, "_frontend_check", lambda: lambda args: None)
