# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-07's rules in the PoC-S2 ownership table (scripts/poc2/check_ownership.py)."""

import pytest

__extra_import_path__ = ["~/scripts/poc2"]
# poc2_common first: check_ownership imports it itself, and only this file's
# imports are resolved through __extra_import_path__.
import poc2_common  # noqa: E402, F401, I001

import check_ownership  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

GLM, TRTLLM = check_ownership.GLM, check_ownership.TRTLLM


def _kind(task, path, repo):
    rule = check_ownership.match_rule(task, path, repo)
    return None if rule is None else rule.kind


@pytest.mark.parametrize(
    "path, repo, kind",
    [
        ("csrc/kernels/sampling.cu", GLM, "own"),
        ("csrc/min_tokens.h", GLM, "own"),
        ("tests/test_min_tokens.cpp", GLM, "own"),
        ("tests/test_ban_logits.py", GLM, "own"),
        ("python/glm5_serve/engine_api.py", GLM, "exception"),
        ("python/glm5_serve/openengine/servicer.py", GLM, "exception"),
        ("tests/openengine/test_openengine_server.py", GLM, "exception"),
        ("README.md", GLM, "exception"),
        ("scripts/poc2/check_ownership.py", TRTLLM, "exception"),
        # nothing else
        ("csrc/kernels/sampling.cu", TRTLLM, None),
        ("README.md", TRTLLM, None),
        ("docs/README.md", GLM, None),
        ("tests/sub/test_min_tokens.cpp", GLM, None),
        ("python/glm5_serve/openengine/stop_strings.py", GLM, None),
        ("tests/openengine/test_stop_strings.py", GLM, None),
        ("scripts/poc2/poc2_common.py", TRTLLM, None),
    ],
)
def test_wt07_rules(path, repo, kind):
    assert _kind("WT-07", path, repo) == kind


def test_wt07_exceptions_leave_the_owners_unchanged():
    assert _kind("WT-06", "python/glm5_serve/openengine/servicer.py", GLM) == "own"
    assert _kind("WT-06", "tests/openengine/test_openengine_server.py", GLM) == "own"
    assert _kind("WT-01", "scripts/poc2/check_ownership.py", TRTLLM) == "own"
