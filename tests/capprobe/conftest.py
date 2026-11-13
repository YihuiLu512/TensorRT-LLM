# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for capprobe.

capprobe reads its switches once at import, so every runtime behaviour is exercised in a fresh
subprocess with an explicit environment. The pytest process itself imports capprobe (for
capprobe.config / overlay / patchcheck) under CAPPROBE_TIER=P1 with every group off, so it never
starts a snapshot writer. Helpers live in probe_testutil (never import from conftest).
"""

import json
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "..", "tools"), _HERE):
    _p = os.path.abspath(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_testutil import IDENTITY, clean_env  # noqa: E402

for _k in [k for k in os.environ if k.startswith("CAPPROBE_")]:
    del os.environ[_k]
os.environ["CAPPROBE_TIER"] = "P1"


@pytest.fixture
def identity_file(tmp_path):
    p = tmp_path / "identity.json"
    p.write_text(json.dumps({"schema": "capkit.identity/1", "identity": IDENTITY}))
    return str(p)


@pytest.fixture
def probe_env(tmp_path, identity_file):
    """Factory: env for a P2 subprocess with the given extra switches."""
    shm = tmp_path / "shm"

    def make(groups="count", **extra):
        base = {"CAPPROBE_TIER": "P2" if groups else "P1", "CAPPROBE_GROUPS": groups,
                "CAPPROBE_SHM_DIR": str(shm), "CAPPROBE_INTERVAL_MS": "10000",
                "CAPPROBE_IDENTITY_FILE": identity_file, "CAPPROBE_PATCH_SHA": "c" * 40,
                "CAPKIT_CODE_SHA": "d" * 40}
        if "lag" in groups.split(","):
            base["CAPPROBE_LAG_PERIOD_US"] = "1000"
        base.update(extra)
        return clean_env(**base)
    make.shm = str(shm)
    return make
