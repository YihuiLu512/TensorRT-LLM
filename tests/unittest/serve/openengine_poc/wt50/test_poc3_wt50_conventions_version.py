# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The conventions version a worker frontend accepts at startup (CS-18, SV-DSG-9), on the gRPC fake engine.

Without ``--server_role`` the frontend accepts an engine on version 3 (as the
glm-5.3-flash engine reports) or on the current version; with a role it needs
the current version.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wt50_chain as chain  # noqa: E402

from tensorrt_llm.grpc.openengine_conventions import (  # noqa: E402
    CONVENTIONS_VERSION,
    CONVENTIONS_VERSION_KEY,
)
from tensorrt_llm.llmapi.disagg_utils import ServerRole  # noqa: E402
from tensorrt_llm.serve.openengine_poc.facade import (  # noqa: E402
    OpenEngineLLM,
    OpenEngineStartupError,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _start(tmp_path: Path, role: ServerRole | None, **profile_changes) -> None:
    """Start a facade with ``role`` on a fake engine changed by ``profile_changes``."""
    model = chain.model_dir(tmp_path / "model")
    fake = chain.engine(model, **profile_changes)

    async def main() -> None:
        async with fake:
            facade = OpenEngineLLM.from_llm_args(
                {"model": str(model)}, fake.target, ready_timeout=30, server_role=role
            )
            try:
                await facade.start()
            finally:
                await facade.aclose()

    asyncio.run(main())


def _reporting(version: object) -> dict:
    return {"server_info_extra": {CONVENTIONS_VERSION_KEY: version}}


def test_engine_version_3_is_accepted_only_without_server_role(tmp_path):
    """NT-219: an engine on version 3 serves a frontend without a role and fails one with a role."""
    assert CONVENTIONS_VERSION == 4
    _start(tmp_path / "agg", None, **_reporting(3))
    for role in (ServerRole.CONTEXT, ServerRole.GENERATION):
        with pytest.raises(OpenEngineStartupError) as error:
            _start(tmp_path / role.name, role, **_reporting(3))
        message = str(error.value)
        assert "conventions version" in message
        assert f"serve={CONVENTIONS_VERSION} engine=3; a {role.name} worker needs 4" in message


@pytest.mark.parametrize(
    "role", [None, ServerRole.CONTEXT, ServerRole.GENERATION], ids=["no-role", "ctx", "gen"]
)
def test_the_current_version_is_accepted_with_or_without_a_role(tmp_path, role):
    _start(tmp_path, role, **_reporting(CONVENTIONS_VERSION))


@pytest.mark.parametrize("version", [2, CONVENTIONS_VERSION + 1])
@pytest.mark.parametrize("role", [None, ServerRole.CONTEXT], ids=["no-role", "ctx"])
def test_another_version_is_refused_with_or_without_a_role(tmp_path, role, version):
    with pytest.raises(OpenEngineStartupError, match=f"engine={version}"):
        _start(tmp_path, role, **_reporting(version))


def test_an_engine_reporting_no_version_serves_only_a_frontend_without_a_role(tmp_path):
    # An engine that does not follow the conventions reports no version.
    _start(tmp_path / "agg", None, follows_conventions=False)
    with pytest.raises(OpenEngineStartupError, match="engine=None; a CONTEXT worker needs 4"):
        _start(tmp_path / "ctx", ServerRole.CONTEXT, follows_conventions=False)
