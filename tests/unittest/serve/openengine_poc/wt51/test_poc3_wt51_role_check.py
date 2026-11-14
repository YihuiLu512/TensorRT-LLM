# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NT-174 (SV-DSG-9, EX-DSG-1): a worker frontend started with a role needs the disagg_request_id capability.

The facade is started in-process on the gRPC fake engine of the CPU chain,
with and without ``server_role``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_DISAGG_REQUEST_ID
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.openengine_poc.facade import OpenEngineLLM, OpenEngineStartupError

__extra_import_path__ = [".."]
from cpu_disagg_chain import (  # noqa: E402
    FakeEngine,
    _chain_profile,
    _fingerprint,
    disagg_profile,
    make_model_dir,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _start(tmp_path: Path, role: ServerRole | None, declares_id: bool) -> None:
    """Start and close a facade with ``role`` on an engine that declares the id capability or not."""
    model = make_model_dir(tmp_path / "model")
    profile = _chain_profile(disagg_profile("trtllm-like"), _fingerprint(model))
    profile = profile.replace(
        capabilities={**profile.capabilities, CAP_REQ_DISAGG_REQUEST_ID: declares_id}
    )

    async def main() -> None:
        async with FakeEngine(profile) as fake:
            facade = OpenEngineLLM.from_llm_args(
                {"model": str(model)}, fake.target, ready_timeout=30, server_role=role
            )
            try:
                await facade.start()
            finally:
                await facade.aclose()

    asyncio.run(main())


@pytest.mark.parametrize("role", [ServerRole.CONTEXT, ServerRole.GENERATION])
def test_a_worker_frontend_needs_the_disagg_request_id_capability(tmp_path, role):
    with pytest.raises(OpenEngineStartupError, match=CAP_REQ_DISAGG_REQUEST_ID):
        _start(tmp_path / "without", role, declares_id=False)
    _start(tmp_path / "with", role, declares_id=True)


def test_a_frontend_without_a_role_does_not_need_it(tmp_path):
    _start(tmp_path, None, declares_id=False)
