# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a worker frontend started with a role checks of its engine (SV-DSG-9), on the gRPC fake engine.

NT-020 (DG-84): the engine must have KV transfer enabled and take remote
prefill; an engine with a role of its own must have the worker's. NT-172
(EP-01): ``trtllm-serve --server_role`` itself refuses to start in front of an
engine without KV transfer, and names the engine and what it lacks. A
frontend without a role makes none of these checks.
"""

from __future__ import annotations

import asyncio
import subprocess  # nosec B404
from pathlib import Path

import pytest

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.openengine_poc.facade import OpenEngineLLM, OpenEngineStartupError

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain import (  # noqa: E402
    MODEL,
    _chain_profile,
    _fingerprint,
    _process_env,
    _serve_command,
    make_model_dir,
)
from fake_openengine import (  # noqa: E402
    FakeEngine,
    FakeKvConnector,
    disagg_profile,
    running_in_thread,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

ROLES = [ServerRole.CONTEXT, ServerRole.GENERATION]


def _profile(model: Path, shape: str = "trtllm", role: str | None = None, **connector):
    """A disaggregated fake profile serving ``model``, its KV connector changed by ``connector``."""
    base = "trtllm-like" if shape == "trtllm" else "ds-like"
    profile = disagg_profile(base, role, kv_connector=FakeKvConnector(**connector))
    return _chain_profile(profile, _fingerprint(model))


def _start(tmp_path: Path, role: ServerRole | None, profile=None, **connector) -> None:
    """Start and close a facade with ``role`` on a fake engine with ``profile``."""
    model = make_model_dir(tmp_path / "model")
    profile = profile or _profile(model, **connector)

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


@pytest.mark.parametrize("role", ROLES, ids=lambda role: role.name)
@pytest.mark.parametrize(
    "connector, missing",
    [
        ({"enabled": False}, "kv_connector.enabled"),
        ({"supports_remote_prefill": False}, "kv_connector.supports_remote_prefill"),
        (
            {"enabled": False, "supports_remote_prefill": False},
            "kv_connector.enabled and kv_connector.supports_remote_prefill",
        ),
    ],
    ids=["disabled", "no-remote-prefill", "neither"],
)
def test_role_frontend_refuses_an_engine_without_kv_connector(tmp_path, role, connector, missing):
    """NT-020: a worker frontend does not start in front of an engine that cannot hand off KV."""
    with pytest.raises(OpenEngineStartupError) as error:
        _start(tmp_path / "refused", role, **connector)
    message = str(error.value)
    assert "step 'kv_connector'" in message
    assert f"engine 'tensorrt_llm' does not declare {missing}; a {role.name} worker" in message
    _start(tmp_path / "accepted", role)


@pytest.mark.parametrize("connector", [{"enabled": False}, {"supports_remote_prefill": False}])
def test_a_frontend_without_a_role_does_not_check_the_kv_connector(tmp_path, connector):
    _start(tmp_path, None, **connector)


def test_an_engine_without_a_kv_connector_is_refused_too(tmp_path):
    """An engine that reports no ``kv_connector`` at all lacks both declarations."""
    model = make_model_dir(tmp_path / "model")
    profile = _profile(model).replace(kv_connector=None)
    for role in ROLES:
        with pytest.raises(OpenEngineStartupError, match="kv_connector.enabled and"):
            _start(tmp_path / role.name, role, profile)
    _start(tmp_path / "none", None, profile)


@pytest.mark.parametrize(
    "role, engine_role, accepted",
    [
        (ServerRole.CONTEXT, "context", True),
        (ServerRole.GENERATION, "generation", True),
        (ServerRole.CONTEXT, "generation", False),
        (ServerRole.GENERATION, "context", False),
    ],
    ids=["ctx-prefill", "gen-decode", "ctx-decode", "gen-prefill"],
)
def test_an_engine_with_another_role_is_refused(tmp_path, role, engine_role, accepted):
    """A ds instance reports PREFILL or DECODE; a worker frontend needs the one it serves."""
    model = make_model_dir(tmp_path / "model")
    profile = _profile(model, shape="ds", role=engine_role)
    if accepted:
        _start(tmp_path, role, profile)
        return
    expected = "prefill" if role is ServerRole.CONTEXT else "decode"
    with pytest.raises(OpenEngineStartupError) as error:
        _start(tmp_path, role, profile)
    message = str(error.value)
    assert "step 'engine role'" in message
    assert f"a {role.name} worker needs {expected}" in message


def test_an_engine_role_without_a_worker_role_is_not_checked(tmp_path):
    model = make_model_dir(tmp_path / "model")
    _start(tmp_path, None, _profile(model, shape="ds", role="generation"))


def test_a_restarted_engine_without_kv_transfer_is_lost(tmp_path):
    """SV-DSG-9 after a restart: the check runs again, and failing it stops the frontend."""
    model = make_model_dir(tmp_path / "model")
    good = _profile(model)
    lost: list[OpenEngineLLM] = []

    async def main() -> None:
        async with FakeEngine(good) as fake:
            facade = OpenEngineLLM.from_llm_args(
                {"model": str(model)},
                fake.target,
                ready_timeout=30,
                server_role=ServerRole.GENERATION,
                health_poll_interval=0.05,
                on_engine_lost=lost.append,
            )
            await facade.start()
            try:
                # The engine comes back as another instance without KV transfer.
                fake.profile = _profile(model, enabled=False)
                fake.instance_id = "restarted"
                for _ in range(200):
                    if lost:
                        break
                    await asyncio.sleep(0.05)
                assert lost == [facade]
                assert not facade.check_health()
                assert "kv_connector.enabled" in str(facade._executor._fatal_error)
            finally:
                await facade.aclose()

    asyncio.run(main())


def test_worker_role_frontend_refuses_an_engine_without_kv_connector(tmp_path):
    """NT-172 (EP-01): ``trtllm-serve --server_role`` exits naming the engine and the declaration."""
    model = make_model_dir(tmp_path / "model")
    with running_in_thread(FakeEngine(_profile(model, enabled=False))) as fake:
        result = subprocess.run(  # nosec B603
            [
                *_serve_command(),
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--served_model_name",
                MODEL,
                "--openengine-endpoint",
                fake.target,
                "--openengine-ready-timeout",
                "60",
                "--server_role",
                "context",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=_process_env({}),
        )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output[-4000:]
    assert "step 'kv_connector'" in output, output[-4000:]
    assert "does not declare kv_connector.enabled; a CONTEXT worker needs KV transfer" in output
