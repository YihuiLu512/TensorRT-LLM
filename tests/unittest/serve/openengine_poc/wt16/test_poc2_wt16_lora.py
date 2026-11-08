# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The create_lora_registry slot (OE-LORA): registration with the engine, checked against the gRPC fake."""

from __future__ import annotations

import asyncio

import grpc
import pytest

from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.grpc.openengine_client import OpenEngineRpcError
from tensorrt_llm.serve.openengine_poc import contracts, lora
from tensorrt_llm.serve.openengine_poc.engine_client import EngineRequest
from tensorrt_llm.serve.openengine_poc.engine_client_impl import OpenEngineEngineClient

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from fake_openengine import PROFILES, FakeEngine  # noqa: E402
from fakes import FakeEngineClient  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

LORA_PROFILE = PROFILES["trtllm-like"].replace(supports_lora=True)


def _request(name="adapter-a", int_id=1, path="/loras/a"):
    return LoRARequest(lora_name=name, lora_int_id=int_id, lora_path=path)


def _load_calls(engine):
    return [request.adapter for rpc, request in engine.control_calls if rpc == "LoadLora"]


def _with_registries(body, profile=LORA_PROFILE, frontends=1):
    """Run ``body(engine, registries, clients)`` with registries connected to one gRPC fake."""

    async def main():
        async with FakeEngine(profile) as engine:
            clients = [
                await OpenEngineEngineClient.connect(engine.target, ready_timeout=10)
                for _ in range(frontends)
            ]
            try:
                registries = [
                    contracts.create_lora_registry(client, client.model_info())
                    for client in clients
                ]
                return await body(engine, registries, clients)
            finally:
                for client in clients:
                    await client.close()

    return asyncio.run(main())


def test_the_slot_resolves_to_this_module():
    contracts.reset_slot_cache()
    try:
        assert contracts.resolve_slot("create_lora_registry")[1] == (
            f"{lora.__name__}:create_lora_registry"
        )
    finally:
        contracts.reset_slot_cache()


def test_first_use_registers_and_later_uses_reuse_it():
    async def body(engine, registries, clients):
        [registry] = registries
        assert await registry.resolve(None) is None
        assert await registry.resolve(_request()) == "adapter-a"
        assert await registry.resolve(_request()) == "adapter-a"
        [adapter] = _load_calls(engine)
        assert (adapter.lora_name, adapter.lora_id, adapter.source_path) == (
            "adapter-a",
            1,
            "/loras/a",
        )
        # The Generate call names the registered adapter, and the engine takes it.
        request = EngineRequest(
            request_id="trtllm-lora", prompt_token_ids=[5, 6], max_tokens=1, lora_name="adapter-a"
        )
        async for _ in clients[0].generate(request):
            pass
        [received] = engine.received
        assert received.rejection is None and received.request.lora_name == "adapter-a"

    _with_registries(body)


def test_concurrent_first_uses_register_once():
    async def body(engine, registries, clients):
        names = await asyncio.gather(*(registries[0].resolve(_request()) for _ in range(8)))
        assert names == ["adapter-a"] * 8
        assert len(_load_calls(engine)) == 1

    _with_registries(body)


@pytest.mark.parametrize(
    "second",
    [_request(path="/loras/other"), _request(int_id=2), _request(name="adapter-b")],
    ids=["same-name-other-path", "same-name-other-id", "same-id-other-name"],
)
def test_a_name_or_id_stays_bound_to_its_adapter(second):
    # OE-LORA-3: refused here, without asking the engine, and the registration stays.
    async def body(engine, registries, clients):
        [registry] = registries
        await registry.resolve(_request())
        with pytest.raises(ValueError, match="conflicts with the registered adapter"):
            await registry.resolve(second)
        assert len(_load_calls(engine)) == 1
        assert await registry.resolve(_request()) == "adapter-a"

    _with_registries(body)


def test_a_conflict_with_a_registration_in_flight_is_refused():
    async def body(engine, registries, clients):
        [registry] = registries
        first = asyncio.ensure_future(registry.resolve(_request()))
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="conflicts"):
            await registry.resolve(_request(path="/loras/other"))
        with pytest.raises(ValueError, match="conflicts"):
            await registry.resolve(_request(name="adapter-b"))
        assert await first == "adapter-a"

    _with_registries(body)


def test_frontends_send_the_clients_id_and_the_engine_refuses_a_conflict():
    # Each frontend has its own registry; the id is the client's, and the
    # engine's registration decides between frontends.
    async def body(engine, registries, clients):
        front_a, front_b = registries
        assert await front_a.resolve(_request()) == "adapter-a"
        # Front end B has not seen the adapter; the engine refuses the other path.
        with pytest.raises(ValueError, match="did not register LoRA adapter 'adapter-a'"):
            await front_b.resolve(_request(path="/loras/other"))
        assert await front_b.resolve(_request()) == "adapter-a"
        assert [a.lora_id for a in _load_calls(engine)] == [1, 1, 1]

    _with_registries(body, frontends=2)


def test_an_engine_without_lora_refuses_without_registering():
    # OE-LORA-1.
    async def body(engine, registries, clients):
        with pytest.raises(ValueError, match="does not declare LoRA support"):
            await registries[0].resolve(_request())
        assert _load_calls(engine) == []

    _with_registries(body, profile=PROFILES["trtllm-like"])


@pytest.mark.parametrize(
    "lora_request, match",
    [
        (_request(name=""), "lora_name"),
        (_request(int_id=-1), "lora_int_id"),
        (_request(int_id=True), "lora_int_id"),
        (_request(int_id=1 << 63), "lora_int_id"),
        (_request(path=""), "lora_path"),
        (object(), "lora_name"),
    ],
)
def test_malformed_requests_are_refused(lora_request, match):
    async def body(engine, registries, clients):
        with pytest.raises(ValueError, match=match):
            await registries[0].resolve(lora_request)
        assert _load_calls(engine) == []

    _with_registries(body)


def test_an_engine_refusal_is_a_bad_request_and_is_not_kept():
    async def body(engine, registries, clients):
        [registry] = registries
        await registry.resolve(_request())
        # The fake keeps at most one adapter.
        with pytest.raises(ValueError, match="RESOURCE_EXHAUSTED"):
            await registry.resolve(_request(name="adapter-b", int_id=2, path="/loras/b"))
        with pytest.raises(ValueError, match="RESOURCE_EXHAUSTED"):
            await registry.resolve(_request(name="adapter-b", int_id=2, path="/loras/b"))
        assert len(_load_calls(engine)) == 3

    _with_registries(body, profile=LORA_PROFILE.replace(max_loras=1))


def test_a_failed_registration_not_yet_cleared_is_tried_again():
    async def body(engine, registries, clients):
        [registry] = registries
        failed = asyncio.get_running_loop().create_future()
        failed.set_exception(ValueError("refused"))
        failed.exception()
        # As the registry holds it between the failure and its done callback.
        registry._loading["adapter-a"] = (lora._fields(_request()), failed)
        registry.check(_request())
        assert await registry.resolve(_request()) == "adapter-a"
        assert len(_load_calls(engine)) == 1

    _with_registries(body)


def test_an_unavailable_engine_keeps_its_status():
    class Unavailable(FakeEngineClient):
        async def load_lora(self, name, int_id, path):
            raise OpenEngineRpcError(
                grpc.StatusCode.UNAVAILABLE, "down", rpc="LoadLora", target="engine"
            )

    client = Unavailable().with_capabilities(supports_lora=True)
    registry = contracts.create_lora_registry(client, client.model_info())
    with pytest.raises(OpenEngineRpcError) as error:
        asyncio.run(registry.resolve(_request()))
    assert error.value.code == grpc.StatusCode.UNAVAILABLE


def test_a_cancelled_waiter_does_not_cancel_the_registration():
    async def body(engine, registries, clients):
        [registry] = registries
        waiter = asyncio.ensure_future(registry.resolve(_request()))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert await registry.resolve(_request()) == "adapter-a"
        assert len(_load_calls(engine)) == 1

    _with_registries(body)


def test_check_refuses_without_asking_the_engine():
    async def body(engine, registries, clients):
        [registry] = registries
        registry.check(_request())  # Not registered by a check.
        assert _load_calls(engine) == []
        await registry.resolve(_request())
        with pytest.raises(ValueError, match="conflicts"):
            registry.check(_request(path="/loras/other"))
        with pytest.raises(ValueError, match="lora_int_id"):
            registry.check(_request(int_id=-1))
        assert len(_load_calls(engine)) == 1

    _with_registries(body)

    async def unsupported(engine, registries, clients):
        with pytest.raises(ValueError, match="does not declare LoRA support"):
            registries[0].check(_request())

    _with_registries(unsupported, profile=PROFILES["trtllm-like"])
