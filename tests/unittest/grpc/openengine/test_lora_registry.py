# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenEngine LoRA registry and the LoadLora, UnloadLora and ListLoras RPCs."""

import asyncio
import os
from types import SimpleNamespace

import pytest

grpc = pytest.importorskip(  # noqa: E402
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.73.1,<2")'
)
pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")
from conftest import AbortError, FakeServicerContext  # noqa: E402

from tensorrt_llm.grpc.openengine.bindings import lora_pb2  # noqa: E402
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer  # noqa: E402
from tensorrt_llm.grpc.openengine.lora_registry import (  # noqa: E402
    LoraConflictError,
    LoraNotRegisteredError,
    LoraRegistry,
    check_adapter_path,
)
from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _make_adapter(path):
    path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    return str(path)


@pytest.fixture
def adapter_dir(tmp_path):
    return _make_adapter(tmp_path / "adapter-a")


def test_register_resolves_to_a_lora_request(adapter_dir):
    registry = LoraRegistry("nemo")

    adapter, already_loaded = registry.register("adapter-a", 3, adapter_dir)
    request = registry.lora_request("adapter-a")

    assert already_loaded is False
    assert (adapter.name, adapter.lora_id, adapter.source_path) == ("adapter-a", 3, adapter_dir)
    assert (request.lora_name, request.lora_int_id, request.lora_path) == (
        "adapter-a",
        3,
        adapter_dir,
    )
    assert request.lora_ckpt_source == "nemo"


def test_registering_the_same_adapter_again_reports_already_loaded(adapter_dir):
    registry = LoraRegistry("hf")
    registry.register("adapter-a", 3, adapter_dir)

    adapter, already_loaded = registry.register("adapter-a", 3, adapter_dir)

    assert already_loaded is True and adapter.lora_id == 3


@pytest.mark.parametrize(
    ("name", "lora_id"),
    [
        # Same name, different adapter.
        ("adapter-a", 4),
        # Same id, different name: the LoRA cache is keyed by id.
        ("adapter-b", 3),
    ],
)
def test_conflicting_registrations_are_refused(adapter_dir, name, lora_id):
    registry = LoraRegistry("hf")
    registry.register("adapter-a", 3, adapter_dir)

    with pytest.raises(LoraConflictError):
        registry.register(name, lora_id, adapter_dir)
    assert [adapter.name for adapter in registry.adapters()] == ["adapter-a"]


@pytest.mark.parametrize(
    ("name", "lora_id", "path"),
    [
        ("", 1, "ADAPTER"),
        ("adapter-a", -1, "ADAPTER"),
        ("adapter-a", 1, ""),
    ],
)
def test_invalid_registrations_are_refused(adapter_dir, name, lora_id, path):
    registry = LoraRegistry("hf")

    with pytest.raises(ValueError):
        registry.register(name, lora_id, adapter_dir if path == "ADAPTER" else path)
    assert registry.adapters() == []


def test_adapter_paths_are_checked_for_their_format(tmp_path, adapter_dir):
    check_adapter_path("hf", adapter_dir)
    nemo_file = tmp_path / "adapter.nemo"
    nemo_file.write_bytes(b"")
    check_adapter_path("nemo", str(nemo_file))

    # An hf adapter is a directory with adapter_config.json.
    with pytest.raises(FileNotFoundError):
        check_adapter_path("hf", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        check_adapter_path("hf", str(nemo_file))
    with pytest.raises(FileNotFoundError):
        check_adapter_path("nemo", str(tmp_path / "missing.nemo"))


def test_registry_bounds_the_ids_it_ever_accepts(adapter_dir):
    registry = LoraRegistry("hf", max_entries=1)
    registry.register("adapter-a", 1, adapter_dir)
    registry.unregister("adapter-a")

    # Unloading does not free the id: the engine still holds its weights.
    with pytest.raises(OverflowError):
        registry.register("adapter-b", 2, adapter_dir)
    registry.register("adapter-a", 1, adapter_dir)


def test_unregistered_names_fail_until_loaded(adapter_dir):
    registry = LoraRegistry("hf")
    with pytest.raises(LoraNotRegisteredError):
        registry.lora_request("adapter-a")

    registry.register("adapter-a", 3, adapter_dir)
    registry.unregister("adapter-a")

    with pytest.raises(LoraNotRegisteredError):
        registry.lora_request("adapter-a")


def test_an_unloaded_id_keeps_its_path(tmp_path, adapter_dir):
    registry = LoraRegistry("hf")
    registry.register("adapter-a", 3, adapter_dir)
    registry.unregister("adapter-a")

    # The engine caches weights by id, so the id cannot name other weights.
    with pytest.raises(LoraConflictError):
        registry.register("adapter-a", 3, _make_adapter(tmp_path / "adapter-b"))
    # The same weights under another name are fine.
    adapter, already_loaded = registry.register("adapter-b", 3, adapter_dir)
    assert (adapter.name, already_loaded) == ("adapter-b", False)


# -- The Control RPCs ----------------------------------------------------------------


def _control(lora_config):
    args = SimpleNamespace(guided_decoding_backend=None, lora_config=lora_config)
    llm = SimpleNamespace(args=args, tokenizer=None)
    return OpenEngineControlServicer(llm, "m", OpenEngineInferenceServicer(llm, "m"))


def _call(control, rpc, request):
    context = FakeServicerContext()
    try:
        response = asyncio.run(getattr(control, rpc)(request, context))
    except AbortError:
        return None, context.abort_code
    return response, None


def _load(name, lora_id, path):
    return lora_pb2.LoadLoraRequest(
        adapter=lora_pb2.LoraAdapter(lora_id=lora_id, lora_name=name, source_path=path)
    )


LORA_CONFIG = SimpleNamespace(lora_ckpt_source="hf", max_loras=2)


@pytest.mark.parametrize(
    ("rpc", "request_message"),
    [
        ("LoadLora", lora_pb2.LoadLoraRequest()),
        ("UnloadLora", lora_pb2.UnloadLoraRequest(lora_name="a")),
        ("ListLoras", lora_pb2.ListLorasRequest()),
    ],
)
def test_lora_rpcs_are_unimplemented_without_lora_config(rpc, request_message):
    _, code = _call(_control(None), rpc, request_message)

    assert code == grpc.StatusCode.UNIMPLEMENTED


def test_load_list_unload_round_trip(adapter_dir):
    control = _control(LORA_CONFIG)

    loaded, code = _call(control, "LoadLora", _load("adapter-a", 3, adapter_dir))
    assert code is None and loaded.already_loaded is False
    assert loaded.adapter == lora_pb2.LoraAdapter(
        lora_id=3, lora_name="adapter-a", source_path=adapter_dir
    )
    again, _ = _call(control, "LoadLora", _load("adapter-a", 3, adapter_dir))
    assert again.already_loaded is True

    listed, _ = _call(control, "ListLoras", lora_pb2.ListLorasRequest())
    assert [adapter.lora_name for adapter in listed.adapters] == ["adapter-a"]

    unloaded, _ = _call(control, "UnloadLora", lora_pb2.UnloadLoraRequest(lora_name="adapter-a"))
    assert unloaded.adapter.lora_id == 3
    listed, _ = _call(control, "ListLoras", lora_pb2.ListLorasRequest())
    assert list(listed.adapters) == []


@pytest.mark.parametrize(
    ("request_message", "expected"),
    [
        (lora_pb2.LoadLoraRequest(), grpc.StatusCode.INVALID_ARGUMENT),
        (_load("", 3, "/no/such/adapter"), grpc.StatusCode.INVALID_ARGUMENT),
        (_load("adapter-b", 3, "/no/such/adapter"), grpc.StatusCode.FAILED_PRECONDITION),
        (_load("adapter-b", 4, "PARENT"), grpc.StatusCode.FAILED_PRECONDITION),
        (_load("adapter-a", 4, "SAME"), grpc.StatusCode.ALREADY_EXISTS),
    ],
)
def test_load_failures_map_to_statuses(adapter_dir, request_message, expected):
    control = _control(LORA_CONFIG)
    _call(control, "LoadLora", _load("adapter-a", 3, adapter_dir))
    if request_message.adapter.source_path == "SAME":
        request_message.adapter.source_path = adapter_dir
    elif request_message.adapter.source_path == "PARENT":
        request_message.adapter.source_path = os.path.dirname(adapter_dir)

    _, code = _call(control, "LoadLora", request_message)

    assert code == expected


def test_loading_a_registered_adapter_again_skips_the_path_check(tmp_path):
    adapter_dir = _make_adapter(tmp_path / "adapter-a")
    control = _control(LORA_CONFIG)
    _call(control, "LoadLora", _load("adapter-a", 3, adapter_dir))
    os.remove(os.path.join(adapter_dir, "adapter_config.json"))

    again, code = _call(control, "LoadLora", _load("adapter-a", 3, adapter_dir))

    assert code is None and again.already_loaded is True


def test_unloading_an_unknown_name_is_not_found():
    _, code = _call(_control(LORA_CONFIG), "UnloadLora", lora_pb2.UnloadLoraRequest(lora_name="x"))

    assert code == grpc.StatusCode.NOT_FOUND
