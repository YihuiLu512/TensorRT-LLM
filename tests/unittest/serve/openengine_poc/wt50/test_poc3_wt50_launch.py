# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How ``trtllm-serve --openengine-endpoint`` hands the worker options to the frontends (CS-01 to CS-03)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import click
import pytest

from tensorrt_llm.commands import serve as serve_cmd
from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig, ServerRole
from tensorrt_llm.serve.openengine_poc import contracts, multi_frontend
from tensorrt_llm.serve.openengine_poc import launch as launch_module
from tensorrt_llm.usage import usage_lib

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

CLUSTER = DisaggClusterConfig(cluster_uri="http://localhost:1")


@pytest.fixture(autouse=True)
def _fresh_slots():
    contracts.reset_slot_cache()
    yield
    contracts.reset_slot_cache()


def _run_cli(tmp_path: Path, *args: str) -> mock.Mock:
    """Run the CLI with one frontend; returns the mocked single-frontend launcher."""
    with (
        mock.patch.object(serve_cmd, "get_is_diffusion_only_model", return_value=False),
        mock.patch.object(serve_cmd, "device_count", side_effect=AssertionError("no GPU here")),
        mock.patch.object(serve_cmd, "launch_server") as launch_server,
        mock.patch.object(launch_module, "launch_openengine_server") as launch,
        mock.patch.object(usage_lib, "_mark_llm_startup"),
        mock.patch.object(usage_lib, "_capture_startup_context"),
    ):
        serve_cmd.main(
            args=[
                str(tmp_path),
                "--openengine-endpoint",
                "localhost:1",
                "--num_serve_frontends",
                "1",
                *args,
            ],
            standalone_mode=False,
        )
    launch_server.assert_not_called()
    return launch


def test_the_yaml_auth_key_and_cluster_reach_the_frontend(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "internal_request_auth_key: secret-key\n"
        "disagg_cluster:\n  cluster_uri: http://localhost:1\n  cluster_name: c\n"
    )
    launch = _run_cli(tmp_path, "--config", str(config), "--server_role", "generation")
    kwargs = launch.call_args.kwargs
    assert kwargs["internal_disagg_auth_key"] == "secret-key"
    assert kwargs["server_role"] is ServerRole.GENERATION
    cluster = kwargs["disagg_cluster_config"]
    assert (cluster.cluster_uri, cluster.cluster_name) == ("http://localhost:1", "c")
    # The key is not an LLM argument.
    assert "internal_request_auth_key" not in launch.call_args.args[2]


@pytest.mark.parametrize("role", ["mm_encoder", "visual_gen", "embedding"])
def test_a_role_other_than_context_or_generation_is_refused(tmp_path, role):
    with pytest.raises(click.UsageError, match="CONTEXT or GENERATION"):
        _run_cli(tmp_path, "--server_role", role)


def test_a_cluster_without_a_role_is_refused(tmp_path):
    with pytest.raises(click.UsageError, match="disagg_cluster needs --server_role"):
        _run_cli(tmp_path, "--disagg_cluster_uri", "http://localhost:1")


def test_a_frontend_without_worker_options_starts_as_before(tmp_path):
    launch = _run_cli(tmp_path)
    kwargs = launch.call_args.kwargs
    assert "server_role" not in kwargs and "internal_disagg_auth_key" not in kwargs


def _config(**changes) -> contracts.FrontendLaunchConfig:
    fields = dict(host="h", port=1, llm_args={"model": "m"}, endpoint="e:2")
    fields.update(changes)
    return contracts.FrontendLaunchConfig(**fields)


@pytest.mark.parametrize(
    "field, value",
    [
        ("server_role", ServerRole.CONTEXT),
        ("disagg_cluster_config", CLUSTER),
        ("internal_disagg_auth_key", "k"),
    ],
    ids=["role", "cluster", "auth-key"],
)
def test_one_worker_frontend_gets_the_worker_options(field, value):
    config = _config(ready_timeout=5.0, report_addr="/tmp/addr", **{field: value})
    with (
        mock.patch.object(contracts, "default_launch_frontends") as default,
        mock.patch.object(launch_module, "launch_openengine_server") as launch,
    ):
        contracts.launch_frontends(config)
    default.assert_not_called()
    kwargs = launch.call_args.kwargs
    assert kwargs[field] == value
    assert kwargs["ready_timeout"] == 5.0 and kwargs["report_addr"] == "/tmp/addr"
    assert launch.call_args.args == ("h", 1, {"model": "m"}, "e:2")


def test_launch_gives_the_worker_options_to_the_server_and_the_facade(tmp_path, monkeypatch):
    built = {}
    order = []
    bind = launch_module._bind

    def recording_bind(host, port):
        order.append("bind")
        return bind(host, port)

    class Facade:
        async def start(self):
            order.append("engine")

        async def aclose(self):
            pass

    def from_llm_args(llm_args, endpoint, **kwargs):
        built["facade"] = kwargs
        return Facade()

    class Server:
        def __init__(self, **kwargs):
            built["server"] = kwargs
            self.app = SimpleNamespace()

        async def __call__(self, host, port, sockets=None):
            built["serve"] = (port, [s.getsockname()[1] for s in sockets or ()])

    monkeypatch.setattr(launch_module.OpenEngineLLM, "from_llm_args", from_llm_args)
    monkeypatch.setattr(
        launch_module, "adapt_server", lambda server, facade: built.setdefault("adapted", server)
    )
    monkeypatch.setattr("tensorrt_llm.serve.openai_server.OpenAIServer", Server)
    monkeypatch.setattr(launch_module.uvloop, "run", asyncio.run)
    monkeypatch.setattr(launch_module, "_bind", recording_bind)

    # Port 0 with a cluster: bound before the engine is awaited, no report address needed.
    launch_module.launch_openengine_server(
        "127.0.0.1",
        0,
        {"model": str(tmp_path)},
        "x:1",
        server_role=ServerRole.GENERATION,
        disagg_cluster_config=CLUSTER,
        internal_disagg_auth_key="k",
    )
    assert built["facade"]["server_role"] is ServerRole.GENERATION
    server = built["server"]
    assert server["server_role"] is ServerRole.GENERATION
    assert server["disagg_cluster_config"] is CLUSTER
    assert server["internal_disagg_auth_key"] == "k"
    assert server["metadata_server_cfg"] is None
    assert isinstance(built["adapted"], Server)
    port, bound = built["serve"]
    assert port > 0 and bound == [port]
    assert order == ["bind", "engine"]


@pytest.mark.parametrize("cluster", [None, {}], ids=["no-cluster", "empty-cluster"])
def test_port_zero_still_needs_a_report_addr_without_a_cluster(tmp_path, cluster):
    with pytest.raises(click.UsageError, match="report_addr or a disagg_cluster"):
        launch_module.launch_openengine_server(
            "127.0.0.1",
            0,
            {"model": str(tmp_path)},
            "x:1",
            server_role=ServerRole.CONTEXT,
            disagg_cluster_config=cluster,
        )


@pytest.mark.parametrize(
    "field, value",
    [("disagg_cluster_config", {}), ("internal_disagg_auth_key", "")],
    ids=["empty-cluster", "empty-auth-key"],
)
def test_empty_worker_options_start_a_frontend_as_before(field, value):
    with (
        mock.patch.object(contracts, "default_launch_frontends") as default,
        mock.patch.object(launch_module, "launch_openengine_server") as launch,
    ):
        contracts.launch_frontends(_config(**{field: value}))
    default.assert_called_once()
    launch.assert_not_called()


def test_every_frontend_of_several_gets_the_worker_options(monkeypatch):
    built = {}

    def from_llm_args(llm_args, endpoint, **kwargs):
        built["facade"] = kwargs
        return SimpleNamespace()

    def server(**kwargs):
        built["server"] = kwargs
        return SimpleNamespace(app=SimpleNamespace())

    from tensorrt_llm.serve.openengine_poc import facade as facade_module

    monkeypatch.setattr(facade_module.OpenEngineLLM, "from_llm_args", from_llm_args)
    # The refusal of a signature the frontend cannot verify comes with launch's adapt_server.
    monkeypatch.setattr(
        launch_module, "adapt_server", lambda server, facade: built.setdefault("adapted", server)
    )
    monkeypatch.setattr("tensorrt_llm.serve.openai_server.OpenAIServer", server)
    monkeypatch.setattr(serve_cmd, "_apply_fastapi_middlewares", lambda app, middleware: None)
    config = _config(
        server_role=ServerRole.CONTEXT, disagg_cluster_config=CLUSTER, internal_disagg_auth_key="k"
    )
    serving = SimpleNamespace(is_launcher=False)
    _, built_server = multi_frontend._build_frontend(config, postproc_workers=0, serving=serving)
    assert built["adapted"] is built_server
    assert built["facade"]["server_role"] is ServerRole.CONTEXT
    kwargs = built["server"]
    assert kwargs["server_role"] is ServerRole.CONTEXT
    # OpenAIServer itself registers from the launcher only.
    assert kwargs["disagg_cluster_config"] is CLUSTER
    assert kwargs["internal_disagg_auth_key"] == "k"
    assert kwargs["multi_frontend_serving"] is serving


# -- A signature the frontend cannot verify -----------------------------------------------

_HANDOFF = {
    "request_type": "generation_only",
    "ctx_request_id": 1,
    "disagg_request_id": 2,
    "encoded_opaque_state": "b3BhcXVl",
}


class _Server:
    """What ``launch.adapt_server`` reads of an ``OpenAIServer``; its own check is recorded."""

    def __init__(self, key, refusal=None):
        self._internal_disagg_auth_key = key
        self.checked = []
        self._refusal = refusal

    def _validate_internal_disagg_request(self, request, raw_request):
        self.checked.append(request)
        if self._refusal is not None:
            raise ValueError(self._refusal)


def _adapted(monkeypatch, server):
    from tensorrt_llm.serve.openengine_poc import facade as facade_module

    adapted = []
    monkeypatch.setattr(facade_module, "adapt_server", lambda *args: adapted.append(args))
    facade = object()
    launch_module.adapt_server(server, facade)
    assert adapted == [(server, facade)]
    return server


def _request(handoff=_HANDOFF):
    from tensorrt_llm.serve.openai_protocol import CompletionRequest

    params = {} if handoff is None else {"disaggregated_params": handoff}
    return CompletionRequest(model="m", prompt="He", **params)


# Handoff fields, none of them protected: nothing a signature covers.
_UNPROTECTED = {key: _HANDOFF[key] for key in ("request_type", "ctx_request_id")}


def _raw(signed: bool, name: str = "x-trtllm-disagg-auth"):
    from starlette.datastructures import Headers

    return SimpleNamespace(headers=Headers({name: "sha256=00"} if signed else {}))


@pytest.mark.parametrize("key", [None, ""], ids=["no-key", "empty-key"])
@pytest.mark.parametrize("name", ["x-trtllm-disagg-auth", "X-TRTLLM-Disagg-Auth"])
def test_a_keyless_frontend_refuses_a_signed_handoff_after_its_own_check(monkeypatch, key, name):
    server = _adapted(monkeypatch, _Server(key))
    request = _request()
    with pytest.raises(ValueError, match="no internal_request_auth_key to verify it"):
        server._validate_internal_disagg_request(request, _raw(True, name))
    assert server.checked == [request]


@pytest.mark.parametrize(
    "key, handoff, raw",
    [
        ("k", _HANDOFF, _raw(True)),
        (None, _HANDOFF, _raw(False)),
        (None, _HANDOFF, None),
        (None, None, _raw(True)),
        (None, _UNPROTECTED, _raw(True)),
    ],
    ids=["keyed", "unsigned", "no-raw-request", "no-handoff-fields", "no-protected-fields"],
)
def test_what_has_no_signature_or_a_key_to_check_it_is_left_to_the_server(
    monkeypatch, key, handoff, raw
):
    server = _adapted(monkeypatch, _Server(key))
    request = _request(handoff)
    server._validate_internal_disagg_request(request, raw)
    assert server.checked == [request]


def test_the_servers_own_refusal_comes_first(monkeypatch):
    server = _adapted(monkeypatch, _Server(None, refusal="needs cache_transceiver_config"))
    with pytest.raises(ValueError, match="needs cache_transceiver_config"):
        server._validate_internal_disagg_request(_request(), _raw(True))
