# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The contract wiring: imports stay light, and OpenAIServer postprocesses where num_postprocess_workers says."""

import asyncio
import importlib.abc
import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tensorrt_llm.executor.postproc_worker import PostprocArgs, PostprocParams
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.openengine_poc import contracts

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _new_modules(setup: str, statement: str) -> list[str]:
    code = (
        f"import sys, json; {setup}; before = set(sys.modules); {statement}; "
        "print(json.dumps(sorted(set(sys.modules) - before)))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_contracts_do_not_load_the_grpc_client():
    loaded = _new_modules(
        "import tensorrt_llm", "import tensorrt_llm.serve.openengine_poc.contracts"
    )
    heavy = [
        m
        for m in loaded
        if m == "grpc"
        or m.startswith(
            ("grpc.", "tensorrt_llm.grpc.openengine_client", "tensorrt_llm.grpc.openengine.")
        )
        or m.startswith("tensorrt_llm.serve.openengine_poc.")
        and not m.endswith(("contracts", "engine_client"))
    ]
    assert heavy == []


def test_openai_server_does_not_load_the_openengine_adapter():
    loaded = _new_modules("import tensorrt_llm", "import tensorrt_llm.serve.openai_server")
    assert "tensorrt_llm.serve.openengine_poc.adapter" not in loaded
    assert "tensorrt_llm.grpc.openengine_client" not in loaded


def test_package_exports_still_resolve():
    import tensorrt_llm.serve.openengine_poc as package
    from tensorrt_llm.serve.openengine_poc.adapter import OpenEngineLLMAdapter

    assert package.OpenEngineLLMAdapter is OpenEngineLLMAdapter
    assert {"OpenEngineLLMAdapter", "OpenEngineRequestOutput", "build_generate_request"} <= set(
        dir(package)
    )
    with pytest.raises(AttributeError):
        package.not_an_export  # noqa: B018


def _server(workers: int) -> OpenAIServer:
    server = OpenAIServer.__new__(OpenAIServer)
    server.generator = SimpleNamespace(args=SimpleNamespace(num_postprocess_workers=workers))
    return server


@pytest.mark.parametrize("workers, enabled", [(0, False), (1, True), (4, True)])
def test_postproc_worker_enabled_is_unchanged(workers, enabled):
    server = _server(workers)
    assert server.postproc_worker_enabled is enabled
    assert isinstance(server._postproc_dispatch, contracts.DefaultPostprocDispatch)


def test_one_dispatch_serves_every_request_of_a_generator():
    server = _server(2)
    assert server._postproc_dispatch is server._postproc_dispatch


def test_the_dispatch_follows_a_swapped_generator():
    server = _server(0)
    assert not server.postproc_worker_enabled
    server.generator = SimpleNamespace(args=SimpleNamespace(num_postprocess_workers=2))
    assert server.postproc_worker_enabled


@pytest.mark.parametrize("workers", [0, 2])
def test_chat_response_comes_from_the_same_place(workers):
    server = _server(workers)

    async def no_metrics(*args, **kwargs):
        return None

    server._extract_metrics = no_metrics

    async def aresult():
        return None

    promise = SimpleNamespace(
        aresult=aresult,
        error=None,
        outputs=[SimpleNamespace(_postprocess_result="from worker")],
    )
    params = PostprocParams(
        post_processor=lambda rsp, args: "from handler", postproc_args=PostprocArgs()
    )
    response = asyncio.run(server._create_chat_response(promise, params, raw_request=None))
    assert response == ("from worker" if workers else "from handler")


class _FailingImport(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Makes importing one module fail as if one of its own imports were missing."""

    def __init__(self, name: str, missing: str) -> None:
        self.name = name
        self.missing = missing

    def find_spec(self, fullname, path, target=None):
        return importlib.util.spec_from_loader(fullname, self) if fullname == self.name else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ModuleNotFoundError(f"No module named {self.missing!r}", name=self.missing)


def test_a_launcher_module_missing_a_dependency_gets_the_install_hint(tmp_path, monkeypatch):
    import click

    from tensorrt_llm.commands import serve as serve_cmd

    launcher = contracts.slots()["launch_frontends"].module
    missing = "poc2_wt01_missing_dependency"
    monkeypatch.delitem(sys.modules, launcher, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_FailingImport(launcher, missing), *sys.meta_path])
    monkeypatch.setattr(serve_cmd, "get_is_diffusion_only_model", lambda *a, **k: False)
    contracts.reset_slot_cache()
    try:
        with pytest.raises(click.ClickException) as error:
            serve_cmd.main(
                args=[str(tmp_path), "--openengine-endpoint", "localhost:1"],
                standalone_mode=False,
            )
    finally:
        contracts.reset_slot_cache()
    # The hint names the real missing module, not one the earlier imports lacked.
    assert "Failed to import the OpenEngine client" in error.value.message
    assert missing in error.value.message
