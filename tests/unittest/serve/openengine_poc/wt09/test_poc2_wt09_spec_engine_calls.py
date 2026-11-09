# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The engine calls ``_test_openengine_wt09_spec.py`` counts, on CPU.

Under several front ends a request and the read of the count after it may
reach different front ends; the count adds up every front end's.
"""

from __future__ import annotations

import contextlib
import importlib.util
import multiprocessing
import sys
import types
from pathlib import Path

import pytest

from tensorrt_llm.serve.openengine_poc import contracts

pytest.importorskip("openai")

pytestmark = pytest.mark.cpu_only

_SPEC = Path(__file__).resolve().parents[3] / "llmapi" / "apps" / "_test_openengine_wt09_spec.py"
_PACKAGE = "wt09_spec_tests"


def _load_spec(monkeypatch, chain: str) -> types.ModuleType:
    """The end-to-end module, imported with ``TRTLLM_TEST_OPENENGINE_CHAIN=chain``.

    The GPU helpers it imports are replaced by stand-ins.
    """
    monkeypatch.setenv("TRTLLM_TEST_OPENENGINE_CHAIN", chain)
    root = types.ModuleType(_PACKAGE)
    root.__path__ = []
    apps = types.ModuleType(f"{_PACKAGE}.apps")
    apps.__path__ = [str(_SPEC.parent)]
    test_llm = types.ModuleType(f"{_PACKAGE}.test_llm")
    test_llm.get_model_path = lambda name: name
    openai_server = types.ModuleType(f"{_PACKAGE}.apps.openai_server")
    openai_server.RemoteOpenAIServer = object
    for module in (root, apps, test_llm, openai_server):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location(f"{_PACKAGE}.apps.{_SPEC.stem}", _SPEC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frontend(counter_dir: str, commands) -> None:
    """Another front end's counter, in a process of its own: counts one call per ``"send"``."""
    counter = contracts.GenerateCounter(counter_dir)
    counter.start()
    commands.send("ready")
    while commands.recv() == "send":
        counter.increment()
        commands.send("sent")
    counter.close()


@pytest.fixture
def frontends(tmp_path):
    """Two front ends counting in ``tmp_path``: this process's, and another process's.

    Yields this process's counter and a function that sends one call through the other.
    """
    # Forked, so that the child need not import TensorRT-LLM again.
    context = multiprocessing.get_context("fork")
    commands, child_end = context.Pipe()
    other = context.Process(target=_frontend, args=(str(tmp_path), child_end))
    other.start()
    own = contracts.GenerateCounter(tmp_path)

    def send_through_other() -> str:
        commands.send("send")
        assert commands.poll(60)
        return commands.recv()

    try:
        assert commands.poll(60) and commands.recv() == "ready"
        own.start()
        yield own, send_through_other
    finally:
        own.close()
        with contextlib.suppress(OSError):
            commands.send("close")
        other.join(60)
        if other.is_alive():
            other.kill()
            other.join(60)
    assert other.exitcode == 0


def _server(counter_dir) -> types.SimpleNamespace:
    return types.SimpleNamespace(_chain_handle=types.SimpleNamespace(counter_dir=str(counter_dir)))


def test_the_count_adds_up_every_front_end(monkeypatch, tmp_path, frontends):
    module = _load_spec(monkeypatch, "trtllm")
    own, send_through_other = frontends
    server = _server(tmp_path)
    own.increment(2)
    assert send_through_other() == "sent"
    assert module._engine_calls(server) == 3

    # A call served by the other front end counts.
    assert module._served(server, send_through_other) == "sent"
    module._served(server, own.increment)
    assert module._engine_calls(server) == 5
    # A call that sends nothing does not.
    with pytest.raises(AssertionError):
        module._served(server, lambda: "nothing sent")

    class Refused(module.openai.BadRequestError):
        def __init__(self, message: str) -> None:
            Exception.__init__(self, message)

    def refuse(send=None):
        def call():
            if send is not None:
                send()
            raise Refused("logprobs are not supported")

        return call

    module._refused(server, refuse(), "logprobs")
    # A refusal after a call sent through the other front end is caught.
    with pytest.raises(AssertionError):
        module._refused(server, refuse(send_through_other), "logprobs")


def test_the_in_process_server_counts_nothing(monkeypatch, tmp_path):
    module = _load_spec(monkeypatch, "none")
    server = _server(tmp_path)
    assert module._engine_calls(server) is None
    assert module._served(server, lambda: "served") == "served"
