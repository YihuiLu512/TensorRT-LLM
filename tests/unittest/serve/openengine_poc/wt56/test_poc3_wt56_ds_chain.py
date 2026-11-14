# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chain harness's ``ds`` chain: the ds-engine started as the engine of a ``trtllm-serve`` command.

The processes are the harness's own stand-in (``openengine_chain/tests/fake_serve.py``),
installed once as ``trtllm-serve`` and once as the ds-engine command, so the tests read
the command lines the harness starts.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_DEFS = Path(__file__).resolve().parents[4] / "integration" / "defs"
_FAKE = _DEFS / "openengine_chain" / "tests" / "fake_serve.py"


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    """The harness's conventions and launch modules, imported under their package as runner.py does."""
    directory = tmp_path_factory.mktemp("pythonpath")
    (directory / "openengine_chain").symlink_to(_DEFS / "openengine_chain")
    sys.path.insert(0, str(directory))
    try:
        from openengine_chain import conventions, launch

        yield conventions, launch
    finally:
        sys.path.remove(str(directory))


def _executable(directory: Path, name: str) -> str:
    path = directory / name
    path.write_text(f"#!{sys.executable}\n" + _FAKE.read_text())
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def fake(harness, tmp_path, monkeypatch):
    """``trtllm-serve`` and ``dsv4-serve`` stand-ins, the record directory and a base environment."""
    cv, launch = harness
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    records = tmp_path / "records"
    records.mkdir()
    monkeypatch.setattr(launch, "require_plugin", lambda: None)
    monkeypatch.setattr(launch, "frontend_supports_ready_timeout", lambda: True)
    monkeypatch.setattr(launch, "engine_supports_metrics_port", lambda: True)
    monkeypatch.setenv(cv.ENV_RESULTS_DIR, str(tmp_path / "results"))
    monkeypatch.delenv(cv.ENV_DS_SERVE_CMD, raising=False)
    env = dict(os.environ, FAKE_RECORD_DIR=str(records))
    env.pop(cv.ENV_COUNTER_DIR, None)
    handles = []
    yield type(
        "Fake",
        (),
        {
            "serve": _executable(bin_dir, "trtllm-serve"),
            "ds": _executable(bin_dir, "dsv4-serve"),
            "records": records,
            "env": env,
            "handles": handles,
        },
    )
    for handle in handles:
        handle.terminate(grace=1)


def _record(records: Path, name: str, timeout: float = 10.0) -> dict:
    path = records / f"{name}.json"
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{name} was not recorded"
        time.sleep(0.05)
    return json.loads(path.read_text())


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(harness, fake, args):
    _, launch = harness
    handle = launch.launch_serve_command([fake.serve, "/model"] + args, env=fake.env, chain="ds")
    fake.handles.append(handle)
    return handle


def test_ds_is_a_chain(harness, monkeypatch):
    cv, launch = harness
    assert cv.CHAIN_DS == "ds" and cv.CHAIN_DS in cv.CHAINS
    assert cv.ENV_DS_SERVE_CMD == "TRTLLM_TEST_DS_SERVE_CMD"
    monkeypatch.setenv(cv.ENV_CHAIN, "ds")
    assert launch.resolve_chain() == "ds"


def test_ds_chain_needs_the_ds_command(harness, fake):
    _, launch = harness
    with pytest.raises(launch.ChainError, match="TRTLLM_TEST_DS_SERVE_CMD"):
        _start(harness, fake, ["--port", str(_port())])
    assert not list(fake.records.iterdir())


def test_ds_chain_starts_the_ds_command_as_the_engine(harness, fake, monkeypatch):
    cv, launch = harness
    monkeypatch.setenv(cv.ENV_DS_SERVE_CMD, f"{fake.ds} --ds-flag")
    port = _port()
    handle = _start(harness, fake, ["--port", str(port), "--tp_size", "4"])
    engine = _record(fake.records, "engine-start")["argv"]
    frontend = _record(fake.records, "frontend-start")["argv"]
    engine_port = handle.engine_endpoint.rsplit(":", 1)[1]
    # dsv4-serve <ckpt> --grpc --grpc-protocol openengine, with the address the harness picks;
    # no TensorRT-LLM-only engine options (metrics port, load metrics).
    assert engine == [
        "--ds-flag",
        "/model",
        "--grpc",
        "--grpc-protocol",
        "openengine",
        "--host",
        "127.0.0.1",
        "--port",
        engine_port,
        "--tp_size",
        "4",
    ]
    # The front end is still the trtllm-serve of the command line, pointed at the ds engine.
    assert frontend[:3] == ["/model", "--openengine-endpoint", handle.engine_endpoint]
    assert frontend[frontend.index("--port") + 1] == str(port)
    assert "--tp_size" not in frontend
    handle.terminate(grace=1)


def test_ds_chain_refuses_a_metrics_port_the_command_gives(harness, fake, monkeypatch):
    cv, launch = harness
    monkeypatch.setattr(launch, "_HANDLES", [])
    monkeypatch.setenv(cv.ENV_DS_SERVE_CMD, fake.ds)
    args = ["--port", str(_port()), launch.ENGINE_METRICS_PORT_OPTION, "18001"]
    with pytest.raises(launch.ChainError, match="not pass it to the ds chain's engine"):
        _start(harness, fake, args)
