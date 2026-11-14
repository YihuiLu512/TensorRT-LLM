# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4.1 module's own machinery on CPU: form K's command lines and start, and collection.

Form K is started here with a stand-in for the ds-engine command: its ``disaggregated``
subcommand answers ``/health`` on the proxy port of its configuration, its workers sleep,
optionally with a child in their process group that ignores SIGTERM.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

_UNITTEST = Path(__file__).resolve().parents[3]
_APPS = _UNITTEST / "llmapi" / "apps"
_BASIC = _APPS / "_test_openengine_ds_disagg_basic.py"
_FRONTEND = _APPS / "_test_openengine_ds_disagg_frontend.py"
_CONVENTIONS = _UNITTEST.parent / "integration" / "defs" / "openengine_chain" / "conventions.py"
_SPLIT = _CONVENTIONS.parent / "split.py"

_FAKE_DS = """\
import http.server, os, sys, time, yaml
args = sys.argv[1:]
if args and args[0] == "disaggregated":
    if os.environ.get("FAKE_PROXY_EXIT"):
        print("proxy refused to start"); sys.exit(3)
    config = yaml.safe_load(open(args[args.index("-c") + 1]))

    class Health(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/health" else 404)
            self.end_headers()

        def log_message(self, *a):
            pass

    http.server.HTTPServer((config["hostname"], config["port"]), Health).serve_forever()
else:
    print("worker", os.environ.get("CUDA_VISIBLE_DEVICES"), " ".join(args), flush=True)
    if os.environ.get("FAKE_WORKER_CHILD"):
        import subprocess
        child = (
            "import os, signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(3600)"
        )
        pid_file = os.environ["FAKE_WORKER_CHILD"] + "." + os.environ["CUDA_VISIBLE_DEVICES"]
        subprocess.Popen([sys.executable, "-c", child, pid_file])
    time.sleep(3600)
"""


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up while the module runs.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[spec.name]
    return module


@pytest.fixture(scope="module")
def basic():
    """The module's basic file, loaded by path: its helpers import nothing from the test package."""
    return _load("ds_disagg_basic_under_test", _BASIC)


def test_environment_names_are_the_harness_ones(basic):
    conventions = _load("openengine_chain_conventions_under_test", _CONVENTIONS)
    assert basic.ENV_CHAIN == conventions.ENV_CHAIN
    assert basic.ENV_DS_SERVE_CMD == conventions.ENV_DS_SERVE_CMD


def test_free_ports_are_distinct(basic):
    ports = basic.free_ports(16)
    assert len(set(ports)) == 16


def test_native_processes_are_test_disagg_sh(basic):
    ports = basic.NativePorts(proxy=8030, context=8031, generation=8032)
    configs = {"context": "c.yaml", "generation": "g.yaml", "proxy": "p.yaml"}
    context, generation, proxy = basic.native_processes(
        ["python3", "-m", "dsv4_serve"], "/ckpt", ports, configs
    )
    common = [
        "--max_batch_size",
        "32",
        "--max_num_tokens",
        "8192",
        "--max_seq_len",
        "32768",
        "--kv_cache_free_gpu_memory_fraction",
        "0.5",
        "--custom_tokenizer",
        "deepseek_v41",
    ]
    for process, role, port, gpus, config in (
        (context, "context", "8031", "0,1", "c.yaml"),
        (generation, "generation", "8032", "2,3", "g.yaml"),
    ):
        assert process.name == role
        assert list(process.argv) == [
            "python3",
            "-m",
            "dsv4_serve",
            "/ckpt",
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--tp_size",
            "2",
            "--server_role",
            role,
            *common,
            "--extra_llm_api_options",
            config,
        ]
        assert dict(process.env) == {"CUDA_VISIBLE_DEVICES": gpus}
    assert proxy.name == "proxy" and not proxy.env
    assert list(proxy.argv) == [
        "python3",
        "-m",
        "dsv4_serve",
        "disaggregated",
        "-c",
        "p.yaml",
        "-t",
        "1200",
    ]
    config = basic.proxy_config(ports)
    assert config["port"] == 8030
    # The workers listen on 127.0.0.1, so the proxy addresses them there.
    assert config["hostname"] == "127.0.0.1"
    assert config["context_servers"]["urls"] == ["127.0.0.1:8031"]
    assert config["generation_servers"]["urls"] == ["127.0.0.1:8032"]
    assert config["context_servers"]["router"] == {"type": "conversation"}


def test_chain_args_share_the_engine_options(basic):
    args = basic.chain_serve_args()
    assert args[:2] == ["--tp_size", "4"]
    assert args[2 : 2 + len(basic.ENGINE_ARGS)] == list(basic.ENGINE_ARGS)
    for option in ("--custom_tokenizer", "--reasoning_parser", "--tool_parser"):
        assert args[args.index(option) + 1] == "deepseek_v41"


def test_chain_args_split_between_the_ds_engine_and_the_front_end(basic):
    split = _load("openengine_chain_split_under_test", _SPLIT)
    if "--custom_tokenizer" not in split.OPTIONS:
        pytest.skip("the chain split table does not list --custom_tokenizer")
    parts = split.split_args(basic.chain_serve_args(), rewrite_yaml=lambda path: path)
    reasoning = ["--reasoning_parser", "deepseek_v41"]
    assert list(parts.engine) == [
        "--tp_size",
        "4",
        *basic.ENGINE_ARGS,
        *basic.TOKENIZER_ARGS,
        *reasoning,
    ]
    assert list(parts.frontend) == [
        *basic.TOKENIZER_ARGS,
        *basic.PARSER_ARGS,
        "--served_model_name",
        basic.MODEL_NAME,
    ]


def test_ds_command_reads_the_environment(basic):
    assert basic.ds_command({}) == []
    env = {basic.ENV_DS_SERVE_CMD: "python3 -m dsv4_serve"}
    assert basic.ds_command(env) == ["python3", "-m", "dsv4_serve"]
    with pytest.raises(ValueError, match=basic.ENV_DS_SERVE_CMD):
        basic.DsNative([], "/ckpt")


@pytest.fixture
def fake_ds(tmp_path):
    script = tmp_path / "fake_ds.py"
    script.write_text(_FAKE_DS)
    return [sys.executable, str(script)]


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_form_k_starts_and_stops(basic, fake_ds, tmp_path):
    native = basic.DsNative(fake_ds, "/ckpt", workdir=str(tmp_path / "k"), ready_timeout=60)
    with native:
        import requests

        assert requests.get(f"{native.url}/health", timeout=5).status_code == 200
        pids = [proc.pid for proc in native._running.values()]
        assert len(pids) == 3
        written = yaml.safe_load((tmp_path / "k" / "context.yaml").read_text())
        assert written == basic.CONTEXT_CONFIG
    assert not any(_alive(pid) for pid in pids)
    log = (tmp_path / "k" / "generation.log").read_text()
    assert log.startswith("worker 2,3 /ckpt --host 127.0.0.1")


def test_form_k_stops_what_outlives_a_worker(basic, fake_ds, tmp_path):
    """A process left in a worker's group after the worker exits on SIGTERM is killed too."""
    env = dict(os.environ, FAKE_WORKER_CHILD=str(tmp_path / "child"))
    native = basic.DsNative(
        fake_ds, "/ckpt", workdir=str(tmp_path / "k"), env=env, ready_timeout=60
    )
    with native:
        pid_files = [tmp_path / f"child.{gpus}" for gpus in ("0,1", "2,3")]
        deadline = time.monotonic() + 30
        while not all(f.is_file() and f.read_text() for f in pid_files):
            assert time.monotonic() < deadline, "the workers' children did not start"
            time.sleep(0.1)
        children = [int(f.read_text()) for f in pid_files]
        assert all(_alive(pid) for pid in children)
    deadline = time.monotonic() + 10
    while any(_alive(pid) for pid in children) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(_alive(pid) for pid in children)


def test_form_k_reports_a_process_that_exited(basic, fake_ds, tmp_path):
    env = dict(os.environ, FAKE_PROXY_EXIT="1")
    native = basic.DsNative(
        fake_ds, "/ckpt", workdir=str(tmp_path / "k"), env=env, ready_timeout=60
    )
    with pytest.raises(RuntimeError, match="the proxy process exited with 3") as info:
        native.__enter__()
    assert "proxy refused to start" in str(info.value)
    assert not native._running


def test_the_module_files_are_collected():
    """The module's two files are collected; their GPU tests skip without the chain and model."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"]
        + [str(path.relative_to(_UNITTEST)) for path in (_BASIC, _FRONTEND)],
        cwd=_UNITTEST,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    collected = [line for line in result.stdout.splitlines() if "::" in line]
    assert any(line.endswith("::test_startup_tokenizer_and_fingerprint") for line in collected)
    assert any("::test_chat_template_and_kwargs" in line for line in collected)
    assert sum("::test_reasoning_and_tools" in line for line in collected) == 2
