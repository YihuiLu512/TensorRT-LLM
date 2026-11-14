# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Perf metrics of a disaggregated request, per leg, and their offline join (DG-40, DG-54, EG-23).

The orchestrator and a context and a generation worker, each with
``--server_role``, register through the orchestrator's cluster and write their
perf records to one directory. The tests run on the in-process workers as well
as on the OpenEngine chain (``TRTLLM_TEST_OPENENGINE_CHAIN``,
``tests/integration/defs/openengine_chain``), where each worker is a frontend
and a TensorRT-LLM OpenEngine server:

- NT-096: once the orchestrator is ready, each worker answers
  ``/steady_clock_offset`` and was given the orchestrator's clock offset.
- NT-203: the three records of a request carry its ``disagg_request_id``, the
  orchestrator's holds both legs' metrics, their times are in order on the
  orchestrator's clock (``validate_timing_metrics``),
  and ``scripts/poc3/wt54_join_perf.py`` joins them. On the chain the KV-cache
  transfer time and size come from the engines' leg records, which the join
  adds (the frontends cannot measure them); in-process the generation
  worker's own record holds them. A streamed response carries no perf
  metrics event.

``test_engine_leg_records_of_a_direct_handoff`` needs no frontend: two engines
complete a handoff over OpenEngine gRPC, and each writes the record of its leg.

Needs one GPU of Hopper or later and ``${LLM_MODELS_ROOT}/Qwen3/Qwen3-0.6B``.
"""

import json
import os
import platform
import secrets
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import pytest
import requests
import torch
from test_common.http_utils import wait_for_endpoint_ready
from test_common.perf_metrics_utils import (
    get_timing_metrics,
    validate_timing_metrics,
    wait_for_perf_metrics_jsonl,
)
from utils.llm_data import llm_models_root

from tensorrt_llm._utils import get_free_ports
from tensorrt_llm.grpc.openengine.streaming import LEG_DURATIONS_FILE_PREFIX
from tensorrt_llm.serve.perf_metrics import (
    RETURN_METRICS_HEADER,
    SERVER_TIMING_HEADER,
    SSE_METRICS_EVENT,
    START_END_TIME_HEADER,
)

from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

__extra_import_path__ = ["~/tests/integration/defs"]
from openengine_chain import direct_replay as dr  # noqa: E402

pytestmark = [pytest.mark.threadleak(enabled=False)]

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
MODEL_SUBDIR = os.path.join("Qwen3", "Qwen3-0.6B")
MODEL_NAME = "Qwen3/Qwen3-0.6B"
JOIN_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "poc3" / "wt54_join_perf.py"
# Logged by a worker when the orchestrator sets its clock offset (OpenAIServer.set_steady_clock_offset).
OFFSET_LOG = "The steady clock offset between local and disagg server"
REQUEST_TIMEOUT_S = 120
RECORD_WAIT_S = 30
ENGINE_READY_TIMEOUT_S = 1200
CALLER_DISAGG_REQUEST_ID = 4_611_686_018_427_387_905


def _model_dir() -> str:
    root = llm_models_root()
    path = None if root is None else os.path.join(str(root), MODEL_SUBDIR)
    assert path is not None and os.path.isdir(path), f"{MODEL_SUBDIR} is not under LLM_MODELS_ROOT"
    return path


def _needs_gpu() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (9, 0):
        pytest.skip("needs a GPU of Hopper or later")


@pytest.fixture(scope="module")
def chain_dirs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("perf_metrics_chain")
    return {"perf": directory / "perf", "logs": directory}


@pytest.fixture(scope="module")
def served(chain_dirs):
    """An orchestrator and two workers with roles, registered through the orchestrator's cluster."""
    _needs_gpu()
    model = _model_dir()
    disagg_port, ctx_port, gen_port = get_free_ports(3)
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}
    auth_key = secrets.token_hex(32)
    perf_dir = str(chain_dirs["perf"])
    worker_config = {
        "model": MODEL_NAME,
        "disagg_cluster": cluster,
        "internal_request_auth_key": auth_key,
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "kv_cache_config": {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False},
        "disable_overlap_scheduler": True,
        "cuda_graph_config": None,
        "return_perf_metrics": True,
        "perf_metrics_output_dir": perf_dir,
        "perf_metrics_max_requests": 1000,
    }
    # Both workers run on this host: loopback is the only TCP device UCX needs.
    env = {**os.environ, "UCX_NET_DEVICES": "lo"}
    logs = {}

    def worker(role: str, port: int) -> RemoteOpenAIServer:
        logs[role] = str(chain_dirs["logs"] / f"output_{role}.log")
        return RemoteOpenAIServer(
            model,
            port=port,
            env=env,
            cli_args=["--server_role", role],
            llmapi_launch=False,
            extra_config=worker_config,
            log_path=logs[role],
            wait=False,
        )

    disagg_config = {
        "hostname": "localhost",
        "port": disagg_port,
        "disagg_cluster": cluster,
        "internal_request_auth_key": auth_key,
        "perf_metrics_max_requests": 1000,
        "return_perf_metrics": True,
        "perf_metrics_output_dir": perf_dir,
    }
    with worker("context", ctx_port) as ctx, worker("generation", gen_port) as gen:
        with RemoteDisaggOpenAIServer(
            ctx_servers=[],
            gen_servers=[],
            port=disagg_port,
            llmapi_launch=False,
            disagg_config=disagg_config,
        ) as orchestrator:
            for server in (ctx, gen):
                server.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)
            wait_for_endpoint_ready(orchestrator.url_root + "/health")
            yield {
                "orchestrator": orchestrator,
                "workers": {"context": ctx, "generation": gen},
                "logs": logs,
            }


def _wait_for_log(path: str, text: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path) and text in Path(path).read_text(errors="replace"):
            return True
        time.sleep(0.5)
    return False


def test_clock_offset_reaches_each_worker_frontend(served):
    """NT-096 (DG-54): each worker answers the clock probe and took the orchestrator's offset."""
    for role, server in served["workers"].items():
        response = requests.get(server.url_for("steady_clock_offset"), timeout=REQUEST_TIMEOUT_S)
        assert response.status_code == 200, (role, response.text)
        body = response.json()
        assert 0 < body["receive_ts"] <= body["transmit_ts"], (role, body)
        assert _wait_for_log(served["logs"][role], OFFSET_LOG, RECORD_WAIT_S), role


def _join(perf_dir: Path, require: str) -> list[dict]:
    output = perf_dir.parent / f"joined-{require}.jsonl"
    result = subprocess.run(  # nosec B603
        [
            sys.executable,
            str(JOIN_SCRIPT),
            str(perf_dir),
            "--require",
            require,
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return [json.loads(line) for line in output.read_text().splitlines()]


def _leg_records(directory: Path, disagg_request_id: int | None = None, count: int = 1):
    """The engines' leg records in ``directory`` (of one request if given), once ``count`` are."""
    deadline = time.monotonic() + RECORD_WAIT_S
    while True:
        records = [
            json.loads(line)
            for path in directory.glob(f"{LEG_DURATIONS_FILE_PREFIX}-*.jsonl")
            for line in path.read_text().splitlines()
            if line
        ]
        if disagg_request_id is not None:
            records = [r for r in records if r.get("disagg_request_id") == disagg_request_id]
        if len(records) >= count or time.monotonic() > deadline:
            return records
        time.sleep(0.5)


def test_engine_side_transfer_stats_join_by_disagg_request_id(served, chain_dirs):
    """NT-203 (EG-23, DG-40): the records of a request join by its disagg_request_id."""
    perf_dir = chain_dirs["perf"]
    before = len(wait_for_perf_metrics_jsonl(perf_dir, expected_count=0))
    payload = {
        "model": MODEL_NAME,
        "prompt": "Reply with one token.",
        "max_tokens": 4,
        "temperature": 0.0,
    }
    response = requests.post(
        f"{served['orchestrator'].url_root}/v1/completions",
        headers={RETURN_METRICS_HEADER: "1"},
        json=payload,
        timeout=REQUEST_TIMEOUT_S,
    )
    assert response.status_code == 200, response.text
    # The frontends' measurements reach the response; the per-step headers are the engine's.
    for header in (SERVER_TIMING_HEADER, START_END_TIME_HEADER):
        assert response.headers.get(header), header

    records = wait_for_perf_metrics_jsonl(perf_dir, expected_count=before + 3)
    disagg = [r for r in records if "ctx_server" in r]
    disagg_request_id = disagg[-1]["disagg_request_id"]
    mine = [r for r in records if r.get("disagg_request_id") == disagg_request_id]
    assert len(mine) == 3, mine
    workers = [r for r in mine if "ctx_server" not in r]
    assert all(r["status"] == "complete" for r in mine), mine
    assert {"ctx_perf_metrics", "gen_perf_metrics"} <= disagg[-1].keys(), disagg[-1]
    assert workers[0]["ctx_request_id"] == workers[1]["ctx_request_id"], workers
    # This module's only request so far, so the joined request is this one.
    validate_timing_metrics(get_timing_metrics(perf_dir), "NT-203")

    if CHAIN:
        legs = _leg_records(perf_dir, disagg_request_id, count=2)
        assert sorted(r["leg"] for r in legs) == ["context", "generation"], legs
        (entry,) = [
            e for e in _join(perf_dir, "engines") if e["disagg_request_id"] == disagg_request_id
        ]
        assert entry["missing"] == [], entry
        assert entry["kv_cache_transfer_ms"] is not None and entry["kv_cache_transfer_ms"] > 0, (
            entry
        )
        assert entry["kv_cache_size"] is not None and entry["kv_cache_size"] > 0, entry
    else:
        (entry,) = [
            e for e in _join(perf_dir, "frontends") if e["disagg_request_id"] == disagg_request_id
        ]
        assert entry["missing"] == [], entry
        timing = entry["generation_frontend"]["perf_metrics"]["timing_metrics"]
        assert {"kv_cache_transfer_start", "kv_cache_transfer_end"} <= timing.keys(), timing

    streamed = requests.post(
        f"{served['orchestrator'].url_root}/v1/completions",
        json={**payload, "stream": True},
        timeout=REQUEST_TIMEOUT_S,
    )
    assert streamed.status_code == 200, streamed.text
    assert "data: [DONE]" in streamed.text
    assert f"event: {SSE_METRICS_EVENT}" not in streamed.text


# -- Two engines without a frontend -------------------------------------------------


def _ucx_tls() -> str:
    """The UCX transports ``get_ucx_tls`` (``tests/integration/defs/common.py``) picks on Hopper or later."""
    if torch.cuda.get_device_capability(0) == (10, 3) and "aarch" in platform.machine().lower():
        return "cuda_copy,cuda_ipc,sm,self,tcp"
    return "^ib,gdr_copy"


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    _needs_gpu()
    model = _model_dir()
    workdir = tmp_path_factory.mktemp("direct_legs")
    kv_cache_config = {"free_gpu_memory_fraction": 0.2, "enable_block_reuse": False}
    started = []
    perf_dirs = {}
    try:
        # One after the other, so that neither measures free memory while the other is loading.
        for name in ("ctx", "gen"):
            perf_dirs[name] = workdir / f"perf-{name}"
            config = {
                "kv_cache_config": kv_cache_config,
                "cache_transceiver_config": {"backend": "DEFAULT"},
                "perf_metrics_output_dir": str(perf_dirs[name]),
            }
            if name == "ctx":
                config["disable_overlap_scheduler"] = True
            env = dict(os.environ, UCX_TLS=_ucx_tls(), UCX_NET_DEVICES="lo")
            env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            started.append(dr.start_engine(name, model, str(workdir), config=config, env=env))
            started[-1].wait_serving(ENGINE_READY_TIMEOUT_S)
        yield started[0], started[1], perf_dirs
    finally:
        # Engine.stop() never raises, so every engine is stopped.
        for engine in started:
            engine.stop()


def test_engine_leg_records_of_a_direct_handoff(engines):
    """CO-08 on real engines: each engine writes one record of its leg, durations only."""
    ctx_engine, gen_engine, perf_dirs = engines
    handoff = dr.run_handoff(
        ctx_engine.endpoint,
        gen_engine.endpoint,
        MODEL_NAME,
        "The capital of France is Paris. The capital of Germany is",
        max_tokens=8,
        timeout=REQUEST_TIMEOUT_S,
        ctx_options={"extra": {"disagg_request_id": str(CALLER_DISAGG_REQUEST_ID)}},
    )
    result = handoff.to_json()
    assert handoff.ctx.status == "OK" and handoff.ctx.terminal() == "prefill_ready", result
    assert handoff.gen is not None and handoff.gen.status == "OK", result

    (ctx_record,) = _leg_records(perf_dirs["ctx"])
    (gen_record,) = _leg_records(perf_dirs["gen"])
    assert ctx_record["leg"] == "context" and gen_record["leg"] == "generation"
    for record in (ctx_record, gen_record):
        assert record["disagg_request_id"] == CALLER_DISAGG_REQUEST_ID, record
        durations = record["durations_ms"]
        assert durations["arrival_to_first_token"] > 0, record
        assert all(value is None or value >= 0 for value in durations.values()), record
    assert ctx_record["ctx_request_id"] == gen_record["ctx_request_id"] is not None
    # The generation engine measured the transfer it received.
    assert gen_record["durations_ms"]["kv_cache_transfer"] is not None, gen_record
    assert gen_record["kv_cache_size"] is not None and gen_record["kv_cache_size"] > 0, gen_record
