# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A context leg sent to a lost context worker is retried with a new id until it is served (DG-45).

The deployment is ``_test_openengine_disagg_common.disagg_deployment`` with two
context workers and one generation worker of Qwen3-0.6B on one GPU, in-process
or on the OpenEngine chain. The orchestrator retries a failed context leg
``max_retries`` times, one second apart, on the worker it first chose, with a
new ``disagg_request_id`` each time; it writes a perf record of each request
with the id it used last. On the chain the worker front ends write every
GenerateRequest they send to ``TRTLLM_OPENENGINE_REQUEST_DUMP_DIR``.

- NT-095 (DG-45): the first context worker is killed (on the chain its front
  end, whose engine lives on) and, right after requests are sent, started
  again on its port. The requests all complete, the orchestrator retried at
  least one context leg, and one of them was served by the restarted worker.
  On the chain the context and the generation engine got each request under
  the id of the orchestrator's record.

The test changes the deployment, so it has one of its own.
"""

import contextlib
import json
import os
import re
import threading
import time
from pathlib import Path

import psutil
import pytest
import requests
import yaml

from tensorrt_llm.grpc import openengine_conventions as conv

from ._test_openengine_disagg_common import (
    CHAIN,
    DUMP_ENV,
    ENGINE_READY_TIMEOUT_S,
    REQUEST_TIMEOUT_S,
    disagg_deployment,
    model_path,
)
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

NUM_CTX = 2
# Round-robin over two context workers: one of two requests goes to the lost one.
NUM_REQUESTS = 2
MAX_TOKENS = 4
# Retries one second apart: enough for an in-process worker to load the model again.
MAX_RETRIES = 900
RETRIED_REQUEST_TIMEOUT_S = MAX_RETRIES + 300
RECORD_WAIT_S = 30
PROMPT = "The capital of France is"
_CTX_RETRIES = re.compile(r"^ctx_retry_requests_total(?:\{[^}]*\})?\s+(\S+)\s*$", re.MULTILINE)


@pytest.fixture(scope="module")
def dirs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("disagg_retry")
    return {name: root / name for name in ("logs", "perf", "dumps")}


@pytest.fixture(scope="module")
def deployment(dirs):
    dirs["logs"].mkdir(parents=True, exist_ok=True)
    with disagg_deployment(
        dirs["logs"],
        counts={"context": NUM_CTX},
        orchestrator_config={
            "max_retries": MAX_RETRIES,
            "perf_metrics_output_dir": str(dirs["perf"]),
        },
        env={DUMP_ENV: str(dirs["dumps"])},
    ) as deployment:
        yield deployment


@pytest.fixture(scope="module")
def restarted(deployment):
    """Workers the test starts again, stopped before the deployment."""
    with contextlib.ExitStack() as stack:
        yield stack


def _ctx_retries(deployment) -> int:
    response = requests.get(f"{deployment.url}/prometheus/metrics", timeout=REQUEST_TIMEOUT_S)
    assert response.ok, response.text
    return sum(int(float(value)) for value in _CTX_RETRIES.findall(response.text))


def _records(perf_dir: Path) -> list[dict]:
    """The orchestrator's perf records so far, in the order it wrote them."""
    records = []
    for path in sorted(perf_dir.glob("perf_metrics-disagg-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _dumps(dump_dir: Path) -> dict[str, dict]:
    return {path.name: json.loads(path.read_text()) for path in dump_dir.glob("*.json")}


def _kill(pid: int) -> None:
    """SIGKILL ``pid`` and the processes below it, as a lost host would stop them."""
    with contextlib.suppress(psutil.NoSuchProcess):
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
        for proc in procs:
            with contextlib.suppress(psutil.NoSuchProcess):
                proc.kill()
        psutil.wait_procs(procs, timeout=60)


def _lose_and_restart(deployment, restarted, logs: Path):
    """Kill the first context worker; return a function that starts it again on its port.

    On the chain only its front end is killed and started again; its engine lives on.
    In-process the worker is started again with the configuration it had.
    """
    worker = deployment.workers["context"][0]
    if CHAIN:
        handle = worker._chain_handle
        _kill(handle.frontend.pid)

        def restart():
            handle.restart_frontend(handle.port)
            worker.proc = handle.frontend

        return restart

    config = yaml.safe_load(Path(worker.extra_config_file).read_text())
    _kill(worker.proc.pid)
    worker.proc.wait()

    def restart():
        server = restarted.enter_context(
            RemoteOpenAIServer(
                model_path(deployment.model),
                port=worker.port,
                # As disagg_deployment starts it; an in-process worker writes no dumps.
                env={**os.environ, "UCX_NET_DEVICES": "lo"},
                cli_args=["--server_role", "context"],
                extra_config=config,
                log_path=str(logs / "context_0_restarted.log"),
                wait=False,
            )
        )
        # Raises as soon as the restarted worker exits, rather than after the retries run out.
        server.wait_for_server(timeout=ENGINE_READY_TIMEOUT_S)

    return restart


def _complete(deployment, results: list, index: int) -> None:
    body = {
        "model": deployment.model,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "temperature": 0.0,
    }
    try:
        results[index] = requests.post(
            f"{deployment.url}/v1/completions", json=body, timeout=RETRIED_REQUEST_TIMEOUT_S
        )
    except requests.RequestException as error:  # reported by the test
        results[index] = error


def test_request_survives_a_lost_ctx_worker_with_a_new_id(deployment, restarted, dirs):
    retries = _ctx_retries(deployment)
    seen_records = len(_records(dirs["perf"]))
    seen_dumps = set(_dumps(dirs["dumps"]))

    restart = _lose_and_restart(deployment, restarted, dirs["logs"])
    # Sent before the orchestrator drops the lost worker, which takes up to its
    # inactive_timeout_sec without a heartbeat.
    results: list = [None] * NUM_REQUESTS
    threads = [
        threading.Thread(target=_complete, args=(deployment, results, i))
        for i in range(NUM_REQUESTS)
    ]
    for thread in threads:
        thread.start()
    restart()
    for thread in threads:
        thread.join(RETRIED_REQUEST_TIMEOUT_S)
        assert not thread.is_alive(), "a request did not finish"

    for result in results:
        assert isinstance(result, requests.Response), result
        assert result.status_code == 200, result.text
        assert result.json()["usage"]["completion_tokens"] == MAX_TOKENS, result.text
    assert _ctx_retries(deployment) > retries, "no context leg was retried"

    deadline = time.monotonic() + RECORD_WAIT_S
    while len(records := _records(dirs["perf"])) < seen_records + NUM_REQUESTS:
        assert time.monotonic() < deadline, "the orchestrator wrote no perf record of a request"
        time.sleep(0.1)
    new_records = records[seen_records:]
    lost_port = deployment.workers["context"][0].port
    assert any(int(r["ctx_server"].rsplit(":", 1)[1]) == lost_port for r in new_records), (
        lost_port,
        new_records,
    )
    final_ids = sorted(int(r["disagg_request_id"]) for r in new_records)

    if CHAIN:
        dumps = [d for name, d in _dumps(dirs["dumps"]).items() if name not in seen_dumps]
        ctx_ids = sorted(
            conv.as_int(d["extra"][conv.REQ_DISAGG_REQUEST_ID])
            for d in dumps
            if d.get("extra", {}).get(conv.REQ_REQUEST_TYPE) == conv.REQUEST_TYPE_CONTEXT_ONLY
        )
        gen_ids = sorted(
            conv.as_int(d["kv"]["session"]["attributes_struct"]["disagg_request_id"])
            for d in dumps
            if "session" in d.get("kv", {})
        )
        # A restarted front end may refuse a context leg it already wrote down; the
        # orchestrator sends that leg again under a new id, so ids may be left unused.
        assert set(final_ids) <= set(ctx_ids), (ctx_ids, final_ids)
        assert gen_ids == final_ids, (gen_ids, final_ids)
