# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A context worker with two serving frontends on the CPU chain (SV-DSG-15; CS-17, UV-68).

The context worker frontend runs with ``num_serve_frontends: 2`` and
``serve_frontend_affinity_port_stride`` (DG-23), and the orchestrator is told
both, so it sends a request with a conversation id to the affinity port of
the frontend ``crc32(conversation id) % 2`` picks. Requests picked for each
frontend go through both legs; each frontend serves its own, with the same
mapping, ``PrefillReady`` path and counters, and the frontends' counts add up
to what the context engine received. The context frontends have no
``internal_request_auth_key``, so they refuse subagent affinity headers.
"""

from __future__ import annotations

import os
import shutil
import socket
import tempfile
import zlib

import pytest
import requests
import yaml

from tensorrt_llm.grpc.openengine.affinity import AFFINITY_AUTH_HEADER, AFFINITY_ID_HEADER
from tensorrt_llm.grpc.openengine_conventions import (
    BUCKET_CONTEXT_ONLY,
    DISAGG_REQUEST_ID_MIN,
    SERVE_COUNTER_DIR_ENV,
    SERVE_COUNTER_GENERATE_SENT,
    SERVE_COUNTER_GENERATE_SENT_BUCKETS,
    SERVE_COUNTER_HTTP_REJECTED,
    SERVE_COUNTER_PREFILL_READY_RECEIVED,
    SERVE_REQUEST_DUMP_DIR_ENV,
)
from tensorrt_llm.serve.openengine_poc import contracts

__extra_import_path__ = [".."]
from cpu_disagg_chain import (  # noqa: E402
    _RECORDER,
    MODEL,
    CpuDisaggChain,
    _Process,
    _process_env,
    _serve_command,
)
from cpu_disagg_chain_recorder import HTTP_LOG_ENV  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

FRONTENDS = 2
STRIDE = 7


def _free_port_with_affinity_ports() -> int:
    """A port P with P + STRIDE * (i + 1) free too, for each frontend i; checked, not reserved."""
    for _ in range(100):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        ports = [port + STRIDE * (i + 1) for i in range(FRONTENDS)]
        if ports[-1] >= 65536:
            continue
        sockets = []
        try:
            for candidate in (port, *ports):
                sock = socket.socket()
                sockets.append(sock)
                sock.bind(("127.0.0.1", candidate))
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
        return port
    raise RuntimeError("no port with free affinity ports")


class _MultiFrontendChain(CpuDisaggChain):
    """The chain, with a context worker frontend of :data:`FRONTENDS` serving frontends.

    Several frontends share a fixed port, which they cannot report, so the
    address is written for them; the orchestrator learns their affinity ports.
    """

    def __init__(self, directory, **kwargs) -> None:
        self.ctx_port = _free_port_with_affinity_ports()
        super().__init__(directory, **kwargs)

    def _start_frontend(self, worker) -> None:
        if worker.role != "context":
            super()._start_frontend(worker)
            return
        # As the base class starts it, on a fixed port and without --report_addr,
        # which several frontends cannot use; the address is written for them.
        for path in (worker.directory, worker.counter_dir, worker.dump_dir):
            path.mkdir(parents=True, exist_ok=True)
        (worker.directory / "addr").write_text(f"127.0.0.1:{self.ctx_port}")
        command = [
            *_serve_command(),
            str(self.model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.ctx_port),
            "--served_model_name",
            MODEL,
            "--openengine-endpoint",
            worker.engine.target,
            "--openengine-ready-timeout",
            "60",
            "--middleware",
            _RECORDER,
            *self.frontend_args[worker.role],
        ]
        env = _process_env(
            {
                SERVE_COUNTER_DIR_ENV: str(worker.counter_dir),
                SERVE_REQUEST_DUMP_DIR_ENV: str(worker.dump_dir),
                HTTP_LOG_ENV: str(worker.http_log),
                **self.env,
            }
        )
        worker.process = _Process(command, env, worker.directory / "frontend.log")

    def _start_orchestrator(self) -> None:
        self.orchestrator_config["context_servers"] = {
            "num_instances": 1,
            "urls": [self.ctx.host_port],
            "router": {"type": "round_robin"},
            "num_serve_frontends": FRONTENDS,
            "serve_frontend_affinity_port_stride": STRIDE,
        }
        super()._start_orchestrator()


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("multi")
    configs = {}
    for role, extra in (
        (
            "context",
            {"num_serve_frontends": FRONTENDS, "serve_frontend_affinity_port_stride": STRIDE},
        ),
        ("generation", {}),
    ):
        path = directory / f"{role}.yaml"
        path.write_text(
            yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}, **extra})
        )
        configs[role] = ["--config", str(path)]
    # The launcher's Unix socket path must stay short.
    ipc = tempfile.mkdtemp(prefix="wt53-", dir="/tmp" if os.path.isdir("/tmp") else None)
    try:
        with _MultiFrontendChain(directory, frontend_args=configs, env={"TMPDIR": ipc}) as chain:
            yield chain
    finally:
        shutil.rmtree(ipc, ignore_errors=True)


def _frontend_for(conversation_id: str) -> int:
    """The frontend the orchestrator sends this conversation's context requests to."""
    return zlib.crc32(conversation_id.encode()) % FRONTENDS


def _conversations() -> dict[int, str]:
    """A conversation id for each frontend."""
    found: dict[int, str] = {}
    for n in range(1000):
        found.setdefault(_frontend_for(f"conversation-{n}"), f"conversation-{n}")
        if len(found) == FRONTENDS:
            return found
    raise AssertionError("no conversation id for some frontend")


def _conversation_of(record: dict) -> str | None:
    """The conversation id of a request the orchestrator sent a worker frontend."""
    body = record.get("body")
    params = body.get("conversation_params") if isinstance(body, dict) else None
    return (params or {}).get("conversation_id") or record["headers"].get("x-session-id")


def _complete(chain, conversation_id: str) -> requests.Response:
    return requests.post(
        f"{chain.url}/v1/completions",
        json={"model": MODEL, "prompt": "He llo", "max_tokens": 3},
        headers={"Connection": "close", "x-session-id": conversation_id},
        timeout=60,
    )


def test_multi_frontend_worker_two_stage(chain):
    """F-39 and DG-23 on CPU: each context frontend serves the two-stage requests sent to it."""
    sent_key = SERVE_COUNTER_GENERATE_SENT_BUCKETS[BUCKET_CONTEXT_ONLY]
    before = contracts.read_serve_counters_by_pid(chain.ctx.counter_dir)
    engine_before = len(chain.ctx.engine.received)
    conversations = _conversations()
    per_frontend = 2
    for _ in range(per_frontend):
        for conversation_id in conversations.values():
            response = _complete(chain, conversation_id)
            assert response.status_code == 200, chain.orchestrator_log()[-4000:]
            assert response.json()["choices"][0]["text"]

    # Each conversation's requests reached one frontend process, and different
    # conversations' requests reached different ones: the affinity ports work.
    served_by: dict[str, set[int]] = {}
    for record in chain.ctx.http_requests():
        if record["path"] != "/v1/completions":
            continue
        served_by.setdefault(_conversation_of(record), set()).add(record["pid"])
    pids = [served_by.get(conversation_id, set()) for conversation_id in conversations.values()]
    assert all(len(p) == 1 for p in pids), served_by
    assert len(set.union(*pids)) == FRONTENDS, served_by

    after = contracts.read_serve_counters_by_pid(chain.ctx.counter_dir)
    assert len(after) == FRONTENDS
    sent = {pid: after[pid][sent_key] - before.get(pid, {}).get(sent_key, 0) for pid in after}
    assert sorted(sent.values()) == [per_frontend] * FRONTENDS, sent
    ready = sum(
        after[pid][SERVE_COUNTER_PREFILL_READY_RECEIVED]
        - before.get(pid, {}).get(SERVE_COUNTER_PREFILL_READY_RECEIVED, 0)
        for pid in after
    )
    # Every frontend's prefill calls reached the one context engine.
    engine_calls = chain.ctx.engine.received[engine_before:]
    assert len(engine_calls) == sum(sent.values()) == ready


def test_affinity_headers_without_a_key_send_no_generate(chain):
    """SV-DSG-14: a worker frontend without internal_request_auth_key refuses affinity headers.

    The context frontends here have no key; the request is refused before any
    Generate call, on whichever frontend took it.
    """
    received = len(chain.ctx.engine.received)
    before = chain.ctx.serve_counters()
    # Chat requests carry the hint; the completions endpoint does not read it.
    response = requests.post(
        f"{chain.ctx.url}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "He llo"}],
            "max_tokens": 3,
            "conversation_params": {"conversation_id": "child-53"},
            "disaggregated_params": {
                "request_type": "context_only",
                "disagg_request_id": DISAGG_REQUEST_ID_MIN + 530,
            },
        },
        headers={
            "Connection": "close",
            AFFINITY_ID_HEADER: "parent-53",
            AFFINITY_AUTH_HEADER: "sha256=" + "0" * 64,
        },
        timeout=60,
    )
    assert 400 <= response.status_code < 500, response.text
    assert "requires internal_request_auth_key" in response.text
    assert len(chain.ctx.engine.received) == received
    after = chain.ctx.serve_counters()
    assert after.get(SERVE_COUNTER_GENERATE_SENT, 0) == before.get(SERVE_COUNTER_GENERATE_SENT, 0)
    assert (
        after.get(SERVE_COUNTER_HTTP_REJECTED, 0) == before.get(SERVE_COUNTER_HTTP_REJECTED, 0) + 1
    )
