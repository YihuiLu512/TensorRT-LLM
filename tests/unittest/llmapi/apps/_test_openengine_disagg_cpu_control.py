# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The control endpoints of an orchestrator in front of two worker frontends, on CPU.

The orchestrator (``trtllm-serve disaggregated``) and the two worker frontends
(``trtllm-serve --openengine-endpoint``) are the real command lines; each worker
frontend talks to its own gRPC fake engine (see ``wt50_chain`` in
``tests/unittest/serve/openengine_poc/wt50/``). No request reaches an engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_CHAIN = Path(__file__).resolve().parents[2] / "serve" / "openengine_poc" / "wt50"
if str(_CHAIN) not in sys.path:
    sys.path.insert(0, str(_CHAIN))

import wt50_chain as chain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False), pytest.mark.timeout(900)]


def test_worker_frontend_models_version_health(tmp_path):
    """NT-001 (F-05a, F-05b, F-05c)."""
    model = chain.model_dir(tmp_path / "model")
    with (
        chain.running_in_thread(chain.engine(model)) as ctx_engine,
        chain.running_in_thread(chain.engine(model)) as gen_engine,
    ):
        port = chain.free_port()
        cluster = chain.cluster_config(port)
        url = f"http://127.0.0.1:{port}"
        orchestrator = chain.start_orchestrator(tmp_path, port, cluster=cluster)
        config = {"disagg_cluster": cluster, "cache_transceiver_config": {"backend": "DEFAULT"}}
        ctx = chain.start_worker(tmp_path, "ctx", model, ctx_engine, role="context", config=config)
        with chain.stopping(orchestrator, ctx):
            ctx_workers = chain.wait(
                lambda: chain.registered(url)[0],
                "the ctx worker frontend to register",
                processes=[orchestrator, ctx],
            )
            ctx_url = f"http://127.0.0.1:{ctx_workers[0]['port']}"
            # F-05c: one healthy worker frontend is not enough.
            assert chain.status(f"{ctx_url}/health") == 200
            assert chain.status(f"{url}/health") == 503

            gen = chain.start_worker(
                tmp_path, "gen", model, gen_engine, role="generation", config=config
            )
            with chain.stopping(gen):
                chain.wait(
                    lambda: chain.status(f"{url}/health") == 200,
                    "the orchestrator to be ready",
                    processes=[orchestrator, ctx, gen],
                )
                gen_url = f"http://127.0.0.1:{chain.registered(url)[1][0]['port']}"
                assert chain.status(f"{gen_url}/health") == 200
                for name in ("ctx", "gen"):
                    assert "GET /health" in chain.requests_seen(tmp_path, name)

                # F-05a: the orchestrator answers /v1/models from the ctx worker frontend.
                response = chain.get(f"{url}/v1/models")
                assert response is not None and response.status_code == 200, response
                assert [entry["id"] for entry in response.json()["data"]] == [chain.MODEL]
                assert "GET /v1/models" in chain.requests_seen(tmp_path, "ctx")

                # F-05b: each worker frontend answers /version.
                for worker_url in (ctx_url, gen_url):
                    version = chain.get(f"{worker_url}/version")
                    assert version is not None and version.status_code == 200
                    assert version.json()["version"]
    assert chain.generate_calls(ctx_engine) == chain.generate_calls(gen_engine) == 0
