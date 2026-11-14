# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker frontends register with a real orchestrator (SV-DSG-1), on CPU against the gRPC fake engine.

The worker frontends and the orchestrator are the real ``trtllm-serve``
command lines (see ``wt50_chain``).
"""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wt50_chain as chain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False), pytest.mark.timeout(900)]


def _worker_config(cluster: dict, auth_key: str | None = None) -> dict:
    config = {"disagg_cluster": cluster, "cache_transceiver_config": {"backend": "DEFAULT"}}
    if auth_key is not None:
        config["internal_request_auth_key"] = auth_key
    return config


def _started_orchestrator(tmp_path: Path, **kwargs) -> tuple[chain.Process, str, dict]:
    port = chain.free_port()
    url = f"http://127.0.0.1:{port}"
    cluster = kwargs.pop("cluster", None) or chain.cluster_config(url)
    orchestrator = chain.start_orchestrator(tmp_path, port, cluster=cluster, **kwargs)
    with contextlib.ExitStack() as on_failure:
        on_failure.callback(orchestrator.stop)
        chain.wait(
            lambda: chain.cluster_info(url), "the orchestrator to answer", processes=[orchestrator]
        )
        on_failure.pop_all()
    return orchestrator, url, cluster


def _role_keys(orchestrator: str, role: str) -> list[str]:
    return [key for key in chain.storage_keys(orchestrator) if f"/{role}-" in key]


def test_worker_frontend_registers_over_http_storage(tmp_path):
    """NT-002 (DG-11, DG-28, DG-31): registration, heartbeat renewal, readiness by role."""
    model = chain.model_dir(tmp_path / "model")
    auth_key = secrets.token_hex(16)
    with (
        chain.running_in_thread(chain.engine(model)) as ctx_engine,
        chain.running_in_thread(chain.engine(model)) as gen_engine,
    ):
        orchestrator, url, cluster = _started_orchestrator(tmp_path, auth_key=auth_key)
        config = _worker_config(cluster, auth_key)
        ctx = chain.start_worker(tmp_path, "ctx", model, ctx_engine, role="context", config=config)
        with chain.stopping(orchestrator, ctx):
            ctx_workers = chain.wait(
                lambda: chain.registered(url)[0],
                "the ctx worker frontend to register",
                processes=[orchestrator, ctx],
            )
            # One role only: registered and listed, but the orchestrator is not ready.
            chain.wait(
                lambda: chain.listed(url)[0],
                "the ctx worker frontend to be listed",
                processes=[orchestrator, ctx],
            )
            assert len(ctx_workers) == 1 and chain.registered(url)[1] == []
            assert "registration successful" in ctx.log()
            assert chain.status(f"{url}/health") == 503
            assert chain.cluster_info(url)["is_ready"] is False
            ctx_port = ctx_workers[0]["port"]
            assert ctx_port > 0 and chain.status(f"http://127.0.0.1:{ctx_port}/health") == 200

            # The registration outlives the inactive timeout without a gap: the heartbeat renews it.
            keys = _role_keys(url, "CONTEXT")
            assert len(keys) == 1 and f":{ctx_port}-" in keys[0]
            deadline = time.monotonic() + chain.INACTIVE_SECONDS * 2
            while time.monotonic() < deadline:
                assert _role_keys(url, "CONTEXT") == keys
                assert len(chain.registered(url)[0]) == 1 and len(chain.listed(url)[0]) == 1
                time.sleep(0.2)

            gen = chain.start_worker(
                tmp_path, "gen", model, gen_engine, role="generation", config=config
            )
            with chain.stopping(gen):
                chain.wait(
                    lambda: chain.status(f"{url}/health") == 200,
                    "the orchestrator to be ready with both roles",
                    processes=[orchestrator, ctx, gen],
                )
                ctx_workers, gen_workers = chain.registered(url)
                assert len(ctx_workers) == 1 and len(gen_workers) == 1
                assert chain.cluster_info(url)["is_ready"] is True
                assert len(_role_keys(url, "GENERATION")) == 1
                # LIM-23: only the worker frontends register, never an engine.
                for key in chain.storage_keys(url):
                    for fake in (ctx_engine, gen_engine):
                        assert f":{fake.port}-" not in key, key
        # Registration sends no Generate.
        assert chain.generate_calls(ctx_engine) == chain.generate_calls(gen_engine) == 0


def test_static_urls_reach_worker_frontends(tmp_path):
    """NT-005 (DG-27): the orchestrator's static ``urls`` reach two worker frontends without roles."""
    pytest.skip(
        "needs a two-leg request, which needs PrefillReady from the fake engine and the "
        "facade's handoff mapping"
    )


@pytest.mark.skipif(shutil.which("etcd") is None, reason="etcd is not installed")
def test_worker_frontend_registers_over_etcd(tmp_path):
    """NT-006 (DG-29): as NT-002's registration, with ``cluster_uri`` set to etcd."""
    import subprocess  # nosec B404

    etcd_port = chain.free_port()
    peer_port = chain.free_port()
    etcd_log = (tmp_path / "etcd.log").open("w")
    etcd = subprocess.Popen(  # nosec B603
        [
            shutil.which("etcd"),
            "--data-dir",
            str(tmp_path / "etcd"),
            "--listen-client-urls",
            f"http://127.0.0.1:{etcd_port}",
            "--advertise-client-urls",
            f"http://127.0.0.1:{etcd_port}",
            "--listen-peer-urls",
            f"http://127.0.0.1:{peer_port}",
        ],
        stdout=etcd_log,
        stderr=subprocess.STDOUT,
    )
    try:
        chain.wait(
            lambda: etcd.poll() is not None
            or chain.status(f"http://127.0.0.1:{etcd_port}/health") == 200,
            "etcd to answer",
        )
        assert etcd.poll() is None, (tmp_path / "etcd.log").read_text()
        model = chain.model_dir(tmp_path / "model")
        with (
            chain.running_in_thread(chain.engine(model)) as ctx_engine,
            chain.running_in_thread(chain.engine(model)) as gen_engine,
        ):
            cluster = chain.cluster_config(f"etcd://127.0.0.1:{etcd_port}")
            orchestrator, url, _ = _started_orchestrator(tmp_path, cluster=cluster)
            config = _worker_config(cluster)
            ctx = chain.start_worker(
                tmp_path, "ctx", model, ctx_engine, role="context", config=config
            )
            gen = chain.start_worker(
                tmp_path, "gen", model, gen_engine, role="generation", config=config
            )
            with chain.stopping(orchestrator, ctx, gen):
                chain.wait(
                    lambda: chain.status(f"{url}/health") == 200,
                    "the orchestrator to be ready with both roles",
                    processes=[orchestrator, ctx, gen],
                )
                ctx_workers, gen_workers = chain.registered(url)
                assert len(ctx_workers) == 1 and len(gen_workers) == 1
    finally:
        etcd.kill()
        etcd.wait(timeout=30)
        etcd_log.close()


def test_new_worker_is_listed_after_clock_sync_and_server_info(tmp_path):
    """NT-007 (DG-32, DG-54): clock sync, then ``/server_info``, then listing; a failing one is not listed."""
    model = chain.model_dir(tmp_path / "model")
    with (
        chain.running_in_thread(chain.engine(model)) as ctx_engine,
        chain.running_in_thread(chain.engine(model)) as gen_engine,
    ):
        orchestrator, url, cluster = _started_orchestrator(tmp_path)
        config = _worker_config(cluster)
        ctx = chain.start_worker(tmp_path, "ctx", model, ctx_engine, role="context", config=config)
        gen = chain.start_worker(
            tmp_path,
            "gen",
            model,
            gen_engine,
            role="generation",
            config=config,
            # The middleware added last runs first: record the request before it fails.
            middleware=("wt50_chain.fail_server_info", "wt50_chain.record_requests"),
        )
        with chain.stopping(orchestrator, ctx, gen):
            chain.wait(
                lambda: chain.listed(url)[0],
                "the ctx worker frontend to be listed",
                processes=[orchestrator, ctx, gen],
            )
            seen = chain.requests_seen(tmp_path, "ctx")
            assert "POST /steady_clock_offset" in seen and "GET /server_info" in seen, seen
            assert seen.index("POST /steady_clock_offset") < seen.index("GET /server_info"), seen

            # The gen worker frontend registers, but its /server_info fails: never listed.
            chain.wait(
                lambda: chain.registered(url)[1]
                and "GET /server_info" in chain.requests_seen(tmp_path, "gen"),
                "the orchestrator to ask the gen worker frontend for /server_info",
                processes=[orchestrator, ctx, gen],
            )
            time.sleep(chain.HEARTBEAT_SECONDS * 3)
            assert chain.listed(url)[1] == []
            assert chain.status(f"{url}/health") == 503


def test_only_the_launcher_frontend_registers(tmp_path):
    """CS-03: with ``num_serve_frontends=2``, the orchestrator sees one registration."""
    model = chain.model_dir(tmp_path / "model")
    # The launcher's Unix socket path must stay short.
    tmpdir = tempfile.mkdtemp(prefix="wt50-", dir="/tmp" if os.path.isdir("/tmp") else None)
    try:
        with chain.running_in_thread(chain.engine(model)) as ctx_engine:
            orchestrator, url, cluster = _started_orchestrator(tmp_path)
            ctx = chain.start_worker(
                tmp_path,
                "ctx",
                model,
                ctx_engine,
                role="context",
                config=_worker_config(cluster),
                port=chain.free_port(),
                frontends=2,
                env={"TMPDIR": tmpdir},
            )
            with chain.stopping(orchestrator, ctx):
                chain.wait(
                    lambda: chain.registered(url)[0],
                    "the ctx worker frontend to register",
                    processes=[orchestrator, ctx],
                )
                launcher = psutil.Process(ctx.proc.pid)
                chain.wait(
                    lambda: [
                        child
                        for child in launcher.children()
                        if "--openengine-endpoint" in child.cmdline()
                    ],
                    "the attached frontend to start",
                    processes=[ctx],
                )
                time.sleep(chain.HEARTBEAT_SECONDS * 3)
                assert len(chain.registered(url)[0]) == 1
                assert len(_role_keys(url, "CONTEXT")) == 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_a_worker_frontend_that_fails_its_startup_checks_never_registers(tmp_path):
    """SV-DSG-1: registration follows the startup checks; a worker on a version-3 engine fails them."""
    from tensorrt_llm.grpc.openengine_conventions import CONVENTIONS_VERSION_KEY

    model = chain.model_dir(tmp_path / "model")
    fake = chain.engine(model, server_info_extra={CONVENTIONS_VERSION_KEY: 3})
    with chain.running_in_thread(fake) as ctx_engine:
        orchestrator, url, cluster = _started_orchestrator(tmp_path)
        ctx = chain.start_worker(
            tmp_path, "ctx", model, ctx_engine, role="context", config=_worker_config(cluster)
        )
        with chain.stopping(orchestrator, ctx):
            ctx.proc.wait(timeout=chain.STARTUP_SECONDS)
            assert ctx.proc.returncode != 0
            assert "conventions version" in ctx.log()
            # Never registered, rather than registered and removed on the way out.
            assert "registration successful" not in ctx.log()
            assert chain.registered(url)[0] == []
            assert _role_keys(url, "CONTEXT") == []
