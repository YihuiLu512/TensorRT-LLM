# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Postprocessing and the post_processor_hook on each leg of a disaggregated request (F-38, F-40).

Each test starts its own ``_test_openengine_disagg_common.disagg_deployment``
(Qwen3-0.6B, one context and one generation worker on one GPU), in-process or
on the OpenEngine chain, and sends every request through the orchestrator.
The hook is ``_test_openengine_wt13_hooks.ProcessRecordingHook``: it appends
"<process name> <pid> <request id>" to a file of its process, so the records
tell which process ran it for which request. A worker's serving process is the
``trtllm-serve`` the test started; on the chain, its worker frontend.

- NT-068 (F-38): with ``--num_postprocess_workers 2`` on both workers and the
  hook on the generation worker, the generation leg is postprocessed by that
  worker's pool processes, never by its serving process; on the chain they are
  the frontend's own ``trtllm-postproc-*`` children. The context worker's pool
  still returns the prompt token ids of a non-streamed context response, which
  the orchestrator sends to the generation worker as the completion prompt.
- NT-070 (F-40, the "same" case of LIM-45): without a pool and with the hook on
  both workers, each leg runs the hook in its own worker's serving process,
  under that worker's request id, so neither leg's per-request state is seen by
  the other. Run in both forms, the records are alike.
"""

import asyncio
import os
from collections import defaultdict
from pathlib import Path

import openai
import psutil
import pytest

from ._test_openengine_disagg_common import CHAIN, disagg_deployment
from ._test_openengine_wt13_hooks import LOG_ENV

pytestmark = pytest.mark.threadleak(enabled=False)

HOOK = "_test_openengine_wt13_hooks.ProcessRecordingHook"
POOL_WORKERS = 2
# The OpenEngine frontend names its pool processes "trtllm-postproc-<index>".
POOL_PROCESS_PREFIX = "trtllm-postproc-"
# Requests are routed to a pool process by a hash of their random ids; with
# this many, the chance that they all go to one of two processes is 2**-15.
NUM_REQUESTS = 16


def _hook_env(log: Path) -> dict[str, str]:
    """Make the hook importable in every worker process and its pool, and name its log."""
    apps_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.environ.get("PYTHONPATH", "")
    return {"PYTHONPATH": apps_dir + (os.pathsep + path if path else ""), LOG_ENV: str(log)}


def _records(log: Path) -> list[tuple[str, int, str]]:
    """Every hook call: (process name, pid, request id)."""
    records = []
    for path in log.parent.glob(f"{log.name}.*"):
        for line in path.read_text().splitlines():
            name, pid, request_id = line.split()
            records.append((name, int(pid), request_id))
    return records


def _serving_pid(worker) -> int:
    """The pid of a worker's serving process: its ``trtllm-serve``, or on the chain its frontend."""
    handle = worker._chain_handle
    return (handle.frontend if handle is not None else worker.proc).pid


async def _send(client: openai.AsyncOpenAI, model: str, count: int) -> list[str]:
    """``count`` requests at once: half non-streamed completions, half streamed chats."""

    async def completion(i: int) -> str:
        response = await client.completions.create(
            model=model, prompt=f"Count from 1 to {i + 3}:", max_tokens=8, temperature=0.0
        )
        return response.choices[0].text

    async def chat_stream(i: int) -> str:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Name {i + 2} colors."}],
            max_tokens=8,
            temperature=0.0,
            stream=True,
        )
        return "".join([chunk.choices[0].delta.content or "" async for chunk in stream])

    half = count // 2
    return await asyncio.gather(
        *(completion(i) for i in range(half)), *(chat_stream(i) for i in range(count - half))
    )


def _async_client(deployment) -> openai.AsyncOpenAI:
    sync = deployment.client()
    return openai.AsyncOpenAI(base_url=sync.base_url, api_key=sync.api_key, timeout=sync.timeout)


@pytest.fixture
def pool_deployment(tmp_path):
    log = tmp_path / "hook.log"
    with disagg_deployment(
        tmp_path,
        worker_args=["--num_postprocess_workers", str(POOL_WORKERS)],
        role_args={"generation": ["--post_processor_hook", HOOK]},
        env=_hook_env(log),
    ) as deployment:
        yield deployment, log


def test_postproc_pool_on_gen_worker_frontend(pool_deployment):
    deployment, log = pool_deployment
    texts = asyncio.run(_send(_async_client(deployment), deployment.model, NUM_REQUESTS))
    # A completion's generation leg gets as its prompt the token ids the context
    # worker's pool returned; without them the request fails.
    completions = texts[: NUM_REQUESTS // 2]
    assert all(completions), texts
    records = _records(log)
    assert records, "the generation worker's hook never ran"
    by_request = defaultdict(set)
    for _, pid, request_id in records:
        by_request[request_id].add(pid)
    # The hook is on the generation worker only: one request id per generation leg.
    assert len(by_request) == NUM_REQUESTS, sorted(by_request)
    # Every request stayed in one process.
    assert all(len(pids) == 1 for pids in by_request.values()), by_request
    used = set().union(*by_request.values())
    assert 1 <= len(used) <= POOL_WORKERS, by_request
    serving = {_serving_pid(deployment.ctx), _serving_pid(deployment.gen)}
    # The pool's processes ran the hook, never a worker's serving process.
    assert not used & serving, (used, serving)
    names = {name for name, _, _ in records}
    assert "MainProcess" not in names, names
    if CHAIN:
        assert all(name.startswith(POOL_PROCESS_PREFIX) for name in names), names
        gen_frontend = _serving_pid(deployment.gen)
        parents = {pid: psutil.Process(pid).ppid() for pid in used}
        assert set(parents.values()) == {gen_frontend}, (parents, gen_frontend)
        # One frontend per worker: the requests used every process of its pool.
        assert len(used) == POOL_WORKERS, by_request


@pytest.fixture
def hook_deployment(tmp_path):
    log = tmp_path / "hook.log"
    with disagg_deployment(
        tmp_path,
        worker_args=["--post_processor_hook", HOOK],
        env=_hook_env(log),
    ) as deployment:
        yield deployment, log


def test_post_processor_hook_state_per_leg(hook_deployment):
    deployment, log = hook_deployment
    count = 4
    texts = asyncio.run(_send(_async_client(deployment), deployment.model, count))
    assert all(texts[: count // 2]), texts
    by_pid = defaultdict(set)
    for _, pid, request_id in _records(log):
        by_pid[pid].add(request_id)
    ctx_pid, gen_pid = _serving_pid(deployment.ctx), _serving_pid(deployment.gen)
    # Without a pool, each worker's serving process runs its own hook, and no other process does.
    assert set(by_pid) == {ctx_pid, gen_pid}, (dict(by_pid), ctx_pid, gen_pid)
    # Each worker's hook saw its leg of every request, under that worker's own request id.
    assert len(by_pid[ctx_pid]) == count, dict(by_pid)
    assert len(by_pid[gen_pid]) == count, dict(by_pid)
