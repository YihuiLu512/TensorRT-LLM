# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The postprocessing processes of the OpenEngine frontend, end to end.

Runs only when ``TRTLLM_TEST_OPENENGINE_CHAIN`` selects the chain harness
(``tests/integration/defs/openengine_chain``): with ``--num_postprocess_workers``
the frontend's own pool processes run the ``post_processor_hook`` and the
endpoints' postprocessors.
"""

import asyncio
import os
from collections import defaultdict

import openai
import pytest

from ..test_llm import get_model_path
from ._test_openengine_wt13_hooks import LOG_ENV
from .openai_server import RemoteOpenAIServer

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
pytestmark = [
    pytest.mark.threadleak(enabled=False),
    pytest.mark.skipif(not CHAIN, reason="checks the OpenEngine frontend; runs on the chain only"),
]

WORKERS = 2
HOOK = "_test_openengine_wt13_hooks.ProcessRecordingHook"
# The pool names its processes "trtllm-postproc-<index>".
POOL_PROCESS_PREFIX = "trtllm-postproc-"
# Requests are routed by a hash of their random ids; with this many, the
# chance that they all go to one of two processes is 2**-15.
NUM_REQUESTS = 16


@pytest.fixture(scope="module")
def model_name() -> str:
    return "Qwen3/Qwen3-0.6B"


@pytest.fixture(scope="module")
def hook_log(tmp_path_factory):
    return tmp_path_factory.mktemp("postproc_hook") / "hook.log"


@pytest.fixture(scope="module")
def server(model_name: str, hook_log):
    args = [
        "--backend",
        "pytorch",
        "--kv_cache_free_gpu_memory_fraction",
        "0.2",
        "--num_postprocess_workers",
        str(WORKERS),
        "--post_processor_hook",
        HOOK,
    ]
    # The hook module must be importable in the frontend and its pool processes.
    apps_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = apps_dir + os.pathsep + env.get("PYTHONPATH", "")
    env[LOG_ENV] = str(hook_log)
    with RemoteOpenAIServer(get_model_path(model_name), args, env=env) as remote_server:
        yield remote_server


async def _send_requests(client: openai.AsyncOpenAI, model_name: str) -> list[str]:
    async def completion(i: int) -> str:
        response = await client.completions.create(
            model=model_name, prompt=f"Count from 1 to {i + 3}:", max_tokens=8, temperature=0.0
        )
        return response.choices[0].text

    async def chat_stream(i: int) -> str:
        stream = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": f"Name {i + 2} colors."}],
            max_tokens=8,
            temperature=0.0,
            stream=True,
        )
        return "".join([chunk.choices[0].delta.content or "" async for chunk in stream])

    half = NUM_REQUESTS // 2
    return await asyncio.gather(
        *(completion(i) for i in range(half)), *(chat_stream(i) for i in range(half))
    )


def test_the_pool_processes_do_the_postprocessing(server, hook_log, model_name: str):
    texts = asyncio.run(_send_requests(server.get_async_client(), model_name))
    assert any(texts), texts
    records = [
        line.split()
        for log in hook_log.parent.glob(f"{hook_log.name}.*")
        for line in log.read_text().splitlines()
    ]
    assert records, "the hook never ran"
    names = {name for name, _, _ in records}
    # Only the pool's processes ran the hook, never the frontend process.
    assert all(name.startswith(POOL_PROCESS_PREFIX) for name in names), names
    by_request = defaultdict(set)
    for _, pid, request_id in records:
        by_request[request_id].add(pid)
    assert len(by_request) == NUM_REQUESTS, sorted(by_request)
    # Every request stayed in one process.
    assert all(len(pids) == 1 for pids in by_request.values()), by_request
    used = set().union(*by_request.values())
    from openengine_chain import proof

    frontends = proof.frontend_count(server._chain_handle)
    if frontends == 1:
        # The requests used every process of the one pool.
        assert len(used) == WORKERS, by_request
    else:
        # Each front end runs its share of the workers, at least one (SV-PP-1).
        assert len(used) <= max(WORKERS, frontends), by_request
