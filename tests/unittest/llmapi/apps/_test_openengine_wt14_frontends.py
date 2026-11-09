# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Several trtllm-serve frontends in front of one OpenEngine engine, end to end (SV-MF).

Runs only when ``TRTLLM_TEST_OPENENGINE_CHAIN`` selects the chain harness
(``tests/integration/defs/openengine_chain``).
"""

import os
import time
from pathlib import Path

import psutil
import pytest

from tensorrt_llm.serve.openengine_poc.contracts import read_generate_sent_by_pid

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

CHAIN = os.environ.get("TRTLLM_TEST_OPENENGINE_CHAIN", "none") not in ("", "none")
pytestmark = pytest.mark.skipif(
    not CHAIN, reason="checks the OpenEngine frontends; runs on the chain only"
)

NUM_FRONTENDS = 4
NUM_REQUESTS = 32


@pytest.fixture(scope="module")
def model_name() -> str:
    return "Qwen3/Qwen3-0.6B"


def _listeners(port: int) -> int:
    return sum(
        1
        for c in psutil.net_connections("tcp")
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port
    )


@pytest.fixture(scope="module")
def server(model_name: str):
    args = [
        "--backend",
        "pytorch",
        "--kv_cache_free_gpu_memory_fraction",
        "0.2",
        "--num_serve_frontends",
        str(NUM_FRONTENDS),
    ]
    with RemoteOpenAIServer(get_model_path(model_name), args) as remote_server:
        # /health answers once the launcher listens; the others follow it.
        deadline = time.monotonic() + 120
        while _listeners(remote_server.port) < NUM_FRONTENDS:
            assert time.monotonic() < deadline, _listeners(remote_server.port)
            time.sleep(0.5)
        yield remote_server


def test_the_frontends_share_the_port_and_the_engine(server: RemoteOpenAIServer, model_name: str):
    log = Path(server._chain_handle.frontend_log).read_text(errors="replace")
    assert f"All {NUM_FRONTENDS} serving frontends are connected" in log
    before = read_generate_sent_by_pid(server._chain_handle.counter_dir)
    for i in range(NUM_REQUESTS):
        # A connection of its own each time, so the kernel picks a frontend for each request.
        with server.get_client() as client:
            output = client.completions.create(
                model=model_name,
                prompt=f"Request {i}: the capital of France is",
                max_tokens=4,
                temperature=0.0,
            )
        assert output.choices[0].text
    after = read_generate_sent_by_pid(server._chain_handle.counter_dir)
    sent = {pid: count - before.get(pid, 0) for pid, count in after.items()}
    assert sum(sent.values()) == NUM_REQUESTS, sent
    assert sum(1 for count in sent.values() if count > 0) >= 2, sent
