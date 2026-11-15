# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A worker frontend's error reaches the caller through the orchestrator (DG-41), on CPU.

NT-010: the CPU chain (``cpu_disagg_chain``) runs the real orchestrator in
front of two real worker frontends on fake engines. When an engine refuses
or fails a leg, the worker frontend answers with its own status and message,
and the orchestrator passes both on without the worker's address. After a
failed context leg no generation leg is sent. The orchestrator here has no
retries, so each leg is tried once (retries: ``test_poc3_wt57_orchestrator_retry``).
"""

from __future__ import annotations

import grpc
import pytest

__extra_import_path__ = [".", ".."]
from wt57_chain import FAILURE, PROMPT, Fail, calls, scripted, start_chain  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wt57-errors")
    with start_chain(directory, orchestrator_config={"max_retries": 0}) as chain:
        yield chain


@pytest.mark.parametrize(
    "path", ["/v1/completions", "/v1/chat/completions"], ids=["completions", "chat"]
)
@pytest.mark.parametrize(
    "code, status",
    [
        (grpc.StatusCode.INVALID_ARGUMENT, 400),
        (grpc.StatusCode.FAILED_PRECONDITION, 500),
        (grpc.StatusCode.INTERNAL, 500),
    ],
    ids=["refused", "precondition", "failed"],
)
@pytest.mark.parametrize("role", ["context", "generation"])
def test_worker_frontend_error_passes_through_the_orchestrator(chain, role, code, status, path):
    worker = chain.ctx if role == "context" else chain.gen
    if path == "/v1/completions":
        body = {"prompt": PROMPT, "max_tokens": 4}
    else:
        body = {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": 4}
    before = calls(chain)
    with scripted(worker.engine, Fail(code, FAILURE)):
        response = chain.post(path, body)
    after = calls(chain)

    assert response.status_code == status, response.text
    assert FAILURE in response.text, response.text
    # The worker's message names the leg; the worker's address stays inside.
    assert f"the {role} leg" in response.text, response.text
    for host_port in (worker.host_port, worker.engine.target):
        assert host_port not in response.text, response.text
    # Each leg is tried once, and a failed context leg has no generation leg.
    assert after["context"] - before["context"] == 1
    assert after["generation"] - before["generation"] == (1 if role == "generation" else 0)
