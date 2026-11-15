# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CPU disaggregated chain as the WT-57 tests start it.

:func:`start_chain` starts a :class:`cpu_disagg_chain.CpuDisaggChain` whose
worker frontends keep ``cache_transceiver_config`` in their YAML (without it
a worker frontend refuses the handoff fields, SV-DSG-1), optionally with an
orchestrator request timeout, which ``trtllm-serve disaggregated`` takes only
on its command line (``-r``).
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
import cpu_disagg_chain  # noqa: E402
from cpu_disagg_chain import CpuDisaggChain  # noqa: E402
from fake_openengine import Fail, Hang  # noqa: E402, F401

PROMPT = "He llo"
FAILURE = "scripted wt57 failure"


class _TimedChain(CpuDisaggChain):
    """A chain whose orchestrator is started with ``-r <request_timeout>``."""

    def __init__(self, *args: Any, request_timeout: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.request_timeout = request_timeout

    def _start_orchestrator(self) -> None:
        # The base class builds the orchestrator's command and starts it as a
        # module-level _Process; add the option to that one command only.
        original = cpu_disagg_chain._Process
        timeout = str(self.request_timeout)

        class _WithTimeout(original):
            def __init__(self, command: Sequence[str], env: Mapping[str, str], log: Path) -> None:
                super().__init__([*command, "-r", timeout], env, log)

        cpu_disagg_chain._Process = _WithTimeout
        try:
            super()._start_orchestrator()
        finally:
            cpu_disagg_chain._Process = original


@contextlib.contextmanager
def start_chain(
    directory: Path,
    *,
    frontend_args: Sequence[str] = (),
    orchestrator_config: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    request_timeout: int | None = None,
) -> Iterator[CpuDisaggChain]:
    """Start the trtllm-shaped chain in ``directory``; ``frontend_args`` go to both frontends."""
    config = directory / "frontend.yaml"
    config.write_text(yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}}))
    args = ["--config", str(config), *frontend_args]
    options: dict[str, Any] = {
        "frontend_args": {"context": args, "generation": args},
        "orchestrator_config": orchestrator_config,
        "env": env,
    }
    if request_timeout is None:
        chain = CpuDisaggChain(directory, "trtllm", **options)
    else:
        chain = _TimedChain(directory, "trtllm", request_timeout=request_timeout, **options)
    with chain:
        yield chain


def calls(chain: CpuDisaggChain) -> dict[str, int]:
    """The Generate calls each engine has received so far, by role."""
    return {worker.role: len(worker.engine.requests) for worker in chain.workers}
