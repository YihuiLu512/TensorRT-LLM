# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chain harness reads the Generate counts the front ends publish, as the contract reads them."""

from __future__ import annotations

import multiprocessing
import sys
import types
from pathlib import Path

import pytest

from tensorrt_llm.serve.openengine_poc import contracts

pytestmark = pytest.mark.cpu_only

_DEFS = Path(__file__).resolve().parents[4] / "integration" / "defs"


@pytest.fixture(scope="module")
def proof(tmp_path_factory):
    """The harness's proof module, imported under its top-level package as runner.py does."""
    directory = tmp_path_factory.mktemp("pythonpath")
    (directory / "openengine_chain").symlink_to(_DEFS / "openengine_chain")
    sys.path.insert(0, str(directory))
    try:
        from openengine_chain import proof

        yield proof
    finally:
        sys.path.remove(str(directory))


def _frontend(counter_dir: str, sent: int, ready, finish) -> None:
    """A front end's counter in a process of its own: started, counted, and closed on ``finish``."""
    counter = contracts.GenerateCounter(counter_dir)
    counter.start()
    counter.increment(sent)
    ready.set()
    finish.wait(60)
    counter.close()


def test_the_harness_reads_the_counts_the_front_ends_publish(proof, tmp_path):
    handle = types.SimpleNamespace(counter_dir=str(tmp_path), alive=True)

    def both() -> tuple[int | None, int]:
        return proof.read_frontend_counter(handle), contracts.read_generate_sent(tmp_path)

    # Forked, so that the child need not import TensorRT-LLM again.
    context = multiprocessing.get_context("fork")
    ready, finish = context.Event(), context.Event()
    other = context.Process(target=_frontend, args=(str(tmp_path), 5, ready, finish))
    other.start()
    own = contracts.GenerateCounter(tmp_path)
    try:
        assert ready.wait(60)
        own.start()
        own.increment(2)
        assert both() == (7, 7)
        # Each read gives the counts of that moment, with no file written in between.
        own.increment()
        assert both() == (8, 8)
        finish.set()
        other.join(60)
        assert other.exitcode == 0
        # The other front end has closed: its file holds its final count.
        assert both() == (8, 8)
        own.close()
        assert both() == (8, 8)
    finally:
        finish.set()
        other.join(60)
        own.close()
