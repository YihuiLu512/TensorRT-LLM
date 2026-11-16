# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time limits for existing tests whose workers start one after the other on the chain.

``test_completion_metrics`` starts a context and a generation worker with
``wait=False``, then the disaggregated server, all inside its own 300 s limit.
Without the chain both workers load at once. On the chain each worker's engine
is ready before the next worker starts, so the limit has to cover two engine
starts; with cold JIT caches one took 108 to 238 s on a B200, with warm ones
about 18 s.
"""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import TIME_LIMIT, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-60-fix1",
        "test_disagg_serving_perf_metrics.py::test_completion_metrics",
        TIME_LIMIT,
        "its two workers' engines start one after the other on the chain, each in up to "
        "about 240 s with cold JIT caches; 900 s covers both and the test itself",
        seconds=900,
        chains=(CHAIN_TRTLLM,),
    ),
]
