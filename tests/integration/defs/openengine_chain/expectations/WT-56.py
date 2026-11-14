# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4.1 module's tests that send no Generate call on the ds chain."""

from openengine_chain.conventions import CHAIN_DS
from openengine_chain.expectations import NO_ENGINE_TRAFFIC, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-56",
        "_test_openengine_ds_disagg_basic.py::test_startup_tokenizer_and_fingerprint",
        NO_ENGINE_TRAFFIC,
        "reads the engine's ServerInfo and ModelInfo and the front end's /v1/models only",
        chains=(CHAIN_DS,),
    ),
]
