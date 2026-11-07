# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash module's tests that send no Generate call on the glm chain.

On case-01 they only check that serve rejects what the glm engine does not declare.
"""

from openengine_chain.conventions import CHAIN_GLM
from openengine_chain.expectations import NO_ENGINE_TRAFFIC, Expectation

_FILE = "_test_openengine_glm53.py"
_REJECTED = "on the glm engine the front end rejects every request of the test before generating"

EXPECTATIONS = [
    Expectation("WT-06", f"{_FILE}::{name}", NO_ENGINE_TRAFFIC, _REJECTED, chains=(CHAIN_GLM,))
    for name in (
        "test_penalties[*",
        "test_logprobs[*21*",
        "test_guided_json_schema[*",
        "test_thinking_token_budget[*",
        "test_video_url[*",
    )
]
