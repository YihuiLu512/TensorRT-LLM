# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure response-analysis helpers make no engine requests."""

from openengine_chain.conventions import CHAIN_TRTLLM
from openengine_chain.expectations import NO_ENGINE_TRAFFIC, Expectation

_HELPER_TESTS = (
    "test_parse_integers_all_same_line",
    "test_evaluate_number_set_all_same_valid",
    "test_evaluate_number_set_all_same_wrong_common",
    "test_evaluate_number_set_all_same_wrong_not_in_pool",
    "test_evaluate_number_set_unique_intersects_pool",
    "test_evaluate_number_set_unique_hallucination_ok",
    "test_evaluate_number_set_freeform_pool_digits",
    "test_evaluate_number_set_weak_invalid_empty",
    "test_evaluate_number_set_all_same_markdown_fence",
    "test_evaluate_number_set_freeform_correct_number_only",
)

EXPECTATIONS = [
    Expectation(
        "WT-20-R2",
        f"_test_openai_kv_cache_contamination.py::{name}",
        NO_ENGINE_TRAFFIC,
        "Pure response-analysis helper; no Generate call is made.",
        chains=(CHAIN_TRTLLM,),
    )
    for name in _HELPER_TESTS
]
