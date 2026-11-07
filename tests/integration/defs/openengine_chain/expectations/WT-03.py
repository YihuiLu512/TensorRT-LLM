# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The harness's own table: tests of lists/case02_full.txt known to send no Generate call by design.

They call endpoints that do not generate, or send only requests the front end rejects.
"""

from openengine_chain.expectations import NO_ENGINE_TRAFFIC, Expectation

_NO_GENERATE = "endpoint that does not generate"
_REJECTED = "the front end rejects the request before generating"

_TESTS = {
    "_test_openai_misc.py": (
        ("test_version", _NO_GENERATE),
        ("test_health[", _NO_GENERATE),
        ("test_model", _NO_GENERATE),
        ("test_tokenize", _NO_GENERATE),
    ),
    "_test_openai_prometheus.py": (
        ("test_kv_cache_metrics_available_before_first_request", _NO_GENERATE),
    ),
    "_test_openai_responses_entrypoint.py": (
        ("test_get_invalid_response_id", _NO_GENERATE),
        ("test_get_non_existent_response_id", _NO_GENERATE),
        ("test_delete_invalid_response_id", _NO_GENERATE),
        ("test_delete_non_existent_response_id", _NO_GENERATE),
    ),
    "_test_openai_completions.py": (("test_completion_with_invalid_logit_bias", _REJECTED),),
    "_test_openai_chat.py": (("test_chat_completion_with_invalid_logit_bias", _REJECTED),),
    "_test_openai_tool_call.py": (
        ("test_named_tool_choice_missing_function_returns_400", _REJECTED),
        ("test_named_tool_choice_with_response_format_is_rejected", _REJECTED),
        ("test_named_tool_choice_with_prompt_token_ids_b64_is_rejected", _REJECTED),
    ),
    "_test_anthropic_messages.py": (
        ("test_count_tokens_grows_with_the_prompt", _NO_GENERATE),
        ("test_invalid_request_returns_an_anthropic_error", _REJECTED),
        ("test_tool_choice_any_is_rejected_with_a_clear_reason", _REJECTED),
    ),
}

EXPECTATIONS = [
    Expectation("WT-03", f"{file}::{name}*", NO_ENGINE_TRAFFIC, reason)
    for file, tests in _TESTS.items()
    for name, reason in tests
]
