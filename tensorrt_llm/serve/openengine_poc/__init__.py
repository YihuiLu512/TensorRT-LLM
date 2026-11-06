# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prototype: serve the OpenAI API from an external engine over OpenEngine v1.

``trtllm-serve <model dir> --openengine-endpoint host:port`` builds
``OpenAIServer`` on :class:`OpenEngineLLMAdapter` instead of ``LLM``.
Only ``/v1/chat/completions``, ``/v1/completions`` and ``/health`` are adapted.

The exports load on first use: ``OpenAIServer`` imports :mod:`.contracts`,
which must not pull in the gRPC client.
"""

import importlib
from typing import Any

_EXPORTS = {
    "OpenEngineLLMAdapter": ".adapter",
    "OpenEngineRequestOutput": ".adapter",
    "build_generate_request": ".mapping",
}

__all__ = ["OpenEngineLLMAdapter", "OpenEngineRequestOutput", "build_generate_request"]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
