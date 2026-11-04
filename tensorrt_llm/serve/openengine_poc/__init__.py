# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prototype: serve the OpenAI API from an external engine over OpenEngine v1.

``trtllm-serve <model dir> --openengine-endpoint host:port`` builds
``OpenAIServer`` on :class:`OpenEngineLLMAdapter` instead of ``LLM``.
Only ``/v1/chat/completions``, ``/v1/completions`` and ``/health`` are adapted.
"""

from .adapter import OpenEngineLLMAdapter, OpenEngineRequestOutput
from .mapping import build_generate_request

__all__ = ["OpenEngineLLMAdapter", "OpenEngineRequestOutput", "build_generate_request"]
