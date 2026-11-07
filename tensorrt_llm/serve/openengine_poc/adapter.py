# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Earlier names of the OpenEngine ``LLM`` facade; see :mod:`.facade`."""

from .facade import LOG_PROMPT_IDS_ENV, OpenEngineLLM, OpenEngineRequestOutput

OpenEngineLLMAdapter = OpenEngineLLM

__all__ = ["LOG_PROMPT_IDS_ENV", "OpenEngineLLMAdapter", "OpenEngineRequestOutput"]
