# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse tests/capload/capload_support.py (one module object even when both directories are
collected in the same session)."""

import os
import sys

_SUP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "capload"))
if _SUP not in sys.path:
    sys.path.insert(0, _SUP)

from capload_support import launch  # noqa: E402,F401  (pytest fixture; also puts tools/ on sys.path)
