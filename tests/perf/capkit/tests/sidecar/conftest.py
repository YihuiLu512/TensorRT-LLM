# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Put <C>/tools (capcommon, capenv) on sys.path. Helpers live in b4_<dir>_util.py."""

import os
import sys

_C = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_tools = os.path.join(_C, "tools")
if _tools not in sys.path:
    sys.path.insert(0, _tools)
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
