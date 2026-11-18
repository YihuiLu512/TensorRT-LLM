# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tests/cape12: put tools/ on sys.path (the wrapper's logic needs no product import)."""

import os
import sys

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
