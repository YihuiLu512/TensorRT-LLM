# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Make <C>/tools and the support module importable; expose the launch fixture."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(_HERE, "..", "..", "tools")), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from capload_support import launch  # noqa: E402,F401  (pytest fixture)
