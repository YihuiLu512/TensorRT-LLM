# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Put <C>/tools on sys.path; expose the judge package whatever its directory name."""

import importlib
import os
import sys

C = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS = os.path.join(C, "tools")
SAMPLES = os.path.join(C, "samples")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
PKG = "capjudge" if os.path.isdir(os.path.join(TOOLS, "capjudge")) else "judge"


def judge_mod(name):
    return importlib.import_module("%s.%s" % (PKG, name))
