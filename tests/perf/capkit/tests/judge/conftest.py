# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Put <C>/tools on sys.path; expose the judge package whatever its directory name."""

import importlib
import os
import sys

import pytest

C = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOOLS = os.path.join(C, "tools")
SAMPLES = os.path.join(C, "samples")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
PKG = "capjudge" if os.path.isdir(os.path.join(TOOLS, "capjudge")) else "judge"

# Tests whose expectation needs I localisation on a judged sample: the calibration v2 judge has
# no I threshold source until P2-13 (INTERFACES erratum calib-v2-A7). strict: an unexpected pass
# fails the run, so the marker has to go when P2-13 lands.
XFAIL_I_P2_13 = pytest.mark.xfail(
    strict=True, reason="P2-13 SB-C: I threshold source (i1_calibration) not yet implemented; "
                        "see INTERFACES calib-v2-A7")


def judge_mod(name):
    return importlib.import_module("%s.%s" % (PKG, name))
