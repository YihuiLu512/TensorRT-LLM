# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scaling analysis entry point (stdlib only; runs with plain python3.11 on the VM).

    python3.11 analysis/scaling.py --boundary <boundary.json> --pair <base>,<scaled> [...]
        --grows-min <ratio> --flat-max <ratio> --out <scaling.json>

Thin wrapper over `capjudge.scaling` (INTERFACES 14.2): T6 and opposite-side doubling
attribution. Writes <scaling>.cmd.txt next to the product.
"""

import importlib
import os
import sys

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def _module():
    return importlib.import_module("capjudge.scaling")


if __name__ == "__main__":
    sys.exit(_module().main())
