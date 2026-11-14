# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary analysis entry point (stdlib only; runs with plain python3.11 on the VM).

    python3.11 analysis/boundary.py --inputs <list> --workdir <W> --out <boundary.json>
        [--compare <armA>,<armB> ...]

Thin wrapper over `capjudge.boundary` (INTERFACES 14.1): per-launch T4, per-arm envelope, T5.
Writes <boundary>.cmd.txt next to the product.
"""

import importlib
import os
import sys

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def _module():
    return importlib.import_module("capjudge.boundary")


if __name__ == "__main__":
    sys.exit(_module().main())
