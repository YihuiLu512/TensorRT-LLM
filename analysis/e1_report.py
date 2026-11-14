# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E1 report entry point (stdlib only; runs with plain python3.11 on the VM).

    python3.11 analysis/e1_report.py --boundary <boundary.json> --workdir <W>
        --out <e1_report.json>

Thin wrapper over `capjudge.e1report` (INTERFACES 14.3): slope reconciliation, bytes and accepts.
Writes <e1_report>.cmd.txt next to the product.
"""

import importlib
import os
import sys

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def _module():
    return importlib.import_module("capjudge.e1report")


if __name__ == "__main__":
    sys.exit(_module().main())
