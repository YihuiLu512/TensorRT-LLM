# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E0 pilot builder entry point (stdlib only; runs with plain python3.11 on the VM).

    python3.11 analysis/pilot_build.py --inputs <list file> --workdir <W> --out <pilot.json>

Thin wrapper over `capjudge.pilot` (INTERFACES 11), so the pilot and the judge compute every
statistic with the same code. See tools/capjudge/README.md for the list-file format and the
output.
"""

import importlib
import os
import sys

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def _pilot_module():
    return importlib.import_module("capjudge.pilot")


if __name__ == "__main__":
    sys.exit(_pilot_module().main())
