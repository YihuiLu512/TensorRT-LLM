# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-2 calibration file builder entry point (stdlib only; runs with plain python3.11 on the
VM).

    python3.11 analysis/calib_build.py near-boundary --inputs <list> --workdir <W>
        --regime {subsat,near_boundary} --out <near_boundary_noise.json>
    python3.11 analysis/calib_build.py loadgen-band --inputs <list> --workdir <W> --out <F>
    python3.11 analysis/calib_build.py stub-calib --inputs <list> --workdir <W> --out <F>

Thin wrapper over `capjudge.calib` (INTERFACES 9.6-9.8), so the builders and the judge compute
every statistic with the same code. See tools/capjudge/README.md for the list-file formats.
"""

import importlib
import os
import sys

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def _calib_module():
    return importlib.import_module("capjudge.calib")


if __name__ == "__main__":
    sys.exit(_calib_module().main())
