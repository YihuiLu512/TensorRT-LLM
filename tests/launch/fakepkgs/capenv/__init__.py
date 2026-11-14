# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capfake capenv (test-only stand-in; shadows the real capenv on the tests/launch path).

Only the modules in this directory are faked (canary, sidecar); every other capenv module (e.g.
capenv.spinners and the capenv.procfs it imports) is found in the real package, which is
searched after this directory."""

import os

_REAL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))), "tools", "capenv")
assert os.path.isfile(os.path.join(_REAL, "spinners.py")), _REAL
__path__.append(_REAL)
