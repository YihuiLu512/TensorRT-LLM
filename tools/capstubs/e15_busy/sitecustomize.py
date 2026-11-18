# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E15-B H-inproc+busy start-up hook (design 13.3): put this directory first on PYTHONPATH of the
engine role only. It runs the image's own sitecustomize (next one on sys.path), then
capstubs.e15_busy_core.start_from_env(), and writes one marker line to stderr. With the busy
variables unset it starts nothing."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "capkit-e15-busy: loaded pid=%d started=%s"


def _chain_next_sitecustomize():
    for d in sys.path:
        if not d or os.path.abspath(d) == _HERE:
            continue
        p = os.path.join(d, "sitecustomize.py")
        if os.path.isfile(p):
            ns = {"__name__": "sitecustomize", "__file__": p}
            with open(p, "rb") as f:
                exec(compile(f.read(), p, "exec"), ns)  # noqa: S102  the image's own hook
            return p
    return None


CHAINED = _chain_next_sitecustomize()
try:
    from capstubs.e15_busy_core import start_from_env
    STARTED = start_from_env() is not None
except ValueError as e:  # bad settings: say so, start nothing
    STARTED = False
    sys.stderr.write("capkit-e15-busy: refused: %s\n" % e)
sys.stderr.write(MARK % (os.getpid(), STARTED) + "\n")
