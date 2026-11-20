# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Front-end item 10 (design card 10', PLAN-R2R5): per-segment CPU timing of the real frontend's
request path without changing product code (capstubs.q1seg).

Used only by putting this directory first on PYTHONPATH of the frontend role. At start-up Python
imports this `sitecustomize`: it first runs the next `sitecustomize.py` on sys.path (the image's
own one, so its behaviour is kept), then, when CAPKIT_Q1SEG names segments or
CAPKIT_Q1SEG_INJECT names the H1 site, installs the wrappers of capstubs.q1seg (an import hook
for modules not loaded yet). Without either variable nothing else is done. Bad switches stop the
process (exit 2); a missing or wrongly typed site stops it when its module loads (exit 3).
Marker lines on stderr: one when this file loads, one per installed site, then the
'capkit-q1seg-report' lines.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
MARK_LOAD = "capkit-q1seg: loaded pid=%d segments=%s inject=%s"


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
STATE = None
try:
    if os.environ.get("CAPKIT_Q1SEG", "").strip() or os.environ.get("CAPKIT_Q1SEG_INJECT",
                                                                    "").strip():
        from capstubs import q1seg as _q1seg  # noqa: E402

        STATE = _q1seg.setup()
except Exception as _e:  # noqa: BLE001 - site.py would swallow it and start without the sites
    sys.stderr.write("capkit-q1seg: refusing to run: %s: %s\n" % (type(_e).__name__, _e))
    sys.stderr.flush()
    os._exit(2)
sys.stderr.write(MARK_LOAD % (os.getpid(), os.environ.get("CAPKIT_Q1SEG", ""),
                              os.environ.get("CAPKIT_Q1SEG_INJECT", "")) + "\n")
