# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SUPP-Q12 logging ablation: switch off uvicorn's access log in a product process without
changing product code.

Used only by putting this directory first on PYTHONPATH of one role (the frontend). At start-up
Python imports this `sitecustomize`; it first runs the next `sitecustomize.py` on sys.path (the
image's own one, so its behaviour is kept), then wraps uvicorn.config.Config.__init__ so that
every Config is built with access_log=False (what uvicorn's --no-access-log does). It writes one
marker line to stderr when it loads and one each time it overrides a Config, so a launch can be
checked from its logs. Nothing else is changed.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
MARK_LOAD = "capkit-q12-noaccess: loaded pid=%d"
MARK_SET = "capkit-q12-noaccess: uvicorn access_log forced off (was %r) pid=%d"


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


def _patch_uvicorn():
    try:
        import uvicorn.config as _cfg  # noqa: PLC0415
    except ImportError:
        return False
    orig = _cfg.Config.__init__
    if getattr(orig, "_capkit_q12", False):
        return True

    def __init__(self, *args, **kwargs):
        sys.stderr.write(MARK_SET % (kwargs.get("access_log", True), os.getpid()) + "\n")
        kwargs["access_log"] = False
        orig(self, *args, **kwargs)

    __init__._capkit_q12 = True
    _cfg.Config.__init__ = __init__
    return True


CHAINED = _chain_next_sitecustomize()
PATCHED = _patch_uvicorn()
sys.stderr.write(MARK_LOAD % os.getpid() + "\n")
