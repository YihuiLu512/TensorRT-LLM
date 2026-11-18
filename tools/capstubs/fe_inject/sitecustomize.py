# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run-time injection sites in the real new-architecture frontend and its postproc pool processes
(R3 V interventions for E5 and E6), without changing product code.

Used only by putting this directory first on PYTHONPATH of the frontend role(s); the pool
processes are spawn children of the frontend and inherit it. At start-up Python imports this
`sitecustomize`: it first runs the next `sitecustomize.py` on sys.path (the image's own one), then,
when CAPKIT_FE_INJECT names sites, adds an import hook. When a site's module finishes loading in
this process, the hook wraps the site's function with capstubs.snx_inject (one counted pass per
call, then the site's busy wait, GIL held). Sites:

    Fr  tensorrt_llm.grpc.openengine_client.stream.GenerateStream._validate: once per response
        read from a Generate call, in the frontend process (snx_inject site Fr, path of F2).
    Pp  tensorrt_llm.serve.openengine_poc.postproc_pool._handle: once per message a pool process
        takes from its inbox (open, each event, close), in every postproc pool process.

CAPKIT_FE_INJECT uses snx_inject's syntax ('Pp:busy:200000;Fr:count'). The counters (hits,
applied, applied_ns per site) are reported every 2 s and at exit as one stderr line
'capkit-fe-inject-report {json}' per process, or, with CAPKIT_FE_INJECT_DIR set, by snx_inject's
own reporter as <dir>/inject.<pid>.jsonl. Marker lines on stderr: one when this file loads, one
per installed site.
"""

import atexit
import importlib.abc
import json
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
ENV_INJECT = "CAPKIT_FE_INJECT"
ENV_DIR = "CAPKIT_FE_INJECT_DIR"
FE_SITES = ("Fr", "Pp")
MARK_LOAD = "capkit-fe-inject: loaded pid=%d specs=%s"
MARK_SITE = "capkit-fe-inject: site %s installed in %s pid=%d"
REPORT = "capkit-fe-inject-report "
REPORT_EVERY_S = 2.0


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


class _State:
    def __init__(self, specs, out_dir):
        self.specs = specs
        self.out_dir = out_dir
        self.installed = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None

    def line(self):
        sites = {}
        for st in self.installed:
            sites.update(st.line()["sites"])
        return {"pid": os.getpid(), "sites": sites}

    def report(self):
        sys.stderr.write(REPORT + json.dumps(self.line(), sort_keys=True) + "\n")
        sys.stderr.flush()

    def start_reporter(self):
        def run():
            while not self.stop.wait(REPORT_EVERY_S):
                try:
                    self.report()
                except (OSError, ValueError):
                    return
        self.thread = threading.Thread(target=run, name="fe_inject_report", daemon=True)
        self.thread.start()
        atexit.register(self.final)

    def final(self):
        self.stop.set()
        try:
            self.report()
        except (OSError, ValueError):
            pass

    def on_loaded(self, modname, module):
        from capstubs import snx_inject  # noqa: PLC0415
        sub = {s: sp for s, sp in self.specs.items() if snx_inject.SITES[s][0] == modname}
        st = snx_inject.install(sub, out_dir=self.out_dir, modules={modname: module},
                                reporter=bool(self.out_dir), header={"via": "fe_inject"})
        with self.lock:
            first = not self.installed
            self.installed.append(st)
        for s in sorted(sub):
            sys.stderr.write(MARK_SITE % (s, modname, os.getpid()) + "\n")
        if first and not self.out_dir:
            self.start_reporter()


class _Hook(importlib.abc.MetaPathFinder):
    """Finds a site module with the other finders and runs state.on_loaded after it executes."""

    def __init__(self, state, modules):
        self.state = state
        self.modules = set(modules)

    def find_spec(self, name, path, target=None):
        if name not in self.modules:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        self.modules.discard(name)
        loader, state = spec.loader, self.state
        orig = loader.exec_module

        def exec_module(module):
            orig(module)
            state.on_loaded(name, module)
        loader.exec_module = exec_module
        return spec


def _setup():
    text = os.environ.get(ENV_INJECT, "")
    if not text.strip():
        return None
    from capstubs import snx_inject  # noqa: PLC0415
    specs = snx_inject.parse_specs(text)
    bad = sorted(set(specs) - set(FE_SITES))
    if bad:
        raise snx_inject.InjectSpecError("fe_inject sites are %s, got %s" % (FE_SITES, bad))
    state = _State(specs, os.environ.get(ENV_DIR) or None)
    mods = {snx_inject.SITES[s][0] for s in specs}
    loaded = [m for m in mods if m in sys.modules]
    hook = _Hook(state, mods - set(loaded))
    sys.meta_path.insert(0, hook)
    for m in loaded:
        state.on_loaded(m, sys.modules[m])
    return state


CHAINED = _chain_next_sitecustomize()
try:
    STATE = _setup()
except Exception as _e:  # noqa: BLE001 - site.py would swallow it and start without the sites
    sys.stderr.write("capkit-fe-inject: refusing to start: %s: %s\n" % (type(_e).__name__, _e))
    sys.stderr.flush()
    os._exit(2)
sys.stderr.write(MARK_LOAD % (os.getpid(), os.environ.get(ENV_INJECT, "")) + "\n")
