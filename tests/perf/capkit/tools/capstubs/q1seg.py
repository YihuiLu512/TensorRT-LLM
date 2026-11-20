# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-segment CPU timing of the real new-architecture frontend's request path (front-end item
10, design card 10', PLAN-R2R5), without changing product code.

Loaded by capstubs/q1_seg/sitecustomize.py (that directory first on the frontend role's
PYTHONPATH). Off unless CAPKIT_Q1SEG names segments. With it on, an import hook wraps each named
SYNCHRONOUS product callable when its module finishes loading; every call on the process's main
thread (the frontend's event loop) adds to that segment's count, thread CPU
(time.thread_time_ns) and wall time (time.perf_counter_ns). Calls on other threads are only
counted (n_other). A timed call made inside another timed call is counted (n, n_nested) but not
timed again, so the segment sums never count the same CPU twice. Segments:

  S1j  tensorrt_llm.serve.openai_server's json.loads: the request body's JSON parse. The
       server's route class _MsgspecRoute makes FastAPI read the body through
       _MsgspecRequest.json, which calls that module's own json.loads (not Starlette's
       Request.json), so the module global json of openai_server is replaced by a proxy whose
       loads is timed; json.loads everywhere else stays the real one
  S1k  tensorrt_llm.serve.kv_hints KvHintsRequest.filter_kv_hints: the hint trust policy applied
       to the decoded body (called by _MsgspecRequest.json)
  S1v  fastapi._compat.v2.ModelField.validate: pydantic validation of the body (CompletionRequest)
  S2   OpenEngineLLM.generate_async (facade): building the engine request (sync)
  S3   completion_stream_post_processor: OpenAI event conversion and SSE encoding, per output
       (wrapped in tensorrt_llm.serve.openai_server and postprocess_handlers, the same object)
  S4   OpenAIServer.await_disconnected: count only (a coroutine function; its CPU is remainder)
  S5   GenerateStream._validate: the gRPC stream's per-response check (sync part of the read)

Injection site H1 (CAPKIT_Q1SEG_INJECT 'H1:count' or 'H1:busy:<ns>'): the frontend's
ServerArrivalTimeMiddleware.__call__ (the location of capprobe site H1): for every ASGI http scope
whose path starts with /v1/, count one hit and, with busy, spin on the main thread for <ns>
(GIL held) before the middleware's coroutine is returned; applied_ns is the measured spin. It is
outside every segment (remainder), so a busy arm checks that the remainder grows by applied_ns.

A requested segment whose callable is missing, is a coroutine function where a timed one is
needed, or (S4) is not one, stops the process at module load (exit 3, marker line on stderr);
one whose module has not loaded by the first timed main-thread call stops it then. The cumulative
counters are written as one stderr line 'capkit-q1seg-report {json}' every CAPKIT_Q1SEG_REPORT_MS
(default 250) and at exit, stamped with time.monotonic_ns() (the clock of the orchestrator's tier
windows); window_delta() interpolates them at a tier's window ends.
"""

import atexit
import functools
import importlib.abc
import inspect
import json
import os
import sys
import threading
import time

ENV = "CAPKIT_Q1SEG"
ENV_INJECT = "CAPKIT_Q1SEG_INJECT"
ENV_REPORT_MS = "CAPKIT_Q1SEG_REPORT_MS"
REPORT = "capkit-q1seg-report "
MARK_LOAD = "capkit-q1seg: loaded pid=%d segments=%s inject=%s"
MARK_SITE = "capkit-q1seg: site %s installed (%s.%s) pid=%d"
MARK_REFUSE = "capkit-q1seg: refusing to run: %s"
EXIT_SPEC, EXIT_SITE = 2, 3
REPORT_MS_DEFAULT = 250

# id -> (module, target, kind). kind: time = sync callable, timed; count_coro = a coroutine
# function, counted only; inject = the H1 injection wrapper.
SEGMENTS = {
    "S1j": ("tensorrt_llm.serve.openai_server", "json.loads", "time"),
    "S1k": ("tensorrt_llm.serve.kv_hints", "KvHintsRequest.filter_kv_hints", "time"),
    "S1v": ("fastapi._compat.v2", "ModelField.validate", "time"),
    "S2": ("tensorrt_llm.serve.openengine_poc.facade", "OpenEngineLLM.generate_async", "time"),
    "S3": ("tensorrt_llm.serve.openai_server", "completion_stream_post_processor", "time"),
    "S4": ("tensorrt_llm.serve.openai_server", "OpenAIServer.await_disconnected", "count_coro"),
    "S5": ("tensorrt_llm.grpc.openengine_client.stream", "GenerateStream._validate", "time"),
}
TIMED = tuple(s for s, v in SEGMENTS.items() if v[2] == "time")
# Segments summed into the card's sum of S (S4 is count only): S1 = S1j + S1k + S1v.
S1_SEGMENTS = ("S1j", "S1k", "S1v")
SUM_SEGMENTS = S1_SEGMENTS + ("S2", "S3", "S5")
# Sites whose target is a module global json's loads (the module's json becomes a _JsonProxy).
JSON_LOADS = "json.loads"
INJECT_SITES = {"H1": ("tensorrt_llm.serve.responses_utils",
                       "ServerArrivalTimeMiddleware.__call__", "inject")}
S3_ALSO = "tensorrt_llm.serve.postprocess_handlers"
# Counter layout per segment: n, cpu_ns, wall_ns, n_other (other threads), n_nested.
SEG_FIELDS = ("n", "cpu_ns", "wall_ns", "n_other", "n_nested")
# Per injection site: hits, applied, applied_ns, hits_other.
INJ_FIELDS = ("hits", "applied", "applied_ns", "hits_other")


class SpecError(ValueError):
    pass


class SiteError(RuntimeError):
    pass


def parse_segments(text):
    """CAPKIT_Q1SEG: 'all' or a comma list of segment ids; empty = off."""
    text = (text or "").strip()
    if not text:
        return ()
    if text == "all":
        return tuple(SEGMENTS)
    out = []
    for x in text.split(","):
        x = x.strip()
        if x not in SEGMENTS:
            raise SpecError("unknown segment %r (known: %s)" % (x, ",".join(SEGMENTS)))
        if x in out:
            raise SpecError("segment %s named twice" % x)
        out.append(x)
    return tuple(out)


def parse_inject(text):
    """CAPKIT_Q1SEG_INJECT: '' (off), 'H1:count' or 'H1:busy:<delta_ns>' -> dict or None."""
    text = (text or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if parts[0] not in INJECT_SITES:
        raise SpecError("unknown injection site %r (known: %s)" % (parts[0], list(INJECT_SITES)))
    if parts[1:] == ["count"]:
        return {"site": parts[0], "mode": "count", "delta_ns": 0}
    if len(parts) == 3 and parts[1] == "busy" and parts[2].isdigit() and int(parts[2]) > 0:
        return {"site": parts[0], "mode": "busy", "delta_ns": int(parts[2])}
    raise SpecError("injection spec %r: want H1:count or H1:busy:<ns > 0>" % text)


# ---------------------------------------------------------------- wrappers
def _no_report():
    return None


def make_timed(fn, c, main_ident, depth, check, next_report=None, report=_no_report):
    """Wrap sync `fn`: main-thread calls add to c = [n, cpu_ns, wall_ns, n_other, n_nested];
    depth = [int] shared by every timed wrapper (nested calls are not timed twice); check =
    [callable or None], called once by the first main-thread call (completeness check);
    report() is called after a top-level main-thread call that ends at or after next_report[0]
    (perf_counter_ns), outside the timed interval."""
    if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
        raise SiteError("%r is a coroutine / async generator function; only synchronous "
                        "callables can be timed with thread CPU" % (fn,))
    if not callable(fn):
        raise SiteError("%r is not callable" % (fn,))
    tt, pc, gi = time.thread_time_ns, time.perf_counter_ns, threading.get_ident
    nxt = next_report if next_report is not None else [1 << 62]

    def q1seg_timed(*args, **kwargs):
        if gi() != main_ident:
            c[3] += 1
            return fn(*args, **kwargs)
        if check[0] is not None:
            check[0]()
        if depth[0]:
            c[0] += 1
            c[4] += 1
            return fn(*args, **kwargs)
        depth[0] = 1
        t0 = tt()
        w0 = pc()
        try:
            return fn(*args, **kwargs)
        finally:
            w1 = pc()
            t1 = tt()
            depth[0] = 0
            c[0] += 1
            c[1] += t1 - t0
            c[2] += w1 - w0
            if w1 >= nxt[0]:
                report()
    functools.update_wrapper(q1seg_timed, fn)
    q1seg_timed._capkit_q1seg = True
    return q1seg_timed


def make_count_coro(fn, c, main_ident):
    """Wrap a coroutine function: count the call (main thread: c[0], else c[3]) and return the
    coroutine object unchanged (no extra frame; its CPU stays in the remainder)."""
    if not inspect.iscoroutinefunction(fn):
        raise SiteError("%r is not a coroutine function (count-only site expects one)" % (fn,))
    gi = threading.get_ident

    def q1seg_count(*args, **kwargs):
        if gi() == main_ident:
            c[0] += 1
        else:
            c[3] += 1
        return fn(*args, **kwargs)
    functools.update_wrapper(q1seg_count, fn)
    q1seg_count._capkit_q1seg = True
    return q1seg_count


def make_h1(fn, c, main_ident, spec):
    """Wrap ServerArrivalTimeMiddleware.__call__ (a coroutine function): for an http scope with a
    /v1/ path count a hit (c = [hits, applied, applied_ns, hits_other]) and with mode busy spin
    delta_ns on the calling (main) thread, then return the original coroutine."""
    if not inspect.iscoroutinefunction(fn):
        raise SiteError("%r is not a coroutine function (ASGI __call__ expected)" % (fn,))
    gi, pc = threading.get_ident, time.perf_counter_ns
    busy = spec["mode"] == "busy"
    delta = spec["delta_ns"]

    def q1seg_h1(self, scope, receive, send):
        if scope.get("type") == "http" and str(scope.get("path", "")).startswith("/v1/"):
            if gi() != main_ident:
                c[3] += 1
            else:
                c[0] += 1
                if busy:
                    t0 = pc()
                    end = t0 + delta
                    t = t0
                    while t < end:
                        t = pc()
                    c[1] += 1
                    c[2] += t - t0
        return fn(self, scope, receive, send)
    functools.update_wrapper(q1seg_h1, fn)
    q1seg_h1._capkit_q1seg = True
    return q1seg_h1


class _JsonProxy:
    """Stands in for the json module in one module's globals: `loads` is the wrapper, every
    other attribute is the real module's."""

    def __init__(self, real, loads):
        self._real = real
        self.loads = loads

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------- state and installation
def _die(msg, code):
    sys.stderr.write(MARK_REFUSE % msg + "\n")
    sys.stderr.flush()
    os._exit(code)


class State:
    """Counters, wrappers and the report line of one process. Reports are written by the main
    thread itself (after a timed call, at most every report_ns, and at exit): no extra thread."""

    def __init__(self, segments, inject, report_ms=REPORT_MS_DEFAULT, main_ident=None,
                 out=None, die=_die):
        self.segments = tuple(segments)
        self.inject = inject
        self.main_ident = main_ident if main_ident is not None else threading.main_thread().ident
        self.counters = {s: [0] * len(SEG_FIELDS) for s in self.segments}
        self.inj = {inject["site"]: [0] * len(INJ_FIELDS)} if inject else {}
        self.installed = {}
        self.depth = [0]
        self.check = [self.check_complete]
        self.report_ns = int(report_ms) * 1_000_000
        self.next_report = [0]
        self.seq = 0
        self.out = out
        self.die = die
        self.pid = os.getpid()

    def wanted(self):
        """{module: [site ids]} still to install."""
        mods = {}
        for s in self.segments:
            if s not in self.installed:
                mods.setdefault(SEGMENTS[s][0], []).append(s)
        if self.inject and self.inject["site"] not in self.installed:
            mods.setdefault(INJECT_SITES[self.inject["site"]][0], []).append(self.inject["site"])
        return mods

    def check_complete(self):
        missing = sorted(set(self.segments) - set(self.installed))
        if self.inject and self.inject["site"] not in self.installed:
            missing.append(self.inject["site"])
        if missing:
            self.die("sites %s were never installed (their modules did not load before the "
                     "first timed call)" % missing, EXIT_SITE)
        self.check[0] = None

    def _timed(self, site, fn):
        return make_timed(fn, self.counters[site], self.main_ident, self.depth, self.check,
                          self.next_report, self.report)

    def install_site(self, site, module):
        """Wrap one site in `module` (raises SiteError when it is missing or of the wrong kind)."""
        modname, target, kind = SEGMENTS.get(site) or INJECT_SITES[site]
        if target == JSON_LOADS:
            real = getattr(module, "json", None)
            if isinstance(real, _JsonProxy):
                raise SiteError("%s.json is already wrapped" % modname)
            if real is None or not callable(getattr(real, "loads", None)) \
                    or getattr(real, "__name__", None) != "json":
                raise SiteError("%s has no module global json with loads" % modname)
            module.json = _JsonProxy(real, self._timed(site, real.loads))
        elif "." in target:
            cname, attr = target.split(".", 1)
            cls = getattr(module, cname, None)
            if not isinstance(cls, type):
                raise SiteError("%s.%s is not a class" % (modname, cname))
            raw = inspect.getattr_static(cls, attr, None)
            if raw is None or isinstance(raw, (staticmethod, classmethod)) \
                    or not inspect.isfunction(raw):
                raise SiteError("%s.%s is not a plain function attribute" % (modname, target))
            if getattr(raw, "_capkit_q1seg", False):
                raise SiteError("%s.%s is already wrapped" % (modname, target))
            if kind == "time":
                new = self._timed(site, raw)
            elif kind == "count_coro":
                new = make_count_coro(raw, self.counters[site], self.main_ident)
            else:
                new = make_h1(raw, self.inj[site], self.main_ident, self.inject)
            setattr(cls, attr, new)
        else:  # S3: module global, the same object in postprocess_handlers
            fn = getattr(module, target, None)
            other = sys.modules.get(S3_ALSO)
            if fn is None or other is None or getattr(other, target, None) is not fn:
                raise SiteError("%s.%s missing or not the same object as %s.%s"
                                % (modname, target, S3_ALSO, target))
            new = self._timed(site, fn)
            setattr(module, target, new)
            setattr(other, target, new)
        self.installed[site] = "%s.%s" % (modname, target)
        sys.stderr.write(MARK_SITE % (site, modname, target, self.pid) + "\n")

    def on_loaded(self, modname, module):
        for site in self.wanted().get(modname, []):
            try:
                self.install_site(site, module)
            except SiteError as e:
                self.die("site %s: %s" % (site, e), EXIT_SITE)

    def line(self):
        return {"v": 1, "pid": os.getpid(), "seq": self.seq, "t_ns": time.monotonic_ns(),
                "seg": {s: list(c) for s, c in self.counters.items()},
                "inject": {s: list(c) for s, c in self.inj.items()},
                "installed": sorted(self.installed), "segments": list(self.segments),
                "inject_spec": self.inject}

    def report(self):
        """Write one report line (main thread, outside every timed interval)."""
        self.next_report[0] = time.perf_counter_ns() + self.report_ns
        doc = self.line()
        self.seq += 1
        out = self.out or sys.stderr
        try:
            out.write(REPORT + json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")
            out.flush()
        except (OSError, ValueError):
            pass

    def final(self):
        if os.getpid() == self.pid:
            self.report()


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


def setup(environ=None):
    """Read the switches; None when off. Installs at once in modules already loaded and an
    import hook for the rest; registers the exit report. Bad switches raise SpecError."""
    env = os.environ if environ is None else environ
    segs = parse_segments(env.get(ENV, ""))
    inj = parse_inject(env.get(ENV_INJECT, ""))
    if not segs and inj is None:
        return None
    rms = env.get(ENV_REPORT_MS, "") or str(REPORT_MS_DEFAULT)
    if not rms.isdigit() or int(rms) < 1:
        raise SpecError("%s=%r: want a positive integer (ms)" % (ENV_REPORT_MS, rms))
    st = State(segs, inj, int(rms))
    mods = st.wanted()
    loaded = [m for m in mods if m in sys.modules]
    sys.meta_path.insert(0, _Hook(st, set(mods) - set(loaded)))
    for m in loaded:
        st.on_loaded(m, sys.modules[m])
    atexit.register(st.final)
    return st


# ---------------------------------------------------------------- reading the reports
def parse_reports(lines, pid=None):
    """Report dicts from log lines (any other line ignored), in file order; only `pid` if
    given. A malformed report line raises ValueError (never skipped silently)."""
    out = []
    for ln in lines:
        i = ln.find(REPORT)
        if i < 0:
            continue
        doc = json.loads(ln[i + len(REPORT):])
        if not isinstance(doc, dict) or doc.get("v") != 1:
            raise ValueError("q1seg report of unknown version: %r" % ln[:200])
        if pid is None or doc["pid"] == pid:
            out.append(doc)
    return out


def _at(reports, t, get):
    """Linear interpolation of a cumulative counter at time t; None outside the reports."""
    for a, b in zip(reports, reports[1:]):
        if a["t_ns"] <= t <= b["t_ns"]:
            va, vb = get(a), get(b)
            if va is None or vb is None:
                return None
            if b["t_ns"] == a["t_ns"]:
                return float(vb)
            return va + (vb - va) * (t - a["t_ns"]) / float(b["t_ns"] - a["t_ns"])
    return None


def window_delta(reports, t0, t1):
    """Counter deltas between monotonic times t0 < t1 (interpolated between the bracketing
    reports of one process): {"seg": {id: {field: delta}}, "inject": {...}, "gap_ns": the
    largest report interval touching the window, "n_reports": reports inside}; None when a
    window end lies outside the reports."""
    rs = sorted(reports, key=lambda r: r["t_ns"])
    if len(rs) < 2 or rs[0]["t_ns"] > t0 or rs[-1]["t_ns"] < t1:
        return None
    out = {"seg": {}, "inject": {}}
    for grp, fields in (("seg", SEG_FIELDS), ("inject", INJ_FIELDS)):
        for k in rs[-1][grp]:
            d = {}
            for i, f in enumerate(fields):
                def get(r, k=k, i=i, grp=grp):
                    v = r[grp].get(k)
                    return None if v is None else v[i]
                a, b = _at(rs, t0, get), _at(rs, t1, get)
                d[f] = None if a is None or b is None else b - a
            out[grp][k] = d
    inside = [r for r in rs if t0 <= r["t_ns"] <= t1]
    lo = max(i for i, r in enumerate(rs) if r["t_ns"] <= t0)
    hi = min(i for i, r in enumerate(rs) if r["t_ns"] >= t1)
    out["gap_ns"] = max(rs[i + 1]["t_ns"] - rs[i]["t_ns"] for i in range(lo, hi))
    out["n_reports"] = len(inside)
    return out


def zero_hit_sites(delta, installed):
    """Installed sites without a call in a window (window_delta() output): a segment whose
    main-thread count n is not > 0, the injection site whose main-thread hits are not > 0, or a
    site missing from the delta. Sorted list; empty = every installed site was hit. An installed
    name this module does not know raises ValueError (a report of another version)."""
    bad = []
    for s in installed:
        if s in SEGMENTS:
            v = (delta.get("seg") or {}).get(s)
            n = None if v is None else v.get("n")
        elif s in INJECT_SITES:
            v = (delta.get("inject") or {}).get(s)
            n = None if v is None else v.get("hits")
        else:
            raise ValueError("unknown q1seg site %r in a report's installed list" % (s,))
        if n is None or not n > 0:
            bad.append(s)
    return sorted(bad)


def arrivals(delta):
    """Requests that arrived in a window (window_delta() output), the per-request denominator:
    (H1 hits + hits_other, "H1_hits") when the H1 site reports, else (S2 n + n_other,
    "S2_calls"), else (None, None)."""
    h1 = (delta.get("inject") or {}).get("H1")
    if h1 is not None and h1.get("hits") is not None:
        return h1["hits"] + (h1.get("hits_other") or 0), "H1_hits"
    s2 = (delta.get("seg") or {}).get("S2")
    if s2 is not None and s2.get("n") is not None:
        return s2["n"] + (s2.get("n_other") or 0), "S2_calls"
    return None, None
