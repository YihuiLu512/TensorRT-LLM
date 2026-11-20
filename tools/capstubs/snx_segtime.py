# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-thread segment CPU timing of the Python engine host's result path (D1 S0 stage record,
TRTLLM-18068 06_staged-comparison section 2 item 4), installed by capstubs.serve_nx --segtime.

No product file is changed: like capstubs.snx_inject, each site replaces one product callable at
run time in the process serve_nx starts (the engine host), before the server is built. Every call
of a wrapped callable adds 1 to the calling thread's count for that site and its thread CPU
(time.thread_time_ns) to the site's total. Timing is inclusive: a site called inside another
site is timed by both (the readout derives exclusive parts from the nesting below). The counters
are kept per thread (name and native id), so the readout can put each site on its thread and
compare the sum with that thread's CPU from procmon (coverage).

Sites (id: owner, kind; thread they run on in the host):

  dtask    proxy.GenerationExecutorProxy.dispatch_result_task  per frame, proxy_dispatch_result_thread
  recv       ipc.ZeroMqQueue._recv_data      inside dtask: ZeroMQ recv + parse
  parse        ipc.ZeroMqQueue._parse_data   inside recv: HMAC check + pickle.loads
  hmac           ipc.ZeroMqQueue._verify_hmac
  pickle         ipc's module global pickle -> proxy whose loads is timed (dumps etc. unchanged)
  notify     llmapi.utils._SyncQueue.notify_many  inside dtask: call_soon_threadsafe
  wake     llmapi.utils._SyncQueue._notify_many   event loop: wakes the waiting result queues
  handle   result.DetokenizedGenerationResultBase._handle_response  per result, event loop
  hbase      result.GenerationResultBase._handle_response  inside handle
  deser        _torch.pyexecutor.llm_request.LlmResult.deserialize  inside hbase (result bytes)
  format   grpc.openengine.servicer's module global _format_result (protobuf messages)
  pbser    the Generate handler's response serializer (GenerateResponse.SerializeToString),
           wrapped where grpc.unary_stream_rpc_method_handler receives it
  pbdeser  the Generate handler's request deserializer (GenerateRequest.FromString), per request
  gen      llmapi.llm.BaseLLM.generate_async  per request, event loop
  prep     ipc.ZeroMqQueue._prepare_data  pickle.dumps + HMAC of a request (per request)

A site whose target is missing or of the wrong kind refuses the start (SiteMissing) before
anything is changed. The header line of <out_dir>/segtime.<pid>.jsonl carries the sites and a
calibration of the wrapper itself (thread CPU one wrapped no-op call adds, and the part of it that
lands inside the timed interval); then a reporter thread appends one line per interval
{"t_ns": monotonic ns, "threads": [{"name", "native_id", "seg": {site: [n, cpu_ns]}}]}
(cumulative) and one more at exit.
"""

import atexit
import inspect
import json
import os
import threading
import time

SCHEMA = "capkit.snx_segtime/1"
ENV = "CAPKIT_SNX_SEGTIME"
P = "tensorrt_llm."
# id -> (module, owner path inside the module or None, attribute, kind)
SITES = {
    "dtask": (P + "executor.proxy", "GenerationExecutorProxy", "dispatch_result_task", "method"),
    "recv": (P + "executor.ipc", "ZeroMqQueue", "_recv_data", "method"),
    "parse": (P + "executor.ipc", "ZeroMqQueue", "_parse_data", "method"),
    "hmac": (P + "executor.ipc", "ZeroMqQueue", "_verify_hmac", "method"),
    "pickle": (P + "executor.ipc", None, "pickle", "loads_proxy"),
    "notify": (P + "llmapi.utils", "_SyncQueue", "notify_many", "static"),
    "wake": (P + "llmapi.utils", "_SyncQueue", "_notify_many", "static"),
    "handle": (P + "executor.result", "DetokenizedGenerationResultBase", "_handle_response",
               "method"),
    "hbase": (P + "executor.result", "GenerationResultBase", "_handle_response", "method"),
    "deser": (P + "_torch.pyexecutor.llm_request", "LlmResult", "deserialize", "method"),
    "format": (P + "grpc.openengine.servicer", None, "_format_result", "func"),
    "pbser": (P + "grpc.openengine.bindings", "generation_pb2.GenerateResponse",
              "SerializeToString", "grpc_ser"),
    "pbdeser": (P + "grpc.openengine.bindings", "generation_pb2.GenerateRequest", "FromString",
                "grpc_deser"),
    "gen": (P + "llmapi.llm", "BaseLLM", "generate_async", "method"),
    "prep": (P + "executor.ipc", "ZeroMqQueue", "_prepare_data", "method"),
}
# exclusive part = site - children (readout); top-level sites per thread have no parent
NESTING = {"recv": "dtask", "notify": "dtask", "parse": "recv", "hmac": "parse",
           "pickle": "parse", "hbase": "handle", "deser": "hbase"}
CALIBRATE_N = 20000


class SiteMissing(ValueError):
    pass


def parse_sites(text):
    """'all' or a comma list of site ids -> tuple in SITES order; '' -> ()."""
    text = (text or "").strip()
    if not text:
        return ()
    if text == "all":
        return tuple(SITES)
    out = [x.strip() for x in text.split(",") if x.strip()]
    bad = sorted(set(out) - set(SITES))
    if bad or len(set(out)) != len(out):
        raise SiteMissing("bad --segtime list %r (unknown %s; known %s)" % (text, bad,
                                                                           list(SITES)))
    return tuple(s for s in SITES if s in out)


class Registry:
    """Per-thread counters: a flat list [n0, ns0, n1, ns1, ...] per thread (thread-local)."""

    def __init__(self, sites):
        self.sites = tuple(sites)
        self.local = threading.local()
        self.threads = []  # (name, native_id, counters)
        self.lock = threading.Lock()

    def new_thread(self):
        c = [0] * (2 * len(self.sites))
        th = threading.current_thread()
        with self.lock:
            self.threads.append((th.name, threading.get_native_id(), c))
        self.local.c = c
        return c

    def snapshot(self):
        with self.lock:
            rows = list(self.threads)
        return [{"name": nm, "native_id": nid,
                 "seg": {s: [c[2 * i], c[2 * i + 1]] for i, s in enumerate(self.sites)
                         if c[2 * i]}} for nm, nid, c in rows]


def make_timed(fn, idx, reg):
    """Sync wrapper: counts the call and adds its thread CPU to counters[idx] of the thread."""
    if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
        raise SiteMissing("%r is a coroutine / async generator function" % (fn,))
    tt = time.thread_time_ns
    local = reg.local
    i_n, i_ns = 2 * idx, 2 * idx + 1

    def wrapped(*args, **kwargs):
        try:
            c = local.c
        except AttributeError:
            c = reg.new_thread()
        t0 = tt()
        try:
            return fn(*args, **kwargs)
        finally:
            c[i_ns] += tt() - t0
            c[i_n] += 1
    wrapped.__wrapped__ = fn
    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    wrapped.__qualname__ = getattr(fn, "__qualname__", wrapped.__name__)
    wrapped.capkit_segtime = True
    return wrapped


class PickleProxy:
    """Stands in for a module's global `pickle`: loads is timed, every other name is the real
    module's."""

    def __init__(self, real, loads):
        self._real = real
        self.loads = loads

    def __getattr__(self, name):
        return getattr(self._real, name)


def _resolve(modname, path, modules):
    mod = modules.get(modname)
    if mod is None:
        import importlib  # noqa: PLC0415
        try:
            mod = importlib.import_module(modname)
        except ImportError as e:
            raise SiteMissing("cannot import %s: %s" % (modname, e))
    owner = mod
    for part in (path.split(".") if path else ()):
        owner = getattr(owner, part, None)
        if owner is None:
            raise SiteMissing("%s.%s not found" % (modname, path))
    return mod, owner


def _plan(site, modules):
    """(owner, attr, kind, original) after checking the target exists with the expected kind."""
    modname, path, attr, kind = SITES[site]
    mod, owner = _resolve(modname, path, modules)
    if kind == "loads_proxy":
        real = getattr(mod, attr, None)
        if real is None or not callable(getattr(real, "loads", None)) or \
                isinstance(real, PickleProxy):
            raise SiteMissing("site %s: %s.%s is not an unwrapped pickle module" % (site, modname,
                                                                                   attr))
        return mod, attr, kind, real
    if kind in ("grpc_ser", "grpc_deser"):
        target = getattr(owner, attr, None)
        if target is None or not callable(target):
            raise SiteMissing("site %s: %s.%s.%s not callable" % (site, modname, path, attr))
        return owner, attr, kind, target
    static = inspect.getattr_static(owner, attr, None)
    if static is None:
        raise SiteMissing("site %s: %s.%s%s not found" % (site, modname, path + "." if path else "",
                                                          attr))
    if kind == "static" and not isinstance(static, staticmethod):
        raise SiteMissing("site %s: %s.%s.%s is not a staticmethod" % (site, modname, path, attr))
    if kind in ("method", "func") and not inspect.isfunction(static):
        raise SiteMissing("site %s: %s.%s%s is not a plain function" % (
            site, modname, path + "." if path else "", attr))
    fn = static.__func__ if kind == "static" else static
    if getattr(fn, "capkit_segtime", False):
        raise SiteMissing("site %s already wrapped" % site)
    return owner, attr, kind, fn


def calibrate(n=CALIBRATE_N):
    """Thread CPU one wrapped no-op call adds (per_call_ns) and the part inside the timed interval
    (inside_ns), on the calling thread."""
    reg = Registry(("cal",))

    def noop():
        return None
    w = make_timed(noop, 0, reg)
    tt = time.thread_time_ns
    w()
    t0 = tt()
    for _ in range(n):
        noop()
    bare = tt() - t0
    c0 = list(reg.local.c)
    t0 = tt()
    for _ in range(n):
        w()
    wrapped = tt() - t0
    inside = reg.local.c[1] - c0[1]
    return {"n": n, "per_call_ns": round((wrapped - bare) / n, 1),
            "inside_ns": round(inside / n, 1)}


class Installed:
    def __init__(self, reg):
        self.reg = reg
        self.originals = []
        self.path = None
        self._stop = threading.Event()
        self._thread = None
        self.grpc_wrapped = []

    def line(self):
        return {"t_ns": time.monotonic_ns(), "threads": self.reg.snapshot()}

    def write_line(self):
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.line(), sort_keys=True) + "\n")

    def start_reporter(self, interval_s):
        def run():
            while not self._stop.wait(interval_s):
                try:
                    self.write_line()
                except OSError:
                    pass
        self._thread = threading.Thread(target=run, name="snx_segtime_report", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self.write_line()
        except OSError:
            pass

    def uninstall(self):
        for owner, attr, val in reversed(self.originals):
            setattr(owner, attr, val)
        self.originals.clear()


def _grpc_handler_patch(grpc_mod, targets, st):
    """Wrap grpc.unary_stream_rpc_method_handler so the Generate handler gets timed serializers.
    targets: {'response_serializer': (original, wrapped), 'request_deserializer': (...)}."""
    orig = grpc_mod.unary_stream_rpc_method_handler

    def handler(behavior, request_deserializer=None, response_serializer=None):
        hit = []
        for key, val in (("request_deserializer", request_deserializer),
                         ("response_serializer", response_serializer)):
            t = targets.get(key)
            if t is not None and val is not None and val == t[0]:
                hit.append(key)
                if key == "request_deserializer":
                    request_deserializer = t[1]
                else:
                    response_serializer = t[1]
        if hit:
            st.grpc_wrapped.append({"behavior": getattr(behavior, "__qualname__", repr(behavior)),
                                    "wrapped": hit})
        return orig(behavior, request_deserializer=request_deserializer,
                    response_serializer=response_serializer)
    handler.__wrapped__ = orig
    st.originals.append((grpc_mod, "unary_stream_rpc_method_handler", orig))
    grpc_mod.unary_stream_rpc_method_handler = handler


def install(sites, out_dir=None, modules=None, reporter_s=1.0, header=None, grpc_mod=None):
    """Wrap every site in `sites`; with out_dir write the header and start the reporter.
    Raises SiteMissing before anything is changed."""
    modules = {} if modules is None else modules
    sites = tuple(s for s in SITES if s in set(sites))
    plans = {s: _plan(s, modules) for s in sites}
    reg = Registry(sites)
    st = Installed(reg)
    grpc_targets = {}
    for i, s in enumerate(sites):
        owner, attr, kind, orig = plans[s]
        if kind == "loads_proxy":
            proxy = PickleProxy(orig, make_timed(orig.loads, i, reg))
            st.originals.append((owner, attr, orig))
            setattr(owner, attr, proxy)
        elif kind in ("grpc_ser", "grpc_deser"):
            key = "response_serializer" if kind == "grpc_ser" else "request_deserializer"
            grpc_targets[key] = (orig, make_timed(orig, i, reg))
        else:
            static = inspect.getattr_static(owner, attr)
            st.originals.append((owner, attr, static))
            w = make_timed(orig, i, reg)
            setattr(owner, attr, staticmethod(w) if kind == "static" else w)
    if grpc_targets:
        if grpc_mod is None:
            import grpc as grpc_mod  # noqa: PLC0415
        _grpc_handler_patch(grpc_mod, grpc_targets, st)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        st.path = os.path.join(out_dir, "segtime.%d.jsonl" % os.getpid())
        with open(st.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(header or {}, schema=SCHEMA, pid=os.getpid(),
                                    sites=list(sites), nesting=NESTING,
                                    calibration=calibrate(), t_ns=time.monotonic_ns()),
                               sort_keys=True) + "\n")
        st.start_reporter(reporter_s)
        atexit.register(st.stop)
    return st


def read(path):
    """(header, rows) of a segtime jsonl file."""
    with open(path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    if not lines or lines[0].get("schema") != SCHEMA:
        raise ValueError("%s: not a %s file" % (path, SCHEMA))
    return lines[0], lines[1:]


def _flat(row):
    return {(t["name"], t["native_id"], s): v for t in row["threads"] for s, v in t["seg"].items()}


def value_at(rows, t):
    """Cumulative counters linearly interpolated at monotonic time t between the two report rows
    around it; None when t is outside the rows or the gap around t exceeds 2.5 s."""
    ts = [r["t_ns"] for r in rows]
    import bisect  # noqa: PLC0415
    j = bisect.bisect_left(ts, t)
    if j >= len(rows) or (j == 0 and ts[0] != t):
        return None
    if ts[j] == t:
        return {k: [float(v[0]), float(v[1])] for k, v in _flat(rows[j]).items()}
    a, b = rows[j - 1], rows[j]
    if b["t_ns"] - a["t_ns"] > 2_500_000_000:
        return None
    f = (t - a["t_ns"]) / (b["t_ns"] - a["t_ns"])
    fa, fb = _flat(a), _flat(b)
    return {k: [fa.get(k, [0, 0])[0] + f * (v[0] - fa.get(k, [0, 0])[0]),
                fa.get(k, [0, 0])[1] + f * (v[1] - fa.get(k, [0, 0])[1])] for k, v in fb.items()}


def window_delta(rows, ws, we):
    """Per (thread name, native id) and site: interpolated counters at we minus those at ws.
    Returns {(name, nid): {site: [dn, dns]}} or None when the rows do not cover the window."""
    a, b = value_at(rows, ws), value_at(rows, we)
    if a is None or b is None:
        return None
    out = {}
    for (nm, nid, s), v in b.items():
        u = a.get((nm, nid, s), [0.0, 0.0])
        if v[0] - u[0] > 0:
            out.setdefault((nm, nid), {})[s] = [v[0] - u[0], v[1] - u[1]]
    return out
