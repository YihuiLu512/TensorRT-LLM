# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX run-time injection and host diagnostics, installed by capstubs.serve_nx (SUPP-Q3).

No product file is changed: each site wraps one product function at run time, in the process
that serve_nx starts (and in every attached frontend, which re-executes serve_nx with the same
environment). A site whose target is missing refuses the start (SiteMissing).

Sites (one call per pass; the pass is counted, then the active spec is applied, then the product
function runs):

    Hs  engine host, event loop: tensorrt_llm.grpc.openengine.servicer._format_result(result,
        format_state) -> responses; the OpenEngine Generate servicer calls it once per engine
        result, then yields each gRPC message (S-NX, stream_interval 1: one message per result).
    Hd  engine host, proxy_dispatch_result_thread: tensorrt_llm.executor.proxy.global_tracer();
        GenerationExecutorProxy.dispatch_result_task calls `global_tracer().log_instant(...)`
        once per result item it dispatches (its only call in proxy.py).
    Fr  new-architecture frontend: tensorrt_llm.grpc.openengine_client.stream.GenerateStream.
        _validate(self, response) -> bool; called once per response read from a Generate call
        (the path of capprobe site F2).
    Pp  new-architecture postproc pool process: tensorrt_llm.serve.openengine_poc.postproc_pool.
        _handle(records, message, tokenizer, hook); called once per message a pool process takes
        from its inbox (open, each event, close). Installed by capstubs/fe_inject (sitecustomize)
        in the pool processes; serve_nx never imports it.

Specs: CAPKIT_SNX_INJECT = 'SITE:count' or 'SITE:busy:NS' items joined by ';'. busy is
capcommon.inject's inline busy wait (the GIL is held). Counters per site: hits / applied /
applied_ns (capcommon.inject.Injector, private slots, one writer thread per site).

Diagnostics (CAPKIT_SNX_DIAG, comma list): gc (gc.callbacks: per generation count, total and
longest pause, and the thread that ran each pause >= 50 ms), threads (Python thread name <->
native id census), conn (ESTABLISHED TCP connections whose local port is one of this process's
listening ports, i.e. accepted connections, with their peer ports).

Every second a reporter thread appends one JSON line to <out_dir>/inject.<pid>.jsonl:
{"t_ns": monotonic ns, "sites": {site: counters}, "gc": {...}, "threads": [...],
"conn": {...}} (keys present when switched on), and one more line at exit.
"""

import atexit
import gc
import json
import os
import threading
import time

from capcommon.inject import Injector, InjectSpec, InjectSpecError

ENV_INJECT = "CAPKIT_SNX_INJECT"
ENV_DIAG = "CAPKIT_SNX_DIAG"
SCHEMA = "capkit.snx_inject/1"
SITES = {
    "Hs": ("tensorrt_llm.grpc.openengine.servicer", "_format_result", None),
    "Hd": ("tensorrt_llm.executor.proxy", "global_tracer", None),
    "Fr": ("tensorrt_llm.grpc.openengine_client.stream", "_validate", "GenerateStream"),
    "Pp": ("tensorrt_llm.serve.openengine_poc.postproc_pool", "_handle", None),
}
DIAGS = ("gc", "threads", "conn")
LONG_PAUSE_NS = 50 * 1000 * 1000
CONN_EVERY_S = 5.0


class SiteMissing(InjectSpecError):
    pass


def parse_specs(text):
    """'Hs:busy:156250;Hd:count' -> {site: InjectSpec or None (count only)}."""
    out = {}
    for item in (text or "").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        site = parts[0]
        if site not in SITES:
            raise InjectSpecError("unknown S-NX inject site %r (known: %s)" % (site, sorted(SITES)))
        if site in out:
            raise InjectSpecError("site %s given twice" % site)
        if parts[1:] == ["count"]:
            out[site] = None
        elif len(parts) == 3 and parts[1] == "busy":
            try:
                ns = int(parts[2], 10)
            except ValueError:
                raise InjectSpecError("busy needs integer ns, got %r" % parts[2])
            out[site] = InjectSpec(site, "busy", delta_ns=ns)
        else:
            raise InjectSpecError("inject item %r is not SITE:count or SITE:busy:NS" % item)
    return out


def parse_diags(text):
    out = [d.strip() for d in (text or "").split(",") if d.strip()]
    bad = sorted(set(out) - set(DIAGS))
    if bad:
        raise InjectSpecError("unknown S-NX diagnostics %s (known: %s)" % (bad, list(DIAGS)))
    return sorted(set(out))


def _target(site, modules):
    """(owner object, attribute name) of a site; modules maps module name -> module."""
    modname, attr, cls = SITES[site]
    mod = modules.get(modname)
    if mod is None:
        import importlib  # noqa: PLC0415
        try:
            mod = importlib.import_module(modname)
        except ImportError as e:
            raise SiteMissing("site %s: cannot import %s: %s" % (site, modname, e))
    owner = mod if cls is None else getattr(mod, cls, None)
    if owner is None or not callable(getattr(owner, attr, None)):
        raise SiteMissing("site %s: %s.%s%s not found or not callable"
                          % (site, modname, cls + "." if cls else "", attr))
    return owner, attr


def _wrap(orig, inj):
    hit = inj.hit

    def wrapped(*args, **kwargs):
        hit()
        return orig(*args, **kwargs)
    wrapped.__wrapped__ = orig
    wrapped.__name__ = getattr(orig, "__name__", "wrapped")
    wrapped.capkit_snx_site = inj.site
    return wrapped


class GcStats:
    """gc.callbacks recorder: per generation count / total / max pause, long pauses listed."""

    def __init__(self, clock=time.monotonic_ns):
        self._clock = clock
        self._t0 = {}
        self.gen = {g: {"n": 0, "ns": 0, "max_ns": 0} for g in (0, 1, 2)}
        self.long = []

    def __call__(self, phase, info):
        tid = threading.get_ident()
        if phase == "start":
            self._t0[tid] = self._clock()
            return
        t0 = self._t0.pop(tid, None)
        if t0 is None:
            return
        t1 = self._clock()
        d = t1 - t0
        g = self.gen.setdefault(info.get("generation", -1), {"n": 0, "ns": 0, "max_ns": 0})
        g["n"] += 1
        g["ns"] += d
        if d > g["max_ns"]:
            g["max_ns"] = d
        if d >= LONG_PAUSE_NS and len(self.long) < 1000:
            th = threading.current_thread()
            self.long.append({"t0_ns": t0, "ns": d, "generation": info.get("generation"),
                              "collected": info.get("collected"),
                              "thread": th.name, "native_id": th.native_id})

    def snapshot(self):
        return {"gen": {str(k): dict(v) for k, v in self.gen.items()},
                "long": list(self.long), "frozen": gc.get_freeze_count(),
                "threshold": list(gc.get_threshold())}


def thread_census():
    return sorted([t.name, t.native_id] for t in threading.enumerate() if t.native_id)


def _socket_inodes(pid="self"):
    out = set()
    d = "/proc/%s/fd" % pid
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        try:
            link = os.readlink(os.path.join(d, n))
        except OSError:
            continue
        if link.startswith("socket:["):
            out.add(link[8:-1])
    return out


def _tcp_rows(paths=("/proc/net/tcp", "/proc/net/tcp6")):
    for p in paths:
        try:
            with open(p, "r", encoding="ascii") as f:
                lines = f.read().splitlines()[1:]
        except OSError:
            continue
        for ln in lines:
            c = ln.split()
            if len(c) < 10:
                continue
            lport = int(c[1].rsplit(":", 1)[1], 16)
            rport = int(c[2].rsplit(":", 1)[1], 16)
            yield c[3], lport, rport, c[9]


def conn_census(pid="self", rows=None):
    """Accepted connections of this process: ESTABLISHED sockets whose local port is one of the
    process's LISTEN ports. Returns {"listen": [ports], "accepted": {port: [peer ports]}}."""
    inodes = _socket_inodes(pid)
    rows = list(_tcp_rows() if rows is None else rows)
    listen = sorted({lp for st, lp, rp, ino in rows if st == "0A" and ino in inodes})
    acc = {}
    for st, lp, rp, ino in rows:
        if st == "01" and ino in inodes and lp in listen:
            acc.setdefault(str(lp), []).append(rp)
    return {"listen": listen, "accepted": {k: sorted(v) for k, v in acc.items()}}


class Installed:
    def __init__(self):
        self.injectors = {}
        self.originals = {}
        self.diags = []
        self.gc = None
        self.path = None
        self._stop = threading.Event()
        self._thread = None
        self._last_conn = None
        self._last_conn_t = 0.0

    def line(self):
        row = {"t_ns": time.monotonic_ns(),
               "sites": {s: {k: v for k, v in inj.counters().items()
                             if k in ("hits", "applied", "applied_ns")}
                         for s, inj in self.injectors.items()}}
        if self.gc is not None:
            row["gc"] = self.gc.snapshot()
        if "threads" in self.diags:
            row["threads"] = thread_census()
        if "conn" in self.diags:
            now = time.monotonic()
            if self._last_conn is None or now - self._last_conn_t >= CONN_EVERY_S:
                self._last_conn, self._last_conn_t = conn_census(), now
                row["conn"] = self._last_conn
        return row

    def write_line(self):
        if self.path is None:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.line(), sort_keys=True) + "\n")

    def start_reporter(self, interval_s=1.0):
        def run():
            while not self._stop.wait(interval_s):
                try:
                    self.write_line()
                except OSError:
                    pass
        self._thread = threading.Thread(target=run, name="snx_inject_report", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self.write_line()
        except OSError:
            pass

    def uninstall(self):
        for site, (owner, attr, orig) in self.originals.items():
            setattr(owner, attr, orig)
        self.originals.clear()
        if self.gc is not None and self.gc in gc.callbacks:
            gc.callbacks.remove(self.gc)


def install(specs, diags=(), out_dir=None, modules=None, reporter=True, header=None):
    """Wrap every site in `specs` ({site: InjectSpec or None}), switch on `diags`, and (with
    out_dir) start the per-second reporter. Returns an Installed. Raises SiteMissing /
    InjectSpecError before anything is changed when a target is missing."""
    modules = {} if modules is None else modules
    targets = {s: _target(s, modules) for s in specs}
    st = Installed()
    st.diags = sorted(set(diags))
    for site, spec in specs.items():
        inj = Injector(site)
        if spec is not None:
            inj.set_spec(spec)
        owner, attr = targets[site]
        orig = getattr(owner, attr)
        st.originals[site] = (owner, attr, orig)
        setattr(owner, attr, _wrap(orig, inj))
        st.injectors[site] = inj
    if "gc" in st.diags:
        st.gc = GcStats()
        gc.callbacks.append(st.gc)
    if out_dir and (specs or st.diags):
        os.makedirs(out_dir, exist_ok=True)
        st.path = os.path.join(out_dir, "inject.%d.jsonl" % os.getpid())
        with open(st.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(header or {}, schema=SCHEMA, pid=os.getpid(),
                                    specs={s: (sp.to_dict() if sp else "count")
                                           for s, sp in specs.items()},
                                    diags=st.diags), sort_keys=True) + "\n")
        if reporter:
            st.start_reporter()
            atexit.register(st.stop)
    return st


def install_from_env(out_dir, environ=None, header=None):
    """install() from CAPKIT_SNX_INJECT / CAPKIT_SNX_DIAG; None when both are empty."""
    env = os.environ if environ is None else environ
    specs = parse_specs(env.get(ENV_INJECT, ""))
    diags = parse_diags(env.get(ENV_DIAG, ""))
    if not specs and not diags:
        return None
    return install(specs, diags, out_dir=out_dir, header=header)


def gc_freeze_now():
    """gc.collect() then gc.freeze(); returns the number of frozen objects."""
    gc.collect()
    gc.freeze()
    return gc.get_freeze_count()
