# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E10 entry: `glm5-serve --grpc --grpc-protocol openengine` with an engine substitute and a
run-time probe layer (design card E10 G0 / G1). Nothing in the glm checkout is changed.

    <glm venv python> -m capstubs.serve_glm [options] -- <glm5-serve serve arguments>

- --engine mock (G0): glm's own MockEngine (GLM5_MOCK_ENGINE=1, options in GLM5_MOCK_CONFIG:
  mock_step_ms, mock_tokens_per_step, mock_pattern). --engine g1 (G1): capstubs.glm_g1.G1Engine
  put in place of glm5_serve.engine_api.MockEngine (cli._create_engine imports it at call time);
  checked after start: the engine the server built must be the substitute asked for.
- --tier-table: every load tier's beat_ns must equal --step-ms (the substitute steps on its own
  fixed period; the tier table's beat is what the load source plans with).
- Sites (capcommon.inject, one call per pass, counted before the spec is applied, GIL held):
    M1  glm5_serve.openengine.servicer._token_output: the Generate servicer builds one token
        message per engine response through it (design E10 "servicer 的每消息路径").
  Spec: --inject off (default: no wrapper) | M1:count | M1:busy:NS.
- Diagnostics, every second one JSONL row to <out-dir>/serve_glm.<pid>.jsonl: M1 counters;
  dispatch (EngineClient._dispatch: calls = call_soon_threadsafe batches, responses, largest
  batch); loop (event-loop lag: a task sleeping --loop-period-ms, lateness histogram, sum, max);
  engine (G1: StepStats; G0: MockEngine iterations and active requests); server (phase, requests
  in flight, generate_accepted); threads (Python name <-> native id); gc (capstubs.snx_inject).
- --pick-port appends `--port N`; --report-addr gets `host:port` once the server reports READY
  and the port accepts TCP (glm's own --report_addr is written before the engine starts).
- --procmon-out (capstubs.affinity): process-tree thread sampler on its own core.
"""

import argparse
import asyncio
import atexit
import gc
import hashlib
import json
import os
import socket
import sys
import threading
import time

SCHEMA_RECORD = "capkit.serve_glm/1"
SCHEMA_JSONL = "capkit.serve_glm.diag/1"
ENGINES = ("mock", "g1")
SITE_M1 = ("glm5_serve.openengine.servicer", "_token_output")
HASHED = ("engine_api.py", "engine_client.py", "cli.py", "openengine/server.py",
          "openengine/servicer.py")
LOOP_BUCKETS_US = (50, 100, 250, 500, 1000, 2000, 5000, 10000, 50000, 100000)


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.serve_glm", allow_abbrev=False,
                                description=__doc__.split("\n")[0])
    p.add_argument("--engine", choices=ENGINES, required=True)
    p.add_argument("--step-ms", type=float, required=True, help="mock_step_ms (G0 and G1)")
    p.add_argument("--tokens-per-step", type=int, default=1, help="mock_tokens_per_step")
    p.add_argument("--pattern", default="1,2,3", help="mock_pattern (comma list of token ids)")
    p.add_argument("--tier-table", help="check: every load tier's beat_ns == --step-ms")
    p.add_argument("--out-dir", required=True, help="diagnostics JSONL and the glm address file")
    p.add_argument("--record", help="write the resolved settings here (JSON)")
    p.add_argument("--pick-port", action="store_true")
    p.add_argument("--report-addr")
    p.add_argument("--report-host", default="127.0.0.1")
    p.add_argument("--report-timeout-s", type=float, default=3600.0)
    p.add_argument("--inject", action="append", default=[], metavar="off|M1:count|M1:busy:NS")
    p.add_argument("--diag", default="dispatch,loop,engine,server,threads,gc")
    p.add_argument("--loop-period-ms", type=float, default=10.0)
    p.add_argument("--report-interval-ms", type=int, default=1000)
    from capstubs.affinity import add_split_args  # noqa: PLC0415
    add_split_args(p)
    return p


DIAGS = ("dispatch", "loop", "engine", "server", "threads", "gc")


def parse_diags(text):
    out = sorted({d.strip() for d in (text or "").split(",") if d.strip()})
    bad = sorted(set(out) - set(DIAGS))
    if bad:
        raise ValueError("unknown diagnostics %s (known: %s)" % (bad, list(DIAGS)))
    return out


def parse_inject(items):
    """['M1:busy:5000'] -> InjectSpec or None (count) for M1; [] or ['off'] -> 'off' (no
    wrapper at all: the probe-free path)."""
    from capcommon.inject import InjectSpec  # noqa: PLC0415
    if not items or items == ["off"]:
        return "off"
    if len(items) != 1:
        raise ValueError("one --inject item at most (site M1)")
    parts = items[0].split(":")
    if parts[0] != "M1":
        raise ValueError("unknown site %r (known: M1)" % parts[0])
    if parts[1:] == ["count"]:
        return None
    if len(parts) == 3 and parts[1] == "busy":
        ns = int(parts[2], 10)
        if ns <= 0:
            raise ValueError("busy needs ns > 0")
        return InjectSpec("M1", "busy", delta_ns=ns)
    raise ValueError("inject item %r is not M1:count or M1:busy:NS" % items[0])


def check_tier_beats(path, step_ms):
    """Every load tier's beat_ns equals the substitute's step (ns, rounded); returns the beats."""
    from capcommon.tiers import TierTable  # noqa: PLC0415
    want = int(round(step_ms * 1e6))
    tt = TierTable.load(path)
    beats = sorted({t.beat_ns for t in tt.tiers if t.is_load})
    if beats != [want]:
        raise ValueError("tier table load beats %s != --step-ms %g (%d ns)" % (beats, step_ms,
                                                                              want))
    return beats


def flag_value(args, flag):
    for i, x in enumerate(args):
        if x == flag and i + 1 < len(args):
            return args[i + 1]
        if x.startswith(flag + "="):
            return x[len(flag) + 1:]
    return None


def write_atomic(path, text):
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def file_hashes(root):
    out = {}
    for rel in HASHED:
        try:
            with open(os.path.join(root, rel), "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
        except OSError as e:
            out[rel] = "unreadable: %s" % e
    return out


def loop_bucket(us):
    for i, b in enumerate(LOOP_BUCKETS_US):
        if us <= b:
            return i
    return len(LOOP_BUCKETS_US)


class Probe:
    """The run-time probe layer inside the glm process (all wrappers installed before the glm
    server starts; every counter has one writer thread)."""

    def __init__(self, diags, loop_period_ms, engine_kind):
        self.diags = diags
        self.engine_kind = engine_kind
        self.loop_period_s = loop_period_ms / 1000.0
        self.m1 = None
        self.state = None
        self.engines = []
        self.dispatch = {"calls": 0, "responses": 0, "max_batch": 0}
        self.loop = {"n": 0, "lag_ns": 0, "max_lag_ns": 0,
                     "hist": [0] * (len(LOOP_BUCKETS_US) + 1)}
        self.gc = None
        self.originals = []

    def _patch(self, owner, attr, new):
        orig = getattr(owner, attr)
        self.originals.append((owner, attr, orig))
        new.__wrapped__ = orig
        setattr(owner, attr, new)
        return orig

    def install(self, spec):
        """Wrap the glm targets; spec = 'off', None (count) or an InjectSpec for M1."""
        import importlib  # noqa: PLC0415

        from capcommon.inject import Injector  # noqa: PLC0415
        from glm5_serve import engine_api, engine_client  # noqa: PLC0415
        from glm5_serve.openengine import servicer  # noqa: PLC0415
        mod = importlib.import_module(SITE_M1[0])
        if not callable(getattr(mod, SITE_M1[1], None)):
            raise RuntimeError("site M1: %s.%s not found" % SITE_M1)
        if spec != "off":
            self.m1 = Injector("M1")
            if spec is not None:
                self.m1.set_spec(spec)
            hit = self.m1.hit
            orig_tok = getattr(mod, SITE_M1[1])

            def token_output(*args, **kwargs):
                hit()
                return orig_tok(*args, **kwargs)
            self._patch(mod, SITE_M1[1], token_output)
        probe = self
        orig_state_init = servicer.ServerState.__init__

        def state_init(st, *args, **kwargs):
            orig_state_init(st, *args, **kwargs)
            probe.state = st
        self._patch(servicer.ServerState, "__init__", state_init)
        if "dispatch" in self.diags:
            orig_dispatch = engine_client.EngineClient._dispatch
            d = self.dispatch

            def dispatch(cl, responses):
                n = len(responses)
                d["calls"] += 1
                d["responses"] += n
                if n > d["max_batch"]:
                    d["max_batch"] = n
                return orig_dispatch(cl, responses)
            self._patch(engine_client.EngineClient, "_dispatch", dispatch)
        if "loop" in self.diags:
            orig_start = engine_client.EngineClient.start_dispatcher

            def start_dispatcher(cl, loop):
                orig_start(cl, loop)
                loop.create_task(probe.loop_lag())
            self._patch(engine_client.EngineClient, "start_dispatcher", start_dispatcher)
        if self.engine_kind == "mock":
            orig_mock_init = engine_api.MockEngine.__init__

            def mock_init(eng, *args, **kwargs):
                orig_mock_init(eng, *args, **kwargs)
                probe.engines.append(eng)
            self._patch(engine_api.MockEngine, "__init__", mock_init)
        else:
            from capstubs import glm_g1  # noqa: PLC0415
            self._patch(engine_api, "MockEngine", glm_g1.G1Engine)
            self.engines = glm_g1.INSTANCES
        if "gc" in self.diags:
            from capstubs.snx_inject import GcStats  # noqa: PLC0415
            self.gc = GcStats()
            gc.callbacks.append(self.gc)

    async def loop_lag(self):
        per, mono, lp = self.loop_period_s, time.monotonic_ns, self.loop
        per_ns = int(per * 1e9)
        while True:
            t0 = mono()
            await asyncio.sleep(per)
            lag = mono() - t0 - per_ns
            if lag < 0:
                lag = 0
            lp["n"] += 1
            lp["lag_ns"] += lag
            lp["hist"][loop_bucket(lag // 1000)] += 1
            if lag > lp["max_lag_ns"]:
                lp["max_lag_ns"] = lag

    def engine_mismatch(self):
        """None when exactly one engine was built and it is the kind asked for, else why not."""
        if len(self.engines) != 1:
            return "%d engines built (want 1)" % len(self.engines)
        e = self.engines[0]
        kind = "g1" if hasattr(e, "stats_snapshot") else type(e).__name__
        want = "g1" if self.engine_kind == "g1" else "MockEngine"
        return None if kind == want else "engine built is %s, asked for %s" % (kind, want)

    def engine_row(self):
        if not self.engines:
            return None
        e = self.engines[-1]
        if hasattr(e, "stats_snapshot"):
            return e.stats_snapshot()
        return {"kind": "mock", "iter": getattr(e, "_iter", None),
                "active": len(getattr(e, "_reqs", {}) or {}),
                "step_s": getattr(e, "step_s", None), "n_engines": len(self.engines)}

    def row(self):
        r = {"t_ns": time.monotonic_ns()}
        if self.m1 is not None:
            r["M1"] = {k: v for k, v in self.m1.counters().items()
                       if k in ("hits", "applied", "applied_ns")}
        if "dispatch" in self.diags:
            r["dispatch"] = dict(self.dispatch)
            self.dispatch["max_batch"] = 0
        if "loop" in self.diags:
            r["loop"] = {"n": self.loop["n"], "lag_ns": self.loop["lag_ns"],
                         "max_lag_ns": self.loop["max_lag_ns"], "hist": list(self.loop["hist"])}
            self.loop["max_lag_ns"] = 0
        if "engine" in self.diags:
            r["engine"] = self.engine_row()
        if "server" in self.diags and self.state is not None:
            st = self.state
            r["server"] = {"phase": str(getattr(st, "phase", None)),
                           "requests": len(getattr(st, "requests", {}) or {}),
                           "generate_accepted": getattr(st, "generate_accepted", None)}
        if "threads" in self.diags:
            r["threads"] = sorted([t.name, t.native_id] for t in threading.enumerate()
                                  if t.native_id)
        if self.gc is not None:
            r["gc"] = self.gc.snapshot()
        return r


def start_reporter(probe, path, interval_ms, header):
    """Daemon thread: one JSONL row per interval (and one at exit)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(header, schema=SCHEMA_JSONL, pid=os.getpid()),
                           sort_keys=True) + "\n")
    stop = threading.Event()

    def write():
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(probe.row(), sort_keys=True) + "\n")
        except OSError:
            pass

    def run():
        while not stop.wait(interval_ms / 1000.0):
            write()
    t = threading.Thread(target=run, name="capglm_report", daemon=True)
    t.start()

    def final():
        stop.set()
        write()
    atexit.register(final)
    return t


def start_addr_reporter(probe, path, host, port, timeout_s):
    """Daemon thread: once the server reports READY and host:port accepts TCP, write
    `host:port\\n` to path (capkit readiness kind report_addr)."""
    def run():
        deadline = time.monotonic() + timeout_s
        h = "[%s]" % host if ":" in host else host
        while time.monotonic() < deadline:
            st = probe.state
            if st is None or getattr(st, "phase", None) != getattr(st, "READY", "ready"):
                time.sleep(0.1)
                continue
            why = probe.engine_mismatch()
            if why:  # never report ready: the launch's readiness wait fails visibly
                print("serve_glm: %s; not reporting the address" % why, file=sys.stderr)
                return
            try:
                with socket.create_connection((host, port), timeout=1.0) as c:
                    if self_connected(c):  # not the server; drop it and keep polling
                        raise OSError("self-connect")
            except OSError:
                time.sleep(0.1)
                continue
            write_atomic(path, "%s:%d\n" % (h, port))
            return
    t = threading.Thread(target=run, name="capglm_report_addr", daemon=True)
    t.start()
    return t


# --pick-port: a port below the kernel's ephemeral range that binds now, and the reporter's
# self-connect check; shared with serve_nx (see serve_nx.pick_free_port for why).
from capstubs.serve_nx import pick_free_port, self_connected  # noqa: E402,F401


def mock_config(a):
    return {"mock_step_ms": a.step_ms, "mock_tokens_per_step": a.tokens_per_step,
            "mock_pattern": [int(x) for x in a.pattern.split(",") if x.strip()]}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("serve_glm: usage: python -m capstubs.serve_glm [options] -- <glm5-serve args>",
              file=sys.stderr)
        return 2
    i = argv.index("--")
    a = build_parser().parse_args(argv[:i])
    serve_args = argv[i + 1:]
    try:
        diags = parse_diags(a.diag)
        spec = parse_inject(a.inject)
        beats = check_tier_beats(a.tier_table, a.step_ms) if a.tier_table else None
        if a.step_ms <= 0 and a.engine == "g1":
            raise ValueError("G1 needs --step-ms > 0")
        if a.tokens_per_step < 1:
            raise ValueError("--tokens-per-step must be >= 1")
        if any(x in ("--mock_engine", "--report_addr") or x.startswith("--report_addr=")
               for x in serve_args):
            raise ValueError("serve arguments must not set --mock_engine / --report_addr")
        if "--grpc" not in serve_args or flag_value(serve_args, "--grpc-protocol") != \
                "openengine":
            raise ValueError("serve arguments need --grpc --grpc-protocol openengine")
        if a.pick_port:
            if flag_value(serve_args, "--port") is not None:
                raise ValueError("--pick-port with --port in the serve arguments")
            serve_args = serve_args + ["--port", str(pick_free_port(a.report_host))]
        port = flag_value(serve_args, "--port")
        if a.report_addr and (port is None or not port.isdigit() or int(port) == 0):
            raise ValueError("--report-addr needs a fixed --port (or --pick-port)")
    except Exception as e:  # noqa: BLE001 -- any option / input problem refuses the start
        print("serve_glm: refused: %s" % e, file=sys.stderr)
        return 3
    os.makedirs(a.out_dir, exist_ok=True)
    affinity = None
    if a.worker_cores or a.procmon_out:
        from capstubs.affinity import split_role_cpus  # noqa: PLC0415
        try:
            _wk, affinity = split_role_cpus(a, sys.executable)
        except Exception as e:  # noqa: BLE001 -- refuse the start like a config problem
            print("serve_glm: core split refused: %s" % e, file=sys.stderr)
            return 3
    os.environ["GLM5_MOCK_ENGINE"] = "1"
    os.environ["GLM5_MOCK_CONFIG"] = json.dumps(mock_config(a), sort_keys=True)
    import glm5_serve  # noqa: PLC0415
    root = os.path.dirname(os.path.abspath(glm5_serve.__file__))
    probe = Probe(diags, a.loop_period_ms, a.engine)
    try:
        probe.install(spec)
    except Exception as e:  # noqa: BLE001 -- a missing target refuses the start
        print("serve_glm: probe install refused: %s" % e, file=sys.stderr)
        return 3
    header = {"engine": a.engine, "step_ms": a.step_ms, "tokens_per_step": a.tokens_per_step,
              "inject": list(a.inject), "diags": diags, "loop_period_ms": a.loop_period_ms,
              "switchinterval_s": sys.getswitchinterval()}
    start_reporter(probe, os.path.join(a.out_dir, "serve_glm.%d.jsonl" % os.getpid()),
                   a.report_interval_ms, header)
    if a.report_addr:
        start_addr_reporter(probe, a.report_addr, a.report_host, int(port), a.report_timeout_s)
    if a.record:
        os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
        write_atomic(a.record, json.dumps(
            {"schema": SCHEMA_RECORD, "pid": os.getpid(), "header": header,
             "mock_config": mock_config(a), "tier_beats": beats, "serve_args": serve_args,
             "glm5_serve_root": root, "glm5_serve_version": getattr(glm5_serve, "__version__",
                                                                    None),
             "glm_file_sha256": file_hashes(root), "python": sys.version,
             "executable": sys.executable, "prefix": sys.prefix, "affinity": affinity},
            indent=1, sort_keys=True) + "\n")
    from glm5_serve.cli import main as glm_main  # noqa: PLC0415
    sys.argv = ["glm5-serve"] + serve_args
    try:
        glm_main()
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
    sys.exit(main())
