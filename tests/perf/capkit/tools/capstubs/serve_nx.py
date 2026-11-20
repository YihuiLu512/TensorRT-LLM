# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX entry: `trtllm-serve` with the rank 0 worker's PyExecutor replaced by a NullExecutor
(design 2.3 S-NX "做法"; E8, E9).

    python -m capstubs.serve_nx [S-NX options] -- <trtllm-serve arguments>

- Puts the S-NX config (capkit.snx_config/1, from --config-json plus --set / --tier-table /
  --origin-ns / --out-dir) into TRTLLM_CAPKIT_SNX as inline JSON (MpiPoolSession forwards TRTLLM*
  variables to the worker, where capstubs.null_worker.NullWorker reads it).
- Replaces the default of GenerationExecutorProxy.__init__'s keyword-only worker_cls with
  NullWorker (__kwdefaults__; the product factory never passes worker_cls) and checks it took.
- Optionally (--patch-cuda-stream) replaces torch.cuda.Stream with a dummy when CUDA is not
  available (the proxy constructs one unconditionally before submitting the workers).
- Optionally picks a free port (--pick-port: appends `--port N` to the serve arguments) and, with
  --report-addr, writes `host:port` to that file once a TCP connect to it succeeds: trtllm-serve's
  own --report_addr refuses multi-frontend serving and gRPC serving, which E9's n = 8 arms and the
  engine host need.
- Optionally (--inject, --diag) installs capstubs.snx_inject's run-time site wrappers and host
  diagnostics; the specs go into CAPKIT_SNX_INJECT / CAPKIT_SNX_DIAG so re-executed attached
  frontends install the same (SUPP-Q3). --gc-freeze-after-ready runs gc.collect() + gc.freeze()
  once the reported address accepts TCP.
- C++ host mode (serve arguments with `--openengine-host-impl cpp`, D1): --cpp-host-bin,
  --cpp-host-arg and --cpp-host-inject set TLLM_OPENENGINE_CPP_HOST_BIN / _ARGS and TLLM_D1_INJECT
  for the launcher, which starts the C++ host as attached frontend 1 on the public port (the host
  inherits this process's core set). --report-target private makes --report-addr name the
  launcher's loopback OpenEngine port instead (its Python Generate on lane 0: the E8-2l / N2l
  arms), written once the public port accepts, i.e. after the C++ host is ready.
- Optionally (--grpc-server-option KEY=VALUE, OpenEngine host only) adds gRPC channel arguments
  to the OpenEngine server's option list (tensorrt_llm.grpc.openengine.server._SERVER_OPTIONS,
  which OpenEngineServer copies when it builds grpc.aio.server), e.g. the admission limits
  grpc.server.max_pending_requests / grpc.server.max_pending_requests_hard_limit (PLAN-R2R5 5,
  B11). A key already in the list is replaced in place, a new key appended; every
  grpc.aio.server call in the process is logged with its options to
  <out-dir>/grpc_server_options.<pid>.jsonl (the read-back). The options also go into
  CAPKIT_SNX_GRPC_SERVER_OPTIONS so re-executed attached frontends apply the same.
- E16-zr (--cpp-host-szr-bin / --cpp-host-szr-dir / --cpp-host-szr-arg, C++ host mode only): writes
  capstubs.szr_launch's wrapper into the S-ZR dir and gives it to the launcher as the host binary;
  the wrapper starts the S-ZR replay stub from TRTLLM_CAPKIT_SNX (this launch's tier table and
  origin) on the rank 0 worker's cpus as D1's request entry and lane-1 result source, then execs
  --cpp-host-bin. rank 0 (S-NX) stays up and idle.
- Then rewrites sys.argv to the serve arguments and calls tensorrt_llm.commands.serve:main.

Both architectures: the original `trtllm-serve <model dir>` (frontends x n + executor proxy <->
ZMQ <-> worker) and the new architecture's host `trtllm-serve <model dir> --grpc --grpc-protocol
openengine` (a new-architecture frontend connects with --openengine-endpoint). Attached frontends
re-execute `[sys.executable] + sys.argv` (serve/_frontend_processes.py), i.e. this file as a script
with the serve arguments only; that re-exec mode (no `--`, TRTLLM_CAPKIT_SNX already set) skips the
option parsing and just applies the patch. Imports are absolute for that reason.
"""

import argparse
import json
import os
import socket
import sys
import threading
import time

SCHEMA_RECORD = "capkit.serve_nx/1"
WORKER_KINDS = ("null", "real", "product")
# the worker kind, for re-executed attached frontends (they never build a worker; recorded only)
ENV_WORKER = "CAPKIT_SERVE_WORKER"
LLM_API_KEY_RE = r"^[a-z_][a-z0-9_]*$"
# C++ host mode (D1): the launcher reads these (tensorrt_llm.grpc.openengine.cpp_host)
ENV_CPP_HOST_BIN = "TLLM_OPENENGINE_CPP_HOST_BIN"
ENV_CPP_HOST_ARGS = "TLLM_OPENENGINE_CPP_HOST_ARGS"
ENV_CPP_HOST_INJECT = "TLLM_D1_INJECT"
CPP_HOST_LAUNCHER_FLAGS = ("--listen", "--launcher-endpoint", "--template")
REPORT_TARGETS = ("public", "private")
# --grpc-server-option: OpenEngine server channel arguments (JSON [[key, value], ...] for re-exec)
ENV_GRPC_SERVER_OPTIONS = "CAPKIT_SNX_GRPC_SERVER_OPTIONS"
GRPC_OPTION_KEY_RE = r"^grpc\.[A-Za-z0-9_.]+$"
GRPC_OPTION_INT_RE = r"^-?[0-9]+$"
GRPC_OPTIONS_OWNED = ("grpc.so_reuseport",)  # the product appends it after the list
OE_SERVER_MODULE = "tensorrt_llm.grpc.openengine.server"


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.serve_nx", allow_abbrev=False,
                                description=__doc__.split("\n")[0])
    p.add_argument("--config-json", help="base capkit.snx_config/1 file")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                   help="override one config key (value parsed as JSON, else taken as a string)")
    p.add_argument("--tier-table", help="config tier_table (also needs --origin-ns)")
    p.add_argument("--origin-ns", type=int, help="config origin_ns (monotonic ns)")
    p.add_argument("--out-dir", help="config out_dir (S-NX self-report files)")
    p.add_argument("--pick-port", action="store_true",
                   help="choose a free port on --report-host and append --port N")
    p.add_argument("--report-addr", help="write host:port here once the port accepts TCP")
    p.add_argument("--report-host", default="127.0.0.1")
    p.add_argument("--report-timeout-s", type=float, default=3600.0)
    p.add_argument("--patch-cuda-stream", choices=("auto", "never"), default="auto",
                   help="auto: dummy torch.cuda.Stream when torch.cuda.is_available() is false")
    p.add_argument("--record", help="write the resolved config and argv here (JSON)")
    p.add_argument("--keep-launcher-env", action="store_true",
                   help="keep SLURM_* / PMI_* / PMIX_* (default: removed before the product "
                        "starts, so its MPI worker spawn runs as a singleton)")
    p.add_argument("--stream-interval", type=int,
                   help="LLM API stream_interval: writes <out-dir>/serve_nx.<pid>.llm_api.yaml "
                        "and appends --config <it> (the serve arguments must not have --config)")
    p.add_argument("--inject", action="append", default=[], metavar="SITE:MODE",
                   help="capstubs.snx_inject site spec (SITE:count or SITE:busy:NS); repeatable; "
                        "worker sites (Wr) go to the rank 0 worker through the S-NX config")
    p.add_argument("--diag", default="", metavar="LIST",
                   help="capstubs.snx_inject diagnostics, comma list of gc,threads,conn,tname")
    p.add_argument("--gc-freeze-after-ready", action="store_true",
                   help="gc.collect() + gc.freeze() once --report-addr has been written")
    p.add_argument("--worker", choices=WORKER_KINDS, default="null",
                   help="null: S-NX NullWorker (default); real: the product worker with the real "
                        "PyExecutor (S-TM, E11 / E13-S / E15-B), pinned to --worker-cores by "
                        "capstubs.tm_worker.PinnedWorker); product: worker_cls untouched (the "
                        "product's own worker, e.g. E15-B where H-inproc builds its worker "
                        "in-process and both arms must run the same worker class)")
    p.add_argument("--llm-api-set", action="append", default=[], metavar="KEY=JSON",
                   help="one more top-level key of the LLM API YAML that --stream-interval "
                        "writes (value parsed as JSON); repeatable")
    p.add_argument("--cpp-host-bin", metavar="PATH",
                   help="C++ host mode: the host executable (%s)" % ENV_CPP_HOST_BIN)
    p.add_argument("--cpp-host-arg", action="append", default=[], metavar="ARG",
                   help="C++ host mode: one more host argument, in order, joined into %s; write "
                        "flags as --cpp-host-arg=--diag-out; repeatable" % ENV_CPP_HOST_ARGS)
    p.add_argument("--cpp-host-inject", metavar="SPECS",
                   help="C++ host mode: %s, e.g. hs:busy:2000" % ENV_CPP_HOST_INJECT)
    p.add_argument("--cpp-host-szr-bin", metavar="PATH",
                   help="C++ host mode, E16-zr: the S-ZR replay stub (trtllm-oehost-szr). The "
                        "launcher then starts <--cpp-host-szr-dir>/szr-d1-wrap (written here, "
                        "capstubs.szr_launch) in place of --cpp-host-bin: it starts S-ZR from this "
                        "process's S-NX config on the rank 0 worker's cpus as D1's request entry "
                        "and result source on lane 1, and execs --cpp-host-bin")
    p.add_argument("--cpp-host-szr-dir", metavar="DIR",
                   help="E16-zr: S-ZR's out dir (templates, timetable, self-report, wrapper)")
    p.add_argument("--cpp-host-szr-arg", action="append", default=[], metavar="ARG",
                   help="E16-zr: one more capstubs.szr_launch run option, in order (write flags "
                        "as --cpp-host-szr-arg=--dump); repeatable")
    p.add_argument("--report-target", choices=REPORT_TARGETS, default="public",
                   help="private (C++ host mode): --report-addr gets the launcher's loopback "
                        "OpenEngine address, written once the public port accepts")
    p.add_argument("--grpc-server-option", action="append", default=[], metavar="KEY=VALUE",
                   help="OpenEngine host (--grpc-protocol openengine): one more gRPC server "
                        "channel argument in %s._SERVER_OPTIONS (VALUE an integer, else a "
                        "string; a key already there is replaced); repeatable" % OE_SERVER_MODULE)
    from capstubs.affinity import add_split_args  # noqa: PLC0415
    add_split_args(p)
    return p


def parse_grpc_server_options(items):
    """[(key, value)] of --grpc-server-option KEY=VALUE items, in order (VALUE an int when it is
    an integer literal, else the string); ValueError on a bad item, a repeated key or a key the
    product sets itself."""
    import re  # noqa: PLC0415
    out, seen = [], set()
    for kv in items:
        k, sep, v = kv.partition("=")
        if not sep or not re.match(GRPC_OPTION_KEY_RE, k) or v == "":
            raise ValueError("--grpc-server-option needs KEY=VALUE with a grpc.* key, got %r" % kv)
        if k in GRPC_OPTIONS_OWNED:
            raise ValueError("--grpc-server-option %s: the product sets that option" % k)
        if k in seen:
            raise ValueError("--grpc-server-option key %r repeated" % k)
        seen.add(k)
        out.append((k, int(v) if re.match(GRPC_OPTION_INT_RE, v) else v))
    return out


def apply_grpc_server_options(opts, out_dir):
    """Put opts into the OpenEngine server module's _SERVER_OPTIONS (in place: a key already
    there is replaced, a new one appended) and wrap grpc.aio.server so every call appends
    {pid, t_ns, options} to <out_dir>/grpc_server_options.<pid>.jsonl (when out_dir is set).
    Returns {"before", "after", "readback"}; RuntimeError when the module has no list."""
    import importlib  # noqa: PLC0415
    import grpc.aio as gaio  # noqa: PLC0415
    srv = importlib.import_module(OE_SERVER_MODULE)
    lst = getattr(srv, "_SERVER_OPTIONS", None)
    if not isinstance(lst, list) or not all(isinstance(x, tuple) and len(x) == 2 for x in lst):
        raise RuntimeError("%s._SERVER_OPTIONS is not a list of (key, value) pairs"
                           % OE_SERVER_MODULE)
    before = [list(x) for x in lst]
    for k, v in opts:
        idx = [i for i, x in enumerate(lst) if x[0] == k]
        if idx:
            for i in idx:
                lst[i] = (k, v)
        else:
            lst.append((k, v))
    after = [list(x) for x in srv._SERVER_OPTIONS]
    path = None if not out_dir else os.path.join(out_dir, "grpc_server_options.%d.jsonl"
                                                 % os.getpid())
    if path:
        os.makedirs(out_dir, exist_ok=True)
    orig = gaio.server

    def server(*args, **kwargs):
        options = kwargs.get("options")
        if options is None and len(args) >= 4:  # server(thread_pool, handlers, interceptors, opts)
            options = args[3]
        if path:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"pid": os.getpid(), "t_ns": time.monotonic_ns(),
                                    "options": [list(x) for x in (options or ())]},
                                   sort_keys=True) + "\n")
        return orig(*args, **kwargs)
    server.__wrapped__ = orig
    gaio.server = server
    return {"before": before, "after": after, "readback": path}


def cpp_host_mode(serve_args):
    """True when the serve arguments select the C++ OpenEngine host."""
    return flag_value(serve_args, "--openengine-host-impl") == "cpp"


def cpp_host_env(a, serve_args):
    """({env var: value} for the C++ host options, None) or ({}, why refused). Also returns the
    --diag-out directory (created by the caller) as env key None."""
    import shlex  # noqa: PLC0415
    given = a.cpp_host_bin or a.cpp_host_arg or a.cpp_host_inject or \
        a.report_target == "private" or a.cpp_host_szr_bin or a.cpp_host_szr_dir or \
        a.cpp_host_szr_arg
    if not given:
        return {}, None
    if not cpp_host_mode(serve_args):
        return {}, ("--cpp-host-* / --report-target private need --openengine-host-impl cpp in "
                    "the serve arguments")
    if (a.cpp_host_szr_dir or a.cpp_host_szr_arg) and not a.cpp_host_szr_bin:
        return {}, "--cpp-host-szr-dir / --cpp-host-szr-arg need --cpp-host-szr-bin"
    if a.cpp_host_szr_bin and not (a.cpp_host_bin and a.cpp_host_szr_dir):
        return {}, "--cpp-host-szr-bin needs --cpp-host-bin (the real host) and --cpp-host-szr-dir"
    if a.cpp_host_szr_bin and a.report_target == "private":
        return {}, "--cpp-host-szr-bin with --report-target private: the load must reach the C++ host"
    for x in a.cpp_host_arg:
        if x.split("=", 1)[0] in CPP_HOST_LAUNCHER_FLAGS:
            return {}, "--cpp-host-arg %s: the launcher sets that flag" % x
    env = {}
    if a.cpp_host_bin:
        env[ENV_CPP_HOST_BIN] = a.cpp_host_bin
    if a.cpp_host_arg:
        env[ENV_CPP_HOST_ARGS] = shlex.join(a.cpp_host_arg)
    if a.cpp_host_inject:
        env[ENV_CPP_HOST_INJECT] = a.cpp_host_inject
    diag = flag_value(a.cpp_host_arg, "--diag-out")
    if diag:
        env[None] = diag
    return env, None


def szr_wrapper(a, cfg):
    """E16-zr: write the S-ZR wrapper (capstubs.szr_launch.write_wrapper, S-NX config from
    TRTLLM_CAPKIT_SNX, S-ZR pinned to the rank 0 worker's cpus when the role was split); returns
    (wrapper path, record dict)."""
    from capstubs import szr_launch  # noqa: PLC0415
    argv = ["--out-dir", a.cpp_host_szr_dir, "--szr-bin", a.cpp_host_szr_bin]
    if cfg.worker_cpus:
        argv += ["--cpus", ",".join(str(c) for c in cfg.worker_cpus)]
    argv += list(a.cpp_host_szr_arg)
    path = szr_launch.write_wrapper(a.cpp_host_szr_dir, a.cpp_host_bin, argv, snx_env=True)
    return path, {"wrapper": path, "szr_bin": a.cpp_host_szr_bin, "d1_bin": a.cpp_host_bin,
                  "argv": argv}


def capture_private_port(holder):
    """Wrap tensorrt_llm.grpc.openengine.cpp_host.cpp_host_command (the launcher imports it at call
    time) so the launcher's loopback port lands in holder["port"]."""
    import tensorrt_llm.grpc.openengine.cpp_host as ch  # noqa: PLC0415
    orig = ch.cpp_host_command

    def cpp_host_command(binary, *, launcher_port, **kw):
        holder["port"] = int(launcher_port)
        return orig(binary, launcher_port=launcher_port, **kw)
    ch.cpp_host_command = cpp_host_command
    return orig


def split_argv(argv):
    """(own options, serve arguments) around the first `--`; None when there is no `--`."""
    if "--" not in argv:
        return None
    i = argv.index("--")
    return argv[:i], argv[i + 1:]


def _coerce(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def resolve_config(a):
    from capcommon.exitcodes import ConfigError  # noqa: PLC0415
    from capstubs.null_executor import SCHEMA_CONFIG, SnxConfig  # noqa: PLC0415
    doc = {"schema": SCHEMA_CONFIG}
    if a.config_json:
        with open(a.config_json, "r", encoding="utf-8") as f:
            doc.update(json.load(f))
    for kv in a.set:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise ConfigError("--set needs KEY=JSON, got %r" % kv)
        doc[k] = _coerce(v)
    for k, v in (("tier_table", a.tier_table), ("origin_ns", a.origin_ns),
                 ("out_dir", a.out_dir)):
        if v is not None:
            doc[k] = v
    cfg = SnxConfig.from_doc(doc)
    if cfg.tier_table:
        from capcommon.tiers import TierTable  # noqa: PLC0415
        from capstubs.null_executor import tier_shape  # noqa: PLC0415
        tier_shape(TierTable.load(cfg.tier_table), cfg)
    return cfg


PORT_RANGE_FILE = "/proc/sys/net/ipv4/ip_local_port_range"
PORT_FLOOR = 10000  # picked ports lie in [PORT_FLOOR, ephemeral low end)


def ephemeral_range(path=PORT_RANGE_FILE):
    """(low, high) of the kernel's ephemeral (local) port range; None when unreadable."""
    try:
        lo, hi = (int(x) for x in open(path).read().split())
        return lo, hi
    except (OSError, ValueError):
        return None


def pick_free_port(host, range_file=PORT_RANGE_FILE, rng=None, tries=200):
    """A port the server can bind later. The server binds tens of seconds after the pick, and
    meanwhile processes poll TCP connects to that port (the address reporter here, the product's
    frontend wait loop); a port inside the ephemeral range can be taken in between as the source
    port of an outgoing connection (on loopback a connect to an unbound port can even
    self-connect), so the server's bind fails with EADDRINUSE. Pick below the ephemeral range,
    where no outgoing connection gets a source port, and check it binds now. Falls back to an
    OS-assigned port (the old behaviour) only when the range is unreadable or leaves no room."""
    import random  # noqa: PLC0415
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    rg = ephemeral_range(range_file)
    rng = rng or random.SystemRandom()
    if rg is not None and rg[0] - PORT_FLOOR >= 1000:
        for _ in range(tries):
            port = rng.randrange(PORT_FLOOR, rg[0])
            with socket.socket(fam, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                except OSError:
                    continue
                return port
    with socket.socket(fam, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def self_connected(sock):
    """True for a loopback TCP self-connect (local and peer address equal): no server answered."""
    try:
        return sock.getsockname()[:2] == sock.getpeername()[:2]
    except OSError:
        return False


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


def tcp_accepts(host, port):
    """True when a TCP connect to host:port succeeds and is not a loopback self-connect."""
    try:
        with socket.create_connection((host, port), timeout=1.0) as c:
            return not self_connected(c)
    except OSError:
        return False


def start_reporter(path, host, port, timeout_s, after=None, target=None):
    """Daemon thread: poll a TCP connect to host:port every 100 ms; write `host:port\\n` to path on
    the first success, call `after()` (when given), then end. With `target` (a callable returning
    (host, port) or None while unknown), the address written is target's once host:port and it
    both accept."""
    def run():
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not tcp_accepts(host, port):
                time.sleep(0.1)
                continue
            th, tp = host, port
            if target is not None:
                t = target()
                if t is None or not tcp_accepts(t[0], t[1]):
                    time.sleep(0.1)
                    continue
                th, tp = t
            h = "[%s]" % th if ":" in th else th
            write_atomic(path, "%s:%d\n" % (h, tp))
            if after is not None:
                after()
            return
    t = threading.Thread(target=run, name="snx_report_addr", daemon=True)
    t.start()
    return t


def parse_llm_api_sets(items):
    """[(key, value)] of --llm-api-set KEY=JSON items, in order; ValueError on a bad item, a
    duplicate key, or stream_interval (that one is --stream-interval)."""
    import re  # noqa: PLC0415
    out, seen = [], set()
    for kv in items:
        k, sep, v = kv.partition("=")
        if not sep or not re.match(LLM_API_KEY_RE, k):
            raise ValueError("--llm-api-set needs KEY=JSON with a lower-case key, got %r" % kv)
        if k == "stream_interval" or k in seen:
            raise ValueError("--llm-api-set key %r repeated or reserved" % k)
        seen.add(k)
        try:
            out.append((k, json.loads(v)))
        except ValueError:
            raise ValueError("--llm-api-set %s: value is not JSON: %r" % (k, v)) from None
    return out


def with_stream_interval(serve_args, s, out_dir, extra=()):
    """(serve args + ["--config", <yaml with stream_interval: s and the extra (key, value)
    pairs, each value written as JSON, which YAML reads as the same value>], None) or
    (args, why)."""
    if s < 1:
        return serve_args, "--stream-interval must be >= 1"
    if any(x in ("--config", "--extra_llm_api_options") or
           x.startswith(("--config=", "--extra_llm_api_options=")) for x in serve_args):
        return serve_args, "--stream-interval with --config in the serve arguments"
    if not out_dir:
        return serve_args, "--stream-interval needs --out-dir (the YAML is written there)"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "serve_nx.%d.llm_api.yaml" % os.getpid())
    write_atomic(path, "stream_interval: %d\n" % s + "".join(
        "%s: %s\n" % (k, json.dumps(v, sort_keys=True)) for k, v in extra))
    return serve_args + ["--config", path], None


# A Slurm step's launcher variables: with them in the environment, the MPI library the product's
# MpiPoolSession uses (MPI_Comm_spawn of the rank 0 worker) may take the process for an srun
# direct launch and hang or abort under `srun --mpi=none`; without them it starts as a singleton.
LAUNCHER_ENV_PREFIXES = ("SLURM_", "SLURMD_", "PMIX_", "PMI_")


def scrub_launcher_env(environ=None):
    """Remove the Slurm / PMI launcher variables from `environ` (default os.environ) before the
    product is imported; returns the sorted names removed."""
    env = os.environ if environ is None else environ
    gone = sorted(k for k in env if k.startswith(LAUNCHER_ENV_PREFIXES))
    for k in gone:
        del env[k]
    return gone


def apply_patches(patch_cuda_stream="auto", worker="null"):
    """Point GenerationExecutorProxy's worker_cls default at NullWorker (worker "null") or at
    capstubs.tm_worker.PinnedWorker (worker "real"); returns what was done."""
    from tensorrt_llm.executor.proxy import GenerationExecutorProxy  # noqa: PLC0415
    if worker == "product":
        kwd = GenerationExecutorProxy.__init__.__kwdefaults__ or {}
        cls = kwd.get("worker_cls")
        done = {"worker_cls": None if cls is None else "%s.%s" % (cls.__module__,
                                                                  cls.__qualname__),
                "worker": worker, "cuda_stream_patched": False}
        if patch_cuda_stream == "auto":
            done["cuda_stream_patched"] = patch_cuda_stream_if_no_cuda()
        return done
    if worker == "real":
        from capstubs.tm_worker import PinnedWorker as cls  # noqa: PLC0415
    elif worker == "null":
        from capstubs.null_worker import NullWorker as cls  # noqa: PLC0415
    else:
        raise ValueError("worker must be one of %s" % (WORKER_KINDS,))
    kwd = GenerationExecutorProxy.__init__.__kwdefaults__
    if not isinstance(kwd, dict) or "worker_cls" not in kwd:
        raise RuntimeError("GenerationExecutorProxy.__init__ has no keyword-only worker_cls")
    kwd["worker_cls"] = cls
    if GenerationExecutorProxy.__init__.__kwdefaults__["worker_cls"] is not cls:
        raise RuntimeError("worker_cls default did not take")
    done = {"worker_cls": "%s.%s" % (cls.__module__, cls.__qualname__), "worker": worker,
            "cuda_stream_patched": False}
    if patch_cuda_stream == "auto":
        done["cuda_stream_patched"] = patch_cuda_stream_if_no_cuda()
    return done


def patch_cuda_stream_if_no_cuda():
    """Replace torch.cuda.Stream with a dummy type when CUDA is not available (the proxy builds
    one unconditionally before it submits the workers); True when patched."""
    import torch  # noqa: PLC0415
    if torch.cuda.is_available():
        return False

    class _NoCudaStream:  # still a type: the product annotates `torch.cuda.Stream | None`
        def __init__(self, *args, **kwargs):
            pass
    torch.cuda.Stream = _NoCudaStream
    return True


def install_inject(out_dir, role):
    """capstubs.snx_inject.install_from_env; (Installed or None, error text or None)."""
    from capstubs import snx_inject  # noqa: PLC0415
    try:
        return snx_inject.install_from_env(out_dir, header={"role": role}), None
    except snx_inject.InjectSpecError as e:
        return None, str(e)


def _snx_out_dir(env_var):
    try:
        return json.loads(os.environ.get(env_var) or "{}").get("out_dir")
    except ValueError:
        return None


def main(argv=None):
    from capstubs.null_executor import ENV_VAR  # noqa: PLC0415
    argv = list(sys.argv[1:] if argv is None else argv)
    parts = split_argv(argv)
    if parts is None:
        if not os.environ.get(ENV_VAR):
            print("serve_nx: usage: python -m capstubs.serve_nx [options] -- <trtllm-serve "
                  "args>", file=sys.stderr)
            return 2
        # re-exec as an attached frontend: no engine here, patch anyway (same worker kind)
        apply_patches("auto", os.environ.get(ENV_WORKER) or "null")
        _inj, why = install_inject(_snx_out_dir(ENV_VAR), "reexec")
        if why:
            print("serve_nx: inject refused: %s" % why, file=sys.stderr)
            return 3
        if os.environ.get(ENV_GRPC_SERVER_OPTIONS):
            apply_grpc_server_options([tuple(x) for x in json.loads(
                os.environ[ENV_GRPC_SERVER_OPTIONS])], _snx_out_dir(ENV_VAR))
        serve_args = argv
    else:
        own, serve_args = parts
        a = build_parser().parse_args(own)
        try:
            cfg = resolve_config(a)
        except Exception as e:  # noqa: BLE001 -- any config problem refuses the start
            print("serve_nx: config refused: %s" % e, file=sys.stderr)
            return 3
        try:
            extra = parse_llm_api_sets(a.llm_api_set)
        except ValueError as e:
            print("serve_nx: %s" % e, file=sys.stderr)
            return 3
        try:
            grpc_opts = parse_grpc_server_options(a.grpc_server_option)
        except ValueError as e:
            print("serve_nx: %s" % e, file=sys.stderr)
            return 3
        if grpc_opts and flag_value(serve_args, "--grpc-protocol") != "openengine":
            print("serve_nx: --grpc-server-option needs --grpc-protocol openengine in the serve "
                  "arguments", file=sys.stderr)
            return 3
        if extra and a.stream_interval is None:
            print("serve_nx: --llm-api-set needs --stream-interval (it writes the YAML)",
                  file=sys.stderr)
            return 3
        if a.stream_interval is not None:
            serve_args, why = with_stream_interval(serve_args, a.stream_interval, cfg.out_dir,
                                                   extra)
            if why:
                print("serve_nx: %s" % why, file=sys.stderr)
                return 3
        if a.pick_port:
            if flag_value(serve_args, "--port") is not None:
                print("serve_nx: --pick-port with --port in the serve arguments", file=sys.stderr)
                return 3
            serve_args = serve_args + ["--port", str(pick_free_port(a.report_host))]
        affinity = None
        if a.worker_cores or a.procmon_out:
            from capstubs.affinity import split_role_cpus  # noqa: PLC0415
            try:
                wk, affinity = split_role_cpus(a, sys.executable)
                if wk:
                    cfg.worker_cpus = wk
                    cfg.validate()
            except Exception as e:  # noqa: BLE001 -- refuse the start like a config problem
                print("serve_nx: core split refused: %s" % e, file=sys.stderr)
                return 3
        from capstubs import snx_inject  # noqa: PLC0415
        try:  # validate before anything starts; the env is what re-executed frontends read
            snx_inject.parse_specs(";".join(a.inject))
            snx_inject.parse_diags(a.diag)
        except snx_inject.InjectSpecError as e:
            print("serve_nx: inject refused: %s" % e, file=sys.stderr)
            return 3
        # worker sites travel in the S-NX config (TRTLLM_CAPKIT_SNX reaches the worker)
        host_inject, worker_inject = snx_inject.split_worker_specs(a.inject)
        if worker_inject:
            if a.worker != "null":
                print("serve_nx: worker inject sites need --worker null", file=sys.stderr)
                return 3
            cfg.worker_inject = ";".join(worker_inject)
        os.environ[ENV_VAR] = json.dumps(cfg.to_doc(), sort_keys=True)
        if a.gc_freeze_after_ready and not a.report_addr:
            print("serve_nx: --gc-freeze-after-ready needs --report-addr", file=sys.stderr)
            return 3
        cpp_env, why = cpp_host_env(a, serve_args)
        if why:
            print("serve_nx: %s" % why, file=sys.stderr)
            return 3
        if a.report_target == "private" and not a.report_addr:
            print("serve_nx: --report-target private needs --report-addr", file=sys.stderr)
            return 3
        diag_dir = cpp_env.pop(None, None)
        if diag_dir:
            os.makedirs(diag_dir, exist_ok=True)
        szr = None
        if a.cpp_host_szr_bin:
            try:
                cpp_env[ENV_CPP_HOST_BIN], szr = szr_wrapper(a, cfg)
            except SystemExit as e:  # argparse refusal of an S-ZR option
                print("serve_nx: --cpp-host-szr-arg refused (%s)" % e, file=sys.stderr)
                return 3
        os.environ.update(cpp_env)
        if host_inject:
            os.environ[snx_inject.ENV_INJECT] = ";".join(host_inject)
        if a.diag:
            os.environ[snx_inject.ENV_DIAG] = a.diag
        scrubbed = [] if a.keep_launcher_env else scrub_launcher_env()
        os.environ[ENV_WORKER] = a.worker
        done = apply_patches(a.patch_cuda_stream, a.worker)
        inj, why = install_inject(cfg.out_dir, "main")
        if why:
            print("serve_nx: inject refused: %s" % why, file=sys.stderr)
            return 3
        done["inject"] = {"specs": list(a.inject), "worker_specs": worker_inject,
                          "diag": a.diag,
                          "jsonl": None if inj is None else inj.path,
                          "gc_freeze_after_ready": bool(a.gc_freeze_after_ready)}
        if grpc_opts:
            try:
                done["grpc_server_options"] = dict(
                    apply_grpc_server_options(grpc_opts, cfg.out_dir),
                    given=[list(x) for x in grpc_opts])
            except (ImportError, RuntimeError) as e:
                print("serve_nx: --grpc-server-option refused: %s" % e, file=sys.stderr)
                return 3
            os.environ[ENV_GRPC_SERVER_OPTIONS] = json.dumps([list(x) for x in grpc_opts])
        done["launcher_env_removed"] = scrubbed
        done["affinity"] = affinity
        private = {}
        if a.report_target == "private":
            capture_private_port(private)
        if cpp_env or a.report_target == "private":
            done["cpp_host"] = {"env": dict(cpp_env), "diag_out": diag_dir,
                                "report_target": a.report_target, "szr": szr}
        if a.report_addr:
            port = flag_value(serve_args, "--port")
            if port is None or not port.isdigit() or int(port) == 0:
                print("serve_nx: --report-addr needs a fixed --port (or --pick-port)",
                      file=sys.stderr)
                return 3
            after = None
            if a.gc_freeze_after_ready:
                def after():
                    n = snx_inject.gc_freeze_now()
                    if cfg.out_dir:
                        write_atomic(os.path.join(cfg.out_dir, "gc_freeze.%d.json" % os.getpid()),
                                     json.dumps({"t_ns": time.monotonic_ns(), "frozen": n}) + "\n")
            target = None
            if a.report_target == "private":
                def target():
                    return None if "port" not in private else ("127.0.0.1", private["port"])
            start_reporter(a.report_addr, a.report_host, int(port), a.report_timeout_s, after,
                           target)
        if a.record:
            os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
            write_atomic(a.record, json.dumps(
                {"schema": SCHEMA_RECORD, "pid": os.getpid(), "config": cfg.to_doc(),
                 "serve_args": serve_args, "patches": done, "argv0": sys.argv[0],
                 "clock": {"monotonic_ns": time.monotonic_ns(), "time_ns": time.time_ns()}},
                indent=1, sort_keys=True) + "\n")
    from tensorrt_llm.commands.serve import main as serve_main  # noqa: PLC0415
    sys.argv = [sys.argv[0]] + serve_args
    return serve_main()


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
    sys.exit(main())
