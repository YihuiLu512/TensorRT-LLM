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
    from capstubs.affinity import add_split_args  # noqa: PLC0415
    add_split_args(p)
    return p


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


def start_reporter(path, host, port, timeout_s, after=None):
    """Daemon thread: poll a TCP connect to host:port every 100 ms; write `host:port\\n` to path on
    the first success, call `after()` (when given), then end."""
    def run():
        deadline = time.monotonic() + timeout_s
        h = "[%s]" % host if ":" in host else host
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0) as c:
                    if self_connected(c):  # not the server; drop it and keep polling
                        raise OSError("self-connect")
            except OSError:
                time.sleep(0.1)
                continue
            write_atomic(path, "%s:%d\n" % (h, port))
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
        done["launcher_env_removed"] = scrubbed
        done["affinity"] = affinity
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
            start_reporter(a.report_addr, a.report_host, int(port), a.report_timeout_s, after)
        if a.record:
            os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
            write_atomic(a.record, json.dumps(
                {"schema": SCHEMA_RECORD, "pid": os.getpid(), "config": cfg.to_doc(),
                 "serve_args": serve_args, "patches": done, "argv0": sys.argv[0]},
                indent=1, sort_keys=True) + "\n")
    from tensorrt_llm.commands.serve import main as serve_main  # noqa: PLC0415
    sys.argv = [sys.argv[0]] + serve_args
    return serve_main()


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
    sys.exit(main())
