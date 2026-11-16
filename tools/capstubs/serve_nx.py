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


def pick_free_port(host):
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(fam, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


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


def start_reporter(path, host, port, timeout_s):
    """Daemon thread: poll a TCP connect to host:port every 100 ms; write `host:port\\n` to path on
    the first success, then end."""
    def run():
        deadline = time.monotonic() + timeout_s
        h = "[%s]" % host if ":" in host else host
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    pass
            except OSError:
                time.sleep(0.1)
                continue
            write_atomic(path, "%s:%d\n" % (h, port))
            return
    t = threading.Thread(target=run, name="snx_report_addr", daemon=True)
    t.start()
    return t


def apply_patches(patch_cuda_stream="auto"):
    """Point GenerationExecutorProxy's worker_cls default at NullWorker; returns what was done."""
    from tensorrt_llm.executor.proxy import GenerationExecutorProxy  # noqa: PLC0415

    from capstubs.null_worker import NullWorker  # noqa: PLC0415
    kwd = GenerationExecutorProxy.__init__.__kwdefaults__
    if not isinstance(kwd, dict) or "worker_cls" not in kwd:
        raise RuntimeError("GenerationExecutorProxy.__init__ has no keyword-only worker_cls")
    kwd["worker_cls"] = NullWorker
    if GenerationExecutorProxy.__init__.__kwdefaults__["worker_cls"] is not NullWorker:
        raise RuntimeError("worker_cls default did not take")
    done = {"worker_cls": "%s.%s" % (NullWorker.__module__, NullWorker.__qualname__),
            "cuda_stream_patched": False}
    if patch_cuda_stream == "auto":
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            class _NoCudaStream:  # still a type: the product annotates `torch.cuda.Stream | None`
                def __init__(self, *args, **kwargs):
                    pass
            torch.cuda.Stream = _NoCudaStream
            done["cuda_stream_patched"] = True
    return done


def main(argv=None):
    from capstubs.null_executor import ENV_VAR  # noqa: PLC0415
    argv = list(sys.argv[1:] if argv is None else argv)
    parts = split_argv(argv)
    if parts is None:
        if not os.environ.get(ENV_VAR):
            print("serve_nx: usage: python -m capstubs.serve_nx [options] -- <trtllm-serve "
                  "args>", file=sys.stderr)
            return 2
        apply_patches("auto")  # re-exec as an attached frontend: no engine here, patch anyway
        serve_args = argv
    else:
        own, serve_args = parts
        a = build_parser().parse_args(own)
        try:
            cfg = resolve_config(a)
        except Exception as e:  # noqa: BLE001 -- any config problem refuses the start
            print("serve_nx: config refused: %s" % e, file=sys.stderr)
            return 3
        if a.pick_port:
            if flag_value(serve_args, "--port") is not None:
                print("serve_nx: --pick-port with --port in the serve arguments", file=sys.stderr)
                return 3
            serve_args = serve_args + ["--port", str(pick_free_port(a.report_host))]
        os.environ[ENV_VAR] = json.dumps(cfg.to_doc(), sort_keys=True)
        done = apply_patches(a.patch_cuda_stream)
        if a.report_addr:
            port = flag_value(serve_args, "--port")
            if port is None or not port.isdigit() or int(port) == 0:
                print("serve_nx: --report-addr needs a fixed --port (or --pick-port)",
                      file=sys.stderr)
                return 3
            start_reporter(a.report_addr, a.report_host, int(port), a.report_timeout_s)
        if a.record:
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
