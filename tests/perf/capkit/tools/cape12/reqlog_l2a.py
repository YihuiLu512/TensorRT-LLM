# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python reqlog_l2a.py <benchmark_serving args>: E12-L2a copy of WT-21's bench_reqlog.py.

The original (J17728 00_plan/scripts/reqlog/bench_reqlog.py, 62 lines) wraps every entry of
backend_request_func.ASYNC_REQUEST_FUNCS and logs one JSON line per request. This copy keeps that
and adds design 12.6(a) W1-W6, all driven by environment variables (one client_env_var set per
client entry of the perf-sanity YAML):

  E12_REQLOG_DIR   output directory (required); one <pid>.jsonl per client process
  E12_MODE         closed | closed_delta | partial_delta | open | replay   (default closed)
  E12_DELTA_MS     W1: closed_delta / partial_delta wait before the real request call
  E12_MASK_SEED    partial_delta: which requests get the delay (fraction E12_MASK_FRAC, default
                   0.5), from random.Random(seed) over request indices
  E12_SCHEDULE     W2: open / replay: JSON {"schema": "cape12.schedule/1", "offsets_s": [...]};
                   request i (in wrapper-entry order) is called at first-entry + offsets_s[i]
  E12_WARMUP_SEED  W4: a client whose argv `--seed` equals this value is the warm-up client: it
                   runs closed, with no delay and no schedule, whatever E12_MODE says

W3: in open / replay mode `--max-concurrency <n>` is removed from argv before benchmark_serving
parses it (the scaffold always writes it; with it the semaphore still caps in-flight requests).
W5: the header line records the final argv, mode, delta, schedule sha256, mask seed / fraction
and whether this is the warm-up client. W6: every request line carries output_sha1 (the
generated text) besides prompt_sha1. Request lines add t_enter (wrapper entry) and t_call (the
real call) so the applied delay / schedule lateness is measurable.
"""

import functools
import hashlib
import itertools
import json
import os
import random
import runpy
import sys
import time

MODES = ("closed", "closed_delta", "partial_delta", "open", "replay")
SCHEDULE_SCHEMA = "cape12.schedule/1"


class WrapperConfigError(ValueError):
    pass


def argv_seed(argv):
    for i, a in enumerate(argv):
        if a == "--seed" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--seed="):
            return a.split("=", 1)[1]
    return None


def strip_max_concurrency(argv):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--max-concurrency":
            i += 2
            continue
        if argv[i].startswith("--max-concurrency="):
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


def load_config(env, argv):
    """The effective wrapper configuration (W4 applied) from the environment and argv."""
    mode = env.get("E12_MODE", "closed")
    if mode not in MODES:
        raise WrapperConfigError("E12_MODE %r not in %s" % (mode, MODES))
    if "E12_REQLOG_DIR" not in env:
        raise WrapperConfigError("E12_REQLOG_DIR is required")
    wseed = env.get("E12_WARMUP_SEED")
    warmup = wseed is not None and argv_seed(argv) == wseed
    cfg = {"out_dir": env["E12_REQLOG_DIR"], "requested_mode": mode, "warmup": warmup,
           "mode": "closed" if warmup else mode, "delta_s": 0.0, "mask_seed": None,
           "mask_frac": None, "schedule_sha256": None, "offsets_s": None}
    m = cfg["mode"]
    if m in ("closed_delta", "partial_delta"):
        if "E12_DELTA_MS" not in env:
            raise WrapperConfigError("%s needs E12_DELTA_MS" % m)
        cfg["delta_s"] = float(env["E12_DELTA_MS"]) / 1000.0
        if cfg["delta_s"] < 0:
            raise WrapperConfigError("E12_DELTA_MS must be >= 0")
    if m == "partial_delta":
        if "E12_MASK_SEED" not in env:
            raise WrapperConfigError("partial_delta needs E12_MASK_SEED")
        cfg["mask_seed"] = int(env["E12_MASK_SEED"])
        cfg["mask_frac"] = float(env.get("E12_MASK_FRAC", "0.5"))
        if not 0.0 < cfg["mask_frac"] < 1.0:
            raise WrapperConfigError("E12_MASK_FRAC must be in (0, 1)")
    if m in ("open", "replay"):
        path = env.get("E12_SCHEDULE")
        if not path:
            raise WrapperConfigError("%s needs E12_SCHEDULE" % m)
        raw = open(path, "rb").read()
        doc = json.loads(raw)
        offs = doc.get("offsets_s") if isinstance(doc, dict) else None
        if doc.get("schema") != SCHEDULE_SCHEMA or not offs or \
                any(not isinstance(x, (int, float)) or x < 0 for x in offs) or \
                any(b < a for a, b in zip(offs, offs[1:])):
            raise WrapperConfigError("%s: need %s with non-empty non-decreasing offsets_s >= 0"
                                     % (path, SCHEDULE_SCHEMA))
        cfg["offsets_s"] = [float(x) for x in offs]
        cfg["schedule_sha256"] = hashlib.sha256(raw).hexdigest()
    cfg["argv"] = strip_max_concurrency(argv) if m in ("open", "replay") else list(argv)
    return cfg


def delayed(cfg, i):
    """Whether request i gets the W1 delay."""
    if cfg["mode"] == "closed_delta":
        return True
    if cfg["mode"] == "partial_delta":
        return random.Random(cfg["mask_seed"] * 1000003 + i).random() < cfg["mask_frac"]
    return False


def _sha1(x):
    return hashlib.sha1(repr(x).encode()).hexdigest()[:12]


class Wrapper:
    """Wraps request functions; writes the header line at construction (W5)."""

    def __init__(self, cfg, log, clock=time.perf_counter, wall=time.time, sleep=None):
        import asyncio  # noqa: PLC0415
        self.cfg, self.log, self.clock, self.wall = cfg, log, clock, wall
        self.sleep = sleep or asyncio.sleep
        self.counter = itertools.count()
        self.perf0, self.wall0 = clock(), wall()
        self.first_entry = None
        hdr = {"hdr": 1, "pid": os.getpid(), "wall0": self.wall0, "perf0": self.perf0}
        hdr.update({k: cfg[k] for k in ("argv", "mode", "requested_mode", "warmup", "delta_s",
                                        "mask_seed", "mask_frac", "schedule_sha256")})
        hdr["n_offsets"] = None if cfg["offsets_s"] is None else len(cfg["offsets_s"])
        log.write(json.dumps(hdr) + "\n")

    async def _hold(self, i, t_enter):
        cfg = self.cfg
        if cfg["mode"] in ("open", "replay"):
            if self.first_entry is None:
                self.first_entry = t_enter
            offs = cfg["offsets_s"]
            if i >= len(offs):
                raise RuntimeError("request %d has no schedule offset (%d)" % (i, len(offs)))
            planned = self.first_entry + offs[i]
            wait = planned - self.clock()
            if wait > 0:
                await self.sleep(wait)
            return planned, False
        if delayed(cfg, i):
            if cfg["delta_s"] > 0:
                await self.sleep(cfg["delta_s"])
            return None, True
        return None, False

    def wrap(self, fn):
        @functools.wraps(fn)
        async def wrapped(request_func_input, *args, **kwargs):
            i = next(self.counter)
            t_enter, w_enter = self.clock(), self.wall()
            planned, got_delta = await self._hold(i, t_enter)
            t_call = self.clock()
            out = await fn(request_func_input, *args, **kwargs)
            t1 = self.clock()
            p0 = self.perf0
            self.log.write(json.dumps({
                "i": i, "t_enter": t_enter - p0, "t_call": t_call - p0, "t0": t_call - p0,
                "planned": None if planned is None else planned - p0, "delayed": got_delta,
                "wall0": w_enter, "dur": t1 - t_call,
                "ttft": getattr(out, "ttft", None), "latency": getattr(out, "latency", None),
                "success": getattr(out, "success", None),
                "prompt_len": getattr(request_func_input, "prompt_len", None),
                "output_len": getattr(request_func_input, "output_len", None),
                "output_tokens": getattr(out, "output_tokens", None),
                "n_itl": len(getattr(out, "itl", []) or []),
                "prompt_sha1": _sha1(getattr(request_func_input, "prompt", "")),
                "output_sha1": _sha1(getattr(out, "generated_text", "")),
                "error": (getattr(out, "error", "") or "")[:300],
            }) + "\n")
            return out

        return wrapped


def main():
    # `python -m` semantics: the working directory, not this script's directory, is sys.path[0].
    sys.path[0] = os.getcwd()
    cfg = load_config(os.environ, sys.argv[1:])
    from tensorrt_llm.serve.scripts import backend_request_func as brf  # noqa: PLC0415
    os.makedirs(cfg["out_dir"], exist_ok=True)
    log = open(os.path.join(cfg["out_dir"], "%d.jsonl" % os.getpid()), "a", buffering=1)
    w = Wrapper(cfg, log)
    for key in list(brf.ASYNC_REQUEST_FUNCS):
        brf.ASYNC_REQUEST_FUNCS[key] = w.wrap(brf.ASYNC_REQUEST_FUNCS[key])
    sys.argv = ["benchmark_serving"] + cfg["argv"]
    runpy.run_module("tensorrt_llm.serve.scripts.benchmark_serving", run_name="__main__",
                     alter_sys=True)


if __name__ == "__main__":
    main()
