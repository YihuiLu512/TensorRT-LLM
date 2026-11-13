# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nvidia-smi queries: GPU list (positive control), compute processes, utilization sampling.

Every check first proves that nvidia-smi lists this node's GPUs (rc 0 and >= 1 GPU). If it does
not, the result is NOT_RUN, never "no compute processes, PASS".

    python -m capenv.gpu check <identity> --out PATH --samples N --every-s S
        -> gpu_idle.json (capkit.gpu_idle/1); exit 0 PASS, 6 RED or NOT_RUN
"""

import argparse
import json
import os
import sys
import time

from capcommon import exitcodes
from capcommon.cli import add_identity_args
from capcommon.exitcodes import ConfigError
from capcommon.identity import Identity
from capcommon.jsonio import write_json_stamped

from . import runcmd

GPU_IDLE_SCHEMA = "capkit.gpu_idle/1"
NVSMI = "nvidia-smi"
GPU_FIELDS = ("index", "uuid", "pci.bus_id", "name", "memory.total", "memory.used",
              "utilization.gpu", "driver_version")
APP_FIELDS = ("gpu_bus_id", "pid", "used_memory")


def _num(v):
    v = v.strip()
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return None  # "[N/A]", "[Not Supported]"


def _csv_rows(text, nfields):
    rows = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != nfields:
            raise ValueError("nvidia-smi row has %d fields, expected %d: %r"
                             % (len(parts), nfields, line))
        rows.append(parts)
    return rows


def _norm_bus(b):
    """Bus ids differ in domain width between queries (00000000:1B:00.0 vs 0000:1B:00.0)."""
    b = b.strip().upper()
    dom, _, rest = b.partition(":")
    return "%s:%s" % (dom[-4:], rest) if rest else b


def query_gpus(timeout_s, runner=runcmd.run):
    """(record, [gpu dict]). Raises nothing; parse errors set record['reason']."""
    rec = runner([NVSMI, "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
                 timeout_s)
    gpus = []
    if runcmd.ok(rec):
        try:
            for p in _csv_rows(rec["stdout"], len(GPU_FIELDS)):
                gpus.append({"index": int(p[0]), "uuid": p[1], "bus_id": _norm_bus(p[2]),
                             "name": p[3], "mem_total_mib": _num(p[4]),
                             "mem_used_mib": _num(p[5]), "util_pct": _num(p[6]),
                             "driver": p[7]})
        except ValueError as e:
            rec = dict(rec, reason="parse error: %s" % e)
            gpus = []
    return rec, gpus


def query_apps(timeout_s, runner=runcmd.run):
    """(record, [{"bus_id", "pid", "used_mib"}])."""
    rec = runner([NVSMI, "--query-compute-apps=" + ",".join(APP_FIELDS),
                  "--format=csv,noheader,nounits"], timeout_s)
    apps = []
    if runcmd.ok(rec):
        try:
            for p in _csv_rows(rec["stdout"], len(APP_FIELDS)):
                apps.append({"bus_id": _norm_bus(p[0]), "pid": _num(p[1]), "used_mib": _num(p[2])})
        except ValueError as e:
            rec = dict(rec, reason="parse error: %s" % e)
            apps = []
    return rec, apps


def positive_control(timeout_s, runner=runcmd.run):
    """(ok, gpu_count, record). ok only if the query ran (rc 0) and listed >= 1 GPU."""
    rec, gpus = query_gpus(timeout_s, runner)
    ok = runcmd.ok(rec) and rec.get("reason") is None and len(gpus) >= 1
    return ok, len(gpus) if ok else 0, rec


def sample(timeout_s, runner=runcmd.run):
    """One sample: (rc, [ {gpu_index, util_pct, mem_used_mib, compute_pids} ]).

    rc is 0 only if both queries succeeded and parsed; otherwise the per-GPU list is empty and
    the caller writes one null row per known GPU with query_rc != 0.
    """
    grec, gpus = query_gpus(timeout_s, runner)
    arec, apps = query_apps(timeout_s, runner)
    if not (runcmd.ok(grec) and grec.get("reason") is None and runcmd.ok(arec)
            and arec.get("reason") is None):
        rc = grec.get("rc") if not runcmd.ok(grec) else arec.get("rc")
        return (rc if rc not in (None, 0) else -1), []
    by_bus = {}
    for a in apps:
        by_bus.setdefault(a["bus_id"], []).append(a["pid"])
    rows = [{"gpu_index": g["index"], "util_pct": g["util_pct"],
             "mem_used_mib": g["mem_used_mib"], "compute_pids": sorted(
                 p for p in by_bus.get(g["bus_id"], []) if p is not None)} for g in gpus]
    unknown_bus = sorted(set(by_bus) - {g["bus_id"] for g in gpus})
    if unknown_bus:  # an app on a GPU the GPU query did not list: do not hide it
        rows.append({"gpu_index": None, "util_pct": None, "mem_used_mib": None,
                     "compute_pids": sorted(p for b in unknown_bus for p in by_bus[b]
                                            if p is not None)})
    return 0, rows


def idle_check(samples, every_s, timeout_s, runner=runcmd.run, sleep=time.sleep):
    """gpu_idle payload: positive control, then `samples` samples; RED on any util > 0 or any
    compute process; NOT_RUN if the positive control fails, samples < 1 or any sample failed."""
    ok, count, pc_rec = positive_control(timeout_s, runner)
    out = {"positive_control_ok": ok, "gpu_count": count,
           "positive_control": {k: pc_rec.get(k) for k in ("cmd", "rc", "stdout_sha256",
                                                           "reason")},
           "expected_samples": samples, "samples": [], "verdict": None, "reasons": []}
    if not ok:
        out["verdict"] = "NOT_RUN"
        out["reasons"].append("nvidia-smi did not list any GPU: %s" % runcmd.why(pc_rec))
        return out
    if samples < 1:
        out["verdict"] = "NOT_RUN"
        out["reasons"].append("samples < 1")
        return out
    failed = 0
    for i in range(samples):
        if i:
            sleep(every_s)
        rc, rows = sample(timeout_s, runner)
        out["samples"].append({"t_mono_ns": time.monotonic_ns(), "query_rc": rc, "gpus": rows})
        if rc != 0:
            failed += 1
    busy = [(s["t_mono_ns"], r) for s in out["samples"] for r in s["gpus"]
            if (r["util_pct"] or 0) > 0 or r["compute_pids"]]
    if failed:
        out["verdict"] = "NOT_RUN"
        out["reasons"].append("%d of %d samples failed" % (failed, samples))
    elif busy:
        out["verdict"] = "RED"
        out["reasons"].append("%d GPU readings busy (util > 0 or compute processes)" % len(busy))
    else:
        out["verdict"] = "PASS"
    out["checked_samples"] = samples - failed
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capenv.gpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    add_identity_args(c)
    c.add_argument("--out", required=True)
    c.add_argument("--samples", type=int, required=True)
    c.add_argument("--every-s", type=float, required=True)
    c.add_argument("--timeout-s", type=float, required=True)
    args = ap.parse_args(argv)
    try:
        ident = Identity(experiment=args.experiment, arm=args.arm, launch_seq=args.launch_seq,
                         job_id=args.job_id, restart_count=args.restart_count,
                         register_sha=args.register_sha, role=args.role, worker=args.worker,
                         pid=os.getpid())
        payload = idle_check(args.samples, args.every_s, args.timeout_s)
        write_json_stamped(args.out, GPU_IDLE_SCHEMA, ident, payload)
    except ConfigError as e:
        print("capenv.gpu: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    print(json.dumps({"verdict": payload["verdict"], "reasons": payload["reasons"]}))
    return exitcodes.OK if payload["verdict"] == "PASS" else exitcodes.ENV_INVALID


if __name__ == "__main__":
    sys.exit(main())
