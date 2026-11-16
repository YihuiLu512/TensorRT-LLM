# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nvidia-smi queries: GPU list (positive control), compute processes, utilization sampling.

The only implementation of the nvidia-smi queries in capkit: the gate (item 5), the hardware
profile, the sidecar and capjobs.envfiles all call these functions. Production callers pass the
nvidia-smi executable explicitly (``nvsmi=``); the default is for unit tests only.

check_visible() compares the GPUs nvidia-smi lists with the number requested from Slurm:
  requested >= 1: the query must list >= 1 GPU (positive control, else NOT_RUN), exactly the
                  requested number (else RED), and no compute process (else RED).
  requested == 0: the query must report no GPU (else RED; unreadable -> NOT_RUN); the positive
                  control and the compute-process check need a listed GPU and are recorded as
                  NOT_RUN with that reason.
A reading that is missing ([N/A], query failure) is never counted as idle.

    python -m capenv.gpu check <identity> --out PATH --samples N --every-s S --timeout-s S
        --expect-gpus N --nvidia-smi PATH
        -> capkit.gpu_idle_check/1 (a sampled idle check; the job's env/gpu_idle.json is
           written by capjobs.envfiles as capkit.gpu_idle/1); exit 0 PASS, 6 RED or NOT_RUN
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

GPU_IDLE_CHECK_SCHEMA = "capkit.gpu_idle_check/1"
NVSMI = "nvidia-smi"  # default for unit tests only; production callers pass nvsmi= explicitly
NO_DEVICES_TEXT = "no devices were found"  # nvidia-smi's message when it sees no GPU
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


def _not_given(argv):
    return {"cmd": list(argv), "rc": None, "stdout": None, "stdout_sha256": None,
            "stderr": None, "reason": "nvidia-smi path not given"}


def _run(nvsmi, args, timeout_s, runner):
    argv = [nvsmi or "<nvidia-smi not given>"] + list(args)
    if not nvsmi:
        return _not_given(argv)
    return runner(argv, timeout_s)


def query_gpus(timeout_s, runner=runcmd.run, nvsmi=NVSMI):
    """(record, [gpu dict]). Raises nothing; parse errors set record['reason']."""
    rec = _run(nvsmi, ["--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
               timeout_s, runner)
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


def query_apps(timeout_s, runner=runcmd.run, nvsmi=NVSMI):
    """(record, [{"bus_id", "pid", "used_mib"}]). An app whose pid reads [N/A] is kept (pid None):
    it is still a compute process."""
    rec = _run(nvsmi, ["--query-compute-apps=" + ",".join(APP_FIELDS),
                       "--format=csv,noheader,nounits"], timeout_s, runner)
    apps = []
    if runcmd.ok(rec):
        try:
            for p in _csv_rows(rec["stdout"], len(APP_FIELDS)):
                apps.append({"bus_id": _norm_bus(p[0]), "pid": _num(p[1]), "used_mib": _num(p[2])})
        except ValueError as e:
            rec = dict(rec, reason="parse error: %s" % e)
            apps = []
    return rec, apps


def positive_control(timeout_s, runner=runcmd.run, nvsmi=NVSMI):
    """(ok, gpu_count, record). ok only if the query ran (rc 0) and listed >= 1 GPU."""
    rec, gpus = query_gpus(timeout_s, runner, nvsmi=nvsmi)
    ok = runcmd.ok(rec) and rec.get("reason") is None and len(gpus) >= 1
    return ok, len(gpus) if ok else 0, rec


def _pids(apps):
    """Sorted known pids, then one None per app whose pid was unreadable (never dropped)."""
    known = sorted(a["pid"] for a in apps if a["pid"] is not None)
    return known + [None] * sum(1 for a in apps if a["pid"] is None)


def sample(timeout_s, runner=runcmd.run, nvsmi=NVSMI):
    """One sample: (rc, [ {gpu_index, util_pct, mem_used_mib, compute_pids} ]).

    rc is 0 only if both queries succeeded and parsed; otherwise the per-GPU list is empty and
    the caller writes one null row per known GPU with query_rc != 0.
    """
    grec, gpus = query_gpus(timeout_s, runner, nvsmi=nvsmi)
    arec, apps = query_apps(timeout_s, runner, nvsmi=nvsmi)
    if not (runcmd.ok(grec) and grec.get("reason") is None and runcmd.ok(arec)
            and arec.get("reason") is None):
        rc = grec.get("rc") if not runcmd.ok(grec) else arec.get("rc")
        return (rc if rc not in (None, 0) else -1), []
    by_bus = {}
    for a in apps:
        by_bus.setdefault(a["bus_id"], []).append(a)
    rows = [{"gpu_index": g["index"], "util_pct": g["util_pct"],
             "mem_used_mib": g["mem_used_mib"], "compute_pids": _pids(by_bus.get(g["bus_id"], []))}
            for g in gpus]
    unknown_bus = sorted(set(by_bus) - {g["bus_id"] for g in gpus})
    if unknown_bus:  # an app on a GPU the GPU query did not list: do not hide it
        rows.append({"gpu_index": None, "util_pct": None, "mem_used_mib": None,
                     "compute_pids": _pids([a for b in unknown_bus for a in by_bus[b]])})
    return 0, rows


def _rec_view(rec):
    return {k: rec.get(k) for k in ("cmd", "rc", "stdout", "stdout_sha256", "stderr", "reason")}


def _status(*subs):
    """RED if any sub-check is RED, else NOT_RUN if any is NOT_RUN, else PASS."""
    st = [x["status"] for x in subs if x is not None]
    return "RED" if "RED" in st else ("NOT_RUN" if "NOT_RUN" in st else "PASS")


def check_visible(expect_gpus, timeout_s, runner=runcmd.run, nvsmi=NVSMI):
    """Visible GPUs == requested GPUs, and no compute process on them (see module docstring).

    Returns {"status": PASS|RED|NOT_RUN, "reasons", "expected_gpus", "visible_gpus" (None when
    unreadable), "gpus", "compute_apps" (None when not queried), "positive_control_ok",
    "count_check", "positive_control", "apps_check" ({"status", "why"} each), "queries"}.
    Raises ConfigError when expect_gpus is not an int >= 0.
    """
    if not isinstance(expect_gpus, int) or isinstance(expect_gpus, bool) or expect_gpus < 0:
        raise ConfigError("expected GPU count must be an int >= 0, got %r" % (expect_gpus,))
    grec, gpus = query_gpus(timeout_s, runner, nvsmi=nvsmi)
    q_ok = runcmd.ok(grec) and grec.get("reason") is None
    out = {"expected_gpus": expect_gpus, "nvidia_smi": nvsmi, "visible_gpus": None,
           "gpus": gpus, "compute_apps": None, "queries": [_rec_view(grec)]}
    if expect_gpus == 0:
        text = ("%s\n%s" % (grec.get("stdout") or "", grec.get("stderr") or "")).lower()
        if q_ok:
            out["visible_gpus"] = len(gpus)
            cnt = {"status": "PASS" if not gpus else "RED",
                   "why": "visible %d, requested 0" % len(gpus)}
        elif grec.get("rc") is not None and NO_DEVICES_TEXT in text:
            out["visible_gpus"] = 0
            cnt = {"status": "PASS", "why": "nvidia-smi rc %s: %r" % (grec["rc"], NO_DEVICES_TEXT)}
        else:
            cnt = {"status": "NOT_RUN", "why": "cannot tell how many GPUs are visible: %s"
                   % runcmd.why(grec)}
        na = "0 GPUs requested: needs a listed GPU, not run"
        pc, apps_chk = {"status": "NOT_RUN", "why": na}, {"status": "NOT_RUN", "why": na}
        out["status"] = cnt["status"]
    else:
        if not q_ok or not gpus:
            why = runcmd.why(grec) if not q_ok else "query listed no GPU"
            pc = {"status": "NOT_RUN", "why": "positive control failed: %s" % why}
            cnt = {"status": "NOT_RUN", "why": "positive control failed"}
            apps_chk = {"status": "NOT_RUN", "why": "positive control failed"}
            out["visible_gpus"] = len(gpus) if q_ok else None
        else:
            out["visible_gpus"] = len(gpus)
            pc = {"status": "PASS", "why": "listed %d GPU(s)" % len(gpus)}
            cnt = {"status": "PASS" if len(gpus) == expect_gpus else "RED",
                   "why": "visible %d, requested %d" % (len(gpus), expect_gpus)}
            arec, apps = query_apps(timeout_s, runner, nvsmi=nvsmi)
            out["queries"].append(_rec_view(arec))
            if runcmd.ok(arec) and arec.get("reason") is None:
                out["compute_apps"] = apps
                apps_chk = {"status": "RED" if apps else "PASS",
                            "why": "%d compute process(es)" % len(apps)}
            else:
                apps_chk = {"status": "NOT_RUN", "why": "compute-apps query failed: %s"
                            % runcmd.why(arec)}
        out["status"] = _status(cnt, pc, apps_chk)
    out["count_check"], out["positive_control"], out["apps_check"] = cnt, pc, apps_chk
    out["positive_control_ok"] = pc["status"] == "PASS"
    out["reasons"] = ["%s %s: %s" % (n, c["status"], c["why"]) for n, c in
                      (("count", cnt), ("positive_control", pc), ("compute_apps", apps_chk))
                      if c["status"] != "PASS"]
    return out


def idle_check(samples, every_s, timeout_s, expect_gpus, runner=runcmd.run, sleep=time.sleep,
               nvsmi=NVSMI):
    """gpu_idle payload: check_visible(), then `samples` utilization samples.

    RED on a visible-count mismatch, any util > 0 or any compute process (samples are still
    taken as evidence); NOT_RUN if the visible check is NOT_RUN, samples < 1, any sample failed or any utilization reading is unreadable
    ([N/A]); PASS with no samples when 0 GPUs were requested and none is visible.
    """
    vis = check_visible(expect_gpus, timeout_s, runner, nvsmi=nvsmi)
    out = {"visible": vis, "positive_control_ok": vis["positive_control_ok"],
           "gpu_count": vis["visible_gpus"] or 0, "expected_gpus": expect_gpus,
           "expected_samples": samples, "samples": [], "verdict": None,
           "reasons": list(vis["reasons"]), "checked_samples": 0, "unreadable_util": 0}
    if vis["status"] == "NOT_RUN" or expect_gpus == 0:
        out["verdict"] = vis["status"]
        if expect_gpus == 0 and vis["status"] == "PASS":
            out["reasons"].append("0 GPUs requested and none visible: no utilization to sample")
        return out
    if samples < 1:
        out["verdict"] = "NOT_RUN"
        out["reasons"].append("samples < 1")
        return out
    failed = 0
    for i in range(samples):
        if i:
            sleep(every_s)
        rc, rows = sample(timeout_s, runner, nvsmi=nvsmi)
        out["samples"].append({"t_mono_ns": time.monotonic_ns(), "query_rc": rc, "gpus": rows})
        if rc != 0:
            failed += 1
    rows = [r for s in out["samples"] for r in s["gpus"]]
    busy = [r for r in rows if (r["util_pct"] is not None and r["util_pct"] > 0)
            or r["compute_pids"]]
    unreadable = [r for r in rows if r["util_pct"] is None and not r["compute_pids"]]
    out["unreadable_util"] = len(unreadable)
    out["checked_samples"] = samples - failed
    if vis["status"] == "RED" or busy:
        out["verdict"] = "RED"
        if busy:
            out["reasons"].append("%d GPU readings busy (util > 0 or compute processes)"
                                  % len(busy))
    elif failed or unreadable:
        out["verdict"] = "NOT_RUN"
        if failed:
            out["reasons"].append("%d of %d samples failed" % (failed, samples))
        if unreadable:
            out["reasons"].append("%d utilization readings unreadable ([N/A]); never counted "
                                  "as idle" % len(unreadable))
    else:
        out["verdict"] = "PASS"
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
    c.add_argument("--expect-gpus", type=int, required=True, help="GPUs requested from Slurm")
    c.add_argument("--nvidia-smi", required=True, help="nvidia-smi executable to run")
    args = ap.parse_args(argv)
    try:
        ident = Identity(experiment=args.experiment, arm=args.arm, launch_seq=args.launch_seq,
                         job_id=args.job_id, restart_count=args.restart_count,
                         register_sha=args.register_sha, role=args.role, worker=args.worker,
                         pid=os.getpid())
        payload = idle_check(args.samples, args.every_s, args.timeout_s, args.expect_gpus,
                             nvsmi=args.nvidia_smi)
        write_json_stamped(args.out, GPU_IDLE_CHECK_SCHEMA, ident, payload)
    except ConfigError as e:
        print("capenv.gpu: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    print(json.dumps({"verdict": payload["verdict"], "reasons": payload["reasons"]}))
    return exitcodes.OK if payload["verdict"] == "PASS" else exitcodes.ENV_INVALID


if __name__ == "__main__":
    sys.exit(main())
