# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the launches of one job inside the container, then copy back, verify, lock raw/ and write
the first STATE line of each launch tree (INTERFACES.md 5.1, 5.7, 5.9).

    python -m capjobs.launches --plan <snap>/launch_plan.json --overlay-manifest <snap>/... \
        --snap S --job-dir JD --local-root L --shm-root SH --run-root R --job-id J \
        --restart-count N --python PY --deadline-epoch E --out <job dir>/launches.json

Per launch: skip (NOT_RUN, recorded) when the remaining allocation time is below
budget_s + margin_s or an earlier launch ended ENV_INVALID; otherwise run the plan's argv with the
identity arguments appended, under a budget_s timeout; copy <L>/runs/<tree> to <R>/<tree>; verify;
chmod a-w raw/; append STATE. Status of the first STATE line:
  copy-back not verified -> VOID; control/stop.json reason not_ready_before_origin -> NOT_RUN;
  launcher exit 0 -> VALID; exit 6 -> ENV_INVALID; timeout or any other exit -> VOID.
The judge appends its own STATE line later; this one only says the launch ran and its data is safe.
"""

import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from capcommon.clock import wall_iso
from capcommon.exitcodes import ENV_INVALID, OK, RUNTIME, CapkitError
from capcommon.identity import Identity
from capcommon.jsonio import write_json_stamped
from capcommon.runtree import append_state, launch_shm_subdir, tree_dir_name

from . import copyback, launchplan

SCHEMA = "capkit.launches/1"
KILL_GRACE_S = 30


def probe_env(manifest_doc, base_env):
    """Version-report variables the inner script must set (INTERFACES.md 2.5)."""
    env = dict(base_env)
    tier = manifest_doc.get("tier")
    env["CAPKIT_CODE_SHA"] = manifest_doc["capkit_code_sha"]
    env["CAPPROBE_TIER"] = tier
    env["CAPPROBE_GROUPS"] = ",".join(manifest_doc.get("groups") or [])
    if tier in ("P1", "P2"):
        env["CAPPROBE_PATCH_SHA"] = manifest_doc["target_commit"]
    else:
        env.pop("CAPPROBE_PATCH_SHA", None)
    return env


def launch_argv(plan, ln, values, job_id, restart_count, run_dir):
    argv = [launchplan.substitute(x, values) for x in ln["argv"]]
    return argv + ["--experiment", plan["experiment"], "--arm", ln["arm"],
                   "--launch-seq", str(ln["launch_seq"]), "--job-id", str(job_id),
                   "--restart-count", str(restart_count), "--register-sha", plan["register_sha"],
                   "--role", "launch", "--run-dir", run_dir]


def run_with_budget(argv, env, budget_s, out_path, err_path):
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        p = subprocess.Popen(argv, env=env, stdout=out, stderr=err, start_new_session=True)
        try:
            return p.wait(timeout=budget_s), False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                return p.wait(timeout=KILL_GRACE_S), True
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                return p.wait(), True


def stop_reason(tree):
    try:
        with open(os.path.join(tree, "control", "stop.json"), "r", encoding="utf-8") as f:
            return json.load(f).get("reason")
    except (OSError, ValueError, AttributeError):
        return None


def classify(rc, timed_out, stop):
    if stop == "not_ready_before_origin":
        return "NOT_RUN", "launcher stopped: not_ready_before_origin (rc=%r)" % rc
    if timed_out:
        return "VOID", "launcher exceeded its budget and was killed (rc=%r)" % rc
    if rc == OK:
        return "VALID", "launcher exit 0"
    if rc == ENV_INVALID:
        return "ENV_INVALID", "launcher exit 6 (ENV_INVALID)"
    return "VOID", "launcher exit %r" % rc


def copy_probe_ts(shm_root, restart_count, launch_seq, tree):
    """INTERFACES.md 6.5: ts rings land in <shm_dir>/<pid>.ts.capbin; copy them into the tree."""
    src_dir = launch_shm_subdir(os.path.join(shm_root, "snap"), restart_count, launch_seq)
    files = sorted(glob.glob(os.path.join(src_dir, "*.ts.capbin")))
    if files:
        dst = os.path.join(tree, "raw", "probe_ts")
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copyfile(f, os.path.join(dst, os.path.basename(f)))
    return len(files)


def run_all(a, plan, manifest_doc, now_fn=time.time):
    results = []
    env = probe_env(manifest_doc, os.environ)
    values = {"PY": a.python, "SNAP": a.snap, "JOB_DIR": a.job_dir, "SHM_ROOT": a.shm_root,
              "LOCAL_ROOT": a.local_root, "JOB_ID": a.job_id, "RESTART_COUNT": a.restart_count}
    os.makedirs(os.path.join(a.local_root, "runs"), exist_ok=True)
    os.makedirs(os.path.join(a.local_root, "logs"), exist_ok=True)
    halted = None
    for ln in plan["launches"]:
        name = tree_dir_name(a.job_id, a.restart_count, ln["launch_seq"], ln["arm"])
        r = {"launch_seq": ln["launch_seq"], "arm": ln["arm"], "tree": name}
        results.append(r)
        left = a.deadline_epoch - now_fn()
        if halted:
            r.update(ran=False, status="NOT_RUN", reason="skipped: " + halted)
            continue
        if left < ln["budget_s"] + plan["margin_s"]:
            r.update(ran=False, status="NOT_RUN", reason="skipped: %.0f s left < budget %d s + "
                     "margin %d s" % (left, ln["budget_s"], plan["margin_s"]))
            continue
        local = os.path.join(a.local_root, "runs", name)
        os.mkdir(local)
        argv = launch_argv(plan, ln, values, a.job_id, a.restart_count, local)
        r["argv"] = argv
        r["started_wall"] = wall_iso()
        logs = os.path.join(a.local_root, "logs", name)
        rc, timed_out = run_with_budget(argv, env, ln["budget_s"], logs + ".launcher.out",
                                        logs + ".launcher.err")
        r.update(ran=True, rc=rc, timed_out=timed_out, ended_wall=wall_iso())
        r["probe_ts_files"] = copy_probe_ts(a.shm_root, a.restart_count, ln["launch_seq"], local)
        status, reason = classify(rc, timed_out, stop_reason(local))
        dst = os.path.join(a.run_root, name)
        try:
            rec = copyback.copy_tree(local, dst)
        except OSError as e:
            rec = {"src": local, "dst": dst, "passed": False, "outcome": "FAIL", "error": str(e)}
        copyback.write_record(os.path.join(a.job_dir, "copyback", name + ".verify.json"), rec)
        r["copyback"] = {k: rec.get(k) for k in ("outcome", "files_expected", "bytes_expected")}
        if rec["passed"]:
            copyback.lock_raw(dst)
            reason = "%s; copy-back verified %d files %d bytes" % (
                reason, rec["files_expected"], rec["bytes_expected"])
        else:
            status, reason = "VOID", "copy-back not verified (%s); %s" % (rec["outcome"], reason)
        if os.path.isdir(dst):
            append_state(dst, status, reason, "capjobs.launches")
        r.update(status=status, reason=reason)
        if status == "ENV_INVALID":
            halted = "launch %d ended ENV_INVALID" % ln["launch_seq"]
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capjobs.launches")
    for k in ("--plan", "--overlay-manifest", "--snap", "--job-dir", "--local-root",
              "--shm-root", "--run-root", "--job-id", "--python", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--restart-count", required=True, type=int)
    ap.add_argument("--deadline-epoch", required=True, type=float)
    a = ap.parse_args(argv)
    try:
        plan = launchplan.load(a.plan)
        with open(a.overlay_manifest, "r", encoding="utf-8") as f:
            manifest_doc = json.load(f)
        ident = Identity(experiment=plan["experiment"], arm=plan["job_arm"], launch_seq=0,
                         job_id=a.job_id, restart_count=a.restart_count,
                         register_sha=plan["register_sha"], role="jobs").with_pid()
        results = run_all(a, plan, manifest_doc)
    except (CapkitError, OSError, ValueError, KeyError) as e:
        print("capjobs.launches: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return RUNTIME
    planned = len(plan["launches"])
    write_json_stamped(a.out, SCHEMA, ident, {
        "planned": planned, "ran": sum(1 for r in results if r.get("ran")),
        "statuses": {s: sum(1 for r in results if r.get("status") == s)
                     for s in ("VALID", "VOID", "NOT_RUN", "ENV_INVALID")},
        "launches": results})
    for r in results:
        print("capjobs.launches: L%d %s %s %s" % (r["launch_seq"], r["arm"], r.get("status"),
                                                 r.get("reason")))
    if any(r.get("copyback", {}).get("outcome") not in (None, "PASS") for r in results):
        return RUNTIME
    if any(r.get("status") == "ENV_INVALID" for r in results):
        return ENV_INVALID
    return OK


if __name__ == "__main__":
    sys.exit(main())
