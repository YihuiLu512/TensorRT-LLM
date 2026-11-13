# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the launches of one job inside the container, then copy back, verify, lock raw/ and write
the first STATE line of each launch tree (INTERFACES.md 5.1, 5.7, 5.9).

    python -m capjobs.launches --plan <snap>/launch_plan.json --overlay-manifest <snap>/... \
        --nvidia-smi PATH --snap S --job-dir JD --local-root L --shm-root SH --run-root R \
        --job-id J --restart-count N --python PY --deadline-epoch E --out <job dir>/launches.json

Per launch: skip (NOT_RUN, recorded) when the remaining allocation time is below
budget_s + kill_grace_s + margin_s, or an earlier launch ended ENV_INVALID or left processes that
survived SIGKILL; otherwise run the plan's argv with --overlay-manifest, --nvidia-smi,
--job-env-dir <job dir>/env and the identity arguments appended, under a budget_s timeout; kill
whatever the launch left behind; copy <L>/runs/<tree> to <R>/<tree>; verify; chmod a-w raw/;
append STATE. Status of that STATE line:
  copy-back not verified -> VOID; budget exceeded (killed) -> VOID;
  otherwise the launcher's own last STATE line in the tree (by "caplaunch") when it wrote one and
  its exit code agrees (0 or 5 for VALID) -> that status; disagreement -> VOID;
  no launcher STATE: a control/stop.json reason in NOT_RUN_STOPS with exit 5 -> NOT_RUN;
  exit 0 -> VALID;
  exit 6 -> ENV_INVALID; any other exit -> VOID.
Process tree: the launcher's descendants are recorded while it runs (and this process is a child
subreaper when run as main, so orphans re-parent here); after the launcher is gone, every recorded
descendant still alive (same pid and start time) gets SIGTERM, then SIGKILL after kill_grace_s,
and is listed in the launch record as a leftover.
The judge appends its own STATE line later; this one only says the launch ran and its data is safe.
"""

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

from capcommon.clock import wall_iso
from capcommon.exitcodes import ENV_INVALID, OK, RUNTIME, STOPPED_EARLY, CapkitError
from capcommon.identity import Identity
from capcommon.jsonio import write_json_stamped
from capcommon.runtree import append_state, read_state, tree_dir_name

from . import copyback, launchplan

SCHEMA = "capkit.launches/1"
POLL_S = 0.5
TERM_GRACE_S = 10  # after the job itself got SIGTERM (Slurm's KillWait is short)
_TERMINATED = []  # set by the SIGTERM handler of main(); polled, never raised into copy-back
PR_SET_CHILD_SUBREAPER = 36
LAUNCHER_BY = "caplaunch"


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


def launch_argv(plan, ln, values, job_id, restart_count, run_dir, overlay_manifest, nvidia_smi,
                job_env_dir):
    argv = [launchplan.substitute(x, values) for x in ln["argv"]]
    return argv + ["--overlay-manifest", overlay_manifest, "--nvidia-smi", nvidia_smi,
                   "--job-env-dir", job_env_dir,
                   "--experiment", plan["experiment"], "--arm", ln["arm"],
                   "--launch-seq", str(ln["launch_seq"]), "--job-id", str(job_id),
                   "--restart-count", str(restart_count), "--register-sha", plan["register_sha"],
                   "--role", "launch", "--run-dir", run_dir]


def become_subreaper():
    """Orphans of the launcher (its roles run in their own sessions) re-parent to this process
    instead of init, so they stay visible as descendants. False when prctl is unavailable."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def proc_table(proc="/proc"):
    """{pid: (ppid, starttime_ticks, state)} of every readable /proc/<pid>/stat."""
    out = {}
    for n in os.listdir(proc):
        if not n.isdigit():
            continue
        try:
            with open(os.path.join(proc, n, "stat"), "rb") as f:
                text = f.read().decode("utf-8", "replace")
            rest = text[text.rindex(")") + 2:].split()
            out[int(n)] = (int(rest[1]), int(rest[19]), rest[0])
        except (OSError, ValueError, IndexError):
            continue
    return out


def descendants(root, table):
    kids = {}
    for pid, (ppid, _st, _s) in table.items():
        kids.setdefault(ppid, []).append(pid)
    out, stack = [], [root]
    while stack:
        for c in kids.get(stack.pop(), []):
            out.append(c)
            stack.append(c)
    return out


def track(roots, seen, proc="/proc"):
    """Add every live descendant of roots to seen {pid: starttime}."""
    table = proc_table(proc)
    for r in roots:
        for pid in descendants(r, table):
            if table[pid][2] != "Z":
                seen.setdefault(pid, table[pid][1])


def _live(pids, proc="/proc"):
    """The subset of {pid: starttime} still running as the same process (not a zombie)."""
    table = proc_table(proc)
    return {pid: st for pid, st in pids.items()
            if pid in table and table[pid][1] == st and table[pid][2] != "Z"}


def _reap(pids):
    for pid in pids:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def _wait_tracking(p, seen, timeout_s, stop_on_term=True, sleep=time.sleep, clock=time.monotonic):
    """Wait for p up to timeout_s (or until this process got SIGTERM, if stop_on_term) while
    recording its descendants. Returns rc or None."""
    end = clock() + timeout_s
    while True:
        track([p.pid, os.getpid()], seen)
        rc = p.poll()
        if rc is not None:
            return rc
        if stop_on_term and _TERMINATED:
            return None
        left = end - clock()
        if left <= 0:
            return None
        sleep(min(POLL_S, left))


def kill_leftovers(seen, grace_s, sleep=time.sleep, clock=time.monotonic):
    """SIGTERM every recorded descendant still alive, SIGKILL after grace_s. Returns records."""
    track([os.getpid()], seen)
    _reap(seen)
    live = _live(seen)
    recs = {pid: {"pid": pid, "starttime_ticks": st, "signals": []} for pid, st in live.items()}
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, grace_s)):
        for pid in _live(live):
            try:
                os.kill(pid, sig)
                recs[pid]["signals"].append(sig.name)
            except ProcessLookupError:
                pass
        end = clock() + wait_s
        while live and clock() < end:
            _reap(live)
            live = _live(live)
            if live:
                sleep(min(POLL_S, max(0.0, end - clock())))
        _reap(live)
        live = _live(live)
        if not live:
            break
    for pid, r in recs.items():
        r["survived_sigkill"] = pid in live
    return sorted(recs.values(), key=lambda r: r["pid"])


def run_with_budget(argv, env, budget_s, grace_s, out_path, err_path):
    """(rc, timed_out, leftovers). On timeout, or when this process got SIGTERM: SIGTERM to the
    launcher's process group, SIGKILL after the grace; in every case the launch's remaining
    descendants are killed afterwards."""
    seen = {}
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        p = subprocess.Popen(argv, env=env, stdout=out, stderr=err, start_new_session=True)
        rc = _wait_tracking(p, seen, budget_s)
        timed_out = rc is None
        if timed_out:
            if _TERMINATED:
                grace_s = min(grace_s, TERM_GRACE_S)
            os.killpg(p.pid, signal.SIGTERM)
            rc = _wait_tracking(p, seen, grace_s, stop_on_term=False)
            if rc is None:
                track([p.pid], seen)
                os.killpg(p.pid, signal.SIGKILL)
                rc = p.wait()
    seen.pop(p.pid, None)
    return rc, timed_out, kill_leftovers(seen, min(grace_s, TERM_GRACE_S) if _TERMINATED
                                         else grace_s)


def _on_sigterm(_signum, _frame):
    _TERMINATED.append(time.monotonic())


def stop_reason(tree):
    try:
        with open(os.path.join(tree, "control", "stop.json"), "r", encoding="utf-8") as f:
            return json.load(f).get("reason")
    except (OSError, ValueError, AttributeError):
        return None


def launcher_state(tree):
    """The launcher's own last STATE line (iso, status, by, reason) in the tree, or None."""
    try:
        mine = [ln for ln in read_state(tree) if ln[2] == LAUNCHER_BY]
    except (CapkitError, OSError) as e:
        return ("", "VOID", LAUNCHER_BY, "unreadable launcher STATE: %s" % e)
    return mine[-1] if mine else None


# caplaunch stop reasons (control/stop.json) that mean the launch never measured anything; it
# records them as STATE NOT_RUN with exit 5, so they matter here only when STATE is missing.
NOT_RUN_STOPS = ("not_ready_before_origin", "overlay_manifest_missing", "job_env_dir_missing",
                 "orchestrator_signalled")


def classify(rc, timed_out, stop, recorded=None):
    """The launcher's recorded STATE is the more specific record and wins unless the launch was
    killed for overrunning its budget or the exit code contradicts it."""
    if timed_out:
        return "VOID", "launcher exceeded its budget and was killed (rc=%r)" % rc
    if recorded is not None:
        status, why = recorded[1], recorded[3]
        if status != "VALID" or rc in (OK, STOPPED_EARLY):
            return status, "launcher STATE %s: %s (rc=%r)" % (status, why, rc)
        return "VOID", "launcher STATE VALID but exit %r: %s" % (rc, why)
    if stop in NOT_RUN_STOPS and rc == STOPPED_EARLY:
        return "NOT_RUN", "launcher stopped: %s (rc=%r)" % (stop, rc)
    if rc == OK:
        return "VALID", "launcher exit 0"
    if rc == ENV_INVALID:
        return "ENV_INVALID", "launcher exit 6 (ENV_INVALID)"
    return "VOID", "launcher exit %r" % rc


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
        if _TERMINATED and not halted:
            halted = "the job received SIGTERM"
        if halted:
            r.update(ran=False, status="NOT_RUN", reason="skipped: " + halted)
            continue
        grace = plan["kill_grace_s"]
        if left < ln["budget_s"] + grace + plan["margin_s"]:
            r.update(ran=False, status="NOT_RUN", reason="skipped: %.0f s left < budget %d s + "
                     "kill grace %d s + margin %d s" % (left, ln["budget_s"], grace,
                                                        plan["margin_s"]))
            continue
        local = os.path.join(a.local_root, "runs", name)
        os.mkdir(local)
        argv = launch_argv(plan, ln, values, a.job_id, a.restart_count, local, a.overlay_manifest,
                           a.nvidia_smi, os.path.join(a.job_dir, "env"))
        r["argv"] = argv
        r["started_wall"] = wall_iso()
        logs = os.path.join(a.local_root, "logs", name)
        rc, timed_out, leftovers = run_with_budget(argv, env, ln["budget_s"], grace,
                                                   logs + ".launcher.out",
                                                   logs + ".launcher.err")
        r.update(ran=True, rc=rc, timed_out=timed_out, ended_wall=wall_iso(),
                 leftovers=leftovers)
        status, reason = classify(rc, timed_out, stop_reason(local), launcher_state(local))
        if timed_out and _TERMINATED:
            r["terminated"] = True
            reason = "the job received SIGTERM; launcher stopped (rc=%r)" % rc
        survived = [x["pid"] for x in leftovers if x["survived_sigkill"]]
        if leftovers:
            reason = "%s; %d leftover process(es) killed after the launcher exited" % (
                reason, len(leftovers))
        if survived:
            status, reason = "VOID", "%s; pids %s survived SIGKILL" % (reason, survived)
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
        elif survived:
            halted = "launch %d left processes that survived SIGKILL" % ln["launch_seq"]
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capjobs.launches")
    for k in ("--plan", "--overlay-manifest", "--nvidia-smi", "--snap", "--job-dir", "--local-root",
              "--shm-root", "--run-root", "--job-id", "--python", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--restart-count", required=True, type=int)
    ap.add_argument("--deadline-epoch", required=True, type=float)
    a = ap.parse_args(argv)
    try:
        plan = launchplan.load(a.plan)
        launchplan.check_kill_grace(plan, a.snap)
        with open(a.overlay_manifest, "r", encoding="utf-8") as f:
            manifest_doc = json.load(f)
        ident = Identity(experiment=plan["experiment"], arm=plan["job_arm"], launch_seq=0,
                         job_id=a.job_id, restart_count=a.restart_count,
                         register_sha=plan["register_sha"], role="jobs").with_pid()
        subreaper = become_subreaper()
        signal.signal(signal.SIGTERM, _on_sigterm)
        results = run_all(a, plan, manifest_doc)
    except (CapkitError, OSError, ValueError, KeyError) as e:
        print("capjobs.launches: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return RUNTIME
    planned = len(plan["launches"])
    write_json_stamped(a.out, SCHEMA, ident, {
        "planned": planned, "ran": sum(1 for r in results if r.get("ran")),
        "child_subreaper": subreaper,
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
