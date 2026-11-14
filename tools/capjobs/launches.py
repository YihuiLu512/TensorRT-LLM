# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the launches of one job inside the container, then copy back, verify, lock raw/ and append
the job's STATE line to each launch tree, after the launcher's own (INTERFACES.md 5.1, 5.7, 5.9).

    python -m capjobs.launches --plan <snap>/launch_plan.json --overlay-manifest <snap>/... \
        --nvidia-smi PATH --hw-profile <job dir>/env/hw_profile.json --snap S --job-dir JD \
        --local-root L --shm-root SH --run-root R --job-id J --restart-count N --python PY \
        --deadline-epoch E --out <job dir>/launches.json

Per launch: skip (NOT_RUN, recorded) when the remaining allocation time is below
budget_s + kill_grace_s + margin_s + launchplan.in_job_bound_s (pin table + job inputs), or an
earlier launch ended ENV_INVALID or left processes that survived SIGKILL; generate its pin table
on this node (pin_table(); a failure, or a scenario that cannot be read, skips the launch as
NOT_RUN, marked pin_table_failed: main then exits 3 as for config_refused); generate the job
inputs its scenario declares (capjobs.jobinputs into <job dir>/job_inputs/<tree>/; a failure
skips the launch as NOT_RUN, marked job_input_failed, exit 3 likewise); the job's SIGTERM while
either helper runs kills it (run_helper) and skips the launch as NOT_RUN, neither flag set;
otherwise run the plan's argv with
--overlay-manifest, --nvidia-smi, --job-env-dir <job dir>/env, --pin-table, one --job-input per
generated kind and the identity arguments appended, under a budget_s timeout; kill
whatever the launch left behind; copy <L>/runs/<tree> to <R>/<tree>; verify; chmod a-w raw/;
append STATE. Status of that STATE line:
  copy-back not verified -> VOID; a process survived SIGKILL -> VOID; budget exceeded (killed) ->
  VOID;
  stopped because the job itself got SIGTERM -> NOT_RUN (as the orchestrator records its own
  signal stop); each of these only makes the launcher's own STATE worse, never better
  (capcommon.runtree.worst_state: VALID < NOT_RUN < VOID < ENV_INVALID);
  otherwise the launcher's own last STATE line in the tree (by "caplaunch") when it wrote one and
  its exit code agrees (0 or 5 for VALID) -> that status; disagreement -> VOID;
  no launcher STATE: a control/stop.json reason in NOT_RUN_STOPS with exit 5 -> NOT_RUN;
  a non-zero exit with no raw/launch.json (caplaunch stopped before any role could start: a
  ConfigError 3, an argparse usage error 2, an uncaught exception 1, ...) -> NOT_RUN, with the
  exit code and the last line of the launcher's stderr as the reason, marked config_refused (main
  then exits 3 unless a copy-back failure or ENV_INVALID decides the exit code); exit 0 -> VALID;
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
import stat
import subprocess
import sys
import time

from capcommon.clock import wall_iso
from capcommon.exitcodes import CONFIG, ENV_INVALID, OK, RUNTIME, STOPPED_EARLY, CapkitError
from capcommon.identity import Identity
from capcommon.jsonio import write_json_stamped
from capcommon.runtree import append_state, read_state, tree_dir_name, worst_state

from . import copyback, jobinputs, launchplan

SCHEMA = "capkit.launches/1"
POLL_S = 0.5
TERM_GRACE_S = 10  # after the job itself got SIGTERM (Slurm's KillWait is short)
_TERMINATED = []  # set by the SIGTERM handler of main(); polled, never raised into copy-back
PR_SET_CHILD_SUBREAPER = 36
LAUNCHER_BY = "caplaunch"


def private_mkdir(path, exist_ok=False):
    """Create `path` as this user's directory with mode exactly 0700 whatever the umask (caplaunch
    refuses a group/other-writable --run-dir): mkdir, chmod, then read the mode back. An existing
    path (exist_ok) must be this user's real directory; it is chmod-ed the same way."""
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        if not exist_ok:
            raise
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid():
        raise OSError("%s is not a real directory owned by this user" % path)
    os.chmod(path, 0o700)
    mode = stat.S_IMODE(os.lstat(path).st_mode)
    if mode != 0o700:
        raise OSError("%s has mode %o after chmod 0700" % (path, mode))
    return path


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
                job_env_dir, pin_table_path, job_inputs=None):
    argv = [launchplan.substitute(x, values) for x in ln["argv"]]
    return argv + ["--overlay-manifest", overlay_manifest, "--nvidia-smi", nvidia_smi,
                   "--job-env-dir", job_env_dir, "--pin-table", pin_table_path] \
        + jobinputs.caplaunch_args(job_inputs or {}) + [
                   "--experiment", plan["experiment"], "--arm", ln["arm"],
                   "--launch-seq", str(ln["launch_seq"]), "--job-id", str(job_id),
                   "--restart-count", str(restart_count), "--register-sha", plan["register_sha"],
                   "--role", "launch", "--run-dir", run_dir]


class JobTerminated(Exception):
    """This job got SIGTERM while an in-job helper command (pin table, job input generator)
    ran; the helper's process group has been stopped."""


def _kill_group(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()


def run_helper(argv, env=None, stdout=None, stderr=None, stdin=None, timeout=None,
               sleep=time.sleep, clock=time.monotonic):
    """subprocess.run for the in-job helper commands (pin table, job input generators), polled
    so that the job's SIGTERM is honoured while they run (within POLL_S): the helper gets its
    own session; on the timeout or this process's SIGTERM its whole process group is killed
    (SIGKILL, as subprocess.run kills on its timeout, so the in-job time bound is unchanged).
    Returns CompletedProcess; raises subprocess.TimeoutExpired, or JobTerminated on SIGTERM."""
    p = subprocess.Popen(argv, env=env, stdout=stdout, stderr=stderr, stdin=stdin,
                         start_new_session=True)
    end = None if timeout is None else clock() + timeout
    while True:
        rc = p.poll()
        if rc is not None:
            return subprocess.CompletedProcess(argv, rc)
        if _TERMINATED or (end is not None and clock() >= end):
            _kill_group(p)
            if _TERMINATED:
                raise JobTerminated("the job received SIGTERM")
            raise subprocess.TimeoutExpired(argv, timeout)
        sleep(POLL_S if end is None else min(POLL_S, max(0.0, end - clock())))


def pin_table(a, plan, ln, env, out_path):
    """Generate one launch's pin table on this node with capenv.pins generate (INTERFACES.md
    5.5.1; plan RESOURCES 5 item 5): demand = the launch scenario's roles
    [{pin_role, cpus_needed, numa}], --source-sha = profile_sha256 of this job's hardware profile
    (--hw-profile), CPUs outside this process's affinity avoided, identity = this launch with role
    "pins". Returns None, or why no table was written; raises JobTerminated (run_helper)."""
    try:
        roles = launchplan.load_scenario(ln)["roles"]
        demand = [{"role": r["pin_role"], "cpus_needed": r["cpus_needed"], "numa": r["numa"]}
                  for r in roles]
        with open(a.hw_profile, "r", encoding="utf-8") as f:
            hw_sha = json.load(f)["profile_sha256"]
    except (CapkitError, OSError, ValueError, KeyError, TypeError) as e:
        return "pin table: %s: %s" % (type(e).__name__, e)
    dpath = out_path + ".demand.json"
    try:
        with open(dpath, "w", encoding="utf-8") as f:
            json.dump(demand, f)
    except OSError as e:
        return "pin table: cannot write %s: %s" % (dpath, e)
    argv = [a.python, "-m", "capenv.pins", "generate", "--experiment", plan["experiment"],
            "--arm", ln["arm"], "--launch-seq", str(ln["launch_seq"]), "--job-id", str(a.job_id),
            "--restart-count", str(a.restart_count), "--register-sha", plan["register_sha"],
            "--role", "pins", "--demand", dpath, "--out", out_path, "--source-sha", str(hw_sha),
            "--only-affinity"]
    try:
        with open(out_path + ".log", "wb") as log:
            rc = run_helper(argv, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                            timeout=launchplan.PINS_TIMEOUT_S).returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        return "pin table: %s" % e
    if rc != 0 or not os.path.isfile(out_path):
        return "pin table: capenv.pins generate exit %r (log %s.log)" % (rc, out_path)
    return None


def job_inputs(a, ln, values, env, out_dir):
    """Generate the job inputs of one launch's scenario (capjobs.jobinputs.generate) with this
    launch's --param values and the job's environment, each generator through run_helper.
    Returns ({kind: path}, records, None) or (None, records, why); raises JobTerminated."""
    try:
        doc = launchplan.load_scenario(ln)
        wd, _rel = launchplan.scenario_path(ln)
        params = jobinputs.params_from_argv([launchplan.substitute(x, values)
                                             for x in ln["argv"]])
        paths, recs = jobinputs.generate(doc, os.path.realpath(wd), params, a.python, out_dir,
                                         env, launchplan.JOB_INPUT_TIMEOUT_S,
                                         run=run_helper)
    except jobinputs.JobInputError as e:
        return None, e.records, "job inputs: %s" % e
    except (CapkitError, OSError, ValueError) as e:
        return None, [], "job inputs: %s: %s" % (type(e).__name__, e)
    return paths, recs, None


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
    """(rc, timed_out, terminated, leftovers). timed_out: the launcher overran budget_s;
    terminated: this process got SIGTERM while the launcher ran. Either way: SIGTERM to the
    launcher's process group, SIGKILL after the grace; in every case the launch's remaining
    descendants are killed afterwards."""
    seen = {}
    terminated = False
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        p = subprocess.Popen(argv, env=env, stdout=out, stderr=err, start_new_session=True)
        rc = _wait_tracking(p, seen, budget_s)
        stopped = rc is None
        if stopped:
            if _TERMINATED:
                terminated = True
                grace_s = min(grace_s, TERM_GRACE_S)
            os.killpg(p.pid, signal.SIGTERM)
            rc = _wait_tracking(p, seen, grace_s, stop_on_term=False)
            if rc is None:
                track([p.pid], seen)
                os.killpg(p.pid, signal.SIGKILL)
                rc = p.wait()
    seen.pop(p.pid, None)
    return rc, stopped and not terminated, terminated, kill_leftovers(
        seen, min(grace_s, TERM_GRACE_S) if _TERMINATED else grace_s)


def _on_sigterm(_signum, _frame):
    _TERMINATED.append(time.monotonic())


def early_config_error(tree, err_path):
    """The launcher's last non-empty stderr line when the tree has no raw/launch.json (else
    None): caplaunch starts roles only after writing it."""
    if os.path.exists(os.path.join(tree, "raw", "launch.json")):
        return None
    try:
        with open(err_path, "rb") as f:
            lines = [ln.strip() for ln in f.read().decode("utf-8", "replace").splitlines()]
    except OSError:
        return "(launcher stderr unreadable)"
    lines = [ln for ln in lines if ln]
    return lines[-1][-500:] if lines else "(launcher stderr empty)"


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


def prestart_config_refusal(rc, timed_out, terminated, recorded, early_config_error):
    """caplaunch exited non-zero on its own (not killed for its budget or the job's SIGTERM),
    the tree has no raw/launch.json (early_config_error is then the launcher's last stderr line,
    else None) and no launcher STATE line: it stopped before starting any role (roles start only
    after launch.json is written), whatever the exit code (3 ConfigError, 2 argparse usage
    error, 1 uncaught exception, ...)."""
    return (not timed_out and not terminated and recorded is None and rc != OK
            and early_config_error is not None)


def classify(rc, timed_out, stop, recorded=None, terminated=False, early_config_error=None):
    """The launcher's recorded STATE is the more specific record and wins unless the launch was
    killed for overrunning its budget or the exit code contradicts it. A launch stopped because
    the job got SIGTERM did not finish measuring: NOT_RUN. A budget kill or SIGTERM only makes the
    launcher's own record worse (runtree.worst_state), never better. early_config_error (the
    launcher's last stderr line, given only when the tree has no raw/launch.json): any non-zero
    exit there means caplaunch stopped before starting any role (roles start after launch.json
    is written), so nothing was measured: NOT_RUN (prestart_config_refusal)."""
    if timed_out or terminated:
        mine = "VOID" if timed_out else "NOT_RUN"
        why = ("launcher exceeded its budget and was killed (rc=%r)" % rc if timed_out else
               "terminated: the job received SIGTERM; launcher stopped (rc=%r)" % rc)
        if recorded is None:
            return mine, why
        return worst_state([mine, recorded[1]]), "%s; launcher STATE %s: %s" % (
            why, recorded[1], recorded[3])
    if recorded is not None:
        status, why = recorded[1], recorded[3]
        if status != "VALID" or rc in (OK, STOPPED_EARLY):
            return status, "launcher STATE %s: %s (rc=%r)" % (status, why, rc)
        return "VOID", "launcher STATE VALID but exit %r: %s" % (rc, why)
    if stop in NOT_RUN_STOPS and rc == STOPPED_EARLY:
        return "NOT_RUN", "launcher stopped: %s (rc=%r)" % (stop, rc)
    if prestart_config_refusal(rc, timed_out, terminated, recorded, early_config_error):
        return "NOT_RUN", "launcher exited before starting any role (rc=%r): %s" % (
            rc, early_config_error)
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
    private_mkdir(os.path.join(a.local_root, "runs"), exist_ok=True)
    private_mkdir(os.path.join(a.local_root, "logs"), exist_ok=True)
    pins_dir = private_mkdir(os.path.join(a.job_dir, "pins"), exist_ok=True)
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
        try:
            bound = launchplan.in_job_bound_s(ln)
        except CapkitError as e:  # the scenario is unreadable: the pin table and the job
            # inputs need it too; counted with pin_table_failed (jobs/README.md)
            r.update(ran=False, status="NOT_RUN", reason="skipped: scenario: %s" % e,
                     pin_table_failed=True)
            continue
        if left < ln["budget_s"] + grace + plan["margin_s"] + bound:
            r.update(ran=False, status="NOT_RUN", reason="skipped: %.0f s left < budget %d s + "
                     "kill grace %d s + margin %d s + pin table and job inputs %d s"
                     % (left, ln["budget_s"], grace, plan["margin_s"], bound))
            continue
        pins = os.path.join(pins_dir, name + ".pin_table.json")
        try:
            why = pin_table(a, plan, ln, env, pins)
        except JobTerminated as e:
            halted = str(e)
            r.update(ran=False, status="NOT_RUN", reason="skipped: %s during the pin table" % e)
            continue
        if why is not None:
            r.update(ran=False, status="NOT_RUN", reason="skipped: " + why, pin_table_failed=True)
            continue
        try:
            jpaths, jrecs, why = job_inputs(a, ln, values, env,
                                            os.path.join(a.job_dir, "job_inputs", name))
        except JobTerminated as e:
            halted = str(e)
            r.update(ran=False, status="NOT_RUN", reason="skipped: %s during the job inputs" % e)
            continue
        if jrecs:
            r["job_inputs"] = [{k: x.get(k) for k in ("kind", "rc", "out_sha256", "log")}
                               for x in jrecs]
        if why is not None:
            r.update(ran=False, status="NOT_RUN", reason="skipped: " + why, job_input_failed=True)
            continue
        local = os.path.join(a.local_root, "runs", name)
        private_mkdir(local)
        argv = launch_argv(plan, ln, values, a.job_id, a.restart_count, local, a.overlay_manifest,
                           a.nvidia_smi, os.path.join(a.job_dir, "env"), pins, jpaths)
        r["argv"] = argv
        r["started_wall"] = wall_iso()
        logs = os.path.join(a.local_root, "logs", name)
        rc, timed_out, terminated, leftovers = run_with_budget(
            argv, env, ln["budget_s"], grace, logs + ".launcher.out", logs + ".launcher.err")
        recorded = launcher_state(local)
        early = early_config_error(local, logs + ".launcher.err")
        r.update(ran=True, rc=rc, timed_out=timed_out, terminated=terminated,
                 ended_wall=wall_iso(), leftovers=leftovers,
                 config_refused=prestart_config_refusal(rc, timed_out, terminated, recorded,
                                                        early))
        status, reason = classify(rc, timed_out, stop_reason(local), recorded, terminated, early)
        survived = [x["pid"] for x in leftovers if x["survived_sigkill"]]
        if leftovers:
            reason = "%s; %d leftover process(es) killed after the launcher exited" % (
                reason, len(leftovers))
        if survived:
            status = worst_state([status, "VOID"])
            reason = "%s; pids %s survived SIGKILL" % (reason, survived)
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
        elif status == "NOT_RUN" and rec.get("outcome") == "NOT_RUN":
            reason = "%s; nothing to copy back (no file in the launch tree)" % reason
        else:
            status = worst_state([status, "VOID"])
            reason = "copy-back not verified (%s); %s" % (rec["outcome"], reason)
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
    for k in ("--plan", "--overlay-manifest", "--nvidia-smi", "--hw-profile", "--snap", "--job-dir",
              "--local-root", "--shm-root", "--run-root", "--job-id", "--python", "--out"):
        ap.add_argument(k, required=True)
    ap.add_argument("--restart-count", required=True, type=int)
    ap.add_argument("--deadline-epoch", required=True, type=float)
    a = ap.parse_args(argv)
    try:
        plan = launchplan.load(a.plan)
        launchplan.check_kill_grace(plan)
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
        "config_refused": sum(1 for r in results if r.get("config_refused")),
        "pin_table_failed": sum(1 for r in results if r.get("pin_table_failed")),
        "job_input_failed": sum(1 for r in results if r.get("job_input_failed")),
        "statuses": {s: sum(1 for r in results if r.get("status") == s)
                     for s in ("VALID", "VOID", "NOT_RUN", "ENV_INVALID")},
        "launches": results})
    for r in results:
        print("capjobs.launches: L%d %s %s %s" % (r["launch_seq"], r["arm"], r.get("status"),
                                                 r.get("reason")))
    if any(r.get("copyback", {}).get("outcome") not in (None, "PASS")
           and not (r.get("status") == "NOT_RUN" and r["copyback"]["outcome"] == "NOT_RUN")
           for r in results):
        return RUNTIME
    if any(r.get("status") == "ENV_INVALID" for r in results):
        return ENV_INVALID
    if any(r.get("config_refused") or r.get("pin_table_failed") or r.get("job_input_failed")
           for r in results):
        # recorded NOT_RUN (nothing was measured), but the job must not look like a success
        return CONFIG
    return OK


if __name__ == "__main__":
    sys.exit(main())
