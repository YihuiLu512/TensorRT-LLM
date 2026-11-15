# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capenv.spinners: pure-Python spin processes, one per listed core (INTERFACES.md errata SB-B1).

Started as an `aux` role of a caplaunch scenario with measure_class `observer`. One spin process
is forked per core of --cpus and bound to that core with sched_setaffinity. With --target-role
ROLE one more spin process is bound to the first core of ROLE in --pin-table (the launch tree's
raw/env/pin_table.json, passed by the orchestrator); the pin table is only read. That extra
spinner is the C13 positive control: busy time on a measured core that no thread of the measured
role explains.

The started process is a supervisor and never spins: it pins itself to --cpus, forks the
spinners, writes the ready file once every spinner has reported its affinity read-back, and on
the first SIGTERM/SIGINT stops them (SIGTERM, then SIGKILL after --stop-grace-s; a second signal
skips the grace). The spinners are not listed in the ready file's `children`: a registered
process needs a version record for C10, and the orchestrator takes those only from snapshots or
the started process's ready file. They stay unregistered descendants of an observer role, which
the orchestrator lists in procs/monitor_children.json. A spinner leaves its loop on SIGTERM or
once its parent is gone.

Output raw/spinners/<role>.json (capkit.spinners/1): per spinner its core, purpose, affinity
read-back, work units, CPU time and wall time; per spun core the /proc/stat tick deltas from
ready to stop. These are a record of what ran, not a performance measurement.

Exit codes: 0 stopped by a signal with every spinner reported, or stopped before it was ready
with every affinity read-back received so far correct (the record then has ready false; missing
reports and results are not counted); 3 configuration (pin table unreadable or not a pin table,
target role absent or without cpus, target core inside --cpus; caplaunch's scenario validation
checks the target role before the launch, this is the fallback); 4 the supervisor could not pin
itself (no spinner started), a spinner could not be pinned, exited without an affinity report or
sent none before the shared deadline, exited before the stop, or reported no result after it.
Every exit 4 writes the record, with the reasons in `problems`.
"""

import argparse
import hashlib
import json
import os
import select
import signal
import sys
import time

from capcommon import exitcodes
from capcommon.cli import add_common_args, common_from_args
from capcommon.exitcodes import CapkitError, ConfigError
from capcommon.jsonio import write_json_stamped
from capcommon.lifecycle import StopController, pin_self, read_stop_file, write_ready_file
from capcommon.runtree import RunTree
from capcommon.versions import version_record

from . import procfs
from .pins import PIN_SCHEMA

SCHEMA = "capkit.spinners/1"
UNIT_OPS = 2000  # LCG steps per work unit; a spinner checks its stop flag once per unit
ORPHAN_CHECK_UNITS = 64  # units between two getppid() checks
PIN_REPORT_WAIT_S = 10.0  # one deadline for all spinners' affinity reports (forked together)
# After a spinner exit seen while no stop was requested yet: how long to wait for a stop request
# before calling it an early exit (a group-wide SIGTERM reaches the spinners and the supervisor
# together, and the supervisor's handler may run after waitpid saw the spinner go).
STOP_SETTLE_S = 0.5
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT)


def output_path(run_dir, role):
    """raw/spinners/<role>.json of the launch tree."""
    return os.path.join(RunTree(run_dir).raw, "spinners", "%s.json" % role)


def check_pin_table(doc, path):
    """doc itself; ConfigError unless it is a capkit.pin_table/1 document with a roles mapping
    (the same acceptance as the orchestrator's --pin-table)."""
    if not isinstance(doc, dict) or doc.get("schema") != PIN_SCHEMA \
            or not isinstance(doc.get("roles"), dict):
        raise ConfigError("--pin-table %s is not a %s file with roles" % (path, PIN_SCHEMA))
    return doc


def target_core(pin_doc, target_role):
    """First core of target_role in the pin table's roles (list order of its `cpus`)."""
    ent = pin_doc["roles"].get(target_role)
    if not isinstance(ent, dict):
        raise ConfigError("--target-role %s is not a role of the pin table (roles: %s)"
                          % (target_role, sorted(pin_doc["roles"])))
    cpus = ent.get("cpus")
    if not isinstance(cpus, list) or not cpus or any(
            not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in cpus):
        raise ConfigError("pin table role %s has no list of cpus: %r" % (target_role, cpus))
    return cpus[0]


def plan_spinners(cpus, target_role=None, target_cpu=None):
    """[{index, cpu, purpose[, target_role]}]: one `own` spinner per --cpus core, then the
    `target` spinner. A target core inside --cpus would put two spinners on one core."""
    plan = [{"index": i, "cpu": c, "purpose": "own"} for i, c in enumerate(cpus)]
    if target_role is not None:
        if target_cpu in cpus:
            raise ConfigError("target role %s's first core %d is one of --cpus %s"
                              % (target_role, target_cpu, cpus))
        plan.append({"index": len(plan), "cpu": target_cpu, "purpose": "target",
                     "target_role": target_role})
    return plan


def spin(stop, parent_pid, unit_ops=UNIT_OPS, check_every=ORPHAN_CHECK_UNITS):
    """Spin until stop[0] is true or the parent is gone. Pure Python, one thread."""
    t0, c0 = time.monotonic_ns(), time.process_time_ns()
    x, units, orphaned = 1, 0, False
    while not stop[0]:
        for _ in range(unit_ops):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        units += 1
        if units % check_every == 0 and os.getppid() != parent_pid:
            orphaned = True
            break
    return {"units": units, "cpu_ns": time.process_time_ns() - c0,
            "wall_ns": time.monotonic_ns() - t0, "orphaned": orphaned, "lcg_state": x}


def _send(fd, obj):
    os.write(fd, (json.dumps(obj, sort_keys=True) + "\n").encode())


def _last_cpu():
    try:
        return procfs.read_stat("/proc/self/stat")["processor"]
    except (procfs.ProcGone, OSError, ValueError, IndexError):
        return None


def _child(cpu, wfd, parent_pid, close_fds):
    """Body of a forked spinner; never returns. Stop signals are blocked on entry. close_fds:
    the inherited read ends (its own pipe's and the earlier spinners'), closed first."""
    rc = exitcodes.INTERNAL
    try:
        for fd in close_fds:
            os.close(fd)
        stop = [False]

        def on_signal(_signum, _frame):
            stop[0] = True
        for s in STOP_SIGNALS:
            signal.signal(s, on_signal)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
        try:
            os.sched_setaffinity(0, {cpu})
            aff = sorted(os.sched_getaffinity(0))
        except OSError as e:
            _send(wfd, {"event": "pinned", "ok": False, "error": str(e)})
            rc = exitcodes.RUNTIME
        else:
            _send(wfd, {"event": "pinned", "ok": aff == [cpu], "affinity": aff})
            if aff != [cpu]:
                rc = exitcodes.RUNTIME
            else:
                res = spin(stop, parent_pid)
                res["last_cpu"] = _last_cpu()
                res["event"] = "result"
                _send(wfd, res)
                rc = exitcodes.OK
    finally:
        os._exit(rc)


# ---- supervisor -----------------------------------------------------------------------------

def start_spinners(plan):
    """Fork one spinner per plan entry (stop signals blocked across the fork, so a signal that
    arrives before the child installed its own handlers stays pending for it)."""
    parent = os.getpid()
    for k, sp in enumerate(plan):
        rfd, wfd = os.pipe()
        inherited = [rfd] + [p["rfd"] for p in plan[:k]]  # read ends the child must not hold
        old = signal.pthread_sigmask(signal.SIG_BLOCK, STOP_SIGNALS)
        try:
            pid = os.fork()
            if pid == 0:
                _child(sp["cpu"], wfd, parent, inherited)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)
        os.close(wfd)
        sp.update(pid=pid, rfd=rfd, buf=b"", events=[], eof=False, exit=None, killed=False)


def _read_some(sp, timeout_s):
    """Read what the spinner's pipe has within timeout_s and parse complete lines into
    sp["events"]. False once the pipe is at EOF."""
    if sp["eof"]:
        return False
    ready, _, _ = select.select([sp["rfd"]], [], [], max(0.0, timeout_s))
    if not ready:
        return True
    data = os.read(sp["rfd"], 65536)
    if not data:
        sp["eof"] = True
        os.close(sp["rfd"])
        return False
    sp["buf"] += data
    while b"\n" in sp["buf"]:
        line, sp["buf"] = sp["buf"].split(b"\n", 1)
        try:
            sp["events"].append(json.loads(line))
        except ValueError:
            sp["events"].append({"event": "unparsable",
                                 "raw": line[:200].decode("utf-8", "replace")})
    return True


def _event(sp, name):
    for ev in sp["events"]:
        if ev.get("event") == name:
            return ev
    return None


def wait_pinned(plan, stop, wait_s=PIN_REPORT_WAIT_S):
    """Problems (strings) for spinners that did not report a correct affinity read-back within
    wait_s, one deadline for all of them. A spinner whose pipe closed without a report is told
    apart from one still silent at the deadline. A stop request ends the wait without counting
    the missing reports."""
    problems = []
    deadline = time.monotonic() + wait_s
    for sp in plan:
        while _event(sp, "pinned") is None and not stop.requested:
            left = deadline - time.monotonic()
            if left <= 0 or not _read_some(sp, min(left, 0.2)):
                break
        ev = _event(sp, "pinned")
        where = "spinner %d (%s, cpu %d)" % (sp["index"], sp["purpose"], sp["cpu"])
        if ev is None and stop.requested:
            continue
        if ev is None and sp["eof"]:
            _reap_soon(sp)
            problems.append("%s: pipe closed before an affinity report (exit %s)"
                            % (where, sp["exit"]))
        elif ev is None:
            problems.append("%s: no affinity report within %.1fs" % (where, wait_s))
        elif not ev.get("ok"):
            problems.append("%s: pinning failed: %s" % (
                where, ev.get("error") or "affinity read-back %s" % (ev.get("affinity"),)))
    return problems


def poll_exits(plan):
    for sp in plan:
        if sp["exit"] is not None:
            continue
        try:
            pid, status = os.waitpid(sp["pid"], os.WNOHANG)
        except ChildProcessError:
            sp["exit"] = {"code": None, "why": "not a child any more"}
            continue
        if pid:
            sp["exit"] = {"code": os.waitstatus_to_exitcode(status)}


def _reap_soon(sp, wait_s=1.0):
    """poll_exits for one spinner until it has an exit status or wait_s passed (its pipe is at
    EOF, so it is exiting)."""
    end = time.monotonic() + wait_s
    poll_exits([sp])
    while sp["exit"] is None and time.monotonic() < end:
        time.sleep(0.01)
        poll_exits([sp])


def watch(plan, stop, settle_s=STOP_SETTLE_S):
    """Wait for the stop request. Problems (strings) for spinners that exited before it; an exit
    followed within settle_s by the stop request is the stop itself, not a problem."""
    while not stop.requested:
        stop.wait(0.2)
        poll_exits(plan)
        if stop.requested or all(sp["exit"] is None for sp in plan):
            continue
        if stop.wait(settle_s):  # the same stop signal reached the spinners first
            break
        return ["spinner %d (%s, cpu %d) exited before the stop: %s"
                % (sp["index"], sp["purpose"], sp["cpu"], sp["exit"])
                for sp in plan if sp["exit"] is not None]
    return []


def _kill(sp, sig):
    try:
        os.kill(sp["pid"], sig)
    except ProcessLookupError:
        pass


def stop_spinners(plan, stop, grace_s):
    """SIGTERM every live spinner (SIGKILL when the stop is already hard); SIGKILL whatever is
    left after grace_s or as soon as a second stop signal arrives; reap all; read the pipes to
    EOF. sp["killed"]: False, "second_signal" or "grace_expired"."""
    poll_exits(plan)
    hard = stop.hard
    for sp in plan:
        if sp["exit"] is None:
            _kill(sp, signal.SIGKILL if hard else signal.SIGTERM)
            sp["killed"] = "second_signal" if hard else False
    deadline = time.monotonic() + grace_s
    while True:
        for sp in plan:
            _read_some(sp, 0)
        poll_exits(plan)
        if all(sp["exit"] is not None for sp in plan):
            break
        if not hard and (stop.hard or time.monotonic() >= deadline):
            hard = True
            why = "second_signal" if stop.hard else "grace_expired"
            for sp in plan:
                if sp["exit"] is None:
                    _kill(sp, signal.SIGKILL)
                    sp["killed"] = why
        time.sleep(0.01)
    for sp in plan:
        while _read_some(sp, 1.0):
            pass


def cpu_ticks(cpus):
    """({cpu: /proc/stat row} for cpus, None) or (None, reason)."""
    try:
        rows = procfs.read_proc_stat_cpus()
    except (OSError, ValueError) as e:
        return None, "/proc/stat: %s" % e
    return {c: rows.get(c) for c in cpus}, None


def core_deltas(t0, t1):
    """{"<cpu>": tick deltas, busy ticks (procfs.BUSY_COLS), total ticks, busy_frac} from two
    cpu_ticks readings; a cpu missing from either reading is recorded as unreadable."""
    out = {}
    for c in sorted(t0):
        a, b = t0.get(c), t1.get(c)
        if a is None or b is None:
            out[str(c)] = {"unreadable": "cpu%d not in /proc/stat" % c}
            continue
        d = {k: b[k] - a[k] for k in procfs.STAT_CPU_COLS}
        total = sum(d.values())
        busy = procfs.busy_ticks(d)
        out[str(c)] = {"ticks": d, "busy_ticks": busy, "total_ticks": total,
                       "busy_frac": busy / total if total > 0 else None}
    return out


def spinner_record(sp, ready):
    """One output row per spinner; problems (strings) for missing results once the tool was
    ready, except for spinners a second stop signal killed (INTERFACES 2.3: write what exists
    and exit)."""
    pinned, res = _event(sp, "pinned"), _event(sp, "result")
    rec = {k: sp.get(k) for k in ("index", "cpu", "purpose", "target_role", "pid", "exit",
                                  "killed")}
    rec["affinity"] = None if pinned is None else pinned.get("affinity")
    rec["pinned_ok"] = None if pinned is None else bool(pinned.get("ok"))
    rec["reported"] = res is not None
    for k in ("units", "cpu_ns", "wall_ns", "orphaned", "last_cpu"):
        rec[k] = None if res is None else res.get(k)
    rec["cpu_frac"] = None if res is None or not res.get("wall_ns") else \
        res["cpu_ns"] / res["wall_ns"]
    problems = []
    if ready and res is None and sp.get("killed") != "second_signal":
        problems.append("spinner %d (%s, cpu %d) did not report (exit %s)"
                        % (sp["index"], sp["purpose"], sp["cpu"], sp["exit"]))
    return rec, problems


def build_parser():
    ap = argparse.ArgumentParser(prog="capenv.spinners", allow_abbrev=False,
                                 description=__doc__.split("\n\n")[0])
    add_common_args(ap, ready=True, cpus=True, inject=False, stop=True)
    g = ap.add_argument_group("spinners (INTERFACES.md errata SB-B1)")
    g.add_argument("--origin-ns", type=int, required=True)
    g.add_argument("--pin-table", required=True,
                   help="this launch's raw/env/pin_table.json (passed by the orchestrator)")
    g.add_argument("--target-role", default=None,
                   help="also bind one spinner to the first core of this pin table role")
    return ap


def _read_pin_table(path):
    """(document, sha256 of the bytes read) of --pin-table; ConfigError when unusable."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise ConfigError("--pin-table %s: %s" % (path, e)) from e
    try:
        doc = json.loads(data)
    except ValueError as e:
        raise ConfigError("--pin-table %s: %s" % (path, e)) from e
    return check_pin_table(doc, path), hashlib.sha256(data).hexdigest()


def main(argv=None):
    args = build_parser().parse_args(argv)
    stop = StopController()
    try:
        common = common_from_args(args)
        pin, pin_sha = _read_pin_table(args.pin_table)
        tcpu = None if args.target_role is None else target_core(pin, args.target_role)
        plan = plan_spinners(common.cpus, args.target_role, tcpu)
    except ConfigError as e:
        print("capenv.spinners: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    origin = args.origin_ns
    problems, ready, t0, t1, cores_why = [], False, None, None, None
    try:
        affinity = pin_self(common.cpus)
    except (OSError, CapkitError) as e:
        problems.append("cannot pin the supervisor to %s: %s" % (common.cpus, e))
        print("capenv.spinners: %s" % problems[-1], file=sys.stderr)
        affinity, plan = None, []
    try:
        if plan:
            start_spinners(plan)
            problems += wait_pinned(plan, stop)
        if plan and not problems and not stop.requested:
            t0, cores_why = cpu_ticks([sp["cpu"] for sp in plan])
            ready_ns = time.monotonic_ns()
            write_ready_file(common.ready_file, common.identity,
                             version_record(include_grpc=False), affinity=affinity, children=[],
                             info={"threads": [], "unit_ops": UNIT_OPS,
                                   "target_role": args.target_role, "target_cpu": tcpu,
                                   "spinners": [{k: sp.get(k) for k in
                                                 ("index", "cpu", "purpose", "pid")}
                                                for sp in plan]})
            ready = True
            problems += watch(plan, stop)
            if t0 is not None:
                t1, cores_why = cpu_ticks([sp["cpu"] for sp in plan])
            stop_ns = time.monotonic_ns()
    finally:
        stop_spinners([sp for sp in plan if "pid" in sp], stop, common.stop_grace_s)
    rows = []
    for sp in plan:
        rec, more = spinner_record(sp, ready)
        rows.append(rec)
        problems += more
    payload = {
        "origin_ns": origin, "unit_ops": UNIT_OPS, "cpus": common.cpus,
        "workload": "31-bit LCG, pure Python, one process per core",
        "pin_table": {"path": args.pin_table, "sha256": pin_sha},
        "target_role": args.target_role, "target_cpu": tcpu, "supervisor_affinity": affinity,
        "ready": ready, "ready_rel_ns": ready_ns - origin if ready else None,
        "stop_rel_ns": stop_ns - origin if ready else None, "spinners": rows,
        "cores": core_deltas(t0, t1) if t0 is not None and t1 is not None else None,
        "cores_unreadable": cores_why, "clk_tck": os.sysconf("SC_CLK_TCK"),
        "stop": read_stop_file(RunTree(common.run_dir).stop_json), "signals": stop.signals,
        "problems": problems}
    path = output_path(common.run_dir, common.identity.role)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_json_stamped(path, SCHEMA, common.identity, payload)
    return exitcodes.RUNTIME if problems else exitcodes.OK


if __name__ == "__main__":
    sys.exit(main())
