# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capjudge: judge one launch tree (INTERFACES 10; design 2.6, 2.7).

    python3 -m capjudge --tree <launch tree> --workdir <W>

Exactly one tree, no globs; a tree is judged once. Every run that gets past the two
already-judged checks (<tree>/judge exists, JUDGED.tsv row) writes into a fresh attempt
directory <tree>/judge.attempts/<attempt>/ with attempt.json. Only a run that produced verdicts
renames its attempt directory to <tree>/judge (atomic; that rename is the judged-once mark),
then appends one JUDGED.tsv row and one STATE line. The STATE line is the worse of the judge's
launch verdict and the conclusion recorded before it (caplaunch / capjobs STATE lines,
launch_end.json): the judge never improves on it. A later run that finds judge/ without its
ledger row or STATE line (the committing run died after the rename) appends the missing ones
from judge/ as written, without judging again. INPUT_INVALID and exceptions before the rename
leave their attempt directory (inputs.json / judge.log / attempt.json) for review and no
judge/, so the tree can be judged once its inputs are fixed.
"""

import argparse
import os
import socket
import subprocess
import sys
import traceback

from capcommon import exitcodes, ledger
from capcommon.clock import wall_iso
from capcommon.hashing import is_git_sha, sha256_file
from capcommon.jsonio import atomic_write_bytes, dumps, read_json, write_json_stamped
from capcommon.runtree import append_state, read_state, worst_state

from . import checks as C
from . import grade as G
from . import inputs as I
from . import intervene, localize, stats
from .symptoms import evaluate_x
from .treedata import LaunchData, TierData

INPUTS_SCHEMA = "capkit.judge.inputs/1"
VERDICTS_SCHEMA = "capkit.judge.verdicts/1"
ATTEMPT_SCHEMA = "capkit.judge.attempt/1"
ATTEMPTS_DIR = "judge.attempts"
CHECK_COLUMNS = ("tier_seq", "check_id", "expected_count", "checked_count", "red_count",
                 "status", "source_of_expected", "detail")
GLOB_CHARS = "*?["
STATE_OF = {"VALID": "VALID", "VOID": "VOID", "NOT_RUN": "NOT_RUN"}
NOT_REQUESTED = "NOT_REQUESTED"


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        self.f.write(msg.rstrip("\n") + "\n")
        self.f.flush()

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def code_sha():
    """(sha or None, how). CAPKIT_CODE_SHA wins; else git HEAD of <C>, only if tools/ is clean."""
    env = os.environ.get("CAPKIT_CODE_SHA")
    if env:
        return (env, "env CAPKIT_CODE_SHA") if is_git_sha(env) else (None, "bad CAPKIT_CODE_SHA")
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30, check=False)
        dirty = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "tools"],
                               capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        return None, "git unavailable: %s" % e
    sha = head.stdout.strip()
    if head.returncode != 0 or not is_git_sha(sha):
        return None, "git rev-parse failed"
    if dirty.returncode != 0 or dirty.stdout.strip():
        return None, "tools/ has uncommitted changes (HEAD %s does not identify the code)" % sha
    return sha, "git HEAD"


def _usage(msg):
    sys.stderr.write("capjudge: %s\n" % msg)
    return exitcodes.JUDGE_USAGE


def parse_args(argv):
    p = argparse.ArgumentParser(prog="capjudge", description=__doc__.split("\n")[0])
    p.add_argument("--tree", required=True)
    p.add_argument("--workdir", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tree, W = args.tree, args.workdir
    if any(c in tree for c in GLOB_CHARS):
        return _usage("--tree must be one literal path (no * ? [)")
    if not os.path.isdir(tree) or not os.path.isdir(W):
        return _usage("--tree and --workdir must be existing directories")
    tree, W = os.path.abspath(tree), os.path.abspath(W)
    launch_path = os.path.join(tree, "raw", "launch.json")
    if not os.path.isfile(launch_path):
        sys.stderr.write("capjudge: NOT_RUN: %s has no raw/launch.json (empty tree)\n" % tree)
        return exitcodes.JUDGE_NOT_RUN
    key = _judged_key(launch_path)
    judged_tsv = os.path.join(W, "JUDGED.tsv")
    jdir = os.path.join(tree, "judge")
    if os.path.lexists(jdir):
        return _finish_commit(tree, W, jdir, key, judged_tsv)
    if key is not None:
        try:
            hit = ledger.judged_lookup(judged_tsv, *key)
        except exitcodes.ConfigError as e:
            return _usage("JUDGED.tsv unreadable: %s" % e)
        if hit is not None:
            sys.stderr.write("capjudge: ALREADY_JUDGED: %s judged at %s (%s)\n"
                             % (key, hit["judged_at"], hit["tree_relpath"]))
            return exitcodes.JUDGE_ALREADY_JUDGED
    adir = _new_attempt_dir(tree)
    att = {"schema": ATTEMPT_SCHEMA, "tree": tree, "workdir": W, "pid": os.getpid(),
           "host": socket.gethostname(), "started_wall": wall_iso()}
    cur = adir  # where the attempt's files are now (adir, or jdir once committed)
    log = Log(os.path.join(adir, "judge.log"))
    try:
        code, commit = _judge(tree, W, adir, log)
        if commit is None:
            _write_attempt(cur, att, "INPUT_INVALID", code)
            sys.stderr.write("capjudge: record of this attempt: %s\n" % adir)
            return code
        if key is not None and ledger.judged_lookup(judged_tsv, *key) is not None:
            return _lost_race(adir, att, log, "JUDGED.tsv got a row for %s meanwhile" % (key,))
        _write_attempt(cur, att, "JUDGED", code)
        try:
            os.rename(adir, jdir)  # the judged-once mark: atomic, and only with full outputs
        except OSError:
            if os.path.lexists(jdir):
                return _lost_race(adir, att, log, "%s was created meanwhile" % jdir)
            raise
        cur = jdir
        _commit(tree, W, commit)
        log("done: launch_verdict=%s exit=%d" % (commit["launch_verdict"], code))
        return code
    except BaseException as e:
        att["error"] = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-8000:]
        log("ERROR: %r" % (e,))
        _write_attempt(cur, att, "ERROR",
                       exitcodes.INTERNAL if isinstance(e, Exception) else None)
        sys.stderr.write("capjudge: interrupted by %r; record of this attempt: %s\n" % (e, cur))
        raise
    finally:
        log.close()


def _new_attempt_dir(tree):
    """A new, empty <tree>/judge.attempts/<UTC stamp>-p<pid>[-k] (os.mkdir: never reused)."""
    parent = os.path.join(tree, ATTEMPTS_DIR)
    try:
        os.mkdir(parent)
    except FileExistsError:
        pass
    if os.path.islink(parent) or not os.path.isdir(parent):
        raise exitcodes.ConfigError("%s is not a plain directory" % parent)
    stamp = wall_iso().replace("-", "").replace(":", "")
    for k in range(100):
        path = os.path.join(parent, "%s-p%d%s" % (stamp, os.getpid(), "-%d" % k if k else ""))
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            continue
    raise exitcodes.ConfigError("cannot create a fresh attempt directory under %s" % parent)


def _write_attempt(adir, att, status, code):
    doc = dict(att, status=status, exit_code=code, ended_wall=wall_iso())
    atomic_write_bytes(os.path.join(adir, "attempt.json"), (dumps(doc) + "\n").encode())


def _lost_race(adir, att, log, why):
    log("ALREADY_JUDGED while this attempt ran: %s" % why)
    att["detail"] = why
    _write_attempt(adir, att, "ALREADY_JUDGED", exitcodes.JUDGE_ALREADY_JUDGED)
    sys.stderr.write("capjudge: ALREADY_JUDGED: %s; record of this attempt: %s\n" % (why, adir))
    return exitcodes.JUDGE_ALREADY_JUDGED


def _commit(tree, W, c):
    _append_judged(tree, W, c["ident"].to_dict(), c["vsha"])
    sl = c["state_line"]
    append_state(tree, sl["status"], _state_reason(sl, c["vsha"]), I.JUDGE_WRITER)


def _append_judged(tree, W, ident, vsha):
    ledger.judged_append(os.path.join(W, "JUDGED.tsv"), {
        "job_id": ident["job_id"], "restart_count": ident["restart_count"],
        "launch_seq": ident["launch_seq"], "arm": ident["arm"],
        "experiment": ident["experiment"], "tree_relpath": os.path.relpath(tree, W),
        "verdicts_sha256": vsha})


def _state_reason(sl, vsha):
    return "%s verdicts_sha=%s" % (sl["reason"], vsha[:12])


def _finish_commit(tree, W, jdir, key, judged_tsv):
    """<tree>/judge exists: the tree is judged. If the run that renamed its attempt directory
    to judge/ died before appending its JUDGED.tsv row or its STATE line, append the missing
    one(s) now from judge/ as written (verdicts.json state_line and sha; nothing is re-judged)
    and return the recorded exit code. Otherwise (both present, or judge/ cannot be read, or
    its writer may still be running) ALREADY_JUDGED."""
    def refuse(why):
        sys.stderr.write("capjudge: ALREADY_JUDGED: %s exists (%s)\n" % (jdir, why))
        return exitcodes.JUDGE_ALREADY_JUDGED

    vpath = os.path.join(jdir, "verdicts.json")
    try:
        att = read_json(os.path.join(jdir, "attempt.json"))
        verdicts = read_json(vpath)
        vsha = sha256_file(vpath)
        sl = verdicts["state_line"]
        code = verdicts["exit_code"]
        ident = verdicts["identity"]
        state_lines = read_state(tree)
        hit = ledger.judged_lookup(judged_tsv, *key) if key is not None else None
        if not isinstance(att, dict) or not isinstance(ident, dict):
            raise TypeError("attempt.json or verdicts.json identity is not an object")
    except (OSError, ValueError, KeyError, TypeError, exitcodes.ConfigError) as e:
        return refuse("cannot check that its ledger row and STATE line exist: %s" % (e,))
    mark = "verdicts_sha=%s" % vsha[:12]
    has_state = any(by == I.JUDGE_WRITER and reason.endswith(mark)
                    for _at, _st, by, reason in state_lines)
    if hit is not None and hit["verdicts_sha256"] != vsha:
        return refuse("JUDGED.tsv row has verdicts_sha256 %s, judge/verdicts.json is %s"
                      % (hit["verdicts_sha256"], vsha))
    if hit is not None and has_state:
        return refuse("judged at %s" % hit["judged_at"])
    if key is None:
        return refuse("no launch key in raw/launch.json: cannot complete the JUDGED.tsv row")
    if att.get("status") == "JUDGED" and att.get("host") == socket.gethostname() \
            and _pid_alive(att.get("pid")):
        return refuse("its writer pid %s may still be committing" % att.get("pid"))
    done = []
    with Log(os.path.join(jdir, "judge.log")) as log:
        log("completing the commit of %s (pid %d): ledger row %s, STATE line %s"
            % (jdir, os.getpid(), "present" if hit else "missing",
               "present" if has_state else "missing"))
        if hit is None:
            try:
                _append_judged(tree, W, ident, vsha)
            except exitcodes.ConfigError as e:  # another run added the row meanwhile
                log("JUDGED.tsv append refused: %s" % e)
                return refuse("JUDGED.tsv append refused: %s" % e)
            done.append("JUDGED.tsv row")
        if not has_state:
            append_state(tree, sl["status"], _state_reason(sl, vsha), I.JUDGE_WRITER)
            done.append("STATE line")
        att = dict(att, status="JUDGED", exit_code=code, commit_completed={
            "pid": os.getpid(), "host": socket.gethostname(), "wall": wall_iso(),
            "appended": done})
        atomic_write_bytes(os.path.join(jdir, "attempt.json"), (dumps(att) + "\n").encode())
        log("done: appended %s from judge/ (not re-judged); exit=%d" % (done, code))
    sys.stderr.write("capjudge: completed the commit of %s: appended %s\n" % (jdir, done))
    return code


def _pid_alive(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _judged_key(launch_path):
    try:
        ident = read_json(launch_path).get("identity") or {}
        return (ident["job_id"], ident["restart_count"], ident["launch_seq"], ident["arm"])
    except (OSError, ValueError, KeyError, AttributeError):
        return None


def _judge(tree, W, adir, log):
    """Write inputs.json (+ verdicts.json, checks.tsv) into the attempt dir `adir`.
    Returns (exit code, commit) with commit None for INPUT_INVALID."""
    log("judge start: tree=%s workdir=%s attempt=%s" % (tree, W, adir))
    try:
        ctx = I.verify(W, tree)
    except I.InputInvalid as e:
        log("INPUT_INVALID: %s" % e)
        doc = {"schema": INPUTS_SCHEMA, "identity": None, "status": "INPUT_INVALID",
               "item": e.item, "detail": e.detail}
        atomic_write_bytes(os.path.join(adir, "inputs.json"), (dumps(doc) + "\n").encode())
        sys.stderr.write("capjudge: INPUT_INVALID: %s\n" % e)
        return exitcodes.JUDGE_INPUT_INVALID, None
    sha, how = code_sha()
    log("code sha: %s (%s)" % (sha, how))
    result = judge_tree(ctx, log, code_ok=sha is not None, code_note=how)
    ident = ctx.ident.with_role("judge")
    write_json_stamped(os.path.join(adir, "inputs.json"), INPUTS_SCHEMA, ident, {
        "status": "OK", "code_sha": sha, "code_sha_source": how,
        "files_read": ctx.files_read, "calibration": ctx.bands.sources,
        "band_lookups": ctx.bands.lookups, "checks_applied": ctx.checks_applied,
        "identity_scan": ctx.identity_scan, "submission_submitted_at":
            ctx.submission["submitted_at"]})
    vpath = os.path.join(adir, "verdicts.json")
    write_json_stamped(vpath, VERDICTS_SCHEMA, ident, result["verdicts"])
    _write_checks_tsv(os.path.join(adir, "checks.tsv"), result["rows"])
    lv = result["verdicts"]["launch_verdict"]
    log("verdicts written: launch_verdict=%s exit=%d" % (lv, result["verdicts"]["exit_code"]))
    return result["verdicts"]["exit_code"], {"ident": ctx.ident, "vsha": sha256_file(vpath),
                                             "launch_verdict": lv,
                                             "state_line": result["verdicts"]["state_line"]}


def _write_checks_tsv(path, rows):
    lines = ["\t".join(CHECK_COLUMNS)]
    for r in rows:
        lines.append("\t".join(str(r[c]) for c in CHECK_COLUMNS))
    atomic_write_bytes(path, ("\n".join(lines) + "\n").encode("utf-8"))


def _fold(results):
    """P status of a list of Results: RED if any RED, else NOT_RUN if any NOT_RUN, else PASS."""
    sts = [r.status for r in results if r is not None]
    if C.RED in sts:
        return C.RED
    if C.NOT_RUN in sts:
        return C.NOT_RUN
    return C.PASS


def judge_tree(ctx, log, code_ok=True, code_note=""):
    ld = LaunchData(ctx)
    tt = ctx.tt
    tds = {t.seq: TierData(ctx, t) for t in tt.tiers}
    applied = set(ctx.checks_applied)
    rows = []
    # ---- launch-level preconditions -----------------------------------------------------
    launch_res = [C.check_launch_state(ctx), C.check_c10(ctx, ld)]
    load_tiers = [t for t in tt.tiers if t.kind in ("up", "down")]
    gpu, gpu_void = C.check_gpu_idle(ctx, ld, load_tiers,
                                     own_pids=C.launch_pids(ld, tds.values()))
    launch_res.append(gpu)
    if "C13" in applied:
        launch_res.append(C.check_c13_launch(ctx, ld))
    pt = C.check_probe_tier(ctx, ld)
    if pt is not None:
        launch_res.append(pt)
    launch_notes = []
    if not code_ok:
        launch_notes.append("judge code sha unavailable: %s" % code_note)
    le = ld.launch_end
    completed = le.get("tiers_completed") if isinstance(le, dict) else None
    if not isinstance(completed, int):
        launch_notes.append("launch_end.json missing or without tiers_completed")
        completed = -1
    base = tt.baseline_seq()
    if base is None:
        launch_notes.append("tier table has no baseline tier (X1/T2 need exactly one)")
    launch_p = _fold(launch_res)
    for r in launch_res:
        rows.append(r.as_row())
        log("launch %s: %s exp=%d chk=%d red=%d %s" % (r.check_id, r.status, r.expected,
                                                       r.checked, r.red, r.detail[:3]))
    # ---- per tier ---------------------------------------------------------------------------
    tier_out = []
    i_cache = {}  # (tier seq, group pids) -> I1 readings, shared: the baseline is read once
    for t in tt.tiers:
        entry = {"tier_seq": t.seq, "tier_label": t.label, "kind": t.kind,
                 "load_value": t.load_value, "judged": t.kind in ("up", "down")}
        if not entry["judged"]:
            entry.update({"verdict": None, "reasons": ["%s tier: not judged for stability"
                                                       % t.kind], "checks": {}})
            tier_out.append(entry)
            continue
        res, record = [], {}
        td = tds[t.seq]
        res.append(C.check_records(td))
        res.append(C.check_pids(td))
        res.append(C.check_sent(ctx, td))
        drain_td = tds.get(t.seq - 1)
        nxt = tds.get(t.seq + 1)
        next_drain = nxt if nxt is not None and nxt.tier.kind == "drain" else None
        for cid in ctx.checks_applied:
            if cid == "C1":
                res.append(C.check_c1(ctx, td, record))
            elif cid == "C2":
                res.append(C.check_c2(ctx, ld, td, record))
            elif cid == "C4":
                res.append(C.check_c4(ctx, td))
            elif cid == "C12":
                res.append(C.check_c12(ctx, ld, td, next_drain))
            elif cid == "C3":
                res.append(C.check_c3(ctx, td))
            elif cid in ("C5", "C6", "C7"):
                kind = {"C5": "requests", "C6": "messages", "C7": "chunks"}[cid]
                res.append(C.check_chains(ctx, ld, td, next_drain, kind))
            elif cid == "C8":
                res.append(C.check_c8(ctx, ld, td))
            elif cid == "C9":
                res.append(C.check_c9(ctx, ld, td))
            elif cid == "C11":
                res.append(C.check_c11(ctx, ld, td, drain_td))
            elif cid == "C13":
                res.append(C.check_c13_tier(ctx, ld, td, record))
        base_td = tds.get(base) if base is not None else None
        x = evaluate_x(ctx, ld, td, base_td, record, hooks=_hooks(ctx, ld, td, base_td, record,
                                                                 i_cache))
        p = _fold(res)
        reasons = list(launch_notes)
        if t.seq > completed:
            reasons.append("tier not completed (launch_end.tiers_completed=%d)" % completed)
        if launch_p == C.RED or gpu_void:
            verdict = "VOID"
            reasons.append("launch-level precondition RED")
        elif p == C.RED:
            verdict = "VOID"
            reasons += ["%s RED" % r.check_id for r in res if r.status == C.RED]
        elif launch_p == C.NOT_RUN or p == C.NOT_RUN or reasons:
            verdict = "NOT_RUN"
            reasons += ["%s NOT_RUN" % r.check_id for r in res + launch_res
                        if r.status == C.NOT_RUN]
        elif x["contradiction"]:
            verdict = "VOID"
            reasons += x["contradiction"]
        elif x["holds"] is None:
            verdict = "NOT_RUN"
            reasons += ["X NOT_RUN"] + x["reasons"]
        else:
            verdict = "UNSTABLE" if x["holds"] else "STABLE"
            reasons += x["reasons"]
            if verdict == "UNSTABLE" and x["I"].get("status") == localize.UNDETERMINED:
                verdict = "UNDETERMINED"  # design 2.6: signals disagree -> no boundary here
                reasons += ["I: %s" % r for r in x["I"]["reasons"]]
        entry.update({"verdict": verdict, "reasons": reasons, "regime": ctx.tier_regime[t.seq],
                      "checks": {r.check_id: r.status for r in res},
                      "P": p, "X": x, "stats": record,
                      "order_key": tt.load_order_key(t.seq)})
        for r in res:
            rows.append(r.as_row())
        log("tier %d %s: %s (P=%s X2=%s holds=%s I=%s) %s" % (
            t.seq, t.label, verdict, p, x["X2_status"], x["holds"], x["I"].get("status"),
            reasons))
        tier_out.append(entry)
    return {"verdicts": _launch_verdict(ctx, tier_out, launch_res, launch_p, gpu_void),
            "rows": rows}


def _launch_verdict(ctx, tier_out, launch_res, launch_p, gpu_void):
    judged = [e for e in tier_out if e["judged"]]
    vs = [e["verdict"] for e in judged]
    if "VOID" in vs or launch_p == C.RED or gpu_void:
        lv, code = "VOID", exitcodes.JUDGE_RED
    elif not judged or "NOT_RUN" in vs:
        lv, code = "NOT_RUN", exitcodes.JUDGE_NOT_RUN
    else:
        lv, code = "VALID", exitcodes.JUDGE_PASS
    up = ctx.upstream
    state = worst_state([STATE_OF[lv], up["status"]])
    reason = "launch_verdict=%s upstream=%s" % (lv, up["status"])
    boundary = stats.t4_boundary([{"seq": e["tier_seq"], "order_key": e.get("order_key"),
                                   "load_value": e["load_value"], "verdict": e["verdict"]}
                                  for e in judged])
    boundary["undetermined_seqs"] = [e["tier_seq"] for e in judged
                                     if e["verdict"] == "UNDETERMINED"]
    out = {"launch_verdict": lv, "exit_code": code, "tiers": tier_out,
           "upstream": up, "state_line": {"status": state, "reason": reason, "by": "capjudge"},
           "launch_checks": {r.check_id: r.status for r in launch_res},
           "axis": ctx.tt.axis, "boundary_T4": boundary, "mode": ctx.mode, "batch": 2}
    if ctx.mode != "P+X+I+V":
        out.update(localisation_I=NOT_REQUESTED, intervention_V=NOT_REQUESTED,
                   grade=NOT_REQUESTED)
        return out
    g = G.launch_grade(tier_out, boundary)
    at = next((e for e in judged if e["tier_seq"] == g.get("tier_seq")), None)
    out["grade"] = g
    i = {} if at is None else at["X"]["I"]
    out["localisation_I"] = dict({k: i.get(k) for k in ("status", "role", "via", "finding",
                                                        "measure_class", "pids", "holding")},
                                 tier_seq=g.get("tier_seq"),
                                 detail="tiers[].X.I of that tier")
    out["intervention_V"] = {
        "doses": intervene.launch_doses(ctx), "heartbeat": intervene.launch_heartbeat(judged),
        "measure_class": {r: d["measure_class"] for r, d in sorted(ctx.roles.items())},
        "dose_response": "NOT_RUN: across launches, python3 -m capjudge.intervene"}
    return out


def _hooks(ctx, ld, td, base_td, record, cache):
    """symptoms.evaluate_x hooks for I and the per-tier V heartbeat (mode P+X+I+V only)."""
    if ctx.mode != "P+X+I+V":
        return None
    return {"I": lambda x: localize.localize_tier(ctx, ld, td, base_td, x, record, cache),
            "V": lambda: intervene.tier_heartbeat(ctx, td)}
