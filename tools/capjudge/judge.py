# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capjudge: judge one launch tree (INTERFACES 10; design 2.6, 2.7).

    python3 -m capjudge --tree <launch tree> --workdir <W>

Exactly one tree, no globs; a tree is judged once (JUDGED.tsv + os.mkdir(<tree>/judge)).
Outputs only under <tree>/judge/, plus one JUDGED.tsv row and one STATE line.
"""

import argparse
import os
import subprocess
import sys

from capcommon import exitcodes, ledger
from capcommon.hashing import is_git_sha, sha256_file
from capcommon.jsonio import atomic_write_bytes, dumps, read_json, write_json_stamped
from capcommon.runtree import append_state

from . import checks as C
from . import inputs as I
from . import stats
from .symptoms import evaluate_x
from .treedata import LaunchData, TierData

INPUTS_SCHEMA = "capkit.judge.inputs/1"
VERDICTS_SCHEMA = "capkit.judge.verdicts/1"
CHECK_COLUMNS = ("tier_seq", "check_id", "expected_count", "checked_count", "red_count",
                 "status", "source_of_expected", "detail")
GLOB_CHARS = "*?["
STATE_OF = {"VALID": "VALID", "VOID": "VOID", "NOT_RUN": "NOT_RUN"}


class Log:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        self.f.write(msg.rstrip("\n") + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


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
    if key is not None:
        try:
            hit = ledger.judged_lookup(judged_tsv, *key)
        except exitcodes.ConfigError as e:
            return _usage("JUDGED.tsv unreadable: %s" % e)
        if hit is not None:
            sys.stderr.write("capjudge: ALREADY_JUDGED: %s judged at %s (%s)\n"
                             % (key, hit["judged_at"], hit["tree_relpath"]))
            return exitcodes.JUDGE_ALREADY_JUDGED
    jdir = os.path.join(tree, "judge")
    try:
        os.mkdir(jdir)
    except FileExistsError:
        sys.stderr.write("capjudge: ALREADY_JUDGED: %s exists\n" % jdir)
        return exitcodes.JUDGE_ALREADY_JUDGED
    log = Log(os.path.join(jdir, "judge.log"))
    try:
        return _judge(tree, W, jdir, log)
    finally:
        log.close()


def _judged_key(launch_path):
    try:
        ident = read_json(launch_path).get("identity") or {}
        return (ident["job_id"], ident["restart_count"], ident["launch_seq"], ident["arm"])
    except (OSError, ValueError, KeyError, AttributeError):
        return None


def _judge(tree, W, jdir, log):
    log("judge start: tree=%s workdir=%s" % (tree, W))
    try:
        ctx = I.verify(W, tree)
    except I.InputInvalid as e:
        log("INPUT_INVALID: %s" % e)
        doc = {"schema": INPUTS_SCHEMA, "identity": None, "status": "INPUT_INVALID",
               "item": e.item, "detail": e.detail}
        atomic_write_bytes(os.path.join(jdir, "inputs.json"), (dumps(doc) + "\n").encode())
        sys.stderr.write("capjudge: INPUT_INVALID: %s\n" % e)
        return exitcodes.JUDGE_INPUT_INVALID
    sha, how = code_sha()
    log("code sha: %s (%s)" % (sha, how))
    result = judge_tree(ctx, log, code_ok=sha is not None, code_note=how)
    ident = ctx.ident.with_role("judge")
    write_json_stamped(os.path.join(jdir, "inputs.json"), INPUTS_SCHEMA, ident, {
        "status": "OK", "code_sha": sha, "code_sha_source": how,
        "files_read": ctx.files_read, "calibration": ctx.bands.sources,
        "band_lookups": ctx.bands.lookups, "checks_applied": ctx.checks_applied,
        "identity_scan": ctx.identity_scan, "submission_submitted_at":
            ctx.submission["submitted_at"]})
    vpath = os.path.join(jdir, "verdicts.json")
    write_json_stamped(vpath, VERDICTS_SCHEMA, ident, result["verdicts"])
    _write_checks_tsv(os.path.join(jdir, "checks.tsv"), result["rows"])
    vsha = sha256_file(vpath)
    rel = os.path.relpath(tree, W)
    ledger.judged_append(os.path.join(W, "JUDGED.tsv"), {
        "job_id": ctx.ident.job_id, "restart_count": ctx.ident.restart_count,
        "launch_seq": ctx.ident.launch_seq, "arm": ctx.ident.arm,
        "experiment": ctx.ident.experiment, "tree_relpath": rel, "verdicts_sha256": vsha})
    lv = result["verdicts"]["launch_verdict"]
    append_state(tree, STATE_OF[lv], "launch_verdict=%s verdicts_sha=%s" % (lv, vsha[:12]),
                 "capjudge")
    log("done: launch_verdict=%s exit=%d" % (lv, result["verdicts"]["exit_code"]))
    return result["verdicts"]["exit_code"]


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
    launch_res = [C.check_c10(ctx, ld)]
    load_tiers = [t for t in tt.tiers if t.kind in ("up", "down")]
    gpu, gpu_void = C.check_gpu_idle(ctx, ld, load_tiers)
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
            if cid in I.BATCH2_CHECKS:
                res.append(C.not_implemented(cid, t.seq))
            elif cid == "C1":
                res.append(C.check_c1(ctx, td, record))
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
        x = evaluate_x(ctx, ld, td, base_td, record)
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
        entry.update({"verdict": verdict, "reasons": reasons,
                      "checks": {r.check_id: r.status for r in res},
                      "P": p, "X": x, "stats": record,
                      "order_key": tt.load_order_key(t.seq)})
        for r in res:
            rows.append(r.as_row())
        log("tier %d %s: %s (P=%s X2=%s holds=%s) %s" % (t.seq, t.label, verdict, p,
                                                         x["X2_status"], x["holds"], reasons))
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
    boundary = stats.t4_boundary([{"seq": e["tier_seq"], "order_key": e.get("order_key"),
                                   "load_value": e["load_value"], "verdict": e["verdict"]}
                                  for e in judged])
    return {"launch_verdict": lv, "exit_code": code, "tiers": tier_out,
            "launch_checks": {r.check_id: r.status for r in launch_res},
            "axis": ctx.tt.axis, "boundary_T4": boundary,
            "localisation_I": "NOT_IMPLEMENTED_BATCH2", "intervention_V": "NOT_IMPLEMENTED_BATCH2",
            "batch": 1}
