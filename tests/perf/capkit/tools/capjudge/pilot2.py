# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m capjudge.pilot2 <subcommand>: calibration v2 pilot (calib-v2 DESIGN-r7a 3.1 P2-09,
P2-09b, P2-09c, 3.2, 3.3, 4.5; INTERFACES erratum calib-v2-A5). capjudge.pilot (v1) is
unchanged.

Subcommands here: `d14` (e0_d14.json, capkit.d14_result/1), `bootstrap` (gate G0c: the bootstrap
capkit.pilot/2) and `shortfall` (capkit.e0_params/1 stage shortfall). The gate subcommands of
later sub-batches (bracket, capacity, recenter, settle, thresholds, drift, summary) use the same
framework below.

Framework (every subcommand):
* the list file names every input, one per line `<tree>\t<tier seqs or ALL>\t<regime>` (seqs
  comma-separated, regime subsat or near); no wildcards; a tree listed twice is an error;
* every tree is in <W>/JUDGED.tsv with the sha256 of its judge/verdicts.json, and all trees were
  judged by one judge code sha (judge/inputs.json code_sha);
* products are new files (open(..., "x")) with a `.cmd.txt` record beside them
  (capjudge.cmdrec);
* every rejection exits 3.
Every subcommand except `d14` and `bootstrap` requires its list file to equal, line for line, the
selection `_select` computes from the registered job plans and the ledgers (DESIGN 4.5): one tree
more, one less or a different order exits 3. `bootstrap` requires its list to hold exactly the
judged trees of every item of its register's budget_id (`bootstrap_universe`); both refuse an
ended job with a launch tree not judged yet (`unjudged_trees`). `d14` and `bootstrap` take their
tiers from capjudge.pilot.select_tiers (the launch-level conclusion recorded before the judge and
the deterministic checks), never from the judge's launch verdict.
"""

import argparse
import glob
import math
import os
import sys

from capcommon import exitcodes, jobplan, ledger, runtree
from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_file
from capcommon.jsonio import dumps, read_json

from . import checks as C
from . import cmdrec
from . import d14 as D14
from . import nullz
from . import pilot as P
from . import stats
from .judge import ATTEMPTS_DIR, BATCH
from .treedata import LaunchData, role_cpus

MODULE = "capjudge.pilot2"
REJECT = exitcodes.CONFIG  # 3
LIST_REGIMES = ("subsat", "near")
GLOB_CHARS = "*?["
SELECT = "capjudge.pilot.select_tiers"
WINDOW_NS = 60 * 10 ** 9
RAMP_BASE_NS = 4 * 10 ** 9
E0_PARAMS_SCHEMA = "capkit.e0_params/1"
D14_RESULT_SCHEMA = nullz.D14_SCHEMA
PILOT2_SCHEMA = nullz.PILOT2_SCHEMA
# DESIGN-r7a 2.11.3: the registered bootstrap rule, verbatim. The V0-cal register's
# expected.bootstrap_rule must equal this one line; `bootstrap` computes exactly what it says.
BOOTSTRAP_RULE = (
    "calib-v2 bootstrap rule (DESIGN-r7a 2.11.3): tiers = capjudge.pilot.select_tiers on the "
    "V0-cal trees, ladder role cal only; max over all keys, configs and roles; lag: z = max(10, "
    "3*max|z_d|), min_D_ns = max(10000000, 3*max|D_ns|); count: z = max(10.705, 3*max|z_d|); "
    "ratio: tol = max(0.02, 3*max|norm-1|), judge applies max(2/planned_requests, tol); c13: "
    "window = max(0.10, 3*max|foreign_busy_frac|), launch_mean = max(0.05, 3*max|mean over "
    "judged load tiers per launch x under_test role|)")
BOOTSTRAP_FACTOR = 3
# the r6 design constants (registered before any E0 v2 data) are the floors of the rule
BOOTSTRAP_FLOORS = {"lag_z": 10.0, "lag_D": 10_000_000, "count_z": 10.705, "ratio": 0.02,
                    "c13_window": 0.10, "c13_launch_mean": 0.05}
BOOTSTRAP_FAMILIES = tuple(BOOTSTRAP_FLOORS)
RATIO_MIN_PER_REQUEST = 2
CAL_ROLE = "cal"
# DESIGN 4.5: a launch is usable unless its job ended with one of these, or the job did not end,
# or the launch has no judged result, or its launch_verdict is VOID
JOB_LEVEL_UNUSABLE = ("ENV_INVALID", "CANCELLED", "SUBMIT_FAILED")
# post-gate rules whose items all enter the selection (N grows); shortfall items queue after the
# original segment and compete for the first N usable
ALL_SELECTED_RULES = ("coverage", "blind_topup", "backstop", "shift")
SHORTFALL_CLASS = jobplan.POST_GATE_PREFIX + "shortfall"
MAX_SHORTFALL_ROUNDS = 2
# selection unit of a block item: "cmp:<comparison>" (jobplan.block_comparison of its block id)
CMP_PREFIX = "cmp:"


class Pilot2Error(ConfigError):
    """A rejection: the subcommand writes nothing and exits 3."""


# ---- framework -----------------------------------------------------------------------------

def read_list(path, allow_empty=False):
    """[(absolute tree, [seq, ...] or "ALL", regime)] of a list file; Pilot2Error on a malformed
    line, a wildcard, a tree listed twice or (unless allow_empty: a selection may be empty when
    every launch was lost) an empty list."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().split("\n")
    except OSError as e:
        raise Pilot2Error("list file %s: %s" % (path, e))
    for n, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise Pilot2Error("%s:%d: need '<tree>\\t<tier seqs or ALL>\\t<regime>'" % (path, n))
        tree, tiers, regime = parts
        if not tree or any(c in tree for c in GLOB_CHARS):
            raise Pilot2Error("%s:%d: the tree must be one literal path (no %s)"
                              % (path, n, GLOB_CHARS))
        if any(c in tiers for c in GLOB_CHARS):
            raise Pilot2Error("%s:%d: no wildcard in the tier column" % (path, n))
        if tiers == "ALL":
            seqs = "ALL"
        else:
            try:
                seqs = [int(x) for x in tiers.split(",")]
            except ValueError:
                raise Pilot2Error("%s:%d: tier seqs must be integers or ALL" % (path, n))
            if len(set(seqs)) != len(seqs):
                raise Pilot2Error("%s:%d: a tier seq is listed twice" % (path, n))
        if regime not in LIST_REGIMES:
            raise Pilot2Error("%s:%d: regime must be one of %s" % (path, n, LIST_REGIMES))
        out.append((os.path.abspath(tree), seqs, regime))
    if not out and not allow_empty:
        raise Pilot2Error("%s lists no tree" % path)
    trees = [t for t, _s, _r in out]
    dup = sorted({t for t in trees if trees.count(t) > 1})
    if dup:
        raise Pilot2Error("a tree is listed twice: %s" % dup)
    return out


def judged(W, tree):
    """{"rel", "vpath", "vsha", "verdicts", "judge_sha"} of a tree judged in <W>/JUDGED.tsv with
    the sha256 of its current judge/verdicts.json; Pilot2Error otherwise."""
    rel = os.path.relpath(tree, W)
    vpath = os.path.join(tree, "judge", "verdicts.json")
    if not os.path.isfile(vpath):
        raise Pilot2Error("%s has not been judged (no judge/verdicts.json)" % rel)
    vsha = sha256_file(vpath)
    rows = [r for r in ledger.read_tsv(os.path.join(W, "JUDGED.tsv"), ledger.JUDGED_COLUMNS)
            if os.path.normpath(r["tree_relpath"]) == os.path.normpath(rel)]
    if not rows:
        raise Pilot2Error("%s is not in %s/JUDGED.tsv" % (rel, W))
    if rows[-1]["verdicts_sha256"] != vsha:
        raise Pilot2Error("%s: judge/verdicts.json sha256 %s != JUDGED.tsv %s"
                          % (rel, vsha, rows[-1]["verdicts_sha256"]))
    try:
        jsha = read_json(os.path.join(tree, "judge", "inputs.json")).get("code_sha")
    except (OSError, ValueError, AttributeError) as e:
        raise Pilot2Error("%s: judge/inputs.json unreadable: %s" % (rel, e))
    if not jsha:
        raise Pilot2Error("%s: judged without a code sha (judge/inputs.json code_sha)" % rel)
    verdicts = read_json(vpath)
    if not isinstance(verdicts, dict):
        raise Pilot2Error("%s: judge/verdicts.json is not a JSON object" % rel)
    return {"rel": rel, "vpath": vpath, "vsha": vsha, "verdicts": verdicts, "judge_sha": jsha}


def unique(values, what):
    """The single distinct value of `values` (dicts compared by content); Pilot2Error when there
    are several or none."""
    distinct = []
    for v in values:
        if v not in distinct:
            distinct.append(v)
    if len(distinct) != 1:
        raise Pilot2Error("%s is not unique over the inputs: %s" % (what, distinct))
    return distinct[0]


def check_v2(rel, verdicts):
    """A tree judged by the calibration v2 judge (verdicts.json batch calib-v2); a v1 tree
    exits 3. bootstrap and _select call it; d14 does not: its inputs are the round-1 trees the
    v1 judge judged (DESIGN-r7a 3.3 table)."""
    if verdicts.get("batch") != BATCH:
        raise Pilot2Error("%s was judged by batch %r, not %r (a v1 tree is no pilot2 input)"
                          % (rel, verdicts.get("batch"), BATCH))


def check_windows(rel, tt, seqs, window_ns=WINDOW_NS):
    """Every listed load tier has the registered window (60 s). For the gate subcommands of
    later sub-batches (no caller here: bootstrap has no window check, DESIGN-r7a 3.3 table)."""
    bad = {s: tt.tiers[s].window_ns for s in seqs if tt.tiers[s].window_ns != window_ns}
    if bad:
        raise Pilot2Error("%s: tier windows %s != %d ns" % (rel, bad, window_ns))


def check_ramp(rel, tt, seqs, settle_ns):
    """Every listed load tier has ramp_ns = 4 s + S (DESIGN 4.1). For the gate subcommands of
    later sub-batches (no caller here)."""
    want = RAMP_BASE_NS + settle_ns
    bad = {s: tt.tiers[s].ramp_ns for s in seqs if tt.tiers[s].ramp_ns != want}
    if bad:
        raise Pilot2Error("%s: tier ramps %s != 4 s + S = %d ns" % (rel, bad, want))


def registered(W, path, kind):
    """(relpath, sha256, doc) of a JSON file REGISTRY.tsv lists under `kind` at this path."""
    if not path or not os.path.isfile(path):
        raise Pilot2Error("%s file %r does not exist" % (kind, path))
    sha = sha256_file(path)
    row = ledger.registry_lookup(os.path.join(W, "REGISTRY.tsv"), sha)
    if row is None:
        raise Pilot2Error("%s (sha256 %s) is not in REGISTRY.tsv" % (path, sha))
    if ledger.registry_kind(row) != kind:
        raise Pilot2Error("%s is registered as kind %r, not %r"
                          % (path, ledger.registry_kind(row), kind))
    if os.path.realpath(os.path.join(W, row["relpath"])) != os.path.realpath(path):
        raise Pilot2Error("%s is not the registered relpath %s" % (path, row["relpath"]))
    if path.endswith(".tsv"):
        return row["relpath"], sha, None
    try:
        doc = read_json(path)
    except (OSError, ValueError) as e:
        raise Pilot2Error("%s: %s" % (path, e))
    return row["relpath"], sha, doc


def load_e0_params(W, path, stage):
    """{relpath, sha256, doc} of a registered capkit.e0_params/1 file of `stage`; a missing or
    unregistered file or another stage exits 3. For the gate subcommands of later sub-batches
    that read an earlier stage's product (no caller here)."""
    if path is None:
        raise Pilot2Error("needs the registered e0_params %s file" % stage)
    rel, sha, doc = registered(W, path, "e0_params")
    if not isinstance(doc, dict) or doc.get("schema") != E0_PARAMS_SCHEMA \
            or doc.get("stage") != stage:
        raise Pilot2Error("%s is not %s stage %r" % (rel, E0_PARAMS_SCHEMA, stage))
    return {"relpath": rel, "sha256": sha, "doc": doc}


def check_list_matches(entries, selected, W):
    """The list file's trees (in order) must equal the `_select` selection (tree relpaths under
    W) exactly: one more, one less or another order exits 3."""
    got = [os.path.normpath(t) for t, _s, _r in entries]
    want = [os.path.normpath(os.path.join(W, t)) for t in selected]
    if got == want:
        return
    extra = [os.path.relpath(t, W) for t in got if t not in want]
    missing = [os.path.relpath(t, W) for t in want if t not in got]
    raise Pilot2Error("the list file is not the _select selection (%d lines, %d selected): "
                      "extra %s, missing %s%s" % (len(got), len(want), extra[:5], missing[:5],
                                                  "" if extra or missing else ", order differs"))


def write_product(out, doc, argv, inputs):
    """Write `out` as a new file (open(..., "x")) and its .cmd.txt record."""
    data = (dumps(doc) + "\n").encode("utf-8")
    with open(out, "xb") as f:
        f.write(data)
    cmdrec.write(out, MODULE, argv, inputs, code=0)


# ---- _select: usable launches and the selection (DESIGN 4.5) --------------------------------

def _plan_files(W, plans):
    """[(relpath, sha256, items, post_gate rule or None)]: the original segment's plans
    (planned / surplus items) first, then the post-gate plans, each group in the given order."""
    if not plans:
        raise Pilot2Error("needs at least one registered job_plan.tsv")
    out = []
    for p in plans:
        rel, sha, _ = registered(W, p, "job_plan")
        try:
            items = jobplan.read(p)
        except (ConfigError, OSError, ValueError) as e:
            raise Pilot2Error("%s: %s" % (rel, e))
        cls = items[0]["item_class"]
        rule = cls[len(jobplan.POST_GATE_PREFIX):] if cls.startswith(
            jobplan.POST_GATE_PREFIX) else None
        out.append((rel, sha, items, rule))
    if all(r is not None for _a, _b, _c, r in out):
        raise Pilot2Error("no plan of the original segment (planned / surplus items)")
    return [x for x in out if x[3] is None] + [x for x in out if x[3] is not None]


def _unit_of(item, config):
    """Selection unit: the comparison of a block item (block_id '<comparison>-bNN'), else the
    configuration."""
    if item["block_id"] is not None:
        return CMP_PREFIX + jobplan.block_comparison(item["block_id"])
    return config


def _node_of(tree):
    try:
        return read_json(os.path.join(tree, "raw", "dims.json")).get("node")
    except (OSError, ValueError, AttributeError):
        return None


def launch_facts(W, plans, void_usable=False):
    """Per launch slot of every plan item, in registration order: {"pos", "plan", "item_id",
    "item_class", "rule", "unit", "config", "block_id", "slot", "tree" (relpath or None),
    "node", "usable", "reason", "job_level", "verdict", "ended" (the item's job has an ENDED
    row)}. Reads only the plans, BUDGET.tsv, submissions.tsv, JUDGED.tsv, each tree's
    verdicts.json launch_verdict / arm_config / batch (never z, b or X) and, for an ended job,
    which launch trees its run root holds (unjudged_trees: one not judged yet exits 3).
    void_usable: a VOID launch counts as usable (shortfall --job-level-only)."""
    files = _plan_files(W, plans)
    reserved, rows = {}, ledger.budget_rows(os.path.join(W, "BUDGET.tsv"))
    for r in rows:
        if r["event"] == "RESERVED":
            if r["item_id"] in reserved:
                raise Pilot2Error("item %s has two RESERVED rows in BUDGET.tsv" % r["item_id"])
            reserved[r["item_id"]] = r
    _check_plans_complete(files, [r for r in rows if r["event"] == "RESERVED"])
    subs = ledger.submissions_current(os.path.join(W, "submissions.tsv"))
    by_job = {}
    for r in ledger.read_tsv(os.path.join(W, "JUDGED.tsv"), ledger.JUDGED_COLUMNS):
        by_job.setdefault(r["job_id"], {})[(int(r["restart_count"]), int(r["launch_seq"]))] = r
    facts, seen = [], set()
    for rel, sha, items, rule in files:
        for it in items:
            if it["item_id"] in seen:
                raise Pilot2Error("item %s is in two plans" % it["item_id"])
            seen.add(it["item_id"])
            facts += _item_facts(W, rel, sha, it, rule, reserved, subs, by_job, void_usable)
    for k, f in enumerate(facts):
        f["pos"] = k
    return facts


def _check_plans_complete(files, reserved_rows):
    """The given plans must be every plan the segment has submitted from (RS28: the shortfall
    round, the cap already used and the selection are counted from them, so a plan left out
    would restart them). Segments of one budget_id (V1-b A, then B) run one after the other,
    so the segment's RESERVED rows are those of its budget_id from the first one of a given
    original-segment plan (of any given plan when the original has none yet) up to the first
    planned / surplus row of a plan not given (the next segment's original plan). Each of
    these rows must carry the sha of a given plan, else Pilot2Error naming the missing plan
    sha(s)."""
    shas = {sha for _rel, sha, _items, _rule in files}
    ids = {it["item_id"] for _rel, _sha, items, _rule in files for it in items}
    mine = [k for k, r in enumerate(reserved_rows) if r["item_id"] in ids]
    if not mine:
        return
    budget_ids = sorted({reserved_rows[k]["budget_id"] for k in mine})
    if len(budget_ids) != 1:
        raise Pilot2Error("the given plans were reserved under several budget ids %s"
                          % budget_ids)
    orig = {sha for _rel, sha, _items, rule in files if rule is None}
    starts = [k for k in mine if reserved_rows[k]["plan_sha256"] in orig]
    start = min(starts or mine)
    missing = set()
    for r in reserved_rows[start:]:
        if r["budget_id"] != budget_ids[0] or r["plan_sha256"] in shas:
            continue
        if not r["item_class"].startswith(jobplan.POST_GATE_PREFIX):
            break  # the next segment's original plan
        missing.add(r["plan_sha256"])
    missing = sorted(missing)
    if missing:
        raise Pilot2Error("BUDGET.tsv has RESERVED rows of budget %s since this segment began "
                          "under plan sha256 %s, not given with --plan: give every plan of "
                          "the segment (rounds, caps and the selection are counted from them)"
                          % (budget_ids[0], missing))


def _job_state(it, sha, reserved, subs):
    """(job id or None, reason it is unusable at job level or None, job ended?, merged
    submissions.tsv row or None)."""
    res = reserved.get(it["item_id"])
    if res is None:
        return None, "not submitted (no RESERVED row)", False, None
    if res["plan_sha256"] != sha:
        raise Pilot2Error("item %s was reserved under plan sha %s, not %s"
                          % (it["item_id"], res["plan_sha256"], sha))
    sub = subs.get(res["plan_id"])
    if sub is None:
        return None, "not submitted (plan id %s not in submissions.tsv)" % res["plan_id"], \
            False, None
    if sub["event"] != "ENDED":
        return sub["job_id"] or None, "job not ended (%s %s)" % (sub["event"], sub["status"]), \
            False, sub
    if sub["status"] in JOB_LEVEL_UNUSABLE:
        return sub["job_id"] or None, "job %s" % sub["status"], True, sub
    return sub["job_id"], None, True, sub


def _attempt_ended(tree):
    """True when a judge attempt on `tree` ended without verdicts: no judge/ and an attempt
    directory under judge.attempts/ holding its attempt.json (written when the attempt ends,
    e.g. INPUT_INVALID). An attempt still running has no attempt.json yet."""
    if os.path.exists(os.path.join(tree, "judge")):
        return False  # judged, its ledger row or STATE line not committed yet: run the judge
    return any(os.path.isfile(p) for p in glob.glob(os.path.join(tree, ATTEMPTS_DIR, "*",
                                                                 "attempt.json")))


def unjudged_trees(W, sub, judged_rels):
    """([not judged yet], [judge attempted, no verdicts]) relpaths of the launch trees of an
    ended job that are not in `judged_rels` (its JUDGED.tsv tree relpaths): the directories
    <CAPJOBS_RUN_ROOT of the plan's job.env>/J<job>-r*-L*/ holding raw/launch.json (the launch
    started; RS24). A run root that cannot be read exits 3: an unknown is never a loss
    (calibv2a-RS35)."""
    plan = sub["plan_id"]
    if not sub.get("snapshot_dir"):
        raise Pilot2Error("plan %s has no snapshot_dir in submissions.tsv: its run root (and "
                          "launch trees) cannot be found" % plan)
    try:
        root = ledger.plan_job_env_value(sub["snapshot_dir"], "CAPJOBS_RUN_ROOT")
    except OSError as e:
        raise Pilot2Error("plan %s: snapshot job.env unreadable: %s" % (plan, e))
    if not root:
        raise Pilot2Error("plan %s: no CAPJOBS_RUN_ROOT in its snapshot job.env" % plan)
    done = {os.path.realpath(os.path.join(W, r)) for r in judged_rels}
    pending, attempted = [], []
    try:
        pattern = runtree.tree_dir_glob(sub["job_id"])
    except ConfigError as e:
        raise Pilot2Error("plan %s: %s" % (plan, e))
    for d in sorted(glob.glob(os.path.join(glob.escape(root), pattern))):
        if not os.path.isfile(os.path.join(d, "raw", "launch.json")) \
                or os.path.realpath(d) in done:
            continue
        (attempted if _attempt_ended(d) else pending).append(os.path.relpath(d, W))
    return pending, attempted


def _check_judged(W, sub, judged_rels):
    """The judge-attempted, no-verdicts trees of an ended job; Pilot2Error (exit 3) when a
    launch tree of it is not judged yet."""
    pending, attempted = unjudged_trees(W, sub, judged_rels)
    if pending:
        raise Pilot2Error("launch tree(s) %s of ended job %s not judged yet: judge them, then "
                          "run this again" % (pending, sub["job_id"]))
    return attempted


def _item_facts(W, rel, sha, it, rule, reserved, subs, by_job, void_usable):
    job_id, why, ended, sub = _job_state(it, sha, reserved, subs)
    trees, attempted = [], []
    if why is None:
        rows = by_job.get(job_id, {})
        if len({rc for rc, _ls in rows}) > 1:
            raise Pilot2Error("job %s has launches judged under several restart counts" % job_id)
        trees = [rows[k] for k in sorted(rows)]
        attempted = _check_judged(W, sub, [r["tree_relpath"] for r in trees])
    slots = {c: n for c, n in it["config_counts"].items()}
    base = {"plan": rel, "item_id": it["item_id"], "item_class": it["item_class"], "rule": rule,
            "block_id": it["block_id"], "slot": it["slot"], "ended": ended}
    out = []
    for row in trees:
        tree = os.path.join(W, row["tree_relpath"])
        j = judged(W, tree)
        check_v2(j["rel"], j["verdicts"])
        cfg = j["verdicts"].get("arm_config")
        if slots.get(cfg, 0) < 1:
            raise Pilot2Error("%s: arm_config %r is not (or no longer) in item %s config_counts %s"
                              % (j["rel"], cfg, it["item_id"], it["config_counts"]))
        slots[cfg] -= 1
        lv = j["verdicts"].get("launch_verdict")
        bad = lv == "VOID" and not void_usable
        out.append(dict(base, unit=_unit_of(it, cfg), config=cfg, tree=j["rel"],
                        node=_node_of(tree), verdict=lv, usable=not bad,
                        reason="launch_verdict VOID" if bad else None, job_level=False))
    for cfg in sorted(slots):
        for _ in range(slots[cfg]):
            if why is None and attempted:
                reason = "judge attempted, no verdicts (%s)" % attempted.pop(0)
            else:
                reason = why or "no judged launch (no launch tree with raw/launch.json)"
            out.append(dict(base, unit=_unit_of(it, cfg), config=cfg, tree=None, node=None,
                            verdict=None, usable=False, reason=reason, job_level=True))
    return out


def planned_n(facts):
    """{unit: N}: the planned launches per configuration (planned blocks per comparison) of the
    original segment."""
    n = {}
    for f in facts:
        if f["item_class"] == "planned" and (f["block_id"] is None or f["slot"] == 1):
            n[f["unit"]] = n.get(f["unit"], 0) + 1
    return n


def _entries(facts):
    """Selection entries in registration order: one per launch, one per block (both slots)."""
    out, blocks = [], {}
    for f in facts:
        if f["block_id"] is None:
            out.append({"unit": f["unit"], "facts": [f], "pos": f["pos"]})
            continue
        key = (f["item_class"], f["block_id"])
        if key not in blocks:
            blocks[key] = {"unit": f["unit"], "facts": [], "pos": f["pos"]}
            out.append(blocks[key])
        blocks[key]["facts"].append(f)
    for e in out:
        fs = e["facts"]
        e["rule"], e["item_class"] = fs[0]["rule"], fs[0]["item_class"]
        bad = [f for f in fs if not f["usable"]]
        if bad:
            e["usable"], e["reason"] = False, "; ".join(
                "%s: %s" % (f["tree"] or f["item_id"], f["reason"]) for f in bad)
            e["job_level"] = any(f["job_level"] for f in bad)
        elif fs[0]["block_id"] is not None and (
                len(fs) != 2 or fs[0]["node"] is None or fs[0]["node"] != fs[1]["node"]):
            e["usable"], e["job_level"] = False, False
            e["reason"] = "block not on one known node: %s" % [f["node"] for f in fs]
        else:
            e["usable"], e["reason"], e["job_level"] = True, None, False
    return out


def select(facts, n_by_unit):
    """The DESIGN 4.5 selection from launch facts. Per unit: the first N usable entries of the
    original segment followed by the shortfall items (registration order) are selected, later
    usable ones are 'surplus, unused'; every entry of a post-gate rule in ALL_SELECTED_RULES
    (coverage, blind top-up, backstop, shift) is selected when usable (N grows by one per
    such entry). Unusable entries are listed with their reason. Returns {"selected": [tree],
    "surplus_unused": [tree], "unusable": [{"unit", "item_id", "tree", "reason",
    "job_level"}], "by_unit": {unit: {"N", "n_all_selected", "usable", "shortfall"}}}."""
    taken, by_unit = {}, {}
    sel, surplus, unusable = [], [], []
    for e in _entries(facts):
        u = e["unit"]
        bu = by_unit.setdefault(u, {"N": n_by_unit.get(u, 0), "n_all_selected": 0,
                                    "usable": 0, "shortfall": None})
        trees = [f["tree"] for f in e["facts"]]
        if not e["usable"]:
            for f in e["facts"]:
                unusable.append({"unit": u, "item_id": f["item_id"], "tree": f["tree"],
                                 "reason": e["reason"], "job_level": e["job_level"]})
            continue
        if e["rule"] in ALL_SELECTED_RULES:
            bu["n_all_selected"] += 1
            sel += trees
            continue
        bu["usable"] += 1
        if taken.get(u, 0) < bu["N"]:
            taken[u] = taken.get(u, 0) + 1
            sel += trees
        else:
            surplus += trees
    for u, n in n_by_unit.items():
        by_unit.setdefault(u, {"N": n, "n_all_selected": 0, "usable": 0, "shortfall": None})
    for bu in by_unit.values():
        bu["shortfall"] = max(0, bu["N"] - bu["usable"])
    return {"selected": sel, "surplus_unused": surplus, "unusable": unusable,
            "by_unit": {u: by_unit[u] for u in sorted(by_unit)}}


def _select(plans, W, void_usable=False):
    """The selection of DESIGN 4.5 over registered job plans (one or more job_plan.tsv of a
    segment and its post-gate plans) and the ledgers of <W>."""
    facts = launch_facts(os.path.abspath(W), plans, void_usable=void_usable)
    out = select(facts, planned_n(facts))
    out["facts"] = facts
    return out


# ---- d14 (P2-09; DESIGN 2.12) ----------------------------------------------------------------

def _select_tree(tree, seqs, W):
    """capjudge.pilot.select_tiers (the v1 builder's tree rule, unchanged)."""
    try:
        return P.select_tiers(tree, seqs, W, drop_verdicts=("VOID",))
    except P.PilotError as e:
        raise Pilot2Error(str(e))


def build_d14(entries, W, rule_path):
    """capkit.d14_result/1 from the listed trees (explicit tier seqs) and the registered rule."""
    W = os.path.abspath(W)
    try:
        rule, rule_rec = D14.load_rule(rule_path, W)
    except D14.D14RuleError as e:
        raise Pilot2Error(str(e))
    for tree, seqs, _regime in entries:
        if seqs == "ALL":
            raise Pilot2Error("%s: d14 takes explicit tier seqs, not ALL" % tree)
    unique([judged(W, tree)["judge_sha"] for tree, _s, _r in entries], "judge code sha")
    envs, inputs, trees, submitted = [], [], [], {}
    for tree, seqs, _regime in entries:
        sel = _select_tree(tree, seqs, W)
        ctx, rec = sel["ctx"], sel["rec"]
        envs.append(P._env_of(ctx))
        inputs.append(rec)
        if "tree_excluded" in rec:
            continue
        ld = LaunchData(ctx)
        trees.append((rec["tree"], D14.tree_readings(ctx, ld, role_cpus(ld.pin_table),
                                                     rec["tiers_used"])))
        submitted[rec["tree"]] = (ctx.launch.get("submission") or {}).get("submitted_at")
    env = unique(envs, "environment (hw_profile_sha, image_sha, capkit_sha, baseline_sha)")
    return {"schema": D14_RESULT_SCHEMA, "rule": {"relpath": rule_rec["relpath"],
                                                  "sha256": rule_rec["sha256"]},
            "select": SELECT, "environment": env, "inputs": inputs,
            "d14_softirq": D14.conclude(rule, rule_rec, trees, submitted)}


# ---- bootstrap (P2-09c; DESIGN 2.11.3, gate G0c) --------------------------------------------

def _register(W, path):
    """{relpath, sha256, doc, expected} of a registered register.json (kind register); the
    expected section, when present, must be an object ({} when absent)."""
    rel, sha, doc = registered(W, path, "register")
    if not isinstance(doc, dict):
        raise Pilot2Error("%s is not a JSON object" % rel)
    exp = doc.get("expected", {})
    if not isinstance(exp, dict):
        raise Pilot2Error("%s expected must be a JSON object" % rel)
    return {"relpath": rel, "sha256": sha, "doc": doc, "expected": exp}


def _d14_ref(W, reg):
    """The one d14_result entry of the register's calibration list, sha checked."""
    cal = reg["doc"].get("calibration", [])
    if not isinstance(cal, list):
        raise Pilot2Error("%s: calibration must be a list" % reg["relpath"])
    refs = [c for c in cal if isinstance(c, dict) and c.get("kind") == "d14_result"]
    if len(refs) != 1:
        raise Pilot2Error("%s: calibration must list exactly one d14_result (got %d)"
                          % (reg["relpath"], len(refs)))
    if not isinstance(refs[0].get("relpath"), str):
        raise Pilot2Error("%s: the d14_result entry needs a relpath" % reg["relpath"])
    path = os.path.join(W, refs[0]["relpath"])
    if not os.path.isfile(path) or sha256_file(path) != refs[0].get("sha256"):
        raise Pilot2Error("%s: d14_result %s missing or sha256 differs"
                          % (reg["relpath"], refs[0].get("relpath")))
    return {"relpath": refs[0]["relpath"], "sha256": refs[0]["sha256"]}


def _family(key):
    try:
        return nullz.family_of(key)
    except ValueError:
        return None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def bootstrap_readings(rel, tiers, used):
    """{family: [(|value|, tree, tier_seq, key)]} of one tree: on the used tiers, X2 / X3 z_d
    (`zstats`, lag and count families), the lag trend D and the X1 |norm - 1| (`stats`) and
    the C13 per-window foreign busy fractions; per under_test role (C13 key), the mean the
    judge's C13-MEAN takes, over every judged load tier of the launch (checks.
    c13_launch_readings; tier_seq None). Reads no X conclusion."""
    by_seq = {t["tier_seq"]: t for t in tiers}
    out = {f: [] for f in BOOTSTRAP_FAMILIES}
    c13 = set()
    for seq in used:
        t = by_seq[seq]
        for key, vals in sorted((t.get("zstats") or {}).items()):
            fam = _family(key)
            if fam in nullz.FAMILIES:
                out[fam + "_z"] += [(abs(v), rel, seq, key) for v in vals if _num(v)]
        for key, vals in sorted((t.get("stats") or {}).items()):
            vals = [v for v in vals if _num(v)]
            if _family(key) == "lag":
                out["lag_D"] += [(abs(v), rel, seq, key) for v in vals]
            elif key.startswith(nullz.RATIO_STAT + ":"):
                out["ratio"] += [(abs(v), rel, seq, key) for v in vals]
            elif key.startswith(nullz.C13_STAT + ":"):
                out["c13_window"] += [(abs(v), rel, seq, key) for v in vals]
    for t in tiers:
        if t.get("judged"):
            c13.update(k for k in (t.get("stats") or {}) if k.startswith(nullz.C13_STAT + ":"))
    for key in sorted(c13):
        vals = [v for v in C.c13_launch_readings(tiers, key) if _num(v)]
        if vals:
            out["c13_launch_mean"].append((abs(math.fsum(vals) / len(vals)), rel, None, key))
    return out


def bootstrap_values(readings):
    """(x_const, ratio_tol, c13_cap, max_abs) by the registered rule: per family
    max(design floor, 3 x max |reading|). A family without any reading exits 3."""
    max_abs = {}
    for fam in BOOTSTRAP_FAMILIES:
        rs = readings[fam]
        if not rs:
            raise Pilot2Error("no %s reading in the inputs: zero readings give no threshold"
                              % fam)
        best = max(rs, key=lambda r: r[0])
        three, floor = BOOTSTRAP_FACTOR * best[0], BOOTSTRAP_FLOORS[fam]
        max_abs[fam] = {"value": best[0], "tree": best[1], "tier_seq": best[2], "key": best[3],
                        "n_readings": len(rs), "three_x_max": three, "design_floor": floor,
                        "bound_by": "design_floor" if floor >= three else "data"}
    th = {f: max(m["design_floor"], m["three_x_max"]) for f, m in max_abs.items()}
    return ({"lag": {"z": th["lag_z"], "min_D_ns": th["lag_D"]}, "count": {"z": th["count_z"]}},
            {nullz.RATIO_STAT: {"min_per_request": RATIO_MIN_PER_REQUEST, "tol": th["ratio"]}},
            {nullz.C13_STAT: {"window": th["c13_window"], "launch_mean": th["c13_launch_mean"]}},
            max_abs)


def build_bootstrap(entries, W, register_path):
    """The bootstrap capkit.pilot/2 (stage bootstrap) from the V0-cal trees (every line ALL:
    the ladder's cal tiers, through capjudge.pilot.select_tiers)."""
    W = os.path.abspath(W)
    reg = _register(W, register_path)
    rule = reg["expected"].get("bootstrap_rule")
    if rule != BOOTSTRAP_RULE:
        raise Pilot2Error("%s expected.bootstrap_rule is not the registered rule text of this "
                          "code (DESIGN-r7a 2.11.3): %r" % (reg["relpath"], rule))
    d14 = _d14_ref(W, reg)
    universe = bootstrap_universe(W, reg)
    listed = {os.path.normpath(t) for t, _s, _r in entries}
    want = {os.path.normpath(os.path.join(W, t)) for t in universe.pop("trees")}
    if listed != want:
        raise Pilot2Error("the list file is not every judged tree of budget %s (calibv2a-RS28): "
                          "extra %s, missing %s" % (
                              universe["budget_id"],
                              sorted(os.path.relpath(t, W) for t in listed - want)[:5],
                              sorted(os.path.relpath(t, W) for t in want - listed)[:5]))
    envs, exps, inputs = [], [], []
    readings = {f: [] for f in BOOTSTRAP_FAMILIES}
    for tree, seqs, regime in entries:
        if seqs != "ALL" or regime != "subsat":
            raise Pilot2Error("%s: bootstrap lines are '<tree>\\tALL\\tsubsat' (the ladder's "
                              "cal tiers)" % tree)
        j = judged(W, tree)
        check_v2(j["rel"], j["verdicts"])
        cal = [t["tier_seq"] for t in j["verdicts"].get("tiers", [])
               if t.get("ladder_role") == CAL_ROLE]
        if not cal:
            raise Pilot2Error("%s has no tier of ladder role %s" % (j["rel"], CAL_ROLE))
        sel = _select_tree(tree, cal, W)
        ctx, rec = sel["ctx"], sel["rec"]
        if ctx.ident.register_sha != reg["sha256"]:
            raise Pilot2Error("%s was launched under register sha %s, not %s"
                              % (j["rel"], ctx.ident.register_sha, reg["sha256"]))
        envs.append(dict(P._env_of(ctx), judge_sha=j["judge_sha"]))
        exps.append(ctx.ident.experiment)
        inputs.append({"tree": rec["tree"], "judge_verdicts_sha256": rec["judge_verdicts_sha256"],
                       "config": j["verdicts"].get("arm_config"), "node": _node_of(tree),
                       "tiers_used": rec["tiers_used"], "tiers_excluded": rec["tiers_excluded"],
                       **({"tree_excluded": rec["tree_excluded"]} if "tree_excluded" in rec
                          else {})})
        if "tree_excluded" not in rec:
            for fam, rs in bootstrap_readings(rec["tree"], j["verdicts"]["tiers"],
                                              rec["tiers_used"]).items():
                readings[fam] += rs
    env = unique(envs, "environment (hw_profile_sha, image_sha, capkit_sha, baseline_sha, "
                       "judge_sha)")
    x_const, ratio_tol, c13_cap, max_abs = bootstrap_values(readings)
    return {"schema": PILOT2_SCHEMA, "experiment": unique(exps, "experiment"),
            "stage": "bootstrap", "environment": env, "d14": d14, "select": SELECT,
            "rule": rule, "register": {"relpath": reg["relpath"], "sha256": reg["sha256"]},
            "universe": universe, "inputs": inputs, "x_const": x_const,
            "ratio_tol": ratio_tol, "c13_cap": c13_cap, "max_abs": max_abs}


def bootstrap_universe(W, reg):
    """Every V0-cal tree, from the ledgers: the items with a RESERVED row under the register's
    expected.budget_id in BUDGET.tsv, their plans in submissions.tsv (each must have ended)
    and the JUDGED.tsv trees of their jobs. A launch tree of an ended job (not ENV_INVALID /
    CANCELLED / SUBMIT_FAILED) that is not judged yet exits 3; one the judge refused (an ended
    attempt, no verdicts) is no input. Returns {"budget_id", "items", "jobs", "trees"
    (relpaths)}."""
    bid = reg["expected"].get("budget_id")
    if not isinstance(bid, str) or not jobplan.ITEM_ID_RE.match(bid):
        raise Pilot2Error("%s expected.budget_id is missing: the V0-cal set is read from its "
                          "BUDGET.tsv rows" % reg["relpath"])
    rows = [r for r in ledger.budget_rows(os.path.join(W, "BUDGET.tsv"))
            if r["event"] == "RESERVED" and r["budget_id"] == bid]
    if not rows:
        raise Pilot2Error("BUDGET.tsv has no RESERVED row of budget %s" % bid)
    subs = ledger.submissions_current(os.path.join(W, "submissions.tsv"))
    by_job = {}
    for r in ledger.read_tsv(os.path.join(W, "JUDGED.tsv"), ledger.JUDGED_COLUMNS):
        by_job.setdefault(r["job_id"], set()).add(os.path.normpath(r["tree_relpath"]))
    trees, jobs, open_items = set(), set(), []
    for r in rows:
        sub = subs.get(r["plan_id"])
        if sub is None or sub["event"] != "ENDED":
            open_items.append(r["item_id"])
            continue
        if not sub["job_id"]:
            continue  # SUBMIT_FAILED: no job, no tree
        jobs.add(sub["job_id"])
        done = sorted(by_job.get(sub["job_id"], ()))
        if sub["status"] not in JOB_LEVEL_UNUSABLE:
            _check_judged(W, sub, done)
        trees.update(done)
    if open_items:
        raise Pilot2Error("budget %s items not ended yet: %s" % (bid, sorted(open_items)))
    return {"budget_id": bid, "items": len(rows), "jobs": len(jobs), "trees": sorted(trees)}


# ---- shortfall (P2-09b; DESIGN 4.5) ----------------------------------------------------------

def _jobs_of(alloc, pin_of, per_job):
    """Jobs laddergen --plan-from packs `alloc` {unit: launches or blocks} into: launches of
    configurations packed per_job to a job (pinned and unpinned separately), two jobs of one
    launch each per block of a comparison unit (DESIGN 4.5)."""
    tot, blocks = {}, 0
    for c, n in alloc.items():
        if c.startswith(CMP_PREFIX):
            blocks += n
            continue
        tot[pin_of[c]] = tot.get(pin_of[c], 0) + n
    return sum(-(-n // per_job) for n in tot.values()) + 2 * blocks


def _cap(topup, pin_of, per_job, budget_jobs):
    """{config: launches} within budget_jobs jobs, configs in name order (each gets the most
    that still fits); and whether anything was cut."""
    alloc = {}
    for c in sorted(topup):
        n = topup[c]
        while n > 0 and _jobs_of(dict(alloc, **{c: n}), pin_of, per_job) > budget_jobs:
            n -= 1
        if n:
            alloc[c] = n
    return alloc, any(alloc.get(c, 0) < n for c, n in topup.items())


def _segment_arms(reg, facts, files):
    """{config: (arm, pin)} of the original segment: the arm is register.json arm_config
    inverted over the segment's arms (one arm per configuration), pin the items' pin."""
    arm_config = reg["doc"].get("arm_config")
    if not isinstance(arm_config, dict):
        raise Pilot2Error("%s has no arm_config (not a calibration v2 register)" % reg["relpath"])
    out = {}
    for _rel, _sha, items, rule in files:
        if rule is not None:
            continue
        for it in items:
            if it["block_id"] is not None:
                continue
            for arm in it["arms"]:
                cfg = arm_config.get(arm)
                if cfg is None:
                    raise Pilot2Error("arm %s of item %s is not in %s arm_config"
                                      % (arm, it["item_id"], reg["relpath"]))
                if cfg not in it["config_counts"]:
                    continue
                prev = out.setdefault(cfg, (arm, it["pin"]))
                if prev != (arm, it["pin"]):
                    raise Pilot2Error("configuration %s has two arms or pin values in the "
                                      "segment: %s, %s" % (cfg, prev, (arm, it["pin"])))
    return out


def _block_arm(it):
    """The arm of a block item without its '.<block_id>.<slot>' suffix (laddergen job_plan_items
    names block arms '<arm>.<block_id>.<slot>')."""
    suffix = ".%s.%d" % (it["block_id"], it["slot"])
    if len(it["arms"]) != 1 or not it["arms"][0].endswith(suffix) \
            or len(it["arms"][0]) == len(suffix) or len(it["config_counts"]) != 1:
        raise Pilot2Error("block item %s must have one arm '<arm>%s' and one configuration"
                          % (it["item_id"], suffix))
    return it["arms"][0][:-len(suffix)]


def _segment_pairs(files):
    """{comparison unit: [[config, arm], [config, arm]]} (sorted by configuration) of the
    original segment's blocks; every block of a comparison must hold the same pair."""
    out, slots = {}, {}
    for _rel, _sha, items, rule in files:
        if rule is not None:
            continue
        for it in items:
            if it["block_id"] is None:
                continue
            key = (_unit_of(it, None), it["block_id"])
            slots.setdefault(key, []).append([next(iter(it["config_counts"])), _block_arm(it)])
    for (unit, bid), pair in slots.items():
        pair = sorted(pair)
        if len(pair) != 2 or pair[0][0] == pair[1][0]:
            raise Pilot2Error("block %s must hold two slots of two configurations, got %s"
                              % (bid, pair))
        if out.setdefault(unit, pair) != pair:
            raise Pilot2Error("comparison %s has blocks of different pairs: %s, %s"
                              % (unit[len(CMP_PREFIX):], out[unit], pair))
    return out


def _loss_units(facts, job_level_only):
    """{loss unit: lost?} of the segment for q_hat (DESIGN 4.5): a job (non-block item) or a
    block (both its jobs) is lost when one of its launches is unusable (with --job-level-only:
    unusable at job level; VOID launches count as usable then). A block on two nodes is
    unusable."""
    out = {}
    for e in _entries(facts):
        bad = (not e["usable"]) and (e["job_level"] or not job_level_only)
        f = e["facts"][0]
        keys = {("block", f["item_class"], f["block_id"])} if f["block_id"] is not None \
            else {("job", x["item_id"]) for x in e["facts"]}
        for k in keys:
            out[k] = out.get(k, False) or bad
    return out


def _shortfall_item(unit, n, arms, pairs):
    if unit.startswith(CMP_PREFIX):
        return {"item_class": SHORTFALL_CLASS, "comparison": unit[len(CMP_PREFIX):],
                "configs": pairs[unit], "blocks": n, "pin": True}
    return {"item_class": SHORTFALL_CLASS, "config": unit, "arm": arms[unit][0], "launches": n,
            "pin": arms[unit][1]}


def build_shortfall(entries, W, plans, register_path, per_job, phase, segment,
                    job_level_only=False):
    """capkit.e0_params/1 stage shortfall for one segment whose items have all ended. Reads
    only usable counts (job states, launch_verdict VOID or not), never z, b or X."""
    W = os.path.abspath(W)
    if not (isinstance(per_job, int) and per_job >= 1):
        raise Pilot2Error("--launches-per-job must be an int >= 1")
    reg = _register(W, register_path)
    cap_jobs = reg["expected"].get("shortfall_cap_jobs")
    if not isinstance(cap_jobs, int) or isinstance(cap_jobs, bool) or cap_jobs < 0:
        raise Pilot2Error("%s expected.shortfall_cap_jobs must be an int >= 0" % reg["relpath"])
    files = _plan_files(W, plans)
    sel = _select(plans, W, void_usable=job_level_only)
    facts = sel["facts"]
    open_items = sorted({f["item_id"] for f in facts if not f["ended"]})
    if open_items:
        raise Pilot2Error("segment items not ended yet: %s" % open_items)
    check_list_matches(entries, sel["selected"], W)
    prior = [x for x in files if x[3] == "shortfall"]
    rnd = len(prior) + 1
    if rnd > MAX_SHORTFALL_ROUNDS:
        raise Pilot2Error("shortfall round %d > %d (N15: stop and report)"
                          % (rnd, MAX_SHORTFALL_ROUNDS))
    units = _loss_units(facts, job_level_only)
    lost, ended = sum(units.values()), len(units)
    arms = _segment_arms(reg, facts, files)
    pairs = _segment_pairs(files)
    topup = {}
    for unit, bu in sel["by_unit"].items():
        n = stats.shortfall_topup(bu["shortfall"], lost, ended)
        if n:
            if unit not in arms and unit not in pairs:
                raise Pilot2Error("%s has a shortfall but no arm (block pair) in the original "
                                  "segment" % unit)
            topup[unit] = n
    prior_jobs = sum(len(x[2]) for x in prior)
    pin_of = {c: (True if c in pairs else arms[c][1]) for c in topup}
    alloc, capped = _cap(topup, pin_of, per_job, max(0, cap_jobs - prior_jobs))
    q = float(stats.shortfall_q(lost, ended))
    return {"schema": E0_PARAMS_SCHEMA, "experiment": reg["doc"].get("experiment"),
            "stage": "shortfall", "phase": phase, "segment": segment, "round": rnd,
            "job_level_only": bool(job_level_only),
            "rule": "ceil(1.25*d/(1-q_hat))+1, q_hat=min(0.5,max(0.05,lost/ended)), "
                    "<=%d rounds (DESIGN-r7a 4.5)" % MAX_SHORTFALL_ROUNDS,
            "register": {"relpath": reg["relpath"], "sha256": reg["sha256"]},
            "plans": [{"relpath": r, "sha256": s, "post_gate_rule": ru}
                      for r, s, _i, ru in files],
            "q_hat": {"lost": lost, "ended": ended, "value": q},
            "by_config": {u: dict(bu, topup=topup.get(u, 0), listed=alloc.get(u, 0))
                          for u, bu in sel["by_unit"].items()},
            "cap": {"shortfall_cap_jobs": cap_jobs, "prior_jobs": prior_jobs,
                    "launches_per_job": per_job, "jobs": _jobs_of(alloc, pin_of, per_job)},
            "capped": capped,
            "items": [_shortfall_item(c, n, arms, pairs) for c, n in sorted(alloc.items())],
            "select": {k: sel[k] for k in ("selected", "surplus_unused", "unusable")}}


# ---- V0 cancel count (DESIGN 4.2 s1-pc criterion; test 23) ----------------------------------

CANCELLED_REASON = "grpc_CANCELLED"


def v0_cancelled(rows, status_codes, start_rel_ns, end_rel_ns, stop_rel_ns):
    """Requests of a reqlog (`rows` columns done_rel_ns / status; `status_codes` the header's
    {str(status id): reason}) that ended grpc_CANCELLED inside [start_rel_ns, end_rel_ns) (the
    overload tier and its drain slot) and before the load generator stopped (stop_rel_ns): a
    cancellation at or after the stop is the stop's own, not a finding. For the V0 analysis
    (DESIGN-r7a 3.5 test 23; no caller here yet)."""
    ids = {int(k) for k, r in (status_codes or {}).items() if r == CANCELLED_REASON}
    return sum(1 for d, s in zip(rows.get("done_rel_ns", []), rows.get("status", []))
               if s in ids and start_rel_ns <= d < end_rel_ns and d < stop_rel_ns)


# ---- command line ----------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(prog="python -m %s" % MODULE, description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("d14", "e0_d14.json (capkit.d14_result/1), DESIGN-r7a 2.12"),
                      ("bootstrap", "bootstrap capkit.pilot/2 from the V0-cal trees (gate G0c)"),
                      ("shortfall", "capkit.e0_params/1 stage shortfall of one ended segment")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--workdir", required=True)
        p.add_argument("--list", required=True,
                       help="list file: <tree>\\t<tier seqs or ALL>\\t<regime> per line")
        p.add_argument("--out", required=True, help="new product file (never overwritten)")
        if name == "d14":
            p.add_argument("--rule", required=True,
                           help="capkit.d14_rule/1 registered in REGISTRY.tsv (kind d14_rule)")
        else:
            p.add_argument("--register", required=True,
                           help="the phase's registered register.json (kind register)")
        if name == "shortfall":
            p.add_argument("--plan", action="append", required=True,
                           help="registered job_plan.tsv of the segment (repeat: the original "
                                "plan and its post-gate plans)")
            p.add_argument("--launches-per-job", type=int, required=True)
            p.add_argument("--phase", required=True)
            p.add_argument("--segment", required=True)
            p.add_argument("--job-level-only", action="store_true",
                           help="V0-cal / V0: only ENV_INVALID, CANCELLED, SUBMIT_FAILED and "
                                "unfinished jobs are lost; a VOID launch is not")
    return ap


def run(a):
    """(product doc, input files) for parsed arguments; Pilot2Error on any rejection."""
    entries = read_list(a.list, allow_empty=a.cmd == "shortfall")
    if a.cmd == "d14":
        return build_d14(entries, a.workdir, a.rule), [a.list, a.rule]
    if a.cmd == "bootstrap":
        return build_bootstrap(entries, a.workdir, a.register), [a.list, a.register]
    doc = build_shortfall(entries, a.workdir, a.plan, a.register, a.launches_per_job, a.phase,
                          a.segment, job_level_only=a.job_level_only)
    led = [os.path.join(a.workdir, f) for f in ("BUDGET.tsv", "submissions.tsv", "JUDGED.tsv")
           if os.path.isfile(os.path.join(a.workdir, f))]
    return doc, [a.list, a.register] + list(a.plan) + led


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    a = build_parser().parse_args(argv)
    why = cmdrec.refuse_existing(a.out)
    if why:
        sys.stderr.write("%s: %s\n" % (MODULE, why))
        return REJECT
    try:
        doc, inputs = run(a)
    except (Pilot2Error, ConfigError, OSError, ValueError) as e:
        sys.stderr.write("%s %s: %s\n" % (MODULE, a.cmd, e))
        return REJECT
    write_product(a.out, doc, argv, inputs)
    print("%s %s: wrote %s" % (MODULE, a.cmd, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
