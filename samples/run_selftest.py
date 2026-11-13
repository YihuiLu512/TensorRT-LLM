# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Judge self-test runner (INTERFACES 10.4; design 2.7 sample classes; pitfalls 1, 7).

For every sample W under --samples: copy it to a fresh temp dir, run the judge exactly as
documented (`python -m <pkg> --tree <T> --workdir <W>`, one tree, no globs), and assert the
exit code, the listed tier verdicts and that every listed check is RED / NOT_RUN. Plus:
  * the good sample judged a second time -> 12 without a new attempt record; its tree copied
    elsewhere (JUDGED.tsv) -> 12;
  * an INPUT_INVALID sample leaves no judge/ but one judge.attempts/*/ record (attempt.json and
    inputs.json both INPUT_INVALID), and judging it again gives 13 again, not 12;
  * a glob in --tree and a missing tree -> 2;
  * sabotage.json applied to a fresh copy of the good sample: assert the edit changed the
    target file (positive control of the sabotage), then assert the expected RED.
Exit code 0 only if every assertion holds and at least one sample ran; --expect-samples is
required and must equal the number of sample dirs (with expect.json) under --samples.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
FIXED_CODE_SHA = "b" * 40


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_judge(args, W, tree):
    env = dict(os.environ, PYTHONPATH=args.tools, CAPKIT_CODE_SHA=FIXED_CODE_SHA)
    p = subprocess.run([sys.executable, "-m", args.package, "--tree", tree, "--workdir", W],
                       env=env, capture_output=True, text=True, timeout=300, check=False)
    return p.returncode, p.stderr


def read_checks(tree):
    path = os.path.join(tree, "judge", "checks.tsv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lines = f.read().splitlines()
    cols = lines[0].split("\t")
    return [dict(zip(cols, ln.split("\t"))) for ln in lines[1:]]


def compare(name, exp, rc, tree, errs):
    if rc != exp["exit_code"]:
        errs.append("%s: exit %d != expected %d" % (name, rc, exp["exit_code"]))
    tv = exp.get("tier_verdicts") or {}
    if tv:
        vpath = os.path.join(tree, "judge", "verdicts.json")
        if not os.path.exists(vpath):
            errs.append("%s: no verdicts.json" % name)
            return
        with open(vpath) as f:
            v = json.load(f)
        got = {str(t["tier_seq"]): t["verdict"] for t in v["tiers"]}
        for seq, want in tv.items():
            if got.get(seq) != want:
                errs.append("%s: tier %s verdict %r != expected %r" % (name, seq, got.get(seq),
                                                                       want))
    rows = read_checks(tree)
    for key, status in (("red_checks", "RED"), ("not_run_checks", "NOT_RUN")):
        for cid in exp.get(key) or []:
            if not any(r["check_id"] == cid and r["status"] == status for r in rows):
                errs.append("%s: check %s never %s" % (name, cid, status))


def copy_sample(src, tmp, name):
    dst = os.path.join(tmp, name)
    shutil.copytree(src, dst, symlinks=True)
    return dst


def run(args):
    errs, ran = [], []
    names = sorted(n for n in os.listdir(args.samples)
                   if os.path.isfile(os.path.join(args.samples, n, "expect.json")))
    if len(names) != args.expect_samples:
        errs.append("found %d samples, expected %d" % (len(names), args.expect_samples))
    tmp = tempfile.mkdtemp(prefix="judge-selftest-", dir=args.tmp)
    for name in names:
        W = copy_sample(os.path.join(args.samples, name), tmp, name)
        with open(os.path.join(W, "expect.json")) as f:
            exp = json.load(f)
        tree = os.path.join(W, exp["tree"])
        if exp.get("create_empty_tree"):
            os.makedirs(tree)
            if os.listdir(tree):
                errs.append("%s: tree is not empty" % name)
        rc, err = run_judge(args, W, tree)
        compare(name, exp, rc, tree, errs)
        ran.append((name, rc))
        if exp["exit_code"] == 13:
            _input_invalid_extras(args, name, W, tree, errs)
        if name == "good":
            _good_extras(args, W, tree, tmp, errs)
    print(json.dumps({"ran": ran, "errors": errs, "tmp": tmp}, indent=1))
    if not ran:
        errs.append("no sample ran")
    return 0 if not errs else 1


def attempts(tree):
    d = os.path.join(tree, "judge.attempts")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _input_invalid_extras(args, name, W, tree, errs):
    if os.path.lexists(os.path.join(tree, "judge")):
        errs.append("%s: INPUT_INVALID left judge/ behind" % name)
    att = attempts(tree)
    if len(att) != 1:
        errs.append("%s: %d attempt records, expected 1" % (name, len(att)))
        return
    a = os.path.join(tree, "judge.attempts", att[0])
    doc, inp = read_json(os.path.join(a, "attempt.json")), read_json(os.path.join(a, "inputs.json"))
    if (doc or {}).get("status") != "INPUT_INVALID" or (inp or {}).get("status") != "INPUT_INVALID":
        errs.append("%s: attempt record %s is not INPUT_INVALID" % (name, att[0]))
    rc, _ = run_judge(args, W, tree)
    if rc != 13:
        errs.append("%s judged again: exit %d != 13 (stuck after INPUT_INVALID?)" % (name, rc))


def _good_extras(args, W, tree, tmp, errs):
    if (read_json(os.path.join(tree, "judge", "attempt.json")) or {}).get("status") != "JUDGED":
        errs.append("good: judge/attempt.json missing or not JUDGED")
    before = attempts(tree)
    rc, _ = run_judge(args, W, tree)
    if rc != 12:
        errs.append("good judged twice: exit %d != 12" % rc)
    if attempts(tree) != before:
        errs.append("good judged twice: the rejected run left an attempt record")
    moved = os.path.join(W, "runs", "copied-elsewhere")
    shutil.copytree(tree, moved, ignore=shutil.ignore_patterns("judge", "STATE"))
    rc, _ = run_judge(args, W, moved)
    if rc != 12:
        errs.append("good tree copied elsewhere: exit %d != 12 (JUDGED.tsv)" % rc)
    rc, _ = run_judge(args, W, os.path.join(W, "runs", "*"))
    if rc != 2:
        errs.append("glob --tree: exit %d != 2" % rc)
    rc, _ = run_judge(args, W, os.path.join(W, "runs", "does-not-exist"))
    if rc != 2:
        errs.append("missing --tree: exit %d != 2" % rc)
    # sabotage on a fresh copy
    W2 = copy_sample(os.path.join(args.samples, "good"), tmp, "good-sabotaged")
    with open(os.path.join(W2, "sabotage.json")) as f:
        sab = json.load(f)
    target = os.path.join(W2, sab["file"])
    before = sha(target)
    with open(target) as f:
        doc = json.load(f)
    cur = doc
    for k in sab["json_path"][:-1]:
        cur = cur[k]
    cur[sab["json_path"][-1]] += sab["add"]
    with open(target, "w") as f:
        json.dump(doc, f, sort_keys=True, indent=1)
    if sha(target) == before:
        errs.append("sabotage did not change %s (positive control failed)" % sab["file"])
    with open(os.path.join(W2, "expect.json")) as f:
        tree2 = os.path.join(W2, json.load(f)["tree"])
    rc, _ = run_judge(args, W2, tree2)
    compare("good+sabotage", sab["expect"], rc, tree2, errs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="judge self-test over the sample work dirs")
    ap.add_argument("--samples", default=HERE, help="directory holding the sample W dirs")
    ap.add_argument("--tools", required=True, help="PYTHONPATH for the judge (… /tools)")
    ap.add_argument("--package", required=True, help="judge package name (capjudge / judge)")
    ap.add_argument("--tmp", required=True, help="existing directory for the temp copies")
    ap.add_argument("--expect-samples", type=int, required=True,
                    help="number of sample dirs that must be found (given from outside)")
    args = ap.parse_args(argv)
    if args.expect_samples < 1:
        ap.error("--expect-samples must be >= 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
