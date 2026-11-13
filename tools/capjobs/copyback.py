# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy node-local data back to the work dir and verify it (plan RESOURCES 7, CONSTRAINTS 6,
INTERFACES.md 5.9).

Verification compares the full listing of source and copy: every relative path with its kind
(file / symlink / dir), file sizes and symlink targets; it records the file count and total bytes
(the expected values come from the source walk, not from the copy). A source with zero regular
files is never "passed" for a tree; the job-level sweep reports outcome EMPTY instead, which is
the only zero-count case allowed to unlock deletion (nothing can be lost).

    python -m capjobs.copyback sweep --local-root L --shm-root S --dest-root R --job-dir JD \
        --out <job dir>/copyback/final.verify.json
"""

import argparse
import json
import os
import shutil
import sys

from capcommon.clock import wall_iso
from capcommon.exitcodes import OK, RUNTIME
from capcommon.jsonio import atomic_write_bytes, dumps

from .snapshot import make_read_only

SCHEMA = "capkit.copyback_verify/1"


def listing(top):
    """{relpath: ["f", size] | ["l", target] | ["d"]} for everything under top (not top)."""
    out = {}
    for root, dirs, files in os.walk(top, followlinks=False):
        for d in list(dirs):
            p = os.path.join(root, d)
            rel = os.path.relpath(p, top)
            if os.path.islink(p):
                out[rel] = ["l", os.readlink(p)]
                dirs.remove(d)
            else:
                out[rel] = ["d"]
        for n in files:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, top)
            if os.path.islink(p):
                out[rel] = ["l", os.readlink(p)]
            else:
                out[rel] = ["f", os.path.getsize(p)]
    return out


# Top-level entries a launch tree gains only after copy-back (STATE is appended to the copy).
DST_ONLY_TOP = ("STATE",)


def compare(src, dst):
    a = listing(src)
    b = listing(dst) if os.path.isdir(dst) else {}
    b = {k: v for k, v in b.items() if k not in DST_ONLY_TOP}
    files = [k for k, v in a.items() if v[0] == "f"]
    diff = sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b) if a[k] != b[k])
    n_files = len(files)
    total = sum(a[k][1] for k in files)
    dst_files = [k for k, v in b.items() if v[0] == "f"]
    passed = n_files > 0 and not diff
    return {"src": src, "dst": dst, "files_expected": n_files, "bytes_expected": total,
            "files_copied": len(dst_files), "bytes_copied": sum(b[k][1] for k in dst_files),
            "entries_expected": len(a), "differences": diff[:200], "n_differences": len(diff),
            "passed": passed, "outcome": "PASS" if passed else
            ("NOT_RUN" if n_files == 0 else "FAIL"), "checked_at": wall_iso()}


def copy_tree(src, dst):
    """Copy src to dst (dst must not exist) and verify. Returns the verify record."""
    if os.path.lexists(dst):
        raise FileExistsError("copy-back destination exists: %s" % dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copytree(src, dst, symlinks=True)
    return compare(src, dst)


def lock_raw(dst_tree):
    """chmod -R a-w on <tree>/raw only (STATE and judge/ stay writable)."""
    raw = os.path.join(dst_tree, "raw")
    if os.path.isdir(raw) and not os.path.islink(raw):
        make_read_only(raw)
        return True
    return False


def write_record(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    full = {"schema": SCHEMA}
    full.update(doc)
    atomic_write_bytes(path, (dumps(full) + "\n").encode("utf-8"))


def _regular_files(top):
    if not os.path.isdir(top):
        return 0
    return sum(1 for v in listing(top).values() if v[0] == "f")


def sweep(local_root, shm_root, dest_root, job_dir):
    """Final pass: every tree under <local_root>/runs and every file under shm_root must have a
    verified copy. Trees already copied are re-verified, not copied again."""
    local_files = _regular_files(local_root)
    shm_files = _regular_files(shm_root)
    if local_files == 0 and shm_files == 0:
        return {"local_root": local_root, "shm_root": shm_root, "local_regular_files": 0,
                "shm_regular_files": 0, "trees": [], "shm": None, "passed": True,
                "outcome": "EMPTY", "checked_at": wall_iso()}
    trees = []
    runs = os.path.join(local_root, "runs")
    pairs = [(os.path.join(runs, n), os.path.join(dest_root, n))
             for n in (sorted(os.listdir(runs)) if os.path.isdir(runs) else [])]
    others = sorted(n for n in os.listdir(local_root) if n != "runs") \
        if os.path.isdir(local_root) else []
    pairs += [(os.path.join(local_root, n), os.path.join(job_dir, "local_leftovers", n))
              for n in others if _regular_files(os.path.join(local_root, n)) > 0
              or os.path.isfile(os.path.join(local_root, n))]
    for src, dst in pairs:
        if os.path.isfile(src) or os.path.islink(src):
            rec = {"src": src, "dst": dst, "passed": False, "outcome": "FAIL",
                   "error": "stray top-level file in the local root (not copied)"}
            trees.append(rec)
            continue
        try:
            rec = compare(src, dst) if os.path.lexists(dst) else copy_tree(src, dst)
        except OSError as e:
            rec = {"src": src, "dst": dst, "passed": False, "outcome": "FAIL", "error": str(e)}
        trees.append(rec)
    shm_rec = None
    if os.path.isdir(shm_root) and _regular_files(shm_root) > 0:
        dst = os.path.join(job_dir, "shm_leftovers")
        try:
            shm_rec = compare(shm_root, dst) if os.path.lexists(dst) else copy_tree(shm_root, dst)
        except OSError as e:
            shm_rec = {"src": shm_root, "dst": dst, "passed": False, "outcome": "FAIL",
                       "error": str(e)}
    parts = [t["passed"] or (t.get("outcome") == "NOT_RUN" and t.get("n_differences") == 0)
             for t in trees] + ([shm_rec["passed"]] if shm_rec else [])
    uncovered = local_files - sum(t.get("files_expected", 0) for t in trees)
    passed = bool(parts) and all(parts) and uncovered == 0
    outcome = "PASS" if passed else "FAIL"
    return {"local_root": local_root, "shm_root": shm_root, "local_regular_files": local_files,
            "shm_regular_files": shm_files, "trees": trees, "shm": shm_rec, "passed": passed,
            "outcome": outcome, "checked_at": wall_iso()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capjobs.copyback")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--local-root", required=True)
    s.add_argument("--shm-root", required=True)
    s.add_argument("--dest-root", required=True)
    s.add_argument("--job-dir", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    doc = sweep(a.local_root, a.shm_root, a.dest_root, a.job_dir)
    write_record(a.out, doc)
    print("capjobs.copyback sweep: %s" % doc["outcome"])
    print(json.dumps({"passed": doc["passed"], "trees": len(doc["trees"])}))
    return OK if doc["passed"] else RUNTIME


if __name__ == "__main__":
    sys.exit(main())
