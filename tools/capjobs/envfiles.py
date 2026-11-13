# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job-level environment files written by the inner script (INTERFACES.md 5.5, B6 rows):
env/image.json, env/mounts.json, env/local_disk.json, env/gpu_idle.json, env/import_origin.json
and <job dir>/job.json. Each check derives its expected item count from its inputs and reports
outcome NOT_RUN (and exit 6) when that count is 0; exit 6 (ENV_INVALID) on any failure.

    python -m capjobs.envfiles <image|mounts|local-disk|gpu-idle|imports|job> ... <identity args>
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

from capcommon.exitcodes import ENV_INVALID, OK, CapkitError
from capcommon.hashing import sha256_bytes
from capcommon.jsonio import write_json_stamped
from capcommon.policy import FORBIDDEN_MOUNT_PATHS

from . import jobident, snapshot


def _outcome(expected, passed):
    if expected == 0:
        return "NOT_RUN", False
    return ("PASS" if passed else "FAIL"), passed


def check_image(check_file):
    """image_check.tsv (written by the batch step with coreutils): key<TAB>value lines."""
    kv = {}
    with open(check_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                k, _, v = line.partition("\t")
                kv[k] = v
    need = ("path", "expected_sha256", "expected_size", "measured_sha256", "measured_size")
    missing = [k for k in need if not kv.get(k)]
    ok = not missing and kv["expected_sha256"] == kv["measured_sha256"] \
        and kv["expected_size"] == kv["measured_size"]
    outcome, passed = _outcome(len(need), ok)
    return {"path": kv.get("path"), "sha256": kv.get("measured_sha256"),
            "size": kv.get("measured_size"), "expected_sha256": kv.get("expected_sha256"),
            "expected_size": kv.get("expected_size"), "missing": missing,
            "passed": passed, "outcome": outcome}


def check_mounts(mounts, must_read, forbidden=FORBIDDEN_MOUNT_PATHS):
    """Model libraries absent or empty; every mount dst and must-read path is readable and
    non-empty (a missing mount looks like an empty directory)."""
    items = []
    for p in forbidden:
        if not os.path.lexists(p):
            items.append({"path": p, "kind": "forbidden", "state": "absent", "ok": True})
            continue
        try:
            empty = os.path.isdir(p) and not os.listdir(p)
        except OSError as e:
            items.append({"path": p, "kind": "forbidden", "state": "unreadable: %s" % e,
                          "ok": False})
            continue
        items.append({"path": p, "kind": "forbidden", "state": "empty" if empty else "present",
                      "ok": empty})
    for kind, paths in (("mount", [m.split(":")[1] for m in mounts]), ("must_read", must_read)):
        for p in paths:
            try:
                ok = os.access(p, os.R_OK) and (not os.path.isdir(p) or bool(os.listdir(p)))
                state = "readable, non-empty" if ok else "missing, unreadable or empty"
            except OSError as e:
                ok, state = False, "error: %s" % e
            items.append({"path": p, "kind": kind, "state": state, "ok": ok})
    outcome, passed = _outcome(len(items), all(i["ok"] for i in items))
    return {"mounts": list(mounts), "items": items, "expected_items": len(items),
            "passed": passed, "outcome": outcome}


def check_local_disk(paths, min_free):
    items = []
    for p in paths:
        try:
            u = shutil.disk_usage(p)
            items.append({"path": p, "total": u.total, "used": u.used, "free": u.free,
                          "ok": u.free >= min_free})
        except OSError as e:
            items.append({"path": p, "error": str(e), "ok": False})
    outcome, passed = _outcome(len(items), all(i["ok"] for i in items))
    return {"min_free_bytes": min_free, "items": items, "passed": passed, "outcome": outcome}


def _run(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
        return r.returncode, r.stdout, r.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, "", str(e)


def check_gpu_idle(nvidia_smi, expect_gpus):
    """Positive control: the GPU query lists at least one GPU. Pass: no compute processes."""
    q1 = [nvidia_smi, "--query-gpu=index,name,utilization.gpu,memory.used",
          "--format=csv,noheader,nounits"]
    q2 = [nvidia_smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]
    rc1, out1, err1 = _run(q1)
    rc2, out2, err2 = _run(q2)
    gpus = [[c.strip() for c in ln.split(",")] for ln in out1.splitlines() if ln.strip()] \
        if rc1 == 0 else []
    apps = [ln.strip() for ln in out2.splitlines() if ln.strip()] if rc2 == 0 else None
    positive = rc1 == 0 and len(gpus) >= 1
    passed = positive and rc2 == 0 and apps == []
    return {"queries": [{"cmd": q1, "rc": rc1, "stdout": out1, "stderr": err1},
                        {"cmd": q2, "rc": rc2, "stdout": out2, "stderr": err2}],
            "gpu_count": len(gpus), "gpus": gpus, "compute_apps": apps,
            "expected_gpus_requested": expect_gpus, "positive_control_ok": positive,
            "passed": passed, "outcome": "PASS" if passed else
            ("NOT_RUN" if not positive else "FAIL")}


def manifest_packages(manifest_doc):
    """Top-level package names of the overlay's capkit_packages entries (capjobs included)."""
    names = sorted({e["dst"].split("/")[0] for e in manifest_doc.get("capkit_packages") or []})
    return names


def check_imports(dest, packages):
    """Every capkit package must be imported from dest (PYTHONPATH unset; INTERFACES.md 1.2)."""
    dest_real = os.path.realpath(dest)
    items = []
    for name in packages:
        spec = importlib.util.find_spec(name)
        origin = spec.origin if spec is not None else None
        ok = origin is not None and os.path.realpath(origin).startswith(dest_real + os.sep)
        items.append({"package": name, "origin": origin, "ok": ok})
    outcome, passed = _outcome(len(items), all(i["ok"] for i in items))
    return {"dest": dest, "pythonpath": os.environ.get("PYTHONPATH"), "items": items,
            "passed": passed, "outcome": outcome}


def check_job(snap, plan_id, node, deadline_epoch):
    ok, problems = snapshot.verify(snap)
    with open(os.path.join(snap, snapshot.MANIFEST), "rb") as f:
        msha = sha256_bytes(f.read())
    return {"plan_id": plan_id, "node": node, "snapshot_dir": snap, "snapshot_sha256": msha,
            "snapshot_verified": ok, "snapshot_problems": problems,
            "deadline_epoch": deadline_epoch, "passed": ok, "outcome": "PASS" if ok else "FAIL"}


SCHEMAS = {"image": "capkit.image/1", "mounts": "capkit.mounts/1",
           "local-disk": "capkit.local_disk/1", "gpu-idle": "capkit.gpu_idle/1",
           "imports": "capkit.import_origin/1", "job": "capkit.job/1"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capjobs.envfiles")
    ap.add_argument("what", choices=sorted(SCHEMAS))
    jobident.add_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-file")
    ap.add_argument("--mounts-json")
    ap.add_argument("--must-read", action="append", default=[])
    ap.add_argument("--path", action="append", default=[])
    ap.add_argument("--min-free-bytes", type=int)
    ap.add_argument("--nvidia-smi")
    ap.add_argument("--expect-gpus", type=int)
    ap.add_argument("--dest")
    ap.add_argument("--manifest", help="imports: overlay manifest naming the packages")
    ap.add_argument("--snap")
    ap.add_argument("--plan-id")
    ap.add_argument("--node")
    ap.add_argument("--deadline-epoch", type=int)
    a = ap.parse_args(argv)
    try:
        ident = jobident.from_args(a)
        if a.what == "image":
            doc = check_image(a.check_file)
        elif a.what == "mounts":
            with open(a.mounts_json, "r", encoding="utf-8") as f:
                doc = check_mounts(json.load(f), a.must_read)
        elif a.what == "local-disk":
            doc = check_local_disk(a.path, a.min_free_bytes)
        elif a.what == "gpu-idle":
            doc = check_gpu_idle(a.nvidia_smi, a.expect_gpus)
        elif a.what == "imports":
            with open(a.manifest, "r", encoding="utf-8") as f:
                doc = check_imports(a.dest, manifest_packages(json.load(f)))
        else:
            doc = check_job(a.snap, a.plan_id, a.node, a.deadline_epoch)
    except (CapkitError, OSError, ValueError, TypeError) as e:
        print("capjobs.envfiles %s: %s" % (a.what, e), file=sys.stderr)
        return ENV_INVALID
    write_json_stamped(a.out, SCHEMAS[a.what], ident, doc)
    print("capjobs.envfiles %s: %s" % (a.what, doc["outcome"]))
    return OK if doc["passed"] else ENV_INVALID


if __name__ == "__main__":
    sys.exit(main())
