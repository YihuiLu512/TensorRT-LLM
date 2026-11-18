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
import socket
import sys

from capcommon.exitcodes import ENV_INVALID, OK, CapkitError
from capcommon.hashing import sha256_bytes
from capcommon.jsonio import write_json_stamped
from capcommon.policy import FORBIDDEN_MOUNT_PATHS, paths_related
from capenv import gpu as capgpu

from . import jobident, snapshot

GPU_QUERY_TIMEOUT_S = 60


def _outcome(expected, passed):
    if expected == 0:
        return "NOT_RUN", False
    return ("PASS" if passed else "FAIL"), passed


def check_image(check_file):
    """image_check.tsv (written by the batch step with coreutils): key<TAB>value lines. The sha256
    was computed by capjobs.submit; the job checks that size and mtime (ns) are unchanged."""
    kv = {}
    with open(check_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                k, _, v = line.partition("\t")
                kv[k] = v
    need = ("path", "expected_sha256", "sha256_source", "expected_size", "expected_mtime_ns",
            "measured_size", "measured_mtime_ns")
    missing = [k for k in need if not kv.get(k)]
    ok = not missing and kv["expected_size"] == kv["measured_size"] \
        and kv["expected_mtime_ns"] == kv["measured_mtime_ns"]
    outcome, passed = _outcome(len(need), ok)
    return {"path": kv.get("path"), "sha256": kv.get("expected_sha256") if ok else None,
            "sha256_source": kv.get("sha256_source"), "size": kv.get("measured_size"),
            "mtime_ns": kv.get("measured_mtime_ns"), "expected_sha256": kv.get("expected_sha256"),
            "expected_size": kv.get("expected_size"),
            "expected_mtime_ns": kv.get("expected_mtime_ns"), "missing": missing,
            "passed": passed, "outcome": outcome}


def _visible_only_toward(root, allowed):
    """For a model library root that exists in the container of a model job: every directory on
    the way from root down to each allowed model dir lists nothing but the next path component
    toward an allowed dir (what pyxis creates for an identity mount of the dir); the allowed dirs
    themselves are readable and non-empty. Returns (ok, state)."""
    want = {}  # directory -> set of child names that may appear in it
    for d in allowed:
        rel = os.path.relpath(d, root)
        if rel == "." or rel.startswith(".."):
            continue
        cur = root
        for part in rel.split(os.sep):
            want.setdefault(cur, set()).add(part)
            cur = os.path.join(cur, part)
    if not want:
        return False, "present, but no allowed model dir lies under it"
    for d, names in sorted(want.items()):
        try:
            seen = set(os.listdir(d))
        except OSError as e:
            return False, "unreadable %s: %s" % (d, e)
        extra = sorted(seen - names)
        if extra:
            return False, "%s shows more than the allowed model dirs: %s" % (d, extra[:5])
    for d in allowed:
        try:
            if not (os.path.isdir(d) and os.listdir(d)):
                return False, "allowed model dir %s missing or empty" % d
        except OSError as e:
            return False, "allowed model dir %s unreadable: %s" % (d, e)
    return True, "only the allowed model dirs %s are visible" % sorted(allowed)


def check_mounts(mounts, must_read, forbidden=FORBIDDEN_MOUNT_PATHS, allow_model_dirs=()):
    """Model libraries absent or empty; every mount dst and must-read path is readable and
    non-empty (a missing mount looks like an empty directory), and neither it nor its realpath
    equals, lies under or contains a model library path.

    A whitelisted model job passes its registered model dirs as allow_model_dirs (job.env
    CAPJOBS_MODEL_DIRS, from policy.check_model_submission): a library root may then exist but
    show nothing except the path to those dirs, and a mount equal to one of them is accepted;
    any other mount that touches a library still fails."""
    allowed = sorted({os.path.normpath(d) for d in allow_model_dirs})
    if any(not d.startswith("/") for d in allowed):
        raise ValueError("allowed model dirs must be absolute: %s" % allowed)
    items = []
    for d in allowed:
        if not any(paths_related(d, f) for f in forbidden):
            items.append({"path": d, "kind": "allowed_model_dir",
                          "state": "not under a model library root", "ok": False})
    for p in forbidden:
        if not os.path.lexists(p):
            items.append({"path": p, "kind": "forbidden", "state": "absent", "ok": True})
            continue
        if any(paths_related(d, p) for d in allowed):
            ok, state = _visible_only_toward(p, [d for d in allowed if paths_related(d, p)])
            items.append({"path": p, "kind": "forbidden_allowing_model_dirs", "state": state,
                          "ok": ok})
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
            real = os.path.realpath(p)
            if kind == "mount" and os.path.normpath(p) in allowed:
                libs = []  # an allowed model dir; its realpath may resolve inside the library
            else:
                libs = [f for f in forbidden for q in (p, real) if paths_related(q, f)]
            try:
                ok = os.access(p, os.R_OK) and (not os.path.isdir(p) or bool(os.listdir(p)))
                state = "readable, non-empty" if ok else "missing, unreadable or empty"
            except OSError as e:
                ok, state = False, "error: %s" % e
            if libs:
                ok, state = False, "path or realpath %s touches model library %s" % (
                    real, sorted(set(libs)))
            items.append({"path": p, "realpath": real, "kind": kind, "state": state, "ok": ok})
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


OUTCOME = {"PASS": "PASS", "RED": "FAIL", "NOT_RUN": "NOT_RUN"}


def check_gpu_idle(nvidia_smi, max_alloc_gpus, timeout_s=GPU_QUERY_TIMEOUT_S):
    """capenv.gpu.check_visible (the one implementation of the nvidia-smi queries): visible GPUs
    == requested GPUs, positive control when >= 1 is requested, no compute process."""
    doc = capgpu.check_visible(max_alloc_gpus, timeout_s, nvsmi=nvidia_smi)
    doc["gpu_count"] = doc["visible_gpus"] or 0
    doc["passed"] = doc["status"] == "PASS"
    doc["outcome"] = OUTCOME[doc["status"]]
    return doc


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


def check_job(snap, plan_id, node, deadline_epoch, expect_node=None):
    """Snapshot verification and, for a pinned job (submit --expect-node), the Slurm node name
    `node` (SLURMD_NODENAME, the name space of sinfo, pin-next and --expect-node): another node,
    or none known ('' or 'unknown'), fails the check, so the job ends ENV_INVALID before any
    measurement. The short host name is recorded for diagnosis only."""
    ok, problems = snapshot.verify(snap)
    with open(os.path.join(snap, snapshot.MANIFEST), "rb") as f:
        msha = sha256_bytes(f.read())
    host = socket.gethostname().split(".")[0]
    node_ok = not expect_node or (node not in (None, "", "unknown") and node == expect_node)
    if not node_ok:
        problems = list(problems) + ["node %s (host %s) is not the expected node %s"
                                     % (node or "unknown", host, expect_node)]
    passed = ok and node_ok
    return {"plan_id": plan_id, "node": node, "snapshot_dir": snap, "snapshot_sha256": msha,
            "snapshot_verified": ok, "snapshot_problems": problems,
            "expect_node": expect_node or None, "hostname_short": host,
            "expect_node_ok": node_ok, "deadline_epoch": deadline_epoch, "passed": passed,
            "outcome": "PASS" if passed else "FAIL"}


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
    ap.add_argument("--allow-model-dir", action="append", default=[],
                    help="mounts: a whitelisted model job's registered model dir (job.env "
                         "CAPJOBS_MODEL_DIRS); repeatable")
    ap.add_argument("--path", action="append", default=[])
    ap.add_argument("--min-free-bytes", type=int)
    ap.add_argument("--nvidia-smi")
    ap.add_argument("--max-alloc-gpus", type=int, help="gpu-idle: GPUs requested (job.env)")
    ap.add_argument("--dest")
    ap.add_argument("--manifest", help="imports: overlay manifest naming the packages")
    ap.add_argument("--snap")
    ap.add_argument("--plan-id")
    ap.add_argument("--node")
    ap.add_argument("--deadline-epoch", type=int)
    ap.add_argument("--expect-node", default="", help="job: the Slurm node name (--node) a "
                                                     "pinned job must run on ('' = not pinned)")
    a = ap.parse_args(argv)
    try:
        ident = jobident.from_args(a)
        if a.what == "image":
            doc = check_image(a.check_file)
        elif a.what == "mounts":
            with open(a.mounts_json, "r", encoding="utf-8") as f:
                doc = check_mounts(json.load(f), a.must_read,
                                   allow_model_dirs=a.allow_model_dir)
        elif a.what == "local-disk":
            doc = check_local_disk(a.path, a.min_free_bytes)
        elif a.what == "gpu-idle":
            if not a.nvidia_smi or a.max_alloc_gpus is None:
                raise ValueError("gpu-idle needs --nvidia-smi and --max-alloc-gpus")
            doc = check_gpu_idle(a.nvidia_smi, a.max_alloc_gpus)
        elif a.what == "imports":
            with open(a.manifest, "r", encoding="utf-8") as f:
                doc = check_imports(a.dest, manifest_packages(json.load(f)))
        else:
            doc = check_job(a.snap, a.plan_id, a.node, a.deadline_epoch, a.expect_node)
    except (CapkitError, OSError, ValueError, TypeError) as e:
        print("capjobs.envfiles %s: %s" % (a.what, e), file=sys.stderr)
        return ENV_INVALID
    write_json_stamped(a.out, SCHEMAS[a.what], ident, doc)
    print("capjobs.envfiles %s: %s" % (a.what, doc["outcome"]))
    return OK if doc["passed"] else ENV_INVALID


if __name__ == "__main__":
    sys.exit(main())
