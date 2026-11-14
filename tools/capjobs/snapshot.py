# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable per-submission snapshot (plan RESOURCES section 4: the job runs the snapshot, later
edits of the sources never reach a queued or running job).

Layout of <snapshot_root>/<plan_id>/:
  jobs/ tools/          copies of <code_root>/jobs and <code_root>/tools (no __pycache__, *.pyc,
                        *.draft, and not jobs/prep_tokenizer.sbatch: the tokenizer copy job has
                        its own plan dir, see capjobs.submit prep-tokenizer)
  <include relpaths>    configs and other inputs named by the caller (relative to code_root)
  <overlay srcs>        every src of the overlay manifest (probe-tree files, capkit packages,
                        stub modules), sha-verified
  overlay_manifest.json launch_plan.json mounts.json job.env   generated / copied inputs
  MANIFEST.json         [[relpath, size, sha256], ...] of every other file, sorted
snapshot_sha256 = sha256 of MANIFEST.json's bytes. Everything is chmod a-w after writing.
"""

import json
import os
import shutil
import stat

from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_bytes, sha256_file
from capcommon.jsonio import dumps

MANIFEST = "MANIFEST.json"
SKIP_DIRS = ("__pycache__",)
SKIP_SUFFIXES = (".pyc", ".draft")
SKIP_RELPATHS = ("jobs/prep_tokenizer.sbatch",)
GENERATED = ("overlay_manifest.json", "launch_plan.json", "mounts.json", "job.env", MANIFEST)


class SnapshotError(ConfigError):
    pass


def _check_rel(rel):
    if os.path.isabs(rel) or rel.startswith("..") or "/../" in rel or rel in ("", "."):
        raise SnapshotError("path must be relative inside the code root: %r" % rel)
    if rel.split("/")[0] in GENERATED:
        raise SnapshotError("path %r collides with a generated snapshot file" % rel)


def _walk_files(base, rel_top):
    """Yield (abs src, relpath) of regular files under base/rel_top; refuse symlinks."""
    top = os.path.join(base, rel_top)
    if os.path.islink(top):
        raise SnapshotError("symlink not allowed in snapshot input: %s" % rel_top)
    if os.path.isfile(top):
        yield top, rel_top
        return
    if not os.path.isdir(top):
        raise SnapshotError("snapshot input does not exist: %s" % top)
    for root, dirs, files in os.walk(top):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for d in dirs:
            if os.path.islink(os.path.join(root, d)):
                raise SnapshotError("symlinked dir in snapshot input: %s" % os.path.join(root, d))
        for name in sorted(files):
            if name.endswith(SKIP_SUFFIXES):
                continue
            src = os.path.join(root, name)
            if os.path.relpath(src, base) in SKIP_RELPATHS:
                continue
            if os.path.islink(src) or not os.path.isfile(src):
                raise SnapshotError("not a regular file: %s" % src)
            yield src, os.path.relpath(src, base)


STUB_KEYS = ("key", "repo_path", "commit", "src", "sha256", "size")


def overlay_entries(manifest_doc):
    """(src relpath, sha256, size) of every file the overlay manifest names: files and
    capkit_packages (installed into site-packages) and stub_modules (used in place from the
    snapshot, never installed; see capjobs.overlay)."""
    out = []
    for key in ("files", "capkit_packages"):
        for e in manifest_doc.get(key) or []:
            if not isinstance(e, dict) or not all(k in e for k in ("src", "dst", "sha256", "size")):
                raise SnapshotError("overlay manifest %s entry lacks src/dst/sha256/size" % key)
            _check_rel(e["src"])
            out.append((e["src"], e["sha256"], e["size"]))
    stubs = manifest_doc.get("stub_modules") or []
    if not isinstance(stubs, list):
        raise SnapshotError("overlay manifest stub_modules must be a list")
    keys = [e.get("key") if isinstance(e, dict) else None for e in stubs]
    if len(set(keys)) != len(keys):
        raise SnapshotError("overlay manifest stub_modules repeats a key: %s" % keys)
    for e in stubs:
        if not isinstance(e, dict) or set(e) != set(STUB_KEYS):
            raise SnapshotError("overlay manifest stub_modules entry must have exactly %s"
                                % (STUB_KEYS,))
        _check_rel(e["src"])
        out.append((e["src"], e["sha256"], e["size"]))
    if not out:
        raise SnapshotError("overlay manifest lists no files")
    return out


def plan_files(code_root, includes, manifest_doc):
    """Ordered {relpath: abs src} of everything to copy (no writes). Duplicates must agree."""
    plan = {}
    tops = ["jobs", "tools"] + list(includes)
    for t in tops:
        _check_rel(t)
        for src, rel in _walk_files(code_root, t):
            plan[rel] = src
    for rel, _sha, _size in overlay_entries(manifest_doc):
        src = os.path.join(code_root, rel)
        if os.path.islink(src) or not os.path.isfile(src):
            raise SnapshotError("overlay source missing or not a regular file: %s" % src)
        plan.setdefault(rel, src)
    return plan


def create(snapshot_root, plan_id, code_root, includes, manifest_path, generated):
    """Copy everything, verify the overlay entries, write MANIFEST.json, make read-only.

    generated: {name: bytes} for the generated top-level files (launch_plan.json, mounts.json,
    job.env); overlay_manifest.json is copied byte-for-byte from manifest_path.
    Returns (snap_dir, snapshot_sha256, manifest list).
    """
    with open(manifest_path, "rb") as f:
        manifest_bytes = f.read()
    manifest_doc = json.loads(manifest_bytes.decode("utf-8"))
    plan = plan_files(code_root, includes, manifest_doc)
    snap = os.path.join(snapshot_root, plan_id)
    os.makedirs(snapshot_root, exist_ok=True)
    os.mkdir(snap)  # fails if it exists: a snapshot is never reused or overwritten
    for rel, src in sorted(plan.items()):
        dst = os.path.join(snap, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
    for rel, sha, size in overlay_entries(manifest_doc):
        p = os.path.join(snap, rel)
        if os.path.getsize(p) != size or sha256_file(p) != sha:
            raise SnapshotError("overlay source %s does not match the overlay manifest" % rel)
    files = dict(generated)
    files["overlay_manifest.json"] = manifest_bytes
    for name, data in files.items():
        if name not in GENERATED or name == MANIFEST:
            raise SnapshotError("unexpected generated file %r" % name)
        with open(os.path.join(snap, name), "wb") as f:
            f.write(data)
    entries = []
    for root, _dirs, names in os.walk(snap):
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, snap)
            entries.append([rel, os.path.getsize(p), sha256_file(p)])
    entries.sort()
    mbytes = (dumps(entries) + "\n").encode("utf-8")
    with open(os.path.join(snap, MANIFEST), "wb") as f:
        f.write(mbytes)
    make_read_only(snap)
    return snap, sha256_bytes(mbytes), entries


def make_read_only(top):
    """chmod a-w on every file and directory under top (directories last)."""
    wbits = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for root, dirs, files in os.walk(top, topdown=False):
        for n in files:
            p = os.path.join(root, n)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode & ~wbits)
        for d in dirs:
            p = os.path.join(root, d)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode & ~wbits)
    os.chmod(top, os.stat(top).st_mode & ~wbits)


def verify(snap):
    """Recompute MANIFEST.json against the files. Returns (ok, problems)."""
    with open(os.path.join(snap, MANIFEST), "rb") as f:
        entries = json.loads(f.read().decode("utf-8"))
    problems = []
    listed = set()
    for rel, size, sha in entries:
        listed.add(rel)
        p = os.path.join(snap, rel)
        if not os.path.isfile(p) or os.path.getsize(p) != size or sha256_file(p) != sha:
            problems.append(rel)
    for root, _dirs, names in os.walk(snap):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), snap)
            if rel != MANIFEST and rel not in listed:
                problems.append("unlisted:" + rel)
    if not entries:
        problems.append("empty manifest")
    return not problems, problems
