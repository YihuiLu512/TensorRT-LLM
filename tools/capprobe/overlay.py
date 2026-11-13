# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overlay manifest generator for probe tiers P0 / P1 / P2 (INTERFACES.md section 6.6).

    python -m capprobe.overlay --tier {P0,P1,P2} --probe-tree PATH --image-commit SHA
        --base-commit SHA --capkit-root <C> --capkit-sha SHA --packages capprobe,capcommon
        --groups count,lag --out PATH
        [--export-dir DIR]           required for P0 (the probe tree's working files are the
                                     patched ones, so P0 copies come from the base commit's blobs)
        [--allow-unshipped PATH]...  exact repo paths of non-Python files under tensorrt_llm/ that
                                     changed but are not installed package data (recorded)
        [--sites LIST]               P2 only: the subset of open-group sites this scenario passes

files = `git diff --name-only <image> <target>` restricted to tensorrt_llm/**.py, each with the
sha256 and size of the TARGET COMMIT's blob; the file at `src` is verified to have exactly those
bytes. P0: target = base commit. P1/P2: target = probe tree HEAD, which must contain the base
commit, have a clean tensorrt_llm/ working tree and pass capprobe.patchcheck.
Exit 3 (no manifest written) when C++ / non-Python package content changed between the image
commit and the target, when a tensorrt_llm/**.py file was deleted or renamed, or on any other
inconsistency.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

from capcommon import clock, exitcodes, jsonio

from . import patchcheck as _patchcheck
from . import sites as _sites

SCHEMA = "capkit.overlay/1"
TIERS = ("P0", "P1", "P2")
REQUIRED_PACKAGES_PROBED = ("capprobe", "capcommon")


class OverlayError(exitcodes.ConfigError):
    pass


def _git(tree, *args, binary=False):
    r = subprocess.run(["git", "-C", tree, *args], capture_output=True, check=False)
    if r.returncode != 0:
        raise OverlayError("git -C %s %s failed: %s"
                           % (tree, " ".join(args), r.stderr.decode(errors="replace").strip()))
    return r.stdout if binary else r.stdout.decode("utf-8")


def resolve_commit(tree, rev):
    return _git(tree, "rev-parse", "--verify", "%s^{commit}" % rev).strip()


def is_ancestor(tree, a, b):
    r = subprocess.run(["git", "-C", tree, "merge-base", "--is-ancestor", a, b],
                       capture_output=True, check=False)
    if r.returncode not in (0, 1):
        raise OverlayError("git merge-base failed: %s" % r.stderr.decode(errors="replace"))
    return r.returncode == 0


def classify_changes(name_status_text):
    """Split `git diff --name-status --no-renames` output into (py, non_python, deleted_py)."""
    py, non_python, deleted_py = [], [], []
    for ln in name_status_text.splitlines():
        st, rel = ln.split("\t", 1)
        in_pkg = rel.startswith("tensorrt_llm/")
        if in_pkg and rel.endswith(".py"):
            (deleted_py if st == "D" else py).append(rel)
        elif in_pkg or rel.startswith("cpp/") or (
                "/" not in rel and rel.startswith("requirements") and rel.endswith(".txt")):
            non_python.append(rel)
    return sorted(py), sorted(non_python), sorted(deleted_py)


def _src_path(path, capkit_root):
    """Path relative to capkit_root when inside it, else absolute."""
    ap, root = os.path.abspath(path), os.path.abspath(capkit_root)
    if ap == root or ap.startswith(root + os.sep):
        return os.path.relpath(ap, root)
    return ap


def _entry(src_abs, dst, blob, capkit_root):
    with open(src_abs, "rb") as f:
        on_disk = f.read()
    if on_disk != blob:
        raise OverlayError("%s differs from the committed blob (uncommitted change?)" % src_abs)
    return {"src": _src_path(src_abs, capkit_root), "dst": dst,
            "sha256": hashlib.sha256(blob).hexdigest(), "size": len(blob)}


def _parse_list(text):
    if text is None or text.strip() == "":
        return []
    items = [x.strip() for x in text.split(",")]
    if any(not x for x in items) or len(set(items)) != len(items):
        raise OverlayError("bad comma list %r (empty or repeated item)" % text)
    return items


def _capkit_files(capkit_root, capkit_sha, packages):
    head = resolve_commit(capkit_root, "HEAD")
    if head != capkit_sha:
        raise OverlayError("capkit repo HEAD %s != --capkit-sha %s" % (head, capkit_sha))
    out = []
    for pkg in packages:
        pdir = "tools/%s" % pkg
        dirty = _git(capkit_root, "status", "--porcelain", "--untracked-files=all", "--", pdir)
        dirty = [ln for ln in dirty.splitlines() if not ln.endswith(".pyc")
                 and "__pycache__/" not in ln]
        if dirty:
            raise OverlayError("%s has uncommitted changes: %s" % (pdir, dirty[:5]))
        names = [n for n in _git(capkit_root, "ls-tree", "-r", "--name-only", capkit_sha, "--",
                                 pdir).splitlines() if n.endswith(".py")]
        if not names:
            raise OverlayError("package %s has no committed .py files at %s" % (pkg, capkit_sha))
        for rel in sorted(names):
            blob = _git(capkit_root, "show", "%s:%s" % (capkit_sha, rel), binary=True)
            dst = rel[len("tools/"):]
            out.append(_entry(os.path.join(capkit_root, rel), dst, blob, capkit_root))
    return out


def build_manifest(tier, probe_tree, image_commit, base_commit, capkit_root, capkit_sha,
                   packages, groups, export_dir=None, allow_unshipped=(), sites=None):
    """Return the manifest dict; raises OverlayError on every rule violation."""
    if tier not in TIERS:
        raise OverlayError("--tier must be one of %s" % (TIERS,))
    image = resolve_commit(probe_tree, image_commit)
    base = resolve_commit(probe_tree, base_commit)
    capkit = resolve_commit(capkit_root, capkit_sha)
    groups = list(groups)
    if tier in ("P0", "P1") and groups:
        raise OverlayError("tier %s requires empty --groups, got %s" % (tier, groups))
    if tier == "P2":
        if not groups:
            raise OverlayError("tier P2 requires non-empty --groups")
        bad = sorted(set(groups) - set(_sites.IMPLEMENTED_GROUPS))
        if bad:
            raise OverlayError("groups %s are not implemented (implemented: %s)"
                               % (bad, list(_sites.IMPLEMENTED_GROUPS)))
        groups = [g for g in _sites.ALL_GROUPS if g in groups]
    if sites is not None and tier != "P2":
        raise OverlayError("--sites is only valid with tier P2")
    if tier in ("P1", "P2"):
        missing = [p for p in REQUIRED_PACKAGES_PROBED if p not in packages]
        if missing:
            raise OverlayError("tier %s needs packages %s in --packages" % (tier, missing))
    patch_report = None
    if tier == "P0":
        if not export_dir:
            raise OverlayError("tier P0 requires --export-dir (blobs of the base commit)")
        target = base
    else:
        target = resolve_commit(probe_tree, "HEAD")
        if not is_ancestor(probe_tree, base, target):
            raise OverlayError("probe tree HEAD %s does not contain base %s" % (target, base))
        dirty = _git(probe_tree, "status", "--porcelain", "--untracked-files=all", "--",
                     "tensorrt_llm")
        if dirty.strip():
            raise OverlayError("probe tree tensorrt_llm/ is not clean: %s"
                               % dirty.splitlines()[:5])
        patch_report = _patchcheck.run_check(probe_tree, base, target)
        if patch_report["status"] != "PASS":
            raise OverlayError("patchcheck %s: %s" % (patch_report["status"],
                                                      patch_report["violations"][:5]))
    ns = _git(probe_tree, "diff", "--name-status", "--no-renames", image, target)
    py, non_python, deleted_py = classify_changes(ns)
    if deleted_py:
        raise OverlayError("tensorrt_llm/**.py deleted between image and target (a copy overlay "
                           "cannot remove them): %s" % deleted_py)
    allowed = sorted(set(allow_unshipped))
    stale = [p for p in allowed if p not in non_python]
    if stale:
        raise OverlayError("--allow-unshipped names paths that did not change: %s" % stale)
    bad_alloc = [p for p in allowed if not p.startswith("tensorrt_llm/")]
    if bad_alloc:
        raise OverlayError("--allow-unshipped only applies to files under tensorrt_llm/: %s"
                           % bad_alloc)
    non_python_changed = [p for p in non_python if p not in allowed]
    if non_python_changed:
        raise OverlayError("C++ / non-Python package content changed between image %s and "
                           "target %s (rebuild the image; ask the user first): %s"
                           % (image, target, non_python_changed))
    if not py:
        raise OverlayError("no tensorrt_llm/**.py differs between image and target; nothing to "
                           "overlay (wrong commits?)")
    files = []
    for rel in py:
        blob = _git(probe_tree, "show", "%s:%s" % (target, rel), binary=True)
        if tier == "P0":
            path = os.path.join(export_dir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            jsonio.atomic_write_bytes(path, blob)
        else:
            path = os.path.join(probe_tree, rel)
        files.append(_entry(path, rel, blob, capkit_root))
    capkit_files = _capkit_files(capkit_root, capkit, packages)
    if tier == "P2":
        open_sites = _sites.sites_for_groups(groups)
        if sites is None:
            chosen, selection = open_sites, "all_sites_of_open_groups"
        else:
            unknown = sorted(set(sites) - set(open_sites))
            if unknown:
                raise OverlayError("--sites %s are not sites of the open groups %s"
                                   % (unknown, groups))
            chosen = {k: open_sites[k] for k in sites}
            selection = "cli"
        if not chosen:
            raise OverlayError("P2 manifest would list no sites")
    else:
        chosen, selection = {}, "none"
    return {"schema": SCHEMA, "tier": tier, "image_commit": image, "base_commit": base,
            "target_commit": target, "image_is_ancestor_of_target": is_ancestor(
                probe_tree, image, target),
            "groups": groups, "files": files, "capkit_packages": capkit_files,
            "capkit_code_sha": capkit, "non_python_changed": [],
            "unshipped_allowed": allowed, "sites": chosen, "sites_selection": selection,
            "patchcheck": None if patch_report is None else {
                k: patch_report[k] for k in ("status", "expected_site_count",
                                             "found_site_count", "rev")},
            "created_wall": clock.wall_iso()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capprobe.overlay",
                                 description="overlay manifest for probe tiers P0/P1/P2")
    ap.add_argument("--tier", required=True, choices=TIERS)
    ap.add_argument("--probe-tree", required=True)
    ap.add_argument("--image-commit", required=True)
    ap.add_argument("--base-commit", required=True)
    ap.add_argument("--capkit-root", required=True)
    ap.add_argument("--capkit-sha", required=True)
    ap.add_argument("--packages", required=True, help="comma list, e.g. capprobe,capcommon")
    ap.add_argument("--groups", required=True, help="comma list; empty string for P0/P1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--allow-unshipped", action="append", default=[])
    ap.add_argument("--sites", default=None, help="P2 only: comma list of site ids")
    args = ap.parse_args(argv)
    try:
        man = build_manifest(args.tier, args.probe_tree, args.image_commit, args.base_commit,
                             args.capkit_root, args.capkit_sha, _parse_list(args.packages),
                             _parse_list(args.groups), export_dir=args.export_dir,
                             allow_unshipped=args.allow_unshipped,
                             sites=None if args.sites is None else _parse_list(args.sites))
    except exitcodes.CapkitError as e:
        print("capprobe.overlay: %s" % e, file=sys.stderr)
        return e.exit_code
    data = (json.dumps(man, indent=1, sort_keys=True) + "\n").encode("utf-8")
    jsonio.atomic_write_bytes(args.out, data)
    print("capprobe.overlay: wrote %s (%d tensorrt_llm files, %d capkit files, %d sites)"
          % (args.out, len(man["files"]), len(man["capkit_packages"]), len(man["sites"])))
    return exitcodes.OK


if __name__ == "__main__":
    sys.exit(main())
