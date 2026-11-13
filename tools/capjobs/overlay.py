# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy the overlay (INTERFACES.md 6.6) into the container's site-packages and verify every file
byte for byte; writes env/overlay_verify.json (capkit.overlay_verify/1). Runs inside the
container with PYTHONPATH=<snapshot>/tools:<snapshot>/jobs (only for this step).

    python -m capjobs.overlay --manifest <snap>/overlay_manifest.json --source-root <snap> \
        --dest <purelib> --out <job dir>/env/overlay_verify.json <identity args>

Exit 0 = every entry copied and read back identical; 6 (ENV_INVALID) otherwise.
"""

import argparse
import json
import os
import sys

from capcommon.exitcodes import ENV_INVALID, OK, CapkitError
from capcommon.hashing import sha256_bytes
from capcommon.jsonio import write_json_stamped

from . import jobident
from .snapshot import overlay_entries

SCHEMA = "capkit.overlay_verify/1"


def _safe_rel(rel):
    return not os.path.isabs(rel) and ".." not in rel.split("/") and rel not in ("", ".")


def apply(manifest_doc, source_root, dest):
    """Returns (results list, passed). Never raises for per-file problems; records them."""
    results = []
    entries = []
    for key in ("files", "capkit_packages"):
        for e in manifest_doc.get(key) or []:
            entries.append((key, e))
    overlay_entries(manifest_doc)  # shape check; raises on a malformed manifest
    for key, e in entries:
        r = {"list": key, "src": e["src"], "dst": e["dst"], "sha256": e["sha256"],
             "size": e["size"], "ok": False, "problem": None}
        results.append(r)
        if not _safe_rel(e["dst"]) or not _safe_rel(e["src"]):
            r["problem"] = "unsafe path"
            continue
        src = os.path.join(source_root, e["src"])
        try:
            with open(src, "rb") as f:
                data = f.read()
        except OSError as ex:
            r["problem"] = "cannot read source: %s" % ex
            continue
        if len(data) != e["size"] or sha256_bytes(data) != e["sha256"]:
            r["problem"] = "source differs from manifest"
            continue
        dst = os.path.join(dest, e["dst"])
        r["existed_before"] = os.path.exists(dst)
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".capjobs-tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            with open(dst, "rb") as f:
                back = f.read()
        except OSError as ex:
            r["problem"] = "copy failed: %s" % ex
            continue
        r["readback_sha256"] = sha256_bytes(back)
        if back != data:
            r["problem"] = "read-back bytes differ"
            continue
        r["ok"] = True
    expected = len(entries)
    passed = expected > 0 and all(r["ok"] for r in results)
    return results, passed, expected


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capjobs.overlay")
    jobident.add_args(ap)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        ident = jobident.from_args(a)
        with open(a.manifest, "r", encoding="utf-8") as f:
            doc = json.load(f)
        results, passed, expected = apply(doc, a.source_root, a.dest)
    except (CapkitError, OSError, ValueError) as e:
        print("capjobs.overlay: %s" % e, file=sys.stderr)
        return ENV_INVALID
    write_json_stamped(a.out, SCHEMA, ident, {
        "manifest": os.path.abspath(a.manifest), "dest": os.path.abspath(a.dest),
        "tier": doc.get("tier"), "expected_files": expected,
        "verified_files": sum(1 for r in results if r["ok"]), "passed": passed,
        "outcome": "PASS" if passed else ("NOT_RUN" if expected == 0 else "FAIL"),
        "files": results})
    print("capjobs.overlay: %s %d/%d" % ("ok" if passed else "FAILED",
                                         sum(1 for r in results if r["ok"]), expected))
    return OK if passed else ENV_INVALID


if __name__ == "__main__":
    sys.exit(main())
