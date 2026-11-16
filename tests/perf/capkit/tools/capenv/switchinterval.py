# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capenv.switchinterval: D13 report (design 2.5 D13; INTERFACES.md 5.5 switchinterval.json).

Collects sys.getswitchinterval() of every process from its D11 version record copies in
raw/env/versions/*.json (each file is either a bare version record or a stamped JSON with a
"versions" object). Expected count comes from the caller (sum of launch.json expected_procs);
0 expected -> NOT_RUN; fewer records than expected or a record without switchinterval_s -> RED.
Values are reported, not judged (the interpreter default is kept on purpose).

    python -m capenv.switchinterval <identity> --versions-dir DIR --expected-count N --out PATH
"""

import argparse
import json
import os
import sys

from capcommon import exitcodes
from capcommon.cli import add_identity_args
from capcommon.exitcodes import ConfigError
from capcommon.identity import Identity
from capcommon.jsonio import read_json, write_json_stamped

SCHEMA = "capkit.switchinterval/1"


def collect(versions_dir, expected):
    rows, bad = [], []
    try:
        names = sorted(n for n in os.listdir(versions_dir) if n.endswith(".json"))
    except FileNotFoundError:
        names = []
    for n in names:
        try:
            doc = read_json(os.path.join(versions_dir, n))
        except (OSError, ValueError) as e:
            bad.append({"file": n, "why": "unreadable: %s" % e})
            continue
        rec = doc.get("versions") if isinstance(doc.get("versions"), dict) else doc
        ident = doc.get("identity") if isinstance(doc.get("identity"), dict) else {}
        v = rec.get("switchinterval_s") if isinstance(rec, dict) else None
        row = {"file": n, "pid": rec.get("pid") if isinstance(rec, dict) else None,
               "role": ident.get("role"), "worker": ident.get("worker"), "switchinterval_s": v}
        rows.append(row)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            bad.append({"file": n, "why": "no switchinterval_s"})
    if expected <= 0:
        verdict = "NOT_RUN"
    elif len(rows) < expected or bad:
        verdict = "RED"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "expected_count": expected, "found": len(rows),
            "problems": bad, "processes": rows,
            "distinct_values": sorted({r["switchinterval_s"] for r in rows
                                       if isinstance(r["switchinterval_s"], (int, float))})}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capenv.switchinterval")
    add_identity_args(ap)
    ap.add_argument("--versions-dir", required=True)
    ap.add_argument("--expected-count", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        ident = Identity(experiment=args.experiment, arm=args.arm, launch_seq=args.launch_seq,
                         job_id=args.job_id, restart_count=args.restart_count,
                         register_sha=args.register_sha, role=args.role, worker=args.worker,
                         pid=os.getpid())
        res = collect(args.versions_dir, args.expected_count)
        write_json_stamped(args.out, SCHEMA, ident, res)
    except ConfigError as e:
        print("capenv.switchinterval: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    print(json.dumps({"verdict": res["verdict"], "found": res["found"]}))
    return exitcodes.OK if res["verdict"] == "PASS" else exitcodes.ENV_INVALID


if __name__ == "__main__":
    sys.exit(main())
