# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode one snapshot file to JSON on stdout (INTERFACES.md section 11, `capprobe.decode`).

    python -m capprobe.decode FILE [--no-hist]

Output: {"file", "seq", "t_mono_ns", "meta_gen", "meta", "values"}; hist slots are bucket lists
unless --no-hist (then only their .n/.sum/.max/.neg counters remain). Exit 3 on a malformed
file (capcommon.snapshot.SnapshotError), 4 when the file cannot be read.

This module is a tool, not part of the import surface of product processes.
"""

import argparse
import json
import sys

from capcommon import exitcodes
from capcommon import snapshot as _snapshot


def decode_file(path, include_hist=True):
    snap = _snapshot.read_snapshot_file(path)
    kinds = {s["name"]: s["kind"] for s in snap.meta["slots"]}
    values = {k: v for k, v in snap.values.items() if include_hist or kinds.get(k) != "hist"}
    return {"file": path, "seq": snap.seq, "t_mono_ns": snap.t_mono_ns,
            "meta_gen": snap.meta_gen, "meta": snap.meta, "values": values}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capprobe.decode",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--no-hist", action="store_true", help="omit hist bucket lists")
    args = ap.parse_args(argv)
    try:
        doc = decode_file(args.file, include_hist=not args.no_hist)
    except exitcodes.CapkitError as e:
        print("capprobe.decode: %s: %s" % (args.file, e), file=sys.stderr)
        return e.exit_code
    except OSError as e:
        print("capprobe.decode: %s: %s" % (args.file, e), file=sys.stderr)
        return exitcodes.RUNTIME
    json.dump(doc, sys.stdout, sort_keys=True, indent=1)
    sys.stdout.write("\n")
    return exitcodes.OK


if __name__ == "__main__":
    sys.exit(main())
