# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compute --exclude from 'cdb find ... -o csv' output (plan RESOURCES 9.1: exclude nodes of the
listed partitions whose profile differs from the registered main-profile spec).

Input text layout (as saved in 00_plan/cluster-probe/<date>/02d_cdb_b200.csv): optional lines
starting with '# cmd:' and '# exit: <rc>', one CSV header, one row per node. The last column
'slurm.partitions' is a Python-list literal and is the only column allowed to contain commas.

The main-profile spec (JSON, schema capkit.cdb_main_profile/1) names the profile columns and their
values explicitly: {"schema": ..., "name": "G0", "fields": {"system.product_name": "DGXB200", ...}}.
"""

import ast
import json

from capcommon.exitcodes import ConfigError
from capcommon.hashing import canonical_json_sha256, sha256_bytes

SPEC_SCHEMA = "capkit.cdb_main_profile/1"
RESULT_SCHEMA = "capkit.cdb_exclude/1"
HOST_COL = "system.hostname"
PART_COL = "slurm.partitions"


class CdbError(ConfigError):
    pass


def parse_cdb_csv(text):
    """Return (header list, rows list of dicts, meta dict). Fails closed on anything unexpected."""
    meta = {"cmd": None, "exit": None}
    header = None
    rows = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        if line.startswith("#"):
            body = line[1:].strip()
            if body.startswith("cmd:"):
                meta["cmd"] = body[4:].strip()
            elif body.startswith("exit:"):
                try:
                    meta["exit"] = int(body[5:].strip())
                except ValueError:
                    raise CdbError("line %d: bad exit marker %r" % (lineno, line))
            continue
        if header is None:
            header = line.split(",")
            if HOST_COL not in header or PART_COL not in header or header[-1] != PART_COL:
                raise CdbError("header must contain %s and end with %s" % (HOST_COL, PART_COL))
            if len(set(header)) != len(header):
                raise CdbError("duplicate header columns")
            continue
        n = len(header)
        parts = line.split(",", n - 1)
        if len(parts) != n:
            raise CdbError("line %d: %d cells, expected %d" % (lineno, len(parts), n))
        row = dict(zip(header, parts))
        try:
            plist = ast.literal_eval(row[PART_COL])
        except (ValueError, SyntaxError):
            raise CdbError("line %d: partitions cell is not a list literal: %r"
                           % (lineno, row[PART_COL]))
        if not isinstance(plist, list) or not all(isinstance(p, str) for p in plist):
            raise CdbError("line %d: partitions must be a list of strings" % lineno)
        row[PART_COL] = plist
        rows.append(row)
    if header is None:
        raise CdbError("no CSV header found")
    if meta["exit"] is not None and meta["exit"] != 0:
        raise CdbError("cdb exit marker is %d (the query failed)" % meta["exit"])
    hosts = [r[HOST_COL] for r in rows]
    if len(set(hosts)) != len(hosts):
        raise CdbError("duplicate hostnames in cdb output")
    return header, rows, meta


def load_spec(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SPEC_SCHEMA:
        raise CdbError("main-profile spec must have schema %s" % SPEC_SCHEMA)
    fields = doc.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise CdbError("main-profile spec needs a non-empty 'fields' object")
    if HOST_COL in fields or PART_COL in fields:
        raise CdbError("spec fields must not include hostname or partitions")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in fields.items()):
        raise CdbError("spec field names and values must be strings (as cdb prints them)")
    if not isinstance(doc.get("name"), str) or not doc["name"]:
        raise CdbError("main-profile spec needs a name")
    return doc


def compute_exclude(text, spec_doc, partitions):
    """Return the result dict. Raises CdbError when the check would be empty (no node of the given
    partitions in the output) or has no positive control (no main-profile node in them)."""
    spec = load_spec(spec_doc)
    if not partitions or not all(isinstance(p, str) and p for p in partitions):
        raise CdbError("partitions must be a non-empty list")
    header, rows, meta = parse_cdb_csv(text)
    missing = [k for k in spec["fields"] if k not in header]
    if missing:
        raise CdbError("spec fields not in cdb output: %s" % missing)
    wanted = set(partitions)
    in_parts = [r for r in rows if wanted & set(r[PART_COL])]
    if not in_parts:
        raise CdbError("NOT_RUN: no node of partitions %s in the cdb output" % sorted(wanted))
    seen_parts = set()
    for r in in_parts:
        seen_parts |= wanted & set(r[PART_COL])
    unseen = sorted(wanted - seen_parts)
    if unseen:
        raise CdbError("partitions with no node in the cdb output: %s" % unseen)
    match, differ = [], []
    for r in in_parts:
        diffs = {k: {"want": v, "got": r[k]} for k, v in spec["fields"].items() if r[k] != v}
        (differ if diffs else match).append((r[HOST_COL], diffs))
    if not match:
        raise CdbError("no node of the given partitions matches main profile %r" % spec["name"])
    exclude = sorted(h for h, _ in differ)
    return {"schema": RESULT_SCHEMA, "spec_name": spec["name"],
            "spec_sha256": canonical_json_sha256(spec), "cdb_text_sha256":
            sha256_bytes(text.encode("utf-8")), "cdb_cmd": meta["cmd"], "cdb_exit": meta["exit"],
            "partitions": sorted(wanted), "nodes_in_partitions": len(in_parts),
            "nodes_matching": sorted(h for h, _ in match), "exclude": exclude,
            "differences": {h: d for h, d in differ}}


def read_result(path):
    """Load a result written by compute_exclude (for submit --exclude-file)."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or doc.get("schema") != RESULT_SCHEMA \
            or not isinstance(doc.get("exclude"), list) or not isinstance(doc.get("partitions"),
                                                                          list):
        raise CdbError("%s is not a %s file" % (path, RESULT_SCHEMA))
    return doc
