# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m caplaunch.tiergen: fill a tier-table TEMPLATE with registered numbers.

Templates (configs/e0/*.tiers.template.json, schema capkit.tiers_template/1) carry the tier
shape; every number E0 must pick itself is a string "TBD:<name>". A values file
(capkit.tier_values/1, {"values": {name: number}}) supplies them; the generator refuses to
write while any TBD remains, computes seq / start_ns / drain defaults / load_value, and
validates the result with capcommon.tiers.TierTable (INTERFACES.md 3.2). Optionally it also
writes the arrival schedule (capcommon.schedule.write_schedule).
"""

import argparse
import fractions
import json
import os
import sys

from capcommon.exitcodes import ConfigError
from capcommon.jsonio import atomic_write_bytes
from capcommon.tiers import TierTable

TEMPLATE_SCHEMA = "capkit.tiers_template/1"
VALUES_SCHEMA = "capkit.tier_values/1"
TOP_COPY = ("experiment", "name", "arrival", "axis", "request_timeout_ns", "default_gate",
            "launch_inject")
LOAD_DEFAULTS = ("max_tokens", "tokens_per_msg", "stream_interval", "beat_ns", "prompt_set",
                 "ramp_ns", "tail_ns", "streams_c")


def substitute(obj, values, missing):
    if isinstance(obj, str) and obj.startswith("TBD:"):
        name = obj[4:]
        if name in values:
            return values[name]
        missing.add(name)
        return obj
    if isinstance(obj, dict):
        return {k: substitute(v, values, missing) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(v, values, missing) for v in obj]
    return obj


def tbd_names(obj):
    out = set()
    substitute(obj, {}, out)
    return out


def _rate_from_streams(t):
    tps = t["tokens_per_msg"] // t["stream_interval"]
    steps = -(-t["max_tokens"] // tps)
    return float(fractions.Fraction(t["streams_c"] * 10 ** 9, steps * t["beat_ns"]))


def _load_value(axis, t):
    k = axis["load_key"]
    if k == "rate_rps":
        return t["rate_rps"]
    if k == "streams_c":
        return t["streams_c"]
    if k == "beat_ns":
        return t["beat_ns"]
    if k == "tokens_per_msg":
        return t["tokens_per_msg"]
    if k == "nominal_msgs_per_s":
        return float(fractions.Fraction(t["streams_c"] * 10 ** 9,
                                        t["beat_ns"] * t["stream_interval"]))
    if "prompt_len" not in t:
        raise ConfigError("tier %s: axis prompt_len needs an explicit prompt_len" % t["label"])
    return t.pop("prompt_len")


def expand(template, values):
    """Template + values -> tier-table document (validated)."""
    if template.get("schema") != TEMPLATE_SCHEMA:
        raise ConfigError("template schema must be %s" % TEMPLATE_SCHEMA)
    missing = set()
    tpl = substitute(template, values, missing)
    if missing:
        raise ConfigError("values missing for TBD names: %s" % sorted(missing))
    unused = set(values) - tbd_names(template)
    if unused:
        raise ConfigError("values not used by the template: %s" % sorted(unused))
    common = tpl.get("common") or {}
    doc = {"schema": "capkit.tiers/1"}
    doc.update({k: tpl[k] for k in TOP_COPY})
    doc["beat_origin_rel_ns"] = 0
    tiers, start, prev_beat = [], 0, None
    for seq, src in enumerate(tpl["tiers"]):
        t = dict(src)
        kind = t["kind"]
        t["seq"], t["start_ns"] = seq, start
        t.setdefault("inject", [])
        t.setdefault("gate", None)
        t.setdefault("baseline", False)
        t.setdefault("note", "")
        if kind == "drain":
            t.update({"ramp_ns": 0, "tail_ns": 0, "rate_rps": 0.0, "max_tokens": 0,
                      "prompt_set": None, "streams_c": None, "load_value": None})
            t.setdefault("tokens_per_msg", common.get("tokens_per_msg", 1))
            t.setdefault("stream_interval", common.get("stream_interval", 1))
            t["beat_ns"] = prev_beat if prev_beat is not None else common.get("beat_ns", 0)
        else:
            for k in LOAD_DEFAULTS:
                if k not in t:
                    t[k] = common.get(k, None if k == "streams_c" else 0)
            if t.get("rate_rps") == "from_streams_c":
                t["rate_rps"] = _rate_from_streams(t)
            t["rate_rps"] = float(t["rate_rps"])
            t["load_value"] = None if kind == "warmup" and "load_value" not in src else \
                _load_value(tpl["axis"], t)
        prev_beat = t["beat_ns"]
        start += t["ramp_ns"] + t["window_ns"] + t["tail_ns"]
        tiers.append(t)
    doc["tiers"] = tiers
    TierTable.from_doc(doc)  # raises TierTableError (exit 3) on any INTERFACES 3.2 violation
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m caplaunch.tiergen", description=__doc__)
    p.add_argument("--template", required=True)
    p.add_argument("--values", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--schedule-out", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    a = p.parse_args(argv)
    try:
        with open(a.template) as f:
            template = json.load(f)
        with open(a.values) as f:
            vals = json.load(f)
        if vals.get("schema") != VALUES_SCHEMA or not isinstance(vals.get("values"), dict):
            raise ConfigError("values file must be %s with a values object" % VALUES_SCHEMA)
        unset = sorted(k for k, v in vals["values"].items() if v is None)
        if unset:  # a *.values.todo.json checklist not yet filled in
            raise ConfigError("values still null: %s" % unset)
        doc = expand(template, vals["values"])
        if os.path.exists(a.out):
            raise ConfigError("%s exists; registered files are never overwritten" % a.out)
        if a.schedule_out and (a.max_rows is None or os.path.exists(a.schedule_out)):
            raise ConfigError("--schedule-out needs --max-rows and a new path")
        atomic_write_bytes(a.out, (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode())
        if a.schedule_out:
            from capcommon.schedule import write_schedule  # noqa: PLC0415
            write_schedule(a.schedule_out, TierTable.load(a.out), a.max_rows)
    except ConfigError as e:
        print("tiergen: %s" % e, file=sys.stderr)
        return e.exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
