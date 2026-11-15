# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""job_plan.tsv: the registered job plan of one segment of a phase (calib-v2 DESIGN-r7a 3.2,
4.5; REGISTRY kind job_plan). caplaunch.laddergen writes it, capjobs.submit's budget gate and
capkit's pilot2 read it.

Columns: order (1, 2, ... contiguous; the two jobs of a block share one order), item_id (unique),
item_class (planned | surplus | post_gate:<rule>), arms (comma list), launches, block_id, slot
(both empty outside blocks), config_counts (JSON {config: launches}), pin (true | false).
There is no replace / resubmit class: every submission after a job result is a post-gate item
of a registered gate product.
"""

import json
import re

from .exitcodes import ConfigError
from .identity import ARM_RE

JOB_PLAN_COLUMNS = ("order", "item_id", "item_class", "arms", "launches", "block_id", "slot",
                    "config_counts", "pin")
POST_GATE_RULES = ("shift", "backstop", "coverage", "blind_topup", "shortfall")
POST_GATE_PREFIX = "post_gate:"
ITEM_CLASSES = ("planned", "surplus") + tuple(POST_GATE_PREFIX + r for r in POST_GATE_RULES)
ITEM_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
BLOCK_ID_RE = ITEM_ID_RE


class JobPlanError(ConfigError):
    pass


def check_item_class(item_class):
    if item_class not in ITEM_CLASSES:
        raise JobPlanError("item_class %r is not one of %s (there is no replace / resubmit "
                           "class)" % (item_class, ITEM_CLASSES))
    return item_class


def block_comparison(block_id):
    """The comparison a block belongs to: block ids are '<comparison>-b<suffix>' (laddergen
    --job-plan and --plan-from), so the text before the last '-b' (the whole id when there is
    none). capjobs.submit pin-next and capjudge.pilot2 group block items by it."""
    return block_id.rsplit("-b", 1)[0]


def _is_pos_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def validate(items):
    """Check a list of typed items (as read() returns them). Returns the list."""
    if not items:
        raise JobPlanError("a job plan has at least one item")
    ids = [it["item_id"] for it in items]
    if len(set(ids)) != len(ids):
        raise JobPlanError("duplicate item_id in the job plan")
    orders = [it["order"] for it in items]
    if orders != sorted(orders) or sorted(set(orders)) != list(range(1, max(orders) + 1)):
        raise JobPlanError("order must start at 1 and increase by 1 in file order")
    classes = [check_item_class(it["item_class"]) for it in items]
    post = [c for c in classes if c.startswith(POST_GATE_PREFIX)]
    if post and len(set(classes)) != 1:
        raise JobPlanError("a plan of post-gate items holds one post-gate rule only")
    if not post and classes != sorted(classes, key=("planned", "surplus").index):
        raise JobPlanError("surplus items come after every planned item")
    for it in items:
        where = "item %s" % it["item_id"]
        if not ITEM_ID_RE.match(it["item_id"]):
            raise JobPlanError("%s: bad item_id" % where)
        check_item_class(it["item_class"])
        if not it["arms"] or any(not ARM_RE.match(a) for a in it["arms"]):
            raise JobPlanError("%s: arms must be a non-empty comma list of arm names" % where)
        if not _is_pos_int(it["launches"]):
            raise JobPlanError("%s: launches must be an int > 0" % where)
        cc = it["config_counts"]
        if not isinstance(cc, dict) or not cc or not all(
                isinstance(k, str) and k and _is_pos_int(v) for k, v in cc.items()) \
                or sum(cc.values()) != it["launches"]:
            raise JobPlanError("%s: config_counts must map configs to ints > 0 summing to "
                               "launches" % where)
        if (it["block_id"] is None) != (it["slot"] is None):
            raise JobPlanError("%s: block_id and slot are both set or both empty" % where)
        if it["block_id"] is not None and (not BLOCK_ID_RE.match(it["block_id"])
                                           or it["slot"] not in (1, 2)):
            raise JobPlanError("%s: bad block_id or slot (1 or 2)" % where)
        if not isinstance(it["pin"], bool):
            raise JobPlanError("%s: pin must be true or false" % where)
    for o in sorted(set(orders)):
        grp = [it for it in items if it["order"] == o]
        if len(grp) == 1:
            continue
        if len(grp) != 2 or grp[0]["block_id"] is None \
                or grp[0]["block_id"] != grp[1]["block_id"] \
                or {grp[0]["slot"], grp[1]["slot"]} != {1, 2}:
            raise JobPlanError("order %d: only the two slots of one block share an order" % o)
    return items


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    if isinstance(v, list):
        return ",".join(v)
    return str(v)


def format_tsv(items):
    """Bytes of a job_plan.tsv for validated typed items."""
    validate(items)
    lines = ["\t".join(JOB_PLAN_COLUMNS)]
    for it in items:
        cells = [_cell(it[c]) for c in JOB_PLAN_COLUMNS]
        if any("\t" in c or "\n" in c for c in cells):
            raise JobPlanError("job plan cell contains a tab or newline")
        lines.append("\t".join(cells))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _int(text, what):
    if not text.isascii() or not text.isdigit():
        raise JobPlanError("%s must be a decimal int, got %r" % (what, text))
    return int(text)


def parse_tsv(text):
    """Typed items of a job_plan.tsv text (validated)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines or tuple(lines[0].split("\t")) != JOB_PLAN_COLUMNS:
        raise JobPlanError("job plan header must be %s" % (JOB_PLAN_COLUMNS,))
    items = []
    for n, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(JOB_PLAN_COLUMNS):
            raise JobPlanError("line %d: %d cells, expected %d" % (n, len(cells),
                                                                  len(JOB_PLAN_COLUMNS)))
        row = dict(zip(JOB_PLAN_COLUMNS, cells))
        try:
            cc = json.loads(row["config_counts"])
        except ValueError:
            raise JobPlanError("line %d: config_counts is not JSON" % n)
        if row["pin"] not in ("true", "false"):
            raise JobPlanError("line %d: pin must be true or false" % n)
        items.append({"order": _int(row["order"], "order"), "item_id": row["item_id"],
                      "item_class": row["item_class"],
                      "arms": row["arms"].split(",") if row["arms"] else [],
                      "launches": _int(row["launches"], "launches"),
                      "block_id": row["block_id"] or None,
                      "slot": _int(row["slot"], "slot") if row["slot"] else None,
                      "config_counts": cc, "pin": row["pin"] == "true"})
    return validate(items)


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return parse_tsv(f.read())
