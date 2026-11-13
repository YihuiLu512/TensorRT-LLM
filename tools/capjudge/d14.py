# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D14: is the softirq time on the tested cores the experiment's own CPU time? (user decision
2026-11-13: decided from E0 readings by a registered rule; INTERFACES erratum R1b).

Readings, per launch tree and per `under_test` role, on that role's cores (pin_table.json):

* idle: the job's gate idle window (raw/env/exclusivity.json item 6, per-core irq / softirq /
  total ticks over --idle-window-s, before any launch of the job);
* load: every load tier the pilot uses for that tree (sidecar cpus.jsonl, first and last row of
  each core inside the tier's steady window).

softirq_frac = softirq ticks / all /proc/stat ticks of those cores (user nice system idle iowait
irq softirq steal) over the same interval.

Rule (`capkit.d14_rule/1`, a file registered in <W>/REGISTRY.tsv with kind=d14_rule before the
first input launch was submitted):

    {"schema": "capkit.d14_rule/1", "idle_softirq_frac_max": F, "load_min_excess_frac": F,
     "monotonic_slack_frac": F, "min_trees": N}

A role satisfies the rule when its idle softirq_frac <= idle_softirq_frac_max ("close to 0"),
every load tier's softirq_frac >= idle softirq_frac + load_min_excess_frac, and the load tiers'
softirq_frac, ordered by load (tier table axis order), never drop by more than
monotonic_slack_frac ("grows with load"). A tree is complete when every under_test role has the
idle reading and at least one load reading. With at least min_trees complete trees the result
is irq_softirq_is_experiment = every role of every complete tree satisfies the rule; otherwise
(or when the rule was registered too late) no conclusion is written and C13 stays NOT_RUN.
"""

import json
import os

from capcommon import ledger
from capcommon.hashing import sha256_file

RULE_SCHEMA = "capkit.d14_rule/1"
REGISTRY_KIND = "d14_rule"
FRAC_KEYS = ("idle_softirq_frac_max", "load_min_excess_frac", "monotonic_slack_frac")
STAT_COLS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
GATE_FILE = "exclusivity.json"
IDLE_ITEM = 6


class D14RuleError(ValueError):
    pass


def load_rule(path, W):
    """(rule, record) for a rule file whose sha256 is registered in <W>/REGISTRY.tsv with
    kind=d14_rule. Raises D14RuleError."""
    try:
        with open(path, "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise D14RuleError("--d14-rule %s unreadable: %s" % (path, e)) from None
    if not isinstance(doc, dict) or set(doc) != {"schema", "min_trees"} | set(FRAC_KEYS) \
            or doc["schema"] != RULE_SCHEMA:
        raise D14RuleError("--d14-rule must be exactly {schema: %s, %s, min_trees}"
                           % (RULE_SCHEMA, ", ".join(FRAC_KEYS)))
    for k in FRAC_KEYS:
        v = doc[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
            raise D14RuleError("--d14-rule %s must be a number in [0, 1], got %r" % (k, v))
    n = doc["min_trees"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise D14RuleError("--d14-rule min_trees must be an int >= 1, got %r" % (n,))
    sha = sha256_file(path)
    row = ledger.registry_lookup(os.path.join(W, "REGISTRY.tsv"), sha)
    if row is None or ledger.registry_kind(row) != REGISTRY_KIND:
        raise D14RuleError("--d14-rule %s (sha256 %s) is not registered in %s/REGISTRY.tsv with "
                           "kind=%s" % (path, sha, W, REGISTRY_KIND))
    return doc, {"relpath": row["relpath"], "sha256": sha, "registered_at": row["registered_at"]}


def _frac(s, t):
    return s / t if t > 0 else None


def idle_reading(gate_doc, cores):
    """Idle-window reading of `cores` from exclusivity.json, or (None, why)."""
    if not isinstance(gate_doc, dict):
        return None, "raw/env/%s missing" % GATE_FILE
    items = [it for it in gate_doc.get("items") or [] if isinstance(it, dict)
             and it.get("id") == IDLE_ITEM]
    if len(items) != 1:
        return None, "%s has no item %d (idle window)" % (GATE_FILE, IDLE_ITEM)
    det = items[0].get("detail") or {}
    per = det.get("per_core") or {}
    out = {"source": "raw/env/%s item %d" % (GATE_FILE, IDLE_ITEM),
           "window_ns": det.get("window_ns"), "per_core": {}}
    for c in sorted(cores):
        row = per.get(str(c))
        if not isinstance(row, dict) or any(not isinstance(row.get(k), int) or
                                            isinstance(row.get(k), bool)
                                            for k in ("softirq_ticks", "irq_ticks",
                                                      "total_ticks")):
            return None, "idle window has no softirq / irq / total ticks for core %d" % c
        out["per_core"][str(c)] = {k: row[k] for k in ("softirq_ticks", "irq_ticks",
                                                       "total_ticks")}
    return _totals(out), None


def _totals(rd):
    pc = rd["per_core"].values()
    rd["softirq_ticks"] = sum(r["softirq_ticks"] for r in pc)
    rd["irq_ticks"] = sum(r["irq_ticks"] for r in pc)
    rd["total_ticks"] = sum(r["total_ticks"] for r in pc)
    rd["softirq_frac"] = _frac(rd["softirq_ticks"], rd["total_ticks"])
    return rd


def load_reading(cpu_rows, cores, ws, we):
    """Reading of `cores` over [ws, we) from sidecar cpus rows, or (None, why)."""
    first, last = {}, {}
    for row in cpu_rows:
        t, c = row.get("t_rel_ns"), row.get("cpu")
        if c not in cores or not isinstance(t, int) or not ws <= t < we:
            continue
        if c not in first or t < first[c]["t_rel_ns"]:
            first[c] = row
        if c not in last or t > last[c]["t_rel_ns"]:
            last[c] = row
    missing = sorted(set(cores) - {c for c in first if last[c] is not first[c]})
    if missing:
        return None, "fewer than two sidecar cpu rows in the window for cores %s" % missing
    out = {"source": "raw/sidecar/cpus.jsonl", "per_core": {}}
    for c in sorted(cores):
        d = {k: last[c][k] - first[c][k] for k in STAT_COLS}
        out["per_core"][str(c)] = {"softirq_ticks": d["softirq"], "irq_ticks": d["irq"],
                                   "total_ticks": sum(d.values()),
                                   "t_first_rel_ns": first[c]["t_rel_ns"],
                                   "t_last_rel_ns": last[c]["t_rel_ns"]}
    return _totals(out), None


def tree_readings(ctx, ld, role_cpus, seqs):
    """Readings of one launch tree: {"roles": {role: {...}}, "complete": bool, "why": ...}."""
    ut = sorted(r["role"] for r in ctx.roles.values() if r.get("measure_class") == "under_test")
    rec = {"roles": {}, "complete": False, "why": None}
    if not ut:
        rec["why"] = "no under_test role"
        return rec
    gate_doc = ld._json(ctx.tree.env(GATE_FILE))
    cpus = ld.sidecar("cpus")
    why = []
    for role in ut:
        cores = role_cpus.get(role) or set()
        r = {"cores": sorted(cores), "idle": None, "load": [], "why": []}
        rec["roles"][role] = r
        if not cores:
            r["why"].append("no cores in pin_table.json")
            why.append("%s: no cores" % role)
            continue
        r["idle"], w = idle_reading(gate_doc, cores)
        if w:
            r["why"].append(w)
        for seq in seqs:
            t = ctx.tt.tiers[seq]
            if cpus is None:
                r["why"].append("raw/sidecar/cpus.jsonl missing")
                break
            rd, w = load_reading(cpus[1], cores, t.window_start_ns, t.window_end_ns)
            if w:
                r["why"].append("tier %d: %s" % (seq, w))
                continue
            rd.update(tier_seq=seq, load_value=t.load_value,
                      load_order_key=ctx.tt.load_order_key(seq))
            r["load"].append(rd)
        if r["idle"] is None or not r["load"]:
            why.append("%s: %s" % (role, "; ".join(r["why"]) or "no load reading"))
    rec["complete"] = not why
    rec["why"] = "; ".join(why) or None
    return rec


def evaluate_role(r, rule):
    """(idle_close_to_zero, grows_with_load) for one complete role reading."""
    idle = r["idle"]["softirq_frac"]
    loads = sorted(r["load"], key=lambda x: (x["load_order_key"] is None,
                                             x["load_order_key"], x["tier_seq"]))
    fr = [x["softirq_frac"] for x in loads]
    if idle is None or any(f is None for f in fr) or \
            any(x["load_order_key"] is None for x in loads):
        return None, None
    near_zero = idle <= rule["idle_softirq_frac_max"]
    grows = all(f >= idle + rule["load_min_excess_frac"] for f in fr) and \
        all(b >= a - rule["monotonic_slack_frac"] for a, b in zip(fr, fr[1:]))
    return near_zero, grows


def conclude(rule, rule_rec, trees, submitted_at):
    """The pilot.json d14_softirq object. trees: [(tree relpath, readings)]; submitted_at:
    {tree relpath: submitted_at or None}."""
    out = {"rule": rule, "rule_file": rule_rec, "readings": {}, "n_trees_complete": 0,
           "conclusion_reason": None}
    late = sorted(t for t, at in submitted_at.items()
                  if not at or not ledger.registered_before(rule_rec, at))
    verdicts = []
    for tree, rec in trees:
        for r in rec["roles"].values():
            if rec["complete"]:
                r["idle_close_to_zero"], r["grows_with_load"] = evaluate_role(r, rule)
                verdicts.append((r["idle_close_to_zero"], r["grows_with_load"]))
        out["readings"][tree] = rec
        out["n_trees_complete"] += bool(rec["complete"])
    if late:
        out["conclusion_reason"] = ("rule not registered before the submission of %s (or no "
                                    "submitted_at): no conclusion" % late)
    elif out["n_trees_complete"] < rule["min_trees"]:
        out["conclusion_reason"] = ("%d complete tree(s) < min_trees %d: no conclusion"
                                    % (out["n_trees_complete"], rule["min_trees"]))
    elif any(a is None or b is None for a, b in verdicts):
        out["conclusion_reason"] = "a reading has no tick total or no load order: no conclusion"
    else:
        ok = all(a and b for a, b in verdicts)
        out["irq_softirq_is_experiment"] = ok
        out["conclusion_reason"] = ("every under_test role: idle softirq close to 0 and growing "
                                    "with load" if ok else
                                    "some under_test role: idle softirq not close to 0, or not "
                                    "growing with load")
    return out
