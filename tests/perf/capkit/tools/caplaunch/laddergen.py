# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m caplaunch.laddergen: calibration v2 tier tables, ladder.json and job plans
(calib-v2 DESIGN-r7a 3.1 P2-07, 3.2, 3.4, 4.1, 4.2, 4.5; INTERFACES erratum calib-v2-A4).

Four modes (exactly one):
  --kind {ladder,refine,bracket,hold,null,pc,c13pc,e2e,cal}
      one launch: a tier-table template + values file for caplaunch.tiergen (expanded and
      validated here with tiergen.expand; tiergen.py and capcommon/tiers.py are not changed) and
      the launch's ladder.json (capkit.ladder/1). The centre comes only from a registered
      e0_params file (--center, --center-key); --center-literal is for the V0 / V0-cal / V1-a1
      prior centre and is recorded as center_source. Drain slots follow DESIGN 4.1 with the
      phase's registered e_c; the table must fit the 4.1 job condition and capjobs' time budget,
      otherwise nothing is written. --request-timeout-ns {null,<int ns>} sets the table's
      request_timeout_ns through the values file (null is written as a registered null that
      tiergen accepts); without it the value comes from --base-values. --kind pc needs it.
  --plan
      the K_fit table of DESIGN 2.7 / 4.1 (the conservative r4 model: every level at the top rho).
  --job-plan <spec>
      job_plan.tsv of one segment: planned then surplus items, mixed by the registered seed
      (layout "mixed"), or with layout "one_each" every job running every configuration once
      in seed order (V0-cal, DESIGN 4.2).
  --plan-from <registered gate product>
      job_plan.tsv of the post-gate items a registered gate product lists (never a replace item;
      launch items and V2-b blocks of a shortfall product). --launches-per-job must equal the
      product's cap.launches_per_job when it has one.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import sys

from capcommon import jobplan, ledger
from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_bytes, sha256_file
from capcommon.identity import ARM_RE

from . import tiergen

KINDS = ("ladder", "refine", "bracket", "hold", "null", "pc", "c13pc", "e2e", "cal")
# --center-literal: V0 / V0-cal (c0, DESIGN 3.1 P2-07, 4.2) and the V1-a1 prior (no e0_params key)
LITERAL_CENTER_KINDS = ("ladder", "pc", "c13pc", "e2e", "cal", "bracket")
# kinds whose table must say its request timeout explicitly (s1-pc null, s*-fix 60 s; DESIGN 4.2)
TIMEOUT_REQUIRED_KINDS = ("pc",)
TIMEOUT_NAME = "request_timeout_ns"
TIMEOUT_TBD = "TBD:" + TIMEOUT_NAME
# --job-plan layouts: "mixed" (seed-mixed launches cut into jobs) and "one_each" (DESIGN 4.2
# V0-cal: every job runs every configuration exactly once, in seed order)
LAYOUTS = ("mixed", "one_each")
ONE_EACH_PHASES = ("V0cal",)
# e0_params stage that registers each centre key (DESIGN 3.2)
CENTER_KEY_STAGE = {"c1": "bracket", "c_hat": "capacity", "c_b": "recenter", "c_n": "settle",
                    "c_lad": "ladder"}
E0_PARAMS_SCHEMA = "capkit.e0_params/1"
LADDER_SCHEMA = "capkit.ladder/1"
SPEC_SCHEMA = "capkit.job_plan_spec/1"
# registered e_c per phase (DESIGN 4.1); other phases give --e-c explicitly
E_C_BY_PHASE = {"V0cal": 0.30, "V0": 0.30, "V1b": 0.02, "V2n": 0.02, "V2b": 0.01}
PHASES = ("V0cal", "V0", "V1a1", "V1a2", "V1b", "V2n", "V2b", "V2c")
NS = 10 ** 9
WINDOW_S, TAIL_S, RAMP_BASE_S, WARMUP_S = 60, 2, 4, 10
SLOPE_SETTLE_S = 4
BASE_RHO = 0.3
SLOPE_RHOS = (1.05, 1.10, 1.20)
H_LADDER, H_REFINE = 0.0025, 0.005
REFINE_LEVELS = 21
PC_OVER_RHO = 1.5
E2E_RHOS = (0.5, 1.0, 1.5)
CAL_RHOS = (0.3, 0.4, 0.5, 0.6)
NEAR_LOW_RHOS = (0.90, 0.95)
HOLD_S = 600
HOLD_RHOS = (0.95, 0.98, 0.99)
DRAIN_MIN_S = 60
BRACKET_SLOT_S = 180  # DESIGN 4.1: unknown capacity, backlog capped by the 60 s request timeout
# packing (DESIGN 4.1): launch overhead (MEASURED max), budget_s pad, QoS limit, job condition
LAUNCH_OVERHEAD_S = 6.5
BUDGET_PAD_S = 60
QOS_LIMIT_S = 14400
JOB_COND_PAD_S = 60
JOB_OVERHEAD_FLOOR_S = 1.5 * 509
PLAN_K_RANGE = tuple(range(12, 34, 2))
PLAN_E_C = E_C_BY_PHASE["V2b"]


class LadderError(ConfigError):
    pass


def ceil10(x):
    return int(math.ceil(x / 10.0 - 1e-12) * 10)


def drain_slot_s(rho, e_c, prev_len_s, bracket=False):
    """DESIGN 4.1: D = max(60, ceil10(1.5 * B + 30)), B = (rho_max - 1)+ * previous tier length,
    rho_max = rho * (1 + e_c); a bracket tier that may be over capacity gets the 180 s slot."""
    rho_max = rho * (1 + e_c)
    if bracket and rho_max > 1:
        return BRACKET_SLOT_S
    return max(DRAIN_MIN_S, ceil10(1.5 * max(0.0, rho_max - 1) * prev_len_s + 30))


def seed_u(seed_base, config, block_id, slot):
    """DESIGN 3.4: u = int(sha256(base|config|block_id|slot)[:13], 16) / 16**13."""
    sha = hashlib.sha256(("%s|%s|%s|%s" % (seed_base, config, block_id, slot)).encode(
        "utf-8")).hexdigest()
    return int(sha[:13], 16) / 16 ** 13, sha


def _pos_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def registered(workdir, path, kind):
    """(relpath, sha256, doc) of a file that REGISTRY.tsv lists under kind with this sha and
    this path; anything else fails."""
    sha = sha256_file(path)
    row = ledger.registry_lookup(os.path.join(workdir, "REGISTRY.tsv"), sha)
    if row is None:
        raise LadderError("%s (sha %s) is not in REGISTRY.tsv" % (path, sha))
    if ledger.registry_kind(row) != kind:
        raise LadderError("%s is registered as kind %r, not %r" % (path, ledger.registry_kind(row),
                                                                   kind))
    if os.path.realpath(os.path.join(workdir, row["relpath"])) != os.path.realpath(path):
        raise LadderError("%s is not the registered relpath %s" % (path, row["relpath"]))
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return row["relpath"], sha, doc


def read_center(a):
    """(centre value, center_source) from --center/--center-key or --center-literal."""
    if (a.center is None) == (a.center_literal is None):
        raise LadderError("give exactly one of --center (with --center-key) or --center-literal")
    if a.center_literal is not None:
        if a.kind not in LITERAL_CENTER_KINDS:
            raise LadderError("--center-literal is for kinds %s; --kind %s reads the centre from "
                              "a registered e0_params file" % (LITERAL_CENTER_KINDS, a.kind))
        if not _pos_num(a.center_literal):
            raise LadderError("--center-literal must be > 0")
        return float(a.center_literal), {"literal": float(a.center_literal)}
    if a.center_key not in CENTER_KEY_STAGE or a.workdir is None:
        raise LadderError("--center needs --center-key %s and --workdir" % sorted(CENTER_KEY_STAGE))
    rel, sha, doc = registered(a.workdir, a.center, "e0_params")
    stage = CENTER_KEY_STAGE[a.center_key]
    if not isinstance(doc, dict) or doc.get("schema") != E0_PARAMS_SCHEMA \
            or doc.get("stage") != stage:
        raise LadderError("%s must be %s stage %r (it holds %s)" % (a.center, E0_PARAMS_SCHEMA,
                                                                   stage, a.center_key))
    by_config = doc.get("by_config")
    entry = by_config.get(a.config) if isinstance(by_config, dict) else None
    val = entry.get(a.center_key) if isinstance(entry, dict) else None
    if not _pos_num(val):
        raise LadderError("%s has no by_config[%r][%r] > 0" % (a.center, a.config, a.center_key))
    return float(val), {"relpath": rel, "sha256": sha, "key": a.center_key}


# ---- tier sequences -----------------------------------------------------------------------------

def _load(label, kind, rho, settle_s, role, **extra):
    return {"label": label, "kind": kind, "rho": rho, "ramp_s": RAMP_BASE_S + settle_s,
            "window_s": WINDOW_S, "tail_s": TAIL_S, "role": role, "extra": extra}


def _slopes():
    return [_load("slope%d" % (i + 1), "up", r, SLOPE_SETTLE_S, "slope")
            for i, r in enumerate(SLOPE_RHOS)]


def _levels(rhos, passes, settle_s):
    """Level k up (k = 0..n-1) then, with two passes, down (n-1..0)."""
    out = [_load("L%02du" % k, "up", r, settle_s, "levels", level=k, passes=1)
           for k, r in enumerate(rhos)]
    if passes == 2:
        out += [_load("L%02dd" % k, "down", rhos[k], settle_s, "levels", level=k, passes=2)
                for k in reversed(range(len(rhos)))]
    return out


def tier_sequence(a, u):
    """[load tier specs] of one launch (warm-up and drains are added by build); h of the kind."""
    S = a.settle_s
    base = [_load("base", "up", BASE_RHO, S, "baseline")]
    h = None
    if a.kind == "ladder":
        if not (isinstance(a.K, int) and a.K >= 2 and _pos_num(a.span_half)):
            raise LadderError("--kind ladder needs --K >= 2 and --span-half > 0")
        h = a.h
        rhos = [1 - a.span_half + h * (k + u) for k in range(a.K)]
        seq = base + _levels(rhos, a.passes, S) + _slopes()
    elif a.kind == "refine":
        h = H_REFINE
        rhos = [1 - h * (REFINE_LEVELS // 2) + h * k for k in range(REFINE_LEVELS)]
        seq = base + _levels(rhos, 1, S) + _slopes()
    elif a.kind == "bracket":
        if not (_pos_num(a.rho_first) and _pos_num(a.ratio) and a.ratio > 1
                and isinstance(a.n_levels, int) and a.n_levels >= 2):
            raise LadderError("--kind bracket needs --rho-first > 0, --ratio > 1, --n-levels >= 2")
        rhos = [a.rho_first * a.ratio ** k for k in range(a.n_levels)]
        seq = base + _levels(rhos, 1, S)
    elif a.kind == "hold":
        order = list(itertools.permutations(HOLD_RHOS))[min(5, int(u * 6))]
        seq = base + [dict(_load("hold%d" % (i + 1), "up", r, 0, "holds", hold_ns=HOLD_S * NS),
                           ramp_s=0, window_s=HOLD_S) for i, r in enumerate(order)] + _slopes()
    elif a.kind == "null":
        if not (_pos_num(a.rho_top) and NEAR_LOW_RHOS[-1] < a.rho_top < 1):
            raise LadderError("--kind null needs %s < --rho-top < 1" % NEAR_LOW_RHOS[-1])
        up = list(NEAR_LOW_RHOS) + [a.rho_top] * 3
        seq = base + [_load("nu%d" % (i + 1), "up", r, S, "near", pass_=1)
                      for i, r in enumerate(up)] \
            + [_load("nd%d" % (i + 1), "down", r, S, "near", pass_=2)
               for i, r in enumerate(reversed(up))] \
            + [_load("sub", "down", BASE_RHO, S, "cal", pass_=2)] + _slopes()
    elif a.kind == "pc":
        seq = base + [_load("over", "up", PC_OVER_RHO, S, "slope")] \
            + [_load("sub%d" % (i + 1), "down", BASE_RHO, S, "cal", pass_=i + 1)
               for i in range(2)]
    elif a.kind == "c13pc":  # DESIGN 4.2 c13-pc: the pc table without the overload tier
        seq = base + [_load("sub%d" % (i + 1), "down", BASE_RHO, S, "cal", pass_=i + 1)
                      for i in range(2)]
    elif a.kind == "e2e":
        h = a.h
        seq = base + _levels([r + h * u for r in E2E_RHOS], 2, S) + _slopes()
    else:  # cal
        seq = base + [_load("calu%d" % (i + 1), "up", r, S, "cal", pass_=1)
                      for i, r in enumerate(CAL_RHOS)] \
            + [_load("cald%d" % (i + 1), "down", r, S, "cal", pass_=2)
               for i, r in enumerate(reversed(CAL_RHOS))]
    return seq, h


# ---- template, values, ladder.json --------------------------------------------------------------

def _tbd(label, field):
    return "TBD:%s_%s" % (label.replace("-", "_").replace(".", "_"), field)


def parse_timeout(text):
    """--request-timeout-ns: "null" (no registered timeout) or a decimal int > 0."""
    if text == "null":
        return None
    if not (text.isascii() and text.isdigit() and int(text) > 0):
        raise LadderError("--request-timeout-ns must be null or an int > 0, got %r" % text)
    return int(text)


def build_template(base, base_values, name, seq, center, e_c, bracket, timeout=()):
    """(template, values, drain slots {drain label: s}) for tiergen.expand. timeout: () = take
    request_timeout_ns from base_values, else (value,) with value None or an int > 0."""
    if base.get("schema") != tiergen.TEMPLATE_SCHEMA:
        raise LadderError("--base-template must be %s" % tiergen.TEMPLATE_SCHEMA)
    if base.get(TIMEOUT_NAME) != TIMEOUT_TBD:
        raise LadderError("--base-template %s must be %r: the timeout comes from "
                          "--request-timeout-ns or --base-values, not a literal in the template "
                          "(got %r)" % (TIMEOUT_NAME, TIMEOUT_TBD, base.get(TIMEOUT_NAME)))
    tpl = {k: base[k] for k in ("schema", "experiment", "arrival", "request_timeout_ns",
                                "default_gate", "launch_inject") if k in base}
    tpl["name"] = name
    tpl["axis"] = {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"}
    tpl["common"] = {k: v for k, v in (base.get("common") or {}).items() if k != "streams_c"}
    values, slots, tiers = {}, {}, []

    def add(label, field, value):
        name_ = _tbd(label, field)
        values[name_[4:]] = value
        return name_

    def drain(after_label, rho, length_s):
        lab = "d%02d" % (len(slots) + 1)
        slot = drain_slot_s(rho, e_c, length_s, bracket=bracket and after_label != "warmup")
        slots[lab] = slot
        tiers.append({"label": lab, "kind": "drain", "window_ns": add(lab, "window_ns",
                                                                      slot * NS)})

    tiers.append({"label": "warmup", "kind": "warmup", "ramp_ns": 0, "tail_ns": 0,
                  "window_ns": add("warmup", "window_ns", WARMUP_S * NS),
                  "rate_rps": add("warmup", "rate_rps", BASE_RHO * center)})
    drain("warmup", BASE_RHO, WARMUP_S)
    for t in seq:
        lab = t["label"]
        tiers.append({"label": lab, "kind": t["kind"], "baseline": t["role"] == "baseline",
                      "ramp_ns": add(lab, "ramp_ns", int(round(t["ramp_s"] * NS))),
                      "window_ns": add(lab, "window_ns", int(round(t["window_s"] * NS))),
                      "tail_ns": add(lab, "tail_ns", int(round(t["tail_s"] * NS))),
                      "rate_rps": add(lab, "rate_rps", t["rho"] * center)})
        drain(lab, t["rho"], t["ramp_s"] + t["window_s"] + t["tail_s"])
    tpl["tiers"] = tiers
    if timeout:
        values[TIMEOUT_NAME] = timeout[0]
    used = tiergen.tbd_names(tpl)
    for k in sorted(used - set(values)):
        if k not in base_values:
            raise LadderError("--base-values has no value for %r" % k)
        if base_values[k] is None:
            raise LadderError("--base-values %r is null (an unfilled checklist?); a table without "
                              "a request timeout is asked for with --request-timeout-ns null"
                              % k)
        values[k] = base_values[k]
    return tpl, values, slots


def ladder_roles(seq, doc):
    """ladder.json role lists for the expanded tier table doc."""
    by_label = {t["label"]: t for t in doc["tiers"]}
    out = {"baseline_seq": None, "levels": [], "near": [], "cal": [], "holds": [], "slope": []}
    levels = {}
    for t in seq:
        tier = by_label[t["label"]]
        role, ex = t["role"], t["extra"]
        if role == "baseline":
            out["baseline_seq"] = tier["seq"]
        elif role == "levels":
            lv = levels.setdefault(ex["level"], {"level": ex["level"], "rho": t["rho"],
                                                 "load_value": tier["load_value"],
                                                 "up_seq": None, "down_seq": None})
            lv["up_seq" if ex["passes"] == 1 else "down_seq"] = tier["seq"]
        elif role in ("near", "cal"):
            out[role].append({"rho": t["rho"], "seq": tier["seq"], "pass": ex["pass_"]})
        elif role == "holds":
            out["holds"].append({"rho": t["rho"], "seq": tier["seq"], "hold_ns": ex["hold_ns"]})
        else:
            out["slope"].append({"rho": t["rho"], "seq": tier["seq"]})
    out["levels"] = [levels[k] for k in sorted(levels)]
    return out


def packing(table_s, a):
    """DESIGN 4.1 job condition and capjobs.submit.time_budget for --launches-per-job launches of
    this table; raises when it does not fit (nothing is written then)."""
    from capjobs import launchplan, submit  # noqa: PLC0415

    n = a.launches_per_job
    if not (isinstance(n, int) and n >= 1):
        raise LadderError("--launches-per-job must be >= 1")
    if not (a.job_overhead_s_max >= JOB_OVERHEAD_FLOOR_S):
        raise LadderError("--job-overhead-s-max must be >= %s (1.5 x 509 s)" % JOB_OVERHEAD_FLOOR_S)
    if not 0 < a.time_limit_s <= QOS_LIMIT_S:
        raise LadderError("--time-limit-s must be in (0, %d]" % QOS_LIMIT_S)
    budget_s = ceil10(table_s + LAUNCH_OVERHEAD_S + BUDGET_PAD_S)
    plan = {"launches": [{"budget_s": budget_s}] * n, "kill_grace_s": a.kill_grace_s,
            "margin_s": a.margin_s}
    # the calibration v2 scenarios declare no job inputs (scenario job_inputs, errata B2B-9)
    tb = submit.time_budget(plan, {"idle_window_s": a.idle_window_s,
                                   "slurm_view_wait_s": a.slurm_view_wait_s},
                            job_input_kinds=0)
    job_cond = n * (table_s + LAUNCH_OVERHEAD_S) + a.job_overhead_s_max
    out = {"launches_per_job": n, "table_s": table_s, "budget_s": budget_s,
           "time_budget_s": tb["total_s"], "time_limit_s": a.time_limit_s,
           "pins_timeout_s": launchplan.PINS_TIMEOUT_S, "kill_grace_s": a.kill_grace_s,
           "margin_s": a.margin_s, "idle_window_s": a.idle_window_s,
           "slurm_view_wait_s": a.slurm_view_wait_s, "job_overhead_s_max": a.job_overhead_s_max,
           "job_condition_s": job_cond, "job_condition_max_s": QOS_LIMIT_S - JOB_COND_PAD_S}
    if job_cond > QOS_LIMIT_S - JOB_COND_PAD_S:
        raise LadderError("does not fit: %d x (table %.1f s + %.1f) + job overhead %.1f = %.1f s > "
                          "%d s (DESIGN 4.1 job condition)" % (n, table_s, LAUNCH_OVERHEAD_S,
                                                               a.job_overhead_s_max, job_cond,
                                                               QOS_LIMIT_S - JOB_COND_PAD_S))
    if tb["total_s"] > a.time_limit_s:
        raise LadderError("does not fit: capjobs time budget %d s > --time-limit-s %d"
                          % (tb["total_s"], a.time_limit_s))
    return out


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_bytes(doc):
    return (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode("utf-8")


def phase_e_c(a):
    reg = E_C_BY_PHASE.get(a.phase)
    if reg is not None:
        if a.e_c is not None and a.e_c != reg:
            raise LadderError("phase %s registers e_c = %s (DESIGN 4.1); --e-c %s differs"
                              % (a.phase, reg, a.e_c))
        return reg
    if a.e_c is None or not 0 < a.e_c < 1:
        raise LadderError("phase %s has no registered e_c: give --e-c in (0, 1)" % a.phase)
    return a.e_c


def generate(a):
    """{file name: bytes} of one launch (template, values, ladder.json) and the summary."""
    if a.phase not in PHASES:
        raise LadderError("--phase must be one of %s" % (PHASES,))
    if not isinstance(a.config, str) or not a.config or not a.arm or not ARM_RE.match(a.arm):
        raise LadderError("--config and a valid --arm are required")
    if a.name is None or not ARM_RE.match(a.name):
        raise LadderError("--name must match %s" % ARM_RE.pattern)
    if not (isinstance(a.settle_s, (int, float)) and a.settle_s >= 0):
        raise LadderError("--settle-s must be >= 0")
    e_c = phase_e_c(a)
    center, center_source = read_center(a)
    u, seed = None, None
    if a.kind in ("ladder", "e2e", "hold"):
        if not (a.seed_base and a.block_id and a.slot is not None):
            raise LadderError("--kind %s needs --seed-base, --block-id and --slot (DESIGN 3.4)"
                              % a.kind)
        u, sha = seed_u(a.seed_base, a.config, a.block_id, a.slot)
        seed = {"base": a.seed_base, "block_id": a.block_id, "slot": a.slot, "sha": sha}
    seq, h = tier_sequence(a, 0.0 if u is None else u)
    if a.request_timeout_ns is None and a.kind in TIMEOUT_REQUIRED_KINDS:
        raise LadderError("--kind %s needs --request-timeout-ns (null or ns; DESIGN 4.2 s1-pc / "
                          "s*-fix)" % a.kind)
    timeout = () if a.request_timeout_ns is None else (parse_timeout(a.request_timeout_ns),)
    base = _read_json(a.base_template)
    bvals = _read_json(a.base_values)
    if bvals.get("schema") != tiergen.VALUES_SCHEMA or not isinstance(bvals.get("values"), dict):
        raise LadderError("--base-values must be %s" % tiergen.VALUES_SCHEMA)
    tpl, values, slots = build_template(base, bvals["values"], a.name, seq, center, e_c,
                                        a.kind == "bracket", timeout)
    doc = tiergen.expand(tpl, values)  # TierTable validation (exit 3 on any violation)
    table_s = doc["tiers"][-1]["start_ns"] / NS + doc["tiers"][-1]["window_ns"] / NS
    pack = packing(table_s, a)
    tpl_b = _json_bytes(tpl)
    vdoc = {"schema": tiergen.VALUES_SCHEMA, "values": values}
    nulls = sorted(k for k, v in values.items() if v is None)
    if nulls:  # only the request timeout can be null here (build_template)
        vdoc[tiergen.REGISTERED_NULLS_KEY] = nulls
    val_b = _json_bytes(vdoc)
    tiers_b = (json.dumps(doc, sort_keys=True, indent=1) + "\n").encode("utf-8")  # = tiergen's
    by_label = {t["label"]: t["seq"] for t in doc["tiers"]}
    lad = {"schema": LADDER_SCHEMA, "kind": a.kind, "phase": a.phase, "config": a.config,
           "arm": a.arm, "c_lad": center, "center_source": center_source, "axis": "rate_rps",
           "h": h, "K": a.K if a.kind == "ladder" else None,
           "span_half": a.span_half if a.kind == "ladder" else None,
           "passes": a.passes if a.kind == "ladder" else (2 if a.kind == "e2e" else 1),
           "u": u, "settle_ns": int(a.settle_s * NS), "ramp_base_ns": RAMP_BASE_S * NS,
           "seed": seed,
           "drain_rule": {"formula": "max(60, ceil10(1.5 * (rho*(1+e_c) - 1)+ * prev_len_s + 30))"
                                     + ("; 180 s when rho*(1+e_c) > 1 (bracket)"
                                        if a.kind == "bracket" else ""),
                          "e_c": e_c, "slot_s_by_seq": {str(by_label[k]): v
                                                        for k, v in sorted(slots.items())}},
           "packing": pack, "template_sha256": sha256_bytes(tpl_b),
           "values_sha256": sha256_bytes(val_b), "tier_table_sha256": sha256_bytes(tiers_b)}
    lad.update(ladder_roles(seq, doc))
    files = {"%s.tiers.template.json" % a.name: tpl_b, "%s.values.json" % a.name: val_b,
             "%s.ladder.json" % a.name: _json_bytes(lad)}
    return files, {"table_s": table_s, "packing": pack, "files": sorted(files)}


def write_new(out_dir, files):
    """Write every file with open(..., "x"); refuse when any exists (nothing written then)."""
    paths = {n: os.path.join(out_dir, n) for n in files}
    clash = [p for p in paths.values() if os.path.lexists(p)]
    if clash:
        raise LadderError("refusing to overwrite %s" % clash)
    for n, p in paths.items():
        with open(p, "xb") as f:
            f.write(files[n])
    return sorted(paths.values())


# ---- --plan: K_fit (DESIGN 2.7, 4.1; the r4 model) ---------------------------------------------

def _model_tier(rho, settle_s):
    """Conservative tier length of the r4 model: load tier + its drain slot (backlog capped by
    the 60 s request timeout)."""
    t = RAMP_BASE_S + settle_s + WINDOW_S + TAIL_S
    return t + max(DRAIN_MIN_S, ceil10(1.5 * min(max(0.0, rho - 1) * t, 60.0) + 30))


def model_ladder_s(K, S, span_half, e_c=PLAN_E_C):
    """Every level at the top rho (1 + span_half)(1 + e_c); warm-up 10 s + 60 s slot; baseline;
    2K levels; 3 slope tiers."""
    rmax = (1 + span_half) * (1 + e_c)
    return WARMUP_S + DRAIN_MIN_S + _model_tier(BASE_RHO, S) + 2 * K * _model_tier(rmax, S) \
        + sum(_model_tier(r * (1 + e_c), SLOPE_SETTLE_S) for r in SLOPE_RHOS)


def model_per_job(table_s, kill_grace_s=90, margin_s=60, pins_s=120, idle_s=30, view_s=300):
    per = ceil10(table_s + LAUNCH_OVERHEAD_S + BUDGET_PAD_S) + kill_grace_s + margin_s + pins_s
    return int((QOS_LIMIT_S - idle_s - view_s) // per)


def model_fits(table_s, job_overhead_s_max):
    table_max = QOS_LIMIT_S - JOB_COND_PAD_S - LAUNCH_OVERHEAD_S - job_overhead_s_max
    return model_per_job(table_s) >= 1 and table_s <= table_max


def k_fit(S, job_overhead_s_max, h=H_LADDER):
    for K in range(30, 10, -2):
        if model_fits(model_ladder_s(K, S, K * h / 2), job_overhead_s_max):
            return K
    return None


def s_fit_by_k(job_overhead_s_max, h=H_LADDER):
    out = {}
    for K in PLAN_K_RANGE:
        best = None
        for S in range(0, 1001, 10):
            if model_fits(model_ladder_s(K, S, K * h / 2), job_overhead_s_max):
                best = S
        out[str(K)] = best
    return out


def plan_report(a):
    if not a.job_overhead_s_max >= JOB_OVERHEAD_FLOOR_S:
        raise LadderError("--job-overhead-s-max must be >= %s" % JOB_OVERHEAD_FLOOR_S)
    out = {"model": "r4 (every level at the top rho; DESIGN 2.7, 4.1)", "e_c": PLAN_E_C,
           "job_overhead_s_max": a.job_overhead_s_max,
           "table_max_s": QOS_LIMIT_S - JOB_COND_PAD_S - LAUNCH_OVERHEAD_S - a.job_overhead_s_max,
           "S_fit_by_K": s_fit_by_k(a.job_overhead_s_max)}
    if a.settle_s is not None:
        out["K_fit"] = k_fit(a.settle_s, a.job_overhead_s_max)
        if a.K is not None:
            span = a.span_half if a.span_half is not None else a.K * H_LADDER / 2
            T = model_ladder_s(a.K, a.settle_s, span)
            out.update({"K": a.K, "span_half": span, "settle_s": a.settle_s, "launch_s": T,
                        "per_job": model_per_job(T),
                        "fits": model_fits(T, a.job_overhead_s_max)})
    return out


# ---- --job-plan / --plan-from: job_plan.tsv (DESIGN 3.2, 4.5) -----------------------------------

def _mix_key(seed_base, *parts):
    return hashlib.sha256("|".join([seed_base] + [str(p) for p in parts]).encode(
        "utf-8")).hexdigest()


def _pack(tokens, per_job, seed_base, tag):
    """tokens [(config, arm, n)] -> jobs [[(config, arm)]]: launches mixed by the seed, then cut
    into jobs of per_job launches (the last one may be shorter)."""
    if not (isinstance(per_job, int) and per_job >= 1):
        raise LadderError("launches_per_job must be an int >= 1")
    flat = [(c, arm, i) for c, arm, n in tokens for i in range(n)]
    flat.sort(key=lambda x: _mix_key(seed_base, tag, x[0], x[1], x[2]))
    return [[(c, arm) for c, arm, _ in flat[i:i + per_job]] for i in range(0, len(flat), per_job)]


def _one_each(tokens, per_job, seed_base, tag):
    """tokens [(config, arm, n)] -> n jobs, each running every configuration exactly once in the
    order of sha256(seed_base|tag|one_each|job|config|arm) (DESIGN 4.2 V0-cal)."""
    if len({c for c, _a, _n in tokens}) != len(tokens) \
            or len({arm for _c, arm, _n in tokens}) != len(tokens):
        raise LadderError("layout one_each needs distinct configs and arms")
    counts = {n for _c, _a, n in tokens}
    if len(counts) > 1:
        raise LadderError("layout one_each needs the same %s count for every config, got %s"
                          % (tag, sorted((c, n) for c, _a, n in tokens)))
    if per_job != len(tokens):
        raise LadderError("layout one_each needs launches_per_job = the number of configs (%d), "
                          "got %r" % (len(tokens), per_job))
    n = counts.pop() if counts else 0
    return [[(c, arm) for c, arm, _n in sorted(
        tokens, key=lambda x, j=j: _mix_key(seed_base, tag, "one_each", j, x[0], x[1]))]
        for j in range(n)]


def _job_item(job, item_class, pin):
    arms, counts = [], {}
    for c, arm in job:
        if arm not in arms:
            arms.append(arm)
        counts[c] = counts.get(c, 0) + 1
    return {"item_class": item_class, "arms": arms, "launches": len(job), "block_id": None,
            "slot": None, "config_counts": counts, "pin": pin}


def job_plan_items(spec):
    """Items of one segment from a capkit.job_plan_spec/1 document."""
    if not isinstance(spec, dict) or spec.get("schema") != SPEC_SCHEMA:
        raise LadderError("--job-plan spec must be %s" % SPEC_SCHEMA)
    pid, sb = spec.get("phase_id"), spec.get("seed_base")
    if not (isinstance(pid, str) and jobplan.ITEM_ID_RE.match(pid) and isinstance(sb, str) and sb):
        raise LadderError("spec needs phase_id and seed_base")
    if not isinstance(spec.get("pin", False), bool):
        raise LadderError("spec pin must be true or false")
    layout, phase = spec.get("layout", "mixed"), spec.get("phase")
    if layout not in LAYOUTS:
        raise LadderError("spec layout must be one of %s" % (LAYOUTS,))
    if phase is not None and phase not in PHASES:
        raise LadderError("spec phase must be one of %s" % (PHASES,))
    if phase in ONE_EACH_PHASES and layout != "one_each":
        raise LadderError("phase %s needs layout one_each: every job runs every configuration "
                          "exactly once (DESIGN 4.2)" % phase)
    units = {"planned": [], "surplus": []}
    toks = {"planned": [], "surplus": []}
    if not all(isinstance(spec.get(k, []), list) for k in ("configs", "blocks")):
        raise LadderError("spec configs and blocks must be lists")
    for c in spec.get("configs") or []:
        if not (isinstance(c, dict) and isinstance(c.get("config"), str)
                and ARM_RE.match(str(c.get("arm")))):
            raise LadderError("spec configs[] need config and arm")
        for cls in units:
            n = c.get(cls, 0)
            if not (isinstance(n, int) and not isinstance(n, bool) and n >= 0):
                raise LadderError("spec %s count must be an int >= 0" % cls)
            toks[cls].append((c["config"], c["arm"], n))
    for cls in units:
        if layout == "one_each":
            jobs = _one_each(toks[cls], spec.get("launches_per_job"), sb, cls)
        else:
            jobs = _pack(toks[cls], spec.get("launches_per_job"), sb, cls)
        for j, job in enumerate(jobs):
            key = _mix_key(sb, cls, *[x for p in job for x in p]) if layout == "mixed" \
                else _mix_key(sb, cls, "one_each", j)
            units[cls].append(("job", key, [_job_item(job, cls, spec.get("pin", False))]))
    for b in spec.get("blocks") or []:
        cmp_ = b.get("comparison") if isinstance(b, dict) else None
        pair = b.get("configs") if isinstance(b, dict) else None
        if not (isinstance(cmp_, str) and jobplan.ITEM_ID_RE.match(cmp_) and isinstance(pair, list)
                and len(pair) == 2 and all(isinstance(p, list) and len(p) == 2 for p in pair)):
            raise LadderError("spec blocks[] need comparison and configs [[config, arm], x2]")
        n_planned, n_surplus = b.get("planned", 0), b.get("surplus", 0)
        if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0
                   for n in (n_planned, n_surplus)):
            raise LadderError("spec blocks[] planned / surplus must be ints >= 0")
        for k in range(n_planned + n_surplus):
            cls = "planned" if k < n_planned else "surplus"
            bid = "%s-b%02d" % (cmp_, k + 1)
            u, _ = seed_u(sb, cmp_, bid, 0)
            first = pair if u < 0.5 else pair[::-1]
            items = []
            for slot, (config, arm) in enumerate(first, start=1):
                barm = "%s.%s.%d" % (arm, bid, slot)
                if not ARM_RE.match(barm):
                    raise LadderError("block arm %r is not a valid arm name" % barm)
                items.append({"item_class": cls, "arms": [barm], "launches": 1, "block_id": bid,
                              "slot": slot, "config_counts": {config: 1}, "pin": True})
            units[cls].append(("block", _mix_key(sb, cls, cmp_, bid), items))
    out, order = [], 0
    for cls in ("planned", "surplus"):
        for kind_, _key, items in sorted(units[cls], key=lambda x: x[1]):
            order += 1
            for it in items:
                it["order"] = order
                it["item_id"] = ("%s-%s-s%d" % (pid, it["block_id"], it["slot"])
                                 if kind_ == "block" else "%s-%04d" % (pid, order))
                out.append(it)
    return jobplan.validate(out)


def plan_from_items(a):
    """Post-gate items of a registered gate product (REGISTRY kind e0_params, sha checked; an
    int `round` >= 1 required): launch items {config, arm, launches, pin} packed
    --launches-per-job to a job (pinned and unpinned apart), item ids
    '<phase_id>-<rule>-r<round>-NNNN', then block items {comparison, configs [[config, arm] x2],
    blocks, pin true} as blocks of two pinned one-launch jobs sharing one order, block ids
    '<comparison>-b<rule><round><NN>'. The round is in both, so no id repeats one of the
    original segment's or of another round of the same phase. A product with
    cap.launches_per_job (shortfall: its job cap was counted with it) needs the same
    --launches-per-job (calibv2a-RS33)."""
    if a.workdir is None or not a.phase_id or not a.seed_base:
        raise LadderError("--plan-from needs --workdir, --phase-id and --seed-base")
    rel, sha, doc = registered(a.workdir, a.plan_from, "e0_params")
    if not isinstance(doc, dict) or doc.get("schema") != E0_PARAMS_SCHEMA:
        raise LadderError("%s is not %s" % (a.plan_from, E0_PARAMS_SCHEMA))
    src = doc.get("items")
    if not isinstance(src, list) or not src:
        raise LadderError("%s lists no post-gate items" % rel)
    classes = {it.get("item_class") if isinstance(it, dict) else None for it in src}
    if len(classes) != 1:
        raise LadderError("a gate product lists one post-gate class, got %s" % sorted(
            map(str, classes)))
    cls = jobplan.check_item_class(classes.pop())
    if not cls.startswith(jobplan.POST_GATE_PREFIX):
        raise LadderError("--plan-from writes post-gate items only, not %r" % cls)
    rule = cls[len(jobplan.POST_GATE_PREFIX):]
    tag = rule.replace("_", "")
    rnd = doc.get("round")
    if not (isinstance(rnd, int) and not isinstance(rnd, bool) and rnd >= 1):
        raise LadderError("%s has no int round >= 1 (item and block ids carry it)" % rel)
    cap = doc.get("cap", {})
    if not isinstance(cap, dict):
        raise LadderError("%s cap must be an object" % rel)
    if "launches_per_job" in cap and cap["launches_per_job"] != a.launches_per_job:
        raise LadderError("--launches-per-job %r != %s cap.launches_per_job %r (the gate sized "
                          "its job cap with it)" % (a.launches_per_job, rel,
                                                    cap["launches_per_job"]))
    launch_items = [it for it in src if "comparison" not in it]
    block_items = [it for it in src if "comparison" in it]
    out = []
    for pin in (False, True):
        toks = []
        for it in launch_items:
            if not (isinstance(it.get("config"), str) and ARM_RE.match(str(it.get("arm")))
                    and isinstance(it.get("launches"), int) and it["launches"] > 0
                    and isinstance(it.get("pin"), bool)):
                raise LadderError("gate product items need config, arm, launches > 0, pin "
                                  "(or comparison, configs, blocks, pin for blocks)")
            if it["pin"] == pin:
                toks.append((it["config"], it["arm"], it["launches"]))
        for job in _pack(toks, a.launches_per_job, a.seed_base, cls):
            out.append(_job_item(job, cls, pin))
    for k, it in enumerate(out, start=1):
        it["order"] = k
        it["item_id"] = "%s-%s-r%d-%04d" % (a.phase_id, tag, rnd, k)
    order = len(out)
    for it in block_items:
        _check_block_item(it)
        for k in range(it["blocks"]):
            order += 1
            bid = "%s-b%s%d%02d" % (it["comparison"], tag, rnd, k + 1)
            out += _block_items(it["comparison"], bid, it["configs"], a.seed_base, cls, order,
                                "%s-%s-%s" % (a.phase_id, tag, bid))
    return jobplan.validate(out), {"relpath": rel, "sha256": sha}


def _check_block_item(it):
    pair = it.get("configs")
    if not (isinstance(it.get("comparison"), str) and jobplan.ITEM_ID_RE.match(it["comparison"])
            and isinstance(pair, list) and len(pair) == 2
            and all(isinstance(p, list) and len(p) == 2 and isinstance(p[0], str)
                    and ARM_RE.match(str(p[1])) for p in pair)
            and isinstance(it.get("blocks"), int) and not isinstance(it["blocks"], bool)
            and it["blocks"] > 0 and it.get("pin") is True):
        raise LadderError("gate product block items need comparison, configs [[config, arm], "
                          "x2], blocks > 0 and pin true")


def _block_items(cmp_, bid, pair, seed_base, cls, order, id_prefix):
    """The two pinned one-launch jobs of block `bid` (slot order by the seed, as in
    job_plan_items); arms '<arm>.<bid>.<slot>', item ids '<id_prefix>-s<slot>'."""
    u, _ = seed_u(seed_base, cmp_, bid, 0)
    first = pair if u < 0.5 else pair[::-1]
    out = []
    for slot, (config, arm) in enumerate(first, start=1):
        barm = "%s.%s.%d" % (arm, bid, slot)
        if not ARM_RE.match(barm):
            raise LadderError("block arm %r is not a valid arm name" % barm)
        out.append({"item_class": cls, "arms": [barm], "launches": 1, "block_id": bid,
                    "slot": slot, "config_counts": {config: 1}, "pin": True, "order": order,
                    "item_id": "%s-s%d" % (id_prefix, slot)})
    return out


# ---- main ---------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="python -m caplaunch.laddergen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--kind", choices=KINDS)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--job-plan", metavar="SPEC")
    mode.add_argument("--plan-from", metavar="GATE_PRODUCT")
    p.add_argument("--workdir", help="<W>: REGISTRY.tsv and the registered files")
    p.add_argument("--phase", help="one of %s" % (PHASES,))
    p.add_argument("--e-c", type=float, help="only for phases without a registered e_c")
    p.add_argument("--config")
    p.add_argument("--arm")
    p.add_argument("--name", help="output file stem (= tier table name)")
    p.add_argument("--base-template", help="capkit.tiers_template/1 of the scenario")
    p.add_argument("--base-values", help="capkit.tier_values/1 for its non-tier TBD names")
    p.add_argument("--center", help="registered e0_params file")
    p.add_argument("--center-key", choices=sorted(CENTER_KEY_STAGE))
    p.add_argument("--center-literal", type=float)
    p.add_argument("--request-timeout-ns", metavar="{null,NS}",
                   help="the table's request_timeout_ns: null (a registered null) or ns > 0")
    p.add_argument("--settle-s", type=float)
    p.add_argument("--K", type=int)
    p.add_argument("--span-half", type=float)
    p.add_argument("--h", type=float, default=H_LADDER)
    p.add_argument("--passes", type=int, choices=(1, 2), default=2)
    p.add_argument("--rho-first", type=float)
    p.add_argument("--ratio", type=float)
    p.add_argument("--n-levels", type=int)
    p.add_argument("--rho-top", type=float)
    p.add_argument("--seed-base")
    p.add_argument("--block-id")
    p.add_argument("--slot", type=int)
    p.add_argument("--launches-per-job", type=int)
    p.add_argument("--job-overhead-s-max", type=float, default=JOB_OVERHEAD_FLOOR_S)
    p.add_argument("--time-limit-s", type=int, default=QOS_LIMIT_S)
    p.add_argument("--kill-grace-s", type=int, default=90)
    p.add_argument("--margin-s", type=int, default=60)
    p.add_argument("--idle-window-s", type=float, default=30)
    p.add_argument("--slurm-view-wait-s", type=float, default=300)
    p.add_argument("--phase-id", help="--plan-from: item id prefix")
    p.add_argument("--out-dir", help="--kind: directory of the three new files")
    p.add_argument("--out", help="--job-plan / --plan-from: the new job_plan.tsv")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        if a.plan:
            print(json.dumps(plan_report(a), indent=1, sort_keys=True))
            return 0
        if a.kind is not None:
            if not a.out_dir or not a.base_template or not a.base_values:
                raise LadderError("--kind needs --out-dir, --base-template and --base-values")
            files, summary = generate(a)
            summary["written"] = write_new(a.out_dir, files)
            print(json.dumps(summary, indent=1, sort_keys=True))
            return 0
        if not a.out:
            raise LadderError("--job-plan / --plan-from need --out")
        if a.job_plan is not None:
            items, source = job_plan_items(_read_json(a.job_plan)), None
        else:
            items, source = plan_from_items(a)
        data = jobplan.format_tsv(items)
        with open(a.out, "xb") as f:
            f.write(data)
        print(json.dumps({"out": os.path.abspath(a.out), "items": len(items),
                          "sha256": sha256_bytes(data), "gate_product": source},
                         indent=1, sort_keys=True))
        return 0
    except (ConfigError, OSError, ValueError) as e:
        print("laddergen: %s" % e, file=sys.stderr)
        return getattr(e, "exit_code", 3)


if __name__ == "__main__":
    sys.exit(main())
