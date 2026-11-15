# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibration v2 threshold sources (calib-v2 DESIGN-r7a 2.3, 2.11, 3.1 P2-02, 3.2).

A launch reads thresholds only from registered calibration files of two schemas:

* `capkit.pilot/2` (kind `pilot`), in one of two stages:
  - `thresholds`: per-key empirical thresholds `null_z` ("<T>:<quantity>:<role>|<config>|
    <regime>|L=<window s>", regime `subsat` or `near`), the X1 tolerances `ratio`
    ("T2:delivery_ratio_norm:<role>|*|*"), the C13 caps `c13_cap` and the I1 calibration
    `i1_calibration`;
  - `bootstrap`: one value per statistic family (`x_const` lag / count, `ratio_tol`, `c13_cap`),
    returned for every configuration and regime.
* `capkit.d14_result/1` (kind `d14_result`): the D14 conclusion (`d14_softirq`).

A v1 calibration file (`capkit.pilot/1`, or any file of kind `near_boundary_noise`) is not a
threshold source: add_source raises ThresholdError(V1_REJECTED). Statistic keys passed to get()
are "<T>:<quantity>:<role>" (thresholds.band_key); the configuration is the caller's (the judge
passes register.json arm_threshold_config[arm]) and the regime is the judge's tier regime
("subsat" / "near_boundary", written "near" in pilot/2 keys).
"""

import re

from . import stats
from .calib import ENV_KEYS, CalibError, _check_i1
from .thresholds import KEY_CODES, REGIMES

PILOT2_SCHEMA = "capkit.pilot/2"
PILOT1_SCHEMA = "capkit.pilot/1"
D14_SCHEMA = "capkit.d14_result/1"
STAGES = ("thresholds", "bootstrap")
KINDS = ("pilot", "d14_result")
V1_REJECTED = "v1 calibration is not a threshold source (calib-v2 erratum)"
# judge tier regime -> the regime word of a pilot/2 null_z key
REGIME_KEY = {"subsat": "subsat", "near_boundary": "near"}
FAMILIES = ("lag", "count")
LAG_QUANTITIES = ("req_first_lag_p50_ns", "send_lag_p50_ns", "msg_lag_p50_ns")
COUNT_QUANTITIES = ("inflight_derived",)
COUNT_PREFIX = "backlog."
RATIO_STAT = "T2:delivery_ratio_norm"
C13_STAT = "T3:foreign_busy_frac"
ENV_KEYS_PILOT2 = ENV_KEYS + ("judge_sha",)
_PART = r"[^:|*]+"
NULL_Z_KEY = re.compile(r"^(%s):(%s):(%s)\|([^|*]+)\|(subsat|near)\|L=([0-9]+)\Z"
                        % ("|".join(KEY_CODES), _PART, _PART))
RATIO_KEY = re.compile(r"^%s:(%s)\|\*\|\*\Z" % (re.escape(RATIO_STAT), _PART))
STAT_KEY = re.compile(r"^(%s):(%s):(%s)\Z" % ("|".join(KEY_CODES), _PART, _PART))
NS_PER_S = 1_000_000_000


class ThresholdError(Exception):
    pass


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def family_of(stat_key):
    """'lag', 'count', 'ratio' or None for a "<T>:<quantity>:<role>" key."""
    m = STAT_KEY.match(stat_key) if isinstance(stat_key, str) else None
    if m is None:
        raise ValueError("bad statistic key %r (want <T>:<quantity>:<role>)" % (stat_key,))
    code, qty = m.group(1), m.group(2)
    if code == "T1" and qty in LAG_QUANTITIES:
        return "lag"
    if code == "T1" and (qty in COUNT_QUANTITIES or qty.startswith(COUNT_PREFIX)):
        return "count"
    if "%s:%s" % (code, qty) == RATIO_STAT:
        return "ratio"
    return None


# ---- schema validation -----------------------------------------------------------------------

def _check_env(doc, keys, where):
    env = doc.get("environment")
    if not isinstance(env, dict) or any(k not in env for k in keys):
        raise ThresholdError("%s: environment must have %s" % (where, keys))


def _check_ref(ref, what, where):
    if not isinstance(ref, dict) or set(ref) != {"relpath", "sha256"} \
            or not isinstance(ref["relpath"], str) or not ref["relpath"] \
            or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64:
        raise ThresholdError("%s: %s must be {relpath, sha256}" % (where, what))


def _num_field(obj, key, where, lo=None, lo_open=False):
    v = obj.get(key) if isinstance(obj, dict) else None
    bad = not _is_num(v) or (lo is not None and (v <= lo if lo_open else v < lo))
    if bad:
        raise ThresholdError("%s.%s must be a number %s %s" % (
            where, key, ">" if lo_open else ">=", lo))
    return v


def _check_c13_cap(cap, where):
    if not isinstance(cap, dict) or set(cap) != {C13_STAT}:
        raise ThresholdError("%s: c13_cap must be {%r: {window, launch_mean}}" % (where,
                                                                                   C13_STAT))
    c = cap[C13_STAT]
    if not isinstance(c, dict) or set(c) != {"window", "launch_mean"}:
        raise ThresholdError("%s: c13_cap[%r] must be {window, launch_mean}" % (where, C13_STAT))
    for k in ("window", "launch_mean"):
        _num_field(c, k, "%s c13_cap" % where, lo=0, lo_open=True)


def _check_null_z(nz, window_s, where):
    if not isinstance(nz, dict):
        raise ThresholdError("%s: null_z must be an object" % where)
    for key, e in nz.items():
        m = NULL_Z_KEY.match(key) if isinstance(key, str) else None
        w = "%s null_z[%r]" % (where, key)
        if m is None:
            raise ThresholdError("%s: key must be <T>:<quantity>:<role>|<config>|subsat|near"
                                 "|L=<s>" % w)
        if int(m.group(6)) != window_s:
            raise ThresholdError("%s: L=%s != window_ns %d s" % (w, m.group(6), window_s))
        if not isinstance(e, dict) or e.get("family") not in FAMILIES \
                or not isinstance(e.get("uncalibrated"), bool):
            raise ThresholdError("%s must have family in %s and a bool uncalibrated"
                                 % (w, FAMILIES))
        if not e["uncalibrated"]:
            hi, lo = e.get("thr_hi"), e.get("thr_lo")
            if not _is_num(hi) or not _is_num(lo) or lo > hi:
                raise ThresholdError("%s: thr_lo <= thr_hi numbers required unless uncalibrated"
                                     % w)


def _check_ratio(ratio, where):
    if not isinstance(ratio, dict):
        raise ThresholdError("%s: ratio must be an object" % where)
    for key, e in ratio.items():
        if not isinstance(key, str) or RATIO_KEY.match(key) is None:
            raise ThresholdError("%s: ratio key %r must be %s:<role>|*|*" % (where, key,
                                                                            RATIO_STAT))
        _num_field(e, "abs_tol", "%s ratio[%r]" % (where, key), lo=0)


def _check_bootstrap(doc, where):
    xc = doc.get("x_const")
    if not isinstance(xc, dict) or set(xc) != {"lag", "count"} \
            or not isinstance(xc["lag"], dict) or set(xc["lag"]) != {"z", "min_D_ns"} \
            or not isinstance(xc["count"], dict) or set(xc["count"]) != {"z"}:
        raise ThresholdError("%s: x_const must be {lag: {z, min_D_ns}, count: {z}}" % where)
    _num_field(xc["lag"], "z", "%s x_const.lag" % where, lo=0, lo_open=True)
    _num_field(xc["lag"], "min_D_ns", "%s x_const.lag" % where, lo=0)
    _num_field(xc["count"], "z", "%s x_const.count" % where, lo=0, lo_open=True)
    rt = doc.get("ratio_tol")
    if not isinstance(rt, dict) or set(rt) != {RATIO_STAT} or not isinstance(rt[RATIO_STAT],
                                                                            dict) \
            or set(rt[RATIO_STAT]) != {"min_per_request", "tol"}:
        raise ThresholdError("%s: ratio_tol must be {%r: {min_per_request, tol}}"
                             % (where, RATIO_STAT))
    mpr = rt[RATIO_STAT]["min_per_request"]
    if not isinstance(mpr, int) or isinstance(mpr, bool) or mpr < 1:
        raise ThresholdError("%s: ratio_tol min_per_request must be an int >= 1" % where)
    _num_field(rt[RATIO_STAT], "tol", "%s ratio_tol" % where, lo=0)


def validate(kind, doc, where):
    """Raise ThresholdError unless `doc` is a calibration v2 threshold source of `kind`.
    Returns the pilot/2 stage, or None for a d14_result."""
    if kind == "near_boundary_noise":
        raise ThresholdError(V1_REJECTED)
    if kind not in KINDS:
        raise ThresholdError("%s: %r is not a calibration v2 threshold kind %s" % (where, kind,
                                                                                    KINDS))
    if not isinstance(doc, dict):
        raise ThresholdError("%s: not a JSON object" % where)
    schema = doc.get("schema")
    if kind == "d14_result":
        if schema != D14_SCHEMA:
            raise ThresholdError("%s: d14_result schema must be %s" % (where, D14_SCHEMA))
        _check_env(doc, ENV_KEYS, where)
        _check_ref(doc.get("rule"), "rule", where)
        d14 = doc.get("d14_softirq")
        if not isinstance(d14, dict) or not isinstance(d14.get("irq_softirq_is_experiment"),
                                                       bool):
            raise ThresholdError("%s: d14_softirq must carry a bool irq_softirq_is_experiment"
                                 % where)
        if not isinstance(doc.get("inputs"), list):
            raise ThresholdError("%s: inputs must be a list" % where)
        return None
    if schema == PILOT1_SCHEMA:
        raise ThresholdError(V1_REJECTED)
    if schema != PILOT2_SCHEMA:
        raise ThresholdError("%s: pilot schema must be %s" % (where, PILOT2_SCHEMA))
    stage = doc.get("stage")
    if stage not in STAGES:
        raise ThresholdError("%s: pilot/2 stage must be one of %s" % (where, STAGES))
    _check_env(doc, ENV_KEYS_PILOT2, where)
    _check_ref(doc.get("d14"), "d14", where)
    _check_c13_cap(doc.get("c13_cap"), where)
    if stage == "bootstrap":
        _check_bootstrap(doc, where)
        return stage
    win = doc.get("window_ns")
    if not isinstance(win, int) or isinstance(win, bool) or win <= 0 or win % NS_PER_S:
        raise ThresholdError("%s: window_ns must be a positive whole number of seconds in ns"
                             % where)
    _check_null_z(doc.get("null_z"), win // NS_PER_S, where)
    _check_ratio(doc.get("ratio"), where)
    try:
        _check_i1(doc.get("i1_calibration"), where)
    except CalibError as e:
        raise ThresholdError(str(e))
    return stage


# ---- the merged view of one launch -------------------------------------------------------------

class Thresholds:
    """The calibration v2 threshold sources of one launch: at most one pilot/2 (either stage)
    and at most one d14_result. A pilot/2 names its D14 file by sha; when both are registered
    the shas must agree."""

    def __init__(self):
        self._docs = {}  # kind -> document
        self._refs = {}  # kind -> (relpath, sha256)
        self.sources = []  # [{"kind", "relpath", "sha256", "schema", "stage"}]
        # "<stat key>|<config>|<regime>" or "cap|<key>" -> found / missing / uncalibrated
        self.lookups = {}

    def add_source(self, kind, relpath, sha256, doc):
        stage = validate(kind, doc, relpath)
        if kind in self._docs:
            raise ThresholdError("two calibration files of kind %s (%s, %s)"
                                 % (kind, self._refs[kind][0], relpath))
        self._docs[kind], self._refs[kind] = doc, (relpath, sha256)
        pilot, d14 = self._docs.get("pilot"), self._refs.get("d14_result")
        if pilot is not None and d14 is not None and pilot["d14"]["sha256"] != d14[1]:
            raise ThresholdError("pilot/2 %s names D14 file sha %s but the registered "
                                 "d14_result %s has sha %s" % (
                                     self._refs["pilot"][0], pilot["d14"]["sha256"], d14[0],
                                     d14[1]))
        self.sources.append({"kind": kind, "relpath": relpath, "sha256": sha256,
                             "schema": doc["schema"], "stage": stage})

    @property
    def stage(self):
        """'thresholds', 'bootstrap' or None (no pilot/2)."""
        p = self._docs.get("pilot")
        return None if p is None else p["stage"]

    def get(self, stat_key, config, regime):
        """The threshold entry of `stat_key` for `config` in `regime`, or None (no source for
        it). Entries are dicts with "form" in null_z / lag_const / count_const / ratio_abs /
        ratio_tol, read by compare(). A thresholds-stage key marked uncalibrated is returned
        (compare gives None for it). `config` None (a v1 register: no arm_config) reads only
        the configuration-free values: a bootstrap family value or a thresholds-stage ratio;
        a null_z key needs a configuration (missing). Lookups record it as "-"."""
        fam = family_of(stat_key)
        if regime not in REGIMES:
            raise ValueError("unknown regime %r" % (regime,))
        if config is not None and (not isinstance(config, str) or not config or "|" in config
                                   or "*" in config):
            raise ValueError("bad configuration name %r" % (config,))
        look = "%s|%s|%s" % (stat_key, "-" if config is None else config, regime)
        e = self._entry(stat_key, fam, config, regime)
        self.lookups[look] = "missing" if e is None else (
            "uncalibrated" if e.get("uncalibrated") else "found")
        return e

    def _entry(self, stat_key, fam, config, regime):
        p = self._docs.get("pilot")
        if p is None:
            return None
        src = {"source": self._refs["pilot"][0], "stage": p["stage"]}
        if p["stage"] == "bootstrap":
            if fam == "lag":
                c = p["x_const"]["lag"]
                return dict(src, form="lag_const", key="x_const.lag", z=c["z"],
                            min_D_ns=c["min_D_ns"])
            if fam == "count":
                return dict(src, form="count_const", key="x_const.count",
                            z=p["x_const"]["count"]["z"])
            if fam == "ratio":
                r = p["ratio_tol"][RATIO_STAT]
                return dict(src, form="ratio_tol", key="ratio_tol.%s" % RATIO_STAT,
                            tol=r["tol"], min_per_request=r["min_per_request"])
            return None
        if fam == "ratio":
            key = "%s|*|*" % stat_key
            r = p["ratio"].get(key)
            return None if r is None else dict(src, form="ratio_abs", key=key,
                                               abs_tol=r["abs_tol"])
        if config is None:
            return None
        key = "%s|%s|%s|L=%d" % (stat_key, config, REGIME_KEY[regime],
                                 p["window_ns"] // NS_PER_S)
        n = p["null_z"].get(key)
        if n is None:
            return None
        return dict(src, form="null_z", key=key, family=n["family"], thr_hi=n.get("thr_hi"),
                    thr_lo=n.get("thr_lo"), uncalibrated=n["uncalibrated"])

    def cap(self, key):
        """{"window", "launch_mean"} of a C13 cap key ("T3:foreign_busy_frac"), or None."""
        p = self._docs.get("pilot")
        c = None if p is None else p["c13_cap"].get(key)
        self.lookups["cap|%s" % key] = "missing" if c is None else "found"
        return None if c is None else dict(c)

    def d14(self):
        """The registered d14_result's d14_softirq, or None."""
        d = self._docs.get("d14_result")
        return None if d is None else d["d14_softirq"]

    def i1_near(self):
        """near_one_core_min of a thresholds-stage pilot/2's i1_calibration, or None."""
        p = self._docs.get("pilot")
        i1 = None if p is None else p.get("i1_calibration")
        return None if not i1 else i1["near_one_core_min"]


def compare(z, entry, D=None, planned_requests=None):
    """'above', 'below', 'inside', or None (no entry, uncalibrated key, or no value).

    `z` is the trend statistic z_d for null_z / lag_const / count_const entries and the signed
    normalised delivery deviation (norm - 1) for ratio entries. A lag_const entry also needs
    `D` (the trend in raw units, ns): above iff z >= z_lag and D >= min_D_ns. Ratio entries
    need `planned_requests` of the tier: the tolerance is max(min_per_request /
    planned_requests, tol) for ratio_tol and max(1 / planned_requests, abs_tol) for ratio_abs
    (DESIGN-r7a 2.9: the V2 quantile has the floor 1 / planned requests). null_z, count_const
    and both ratio forms compare strictly (>, <)."""
    if entry is None or z is None or isinstance(z, stats.NotRun):
        return None
    if not _is_num(z):
        raise ValueError("value must be a number, got %r" % (z,))
    form = entry["form"]
    if form == "null_z":
        if entry["uncalibrated"]:
            return None
        return "above" if z > entry["thr_hi"] else "below" if z < entry["thr_lo"] else "inside"
    if form == "lag_const":
        if not _is_num(D):
            raise ValueError("a lag_const comparison needs D (ns), got %r" % (D,))
        return "above" if z >= entry["z"] and D >= entry["min_D_ns"] else "inside"
    if form == "count_const":
        return "above" if z > entry["z"] else "below" if z < -entry["z"] else "inside"
    if form in ("ratio_abs", "ratio_tol"):
        if not isinstance(planned_requests, int) or isinstance(planned_requests, bool) \
                or planned_requests < 1:
            raise ValueError("a %s comparison needs planned_requests >= 1, got %r"
                             % (form, planned_requests))
        if form == "ratio_tol":
            tol = max(entry["min_per_request"] / planned_requests, entry["tol"])
        else:
            tol = max(1 / planned_requests, entry["abs_tol"])
        if abs(z) > tol:
            return "above" if z > 0 else "below"
        return "inside"
    raise ValueError("unknown threshold form %r" % (form,))
