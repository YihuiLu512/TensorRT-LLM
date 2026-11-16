# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tier table load/validate/query (INTERFACES.md section 3)."""

import bisect
import dataclasses
import fractions
import hashlib
import json
import math

from .exitcodes import ConfigError
from .hashing import sha256_file
from .identity import TIER_LABEL_RE
from .inject import InjectSpec, InjectSpecError

SCHEMA = "capkit.tiers/1"
KINDS = ("warmup", "up", "down", "drain")
LOAD_KINDS = ("warmup", "up", "down")
PRE_SEQ = -1  # events before the origin
POST_SEQ = -2  # events after the last tier
PRE_LABEL = "pre"
POST_LABEL = "post"
GATE_MODES = ("enforce", "record_only")
GATE_RULES = ("abs_max", "baseline_band")
GATE_AGGS = ("sum", "max")

_TIER_FIELDS = ("seq", "label", "kind", "start_ns", "ramp_ns", "window_ns", "tail_ns", "rate_rps",
                "streams_c", "max_tokens", "tokens_per_msg", "stream_interval", "beat_ns",
                "prompt_set", "inject", "gate", "baseline", "note", "load_value")
_TOP_FIELDS = ("schema", "experiment", "name", "arrival", "axis", "request_timeout_ns",
               "beat_origin_rel_ns", "default_gate", "launch_inject", "tiers")
AXIS_NAMES = ("R", "M", "K", "L")
AXIS_LOAD_KEYS = ("rate_rps", "nominal_msgs_per_s", "streams_c", "beat_ns", "tokens_per_msg",
                  "prompt_len")
AXIS_ORDERS = ("asc", "desc")  # asc: a larger load_value is more load; desc: smaller is more


class TierTableError(ConfigError):
    pass


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


@dataclasses.dataclass(frozen=True)
class Tier:
    seq: int
    label: str
    kind: str
    start_ns: int
    ramp_ns: int
    window_ns: int
    tail_ns: int
    rate_rps: float
    streams_c: object
    max_tokens: int
    tokens_per_msg: int
    stream_interval: int
    beat_ns: int
    prompt_set: object
    inject: tuple
    gate: object
    baseline: bool
    note: str
    load_value: object = None

    @property
    def duration_ns(self):
        return self.ramp_ns + self.window_ns + self.tail_ns

    @property
    def end_ns(self):
        return self.start_ns + self.duration_ns

    @property
    def window_start_ns(self):
        return self.start_ns + self.ramp_ns

    @property
    def window_end_ns(self):
        return self.start_ns + self.ramp_ns + self.window_ns

    @property
    def is_load(self):
        return self.kind in LOAD_KINDS

    @property
    def tokens_per_step(self):
        return self.tokens_per_msg // self.stream_interval

    def steps_per_stream(self):
        """Steps a stream of max_tokens needs: ceil(max_tokens / tokens_per_step)."""
        if self.max_tokens <= 0:
            return 0
        return -(-self.max_tokens // self.tokens_per_step)

    def stream_duration_ns(self):
        return self.steps_per_stream() * self.beat_ns


def _validate_gate(g, where):
    if not isinstance(g, dict):
        raise TierTableError("%s: gate must be an object" % where)
    allowed = {"mode", "rule", "readings", "poll_ns", "settle_polls", "deadline_guard_ns",
               "max_snapshot_age_ns", "band_ref"}
    extra = set(g) - allowed
    if extra:
        raise TierTableError("%s: unknown gate keys %s" % (where, sorted(extra)))
    if g.get("mode") not in GATE_MODES:
        raise TierTableError("%s: gate.mode must be one of %s" % (where, GATE_MODES))
    if g.get("rule") not in GATE_RULES:
        raise TierTableError("%s: gate.rule must be one of %s" % (where, GATE_RULES))
    for k in ("poll_ns", "settle_polls", "max_snapshot_age_ns"):
        if not _is_int(g.get(k)) or g[k] <= 0:
            raise TierTableError("%s: gate.%s must be int > 0" % (where, k))
    if not _is_int(g.get("deadline_guard_ns")) or g["deadline_guard_ns"] < 0:
        raise TierTableError("%s: gate.deadline_guard_ns must be int >= 0" % where)
    if g["rule"] == "baseline_band":
        ref = g.get("band_ref")
        if not isinstance(ref, dict) or not isinstance(ref.get("calib_kind"), str):
            raise TierTableError("%s: baseline_band needs band_ref.calib_kind" % where)
    rd = g.get("readings")
    if not isinstance(rd, list) or not rd:
        raise TierTableError("%s: gate.readings must be a non-empty list" % where)
    names = set()
    for r in rd:
        if not isinstance(r, dict) or not isinstance(r.get("name"), str) or not r["name"]:
            raise TierTableError("%s: each reading needs a name" % where)
        if r["name"] in names:
            raise TierTableError("%s: duplicate reading %s" % (where, r["name"]))
        names.add(r["name"])
        terms = r.get("terms")
        if not isinstance(terms, list) or not terms:
            raise TierTableError("%s: reading %s needs terms" % (where, r["name"]))
        for t in terms:
            if not isinstance(t, dict) or set(t) != {"role", "slot", "sign", "agg"}:
                raise TierTableError("%s: reading %s term needs exactly role/slot/sign/agg"
                                     % (where, r["name"]))
            if not isinstance(t["slot"], str) or not t["slot"].startswith("backlog."):
                raise TierTableError("%s: gate slot must start with 'backlog.': %r"
                                     % (where, t["slot"]))
            if t["sign"] not in (1, -1) or isinstance(t["sign"], bool):
                raise TierTableError("%s: term sign must be 1 or -1" % where)
            if t["agg"] not in GATE_AGGS:
                raise TierTableError("%s: term agg must be one of %s" % (where, GATE_AGGS))
            if not isinstance(t["role"], str) or not t["role"]:
                raise TierTableError("%s: term role must be a non-empty string" % where)
        if g["rule"] == "abs_max" and not _is_num(r.get("abs_max")):
            raise TierTableError("%s: reading %s needs numeric abs_max" % (where, r["name"]))


def _parse_inject_list(lst, where, need_role):
    if not isinstance(lst, list):
        raise TierTableError("%s: inject must be a list" % where)
    out = []
    for d in lst:
        try:
            spec = InjectSpec.from_dict(d, require_role=need_role)
        except InjectSpecError as e:
            raise TierTableError("%s: %s" % (where, e))
        out.append(spec)
    return tuple(out)


class TierTable:
    """A validated tier table. Times are ns relative to the launch origin."""

    def __init__(self, doc, sha256=None, path=None):
        self.doc = doc
        self.sha256 = sha256
        self.path = path
        self._validate_top(doc)
        self.experiment = doc["experiment"]
        self.name = doc["name"]
        self.dist = doc["arrival"]["dist"]
        self.seed = doc["arrival"]["seed"]
        self.beat_origin_rel_ns = doc["beat_origin_rel_ns"]
        self.axis = doc["axis"]
        self.request_timeout_ns = doc["request_timeout_ns"]
        self.default_gate = doc["default_gate"]
        self.launch_inject = _parse_inject_list(doc["launch_inject"], "launch_inject", True)
        self.tiers = tuple(self._parse_tier(i, t) for i, t in enumerate(doc["tiers"]))
        self._validate_sequence()
        self._starts = [t.start_ns for t in self.tiers]

    # ---- loading -------------------------------------------------------------------------
    @classmethod
    def load(cls, path):
        try:
            with open(path, "rb") as f:
                raw = f.read()
            doc = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError) as e:
            raise TierTableError("%s: cannot read tier table: %s" % (path, e))
        return cls(doc, sha256=hashlib.sha256(raw).hexdigest(), path=path)

    @classmethod
    def from_doc(cls, doc):
        return cls(doc)

    # ---- validation ----------------------------------------------------------------------
    @staticmethod
    def _validate_top(doc):
        if not isinstance(doc, dict):
            raise TierTableError("tier table must be a JSON object")
        missing = [k for k in _TOP_FIELDS if k not in doc]
        extra = [k for k in doc if k not in _TOP_FIELDS]
        if missing or extra:
            raise TierTableError("top-level fields missing=%s extra=%s" % (missing, extra))
        if doc["schema"] != SCHEMA:
            raise TierTableError("schema must be %s" % SCHEMA)
        for k in ("experiment", "name"):
            if not isinstance(doc[k], str) or not doc[k]:
                raise TierTableError("%s must be a non-empty string" % k)
        arr = doc["arrival"]
        if not isinstance(arr, dict) or set(arr) != {"dist", "seed"}:
            raise TierTableError("arrival must be exactly {dist, seed}")
        if arr["dist"] not in ("fixed", "poisson"):
            raise TierTableError("arrival.dist must be fixed or poisson")
        if not _is_int(arr["seed"]):
            raise TierTableError("arrival.seed must be an int")
        ax = doc["axis"]
        if not isinstance(ax, dict) or set(ax) != {"name", "load_key", "unit", "order"}:
            raise TierTableError("axis must be exactly {name, load_key, unit, order}")
        if ax["name"] not in AXIS_NAMES:
            raise TierTableError("axis.name must be one of %s" % (AXIS_NAMES,))
        if ax["load_key"] not in AXIS_LOAD_KEYS:
            raise TierTableError("axis.load_key must be one of %s" % (AXIS_LOAD_KEYS,))
        if not isinstance(ax["unit"], str) or not ax["unit"]:
            raise TierTableError("axis.unit must be a non-empty string")
        if ax["order"] not in AXIS_ORDERS:
            raise TierTableError("axis.order must be one of %s" % (AXIS_ORDERS,))
        rt = doc["request_timeout_ns"]
        if rt is not None and (not _is_int(rt) or rt <= 0):
            raise TierTableError("request_timeout_ns must be null or an int > 0")
        if not _is_int(doc["beat_origin_rel_ns"]):
            raise TierTableError("beat_origin_rel_ns must be an int")
        if doc["default_gate"] is not None:
            _validate_gate(doc["default_gate"], "default_gate")
        if not isinstance(doc["tiers"], list) or not doc["tiers"]:
            raise TierTableError("tiers must be a non-empty list")

    def _parse_tier(self, i, t):
        where = "tiers[%d]" % i
        if not isinstance(t, dict):
            raise TierTableError("%s must be an object" % where)
        missing = [k for k in _TIER_FIELDS if k not in t]
        extra = [k for k in t if k not in _TIER_FIELDS]
        if missing or extra:
            raise TierTableError("%s fields missing=%s extra=%s" % (where, missing, extra))
        for k in ("seq", "start_ns", "ramp_ns", "window_ns", "tail_ns", "max_tokens",
                  "tokens_per_msg", "stream_interval", "beat_ns"):
            if not _is_int(t[k]):
                raise TierTableError("%s.%s must be an int" % (where, k))
        for k in ("ramp_ns", "window_ns", "tail_ns", "max_tokens", "beat_ns"):
            if t[k] < 0:
                raise TierTableError("%s.%s must be >= 0" % (where, k))
        if not _is_num(t["rate_rps"]) or t["rate_rps"] < 0:
            raise TierTableError("%s.rate_rps must be a finite number >= 0" % where)
        if t["streams_c"] is not None and (not _is_int(t["streams_c"]) or t["streams_c"] <= 0):
            raise TierTableError("%s.streams_c must be null or int > 0" % where)
        if not isinstance(t["label"], str) or not TIER_LABEL_RE.match(t["label"]):
            raise TierTableError("%s.label invalid: %r" % (where, t["label"]))
        if t["label"] in (PRE_LABEL, POST_LABEL):
            raise TierTableError("%s.label %r is reserved" % (where, t["label"]))
        if t["kind"] not in KINDS:
            raise TierTableError("%s.kind must be one of %s" % (where, KINDS))
        if not isinstance(t["baseline"], bool):
            raise TierTableError("%s.baseline must be a bool" % where)
        if not isinstance(t["note"], str):
            raise TierTableError("%s.note must be a string" % where)
        if t["prompt_set"] is not None and (not isinstance(t["prompt_set"], str)
                                            or not t["prompt_set"]):
            raise TierTableError("%s.prompt_set must be null or a non-empty string" % where)
        if t["tokens_per_msg"] < 1 or t["stream_interval"] < 1:
            raise TierTableError("%s: tokens_per_msg and stream_interval must be >= 1" % where)
        if t["tokens_per_msg"] % t["stream_interval"] != 0:
            raise TierTableError("%s: tokens_per_msg %% stream_interval must be 0" % where)
        inject = _parse_inject_list(t["inject"], where + ".inject", True)
        if t["gate"] is not None:
            _validate_gate(t["gate"], where + ".gate")
        tier = Tier(seq=t["seq"], label=t["label"], kind=t["kind"], start_ns=t["start_ns"],
                    ramp_ns=t["ramp_ns"], window_ns=t["window_ns"], tail_ns=t["tail_ns"],
                    rate_rps=float(t["rate_rps"]), streams_c=t["streams_c"],
                    max_tokens=t["max_tokens"], tokens_per_msg=t["tokens_per_msg"],
                    stream_interval=t["stream_interval"], beat_ns=t["beat_ns"],
                    prompt_set=t["prompt_set"], inject=inject, gate=t["gate"],
                    baseline=t["baseline"], note=t["note"], load_value=t["load_value"])
        self._check_load_value(tier, where)
        if tier.is_load:
            if not tier.rate_rps > 0:
                raise TierTableError("%s: load tier needs rate_rps > 0" % where)
            if tier.max_tokens < 1 or tier.window_ns <= 0 or tier.prompt_set is None:
                raise TierTableError("%s: load tier needs max_tokens >= 1, window_ns > 0, "
                                     "prompt_set" % where)
            if tier.gate is not None:
                raise TierTableError("%s: load tier must have gate null" % where)
        else:
            if tier.rate_rps != 0 or tier.max_tokens != 0:
                raise TierTableError("%s: drain tier needs rate_rps == 0 and max_tokens == 0"
                                     % where)
            if tier.ramp_ns != 0 or tier.tail_ns != 0 or tier.window_ns <= 0:
                raise TierTableError("%s: drain tier needs ramp_ns == tail_ns == 0, window_ns > 0"
                                     % where)
            if tier.gate is None and self.default_gate is None:
                raise TierTableError("%s: drain tier needs gate (or a default_gate)" % where)
            if tier.baseline:
                raise TierTableError("%s: drain tier cannot be the baseline" % where)
        if tier.baseline and tier.kind not in ("up", "down"):
            raise TierTableError("%s: only an up/down tier can be the baseline" % where)
        if tier.beat_ns > 0 and tier.duration_ns % tier.beat_ns != 0:
            raise TierTableError("%s: duration %d is not a multiple of beat_ns %d"
                                 % (where, tier.duration_ns, tier.beat_ns))
        if tier.streams_c is not None:
            if tier.beat_ns <= 0:
                raise TierTableError("%s: streams_c requires beat_ns > 0" % where)
            dur = fractions.Fraction(tier.stream_duration_ns(), 10 ** 9)
            want = fractions.Fraction(tier.streams_c) / dur
            got = fractions.Fraction(tier.rate_rps)
            if abs(got - want) > want * fractions.Fraction(1, 10 ** 9):
                raise TierTableError("%s: rate_rps %r != streams_c / stream duration = %r"
                                     % (where, tier.rate_rps, float(want)))
        return tier

    def nominal_load(self, tier):
        """The raw-parameter value that tier.load_value must equal for this table's axis.

        Returns None for load_key prompt_len (prompt length is not a tier-table parameter; the
        generator's declared value is accepted as is) and raises when the key cannot be computed.
        """
        key = self.axis["load_key"]
        if key == "rate_rps":
            return fractions.Fraction(tier.rate_rps)
        if key == "streams_c":
            return None if tier.streams_c is None else fractions.Fraction(tier.streams_c)
        if key == "beat_ns":
            return fractions.Fraction(tier.beat_ns)
        if key == "tokens_per_msg":
            return fractions.Fraction(tier.tokens_per_msg)
        if key == "nominal_msgs_per_s":
            if tier.streams_c is None or tier.beat_ns <= 0:
                return None
            return fractions.Fraction(tier.streams_c * 10 ** 9,
                                      tier.beat_ns * tier.stream_interval)
        return None  # prompt_len

    def _check_load_value(self, tier, where):
        lv = tier.load_value
        if tier.kind == "drain":
            if lv is not None:
                raise TierTableError("%s: drain tier needs load_value null" % where)
            return
        if lv is None:
            if tier.kind == "warmup":
                return
            raise TierTableError("%s: up/down tier needs a load_value" % where)
        if not _is_num(lv) or lv <= 0:
            raise TierTableError("%s.load_value must be a finite number > 0" % where)
        key = self.axis["load_key"]
        want = self.nominal_load(tier)
        if want is None:
            if key != "prompt_len":
                raise TierTableError("%s: load_key %s cannot be computed for this tier "
                                     "(missing streams_c or beat_ns)" % (where, key))
            if not _is_int(lv):
                raise TierTableError("%s: prompt_len load_value must be an int" % where)
            return
        got = fractions.Fraction(lv)
        if abs(got - want) > want * fractions.Fraction(1, 10 ** 9):
            raise TierTableError("%s: load_value %r != %s from the tier's parameters (%r)"
                                 % (where, lv, key, float(want)))

    def load_order_key(self, seq):
        """Sort key that increases with load (uses axis.order); None for drain tiers."""
        lv = self.tiers[seq].load_value
        if lv is None:
            return None
        return lv if self.axis["order"] == "asc" else -lv

    def _validate_sequence(self):
        ts = self.tiers
        labels = set()
        for i, t in enumerate(ts):
            if t.seq != i:
                raise TierTableError("tiers[%d].seq must be %d" % (i, i))
            if t.label in labels:
                raise TierTableError("duplicate tier label %r" % t.label)
            labels.add(t.label)
        if ts[0].start_ns != 0 or self.beat_origin_rel_ns != 0:
            raise TierTableError("tiers[0].start_ns and beat_origin_rel_ns must both be 0")
        if ts[0].kind != "warmup":
            raise TierTableError("the first tier must be warmup")
        if ts[-1].kind != "drain":
            raise TierTableError("the last tier must be drain")
        for prev, cur in zip(ts, ts[1:]):
            if cur.start_ns != prev.end_ns:
                raise TierTableError("tier %d starts at %d, previous ends at %d (must abut)"
                                     % (cur.seq, cur.start_ns, prev.end_ns))
            if cur.kind in ("up", "down") and prev.kind != "drain":
                raise TierTableError("tier %d (%s) must be preceded by a drain tier"
                                     % (cur.seq, cur.kind))
            if cur.kind == "drain" and cur.beat_ns != prev.beat_ns:
                raise TierTableError("drain tier %d beat_ns must equal previous tier's" % cur.seq)
        if sum(1 for t in ts if t.baseline) > 1:
            raise TierTableError("at most one tier may be the baseline")
        launch_sites = {(s.role, s.site) for s in self.launch_inject}
        if len(launch_sites) != len(self.launch_inject):
            raise TierTableError("launch_inject has duplicate (role, site)")
        for t in ts:
            sites = [(s.role, s.site) for s in t.inject]
            if len(set(sites)) != len(sites):
                raise TierTableError("tier %d inject has duplicate (role, site)" % t.seq)
            clash = launch_sites.intersection(sites)
            if clash:
                raise TierTableError("tier %d inject clashes with launch_inject on %s"
                                     % (t.seq, sorted(clash)))

    # ---- queries -------------------------------------------------------------------------
    @property
    def end_ns(self):
        return self.tiers[-1].end_ns

    def __len__(self):
        return len(self.tiers)

    def __iter__(self):
        return iter(self.tiers)

    def __getitem__(self, seq):
        return self.tiers[seq]

    def abs_ns(self, origin_ns, rel_ns):
        return origin_ns + rel_ns

    def tier_at(self, rel_ns):
        """The tier containing rel_ns, or None before the origin / after the last tier."""
        if rel_ns < 0 or rel_ns >= self.end_ns:
            return None
        return self.tiers[bisect.bisect_right(self._starts, rel_ns) - 1]

    def tier_seq_at(self, rel_ns):
        """tier seq for an event time; PRE_SEQ (-1) before the origin, POST_SEQ (-2) after."""
        if rel_ns < 0:
            return PRE_SEQ
        if rel_ns >= self.end_ns:
            return POST_SEQ
        return bisect.bisect_right(self._starts, rel_ns) - 1

    def label_of(self, seq):
        if seq == PRE_SEQ:
            return PRE_LABEL
        if seq == POST_SEQ:
            return POST_LABEL
        return self.tiers[seq].label

    def window_of(self, seq):
        t = self.tiers[seq]
        return (t.window_start_ns, t.window_end_ns)

    def gate_of(self, seq):
        """Effective gate config of a drain tier: (gate_dict, source) with source tier|default."""
        t = self.tiers[seq]
        if t.kind != "drain":
            raise TierTableError("tier %d is not a drain tier" % seq)
        if t.gate is not None:
            return t.gate, "tier"
        return self.default_gate, "default"

    def baseline_seq(self):
        for t in self.tiers:
            if t.baseline:
                return t.seq
        return None

    def load_tiers(self):
        return [t for t in self.tiers if t.is_load]

    def inject_for(self, role, seq):
        """InjectSpecs active for `role` during tier `seq` (launch-level + tier-level)."""
        out = [s for s in self.launch_inject if s.role == role]
        if seq is not None and 0 <= seq < len(self.tiers):
            out += [s for s in self.tiers[seq].inject if s.role == role]
        return out


    def all_inject_for(self, role):
        """Every InjectSpec `role` may activate during the launch (for InjectorSet.prestart_aux)."""
        out = [s for s in self.launch_inject if s.role == role]
        for t in self.tiers:
            out += [s for s in t.inject if s.role == role]
        return out


def file_sha256(path):
    return sha256_file(path)
