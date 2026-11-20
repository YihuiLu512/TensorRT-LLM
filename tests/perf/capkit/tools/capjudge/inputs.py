# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input verification of one launch tree (INTERFACES 10.1; DATA-AND-ANALYSIS 4.5, 4.6).

Any failure raises InputInvalid naming the item; the judge then writes inputs.json into its
attempt directory and exits 13. Nothing here relaxes a rule for synthetic or local runs.
"""

import json
import math
import os
import re

from capcommon import expect, jobinputs, ledger
from capcommon.capbin import CapbinError, read_capbin
from capcommon.exitcodes import ConfigError
from capcommon.hashing import is_sha256_hex, sha256_bytes, sha256_file
from capcommon.identity import LAUNCH_KEY_FIELDS
from capcommon.jsonio import read_json, read_json_stamped
from capcommon.runtree import STATE_VALUES, RunTree, read_state, worst_state
from capcommon.snapshot import SnapshotError, read_snapshot_file
from capcommon.tiers import TierTable, TierTableError

from . import calib, nullz
from .thresholds import BandError, Bands
from .treedata import FRAME_CAPTURE_FILES

LAUNCH_SCHEMA = "capkit.launch/1"
REGISTER_SCHEMA = "capkit.register/1"
REGISTER_FILE_KEYS = ("tier_table", "schedule", "prompts", "scenario")
# files_by_arm[<arm>] of a v2 register (one with arm_config) may also name the arm's ladder.json
LADDER_FILE_KEY = "ladder"
LADDER_SCHEMA = "capkit.ladder/1"
# ladder.json role lists -> the tier regime (calib-v2 DESIGN-r7a 2.11.1, P2-03). holds and
# slope tiers are judged and their X statistics computed, but their X conclusion is not used
# (DESIGN-r7a 2.11.1 r7a note): they read near-boundary thresholds like the levels.
LADDER_ROLE_REGIME = {"levels": "near_boundary", "near": "near_boundary", "cal": "subsat",
                      "baseline": "subsat", "holds": "near_boundary", "slope": "near_boundary"}
LADDER_SEQ_LISTS = ("near", "cal", "holds", "slope")
# near_boundary_noise is listed so that its rejection names the v1 rule (nullz.V1_REJECTED).
CALIB_KINDS = ("pilot", "loadgen_band", "stub_calib", "near_boundary_noise", "d14_result")
# kinds read by capjudge.nullz (calibration v2 threshold sources); the shape kinds keep their
# v1 reading (calib.validate, Bands)
NULLZ_KINDS = nullz.KINDS
INLINE_V2_REJECTED = "inline thresholds are not a calibration file (CONSTRAINTS 4.2; calib-v2 r7)"
MEASURE_CLASSES = ("load_source", "stub_out_of_domain", "stub_in_domain", "under_test",
                   "observer")
ROLE_KEYS = ("role", "workers", "expected_procs", "measure_class", "asyncio", "uses_grpc",
             "loop", "probe_tier")
# Checks the judge runs on every launch whatever the register says (C10 = D11, C11 drain gate,
# C3 non-overload failures, GPU-IDLE = CONSTRAINTS 2.4). The register can add checks, not remove
# these.
MANDATORY_CHECKS = ("C3", "C10", "C11", "GPU-IDLE")
KNOWN_CHECKS = ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12",
                "C13", "GPU-IDLE")
# "P+X": preconditions and external symptom only. "P+X+I+V": also internal localisation I,
# the per-tier intervention heartbeat and the launch grade (INTERFACES errata B2-2E).
JUDGE_MODES = ("P+X", "P+X+I+V")
READING_NAME = re.compile(r"^[a-z0-9_.-]{1,48}$")
# Calibration kinds whose register entry binds this experiment's roles to the file's shapes.
BIND_KINDS = tuple(calib.SHAPE_KINDS)
CALIB_ENTRY_KEYS = ("kind", "relpath", "sha256")


class InputInvalid(Exception):
    def __init__(self, item, detail):
        Exception.__init__(self, "%s: %s" % (item, detail))
        self.item = item
        self.detail = detail


class Ctx:
    """Everything the checks need, loaded once and verified."""

    def __init__(self, workdir, tree_root):
        self.W = os.path.abspath(workdir)
        self.root = os.path.abspath(tree_root)
        self.tree = RunTree(self.root)
        self.files_read = {}  # relpath (to tree root, or W:<relpath>) -> sha256
        self.launch = None
        self.ident = None
        self.register = None
        self.tt = None
        self.schedule_meta = None
        self.schedule_tier = None  # array('h'): planned tier of every req_id
        self.roles = {}  # role -> launch.json roles[] entry
        self.bands = Bands()  # loadgen_band / stub_calib sources and a v1 register's inline
        self.thresholds = nullz.Thresholds()  # pilot/2 and d14_result (calibration v2)
        self.calib_docs = {}  # kind -> validated calibration document
        self.calib_bind = {}  # kind in BIND_KINDS -> {role: shape}
        self.tier_regime = {}  # tier seq -> "subsat" / "near_boundary" (load tiers)
        # register.json arm_config / arm_threshold_config of this arm (None: a v1 register)
        self.config = None
        self.threshold_config = None
        self.ladder_path = None
        self.ladder = None  # ladder.json of this arm (v2 register with files_by_arm ladder)
        self.tier_role = {}  # load tier seq -> its ladder.json role (empty without a ladder)
        self.mode = None  # register.json judge.mode
        self.checks_applied = []
        self.submission = None
        self.identity_scan = None
        self.upstream = None  # launch-level conclusion of the writers before the judge

    def note_file(self, path, sha=None):
        rel = os.path.relpath(path, self.root)
        if rel.startswith(".."):
            rel = "W:" + os.path.relpath(path, self.W)
        self.files_read[rel] = sha if sha is not None else sha256_file(path)


def _wpath(ctx, relpath, item):
    if not isinstance(relpath, str) or not relpath or os.path.isabs(relpath) \
            or relpath.split("/")[0] == "..":
        raise InputInvalid(item, "relpath must be relative to the work dir: %r" % (relpath,))
    p = os.path.join(ctx.W, relpath)
    if not os.path.isfile(p):
        raise InputInvalid(item, "file not found: %s" % relpath)
    return p


def _registered_file(ctx, ref, item, kinds=None):
    """ref = {"relpath", "sha256"}: recompute sha, require a REGISTRY row registered strictly
    before the job's submission. Returns the absolute path."""
    if not isinstance(ref, dict) or not is_sha256_hex(ref.get("sha256")):
        raise InputInvalid(item, "needs {relpath, sha256}")
    p = _wpath(ctx, ref.get("relpath"), item)
    got = sha256_file(p)
    if got != ref["sha256"]:
        raise InputInvalid(item, "sha256 %s != listed %s" % (got, ref["sha256"]))
    row = ledger.registry_lookup(os.path.join(ctx.W, "REGISTRY.tsv"), got)
    if row is None:
        raise InputInvalid(item, "sha %s not in REGISTRY.tsv" % got)
    if kinds is not None and ledger.registry_kind(row) not in kinds:
        raise InputInvalid(item, "registered as kind %r, expected %s"
                           % (ledger.registry_kind(row), kinds))
    try:
        before = ledger.registered_before(row, ctx.submission["submitted_at"])
    except ValueError as e:
        raise InputInvalid(item, "timestamp: %s" % e)
    if not before:
        raise InputInvalid(item, "registered_at %s is not before submitted_at %s"
                           % (row["registered_at"], ctx.submission["submitted_at"]))
    ctx.note_file(p, got)
    return p


def _load_launch(ctx):
    path = ctx.tree.launch_json
    try:
        ident, doc = read_json_stamped(path, LAUNCH_SCHEMA)
    except (ConfigError, ValueError, OSError) as e:
        raise InputInvalid("launch.json", str(e))
    if ident.role != "launch":
        raise InputInvalid("launch.json", "identity.role must be 'launch'")
    for k in ("origin_ns", "register", "tier_table", "schedule", "prompts", "scenario",
              "roles", "reconcile", "hw_profile_sha", "image", "probe_tier"):
        if k not in doc:
            raise InputInvalid("launch.json", "missing key %s" % k)
    if doc["register"].get("sha256") != ident.register_sha:
        raise InputInvalid("launch.json", "register.sha256 != identity.register_sha")
    ctx.launch, ctx.ident = doc, ident
    ctx.note_file(path)


def _load_submission(ctx):
    path = os.path.join(ctx.W, "submissions.tsv")
    try:
        row = ledger.submission_for_job(path, ctx.ident.job_id)
    except ConfigError as e:
        raise InputInvalid("submissions.tsv", str(e))
    if row is None or not row.get("submitted_at"):
        raise InputInvalid("submissions.tsv", "no submitted_at for job %s" % ctx.ident.job_id)
    ctx.submission = row


def _load_roles(ctx):
    roles = ctx.launch["roles"]
    if not isinstance(roles, list) or not roles:
        raise InputInvalid("launch.json roles", "must be a non-empty list")
    for r in roles:
        miss = [k for k in ROLE_KEYS if k not in r] if isinstance(r, dict) else ROLE_KEYS
        if miss:
            raise InputInvalid("launch.json roles", "entry lacks %s" % (list(miss),))
        if r["role"] in ctx.roles:
            raise InputInvalid("launch.json roles", "duplicate role %s" % r["role"])
        if r["measure_class"] not in MEASURE_CLASSES:
            raise InputInvalid("launch.json roles", "bad measure_class %r" % r["measure_class"])
        for k in ("workers", "expected_procs"):
            if not isinstance(r[k], int) or isinstance(r[k], bool) or r[k] < 1:
                raise InputInvalid("launch.json roles", "%s.%s must be int >= 1" % (r["role"], k))
        ctx.roles[r["role"]] = r


def _load_register(ctx):
    ref = ctx.launch["register"]
    p = _registered_file(ctx, ref, "register.json", kinds=("register",))
    try:
        doc = read_json(p)
    except (OSError, ValueError) as e:
        raise InputInvalid("register.json", str(e))
    if not isinstance(doc, dict) or doc.get("schema") != REGISTER_SCHEMA:
        raise InputInvalid("register.json", "schema must be %s" % REGISTER_SCHEMA)
    if doc.get("experiment") != ctx.ident.experiment:
        raise InputInvalid("register.json", "experiment %r != launch %r"
                           % (doc.get("experiment"), ctx.ident.experiment))
    if ctx.ident.arm not in (doc.get("arms") or []):
        raise InputInvalid("register.json", "arm %r not in arms" % ctx.ident.arm)
    has_files, has_by_arm = "files" in doc, "files_by_arm" in doc
    if has_files == has_by_arm:
        raise InputInvalid("register.json", "exactly one of files / files_by_arm")
    _load_arm_config(ctx, doc)
    files = doc["files"] if has_files else (doc["files_by_arm"] or {}).get(ctx.ident.arm)
    v2_ladder = ctx.config is not None and has_by_arm and isinstance(files, dict) \
        and LADDER_FILE_KEY in files
    want = set(REGISTER_FILE_KEYS) | ({LADDER_FILE_KEY} if v2_ladder else set())
    if not isinstance(files, dict) or set(files) != want:
        raise InputInvalid("register.json", "files must have exactly %s%s" % (
            REGISTER_FILE_KEYS, " (+ ladder in files_by_arm of a register with arm_config)"
            if ctx.config is not None else ""))
    if v2_ladder:
        ctx.ladder_path = _registered_file(ctx, files[LADDER_FILE_KEY],
                                           "register.files_by_arm.%s.ladder" % ctx.ident.arm,
                                           kinds=(LADDER_FILE_KEY,))
    for k in REGISTER_FILE_KEYS:
        _registered_file(ctx, files[k], "register.files.%s" % k, kinds=(k,))
        lk = ctx.launch[k]
        if not isinstance(lk, dict) or lk.get("sha256") != files[k]["sha256"] \
                or lk.get("relpath") != files[k]["relpath"]:
            raise InputInvalid("launch.json %s" % k, "does not match register.json files.%s" % k)
    if "register_md" in doc:
        _registered_file(ctx, doc["register_md"], "register.register_md", kinds=("register_md",))
    judge = doc.get("judge")
    if not isinstance(judge, dict) or judge.get("mode") not in JUDGE_MODES \
            or not isinstance(judge.get("checks"), list):
        raise InputInvalid("register.json", "judge must be {mode: one of %s, checks: [...]}"
                           % (JUDGE_MODES,))
    unknown = [c for c in judge["checks"] if c not in KNOWN_CHECKS]
    if unknown:
        raise InputInvalid("register.json", "unknown checks %s" % unknown)
    if ctx.config is not None and "near_boundary_tiers" in judge:
        raise InputInvalid("register.json", "judge.near_boundary_tiers is not read for a "
                                            "register with arm_config: the regime of a tier "
                                            "comes from its arm's ladder.json roles")
    nbt = judge.get("near_boundary_tiers", {})
    if not isinstance(nbt, dict) or any(
            a not in (doc.get("arms") or []) or not isinstance(v, list)
            or not all(isinstance(x, str) and x for x in v) or len(set(v)) != len(v)
            for a, v in nbt.items()):
        raise InputInvalid("register.json", "judge.near_boundary_tiers must map arms of "
                                            "`arms` to lists of distinct tier labels")
    ctx.checks_applied = sorted(set(MANDATORY_CHECKS) | set(judge["checks"]),
                                key=lambda c: (len(c), c))
    if "localize" in judge and judge["mode"] != "P+X+I+V":
        raise InputInvalid("register.json", "judge.localize needs mode 'P+X+I+V'")
    ctx.mode = judge["mode"]
    ctx.register = doc
    ctx.register_files = files


def _arm_map(doc, key, arms):
    m = doc.get(key)
    if not isinstance(m, dict) or set(m) != set(arms) or not all(
            isinstance(v, str) and v and "|" not in v and "*" not in v for v in m.values()):
        raise InputInvalid("register.json", "%s must map exactly the arms of `arms` %s to "
                                            "configuration names (non-empty, no '|' or '*')"
                           % (key, sorted(arms)))
    return m


def _load_arm_config(ctx, doc):
    """register.json arm_config / arm_threshold_config (calib-v2 DESIGN-r7a P2-03). Without
    arm_config the register is read with v1 semantics (no configuration, no ladder). With it,
    ctx.config = arm_config[arm] (ladder check, verdicts.json, batching) and
    ctx.threshold_config = arm_threshold_config[arm] (default arm_config[arm]): the only
    configuration thresholds are read for. Arm names are never parsed."""
    arms = doc.get("arms") or []
    ctx.config = ctx.threshold_config = ctx.ladder_path = None
    if "arm_config" not in doc:
        if "arm_threshold_config" in doc:
            raise InputInvalid("register.json", "arm_threshold_config needs arm_config")
        return
    ac = _arm_map(doc, "arm_config", arms)
    atc = _arm_map(doc, "arm_threshold_config", arms) if "arm_threshold_config" in doc else ac
    ctx.config = ac[ctx.ident.arm]
    ctx.threshold_config = atc[ctx.ident.arm]


def _load_calibration(ctx):
    cal = ctx.register.get("calibration", [])
    if not isinstance(cal, list):
        raise InputInvalid("register.calibration", "must be a list")
    for i, c in enumerate(cal):
        item = "register.calibration[%d]" % i
        if not isinstance(c, dict) or c.get("kind") not in CALIB_KINDS:
            raise InputInvalid(item, "kind must be one of %s" % (CALIB_KINDS,))
        if c["kind"] == "near_boundary_noise":
            raise InputInvalid(item, nullz.V1_REJECTED)
        allowed = CALIB_ENTRY_KEYS + (("bind",) if c["kind"] in BIND_KINDS else ())
        extra = sorted(set(c) - set(allowed))
        if extra:
            raise InputInvalid(item, "unknown keys %s for kind %s" % (extra, c["kind"]))
        p = _registered_file(ctx, c, item, kinds=(c["kind"],))
        try:
            doc = read_json(p)
        except (OSError, ValueError) as e:
            raise InputInvalid(item, str(e))
        if c["kind"] in NULLZ_KINDS:
            # calibration v2, by schema (nullz.validate): capkit.pilot/1 -> nullz.V1_REJECTED
            try:
                ctx.thresholds.add_source(c["kind"], c["relpath"], c["sha256"], doc)
            except nullz.ThresholdError as e:
                raise InputInvalid(item, str(e))
            _check_env(ctx, doc, item)
            ctx.calib_docs[c["kind"]] = doc
            continue
        try:
            calib.validate(c["kind"], doc, item)
        except calib.CalibError as e:
            raise InputInvalid(item, str(e))
        _check_env(ctx, doc, item)
        if c["kind"] in BIND_KINDS:
            _load_bind(ctx, c, doc, item)
        try:
            ctx.bands.add_source(c["kind"], c["relpath"], c["sha256"], doc)
        except BandError as e:
            raise InputInvalid(item, str(e))
        ctx.calib_docs[c["kind"]] = doc
    inline = ctx.register.get("inline_thresholds")
    if inline and ctx.config is not None:
        raise InputInvalid("register.inline_thresholds", INLINE_V2_REJECTED)
    if inline:
        if ctx.ident.experiment != "E0":
            raise InputInvalid("register.inline_thresholds", "only E0 may use inline thresholds")
        try:
            ctx.bands.add_source("inline", ctx.launch["register"]["relpath"]
                                 + "#inline_thresholds", ctx.ident.register_sha, inline)
        except BandError as e:
            raise InputInvalid("register.inline_thresholds", str(e))


def _load_bind(ctx, entry, doc, item):
    """register.calibration[] `bind` of a loadgen_band / stub_calib entry: {role: shape}. Every
    named shape must exist in the file; roles of other arms may be named. A role of this launch
    left unbound has no calibration (its C1 / C2 items are NOT_RUN)."""
    bind = entry.get("bind")
    shapes = calib.shapes_of(entry["kind"], doc)
    if not isinstance(bind, dict) or not bind or not all(
            isinstance(r, str) and r and isinstance(sh, str) for r, sh in bind.items()):
        raise InputInvalid(item, "bind must be a non-empty {role: shape} object")
    missing = sorted(sh for sh in bind.values() if sh not in shapes)
    if missing:
        raise InputInvalid(item, "bind names shapes %s not in the file (has %s)"
                           % (missing, sorted(shapes)))
    ctx.calib_bind[entry["kind"]] = dict(bind)


def _check_env(ctx, doc, item):
    env = doc.get("environment") or {}
    want_hw = ctx.launch.get("hw_profile_sha")
    want_img = (ctx.launch.get("image") or {}).get("sha256")
    if env.get("hw_profile_sha") != want_hw or env.get("image_sha") != want_img:
        # Registered comparability (DATA-AND-ANALYSIS 8.1) is not read: a calibration file from
        # a different hardware profile or image is rejected outright.
        raise InputInvalid(item, "calibration environment (hw %r, image %r) != launch (hw %r, "
                                 "image %r)" % (env.get("hw_profile_sha"), env.get("image_sha"),
                                                want_hw, want_img))


def _load_localize(ctx):
    """register.json judge.localize (mode P+X+I+V, optional): {"readings": [{"name", "terms":
    [{"role", "slot", "sign", "agg"}], "receiver", "sender"}]}. Terms as in a drain-gate
    reading (INTERFACES 8.1: backlog.* slots of launch roles); receiver / sender are launch
    roles or null, not both null and not equal."""
    loc = ctx.register["judge"].get("localize")
    if loc is None:
        return
    item = "register.json judge.localize"
    if not isinstance(loc, dict) or set(loc) != {"readings"} \
            or not isinstance(loc["readings"], list):
        raise InputInvalid(item, "must be {readings: [...]}")
    names = set()
    for i, r in enumerate(loc["readings"]):
        where = "%s.readings[%d]" % (item, i)
        if not isinstance(r, dict) or set(r) != {"name", "terms", "receiver", "sender"}:
            raise InputInvalid(where, "keys must be name, terms, receiver, sender")
        if not isinstance(r["name"], str) or not READING_NAME.match(r["name"]) \
                or r["name"] in names:
            raise InputInvalid(where, "name must match %s and be unique" % READING_NAME.pattern)
        names.add(r["name"])
        ends = [r["receiver"], r["sender"]]
        if all(e is None for e in ends) or r["receiver"] == r["sender"] or any(
                e is not None and e not in ctx.roles for e in ends):
            raise InputInvalid(where, "receiver / sender must be distinct launch roles or null"
                                      " (not both null)")
        if not isinstance(r["terms"], list) or not r["terms"]:
            raise InputInvalid(where, "terms must be a non-empty list")
        for t in r["terms"]:
            if not isinstance(t, dict) or set(t) != {"role", "slot", "sign", "agg"} \
                    or t["role"] not in ctx.roles or not isinstance(t["slot"], str) \
                    or not t["slot"].startswith("backlog.") or t["sign"] not in (1, -1) \
                    or isinstance(t["sign"], bool) or t["agg"] not in ("sum", "max"):
                raise InputInvalid(where, "term %r: {role: launch role, slot: backlog.*, "
                                          "sign: 1 | -1, agg: sum | max}" % (t,))


def _load_tier_regimes(ctx):
    """ctx.tier_regime for every load tier.

    v1 register (no arm_config): near_boundary for the labels register.json
    judge.near_boundary_tiers lists for this arm, subsat otherwise. A listed label must be an
    up / down tier of this launch's tier table.
    v2 register: from the roles of the arm's ladder.json (LADDER_ROLE_REGIME; ctx.tier_role
    records each load tier's role); an arm without a ladder has every load tier subsat."""
    if ctx.config is not None:
        load = {t.seq: "subsat" for t in ctx.tt.tiers if t.kind in ("up", "down")}
        if ctx.ladder_path is not None:
            ctx.ladder, ctx.tier_role = _load_ladder(ctx)
            load = {seq: LADDER_ROLE_REGIME[role] for seq, role in ctx.tier_role.items()}
        ctx.tier_regime = load
        return
    labels = (ctx.register["judge"].get("near_boundary_tiers") or {}).get(ctx.ident.arm, [])
    by_label = {t.label: t for t in ctx.tt.tiers}
    bad = [x for x in labels if x not in by_label or by_label[x].kind not in ("up", "down")]
    if bad:
        raise InputInvalid("register.json", "judge.near_boundary_tiers[%s]: %s are not load "
                                            "tiers of the tier table" % (ctx.ident.arm, bad))
    ctx.tier_regime = {t.seq: "near_boundary" if t.label in labels else "subsat"
                       for t in ctx.tt.tiers if t.kind in ("up", "down")}


def _check_threshold_window(ctx):
    """A thresholds-stage pilot/2 holds per-key thresholds for one window length (its
    window_ns, the L= of its null_z keys): every load tier whose X is judged (all but ladder
    hold tiers) must have that window (calib-v2 DESIGN-r7a 2.5, 3.5 test 22). A bootstrap
    pilot/2 has one value per family and no window."""
    doc = ctx.calib_docs.get("pilot")
    if not isinstance(doc, dict) or doc.get("stage") != "thresholds":
        return
    bad = {t.seq: t.window_ns for t in ctx.tt.tiers if t.kind in ("up", "down")
           and ctx.tier_role.get(t.seq) != "holds" and t.window_ns != doc["window_ns"]}
    if bad:
        raise InputInvalid("register.calibration", "pilot/2 thresholds are for window_ns %d; "
                           "load tiers %s have other windows" % (doc["window_ns"], bad))


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_seq(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _load_ladder(ctx):
    """Read and check the arm's ladder.json (capkit.ladder/1, calib-v2 DESIGN-r7a 3.4) against
    this launch's tier table. Returns (document, {load tier seq: role}). Every up / down tier
    must be named by exactly one role (levels up_seq / down_seq, near, cal, holds, slope, or
    baseline_seq): a load tier no role names would have no regime. The ladder must have been
    generated for this launch: its tier_table_sha256 is the registered tier table's sha and
    its arm the launch's arm (laddergen --arm, the files_by_arm key)."""
    item = "ladder.json"
    try:
        doc = read_json(ctx.ladder_path)
    except (OSError, ValueError) as e:
        raise InputInvalid(item, str(e))
    if not isinstance(doc, dict) or doc.get("schema") != LADDER_SCHEMA:
        raise InputInvalid(item, "schema must be %s" % LADDER_SCHEMA)
    if doc.get("config") != ctx.config:
        raise InputInvalid(item, "config %r != register.json arm_config[%s] %r"
                           % (doc.get("config"), ctx.ident.arm, ctx.config))
    want = ctx.register_files["tier_table"]["sha256"]
    if doc.get("tier_table_sha256") != want:
        raise InputInvalid(item, "tier_table_sha256 %r != the registered tier table %s"
                           % (doc.get("tier_table_sha256"), want))
    if doc.get("arm") != ctx.ident.arm:
        raise InputInvalid(item, "arm %r != this launch's arm %r" % (doc.get("arm"),
                                                                    ctx.ident.arm))
    kinds = {t.seq: t.kind for t in ctx.tt.tiers}
    named = []  # (seq, role, kind it must have or None for any load kind)
    levels = doc.get("levels")
    if not isinstance(levels, list) or not all(isinstance(lv, dict) for lv in levels):
        raise InputInvalid(item, "levels must be a list of objects")
    loads = {t.seq: t.load_value for t in ctx.tt.tiers}
    for lv in levels:
        if not _is_seq(lv.get("up_seq")) or not (lv.get("down_seq") is None
                                                 or _is_seq(lv.get("down_seq"))):
            raise InputInvalid(item, "levels[] needs an int up_seq and an int or null down_seq")
        # the judge composes levels[] and b from these (calib-v2 DESIGN-r7a 2.4, P2-06)
        if not _is_seq(lv.get("level")) or not _is_num(lv.get("rho")) \
                or not _is_num(lv.get("load_value")):
            raise InputInvalid(item, "levels[] needs an int level and numeric rho, load_value")
        named.append((lv["up_seq"], "levels", "up"))
        if lv["down_seq"] is not None:
            named.append((lv["down_seq"], "levels", "down"))
    if len({lv["level"] for lv in levels}) != len(levels):
        raise InputInvalid(item, "levels[] level numbers must be distinct")
    for role in LADDER_SEQ_LISTS:
        lst = doc.get(role)
        if not isinstance(lst, list) or not all(isinstance(e, dict) and _is_seq(e.get("seq"))
                                                for e in lst):
            raise InputInvalid(item, "%s must be a list of objects with an int seq" % role)
        named += [(e["seq"], role, None) for e in lst]
    base = doc.get("baseline_seq")
    if base is not None:
        if not _is_seq(base):
            raise InputInvalid(item, "baseline_seq must be an int or null")
        named.append((base, "baseline", None))
    roles = {}
    for seq, role, want in named:
        k = kinds.get(seq)
        if k not in ("up", "down") or (want is not None and k != want):
            raise InputInvalid(item, "%s seq %d must be a %s tier of the tier table (kind %r)"
                               % (role, seq, want or "load (up / down)", k))
        if seq in roles:
            raise InputInvalid(item, "seq %d named by both %s and %s" % (seq, roles[seq], role))
        roles[seq] = role
    unnamed = sorted(s for s, k in kinds.items() if k in ("up", "down") and s not in roles)
    if unnamed:
        raise InputInvalid(item, "load tiers %s are named by no ladder role" % unnamed)
    for lv in levels:
        for q in (lv["up_seq"], lv["down_seq"]):
            want = loads.get(q)
            if q is not None and not (_is_num(want) and math.isclose(
                    lv["load_value"], want, rel_tol=1e-9, abs_tol=0.0)):
                raise InputInvalid(item, "level %d load_value %r != tier %d load_value %r"
                                   % (lv["level"], lv["load_value"], q, want))
    return doc, roles


def _load_tier_table_and_schedule(ctx):
    ttp = os.path.join(ctx.W, ctx.register_files["tier_table"]["relpath"])
    try:
        ctx.tt = TierTable.load(ttp)
    except TierTableError as e:
        raise InputInvalid("tier_table", str(e))
    sp = os.path.join(ctx.W, ctx.register_files["schedule"]["relpath"])
    try:
        header, cols = read_capbin(sp, columns=["tier_seq"])
    except (CapbinError, OSError) as e:
        raise InputInvalid("schedule", str(e))
    meta = header.get("meta") or {}
    if header.get("kind") != "schedule" or meta.get("tier_table_sha256") != ctx.tt.sha256:
        raise InputInvalid("schedule", "not a schedule of this tier table")
    per = meta.get("per_tier")
    if not isinstance(per, dict):
        raise InputInvalid("schedule", "meta.per_tier missing")
    ctx.schedule_meta = meta
    ctx.schedule_tier = cols["tier_seq"]
    if len(ctx.schedule_tier) != meta.get("n_requests"):
        raise InputInvalid("schedule", "row count != meta.n_requests")


def _load_job_inputs(ctx):
    """Job inputs (errata B2B-9): launch.json job_inputs lists exactly the kinds the registered
    scenario file declares; each copy in the tree (raw/env/job_inputs/<kind>.json) has the listed
    sha256 and its kind's schema, and its record (<kind>.record.json, record_sha256) says the
    kind's module built it from this launch's registered tier table / schedule / prompts.
    A launch.json without the key (written before job inputs existed) counts as none."""
    sp = os.path.join(ctx.W, ctx.register_files["scenario"]["relpath"])
    try:
        want = jobinputs.kinds_of(read_json(sp))
    except (OSError, ValueError) as e:
        raise InputInvalid("scenario", str(e))
    got = ctx.launch.get("job_inputs", {})
    if not isinstance(got, dict) or sorted(got) != sorted(want):
        raise InputInvalid("launch.json job_inputs", "kinds %s != scenario job_inputs %s"
                           % (sorted(got) if isinstance(got, dict) else got, want))
    reg = {k: ctx.launch[k]["sha256"] for k in jobinputs.REGISTERED_INPUTS}
    for k in want:
        item = "job_inputs.%s" % k
        ent = got[k]
        rel = "raw/env/job_inputs/%s.json" % k
        if not isinstance(ent, dict) or ent.get("relpath") != rel:
            raise InputInvalid(item, "relpath must be %s" % rel)
        p = os.path.join(ctx.root, rel)
        rp = p[:-len(".json")] + ".record.json"
        try:
            with open(p, "rb") as f:
                data = f.read()
            with open(rp, "rb") as f:
                rec_bytes = f.read()
        except OSError as e:
            raise InputInvalid(item, str(e))
        sha, rsha = sha256_bytes(data), sha256_bytes(rec_bytes)
        if sha != ent.get("sha256") or rsha != ent.get("record_sha256"):
            raise InputInvalid(item, "sha256 %s / record %s != launch.json %s / %s"
                               % (sha, rsha, ent.get("sha256"), ent.get("record_sha256")))
        try:
            jobinputs.load_verified(k, data, rec_bytes, reg)
        except ConfigError as e:
            raise InputInvalid(item, str(e))
        ctx.note_file(p, sha)
        ctx.note_file(rp, rsha)


def _check_reconcile_and_scenario(ctx):
    try:
        expect.validate_reconcile(ctx.launch["reconcile"], set(ctx.roles))
    except ConfigError as e:
        raise InputInvalid("launch.json reconcile", str(e))


# Files directly in raw/env/ are copied in verbatim (INTERFACES 5.1): job-level files written
# once per job (e.g. the gate results) carry the job's arm and launch_seq 0, and the pin table
# (generated per launch in the job, errata R4-6) this launch's arm and launch_seq; only the job
# key, common to both, is compared for files directly in raw/env/.
JOB_KEY_FIELDS = ("experiment", "job_id", "restart_count", "register_sha")


def _scan_identities(ctx):
    """Every identity-bearing file under raw/ must carry this launch's six key fields
    (files directly in raw/env/: the four job key fields). The frame files of
    capstubs.frame_capture (treedata.FRAME_CAPTURE_FILES) are not opened; their count is
    recorded as frame_capture_files_skipped when there are any."""
    env_dir = os.path.join(ctx.tree.raw, "env")
    n = 0
    skipped = 0
    bad = []
    for d, dirs, files in os.walk(ctx.tree.raw):
        dirs.sort()
        fields = JOB_KEY_FIELDS if os.path.abspath(d) == os.path.abspath(env_dir) \
            else LAUNCH_KEY_FIELDS
        want = [getattr(ctx.ident, f) for f in fields]
        for name in sorted(files):
            p = os.path.join(d, name)
            if os.path.islink(p):
                bad.append((p, "symlink"))
                continue
            if name in FRAME_CAPTURE_FILES:
                skipped += 1
                continue
            ident = _file_identity(p)
            if ident is _NO_IDENTITY:
                continue
            n += 1
            got = None if not isinstance(ident, dict) else [ident.get(f) for f in fields]
            if got != want:
                bad.append((os.path.relpath(p, ctx.root), got))
    if bad:
        raise InputInvalid("identity", "%d file(s) differ from launch.json, first: %r"
                           % (len(bad), bad[0]))
    ctx.identity_scan = {"files_with_identity": n}
    if skipped:
        ctx.identity_scan["frame_capture_files_skipped"] = skipped


_NO_IDENTITY = object()


def _file_identity(p):
    try:
        if p.endswith(".json"):
            doc = read_json(p)
            return doc["identity"] if isinstance(doc, dict) and "identity" in doc \
                else _NO_IDENTITY
        if p.endswith(".jsonl"):
            with open(p, "r", encoding="utf-8") as f:
                head = json.loads(f.readline())
            return head.get("identity", _NO_IDENTITY) if isinstance(head, dict) \
                else _NO_IDENTITY
        if p.endswith(".capbin"):
            header, _ = read_capbin(p, columns=[])
            return header.get("identity") if header.get("identity") is not None \
                else _NO_IDENTITY
        if p.endswith(".bin"):
            snap = read_snapshot_file(p)
            ident = snap.meta.get("identity")
            return ident if ident is not None else _NO_IDENTITY
    except (OSError, ValueError, CapbinError, SnapshotError) as e:
        raise InputInvalid("identity", "%s unreadable: %s" % (p, e))
    return _NO_IDENTITY


JUDGE_WRITER = "capjudge"


def _load_upstream(ctx):
    """The launch-level conclusion recorded before the judge (INTERFACES 5.7): every STATE line
    not written by capjudge (caplaunch, capjobs.launches, ...) plus raw/launch_end.json `state`
    and `leftovers.after`. ctx.upstream = {status: the worst of them, sources: [...]}. The judge
    can make this worse, never better (it does not see leftovers, budget kills or copy-back
    checks). No STATE line from any earlier writer -> INPUT_INVALID (the tree is not finished:
    judge it once the job has recorded its outcome)."""
    try:
        lines = read_state(ctx.root)
    except (ConfigError, OSError, UnicodeDecodeError) as e:
        raise InputInvalid("STATE", "unreadable: %s" % e)
    srcs = [{"source": "STATE", "by": by, "status": st, "reason": reason, "at": at}
            for at, st, by, reason in lines if by != JUDGE_WRITER]
    if not srcs:
        raise InputInvalid("STATE", "no line from caplaunch or capjobs (%d line(s) in all): the "
                                    "launch outcome is not recorded yet" % len(lines))
    ctx.note_file(ctx.tree.state_file)
    le_path = ctx.tree.launch_end_json
    try:
        le = read_json(le_path) if os.path.isfile(le_path) else None
    except (OSError, ValueError) as e:
        raise InputInvalid("launch_end.json", "unreadable: %s" % e)
    if not isinstance(le, dict):
        srcs.append({"source": "launch_end.json", "status": "NOT_RUN",
                     "reason": "raw/launch_end.json missing or not an object"})
    else:
        ctx.note_file(le_path)
        st = le.get("state")
        srcs.append({"source": "launch_end.json state", "status": st if st in STATE_VALUES
                     else "NOT_RUN", "reason": "state %r: %s" % (st, le.get("reason"))})
        after = (le.get("leftovers") or {}).get("after") if isinstance(le.get("leftovers"),
                                                                        dict) else None
        if after:
            srcs.append({"source": "launch_end.json leftovers.after", "status": "VOID",
                         "reason": "%d process(es) of this launch outlived it" % len(after)})
    ctx.upstream = {"status": worst_state(s["status"] for s in srcs), "sources": srcs}


def verify(workdir, tree_root):
    """Run every input check in the fixed order of INTERFACES 10.1; return the Ctx."""
    ctx = Ctx(workdir, tree_root)
    _load_launch(ctx)
    _load_submission(ctx)
    _load_roles(ctx)
    _load_register(ctx)
    _load_calibration(ctx)
    _load_tier_table_and_schedule(ctx)
    _load_tier_regimes(ctx)
    _check_threshold_window(ctx)
    _load_localize(ctx)
    _load_job_inputs(ctx)
    _check_reconcile_and_scenario(ctx)
    _scan_identities(ctx)
    _load_upstream(ctx)
    return ctx
