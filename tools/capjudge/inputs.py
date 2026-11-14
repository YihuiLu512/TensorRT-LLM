# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input verification of one launch tree (INTERFACES 10.1; DATA-AND-ANALYSIS 4.5, 4.6).

Any failure raises InputInvalid naming the item; the judge then writes inputs.json into its
attempt directory and exits 13. Nothing here relaxes a rule for synthetic or local runs.
"""

import json
import os

from capcommon import expect, ledger
from capcommon.capbin import CapbinError, read_capbin
from capcommon.exitcodes import ConfigError
from capcommon.hashing import is_sha256_hex, sha256_file
from capcommon.identity import LAUNCH_KEY_FIELDS
from capcommon.jsonio import read_json, read_json_stamped
from capcommon.runtree import STATE_VALUES, RunTree, read_state, worst_state
from capcommon.snapshot import SnapshotError, read_snapshot_file
from capcommon.tiers import TierTable, TierTableError

from .thresholds import BandError, Bands

LAUNCH_SCHEMA = "capkit.launch/1"
REGISTER_SCHEMA = "capkit.register/1"
PILOT_SCHEMA = "capkit.pilot/1"
REGISTER_FILE_KEYS = ("tier_table", "schedule", "prompts", "scenario")
CALIB_KINDS = ("pilot", "loadgen_band", "stub_calib", "near_boundary_noise")
MEASURE_CLASSES = ("load_source", "stub_out_of_domain", "stub_in_domain", "under_test",
                   "observer")
ROLE_KEYS = ("role", "workers", "expected_procs", "measure_class", "asyncio", "uses_grpc",
             "loop", "probe_tier")
# Checks the judge runs on every launch whatever the register says (C10 = D11, C11 drain gate,
# C3 non-overload failures, GPU-IDLE = CONSTRAINTS 2.4). The register can add checks, not remove
# these.
MANDATORY_CHECKS = ("C3", "C10", "C11", "GPU-IDLE")
BATCH1_CHECKS = ("C1", "C3", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C13", "GPU-IDLE")
BATCH2_CHECKS = ("C2", "C4", "C12")


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
        self.bands = Bands()
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
    files = doc["files"] if has_files else (doc["files_by_arm"] or {}).get(ctx.ident.arm)
    if not isinstance(files, dict) or set(files) != set(REGISTER_FILE_KEYS):
        raise InputInvalid("register.json", "files must have exactly %s" % (REGISTER_FILE_KEYS,))
    for k in REGISTER_FILE_KEYS:
        _registered_file(ctx, files[k], "register.files.%s" % k, kinds=(k,))
        lk = ctx.launch[k]
        if not isinstance(lk, dict) or lk.get("sha256") != files[k]["sha256"] \
                or lk.get("relpath") != files[k]["relpath"]:
            raise InputInvalid("launch.json %s" % k, "does not match register.json files.%s" % k)
    if "register_md" in doc:
        _registered_file(ctx, doc["register_md"], "register.register_md", kinds=("register_md",))
    judge = doc.get("judge")
    if not isinstance(judge, dict) or judge.get("mode") != "P+X" \
            or not isinstance(judge.get("checks"), list):
        raise InputInvalid("register.json", "judge must be {mode: 'P+X', checks: [...]}")
    unknown = [c for c in judge["checks"] if c not in BATCH1_CHECKS + BATCH2_CHECKS]
    if unknown:
        raise InputInvalid("register.json", "unknown checks %s" % unknown)
    ctx.checks_applied = sorted(set(MANDATORY_CHECKS) | set(judge["checks"]),
                                key=lambda c: (len(c), c))
    ctx.register = doc
    ctx.register_files = files


def _load_calibration(ctx):
    cal = ctx.register.get("calibration", [])
    if not isinstance(cal, list):
        raise InputInvalid("register.calibration", "must be a list")
    for i, c in enumerate(cal):
        item = "register.calibration[%d]" % i
        if not isinstance(c, dict) or c.get("kind") not in CALIB_KINDS:
            raise InputInvalid(item, "kind must be one of %s" % (CALIB_KINDS,))
        p = _registered_file(ctx, c, item, kinds=(c["kind"],))
        try:
            doc = read_json(p)
        except (OSError, ValueError) as e:
            raise InputInvalid(item, str(e))
        if c["kind"] == "pilot":
            _check_pilot_env(ctx, doc, item)
        try:
            ctx.bands.add_source(c["kind"], c["relpath"], c["sha256"], doc)
        except BandError as e:
            raise InputInvalid(item, str(e))
    inline = ctx.register.get("inline_thresholds")
    if inline:
        if ctx.ident.experiment != "E0":
            raise InputInvalid("register.inline_thresholds", "only E0 may use inline thresholds")
        try:
            ctx.bands.add_source("inline", ctx.launch["register"]["relpath"]
                                 + "#inline_thresholds", ctx.ident.register_sha, inline)
        except BandError as e:
            raise InputInvalid("register.inline_thresholds", str(e))


def _check_pilot_env(ctx, doc, item):
    if not isinstance(doc, dict) or doc.get("schema") != PILOT_SCHEMA:
        raise InputInvalid(item, "pilot schema must be %s" % PILOT_SCHEMA)
    env = doc.get("environment") or {}
    want_hw = ctx.launch.get("hw_profile_sha")
    want_img = (ctx.launch.get("image") or {}).get("sha256")
    if env.get("hw_profile_sha") != want_hw or env.get("image_sha") != want_img:
        # Registered comparability (DATA-AND-ANALYSIS 8.1) is not read in batch 1: a pilot from a
        # different hardware profile or image is rejected outright.
        raise InputInvalid(item, "pilot environment (hw %r, image %r) != launch (hw %r, image %r)"
                           % (env.get("hw_profile_sha"), env.get("image_sha"), want_hw, want_img))


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
    (files directly in raw/env/: the four job key fields)."""
    env_dir = os.path.join(ctx.tree.raw, "env")
    n = 0
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
    _check_reconcile_and_scenario(ctx)
    _scan_identities(ctx)
    _load_upstream(ctx)
    return ctx
