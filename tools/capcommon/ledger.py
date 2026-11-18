# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append-only TSV ledgers: REGISTRY.tsv, submissions.tsv, BUDGET.tsv, JUDGED.tsv (INTERFACES.md
9.3, 9.5, 10.1; BUDGET.tsv: erratum calib-v2-A4).

Files are only ever opened in append mode; nothing here rewrites a ledger.
"""

import fcntl
import json
import os
import shlex

from .clock import parse_wall_iso, wall_iso
from .exitcodes import ConfigError
from .hashing import is_sha256_hex
from .jobplan import ITEM_ID_RE, check_item_class
from .policy import (ASSERTION_KEYS, MODEL_ASSERTION_KEYS, NO_MODEL_EXPERIMENTS,
                     PREP_ASSERTION_KEYS, PREP_TOKENIZER_EXPERIMENT, PREP_TOKENIZER_PURPOSE,
                     PolicyError, check_model_mounts, check_mounts, check_prep_tokenizer_mounts,
                     is_model_experiment, max_gpus_for)

REGISTRY_COLUMNS = ("registered_at", "experiment", "relpath", "sha256", "note")
REGISTRY_KINDS = ("register", "register_md", "tier_table", "schedule", "prompts", "scenario",
                  "pilot", "loadgen_band", "stub_calib", "near_boundary_noise", "prediction",
                  "hw_profile", "d14_rule",
                  # calibration v2 (calib-v2 DESIGN-r7a 3.1 P2-02, 3.2)
                  "errata", "ladder", "e0_params", "e0_summary", "d14_result", "e0_list",
                  "job_plan", "node_exclusions", "node_list")
SUBMISSION_COLUMNS = ("event_at", "plan_id", "event", "job_id", "status", "reason", "experiment",
                      "arms", "launches", "basis", "account", "partition", "qos", "exclude",
                      "time_limit", "mem", "cpus_per_task", "gpus", "exclusive", "snapshot_dir",
                      "snapshot_sha256", "sbatch_argv", "submitted_at", "mounts_json",
                      "tokenizer_dir_sha256", "assertions_json")
SUBMISSION_EVENTS = ("PLANNED", "SUBMITTED", "ENDED", "NOTE")
SUBMISSION_STATUSES = ("", "VALID", "VOID", "NOT_RUN", "ENV_INVALID", "CANCELLED",
                       "SUBMIT_FAILED", "UNKNOWN_SUBMIT")
# sbatch outcome unknown (the job may be queued): only on a NOTE row with submitted_at; a later
# SUBMITTED or ENDED row (written from evidence, never by resubmitting) supersedes it.
UNKNOWN_SUBMIT = "UNKNOWN_SUBMIT"
# BUDGET.tsv (<W>/BUDGET.tsv, append-only; calib-v2 DESIGN-r7a 3.2, P2-08b): RESERVED is written
# before sbatch, JOB_ID once the job id is known. The budget count of a budget_id is the number of
# distinct item_id with a RESERVED row (never the row count).
BUDGET_COLUMNS = ("event_at", "event", "budget_id", "item_id", "item_class", "plan_sha256",
                  "register_sha256", "plan_id", "job_id")
BUDGET_EVENTS = ("RESERVED", "JOB_ID")
JUDGED_COLUMNS = ("judged_at", "job_id", "restart_count", "launch_seq", "arm", "experiment",
                  "tree_relpath", "verdicts_sha256")


class LedgerError(ConfigError):
    pass


def _cell(v):
    s = "" if v is None else str(v)
    if "\t" in s or "\n" in s or "\r" in s:
        raise LedgerError("ledger cell contains tab/newline: %r" % s)
    return s


def read_tsv(path, columns):
    """Return list of dicts. Missing file -> []. Header must equal `columns` exactly."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return []
    header = tuple(lines[0].split("\t"))
    if header != tuple(columns):
        raise LedgerError("%s: header %s != expected %s" % (path, header, tuple(columns)))
    rows = []
    for i, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        if len(cells) != len(columns):
            raise LedgerError("%s:%d: %d cells, expected %d" % (path, i, len(cells), len(columns)))
        rows.append(dict(zip(columns, cells)))
    return rows


def append_tsv(path, columns, row):
    """Append one row (dict with exactly `columns`), writing the header first if the file is new.
    Uses an exclusive flock so concurrent appenders do not interleave."""
    if set(row) != set(columns):
        raise LedgerError("row keys %s != columns %s" % (sorted(row), sorted(columns)))
    line = "\t".join(_cell(row[c]) for c in columns) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        if os.fstat(fd).st_size == 0:
            os.write(fd, ("\t".join(columns) + "\n").encode("utf-8"))
        else:
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline().rstrip("\n")
            if tuple(first.split("\t")) != tuple(columns):
                raise LedgerError("%s: existing header does not match" % path)
        os.write(fd, line.encode("utf-8"))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---- REGISTRY.tsv ------------------------------------------------------------------------------

def registry_append(path, experiment, relpath, sha256, note, registered_at=None):
    if not is_sha256_hex(sha256):
        raise LedgerError("sha256 must be 64 lowercase hex")
    if not note.startswith("kind="):
        raise LedgerError("note must start with 'kind=<kind>;'")
    kind = note[5:].split(";", 1)[0]
    if kind not in REGISTRY_KINDS:
        raise LedgerError("unknown registry kind %r" % kind)
    if os.path.isabs(relpath) or relpath.startswith(".."):
        raise LedgerError("relpath must be relative to the work dir: %r" % relpath)
    existing = registry_lookup(path, sha256)
    if existing is not None:
        raise LedgerError("sha %s already registered at %s (%s)"
                          % (sha256, existing["registered_at"], existing["relpath"]))
    registered_at = registered_at or wall_iso()
    _check_ts("registered_at", registered_at)
    row = {"registered_at": registered_at, "experiment": experiment,
           "relpath": relpath, "sha256": sha256, "note": note}
    append_tsv(path, REGISTRY_COLUMNS, row)
    return row


def _check_ts(name, value):
    try:
        parse_wall_iso(value)
    except ValueError as e:
        raise LedgerError("%s: %s" % (name, e))


def registry_lookup(path, sha256):
    for r in read_tsv(path, REGISTRY_COLUMNS):
        if r["sha256"] == sha256:
            return r
    return None


def registry_kind(row):
    return row["note"][5:].split(";", 1)[0] if row["note"].startswith("kind=") else None


def registered_before(row, other_iso):
    """True if the registry row's registered_at is strictly earlier than other_iso (both are
    millisecond timestamps; equal means not before)."""
    return parse_wall_iso(row["registered_at"]) < parse_wall_iso(other_iso)


# ---- submissions.tsv ---------------------------------------------------------------------------

def submissions_append(path, row):
    if row.get("event") not in SUBMISSION_EVENTS:
        raise LedgerError("event must be one of %s" % (SUBMISSION_EVENTS,))
    if row.get("status", "") not in SUBMISSION_STATUSES:
        raise LedgerError("status must be one of %s" % (SUBMISSION_STATUSES,))
    if row.get("status") == UNKNOWN_SUBMIT and (row["event"] != "NOTE"
                                                  or not row.get("submitted_at")):
        raise LedgerError("%s is only a NOTE status and needs submitted_at" % UNKNOWN_SUBMIT)
    if row["event"] == "PLANNED" and not row.get("basis"):
        raise LedgerError("PLANNED rows need the approval basis text")
    gpus = row.get("gpus", "")
    if row["event"] == "PLANNED":
        try:
            g = int(gpus)
        except (TypeError, ValueError):
            raise LedgerError("gpus must be an integer on PLANNED rows")
        if g > max_gpus_for(row.get("experiment")):
            raise LedgerError("gpus %d is above the ceiling %d of experiment %r"
                              % (g, max_gpus_for(row.get("experiment")), row.get("experiment")))
        _check_planned_assertions(row)
    argv = row.get("sbatch_argv", "")
    if argv:
        try:
            parsed = json.loads(argv)
        except ValueError:
            raise LedgerError("sbatch_argv must be a JSON array")
        if not isinstance(parsed, list):
            raise LedgerError("sbatch_argv must be a JSON array")
    full = dict(row)
    full.setdefault("event_at", wall_iso())
    for k in ("event_at", "submitted_at"):
        if full.get(k):
            _check_ts(k, full[k])
    append_tsv(path, SUBMISSION_COLUMNS, full)
    return full


def _check_planned_assertions(row):
    """PLANNED rows must carry the evidence that policy.check_no_model_submission (or, for the
    tokenizer copy, policy.check_prep_tokenizer_submission) ran and passed."""
    if row.get("reason") == PREP_TOKENIZER_PURPOSE:
        return _check_prep_tokenizer_planned(row)
    if is_model_experiment(row.get("experiment")):
        return _check_model_planned(row)
    if row.get("experiment") not in NO_MODEL_EXPERIMENTS:
        raise LedgerError("experiment %r is not in the no-model list" % row.get("experiment"))
    try:
        mounts = json.loads(row.get("mounts_json") or "")
        assertions = json.loads(row.get("assertions_json") or "")
    except ValueError:
        raise LedgerError("PLANNED rows need mounts_json and assertions_json as JSON")
    try:
        check_mounts(mounts)
    except PolicyError as e:
        raise LedgerError(str(e))
    if not isinstance(assertions, dict) or set(assertions) != set(ASSERTION_KEYS) \
            or not all(v is True for v in assertions.values()):
        raise LedgerError("assertions_json must have every key of %s set to true"
                          % (ASSERTION_KEYS,))
    tok = row.get("tokenizer_dir_sha256", "")
    if tok != "none" and not is_sha256_hex(tok):
        raise LedgerError("tokenizer_dir_sha256 must be 64 hex or 'none'")


def _check_model_planned(row):
    """A whitelisted model experiment's PLANNED row (policy.check_model_submission ran)."""
    exp = row.get("experiment")
    try:
        g = int(str(row.get("gpus")))
    except ValueError:
        raise LedgerError("gpus must be an integer, got %r" % row.get("gpus"))
    if not 1 <= g <= max_gpus_for(exp):
        raise LedgerError("%s: gpus %d outside [1, %d]" % (exp, g, max_gpus_for(exp)))
    try:
        mounts = json.loads(row.get("mounts_json") or "")
        assertions = json.loads(row.get("assertions_json") or "")
    except ValueError:
        raise LedgerError("PLANNED rows need mounts_json and assertions_json as JSON")
    try:
        check_model_mounts(exp, mounts)
    except PolicyError as e:
        raise LedgerError(str(e))
    if not isinstance(assertions, dict) or set(assertions) != set(MODEL_ASSERTION_KEYS) \
            or not all(v is True for v in assertions.values()):
        raise LedgerError("assertions_json must have every key of %s set to true"
                          % (MODEL_ASSERTION_KEYS,))
    tok = row.get("tokenizer_dir_sha256", "")
    if tok != "none" and not is_sha256_hex(tok):
        raise LedgerError("tokenizer_dir_sha256 must be 64 hex or 'none'")


def _check_prep_tokenizer_planned(row):
    """The one PLANNED row kind allowed to mount (part of) a model library."""
    if row.get("experiment") != PREP_TOKENIZER_EXPERIMENT:
        raise LedgerError("a %s row must have experiment %s"
                          % (PREP_TOKENIZER_PURPOSE, PREP_TOKENIZER_EXPERIMENT))
    if str(row.get("gpus")) != "0":
        raise LedgerError("a %s row requests no GPU" % PREP_TOKENIZER_PURPOSE)
    try:
        mounts = json.loads(row.get("mounts_json") or "")
        assertions = json.loads(row.get("assertions_json") or "")
    except ValueError:
        raise LedgerError("PLANNED rows need mounts_json and assertions_json as JSON")
    try:
        check_prep_tokenizer_mounts(mounts)
    except PolicyError as e:
        raise LedgerError(str(e))
    if not isinstance(assertions, dict) or set(assertions) != set(PREP_ASSERTION_KEYS) \
            or not all(v is True for v in assertions.values()):
        raise LedgerError("assertions_json must have every key of %s set to true"
                          % (PREP_ASSERTION_KEYS,))
    if row.get("tokenizer_dir_sha256") != "none":
        raise LedgerError("a %s row has tokenizer_dir_sha256 'none' (it creates the copy)"
                          % PREP_TOKENIZER_PURPOSE)


def submissions_current(path):
    """{plan_id: merged row} where later events override non-empty fields of earlier ones; the
    status of a SUBMITTED or ENDED row replaces the earlier status even when empty (a SUBMITTED
    row closes an UNKNOWN_SUBMIT)."""
    cur = {}
    for r in read_tsv(path, SUBMISSION_COLUMNS):
        pid = r["plan_id"]
        if pid not in cur:
            cur[pid] = dict(r)
        else:
            for k, v in r.items():
                if v != "" or (k == "status" and r["event"] in ("SUBMITTED", "ENDED")):
                    cur[pid][k] = v
    return cur


def submission_unknown(cur):
    """True when a merged row (submissions_current) is waiting for reconciliation."""
    return cur.get("status") == UNKNOWN_SUBMIT


def plan_job_env_value(snapshot_dir, key):
    """A value of a submitted plan's own job.env (<snapshot_dir>/job.env, `export KEY=<shell
    word>`), '' when the key is absent; OSError when job.env cannot be read."""
    with open(os.path.join(snapshot_dir, "job.env"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("export %s=" % key):
                vals = shlex.split(line.split("=", 1)[1])
                return vals[0] if vals else ""
    return ""


def submission_for_job(path, job_id):
    """The merged row whose job_id matches, or None (error if more than one plan has it)."""
    hits = [r for r in submissions_current(path).values() if r["job_id"] == str(job_id)]
    if len(hits) > 1:
        raise LedgerError("job %s appears under more than one plan_id" % job_id)
    return hits[0] if hits else None


# ---- BUDGET.tsv --------------------------------------------------------------------------------

def budget_append(path, row):
    """Append one BUDGET.tsv row (every column but event_at given; event_at is set here)."""
    full = dict(row)
    if full.get("event") not in BUDGET_EVENTS:
        raise LedgerError("BUDGET event must be one of %s" % (BUDGET_EVENTS,))
    if not full.get("budget_id") or not ITEM_ID_RE.match(full["budget_id"]):
        raise LedgerError("BUDGET budget_id must match %s" % ITEM_ID_RE.pattern)
    if not full.get("item_id") or not ITEM_ID_RE.match(full["item_id"]):
        raise LedgerError("BUDGET item_id must match %s" % ITEM_ID_RE.pattern)
    check_item_class(full.get("item_class"))
    for k in ("plan_sha256", "register_sha256"):
        if not is_sha256_hex(full.get(k, "")):
            raise LedgerError("BUDGET %s must be 64 lowercase hex" % k)
    if not full.get("plan_id"):
        raise LedgerError("BUDGET rows name the submission plan_id")
    if (full["event"] == "JOB_ID") != bool(full.get("job_id")):
        raise LedgerError("BUDGET: a JOB_ID row has a job id, a RESERVED row has none")
    full.setdefault("event_at", wall_iso())
    _check_ts("event_at", full["event_at"])
    append_tsv(path, BUDGET_COLUMNS, full)
    return full


def budget_rows(path):
    return read_tsv(path, BUDGET_COLUMNS)


def budget_reserved_items(path, budget_id=None):
    """{item_id} with a RESERVED row (in budget_id when given)."""
    return {r["item_id"] for r in budget_rows(path) if r["event"] == "RESERVED"
            and (budget_id is None or r["budget_id"] == budget_id)}


def budget_count(path, budget_id):
    """The budget count: distinct item_id with a RESERVED row for budget_id."""
    return len(budget_reserved_items(path, budget_id))


# ---- JUDGED.tsv --------------------------------------------------------------------------------

def judged_key(row):
    return (row["job_id"], str(row["restart_count"]), str(row["launch_seq"]), row["arm"])


def judged_lookup(path, job_id, restart_count, launch_seq, arm):
    key = (str(job_id), str(restart_count), str(launch_seq), arm)
    for r in read_tsv(path, JUDGED_COLUMNS):
        if judged_key(r) == key:
            return r
    return None


def judged_append(path, row):
    full = dict(row)
    full.setdefault("judged_at", wall_iso())
    if judged_lookup(path, full["job_id"], full["restart_count"], full["launch_seq"],
                     full["arm"]) is not None:
        raise LedgerError("launch already judged")
    append_tsv(path, JUDGED_COLUMNS, full)
    return full
