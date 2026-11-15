# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The submission gates of a registered job plan (calib-v2 DESIGN-r7a 3.1 P2-08b, 3.2, 4.5, 4.6;
INTERFACES errata calib-v2-A4, calib-v2-A6). capjobs.submit runs them, in this order, before it
writes anything, for every gated register (`gated`: an "arm_config" section, or an "expected"
section with a key other than the v1 pair job_budget / launches_per_arm):

  1 registration  expected.{budget_id, job_budget, plan, disk}; plan = a registered job_plan.tsv
                  (sha checked); --exclude-file holds every node of the latest registered
                  node_exclusions                                                     (exit 3)
  2 plan item     --plan-item in the plan, no RESERVED row yet (no replace / resubmit path), every
                  item of a smaller order already RESERVED; --arms / --launches / pin match (3)
  3 budget        distinct item_id with a RESERVED row for budget_id, + 1 <= job_budget  (3)
  4 concurrency   open jobs in submissions.tsv + unreserved items of this order <= 8     (3)
  5 disk          statvfs free >= floor_bytes + 1.5 x per_job_bytes x (open + this order) (5)
  6 maintenance   `scontrol show reservation` (logged to 02_e0/E0/cluster/reservations.log)
                  against the partitions' live sinfo nodes; read or parse failure too   (4)

Then submit writes the BUDGET.tsv RESERVED row, runs the usual submissions.tsv flow, and adds a
JOB_ID row once the job id is known. A real submission runs the gates after its dirty-tree check
and holds `reservation_lock` from the first gate until it returns (its RESERVED row included).
--dry-run runs the checks, takes no lock and writes nothing.
"""

import contextlib
import datetime
import fcntl
import json
import os
import re
import time

from capcommon import jobplan, ledger
from capcommon.clock import wall_iso
from capcommon.exitcodes import CONFIG, RUNTIME, STOPPED_EARLY, CapkitError
from capcommon.hashing import is_sha256_hex, sha256_file

from . import nodes, remote

MAX_CONCURRENT_JOBS = 8
DISK_MARGIN = 1.5
MAINTENANCE_PAD_S = 600
RESERVATIONS_LOG_REL = "02_e0/E0/cluster/reservations.log"
BUDGET_REL = "BUDGET.tsv"
# Held by a real gated submission from its first gate until it returns, so gate 2's read of the
# RESERVED rows and its own RESERVED row are under one lock. Not BUDGET.tsv itself:
# ledger.append_tsv flocks that file while appending, and a second flock on it from this
# process would wait for itself.
BUDGET_LOCK_REL = BUDGET_REL + ".lock"
OFFSET_RE = re.compile(r"^([+-])(\d\d):(\d\d)\Z")
SLURM_TIME_FMT = "%Y-%m-%dT%H:%M:%S"
_now = time.time  # injectable for tests
_statvfs = os.statvfs  # injectable for tests


class GateError(CapkitError):
    def __init__(self, message, exit_code=CONFIG):
        super().__init__(message)
        self.exit_code = exit_code


# The whole "expected" section of a v1 register (E0 round 1, e0w60, E1). A register without
# arm_config whose expected holds only these keys submits through the ungated flow.
V1_EXPECTED_KEYS = frozenset({"job_budget", "launches_per_arm"})


def gated(register_doc):
    if not isinstance(register_doc, dict):
        return False
    if "arm_config" in register_doc:
        return True
    if "expected" not in register_doc:
        return False
    exp = register_doc["expected"]
    return not (isinstance(exp, dict) and set(exp) <= V1_EXPECTED_KEYS)


def _expected(doc):
    exp = doc.get("expected")
    if not isinstance(exp, dict):
        raise GateError("register has arm_config but no expected section (DESIGN 3.2)")
    ok_int = lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0  # noqa: E731
    plan, disk = exp.get("plan"), exp.get("disk")
    if not (isinstance(exp.get("budget_id"), str) and jobplan.ITEM_ID_RE.match(exp["budget_id"])
            and ok_int(exp.get("job_budget")) and isinstance(plan, dict)
            and isinstance(plan.get("relpath"), str) and is_sha256_hex(plan.get("sha256", ""))
            and isinstance(disk, dict) and isinstance(disk.get("fs_path"), str)
            and os.path.isabs(disk["fs_path"]) and ok_int(disk.get("floor_bytes"))
            and ok_int(disk.get("per_job_bytes"))):
        raise GateError("register expected needs budget_id, job_budget > 0, plan {relpath, "
                        "sha256}, disk {fs_path (absolute), floor_bytes, per_job_bytes}")
    return exp


def _plan(workdir, exp):
    rel = exp["plan"]["relpath"]
    path = os.path.join(workdir, rel)
    sha = sha256_file(path)
    if sha != exp["plan"]["sha256"]:
        raise GateError("job plan %s sha256 %s != register expected.plan" % (rel, sha))
    row = ledger.registry_lookup(os.path.join(workdir, "REGISTRY.tsv"), sha)
    if row is None or ledger.registry_kind(row) != "job_plan" or row["relpath"] != rel:
        raise GateError("job plan %s is not in REGISTRY.tsv as kind=job_plan" % rel)
    return jobplan.read(path), sha


def _exclusions(workdir, res):
    src = res.get("exclude_source") or {}
    if "file" not in src:
        raise GateError("a registered plan item needs --exclude-file (written by `exclude`)")
    latest = nodes.latest_node_exclusions(workdir)
    if latest is None:
        return None
    missing = sorted(set(latest["nodes"]) - set(res["exclude"]))
    if missing:
        raise GateError("--exclude-file lacks %s of the latest node_exclusions %s"
                        % (missing, latest["relpath"]))
    return {"relpath": latest["relpath"], "sha256": latest["sha256"]}


def _item(a, items, reserved, exclude_doc):
    if not a.plan_item:
        raise GateError("--plan-item is required for a register with a job plan")
    it = next((x for x in items if x["item_id"] == a.plan_item), None)
    if it is None:
        raise GateError("plan item %s is not in the registered job plan" % a.plan_item)
    if it["item_id"] in reserved:
        raise GateError("plan item %s already has a RESERVED row: there is no replace or "
                        "resubmit path" % it["item_id"])
    behind = [x["item_id"] for x in items if x["order"] < it["order"]
              and x["item_id"] not in reserved]
    if behind:
        raise GateError("items of a smaller order are not submitted yet: %s" % behind)
    if a.arms.split(",") != it["arms"] or a.launches != it["launches"]:
        raise GateError("--arms / --launches differ from plan item %s (%s, %d)"
                        % (it["item_id"], ",".join(it["arms"]), it["launches"]))
    if it["pin"]:
        if not a.expect_node or (exclude_doc or {}).get("all_but") != a.expect_node:
            raise GateError("pinned item %s needs --expect-node and an `exclude --all-but` file "
                            "for the same node" % it["item_id"])
    elif a.expect_node:
        raise GateError("item %s is not pinned: --expect-node is not allowed" % it["item_id"])
    return it


def open_jobs(ledger_path):
    """Plans in submissions.tsv not ENDED (queued, running or of unknown outcome); local runs
    never sit in the queue."""
    return sorted(pid for pid, r in ledger.submissions_current(ledger_path).items()
                  if r["event"] != "ENDED" and not r["job_id"].startswith("local-"))


def parse_offset(text):
    m = OFFSET_RE.match(text or "")
    if not m or int(m.group(3)) >= 60 or int(m.group(2)) > 14:
        raise GateError("--cluster-utc-offset must be +HH:MM or -HH:MM")
    s = int(m.group(2)) * 3600 + int(m.group(3)) * 60
    return s if m.group(1) == "+" else -s


def _slurm_time(text, end):
    if end and text in ("Unknown", "None", "INFINITE"):
        return float("inf")
    try:
        return datetime.datetime.strptime(text, SLURM_TIME_FMT).replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        raise nodes.NodesError("bad reservation time %r" % text)


def maintenance(a, runner, partitions, time_s, log_path):
    """Gate 6. Returns the decision record; raises GateError(exit 4) to defer. The raw output is
    appended to log_path (None: dry run, nothing written)."""
    offset_s = parse_offset(a.cluster_utc_offset)
    now_local = _now() + offset_s
    r = remote.run_remote(runner, a.login_host, ["scontrol", "show", "reservation"])
    rec = {"t_wall": wall_iso(), "cluster_utc_offset": a.cluster_utc_offset,
           "now_cluster_local": datetime.datetime.fromtimestamp(
               now_local, datetime.timezone.utc).strftime(SLURM_TIME_FMT),
           "remote_argv": r["remote_argv"], "rc": r["rc"], "stdout": r["stdout"],
           "stderr": r["stderr"], "time_s": time_s, "pad_s": MAINTENANCE_PAD_S}
    problem, conflicts = None, []
    try:
        if r["rc"] != 0:
            raise nodes.NodesError("scontrol show reservation rc=%r" % (r["rc"],))
        resv = nodes.parse_reservations(r["stdout"])
        mine = set(nodes.sinfo_nodes(runner, a.login_host, partitions)[0]) if resv else set()
        for x in resv:
            hits = x["nodes"] == "ALL" or bool(mine.intersection(x["nodes"])) \
                or x["partition"] in partitions
            start, end = _slurm_time(x["start"], False), _slurm_time(x["end"], True)
            if hits and now_local + time_s + MAINTENANCE_PAD_S > start and now_local < end:
                conflicts.append(x["name"])
    except (nodes.NodesError, remote.RemoteError) as e:
        problem = str(e)
    rec.update({"conflicts": conflicts, "problem": problem,
                "decision": "defer" if problem or conflicts else "pass"})
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    if problem:
        raise GateError("maintenance gate cannot read the reservations (%s): defer" % problem,
                        RUNTIME)
    if conflicts:
        raise GateError("maintenance reservation(s) %s overlap now + --time + %d s: defer"
                        % (conflicts, MAINTENANCE_PAD_S), RUNTIME)
    return rec


@contextlib.contextmanager
def reservation_lock(workdir):
    """fcntl.flock(LOCK_EX) on <W>/BUDGET.tsv.lock (created empty when missing); a concurrent
    real submission waits here, then reads the RESERVED row the first one wrote."""
    fd = os.open(os.path.join(workdir, BUDGET_LOCK_REL), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def gates(a, prep, runner, dry_run):
    """Run gates 1-6 for a gated register; None for an ungated one (then --plan-item is
    refused). Nothing is written except the reservations log line of a real submission."""
    workdir, doc = prep["workdir"], prep["register"]["doc"]
    if not gated(doc):
        if a.plan_item:
            raise GateError("--plan-item needs a gated register (arm_config, or expected "
                            "beyond job_budget / launches_per_arm)")
        return None
    exp = _expected(doc)  # 1
    items, plan_sha = _plan(workdir, exp)
    excl = _exclusions(workdir, prep["res"])
    exclude_doc = prep["res"].get("exclude_doc")
    budget_path = os.path.join(workdir, BUDGET_REL)
    reserved = ledger.budget_reserved_items(budget_path)
    it = _item(a, items, reserved, exclude_doc)  # 2
    count = ledger.budget_count(budget_path, exp["budget_id"])  # 3
    if count + 1 > exp["job_budget"]:
        raise GateError("budget %s: %d of %d items reserved; no room for %s"
                        % (exp["budget_id"], count, exp["job_budget"], it["item_id"]))
    n_open = len(open_jobs(os.path.join(workdir, "submissions.tsv")))  # 4
    n_order = sum(1 for x in items if x["order"] == it["order"] and x["item_id"] not in reserved)
    if n_open + n_order > MAX_CONCURRENT_JOBS:
        raise GateError("concurrency: %d open + %d of this order > %d: try again later"
                        % (n_open, n_order, MAX_CONCURRENT_JOBS))
    disk = exp["disk"]  # 5
    st = _statvfs(disk["fs_path"])
    free = st.f_bavail * st.f_frsize
    need = disk["floor_bytes"] + DISK_MARGIN * disk["per_job_bytes"] * (n_open + n_order)
    if free < need:
        raise GateError("disk: %d bytes free on %s < %d needed (N17): stop and report"
                        % (free, disk["fs_path"], need), STOPPED_EARLY)
    resv = maintenance(a, runner, prep["res"]["partition"].split(","), prep["res"]["time_s"],
                       None if dry_run else os.path.join(workdir, RESERVATIONS_LOG_REL))  # 6
    return {"budget_id": exp["budget_id"], "job_budget": exp["job_budget"], "count": count,
            "item": it, "plan_relpath": exp["plan"]["relpath"], "plan_sha256": plan_sha,
            "register_sha256": prep["register"]["sha256"], "n_open": n_open,
            "n_order": n_order, "free_bytes": free, "need_bytes": need,
            "node_exclusions": excl, "maintenance": resv["decision"],
            "budget_path": budget_path}


def budget_row(gate, event, plan_id, job_id=""):
    return {"event": event, "budget_id": gate["budget_id"], "item_id": gate["item"]["item_id"],
            "item_class": gate["item"]["item_class"], "plan_sha256": gate["plan_sha256"],
            "register_sha256": gate["register_sha256"], "plan_id": plan_id, "job_id": job_id}
