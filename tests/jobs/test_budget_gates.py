# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capjobs.submit's job-plan gates (calib-v2 DESIGN-r7a P2-08b; tests 24, 33 and 36 of its 3.5):
registration, plan item and order, budget by distinct item_id, concurrency, disk, maintenance.
A stub runner answers ssh; nothing is ever executed on a cluster."""

import datetime
import json
import os
import subprocess
import types

import pytest
import test_submit as ts

from capcommon import jobplan, ledger
from capcommon.hashing import sha256_file
from capjobs import budget, remote, submit

T0 = datetime.datetime(2026, 11, 16, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
NO_RESV = (0, "No reservations in the system\n", "")
SINFO = (0, "n1\nn2\nn3\n", "")
EXCL_V01 = ("umbriel-b200-026", "umbriel-b200-074", "umbriel-b200-081")
SUBMISSION_HEADER = ("event_at\tplan_id\tevent\tjob_id\tstatus\treason\texperiment\tarms\t"
                     "launches\tbasis\taccount\tpartition\tqos\texclude\ttime_limit\tmem\t"
                     "cpus_per_task\tgpus\texclusive\tsnapshot_dir\tsnapshot_sha256\t"
                     "sbatch_argv\tsubmitted_at\tmounts_json\ttokenizer_dir_sha256\t"
                     "assertions_json")


def item(order, item_id, cls="planned", block=None, slot=None, pin=False):
    return {"order": order, "item_id": item_id, "item_class": cls, "arms": ["nullsse-a", "sfe-a"],
            "launches": 2, "block_id": block, "slot": slot,
            "config_counts": {"nullsse": 1, "sfe": 1}, "pin": pin}


class World:
    """A test_submit.Env plus a registered node_exclusions, job plans and gated registers."""

    def __init__(self, tmp_path, monkeypatch):
        self.env = ts.Env(tmp_path)
        self.w = self.env.w
        self.free = 10 ** 15
        monkeypatch.setattr(budget, "_now", lambda: T0)
        monkeypatch.setattr(budget, "_statvfs", lambda p: types.SimpleNamespace(
            f_bavail=self.free, f_frsize=1))
        self.excl = self.reg_json("02_e0/E0/register/v2/node_exclusions.v01.json", {
            "schema": "capkit.node_exclusions/1", "version": 1, "previous": None,
            "nodes": [{"node": n, "reason": "r", "evidence": "e"} for n in EXCL_V01]},
            "node_exclusions")
        self.exclude_file = self.write("exclude.json", {
            "schema": "capkit.cdb_exclude/1", "partitions": sorted(ts.PARTS.split(",")),
            "exclude": list(EXCL_V01) + ["n9"]})

    def write(self, name, doc):
        return self.env.write_json(name, doc)

    def reg_file(self, rel, data, kind):
        p = os.path.join(self.w, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
        ledger.registry_append(os.path.join(self.w, "REGISTRY.tsv"), "E0", rel, sha256_file(p),
                               "kind=%s;" % kind)
        return p

    def reg_json(self, rel, doc, kind):
        return self.reg_file(rel, json.dumps(doc).encode(), kind)

    def plan(self, rel, items):
        return self.reg_file(rel, jobplan.format_tsv(items), "job_plan")

    def register(self, name, plan_path, budget_id="E0v2-V0cal", job_budget=12, plan_sha=None,
                 disk=None):
        rel = "02_e0/E0/register/v2/" + name
        doc = {"schema": "capkit.register/1", "experiment": "E0", "arms": ["nullsse-a"],
               "files": {"scenario": {"relpath": self.env.scen_rel, "sha256": sha256_file(
                   os.path.join(self.w, self.env.scen_rel))}},
               "environment": ts.REG_ENV,
               "expected": {"budget_id": budget_id, "job_budget": job_budget,
                            "plan": {"relpath": os.path.relpath(plan_path, self.w),
                                     "sha256": plan_sha or sha256_file(plan_path)},
                            "disk": disk or {"fs_path": self.w, "floor_bytes": 100 * 10 ** 9,
                                             "per_job_bytes": 2 * 10 ** 9}}}
        p = self.reg_json(rel, doc, "register")
        sha = sha256_file(p)
        return sha, self.env.plan_for("launch_plan.%s" % name.replace("/", "."), sha)

    def args(self, reg, item_id, **over):
        sha, lp = reg
        a = {"--exclude": False, "--exclude-file": self.exclude_file, "--register-sha": sha,
             "--launch-plan": lp, "--plan-item": item_id, "--cluster-utc-offset": "+00:00"}
        if item_id is None:
            a["--plan-item"] = False
        a.update(over)
        return self.env.submit_args(**a)

    def submit(self, reg, item_id, answers=None, **over):
        ans = {"sbatch": (0, "%d\n" % (5600000 + len(self.budget_rows())), ""),
               "scontrol": NO_RESV, "sinfo": SINFO}
        ans.update(answers or {})
        runner = StubSsh(ans)
        return ts.run(self.args(reg, item_id, **over), runner), runner

    def budget_rows(self):
        return ledger.budget_rows(os.path.join(self.w, "BUDGET.tsv"))

    def state(self):
        """Bytes of every file the gates must not write when they refuse."""
        out = {}
        for rel in ("submissions.tsv", "BUDGET.tsv", budget.RESERVATIONS_LOG_REL):
            p = os.path.join(self.w, rel)
            out[rel] = open(p, "rb").read() if os.path.exists(p) else None
        return out


class StubSsh:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def run(self, argv):
        assert argv[0] == "ssh", argv
        self.calls.append(list(argv))
        ans = self.answers[argv[-1].split()[0]]
        return ans(argv) if callable(ans) else ans


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def three_items(world, rel="02_e0/E0/register/v2/V0cal/job_plan.A.tsv", prefix="V0cal-A"):
    return world.plan(rel, [item(k, "%s-%04d" % (prefix, k)) for k in (1, 2, 3)])


# ---- test 24: budget by distinct item_id, order, refusals that write nothing ---------------------

def test_budget_n_items_pass_and_the_next_exits_3(world):
    reg = world.register("V0cal/register.json", three_items(world), job_budget=2)
    for k in (1, 2):
        rc, r = world.submit(reg, "V0cal-A-%04d" % k)
        assert rc == 0
    rows = world.budget_rows()
    assert [(x["event"], x["item_id"]) for x in rows] == [
        ("RESERVED", "V0cal-A-0001"), ("JOB_ID", "V0cal-A-0001"),
        ("RESERVED", "V0cal-A-0002"), ("JOB_ID", "V0cal-A-0002")]
    assert rows[1]["job_id"] == "5600000" and rows[0]["job_id"] == ""
    assert rows[0]["plan_id"] == rows[1]["plan_id"] == ts.ledger.read_tsv(
        os.path.join(world.w, "submissions.tsv"), ledger.SUBMISSION_COLUMNS)[0]["plan_id"]
    assert ledger.budget_count(os.path.join(world.w, "BUDGET.tsv"), "E0v2-V0cal") == 2
    before = world.state()
    rc, r = world.submit(reg, "V0cal-A-0003")
    assert rc == 3 and world.state() == before and not any("sbatch" in c[-1] for c in r.calls)


def test_unknown_submit_then_reconcile_keeps_the_count(world):
    reg = world.register("V0cal/register.json", three_items(world))
    rc, _ = world.submit(reg, "V0cal-A-0001", answers={"sbatch": (None, "", "timeout")})
    assert rc == 4
    bpath = os.path.join(world.w, "BUDGET.tsv")
    assert [x["event"] for x in world.budget_rows()] == ["RESERVED"]
    plan_id = world.budget_rows()[0]["plan_id"]
    count = ledger.budget_count(bpath, "E0v2-V0cal")
    sc = (0, "JobId=777 Command=%s/jobs/capkit_job.sbatch\n" % ledger.submissions_current(
        os.path.join(world.w, "submissions.tsv"))[plan_id]["snapshot_dir"], "")
    assert ts.run(["reconcile-submit", "--login-host", "fe", "--ssh-timeout-s", "5",
                   "--workdir", world.w, "--plan-id", plan_id, "--job-id", "777", "--reason",
                   "squeue shows it"], StubSsh({"scontrol": sc})) == 0
    assert [(x["event"], x["job_id"]) for x in world.budget_rows()] == [("RESERVED", ""),
                                                                        ("JOB_ID", "777")]
    assert ledger.budget_count(bpath, "E0v2-V0cal") == count == 1
    # the item is consumed: submitting it again is a resubmit
    assert world.submit(reg, "V0cal-A-0001")[0] == 3


def test_one_budget_id_across_two_plan_files(world):
    a = world.register("V1b/register.A.json", three_items(world), budget_id="E0v2-V1b",
                       job_budget=3)
    pb = world.plan("02_e0/E0/register/v2/V1b/job_plan.B.tsv",
                    [item(k, "V1b-B-%04d" % k) for k in (1, 2)])
    b = world.register("V1b/register.B.json", pb, budget_id="E0v2-V1b", job_budget=3)
    assert world.submit(a, "V0cal-A-0001")[0] == 0
    assert world.submit(a, "V0cal-A-0002")[0] == 0
    assert world.submit(b, "V1b-B-0001")[0] == 0  # count 2 + 1 <= 3
    before = world.state()
    assert world.submit(b, "V1b-B-0002")[0] == 3  # count 3 + 1 > 3
    assert world.state() == before


@pytest.mark.parametrize("case", ["missing", "unknown", "resubmit", "skips_order", "plan_sha",
                                  "arms", "literal_exclude", "ungated_item"])
def test_plan_item_refusals_exit_3_and_write_nothing(world, case):
    p = three_items(world)
    reg = world.register("V0cal/register.json", p)
    assert world.submit(reg, "V0cal-A-0001")[0] == 0
    over, item_id = {}, "V0cal-A-0002"
    if case == "missing":
        item_id = None
    elif case == "unknown":
        item_id = "V0cal-A-0099"
    elif case == "resubmit":
        item_id = "V0cal-A-0001"
    elif case == "skips_order":
        item_id = "V0cal-A-0003"
    elif case == "plan_sha":
        reg = world.register("V0cal/register.bad.json", p, plan_sha="0" * 64)
    elif case == "arms":
        over = {"--arms": "nullsse-a", "--launches": "1"}
    elif case == "literal_exclude":
        over = {"--exclude-file": False, "--exclude": ",".join(EXCL_V01)}
    else:  # --plan-item with a register without expected / arm_config
        reg = (world.env.reg_sha, world.env.plan_path)
    before = world.state()
    rc, r = world.submit(reg, item_id, **over)
    assert rc == 3, case
    assert world.state() == before and not any("sbatch" in c[-1] for c in r.calls)


def test_block_pair_submits_consecutively_and_job_env_carries_the_node(world):
    p = world.plan("02_e0/E0/register/v2/V2b/job_plan.AA.tsv", [
        item(1, "V2b-AA-b01-s1", block="AA-b01", slot=1, pin=True),
        item(1, "V2b-AA-b01-s2", block="AA-b01", slot=2, pin=True)])
    reg = world.register("V2b/register.json", p, budget_id="E0v2-V2b")
    pinned = world.write("allbut.json", {"schema": "capkit.cdb_exclude/1", "all_but": "n1",
                                         "partitions": sorted(ts.PARTS.split(",")),
                                         "exclude": sorted(list(EXCL_V01) + ["n2", "n3"])})
    over = {"--exclude-file": pinned, "--expect-node": "n1"}
    assert world.submit(reg, "V2b-AA-b01-s1", **over)[0] == 0
    assert world.submit(reg, "V2b-AA-b01-s2", **over)[0] == 0
    snap = ledger.read_tsv(os.path.join(world.w, "submissions.tsv"),
                           ledger.SUBMISSION_COLUMNS)[0]["snapshot_dir"]
    sh = subprocess.run(["bash", "-c", 'source "$1/job.env"; printf %s "$CAPJOBS_EXPECT_NODE"',
                         "x", snap], capture_output=True, text=True, check=True)
    assert sh.stdout == "n1"


@pytest.mark.parametrize("over", [{"--expect-node": False},
                                  {"--expect-node": "n2"}])
def test_pinned_item_needs_the_expect_node_of_its_all_but_file(world, over):
    p = world.plan("02_e0/E0/register/v2/V2b/job_plan.AA.tsv", [
        item(1, "V2b-AA-b01-s1", block="AA-b01", slot=1, pin=True),
        item(1, "V2b-AA-b01-s2", block="AA-b01", slot=2, pin=True)])
    reg = world.register("V2b/register.json", p, budget_id="E0v2-V2b")
    pinned = world.write("allbut.json", {"schema": "capkit.cdb_exclude/1", "all_but": "n1",
                                         "partitions": sorted(ts.PARTS.split(",")),
                                         "exclude": sorted(list(EXCL_V01) + ["n2", "n3"])})
    a = {"--exclude-file": pinned, "--expect-node": "n1"}
    a.update(over)
    assert world.submit(reg, "V2b-AA-b01-s1", **a)[0] == 3
    reg1 = world.register("V0cal/register.json", three_items(world))
    assert world.submit(reg1, "V0cal-A-0001", **{"--expect-node": "n1"})[0] == 3  # not pinned


def test_dry_run_writes_nothing_and_the_submissions_header_is_unchanged(world, capsys):
    reg = world.register("V0cal/register.json", three_items(world))
    rc, r = world.submit(reg, "V0cal-A-0001", **{"--dry-run": None})
    assert rc == 0 and world.state() == {k: None for k in world.state()}
    out = json.loads(capsys.readouterr().out)
    assert out["budget_gate"]["item"]["item_id"] == "V0cal-A-0001"
    assert [c[-1].split()[0] for c in r.calls] == ["scontrol"]
    assert world.submit(reg, "V0cal-A-0001")[0] == 0
    with open(os.path.join(world.w, "submissions.tsv")) as f:
        assert f.readline().rstrip("\n") == SUBMISSION_HEADER


def test_void_item_cannot_be_submitted_again(world):
    """test 33: there is no replace path; a VOID job's item stays consumed."""
    reg = world.register("V0cal/register.json", three_items(world))
    assert world.submit(reg, "V0cal-A-0001")[0] == 0
    plan_id = world.budget_rows()[0]["plan_id"]
    ledger.submissions_append(os.path.join(world.w, "submissions.tsv"), submit.ledger_row(
        event="ENDED", plan_id=plan_id, job_id="5600000", status="VOID", reason="judged",
        experiment="E0"))
    before = world.state()
    assert world.submit(reg, "V0cal-A-0001")[0] == 3
    assert world.state() == before
    assert not any(a.startswith("--replace") or a.startswith("--resubmit")
                   for a in submit.build_parser()._subparsers._group_actions[0].choices[
                       "submit"]._option_string_actions)


def test_remote_allows_sinfo_and_never_scancel():
    argv = remote.ssh_argv("fe", ["sinfo", "-N", "-h", "-o", "%N", "-p", "b200@x/y"])
    assert argv[-1].startswith("sinfo -N -h -o %N -p")
    with pytest.raises(remote.RemoteError):
        remote.ssh_argv("fe", ["scancel", "1"])


# ---- test 36: concurrency, disk, maintenance -----------------------------------------------------

def open_jobs(world, n):
    for k in range(n):
        ledger.submissions_append(os.path.join(world.w, "submissions.tsv"), submit.ledger_row(
            event="SUBMITTED", plan_id="Popen%d" % k, job_id=str(100 + k), experiment="E0",
            submitted_at="2026-11-15T00:00:00.000Z"))


def test_concurrency_counts_the_block_as_two(world):
    single = world.register("V0cal/register.json", three_items(world))
    p = world.plan("02_e0/E0/register/v2/V2b/job_plan.AA.tsv", [
        item(1, "V2b-AA-b01-s1", block="AA-b01", slot=1, pin=True),
        item(1, "V2b-AA-b01-s2", block="AA-b01", slot=2, pin=True)])
    blk = world.register("V2b/register.json", p, budget_id="E0v2-V2b")
    pinned = world.write("allbut.json", {"schema": "capkit.cdb_exclude/1", "all_but": "n1",
                                         "partitions": sorted(ts.PARTS.split(",")),
                                         "exclude": sorted(list(EXCL_V01) + ["n2"])})
    open_jobs(world, 7)
    before = world.state()
    assert world.submit(blk, "V2b-AA-b01-s1", **{"--exclude-file": pinned,
                                                 "--expect-node": "n1"})[0] == 3
    assert world.state() == before
    assert world.submit(single, "V0cal-A-0001")[0] == 0  # 7 + 1 <= 8
    assert world.submit(single, "V0cal-A-0002")[0] == 3  # 8 + 1 > 8


def test_disk_gate_exits_5_and_writes_nothing(world):
    reg = world.register("V0cal/register.json", three_items(world))
    open_jobs(world, 2)
    need = 100 * 10 ** 9 + 1.5 * 2 * 10 ** 9 * 3  # floor + 1.5 x per_job x (2 open + 1)
    world.free = int(need) - 1
    before = world.state()
    assert world.submit(reg, "V0cal-A-0001")[0] == 5
    assert world.state() == before
    world.free = int(need)
    assert world.submit(reg, "V0cal-A-0001")[0] == 0


def resv(start, end="2026-11-16T22:00:00", nodes="n[1-2]", part="(null)"):
    return (0, "ReservationName=maint StartTime=%s EndTime=%s Duration=11:55:00\n"
               "   Nodes=%s NodeCnt=2 CoreCnt=448 Features=(null) PartitionName=%s "
               "Flags=MAINT,SPEC_NODES\n   TRES=cpu=448\n   Users=root Groups=(null) "
               "Accounts=(null) Licenses=(null) State=INACTIVE BurstBuffer=(null) Watts=n/a\n"
               "   MaxStartDelay=(null)\n" % (start, end, nodes, part), "")


@pytest.mark.parametrize("answer,rc", [
    (resv("2026-11-16T09:05:00"), 4),            # now + --time 3600 + 300 s: defer
    (resv("2026-11-16T09:15:00"), 0),            # now + 3600 + 900 s: passes
    (resv("2026-11-16T07:00:00"), 4),            # inside the window
    (resv("2026-11-16T07:00:00", end="2026-11-16T07:30:00"), 0),  # already over
    (resv("2026-11-16T09:05:00", nodes="m[1-2]"), 0),             # disjoint node set
    (resv("2026-11-16T09:05:00", nodes="ALL"), 4),
    ((255, "", "ssh: connect to host fe: timed out"), 4),
    ((0, "garbage without fields\n", ""), 4),
    ((0, "", ""), 4)])
def test_maintenance_gate(world, answer, rc):
    reg = world.register("V0cal/register.json", three_items(world))
    before = world.state()
    got, r = world.submit(reg, "V0cal-A-0001", answers={"scontrol": answer})
    assert got == rc
    log = os.path.join(world.w, budget.RESERVATIONS_LOG_REL)
    recs = [json.loads(x) for x in open(log)]
    assert len(recs) == 1 and recs[0]["stdout"] == answer[1]
    assert recs[0]["decision"] == ("pass" if rc == 0 else "defer")
    if rc:
        after = world.state()
        assert after["submissions.tsv"] == before["submissions.tsv"]
        assert after["BUDGET.tsv"] == before["BUDGET.tsv"]
        assert not any("sbatch" in c[-1] for c in r.calls)


def test_maintenance_reads_the_partitions_live(world):
    reg = world.register("V0cal/register.json", three_items(world))
    rc, r = world.submit(reg, "V0cal-A-0001", answers={
        "scontrol": resv("2026-11-16T09:05:00"), "sinfo": (1, "", "sinfo: error")})
    assert rc == 4
    assert [c[-1].split()[0] for c in r.calls] == ["scontrol", "sinfo"]


def test_utc_offset_moves_the_cluster_clock(world):
    reg = world.register("V0cal/register.json", three_items(world))
    # now is 08:00 UTC. A controller 8 h behind UTC prints 00:00: a 01:05-05:00 window starts
    # within --time + 600 s -> defer; read as UTC the same window is over -> pass
    win = resv("2026-11-16T01:05:00", end="2026-11-16T05:00:00")
    assert world.submit(reg, "V0cal-A-0001", answers={"scontrol": win},  # "-08:00" as one token
                        **{"--cluster-utc-offset": False, "--cluster-utc-offset=-08:00": None}
                        )[0] == 4
    assert world.submit(reg, "V0cal-A-0001", answers={"scontrol": win},
                        **{"--cluster-utc-offset": "+00:00"})[0] == 0
    assert world.submit(reg, "V0cal-A-0002", **{"--cluster-utc-offset": "8"})[0] == 3
