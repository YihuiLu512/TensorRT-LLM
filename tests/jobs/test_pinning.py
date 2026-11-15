# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinning a job to a node (calib-v2 DESIGN-r7a P2-08a; tests 24, 37 and 39 of its 3.5):
exclude --all-but / --also-exclude, node_exclusions and node_list, pin-next, and the job's
--expect-node check. A stub runner answers ssh; nothing runs on a cluster."""

import json
import os
import shlex
import socket

import pytest
import test_budget_gates as bg
import test_submit as ts
from conftest import DATA, SAVED_CDB, external_input

from capcommon import ledger
from capjobs import envfiles, nodes, snapshot, submit

NODES5 = ["n1", "n2", "n3", "n4", "n5"]
PARTS = ts.PARTS.split(",")


@pytest.fixture
def world(tmp_path, monkeypatch):
    return bg.World(tmp_path, monkeypatch)


def node_list(world, names=NODES5, rel="02_e0/E0/register/v2/V2b/node_list.json"):
    return world.reg_json(rel, {"schema": "capkit.node_list/1",
                                "taken_at": "2026-11-15T00:00:00.000Z",
                                "source": {"relpath": "x", "sha256": "0" * 64},
                                "nodes": sorted(names)}, "node_list")


def exclusions(world, names, version=2, rel=None):
    rel = rel or "02_e0/E0/register/v2/node_exclusions.v%02d.json" % version
    return world.reg_json(rel, {"schema": "capkit.node_exclusions/1", "version": version,
                                "previous": None,
                                "nodes": [{"node": n, "reason": "microcode", "evidence": "J1"}
                                          for n in names]}, "node_exclusions")


def sinfo_answer(states):
    """Stub sinfo: '%N' or '%N %t' per partition; the first partition lists every node."""
    def ans(argv):
        args = shlex.split(argv[-1])
        if args[-1] != PARTS[0]:
            return 0, "", ""
        if args[args.index("-o") + 1] == "%N %t":
            return 0, "".join("%s %s\n" % kv for kv in states.items()), ""
        return 0, "".join("%s\n" % n for n in states), ""
    return ans


def exclude_args(world, *extra):
    return ["exclude", "--partition", ts.PARTS, "--workdir", world.w, "--login-host", "fe",
            "--ssh-timeout-s", "5"] + list(extra)


# ---- exclude --all-but / --also-exclude (tests 24, 37) -------------------------------------------

def test_all_but_is_the_live_sinfo_list_minus_the_target(world):
    nl = node_list(world)
    out = os.path.join(world.env.root, "ab.json")
    r = bg.StubSsh({"sinfo": sinfo_answer({n: "idle" for n in NODES5})})
    assert ts.run(exclude_args(world, "--all-but", "n3", "--node-list", nl, "--out", out),
                  r) == 0
    doc = json.load(open(out))
    assert doc["exclude"] == ["n1", "n2", "n4", "n5"] and doc["all_but"] == "n3"
    assert doc["partition_nodes"] == NODES5 and doc["partitions"] == sorted(PARTS)
    assert [c[-1].split()[:6] for c in r.calls] == [["sinfo", "-N", "-h", "-o", "%N", "-p"]] * 2
    # submit reads it as an exclude file (same schema)
    assert submit.cdbexclude.read_result(out)["exclude"] == doc["exclude"]


def test_all_but_sinfo_failure_writes_no_file(world):
    nl = node_list(world)
    out = os.path.join(world.env.root, "ab.json")
    r = bg.StubSsh({"sinfo": (1, "", "sinfo: error: slurm_load_partitions")})
    assert ts.run(exclude_args(world, "--all-but", "n3", "--node-list", nl, "--out", out),
                  r) == 3
    assert not os.path.exists(out)


@pytest.mark.parametrize("target", ["n9", "n6"])
def test_all_but_target_must_be_in_the_node_list_and_the_partitions(world, target):
    """n9: live in the partitions but not in the registered node_list; n6: in the node_list but
    not in the partitions."""
    nl = node_list(world, NODES5 + (["n6"] if target == "n6" else []),
                   rel="nl.%s.json" % target)
    out = os.path.join(world.env.root, "ab.json")
    r = bg.StubSsh({"sinfo": sinfo_answer({n: "idle" for n in NODES5 + ["n9"]})})
    assert ts.run(exclude_args(world, "--all-but", target, "--node-list", nl, "--out", out),
                  r) == 3
    assert not os.path.exists(out)


def test_also_exclude_merges_the_registered_list(world):
    out = os.path.join(world.env.root, "live.json")
    assert ts.run(["exclude", "--cdb-file", external_input(SAVED_CDB), "--main-profile-spec",
                   os.path.join(DATA, "g0_main_profile.json"), "--partition", ts.PARTS,
                   "--workdir", world.w, "--also-exclude", world.excl, "--out", out]) == 0
    doc = json.load(open(out))
    assert set(bg.EXCL_V01) <= set(doc["exclude"]) and len(doc["exclude"]) >= 12
    assert doc["also_exclude"]["sha256"] == ledger.registry_lookup(
        os.path.join(world.w, "REGISTRY.tsv"), doc["also_exclude"]["sha256"])["sha256"]
    nl = node_list(world, NODES5 + list(bg.EXCL_V01))
    out2 = os.path.join(world.env.root, "ab.json")
    r = bg.StubSsh({"sinfo": sinfo_answer({n: "idle" for n in NODES5 + list(bg.EXCL_V01)})})
    assert ts.run(exclude_args(world, "--all-but", "n1", "--node-list", nl, "--also-exclude",
                               world.excl, "--out", out2), r) == 0
    assert set(bg.EXCL_V01) <= set(json.load(open(out2))["exclude"])
    # the target may not be on the list
    assert ts.run(exclude_args(world, "--all-but", "umbriel-b200-026", "--node-list", nl,
                               "--also-exclude", world.excl, "--out", out2 + ".x"), r) == 3
    assert not os.path.exists(out2 + ".x")


def test_also_exclude_must_be_registered_with_its_sha(world):
    base = ["exclude", "--cdb-file", external_input(SAVED_CDB), "--main-profile-spec",
            os.path.join(DATA, "g0_main_profile.json"), "--partition", ts.PARTS,
            "--workdir", world.w]
    unreg = os.path.join(world.env.root, "unreg.json")
    with open(world.excl) as f, open(unreg, "w") as g:
        g.write(f.read())
    assert ts.run(base + ["--also-exclude", unreg, "--out", unreg + ".o"]) == 3
    with open(world.excl, "a") as f:
        f.write(" ")  # sha no longer the registered one
    assert ts.run(base + ["--also-exclude", world.excl, "--out", unreg + ".o2"]) == 3
    assert not os.path.exists(unreg + ".o") and not os.path.exists(unreg + ".o2")


def test_submit_exclude_file_must_hold_the_latest_list(world):
    reg = world.register("V0cal/register.json", bg.three_items(world))
    exclusions(world, list(bg.EXCL_V01) + ["n7"])  # v02 adds n7; the exclude file lacks it
    before = world.state()
    assert world.submit(reg, "V0cal-A-0001")[0] == 3
    assert world.state() == before
    full = world.write("exclude2.json", {"schema": "capkit.cdb_exclude/1",
                                         "partitions": sorted(PARTS),
                                         "exclude": list(bg.EXCL_V01) + ["n7"]})
    assert world.submit(reg, "V0cal-A-0001", **{"--exclude-file": full})[0] == 0


# ---- pin-next (test 39) --------------------------------------------------------------------------

def pinned_plan(world, n=6, rel="02_e0/E0/register/v2/V2b/job_plan.pin.tsv", block=False):
    if block:
        items = [bg.item(1, "V2b-AA-b01-s1", block="AA-b01", slot=1, pin=True),
                 bg.item(1, "V2b-AA-b01-s2", block="AA-b01", slot=2, pin=True)]
    else:
        items = [bg.item(k, "V2b-AA-%04d" % k, pin=True) for k in range(1, n + 1)]
    return world.plan(rel, items)


def fake_submitted(world, item_id, node, k):
    """BUDGET RESERVED + submissions SUBMITTED rows and a snapshot job.env with the node, as
    `submit --expect-node <node> --plan-item <item_id>` leaves them."""
    plan_id = "Ppin%02d" % k
    snap = os.path.join(world.env.root, "snaps", plan_id)
    os.makedirs(snap)
    with open(os.path.join(snap, "job.env"), "w") as f:
        f.write("export CAPJOBS_EXPECT_NODE=%s\nexport CAPJOBS_RUN_ROOT=%s\n"
                % (node, os.path.join(world.env.root, "runs")))
    ledger.budget_append(os.path.join(world.w, "BUDGET.tsv"), {
        "event": "RESERVED", "budget_id": "E0v2-V2b", "item_id": item_id,
        "item_class": "planned", "plan_sha256": "a" * 64, "register_sha256": "b" * 64,
        "plan_id": plan_id, "job_id": ""})
    ledger.submissions_append(os.path.join(world.w, "submissions.tsv"), submit.ledger_row(
        event="SUBMITTED", plan_id=plan_id, job_id=str(7000 + k), experiment="E0",
        snapshot_dir=snap, submitted_at="2026-11-15T00:00:00.000Z"))


def pin_next(world, nl, ex, plan, item_id, states, key="AA", seed="E0v2-V2b"):
    log = os.path.join(world.w, "02_e0/E0/cluster/pin_next.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    r = bg.StubSsh({"sinfo": sinfo_answer(states)})
    argv = ["pin-next", "--login-host", "fe", "--ssh-timeout-s", "5", "--workdir", world.w,
            "--node-list", nl, "--exclusions", ex, "--key", key, "--seed-base", seed,
            "--plan", plan, "--item", item_id, "--partition", ts.PARTS, "--log", log]
    rc = ts.run(argv, r)
    recs = [json.loads(x) for x in open(log)] if os.path.exists(log) else []
    return rc, (recs[-1] if recs else None)


STATES = {"n1": "idle", "n2": "idle", "n3": "mix", "n4": "drain", "n5": "alloc"}


def run_six(world, key="AA", seed="E0v2-V2b", judged=False):
    nl = node_list(world)
    ex = exclusions(world, ["n2"])
    plan = pinned_plan(world)
    chosen = []
    for k in range(1, 7):
        if judged:  # judge results in the work dir: pin-next reads none of them
            with open(os.path.join(world.w, "JUDGED.tsv"), "a") as f:
                f.write("anything\t%d\n" % k)
        rc, rec = pin_next(world, nl, ex, plan, "V2b-AA-%04d" % k, STATES, key, seed)
        assert rc == 0, rec
        chosen.append(rec["chosen"])
        fake_submitted(world, "V2b-AA-%04d" % k, rec["chosen"], k)
    return chosen


def test_pin_next_rotates_over_usable_nodes_then_repeats(world):
    chosen = run_six(world)
    order = nodes.rotation(NODES5, "E0v2-V2b", "AA")
    usable = [n for n in order if n not in ("n2", "n4")]
    assert chosen == usable + usable
    log = [json.loads(x) for x in open(os.path.join(world.w, "02_e0/E0/cluster/pin_next.log"))]
    reasons = {s["node"]: s["reason"] for rec in log for s in rec["skipped"]}
    assert reasons["n2"].startswith("node_exclusions") and reasons["n4"] == "sinfo state drain"
    assert log[0]["previous"] is None and log[1]["previous"] == chosen[0]


def test_pin_next_is_deterministic_per_seed_and_comparison(tmp_path, monkeypatch):
    a = run_six(bg.World(tmp_path / "a", monkeypatch))
    b = run_six(bg.World(tmp_path / "b", monkeypatch), judged=True)
    assert a == b
    rots = {k: nodes.rotation(NODES5, "E0v2-V2b", k) for k in ("AA", "RES3", "RES6", "SWAP")}
    assert len({tuple(v) for v in rots.values()}) > 1
    c = run_six(bg.World(tmp_path / "c", monkeypatch), key="RES6")
    assert c[:3] == [n for n in rots["RES6"] if n not in ("n2", "n4")]


def test_pin_next_block_partner_gets_the_same_node(world):
    nl, ex = node_list(world), exclusions(world, ["n2"])
    plan = pinned_plan(world, block=True, rel="02_e0/E0/register/v2/V2b/job_plan.blk.tsv")
    rc, rec = pin_next(world, nl, ex, plan, "V2b-AA-b01-s1", STATES)
    assert rc == 0 and rec["rule"] == "rotation"
    fake_submitted(world, "V2b-AA-b01-s1", rec["chosen"], 1)
    rc, rec2 = pin_next(world, nl, ex, plan, "V2b-AA-b01-s2", STATES)
    assert rc == 0 and rec2["chosen"] == rec["chosen"] and rec2["rule"].startswith("block")


def test_pin_next_refusals(world):
    nl, ex = node_list(world), exclusions(world, ["n2"])
    plan = pinned_plan(world)
    assert pin_next(world, nl, world.excl, plan, "V2b-AA-0001", STATES)[0] == 3  # not latest
    fake_submitted(world, "V2b-AA-0001", "n1", 1)
    assert pin_next(world, nl, ex, plan, "V2b-AA-0001", STATES)[0] == 3  # already reserved
    assert pin_next(world, nl, ex, plan, "V2b-AA-0099", STATES)[0] == 3  # not in the plans
    down = {n: "down*" for n in NODES5}
    rc, rec = pin_next(world, nl, ex, plan, "V2b-AA-0002", down)
    assert rc == 3 and rec["chosen"] is None and len(rec["skipped"]) == 5


def test_new_anchored_patterns_refuse_a_trailing_newline():
    """NODE_RE, ITEM_ID_RE and the --cluster-utc-offset pattern end in \\Z: `$` would also
    match before a final newline."""
    from capcommon import jobplan
    from capjobs import budget
    assert nodes.NODE_RE.match("node1") and not nodes.NODE_RE.match("node1\n")
    assert jobplan.ITEM_ID_RE.match("i1") and not jobplan.ITEM_ID_RE.match("i1\n")
    assert budget.parse_offset("+08:00") == 8 * 3600
    with pytest.raises(budget.GateError):
        budget.parse_offset("+08:00\n")
    with pytest.raises(nodes.NodesError):
        nodes.expand_hostlist("node1\n")


def test_usable_state_and_hostlists():
    """An allow-list of sinfo %t states: idle / mix / alloc / comp, bare or with the backfill
    "-" suffix; any other base state or flag suffix is unusable."""
    for s in ("idle", "mix", "mixed", "alloc", "allocated", "comp", "completing", "mix-",
              "IDLE", "idle-"):
        assert nodes.usable_state(s), s
    for s in ("down", "down*", "drain", "drng", "fail~", "idle*", "mix*", "idle~", "idle#",
              "idle!", "idle%", "idle$", "idle@", "idle^", "mix-*", "maint", "resv", "unk",
              "unk*", "inval", "boot", "npc", "plnd", "pow_dn", "future", "", None,
              "idle--"):
        assert not nodes.usable_state(s), s
    rot = ["a", "b", "c"]
    assert nodes.pin_choice(rot, None, {}, {"a": "idle*", "b": "maint", "c": "mix-"})[0] == "c"
    assert nodes.expand_hostlist("umbriel-b200-[001-003,026],x1") == [
        "umbriel-b200-001", "umbriel-b200-002", "umbriel-b200-003", "umbriel-b200-026", "x1"]
    for bad in ("a-[3-1]", "a-[1-2", "a-[x]"):
        with pytest.raises(nodes.NodesError):
            nodes.expand_hostlist(bad)


# ---- --expect-node in the job (test 24) ----------------------------------------------------------

def test_job_check_fails_on_another_node(tmp_path, monkeypatch):
    """The pinned name is a Slurm node name (sinfo, pin-next): the check compares --node
    (SLURMD_NODENAME), never the host name, which is only recorded."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / snapshot.MANIFEST).write_text("{}")
    monkeypatch.setattr(envfiles.snapshot, "verify", lambda s: (True, []))
    monkeypatch.setattr(socket, "gethostname", lambda: "b200-host-12.cluster")
    ok = envfiles.check_job(str(snap), "P1", "umbriel-b200-012", 1, "umbriel-b200-012")
    assert ok["passed"] and ok["expect_node_ok"] and ok["hostname_short"] == "b200-host-12"
    assert envfiles.check_job(str(snap), "P1", "unknown", 1, "")["passed"]  # not pinned
    bad = envfiles.check_job(str(snap), "P1", "umbriel-b200-012", 1, "umbriel-b200-013")
    assert not bad["passed"] and bad["outcome"] == "FAIL"
    assert "expected node umbriel-b200-013" in bad["snapshot_problems"][-1]
    assert "node umbriel-b200-012 (host b200-host-12)" in bad["snapshot_problems"][-1]
    monkeypatch.setattr(socket, "gethostname", lambda: "umbriel-b200-013")
    assert not envfiles.check_job(str(snap), "P1", "umbriel-b200-012", 1,
                                  "umbriel-b200-013")["passed"]  # the host name is not read
    for unknown in ("unknown", "", None):
        bad = envfiles.check_job(str(snap), "P1", unknown, 1, "umbriel-b200-013")
        assert not bad["passed"] and not bad["expect_node_ok"], unknown
    # the placeholder of an unset SLURMD_NODENAME never satisfies a pin, even one so named
    assert not envfiles.check_job(str(snap), "P1", "unknown", 1, "unknown")["passed"]
    ident = ["--experiment", "E0", "--arm", "job", "--launch-seq", "0", "--job-id", "1",
             "--restart-count", "0", "--register-sha", "a" * 64, "--role", "jobs"]
    out = tmp_path / "job.json"
    assert envfiles.main(["job", "--snap", str(snap), "--plan-id", "P1", "--node", "n",
                          "--deadline-epoch", "1", "--expect-node", "umbriel-b200-013",
                          "--out", str(out)] + ident) == 6
    assert json.load(open(out))["expect_node"] == "umbriel-b200-013"
