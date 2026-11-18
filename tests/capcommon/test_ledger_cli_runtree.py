# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os

import pytest

from capcommon import cli, ledger, policy, runtree
from capcommon.exitcodes import ConfigError

from conftest import SHA

B = "b" * 64


def test_registry(tmp_path):
    p = str(tmp_path / "REGISTRY.tsv")
    row = ledger.registry_append(p, "E0", "02_e0/E0/register/register.json", B,
                                 "kind=register;first", registered_at="2026-11-13T00:00:00.000Z")
    assert ledger.registry_lookup(p, B) == row
    assert ledger.registry_kind(row) == "register"
    assert ledger.registered_before(row, "2026-11-13T00:00:00.001Z")  # same second, 1 ms later
    assert not ledger.registered_before(row, "2026-11-13T00:00:00.000Z")
    with pytest.raises(ValueError):  # second-precision stamps are not accepted any more
        ledger.registered_before(row, "2026-11-13T00:00:01Z")
    with pytest.raises(ledger.LedgerError):
        ledger.registry_append(p, "E0", "y", "c" * 64, "kind=pilot;", registered_at="2026-11-13")
    with pytest.raises(ledger.LedgerError):  # duplicate sha
        ledger.registry_append(p, "E0", "x", B, "kind=register;")
    for bad in [("E0", "x", "B" * 64, "kind=register;"), ("E0", "x", SHA, "register"),
                ("E0", "x", SHA, "kind=nope;"), ("E0", "/abs", SHA, "kind=pilot;"),
                ("E0", "x\ty", SHA, "kind=pilot;")]:
        with pytest.raises(ledger.LedgerError):
            ledger.registry_append(p, *bad)
    assert ledger.registry_lookup(p, SHA) is None
    lines = open(p).read().splitlines()
    assert lines[0] == "\t".join(ledger.REGISTRY_COLUMNS) and len(lines) == 2


CALIB_V2_KINDS = ("errata", "ladder", "e0_params", "e0_summary", "d14_result", "e0_list",
                  "job_plan", "node_exclusions", "node_list")


def test_registry_takes_the_calibration_v2_kinds(tmp_path):
    # calib-v2 DESIGN-r7a P2-02: the kinds E0 v2 registers
    p = str(tmp_path / "REGISTRY.tsv")
    for i, kind in enumerate(CALIB_V2_KINDS):
        sha = "%064x" % (i + 1)
        row = ledger.registry_append(p, "E0", "02_e0/E0/register/v2/%s" % kind, sha,
                                     "kind=%s;calib-v2" % kind,
                                     registered_at="2026-11-15T00:00:00.%03dZ" % i)
        assert ledger.registry_kind(row) == kind
    with pytest.raises(ledger.LedgerError, match="unknown registry kind"):
        ledger.registry_append(p, "E0", "x", "f" * 64, "kind=ladders;")


def _sub(**kw):
    row = {c: "" for c in ledger.SUBMISSION_COLUMNS}
    row.pop("event_at")
    row.update(kw)
    return row


_EVIDENCE = {"mounts_json": json.dumps(["/home/scratch.x/w:/w"]),
             "tokenizer_dir_sha256": "none",
             "assertions_json": json.dumps({k: True for k in policy.ASSERTION_KEYS})}


def test_submissions(tmp_path):
    p = str(tmp_path / "submissions.tsv")
    ledger.submissions_append(p, _sub(plan_id="p1", event="PLANNED", experiment="E0", gpus="0",
                                      basis="STATUS.md line 12: user approved E0 on 11-13",
                                      sbatch_argv=json.dumps(["sbatch", "--parsable"]),
                                      **_EVIDENCE))
    ledger.submissions_append(p, _sub(plan_id="p1", event="SUBMITTED", job_id="5601234",
                                      submitted_at="2026-11-13T01:00:00.000Z"))
    ledger.submissions_append(p, _sub(plan_id="p1", event="ENDED", status="VALID"))
    cur = ledger.submissions_current(p)["p1"]
    assert cur["job_id"] == "5601234" and cur["status"] == "VALID" and cur["experiment"] == "E0"
    assert ledger.submission_for_job(p, "5601234")["plan_id"] == "p1"
    assert ledger.submission_for_job(p, "999") is None
    bad_assert = dict(_EVIDENCE, assertions_json=json.dumps(
        dict(json.loads(_EVIDENCE["assertions_json"]), gpus_le_1=False)))
    for bad in [_sub(plan_id="p2", event="PLANNED", gpus="0", experiment="E0", **_EVIDENCE),
                _sub(plan_id="p2", event="PLANNED", gpus="2", basis="x", experiment="E0",
                     **_EVIDENCE),
                _sub(plan_id="p2", event="PLANNED", gpus="0", basis="x", experiment="E0"),
                _sub(plan_id="p2", event="PLANNED", gpus="0", basis="x", experiment="E11",
                     **_EVIDENCE),
                _sub(plan_id="p2", event="PLANNED", gpus="0", basis="x", experiment="E0",
                     **bad_assert),
                _sub(plan_id="p2", event="PLANNED", gpus="0", basis="x", experiment="E0",
                     **dict(_EVIDENCE, mounts_json=json.dumps(["/home:/home"]))),
                _sub(plan_id="p2", event="SUBMITTED", submitted_at="2026-11-13T01:00:00Z"),
                _sub(plan_id="p2", event="BOGUS"),
                _sub(plan_id="p2", event="ENDED", status="DONE"),
                _sub(plan_id="p2", event="NOTE", sbatch_argv="not json")]:
        with pytest.raises(ledger.LedgerError):
            ledger.submissions_append(p, bad)
    with pytest.raises(ledger.LedgerError):
        ledger.append_tsv(p, ledger.SUBMISSION_COLUMNS, {"x": 1})


def test_unknown_submit_status(tmp_path):
    """UNKNOWN_SUBMIT is a formal status: only on a NOTE row with submitted_at, and a later
    SUBMITTED (reconciled) or ENDED row supersedes it."""
    p = str(tmp_path / "submissions.tsv")
    planned = _sub(plan_id="p1", event="PLANNED", experiment="E0", gpus="1", basis="x",
                   **_EVIDENCE)
    ledger.submissions_append(p, planned)
    for bad in [_sub(plan_id="p1", event="NOTE", status="UNKNOWN_SUBMIT"),
                _sub(plan_id="p1", event="ENDED", status="UNKNOWN_SUBMIT",
                     submitted_at="2026-11-13T01:00:00.000Z"),
                _sub(plan_id="p1", event="SUBMITTED", status="UNKNOWN_SUBMIT", job_id="7",
                     submitted_at="2026-11-13T01:00:00.000Z")]:
        with pytest.raises(ledger.LedgerError):
            ledger.submissions_append(p, bad)
    ledger.submissions_append(p, _sub(plan_id="p1", event="NOTE", status="UNKNOWN_SUBMIT",
                                      submitted_at="2026-11-13T01:00:00.000Z"))
    assert ledger.submission_unknown(ledger.submissions_current(p)["p1"])
    ledger.submissions_append(p, _sub(plan_id="p1", event="SUBMITTED", job_id="7",
                                      submitted_at="2026-11-13T01:00:00.000Z"))
    cur = ledger.submissions_current(p)["p1"]
    assert cur["status"] == "" and not ledger.submission_unknown(cur)
    ledger.submissions_append(p, planned | {"plan_id": "p2"})
    ledger.submissions_append(p, _sub(plan_id="p2", event="NOTE", status="UNKNOWN_SUBMIT",
                                      submitted_at="2026-11-13T01:00:00.000Z"))
    ledger.submissions_append(p, _sub(plan_id="p2", event="ENDED", status="SUBMIT_FAILED"))
    assert ledger.submissions_current(p)["p2"]["status"] == "SUBMIT_FAILED"


def test_judged(tmp_path):
    p = str(tmp_path / "JUDGED.tsv")
    row = {"job_id": "1", "restart_count": 0, "launch_seq": 2, "arm": "a", "experiment": "E0",
           "tree_relpath": "x", "verdicts_sha256": B}
    ledger.judged_append(p, row)
    assert ledger.judged_lookup(p, "1", 0, 2, "a") is not None
    with pytest.raises(ledger.LedgerError):
        ledger.judged_append(p, row)
    other = tmp_path / "other.tsv"
    other.write_text("wrong\theader\n")
    with pytest.raises(ledger.LedgerError):
        ledger.read_tsv(str(other), ledger.JUDGED_COLUMNS)


def _parser(**kw):
    p = argparse.ArgumentParser()
    cli.add_common_args(p, **kw)
    return p


BASE = ["--experiment", "E0", "--arm", "a1", "--launch-seq", "0", "--job-id", "local-x",
        "--restart-count", "0", "--register-sha", SHA, "--role", "capload", "--cpus", "0-1,4",
        "--stop-grace-s", "1.5"]


def test_cli_common(tmp_path, table_path):
    tp = table_path()
    p = _parser(loop=True, tiers=True, snapshot=True, ready=True)
    args = p.parse_args(BASE + ["--run-dir", str(tmp_path), "--loop", "asyncio",
                                "--tier-table", tp, "--origin-ns", "123", "--shm-dir",
                                str(tmp_path), "--snapshot-interval-ms", "100", "--ready-file",
                                str(tmp_path / "r.json"), "--inject",
                                "site=send.req,mode=busy,delta_ns=10"])
    c = cli.common_from_args(args)
    assert c.cpus == [0, 1, 4] and c.snapshot_interval_ns == 100_000_000
    assert c.identity.role == "capload" and c.identity.pid == os.getpid()
    assert c.inject_specs[0].site == "send.req" and c.load_tier_table().sha256
    with pytest.raises(SystemExit) as e:  # missing --loop
        p.parse_args(BASE + ["--run-dir", str(tmp_path)])
    assert e.value.code == 2
    bad = p.parse_args(BASE + ["--run-dir", str(tmp_path / "nope"), "--loop", "asyncio",
                               "--tier-table", tp, "--origin-ns", "1", "--shm-dir", "x",
                               "--snapshot-interval-ms", "1", "--ready-file", "r"])
    with pytest.raises(ConfigError):
        cli.common_from_args(bad)
    bad2 = _parser().parse_args(BASE[:-6] + ["--register-sha", "x" * 64, "--role", "capload",
                                             "--cpus", "1", "--stop-grace-s", "1", "--run-dir",
                                             str(tmp_path)])
    with pytest.raises(ConfigError):
        cli.common_from_args(bad2)
    for text in ("1-", "a", "3-1", "1,1"):
        with pytest.raises(ConfigError):
            cli.parse_cpu_list(text)
    assert cli.format_cpu_list([1, 2]) == "1,2"


def test_runtree(tmp_path, ident):
    assert runtree.tree_dir_name("5601234", 0, 3, "nullsse-a") == "J5601234-r0-L3_nullsse-a"
    assert runtree.tier_dir_name(3, "up-r2000") == "003_up-r2000"
    assert runtree.tier_dir_name(-1, "pre") == "pre" and runtree.tier_dir_name(-2, "x") == "post"
    t = runtree.RunTree(str(tmp_path / "T")).make_skeleton()
    assert os.path.isdir(t.procs("ready")) and os.path.isdir(t.control_dir)
    assert t.tier_file(3, "up", "capload", 1, "requests", "capbin").endswith(
        os.path.join("raw", "tiers", "003_up", "capload.w1.requests.capbin"))
    assert t.ready_file("capload", 2).endswith("capload.w2.json")
    assert runtree.read_state(t.root) == []
    runtree.append_state(t.root, "VALID", "copied back and verified", "inner.sh")
    runtree.append_state(t.root, "VOID", "judge: RED abc", "capjudge")
    st = runtree.read_state(t.root)
    assert [s[1] for s in st] == ["VALID", "VOID"]
    with pytest.raises(ConfigError):
        runtree.append_state(t.root, "OK", "x", "y")
    with pytest.raises(ConfigError):
        t.log_file("capload", 0, 1, "stdout")


def test_state_severity_is_one_order():
    """RS15: the launch-level severity order lives only in runtree; every STATE value has a rank,
    and worst_state never returns a better status than any of its inputs."""
    assert runtree.STATE_SEVERITY == ("VALID", "NOT_RUN", "VOID", "ENV_INVALID")
    assert set(runtree.STATE_SEVERITY) == set(runtree.STATE_VALUES)
    vals = runtree.STATE_SEVERITY
    for a in vals:
        for b in vals:
            w = runtree.worst_state([a, b])
            assert runtree.state_rank(w) == max(runtree.state_rank(a), runtree.state_rank(b))
    assert runtree.worst_state(iter(["NOT_RUN", "VALID"])) == "NOT_RUN"
    for bad in ([], ["VALID", "PASS"], [None]):
        with pytest.raises(ConfigError):
            runtree.worst_state(bad)


def test_no_second_state_order_in_the_tools():
    """RS15: no tool keeps its own copy of the STATE ranking; capjudge and capjobs.launches use
    runtree.worst_state."""
    from capjobs import launches
    from capjudge import inputs, judge
    assert launches.worst_state is runtree.worst_state
    assert judge.worst_state is runtree.worst_state
    assert inputs.worst_state is runtree.worst_state
    assert not hasattr(inputs, "STATE_RANK")


def test_registry_takes_the_e12_kinds(tmp_path):
    p = str(tmp_path / "REGISTRY.tsv")
    for i, kind in enumerate(("e12_config", "e12_runplan")):
        row = ledger.registry_append(p, "E12-L2a", "07_gpu/E12/x/%s" % kind, "%064x" % (i + 1),
                                     "kind=%s;" % kind)
        assert ledger.registry_kind(row) == kind
    with pytest.raises(ledger.LedgerError, match="unknown registry kind"):
        ledger.registry_append(p, "E12-L2a", "07_gpu/E12/x/y", "%064x" % 9, "kind=e12_other;")
