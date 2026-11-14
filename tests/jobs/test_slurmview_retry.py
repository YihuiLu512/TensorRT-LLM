# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""slurm-view retries each front-end query a bounded number of times and writes
env/slurm_view.json only when every query that ran succeeded (SEAMS SM-17: one ssh rc 255 used to
be written as-is and exit 0, so the caller never retried and the gate read item rc 255)."""

import json
import math
import os

import pytest

from capenv.gate import SlurmSource
from capjobs import slurmview, submit
from capcommon.exitcodes import ConfigError

JOB = "JobId=77 JobName=x\n   NodeList=n1 BatchHost=n1\n"
KEX = "kex_exchange_identification: Connection closed by remote host\r\n"


def answers(squeue_rcs, job_rcs=None, node_rcs=None):
    """scontrol show job -> job_rcs in turn, show node -> node_rcs in turn (both default always
    0), squeue -> squeue_rcs in turn (the last one repeats)."""
    seq = {"job": list(job_rcs or [0]), "node": list(node_rcs or [0]),
           "squeue": list(squeue_rcs)}

    def take(name):
        rc = seq[name][0]
        if len(seq[name]) > 1:
            seq[name].pop(0)
        return rc

    def scontrol(argv):
        if " job " in argv[-1]:
            rc = take("job")
            return (rc, JOB if rc == 0 else "", "" if rc == 0 else KEX)
        rc = take("node")
        return (rc, "NodeName=n1 SystemProductName=X\n" if rc == 0 else "",
                "" if rc == 0 else KEX)

    def squeue(argv):
        rc = take("squeue")
        return (rc, "77 me RUNNING\n" if rc == 0 else "", "" if rc == 0 else KEX)

    return {"scontrol": scontrol, "squeue": squeue}


class Sleeps(list):
    def __call__(self, s):
        self.append(s)


def collect(stub_runner, squeue_rcs, job_rcs=None, attempts=3, wait=2.0, node_rcs=None):
    r = stub_runner(answers(squeue_rcs, job_rcs, node_rcs))
    sl = Sleeps()
    doc = slurmview.collect(r, "fe", "77", attempts, wait, sleep=sl)
    return doc, r, sl


# ---- slurmview.collect -------------------------------------------------------------------------

def test_transient_failure_is_retried_and_recorded(stub_runner):
    doc, r, sl = collect(stub_runner, [255, 0])
    assert [i["rc"] for i in doc["items"]] == [0, 0, 0]
    assert [i["attempts"] for i in doc["items"]] == [1, 1, 2]
    assert doc["items"][2]["retried_rcs"] == [255] and doc["items"][2]["stderr"] == ""
    assert doc["items"][2]["retried_stderr"] == [KEX]
    assert sl == [2.0] and len(r.calls) == 4
    assert slurmview.failed_items(doc) == []


@pytest.mark.parametrize("rc", [255, None, 1])
def test_persistent_failure_is_bounded_and_named(stub_runner, rc):
    doc, r, sl = collect(stub_runner, [rc])
    it = doc["items"][2]
    assert it["rc"] == rc and it["attempts"] == 3 and it["retried_rcs"] == [rc, rc]
    assert it["retried_stderr"] == [KEX, KEX]
    assert sl == [2.0, 4.0] and len(r.calls) == 2 + 3
    assert slurmview.failed_items(doc) == [it]


def test_one_attempt_does_not_retry(stub_runner):
    doc, r, sl = collect(stub_runner, [255, 0], attempts=1)
    assert doc["items"][2]["rc"] == 255 and doc["items"][2]["attempts"] == 1
    assert sl == [] and len(r.calls) == 3
    assert [i["id"] for i in slurmview.failed_items(doc)] == [3]


def test_item1_failure_skips_items_2_3_and_only_item1_counts(stub_runner):
    doc, r, sl = collect(stub_runner, [0], job_rcs=[255])
    assert [i["rc"] for i in doc["items"]] == [255, None, None]
    assert [i["attempts"] for i in doc["items"]] == [3, 0, 0]
    assert [i["id"] for i in slurmview.failed_items(doc)] == [1]
    assert len(r.calls) == 3
    assert [i["stderr"] for i in doc["items"][1:]] == ["not run: item 1 still failed"] * 2


def test_item2_failure_stops_before_item3(stub_runner):
    """Once an item still fails the file will not be written: item 3 is not queried."""
    doc, r, sl = collect(stub_runner, [0], node_rcs=[255])
    assert [i["rc"] for i in doc["items"]] == [0, 255, None]
    assert [i["attempts"] for i in doc["items"]] == [1, 3, 0]
    assert [i["id"] for i in slurmview.failed_items(doc)] == [2]
    assert len(r.calls) == 1 + 3 and sl == [2.0, 4.0]
    it3 = doc["items"][2]
    assert it3["cmd"] == ["squeue", "-h", "-w", "n1", "-o", "%i %u %T"]
    assert it3["stderr"] == "not run: item 2 still failed"
    assert it3["retried_rcs"] == [] and it3["retried_stderr"] == []


def test_each_earlier_try_keeps_its_stderr(stub_runner):
    """A timeout (rc null) and an ssh rc 255 stay distinguishable in the retried tries."""
    tries = [(None, "", "TimeoutExpired: Command '['ssh']' timed out after 60 seconds"),
             (255, "", KEX), (0, "77 me RUNNING\n", "")]
    r = stub_runner({"scontrol": answers([0])["scontrol"],
                     "squeue": lambda argv: tries.pop(0)})
    doc = slurmview.collect(r, "fe", "77", 3, 0, sleep=Sleeps())
    it = doc["items"][2]
    assert it["rc"] == 0 and it["attempts"] == 3 and it["retried_rcs"] == [None, 255]
    assert it["retried_stderr"][0].startswith("TimeoutExpired:")
    assert it["retried_stderr"][1] == KEX


def test_item1_without_single_node_is_not_a_failed_query(stub_runner):
    r = stub_runner({"scontrol": (0, "JobId=77 NodeList=(null)\n", "")})
    doc = slurmview.collect(r, "fe", "77", 3, 0, sleep=Sleeps())
    assert [i["rc"] for i in doc["items"]] == [0, None, None]
    assert slurmview.failed_items(doc) == []
    assert len(r.calls) == 1


@pytest.mark.parametrize("attempts,wait", [(0, 1.0), (11, 1.0), (True, 1.0), (2.0, 1.0),
                                           (3, -0.1), (3, 60.1), (3, math.nan), (3, True)])
def test_retry_settings_refused(stub_runner, attempts, wait):
    r = stub_runner({})
    with pytest.raises(ConfigError):
        slurmview.collect(r, "fe", "77", attempts, wait)
    assert r.calls == []


# ---- capjobs.submit slurm-view -----------------------------------------------------------------

def cli(stub_runner, jd, squeue_rcs, *extra, job_rcs=None):
    r = stub_runner(answers(squeue_rcs, job_rcs))
    argv = ["slurm-view", "--login-host", "fe", "--ssh-timeout-s", "60", "--job-id", "77",
            "--job-dir", jd, "--retry-wait-s", "0"] + list(extra)
    return submit.main(argv, runner=r), r


def test_cli_still_failing_writes_nothing_and_exits_4(tmp_path, stub_runner, capsys):
    jd = str(tmp_path / "J77-r0")
    os.makedirs(jd)
    rc, r = cli(stub_runner, jd, [255])
    assert rc == 4
    assert not os.path.exists(os.path.join(jd, "env", "slurm_view.json"))
    assert os.listdir(jd) == []
    cap = capsys.readouterr()
    out = json.loads(cap.out)
    assert out["path"] is None and out["rcs"] == [0, 0, 255] and out["attempts"] == [1, 1, 3]
    assert out["failed"] == [{"id": 3, "cmd": ["squeue", "-h", "-w", "n1", "-o", "%i %u %T"],
                              "rc": 255, "stderr": KEX, "retried_rcs": [255, 255],
                              "retried_stderr": [KEX, KEX]}]
    assert "not written" in cap.err


def test_cli_transient_failure_writes_the_view(tmp_path, stub_runner, capsys):
    jd = str(tmp_path / "J77-r0")
    os.makedirs(jd)
    rc, _r = cli(stub_runner, jd, [255, 0])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    path = os.path.join(jd, "env", "slurm_view.json")
    assert out["path"] == path and out["rcs"] == [0, 0, 0] and "failed" not in out
    doc = json.load(open(path))
    assert doc["items"][2]["attempts"] == 2 and doc["items"][2]["retried_rcs"] == [255]


def test_cli_attempts_flag_reaches_collect(tmp_path, stub_runner, capsys):
    jd = str(tmp_path / "J77-r0")
    os.makedirs(jd)
    rc, r = cli(stub_runner, jd, [255, 255, 255, 255, 0], "--attempts", "5")
    assert rc == 0 and len(r.calls) == 2 + 5
    rc, r = cli(stub_runner, jd, [255], "--attempts", "0")
    assert rc == 3 and r.calls == []


def test_cli_missing_job_dir_makes_no_ssh_call(tmp_path, stub_runner, capsys):
    rc, r = cli(stub_runner, str(tmp_path / "missing"), [0])
    assert rc == 3 and r.calls == []
    assert "does not exist (the job creates it)" in capsys.readouterr().err


def test_cli_item2_failure_names_only_item2(tmp_path, stub_runner, capsys):
    jd = str(tmp_path / "J77-r0")
    os.makedirs(jd)
    r = stub_runner(answers([0], node_rcs=[255]))
    argv = ["slurm-view", "--login-host", "fe", "--ssh-timeout-s", "60", "--job-id", "77",
            "--job-dir", jd, "--retry-wait-s", "0"]
    assert submit.main(argv, runner=r) == 4 and len(r.calls) == 1 + 3
    out = json.loads(capsys.readouterr().out)
    assert out["rcs"] == [0, 255, None] and out["attempts"] == [1, 3, 0]
    assert [f["id"] for f in out["failed"]] == [2] and os.listdir(jd) == []


# ---- the gate reads the written view as before -------------------------------------------------

def test_gate_reads_a_retried_view(tmp_path, stub_runner):
    doc, _r, _sl = collect(stub_runner, [255, 0])
    jd = tmp_path / "J77-r0"
    jd.mkdir()
    path = slurmview.write(str(jd), doc)

    def container_fails(argv, timeout_s):
        return {"cmd": argv, "rc": None, "stdout": None, "reason": "not found"}

    src = SlurmSource(path, 1.0, container_fails, 5.0)
    out, rc, cmd, source, why = src.get(3, ["squeue", "-h", "-w", "n1", "-o", "%i %u %T"])
    assert (out, rc, source, why) == ("77 me RUNNING\n", 0, "slurm_view", None)
    assert cmd == ["squeue", "-h", "-w", "n1", "-o", "%i %u %T"]
