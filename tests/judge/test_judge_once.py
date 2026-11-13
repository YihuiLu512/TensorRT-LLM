# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tree is judged once: only a run that produced verdicts creates <tree>/judge (and the
JUDGED.tsv row). INPUT_INVALID and exceptions leave a reviewable judge.attempts/*/ record and
no judge/, so the tree can be judged after the cause is fixed; a second judgement is refused."""

import json
import os
import sys

import pytest
from conftest import SAMPLES, judge_mod

J = judge_mod("judge")


@pytest.fixture
def good(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPKIT_CODE_SHA", "b" * 40)
    sys.path.insert(0, SAMPLES)
    try:
        import make_samples
    finally:
        sys.path.remove(SAMPLES)
    W = make_samples.build(str(tmp_path), "good", {}, make_samples.SAMPLES["good"][1])
    with open(os.path.join(W, "expect.json")) as f:
        tree = os.path.join(W, json.load(f)["tree"])
    return W, tree


def run(W, tree):
    return J.main(["--tree", tree, "--workdir", W])


def attempts(tree):
    d = os.path.join(tree, J.ATTEMPTS_DIR)
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def attempt_doc(tree, name):
    with open(os.path.join(tree, J.ATTEMPTS_DIR, name, "attempt.json")) as f:
        return json.load(f)


def judged_rows(W):
    with open(os.path.join(W, "JUDGED.tsv")) as f:
        return [ln for ln in f.read().splitlines()[1:] if ln.strip()]


def test_input_invalid_leaves_a_record_and_the_tree_stays_judgeable(good):
    W, tree = good
    sub = os.path.join(W, "submissions.tsv")
    os.rename(sub, sub + ".aside")
    assert run(W, tree) == 13
    assert not os.path.lexists(os.path.join(tree, "judge"))
    (a,) = attempts(tree)
    assert attempt_doc(tree, a)["status"] == "INPUT_INVALID"
    with open(os.path.join(tree, J.ATTEMPTS_DIR, a, "inputs.json")) as f:
        assert json.load(f)["status"] == "INPUT_INVALID"
    assert os.path.getsize(os.path.join(tree, J.ATTEMPTS_DIR, a, "judge.log")) > 0
    os.rename(sub + ".aside", sub)
    assert run(W, tree) == 0
    with open(os.path.join(tree, "judge", "attempt.json")) as f:
        assert json.load(f)["status"] == "JUDGED"
    assert os.path.isfile(os.path.join(tree, "judge", "verdicts.json"))
    assert len(judged_rows(W)) == 1
    assert attempts(tree) == [a]  # the successful attempt became judge/


def test_second_judgement_is_refused_and_writes_nothing(good):
    W, tree = good
    assert run(W, tree) == 0
    before = (sorted(os.listdir(os.path.join(tree, "judge"))), attempts(tree), judged_rows(W))
    assert run(W, tree) == 12
    after = (sorted(os.listdir(os.path.join(tree, "judge"))), attempts(tree), judged_rows(W))
    assert before == after


def test_exception_leaves_an_error_record_and_no_judge_dir(good, monkeypatch):
    W, tree = good

    def boom(*_a, **_k):
        raise RuntimeError("boom in judge_tree")

    monkeypatch.setattr(J, "judge_tree", boom)
    with pytest.raises(RuntimeError):
        run(W, tree)
    assert not os.path.lexists(os.path.join(tree, "judge"))
    (a,) = attempts(tree)
    doc = attempt_doc(tree, a)
    assert doc["status"] == "ERROR" and doc["exit_code"] == 1 and "boom" in doc["error"]
    assert judged_rows(W) == []
    monkeypatch.undo()
    monkeypatch.setenv("CAPKIT_CODE_SHA", "b" * 40)
    assert run(W, tree) == 0


def test_judge_dir_created_meanwhile_loses_the_race(good, monkeypatch):
    W, tree = good
    real_verify = J.I.verify
    jdir = os.path.join(tree, "judge")

    def verify_then_race(w, t):
        ctx = real_verify(w, t)
        os.mkdir(jdir)
        with open(os.path.join(jdir, "other.txt"), "w") as f:
            f.write("written by another judge run\n")
        return ctx

    monkeypatch.setattr(J.I, "verify", verify_then_race)
    assert run(W, tree) == 12
    assert os.listdir(jdir) == ["other.txt"]
    (a,) = attempts(tree)
    assert attempt_doc(tree, a)["status"] == "ALREADY_JUDGED"
    assert judged_rows(W) == []


def test_ledger_row_added_meanwhile_loses_the_race(good, monkeypatch):
    W, tree = good
    real_verify = J.I.verify

    def verify_then_ledger(w, t):
        ctx = real_verify(w, t)
        J.ledger.judged_append(os.path.join(w, "JUDGED.tsv"), {
            "job_id": ctx.ident.job_id, "restart_count": ctx.ident.restart_count,
            "launch_seq": ctx.ident.launch_seq, "arm": ctx.ident.arm,
            "experiment": ctx.ident.experiment, "tree_relpath": "elsewhere",
            "verdicts_sha256": "0" * 64})
        return ctx

    monkeypatch.setattr(J.I, "verify", verify_then_ledger)
    assert run(W, tree) == 12
    assert not os.path.lexists(os.path.join(tree, "judge"))
    (a,) = attempts(tree)
    assert attempt_doc(tree, a)["status"] == "ALREADY_JUDGED"
