# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tree is judged once: only a run that produced verdicts creates <tree>/judge (and the
JUDGED.tsv row). INPUT_INVALID and exceptions leave a reviewable judge.attempts/*/ record and
no judge/, so the tree can be judged after the cause is fixed; a second judgement is refused,
except that a run which died after creating judge/ gets its ledger row / STATE line completed
from judge/. The judge's STATE line never improves on the conclusion recorded before it."""

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


# ---- the launch-level conclusion recorded before the judge (STATE, launch_end.json) ----------

def state_lines(tree):
    with open(os.path.join(tree, "STATE")) as f:
        return [ln.split("\t") for ln in f.read().splitlines() if ln.strip()]


def test_judge_never_improves_on_an_earlier_void(good):
    W, tree = good
    J.append_state(tree, "VOID", "copy-back not verified (test)", "capjobs.launches")
    assert run(W, tree) == 10
    last = state_lines(tree)[-1]
    assert last[1:3] == ["VOID", "capjudge"] and "upstream=VOID" in last[3]
    with open(os.path.join(tree, "judge", "verdicts.json")) as f:
        v = json.load(f)
    assert v["launch_checks"]["LAUNCH-STATE"] == "RED" and v["launch_verdict"] == "VOID"
    assert {t["verdict"] for t in v["tiers"] if t["judged"]} == {"VOID"}


def test_env_invalid_before_the_judge_stays_env_invalid(good):
    W, tree = good
    J.append_state(tree, "ENV_INVALID", "gate failed (test)", "capjobs.launches")
    assert run(W, tree) == 10
    assert state_lines(tree)[-1][1] == "ENV_INVALID"


def test_upstream_not_run_makes_every_tier_not_run(good):
    W, tree = good
    J.append_state(tree, "NOT_RUN", "terminated (test)", "capjobs.launches")
    assert run(W, tree) == 11
    assert state_lines(tree)[-1][1] == "NOT_RUN"


def test_leftovers_in_launch_end_void_the_launch(good):
    W, tree = good
    p = os.path.join(tree, "raw", "launch_end.json")
    with open(p) as f:
        doc = json.load(f)
    doc["leftovers"]["after"] = [[4242, 1, "survived"]]
    os.chmod(p, 0o644)
    with open(p, "w") as f:
        json.dump(doc, f)
    assert run(W, tree) == 10
    assert state_lines(tree)[-1][1] == "VOID"


def test_no_state_line_before_the_judge_is_input_invalid(good):
    W, tree = good
    os.rename(os.path.join(tree, "STATE"), os.path.join(W, "STATE.aside"))
    assert run(W, tree) == 13
    assert not os.path.lexists(os.path.join(tree, "judge"))
    os.rename(os.path.join(W, "STATE.aside"), os.path.join(tree, "STATE"))
    assert run(W, tree) == 0


# ---- judge/ exists but the run died before its ledger row / STATE line -------------------------

def _die_in(monkeypatch, where):
    def boom(*_a, **_k):
        raise OSError("disk gone while appending (%r)" % (where,))
    monkeypatch.setattr(*where, boom)


@pytest.mark.parametrize("where", ["state", "ledger"])
def test_commit_interrupted_after_the_rename_is_completed_by_the_next_run(good, monkeypatch,
                                                                         where):
    W, tree = good
    n_state = len(state_lines(tree))
    target = (J, "append_state") if where == "state" else (J.ledger, "judged_append")
    _die_in(monkeypatch, target)
    with pytest.raises(OSError):
        run(W, tree)
    monkeypatch.undo()
    monkeypatch.setenv("CAPKIT_CODE_SHA", "b" * 40)
    assert os.path.isdir(os.path.join(tree, "judge"))
    assert len(state_lines(tree)) == n_state  # no capjudge line yet
    assert len(judged_rows(W)) == (0 if where == "ledger" else 1)
    with open(os.path.join(tree, "judge", "verdicts.json")) as f:
        recorded = json.load(f)
    before = sorted(os.listdir(os.path.join(tree, "judge")))
    assert run(W, tree) == recorded["exit_code"] == 0  # completed, not re-judged
    assert sorted(os.listdir(os.path.join(tree, "judge"))) == before
    assert len(judged_rows(W)) == 1
    lines = state_lines(tree)
    assert len(lines) == n_state + 1 and lines[-1][2] == "capjudge"
    with open(os.path.join(tree, "judge", "attempt.json")) as f:
        att = json.load(f)
    assert att["status"] == "JUDGED" and att["commit_completed"]["appended"]
    assert run(W, tree) == 12  # now complete: refused, nothing appended
    assert len(state_lines(tree)) == n_state + 1 and len(judged_rows(W)) == 1


def test_commit_of_a_live_writer_is_not_completed(good, monkeypatch):
    W, tree = good
    assert run(W, tree) == 0
    # simulate "renamed, still committing": drop the STATE line, mark the writer as this process
    with open(os.path.join(tree, "STATE")) as f:
        keep = [ln for ln in f if "\tcapjudge\t" not in ln]
    with open(os.path.join(tree, "STATE"), "w") as f:
        f.writelines(keep)
    p = os.path.join(tree, "judge", "attempt.json")
    with open(p) as f:
        att = json.load(f)
    att["pid"] = os.getpid()
    with open(p, "w") as f:
        json.dump(att, f)
    assert run(W, tree) == 12
    assert not any(ln[2] == "capjudge" for ln in state_lines(tree))
