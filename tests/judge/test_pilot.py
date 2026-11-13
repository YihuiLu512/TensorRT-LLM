# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pilot builder: bands from judged sub-saturation tiers; RED tiers and trees recorded
non-VALID before the judge excluded; explicit inputs; D14 from the registered rule (idle window
per job run vs load tiers)."""

import json
import os
import subprocess
import sys

from conftest import C, PKG, SAMPLES, TOOLS

if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from capcommon import ledger  # noqa: E402
from capcommon.hashing import sha256_file  # noqa: E402
from capjudge import d14 as D14  # noqa: E402

ENV = dict(os.environ, PYTHONPATH=TOOLS, CAPKIT_CODE_SHA="b" * 40)


def _judged(tmp_path, name):
    out = tmp_path / "s"
    subprocess.run([sys.executable, os.path.join(SAMPLES, "make_samples.py"), "--out",
                    str(out), "--only", name], check=True, capture_output=True, timeout=300)
    W = out / name
    tree = W / json.loads((W / "expect.json").read_text())["tree"]
    subprocess.run([sys.executable, "-m", PKG, "--tree", str(tree), "--workdir", str(W)],
                   env=ENV, capture_output=True, timeout=300, check=False)
    return W, tree


RULE = {"schema": "capkit.d14_rule/1", "idle_softirq_frac_max": 0.002,
        "load_min_excess_frac": 0.005, "monotonic_slack_frac": 0.002, "min_trees": 1}
EARLY = "2000-01-01T00:00:00.000Z"  # before every sample submission


def _rule(W, doc=None, registered_at=EARLY, register=True, name="d14_rule.json"):
    p = W / "register" / name
    p.write_text(json.dumps(doc or RULE))
    if register:
        ledger.registry_append(str(W / "REGISTRY.tsv"), "E0", "register/" + name,
                               sha256_file(str(p)), "kind=d14_rule;test", registered_at)
    return p


def _pilot(tmp_path, W, lines, out_name="pilot.json", rule=None):
    lst = tmp_path / "inputs.tsv"
    lst.write_text("".join(lines))
    out = tmp_path / out_name
    rule = rule or W / "register" / "d14_rule.json"
    if not rule.exists():
        _rule(W)
    p = subprocess.run([sys.executable, os.path.join(C, "analysis", "pilot_build.py"),
                        "--inputs", str(lst), "--workdir", str(W), "--d14-rule", str(rule),
                        "--out", str(out)],
                       capture_output=True, text=True, timeout=300, check=False)
    return p, out


def test_pilot_from_good_sample(tmp_path):
    W, tree = _judged(tmp_path, "good")
    p, out = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 0, p.stderr
    d = json.loads(out.read_text())
    assert d["schema"] == "capkit.pilot/1"
    nb = d["noise_bands"]
    assert nb["T1:msg_lag_p50_ns:capload"]["n"] == 2
    assert nb["T3:send_lag_p99_ns:capload"]["n"] == 4  # two workers x two tiers
    # the baseline tier's own normalised ratio is not a sample
    assert nb["T2:delivery_ratio_norm:capload"]["n"] == 1
    assert d["window_ns"] is not None and d["inputs"][0]["tiers_used"] == [2, 4]
    # refuses to overwrite
    p2, _ = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p2.returncode == 3


def test_pilot_excludes_red_tier(tmp_path):
    W, tree = _judged(tmp_path, "c5_requests")
    p, out = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 0, p.stderr
    d = json.loads(out.read_text())
    assert d["inputs"][0]["tiers_used"] == [2]
    assert "4" in d["inputs"][0]["tiers_excluded"]


def test_pilot_rejects_glob_unjudged_and_missing_tiers(tmp_path):
    W, tree = _judged(tmp_path, "good")
    p, _ = _pilot(tmp_path, W, ["%s\n" % tree], "a.json")
    assert p.returncode == 3  # tier column is required
    p, _ = _pilot(tmp_path, W, ["%s/*\t2\n" % os.path.dirname(str(tree))], "b.json")
    assert p.returncode == 3
    p, _ = _pilot(tmp_path, W, ["%s\t1\n" % tree], "c.json")
    assert p.returncode == 3  # tier 1 is a drain tier


# ---- D14 (INTERFACES erratum R1b) ----------------------------------------------------------------

def _d14(out):
    return json.loads(out.read_text())["d14_softirq"]


def test_d14_true_from_idle_window_and_load_tiers(tmp_path):
    W, tree = _judged(tmp_path, "good")
    p, out = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 0, p.stderr
    d = _d14(out)
    assert d["rule"] == RULE and d["rule_file"]["relpath"] == "register/d14_rule.json"
    rec = d["readings"][os.path.relpath(str(tree), str(W))]
    assert rec["complete"] and d["n_trees_complete"] == 1
    r = rec["roles"]["nullsse"]  # the sample's only under_test role, on core 4
    assert r["cores"] == [4]
    assert r["idle"]["softirq_ticks"] == 0 and r["idle"]["total_ticks"] == 3000
    assert r["idle"]["source"] == "raw/env/exclusivity.json item 6"
    assert [x["tier_seq"] for x in r["load"]] == [2, 4]
    assert all(abs(x["softirq_frac"] - 0.01) < 1e-9 for x in r["load"])  # 1 tick of 100
    assert r["idle_close_to_zero"] is True and r["grows_with_load"] is True
    assert d["irq_softirq_is_experiment"] is True


def test_d14_false_when_idle_softirq_is_not_close_to_zero(tmp_path):
    W, tree = _judged(tmp_path, "good")
    ex = tree / "raw" / "env" / "exclusivity.json"
    doc = json.loads(ex.read_text())
    doc["items"][0]["detail"]["per_core"]["4"]["softirq_ticks"] = 300  # 10 % while idle
    ex.chmod(0o644)
    ex.write_text(json.dumps(doc))
    p, out = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 0, p.stderr
    d = _d14(out)
    assert d["irq_softirq_is_experiment"] is False
    r = d["readings"][os.path.relpath(str(tree), str(W))]["roles"]["nullsse"]
    assert r["idle_close_to_zero"] is False


def test_d14_no_conclusion_without_the_idle_reading(tmp_path):
    W, tree = _judged(tmp_path, "good")
    ex = tree / "raw" / "env" / "exclusivity.json"
    ex.rename(tmp_path / "moved_exclusivity.json")
    p, out = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 0, p.stderr
    d = _d14(out)
    assert "irq_softirq_is_experiment" not in d  # C13 stays NOT_RUN
    assert d["n_trees_complete"] == 0 and "min_trees" in d["conclusion_reason"]
    rec = d["readings"][os.path.relpath(str(tree), str(W))]
    assert not rec["complete"] and "exclusivity.json missing" in rec["why"]


def test_d14_rule_registered_after_the_data_gives_no_conclusion(tmp_path):
    W, tree = _judged(tmp_path, "good")
    rule = _rule(W, registered_at=None)  # now: after the sample's submission
    p, out = _pilot(tmp_path, W, ["%s\t2,4\n" % tree], rule=rule)
    assert p.returncode == 0, p.stderr
    d = _d14(out)
    assert "irq_softirq_is_experiment" not in d
    assert "not registered before" in d["conclusion_reason"]


def test_d14_rule_must_be_registered_and_well_formed(tmp_path):
    W, tree = _judged(tmp_path, "good")
    rule = _rule(W, register=False, name="unregistered.json")
    p, _ = _pilot(tmp_path, W, ["%s\t2,4\n" % tree], "a.json", rule=rule)
    assert p.returncode == 3 and "not registered" in p.stderr
    bad = dict(RULE, idle_softirq_frac_max=2)
    rule = _rule(W, doc=bad, name="bad.json")
    p, _ = _pilot(tmp_path, W, ["%s\t2,4\n" % tree], "b.json", rule=rule)
    assert p.returncode == 3 and "idle_softirq_frac_max" in p.stderr


def _role(idle, loads):
    return {"idle": {"softirq_frac": idle},
            "load": [{"softirq_frac": f, "load_order_key": k, "tier_seq": i}
                     for i, (k, f) in enumerate(loads)]}


def test_d14_rule_growth_and_monotonicity():
    assert D14.evaluate_role(_role(0.0, [(10, 0.01), (20, 0.02)]), RULE) == (True, True)
    # not above the idle reading by the registered excess
    assert D14.evaluate_role(_role(0.0, [(10, 0.004)]), RULE) == (True, False)
    # drops with load by more than the slack (order by load, not by tier seq)
    assert D14.evaluate_role(_role(0.0, [(20, 0.01), (10, 0.03)]), RULE) == (True, False)
    assert D14.evaluate_role(_role(0.0, [(20, 0.0285), (10, 0.03)]), RULE) == (True, True)
    assert D14.evaluate_role(_role(0.01, [(10, 0.05)]), RULE) == (False, True)


# ---- trees whose launch was recorded VOID / NOT_RUN before the judge never enter a band ------

def test_pilot_excludes_a_tree_recorded_void_before_the_judge(tmp_path):
    W, tree = _judged(tmp_path, "upstream_void")
    p, _ = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 3 and "no tier survived" in p.stderr


def test_pilot_excludes_a_tree_whose_state_turned_void_after_judging(tmp_path):
    W, tree = _judged(tmp_path, "good")
    with open(tree / "STATE", "a") as f:
        f.write("2026-11-13T00:10:00.000Z\tVOID\tcapjobs.launches\tlate copy-back mismatch\n")
    p, _ = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 3 and "no tier survived" in p.stderr


def test_pilot_refuses_verdicts_without_the_launch_state_check(tmp_path):
    W, tree = _judged(tmp_path, "good")
    vp = tree / "judge" / "verdicts.json"
    doc = json.loads(vp.read_text())
    del doc["launch_checks"]["LAUNCH-STATE"]
    vp.chmod(0o644)
    vp.write_text(json.dumps(doc))
    p, _ = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 3 and "LAUNCH-STATE" in p.stderr


# ---- D14: an offline core, and independent idle readings ---------------------------------------

def test_d14_offline_core_makes_the_tier_reading_incomplete_not_a_crash(tmp_path):
    W, tree = _judged(tmp_path, "good")
    cpus = tree / "raw" / "sidecar" / "cpus.jsonl"
    lines = cpus.read_text().splitlines()
    head, rows = lines[0], [json.loads(x) for x in lines[1:]]
    cols = json.loads(head)["columns"]
    i_t, i_c = cols.index("t_rel_ns"), cols.index("cpu")
    for r in rows:  # core 4 offline at t = 14 s (inside tier 4's window)
        if r[i_t] == 14 * 10 ** 9 and r[i_c] == 4:
            for k in ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"):
                r[cols.index(k)] = None
    cpus.chmod(0o644)
    cpus.write_text("\n".join([head] + [json.dumps(r) for r in rows]) + "\n")
    p, out = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 0, p.stderr
    r = _d14(out)["readings"][os.path.relpath(str(tree), str(W))]["roles"]["nullsse"]
    assert [x["tier_seq"] for x in r["load"]] == [2]
    assert any("tier 4" in w and "offline" in w for w in r["why"])


def _complete(job):
    role = _role(0.0, [(10, 0.02)])
    return {"roles": {"fe": role}, "complete": True, "why": None, "job_run": job}


def test_d14_trees_of_one_job_run_share_one_idle_reading():
    rule = dict(RULE, min_trees=2)
    rec = {"relpath": "r", "sha256": "0" * 64, "registered_at": EARLY}
    same = [("t1", _complete(["j1", 0])), ("t2", _complete(["j1", 0]))]
    d = D14.conclude(rule, rec, same, {"t1": "2026-11-13T00:00:10.000Z",
                                       "t2": "2026-11-13T00:00:10.000Z"})
    assert "irq_softirq_is_experiment" not in d
    assert d["n_trees_complete"] == 2 and d["n_job_runs_complete"] == 1
    assert "job run" in d["conclusion_reason"]
    two = [("t1", _complete(["j1", 0])), ("t2", _complete(["j1", 1]))]
    d = D14.conclude(rule, rec, two, {"t1": "2026-11-13T00:00:10.000Z",
                                      "t2": "2026-11-13T00:00:10.000Z"})
    assert d["irq_softirq_is_experiment"] is True and d["n_job_runs_complete"] == 2
