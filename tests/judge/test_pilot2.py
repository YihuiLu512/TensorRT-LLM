# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capjudge.pilot2 (calib-v2 DESIGN-r7a P2-09, P2-09b, P2-09c; tests 13, 23, 27, 31, 34, 35, 40).

Synthetic work dirs hold judged trees as judge/verdicts.json + judge/inputs.json + JUDGED.tsv
rows (the judge's own outputs; the raw data behind them are not needed by bootstrap, shortfall
or _select). capjudge.inputs.verify, which select_tiers calls for each tree, is replaced by a
stub returning the launch facts the tree's `fake_ctx.json` names; test 27 runs d14 on the real
round-1 trees (read only)."""

import json
import os
import sys
import types

import pytest

from conftest import TOOLS

if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)
from capcommon import jobplan, ledger  # noqa: E402
from capcommon.hashing import sha256_file  # noqa: E402
from capjudge import inputs as I  # noqa: E402
from capjudge import nullz, stats  # noqa: E402
from capjudge import pilot2 as P2  # noqa: E402

MS = 1_000_000
JSHA = "b" * 40
ENV = {"hw_profile_sha": "d" * 64, "image_sha": "e" * 64, "capkit_sha": "c" * 40,
       "baseline_sha": "2f22a98d6e"}
TS = "2026-11-15T00:00:%02d.000Z"
LAG = "T1:req_first_lag_p50_ns:capload"
CNT = "T1:inflight_derived:capload"
RATIO = "T2:delivery_ratio_norm:capload"
C13 = "T3:foreign_busy_frac:relay1"


def tier(seq, zl=(), zc=(), D=(), ratio=(), c13=(), role="cal", c3="PASS", verdict="NOT_RUN",
         holds=None):
    return {"tier_seq": seq, "judged": True, "verdict": verdict, "ladder_role": role,
            "checks": {"C3": c3, "C13": "NOT_RUN"}, "X": {"holds": holds, "reasons": []},
            "zstats": {LAG: list(zl), CNT: list(zc)},
            "stats": {LAG: list(D), RATIO: list(ratio), C13: list(c13)}}


def verdicts(tiers, lv="NOT_RUN", state="PASS", cfg="s1", batch="calib-v2"):
    return {"batch": batch, "launch_verdict": lv, "arm_config": cfg,
            "arm_threshold_config": cfg, "launch_checks": {"LAUNCH-STATE": state},
            "tiers": tiers, "boundary": {"b": None}}


class FakeW:
    """A work dir with registered files, ledgers and judged trees (verdicts only)."""

    def __init__(self, root):
        self.W = str(root)
        self.n = 0
        os.makedirs(os.path.join(self.W, "register"), exist_ok=True)

    def ts(self):
        self.n += 1
        return "2026-11-15T00:%02d:%02d.000Z" % divmod(self.n, 60)

    def path(self, rel):
        return os.path.join(self.W, rel)

    def put(self, rel, data, kind=None):
        p = self.path(rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data if isinstance(data, bytes) else
                    (json.dumps(data, sort_keys=True, indent=1) + "\n").encode())
        if kind is not None:
            ledger.registry_append(self.path("REGISTRY.tsv"), "E0", rel, sha256_file(p),
                                   "kind=%s;test" % kind, registered_at=self.ts())
        return p

    def tree(self, name, vdoc, job_id="1", seq=0, arm="a", env=None, upstream="VALID",
             judge_sha=JSHA, register_sha="f" * 64, node=None, experiment="E0"):
        rel = "runs/%s" % name
        vp = self.put(rel + "/judge/verdicts.json", vdoc)
        self.put(rel + "/judge/inputs.json", {"status": "OK", "code_sha": judge_sha})
        self.put(rel + "/fake_ctx.json", {"env": env or ENV, "upstream": upstream,
                                          "register_sha": register_sha, "arm": arm,
                                          "experiment": experiment})
        if node is not None:
            self.put(rel + "/raw/dims.json", {"node": node})
        ledger.append_tsv(self.path("JUDGED.tsv"), ledger.JUDGED_COLUMNS, {
            "judged_at": self.ts(), "job_id": job_id, "restart_count": "0",
            "launch_seq": str(seq), "arm": arm, "experiment": "E0", "tree_relpath": rel,
            "verdicts_sha256": sha256_file(vp)})
        return self.path(rel)

    def rejudge(self, tree, vdoc):
        """Rewrite a tree's verdicts.json and add the matching JUDGED.tsv row."""
        rel = os.path.relpath(tree, self.W)
        vp = self.put(rel + "/judge/verdicts.json", vdoc)
        row = [r for r in ledger.read_tsv(self.path("JUDGED.tsv"), ledger.JUDGED_COLUMNS)
               if r["tree_relpath"] == rel][-1]
        ledger.append_tsv(self.path("JUDGED.tsv"), ledger.JUDGED_COLUMNS,
                          dict(row, judged_at=self.ts(), verdicts_sha256=sha256_file(vp)))

    def listfile(self, name, lines):
        return self.put("lists/" + name, "".join("%s\t%s\t%s\n" % x for x in lines).encode())


def fake_verify(W, tree):
    d = json.load(open(os.path.join(tree, "fake_ctx.json")))
    up = d["upstream"]
    env = d["env"]
    return types.SimpleNamespace(
        upstream={"status": up, "sources": [] if up == "VALID" else [
            {"source": "STATE", "status": up, "reason": "synthetic"}]},
        launch={"hw_profile_sha": env["hw_profile_sha"], "image": {"sha256": env["image_sha"]},
                "code": {"capkit_sha": env["capkit_sha"], "baseline_sha": env["baseline_sha"]},
                "submission": {}},
        ident=types.SimpleNamespace(experiment=d["experiment"], arm=d["arm"],
                                    register_sha=d["register_sha"]))


@pytest.fixture
def fake_trees(monkeypatch):
    monkeypatch.setattr(I, "verify", fake_verify)
    monkeypatch.setenv("CAPKIT_CODE_SHA", JSHA)


# ---- bootstrap (test 40) -----------------------------------------------------------------------

def v0cal(tmp_path, set_no=1, rule=None, extra=None, env_b=None):
    """A V0-cal work dir: register (bootstrap_rule, d14_result), trees A, B (two configs) and
    C (launch-level conclusion not PASS, huge readings: must be dropped). Set 1: max |z| 3.0,
    max |D| 2 ms, |norm - 1| all 0, C13 window max 0.03, launch mean max 0.02; set 2: 4.0,
    5 ms, 0.01, 0.04, 0.01."""
    fw = FakeW(tmp_path / "W")
    d14 = fw.put("register/e0_d14.json", {"schema": "capkit.d14_result/1"}, "d14_result")
    reg = fw.put("register/register.json", {
        "schema": "capkit.register/1", "experiment": "E0", "arm_config": {"a": "s1", "b": "s2"},
        "calibration": [{"kind": "d14_result", "relpath": "register/e0_d14.json",
                         "sha256": sha256_file(d14)}],
        "expected": {"bootstrap_rule": P2.BOOTSTRAP_RULE if rule is None else rule}},
        "register")
    rs = sha256_file(reg)
    junk = dict(zl=[500.0], zc=[-400.0], D=[900 * MS], ratio=[0.5], c13=[0.9])
    if set_no == 1:
        a = [tier(1, role=None, **junk), tier(2, zl=[3.0, -1.0], zc=[-2.5], D=[2 * MS, -MS],
                                               ratio=[0.0], c13=[0.03]),
             tier(3, zl=[0.5], zc=[1.0], D=[0.5 * MS], ratio=[0.0], c13=[0.01]),
             tier(4, c3="RED", verdict="VOID", **junk)]
        b = [tier(2, zl=[-2.0], zc=[3.0], D=[-1.5 * MS], ratio=[0.0], c13=[-0.02]),
             tier(3, zl=[1.0], zc=[0.0], D=[0.0], ratio=[0.0], c13=[0.0])]
    else:
        a = [tier(1, role=None, **junk), tier(2, zl=[4.0], zc=[-1.0], D=[5 * MS], ratio=[0.01],
                                               c13=[0.04]),
             tier(3, zl=[1.0], zc=[-4.0], D=[MS], ratio=[0.0], c13=[-0.02]),
             tier(4, c3="RED", verdict="VOID", **junk)]
        b = [tier(2, zl=[2.0], zc=[1.0], D=[-3 * MS], ratio=[0.005], c13=[0.01]),
             tier(3, zl=[0.0], zc=[0.0], D=[0.0], ratio=[0.0], c13=[0.0])]
    ta = fw.tree("A", verdicts(a, cfg="s1"), job_id="11", arm="a", register_sha=rs, node="n1")
    tb = fw.tree("B", verdicts(b, cfg="s2"), job_id="11", seq=1, arm="b", register_sha=rs,
                 env=env_b)
    tc = fw.tree("C", verdicts([tier(2, **junk)], cfg="s1", state="RED"), job_id="12",
                 arm="a", register_sha=rs, upstream="VOID")
    trees = [ta, tb, tc] + (extra(fw, rs) if extra else [])
    lst = fw.listfile("cal.tsv", [(t, "ALL", "subsat") for t in trees])
    return fw, reg, lst, trees


def run_cli(args):
    return P2.main([str(a) for a in args])


def boot(fw, reg, lst, out="boot.json"):
    o = fw.path(out)
    code = run_cli(["bootstrap", "--workdir", fw.W, "--list", lst, "--register", reg,
                    "--out", o])
    return code, (json.load(open(o)) if code == 0 else None), o


def test_bootstrap_set1_design_floors_bind(tmp_path, fake_trees):
    fw, reg, lst, trees = v0cal(tmp_path, 1)
    code, d, out = boot(fw, reg, lst)
    assert code == 0
    assert d["schema"] == "capkit.pilot/2" and d["stage"] == "bootstrap"
    assert d["rule"] == P2.BOOTSTRAP_RULE and d["select"] == "capjudge.pilot.select_tiers"
    assert d["x_const"] == {"lag": {"z": 10.0, "min_D_ns": 10_000_000}, "count": {"z": 10.705}}
    assert d["ratio_tol"] == {"T2:delivery_ratio_norm": {"min_per_request": 2, "tol": 0.02}}
    cap = d["c13_cap"]["T3:foreign_busy_frac"]
    assert cap["window"] == 0.10 and cap["launch_mean"] == pytest.approx(0.06, abs=1e-12)
    m = d["max_abs"]
    assert {f: m[f]["bound_by"] for f in m} == {
        "lag_z": "design_floor", "lag_D": "design_floor", "count_z": "design_floor",
        "ratio": "design_floor", "c13_window": "design_floor", "c13_launch_mean": "data"}
    assert m["lag_z"]["value"] == 3.0 and m["lag_z"]["three_x_max"] == 9.0
    assert m["lag_D"]["three_x_max"] == pytest.approx(6 * MS)
    assert (m["lag_z"]["tree"], m["lag_z"]["tier_seq"], m["lag_z"]["key"]) == ("runs/A", 2, LAG)
    assert m["c13_launch_mean"]["value"] == pytest.approx(0.02, abs=1e-12)
    assert m["c13_launch_mean"]["tier_seq"] is None
    # readings: cal tiers 2 and 3 of A and B only (C excluded, A's tier 4 RED, tier 1 not cal)
    assert m["lag_z"]["n_readings"] == 5 and m["c13_window"]["n_readings"] == 4
    assert m["c13_launch_mean"]["n_readings"] == 2
    inp = {i["tree"]: i for i in d["inputs"]}
    assert inp["runs/A"]["tiers_used"] == [2, 3] and "4" in inp["runs/A"]["tiers_excluded"]
    assert inp["runs/A"]["config"] == "s1" and inp["runs/A"]["node"] == "n1"
    assert inp["runs/C"]["tiers_used"] == [] and "tree_excluded" in inp["runs/C"]
    assert d["environment"] == dict(ENV, judge_sha=JSHA)
    assert d["d14"]["relpath"] == "register/e0_d14.json"
    assert os.path.isfile(out[:-5] + ".cmd.txt")
    # never overwritten
    assert run_cli(["bootstrap", "--workdir", fw.W, "--list", lst, "--register", reg,
                    "--out", out]) == 3


def test_bootstrap_set2_data_terms_bind(tmp_path, fake_trees):
    fw, reg, lst, _ = v0cal(tmp_path, 2)
    code, d, _o = boot(fw, reg, lst)
    assert code == 0
    tol = 1e-12
    assert d["x_const"]["lag"]["z"] == pytest.approx(12.0, abs=tol)
    assert d["x_const"]["lag"]["min_D_ns"] == pytest.approx(15 * MS, abs=tol)
    assert d["x_const"]["count"]["z"] == pytest.approx(12.0, abs=tol)
    assert d["ratio_tol"]["T2:delivery_ratio_norm"]["tol"] == pytest.approx(0.03, abs=tol)
    cap = d["c13_cap"]["T3:foreign_busy_frac"]
    assert cap["window"] == pytest.approx(0.12, abs=tol)
    assert cap["launch_mean"] == pytest.approx(0.05, abs=tol)
    assert {f: v["bound_by"] for f, v in d["max_abs"].items()} == {
        "lag_z": "data", "lag_D": "data", "count_z": "data", "ratio": "data",
        "c13_window": "data", "c13_launch_mean": "design_floor"}


def test_bootstrap_output_is_a_threshold_source_for_every_config_and_regime(tmp_path, fake_trees):
    fw, reg, lst, _ = v0cal(tmp_path, 2)
    code, d, out = boot(fw, reg, lst, out="register/boot.json")
    assert code == 0
    ledger.registry_append(fw.path("REGISTRY.tsv"), "E0", "register/boot.json",
                           sha256_file(out), "kind=pilot;test", registered_at=fw.ts())
    th = nullz.Thresholds()
    th.add_source("pilot", "register/boot.json", sha256_file(out), d)
    seen = set()
    for cfg in ("s1", "s2", "s3", None):
        for regime in ("subsat", "near_boundary"):
            lag = th.get(LAG, cfg, regime)
            cnt = th.get(CNT, cfg, regime)
            rat = th.get(RATIO, cfg, regime)
            seen.add((lag["z"], lag["min_D_ns"], cnt["z"], rat["tol"]))
    assert len(seen) == 1
    assert th.cap("T3:foreign_busy_frac")["window"] == pytest.approx(0.12)


def test_bootstrap_ignores_x_conclusions(tmp_path, fake_trees):
    """Changing only the tiers' X conclusions (verdict and X, not the readings) changes nothing
    but the verdicts sha each input records."""
    fw, reg, lst, trees = v0cal(tmp_path, 1)
    code, d1, o1 = boot(fw, reg, lst, out="b1.json")
    assert code == 0
    for t in trees:
        v = json.load(open(os.path.join(t, "judge", "verdicts.json")))
        for e in v["tiers"]:
            if e["verdict"] != "VOID":
                e["verdict"], e["X"] = "UNSTABLE", {"holds": True, "reasons": ["changed"]}
        v["launch_verdict"] = "VALID"
        fw.rejudge(t, v)
    code, d2, o2 = boot(fw, reg, lst, out="b2.json")
    assert code == 0

    def strip(doc):
        doc = json.loads(json.dumps(doc))
        for i in doc["inputs"]:
            i.pop("judge_verdicts_sha256")
        return json.dumps(doc, sort_keys=True)
    assert d1["inputs"][0]["judge_verdicts_sha256"] != d2["inputs"][0]["judge_verdicts_sha256"]
    assert strip(d1) == strip(d2)


def _extra_other_env(fw, rs):
    return [fw.tree("D", verdicts([tier(2, zl=[1.0], zc=[1.0], D=[MS], ratio=[0.0],
                                        c13=[0.0])], cfg="s4"), job_id="13", arm="a",
                    register_sha=rs, env=dict(ENV, hw_profile_sha="9" * 64))]


def _extra_other_judge(fw, rs):
    return [fw.tree("D", verdicts([tier(2, zl=[1.0], zc=[1.0], D=[MS], ratio=[0.0],
                                        c13=[0.0])], cfg="s4"), job_id="13", arm="a",
                    register_sha=rs, judge_sha="a" * 40)]


def _extra_v1(fw, rs):
    return [fw.tree("D", verdicts([tier(2)], cfg="s4", batch="b2-judge"), job_id="13",
                    arm="a", register_sha=rs)]


def _extra_other_register(fw, rs):
    return [fw.tree("D", verdicts([tier(2, zl=[1.0])], cfg="s4"), job_id="13", arm="a",
                    register_sha="0" * 64)]


def _extra_no_cal(fw, rs):
    return [fw.tree("D", verdicts([tier(2, role="levels")], cfg="s4"), job_id="13", arm="a",
                    register_sha=rs)]


@pytest.mark.parametrize("extra", [_extra_other_env, _extra_other_judge, _extra_v1,
                                   _extra_other_register, _extra_no_cal])
def test_bootstrap_rejects_mixed_or_foreign_inputs(tmp_path, fake_trees, extra):
    fw, reg, lst, _ = v0cal(tmp_path, 1, extra=extra)
    code, _d, out = boot(fw, reg, lst)
    assert code == 3 and not os.path.exists(out)


def test_bootstrap_rejects_a_rule_text_that_differs(tmp_path, fake_trees):
    fw, reg, lst, _ = v0cal(tmp_path, 1, rule=P2.BOOTSTRAP_RULE.replace("3*max", "2*max"))
    code, _d, out = boot(fw, reg, lst)
    assert code == 3 and not os.path.exists(out)
    fw, reg, lst, _ = v0cal(tmp_path / "x", 1, rule=P2.BOOTSTRAP_RULE + " ")
    assert boot(fw, reg, lst)[0] == 3


def test_bootstrap_zero_readings_of_a_family_exit_3(tmp_path, fake_trees):
    fw, reg, lst, trees = v0cal(tmp_path, 1)
    for t in trees:
        v = json.load(open(os.path.join(t, "judge", "verdicts.json")))
        for e in v["tiers"]:
            e["stats"].pop(C13)
        fw.rejudge(t, v)
    code, _d, out = boot(fw, reg, lst)
    assert code == 3 and not os.path.exists(out)


def test_bootstrap_list_lines_are_all_subsat(tmp_path, fake_trees):
    fw, reg, _lst, trees = v0cal(tmp_path, 1)
    lst = fw.listfile("bad.tsv", [(trees[0], "2,3", "subsat")])
    assert boot(fw, reg, lst)[0] == 3
    lst = fw.listfile("bad2.tsv", [(trees[0], "ALL", "near")])
    assert boot(fw, reg, lst)[0] == 3


def test_bootstrap_values_follow_the_rule_exactly():
    """bootstrap_values on hand-made readings: each family max(floor, 3 x max |reading|)."""
    r = {f: [(0.0, "t", 1, "k")] for f in P2.BOOTSTRAP_FAMILIES}
    r["lag_z"] = [(3.4, "t", 1, LAG), (1.0, "t", 2, LAG)]
    r["c13_window"] = [(0.0334, "t", 1, C13)]
    xc, rt, cap, m = P2.bootstrap_values(r)
    assert xc["lag"]["z"] == pytest.approx(10.2) and m["lag_z"]["bound_by"] == "data"
    assert cap["T3:foreign_busy_frac"]["window"] == pytest.approx(0.1002)
    assert xc["count"]["z"] == 10.705 and rt["T2:delivery_ratio_norm"]["tol"] == 0.02
    assert P2.BOOTSTRAP_FLOORS == {"lag_z": 10.0, "lag_D": 10_000_000, "count_z": 10.705,
                                   "ratio": 0.02, "c13_window": 0.10, "c13_launch_mean": 0.05}
    r["ratio"] = []
    with pytest.raises(P2.Pilot2Error, match="zero readings"):
        P2.bootstrap_values(r)


# ---- _select (test 31) and shortfall (test 35) -------------------------------------------------

def item(order, item_id, counts, cls="planned", arms=None, pin=False, block=None, slot=None):
    return {"order": order, "item_id": item_id, "item_class": cls,
            "arms": arms or sorted({{"s1": "a", "s2": "b"}[c] for c in counts}),
            "launches": sum(counts.values()), "block_id": block, "slot": slot,
            "config_counts": dict(counts), "pin": pin}


class Seg:
    """A segment: registered register (arm_config a -> s1, b -> s2) and job plans, BUDGET.tsv
    RESERVED rows, submissions.tsv rows and judged trees."""

    def __init__(self, root, cap_jobs=40, arm_config=None):
        self.fw = FakeW(root)
        self.reg = self.fw.put("register/register.json", {
            "schema": "capkit.register/1", "experiment": "E0",
            "arm_config": arm_config or {"a": "s1", "b": "s2"},
            "expected": {"shortfall_cap_jobs": cap_jobs}}, "register")
        self.plans, self.sha = [], {}
        self.trees = {}

    def plan(self, name, items):
        p = self.fw.put("register/%s.tsv" % name, jobplan.format_tsv(items), "job_plan")
        self.plans.append(p)
        for it in items:
            self.sha[it["item_id"]] = (sha256_file(p), it)
        return p

    def run(self, item_id, status="VALID", launches=None, ended=True, node=None):
        """Submit and end the job of an item; launches = [(config, launch_verdict)]."""
        sha, it = self.sha[item_id]
        pid = "p-" + item_id
        ledger.budget_append(self.fw.path("BUDGET.tsv"), {
            "event": "RESERVED", "budget_id": "E0v2-T", "item_id": item_id,
            "item_class": it["item_class"], "plan_sha256": sha,
            "register_sha256": sha256_file(self.reg), "plan_id": pid, "job_id": ""})
        job = "9%04d" % (len(self.trees) + 1)
        base = {c: "" for c in ledger.SUBMISSION_COLUMNS}
        sub = self.fw.path("submissions.tsv")
        ledger.submissions_append(sub, dict(base, event="SUBMITTED", plan_id=pid, job_id=job,
                                            event_at=self.fw.ts()))
        if ended:
            ledger.submissions_append(sub, dict(base, event="ENDED", plan_id=pid, job_id=job,
                                                status=status, event_at=self.fw.ts()))
        if launches is None:
            launches = [] if status in P2.JOB_LEVEL_UNUSABLE else [
                (c, "NOT_RUN") for c, n in sorted(it["config_counts"].items())
                for _ in range(n)]
        out = []
        for k, (cfg, lv) in enumerate(launches):
            arm = {"s1": "a", "s2": "b"}[cfg]
            out.append(self.fw.tree("%s-L%d" % (item_id, k), verdicts(
                [tier(2, zl=[1.0])], lv=lv, cfg=cfg), job_id=job, seq=k, arm=arm, node=node))
        self.trees[item_id] = out
        return out

    def rel(self, item_id):
        return ["runs/%s-L%d" % (item_id, k) for k in range(len(self.trees[item_id]))]

    def select(self, void_usable=False):
        return P2._select(self.plans, self.fw.W, void_usable=void_usable)


def five_plus_two(root):
    s = Seg(root)
    s.plan("plan", [item(k, "T-%04d" % k, {"s1": 1},
                         cls="planned" if k <= 5 else "surplus") for k in range(1, 8)])
    return s


def test_select_first_n_usable_in_plan_order(tmp_path):
    s = five_plus_two(tmp_path)
    for k in range(1, 8):
        iid = "T-%04d" % k
        if k == 2:
            s.run(iid, status="ENV_INVALID")
        else:
            s.run(iid, launches=[("s1", "VOID" if k == 4 else "NOT_RUN" if k == 3 else
                                  "VALID")])
    sel = s.select()
    want = [r for k in (1, 3, 5, 6, 7) for r in s.rel("T-%04d" % k)]
    assert sel["selected"] == want and sel["surplus_unused"] == []
    un = {u["item_id"]: u["reason"] for u in sel["unusable"]}
    assert un == {"T-0002": "T-0002: job ENV_INVALID",
                  "T-0004": "runs/T-0004-L0: launch_verdict VOID"}
    assert sel["by_unit"]["s1"] == {"N": 5, "n_all_selected": 0, "usable": 5, "shortfall": 0}


def test_select_does_not_read_z_or_b(tmp_path):
    s = five_plus_two(tmp_path)
    for k in range(1, 8):
        s.run("T-%04d" % k, launches=[("s1", "VOID" if k == 2 else "NOT_RUN")])
    before = s.select()
    for k in range(1, 8):
        for t in s.trees["T-%04d" % k]:
            v = json.load(open(os.path.join(t, "judge", "verdicts.json")))
            v["tiers"][0]["zstats"][LAG] = [99.0]
            v["boundary"] = {"b": 1.5, "censored": True}
            s.fw.rejudge(t, v)
    after = s.select()
    for k in ("selected", "surplus_unused", "unusable", "by_unit"):
        assert before[k] == after[k]
    assert len(before["surplus_unused"]) == 1


def test_select_reports_the_shortfall_when_usable_launches_run_out(tmp_path):
    s = five_plus_two(tmp_path)
    for k in range(1, 8):
        s.run("T-%04d" % k, status="CANCELLED" if k in (1, 2, 3) else "VALID",
              launches=None if k in (1, 2, 3) else [("s1", "VALID")])
    sel = s.select()
    assert sel["selected"] == [r for k in (4, 5, 6, 7) for r in s.rel("T-%04d" % k)]
    assert sel["by_unit"]["s1"]["shortfall"] == 1


def test_select_coverage_items_all_enter_and_n_grows(tmp_path):
    s = five_plus_two(tmp_path)
    s.plan("cov", [item(1, "T-cov-0001", {"s1": 1}, cls="post_gate:coverage", pin=True)])
    for k in range(1, 8):
        s.run("T-%04d" % k, launches=[("s1", "VALID")])
    s.run("T-cov-0001", launches=[("s1", "VALID")])
    sel = s.select()
    assert sel["selected"] == [r for k in (1, 2, 3, 4, 5) for r in s.rel("T-%04d" % k)] \
        + s.rel("T-cov-0001")
    assert sel["surplus_unused"] == s.rel("T-0006") + s.rel("T-0007")
    assert sel["by_unit"]["s1"]["n_all_selected"] == 1


def test_select_shortfall_items_queue_after_the_original_segment(tmp_path):
    s = Seg(tmp_path)
    s.plan("plan", [item(k, "T-%04d" % k, {"s1": 1}) for k in range(1, 4)])
    s.plan("sf", [item(k, "T-sf-%04d" % k, {"s1": 1}, cls="post_gate:shortfall")
                  for k in range(1, 4)])
    s.run("T-0001", status="ENV_INVALID")
    s.run("T-0002", launches=[("s1", "VOID")])
    s.run("T-0003", launches=[("s1", "VALID")])
    for k in range(1, 4):
        s.run("T-sf-%04d" % k, launches=[("s1", "VALID")])
    sel = s.select()
    assert sel["selected"] == s.rel("T-0003") + s.rel("T-sf-0001") + s.rel("T-sf-0002")
    assert sel["surplus_unused"] == s.rel("T-sf-0003")


def blocks(root, nodes=None):
    s = Seg(root)
    its = []
    for k in range(1, 5):
        bid = "AA-b%02d" % k
        cls = "planned" if k <= 3 else "surplus"
        for slot, cfg, arm in ((1, "s1", "a"), (2, "s2", "b")):
            its.append(item(k, "%s-s%d" % (bid, slot), {cfg: 1}, cls=cls, pin=True,
                            arms=["%s.%s.%d" % (arm, bid, slot)], block=bid, slot=slot))
    s.plan("blocks", its)
    return s


def test_select_blocks_need_both_jobs_usable_on_one_node(tmp_path):
    s = blocks(tmp_path)
    s.run("AA-b01-s1", launches=[("s1", "VALID")], node="n1")
    s.run("AA-b01-s2", status="CANCELLED")
    s.run("AA-b02-s1", launches=[("s1", "NOT_RUN")], node="n2")  # censored: still usable
    s.run("AA-b02-s2", launches=[("s2", "NOT_RUN")], node="n2")
    s.run("AA-b03-s1", launches=[("s1", "VALID")], node="n3")
    s.run("AA-b03-s2", launches=[("s2", "VALID")], node="n4")  # two nodes: unusable
    s.run("AA-b04-s1", launches=[("s1", "VALID")], node="n5")
    s.run("AA-b04-s2", launches=[("s2", "VALID")], node="n5")
    sel = s.select()
    assert sel["selected"] == s.rel("AA-b02-s1") + s.rel("AA-b02-s2") + s.rel("AA-b04-s1") \
        + s.rel("AA-b04-s2")
    assert sel["by_unit"]["cmp:AA"] == {"N": 3, "n_all_selected": 0, "usable": 2,
                                        "shortfall": 1}
    assert {u["item_id"] for u in sel["unusable"]} == {"AA-b01-s1", "AA-b01-s2", "AA-b03-s1",
                                                       "AA-b03-s2"}


def segment31(root, cap_jobs=40):
    """s1: 7 planned jobs x 4 launches (N = 28) + 1 surplus job of 3 launches (31 launches)."""
    s = Seg(root, cap_jobs=cap_jobs)
    s.plan("plan", [item(k, "S-%04d" % k, {"s1": 4 if k <= 7 else 3},
                         cls="planned" if k <= 7 else "surplus") for k in range(1, 9)])
    return s


def run31(s, void_in=(2, 5), env_invalid=(3,), unended=()):
    for k in range(1, 9):
        iid = "S-%04d" % k
        n = 4 if k <= 7 else 3
        if k in env_invalid:
            s.run(iid, status="ENV_INVALID")
        else:
            s.run(iid, ended=k not in unended, launches=[
                ("s1", "VOID" if (k in void_in and j == 0) else "NOT_RUN") for j in range(n)])


def shortfall(s, lst, out="sf.json", job_level_only=False, per_job=4, plans=None):
    o = s.fw.path(out)
    args = ["shortfall", "--workdir", s.fw.W, "--list", lst, "--register", s.reg,
            "--launches-per-job", per_job, "--phase", "V1b", "--segment", "A", "--out", o]
    for p in plans or s.plans:
        args += ["--plan", p]
    if job_level_only:
        args.append("--job-level-only")
    code = run_cli(args)
    return code, (json.load(open(o)) if code == 0 else None), o


def sel_list(s, name="sel.tsv", job_level_only=False, edit=None):
    trees = [s.fw.path(t) for t in s.select(void_usable=job_level_only)["selected"]]
    if edit:
        trees = edit(trees)
    return s.fw.listfile(name, [(t, "ALL", "subsat") for t in trees])


def test_shortfall_tops_up_the_missing_launches(tmp_path, fake_trees):
    s = segment31(tmp_path)
    run31(s)  # usable: 28 - 4 (ENV_INVALID) - 2 (VOID) + 3 surplus = 25 -> d = 3
    code, d, _o = shortfall(s, sel_list(s))
    assert code == 0
    assert d["schema"] == "capkit.e0_params/1" and d["stage"] == "shortfall"
    assert d["by_config"]["s1"]["shortfall"] == 3 and d["round"] == 1
    assert d["q_hat"] == {"lost": 3, "ended": 8, "value": 0.375}
    n = stats.shortfall_topup(3, 3, 8)
    assert n == 7
    assert d["items"] == [{"item_class": "post_gate:shortfall", "config": "s1", "arm": "a",
                           "launches": n, "pin": False}]
    assert d["capped"] is False and d["cap"]["jobs"] == 2


def test_shortfall_refuses_a_segment_with_an_unended_item(tmp_path, fake_trees):
    s = segment31(tmp_path)
    run31(s, unended=(6,))
    code, _d, o = shortfall(s, sel_list(s))
    assert code == 3 and not os.path.exists(o)


def test_shortfall_third_round_exits_3(tmp_path, fake_trees):
    s = segment31(tmp_path)
    s.plan("sf1", [item(1, "S-sf1-0001", {"s1": 1}, cls="post_gate:shortfall")])
    s.plan("sf2", [item(1, "S-sf2-0001", {"s1": 1}, cls="post_gate:shortfall")])
    run31(s)
    s.run("S-sf1-0001", status="ENV_INVALID")
    s.run("S-sf2-0001", status="ENV_INVALID")
    code, _d, o = shortfall(s, sel_list(s))
    assert code == 3 and not os.path.exists(o)
    code, d, _o = shortfall(s, sel_list(s, "two.tsv"), out="r2.json", plans=s.plans[:2])
    assert code == 0 and d["round"] == 2


def test_shortfall_caps_the_cumulative_jobs(tmp_path, fake_trees):
    s = segment31(tmp_path, cap_jobs=1)
    run31(s)
    code, d, _o = shortfall(s, sel_list(s))
    assert code == 0 and d["capped"] is True
    assert d["items"][0]["launches"] == 4 and d["cap"]["jobs"] == 1
    assert d["by_config"]["s1"]["topup"] == 7 and d["by_config"]["s1"]["listed"] == 4


def test_shortfall_q_hat_is_clamped_when_every_job_is_lost(tmp_path, fake_trees):
    s = segment31(tmp_path, cap_jobs=1000)
    run31(s, void_in=(), env_invalid=tuple(range(1, 9)))
    code, d, _o = shortfall(s, sel_list(s))
    assert code == 0
    assert d["q_hat"] == {"lost": 8, "ended": 8, "value": 0.5}
    assert d["items"][0]["launches"] == stats.shortfall_topup(28, 8, 8) == 71


def test_shortfall_output_does_not_depend_on_z_b_or_x(tmp_path, fake_trees):
    s = segment31(tmp_path)
    run31(s)
    lst = sel_list(s)
    code, _d, o1 = shortfall(s, lst, out="a.json")
    assert code == 0
    for trees in s.trees.values():
        for t in trees:
            v = json.load(open(os.path.join(t, "judge", "verdicts.json")))
            v["tiers"][0]["zstats"][LAG] = [123.0]
            v["tiers"][0]["X"] = {"holds": True, "reasons": ["changed"]}
            v["boundary"] = {"b": 3.0}
            s.fw.rejudge(t, v)
    code, _d, o2 = shortfall(s, lst, out="b.json")
    assert code == 0
    assert open(o1, "rb").read() == open(o2, "rb").read()


def test_shortfall_job_level_only_ignores_void(tmp_path, fake_trees):
    s = Seg(tmp_path)
    s.plan("plan", [item(k, "V-%04d" % k, {"s1": 1}) for k in (1, 2)])
    s.run("V-0001", launches=[("s1", "VOID")])
    s.run("V-0002", status="ENV_INVALID")
    code, d, _o = shortfall(s, sel_list(s, job_level_only=True), job_level_only=True,
                            per_job=1)
    assert code == 0 and d["job_level_only"] is True
    assert d["by_config"]["s1"]["shortfall"] == 1 and d["q_hat"]["lost"] == 1
    assert d["items"][0]["launches"] == stats.shortfall_topup(1, 1, 2)
    code, d, _o = shortfall(s, sel_list(s, "all.tsv"), out="all.json", per_job=1)
    assert code == 0 and d["by_config"]["s1"]["shortfall"] == 2 and d["q_hat"]["lost"] == 2


@pytest.mark.parametrize("edit", [
    lambda t: t + [t[0].replace("S-0001-L0", "S-0008-L2")],  # one more (a surplus tree)
    lambda t: t[1:],                                          # one missing
    lambda t: [t[1], t[0]] + t[2:],                           # reordered
])
def test_list_must_equal_the_select_list(tmp_path, fake_trees, edit):
    s = segment31(tmp_path)
    run31(s, void_in=(), env_invalid=())
    assert len(s.select()["surplus_unused"]) == 3
    code, _d, o = shortfall(s, sel_list(s, edit=edit))
    assert code == 3 and not os.path.exists(o)


# ---- framework rejections (test 13) ------------------------------------------------------------

@pytest.mark.parametrize("text, why", [
    ("runs/*\tALL\tsubsat\n", "literal path"),
    ("runs/A\t2,*\tsubsat\n", "wildcard"),
    ("runs/A\tALL\tsubsat\nruns/A\t2\tsubsat\n", "listed twice"),
    ("runs/A\t2\n", "need"),
    ("runs/A\t2,x\tsubsat\n", "integers"),
    ("runs/A\t2,2\tsubsat\n", "listed twice"),
    ("runs/A\tALL\tslope\n", "regime"),
    ("# only a comment\n", "lists no tree"),
])
def test_list_file_rejections(tmp_path, text, why):
    p = tmp_path / "l.tsv"
    p.write_text(text)
    with pytest.raises(P2.Pilot2Error, match=why):
        P2.read_list(str(p))


def test_list_file_reading(tmp_path):
    p = tmp_path / "l.tsv"
    p.write_text("# c\n%s/A\t2,10\tsubsat\n\n%s/B\tALL\tnear\n" % (tmp_path, tmp_path))
    assert P2.read_list(str(p)) == [(str(tmp_path / "A"), [2, 10], "subsat"),
                                    (str(tmp_path / "B"), "ALL", "near")]
    p.write_text("")
    assert P2.read_list(str(p), allow_empty=True) == []


def _tt(windows, ramps):
    return types.SimpleNamespace(tiers=[types.SimpleNamespace(window_ns=w, ramp_ns=r)
                                        for w, r in zip(windows, ramps)])


def test_window_and_ramp_checks():
    S = 30 * 10 ** 9
    tt = _tt([60 * 10 ** 9, 60 * 10 ** 9, 30 * 10 ** 9], [34 * 10 ** 9, 30 * 10 ** 9, 0])
    P2.check_windows("t", tt, [0, 1])
    with pytest.raises(P2.Pilot2Error, match="tier windows"):
        P2.check_windows("t", tt, [0, 2])
    P2.check_ramp("t", tt, [0], S)
    with pytest.raises(P2.Pilot2Error, match="4 s \\+ S"):
        P2.check_ramp("t", tt, [0, 1], S)


def test_unique_environment_and_judge_sha():
    assert P2.unique([dict(ENV), dict(ENV)], "environment") == ENV
    with pytest.raises(P2.Pilot2Error, match="environment is not unique"):
        P2.unique([dict(ENV), dict(ENV, image_sha="0" * 64)], "environment")
    with pytest.raises(P2.Pilot2Error, match="judge code sha"):
        P2.unique(["a" * 40, "b" * 40], "judge code sha")


def test_judged_needs_the_judged_tsv_row_and_sha(tmp_path):
    fw = FakeW(tmp_path)
    t = fw.tree("A", verdicts([tier(2)]))
    assert P2.judged(fw.W, t)["judge_sha"] == JSHA
    with open(os.path.join(t, "judge", "verdicts.json"), "a") as f:
        f.write(" ")
    with pytest.raises(P2.Pilot2Error, match="!= JUDGED.tsv"):
        P2.judged(fw.W, t)
    t2 = fw.tree("B", verdicts([tier(2)]))
    fw.put("runs/C/judge/verdicts.json", verdicts([tier(2)]))
    with pytest.raises(P2.Pilot2Error, match="not in"):
        P2.judged(fw.W, fw.path("runs/C"))
    fw.put("runs/B/judge/inputs.json", {"status": "OK", "code_sha": None})
    with pytest.raises(P2.Pilot2Error, match="code sha"):
        P2.judged(fw.W, t2)


def test_v1_tree_is_rejected():
    P2.check_v2("t", {"batch": "calib-v2"})
    for b in ("b2-judge", None):
        with pytest.raises(P2.Pilot2Error, match="v1 tree"):
            P2.check_v2("t", {"batch": b})


def test_e0_params_must_be_registered_and_of_the_stage(tmp_path):
    fw = FakeW(tmp_path)
    good = fw.put("register/cap.json", {"schema": "capkit.e0_params/1", "stage": "capacity"},
                  "e0_params")
    assert P2.load_e0_params(fw.W, good, "capacity")["relpath"] == "register/cap.json"
    with pytest.raises(P2.Pilot2Error, match="needs the registered"):
        P2.load_e0_params(fw.W, None, "capacity")
    with pytest.raises(P2.Pilot2Error, match="does not exist"):
        P2.load_e0_params(fw.W, fw.path("register/none.json"), "capacity")
    with pytest.raises(P2.Pilot2Error, match="stage 'bracket'"):
        P2.load_e0_params(fw.W, good, "bracket")
    loose = fw.put("register/loose.json", {"schema": "capkit.e0_params/1", "stage": "capacity",
                                           "note": "never registered"})
    with pytest.raises(P2.Pilot2Error, match="not in REGISTRY"):
        P2.load_e0_params(fw.W, loose, "capacity")
    other = fw.put("register/other.json", {"schema": "capkit.e0_params/1", "stage": "x"},
                   "ladder")
    with pytest.raises(P2.Pilot2Error, match="kind 'ladder'"):
        P2.load_e0_params(fw.W, other, "capacity")


def test_plans_must_be_registered_job_plans(tmp_path):
    s = Seg(tmp_path)
    p = s.fw.put("register/loose.tsv", jobplan.format_tsv([item(1, "L-0001", {"s1": 1})]))
    with pytest.raises(P2.Pilot2Error, match="not in REGISTRY"):
        P2._select([p], s.fw.W)
    s.plan("sf", [item(1, "L-sf-0001", {"s1": 1}, cls="post_gate:shortfall")])
    with pytest.raises(P2.Pilot2Error, match="original segment"):
        P2._select(s.plans, s.fw.W)


# ---- V0 cancel count (test 23) ----------------------------------------------------------------

def test_v0_cancel_count_window_and_loader_stop():
    codes = {"1": "timeout_registered", "7": "grpc_CANCELLED"}
    S = 10 ** 9
    rows = {"done_rel_ns": [10 * S, 15 * S, 25 * S, 31 * S, 40 * S],
            "status": [7, 7, 1, 7, 7]}
    # overload tier + drain slot [10 s, 35 s), load generator stopped at 30 s
    assert P2.v0_cancelled(rows, codes, 10 * S, 35 * S, 30 * S) == 2
    # a cancellation at or after the stop is the stop's own
    rows2 = {"done_rel_ns": [30 * S, 32 * S], "status": [7, 7]}
    assert P2.v0_cancelled(rows2, codes, 10 * S, 35 * S, 30 * S) == 0
    assert P2.v0_cancelled(rows2, codes, 10 * S, 35 * S, 40 * S) == 2


# ---- d14 on the round-1 trees (tests 27, 34) ---------------------------------------------------

REAL_W = "/home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02"
REAL_RULE = os.path.join(REAL_W, "02_e0/E0/register/d14_rule.json")
REAL = os.path.join(os.path.dirname(__file__), "..", "..", "samples", "calibv2_real")
needs_real_w = pytest.mark.skipif(not os.path.isfile(REAL_RULE),
                                  reason="round-1 work dir not on this machine")


def _real_list(tmp_path):
    src = os.path.join(REAL, "e0_round1.pilot_inputs.tsv")
    assert sha256_file(src) == \
        "344e0c45fce623aafb670e1aa3a66368b15d0a20754edd8fb8e0173a02c0fd24"
    assert sha256_file(REAL_RULE) == \
        "ae28dec44bc0579dd34e9774c64569a86a6bcbfdef135b46885c5cce204d8444"
    lines = [ln.split("\t") for ln in open(src).read().splitlines() if ln.strip()]
    p = tmp_path / "d14.list"
    p.write_text("".join("%s\t%s\tsubsat\n" % (t, s) for t, s in lines))
    return str(p), [t for t, _s in lines]


@needs_real_w
def test_d14_reproduces_the_registered_round_1_readings(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPKIT_CODE_SHA", JSHA)
    lst, trees = _real_list(tmp_path)
    out = tmp_path / "e0_d14.json"
    assert P2.main(["d14", "--workdir", REAL_W, "--list", lst, "--rule", REAL_RULE,
                    "--out", str(out)]) == 0
    d = json.load(open(out))
    reg = json.load(open(os.path.join(REAL, "e0_round1.pilot.json")))
    assert d["schema"] == "capkit.d14_result/1" and d["select"] == "capjudge.pilot.select_tiers"
    assert d["d14_softirq"]["readings"] == reg["d14_softirq"]["readings"]
    assert d["d14_softirq"]["irq_softirq_is_experiment"] is False
    assert d["d14_softirq"]["n_trees_complete"] == 12 and len(d["inputs"]) == 15
    assert d["rule"]["sha256"] == sha256_file(REAL_RULE)
    # environment = the input trees' profile / image / capkit / baseline (test 34)
    launch = json.load(open(os.path.join(trees[0], "raw", "launch.json")))
    assert d["environment"] == {"hw_profile_sha": launch["hw_profile_sha"],
                                "image_sha": launch["image"]["sha256"],
                                "capkit_sha": launch["code"]["capkit_sha"],
                                "baseline_sha": launch["code"]["baseline_sha"]}
    # the judge's launch verdict would select nothing: no round-1 tree is VALID
    lvs = [json.load(open(os.path.join(t, "judge", "verdicts.json")))["launch_verdict"]
           for t in trees]
    assert lvs.count("VALID") == 0
    # the product is accepted as a d14_result source
    assert nullz.validate("d14_result", d, "e0_d14.json") is None


@needs_real_w
def test_d14_rejects_a_tree_of_another_profile(tmp_path, monkeypatch):
    lst, trees = _real_list(tmp_path)
    real_env = P2.P._env_of
    odd = os.path.abspath(trees[3])

    def env_of(ctx):
        e = real_env(ctx)
        if os.path.abspath(ctx.tree.root) == odd:
            e = dict(e, hw_profile_sha="0" * 64)
        return e
    monkeypatch.setattr(P2.P, "_env_of", env_of)
    with pytest.raises(P2.Pilot2Error, match="environment .* is not unique"):
        P2.build_d14(P2.read_list(lst), REAL_W, REAL_RULE)


def test_d14_rule_must_be_registered_as_d14_rule(tmp_path, fake_trees):
    fw = FakeW(tmp_path)
    t = fw.tree("A", verdicts([tier(2)]))
    doc = {"schema": "capkit.d14_rule/1", "idle_softirq_frac_max": 0.002,
           "load_min_excess_frac": 0.005, "monotonic_slack_frac": 0.002, "min_trees": 1}
    rule = fw.put("register/rule.json", doc, "pilot")
    lst = fw.listfile("l.tsv", [(t, "2", "subsat")])
    o = fw.path("d.json")
    assert P2.main(["d14", "--workdir", fw.W, "--list", lst, "--rule", rule, "--out", o]) == 3
    assert not os.path.exists(o)
    good = fw.put("register/rule2.json", dict(doc, min_trees=2), "d14_rule")
    lst = fw.listfile("all.tsv", [(t, "ALL", "subsat")])
    with pytest.raises(P2.Pilot2Error, match="explicit tier seqs"):
        P2.build_d14(P2.read_list(lst), fw.W, good)


def test_d14_needs_one_judge_sha(tmp_path, fake_trees):
    fw = FakeW(tmp_path)
    a = fw.tree("A", verdicts([tier(2)]))
    b = fw.tree("B", verdicts([tier(2)]), judge_sha="a" * 40)
    doc = {"schema": "capkit.d14_rule/1", "idle_softirq_frac_max": 0.002,
           "load_min_excess_frac": 0.005, "monotonic_slack_frac": 0.002, "min_trees": 1}
    rule = fw.put("register/rule.json", doc, "d14_rule")
    lst = fw.listfile("l.tsv", [(a, "2", "subsat"), (b, "2", "subsat")])
    with pytest.raises(P2.Pilot2Error, match="judge code sha is not unique"):
        P2.build_d14(P2.read_list(lst), fw.W, rule)


def test_shortfall_product_feeds_laddergen_plan_from(tmp_path, fake_trees):
    """The e0_params shortfall items are what laddergen --plan-from reads (post_gate:shortfall
    job plan, pin and arm kept)."""
    from caplaunch import laddergen
    s = segment31(tmp_path)
    run31(s)
    code, d, o = shortfall(s, sel_list(s), out="register/sf.json")
    assert code == 0
    ledger.registry_append(s.fw.path("REGISTRY.tsv"), "E0", "register/sf.json", sha256_file(o),
                           "kind=e0_params;test", registered_at=s.fw.ts())
    out = s.fw.path("register/plan.sf1.tsv")
    assert laddergen.main(["--plan-from", o, "--workdir", s.fw.W, "--phase-id", "S",
                           "--seed-base", "t", "--launches-per-job", "4", "--out", out]) == 0
    items = jobplan.read(out)
    assert {it["item_class"] for it in items} == {"post_gate:shortfall"}
    assert sum(it["launches"] for it in items) == d["items"][0]["launches"] == 7
    assert len(items) == d["cap"]["jobs"] == 2
    assert all(it["arms"] == ["a"] and it["pin"] is False for it in items)
