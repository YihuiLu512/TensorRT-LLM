# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analysis scripts: capjudge.boundary (T4 per launch, envelope, T5), capjudge.scaling (T6,
opposite-side doubling attribution) and the .cmd.txt recompute records (INTERFACES 14)."""

import json
import os
import shutil
import subprocess
import sys

import pytest

from conftest import SAMPLES, TOOLS, XFAIL_I_P2_13, judge_mod

B = judge_mod("boundary")
S = judge_mod("scaling")
JENV = dict(os.environ, PYTHONPATH=TOOLS, CAPKIT_CODE_SHA="0" * 40)


# ---- per-launch interval ----------------------------------------------------------------------

def _t(seq, verdict, lv=None, red=(), I=None):
    return {"tier_seq": seq, "judged": True, "order_key": (lv or seq * 100.0, seq),
            "load_value": lv or seq * 100.0, "verdict": verdict,
            "checks": {c: "RED" for c in red}, "X": {"I": I or {"status": "NOT_APPLICABLE"}}}


def _verdicts(tiers):
    v = {"tiers": tiers}
    v["boundary_T4"] = B._t4_of(B._judged_tiers(v))
    return v


def test_interval_is_the_judges_t4_without_a_tool_limit():
    v = _verdicts([_t(2, "STABLE"), _t(4, "NOT_RUN"), _t(6, "UNSTABLE"), _t(8, "UNSTABLE")])
    t4, used = B.launch_interval(v)
    assert (t4["lower"], t4["upper"]) == (200.0, 600.0)
    assert (used["lower"], used["upper"], used["censored_by"]) == (200.0, 600.0, None)
    assert used["skipped"] == [{"tier_seq": 4, "verdict": "NOT_RUN"}]


@pytest.mark.parametrize("tool_tier", [
    _t(4, "VOID", red=("C1",)),
    _t(4, "UNSTABLE", I={"status": "LOCALIZED", "role": "ld", "measure_class": "load_source"}),
])
def test_tool_limit_below_the_first_unstable_tier_censors_the_upper_end(tool_tier):
    v = _verdicts([_t(2, "STABLE"), tool_tier, _t(6, "UNSTABLE")])
    t4, used = B.launch_interval(v)
    assert used["upper"] is None and used["lower"] == 200.0
    assert used["kind"] == "lower_bound_only" and used["censored_by"]["tier_seq"] == 4
    assert t4["upper"] in (400.0, 600.0)  # the judge's own T4 is reported unchanged


def test_tool_limit_above_the_boundary_keeps_the_interval_and_other_void_is_skipped():
    v = _verdicts([_t(2, "STABLE"), _t(4, "VOID", red=("C3",)), _t(6, "UNSTABLE"),
                   _t(8, "VOID", red=("C2",))])
    _t4, used = B.launch_interval(v)
    assert (used["lower"], used["upper"]) == (200.0, 600.0)
    assert used["censored_by"]["tier_seq"] == 8
    v2 = _verdicts([_t(2, "STABLE"), _t(4, "VOID", red=("C1", "C3"))])  # not only C1 / C2
    assert B.tool_limited(v2["tiers"][1]) is None
    assert B.launch_interval(v2)[1]["censored_by"] is None


def test_recorded_t4_must_follow_from_the_tiers():
    v = _verdicts([_t(2, "STABLE"), _t(6, "UNSTABLE")])
    v["boundary_T4"]["upper"] = 999.0
    with pytest.raises(B.BoundaryError, match="boundary_T4.upper"):
        B.launch_interval(v)


# ---- arms, envelopes and T5 -------------------------------------------------------------------

def _env(lo, hi):
    return {"lower": lo, "upper": hi}


def test_compare_uses_common_blocks_only():
    arms = {"a": {"by_block": {"1": _env(100, 200), "2": _env(110, 210), "3": _env(1, 2)}},
            "b": {"by_block": {"1": _env(300, 400), "2": _env(320, 420)}}}
    r = B.compare(arms, "a", "b")
    assert r["blocks"] == ["1", "2"] and r["blocks_missing"] == ["3"]
    assert r["result"] == "different" and r["t5"]["direction"] == "b_higher"
    arms["b"]["by_block"]["2"] = _env(150, 420)  # overlaps in one block
    assert B.compare(arms, "a", "b")["result"] == "indistinguishable"
    arms["c"] = {"by_block": {"9": _env(1, 2)}}
    assert B.compare(arms, "a", "c")["result"] == "NOT_RUN"
    # every common block has an open end: nothing to compare, not "indistinguishable"
    arms["d"] = {"by_block": {"1": _env(100, None), "2": _env(110, None)}}
    r = B.compare(arms, "a", "d")
    assert r["result"] == "NOT_RUN" and r["t5"]["min_resolvable_diff"] is None
    with pytest.raises(B.BoundaryError):
        B.compare(arms, "a", "zz")


@pytest.mark.parametrize("bad", ["a", "a,a", "a,b,c", ",b"])
def test_parse_compare_rejects(bad):
    with pytest.raises(B.BoundaryError):
        B.parse_compare([bad])


# ---- scaling -----------------------------------------------------------------------------------

def _roles(w, z):
    return {"ld": {"kind": "client", "measure_class": "under_test", "workers": w},
            "sink": {"kind": "server", "measure_class": "stub_out_of_domain", "workers": z},
            "sidecar": {"kind": "aux", "measure_class": "observer", "workers": 1}}


def _bdoc(arms, cut=None):
    """arms: {arm: ((w, z), {block: (lo, hi)})} - the judge's T4 per block (by_block_t4);
    cut: {arm: {block: (lo, hi)}} the interval used across launches where it differs (a tool
    limit cut it), else the same as T4."""
    doc = {"schema": B.SCHEMA, "axis": {"name": "R", "load_key": "rate_rps", "unit": "req/s",
                                        "order": "asc"}, "launches": [], "arms": {}}
    for arm, ((w, z), blocks) in arms.items():
        for blk in blocks:
            doc["launches"].append({"identity": {"arm": arm}, "block": blk, "roles": _roles(w, z)})
        used = dict(blocks, **(cut or {}).get(arm, {}))
        doc["arms"][arm] = {"by_block": {b: _env(*iv) for b, iv in used.items()},
                            "by_block_t4": {b: _env(*iv) for b, iv in blocks.items()}}
    return doc


def test_scaled_side_and_factor():
    assert S.scaled_side(_roles(1, 4), _roles(2, 4))[:2] == ("client", 2.0)
    assert S.scaled_side(_roles(1, 4), _roles(1, 8))[:2] == ("server", 2.0)
    for scaled in (_roles(2, 8), _roles(1, 2), _roles(1, 4)):
        with pytest.raises(S.ScalingError):
            S.scaled_side(_roles(1, 4), scaled)
    base, scaled = _roles(1, 4), _roles(1, 8)  # a second server role left unscaled
    base["gsink"] = scaled["gsink"] = dict(base["sink"])
    with pytest.raises(S.ScalingError, match="not scaled"):
        S.scaled_side(base, scaled)


def _pairs(w_scaled, z_scaled):
    doc = _bdoc({"w1z4": ((1, 4), {"1": (1000, 1200), "2": (1000, 1100)}),
                 "w2z4": ((2, 4), {"1": w_scaled, "2": w_scaled}),
                 "w1z8": ((1, 8), {"1": z_scaled, "2": z_scaled})})
    return [S.pair(doc, "w1z4", "w2z4", 1.6, 1.25), S.pair(doc, "w1z4", "w1z8", 1.6, 1.25)]


def test_loadgen_limited_when_w_grows_and_z_is_flat():
    pw, pz = _pairs((2000, 2400), (1000, 1200))
    assert pw["claim"] == "grows" and pw["limit_side"] == "client"
    assert pw["ratio"] == {"min": 2000 / 1200, "max": 2400 / 1000, "min_unknown_blocks": [],
                           "max_unknown_blocks": []}
    assert pw["per_block"][0]["efficiency_min"] == pytest.approx(2000 / 1200 / 2)
    assert pz["claim"] == "flat" and pz["limit_side"] == "client"
    assert S.attribute([pw, pz])["result"] == "client"


def test_sink_limited_mirror_and_contradictions():
    pw, pz = _pairs((1000, 1200), (2000, 2400))
    assert S.attribute([pw, pz])["result"] == "server"
    pw, pz = _pairs((2000, 2400), (2000, 2400))  # both grow
    assert S.attribute([pw, pz])["result"] == "UNDETERMINED"
    pw, pz = _pairs((1300, 1500), (1000, 1200))  # W unresolved
    assert pw["claim"] == "unresolved"
    assert S.attribute([pw, pz])["result"] == "NOT_RUN"
    assert S.attribute([pz])["result"] == "NOT_RUN"


def test_scaling_reads_the_judges_t4_not_the_cut_interval():
    # E1 sink limit: the out-of-domain sink saturates first, I localises the unstable tier to
    # it, so the interval used across launches is cut to a lower bound; the doubling must
    # still see the closed T4 to attribute the limit to the server side
    sink_cut = {"w1z4": {"1": (1000, None)}, "w2z4": {"1": (1000, None)}}
    doc = _bdoc({"w1z4": ((1, 4), {"1": (1000, 1200)}), "w2z4": ((2, 4), {"1": (1000, 1200)}),
                 "w1z8": ((1, 8), {"1": (2000, 2400)})}, cut=sink_cut)
    pw, pz = S.pair(doc, "w1z4", "w2z4", 1.6, 1.25), S.pair(doc, "w1z4", "w1z8", 1.6, 1.25)
    assert (pw["claim"], pz["claim"]) == ("flat", "grows")
    assert S.attribute([pw, pz])["result"] == "server"
    del doc["arms"]["w1z8"]["by_block_t4"]
    with pytest.raises(S.ScalingError, match="by_block_t4"):
        S.pair(doc, "w1z4", "w1z8", 1.6, 1.25)


def test_open_ends_decide_what_can_be_claimed():
    pw, _pz = _pairs((2000, None), (1000, 1200))  # scaled arm never became unstable
    assert pw["claim"] == "grows" and pw["ratio"]["max"] is None
    assert pw["ratio"]["max_unknown_blocks"] == ["1", "2"]
    doc = _bdoc({"a": ((1, 4), {"1": (1000, None)}), "b": ((1, 8), {"1": (1000, 1100)})})
    r = S.pair(doc, "a", "b", 1.6, 1.25)
    assert r["claim"] == "NOT_RUN" and r["ratio"] is None
    doc["axis"]["order"] = "desc"
    with pytest.raises(S.ScalingError):
        S.pair(doc, "a", "b", 1.6, 1.25)


def test_a_block_with_an_unknown_end_is_never_skipped():
    # block 2's base never became unstable (upper open): its lower ratio is unknown, its upper
    # ratio 3.2 says the doubling may have tripled the boundary there; block 1 alone looks flat
    doc = _bdoc({"base": ((1, 4), {"1": (100, 110), "2": (100, None)}),
                 "x2": ((1, 8), {"1": (100, 110), "2": (300, 320)})})
    r = S.pair(doc, "base", "x2", 1.5, 1.2)
    assert r["claim"] == "unresolved" and r["limit_side"] is None, r
    assert (r["ratio"]["min"], r["ratio"]["min_unknown_blocks"]) == (None, ["2"])
    assert r["ratio"]["max"] == 3.2
    # the scaled arm's upper end open in one block: no "flat" from the others
    doc = _bdoc({"base": ((1, 4), {"1": (100, 110), "2": (100, 110)}),
                 "x2": ((1, 8), {"1": (100, 110), "2": (100, None)})})
    r = S.pair(doc, "base", "x2", 1.5, 1.2)
    assert r["claim"] == "unresolved" and r["ratio"]["max_unknown_blocks"] == ["2"]
    # both ends known everywhere: flat
    doc["arms"]["x2"]["by_block_t4"]["2"] = _env(100, 110)
    assert S.pair(doc, "base", "x2", 1.5, 1.2)["claim"] == "flat"


# ---- CLIs on judged samples ---------------------------------------------------------------------

def _judged(tmp_path, name):
    W = str(tmp_path / name)
    shutil.copytree(os.path.join(SAMPLES, name), W)
    exp = json.load(open(os.path.join(W, "expect.json")))
    tree = os.path.join(W, exp["tree"])
    p = subprocess.run([sys.executable, "-m", "capjudge", "--tree", tree, "--workdir", W],
                       env=JENV, capture_output=True, text=True, timeout=300, check=False)
    assert p.returncode == exp["exit_code"], p.stderr
    return W, tree


def _run(mod, *args):
    return subprocess.run([sys.executable, "-m", mod] + list(args), env=JENV,
                          capture_output=True, text=True, timeout=300, check=False)


@XFAIL_I_P2_13
def test_boundary_cli_writes_product_and_record(tmp_path):
    W, tree = _judged(tmp_path, "i_server_slow")
    lst = tmp_path / "trees.txt"
    lst.write_text(tree + "\n")
    out = str(tmp_path / "boundary.json")
    p = _run("capjudge.boundary", "--inputs", str(lst), "--workdir", W, "--out", out)
    assert p.returncode == 0, p.stderr
    doc = json.load(open(out))
    v = json.load(open(os.path.join(tree, "judge", "verdicts.json")))
    (x,) = doc["launches"]
    assert x["t4"]["upper"] == v["boundary_T4"]["upper"] and x["interval"]["censored_by"] is None
    assert x["grade"]["grade"] == "LOCALIZED" and doc["excluded"] == []
    arm = v["identity"]["arm"]
    assert doc["arms"][arm]["envelope"]["upper"] == v["boundary_T4"]["upper"]
    assert doc["arms"][arm]["envelope_t4"]["upper"] == v["boundary_T4"]["upper"]
    assert doc["arms"][arm]["by_block_t4"] == doc["arms"][arm]["by_block"]
    rec = open(str(tmp_path / "boundary.cmd.txt")).read()
    assert "\ncommand\t" in rec and "exit_code\t0" in rec and "code_sha\t" + "0" * 40 in rec
    assert "input\t%s\t" % os.path.join(tree, "judge", "verdicts.json") in rec
    p2 = _run("capjudge.boundary", "--inputs", str(lst), "--workdir", W, "--out", out)
    assert p2.returncode == 3 and "exists" in p2.stderr


@XFAIL_I_P2_13
def test_boundary_keeps_the_judges_t4_next_to_the_cut_interval(tmp_path):
    W, tree = _judged(tmp_path, "i_reader_slow")  # tier 4 localised to the load source
    lst = tmp_path / "trees.txt"
    lst.write_text(tree + "\n")
    out = str(tmp_path / "boundary.json")
    assert _run("capjudge.boundary", "--inputs", str(lst), "--workdir", W,
                "--out", out).returncode == 0
    doc = json.load(open(out))
    (arm,) = doc["arms"].values()
    v = json.load(open(os.path.join(tree, "judge", "verdicts.json")))
    assert arm["envelope"]["upper"] is None
    assert arm["envelope_t4"]["upper"] == v["boundary_T4"]["upper"] is not None
    (blk,) = arm["by_block_t4"].values()
    assert blk["upper"] == v["boundary_T4"]["upper"]
    assert all(b["upper"] is None for b in arm["by_block"].values())


def test_boundary_cli_excludes_a_launch_marked_void_after_judging(tmp_path):
    W, tree = _judged(tmp_path, "good")
    with open(os.path.join(tree, "STATE"), "a") as f:
        f.write("2026-11-14T00:00:00.000Z\tVOID\tcapjobs.launches\tcopy-back mismatch\n")
    lst = tmp_path / "trees.txt"
    lst.write_text(tree + "\n")
    p = _run("capjudge.boundary", "--inputs", str(lst), "--workdir", W, "--out",
             str(tmp_path / "b.json"))
    assert p.returncode == 3 and "no launch left" in p.stderr
    assert not os.path.exists(str(tmp_path / "b.json"))
    assert not os.path.exists(str(tmp_path / "b.cmd.txt"))


def test_boundary_cli_rejects_uncommitted_judgements(tmp_path):
    W, tree = _judged(tmp_path, "good")
    vp = os.path.join(tree, "judge", "verdicts.json")
    v = json.load(open(vp))
    v["note"] = "edited after judging"
    json.dump(v, open(vp, "w"))
    lst = tmp_path / "trees.txt"
    lst.write_text(tree + "\n")
    p = _run("capjudge.boundary", "--inputs", str(lst), "--workdir", W, "--out",
             str(tmp_path / "b.json"))
    assert p.returncode == 3 and "JUDGED.tsv" in p.stderr


def test_scaling_cli_and_wrapper(tmp_path):
    bf = str(tmp_path / "boundary.json")
    doc = _bdoc({"w1z4": ((1, 4), {"1": (1000, 1200)}), "w2z4": ((2, 4), {"1": (2000, 2400)})})
    json.dump(doc, open(bf, "w"))
    out = str(tmp_path / "scaling.json")
    wrapper = os.path.join(os.path.dirname(TOOLS), "analysis", "scaling.py")
    p = subprocess.run([sys.executable, wrapper, "--boundary", bf, "--pair", "w1z4,w2z4",
                        "--grows-min", "1.6", "--flat-max", "1.25", "--out", out],
                       env=dict(os.environ, CAPKIT_CODE_SHA="0" * 40), capture_output=True,
                       text=True, timeout=300, check=False)
    assert p.returncode == 0, p.stderr
    d = json.load(open(out))
    assert d["pairs"][0]["claim"] == "grows" and d["attribution"]["result"] == "NOT_RUN"
    rec = open(str(tmp_path / "scaling.cmd.txt")).read()
    assert "invoked\t" in rec and "analysis/scaling.py" in rec and "input\t%s\t" % bf in rec
    p = _run("capjudge.scaling", "--boundary", bf, "--pair", "w1z4,w2z4", "--grows-min",
             "1.0", "--flat-max", "1.25", "--out", str(tmp_path / "s2.json"))
    assert p.returncode == 3 and "flat-max" in p.stderr


# ---- E1 report: slope reconciliation ----------------------------------------------------------

E = judge_mod("e1report")


def _slope_launch(arm, block, lo, hi, delta_ns=None, hb="PASS", mode="P+X+I+V"):
    doses = {} if delta_ns is None else {"ld": [{"site": "recv.chunk", "mode": "busy",
                                                 "delta_ns": delta_ns, "every_n": 1,
                                                 "phase": 0, "where": "inline"}]}
    return {"tree": "/t/%s-%s" % (arm, block), "identity": {"arm": arm}, "block": block,
            "mode": mode, "interval": {"lower": lo, "upper": hi}, "doses": doses,
            "heartbeat": hb if doses else "NONE", "roles": _roles(1, 4)}


def _slope_doc(points, **kw):
    """points: [(delta_ns or None, lo, hi)]; capacity per worker = 1 / (t0 + delta)."""
    doc = {"axis": {"order": "asc"}, "launches": []}
    for i, (d, lo, hi) in enumerate(points):
        doc["launches"].append(_slope_launch("a%s" % d, str(i), lo, hi, d, **kw))
    return doc


def _known(t0, deltas, slope=1.0, rel=0.02, p=2.0):
    """Intervals of +-rel around the boundary implied by passes p per request and slope."""
    out = []
    for d in deltas:
        cap = 1.0 / (p * (t0 + slope * (d or 0) / 1e9))
        out.append((d, cap * (1 - rel), cap * (1 + rel)))
    return out


@pytest.fixture
def two_passes(monkeypatch):
    monkeypatch.setattr(E, "passes_per_unit", lambda tree, W, role, site: (2.0, None))


def test_slope_pass_when_one_over_boundary_rises_with_unit_slope(two_passes):
    s = E.slope(_slope_doc(_known(100e-6, [None, 100000, 200000, 400000])), "/W")
    assert s["status"] == "PASS", s
    assert [lv["delta_s"] for lv in s["levels"]] == [0.0, 1e-4, 2e-4, 4e-4]
    assert all(p["slope_min"] < 1 < p["slope_max"] for p in s["pairs"])


def test_slope_fail_when_the_slope_is_not_one(two_passes):
    s = E.slope(_slope_doc(_known(100e-6, [None, 100000, 200000, 400000], slope=2.0)), "/W")
    assert s["status"] == "FAIL"


def test_slope_not_run_and_exclusions(two_passes):
    s = E.slope(_slope_doc(_known(100e-6, [None, 100000])), "/W")
    assert s["status"] == "NOT_RUN" and "at least 3" in s["reasons"][0]
    doc = _slope_doc(_known(100e-6, [None, 100000, 200000, 400000]))
    doc["launches"][1]["heartbeat"] = "FAIL"
    doc["launches"].append(_slope_launch("x", "9", 1, 2, mode="P+X"))
    s = E.slope(doc, "/W")
    assert s["status"] == "PASS" and len(s["levels"]) == 3
    assert {x["why"] for x in s["excluded"]} == {"heartbeat FAIL", "mode P+X"}
    doc = _slope_doc(_known(100e-6, [None, 100000, 200000]))
    doc["launches"][-1]["interval"]["upper"] = None
    assert E.slope(doc, "/W")["status"] == "INCONCLUSIVE"


def test_slope_pools_the_dosed_launches_topology_only(two_passes):
    # a control launch of another arm (2 load-generator workers: about twice the boundary)
    # is no delta = 0 point of this arm's dose line
    doc = _slope_doc(_known(100e-6, [None, 100000, 200000, 400000]))
    other = _slope_launch("w2", "8", None, None)
    other.update(roles=_roles(2, 4), interval=dict(
        (k, 2 * v) for k, v in doc["launches"][0]["interval"].items()))
    doc["launches"].append(other)
    s = E.slope(doc, "/W")
    assert s["status"] == "PASS", s
    assert s["levels"][0]["trees"] == [doc["launches"][0]["tree"]]
    assert "another topology" in s["excluded"][0]["why"]
    # dosed launches of two topologies: no single dose line
    doc["launches"][1]["roles"] = _roles(2, 4)
    s = E.slope(doc, "/W")
    assert s["status"] == "NOT_RUN" and "2 topologies" in s["reasons"][0]


def test_slope_interval_holding_zero_is_inconclusive(two_passes):
    # boundaries so noisy that slope intervals span 0 and 1: a dose that did not move the
    # boundary would give such an interval too, so it is not a PASS
    s = E.slope(_slope_doc(_known(100e-6, [None, 100000, 200000, 400000], rel=0.4)), "/W")
    assert all(p["slope_min"] < 1 < p["slope_max"] for p in s["pairs"])  # all contain 1
    zero = [p for p in s["pairs"] if p["slope_min"] <= 0]
    assert zero and all(p["status"] == "INCONCLUSIVE" for p in zero), s["pairs"]
    assert s["status"] == "INCONCLUSIVE"


# ---- E1 report: bytes and accepts -------------------------------------------------------------

def _sink_tree(tmp_path, client_bytes, rids, accepts=(3, 3), foreign=0, failed=0, drop=()):
    """drop: (file, dotted key) summary keys to delete."""
    from array import array
    from capcommon.capbin import write_capbin
    from capload.sink import response_head
    tree = tmp_path / "tree"
    td = tree / "raw" / "tiers" / "002_up"
    td.mkdir(parents=True)
    roles = [{"role": "ld", "kind": "client", "module": "capload",
              "measure_class": "under_test", "workers": 1},
             {"role": "sink", "kind": "server", "module": "capload.nullsse",
              "measure_class": "stub_out_of_domain", "workers": 1}]
    json.dump({"roles": roles}, open(str(tree / "raw" / "launch.json"), "w"))
    heads = sum(len(response_head(r, False)) for r in rids)
    docs = {"ld": {"frames": {"bytes": client_bytes}, "conn_pool": {"new_conns": accepts[0]},
                   "counts": {"failed": failed, "unfinished": 0}},
            "sink": {"counts": {"bytes_sent": 1000 + heads, "requests": len(rids),
                                "accepts": accepts[1], "foreign_requests": foreign},
                     "overflow": {"streams": 0}}}
    for who, key in drop:
        d = docs[who]
        *path, last = key.split(".")
        for k in path:
            d = d[k]
        del d[last]
    for who, doc in docs.items():
        json.dump(doc, open(str(td / ("%s.w0.summary.json" % who)), "w"))
    ident = {"experiment": "E1", "arm": "a", "launch_seq": 0, "job_id": "local-1",
             "restart_count": 0, "register_sha": "0" * 64, "role": "sink", "worker": 0,
             "pid": 1, "tier_seq": None, "tier_label": None}
    write_capbin(str(td / "sink.w0.streams.capbin"), "streams", ident,
                 [("req_id", array("q", rids))])
    return str(tree)


def test_counts_bytes_reconcile_with_rebuilt_heads(tmp_path):
    c = E.counts(_sink_tree(tmp_path, 1000, [7, 12345, 99]))
    assert c["bytes"]["status"] == "PASS" and c["accepts"]["status"] == "PASS"
    assert c["bytes"]["server_bytes"] - c["bytes"]["head_bytes"] == 1000


@pytest.mark.parametrize("kw, part, status", [
    ({"client_bytes": 999}, "bytes", "FAIL"),  # sink sent more, no request failed
    ({"client_bytes": 1001}, "bytes", "FAIL"),  # the load generator received more
    ({"client_bytes": 999, "failed": 2}, "bytes", "NOT_RUN"),  # bytes in flight when dropped
    ({"client_bytes": 999, "drop": [("ld", "counts.failed")]}, "bytes", "NOT_RUN"),
    ({"accepts": (3, 2)}, "accepts", "FAIL"),
    ({"foreign": 1}, "bytes", "NOT_RUN"),
    # a missing counter is never summed as 0
    ({"drop": [("ld", "conn_pool.new_conns")]}, "accepts", "NOT_RUN"),
    ({"drop": [("ld", "conn_pool.new_conns"), ("sink", "counts.accepts")]}, "accepts",
     "NOT_RUN"),
    ({"drop": [("sink", "counts.foreign_requests")]}, "bytes", "NOT_RUN"),
    ({"drop": [("sink", "counts.requests")]}, "bytes", "NOT_RUN"),
    ({"drop": [("sink", "overflow.streams")]}, "bytes", "NOT_RUN"),
])
def test_counts_mismatches(tmp_path, kw, part, status):
    args = {"client_bytes": 1000, "rids": [7, 12345, 99]}
    args.update(kw)
    c = E.counts(_sink_tree(tmp_path, **args))
    assert c[part]["status"] == status, c
    if status == "NOT_RUN" and part == "bytes" and "client_bytes" in kw:
        assert "1 bytes more" in c[part]["why"]


def test_counts_need_every_per_tier_summary(tmp_path):
    tree = _sink_tree(tmp_path, 1000, [7])
    td = os.path.join(tree, "raw", "tiers", "002_up")
    os.makedirs(os.path.join(tree, "raw", "tiers", "003_drain"))  # a tier without summaries
    c = E.counts(tree)
    assert c["bytes"]["status"] == c["accepts"]["status"] == "NOT_RUN"
    assert "1 of 2 per-tier summaries" in c["bytes"]["why"]
    assert os.path.isfile(os.path.join(td, "ld.w0.summary.json"))


def test_counts_need_the_e1_topology(tmp_path):
    tree = _sink_tree(tmp_path, 1000, [1])
    lj = os.path.join(tree, "raw", "launch.json")
    d = json.load(open(lj))
    d["roles"].append({"role": "relay", "kind": "server", "module": "capselfcheck.relay",
                       "measure_class": "under_test", "workers": 1})
    json.dump(d, open(lj, "w"))
    c = E.counts(tree)
    assert c["bytes"]["status"] == "NOT_RUN" and "exactly one" in c["bytes"]["why"]


# ---- E1 report CLI on a judged sample: counts and sink health ---------------------------------

def _e1_cli(tmp_path, W, bdoc_path, out, *extra):
    return _run("capjudge.e1report", "--boundary", bdoc_path, "--workdir", W, "--out", out,
                *extra)


def test_e1_report_cli_sink_health_against_registered_bands(tmp_path):
    from capcommon import ledger
    from capcommon.hashing import sha256_file
    W, tree = _judged(tmp_path, "c2_stub_good")
    lst = tmp_path / "trees.txt"
    lst.write_text(tree + "\n")
    bf = str(tmp_path / "b.json")
    assert _run("capjudge.boundary", "--inputs", str(lst), "--workdir", W,
                "--out", bf).returncode == 0
    sink = next(r["role"] for r in json.load(open(os.path.join(tree, "raw", "launch.json")))
                ["roles"] if r["measure_class"] == "stub_out_of_domain")
    p = _e1_cli(tmp_path, W, bf, str(tmp_path / "r0.json"))
    assert p.returncode == 0, p.stderr
    assert json.load(open(str(tmp_path / "r0.json")))["sink_health"]["status"] == "NOT_RUN"
    launch = json.load(open(os.path.join(tree, "raw", "launch.json")))
    env = {"hw_profile_sha": launch.get("hw_profile_sha"),
           "image_sha": (launch.get("image") or {}).get("sha256")}
    for name, hi, e in (("wide", 1e12, env), ("tight", 0.0, env),
                        ("elsewhere", 1e12, dict(env, hw_profile_sha="f" * 64))):
        bands = {"schema": "capkit.near_boundary_noise/1", "regime": "subsat", "noise_bands": {
            "T3:loop_lag_p99_ns:%s" % sink: {"min": 0, "max": hi, "n": 3, "regime": "subsat"}},
            "environment": e}
        path = os.path.join(W, "%s.json" % name)
        json.dump(bands, open(path, "w"))
        out = str(tmp_path / ("r-%s.json" % name))
        p = _e1_cli(tmp_path, W, bf, out, "--sink-bands", path)
        assert p.returncode == 3 and "not registered" in p.stderr
        ledger.registry_append(os.path.join(W, "REGISTRY.tsv"), "E0", "%s.json" % name,
                               sha256_file(path), "kind=near_boundary_noise")
        p = _e1_cli(tmp_path, W, bf, out, "--sink-bands", path)
        assert p.returncode == 0, p.stderr
        (h,) = json.load(open(out))["sink_health"]
        if name == "elsewhere":  # bands of another hardware profile: never compared
            assert h["status"] == "NOT_RUN" and "calibration environment" in h["why"], h
            assert "tiers" not in h
            continue
        lag = [r["items"]["T3:loop_lag_p99_ns"]["status"] for r in h["tiers"]]
        assert lag and set(lag) == ({"INSIDE"} if name == "wide" else {"OUTSIDE"}), h
        assert all(r["items"]["T3:thread_cpu_cores"]["status"] == "NOT_RUN"
                   for r in h["tiers"])  # no band for it
        assert h["status"] == ("NOT_RUN" if name == "wide" else "OUTSIDE")
        assert "input\t%s\t" % path in open(out[:-5] + ".cmd.txt").read()
