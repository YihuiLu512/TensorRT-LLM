# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registration of product-side injection (INTERFACES.md errata FT1-18): the tier table's
launch_inject naming a product role, checked by caplaunch.scenario.cross_check before any
process starts (no processes here)."""

import copy
import json
import os

import pytest

import conftest as cf
from capcommon.exitcodes import ConfigError
from capcommon.tiers import TierTable
from caplaunch import scenario as sc

F2 = {"role": "product", "site": "F2", "mode": "busy", "delta_ns": 20000}


def _setup(tmp_path, tier="P1", snapshots="all", proc_kinds=("frontend_openengine",),
           groups=(), launch_inject=(), tier_inject=None):
    """(scenario doc, validated roles, TierTable, pin roles) of the default roles with the
    product role at `tier`."""
    roles = cf.default_roles(with_observers=False)
    for r in roles:
        if r["launcher"] == "product":
            r.update(probe_tier=tier, snapshots=snapshots, proc_kinds=list(proc_kinds))
    tdoc = cf.tier_doc(cf.gate([("nullsse", "backlog.active_streams")]))
    tdoc["launch_inject"] = [dict(d) for d in launch_inject]
    if tier_inject is not None:
        tdoc["tiers"][2]["inject"] = [dict(tier_inject)]
    wd = cf.make_workdir(tmp_path, tdoc, roles, probe_tier=tier)
    with open(wd["scenario"]) as f:
        doc = json.load(f)
    doc["probe_groups"] = list(groups)
    rs = sc.validate_scenario(doc)
    tt = TierTable.load(os.path.join(wd["W"], "reg", "tiers.json"))
    with open(wd["pin"]) as f:
        pin = json.load(f)["roles"]
    return doc, rs, tt, pin


def _cross(tmp_path, **kw):
    doc, rs, tt, pin = _setup(tmp_path, **kw)
    return sc.cross_check(doc, rs, tt, pin, 40 * cf.MS), rs, tt


@pytest.mark.parametrize("tier,groups,specs", [
    ("P1", (), [F2]),
    ("P1", (), [dict(F2, site="F4", mode="sleep", every_n=4, phase=3)]),
    ("P1", (), [dict(F2, where="aux_thread")]),
    ("P1", (), [F2, dict(F2, site="F4")]),
    ("P2", ("count",), [{"role": "product", "site": "F2", "mode": "miscount",
                         "miscount_delta": -1}]),
])
def test_registered_product_injection_is_accepted(tmp_path, tier, groups, specs):
    _cpus, rs, tt = _cross(tmp_path, tier=tier, groups=groups, launch_inject=specs)
    got = sc.product_inject_specs(rs["product"], tt)
    assert [s.to_dict() for s in got] == [s.to_dict() for s in tt.launch_inject]
    assert sc.product_writes_snapshots(rs["product"], tt) is True
    assert sc.product_inject_specs(rs["nullsse"], tt) == []
    assert sc.product_writes_snapshots(rs["nullsse"], tt) is False


@pytest.mark.parametrize("kw,needle", [
    ({"tier": "P0", "snapshots": "none"}, "probe_tier P1 or P2"),
    ({"tier": "P0", "snapshots": "all"}, "probe_tier P1 or P2"),
    ({"snapshots": "none"}, "snapshots must be all"),
    ({"snapshots": "workers"}, "snapshots must be all"),
    ({"spec": dict(F2, site="G3")}, "capprobe injects only at"),
    ({"spec": dict(F2, site="send.msg")}, "capprobe injects only at"),
    ({"spec": dict(F2, site="Lfe"), "proc_kinds": ("frontend_openengine", "frontend_single")},
     "capprobe injects only at"),  # a capprobe site, but a lag install point, not a count site
    ({"spec": {"role": "product", "site": "F2", "mode": "drop"}}, "not supported in product"),
    ({"spec": {"role": "product", "site": "F2", "mode": "short", "short_tokens": 1}},
     "not supported in product"),
    ({"spec": {"role": "product", "site": "F2", "mode": "miscount", "miscount_delta": 1}},
     "count probe group"),
    ({"spec": dict(F2, site="H1")}, "process kind frontend,"),
    ({"proc_kinds": ()}, "process kind frontend_openengine"),
    ({"proc_kinds": ("frontend",)}, "process kind frontend_openengine"),
])
def test_unusable_product_injection_is_refused_before_start(tmp_path, kw, needle):
    """Each rule's negative case: the switch would reach nothing (P0: no patch), its heartbeat
    could only stay at zero (site outside the declared kinds), capprobe would refuse it at import
    (site, mode, miscount without count), or the snapshots it writes would not be copied."""
    kw = dict(kw)
    spec = kw.pop("spec", F2)
    with pytest.raises(ConfigError, match=needle):
        _cross(tmp_path, launch_inject=[spec], **kw)


def test_per_tier_product_injection_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="launch-level only"):
        _cross(tmp_path, tier_inject=F2)


def test_capkit_role_injection_is_left_to_the_tool(tmp_path):
    """Specs naming a capkit role keep their old path (the tool checks its own sites at start):
    no product rule applies, the product role gets no injection and writes no snapshot."""
    spec = {"role": "nullsse", "site": "send.chunk", "mode": "busy", "delta_ns": 5}
    _cpus, rs, tt = _cross(tmp_path, tier="P1", snapshots="none", proc_kinds=(),
                           launch_inject=[spec], tier_inject=spec | {"site": "other"})
    assert sc.product_inject_specs(rs["product"], tt) == []
    assert sc.product_writes_snapshots(rs["product"], tt) is False


@pytest.mark.parametrize("tier,has_spec,with_tt,expected", [
    ("P0", False, True, False), ("P1", False, True, False), ("P2", False, True, True),
    ("P1", True, True, True), ("P2", True, True, True),
    ("P1", True, False, False), ("P2", True, False, True),
])
def test_product_writes_snapshots_rule(tier, has_spec, with_tt, expected):
    """RS19: every value of the rule. Without the tier table only the P2 rule can apply."""
    r = copy.deepcopy([x for x in cf.default_roles(with_observers=False)
                       if x["launcher"] == "product"][0])
    r["probe_tier"] = tier
    tdoc = cf.tier_doc(cf.gate([("nullsse", "backlog.active_streams")]))
    tdoc["launch_inject"] = [F2] if has_spec else []
    tt = TierTable.from_doc(tdoc)
    assert sc.product_writes_snapshots(r, tt if with_tt else None) is expected
