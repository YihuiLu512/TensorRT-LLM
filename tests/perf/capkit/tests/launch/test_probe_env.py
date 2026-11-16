# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the orchestrator gives product roles and which processes it expects snapshots from
(INTERFACES.md errata R4-1..R4-3, FT1-18), checked on the orchestrator's own methods without
processes."""

import json
import os
import types

import pytest

import conftest as cf
from capcommon import inject as cinject
from capcommon.exitcodes import ConfigError
from capprobe import config as probe_config
from caplaunch import orchestrator as orch

CAPKIT_SHA = "1" * 40
TARGET = "d" * 40
INJECT = "site=F2,mode=busy,delta_ns=20000,every_n=1"


def bare_launch(tier, groups=(), snapshots="all", params=None, target=None, inject=()):
    """A Launch with only the fields these methods read (no inputs, no tree, no processes).
    inject: tier-table launch_inject spec dicts (with role)."""
    L = object.__new__(orch.Launch)
    L.tt = types.SimpleNamespace(
        launch_inject=[cinject.InjectSpec.from_dict(d, require_role=True) for d in inject])
    roles = cf.default_roles(with_observers=False)
    for r in roles:
        if r["launcher"] == "product":
            r.update(probe_tier=tier, snapshots=snapshots)
    L.roles = {r["role"]: r for r in roles}
    L.scen = {"probe_tier": tier, "probe_groups": list(groups)}
    L.params = dict(params or {})
    L.a = types.SimpleNamespace(capkit_sha=CAPKIT_SHA, snapshot_interval_ms=40)
    L.overlay_target = target
    L.mark = "%s=x" % orch.MARK_ENV
    L.shm_dir = "/shm/x"
    L.tree = types.SimpleNamespace(procs=lambda *p: os.path.join("/T/procs", *p))
    L.procs = [{"pid": 10, "role": "product", "discovered_via": "launched"},
               {"pid": 11, "role": "product", "discovered_via": "probe.snapshot"},
               {"pid": 20, "role": "nullsse", "discovered_via": "launched"}]
    return L


@pytest.fixture
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("CAPPROBE_") or k == "CAPKIT_CODE_SHA":
            monkeypatch.delenv(k)
    return monkeypatch


PRODUCT_F2 = {"role": "product", "site": "F2", "mode": "busy", "delta_ns": 20000}
TOOL_SPEC = {"role": "nullsse", "site": "send.chunk", "mode": "busy", "delta_ns": 5}


@pytest.mark.parametrize("tier,scope,inject,expected", [
    ("P0", "none", (), [20]), ("P0", "all", (), [20]), ("P0", "workers", (), [20]),
    ("P1", "none", (), [20]), ("P1", "all", (), [20]), ("P1", "workers", (), [20]),
    ("P1", "all", (TOOL_SPEC,), [20]),
    ("P2", "all", (), [10, 11, 20]),
    ("P1", "all", (PRODUCT_F2,), [10, 11, 20]), ("P2", "all", (PRODUCT_F2,), [10, 11, 20]),
])
def test_expected_snapshot_writers_follow_what_writes(tier, scope, inject, expected):
    """S1 / RS19: a P0 or P1 product role without a registered injection never writes snapshots
    (no probe group, no CAPPROBE_INJECT), so whatever its scope says it is not expected to; a P2
    role, or a P1 / P2 role named in launch_inject (FT1-18; scope all), is, top process
    included. An injection on a capkit role changes nothing for the product role."""
    L = bare_launch(tier, ["count"] if tier == "P2" else [], scope, inject=inject)
    assert [p["pid"] for p in L._snapshot_procs()] == expected


@pytest.mark.parametrize("tier", ["P1", "P2"])
def test_registered_product_inject_reaches_only_its_product_role(clean_env, tier):
    """FT1-18: the tier table's launch_inject entry for a product role becomes exactly that
    role's CAPPROBE_INJECT (role dropped), in a form capprobe's own parser accepts with the same
    spec; capkit roles never get one, and a launch without such an entry gives none."""
    spec2 = dict(PRODUCT_F2, site="F4", mode="sleep", every_n=3, phase=2)
    L = bare_launch(tier, ["count"] if tier == "P2" else [], inject=(PRODUCT_F2, spec2,
                                                                     TOOL_SPEC),
                    target=TARGET)
    env = L._role_env(L.roles["product"])
    got = cinject.parse_inject_specs(env["CAPPROBE_INJECT"])
    want = [cinject.InjectSpec.from_dict({k: v for k, v in d.items() if k != "role"})
            for d in (PRODUCT_F2, spec2)]
    assert got == want
    _tier, _groups, specs, _entry = probe_config._static_checks(env)
    assert specs == want
    assert "CAPPROBE_INJECT" not in L._role_env(L.roles["nullsse"])
    none = bare_launch(tier, ["count"] if tier == "P2" else [], inject=(TOOL_SPEC,),
                       target=TARGET)
    assert "CAPPROBE_INJECT" not in none._role_env(none.roles["product"])


def test_p2_lag_product_env_has_every_switch_capprobe_needs(clean_env):
    """S2 / RS20: with the lag group the product role gets CAPPROBE_LAG_PERIOD_US from the
    scenario param lag_period_us; every role gets CAPKIT_CODE_SHA; P1/P2 roles the patch sha."""
    L = bare_launch("P2", ["count", "lag"], params={"lag_period_us": "1000"}, target=TARGET)
    env = L._role_env(L.roles["product"])
    assert env["CAPPROBE_TIER"] == "P2" and env["CAPPROBE_GROUPS"] == "count,lag"
    assert probe_config._positive_int(env, probe_config.ENV_LAG_PERIOD_US) == 1000
    assert env["CAPKIT_CODE_SHA"] == CAPKIT_SHA and env["CAPPROBE_PATCH_SHA"] == TARGET
    tool = L._role_env(L.roles["nullsse"])
    assert tool["CAPKIT_CODE_SHA"] == CAPKIT_SHA and tool["CAPPROBE_PATCH_SHA"] == TARGET
    assert "CAPPROBE_LAG_PERIOD_US" not in tool and "CAPPROBE_TIER" not in tool
    no_lag = bare_launch("P2", ["count"], target=TARGET)._role_env(L.roles["product"])
    assert "CAPPROBE_LAG_PERIOD_US" not in no_lag


def test_p0_launch_drops_an_outer_patch_sha(clean_env):
    clean_env.setenv("CAPPROBE_PATCH_SHA", "stale")
    L = bare_launch("P0", snapshots="none")
    for r in L.roles.values():
        env = L._role_env(r)
        assert "CAPPROBE_PATCH_SHA" not in env and env["CAPKIT_CODE_SHA"] == CAPKIT_SHA


@pytest.mark.parametrize("key,val,tier,needle", [
    ("CAPPROBE_INJECT", INJECT, "P0", "CAPPROBE_INJECT"),
    ("CAPPROBE_INJECT", INJECT, "P2", "CAPPROBE_INJECT"),
    ("CAPKIT_CODE_SHA", "2" * 40, "P0", "CAPKIT_CODE_SHA"),
    ("CAPPROBE_PATCH_SHA", "e" * 40, "P2", "CAPPROBE_PATCH_SHA"),
    ("CAPPROBE_LAG_PERIOD_US", "999", "P2", "CAPPROBE_LAG_PERIOD_US"),
    ("CAPPROBE_TIER", "P1", "P2", "CAPPROBE_TIER"),
])
def test_outer_values_that_contradict_the_orchestrator_are_refused(clean_env, key, val, tier,
                                                                   needle):
    L = bare_launch(tier, ["count", "lag"] if tier == "P2" else [],
                    snapshots="all" if tier == "P2" else "none",
                    params={"lag_period_us": "1000"}, target=TARGET if tier == "P2" else None)
    L._check_outer_probe_env()
    clean_env.setenv(key, val)
    with pytest.raises(ConfigError, match=needle):
        L._check_outer_probe_env()


def test_outer_values_that_agree_are_accepted(clean_env):
    L = bare_launch("P2", ["count", "lag"], params={"lag_period_us": "1000"}, target=TARGET)
    for k, v in (("CAPKIT_CODE_SHA", CAPKIT_SHA), ("CAPPROBE_PATCH_SHA", TARGET),
                 ("CAPPROBE_LAG_PERIOD_US", "1000"), ("CAPPROBE_TIER", "P2"),
                 ("CAPPROBE_GROUPS", "count,lag")):
        clean_env.setenv(k, v)
    L._check_outer_probe_env()


def test_outer_inject_without_product_roles_is_not_checked(clean_env):
    L = bare_launch("P0")
    L.roles = {n: r for n, r in L.roles.items() if r["launcher"] != "product"}
    clean_env.setenv("CAPPROBE_INJECT", INJECT)
    L._check_outer_probe_env()


@pytest.mark.parametrize("doc,ok", [
    ({"target_commit": TARGET}, True), ({"target_commit": None}, False), ({}, False),
    ([], False),
])
def test_patch_sha_comes_from_the_overlay_manifest(tmp_path, doc, ok):
    """RS20: the P1/P2 patch sha has one source, the overlay manifest's target_commit."""
    p = tmp_path / "overlay.json"
    p.write_text(json.dumps(doc))
    L = bare_launch("P2", ["count"])
    L.a = types.SimpleNamespace(overlay_manifest=str(p), job_env_dir=str(tmp_path / "none"),
                                capkit_sha=CAPKIT_SHA)
    if ok:
        L._load_overlay()
        assert L.overlay_target == TARGET
    else:
        with pytest.raises(ConfigError, match="target_commit|JSON object"):
            L._load_overlay()
    L.a.overlay_manifest = str(tmp_path / "missing.json")  # handled later as NOT_RUN
    L._load_overlay()
    assert L.overlay_target is None and L.overlay_record["present"] is False
