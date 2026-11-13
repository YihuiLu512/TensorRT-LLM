# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capprobe.config: every switch rule (pure function, no subprocess)."""

import json

import pytest
from probe_testutil import IDENTITY

from capprobe import sites
from capprobe.config import CapprobeConfigError, parse_config
from capcommon.snapshot import validate_slot_decls


def env(identity_file, **kw):
    base = {"CAPPROBE_TIER": "P2", "CAPPROBE_GROUPS": "count", "CAPPROBE_SHM_DIR": "/dev/shm/x",
            "CAPPROBE_INTERVAL_MS": "1000", "CAPPROBE_IDENTITY_FILE": identity_file}
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


def test_good_p2(identity_file):
    cfg = parse_config(env(identity_file, CAPPROBE_GROUPS="lag,count",
                           CAPPROBE_LAG_PERIOD_US="500", CAPPROBE_ROLE_HINT="frontend"))
    assert cfg.groups == ("count", "lag")  # canonical order
    assert cfg.count and cfg.lag and cfg.active
    assert cfg.interval_ns == 1_000_000_000 and cfg.lag_period_ns == 500_000
    assert cfg.identity["experiment"] == "E0" and cfg.role_hint == "frontend"


def test_p1_all_off_needs_nothing_else():
    cfg = parse_config({"CAPPROBE_TIER": "P1"})
    assert not cfg.active and cfg.groups == () and cfg.shm_dir is None


@pytest.mark.parametrize("tier", [None, "", "P3", "p2"])
def test_tier_required_and_valid(identity_file, tier):
    with pytest.raises(CapprobeConfigError):
        parse_config(env(identity_file, CAPPROBE_TIER=tier))


def test_p0_import_is_an_error(identity_file):
    with pytest.raises(CapprobeConfigError, match="P0"):
        parse_config(env(identity_file, CAPPROBE_TIER="P0", CAPPROBE_GROUPS=""))


def test_tier_group_consistency(identity_file):
    with pytest.raises(CapprobeConfigError, match="P1 requires"):
        parse_config(env(identity_file, CAPPROBE_TIER="P1"))
    with pytest.raises(CapprobeConfigError, match="P2 requires"):
        parse_config(env(identity_file, CAPPROBE_GROUPS=""))


@pytest.mark.parametrize("groups,msg", [("count,nosuch", "unknown"),
                                        ("bytes", "does not implement"),
                                        ("count,ts", "does not implement"),
                                        ("count,,lag", "empty item"),
                                        ("count,count", "repeats")])
def test_bad_groups(identity_file, groups, msg):
    with pytest.raises(CapprobeConfigError, match=msg):
        parse_config(env(identity_file, CAPPROBE_GROUPS=groups, CAPPROBE_LAG_PERIOD_US="1"))


@pytest.mark.parametrize("drop,msg", [("CAPPROBE_SHM_DIR", "SHM_DIR"),
                                      ("CAPPROBE_INTERVAL_MS", "INTERVAL_MS"),
                                      ("CAPPROBE_IDENTITY_FILE", "IDENTITY_FILE")])
def test_required_when_active(identity_file, drop, msg):
    with pytest.raises(CapprobeConfigError, match=msg):
        parse_config(env(identity_file, **{drop: None}))


def test_relative_shm_dir_rejected(identity_file):
    with pytest.raises(CapprobeConfigError, match="absolute"):
        parse_config(env(identity_file, CAPPROBE_SHM_DIR="shm"))


@pytest.mark.parametrize("v", ["0", "-5", "1.5", "x"])
def test_interval_must_be_positive_int(identity_file, v):
    with pytest.raises(CapprobeConfigError, match="INTERVAL_MS"):
        parse_config(env(identity_file, CAPPROBE_INTERVAL_MS=v))


def test_lag_needs_period(identity_file):
    with pytest.raises(CapprobeConfigError, match="LAG_PERIOD_US"):
        parse_config(env(identity_file, CAPPROBE_GROUPS="lag"))


@pytest.mark.parametrize("spec,msg", [
    ("site=G3,mode=busy,delta_ns=10", "unknown site"),
    ("site=F2,mode=drop", "not supported"),
    ("site=F2,mode=short,short_tokens=1", "not supported"),
    ("site=F2,mode=busy", "delta_ns"),
    ("site=F2,mode=busy,delta_ns=10;site=F2,mode=sleep,delta_ns=10", "more than one"),
])
def test_bad_inject(identity_file, spec, msg):
    with pytest.raises(CapprobeConfigError, match=msg):
        parse_config(env(identity_file, CAPPROBE_INJECT=spec))


def test_miscount_needs_count_group(identity_file):
    with pytest.raises(CapprobeConfigError, match="miscount"):
        parse_config(env(identity_file, CAPPROBE_GROUPS="lag", CAPPROBE_LAG_PERIOD_US="1",
                         CAPPROBE_INJECT="site=H3,mode=miscount,miscount_delta=1"))


def test_inject_alone_activates_and_needs_shm(identity_file):
    e = {"CAPPROBE_TIER": "P1", "CAPPROBE_INJECT": "site=F4,mode=sleep,delta_ns=1000"}
    with pytest.raises(CapprobeConfigError, match="SHM_DIR"):
        parse_config(e)
    e.update(CAPPROBE_SHM_DIR="/dev/shm/x", CAPPROBE_INTERVAL_MS="5",
             CAPPROBE_IDENTITY_FILE=identity_file)
    cfg = parse_config(e)
    assert cfg.active and not cfg.count and len(cfg.inject_specs) == 1


@pytest.mark.parametrize("doc", [
    {"schema": "capkit.identity/0", "identity": IDENTITY},
    {"schema": "capkit.identity/1", "identity": {k: v for k, v in IDENTITY.items()
                                                 if k != "arm"}},
    {"schema": "capkit.identity/1", "identity": dict(IDENTITY, register_sha="xyz")},
    {"schema": "capkit.identity/1", "identity": dict(IDENTITY, extra=1)},
])
def test_identity_file_validated(tmp_path, identity_file, doc):
    p = tmp_path / "bad_identity.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(CapprobeConfigError, match="IDENTITY_FILE"):
        parse_config(env(identity_file, CAPPROBE_IDENTITY_FILE=str(p)))


def test_identity_file_missing(identity_file, tmp_path):
    with pytest.raises(CapprobeConfigError, match="unreadable"):
        parse_config(env(identity_file, CAPPROBE_IDENTITY_FILE=str(tmp_path / "none.json")))


def test_slot_and_site_tables_consistent():
    validate_slot_decls(sites.SLOTS)
    assert set(sites.SITES) == set(sites.COUNT_SITES) | set(sites.LAG_SITES)
    assert set(sites.SITE_NOTES) == set(sites.SITES)
    for site, d in sites.SITES.items():
        assert set(d) == {"group", "proc_kind", "file", "lines", "slots"}
        assert d["lines"] and d["lines"] == sorted(set(d["lines"])), site
        assert set(sites.SITE_NOTES[site]) == {"what"}  # no line numbers outside SITES
        for slot in d["slots"]:
            assert sites.SLOTS[slot]["site"] == site and sites.SLOTS[slot]["group"] == d["group"]
    assert set(sites.IMPLEMENTED_GROUPS) | set(sites.RESERVED_GROUPS) == set(sites.ALL_GROUPS)


def test_sites_for_groups_returns_copies():
    got = sites.sites_for_groups(("count",))
    assert set(got) == set(sites.COUNT_SITES)
    got["F4"]["lines"].append(1)
    got["F4"]["slots"].append("x")
    assert 1 not in sites.SITES["F4"]["lines"] and "x" not in sites.SITES["F4"]["slots"]
