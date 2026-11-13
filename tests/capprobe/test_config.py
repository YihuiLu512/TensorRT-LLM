# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capprobe.config: every switch rule (pure function, no subprocess). Unless a test says
otherwise the parsed process is the registered entry (ARGV matches ENTRY)."""

import json
import os

import pytest
from probe_testutil import IDENTITY

from capcommon import probe_entry
from capcommon.snapshot import validate_slot_decls
from capprobe import config as C
from capprobe import sites
from capprobe.config import CapprobeConfigError

PROGRAM = os.path.realpath("/usr/bin/env")
ARGV = ["/usr/bin/env", "probe-unit", "--x", "1"]
ENTRY = probe_entry.encode({"schema": probe_entry.SCHEMA, "program": PROGRAM,
                            "args": ["probe-unit", "--x", "1"]})


def parse_config(e, argv=ARGV):
    return C.parse_config(e, argv=argv)


def env(identity_file, **kw):
    base = {"CAPPROBE_TIER": "P2", "CAPPROBE_GROUPS": "count", "CAPPROBE_SHM_DIR": "/dev/shm/x",
            "CAPPROBE_INTERVAL_MS": "1000", "CAPPROBE_IDENTITY_FILE": identity_file,
            "CAPPROBE_ENTRY": ENTRY}
    base.update(kw)
    return {k: v for k, v in base.items() if v is not None}


def test_good_p2(identity_file):
    cfg = parse_config(env(identity_file, CAPPROBE_GROUPS="lag,count",
                           CAPPROBE_LAG_PERIOD_US="500", CAPPROBE_ROLE_HINT="frontend"))
    assert cfg.groups == ("count", "lag")  # canonical order
    assert cfg.count and cfg.lag and cfg.active
    assert cfg.interval_ns == 1_000_000_000 and cfg.lag_period_ns == 500_000
    assert cfg.identity["experiment"] == "E0" and cfg.role_hint == "frontend"
    assert cfg.enabled and cfg.reason == C.REASON_ENTRY


def test_p1_all_off_needs_nothing_else():
    cfg = parse_config({"CAPPROBE_TIER": "P1", "CAPPROBE_ENTRY": ENTRY})
    assert cfg.enabled and not cfg.active and cfg.groups == () and cfg.shm_dir is None


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
    e = {"CAPPROBE_TIER": "P1", "CAPPROBE_INJECT": "site=F4,mode=sleep,delta_ns=1000",
         "CAPPROBE_ENTRY": ENTRY}
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


# ---- activation only in the registered entry process (INTERFACES erratum R1b) -------------------

def test_no_capprobe_env_is_silently_off():
    cfg = parse_config({"PATH": "/usr/bin", "OTHER": "1"})
    assert not cfg.enabled and not cfg.active and cfg.tier is None
    assert cfg.reason == C.REASON_NO_ENV and cfg.groups == () and not cfg.count and not cfg.lag


def test_no_entry_rule_is_off_but_values_are_still_checked(identity_file):
    e = env(identity_file, CAPPROBE_ENTRY=None)
    cfg = parse_config(e)
    assert not cfg.enabled and not cfg.active and cfg.reason == C.REASON_NO_ENTRY
    with pytest.raises(CapprobeConfigError, match="P2 requires"):
        parse_config(dict(e, CAPPROBE_GROUPS=""))


@pytest.mark.parametrize("argv", [
    [ARGV[0], "-c", "from multiprocessing.spawn import spawn_main; spawn_main()",
     "--multiprocessing-fork"],                      # spawn pool child
    ["/usr/bin/env", "probe-unit", "--x", "2"],      # same program, other arguments
    ["/usr/bin/env", "probe-unit", "--x"],           # a prefix of the arguments
    ["/bin/true", "probe-unit", "--x", "1"],         # other program, same arguments
    [],                                              # unreadable command line
])
def test_other_processes_stay_off(tmp_path, argv):
    # identity file deliberately missing: a non-entry process must not even read it
    e = env(str(tmp_path / "no_such_identity.json"))
    cfg = parse_config(e, argv=argv)
    assert not cfg.enabled and not cfg.active and cfg.reason == C.REASON_NOT_ENTRY
    assert cfg.tier is None and not cfg.count and cfg.identity is None


def test_entry_matches_a_shebang_start_and_a_path_lookup(identity_file):
    # `#!` script: the kernel puts the interpreter first
    cfg = parse_config(env(identity_file), argv=["/usr/bin/python3", "-u"] + ARGV)
    assert cfg.enabled
    # bare name at position 0 is looked up on the process PATH
    cfg = parse_config(env(identity_file, PATH="/usr/bin"),
                       argv=["env", "probe-unit", "--x", "1"])
    assert cfg.enabled
    # a bare name later in the line is never resolved
    cfg = parse_config(env(identity_file, PATH="/usr/bin"),
                       argv=["python3", "env", "probe-unit", "--x", "1"])
    assert not cfg.enabled


@pytest.mark.parametrize("kw,msg", [
    ({"CAPPROBE_TIER": "P3"}, "CAPPROBE_TIER"),
    ({"CAPPROBE_TIER": "P1"}, "P1 requires"),
    ({"CAPPROBE_TIER": "P0"}, "P0 requires"),
    ({"CAPPROBE_TIER": None}, "without CAPPROBE_TIER"),
    ({"CAPPROBE_GROUPS": "count,nosuch"}, "unknown"),
    ({"CAPPROBE_INJECT": "site=G3,mode=busy,delta_ns=10"}, "unknown site"),
    ({"CAPPROBE_INTERVAL_MS": "0"}, "INTERVAL_MS"),
    ({"CAPPROBE_SHM_DIR": "rel"}, "absolute"),
    ({"CAPPROBE_ENTRY": "{not json"}, "not JSON"),
    ({"CAPPROBE_ENTRY": json.dumps({"schema": probe_entry.SCHEMA, "program": "rel/x",
                                    "args": []})}, "absolute"),
    ({"CAPPROBE_GROUP": "count"}, "unknown capprobe variable"),
])
def test_contradictions_fail_in_every_process(identity_file, kw, msg):
    for argv in (ARGV, ["/bin/true"]):  # the entry and a process that is not the entry
        with pytest.raises(CapprobeConfigError, match=msg):
            parse_config(env(identity_file, **kw), argv=argv)


def test_matched_entry_without_tier_fails():
    with pytest.raises(CapprobeConfigError, match="registered entry .* CAPPROBE_TIER is missing"):
        parse_config({"CAPPROBE_ENTRY": ENTRY})
    # the same environment in a process that is not the entry stays off
    cfg = parse_config({"CAPPROBE_ENTRY": ENTRY}, argv=["/bin/true"])
    assert not cfg.enabled


def test_build_rule_resolves_like_exec(tmp_path):
    rule = probe_entry.build_rule(["env", "a"], str(tmp_path), "/usr/bin")
    assert rule == {"schema": probe_entry.SCHEMA, "program": PROGRAM, "args": ["a"]}
    (tmp_path / "s.py").write_text("")
    rule = probe_entry.build_rule(["./s.py", "b"], str(tmp_path), "/usr/bin")
    assert rule["program"] == os.path.realpath(str(tmp_path / "s.py"))
    with pytest.raises(probe_entry.EntryRuleError):
        probe_entry.build_rule(["no-such-program-xyz"], str(tmp_path), "/usr/bin")
    assert probe_entry.parse(probe_entry.encode(rule)) == rule
