# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""configs/snx (configs/gen_snx_configs.py): the generator reproduces the committed files byte for
byte, every scenario validates, the S-NX roles get the tier table and the launch origin, and the
{origin_ns} placeholder resolves to the launch origin."""

import json
import os
import subprocess
import sys

import pytest

from caplaunch import scenario as sc

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.abspath(os.path.join(HERE, "..", ".."))
SNX = os.path.join(C, "configs", "snx")
SCENARIOS = sorted(n for n in os.listdir(SNX) if n.endswith(".scenario.json"))


def test_generator_reproduces_configs(tmp_path):
    out = str(tmp_path / "gen")
    subprocess.run([sys.executable, os.path.join(C, "configs", "gen_snx_configs.py"), out],
                   check=True)
    made = sorted(os.listdir(os.path.join(out, "snx")))
    assert made == sorted(n for n in os.listdir(SNX) if n != "README.md")
    for n in made:
        with open(os.path.join(out, "snx", n), "rb") as a, open(os.path.join(SNX, n), "rb") as b:
            assert a.read() == b.read(), n


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_validates(name):
    with open(os.path.join(SNX, name)) as f:
        doc = json.load(f)
    roles = sc.validate_scenario(doc)
    snx_roles = [r for r in roles.values()
                 if r.get("module") == "capstubs.nx_role"
                 or "capstubs.serve_nx" in (r.get("command") or [])]
    assert snx_roles, name
    for r in snx_roles:
        if r["launcher"] == "product":
            a = r["args"]
            assert a[a.index("--tier-table") + 1] == "{input:tier_table}"
            assert a[a.index("--origin-ns") + 1] == "{origin_ns}"
            assert "--" in a and "--gpus_per_node" in a[a.index("--"):]
        else:
            assert r["kind"] == "client" and r["expected_procs"] == 1


def test_origin_ns_placeholder():
    ctx = sc.ArgvContext(identity=None, run_dir="/T", raw_dir="/T/raw", params={},
                         stop_grace_s=1.0, shm_dir="/s", snapshot_interval_ms=50,
                         tier_table_path="/t.json", origin_ns=12345, own_files={},
                         port_file_of=None, addr_of=None)
    assert sc.resolve("--origin-ns={origin_ns}", "r", ctx) == "--origin-ns=12345"


@pytest.mark.parametrize("shape", ["mc", "r"])
def test_e9_n4f0_differs_from_n8f0_only_in_frontend_count(shape):
    """E13 4-frontend arms: host, S-NX worker, capload (8 workers), gates and tier template are
    those of e9-n8f0; only --num_serve_frontends, the frontend role's cores and the texts differ."""
    def load(n, kind):
        with open(os.path.join(SNX, "e9-n%df0-%s.%s.json" % (n, shape, kind))) as f:
            return json.load(f)

    s4, s8 = load(4, "scenario"), load(8, "scenario")
    r4, r8 = ({r["role"]: r for r in s["roles"]} for s in (s4, s8))
    assert sorted(r4) == sorted(r8)
    assert r4["capload"]["workers"] == r8["capload"]["workers"] == 8
    for name in r8:
        a, b = dict(r4[name]), dict(r8[name])
        if name == "frontend":
            i = b["args"].index("--num_serve_frontends")
            assert (a["args"][i + 1], b["args"][i + 1]) == ("4", "8")
            a["args"] = a["args"][:i + 1] + a["args"][i + 2:]
            b["args"] = b["args"][:i + 1] + b["args"][i + 2:]
            assert (a.pop("cpus_needed"), b.pop("cpus_needed")) == (44, 88)
            a.pop("note"), b.pop("note")
        assert a == b, name
    for k in ("name", "note", "tier_table", "schedule", "prompts", "roles"):
        s4.pop(k), s8.pop(k)
    assert s4 == s8
    t4, t8 = load(4, "tiers.template"), load(8, "tiers.template")
    assert t4.pop("name") == "e9-n4f0-%s" % shape and t8.pop("name") == "e9-n8f0-%s" % shape
    assert t4 == t8
