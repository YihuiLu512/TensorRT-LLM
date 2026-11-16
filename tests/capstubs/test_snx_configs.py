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
