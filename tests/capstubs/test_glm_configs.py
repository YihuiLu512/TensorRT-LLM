# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""configs/glm (configs/gen_glm_configs.py, E10-S): the generator reproduces the committed files
byte for byte, every scenario validates, the host runs capstubs.serve_glm with the glm venv
python, checks its step against the tier table, and the reader plans message lag."""

import json
import os
import subprocess
import sys

import pytest

from caplaunch import scenario as sc
from capcommon import jobinputs, policy

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.abspath(os.path.join(HERE, "..", ".."))
GLM = os.path.join(C, "configs", "glm")
SCENARIOS = sorted(n for n in os.listdir(GLM) if n.endswith(".scenario.json"))


def test_generator_reproduces_configs(tmp_path):
    out = str(tmp_path / "gen")
    subprocess.run([sys.executable, os.path.join(C, "configs", "gen_glm_configs.py"), out],
                   check=True)
    made = sorted(os.listdir(os.path.join(out, "glm")))
    assert made == sorted(n for n in os.listdir(GLM) if n != "README.md")
    for n in made:
        with open(os.path.join(out, "glm", n), "rb") as a, open(os.path.join(GLM, n), "rb") as b:
            assert a.read() == b.read(), n


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_validates(name):
    with open(os.path.join(GLM, name)) as f:
        doc = json.load(f)
    assert doc["experiment"] == "E10-S" and "E10-S" in policy.NO_MODEL_EXPERIMENTS
    roles = sc.validate_scenario(doc)
    jobinputs.check_scenario(doc)
    h = roles["host"]
    assert h["launcher"] == "product" and h["command"] == ["{param:glm_python}", "-m",
                                                           "capstubs.serve_glm"]
    a = h["args"]
    engine = a[a.index("--engine") + 1]
    assert engine == {"g0": "mock", "g1": "g1"}[doc["name"].split("-")[1]]
    assert a[a.index("--tier-table") + 1] == "{input:tier_table}"
    assert a[a.index("--step-ms") + 1] == "{param:glm_step_ms}"
    serve = a[a.index("--") + 1:]
    assert serve[0] == "{param:tokenizer_dir}" and "--grpc" in serve
    assert serve[serve.index("--grpc-protocol") + 1] == "openengine"
    assert "--planned-msgs" in roles["reader"]["args"]
    assert not any("trt_llm_data" in x for x in json.dumps(doc).split('"'))


def test_vload_templates_inject_at_the_reader_only():
    names = sorted(n for n in os.listdir(GLM) if n.endswith(".tiers.template.json")
                   and "-k8" not in n and "-tps4" not in n)
    assert len(names) == 8
    for n in names:
        with open(os.path.join(GLM, n)) as f:
            t = json.load(f)
        if n.endswith("-vload.tiers.template.json"):
            assert t["launch_inject"] == [{"role": "reader", "site": "recv.msg", "mode": "busy",
                                           "delta_ns": "TBD:delta_ns"}]
            with open(os.path.join(GLM, n.replace("-vload", ""))) as f:
                base = json.load(f)
            assert t["tiers"] == base["tiers"] and t["common"] == base["common"]
        else:
            assert t["launch_inject"] == []


def test_k8_arms_double_the_load_source_only():
    names = sorted(n for n in os.listdir(GLM) if n.endswith("-k8.scenario.json"))
    assert len(names) == 4
    for n in names:
        with open(os.path.join(GLM, n)) as f:
            k8 = json.load(f)
        with open(os.path.join(GLM, n.replace("-k8", ""))) as f:
            base = json.load(f)
        r8 = [r for r in k8["roles"] if r["role"] == "reader"][0]
        r4 = [r for r in base["roles"] if r["role"] == "reader"][0]
        assert (r8["workers"], r8["cpus_needed"], r8["expected_procs"]) == (8, 8, 9)
        assert (r4["workers"], r4["cpus_needed"]) == (4, 4)
        assert r8["args"] == r4["args"]
        assert [r for r in k8["roles"] if r["role"] == "host"] == \
            [r for r in base["roles"] if r["role"] == "host"]
        t8 = n.replace(".scenario.json", ".tiers.template.json")
        with open(os.path.join(GLM, t8)) as f:
            a = json.load(f)
        with open(os.path.join(GLM, t8.replace("-k8", ""))) as f:
            b = json.load(f)
        assert a["tiers"] == b["tiers"] and a["launch_inject"] == b["launch_inject"] == []


def test_tps4_arms_change_tokens_per_msg_only():
    names = sorted(n for n in os.listdir(GLM) if n.endswith("-tps4.tiers.template.json"))
    assert names == ["e10-g0-mc-tps4.tiers.template.json", "e10-g1-mc-tps4.tiers.template.json"]
    for n in names:
        with open(os.path.join(GLM, n)) as f:
            a = json.load(f)
        with open(os.path.join(GLM, n.replace("-tps4", ""))) as f:
            b = json.load(f)
        assert a["common"]["tokens_per_msg"] == 4 and b["common"]["tokens_per_msg"] == 1
        assert dict(a["common"], tokens_per_msg=1) == b["common"]
        assert a["tiers"] == b["tiers"] and a["launch_inject"] == b["launch_inject"] == []
        s = n.replace(".tiers.template.json", ".scenario.json")
        with open(os.path.join(GLM, s)) as f:
            sa = json.load(f)
        with open(os.path.join(GLM, s.replace("-tps4", ""))) as f:
            sb = json.load(f)
        assert sa["roles"] == sb["roles"]
        assert "glm_tokens_per_step" in sa["params_required"]
