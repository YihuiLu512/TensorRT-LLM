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
    names = sorted(n for n in os.listdir(GLM) if n.endswith(".tiers.template.json"))
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
