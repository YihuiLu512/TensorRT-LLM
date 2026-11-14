# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""configs/e0s (E0 supplementary smoke items 1 and 2) and the job-input form of the S-FE
fingerprint in configs/e0 (SEAMS SM-6): generators reproduce the shipped files, scenarios
validate, templates expand, and the S-PB arm's numbers fit what S-PB accepts (no processes)."""

import json
import os
import subprocess
import sys

import pytest

import conftest as cf
from capcommon import jobinputs as JI
from capcommon.tiers import TierTable
from caplaunch import scenario as sc
from caplaunch import tiergen

E0 = os.path.join(cf.C, "configs", "e0")
E0S = os.path.join(cf.C, "configs", "e0s")


def _load(d, n):
    with open(os.path.join(d, n)) as f:
        return json.load(f)


@pytest.mark.parametrize("gen,shipped", [("gen_e0_configs.py", E0), ("gen_e0s_configs.py", E0S)])
def test_generator_reproduces_the_shipped_files(tmp_path, gen, shipped):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-B", os.path.join(cf.C, "configs", gen), str(out)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    made = sorted(os.listdir(str(out)))
    assert made == sorted(n for n in os.listdir(shipped) if n != "README.md")
    for n in made:
        with open(os.path.join(str(out), n), "rb") as a, open(os.path.join(shipped, n), "rb") as b:
            assert a.read() == b.read(), n


def _scenarios(d):
    return sorted(n for n in os.listdir(d) if n.endswith(".scenario.json"))


def test_e0s_scenarios_validate_and_templates_expand():
    assert _scenarios(E0S) == ["e0s-import-probe.scenario.json",
                               "e0s-spb-frontend.scenario.json"]
    for n in _scenarios(E0S):
        doc = _load(E0S, n)
        roles = sc.validate_scenario(doc)
        assert roles["importprobe"]["module"] == "caplaunch.smoke"
        tpl = _load(E0S, n.replace(".scenario.json", ".tiers.template.json"))
        todo = _load(E0S, n.replace(".scenario.json", ".values.todo.json"))
        assert set(todo["values"]) == tiergen.tbd_names(tpl)
        vals = {k: 1000 * cf.MS if k.endswith("_ns") else (2.0 if k.startswith("rate") else 4)
                for k in todo["values"]}
        vals.update(gate_poll_ns=100 * cf.MS, gate_max_snapshot_age_ns=200 * cf.MS,
                    beat_ns=10 * cf.MS)
        tt = TierTable.from_doc(tiergen.expand(tpl, vals))
        assert tt.tiers[-1].kind == "drain"
        for rd in tt.default_gate["readings"]:
            for term in rd["terms"]:
                assert term["slot"] in roles[term["role"]]["backlog_slots"]


def test_spb_arm_fits_what_spb_accepts():
    doc = _load(E0S, "e0s-spb-frontend.scenario.json")
    assert JI.kinds_of(doc) == ["fingerprint", "prebaked"]
    spb = {r["role"]: r for r in doc["roles"]}["spb"]
    args = [a for a in spb["args"] if isinstance(a, str)]
    assert "{input:prebaked}" in args and "--sfe-module" in args and "--schedule" not in args
    assert {"splice_json": "{input:fingerprint}", "key": "sfe_args"} in spb["args"]
    assert spb["workers"] == spb["expected_procs"] == 1
    tpl = _load(E0S, "e0s-spb-frontend.tiers.template.json")
    loads = [t for t in tpl["tiers"] if t["kind"] != "drain"]
    common = tpl["common"]
    # one (k, s) for every load tier (S-PB's requests carry only max_tokens); beat pacing only
    assert all("tokens_per_msg" not in t and "stream_interval" not in t for t in loads)
    assert common["beat_ns"] == "TBD:beat_ns"
    # every httpprobe request's max_tokens has a prebaked stream: the tier table's max_tokens
    reqs = _load(E0, "e0c-httpprobe.requests.json")
    assert {r["body"]["max_tokens"] for r in reqs} == {common["max_tokens"]}
    pre = doc["job_inputs"]["prebaked"]["args"]
    assert pre[pre.index("--tier-table") + 1] == "{input:tier_table}"


def test_frontend_arms_take_the_fingerprint_from_the_job_input():
    for d in (E0, E0S):
        for n in _scenarios(d):
            doc = _load(d, n)
            text = json.dumps(doc)
            assert "fingerprint_json" not in text, n
            if "sfe_args" in text:
                assert "fingerprint" in JI.kinds_of(doc), n
                fp = doc["job_inputs"]["fingerprint"]["args"]
                assert fp == ["--tokenizer", "{param:tokenizer_dir}", "--trust-remote-code",
                              "{param:trust_remote_code}", "--tokenizer-mode",
                              "{param:tokenizer_mode}"], n
