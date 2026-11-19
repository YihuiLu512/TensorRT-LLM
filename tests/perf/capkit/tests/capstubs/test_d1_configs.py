# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""configs/d1 (configs/gen_d1_configs.py, the C++ engine host arms of D1): the generator reproduces
the committed files byte for byte, every scenario validates, each arm differs from its configs/snx
source only by the D1 delta, and the host role's serve_nx options parse and pass serve_nx's own
C++ host checks (with a positive control that the checks refuse a broken arm)."""

import copy
import json
import os
import subprocess
import sys

import pytest

from caplaunch import scenario as sc
from capstubs import serve_nx

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.abspath(os.path.join(HERE, "..", ".."))
D1 = os.path.join(C, "configs", "d1")
SNX = os.path.join(C, "configs", "snx")
SCENARIOS = sorted(n for n in os.listdir(D1) if n.endswith(".scenario.json"))
SOURCE = {"e16": "e8-k4-%s", "e9-nc1f0": "e9-n1f0-%s", "e9-nc8f0": "e9-n8f0-%s",
          "e9-n2l8f0": "e9-n8f0-%s"}


def load(d, n):
    with open(os.path.join(d, n)) as f:
        return json.load(f)


def source_of(name):
    shape = name.rsplit("-", 1)[1]
    key = "e16" if name.startswith("e16-") else name.rsplit("-", 1)[0]
    return SOURCE[key] % shape


def host_parts(doc):
    host = next(r for r in doc["roles"] if r["role"] == "host")
    return serve_nx.split_argv(host["args"])


def test_generator_reproduces_configs(tmp_path):
    out = str(tmp_path / "gen")
    subprocess.run([sys.executable, os.path.join(C, "configs", "gen_d1_configs.py"), out],
                   check=True)
    made = sorted(os.listdir(os.path.join(out, "d1")))
    assert made == sorted(n for n in os.listdir(D1) if n != "README.md")
    assert len(made) == 3 * 16
    for n in made:
        with open(os.path.join(out, "d1", n), "rb") as a, open(os.path.join(D1, n), "rb") as b:
            assert a.read() == b.read(), n


@pytest.mark.parametrize("name", SCENARIOS)
def test_scenario_validates_and_differs_only_by_the_d1_delta(name):
    doc = load(D1, name)
    sc.validate_scenario(doc)
    arm = name[:-len(".scenario.json")]
    src = load(SNX, source_of(arm) + ".scenario.json")
    assert doc["experiment"] == ("E16" if arm.startswith("e16-") else "E9")
    assert doc["name"] == arm and doc["tier_table"].endswith("/%s.tiers.json" % arm)
    a, b = copy.deepcopy(doc), copy.deepcopy(src)
    ha = next(r for r in a["roles"] if r["role"] == "host")
    hb = next(r for r in b["roles"] if r["role"] == "host")
    own, serve = serve_nx.split_argv(ha["args"])
    own0, serve0 = serve_nx.split_argv(hb["args"])
    cpp = "--openengine-host-impl" in serve
    if cpp:
        assert serve == serve0 + ["--openengine-host-impl", "cpp"]
        assert own[:len(own0)] == own0
        extra = own[len(own0):]
        assert extra[:5] == ["--cpp-host-bin", "{param:d1_bin}", "--cpp-host-arg=--grpc-threads",
                             "--cpp-host-arg", "{param:d1_grpc_threads}"]
        diag = ["--cpp-host-arg=--diag-out", "--cpp-host-arg", "{raw}/d1"]
        private = ["--report-target", "private"]
        want = {"nx": diag, "nc": diag, "nx-nodiag": [], "2l": private, "n2l": private}
        kind = ("nx-nodiag" if "-nodiag-" in arm else "2l" if arm.startswith("e16-2l")
                else "nx" if arm.startswith("e16-nx") else "n2l" if "-n2l" in arm else "nc")
        assert extra[5:] == want[kind], arm
        assert {"d1_bin", "d1_grpc_threads"} <= set(doc["params_required"])
    else:
        assert arm.startswith(("e16-py-", "e16-pynd-")) and ha["args"] == hb["args"]
    # everything else is the source's: roles minus the host's args / note, job inputs, stop order
    for r in (ha, hb):
        r.pop("args")
        r.pop("note")
    ji_a = a.get("job_inputs", {}).get("request_bytes", {}).get("args")
    ji_b = b.get("job_inputs", {}).get("request_bytes", {}).get("args")
    if arm.startswith("e16-pynd-"):
        assert ji_a == ji_b + ["--extra-detokenize", "false"]
    else:
        assert ji_a == ji_b
    for k in ("experiment", "name", "tier_table", "schedule", "prompts", "note",
              "params_required", "job_inputs"):
        a.pop(k, None)
        b.pop(k, None)
    assert a == b


INT_PLACEHOLDERS = {"{origin_ns}": "1", "{param:nx_stream_interval}": "1"}


@pytest.mark.parametrize("name", [n for n in SCENARIOS if "-py" not in n])
def test_cpp_host_options_pass_serve_nx_checks(name):
    own, serve = host_parts(load(D1, name))
    own = [INT_PLACEHOLDERS.get(x, x) for x in own]
    env, why = serve_nx.cpp_host_env(serve_nx.build_parser().parse_args(own), serve)
    assert why is None
    assert env["TLLM_OPENENGINE_CPP_HOST_BIN"] == "{param:d1_bin}"
    assert env["TLLM_OPENENGINE_CPP_HOST_ARGS"].startswith("--grpc-threads '{param:d1_grpc_threads}'")
    # positive control: the same options without the cpp host impl are refused
    serve_py = serve[:serve.index("--openengine-host-impl")]
    assert serve_nx.cpp_host_env(serve_nx.build_parser().parse_args(own), serve_py)[1]


def test_pynd_job_input_parses():
    from capgrpc import reqbytes
    doc = load(D1, "e16-pynd-k4-mc.scenario.json")
    args = doc["job_inputs"]["request_bytes"]["args"] + ["--out", "/x"]
    seen = {}

    def fake_build(pb2, prompts, tt, model, method, src, extra=None):
        seen["extra"] = extra
        raise reqbytes.ConfigError("stop here")
    # parse only: argparse accepts the arm's arguments (the build needs the product module)
    p_args = [a.replace("{input:prompts}", "/p").replace("{input:tier_table}", "/t")
              .replace("{param:tokenizer_dir}", "/m") for a in args]
    assert reqbytes.main(p_args) == 3  # cannot import the product pb2 here: exit 3, not a usage error
