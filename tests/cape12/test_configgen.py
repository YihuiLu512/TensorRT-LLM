# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cape12.configgen: the server entry is copied unchanged; M-type / I-type client lists, their
client_env_var (parsed the way test_perf_sanity.to_env_dict does) and the refusals."""

import json

import pytest
import yaml

from cape12 import configgen as cg
from cape12 import reqlog_l2a as rl

SERVER = {"name": "u4_fp8_tp4_isl13312", "tensor_parallel_size": 4, "stream_interval": 20,
          "client_configs": [{"name": "old", "concurrency": 8, "iterations": 20}]}
SOURCE = {"metadata": {"model_name": "glm_5.3_flash_fp8", "supported_gpus": ["B200"]},
          "hardware": {"gpus_per_node": 4}, "server_configs": [SERVER]}


def to_env_dict(s):  # tests/integration/defs/perf/test_perf_sanity.py:1603, copied
    return dict(x.split("=", 1) for x in s.split() if "=" in x)


def _src(tmp_path):
    p = tmp_path / "src.yaml"
    p.write_text(yaml.safe_dump(SOURCE))
    return str(p)


def test_mtype(tmp_path):
    out = tmp_path / "m.yaml"
    assert cg.main(["mtype", "--source", _src(tmp_path), "--server", SERVER["name"],
                    "--k", "3", "--text-first", "0", "--text-dataset", "/d/t.jsonl",
                    "--out", str(out)]) == 0
    doc = yaml.safe_load(out.read_text())
    s = doc["server_configs"][0]
    assert {k: v for k, v in s.items() if k != "client_configs"} == \
        {k: v for k, v in SERVER.items() if k != "client_configs"}
    assert doc["metadata"] == SOURCE["metadata"] and doc["hardware"] == SOURCE["hardware"]
    names = [c["name"] for c in s["client_configs"]]
    assert names == ["c8_r1_tok", "c8_r1_text", "c8_r2_tok", "c8_r2_text", "c8_r3_tok",
                     "c8_r3_text", "c1_text", "c1_tok"]
    for c in s["client_configs"]:
        env = to_env_dict(c["client_env_var"])
        assert env["E12_MODE"] == "closed" and env["E12_WARMUP_SEED"] == "1"
        assert env["E12_RUN"] == c["name"]
        assert ("dataset_file" in c) == c["name"].endswith("text")
        assert c["concurrency"] * c["iterations"] == (160 if c["concurrency"] == 8 else 30)


def test_itype_with_doses_and_schedules(tmp_path):
    sched = tmp_path / "s.json"
    sched.write_text(json.dumps({"schema": rl.SCHEDULE_SCHEMA, "kind": "poisson",
                                 "offsets_s": [0.0, 1.0]}))
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps({"schema": rl.SCHEDULE_SCHEMA, "kind": "replay",
                               "offsets_s": [0.0, 0.5]}))
    out = tmp_path / "i.yaml"
    assert cg.main(["itype", "--source", _src(tmp_path), "--server", SERVER["name"],
                    "--text-dataset", "/d/t.jsonl", "--doses-ms", "0,3,8",
                    "--schedule", str(sched), "--schedule", str(rep), "--out", str(out)]) == 0
    cs = yaml.safe_load(out.read_text())["server_configs"][0]["client_configs"]
    assert [c["name"] for c in cs] == ["c8_front_text", "c8_front_tok", "c1_text", "c1_tok",
                                       "c8_dose01_text", "c8_dose02_text", "c8_dose03_text",
                                       "c8_open01_text", "c8_replay02_text"]
    envs = [to_env_dict(c["client_env_var"]) for c in cs]
    assert [e["E12_DELTA_MS"] for e in envs[4:7]] == ["0.0", "3.0", "8.0"]
    assert all(e["E12_MODE"] == "closed_delta" for e in envs[4:7])
    assert envs[7]["E12_MODE"] == "open" and envs[7]["E12_SCHEDULE"] == str(sched)
    assert envs[8]["E12_MODE"] == "replay"
    # each env is one the wrapper accepts (the schedule is read and checked)
    for e in envs:
        rl.load_config(dict(e, E12_REQLOG_DIR=str(tmp_path)), ["--seed", "0"])


def test_refusals(tmp_path):
    src = _src(tmp_path)
    assert cg.main(["mtype", "--source", src, "--server", "nope", "--k", "3", "--text-first",
                    "1", "--text-dataset", "/d", "--out", str(tmp_path / "a.yaml")]) == 3
    assert cg.main(["mtype", "--source", src, "--server", SERVER["name"], "--k", "1",
                    "--text-first", "1", "--text-dataset", "/d",
                    "--out", str(tmp_path / "b.yaml")]) == 3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kind": "weird"}))
    assert cg.main(["itype", "--source", src, "--server", SERVER["name"], "--text-dataset",
                    "/d", "--schedule", str(bad), "--out", str(tmp_path / "c.yaml")]) == 3
    out = tmp_path / "d.yaml"
    args = ["mtype", "--source", src, "--server", SERVER["name"], "--k", "2", "--text-first",
            "1", "--text-dataset", "/d", "--out", str(out)]
    assert cg.main(args) == 0 and cg.main(args) == 3  # never overwrites
