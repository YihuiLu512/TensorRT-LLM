# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.szr_launch without the product: template cutting and the Python frame model on the
fake factory, the CPython int opcodes, refused S-NX settings, and the wrap mode (attach info
rewrite + exec of D1) with a fake S-ZR and a fake D1 binary."""

import json
import os
import pickle
import subprocess
import sys

import pytest

from capstubs import szr_launch as L

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "..", "tools"))


def test_pickle_int_matches_cpython():
    for v in (0, 1, 255, 256, 65535, 65536, 2**31 - 1, 2**31, 2**57 + 3, 2**63, 2**64 - 1):
        data = pickle.dumps(v, protocol=4)
        body = L.strip_frames(data)[2:-1]  # PROTO 4 ... STOP
        assert L.pickle_int(v) == body, v


def test_fake_templates_self_check():
    doc = L.make_templates(L.FakeFactory(), 3, self_check_frames=40, seed=3)
    assert doc["schema"] == L.SCHEMA_TEMPLATES
    assert {(s["n"], s["state"]) for s in doc["shapes"]} == (
        {(n, st) for n in range(1, 4) for st in ("gen", "length")} | {(n, "cancelled") for n in range(4)})
    assert doc["self_check"]["elements"] > 40
    for s in doc["shapes"]:
        for part in ("head", "tail"):
            assert len(s[part]["tokens"]) == s["n"]
            assert sorted(k for _p, _l, k in s[part]["slots"]) == [0, 1]


def test_assemble_py_loads_back():
    f = L.FakeFactory()
    doc = L.make_templates(f, 2, self_check_frames=1)
    specs = [(7, (1 << 57) | 9, 2, "gen", 1), (70000, (1 << 57) | 10, 0, "cancelled", 0),
             (2**40, (1 << 57) | 11, 1, "length", 5)]
    got = pickle.loads(L.assemble_py(doc, specs, [5, 6, 7]))
    assert [(r.request_id, r.client_id, r.result.is_final) for r in got] == \
        [(7, (1 << 57) | 9, False), (70000, (1 << 57) | 10, True), (2**40, (1 << 57) | 11, True)]
    assert got[0].result._result == f.response(7, 0, [6, 7], "gen").result._result


def test_refuses_unreplayable_snx_settings():
    from capstubs.null_executor import SnxConfig
    cfg = SnxConfig.from_doc({"schema": "capkit.snx_config/1", "k_mix": [[1, 1.0]]})
    with pytest.raises(L.LaunchError, match="k_mix"):
        L.check_replayable(cfg)


def test_required_tokens():
    from capstubs.null_executor import SnxConfig
    cfg = SnxConfig.from_doc({"schema": "capkit.snx_config/1", "tokens_per_step": 3,
                              "stream_interval": 2, "first_step_tokens": 7})
    assert L.required_tokens(cfg, None) == 7
    assert L.required_tokens(cfg, {"k_of_tier": {"1": 5}, "stream_interval": 2}) == 10


FAKE_SZR = r"""#!%s
import json, os, sys, time
a = sys.argv[1:]
ready = a[a.index("--ready-file") + 1]
json.dump({"argv": a, "key": os.environ.get("SZR_HMAC_KEY_HEX"), "ppid": os.getppid()},
          open(ready + ".argv", "w"))
open(ready, "w").write("{}")
time.sleep(1)
"""

FAKE_D1 = r"""#!%s
import json, os, sys
json.dump({"argv": sys.argv[1:], "attach": json.loads(os.environ["TLLM_EXECUTOR_ATTACH_INFO"])},
          open(os.environ["FAKE_D1_OUT"], "w"))
"""


def test_wrap_rewrites_request_addr_and_execs_d1(tmp_path):
    tpl = tmp_path / "tpl.json"
    tpl.write_text(json.dumps(L.make_templates(L.FakeFactory(), 2, self_check_frames=1)))
    szr = tmp_path / "fake_szr.py"
    szr.write_text(FAKE_SZR % sys.executable)
    d1 = tmp_path / "fake_d1.py"
    d1.write_text(FAKE_D1 % sys.executable)
    for p in (szr, d1):
        p.chmod(0o755)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "fields.json").write_text(json.dumps(
        {"sampling_params": {"fields": [{"index": 3, "name": "max_tokens"}]}}))
    out = tmp_path / "out"
    rc = L.main(["make-wrapper", "--out-dir", str(out), "--szr-bin", str(szr), "--templates",
                 str(tpl), "--ipc-dir", str(tmp_path), "--d1-bin", str(d1), "--set", "beat_ns=1000000"])
    assert rc == 0
    attach = {"mode": "classic", "request_addr": "ipc:///rank0/request.sock",
              "result_addrs": ["ipc:///x/r0.sock", "ipc:///x/r1.sock"], "hmac_key": "ab" * 32,
              "model_world_size": 1}
    env = dict(os.environ, TLLM_EXECUTOR_ATTACH_INFO=json.dumps(attach), TLLM_EXECUTOR_FRONTEND_ID="1",
               FAKE_D1_OUT=str(tmp_path / "d1.json"), PYTHONPATH=TOOLS)
    subprocess.run([str(out / "szr-d1-wrap"), "--listen", "0:1", "--template", str(pkg)], env=env,
                   check=True, timeout=60)
    seen = json.loads((tmp_path / "d1.json").read_text())
    assert seen["argv"] == ["--listen", "0:1", "--template", str(pkg)]
    assert seen["attach"]["request_addr"] == "ipc://%s/request.sock" % tmp_path
    assert seen["attach"]["result_addrs"] == attach["result_addrs"]
    szr_seen = json.loads((out / "szr.ready.json.argv").read_text())
    a = szr_seen["argv"]
    assert a[a.index("--result-addr") + 1] == "ipc:///x/r1.sock"
    assert a[a.index("--request-addr") + 1] == "ipc://%s/request.sock" % tmp_path
    assert "--exit-with-parent" in a
    assert szr_seen["key"] == "ab" * 32
    rec = json.loads((out / "szr_launch.json").read_text())
    assert rec["rank0_request_addr"] == "ipc:///rank0/request.sock"
    assert oct((out / "hmac_key.hex").stat().st_mode & 0o777) == "0o600"
