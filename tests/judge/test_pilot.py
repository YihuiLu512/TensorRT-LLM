# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pilot builder: bands from judged sub-saturation tiers; RED tiers excluded; explicit inputs."""

import json
import os
import subprocess
import sys

from conftest import C, PKG, SAMPLES, TOOLS

ENV = dict(os.environ, PYTHONPATH=TOOLS, CAPKIT_CODE_SHA="b" * 40)


def _judged(tmp_path, name):
    out = tmp_path / "s"
    subprocess.run([sys.executable, os.path.join(SAMPLES, "make_samples.py"), "--out",
                    str(out), "--only", name], check=True, capture_output=True, timeout=300)
    W = out / name
    tree = W / json.loads((W / "expect.json").read_text())["tree"]
    subprocess.run([sys.executable, "-m", PKG, "--tree", str(tree), "--workdir", str(W)],
                   env=ENV, capture_output=True, timeout=300, check=False)
    return W, tree


def _pilot(tmp_path, W, lines, out_name="pilot.json"):
    lst = tmp_path / "inputs.tsv"
    lst.write_text("".join(lines))
    out = tmp_path / out_name
    p = subprocess.run([sys.executable, os.path.join(C, "analysis", "pilot_build.py"),
                        "--inputs", str(lst), "--workdir", str(W), "--out", str(out)],
                       capture_output=True, text=True, timeout=300, check=False)
    return p, out


def test_pilot_from_good_sample(tmp_path):
    W, tree = _judged(tmp_path, "good")
    p, out = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 0, p.stderr
    d = json.loads(out.read_text())
    assert d["schema"] == "capkit.pilot/1"
    nb = d["noise_bands"]
    assert nb["T1:msg_lag_p50_ns:capload"]["n"] == 2
    assert nb["T3:send_lag_p99_ns:capload"]["n"] == 4  # two workers x two tiers
    # the baseline tier's own normalised ratio is not a sample
    assert nb["T2:delivery_ratio_norm:capload"]["n"] == 1
    assert d["window_ns"] is not None and d["inputs"][0]["tiers_used"] == [2, 4]
    # refuses to overwrite
    p2, _ = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p2.returncode == 3


def test_pilot_excludes_red_tier(tmp_path):
    W, tree = _judged(tmp_path, "c5_requests")
    p, out = _pilot(tmp_path, W, ["%s\t2,4\n" % tree])
    assert p.returncode == 0, p.stderr
    d = json.loads(out.read_text())
    assert d["inputs"][0]["tiers_used"] == [2]
    assert "4" in d["inputs"][0]["tiers_excluded"]


def test_pilot_rejects_glob_unjudged_and_missing_tiers(tmp_path):
    W, tree = _judged(tmp_path, "good")
    p, _ = _pilot(tmp_path, W, ["%s\n" % tree], "a.json")
    assert p.returncode == 3  # tier column is required
    p, _ = _pilot(tmp_path, W, ["%s/*\t2\n" % os.path.dirname(str(tree))], "b.json")
    assert p.returncode == 3
    p, _ = _pilot(tmp_path, W, ["%s\t1\n" % tree], "c.json")
    assert p.returncode == 3  # tier 1 is a drain tier
