# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tools/cape12/bin/python: only `python -m tensorrt_llm.serve.scripts.benchmark_serving` with
E12_REQLOG_DIR set goes through reqlog_l2a.py; everything else reaches the next python."""

import os
import subprocess

SHIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools", "cape12",
                                        "bin"))


def _fake_python(tmp_path):
    d = tmp_path / "realbin"
    d.mkdir(exist_ok=True)
    p = d / "python"
    p.write_text("#!/bin/bash\nprintf 'REAL %s\\n' \"$*\"\n")
    p.chmod(0o755)
    return str(d)


def _run(tmp_path, args, reqlog=None):
    env = {"PATH": "%s:%s:/usr/bin:/bin" % (SHIM_DIR, _fake_python(tmp_path))}
    if reqlog:
        env["E12_REQLOG_DIR"] = reqlog
    return subprocess.run([os.path.join(SHIM_DIR, "python")] + args, env=env,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_benchmark_serving_goes_through_the_wrapper(tmp_path):
    out = _run(tmp_path, ["-m", "tensorrt_llm.serve.scripts.benchmark_serving", "--x", "1"],
               reqlog=str(tmp_path))
    assert out.startswith("REAL ") and out.endswith("reqlog_l2a.py --x 1")


def test_other_calls_pass_through(tmp_path):
    assert _run(tmp_path, ["-c", "pass"], reqlog=str(tmp_path)) == "REAL -c pass"
    # without E12_REQLOG_DIR even benchmark_serving passes through unchanged
    assert _run(tmp_path, ["-m", "tensorrt_llm.serve.scripts.benchmark_serving"]) == \
        "REAL -m tensorrt_llm.serve.scripts.benchmark_serving"
