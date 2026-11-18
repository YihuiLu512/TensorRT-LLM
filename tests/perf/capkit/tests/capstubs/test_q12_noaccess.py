# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs/q12_noaccess/sitecustomize.py: forces uvicorn access_log off and keeps the next
sitecustomize on sys.path (with a stand-in uvicorn package)."""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MOD = os.path.abspath(os.path.join(HERE, "..", "..", "tools", "capstubs", "q12_noaccess"))


def test_forces_access_log_off_and_chains(tmp_path):
    fake = tmp_path / "pkgs"
    (fake / "uvicorn").mkdir(parents=True)
    (fake / "uvicorn" / "__init__.py").write_text("from .config import Config\n")
    (fake / "uvicorn" / "config.py").write_text(
        "class Config:\n"
        "    def __init__(self, app, host='127.0.0.1', access_log=True, log_level=None):\n"
        "        self.access_log = access_log\n        self.log_level = log_level\n")
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("import builtins\nbuiltins._q12_chained = True\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.pathsep.join([MOD, str(fake), str(other)])
    code = ("import builtins, uvicorn\n"
            "c = uvicorn.Config(None, log_level='info')\n"
            "d = uvicorn.config.Config(None, access_log=True)\n"
            "print(c.access_log, d.access_log, c.log_level, getattr(builtins, '_q12_chained', False))")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                       timeout=60, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "False", "info", "True"]
    assert r.stderr.count("capkit-q12-noaccess: loaded pid=") == 1
    assert r.stderr.count("uvicorn access_log forced off") == 2


def test_without_uvicorn_only_marks(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = MOD
    r = subprocess.run([sys.executable, "-c", "import sitecustomize as s; print(s.PATCHED)"],
                       env=env, capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False" and "loaded pid=" in r.stderr
