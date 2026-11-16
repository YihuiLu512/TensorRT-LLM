# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CAPPROBE_ENTRY against real processes started every way a product role can be started.

Each case builds the rule exactly as the orchestrator does (capcommon.probe_entry.build_rule on
the role's argv, cwd and PATH), starts that argv in the role's cwd, and reads capprobe.ENABLED in
the started process and, where the case asks for it, in a re-exec of `[sys.executable] +
sys.argv` (what the multi-frontend launcher does to start attached frontends). A process may
change directory before it imports capprobe."""

import json
import os
import subprocess
import sys

import pytest
from probe_testutil import clean_env

from capcommon import probe_entry

PROG = r'''
import json, os, subprocess, sys


def main():
    if os.environ.get("FX_CHDIR"):
        os.chdir(os.environ["FX_CHDIR"])
    import capprobe
    print(json.dumps({"tag": os.environ.get("FX_TAG", "parent"), "enabled": capprobe.ENABLED,
                      "reason": capprobe.CONFIG.reason, "orig_argv": list(sys.orig_argv)}),
          flush=True)
    if os.environ.get("FX_REEXEC") == "1":
        env = dict(os.environ, FX_REEXEC="0", FX_TAG="reexec")
        sys.exit(subprocess.run([sys.executable] + sys.argv, env=env, timeout=60).returncode)


if __name__ == "__main__":
    main()
'''


@pytest.fixture
def layout(tmp_path):
    role = tmp_path / "role"  # the role's cwd (control/cwd/<role> in a launch)
    (role / "bin").mkdir(parents=True)
    script = role / "bin" / "fx-serve"  # a console script: the kernel runs its #! interpreter
    script.write_text("#!%s\n%s" % (sys.executable, PROG))
    script.chmod(0o755)
    (role / "entry.py").write_text(PROG)
    (role / "py").mkdir()
    os.symlink(sys.executable, str(role / "py" / "python"))  # an interpreter named relatively
    mods = tmp_path / "mods" / "fxpkg"
    mods.mkdir(parents=True)
    (mods / "__init__.py").write_text("")
    (mods / "serve.py").write_text(PROG)
    (mods / "other.py").write_text(PROG)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = "%s:%s" % (role / "bin", os.environ.get("PATH", os.defpath))
    env = clean_env(CAPPROBE_TIER="P1", PATH=path)
    env["PYTHONPATH"] += ":" + str(tmp_path / "mods")
    return {"role": str(role), "env": env, "path": path, "elsewhere": str(elsewhere)}


def run(layout, argv, registered=None, chdir=False, reexec=False):
    """Start `argv` in the role cwd under the rule registered for `registered` (default: argv
    itself); returns {tag: process state} for the process and its re-exec."""
    rule = probe_entry.build_rule(list(registered or argv), layout["role"], layout["path"])
    env = dict(layout["env"], CAPPROBE_ENTRY=probe_entry.encode(rule),
               FX_REEXEC="1" if reexec else "0")
    if chdir:
        env["FX_CHDIR"] = layout["elsewhere"]
    r = subprocess.run(argv, env=env, cwd=layout["role"], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, "rc=%d\nstdout=%s\nstderr=%s" % (r.returncode, r.stdout, r.stderr)
    out = {}
    for line in r.stdout.splitlines():
        d = json.loads(line)
        out[d["tag"]] = d
    assert set(out) == ({"parent", "reexec"} if reexec else {"parent"}), r.stdout
    return out


def enabled(out):
    return {tag: (d["enabled"], d["reason"]) for tag, d in out.items()}


ON = (True, "entry")
OFF = (False, "not_the_registered_entry")


def test_console_script_and_its_reexec(layout):
    out = run(layout, ["fx-serve", "serve", "--n", "2"], reexec=True)
    assert enabled(out) == {"parent": ON, "reexec": ON}, out
    assert out["parent"]["orig_argv"][0] != "fx-serve"  # the kernel put the #! interpreter first


def test_python_dash_m_and_its_reexec(layout):
    out = run(layout, [sys.executable, "-m", "fxpkg.serve", "serve", "--n", "2"], reexec=True)
    assert enabled(out) == {"parent": ON, "reexec": ON}, out
    assert out["reexec"]["orig_argv"][1].endswith("/fxpkg/serve.py")  # -m <mod> became a file


def test_python_script_and_its_reexec(layout):
    out = run(layout, [sys.executable, os.path.join(layout["role"], "entry.py"), "serve"],
              reexec=True)
    assert enabled(out) == {"parent": ON, "reexec": ON}, out


def test_relative_script_path_and_its_reexec(layout):
    out = run(layout, [sys.executable, "./entry.py", "serve"], reexec=True)
    assert enabled(out) == {"parent": ON, "reexec": ON}, out


@pytest.mark.parametrize("argv", [
    ["./bin/fx-serve", "serve"],          # console script named relatively (#!)
    ["./py/python", "entry.py", "serve"],  # interpreter named relatively
    ["fx-serve", "serve"],                 # found on PATH
], ids=["relative-shebang", "relative-interpreter", "path-lookup"])
def test_chdir_before_import(layout, argv):
    assert enabled(run(layout, argv, chdir=True)) == {"parent": ON}


def test_dash_m_after_interpreter_options_reexec_is_off(layout):
    # documented limit: options before -m are not in sys.argv, so the re-exec line differs
    out = run(layout, [sys.executable, "-u", "-m", "fxpkg.serve", "serve"], reexec=True)
    assert enabled(out) == {"parent": ON, "reexec": OFF}, out


def test_other_module_with_the_same_arguments_is_off(layout):
    out = run(layout, [sys.executable, "-m", "fxpkg.other", "serve", "--n", "2"], reexec=True,
              registered=[sys.executable, "-m", "fxpkg.serve", "serve", "--n", "2"])
    assert enabled(out) == {"parent": OFF, "reexec": OFF}, out
