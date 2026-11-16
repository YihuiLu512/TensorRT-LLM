# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""caplaunch.orchestrator.aux_helper_rule: the runtime helpers multiprocessing starts are matched
with any CPython interpreter options in front of `-c` (the job exports PYTHONDONTWRITEBYTECODE=1,
so they run as `python -B -c ...`), and nothing else is (INTERFACES.md errata R5-1)."""

import json
import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (puts tools/ on sys.path)
from caplaunch import orchestrator as orch

PY = "/usr/bin/python3"
RT = "from multiprocessing.resource_tracker import main;main(9)"
FS = "from multiprocessing.forkserver import main; main(3, 4, ['__main__'], **{})"


@pytest.mark.parametrize("opts", [
    [],
    ["-B"],
    ["-OO"],
    ["-O", "-B"],
    ["-BE"],
    ["-B", "-s", "-E"],
    ["-I"],
    ["-u"],
    ["-vv"],
    ["-b", "-d", "-q", "-S", "-P"],
    ["-X", "dev"],
    ["-Xdev"],
    ["-X", "faulthandler", "-X", "utf8=1"],
    ["-W", "ignore"],
    ["-Wignore"],
    ["-Werror::DeprecationWarning", "-W", "default"],
    ["-BX", "dev"],
    ["-BWerror"],
    ["-B", "-W", "-c"],  # -W takes "-c" as its argument; the real -c follows
], ids=lambda o: " ".join(o) or "none")
@pytest.mark.parametrize("code,rule", [(RT, "mp_resource_tracker"), (FS, "mp_forkserver")],
                         ids=["resource_tracker", "forkserver"])
def test_interpreter_option_forms_are_matched(opts, code, rule):
    assert orch.aux_helper_rule([PY] + opts + ["-c", code]) == rule


@pytest.mark.parametrize("argv", [
    [],
    [PY],
    [PY, "-c"],
    [PY, RT],
    [PY, "-c", RT, "extra"],                # an argument after the code
    [PY, "-B", "-c", RT, "-c", RT],
    [PY, "-m", "multiprocessing.resource_tracker", "-c", RT],
    [PY, "script.py", "-c", RT],             # a script, then -c as its argument
    [PY, "--check-hash-based-pycs", "always", "-c", RT],
    [PY, "-i", "-c", RT],                    # -i (interactive) is not an option mp copies
    [PY, "-Bi", "-c", RT],
    [PY, "-Bc", RT],                         # -c grouped with other options
    [PY, "-W", "-c", RT],                    # -W takes "-c", so RT is a script name
    [PY, "-X"],
    [PY, "-B", "-X", "-c"],
    [PY, "-", "-c", RT],
    [PY, "", "-c", RT],
    [PY, "-B", "-c", "from multiprocessing.resource_tracker import main;main(9); evil()"],
    [PY, "-B", "-c", "import os"],
], ids=lambda a: " ".join(a[1:]) or "empty")
def test_other_command_lines_are_not_matched(argv):
    assert orch.aux_helper_rule(argv) is None


# The real writer: multiprocessing builds the helper's argv from the parent's flags. Each child
# below starts the helper with util.spawnv_passfds replaced by a recorder (no helper runs).
_RECORD = r"""
import json, sys
from multiprocessing import util, resource_tracker, forkserver
seen = []
def rec(path, args, passfds):
    seen.append([a.decode() if isinstance(a, bytes) else a for a in args])
    raise RuntimeError("recorded")
util.spawnv_passfds = rec
if sys.argv[1] == "forkserver":
    resource_tracker.ensure_running = lambda: None  # only the forkserver's own start
    start = forkserver.ensure_running
else:
    start = resource_tracker.ensure_running
try:
    start()
except RuntimeError:
    pass
print(json.dumps(seen))
"""


@pytest.mark.parametrize("flags,env", [
    ([], {}),
    ([], {"PYTHONDONTWRITEBYTECODE": "1"}),
    (["-B"], {}),
    (["-OO", "-s", "-E"], {}),
    (["-I"], {}),
    (["-X", "dev", "-W", "error::ResourceWarning"], {}),
    ([], {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONOPTIMIZE": "1", "PYTHONWARNINGS": "ignore",
          "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}),
], ids=["plain", "env-B", "flag-B", "OO-s-E", "isolated", "devmode-W", "env-mix"])
def test_helpers_started_by_multiprocessing_are_matched(flags, env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    e.update(env)
    seen = []
    for helper in ("resource_tracker", "forkserver"):
        r = subprocess.run([sys.executable] + flags + ["-c", _RECORD, helper], env=e,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        argvs = json.loads(r.stdout.strip().splitlines()[-1])
        assert len(argvs) == 1, argvs
        seen += argvs
    assert [orch.aux_helper_rule(a) for a in seen] == ["mp_resource_tracker",
                                                         "mp_forkserver"], seen
    if env.get("PYTHONDONTWRITEBYTECODE") or "-B" in flags:
        assert all("-B" in a[1:-2] for a in seen), seen
