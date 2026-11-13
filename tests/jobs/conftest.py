# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for capjobs. Nothing here may execute ssh or any Slurm command: the autouse
guard makes subprocess refuse ssh/sbatch/srun/squeue/sacct/scontrol/scancel/salloc/cdb."""

import os
import subprocess
import sys

import pytest

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("tools",):
    p = os.path.join(CODE_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

FORBIDDEN = ("ssh", "sbatch", "srun", "squeue", "sacct", "scontrol", "scancel", "salloc", "cdb")
SAVED_CDB = os.path.join(os.path.dirname(CODE_ROOT), "00_plan", "cluster-probe", "20261113",
                         "02d_cdb_b200.csv")
DATA = os.path.join(os.path.dirname(__file__), "data")


def _guard(real):
    def wrapped(args, *a, **kw):
        argv = args if isinstance(args, (list, tuple)) else [args]
        head = os.path.basename(str(argv[0])) if argv else ""
        if head in FORBIDDEN:
            raise AssertionError("test tried to execute %r" % (argv,))
        return real(args, *a, **kw)
    return wrapped


@pytest.fixture(autouse=True)
def no_cluster_commands(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _guard(subprocess.run))
    monkeypatch.setattr(subprocess, "Popen", _guard(subprocess.Popen))


class StubRunner:
    """Answers by the first token of the remote command (last ssh argv element)."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def run(self, argv):
        assert argv[0] == "ssh", argv
        self.calls.append(list(argv))
        head = argv[-1].split()[0]
        ans = self.answers[head]
        if callable(ans):
            return ans(argv)
        return ans


@pytest.fixture
def stub_runner():
    return StubRunner
