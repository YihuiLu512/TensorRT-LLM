# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Throw-away git repositories for patchcheck / overlay tests (never touches global git config)."""

import os
import subprocess

GIT_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@example.invalid",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}

BASE_A = "import asyncio\n\n\ndef read(x):\n    y = x + 1\n    return y\n"
PATCHED_A = ("import asyncio\nimport capprobe as _capprobe\n\n\ndef read(x):\n    y = x + 1\n"
             "    if _capprobe.ON_F2:\n        _capprobe.F2()\n    return y\n")
SITE_LINE = 7
TEST_SITES = {"F2": {"group": "count", "proc_kind": "frontend", "file": "tensorrt_llm/a.py",
                     "line": SITE_LINE, "slots": ["count.F2.msgs"]}}


def git(repo, *args):
    env = dict(os.environ, **GIT_ENV)
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True, env=env).stdout.strip()


def write(repo, rel, text):
    p = os.path.join(str(repo), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text)


def commit_all(repo, msg):
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def init(repo):
    os.makedirs(str(repo), exist_ok=True)
    git(repo, "init", "-q", "-b", "main")
