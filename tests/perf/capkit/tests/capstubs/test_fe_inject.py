# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs/fe_inject/sitecustomize.py: wraps the Fr / Pp sites when their modules load (also in
a spawn child), applies the busy wait, reports counters on stderr, keeps the next sitecustomize."""

import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "..", "tools"))
MOD = os.path.join(TOOLS, "capstubs", "fe_inject")

STREAM = """
class GenerateStream:
    def _validate(self, response):
        return response is not None
"""
POOL = """
def _handle(records, message, tokenizer, hook):
    records.append(message)
    return None


def _process_main(n):
    records = []
    for i in range(n):
        _handle(records, i, None, None)
    return len(records)
"""


def _fake_tree(tmp_path):
    root = tmp_path / "pkgs"
    for d in ("tensorrt_llm", "tensorrt_llm/grpc", "tensorrt_llm/grpc/openengine_client",
              "tensorrt_llm/serve", "tensorrt_llm/serve/openengine_poc"):
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / "__init__.py").write_text("")
    (root / "tensorrt_llm/grpc/openengine_client/stream.py").write_text(STREAM)
    (root / "tensorrt_llm/serve/openengine_poc/postproc_pool.py").write_text(POOL)
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("import builtins\nbuiltins._fe_chained = True\n")
    return root, other


def _run(tmp_path, code, inject):
    root, other = _fake_tree(tmp_path)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CAPKIT_FE_INJECT",
                                                            "CAPKIT_FE_INJECT_DIR")}
    env["PYTHONPATH"] = os.pathsep.join([MOD, TOOLS, str(root), str(other)])
    if inject is not None:
        env["CAPKIT_FE_INJECT"] = inject
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                          capture_output=True, text=True, timeout=120, cwd=str(tmp_path))


def _reports(stderr):
    out = {}
    for line in stderr.splitlines():
        if line.startswith("capkit-fe-inject-report "):
            d = json.loads(line[len("capkit-fe-inject-report "):])
            out[d["pid"]] = d["sites"]  # the last report of each process wins
    return out


def test_sites_wrapped_in_parent_and_spawn_child(tmp_path):
    code = """
    import builtins, multiprocessing as mp, os, time
    from tensorrt_llm.grpc.openengine_client.stream import GenerateStream
    from tensorrt_llm.serve.openengine_poc import postproc_pool
    s = GenerateStream()
    assert all(s._validate(i) for i in range(5))
    t0 = time.monotonic_ns()
    assert postproc_pool._process_main(4) == 4
    busy = time.monotonic_ns() - t0
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=postproc_pool._process_main, args=(3,))
    p.start(); p.join(60)
    print(os.getpid(), p.pid, p.exitcode, busy, getattr(builtins, "_fe_chained", False))
    """
    r = _run(tmp_path, code, "Fr:count;Pp:busy:2000000")
    assert r.returncode == 0, r.stderr
    parent, child, exitcode, busy, chained = r.stdout.split()
    assert exitcode == "0" and chained == "True"
    assert int(busy) >= 4 * 2000000  # four busy waits of 2 ms in the parent's _handle calls
    rep = _reports(r.stderr)
    assert rep[int(parent)]["Fr"]["hits"] == 5 and rep[int(parent)]["Fr"]["applied"] == 0
    assert rep[int(parent)]["Pp"]["hits"] == 4 and rep[int(parent)]["Pp"]["applied"] == 4
    assert rep[int(child)]["Pp"]["hits"] == 3 and rep[int(child)]["Pp"]["applied"] == 3
    assert rep[int(child)]["Pp"]["applied_ns"] >= 3 * 2000000
    assert "Fr" not in rep[int(child)]  # the child never imports the stream module
    assert r.stderr.count("capkit-fe-inject: site Pp installed") == 2


def test_no_spec_only_marks(tmp_path):
    code = """
    import sitecustomize as s
    from tensorrt_llm.serve.openengine_poc import postproc_pool
    print(s.STATE, postproc_pool._handle.__name__, hasattr(postproc_pool._handle, '__wrapped__'))
    """
    r = _run(tmp_path, code, None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["None", "_handle", "False"]
    assert "capkit-fe-inject: loaded pid=" in r.stderr and "report" not in r.stderr


def test_unknown_site_refuses_start(tmp_path):
    r = _run(tmp_path, "print('started')", "Hs:count")
    assert r.returncode != 0 and "started" not in r.stdout
    assert "fe_inject sites are" in r.stderr
