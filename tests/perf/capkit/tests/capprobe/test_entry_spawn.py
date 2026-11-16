# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probes switch on only in the registered entry process (INTERFACES erratum R1b, decision P3).

A real `python demo.py` parent and children that inherit its environment: a multiprocessing
spawn child that imports capprobe lazily (as the postproc pool does), and a plain subprocess
with another command line (as an MPI worker or any other importer)."""

import json
import os
import subprocess
import sys

from probe_testutil import clean_env, entry_rule

from capcommon.snapshot import list_snapshot_files

DEMO = r'''
import json, multiprocessing, os, subprocess, sys, threading

def state():
    import capprobe
    # getattr: the same demo runs against capprobe builds without ENABLED / CONFIG.reason
    return {"pid": os.getpid(), "enabled": getattr(capprobe, "ENABLED", None),
            "active": capprobe.ACTIVE, "tier": capprobe.TIER, "count": capprobe.COUNT,
            "reason": getattr(capprobe.CONFIG, "reason", None),
            "on_f2": capprobe.ON_F2, "path": capprobe.snapshot_path(),
            "threads": sorted(t.name for t in threading.enumerate())}

def child(q):
    try:
        q.put(state())  # lazy import inside the spawn child
    except BaseException as e:
        q.put({"pid": os.getpid(), "error": "%s: %s" % (type(e).__name__, e)})

if __name__ == "__main__":
    out = {}
    try:
        out["parent"] = state()
        import capprobe
        capprobe.F2(); capprobe.flush()
    except BaseException as e:
        out["parent"] = {"error": "%s: %s" % (type(e).__name__, e)}
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue(); p = ctx.Process(target=child, args=(q,)); p.start()
    out["spawn"] = q.get(timeout=60); p.join(60); out["spawn_exit"] = p.exitcode
    w = subprocess.run([sys.executable, sys.argv[1]], capture_output=True, text=True, timeout=60)
    out["other"] = {"rc": w.returncode, "stdout": w.stdout.strip(), "stderr": w.stderr[-400:]}
    print(json.dumps(out))
'''

OTHER = r'''
import json, os, threading
import capprobe
print(json.dumps({"pid": os.getpid(), "enabled": getattr(capprobe, "ENABLED", None),
                  "active": capprobe.ACTIVE, "reason": getattr(capprobe.CONFIG, "reason", None),
                  "path": capprobe.snapshot_path()}))
'''


def _run(tmp_path, env):
    demo, other = tmp_path / "demo.py", tmp_path / "other_importer.py"
    demo.write_text(DEMO)
    other.write_text(OTHER)
    argv = [sys.executable, str(demo), str(other)]
    if env.get("CAPPROBE_ENTRY") == "THIS":
        env = dict(env, CAPPROBE_ENTRY=entry_rule(argv, str(tmp_path)))
    r = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=180,
                       cwd=str(tmp_path))
    assert r.returncode == 0, "rc=%d\n%s\n%s" % (r.returncode, r.stdout, r.stderr)
    return json.loads(r.stdout.strip().splitlines()[-1])


def _p2_env(tmp_path, identity_file, entry):
    return clean_env(CAPPROBE_TIER="P2", CAPPROBE_GROUPS="count",
                     CAPPROBE_SHM_DIR=str(tmp_path / "shm"), CAPPROBE_INTERVAL_MS="10000",
                     CAPPROBE_IDENTITY_FILE=identity_file, CAPPROBE_ROLE_HINT="frontend",
                     CAPPROBE_ENTRY=entry)


def test_only_the_entry_writes_a_snapshot(tmp_path, identity_file):
    out = _run(tmp_path, _p2_env(tmp_path, identity_file, "THIS"))
    par, ch, oth = out["parent"], out["spawn"], json.loads(out["other"]["stdout"])
    # spawn child: same environment, not the entry -> everything off, no thread, no file
    assert "error" not in ch and out["spawn_exit"] == 0
    files = list_snapshot_files(str(tmp_path / "shm"))
    assert set(files) == {par["pid"]}, files  # exactly one writer: the entry process
    assert par["active"] and par["on_f2"] and "capkit-snapshot" in par["threads"]
    assert par["enabled"] is True
    assert ch["active"] is False and ch["path"] is None
    assert ch["enabled"] is False and ch["tier"] is None
    assert ch["count"] is False and ch["on_f2"] is False and ch["path"] is None
    assert ch["reason"] == "not_the_registered_entry"
    assert "capkit-snapshot" not in ch["threads"]
    # another command line with the inherited environment (MPI worker, other importer)
    assert out["other"]["rc"] == 0, out["other"]["stderr"]
    assert oth["enabled"] is False and oth["path"] is None


def test_entry_registered_for_another_command_line_switches_nothing_on(tmp_path, identity_file):
    rule = entry_rule([sys.executable, str(tmp_path / "not_this.py")], str(tmp_path))
    out = _run(tmp_path, _p2_env(tmp_path, identity_file, rule))
    assert not os.path.exists(tmp_path / "shm") or not os.listdir(tmp_path / "shm")
    assert out["parent"]["active"] is False and out["spawn"]["active"] is False
    assert out["parent"]["enabled"] is False and out["spawn"]["enabled"] is False


def test_no_capprobe_env_is_silent_everywhere(tmp_path):
    out = _run(tmp_path, clean_env())
    assert out["other"]["rc"] == 0, out["other"]["stderr"]
    for st in (out["parent"], out["spawn"], json.loads(out["other"]["stdout"])):
        assert "error" not in st, st
        assert st["enabled"] is False and st["active"] is False and st["path"] is None
        assert st["reason"] == "no_capprobe_env"


def test_entry_without_tier_fails_only_in_the_entry(tmp_path, identity_file):
    env = _p2_env(tmp_path, identity_file, "THIS")
    for k in ("CAPPROBE_TIER", "CAPPROBE_GROUPS"):
        env.pop(k)
    out = _run(tmp_path, env)
    assert "CAPPROBE_TIER is missing" in out["parent"]["error"]
    assert out["spawn"]["enabled"] is False and "error" not in out["spawn"]
