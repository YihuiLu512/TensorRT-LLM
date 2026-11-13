# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container-side modules exercised on local temp dirs (functional only)."""

import json
import os
import stat
import sys
import types

import pytest

from capcommon.hashing import sha256_bytes
from capcommon.runtree import read_state
from capjobs import copyback, envfiles, launches, overlay

REG = "a" * 64


def ident_args(role="jobs"):
    return ["--experiment", "E0", "--arm", "job", "--launch-seq", "0", "--job-id", "local-t",
            "--restart-count", "0", "--register-sha", REG, "--role", role]


# ---- overlay ------------------------------------------------------------------------------------

def make_overlay(tmp_path):
    src = tmp_path / "snap"
    (src / "probe-tree" / "tensorrt_llm").mkdir(parents=True)
    (src / "tools" / "capx").mkdir(parents=True)
    files = {"probe-tree/tensorrt_llm/a.py": b"print('a')\n", "tools/capx/__init__.py": b"X=1\n"}
    for rel, data in files.items():
        (src / rel).write_bytes(data)
    doc = {"tier": "P2", "files": [{"src": "probe-tree/tensorrt_llm/a.py",
                                    "dst": "tensorrt_llm/a.py",
                                    "sha256": sha256_bytes(files["probe-tree/tensorrt_llm/a.py"]),
                                    "size": 11}],
           "capkit_packages": [{"src": "tools/capx/__init__.py", "dst": "capx/__init__.py",
                                "sha256": sha256_bytes(b"X=1\n"), "size": 4}]}
    dest = tmp_path / "site"
    (dest / "tensorrt_llm").mkdir(parents=True)
    (dest / "tensorrt_llm" / "a.py").write_bytes(b"old\n")
    return src, dest, doc


def test_overlay_apply_and_verify(tmp_path):
    src, dest, doc = make_overlay(tmp_path)
    m = tmp_path / "m.json"
    m.write_text(json.dumps(doc))
    out = tmp_path / "ov.json"
    rc = overlay.main(["--manifest", str(m), "--source-root", str(src), "--dest", str(dest),
                       "--out", str(out)] + ident_args())
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["passed"] and rec["expected_files"] == 2 and rec["verified_files"] == 2
    assert (dest / "tensorrt_llm" / "a.py").read_bytes() == b"print('a')\n"
    assert rec["files"][0]["existed_before"] is True


def test_overlay_detects_tampered_source(tmp_path):
    src, dest, doc = make_overlay(tmp_path)
    (src / "tools" / "capx" / "__init__.py").write_bytes(b"X=2\n")
    results, passed, expected = overlay.apply(doc, str(src), str(dest))
    assert not passed and expected == 2
    assert [r["ok"] for r in results] == [True, False]
    doc["capkit_packages"][0]["dst"] = "../escape.py"
    (src / "tools" / "capx" / "__init__.py").write_bytes(b"X=1\n")
    results, passed, _ = overlay.apply(doc, str(src), str(dest))
    assert not passed and results[1]["problem"] == "unsafe path"


# ---- envfiles -----------------------------------------------------------------------------------

def test_check_mounts(tmp_path):
    good = tmp_path / "m1"
    good.mkdir()
    (good / "f").write_text("x")
    empty_forbidden = tmp_path / "lib_empty"
    empty_forbidden.mkdir()
    full_forbidden = tmp_path / "lib_full"
    full_forbidden.mkdir()
    (full_forbidden / "weights").write_text("x")
    absent = str(tmp_path / "absent")
    mounts = ["%s:%s" % (good, good)]
    r = envfiles.check_mounts(mounts, [str(good)], forbidden=(absent, str(empty_forbidden)))
    assert r["passed"] and r["expected_items"] == 4
    r = envfiles.check_mounts(mounts, [], forbidden=(str(full_forbidden),))
    assert not r["passed"] and r["outcome"] == "FAIL"
    empty_mount = tmp_path / "m2"
    empty_mount.mkdir()
    r = envfiles.check_mounts(["%s:%s" % (empty_mount, empty_mount)], [], forbidden=())
    assert not r["passed"]
    r = envfiles.check_mounts([], [], forbidden=())
    assert r["outcome"] == "NOT_RUN" and not r["passed"]


def fake_smi(tmp_path, gpus, apps, rc=0):
    tag = "%d-%d-%d" % (len(gpus), len(apps), rc)
    g, c = tmp_path / ("gpus-" + tag), tmp_path / ("apps-" + tag)
    g.write_text("".join(x + "\n" for x in gpus))
    c.write_text("".join(x + "\n" for x in apps))
    p = tmp_path / ("smi-" + tag)
    p.write_text('#!/bin/bash\nif [[ $1 == --query-gpu=* ]]; then cat "%s"; else cat "%s"; fi\n'
                 'exit %d\n' % (g, c, rc))
    p.chmod(0o755)
    return str(p)


def test_gpu_idle(tmp_path):
    two = ["0, NVIDIA B200, 0, 0", "1, NVIDIA B200, 0, 0"]
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, two, []), 1)
    assert r["passed"] and r["positive_control_ok"] and r["gpu_count"] == 2
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, two, ["4242, 1304 MiB"]), 1)
    assert not r["passed"] and r["outcome"] == "FAIL" and r["compute_apps"] == ["4242, 1304 MiB"]
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, [], []), 1)
    assert not r["passed"] and r["outcome"] == "NOT_RUN" and not r["positive_control_ok"]
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, two, [], rc=9), 1)
    assert not r["passed"] and r["outcome"] == "NOT_RUN"
    r = envfiles.check_gpu_idle(str(tmp_path / "no-such-smi"), 1)
    assert not r["passed"]


def test_image_and_disk_and_imports(tmp_path):
    f = tmp_path / "check.tsv"
    rows = {"path": "/x.sqsh", "expected_sha256": "a" * 64, "expected_size": "10",
            "measured_sha256": "a" * 64, "measured_size": "10"}
    f.write_text("".join("%s\t%s\n" % kv for kv in rows.items()))
    assert envfiles.check_image(str(f))["passed"]
    rows["measured_sha256"] = "b" * 64
    f.write_text("".join("%s\t%s\n" % kv for kv in rows.items()))
    assert not envfiles.check_image(str(f))["passed"]
    rows["measured_sha256"] = ""
    rows["expected_sha256"] = ""
    f.write_text("".join("%s\t%s\n" % kv for kv in rows.items()))
    assert not envfiles.check_image(str(f))["passed"]
    assert envfiles.check_local_disk([str(tmp_path)], 1)["passed"]
    assert not envfiles.check_local_disk([str(tmp_path)], 1 << 62)["passed"]
    assert envfiles.check_local_disk([], 1)["outcome"] == "NOT_RUN"
    site = tmp_path / "site"
    (site / "capzzq").mkdir(parents=True)
    (site / "capzzq" / "__init__.py").write_text("")
    sys.path.insert(0, str(site))
    try:
        r = envfiles.check_imports(str(site), ["capzzq", "json"])
        assert [i["ok"] for i in r["items"]] == [True, False] and not r["passed"]
        assert envfiles.check_imports(str(site), ["capzzq"])["passed"]
    finally:
        sys.path.remove(str(site))
    assert envfiles.manifest_packages({"capkit_packages": [
        {"dst": "capjobs/a.py"}, {"dst": "capcommon/b.py"}, {"dst": "capjobs/c.py"}]}) == [
        "capcommon", "capjobs"]


# ---- copyback -----------------------------------------------------------------------------------

def make_tree(root, files):
    for rel, data in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)


def test_copy_and_compare(tmp_path):
    src = str(tmp_path / "local" / "runs" / "T1")
    make_tree(src, {"raw/launch.json": b"{}", "raw/tiers/000_w/a.bin": b"12345"})
    os.symlink("launch.json", os.path.join(src, "raw", "link"))
    dst = str(tmp_path / "dest" / "T1")
    rec = copyback.copy_tree(src, dst)
    assert rec["passed"] and rec["files_expected"] == 2 and rec["bytes_expected"] == 7
    assert rec["files_copied"] == 2
    with pytest.raises(FileExistsError):
        copyback.copy_tree(src, dst)
    with open(os.path.join(dst, "STATE"), "w") as f:  # appended after copy-back: ignored
        f.write("x")
    assert copyback.compare(src, dst)["passed"]
    with open(os.path.join(dst, "raw", "launch.json"), "wb") as f:
        f.write(b"{}{}")
    bad = copyback.compare(src, dst)
    assert not bad["passed"] and bad["differences"] == ["raw/launch.json"]
    empty = str(tmp_path / "empty")
    os.makedirs(os.path.join(empty, "raw"))
    assert copyback.compare(empty, str(tmp_path / "nowhere"))["outcome"] == "NOT_RUN"
    assert copyback.lock_raw(dst)
    assert not os.access(os.path.join(dst, "raw"), os.W_OK)
    assert os.access(dst, os.W_OK)  # tree root stays writable (STATE, judge/)


def test_sweep(tmp_path):
    local, shm = str(tmp_path / "L"), str(tmp_path / "S")
    dest, jd = str(tmp_path / "runs"), str(tmp_path / "runs" / "J1-r0")
    os.makedirs(jd)
    r = copyback.sweep(local, shm, dest, jd)
    assert r["passed"] and r["outcome"] == "EMPTY"
    os.makedirs(os.path.join(local, "runs", "J1-r0-L0_a", "raw"))
    os.makedirs(os.path.join(local, "logs"))
    r = copyback.sweep(local, shm, dest, jd)
    assert r["outcome"] == "EMPTY"  # only empty dirs: nothing can be lost
    make_tree(os.path.join(local, "runs", "J1-r0-L0_a"), {"raw/x": b"1"})
    make_tree(os.path.join(local, "logs"), {"J1-r0-L0_a.launcher.out": b"hello"})
    make_tree(os.path.join(shm, "snap", "r0-L0"), {"123.bin": b"snap"})
    r = copyback.sweep(local, shm, dest, jd)
    assert r["passed"] and r["outcome"] == "PASS" and len(r["trees"]) == 2
    assert os.path.isfile(os.path.join(dest, "J1-r0-L0_a", "raw", "x"))
    assert os.path.isfile(os.path.join(jd, "local_leftovers", "logs", "J1-r0-L0_a.launcher.out"))
    assert os.path.isfile(os.path.join(jd, "shm_leftovers", "snap", "r0-L0", "123.bin"))
    # second sweep re-verifies (no copy) and still passes
    assert copyback.sweep(local, shm, dest, jd)["passed"]
    # a file that appears later in the local tree breaks the verification
    make_tree(os.path.join(local, "runs", "J1-r0-L0_a"), {"raw/late": b"2"})
    r = copyback.sweep(local, shm, dest, jd)
    assert not r["passed"] and r["outcome"] == "FAIL"
    # a stray top-level file is never silently ignored
    local2 = str(tmp_path / "L2")
    make_tree(local2, {"stray.txt": b"s"})
    r = copyback.sweep(local2, str(tmp_path / "S2"), dest, jd)
    assert not r["passed"]


# ---- launches ------------------------------------------------------------------------------------

FAKE = r'''
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--mode")
for k in ("--experiment", "--arm", "--job-id", "--register-sha", "--role", "--run-dir",
          "--snap"):
    ap.add_argument(k)
ap.add_argument("--launch-seq", type=int)
ap.add_argument("--restart-count", type=int)
a = ap.parse_args()
os.makedirs(os.path.join(a.run_dir, "raw", "tiers", "000_w"), exist_ok=True)
with open(os.path.join(a.run_dir, "raw", "launch.json"), "w") as f:
    json.dump({"arm": a.arm, "seq": a.launch_seq, "role": a.role, "snap": a.snap,
               "code_sha": os.environ.get("CAPKIT_CODE_SHA"),
               "tier": os.environ.get("CAPPROBE_TIER"),
               "patch": os.environ.get("CAPPROBE_PATCH_SHA")}, f)
if a.mode == "notready":
    os.makedirs(os.path.join(a.run_dir, "control"), exist_ok=True)
    with open(os.path.join(a.run_dir, "control", "stop.json"), "w") as f:
        json.dump({"reason": "not_ready_before_origin"}, f)
    sys.exit(5)
if a.mode == "sleep":
    time.sleep(60)
sys.exit({"ok": 0, "env": 6, "fail": 4, "notready": 5}[a.mode])
'''


def launch_env(tmp_path, modes, budgets=None, margin=1):
    fake = tmp_path / "fake_launch.py"
    fake.write_text(FAKE)
    reg = "b" * 64
    plan = {"schema": "capkit.launch_plan/1", "experiment": "E0", "register_sha": reg,
            "job_arm": "job", "margin_s": margin,
            "launches": [{"launch_seq": i, "arm": "arm%d" % i,
                          "budget_s": (budgets or {}).get(i, 60),
                          "argv": ["{PY}", str(fake), "--mode", m, "--snap", "{SNAP}"]}
                         for i, m in enumerate(modes)]}
    a = types.SimpleNamespace(
        python=sys.executable, snap=str(tmp_path / "snap"),
        job_dir=str(tmp_path / "runs" / "J9-r0"),
        shm_root=str(tmp_path / "shm"), local_root=str(tmp_path / "local"),
        run_root=str(tmp_path / "runs"), job_id="9", restart_count=0, deadline_epoch=1e12)
    os.makedirs(a.job_dir)
    manifest = {"tier": "P2", "groups": ["count", "lag"], "capkit_code_sha": "c" * 40,
                "target_commit": "d" * 40}
    return a, plan, manifest


def test_launches_statuses(tmp_path):
    a, plan, manifest = launch_env(tmp_path, ["ok", "fail", "notready", "env", "ok"])
    os.makedirs(os.path.join(a.shm_root, "snap", "r0-L0"))
    with open(os.path.join(a.shm_root, "snap", "r0-L0", "77.ts.capbin"), "wb") as f:
        f.write(b"ts")
    res = launches.run_all(a, plan, manifest)
    assert [r["status"] for r in res] == ["VALID", "VOID", "NOT_RUN", "ENV_INVALID", "NOT_RUN"]
    assert [r["ran"] for r in res] == [True, True, True, True, False]
    assert "ENV_INVALID" in res[4]["reason"]
    t0 = os.path.join(a.run_root, "J9-r0-L0_arm0")
    states = read_state(t0)
    assert len(states) == 1 and states[0][1] == "VALID" and "copy-back verified" in states[0][3]
    doc = json.load(open(os.path.join(t0, "raw", "launch.json")))
    assert doc == {"arm": "arm0", "seq": 0, "role": "launch", "snap": a.snap,
                   "code_sha": "c" * 40, "tier": "P2", "patch": "d" * 40}
    assert os.path.isfile(os.path.join(t0, "raw", "probe_ts", "77.ts.capbin"))
    assert not os.access(os.path.join(t0, "raw"), os.W_OK)
    assert read_state(os.path.join(a.run_root, "J9-r0-L2_arm2"))[0][1] == "NOT_RUN"
    assert not os.path.exists(os.path.join(a.run_root, "J9-r0-L4_arm4"))
    rec = json.load(open(os.path.join(a.job_dir, "copyback", "J9-r0-L0_arm0.verify.json")))
    assert rec["passed"] and rec["files_expected"] == 2


def test_launches_deadline_and_timeout(tmp_path):
    a, plan, manifest = launch_env(tmp_path, ["sleep", "ok"], budgets={0: 1, 1: 100}, margin=5)
    res = launches.run_all(a, plan, manifest, now_fn=lambda: a.deadline_epoch - 50)
    assert [r["status"] for r in res] == ["VOID", "NOT_RUN"]
    assert res[0]["timed_out"] is True and "budget" in res[0]["reason"]
    assert "left < budget" in res[1]["reason"]


def test_launches_p0_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPPROBE_PATCH_SHA", "stale")
    a, plan, manifest = launch_env(tmp_path, ["ok"])
    manifest = dict(manifest, tier="P0", groups=[])
    res = launches.run_all(a, plan, manifest)
    doc = json.load(open(os.path.join(a.run_root, "J9-r0-L0_arm0", "raw", "launch.json")))
    assert res[0]["status"] == "VALID" and doc["tier"] == "P0" and doc["patch"] is None


def test_launches_main_writes_record(tmp_path):
    a, plan, manifest = launch_env(tmp_path, ["ok"])
    pp, mp = tmp_path / "plan.json", tmp_path / "manifest.json"
    pp.write_text(json.dumps(plan))
    mp.write_text(json.dumps(manifest))
    out = os.path.join(a.job_dir, "launches.json")
    rc = launches.main(["--plan", str(pp), "--overlay-manifest", str(mp), "--snap", a.snap,
                        "--job-dir", a.job_dir, "--local-root", a.local_root, "--shm-root",
                        a.shm_root, "--run-root", a.run_root, "--job-id", "9", "--python",
                        sys.executable, "--out", out, "--restart-count", "0",
                        "--deadline-epoch", "1e12"])
    assert rc == 0
    doc = json.load(open(out))
    assert doc["planned"] == 1 and doc["ran"] == 1 and doc["statuses"]["VALID"] == 1
    assert doc["identity"]["arm"] == "job" and doc["identity"]["role"] == "jobs"
    mode = os.stat(os.path.join(a.run_root, "J9-r0-L0_arm0", "raw", "launch.json")).st_mode
    assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def test_overlay_detects_readback_mismatch(tmp_path, monkeypatch):
    """Simulate a destination that does not return what was written (e.g. a broken overlay fs)."""
    import builtins
    import io
    src, dest, doc = make_overlay(tmp_path)
    real_open = builtins.open
    target = os.path.join(str(dest), "capx", "__init__.py")

    def fake_open(path, mode="r", *a, **kw):
        if str(path) == target and mode == "rb":
            return io.BytesIO(b"X=9\n")
        return real_open(path, mode, *a, **kw)
    monkeypatch.setattr(overlay, "open", fake_open, raising=False)
    results, passed, expected = overlay.apply(doc, str(src), str(dest))
    assert not passed and expected == 2
    assert results[1]["problem"] == "read-back bytes differ" and results[0]["ok"]
