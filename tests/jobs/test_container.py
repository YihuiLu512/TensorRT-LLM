# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container-side modules exercised on local temp dirs (functional only)."""

import json
import os
import stat
import subprocess
import sys
import time
import types

import pytest

from capcommon.hashing import sha256_bytes
from capcommon.runtree import read_state
from conftest import CODE_ROOT

from capjobs import copyback, envfiles, launches, launchplan, overlay

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
    # K9: a mounted path whose realpath is inside a model library fails even if it is readable
    lib = tmp_path / "lib"
    (lib / "m").mkdir(parents=True)
    (lib / "m" / "w").write_text("x")
    link = tmp_path / "data"
    os.symlink(str(lib / "m"), str(link))
    r = envfiles.check_mounts(["%s:%s" % (link, link)], [], forbidden=(str(lib),))
    assert not r["passed"] and "model library" in r["items"][-1]["state"]
    r = envfiles.check_mounts(["%s:%s" % (good, good)], [], forbidden=(str(lib),))
    assert not r["passed"]  # positive control of the fixture: lib itself is present, non-empty
    r = envfiles.check_mounts(["%s:%s" % (good, good)], [], forbidden=(absent,))
    assert r["passed"]


def fake_smi(tmp_path, gpus, apps, rc=0, text=""):
    tag = "%d-%d-%d-%d" % (len(gpus), len(apps), rc, len(text))
    g, c = tmp_path / ("gpus-" + tag), tmp_path / ("apps-" + tag)
    g.write_text("".join(x + "\n" for x in gpus) + text)
    c.write_text("".join(x + "\n" for x in apps))
    p = tmp_path / ("smi-" + tag)
    p.write_text('#!/bin/bash\nif [[ $1 == --query-gpu=* ]]; then cat "%s"; else cat "%s"; fi\n'
                 'exit %d\n' % (g, c, rc))
    p.chmod(0o755)
    return str(p)


def gpu_row(i):
    return "%d, GPU-%d, 00000000:%02X:00.0, NVIDIA B200, 183359, 0, 0, 580.1" % (i, i, 0x1B + i)


def test_gpu_idle(tmp_path):
    one, two = [gpu_row(0)], [gpu_row(0), gpu_row(1)]
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, one, []), 1)
    assert r["passed"] and r["positive_control_ok"] and r["gpu_count"] == 1
    assert r["queries"][0]["cmd"][0].startswith(str(tmp_path))  # the explicit path was run
    # K6: visible != requested is a failure (was accepted before: 8 visible, 1 requested)
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, two, []), 1)
    assert not r["passed"] and r["outcome"] == "FAIL" and r["gpu_count"] == 2
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, one, ["00000000:1B:00.0, 4242, 1304"]), 1)
    assert not r["passed"] and r["outcome"] == "FAIL" and len(r["compute_apps"]) == 1
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, [], []), 1)
    assert not r["passed"] and r["outcome"] == "NOT_RUN" and not r["positive_control_ok"]
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, one, [], rc=9), 1)
    assert not r["passed"] and r["outcome"] == "NOT_RUN"
    r = envfiles.check_gpu_idle(str(tmp_path / "no-such-smi"), 1)
    assert not r["passed"]
    # K6: 0 requested -> 0 must be visible; no positive control is possible (recorded NOT_RUN)
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, [], [], rc=6, text="No devices were found\n"),
                                0)
    assert r["passed"] and r["gpu_count"] == 0 and not r["positive_control_ok"]
    assert r["positive_control"]["status"] == "NOT_RUN"
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, one, []), 0)
    assert not r["passed"] and r["outcome"] == "FAIL"
    r = envfiles.check_gpu_idle(fake_smi(tmp_path, [], [], rc=9), 0)
    assert not r["passed"] and r["outcome"] == "NOT_RUN"


def test_gpu_idle_cli_requires_explicit_inputs(tmp_path):
    out = str(tmp_path / "g.json")
    assert envfiles.main(["gpu-idle", "--out", out, "--nvidia-smi", "x"] + ident_args()) == 6
    assert envfiles.main(["gpu-idle", "--out", out, "--max-alloc-gpus", "1"] + ident_args()) == 6
    assert not os.path.exists(out)


def test_image_and_disk_and_imports(tmp_path):
    f = tmp_path / "check.tsv"
    rows = {"path": "/x.sqsh", "expected_sha256": "a" * 64, "sha256_source": "capjobs.submit",
            "expected_size": "10", "expected_mtime_ns": "5", "measured_size": "10",
            "measured_mtime_ns": "5"}
    f.write_text("".join("%s\t%s\n" % kv for kv in rows.items()))
    r = envfiles.check_image(str(f))
    assert r["passed"] and r["sha256"] == "a" * 64
    rows["measured_mtime_ns"] = "6"
    f.write_text("".join("%s\t%s\n" % kv for kv in rows.items()))
    r = envfiles.check_image(str(f))
    assert not r["passed"] and r["sha256"] is None
    rows["measured_mtime_ns"] = ""
    rows["expected_mtime_ns"] = ""
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
    # K8: same size, different bytes is a difference too
    with open(os.path.join(dst, "raw", "launch.json"), "wb") as f:
        f.write(b"[]")
    bad = copyback.compare(src, dst)
    assert not bad["passed"] and bad["differences"] == ["raw/launch.json"]
    assert bad["content_check"] == "sha256"
    empty = str(tmp_path / "empty")
    os.makedirs(os.path.join(empty, "raw"))
    assert copyback.compare(empty, str(tmp_path / "nowhere"))["outcome"] == "NOT_RUN"
    assert copyback.lock_raw(dst)
    assert not os.access(os.path.join(dst, "raw"), os.W_OK)
    assert os.access(dst, os.W_OK)  # tree root stays writable (STATE, judge/)


def test_copy_tree_with_launcher_state(tmp_path):
    """The launcher writes STATE into the local tree; the copy gains lines after it. Only an
    append is accepted (the copy's STATE must start with the source's bytes)."""
    src = str(tmp_path / "T")
    make_tree(src, {"raw/a": b"1", "STATE": b"t0\tVALID\tcaplaunch\tok\n"})
    dst = str(tmp_path / "D" / "T")
    rec = copyback.copy_tree(src, dst)
    assert rec["passed"] and rec["files_expected"] == 2, rec["differences"]
    with open(os.path.join(dst, "STATE"), "ab") as f:
        f.write(b"t1\tVALID\tcapjobs.launches\tcopied\n")
    assert copyback.compare(src, dst)["passed"]
    with open(os.path.join(dst, "STATE"), "wb") as f:
        f.write(b"t0\tVOID\tcaplaunch\tok\nmore\n")
    assert copyback.compare(src, dst)["differences"] == ["STATE"]


def test_sweep_skips_roots_not_owned(tmp_path):
    local = str(tmp_path / "L")
    make_tree(os.path.join(local, "runs", "J1-r0-L0_a"), {"raw/x": b"1"})
    jd = str(tmp_path / "runs" / "J1-r0")
    os.makedirs(jd)
    r = copyback.sweep(None, None, str(tmp_path / "runs"), jd)
    assert r["outcome"] == "EMPTY" and r["local_root"] is None
    assert not os.path.exists(os.path.join(str(tmp_path / "runs"), "J1-r0-L0_a"))


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
import argparse, json, os, signal, subprocess, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--mode")
for k in ("--experiment", "--arm", "--job-id", "--register-sha", "--role", "--run-dir",
          "--snap", "--overlay-manifest", "--nvidia-smi", "--job-env-dir", "--scenario",
          "--kill-after-s", "--workdir", "--pin-table"):
    ap.add_argument(k, required=k != "--mode")
ap.add_argument("--launch-seq", type=int)
ap.add_argument("--restart-count", type=int)
a = ap.parse_args()
if a.mode == "config":  # caplaunch refusing its inputs before it writes anything
    print("caplaunch: scenario x does not match register entry", file=sys.stderr)
    sys.exit(3)
os.makedirs(os.path.join(a.run_dir, "raw", "tiers", "000_w"), exist_ok=True)
with open(os.path.join(a.run_dir, "raw", "launch.json"), "w") as f:
    json.dump({"arm": a.arm, "seq": a.launch_seq, "role": a.role, "snap": a.snap,
               "code_sha": os.environ.get("CAPKIT_CODE_SHA"),
               "tier": os.environ.get("CAPPROBE_TIER"),
               "patch": os.environ.get("CAPPROBE_PATCH_SHA"),
               "overlay_manifest": a.overlay_manifest, "nvidia_smi": a.nvidia_smi,
               "job_env_dir": a.job_env_dir, "pin_table": a.pin_table}, f)
def orphan(ignore_term):
    # a role in its own session, as caplaunch starts them; it outlives this launcher
    code = ("import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN) if %r else None; "
            "time.sleep(60)" % ignore_term)
    p = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    with open(os.path.join(a.run_dir, "raw", "orphan.pid"), "w") as f:
        f.write(str(p.pid))
if a.mode == "drain":  # caplaunch's own record: VALID, stopped early at a drain gate, exit 5
    with open(os.path.join(a.run_dir, "STATE"), "a") as f:
        f.write("2026-11-13T00:00:00.000Z\tVALID\tcaplaunch\tstopped: drain_gate_timeout\n")
    sys.exit(5)
if a.mode == "voidstate":  # caplaunch recorded VOID but exited 0
    with open(os.path.join(a.run_dir, "STATE"), "a") as f:
        f.write("2026-11-13T00:00:00.000Z\tVOID\tcaplaunch\tleftover processes\n")
    sys.exit(0)
if a.mode == "orphan":
    orphan(False)
    sys.exit(0)
if a.mode == "hang_orphan":  # overruns its budget and leaves a role that ignores SIGTERM
    orphan(True)
    time.sleep(60)
if a.mode == "notready":
    os.makedirs(os.path.join(a.run_dir, "control"), exist_ok=True)
    with open(os.path.join(a.run_dir, "control", "stop.json"), "w") as f:
        json.dump({"reason": "not_ready_before_origin"}, f)
    sys.exit(5)
if a.mode == "sleep":
    time.sleep(60)
if a.mode == "owndir":  # caplaunch's own --run-dir check (capcommon.snapshot.ensure_own_dir)
    from capcommon.snapshot import SnapshotError, ensure_own_dir
    try:
        ensure_own_dir(a.run_dir)
    except SnapshotError as e:
        print(e, file=sys.stderr)
        sys.exit(3)
    sys.exit(0)
sys.exit({"ok": 0, "env": 6, "fail": 4, "notready": 5, "config_late": 3}[a.mode])
'''


HW_SHA = "e" * 64


def launch_env(tmp_path, modes, budgets=None, margin=1):
    fake = tmp_path / "fake_launch.py"
    fake.write_text(FAKE)
    # {PY}: the container python; here a wrapper that finds the capkit packages (capenv.pins)
    py = tmp_path / "py"
    py.write_text("#!/bin/sh\nPYTHONPATH=%s exec %s \"$@\"\n"
                  % (os.path.join(CODE_ROOT, "tools"), sys.executable))
    py.chmod(0o755)
    w = tmp_path / "W"
    reg = "b" * 64
    plan = {"schema": "capkit.launch_plan/1", "experiment": "E0", "register_sha": reg,
            "job_arm": "job", "margin_s": margin, "kill_grace_s": 1,
            "launches": [{"launch_seq": i, "arm": "arm%d" % i,
                          "budget_s": (budgets or {}).get(i, 60),
                          "argv": ["{PY}", str(fake), "--mode", m, "--snap", "{SNAP}",
                                   "--workdir", str(w), "--scenario", "scen.json",
                                   "--kill-after-s", "0.1"]}
                         for i, m in enumerate(modes)]}
    a = types.SimpleNamespace(
        python=str(py), snap=str(tmp_path / "snap"),
        job_dir=str(tmp_path / "runs" / "J9-r0"),
        shm_root=str(tmp_path / "shm"), local_root=str(tmp_path / "local"),
        run_root=str(tmp_path / "runs"), job_id="9", restart_count=0, deadline_epoch=1e12,
        overlay_manifest=str(tmp_path / "snap" / "overlay_manifest.json"),
        nvidia_smi="/opt/x/nvidia-smi",
        hw_profile=str(tmp_path / "runs" / "J9-r0" / "env" / "hw_profile.json"))
    os.makedirs(os.path.dirname(a.hw_profile))
    with open(a.hw_profile, "w") as f:
        json.dump({"profile_sha256": HW_SHA}, f)
    os.makedirs(a.snap)
    os.makedirs(str(w))
    os.mkdir(a.local_root, 0o700)  # capkit_inner.sh creates it (mkdir -m 700) before launches
    with open(os.path.join(str(w), "scen.json"), "w") as f:  # worst-case stop 0.4 s <= 1 s
        json.dump({"stop_grace_s": 0.1, "stop_order": ["role"],
                   "roles": [{"pin_role": "role", "cpus_needed": 1, "numa": None}]}, f)
    manifest = {"tier": "P2", "groups": ["count", "lag"], "capkit_code_sha": "c" * 40,
                "target_commit": "d" * 40}
    return a, plan, manifest


def test_launches_statuses(tmp_path):
    a, plan, manifest = launch_env(tmp_path, ["ok", "fail", "notready", "env", "ok"])
    res = launches.run_all(a, plan, manifest)
    assert [r["status"] for r in res] == ["VALID", "VOID", "NOT_RUN", "ENV_INVALID", "NOT_RUN"]
    assert [r["ran"] for r in res] == [True, True, True, True, False]
    assert "ENV_INVALID" in res[4]["reason"]
    t0 = os.path.join(a.run_root, "J9-r0-L0_arm0")
    states = read_state(t0)
    assert len(states) == 1 and states[0][1] == "VALID" and "copy-back verified" in states[0][3]
    doc = json.load(open(os.path.join(t0, "raw", "launch.json")))
    assert doc == {"arm": "arm0", "seq": 0, "role": "launch", "snap": a.snap,
                   "code_sha": "c" * 40, "tier": "P2", "patch": "d" * 40,
                   "overlay_manifest": a.overlay_manifest, "nvidia_smi": "/opt/x/nvidia-smi",
                   "job_env_dir": os.path.join(a.job_dir, "env"),
                   "pin_table": os.path.join(a.job_dir, "pins", "J9-r0-L0_arm0.pin_table.json")}
    assert not os.access(os.path.join(t0, "raw"), os.W_OK)
    assert read_state(os.path.join(a.run_root, "J9-r0-L2_arm2"))[0][1] == "NOT_RUN"
    assert not os.path.exists(os.path.join(a.run_root, "J9-r0-L4_arm4"))
    rec = json.load(open(os.path.join(a.job_dir, "copyback", "J9-r0-L0_arm0.verify.json")))
    assert rec["passed"] and rec["files_expected"] == 1


def test_launches_generate_the_pin_table_on_the_node(tmp_path):
    """SM-2: each launch gets a pin table generated in the job (capenv.pins on this node) with
    this launch's identity and the job's hardware profile sha, inside this process's affinity;
    it is passed as --pin-table (a plan cannot give one)."""
    a, plan, manifest = launch_env(tmp_path, ["ok", "ok"])
    res = launches.run_all(a, plan, manifest)
    assert [r["status"] for r in res] == ["VALID", "VALID"], [r["reason"] for r in res]
    for i in (0, 1):
        path = os.path.join(a.job_dir, "pins", "J9-r0-L%d_arm%d.pin_table.json" % (i, i))
        doc = json.load(open(path))
        ident = doc["identity"]
        assert (ident["experiment"], ident["arm"], ident["launch_seq"], ident["job_id"],
                ident["restart_count"], ident["register_sha"], ident["role"]) == \
            ("E0", "arm%d" % i, i, "9", 0, "b" * 64, "pins")
        assert doc["source_sha"] == HW_SHA and doc["demand"][0]["role"] == "role"
        assert set(doc["roles"]["role"]["cpus"]) <= os.sched_getaffinity(0)
        assert res[i]["argv"][res[i]["argv"].index("--pin-table") + 1] == path
    bad = dict(plan, launches=[dict(plan["launches"][0], argv=plan["launches"][0]["argv"]
                                    + ["--pin-table", "/x.json"])])
    with pytest.raises(launchplan.PlanError, match="--pin-table"):
        launchplan.validate(bad)


def test_launch_without_pin_table_is_not_run(tmp_path):
    """No hardware profile (or a demand that does not fit) -> no pin table -> the launch is
    skipped as NOT_RUN; nothing is started."""
    a, plan, manifest = launch_env(tmp_path, ["ok"])
    os.remove(a.hw_profile)
    res = launches.run_all(a, plan, manifest)
    assert res[0]["status"] == "NOT_RUN" and res[0]["ran"] is False
    assert "pin table" in res[0]["reason"] and res[0]["pin_table_failed"] is True
    (tmp_path / "big").mkdir()
    a, plan, manifest = launch_env(tmp_path / "big", ["ok"])
    with open(os.path.join(str(tmp_path / "big" / "W"), "scen.json"), "w") as f:
        json.dump({"stop_grace_s": 0.1, "stop_order": ["role"],
                   "roles": [{"pin_role": "role", "cpus_needed": 100000, "numa": None}]}, f)
    res = launches.run_all(a, plan, manifest)
    assert res[0]["status"] == "NOT_RUN" and "capenv.pins generate exit 3" in res[0]["reason"]
    assert res[0]["pin_table_failed"] is True


def test_config_error_before_launch_json_is_not_run(tmp_path, monkeypatch):
    """SM-5: caplaunch exit 3 before it wrote raw/launch.json (no role can have started) is
    NOT_RUN with the launcher's last stderr line, and its empty tree is not a copy-back failure;
    exit 3 once launch.json exists stays VOID. R5-3: such a refusal is marked config_refused and
    makes capjobs.launches exit 3 (CONFIG), so the job does not end as a success."""
    monkeypatch.setattr(launches, "become_subreaper", lambda: False)  # not on pytest itself
    a, plan, manifest = launch_env(tmp_path, ["config", "config_late"])
    res = launches.run_all(a, plan, manifest)
    assert [r["status"] for r in res] == ["NOT_RUN", "VOID"], [r["reason"] for r in res]
    assert [r["config_refused"] for r in res] == [True, False]
    assert "does not match register entry" in res[0]["reason"]
    assert "copy-back not verified" not in res[0]["reason"]
    assert "nothing to copy back" in res[0]["reason"]
    assert res[1]["reason"].startswith("launcher exit 3") or "exit 3" in res[1]["reason"]
    (tmp_path / "m").mkdir()
    b, plan, manifest = launch_env(tmp_path / "m", ["config"])
    pp, mp = tmp_path / "plan.json", tmp_path / "manifest.json"
    pp.write_text(json.dumps(plan))
    mp.write_text(json.dumps(manifest))
    out = os.path.join(b.job_dir, "launches.json")
    assert launches.main(["--plan", str(pp), "--overlay-manifest", str(mp), "--nvidia-smi",
                          b.nvidia_smi, "--snap", b.snap, "--hw-profile", b.hw_profile,
                          "--job-dir", b.job_dir, "--local-root", b.local_root, "--shm-root",
                          b.shm_root, "--run-root", b.run_root, "--job-id", "9", "--python",
                          b.python, "--out", out, "--restart-count", "0",
                          "--deadline-epoch", "1e12"]) == 3  # CONFIG, not a copy-back failure
    doc = json.load(open(out))
    assert doc["config_refused"] == 1 and doc["statuses"]["NOT_RUN"] == 1


@pytest.mark.parametrize("kinds,want", [
    (["ok", "config"], 3),       # one refused launch among measured ones still exits CONFIG
    (["ok"], 0),
    (["nopins"], 3),             # RS22: a pin table that cannot be generated, the same way
])
def test_launches_exit_code_with_config_refusals(tmp_path, monkeypatch, kinds, want):
    monkeypatch.setattr(launches, "become_subreaper", lambda: False)  # not on pytest itself
    nopins = kinds == ["nopins"]
    b, plan, manifest = launch_env(tmp_path, ["ok"] if nopins else kinds)
    if nopins:
        os.remove(b.hw_profile)
    pp, mp = tmp_path / "plan.json", tmp_path / "manifest.json"
    pp.write_text(json.dumps(plan))
    mp.write_text(json.dumps(manifest))
    out = os.path.join(b.job_dir, "launches.json")
    assert launches.main(["--plan", str(pp), "--overlay-manifest", str(mp), "--nvidia-smi",
                          b.nvidia_smi, "--snap", b.snap, "--hw-profile", b.hw_profile,
                          "--job-dir", b.job_dir, "--local-root", b.local_root, "--shm-root",
                          b.shm_root, "--run-root", b.run_root, "--job-id", "9", "--python",
                          b.python, "--out", out, "--restart-count", "0",
                          "--deadline-epoch", "1e12"]) == want
    doc = json.load(open(out))
    assert doc["config_refused"] == kinds.count("config")
    assert doc["pin_table_failed"] == kinds.count("nopins")
    if not nopins:
        assert [r["config_refused"] for r in doc["launches"]] == [k == "config" for k in kinds]


def test_launch_trees_are_private_under_umask_002(tmp_path, monkeypatch):
    """The local launch tree (caplaunch's --run-dir) and its parents are 0700 whatever the
    container's umask: under umask 002 caplaunch's ensure_own_dir still accepts --run-dir."""
    monkeypatch.setenv("PYTHONPATH", os.path.join(CODE_ROOT, "tools"))
    a, plan, manifest = launch_env(tmp_path, ["owndir"])
    old = os.umask(0o002)
    try:
        res = launches.run_all(a, plan, manifest)
    finally:
        os.umask(old)
    assert res[0]["status"] == "VALID", res[0]["reason"]
    for d in ("runs", "logs", os.path.join("runs", "J9-r0-L0_arm0")):
        assert stat.S_IMODE(os.lstat(os.path.join(a.local_root, d)).st_mode) == 0o700, d
    os.symlink(str(tmp_path), os.path.join(str(tmp_path), "lnk"))
    with pytest.raises(OSError):
        launches.private_mkdir(os.path.join(str(tmp_path), "lnk"), exist_ok=True)


def test_launches_deadline_and_timeout(tmp_path):
    a, plan, manifest = launch_env(tmp_path, ["sleep", "ok"], budgets={0: 1, 1: 100}, margin=5)
    left = 1 + plan["kill_grace_s"] + 5 + launchplan.PINS_TIMEOUT_S + 20
    res = launches.run_all(a, plan, manifest, now_fn=lambda: a.deadline_epoch - left)
    assert [r["status"] for r in res] == ["VOID", "NOT_RUN"]
    assert res[0]["timed_out"] is True and "budget" in res[0]["reason"]
    assert res[0]["terminated"] is False
    assert "left < budget" in res[1]["reason"]


@pytest.mark.parametrize("short", [1, 0])
def test_remaining_time_check_counts_the_pin_table_bound(tmp_path, short):
    """U10 / RS7: the remaining-time check before a launch includes the pin-table generation
    bound (launchplan.PINS_TIMEOUT_S), which runs inside the allocation before the launch."""
    a, plan, manifest = launch_env(tmp_path, ["ok"], budgets={0: 30}, margin=5)
    need = 30 + plan["kill_grace_s"] + 5 + launchplan.PINS_TIMEOUT_S
    res = launches.run_all(a, plan, manifest, now_fn=lambda: a.deadline_epoch - need + short)
    if short:
        assert res[0]["status"] == "NOT_RUN" and res[0]["ran"] is False
        assert "+ pin table %d s" % launchplan.PINS_TIMEOUT_S in res[0]["reason"]
    else:
        assert res[0]["ran"] is True and res[0]["status"] == "VALID", res[0]["reason"]


def test_unwritable_demand_file_is_not_run(tmp_path, monkeypatch):
    """U5: an OSError writing the pin-table demand file skips that launch as NOT_RUN instead of
    escaping run_all (which lost launches.json for the whole job)."""
    a, plan, manifest = launch_env(tmp_path, ["ok", "ok"])
    real_open = open

    def fake_open(path, *args, **kw):
        if str(path).endswith(".demand.json") and "L0_" in str(path):
            raise OSError(28, "No space left on device")
        return real_open(path, *args, **kw)
    monkeypatch.setattr("builtins.open", fake_open)
    res = launches.run_all(a, plan, manifest)
    monkeypatch.undo()
    assert res[0]["status"] == "NOT_RUN" and res[0]["ran"] is False
    assert "cannot write" in res[0]["reason"] and "No space left" in res[0]["reason"]
    assert res[1]["status"] == "VALID", res[1]["reason"]


def _gone(pid, timeout_s=10):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        try:
            with open("/proc/%d/stat" % pid) as f:
                if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True
        except OSError:
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("stop", ["not_ready_before_origin", "overlay_manifest_missing",
                                  "job_env_dir_missing", "orchestrator_signalled"])
def test_launcher_stop_reasons_are_not_run(stop):
    """caplaunch's early stops are NOT_RUN: from its STATE line, and from stop.json with exit 5
    when STATE is missing; any other exit code with such a stop stays VOID."""
    rec = ("", "NOT_RUN", "caplaunch", "stopped: " + stop)
    assert launches.classify(5, False, stop, rec)[0] == "NOT_RUN"
    assert launches.classify(5, False, stop, None)[0] == "NOT_RUN"
    assert launches.classify(1, False, stop, None)[0] == "VOID"
    assert launches.classify(5, True, stop, rec)[0] == "VOID"  # killed for its budget


def test_job_sigterm_is_not_run_unless_the_launcher_recorded_worse():
    """Q10: a launch stopped because the job got SIGTERM did not finish measuring: NOT_RUN (as
    caplaunch records its own signal stop), never VOID by itself; a non-VALID launcher STATE is
    kept, a VALID one is not upgraded past NOT_RUN."""
    def st(status):
        return ("", status, "caplaunch", "x")
    assert launches.classify(None, False, None, None, terminated=True)[0] == "NOT_RUN"
    assert launches.classify(-15, False, None, None, terminated=True)[1].startswith("terminated")
    assert launches.classify(5, False, "orchestrator_signalled", st("NOT_RUN"),
                             terminated=True)[0] == "NOT_RUN"
    assert launches.classify(0, False, None, st("VALID"), terminated=True)[0] == "NOT_RUN"
    assert launches.classify(6, False, None, st("ENV_INVALID"), terminated=True)[0] == \
        "ENV_INVALID"
    assert launches.classify(0, False, None, st("VOID"), terminated=True)[0] == "VOID"
    assert launches.classify(None, True, None, None, terminated=False)[0] == "VOID"


def test_budget_kill_never_improves_the_launcher_record():
    """RS14: a budget kill makes the launcher's record worse (VALID/NOT_RUN -> VOID) but keeps an
    ENV_INVALID it already recorded."""
    def st(status):
        return ("", status, "caplaunch", "x")
    assert launches.classify(None, True, None, st("VALID"))[0] == "VOID"
    assert launches.classify(None, True, None, st("NOT_RUN"))[0] == "VOID"
    assert launches.classify(-9, True, None, st("ENV_INVALID"))[0] == "ENV_INVALID"
    assert launches.classify(6, True, None, st("ENV_INVALID"), terminated=True)[0] == \
        "ENV_INVALID"


def test_launcher_state_is_kept(tmp_path):
    """K15 + copy-back of a tree that holds the launcher's STATE: the launcher's own VALID
    (exit 5, drain gate timeout) stays VALID; its VOID is not upgraded by exit 0."""
    a, plan, manifest = launch_env(tmp_path, ["drain", "voidstate"])
    res = launches.run_all(a, plan, manifest)
    assert [r["status"] for r in res] == ["VALID", "VOID"], [r["reason"] for r in res]
    assert "drain_gate_timeout" in res[0]["reason"] and "copy-back verified" in res[0]["reason"]
    states = read_state(os.path.join(a.run_root, "J9-r0-L0_arm0"))
    assert [(s[1], s[2]) for s in states] == [("VALID", "caplaunch"),
                                               ("VALID", "capjobs.launches")]
    assert launches.classify(4, False, None, ("", "VALID", "caplaunch", "x"))[0] == "VOID"


def run_main(a, plan, manifest, tmp_path):
    """capjobs.launches as the inner script runs it: its own process (a child subreaper)."""
    pp, mp = tmp_path / "plan.json", tmp_path / "manifest.json"
    pp.write_text(json.dumps(plan))
    mp.write_text(json.dumps(manifest))
    out = os.path.join(a.job_dir, "launches.json")
    env = dict(os.environ, PYTHONPATH=os.path.join(CODE_ROOT, "tools"))
    r = subprocess.run([sys.executable, "-B", "-m", "capjobs.launches", "--plan", str(pp),
                        "--overlay-manifest", str(mp), "--nvidia-smi", a.nvidia_smi, "--snap",
                        a.snap, "--job-dir", a.job_dir, "--local-root", a.local_root,
                        "--shm-root", a.shm_root, "--run-root", a.run_root, "--job-id", "9",
                        "--hw-profile", a.hw_profile,
                        "--python", a.python, "--out", out, "--restart-count", "0",
                        "--deadline-epoch", "1e12"], capture_output=True, text=True, env=env,
                       timeout=120)
    doc = json.load(open(out))
    return r, doc


def test_leftover_processes_are_killed(tmp_path):
    """K14: roles in their own sessions do not outlive the launch (normal exit and timeout)."""
    a, plan, manifest = launch_env(tmp_path, ["orphan", "hang_orphan", "ok"],
                                   budgets={1: 2})
    r, doc = run_main(a, plan, manifest, tmp_path)
    assert doc["child_subreaper"] is True, r.stderr
    res = doc["launches"]
    pids = [int(open(os.path.join(a.run_root, "J9-r0-L%d_arm%d" % (i, i), "raw",
                                  "orphan.pid")).read()) for i in (0, 1)]
    assert all(_gone(p) for p in pids), pids
    assert res[0]["status"] == "VALID" and [x["pid"] for x in res[0]["leftovers"]] == [pids[0]]
    assert res[0]["leftovers"][0]["signals"] == ["SIGTERM"]
    assert res[1]["status"] == "VOID" and res[1]["timed_out"] is True
    left1 = {x["pid"]: x for x in res[1]["leftovers"]}
    assert left1[pids[1]]["signals"] == ["SIGTERM", "SIGKILL"]
    assert not left1[pids[1]]["survived_sigkill"]
    assert res[2]["status"] == "VALID" and res[2]["leftovers"] == []


def test_launches_p0_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPPROBE_PATCH_SHA", "stale")
    a, plan, manifest = launch_env(tmp_path, ["ok"])
    manifest = dict(manifest, tier="P0", groups=[])
    res = launches.run_all(a, plan, manifest)
    doc = json.load(open(os.path.join(a.run_root, "J9-r0-L0_arm0", "raw", "launch.json")))
    assert res[0]["status"] == "VALID" and doc["tier"] == "P0" and doc["patch"] is None


def test_launches_main_writes_record(tmp_path, monkeypatch):
    monkeypatch.setattr(launches, "become_subreaper", lambda: False)  # not on pytest itself
    a, plan, manifest = launch_env(tmp_path, ["ok"])
    pp, mp = tmp_path / "plan.json", tmp_path / "manifest.json"
    pp.write_text(json.dumps(plan))
    mp.write_text(json.dumps(manifest))
    out = os.path.join(a.job_dir, "launches.json")
    rc = launches.main(["--plan", str(pp), "--overlay-manifest", str(mp), "--nvidia-smi",
                        "/opt/x/nvidia-smi", "--snap", a.snap, "--hw-profile", a.hw_profile,
                        "--job-dir", a.job_dir, "--local-root", a.local_root, "--shm-root",
                        a.shm_root, "--run-root", a.run_root, "--job-id", "9", "--python",
                        a.python, "--out", out, "--restart-count", "0",
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


def test_sigterm_stops_launch_and_skips_the_rest(tmp_path):
    """RS4: Slurm's SIGTERM to the step reaches capjobs.launches; it stops the running launch
    (its whole tree), skips the rest, and still writes its record."""
    a, plan, manifest = launch_env(tmp_path, ["hang_orphan", "ok"], budgets={0: 60})
    pp, mp = tmp_path / "plan.json", tmp_path / "manifest.json"
    pp.write_text(json.dumps(plan))
    mp.write_text(json.dumps(manifest))
    out = os.path.join(a.job_dir, "launches.json")
    env = dict(os.environ, PYTHONPATH=os.path.join(CODE_ROOT, "tools"))
    p = subprocess.Popen([sys.executable, "-B", "-m", "capjobs.launches", "--plan", str(pp),
                          "--overlay-manifest", str(mp), "--nvidia-smi", a.nvidia_smi, "--snap",
                          a.snap, "--job-dir", a.job_dir, "--local-root", a.local_root,
                          "--shm-root", a.shm_root, "--run-root", a.run_root, "--job-id", "9",
                          "--hw-profile", a.hw_profile,
                          "--python", a.python, "--out", out, "--restart-count", "0",
                          "--deadline-epoch", "1e12"], env=env)
    pid_file = os.path.join(a.local_root, "runs", "J9-r0-L0_arm0", "raw", "orphan.pid")
    end = time.monotonic() + 30
    while not os.path.exists(pid_file) and time.monotonic() < end:
        time.sleep(0.1)
    time.sleep(1.0)  # let launches record the orphan
    p.send_signal(15)
    assert p.wait(timeout=60) == 0
    doc = json.load(open(out))
    res = doc["launches"]
    assert res[0]["status"] == "NOT_RUN" and res[0]["terminated"] is True, res[0]["reason"]
    assert res[0]["timed_out"] is False and res[0]["reason"].startswith("terminated")
    assert doc["statuses"]["VOID"] == 0
    assert read_state(os.path.join(a.run_root, "J9-r0-L0_arm0"))[-1][1] == "NOT_RUN"
    assert res[1]["status"] == "NOT_RUN" and "SIGTERM" in res[1]["reason"]
    orphan = int(open(os.path.join(a.run_root, "J9-r0-L0_arm0", "raw", "orphan.pid")).read())
    assert _gone(orphan)
