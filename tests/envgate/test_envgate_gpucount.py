# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate item 5 visible == requested GPUs, explicit nvidia-smi path, unreadable GPU readings,
--registered-hw-sha UNREGISTERED and the required gate / hwprofile arguments.

nvidia-smi is never run for real: fake runners, or fake scripts at absolute paths."""

import json
import os
import stat
import subprocess

import pytest

from b4_envgate_util import PY, capenv_env, id_args
from capcommon import exitcodes
from capcommon.exitcodes import ConfigError
from capcommon.hwprofile import profile_sha256
from capcommon.jsonio import read_json
from capenv import gate, gpu, hwprofile
from test_envgate_gate import (GPU_LINE, JOB, Args, _ident, fake_runner, good_table,  # noqa: F401
                               jobdir, nvsmi)

GPU_LINE2 = GPU_LINE.replace("0, GPU-abc, 00000000:1B:00.0", "1, GPU-def, 00000000:2B:00.0")
NO_DEV = "No devices were found\n"


def _gate(d, prof, table, **kw):
    kw.setdefault("registered_hw_sha", profile_sha256(prof))
    return gate.run_gate(Args(d, **kw), _ident(), runner=fake_runner(table))


# ---- K6: visible == requested -----------------------------------------------------------------

def test_item5_more_gpus_visible_than_requested_is_red(jobdir):
    d, prof = jobdir
    out = _gate(d, prof, good_table(**{"nvidia-smi": lambda a: (0, "") if any(
        x.startswith("--query-compute-apps") for x in a) else (0, GPU_LINE + GPU_LINE2)}))
    assert out["items"]["5"] == "RED" and out["verdict"] == "ENV_INVALID"
    it5 = read_json(os.path.join(d, "env", "exclusivity.json"))["items"][4]
    assert it5["detail"]["visible_gpus"] == 2 and it5["detail"]["expected_gpus"] == 1
    assert it5["detail"]["count_check"]["status"] == "RED"


@pytest.mark.parametrize("smi,status", [
    ((6, NO_DEV), "PASS"),            # nvidia-smi sees no device: visible 0 == requested 0
    ((0, ""), "PASS"),                # query ran and listed nothing
    (nvsmi(), "RED"),                 # a GPU is visible although 0 were requested
    (None, "NOT_RUN"),                # nvidia-smi missing: cannot tell what is visible
    ((9, "NVIDIA-SMI has failed because it couldn't communicate with the driver\n"),
     "NOT_RUN"),
])
def test_item5_zero_requested(jobdir, smi, status):
    d, prof = jobdir
    t = good_table(scontrol=lambda a: (0, (
        "JobId=4242 OverSubscribe=NO NodeList=node-a TRES=cpu=24,mem=60G,node=1 Partition=p1"))
        if a[2] == "job" else (0, "NodeName=node-a CPUAlloc=24 CPUTot=24"))
    if smi is None:
        del t["nvidia-smi"]
    else:
        t["nvidia-smi"] = smi
    out = _gate(d, prof, t, max_alloc_gpus=0)
    assert out["items"]["5"] == status, out["reasons"]
    assert out["items"]["1"] == "PASS"
    it5 = read_json(os.path.join(d, "env", "exclusivity.json"))["items"][4]
    # the positive control and the compute-process check need a listed GPU: recorded NOT_RUN
    assert it5["detail"]["positive_control"]["status"] == "NOT_RUN"
    assert it5["detail"]["apps_check"]["status"] == "NOT_RUN"
    assert it5["positive_control_ok"] is False
    assert (out["verdict"] == "PASS") == (status == "PASS")


def test_check_visible_rejects_bad_request():
    for bad in (-1, None, True, 1.0):
        with pytest.raises(ConfigError):
            gpu.check_visible(bad, 5, runner=fake_runner({"nvidia-smi": nvsmi()}))


def test_check_visible_apps_query_failure_is_not_run():
    def smi(a):
        return (3, "") if any(x.startswith("--query-compute-apps") for x in a) else (0, GPU_LINE)
    v = gpu.check_visible(1, 5, runner=fake_runner({"nvidia-smi": smi}))
    assert v["status"] == "NOT_RUN" and v["count_check"]["status"] == "PASS"


# ---- K18: nvidia-smi only from the explicit path --------------------------------------------

def test_gate_uses_the_explicit_nvidia_smi_path(jobdir):
    d, prof = jobdir
    t = good_table()
    smi = t.pop("nvidia-smi")
    t["/opt/drv/bin/nvidia-smi"] = smi
    out = _gate(d, prof, t, nvidia_smi="/opt/drv/bin/nvidia-smi")
    assert out["verdict"] == "PASS", out["reasons"]
    # the bare name is not on the fake PATH: item 5 must not silently use something else
    out = _gate(d, prof, t, nvidia_smi="nvidia-smi")
    assert out["items"]["5"] == "NOT_RUN"


def test_gpu_queries_without_a_path_are_not_run():
    calls = fake_runner({"nvidia-smi": nvsmi()})
    ok, n, rec = gpu.positive_control(5, calls, nvsmi=None)
    assert not ok and rec["reason"] == "nvidia-smi path not given" and calls.calls == []
    rc, rows = gpu.sample(5, calls, nvsmi="")
    assert rc != 0 and rows == [] and calls.calls == []


def test_hwprofile_uses_the_explicit_nvidia_smi_path(tmp_path):
    r = fake_runner({"/x/nvidia-smi": (0, GPU_LINE)})
    p = hwprofile.Collector(str(tmp_path), "dmi", runner=r, nvidia_smi="/x/nvidia-smi").collect()
    assert p["gpu_count"] == 1 and p["gpu_model"] == "NVIDIA B200"
    p = hwprofile.Collector(str(tmp_path), "dmi", runner=r).collect()
    assert "not given" in p["gpu_count"]["unreadable"]
    # an unreadable memory.total must not crash the collector nor be hashed as a number
    na = GPU_LINE.replace("183359", "[N/A]")
    p = hwprofile.Collector(str(tmp_path), "dmi", runner=fake_runner({"/x/nvidia-smi": (0, na)}),
                            nvidia_smi="/x/nvidia-smi").collect()
    assert "unreadable" in p["gpu_mem_mib"]


# ---- K7: unreadable readings are never idle -------------------------------------------------

def test_idle_check_unreadable_util_is_not_run():
    na = GPU_LINE.replace(", 0, 0, 570.1", ", 0, [N/A], 570.1")
    out = gpu.idle_check(2, 0.0, 5, 1, runner=fake_runner({"nvidia-smi": lambda a: (0, "") if any(
        x.startswith("--query-compute-apps") for x in a) else (0, na)}), sleep=lambda s: None)
    assert out["verdict"] == "NOT_RUN" and out["unreadable_util"] == 2


def test_compute_app_with_unreadable_pid_is_busy():
    t = {"nvidia-smi": nvsmi("00000000:1B:00.0, [N/A], 512\n")}
    rc, rows = gpu.sample(5, fake_runner(t))
    assert rc == 0 and rows[0]["compute_pids"] == [None]
    out = gpu.idle_check(1, 0.0, 5, 1, runner=fake_runner(t), sleep=lambda s: None)
    assert out["verdict"] == "RED"
    assert gpu.check_visible(1, 5, runner=fake_runner(t))["status"] == "RED"


def test_idle_check_zero_requested():
    out = gpu.idle_check(3, 0.0, 5, 0, runner=fake_runner({"nvidia-smi": (6, NO_DEV)}),
                         sleep=lambda s: None)
    assert out["verdict"] == "PASS" and out["samples"] == [] and not out["positive_control_ok"]
    out = gpu.idle_check(3, 0.0, 5, 0, runner=fake_runner({"nvidia-smi": nvsmi()}),
                         sleep=lambda s: None)
    assert out["verdict"] == "RED"


# ---- K1: UNREGISTERED never passes; required arguments ---------------------------------------

def _argv(d, sha, extra=()):
    a = id_args("gate") + ["--run-dir", d, "--phase", "pre", "--registered-hw-sha", sha,
                           "--no-comparable",
                           "--slurm-view-file", os.path.join(d, "env", "slurm_view.json"),
                           "--slurm-view-wait-s", "0.2", "--max-alloc-gpus", "1",
                           "--nvidia-smi", "nvidia-smi", "--idle-window-s", "0.2",
                           "--idle-busy-max-frac", "1.0"] + list(extra)
    a[a.index("--job-id") + 1] = JOB
    return a


def test_unregistered_collects_profile_and_exits_env_invalid(jobdir, capsys):
    d, prof = jobdir
    sha = profile_sha256(prof)
    # positive control: the same in-process call with the registered sha passes
    assert gate.main(_argv(d, sha), runner=fake_runner(good_table())) == exitcodes.OK
    capsys.readouterr()
    rc = gate.main(_argv(d, "UNREGISTERED"), runner=fake_runner(good_table()))
    out = json.loads(capsys.readouterr().out)
    assert rc == exitcodes.ENV_INVALID == 6 and out["verdict"] == "ENV_INVALID"
    assert all(v == "PASS" for v in out["items"].values())  # only the registration is missing
    assert out["hw_gate"] == {"status": "NOT_RUN", "node_hw_sha": sha}
    hg = read_json(os.path.join(d, "env", "hw_gate.json"))
    assert hg["passed"] is False and hg["status"] == "NOT_RUN" and hg["node_hw_sha"] == sha
    assert "not registered" in hg["why"]
    # even a comparable file cannot turn it into a pass
    cf = os.path.join(d, "cmp.json")
    with open(cf, "w") as f:
        json.dump({"schema": "capkit.hw_comparable/1", "registered_profile": prof,
                   "comparable_fields": sorted(prof)}, f)
    a = _argv(d, "UNREGISTERED")
    a[a.index("--no-comparable")] = "--comparable-file"
    a.insert(a.index("--comparable-file") + 1, cf)
    assert gate.main(a, runner=fake_runner(good_table())) == 6


def test_unregistered_without_profile_is_env_invalid(jobdir):
    d, _prof = jobdir
    os.remove(os.path.join(d, "env", "hw_profile.json"))
    out = gate.run_gate(Args(d, registered_hw_sha="UNREGISTERED"), _ident(),
                        runner=fake_runner(good_table()))
    assert out["verdict"] == "ENV_INVALID" and out["hw_gate"]["status"] == "NOT_RUN"


@pytest.mark.parametrize("drop,extra,rc", [
    ("--max-alloc-gpus", (), 2),
    ("--nvidia-smi", (), 2),
    ("--idle-window-s", (), 3),
    ("--idle-busy-max-frac", (), 3),
    (None, ("--max-alloc-gpus", "-1"), 3),
    (None, ("--idle-busy-max-frac", "1.5"), 3),
    (None, ("--registered-hw-sha", "unregistered"), 3),
])
def test_gate_required_arguments(jobdir, drop, extra, rc):
    d, prof = jobdir
    a = _argv(d, profile_sha256(prof))
    if drop:
        i = a.index(drop)
        del a[i:i + 2]
    for k, v in zip(extra[::2], extra[1::2]):
        a[a.index(k) + 1] = v
    if rc == 2:
        with pytest.raises(SystemExit) as e:
            gate.main(a, runner=fake_runner(good_table()))
        assert e.value.code == 2
    else:
        assert gate.main(a, runner=fake_runner(good_table())) == rc


def test_gate_post_phase_needs_no_idle_window(jobdir):
    d, prof = jobdir
    a = _argv(d, profile_sha256(prof))
    a[a.index("pre")] = "post"
    for k in ("--idle-window-s", "--idle-busy-max-frac"):
        i = a.index(k)
        del a[i:i + 2]
    assert gate.main(a, runner=fake_runner(good_table())) == 0


def test_hwprofile_cli_requires_nvidia_smi(tmp_path):
    base = [PY, "-m", "capenv.hwprofile"] + id_args("hwprofile") + [
        "--out", str(tmp_path / "p.json"), "--raw-out", str(tmp_path / "r.json"),
        "--local-dir", str(tmp_path), "--chassis-source", "dmi"]
    r = subprocess.run(base, env=capenv_env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "--nvidia-smi" in r.stderr
    smi = tmp_path / "smi"
    smi.write_text("#!/bin/sh\nprintf '%s'\n" % GPU_LINE.replace("\n", "\\n"))
    smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
    r = subprocess.run(base + ["--nvidia-smi", str(smi)], env=capenv_env(path="/usr/bin:/bin"),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    doc = read_json(str(tmp_path / "p.json"))
    assert doc["profile"]["gpu_count"] == 1 and doc["profile"]["sources"]["gpu_count"] == \
        "container:nvidia-smi"


def test_item1_without_tres_is_not_zero_gpus(jobdir):
    d, prof = jobdir
    no_tres = "JobId=4242 OverSubscribe=NO NodeList=node-a Partition=p1"
    out = _gate(d, prof, good_table(scontrol=lambda a: (0, no_tres) if a[2] == "job" else (
        0, "NodeName=node-a CPUAlloc=24 CPUTot=24")))
    assert out["items"]["1"] == "RED" and "TRES" in out["reasons"][0]
