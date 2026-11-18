# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate, hardware profile, memory guard and GPU checks.

Slurm and nvidia-smi are never run for real: in-process tests use a fake runner, the CLI test
puts fake scontrol / squeue / nvidia-smi scripts first on PATH (this VM has none of the real
ones)."""

import copy
import json
import os
import stat
import subprocess
import time

import pytest

from b4_envgate_util import PY, SHA, capenv_env, id_args, kill_all, make_ident, spawn_py
from capcommon import hwprofile as hwprofile_mod
from capcommon.hwprofile import profile_sha256
from capcommon.jsonio import read_json, write_json_stamped
from capenv import gate, gpu, hwprofile, memguard, procfs, runcmd

JOB = "4242"
NODE = "node-a"
JOB_OK = ("JobId=4242 JobName=x UserId=u OverSubscribe=NO NumNodes=1 NumCPUs=24 "
          "NodeList=node-a TRES=cpu=24,mem=60G,node=1,gres/gpu=1 Partition=p1")
NODE_OK = "NodeName=node-a CPUAlloc=24 CPUTot=24 Sockets=1 ThreadsPerCore=1 State=ALLOCATED"
SQ_OK = "4242 u RUNNING\n"
GPU_LINE = "0, GPU-abc, 00000000:1B:00.0, NVIDIA B200, 183359, 0, 0, 570.1\n"


def fake_runner(table):
    """table: {argv0: (rc, stdout) or callable(argv)->(rc, stdout)}; unknown -> not found."""
    calls = []

    def run(argv, timeout_s, env=None):
        calls.append(list(argv))
        ent = table.get(argv[0])
        if ent is None:
            return {"cmd": list(argv), "rc": None, "stdout": None, "stdout_sha256": None,
                    "stderr": None, "reason": "not found: %s" % argv[0]}
        rc, out = ent(argv) if callable(ent) else ent
        return {"cmd": list(argv), "rc": rc, "stdout": out, "stdout_sha256": "0" * 64,
                "stderr": "", "reason": None if rc == 0 else "exit code %d" % rc}
    run.calls = calls
    return run


def nvsmi(apps=""):
    def f(argv):
        if any(a.startswith("--query-compute-apps") for a in argv):
            return 0, apps
        return 0, GPU_LINE
    return f


def good_table(**over):
    t = {"scontrol": lambda a: (0, JOB_OK) if a[2] == "job" else (0, NODE_OK),
         "squeue": (0, SQ_OK), "nvidia-smi": nvsmi()}
    t.update(over)
    return t


class Args:
    def __init__(self, run_dir, **kw):
        self.run_dir = run_dir
        self.phase = "pre"
        self.registered_hw_sha = None
        self.comparable_file = None
        self.slurm_view_file = os.path.join(run_dir, "env", "slurm_view.json")
        self.slurm_view_wait_s = 0.2
        self.max_alloc_gpus = 1
        self.nvidia_smi = "nvidia-smi"
        self.idle_window_s = 0.2
        self.idle_busy_max_frac = 1.0
        self.hw_profile = None
        self.roles_file = None
        self.timeout_s = 5.0
        self.job_id = JOB
        self.__dict__.update(kw)


@pytest.fixture
def jobdir(tmp_path, monkeypatch):
    d = tmp_path / "job"
    (d / "env").mkdir(parents=True)
    monkeypatch.setenv("SLURMD_NODENAME", NODE)
    # memory guard read-back against a synthetic cgroup tree with a finite limit (this VM's real
    # cgroup has none, see test_memguard_real_matches_independent_read)
    sp, cg = tmp_path / "mg_proc", tmp_path / "mg_cg"
    (cg / "memory" / "job").mkdir(parents=True)
    (cg / "memory" / "job" / "memory.limit_in_bytes").write_text("1073741824\n")
    real_check = memguard.check

    def synth_check(pids):
        for pid in pids:
            (sp / str(pid)).mkdir(parents=True, exist_ok=True)
            (sp / str(pid) / "cgroup").write_text("3:memory:/job\n")
        return real_check(pids, str(sp), str(cg))
    monkeypatch.setattr(gate.memguard, "check", synth_check)
    col = hwprofile.Collector(str(tmp_path), "dmi", runner=fake_runner({}))
    prof = col.collect()
    write_json_stamped(str(d / "env" / "hw_profile.json"), "capkit.hw_profile/1",
                       make_ident("hwprofile"), {"profile": prof,
                                                 "profile_sha256": profile_sha256(prof)})
    return str(d), prof


def _ident():
    return make_ident("gate")


def test_gate_pass_with_fake_tools(jobdir):
    d, prof = jobdir
    out = gate.run_gate(Args(d, registered_hw_sha=profile_sha256(prof)), _ident(),
                        runner=fake_runner(good_table()))
    assert out["verdict"] == "PASS", out["reasons"]
    ex = read_json(os.path.join(d, "env", "exclusivity.json"))
    assert [it["id"] for it in ex["items"]] == [1, 2, 3, 4, 5, 6]
    assert ex["positive_control_ok"] is True
    assert all(it["source"] == "container" for it in ex["items"])
    assert read_json(os.path.join(d, "env", "hw_gate.json"))["passed"] is True
    mg = read_json(os.path.join(d, "env", "memguard.json"))
    assert mg["verdict"] == "PASS"  # tests run under systemd-run -p MemoryMax=6G


@pytest.mark.parametrize("table,item,status", [
    (dict(squeue=(0, "4242 u RUNNING\n999 v RUNNING\n")), 3, "RED"),       # another job
    (dict(squeue=(0, "999 v RUNNING\n")), 3, "NOT_RUN"),                    # own job missing
    (dict(squeue=(1, "")), 3, "RED"),                                       # rc != 0
    (dict(scontrol=lambda a: (0, JOB_OK.replace("OverSubscribe=NO", "OverSubscribe=YES"))
          if a[2] == "job" else (0, NODE_OK)), 1, "RED"),
    (dict(scontrol=lambda a: (0, JOB_OK.replace("gres/gpu=1", "gres/gpu=8"))
          if a[2] == "job" else (0, NODE_OK)), 1, "RED"),
    (dict(scontrol=lambda a: (0, JOB_OK) if a[2] == "job"
          else (0, NODE_OK.replace("CPUAlloc=24", "CPUAlloc=8"))), 2, "RED"),
    (dict(**{"nvidia-smi": nvsmi("00000000:1B:00.0, 777, 1024\n")}), 5, "RED"),
    (dict(**{"nvidia-smi": (0, "")}), 5, "NOT_RUN"),                       # lists no GPU
])
def test_gate_items_go_red_or_not_run(jobdir, table, item, status):
    d, prof = jobdir
    out = gate.run_gate(Args(d, registered_hw_sha=profile_sha256(prof)), _ident(),
                        runner=fake_runner(good_table(**table)))
    assert out["verdict"] == "ENV_INVALID"
    assert out["items"][str(item)] == status
    assert all(v == "PASS" for k, v in out["items"].items() if k != str(item))


def test_gate_nvidia_smi_missing_is_not_run(jobdir):
    d, prof = jobdir
    t = good_table()
    del t["nvidia-smi"]
    out = gate.run_gate(Args(d, registered_hw_sha=profile_sha256(prof)), _ident(),
                        runner=fake_runner(t))
    assert out["items"]["5"] == "NOT_RUN" and out["verdict"] == "ENV_INVALID"


def test_gate_slurm_view_fallback(jobdir):
    d, prof = jobdir
    t = good_table()
    del t["scontrol"], t["squeue"]
    sv = os.path.join(d, "env", "slurm_view.json")
    # no file -> wait expires -> items 1-3 fail, gate ENV_INVALID
    out = gate.run_gate(Args(d, registered_hw_sha=profile_sha256(prof)), _ident(),
                        runner=fake_runner(t))
    assert out["verdict"] == "ENV_INVALID" and out["items"]["1"] == "RED"
    items = [{"id": 1, "cmd": ["scontrol", "show", "job", JOB], "rc": 0, "stdout": JOB_OK},
             {"id": 2, "cmd": ["scontrol", "show", "node", NODE], "rc": 0, "stdout": NODE_OK},
             {"id": 3, "cmd": ["squeue"], "rc": 0, "stdout": SQ_OK}]
    with open(sv, "w") as f:
        json.dump({"schema": "capkit.slurm_view/1", "job_id": JOB, "items": items}, f)
    out = gate.run_gate(Args(d, registered_hw_sha=profile_sha256(prof)), _ident(),
                        runner=fake_runner(t))
    assert out["verdict"] == "PASS", out["reasons"]
    ex = read_json(os.path.join(d, "env", "exclusivity.json"))
    assert [it["source"] for it in ex["items"][:3]] == ["slurm_view"] * 3


def test_gate_item6_not_run_without_window(jobdir):
    d, prof = jobdir
    out = gate.run_gate(Args(d, registered_hw_sha=profile_sha256(prof), idle_window_s=None),
                        _ident(), runner=fake_runner(good_table()))
    assert out["items"]["6"] == "NOT_RUN" and out["verdict"] == "ENV_INVALID"


def test_gate_post_phase_skips_item6_and_hwgate(jobdir):
    d, prof = jobdir
    out = gate.run_gate(Args(d, phase="post", registered_hw_sha="e" * 64), _ident(),
                        runner=fake_runner(good_table()))
    assert out["verdict"] == "PASS" and set(out["items"]) == {"1", "2", "3", "4", "5"}
    assert os.path.exists(os.path.join(d, "env", "exclusivity.post.json"))


def test_hw_gate_mismatch_and_comparable(jobdir, tmp_path):
    d, prof = jobdir
    reg = copy.deepcopy(prof)
    reg["kernel"] = "0.0.0-other"
    reg_sha = profile_sha256(reg)
    p = os.path.join(d, "env", "hw_profile.json")
    hg, ok = gate.hw_gate(p, reg_sha, None)
    assert not ok and "no-comparable" in hg["why"]
    cf = str(tmp_path / "comparable.json")
    with open(cf, "w") as f:
        json.dump({"schema": "capkit.hw_comparable/1", "registered_profile": reg,
                   "comparable_fields": ["kernel"]}, f)
    hg, ok = gate.hw_gate(p, reg_sha, cf)
    assert ok and hg["diff_fields"] == ["kernel"]
    with open(cf, "w") as f:
        json.dump({"schema": "capkit.hw_comparable/1", "registered_profile": reg,
                   "comparable_fields": ["nics"]}, f)
    hg, ok = gate.hw_gate(p, reg_sha, cf)
    assert not ok and "kernel" in hg["why"]
    hg, ok = gate.hw_gate(p, "f" * 64, cf)  # registered_profile does not hash to the sha
    assert not ok and "does not hash" in hg["why"]


def _fake_stat(path, rows):
    with open(path, "w") as f:
        f.write("cpu  0 0 0 0 0 0 0 0 0 0\n")
        for c, (busy, idle) in rows.items():
            f.write("cpu%d %d 0 0 %d 0 0 0 0 0 0\n" % (c, busy, idle))


def test_item6_synthetic_red_and_pass(tmp_path):
    proc, sysr = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (sysr / "devices/system/cpu").mkdir(parents=True)
    (sysr / "devices/system/cpu/online").write_text("0-1\n")
    _fake_stat(str(proc / "stat"), {0: (0, 0), 1: (0, 0)})

    def sleep_busy(_s):
        _fake_stat(str(proc / "stat"), {0: (90, 10), 1: (1, 99)})
    it = gate.item6(1.0, 0.5, str(proc), str(sysr), sleep=sleep_busy)
    assert it["status"] == "RED" and it["detail"]["busy_cores"] == [0]
    assert it["detail"]["expected_cores"] == 2 and it["detail"]["checked_cores"] == 2
    _fake_stat(str(proc / "stat"), {0: (0, 0), 1: (0, 0)})
    it = gate.item6(1.0, 0.5, str(proc), str(sysr),
                    sleep=lambda s: _fake_stat(str(proc / "stat"), {0: (1, 99), 1: (2, 98)}))
    assert it["status"] == "PASS"


def _stat_env(tmp_path):
    proc, sysr = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (sysr / "devices/system/cpu").mkdir(parents=True)
    (sysr / "devices/system/cpu/online").write_text("0-1\n")
    return proc, sysr


class _Ticker:
    """A /proc/stat that advances by one row set per window: rows[k] is added during the k-th
    window (calls alternate window, pause, window, ...; a pause adds nothing)."""

    def __init__(self, proc, rows):
        self.proc, self.rows, self.calls, self.acc = proc, rows, [], {0: [0, 0], 1: [0, 0]}
        self.write()

    def write(self):
        _fake_stat(str(self.proc / "stat"), {c: tuple(v) for c, v in self.acc.items()})

    def __call__(self, s):
        self.calls.append(s)
        if len(self.calls) % 2 == 1:  # a window
            for c, (b, i) in self.rows[min(len(self.calls) // 2, len(self.rows) - 1)].items():
                self.acc[c][0] += b
                self.acc[c][1] += i
            self.write()


def test_item6_resample_persistent_busy_stays_red(tmp_path):
    """Positive control: a core busy in every window stays red after all resamples."""
    proc, sysr = _stat_env(tmp_path)
    t = _Ticker(proc, [{0: (50, 50), 1: (1, 99)}])
    it = gate.item6(30, 0.02, str(proc), str(sysr), sleep=t)
    d = it["detail"]
    assert it["status"] == "RED" and d["passed_on_window"] is None
    assert d["windows_read"] == 1 + gate.IDLE_RESAMPLE_MAX == len(d["windows"]) == 4
    assert t.calls == [30] * 7  # 4 windows and the 3 pauses between them
    assert all(w["busy_cores"] == [0] and not w["passed"] for w in d["windows"])
    assert d["busy_cores"] == [0] and d["windows"][0]["busy_frac_of_busy_cores"] == {"0": 0.5}
    assert d["windows"][2]["max_busy_frac"] == 0.5 and d["note"] is None


def test_item6_resample_passes_on_second_window(tmp_path):
    """Transient noise: red in the first window, clean in the second -> pass on window 2."""
    proc, sysr = _stat_env(tmp_path)
    t = _Ticker(proc, [{0: (50, 50), 1: (1, 99)}, {0: (1, 99), 1: (0, 100)}])
    it = gate.item6(30, 0.02, str(proc), str(sysr), sleep=t)
    d = it["detail"]
    assert it["status"] == "PASS" and d["passed_on_window"] == 2 and d["windows_read"] == 2
    assert t.calls == [30, 30, 30]
    assert [w["passed"] for w in d["windows"]] == [False, True]
    assert d["windows"][0]["busy_cores"] == [0] and d["busy_cores"] == []
    assert d["per_core"]["0"]["busy_frac"] == 0.01 and "window 2 of 2" in d["note"]


def test_item6_clean_first_window_does_not_resample(tmp_path):
    proc, sysr = _stat_env(tmp_path)
    t = _Ticker(proc, [{0: (1, 99), 1: (2, 98)}, {0: (99, 1), 1: (99, 1)}])
    it = gate.item6(30, 0.02, str(proc), str(sysr), sleep=t)
    d = it["detail"]
    assert it["status"] == "PASS" and t.calls == [30]
    assert d["passed_on_window"] == 1 and d["windows_read"] == 1 and d["note"] is None


def test_item6_resample_max_zero_is_single_window(tmp_path):
    proc, sysr = _stat_env(tmp_path)
    t = _Ticker(proc, [{0: (50, 50), 1: (1, 99)}, {0: (0, 100), 1: (0, 100)}])
    it = gate.item6(30, 0.02, str(proc), str(sysr), sleep=t, resample_max=0)
    assert it["status"] == "RED" and t.calls == [30]


def test_item6_records_irq_and_softirq_per_core(tmp_path):
    """R1b (D14): the idle window keeps each core's irq and softirq ticks for capjudge.pilot."""
    proc, sysr = tmp_path / "proc", tmp_path / "sys"
    proc.mkdir()
    (sysr / "devices/system/cpu").mkdir(parents=True)
    (sysr / "devices/system/cpu/online").write_text("0-1\n")

    def stat(rows):  # cpuN user nice system idle iowait irq softirq steal guest guest_nice
        with open(str(proc / "stat"), "w") as f:
            f.write("cpu  0 0 0 0 0 0 0 0 0 0\n")
            for c, (idle, irq, sirq) in rows.items():
                f.write("cpu%d 0 0 0 %d 0 %d %d 0 0 0\n" % (c, idle, irq, sirq))
    stat({0: (0, 0, 0), 1: (0, 0, 0)})
    it = gate.item6(1.0, 0.5, str(proc), str(sysr),
                    sleep=lambda s: stat({0: (97, 1, 2), 1: (100, 0, 0)}))
    per = it["detail"]["per_core"]
    assert per["0"]["softirq_ticks"] == 2 and per["0"]["irq_ticks"] == 1
    assert per["0"]["total_ticks"] == 100 and per["1"]["softirq_ticks"] == 0


def test_item6_real_spinner_positive_control():
    cpus = sorted(os.sched_getaffinity(0))
    target = cpus[-1]
    sp = spawn_py("while True:\n    pass\n", [target])
    try:
        time.sleep(0.2)
        it = gate.item6(0.6, 0.5)
    finally:
        kill_all([sp])
    assert target in it["detail"]["busy_cores"]
    assert it["detail"]["per_core"][str(target)]["busy_frac"] > 0.8


def test_memguard_real_matches_independent_read():
    """The tool's reading of this process's memory limit equals an independent read of the
    same cgroup file (no claim about whether a guard exists on this VM)."""
    rec = memguard.read_pid(os.getpid())
    assert rec["version"] in (1, 2) and rec["levels"], rec
    leaf = rec["levels"][0]["file"]
    raw = subprocess.run(["cat", leaf], capture_output=True, text=True, check=True).stdout.strip()
    finite = raw != "max" and int(raw) < (1 << 62)
    assert (rec["levels"][0]["bytes"] is not None) == finite
    if not any(lv["bytes"] is not None for lv in rec["levels"]):
        assert rec["effective_bytes"] is None and "no finite limit" in rec["reason"]


def test_memguard_synthetic(tmp_path):
    # synthetic v2 tree with no finite limit -> RED; no pid -> NOT_RUN
    proc = tmp_path / "proc" / "77"
    proc.mkdir(parents=True)
    (proc / "cgroup").write_text("0::/job/step\n")
    cg = tmp_path / "cg"
    (cg / "job" / "step").mkdir(parents=True)
    (cg / "job" / "step" / "memory.max").write_text("max\n")
    (cg / "job" / "memory.max").write_text("max\n")
    res = memguard.check([77], str(tmp_path / "proc"), str(cg))
    assert res["verdict"] == "RED"
    (cg / "job" / "memory.max").write_text("1073741824\n")
    res = memguard.check([77], str(tmp_path / "proc"), str(cg))
    assert res["verdict"] == "PASS" and res["pids"][0]["effective_bytes"] == 1 << 30
    assert memguard.check([], str(tmp_path / "proc"), str(cg))["verdict"] == "NOT_RUN"


def test_gpu_idle_check_paths():
    out = gpu.idle_check(2, 0.0, 5, 1, runner=fake_runner({}), sleep=lambda s: None)
    assert out["verdict"] == "NOT_RUN" and out["positive_control_ok"] is False
    out = gpu.idle_check(2, 0.0, 5, 1, runner=fake_runner({"nvidia-smi": nvsmi()}),
                         sleep=lambda s: None)
    assert out["verdict"] == "PASS" and out["gpu_count"] == 1 and out["checked_samples"] == 2
    busy = GPU_LINE.replace(", 0, 0, 570.1", ", 10, 30, 570.1")

    def busy_smi(argv):
        return (0, "") if any(a.startswith("--query-compute-apps") for a in argv) else (0, busy)
    out = gpu.idle_check(1, 0.0, 5, 1, runner=fake_runner({"nvidia-smi": busy_smi}),
                         sleep=lambda s: None)
    assert out["verdict"] == "RED"
    out = gpu.idle_check(1, 0.0, 5, 1, runner=fake_runner(
        {"nvidia-smi": nvsmi("0000:1B:00.0, 31337, 2\n")}), sleep=lambda s: None)
    assert out["verdict"] == "RED" and out["samples"][0]["gpus"][0]["compute_pids"] == [31337]
    # nvidia-smi really absent on this VM -> NOT_RUN, never PASS
    assert gpu.idle_check(1, 0.0, 5, 1)["verdict"] == "NOT_RUN"


def test_hwprofile_missing_commands_and_sha_stability(tmp_path):
    a = hwprofile.Collector(str(tmp_path), "dmi", runner=fake_runner({})).collect()
    for k in ("cpu_arch", "cpu_model", "sockets", "gpu_count"):
        assert "unreadable" in a[k]
    time.sleep(0.01)
    b = hwprofile.Collector(str(tmp_path), "dmi", runner=fake_runner({})).collect()
    assert a["collected_mono_ns"] != b["collected_mono_ns"]
    assert profile_sha256(a) == profile_sha256(b)  # observed fields never hashed
    real = hwprofile.Collector(str(tmp_path), "dmi").collect()
    assert isinstance(real["sockets"], int) and real["kernel"] == os.uname().release
    assert profile_sha256(real) != profile_sha256(a)  # unreadable markers are part of the sha
    sv = tmp_path / "sv.json"
    sv.write_text(json.dumps({"schema": "capkit.slurm_view/1", "items": [
        {"id": 1, "rc": 0, "stdout": "JobId=1 Partition=b200x8"},
        {"id": 2, "rc": 0, "stdout": "NodeName=n SystemProductName=DGXB200 CPUTot=224"}]}))
    c = hwprofile.Collector(str(tmp_path), "slurm_view", slurm_view_file=str(sv),
                            runner=fake_runner({})).collect()
    assert c["chassis_model"] == "DGXB200" and c["partition"] == "b200x8"


def _fake_bin(d):
    d.mkdir()
    scripts = {
        "scontrol": "#!/bin/sh\nif [ \"$2\" = job ]; then echo '%s'; else echo '%s'; fi\n"
                    % (JOB_OK, NODE_OK),
        "squeue": "#!/bin/sh\nprintf '%s'\n[ -n \"$EXTRA_SQ\" ] && cat \"$EXTRA_SQ\"\nexit 0\n"
                  % SQ_OK.replace("\n", "\\n"),
        "nvidia-smi": "#!/bin/sh\ncase \"$1\" in --query-compute-apps*) exit 0;; esac\n"
                      "printf '%s'\n" % GPU_LINE.replace("\n", "\\n"),
    }
    for n, body in scripts.items():
        p = d / n
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)


def test_gate_cli_with_fake_bin(jobdir, tmp_path):
    d, prof = jobdir
    fb = tmp_path / "fakebin"
    _fake_bin(fb)
    env = capenv_env({"SLURMD_NODENAME": NODE}, path="%s:/usr/bin:/bin" % fb)
    cmd = [PY, "-m", "capenv.gate"] + id_args("gate") + [
        "--run-dir", d, "--phase", "pre",
        "--registered-hw-sha", profile_sha256(prof), "--no-comparable",
        "--slurm-view-file", os.path.join(d, "env", "slurm_view.json"),
        "--slurm-view-wait-s", "0.2", "--max-alloc-gpus", "1", "--idle-window-s", "0.2",
        "--idle-busy-max-frac", "1.0", "--nvidia-smi", str(fb / "nvidia-smi")]
    cmd[cmd.index("--job-id") + 1] = JOB  # the gate checks this job id against scontrol
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    out = json.loads(r.stdout)
    # items 1-6 pass with the fake tools; the verdict then depends only on this VM's real
    # memory guard read-back (no finite limit here -> ENV_INVALID with exactly that reason)
    assert all(v == "PASS" for v in out["items"].values()), out
    mg = read_json(os.path.join(d, "env", "memguard.json"))
    if mg["verdict"] == "PASS":
        assert r.returncode == 0 and out["verdict"] == "PASS"
    else:
        assert r.returncode == 6 and len(out["reasons"]) == 1
        assert out["reasons"][0].startswith("memguard")
    extra = tmp_path / "extra.txt"
    extra.write_text("5555 other RUNNING\n")
    env["EXTRA_SQ"] = str(extra)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 6 and json.loads(r.stdout)["items"]["3"] == "RED"
    assert runcmd.ok({"rc": 0})


def _pci(root, addr, cls, ven, dev, vf=False):
    d = os.path.join(root, "bus", "pci", "devices", addr)
    os.makedirs(d)
    for name, text in (("class", cls), ("vendor", ven), ("device", dev)):
        with open(os.path.join(d, name), "w") as f:
            f.write(text + "\n")
    if vf:
        os.symlink("../0000:01:00.0", os.path.join(d, "physfn"))


def _netdev(root, name, ven, dev, speed):
    d = os.path.join(root, "class", "net", name)
    os.makedirs(os.path.join(d, "device"))
    for name_, text in (("device/vendor", ven), ("device/device", dev), ("speed", speed),
                        ("operstate", "up")):
        with open(os.path.join(d, name_), "w") as f:
            f.write(text + "\n")


def test_nic_identity_is_the_pci_device_list_not_the_netdevs(tmp_path):
    """SM-12: two nodes with the same network controllers but different network interfaces
    (drivers, links, speeds) get the same identity "nics"; netdevs are observed only."""
    profs = []
    for node, netdevs in (("a", [("ib0", "50000"), ("eth0", "-1")]), ("b", [("eth0", "100")])):
        root = str(tmp_path / node)
        _pci(root, "0000:01:00.0", "0x020700", "0x15b3", "0x1021")
        _pci(root, "0000:02:00.0", "0x020700", "0x15b3", "0x1021")
        _pci(root, "0000:03:00.0", "0x030200", "0x10de", "0x2901")  # a GPU: not a NIC
        _pci(root, "0000:01:00.2", "0x020000", "0x15b3", "0x101e", vf=True)  # SR-IOV VF
        for name, speed in netdevs:
            _netdev(root, name, "0x15b3", "0x1021", speed)
        col = hwprofile.Collector(str(tmp_path), "dmi", runner=fake_runner({}), sys_root=root)
        col.nics()
        profs.append(dict(col.prof))
    assert profs[0]["nics"] == profs[1]["nics"] == [{"model": "0x15b3:0x1021", "count": 2}]
    assert profs[0]["netdevs"] != profs[1]["netdevs"]
    assert "nics" in hwprofile_mod.IDENTITY_FIELDS and "netdevs" in hwprofile_mod.OBSERVED_FIELDS
    col = hwprofile.Collector(str(tmp_path), "dmi", runner=fake_runner({}),
                              sys_root=str(tmp_path / "none"))
    col.nics()
    assert "unreadable" in col.prof["nics"]


@pytest.mark.parametrize("missing", ["vendor", "device", "vendor-empty"])
def test_nic_with_unreadable_vendor_or_device_is_unreadable_identity(tmp_path, missing):
    """U9: a network controller whose vendor or device id cannot be read makes the identity field
    "nics" unreadable (listed in unreadable_identity_fields), never a '?' model."""
    root = str(tmp_path / "sys")
    _pci(root, "0000:01:00.0", "0x020700", "0x15b3", "0x1021")
    _pci(root, "0000:02:00.0", "0x020700", "0x15b3", "0x1021")
    f = os.path.join(root, "bus", "pci", "devices", "0000:02:00.0", missing.split("-")[0])
    if missing.endswith("-empty"):
        open(f, "w").close()
    else:
        os.remove(f)
    col = hwprofile.Collector(str(tmp_path), "dmi", runner=fake_runner({}), sys_root=root)
    col.nics()
    assert "unreadable" in col.prof["nics"], col.prof["nics"]
    assert "0000:02:00.0/%s" % missing.split("-")[0] in str(col.prof["nics"])
