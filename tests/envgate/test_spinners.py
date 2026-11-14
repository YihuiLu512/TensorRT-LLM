# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capenv.spinners (INTERFACES errata SB-B1, design test #26): pin table reading, the target
core, the spinner plan, and the CLI against real forked spinners on this machine. Functional
only: no assertion is about how fast or how busy a core is."""

import hashlib
import json
import os
import signal
import subprocess
import time

import pytest

from b4_envgate_util import PY, capenv_env, id_args, wait_for
from capcommon.exitcodes import ConfigError
from capcommon.jsonio import read_json_stamped
from capcommon.lifecycle import READY_SCHEMA
from capcommon.runtree import RunTree
from capenv import procfs, spinners

PIN = {"schema": "capkit.pin_table/1", "source_sha": None, "rules": "test",
       "siblings_left_idle": [], "unassigned": []}


def _pin(roles):
    d = dict(PIN)
    d["roles"] = roles
    return d


# ---- pure functions --------------------------------------------------------------------------

def test_target_core_is_the_first_listed_cpu_of_the_role():
    pin = _pin({"relay1": {"cpus": [7, 3], "numa_node": 0}, "beat": {"cpus": [2]}})
    assert spinners.target_core(pin, "relay1") == 7
    assert spinners.target_core(pin, "beat") == 2


@pytest.mark.parametrize("roles,needle", [
    ({"beat": {"cpus": [2]}}, "not a role of the pin table"),
    ({"relay1": {"cpus": []}}, "no list of cpus"),
    ({"relay1": {"cpus": "3"}}, "no list of cpus"),
    ({"relay1": {"cpus": [True]}}, "no list of cpus"),
    ({"relay1": {"cpus": [-1]}}, "no list of cpus"),
    ({"relay1": {}}, "no list of cpus"),
    ({"relay1": [3]}, "not a role of the pin table"),
])
def test_target_core_rejects_absent_or_cpu_less_roles(roles, needle):
    with pytest.raises(ConfigError, match=needle):
        spinners.target_core(_pin(roles), "relay1")


@pytest.mark.parametrize("doc", [
    [], {"schema": "capkit.pin_table/2", "roles": {}}, {"schema": "capkit.pin_table/1"},
    {"schema": "capkit.pin_table/1", "roles": []},
])
def test_check_pin_table_rejects_non_pin_tables(doc):
    with pytest.raises(ConfigError, match="is not a capkit.pin_table/1"):
        spinners.check_pin_table(doc, "x.json")
    good = _pin({"a": {"cpus": [1]}})
    assert spinners.check_pin_table(good, "x.json") is good


def test_plan_own_cores_then_target_and_target_inside_cpus_refused():
    assert spinners.plan_spinners([4, 5]) == [
        {"index": 0, "cpu": 4, "purpose": "own"}, {"index": 1, "cpu": 5, "purpose": "own"}]
    plan = spinners.plan_spinners([4, 5], "relay1", 9)
    assert [(p["cpu"], p["purpose"]) for p in plan] == [(4, "own"), (5, "own"), (9, "target")]
    assert plan[-1]["target_role"] == "relay1"
    with pytest.raises(ConfigError, match="is one of --cpus"):
        spinners.plan_spinners([4, 5], "relay1", 5)


def test_core_deltas_busy_fraction_and_unreadable_cpu():
    z = dict.fromkeys(procfs.STAT_CPU_COLS, 0)
    t0 = {1: dict(z), 2: dict(z), 3: None}
    t1 = {1: dict(z, user=30, system=10, idle=60), 2: dict(z), 3: dict(z, user=5)}
    d = spinners.core_deltas(t0, t1)
    assert d["1"]["busy_ticks"] == 40 and d["1"]["total_ticks"] == 100
    assert d["1"]["busy_frac"] == pytest.approx(0.4)
    assert d["2"]["busy_frac"] is None and d["2"]["total_ticks"] == 0
    assert d["3"] == {"unreadable": "cpu3 not in /proc/stat"}
    assert spinners.output_path("/T", "spin") == "/T/raw/spinners/spin.json"


# ---- the CLI with real spinners --------------------------------------------------------------

def _cpus(n):
    avail = sorted(os.sched_getaffinity(0))
    if len(avail) < n:
        pytest.skip("needs %d cpus, have %d" % (n, len(avail)))
    return avail[:n]


def _allowed(pid):
    with open("/proc/%d/status" % pid) as f:
        for line in f:
            if line.startswith("Cpus_allowed_list:"):
                return procfs.parse_cpu_list_text(line.split(":", 1)[1])
    return None


def _alive(pid):
    """True while pid exists and is not a zombie."""
    try:
        return procfs.read_stat("/proc/%d/stat" % pid)["state"] != "Z"
    except (procfs.ProcGone, OSError):
        return False


class Run:
    """One capenv.spinners process on a fresh launch tree under tmp_path."""

    def __init__(self, tmp_path, own, pin_roles, target=None, pin_doc=None, pin_arg=None):
        self.tree = RunTree(str(tmp_path / "T"))
        self.tree.make_skeleton()
        self.pin_path = self.tree.env("pin_table.json")
        with open(self.pin_path, "w") as f:
            json.dump(pin_doc if pin_doc is not None else _pin(pin_roles), f)
        self.ready_file = self.tree.ready_file("spin")
        argv = [PY, "-B", "-m", "capenv.spinners"] + id_args("spin") + [
            "--run-dir", self.tree.root, "--cpus", ",".join(map(str, own)),
            "--stop-grace-s", "2", "--origin-ns", str(time.monotonic_ns()),
            "--ready-file", self.ready_file, "--pin-table", pin_arg or self.pin_path]
        if target is not None:
            argv += ["--target-role", target]
        self.err_path = str(tmp_path / "spin.err")
        with open(self.err_path, "w") as err:
            self.p = subprocess.Popen(argv, env=capenv_env(), stdout=subprocess.DEVNULL,
                                      stderr=err)

    def ready(self, timeout=20):
        assert wait_for(lambda: os.path.exists(self.ready_file) or self.p.poll() is not None,
                        timeout), "no ready file"
        assert os.path.exists(self.ready_file), self.stderr()
        return read_json_stamped(self.ready_file, READY_SCHEMA)[1]

    def wait(self, timeout=30):
        try:
            return self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
            raise

    def stderr(self):
        with open(self.err_path) as f:
            return f.read()

    def output(self):
        return read_json_stamped(spinners.output_path(self.tree.root, "spin"),
                                 spinners.SCHEMA)[1]


def test_target_role_binds_one_spinner_to_the_roles_first_core(tmp_path):
    """#26: --target-role reads the launch's pin table and binds one spinner to the target
    role's first core; one spinner per own core; the pin table is only read; SIGTERM stops all
    of them and every spinner reports."""
    a, b, c, d = _cpus(4)
    r = Run(tmp_path, [a, b], {"spin": {"cpus": [a, b]}, "relay1": {"cpus": [c, d]}},
            target="relay1")
    pin_before = open(r.pin_path, "rb").read()
    try:
        ready = r.ready()
        assert ready["children"] == [] and ready["affinity"] == [a, b]
        assert ready["versions"]["python"] and ready["info"]["threads"] == []
        info = ready["info"]
        assert info["target_role"] == "relay1" and info["target_cpu"] == c
        sp = info["spinners"]
        assert [(s["cpu"], s["purpose"]) for s in sp] == [(a, "own"), (b, "own"), (c, "target")]
        assert [_allowed(s["pid"]) for s in sp] == [[a], [b], [c]]
        assert all(_allowed(s["pid"]) != [d] for s in sp)
        r.p.send_signal(signal.SIGTERM)
        assert r.wait() == 0, r.stderr()
    finally:
        if r.p.poll() is None:
            r.p.kill()
    assert not [s["pid"] for s in sp if _alive(s["pid"])]
    out = r.output()
    assert out["problems"] == [] and out["ready"] is True
    assert out["target_role"] == "relay1" and out["target_cpu"] == c
    assert out["pin_table"] == {"path": r.pin_path,
                                "sha256": hashlib.sha256(pin_before).hexdigest()}
    assert open(r.pin_path, "rb").read() == pin_before
    rows = out["spinners"]
    assert [(x["cpu"], x["purpose"], x["affinity"], x["pinned_ok"], x["reported"])
            for x in rows] == [(a, "own", [a], True, True), (b, "own", [b], True, True),
                               (c, "target", [c], True, True)]
    assert all(x["units"] > 0 and x["wall_ns"] > 0 and x["exit"] == {"code": 0}
               and x["killed"] is False and x["orphaned"] is False for x in rows)
    assert rows[2]["last_cpu"] == c
    assert sorted(out["cores"]) == sorted(str(x) for x in (a, b, c))
    assert out["stop"] is None and out["signals"] == [signal.SIGTERM]
    assert out["identity"]["role"] == "spin" and out["identity"]["pid"] == r.p.pid


def test_without_target_role_only_own_cores_spin(tmp_path):
    a, b, c = _cpus(3)
    r = Run(tmp_path, [a, b], {"spin": {"cpus": [a, b]}, "relay1": {"cpus": [c]}})
    try:
        sp = r.ready()["info"]["spinners"]
        assert [(s["cpu"], s["purpose"]) for s in sp] == [(a, "own"), (b, "own")]
        r.p.send_signal(signal.SIGTERM)
        assert r.wait() == 0, r.stderr()
    finally:
        if r.p.poll() is None:
            r.p.kill()
    out = r.output()
    assert out["target_role"] is None and out["target_cpu"] is None
    assert [x["cpu"] for x in out["spinners"]] == [a, b] and out["problems"] == []


@pytest.mark.parametrize("case,needle", [
    ("absent", "is not a role of the pin table"),
    ("inside", "is one of --cpus"),
    ("schema", "is not a capkit.pin_table/1"),
    ("missing", "--pin-table"),
])
def test_config_errors_exit_3_before_any_spinner(tmp_path, case, needle):
    a, b = _cpus(2)
    roles = {"spin": {"cpus": [a]}, "relay1": {"cpus": [b if case != "inside" else a]}}
    doc = {"schema": "capkit.pin_table/0", "roles": roles} if case == "schema" else None
    r = Run(tmp_path, [a], roles, target="nobody" if case == "absent" else "relay1",
            pin_doc=doc, pin_arg=str(tmp_path / "no-such.json") if case == "missing" else None)
    assert r.wait() == 3, r.stderr()
    assert needle in r.stderr()
    assert not os.path.exists(r.ready_file)
    assert not os.path.exists(spinners.output_path(r.tree.root, "spin"))


def test_unpinnable_target_core_exits_4_without_ready_file(tmp_path):
    """A target core the kernel refuses (not online here): the spinner reports the failure, the
    supervisor writes no ready file, reaps every spinner and exits 4 with the reason recorded."""
    (a,) = _cpus(1)
    r = Run(tmp_path, [a], {"spin": {"cpus": [a]}, "relay1": {"cpus": [4095]}},
            target="relay1")
    assert r.wait() == 4, r.stderr()
    assert not os.path.exists(r.ready_file)
    out = r.output()
    assert out["ready"] is False and out["cores"] is None
    assert len(out["problems"]) == 1 and "spinner 1 (target, cpu 4095): pinning failed" \
        in out["problems"][0]
    rows = out["spinners"]
    assert rows[1]["pinned_ok"] is False and rows[1]["exit"] == {"code": 4}
    assert rows[0]["pinned_ok"] is True and rows[0]["exit"]["code"] in (0, -signal.SIGKILL)
    assert not [x["pid"] for x in rows if _alive(x["pid"])]


def test_spinner_that_dies_before_the_stop_fails_the_tool(tmp_path):
    a, b = _cpus(2)
    r = Run(tmp_path, [a, b], {"spin": {"cpus": [a, b]}})
    try:
        sp = r.ready()["info"]["spinners"]
        os.kill(sp[1]["pid"], signal.SIGKILL)
        assert r.wait() == 4, r.stderr()
    finally:
        if r.p.poll() is None:
            r.p.kill()
    out = r.output()
    assert any("spinner 1 (own, cpu %d) exited before the stop" % b in x
               for x in out["problems"]), out["problems"]
    assert out["spinners"][1]["exit"] == {"code": -signal.SIGKILL}
    assert out["spinners"][0]["reported"] is True
    assert not [s["pid"] for s in sp if _alive(s["pid"])]


def test_spinners_leave_when_the_supervisor_is_killed(tmp_path):
    """SIGKILL of the supervisor (no chance to stop them): every spinner notices its parent is
    gone and exits on its own."""
    a, b = _cpus(2)
    r = Run(tmp_path, [a, b], {"spin": {"cpus": [a, b]}})
    try:
        sp = r.ready()["info"]["spinners"]
        assert all(_alive(s["pid"]) for s in sp)
    finally:
        r.p.kill()
        r.p.wait()
    try:
        assert wait_for(lambda: not [s["pid"] for s in sp if _alive(s["pid"])], 10)
    finally:  # never leave a spinner behind, whatever the outcome
        for s in sp:
            if _alive(s["pid"]):
                os.kill(s["pid"], signal.SIGKILL)


class _Stop:
    def __init__(self, hard):
        self.requested, self.hard = True, hard


def _deaf_children(n):
    """n children that ignore SIGTERM and hold the write end of a pipe, as plan entries."""
    plan = []
    for i in range(n):
        rfd, wfd = os.pipe()
        p = subprocess.Popen([PY, "-B", "-c", "import signal, time\n"
                              "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                              "print('up', flush=True)\ntime.sleep(60)\n"],
                             stdout=wfd, pass_fds=(wfd,))
        os.close(wfd)
        plan.append({"index": i, "cpu": i, "purpose": "own", "pid": p.pid, "rfd": rfd,
                     "buf": b"", "events": [], "eof": False, "exit": None, "killed": False})
    for sp in plan:  # SIGTERM ignored from here on
        assert wait_for(lambda: spinners._read_some(sp, 0.1) and sp["events"], 20)
        sp["events"] = []
    return plan


@pytest.mark.parametrize("hard,why,problem", [(False, "grace_expired", True),
                                              (True, "second_signal", False)])
def test_stop_kills_after_grace_or_at_once_on_a_second_signal(hard, why, problem):
    """Spinners that ignore SIGTERM: SIGKILL after the grace (a problem: no result), or at once
    after a second stop signal (no problem: INTERFACES 2.3 hard stop). All are reaped."""
    plan = _deaf_children(2)
    t0 = time.monotonic()
    spinners.stop_spinners(plan, _Stop(hard), 0.5)
    took = time.monotonic() - t0
    assert (took >= 0.5) if not hard else (took < 0.5)
    assert [(sp["killed"], sp["exit"], sp["eof"]) for sp in plan] == \
        [(why, {"code": -signal.SIGKILL}, True)] * 2
    probs = [spinners.spinner_record(sp, True)[1] for sp in plan]
    assert all(bool(pr) is problem for pr in probs), probs
    assert not [sp["pid"] for sp in plan if _alive(sp["pid"])]
