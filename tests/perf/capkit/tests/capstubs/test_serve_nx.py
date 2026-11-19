# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""serve_nx and nx_driver parts that need no tensorrt_llm: argv split, config resolution, port
picking and the address reporter, refusals; nx_driver's batch parsing, lane accounting, the raw /
deserialize drain on a fake ZMQ queue, the open-loop sender. The product paths (patch, proxy,
worker) run only in the container (01_code/_local/wt-snx/_progress/snx.md: local chroot runs
and the cluster smoke)."""

import json
import os
import pickle
import socket
import threading
import time
import types

import pytest

from capcommon.exitcodes import ConfigError

from capstubs import nx_driver, serve_nx
from capstubs.null_executor import ENV_VAR, SnxConfig


def test_split_argv_and_flag_value():
    assert serve_nx.split_argv(["--a", "x"]) is None
    assert serve_nx.split_argv(["--x", "--", "m", "--port", "5", "--"]) == \
        (["--x"], ["m", "--port", "5", "--"])
    assert serve_nx.flag_value(["m", "--port", "5"], "--port") == "5"
    assert serve_nx.flag_value(["m", "--port=7"], "--port") == "7"
    assert serve_nx.flag_value(["m"], "--port") is None


def test_resolve_config_layers(tmp_path):
    base = tmp_path / "c.json"
    base.write_text(json.dumps({"beat_ns": 5000000, "tokens_per_step": 2}))
    a = serve_nx.build_parser().parse_args(
        ["--config-json", str(base), "--set", "tokens_per_step=3", "--set",
         "token_source=prompt", "--out-dir", "/x"])
    cfg = serve_nx.resolve_config(a)
    assert (cfg.beat_ns, cfg.tokens_per_step, cfg.token_source, cfg.out_dir) == \
        (5000000, 3, "prompt", "/x")
    with pytest.raises(ConfigError):
        serve_nx.resolve_config(serve_nx.build_parser().parse_args(["--set", "nokey"]))
    with pytest.raises(ConfigError):
        serve_nx.resolve_config(serve_nx.build_parser().parse_args(["--set", "bogus=1"]))


def test_main_refusals(monkeypatch, capsys):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert serve_nx.main(["model"]) == 2
    assert serve_nx.main(["--set", "beat_ns=0", "--", "model"]) == 3
    assert serve_nx.main(["--pick-port", "--", "model", "--port", "1"]) == 3
    assert "config refused" in capsys.readouterr().err


def test_reporter_writes_once_port_accepts(tmp_path):
    port = serve_nx.pick_free_port("127.0.0.1")
    out = str(tmp_path / "addr")
    t = serve_nx.start_reporter(out, "127.0.0.1", port, 10.0)
    time.sleep(0.3)
    assert not os.path.exists(out)  # nothing listening yet
    srv = socket.socket()
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    try:
        t.join(5)
        with open(out) as f:
            assert f.read() == "127.0.0.1:%d\n" % port
    finally:
        srv.close()


def test_parse_batch():
    ok = types.SimpleNamespace(error_msg=None, result=types.SimpleNamespace(is_final=False))
    fin = types.SimpleNamespace(error_msg=None, result=types.SimpleNamespace(is_final=True))
    err = types.SimpleNamespace(error_msg="x", result=None)
    assert nx_driver.parse_batch(None) is None
    assert nx_driver.parse_batch([ok, fin, err, None]) == (3, 2, 1)
    assert nx_driver.parse_batch(fin) == (1, 1, 0)


class FakePlan:
    def __init__(self, origin, end):
        self.origin, self.end_abs = origin, end
        self.rows = []
        self.tt = types.SimpleNamespace(sha256="t" * 64)
        self.max_tokens = {0: 4}
        self.prompt = [1, 2]

    def tier_seq_at(self, t):
        return 0 if self.origin <= t < self.end_abs else (-1 if t < self.origin else -2)


class FakeSock:
    def __init__(self, frames):
        self.frames = list(frames)

    def poll(self, _ms):
        return bool(self.frames)

    def recv(self):
        return self.frames.pop(0)


class FakeZq:
    def __init__(self, frames, hmac):
        self.socket = FakeSock(frames)
        self.use_hmac_encryption = hmac

    def setup_lazily(self):
        pass

    def _parse_data(self, data):
        return pickle.loads(data[:-32])


def _frames(hmac):
    fin = types.SimpleNamespace(error_msg=None, result=types.SimpleNamespace(is_final=True))
    pad = b"h" * 32 if hmac else b""
    batch = pickle.dumps([fin, fin]) + pad
    return [batch, batch, pickle.dumps(None) + pad]


@pytest.mark.parametrize("consumer,hmac", [("raw", True), ("raw", False),
                                           ("deserialize", True), ("deserialize", False)])
def test_drain_loop_counts(tmp_path, consumer, hmac):
    now = time.monotonic_ns()
    plan = FakePlan(now - 10 ** 9, now + 10 ** 10)
    st = nx_driver.LaneStats(plan, str(tmp_path), "lane0", consumer, 0)
    nx_driver.drain_loop(FakeZq(_frames(hmac), hmac), st, consumer, 0, threading.Event())
    t = st.tiers[0]
    assert st.sentinel and t["msgs"] == 2 and t["bytes"] > 0
    if consumer == "deserialize":
        assert (t["parsed_msgs"], t["responses"], t["finals"]) == (2, 4, 4)
    else:
        assert t["parsed_msgs"] == 0
    doc = st.summary({"mode": "test"})
    assert doc["tiers"]["0"]["msgs"] == 2 and doc["mode"] == "test"
    rec = open(os.path.join(str(tmp_path), "nxd.lane0.recvs.rec"), "rb").read()
    assert len(rec) == 2 * nx_driver.RECV_REC.size


def test_send_loop_open_loop(tmp_path, monkeypatch):
    now = time.monotonic_ns()
    plan = FakePlan(now, now + 10 ** 10)
    rows = [(0, now - 10 ** 8, 0), (1, now + 30 * 10 ** 6, 0), (2, now + 60 * 10 ** 6, 0)]
    st = nx_driver.LaneStats(plan, str(tmp_path), "lane0", "raw", 0)
    st.planned(rows)
    monkeypatch.setattr(nx_driver, "make_request", lambda p, i, s, e: ("req", i))
    sent = []
    nx_driver.send_loop(plan, rows, sent.append, st, threading.Event(), 2)
    assert [r[1] for r in sent] == [0, 1, 2]
    t = st.tiers[0]
    assert t["planned"] == 3 and t["sent"] == 3 and t["send_lag_ns_max"] >= 10 ** 8
    st.sends.flush()
    assert os.path.getsize(os.path.join(str(tmp_path), "nxd.lane0.sends.rec")) == \
        3 * nx_driver.SEND_REC.size


def test_driver_snx_config_forces_table_and_origin(tmp_path):
    a = nx_driver.build_parser().parse_args(
        ["standalone", "--tier-table", "/t.json", "--schedule", "/s", "--origin-ns", "123",
         "--out-dir", str(tmp_path), "--set", "beat_ns=3000000", "--snx-out-dir", "/o"])
    cfg = nx_driver.snx_config(a)
    assert isinstance(cfg, SnxConfig)
    assert (cfg.tier_table, cfg.origin_ns, cfg.beat_ns, cfg.out_dir) == \
        ("/t.json", 123, 3000000, "/o")


def test_stream_interval_yaml(tmp_path):
    args, why = serve_nx.with_stream_interval(["m"], 20, str(tmp_path))
    assert why is None and args[:2] == ["m", "--config"]
    assert open(args[2]).read() == "stream_interval: 20\n"
    assert serve_nx.with_stream_interval(["m", "--config", "x"], 20, str(tmp_path))[1]
    assert serve_nx.with_stream_interval(["m"], 20, "")[1]
    assert serve_nx.with_stream_interval(["m"], 0, str(tmp_path))[1]


def test_scrub_launcher_env():
    env = {"SLURM_JOB_ID": "1", "SLURMD_NODENAME": "n", "PMIX_RANK": "0", "PMI_FD": "3",
           "OMPI_MCA_x": "y", "OPAL_PREFIX": "/o", "TRTLLM_CAPKIT_SNX": "{}"}
    assert serve_nx.scrub_launcher_env(env) == ["PMIX_RANK", "PMI_FD", "SLURMD_NODENAME",
                                                "SLURM_JOB_ID"]
    assert set(env) == {"OMPI_MCA_x", "OPAL_PREFIX", "TRTLLM_CAPKIT_SNX"}


def test_pick_free_port_stays_below_the_ephemeral_range(tmp_path):
    rf = tmp_path / "range"
    rf.write_text("32768\t60999\n")
    ports = {serve_nx.pick_free_port("127.0.0.1", range_file=str(rf)) for _ in range(20)}
    assert all(serve_nx.PORT_FLOOR <= p < 32768 for p in ports)
    # positive control: the old OS-assigned pick lands in the ephemeral range
    rf.write_text("10500 60999\n")  # no room below -> fallback to an OS-assigned port
    assert serve_nx.pick_free_port("127.0.0.1", range_file=str(rf)) >= 10500
    assert serve_nx.pick_free_port("127.0.0.1", range_file=str(tmp_path / "none")) > 0


def test_pick_free_port_skips_a_bound_port(tmp_path):
    rf = tmp_path / "range"
    rf.write_text("32768 60999\n")
    busy = socket.socket()
    busy.bind(("127.0.0.1", 0))
    taken = busy.getsockname()[1]
    seq = iter([taken, 20001])

    class Rng:
        def randrange(self, a, b):
            return next(seq)
    try:
        if not (serve_nx.PORT_FLOOR <= taken < 32768):
            busy.close()
            busy = socket.socket()
            busy.bind(("127.0.0.1", 20000))
            taken = 20000
            seq = iter([taken, 20001])
        assert serve_nx.pick_free_port("127.0.0.1", range_file=str(rf), rng=Rng()) == 20001
    finally:
        busy.close()


def test_self_connected_detects_a_loopback_self_connect():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.connect(("127.0.0.1", port))  # TCP simultaneous open with itself
    try:
        assert serve_nx.self_connected(s)
    finally:
        s.close()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = socket.create_connection(srv.getsockname()[:2])
    try:
        assert not serve_nx.self_connected(c)  # a real server answered
    finally:
        c.close()
        srv.close()


def test_drain_inject_dr_busy_counts_per_tier(tmp_path):
    now = time.monotonic_ns()
    plan = FakePlan(now - 10 ** 9, now + 10 ** 10)
    st = nx_driver.LaneStats(plan, str(tmp_path), "lane1", "raw", 0)
    inj = nx_driver.drain_injector("Dr:busy:300000")
    t0 = time.monotonic_ns()
    nx_driver.drain_loop(FakeZq(_frames(False), False), st, "raw", 0, threading.Event(), inj)
    assert time.monotonic_ns() - t0 >= 3 * 300000  # the sentinel frame is received too
    t = st.tiers[0]
    assert t["msgs"] == 2 and t["dr_hits"] == 2 and t["dr_applied"] == 2
    assert t["dr_applied_ns"] >= 2 * 300000
    c = inj.counters()
    assert c["hits"] == 3 and c["applied"] == 3


def test_drain_inject_count_and_refusals(tmp_path):
    now = time.monotonic_ns()
    plan = FakePlan(now - 10 ** 9, now + 10 ** 10)
    st = nx_driver.LaneStats(plan, str(tmp_path), "lane1", "deserialize", 0)
    nx_driver.drain_loop(FakeZq(_frames(True), True), st, "deserialize", 0, threading.Event(),
                         nx_driver.drain_injector("Dr:count"))
    assert st.tiers[0]["dr_hits"] == 2 and "dr_applied" not in st.tiers[0]
    st2 = nx_driver.LaneStats(plan, str(tmp_path), "lane2", "raw", 0)
    nx_driver.drain_loop(FakeZq(_frames(False), False), st2, "raw", 0, threading.Event())
    assert "dr_hits" not in st2.tiers[0]  # no Dr: summaries as before
    assert nx_driver.drain_injector("") is None
    for bad in ("Dr:busy:0", "Dr:busy", "Wr:count", "Dr:sleep:5", "Dr:busy:x"):
        with pytest.raises(nx_driver.ConfigError):
            nx_driver.drain_injector(bad)
    a = nx_driver.build_parser().parse_args(["drain", "--attach-info", "x", "--frontend-id", "1",
                                             "--lanes", "2", "--drain-inject", "Dr:busy:5",
                                             "--tier-table", "t", "--schedule", "s",
                                             "--origin-ns", "1", "--out-dir", "o"])
    assert a.drain_inject == "Dr:busy:5"


# ---------------------------------------------------------------- C++ host mode (D1)
def _cpp(argv):
    own, serve = serve_nx.split_argv(argv)
    return serve_nx.cpp_host_env(serve_nx.build_parser().parse_args(own), serve)


def test_cpp_host_env_sets_the_launcher_variables():
    env, why = _cpp(["--cpp-host-bin", "/usr/local/bin/trtllm-openengine-host",
                     "--cpp-host-arg=--grpc-threads", "--cpp-host-arg", "4",
                     "--cpp-host-arg=--diag-out", "--cpp-host-arg", "/r/d1 x",
                     "--cpp-host-inject", "hs:busy:2000", "--",
                     "m", "--grpc", "--openengine-host-impl", "cpp"])
    assert why is None
    assert env == {"TLLM_OPENENGINE_CPP_HOST_BIN": "/usr/local/bin/trtllm-openengine-host",
                   "TLLM_OPENENGINE_CPP_HOST_ARGS": "--grpc-threads 4 --diag-out '/r/d1 x'",
                   "TLLM_D1_INJECT": "hs:busy:2000", None: "/r/d1 x"}
    import shlex
    assert shlex.split(env["TLLM_OPENENGINE_CPP_HOST_ARGS"]) == \
        ["--grpc-threads", "4", "--diag-out", "/r/d1 x"]


def test_cpp_host_env_refusals_and_default():
    # no C++ host option: nothing set, Python host arms unchanged
    assert _cpp(["--", "m", "--grpc"]) == ({}, None)
    assert _cpp(["--", "m", "--openengine-host-impl", "cpp"]) == ({}, None)
    # options without the cpp host impl
    for own in (["--cpp-host-bin", "/b"], ["--cpp-host-arg=--grpc-threads"],
                ["--cpp-host-inject", "hd:busy:1"], ["--report-target", "private"]):
        env, why = _cpp(own + ["--", "m", "--grpc", "--openengine-host-impl", "python"])
        assert env == {} and "openengine-host-impl cpp" in why, own
    # flags the launcher owns
    for flag in ("--listen", "--launcher-endpoint=1", "--template"):
        env, why = _cpp(["--cpp-host-arg=%s" % flag, "--", "m", "--openengine-host-impl=cpp"])
        assert env == {} and "launcher sets" in why, flag


def test_main_refuses_bad_cpp_host_options(monkeypatch, capsys):
    monkeypatch.delenv(ENV_VAR, raising=False)
    for k in ("TLLM_OPENENGINE_CPP_HOST_BIN", "TLLM_OPENENGINE_CPP_HOST_ARGS", "TLLM_D1_INJECT"):
        monkeypatch.delenv(k, raising=False)
    assert serve_nx.main(["--cpp-host-bin", "/b", "--", "model", "--grpc"]) == 3
    assert serve_nx.main(["--report-target", "private", "--", "model",
                          "--openengine-host-impl", "cpp"]) == 3
    err = capsys.readouterr().err
    assert "openengine-host-impl cpp" in err and "needs --report-addr" in err
    assert "TLLM_OPENENGINE_CPP_HOST_BIN" not in os.environ


def test_reporter_private_target_waits_for_both_ports(tmp_path):
    pub, priv = serve_nx.pick_free_port("127.0.0.1"), serve_nx.pick_free_port("127.0.0.1")
    assert pub != priv
    holder = {}
    out = str(tmp_path / "addr")
    t = serve_nx.start_reporter(out, "127.0.0.1", pub, 10.0,
                                target=lambda: None if "port" not in holder
                                else ("127.0.0.1", holder["port"]))
    socks = []
    try:
        s = socket.socket()
        s.bind(("127.0.0.1", priv))
        s.listen(1)
        socks.append(s)
        holder["port"] = priv
        time.sleep(0.3)
        assert not os.path.exists(out)  # the public port (the C++ host) is not up yet
        s = socket.socket()
        s.bind(("127.0.0.1", pub))
        s.listen(1)
        socks.append(s)
        t.join(5)
        with open(out) as f:
            assert f.read() == "127.0.0.1:%d\n" % priv
    finally:
        for s in socks:
            s.close()


def test_reporter_private_target_needs_the_port(tmp_path):
    """Positive control of the wait: the public port accepts but the private port is unknown, so
    nothing is written."""
    pub = serve_nx.pick_free_port("127.0.0.1")
    out = str(tmp_path / "addr")
    srv = socket.socket()
    srv.bind(("127.0.0.1", pub))
    srv.listen(1)
    try:
        serve_nx.start_reporter(out, "127.0.0.1", pub, 1.0, target=lambda: None).join(3)
        assert not os.path.exists(out)
    finally:
        srv.close()


def test_capture_private_port(monkeypatch):
    mod = types.ModuleType("tensorrt_llm.grpc.openengine.cpp_host")
    calls = []

    def cpp_host_command(binary, *, listen, launcher_port, template_dir, extra_args=None):
        calls.append((binary, listen, launcher_port, template_dir, extra_args))
        return [binary, "--launcher-endpoint", "127.0.0.1:%d" % launcher_port]
    mod.cpp_host_command = cpp_host_command
    for name in ("tensorrt_llm", "tensorrt_llm.grpc", "tensorrt_llm.grpc.openengine"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(__import__("sys").modules, name, pkg)
    monkeypatch.setitem(__import__("sys").modules, mod.__name__, mod)
    __import__("sys").modules["tensorrt_llm.grpc.openengine"].cpp_host = mod
    holder = {}
    serve_nx.capture_private_port(holder)
    assert mod.cpp_host_command("/b", listen=("127.0.0.1", 1), launcher_port=4321,
                                template_dir="/t") == ["/b", "--launcher-endpoint",
                                                       "127.0.0.1:4321"]
    assert holder == {"port": 4321} and calls == [("/b", ("127.0.0.1", 1), 4321, "/t", None)]
