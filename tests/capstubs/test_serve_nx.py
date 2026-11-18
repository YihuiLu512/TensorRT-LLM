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
