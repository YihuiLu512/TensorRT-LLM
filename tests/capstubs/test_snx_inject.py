# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.snx_inject on fake product modules (no tensorrt_llm here): spec parsing, the three
site wrappers (count and busy), refusals, the JSONL reporter, gc statistics, the connection census
and serve_nx's option plumbing."""

import gc
import json
import os
import socket
import threading
import time
import types

import pytest

from capcommon.inject import InjectSpecError

from capstubs import serve_nx, snx_inject


def fake_modules():
    servicer = types.ModuleType("tensorrt_llm.grpc.openengine.servicer")
    servicer._format_result = lambda result, state: [("msg", result)]

    def run_servicer(results):  # looks the name up in the module globals at call time
        out = []
        for r in results:
            out += servicer._format_result(r, None)
        return out
    servicer.run = run_servicer
    proxy = types.ModuleType("tensorrt_llm.executor.proxy")
    tracer = types.SimpleNamespace(n=0)

    def log_instant(name):
        tracer.n += 1
    tracer.log_instant = log_instant
    proxy.global_tracer = lambda: tracer
    proxy.tracer = tracer
    stream = types.ModuleType("tensorrt_llm.grpc.openengine_client.stream")

    class GenerateStream:
        def __init__(self):
            self.seen = []

        def _validate(self, response):
            self.seen.append(response)
            return True
    stream.GenerateStream = GenerateStream
    return {"tensorrt_llm.grpc.openengine.servicer": servicer,
            "tensorrt_llm.executor.proxy": proxy,
            "tensorrt_llm.grpc.openengine_client.stream": stream}


def test_parse_specs_and_diags():
    sp = snx_inject.parse_specs("Hs:busy:156250; Hd:count;Fr:count")
    assert sp["Hs"].mode == "busy" and sp["Hs"].delta_ns == 156250
    assert sp["Hd"] is None and sp["Fr"] is None
    assert snx_inject.parse_specs("") == {}
    for bad in ("Hx:count", "Hs:busy", "Hs:busy:x", "Hs:sleep:5", "Hs:count;Hs:count",
                "Hs:busy:0"):
        with pytest.raises(InjectSpecError):
            snx_inject.parse_specs(bad)
    assert snx_inject.parse_diags("threads, gc") == ["gc", "threads"]
    with pytest.raises(InjectSpecError):
        snx_inject.parse_diags("gc,heap")


def test_sites_count_and_restore():
    mods = fake_modules()
    st = snx_inject.install({"Hs": None, "Hd": None, "Fr": None}, modules=mods)
    try:
        assert mods["tensorrt_llm.grpc.openengine.servicer"].run(range(7)) == \
            [("msg", i) for i in range(7)]
        prx = mods["tensorrt_llm.executor.proxy"]
        for _ in range(5):
            prx.global_tracer().log_instant("IPC.get")
        gs = mods["tensorrt_llm.grpc.openengine_client.stream"].GenerateStream()
        for i in range(3):
            assert gs._validate(i) is True
        assert gs.seen == [0, 1, 2] and prx.tracer.n == 5
        c = {s: i.counters() for s, i in st.injectors.items()}
        assert (c["Hs"]["hits"], c["Hd"]["hits"], c["Fr"]["hits"]) == (7, 5, 3)
        assert all(c[s]["applied"] == 0 for s in c)
    finally:
        st.uninstall()
    assert not hasattr(mods["tensorrt_llm.grpc.openengine.servicer"]._format_result,
                       "capkit_snx_site")
    assert not hasattr(mods["tensorrt_llm.grpc.openengine_client.stream"].GenerateStream._validate,
                       "capkit_snx_site")


def test_busy_applies_delta():
    mods = fake_modules()
    st = snx_inject.install(snx_inject.parse_specs("Hs:busy:200000"), modules=mods)
    try:
        t0 = time.monotonic_ns()
        mods["tensorrt_llm.grpc.openengine.servicer"].run(range(10))
        took = time.monotonic_ns() - t0
        c = st.injectors["Hs"].counters()
        assert c["hits"] == c["applied"] == 10
        assert c["applied_ns"] >= 10 * 200000 and took >= 10 * 200000
    finally:
        st.uninstall()


def test_missing_target_refuses_before_change():
    mods = fake_modules()
    del mods["tensorrt_llm.executor.proxy"].global_tracer
    orig = mods["tensorrt_llm.grpc.openengine.servicer"]._format_result
    with pytest.raises(snx_inject.SiteMissing):
        snx_inject.install({"Hs": None, "Hd": None}, modules=mods)
    assert mods["tensorrt_llm.grpc.openengine.servicer"]._format_result is orig


def test_reporter_jsonl_and_gc(tmp_path):
    mods = fake_modules()
    st = snx_inject.install({"Hs": None}, ["gc", "threads", "conn"], out_dir=str(tmp_path),
                            modules=mods, reporter=False, header={"role": "t"})
    try:
        mods["tensorrt_llm.grpc.openengine.servicer"].run(range(4))
        gc.collect()
        st.write_line()
        st.write_line()
    finally:
        st.uninstall()
    rows = [json.loads(x) for x in open(st.path)]
    assert rows[0]["schema"] == snx_inject.SCHEMA and rows[0]["role"] == "t"
    assert rows[0]["specs"] == {"Hs": "count"} and rows[0]["diags"] == ["conn", "gc", "threads"]
    assert rows[1]["sites"]["Hs"]["hits"] == 4
    assert rows[1]["gc"]["gen"]["2"]["n"] >= 1
    assert any(n == "MainThread" for n, _ in rows[1]["threads"])
    assert "conn" in rows[1] and "conn" not in rows[2]  # census at most every CONN_EVERY_S
    assert snx_inject.GcStats not in gc.callbacks


def test_gcstats_long_pause_records_thread():
    t = [0]
    g = snx_inject.GcStats(clock=lambda: t[0])
    g("start", {"generation": 2})
    t[0] = 60 * 1000 * 1000
    g("stop", {"generation": 2, "collected": 3})
    s = g.snapshot()
    assert s["gen"]["2"] == {"n": 1, "ns": 60000000, "max_ns": 60000000}
    assert s["long"][0]["thread"] == threading.current_thread().name


def test_conn_census_counts_accepted():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    cl = [socket.create_connection(("127.0.0.1", port)) for _ in range(3)]
    acc = [srv.accept()[0] for _ in range(3)]
    try:
        c = snx_inject.conn_census()
        assert port in c["listen"]
        assert sorted(c["accepted"][str(port)]) == sorted(s.getsockname()[1] for s in cl)
    finally:
        for s in cl + acc + [srv]:
            s.close()


def test_conn_census_rows_filter():
    rows = [("0A", 50051, 0, "10"), ("01", 50051, 4000, "11"), ("01", 50051, 4001, "99"),
            ("01", 6000, 50051, "12")]
    orig = snx_inject._socket_inodes
    snx_inject._socket_inodes = lambda pid="self": {"10", "11", "12"}
    try:
        assert snx_inject.conn_census(rows=rows) == {"listen": [50051],
                                                     "accepted": {"50051": [4000]}}
    finally:
        snx_inject._socket_inodes = orig


def test_install_from_env(tmp_path):
    assert snx_inject.install_from_env(str(tmp_path), environ={}) is None
    with pytest.raises(InjectSpecError):
        snx_inject.install_from_env(str(tmp_path), environ={snx_inject.ENV_INJECT: "Zz:count"})


def test_serve_nx_options_parse():
    a = serve_nx.build_parser().parse_args(["--inject", "Hs:busy:156250", "--inject", "Hd:count",
                                            "--diag", "gc,conn", "--gc-freeze-after-ready"])
    assert a.inject == ["Hs:busy:156250", "Hd:count"] and a.diag == "gc,conn"
    assert a.gc_freeze_after_ready


def test_serve_nx_refuses_bad_inject(monkeypatch, capsys, tmp_path):
    from capstubs.null_executor import ENV_VAR
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv(snx_inject.ENV_INJECT, raising=False)
    assert serve_nx.main(["--inject", "Hq:count", "--out-dir", str(tmp_path), "--", "m"]) == 3
    assert "inject refused" in capsys.readouterr().err
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert serve_nx.main(["--gc-freeze-after-ready", "--out-dir", str(tmp_path), "--", "m"]) == 3


def test_reporter_after_callback(tmp_path):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    done = threading.Event()
    path = str(tmp_path / "addr")
    try:
        t = serve_nx.start_reporter(path, "127.0.0.1", port, 5.0, after=done.set)
        t.join(5)
        assert done.is_set() and open(path).read() == "127.0.0.1:%d\n" % port
    finally:
        srv.close()


def test_gc_freeze_now():
    try:
        assert snx_inject.gc_freeze_now() > 0
    finally:
        gc.unfreeze()


def test_sites_table_names():
    assert set(snx_inject.SITES) == {"Hs", "Hd", "Fr", "Pp"}
    assert os.path.basename(snx_inject.__file__) == "snx_inject.py"
