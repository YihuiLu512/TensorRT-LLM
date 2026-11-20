# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.snx_segtime on stand-in product modules (no tensorrt_llm here): known-CPU positive
control, per-thread accounting, inclusive nesting, staticmethod / module-global / pickle-proxy /
gRPC-handler sites, refusals before any change, the JSONL reporter and window interpolation, the
wrapper's calibration, serve_nx's option checks, and the sites against the probe tree's source."""

import ast
import json
import os
import pickle as real_pickle
import threading
import time
import types

import pytest

from capstubs import serve_nx, snx_segtime

C = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def spin_cpu(ns):
    t0 = time.thread_time_ns()
    while time.thread_time_ns() - t0 < ns:
        pass


def fake_modules():
    """Stand-ins with the product's shapes for every site (same owners, kinds, call paths)."""
    P = "tensorrt_llm."
    ipc = types.ModuleType(P + "executor.ipc")
    ipc.pickle = real_pickle

    class ZeroMqQueue:
        def __init__(self, frames):
            self.frames = list(frames)

        def _verify_hmac(self, data, mac):
            spin_cpu(20_000)
            return mac == b"ok"

        def _parse_data(self, data):
            if not self._verify_hmac(data, b"ok"):
                raise RuntimeError("hmac")
            return ipc.pickle.loads(data)  # module global looked up at call time

        def _recv_data(self):
            return self._parse_data(self.frames.pop(0))

        def get(self):
            return self._recv_data()

        def _prepare_data(self, obj):
            return ipc.pickle.dumps(obj)
    ipc.ZeroMqQueue = ZeroMqQueue
    utils = types.ModuleType(P + "llmapi.utils")

    class _SyncQueue:
        woken = []

        @staticmethod
        def _notify_many(queues):
            _SyncQueue.woken.extend(queues)

        @staticmethod
        def notify_many(loop, queues):
            _SyncQueue._notify_many(frozenset(queues))
    utils._SyncQueue = _SyncQueue
    proxy = types.ModuleType(P + "executor.proxy")

    class GenerationExecutorProxy:
        def __init__(self, q):
            self.result_queue = q
            self.got = []

        def dispatch_result_task(self):
            res = self.result_queue.get()
            self.got.extend(res)
            utils._SyncQueue.notify_many(None, [len(self.got)])
            return True
    proxy.GenerationExecutorProxy = GenerationExecutorProxy
    llm_request = types.ModuleType(P + "_torch.pyexecutor.llm_request")

    class LlmResult:
        def deserialize(self):
            spin_cpu(30_000)
    llm_request.LlmResult = LlmResult
    result = types.ModuleType(P + "executor.result")

    class GenerationResultBase:
        def _handle_response(self, response):
            response.deserialize()

    class DetokenizedGenerationResultBase(GenerationResultBase):
        def _handle_response(self, response):
            GenerationResultBase._handle_response(self, response)
            spin_cpu(10_000)
    result.GenerationResultBase = GenerationResultBase
    result.DetokenizedGenerationResultBase = DetokenizedGenerationResultBase
    servicer = types.ModuleType(P + "grpc.openengine.servicer")
    servicer._format_result = lambda r, s: [("msg", r)]
    bindings = types.ModuleType(P + "grpc.openengine.bindings")

    class GenerateResponse:
        @staticmethod
        def SerializeToString(msg):
            return repr(msg).encode()

    class GenerateRequest:
        @classmethod
        def FromString(cls, data):
            return ("req", data)
    bindings.generation_pb2 = types.SimpleNamespace(GenerateResponse=GenerateResponse,
                                                    GenerateRequest=GenerateRequest)
    llm = types.ModuleType(P + "llmapi.llm")

    class BaseLLM:
        def generate_async(self, inputs):
            return ("handle", inputs)
    llm.BaseLLM = BaseLLM
    return {m.__name__: m for m in (ipc, utils, proxy, llm_request, result, servicer, bindings,
                                    llm)}


class FakeGrpc:
    def __init__(self):
        self.handlers = []

    def unary_stream_rpc_method_handler(self, behavior, request_deserializer=None,
                                        response_serializer=None):
        h = types.SimpleNamespace(behavior=behavior, request_deserializer=request_deserializer,
                                  response_serializer=response_serializer)
        self.handlers.append(h)
        return h


def seg_of(st, thread_name=None):
    """{site: [n, ns]} summed over threads (or one thread name)."""
    out = {}
    for t in st.reg.snapshot():
        if thread_name is not None and t["name"] != thread_name:
            continue
        for s, v in t["seg"].items():
            o = out.setdefault(s, [0, 0])
            o[0] += v[0]
            o[1] += v[1]
    return out


def test_parse_sites():
    assert snx_segtime.parse_sites("") == ()
    assert snx_segtime.parse_sites("all") == tuple(snx_segtime.SITES)
    assert snx_segtime.parse_sites("format, dtask") == ("dtask", "format")
    for bad in ("nope", "dtask,dtask", "all,dtask"):
        with pytest.raises(snx_segtime.SiteMissing):
            snx_segtime.parse_sites(bad)
    for child, parent in snx_segtime.NESTING.items():
        assert child in snx_segtime.SITES and parent in snx_segtime.SITES


def test_positive_control_known_cpu_and_inclusive_nesting():
    mods = fake_modules()
    st = snx_segtime.install(("handle", "hbase", "deser"), modules=mods)
    try:
        r = mods["tensorrt_llm.executor.result"].DetokenizedGenerationResultBase()
        res = mods["tensorrt_llm._torch.pyexecutor.llm_request"].LlmResult()
        for _ in range(200):
            r._handle_response(res)
        seg = seg_of(st)
        assert seg["handle"][0] == seg["hbase"][0] == seg["deser"][0] == 200
        per = {s: v[1] / v[0] for s, v in seg.items()}
        assert 30_000 <= per["deser"] <= 30_000 * 1.25, per
        assert per["hbase"] >= per["deser"] and per["handle"] >= per["hbase"] + 10_000, per
        assert per["handle"] <= 40_000 * 1.3, per
    finally:
        st.uninstall()
    assert not getattr(mods["tensorrt_llm.executor.result"].GenerationResultBase.__dict__[
        "_handle_response"], "capkit_segtime", False)


def test_dispatch_path_pickle_proxy_static_and_threads():
    mods = fake_modules()
    ipc = mods["tensorrt_llm.executor.ipc"]
    orig_static = mods["tensorrt_llm.llmapi.utils"]._SyncQueue.__dict__["notify_many"]
    st = snx_segtime.install(("dtask", "recv", "parse", "hmac", "pickle", "notify", "wake",
                              "prep"), modules=mods)
    try:
        assert isinstance(ipc.pickle, snx_segtime.PickleProxy)
        assert ipc.pickle.dumps is real_pickle.dumps  # other names pass through
        frames = [real_pickle.dumps([i, i + 1]) for i in range(50)]
        px = mods["tensorrt_llm.executor.proxy"].GenerationExecutorProxy(ipc.ZeroMqQueue(frames))

        def run():
            for _ in range(50):
                px.dispatch_result_task()
        t = threading.Thread(target=run, name="proxy_dispatch_result_thread")
        t.start()
        t.join()
        assert px.got[:4] == [0, 1, 1, 2]
        d = seg_of(st, "proxy_dispatch_result_thread")
        for s in ("dtask", "recv", "parse", "hmac", "pickle", "notify", "wake"):
            assert d[s][0] == 50, (s, d)
        assert d["dtask"][1] >= d["recv"][1] >= d["parse"][1] >= d["hmac"][1] >= 50 * 20_000
        assert seg_of(st, threading.current_thread().name) == {}  # nothing on this thread
        ipc.ZeroMqQueue([])._prepare_data({"a": 1})
        assert seg_of(st, threading.current_thread().name)["prep"][0] == 1
        static = mods["tensorrt_llm.llmapi.utils"]._SyncQueue.__dict__["notify_many"]
        assert isinstance(static, staticmethod)
    finally:
        st.uninstall()
    assert ipc.pickle is real_pickle
    assert mods["tensorrt_llm.llmapi.utils"]._SyncQueue.__dict__["notify_many"] is orig_static


def test_grpc_handler_serializers_wrapped_only_for_generate():
    mods = fake_modules()
    g = FakeGrpc()
    st = snx_segtime.install(("pbser", "pbdeser", "format", "gen"), modules=mods, grpc_mod=g)
    try:
        pb = mods["tensorrt_llm.grpc.openengine.bindings"].generation_pb2
        h = g.unary_stream_rpc_method_handler(
            "Generate", request_deserializer=pb.GenerateRequest.FromString,
            response_serializer=pb.GenerateResponse.SerializeToString)
        other = g.unary_stream_rpc_method_handler("Watch", request_deserializer=bytes,
                                                  response_serializer=bytes)
        assert other.response_serializer is bytes and other.request_deserializer is bytes
        assert h.request_deserializer(b"x") == ("req", b"x")
        for i in range(7):
            h.response_serializer(i)
            mods["tensorrt_llm.grpc.openengine.servicer"]._format_result(i, None)
        mods["tensorrt_llm.llmapi.llm"].BaseLLM().generate_async("p")
        seg = seg_of(st)
        assert seg["pbser"][0] == 7 and seg["pbdeser"][0] == 1 and seg["format"][0] == 7
        assert seg["gen"][0] == 1
        assert st.grpc_wrapped == [{"behavior": "'Generate'",
                                    "wrapped": ["request_deserializer", "response_serializer"]}]
    finally:
        st.uninstall()
    assert g.unary_stream_rpc_method_handler.__name__ == "unary_stream_rpc_method_handler"


@pytest.mark.parametrize("site,breaker", [
    ("dtask", lambda m: delattr(m["tensorrt_llm.executor.proxy"].GenerationExecutorProxy,
                                "dispatch_result_task")),
    ("notify", lambda m: setattr(m["tensorrt_llm.llmapi.utils"]._SyncQueue, "notify_many",
                                 lambda loop, q: None)),
    ("pickle", lambda m: setattr(m["tensorrt_llm.executor.ipc"], "pickle", None)),
    ("format", lambda m: setattr(m["tensorrt_llm.grpc.openengine.servicer"], "_format_result",
                                 None)),
    ("handle", lambda m: setattr(m["tensorrt_llm.executor.result"],
                                 "DetokenizedGenerationResultBase", None)),
])
def test_missing_or_wrong_site_refuses_before_any_change(site, breaker):
    mods = fake_modules()
    breaker(mods)
    before = mods["tensorrt_llm.executor.ipc"].ZeroMqQueue.__dict__["_recv_data"]
    with pytest.raises(snx_segtime.SiteMissing):
        snx_segtime.install(("recv", site), modules=mods)
    assert mods["tensorrt_llm.executor.ipc"].ZeroMqQueue.__dict__["_recv_data"] is before


def test_refuses_coroutines_and_double_wrapping():
    async def co():
        return None
    with pytest.raises(snx_segtime.SiteMissing):
        snx_segtime.make_timed(co, 0, snx_segtime.Registry(("x",)))
    mods = fake_modules()
    st = snx_segtime.install(("pickle", "recv"), modules=mods)
    try:
        for s in ("pickle", "recv"):
            with pytest.raises(snx_segtime.SiteMissing):
                snx_segtime.install((s,), modules=mods)
    finally:
        st.uninstall()


def test_reporter_lines_and_window_delta(tmp_path):
    mods = fake_modules()
    st = snx_segtime.install(("format",), modules=mods, out_dir=str(tmp_path), reporter_s=0.05,
                             header={"role": "test"})
    fmt = mods["tensorrt_llm.grpc.openengine.servicer"]._format_result
    t_start = time.monotonic_ns()
    for _ in range(30):
        fmt(1, None)
        time.sleep(0.01)
    t_end = time.monotonic_ns()
    time.sleep(0.12)
    st.stop()
    st.uninstall()
    hdr, rows = snx_segtime.read(st.path)
    assert hdr["role"] == "test" and hdr["sites"] == ["format"]
    assert hdr["calibration"]["n"] == snx_segtime.CALIBRATE_N
    assert len(rows) >= 4 and rows[-1]["threads"][0]["seg"]["format"][0] == 30
    d = snx_segtime.window_delta(rows, rows[0]["t_ns"], rows[-1]["t_ns"])
    [(key, seg)] = d.items()
    assert key[0] == threading.current_thread().name
    assert seg["format"][0] == 30 - sum(t["seg"].get("format", [0])[0]
                                         for t in rows[0]["threads"])
    assert snx_segtime.window_delta(rows, rows[0]["t_ns"] - 10**9, rows[-1]["t_ns"]) is None
    assert t_start < rows[-1]["t_ns"] and t_end <= rows[-1]["t_ns"]


def test_value_at_interpolates():
    rows = [{"t_ns": 0, "threads": [{"name": "m", "native_id": 1, "seg": {"a": [0, 0]}}]},
            {"t_ns": 1000, "threads": [{"name": "m", "native_id": 1, "seg": {"a": [10, 500]}}]},
            {"t_ns": 4_000_000_000, "threads": [{"name": "m", "native_id": 1,
                                                 "seg": {"a": [20, 900]}}]}]
    assert snx_segtime.value_at(rows, 500) == {("m", 1, "a"): [5.0, 250.0]}
    assert snx_segtime.window_delta(rows, 250, 1000) == {("m", 1): {"a": [7.5, 375.0]}}
    assert snx_segtime.value_at(rows, 2000) is None  # gap > 2.5 s
    assert snx_segtime.value_at(rows, -1) is None and snx_segtime.value_at(rows, 5 * 10**9) is None


def test_calibration_and_overhead_bound():
    cal = snx_segtime.calibrate(5000)
    assert 0 < cal["inside_ns"] < 5000 and 0 < cal["per_call_ns"] < 5000, cal


def test_serve_nx_segtime_option_checks(monkeypatch, capsys):
    monkeypatch.setattr(serve_nx, "apply_patches", lambda *a, **k: {})
    assert serve_nx.main(["--segtime", "nope", "--", "model"]) == 3
    assert "--segtime refused" in capsys.readouterr().err
    assert serve_nx.main(["--segtime", "all", "--", "model"]) == 3  # no --out-dir
    assert "--segtime needs --out-dir" in capsys.readouterr().err
    assert serve_nx.main(["--segtime", "all", "--segtime-report-ms", "0", "--out-dir", "/tmp/x",
                          "--", "model"]) == 3


SRC = {"tensorrt_llm.executor.proxy": "tensorrt_llm/executor/proxy.py",
       "tensorrt_llm.executor.ipc": "tensorrt_llm/executor/ipc.py",
       "tensorrt_llm.llmapi.utils": "tensorrt_llm/llmapi/utils.py",
       "tensorrt_llm.executor.result": "tensorrt_llm/executor/result.py",
       "tensorrt_llm._torch.pyexecutor.llm_request":
       "tensorrt_llm/_torch/pyexecutor/llm_request.py",
       "tensorrt_llm.grpc.openengine.servicer": "tensorrt_llm/grpc/openengine/servicer.py",
       "tensorrt_llm.llmapi.llm": "tensorrt_llm/llmapi/llm.py"}


def test_sites_exist_with_their_kind_in_the_probe_tree():
    """Owners and kinds against the probe tree's source (the product tree of the S-NX images);
    plus the call paths the timing relies on: dispatch_result_task -> result_queue.get, the
    module-global names (ipc pickle.loads, servicer _format_result), the explicit base call in
    the detokenizing _handle_response, and grpc.unary_stream_rpc_method_handler with
    GenerateResponse.SerializeToString in the generated Generate handler."""
    root = os.path.join(C, "probe-tree")
    assert os.path.isdir(os.path.join(root, "tensorrt_llm")), "needs the probe-tree worktree"
    trees = {m: ast.parse(open(os.path.join(root, p)).read()) for m, p in SRC.items()}
    for site, (mod, path, attr, kind) in snx_segtime.SITES.items():
        if kind in ("grpc_ser", "grpc_deser"):
            continue
        tree = trees[mod]
        if kind == "loads_proxy":
            assert any(isinstance(n, ast.Import) and [a.name for a in n.names] == ["pickle"]
                       for n in tree.body), site
            continue
        if path is None:
            [f] = [n for n in tree.body if isinstance(n, ast.ImportFrom)
                   and any(a.name == attr for a in n.names)] or [
                n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == attr]
            continue
        [cls] = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == path]
        [f] = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == attr]
        assert isinstance(f, ast.FunctionDef), site
        deco = [ast.unparse(d) for d in f.decorator_list]
        assert ("staticmethod" in deco) == (kind == "static"), (site, deco)
    src = {m: open(os.path.join(root, p)).read() for m, p in SRC.items()}
    assert "self.result_queue.get()" in src["tensorrt_llm.executor.proxy"]
    assert "return pickle.loads(message_data)" in src["tensorrt_llm.executor.ipc"]
    assert "GenerationResultBase._handle_response(self, response)" in \
        src["tensorrt_llm.executor.result"]
    assert "responses = _format_result(result, format_state)" in \
        src["tensorrt_llm.grpc.openengine.servicer"]
    gen = open(os.path.join(root, "tensorrt_llm/grpc/openengine/_generated/"
                                  "openengine_pb2_grpc.py")).read()
    i = gen.index("'Generate': grpc.unary_stream_rpc_method_handler(")
    assert "GenerateResponse.SerializeToString" in gen[i:i + 400]
    assert "GenerateRequest.FromString" in gen[i:i + 400]
