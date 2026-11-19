# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E15-A2 frame capture (capstubs.frame_capture, nx_role --frame-capture-*, the nx_driver drain
hook) and the offline replay (capstubs.frame_replay) without tensorrt_llm: frames are pickles of
plain objects signed with a real HMAC key, the expected values are the drain's and a fake
worker's record files written with the production record formats. The product proxy path runs
only in the container (E7 e15a capture job)."""

import hashlib
import hmac
import json
import os
import pickle
import stat
import sys
import threading
import time
import types

import pytest

from capcommon.exitcodes import ConfigError

from capstubs import frame_capture as fc
from capstubs import frame_replay as fr
from capstubs import nx_driver, nx_role
from capstubs.null_executor import REQ_REC, TAKE_REC, RecordFile

KEY_RES, KEY_REQ = b"r" * 32, b"q" * 32


class Plan:
    """Two tiers: seq 1 before t = 1000, seq 2 from 1000 on."""

    def __init__(self):
        self.origin, self.end_abs, self.rows = 0, 10 ** 18, []
        self.tt = types.SimpleNamespace(sha256="t" * 64)

    def tier_seq_at(self, t):
        return 1 if t < 1000 else 2


def sign(key, obj, protocol=None):
    d = pickle.dumps(obj, protocol=protocol)
    return d + hmac.new(key, d, hashlib.sha256).digest()


def resp(i):
    return types.SimpleNamespace(client_id=i, result=types.SimpleNamespace(is_final=False),
                                 error_msg=None)


def req(cid, L):
    return types.SimpleNamespace(id=cid, prompt_token_ids=list(range(L)))


# ---- FrameCapture --------------------------------------------------------------------------

def test_capture_every_per_tier_and_files(tmp_path):
    cap = fc.FrameCapture(str(tmp_path / "c"), Plan(), every=2, per_tier=2, max_bytes=1 << 20)
    for i in range(10):
        cap.record("result", b"x%d" % i, 10 * i if i < 5 else 1000 + i)
    cap.set_queue("result", types.SimpleNamespace(hmac_key=KEY_RES, use_hmac_encryption=True),
                  "proxy_result_queue")
    cap.set_queue("request", types.SimpleNamespace(hmac_key=None), "q")
    meta = cap.close({"prompt_len": 128})
    assert cap.close() is None  # idempotent
    assert not cap.record("result", b"late", 5)
    d = meta["directions"]["result"]
    assert d["frames_seen"] == 10 and d["frames_captured"] == 4
    assert d["by_tier"] == {"1": {"seen": 5, "captured": 2}, "2": {"seen": 5, "captured": 2}}
    got = fc.read_frames(str(tmp_path / "c" / "frames.result.bin"))
    assert [(s, t, q, x) for s, t, q, x in got] == [(2, 10, 1, b"x1"), (4, 30, 1, b"x3"),
                                                     (6, 1005, 2, b"x5"), (8, 1007, 2, b"x7")]
    kp = tmp_path / "c" / "frames.keys.json"
    assert stat.S_IMODE(os.stat(kp).st_mode) == 0o600
    keys = json.load(open(kp))["hmac_key_hex"]
    assert keys == {"result": KEY_RES.hex(), "request": None}
    m2 = json.load(open(tmp_path / "c" / "frames.meta.json"))
    assert m2["prompt_len"] == 128 and m2["header"]["format"] == fc.FRAME_HDR.format


def test_capture_byte_cap_and_truncation(tmp_path):
    cap = fc.FrameCapture(str(tmp_path), Plan(), per_tier=100, max_bytes=10)
    assert cap.record("result", b"123456", 1) and cap.record("request", b"123456", 1)
    assert not cap.record("result", b"1", 2)  # 12 bytes captured >= 10
    cap.close()
    assert cap.dirs["result"].seq == 2 and cap.dirs["result"].captured == 1
    p = tmp_path / "frames.result.bin"
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError):
        fc.read_frames(str(p))
    with pytest.raises(FileExistsError):
        fc.FrameCapture(str(tmp_path), Plan())
    with pytest.raises(ValueError):
        fc.FrameCapture(str(tmp_path / "x"), Plan(), every=0)


class FakeQueue:
    def __init__(self, key, name):
        self.hmac_key, self.name, self.use_hmac_encryption = key, name, True

    def _prepare_data(self, obj):
        return sign(self.hmac_key, obj)


def test_request_wrapper_target_kinds_and_restore(tmp_path):
    orig = FakeQueue.__dict__["_prepare_data"]
    cap = fc.FrameCapture(str(tmp_path / "a"), Plan(), clock=lambda: 7)
    q, other = FakeQueue(KEY_REQ, "proxy_request_queue"), FakeQueue(KEY_REQ, "x")
    proxy = types.SimpleNamespace(request_queue=q, result_queue=types.SimpleNamespace(
        queue=FakeQueue(KEY_RES, "proxy_result_queue")))
    fc.attach_proxy(cap, proxy, FakeQueue)
    assert FakeQueue.__dict__["_prepare_data"] is not orig
    out = q._prepare_data(req(1, 3))  # SimpleNamespace: not a GenerationRequest by type name
    assert out == sign(KEY_REQ, req(1, 3))
    meta = cap.close()
    assert FakeQueue.__dict__["_prepare_data"] is orig  # restored
    assert meta["directions"]["request"]["frames_seen"] == 0
    assert meta["directions"]["request"]["other_objects"] == {"SimpleNamespace": 1}
    assert meta["queues"]["request"]["name"] == "proxy_request_queue"
    assert json.load(open(tmp_path / "a" / "frames.keys.json"))["hmac_key_hex"] == {
        "result": KEY_RES.hex(), "request": KEY_REQ.hex()}

    cap = fc.FrameCapture(str(tmp_path / "b"), Plan(), clock=lambda: 7)
    fc.install_request_capture(cap, FakeQueue, q, is_request=lambda o: o is not None)
    q._prepare_data(req(1, 3))
    q._prepare_data(None)
    other._prepare_data(req(2, 3))  # not the target queue: neither counted nor captured
    meta = cap.close()
    assert FakeQueue.__dict__["_prepare_data"] is orig
    d = meta["directions"]["request"]
    assert d["frames_seen"] == 1 and d["other_objects"] == {"NoneType": 1}
    got = fc.read_frames(str(tmp_path / "b" / "frames.request.bin"))
    assert [(x[0], x[1], x[3]) for x in got] == [(1, 7, sign(KEY_REQ, req(1, 3)))]


class FakeSock:
    def __init__(self, frames):
        self.frames = list(frames)

    def poll(self, _ms):
        return bool(self.frames)

    def recv(self):
        return self.frames.pop(0)


class FakeZq:
    def __init__(self, frames, key):
        self.socket, self.hmac_key, self.use_hmac_encryption = FakeSock(frames), key, True

    def setup_lazily(self):
        pass

    def _parse_data(self, data):
        return pickle.loads(data[:-32])


@pytest.mark.parametrize("consumer", ["raw", "deserialize"])
def test_drain_hook_aligns_with_recvs(tmp_path, consumer):
    frames = [sign(KEY_RES, [resp(i)] * (i + 1)) for i in range(4)] + [sign(KEY_RES, None)]
    cap = fc.FrameCapture(str(tmp_path / "c"), Plan())
    st = nx_driver.LaneStats(Plan(), str(tmp_path), "lane0", consumer, 0)
    nx_driver.drain_loop(FakeZq(frames, KEY_RES), st, consumer, 0, threading.Event(), cap=cap)
    cap.close()
    st.summary()
    got = fc.read_frames(str(tmp_path / "c" / "frames.result.bin"))
    raw = open(tmp_path / "nxd.lane0.recvs.rec", "rb").read()
    recvs = [nx_driver.RECV_REC.unpack_from(raw, i * nx_driver.RECV_REC.size)
             for i in range(len(raw) // nx_driver.RECV_REC.size)]
    assert st.sentinel and len(got) == len(recvs) == 4  # sentinel neither counted nor captured
    for (seq, t, _q, data), (rn, nb, _r) in zip(got, recvs):
        assert (t, len(data)) == (rn, nb) and data == frames[seq - 1]
    os.makedirs(tmp_path / "d")
    st2 = nx_driver.LaneStats(Plan(), str(tmp_path / "d"), "lane0", "raw", 0)
    nx_driver.drain_loop(FakeZq([frames[0], frames[-1]], KEY_RES), st2, "raw", 0,
                         threading.Event())
    assert sum(t["msgs"] for t in st2.tiers.values()) == 1  # no cap: as before


# ---- nx_role --frame-capture-* ----------------------------------------------------------------

def _args(extra=()):
    return nx_role.build_parser().parse_args(
        ["--experiment", "E", "--arm", "a", "--launch-seq", "0", "--job-id", "j",
         "--restart-count", "0", "--register-sha", "s", "--role", "r", "--run-dir", "/",
         "--loop", "asyncio", "--tier-table", "/t", "--origin-ns", "1", "--shm-dir", "/",
         "--snapshot-interval-ms", "50", "--ready-file", "/r", "--cpus", "0",
         "--stop-grace-s", "1", "--mode", "proxy", "--model", "/m", "--schedule", "/s",
         "--nxd-dir", "/n", "--snx-out-dir", "/x", "--lag-period-us", "1000"] + list(extra))


def test_nx_role_capture_args_and_refusals(tmp_path):
    a = _args()
    assert a.frame_capture_dir == ""  # default off
    nx_role.validate_args(a)
    a = _args(["--frame-capture-dir", str(tmp_path / "c"), "--frame-capture-every", "3"])
    nx_role.validate_args(a)
    cap = nx_role.make_capture(a, Plan())
    assert (cap.every, cap.per_tier, cap.max_bytes) == (3, 1000, 512 << 20)
    cap.close()
    with pytest.raises(ConfigError):  # the directory already holds frames
        nx_role.make_capture(a, Plan())
    for bad in (["--lanes", "2"], ["--consumer", "product"], ["--frame-capture-every", "0"],
                ["--frame-capture-per-tier", "0"], ["--frame-capture-max-mb", "0"]):
        with pytest.raises(ConfigError):
            nx_role.validate_args(_args(["--frame-capture-dir", "/c"] + bad))
    sa = _args(["--frame-capture-dir", "/c"])
    sa.mode = "standalone"
    with pytest.raises(ConfigError):
        nx_role.validate_args(sa)


# ---- frame_replay ---------------------------------------------------------------------------

def fake_launch(tmp, n_req=6, batches=(1, 1, 3, 4, 8, 9), protocol=None, take_fix=None,
                send_fix=None):
    """A capture dir with real HMAC frames plus the drain's (recvs, sends) and a fake worker's
    (takes, reqs) records in the production formats; returns the replay argv."""
    nxd, snx, capd = (os.path.join(str(tmp), x) for x in ("nxd", "snx", "cap"))
    os.makedirs(nxd)
    os.makedirs(snx)
    plan = Plan()
    cap = fc.FrameCapture(capd, plan)
    st = nx_driver.LaneStats(plan, nxd, "lane0", "raw", 0)
    q = FakeQueue(KEY_REQ, "proxy_request_queue")
    proxy = types.SimpleNamespace(request_queue=q, result_queue=types.SimpleNamespace(
        queue=FakeQueue(KEY_RES, "proxy_result_queue")))
    cap.set_queue("result", proxy.result_queue.queue, "proxy_result_queue")
    cap.set_queue("request", q, "proxy_request_queue")
    orig = FakeQueue.__dict__["_prepare_data"]
    fc.install_request_capture(cap, FakeQueue, q,
                               is_request=lambda o: hasattr(o, "prompt_token_ids"))
    reqs = RecordFile(os.path.join(snx, "snx.4242.reqs.rec"), REQ_REC, 1)
    for k in range(n_req):
        cid, L = 100 + k, (8 if k % 2 else 128)
        q._prepare_data(req(cid, L))
        st.sent(k, 0, time.monotonic_ns(), 1, (send_fix or {}).get(k, cid))
        reqs.add(k, cid, 0, 0, 0, 0, 0, 0, 1, 1, L, 0)
    q._prepare_data(None)  # the shutdown sentinel: counted, not captured
    frames = [sign(KEY_RES, [resp(i) for i in range(n)], protocol) for n in batches]
    takes = RecordFile(os.path.join(snx, "snx.4242.takes.rec"), TAKE_REC, 1)
    for i, n in enumerate(batches):
        takes.add(0, 0, 0, (take_fix or {}).get(i, n), 1)
    nx_driver.drain_loop(FakeZq(frames + [sign(KEY_RES, None)], KEY_RES), st, "raw", 0,
                         threading.Event(), cap=cap)
    st.summary()
    cap.close({"recvs": "nxd.lane0.recvs.rec", "sends": "nxd.lane0.sends.rec",
               "engine_pid": 4242, "prompt_len": 128})
    assert FakeQueue.__dict__["_prepare_data"] is orig
    return ["--capture-dir", capd, "--nxd-dir", nxd, "--snx-dir", snx,
            "--out", os.path.join(str(tmp), "replay.json"), "--impl", "local", "--reps", "2"]


def _doc(argv):
    return json.load(open(argv[argv.index("--out") + 1]))


def test_replay_end_to_end(tmp_path, capsys):
    argv = fake_launch(tmp_path)
    assert fr.main(argv) == fr.RC_OK
    doc = _doc(argv)
    assert doc["ok"] and doc["impl"] == "local" and doc["sampled_frames"] == {"result": 6,
                                                                              "request": 6}
    res, rq = doc["directions"]["result"], doc["directions"]["request"]
    assert res["reconcile"]["checked"] == 6 and res["reconcile"]["mismatches"] == 0
    assert res["tamper"] == {"frames": 6, "sig_rejected": 6, "payload_rejected": 6,
                             "failures": [], "ok": True}
    assert res["equivalence"]["unequal"] == 0 and res["equivalence"]["dumps_reported"]
    a = res["strata"]["all"]
    assert a["responses"] == 26 and a["frames"] == 6
    assert sorted(res["strata"]["by_responses_per_frame"]) == ["1", "2-3", "4-7", "8-15"]
    assert res["strata"]["by_responses_per_frame"]["8-15"]["responses"] == 17
    for op in fr.OPS:
        assert a[op]["cpu_us_per_response"] > 0 and a[op]["wall_us_per_response"] > 0
    assert sorted(rq["strata"]["by_prompt_len"]) == ["128", "8"]
    assert rq["strata"]["all"]["responses"] == 6  # one request per frame
    assert "result_dumps" in capsys.readouterr().out
    assert fr.main(argv) == fr.RC_INPUT  # --out is never overwritten


def test_replay_no_frames_reconcile_and_tamper_failures(tmp_path, monkeypatch):
    argv = fake_launch(tmp_path / "a", n_req=0)
    assert fr.main(argv) == fr.RC_NO_FRAMES
    doc = _doc(argv)
    assert not doc["ok"] and doc["sampled_frames"]["request"] == 0 and not doc["directions"]

    argv = fake_launch(tmp_path / "b", take_fix={2: 5})  # the worker took 5, the frame has 3
    assert fr.main(argv) == fr.RC_RECONCILE
    doc = _doc(argv)
    ex = doc["directions"]["result"]["reconcile"]["examples"]
    assert ex == [{"seq": 3, "item": "responses", "frame": 3, "expected": 5}]
    assert "strata" not in doc["directions"]["result"]

    argv = fake_launch(tmp_path / "c", send_fix={1: 999})
    assert fr.main(argv) == fr.RC_RECONCILE
    ex = _doc(argv)["directions"]["request"]["reconcile"]["examples"]
    assert ex[0]["item"] == "client_id" and ex[0]["expected"] == 999

    argv = fake_launch(tmp_path / "d")
    prep, _parse = fr.local_codec(KEY_RES)
    monkeypatch.setattr(fr, "make_codec", lambda impl, key: (
        prep, lambda data: pickle.loads(data[:-32]), "lax"))  # accepts any signature
    assert fr.main(argv) == fr.RC_TAMPER
    t = _doc(argv)["directions"]["result"]["tamper"]
    assert t["sig_rejected"] == 0 and not t["ok"]


def test_replay_unequal_dumps_not_reported(tmp_path):
    argv = fake_launch(tmp_path, protocol=2)  # re-dumped with the default protocol: unequal
    assert fr.main(argv) == fr.RC_OK
    doc = _doc(argv)
    res = doc["directions"]["result"]
    assert res["equivalence"]["unequal"] == 6 and not res["equivalence"]["dumps_reported"]
    assert res["strata"]["all"]["dumps"] is None and res["strata"]["all"]["loads"] is not None
    assert all(v["dumps"] is None for v in res["strata"]["by_tier"].values())
    assert doc["directions"]["request"]["strata"]["all"]["dumps"] is not None


def test_product_codec_matches_local(monkeypatch):
    class ZeroMqQueue:  # the product's four methods (tensorrt_llm/executor/ipc.py), verbatim
        def _verify_hmac(self, data, actual_hmac):
            expected_hmac = hmac.new(self.hmac_key, data, hashlib.sha256).digest()
            return hmac.compare_digest(expected_hmac, actual_hmac)

        def _sign_data(self, data_before_encoding):
            hmac_signature = hmac.new(self.hmac_key, data_before_encoding,
                                      hashlib.sha256).digest()
            return data_before_encoding + hmac_signature

        def _prepare_data(self, obj):
            data = pickle.dumps(obj)
            if self.use_hmac_encryption:
                return self._sign_data(data)
            return data

        def _parse_data(self, data):
            if self.use_hmac_encryption:
                message_data = data[:-32]
                actual_hmac = data[-32:]
                if not self._verify_hmac(message_data, actual_hmac):
                    raise RuntimeError("HMAC verification failed")
                return pickle.loads(message_data)
            return pickle.loads(data)

        def __del__(self):
            raise AssertionError("never closed: the replay object holds no socket")
    mod = types.ModuleType("tensorrt_llm.executor.ipc")
    mod.ZeroMqQueue = ZeroMqQueue
    for name, m in (("tensorrt_llm", types.ModuleType("tensorrt_llm")),
                    ("tensorrt_llm.executor", types.ModuleType("tensorrt_llm.executor")),
                    ("tensorrt_llm.executor.ipc", mod)):
        monkeypatch.setitem(sys.modules, name, m)
    p1, q1, impl = fr.make_codec("auto", KEY_RES)
    p2, q2 = fr.local_codec(KEY_RES)
    obj = [resp(1), resp(2)]
    assert impl == "product" and p1(obj) == p2(obj) == sign(KEY_RES, obj)
    assert pickle.dumps(q1(p2(obj))) == pickle.dumps(obj)
    with pytest.raises(RuntimeError):
        q1(fr.flip(p2(obj), len(p2(obj)) - 1))
    assert fr.tamper_check([(1, 0, 0, p2(obj))], q1)["ok"]
    assert fr.make_codec("local", KEY_RES)[2] == "local"


def test_rpf_bucket_and_refusals(tmp_path):
    assert [fr.rpf_bucket(n) for n in (0, 1, 2, 3, 4, 7, 8, 128, 255)] == \
        ["0", "1", "2-3", "2-3", "4-7", "4-7", "8-15", "128-255", "128-255"]
    assert fr.main(["--capture-dir", str(tmp_path), "--nxd-dir", "x", "--snx-dir", "y",
                    "--out", str(tmp_path / "o.json")]) == fr.RC_INPUT  # no capture there
    assert fr.main(["--capture-dir", "a", "--nxd-dir", "x", "--snx-dir", "y", "--out", "o",
                    "--reps", "0"]) == fr.RC_INPUT
