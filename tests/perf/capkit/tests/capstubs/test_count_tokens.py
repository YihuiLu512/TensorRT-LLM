# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capgrpc.load --count-tokens (D1 stage S3, merged token events): capgrpc.tokcount against the
product's generated generation_pb2 (probe tree), MsgPlan.msg_of_token, and LoadWorker.one on
hand-made streams with a fixed clock: for k = 1 every record, counter and the lag of
--planned-msgs are identical with and without counting (only the token column and the new
summary block are added); for merged streams the tokens reconcile, the merge-factor histogram is
exact and each message is planned at its first token's step (negative control: without counting
the same stream is planned by message index)."""

import asyncio
import types

import grpc
import pytest

from capcommon import expect, hist as _hist
from capcommon.capbin import RecordArray
from capcommon.tiers import TierTable

from conftest import MS, real_pb2, table_doc

BEAT = 10 * MS


def _tok_msg(pb2, ids, cand=False):
    infos = [pb2.TokenInfo(token_id=x, token="t%d" % x, logprob=-0.5, rank=1) for x in ids]
    if cand:
        for ti in infos:
            ti.candidates.add(token_id=7, logprob=-1.0, token="c")
    return pb2.GenerateResponse(request_id="r1", token=pb2.TokenOutput(
        output_index=0, tokens=infos, text="x" * len(ids))).SerializeToString()


def _fin(pb2, n):
    return pb2.GenerateResponse(
        request_id="r1", finished=pb2.GenerationFinished(output_index=0, reason=1),
        usage=pb2.Usage(prompt_tokens=4, completion_tokens=n, total_tokens=n + 4)
    ).SerializeToString()


def test_response_tokens_on_real_messages():
    from capgrpc.tokcount import KHist, response_tokens
    pb2, _oe = real_pb2()
    for k in range(0, 6):
        assert response_tokens(_tok_msg(pb2, list(range(k)))) == (k, True)
        assert response_tokens(_tok_msg(pb2, list(range(k)), cand=True)) == (k, True)
    assert response_tokens(_fin(pb2, 5)) == (0, False)
    assert response_tokens(pb2.GenerateResponse(prompt=pb2.PromptOutput()).SerializeToString()) \
        == (0, False)
    m = _tok_msg(pb2, [1, 2, 3])
    for bad in (m[:-1], m[:5], m + b"\x0b", b"\x1a\x7f"):
        with pytest.raises(ValueError):
            response_tokens(bad)
    h = KHist()
    for k in (1, 2, 2, 3):
        h.add(k)
    assert h.summary() == {"token_msgs": 4, "tokens": 8, "k_mean": 2.0, "k_max": 3,
                           "k_hist": {"1": 1, "2": 2, "3": 1}}


def _tt(n=5, k=1):
    return TierTable.from_doc(table_doc([("warmup", 400, 10.0, {"n": n, "k": k, "beat": BEAT}),
                                         ("drain", 300, 0.0, {"beat": BEAT})]))


def test_msg_of_token_identity_for_k1_and_first_token_for_merged():
    from capgrpc.msgplan import MsgPlan
    mp = MsgPlan(_tt(5))
    n_msgs = expect.msgs_per_request(5, 1, 1, "product")
    assert [mp.msg_of_token(0, i) for i in range(n_msgs + 1)] == list(range(n_msgs + 1))
    # merged messages carrying tokens [1, 2, 2]: first tokens 0, 1, 3; finished at 5
    assert [mp.msg_of_token(0, i) for i in (0, 1, 3, 5)] == [0, 1, 3, 5]
    mp2 = MsgPlan(_tt(6, 2))  # the tier itself plans 2 tokens per message
    assert [mp2.msg_of_token(0, i) for i in (0, 2, 4, 6)] == [0, 1, 2, 3]


class FakeCall:
    def __init__(self, msgs):
        self.msgs = list(msgs)

    async def read(self):
        return self.msgs.pop(0) if self.msgs else grpc.aio.EOF

    async def code(self):
        return grpc.StatusCode.OK

    def cancel(self):
        pass


def _worker(monkeypatch, tt, msgs, count_tokens, planned_msgs=True):
    """A LoadWorker without process or channel: one request (rid 0, tier 0, planned at 15 ms),
    a fixed clock that advances 1 ms per call."""
    from capgrpc import loader
    from capgrpc.common import TierCursor
    from capgrpc.msgplan import MsgPlan
    from capgrpc.tokcount import KHist
    clock = [10**12]

    def now():
        clock[0] += MS
        return clock[0]
    monkeypatch.setattr(loader, "now_ns", now)
    w = object.__new__(loader.LoadWorker)
    nt = len(tt)
    nop = types.SimpleNamespace(hit=lambda: None, should_drop=lambda: False)
    w.plan = types.SimpleNamespace(req_ids=[0], tseq=[0], planned=[15 * MS],
                                   build=lambda rid, seq: b"req", expected_msgs=[None] * nt)
    w.tt, w.origin, w.timeout_ns, w.nch = tt, 10**12, None, 1
    w.calls = [lambda req, timeout=None: FakeCall(msgs)]
    w.v = [0] * 8
    (w.i_sent, w.i_inflight, w.i_active, w.i_read, w.i_ok, w.i_failed, w.i_bytes) = range(7)
    lags = []
    w.h_send = types.SimpleNamespace(add=lambda x: None)
    w.h_msg = types.SimpleNamespace(add=lags.append)
    w.inj_send = w.inj_recv = w.inj_drop = nop
    w.send_hist = [_hist.new_counts() for _ in range(nt)]
    w.send_max = [None] * nt
    for name in ("sent_by_tier", "ok_by_tier", "failed_by_tier", "msgs_by_tier", "bytes_by_tier",
                 "mismatch_by_tier", "unplanned_by_tier"):
        setattr(w, name, [0] * nt)
    w.fail_reasons = [dict() for _ in range(nt)]
    w.fail_zero = [dict() for _ in range(nt)]
    w.fail_max_ns = [dict() for _ in range(nt)]
    w.recv_types = [dict() for _ in range(nt)]
    w.status_ids = {}
    w.stamped, w.mod = False, 1
    w.msgplan = MsgPlan(tt) if planned_msgs else None
    w.count_tokens = count_tokens
    if count_tokens:
        w.khist_by_tier = [KHist() for _ in range(nt)]
        w.tokens_ok_by_tier = [0] * nt
        w.tok_mismatch_by_tier = [0] * nt
    w.req_rec = RecordArray(loader.REQ_FIELDS, 10)
    w.msg_rec = RecordArray(loader.MSG_FIELDS, 100)
    w.cursor = TierCursor(tt)
    asyncio.run(w.one(0))
    cols = lambda ra: {k: list(v[:ra.n]) for k, v in ra.cols.items()}  # noqa: E731
    return w, cols(w.req_rec), cols(w.msg_rec), lags


def test_k1_stream_identical_with_and_without_counting(monkeypatch):
    pb2, _oe = real_pb2()
    tt = _tt(5)
    msgs = [_tok_msg(pb2, [i]) for i in range(5)] + [_fin(pb2, 5)]
    w0, r0, m0, l0 = _worker(monkeypatch, tt, msgs, False)
    w1, r1, m1, l1 = _worker(monkeypatch, tt, msgs, True)
    assert m0 == m1 and l0 == l1 and len(m0["idx"]) == 6
    assert r0["tokens"] == [-1] and r1["tokens"] == [5]
    r0.pop("tokens")
    r1.pop("tokens")
    assert r0 == r1
    for name in ("ok_by_tier", "failed_by_tier", "msgs_by_tier", "bytes_by_tier",
                 "unplanned_by_tier", "recv_types", "fail_reasons"):
        assert getattr(w0, name) == getattr(w1, name), name
    assert w1.khist_by_tier[0].summary()["k_hist"] == {"1": 5}
    assert w1.tokens_ok_by_tier[0] == 5 and w1.tok_mismatch_by_tier[0] == 0


def test_merged_stream_reconciles_by_tokens_and_plans_by_first_token(monkeypatch):
    pb2, _oe = real_pb2()
    tt = _tt(5)
    msgs = [_tok_msg(pb2, [0]), _tok_msg(pb2, [1, 2]), _tok_msg(pb2, [3, 4]), _fin(pb2, 5)]
    w, r, m, lags = _worker(monkeypatch, tt, msgs, True)
    assert r["tokens"] == [5] and r["frames"] == [4] and r["status"] == [0]
    assert w.khist_by_tier[0].summary() == {"token_msgs": 3, "tokens": 5, "k_mean": 1.6667,
                                            "k_max": 2, "k_hist": {"1": 1, "2": 2}}
    assert w.tokens_ok_by_tier[0] == 5 and w.tok_mismatch_by_tier[0] == 0
    assert m["idx"] == [0, 1, 3, 5]  # first token of each message; finished at its own index
    offs = expect.msg_step_offsets(5, 1, 1, "product")
    first = 2  # 15 ms rounds up to step 2
    assert m["step"] == [first + offs[0], first + offs[1], first + offs[3], first + offs[-1]]
    # negative control: without counting the same stream is planned by message index
    _w0, _r0, m0, _l0 = _worker(monkeypatch, tt, msgs, False)
    assert m0["idx"] == [0, 1, 2, 3] and m0["step"] != m["step"]
    # a short merged stream is an OK request with an unexpected token count
    w2, r2, _m2, _l2 = _worker(monkeypatch, tt, [_tok_msg(pb2, [0, 1, 2]), _fin(pb2, 3)], True)
    assert r2["tokens"] == [3] and w2.tok_mismatch_by_tier[0] == 1


def test_malformed_response_is_a_protocol_error_and_no_token_count(monkeypatch):
    pb2, _oe = real_pb2()
    w, r, _m, _l = _worker(monkeypatch, _tt(5), [_tok_msg(pb2, [0])[:-1]], True)
    assert r["tokens"] == [-1] and r["status"] != [0]
    assert w.fail_reasons[0] == {"protocol_error": 1}


def test_count_tokens_option_checks():
    from capgrpc import loader
    assert "--count-tokens" in loader.build_parser().format_help()

    def args(fmt, **kw):
        d = dict(workers=1, in_process=False, channels_per_worker=1, msg_sample_mod=1,
                 max_records_per_worker=10, lag_records_capacity=0, target="127.0.0.1:1",
                 target_port_file=None, method="/a.B/C", request_format=fmt,
                 prompts="p" if fmt == "capkit" else None,
                 request_bytes=None if fmt == "capkit" else "b", planned_msgs=False,
                 stamped=False, consume="raw", count_tokens=True)
        d.update(kw)
        return types.SimpleNamespace(**d)
    common = types.SimpleNamespace(cpus=[0])
    for bad in (args("capkit"), args("openengine", consume="client"),
                args("openengine", stamped=True)):
        with pytest.raises(loader.ConfigError):
            loader.validate_args(bad, common)
    loader.validate_args(args("openengine"), common)  # positive control
    loader.validate_args(args("capkit", count_tokens=False), common)
