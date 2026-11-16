# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX NullExecutor with a fake product adapter (no tensorrt_llm): the product gate per request,
k / MTP mixture, absolute grid and idle restart, cancel, prefill budget / first-response delay,
max_active, busy loop, self-drain + arrival hook, records and the offline summary, config."""

import json
import os
import threading
import time

import pytest

from capcommon import expect
from capcommon.exitcodes import ConfigError
from capcommon.tiers import TierTable

from capstubs import null_executor as nx
from capstubs.null_executor import (ENV_VAR, NullExecutor, SnxConfig, StepClock, StepTimes,
                                    read_records, summarize, tier_shape)

MS = 1_000_000


class FakeLlm:
    def __init__(self, req_id, ereq):
        self.req_id = req_id
        self.client_id = ereq["client_id"]
        self.max_new = ereq["max_tokens"]
        self.prompt = ereq.get("prompt", [7, 8, 9])
        self.tokens = []
        self.iters = []
        self.sent = 0
        self.reason = None


class FakeAdapter:
    name = "fake"

    def __init__(self):
        self.lock = threading.Lock()
        self.converted = []

    def convert(self, req_id, ereq):
        self.converted.append((req_id, threading.current_thread().name))
        return FakeLlm(req_id, ereq)

    def info(self, llm, want_prompt):
        return llm.client_id, llm.max_new, len(llm.prompt), list(llm.prompt) if want_prompt \
            else None

    def add_tokens(self, llm, ids, it):
        llm.tokens += ids
        llm.iters.append(it)

    def finish(self, llm, cancelled):
        llm.reason = "cancelled" if cancelled else "length"

    def create_response(self, llm):
        new = llm.tokens[llm.sent:]
        llm.sent = len(llm.tokens)
        return {"client_id": llm.client_id, "tokens": new, "final": llm.reason is not None,
                "reason": llm.reason}


def cfg(**kw):
    kw.setdefault("beat_ns", 2 * MS)
    return SnxConfig(**kw).validate()


def collect(ex, n_final, timeout_s=10.0):
    """Await responses until n_final finals arrived: {client_id: [resp...]}."""
    out = {}
    finals = 0
    end = time.monotonic() + timeout_s
    while finals < n_final and time.monotonic() < end:
        for r in ex.await_responses(timeout=0.2):
            out.setdefault(r["client_id"], []).append(r)
            finals += r["final"]
    assert finals == n_final, out
    return out


@pytest.fixture
def run():
    made = []

    def make(c, **kw):
        ex = NullExecutor(c, FakeAdapter(), **kw)
        made.append(ex)
        return ex
    yield make
    for ex in made:
        ex.shutdown()


@pytest.mark.parametrize("n,k,s", [(10, 1, 1), (10, 1, 4), (33, 1, 20), (10, 3, 1), (7, 2, 3)])
def test_product_gate_per_request(run, n, k, s):
    ex = run(cfg(tokens_per_step=k, stream_interval=s))
    ex.enqueue_request({"client_id": 5, "max_tokens": n})
    got = collect(ex, 1)[5]
    assert sum(len(r["tokens"]) for r in got) == n
    assert got[-1]["final"] and got[-1]["reason"] == "length"
    assert len(got) == expect.msgs_per_request(n, k * s, s, "product")
    assert [len(r["tokens"]) for r in got] == expect.msg_tokens(n, k * s, s, "product")


def test_mtp_mixture_first_one_then_3_or_4(run):
    ex = run(cfg(first_step_tokens=1, k_mix=[[3, 0.6], [4, 0.4]], k_mix_seed=7))
    for cid in range(6):
        ex.enqueue_request({"client_id": cid, "max_tokens": 33})
    got = collect(ex, 6)
    sizes = set()
    for cid, rs in got.items():
        toks = [len(r["tokens"]) for r in rs]
        assert sum(toks) == 33 and toks[0] == 1
        sizes |= set(toks[1:-1])
        assert 9 <= len(rs) <= 12  # 12.7: 33 tokens at s = 1 are 9..12 chunks
    assert sizes == {3, 4}


def test_conversion_on_step_thread_and_ids(run):
    ex = run(cfg())
    ids = [ex.enqueue_request({"client_id": c, "max_tokens": 2}) for c in range(3)]
    assert ids == [1, 2, 3]
    collect(ex, 3)
    assert {t for _i, t in ex.adapter.converted} == {"snx_step"}


def test_grid_wake_lag_and_idle_block(run, tmp_path):
    ex = run(cfg(beat_ns=5 * MS, out_dir=str(tmp_path)))
    ex.enqueue_request({"client_id": 1, "max_tokens": 4})
    collect(ex, 1)
    n = ex.counts["steps"]
    restarts = ex.counts["idle_restarts"]
    time.sleep(0.1)
    assert ex.counts["steps"] == n  # idle: blocked, no empty steps
    t_arrive = time.monotonic_ns()
    ex.enqueue_request({"client_id": 2, "max_tokens": 3})
    collect(ex, 1)
    ex.shutdown()
    steps = read_records(os.path.join(str(tmp_path), "snx.%d.steps.bin" % os.getpid()), "steps")
    anchor = steps[0]["planned_ns"]
    for r in steps:
        assert (r["planned_ns"] - anchor) % (5 * MS) == 0  # one absolute grid
        assert r["wake_lag_ns"] == r["start_ns"] - r["planned_ns"] >= 0
    restart = [r for r in steps if r["taken"] and r["planned_ns"] >= t_arrive]
    assert restart and restart[0]["planned_ns"] - t_arrive < 5 * MS
    assert ex.counts["idle_restarts"] == restarts + 1


def test_idle_restart_immediate_is_off_grid(run):
    ex = run(cfg(beat_ns=50 * MS, idle_restart="immediate"))
    t = time.monotonic_ns()
    ex.enqueue_request({"client_id": 1, "max_tokens": 1})
    collect(ex, 1)
    assert ex.max_wake_lag_ns < 20 * MS and time.monotonic_ns() - t < 40 * MS


def test_cancel_emits_a_cancelled_final(run, tmp_path):
    ex = run(cfg(beat_ns=5 * MS, out_dir=str(tmp_path)))
    rid = ex.enqueue_request({"client_id": 9, "max_tokens": 1000})
    first = []
    while not first:
        first = ex.await_responses(timeout=0.5)
    ex.cancel_request(rid)
    got = collect(ex, 1)[9]
    assert got[-1]["reason"] == "cancelled" and ex.counts["cancelled"] == 1
    ex.shutdown()
    reqs = read_records(os.path.join(str(tmp_path), "snx.%d.reqs.bin" % os.getpid()), "reqs")
    assert reqs[-1]["cancelled"] == 1 and reqs[-1]["tokens"] < 1000


def test_prefill_budget_fcfs_chunks_and_delay(run, tmp_path):
    ex = run(cfg(beat_ns=3 * MS, prefill_token_budget=4, first_response_delay_steps=1,
                 out_dir=str(tmp_path)), start=False)
    ex.enqueue_request({"client_id": 1, "max_tokens": 1, "prompt": list(range(10))})
    ex.enqueue_request({"client_id": 2, "max_tokens": 1, "prompt": [1, 2, 3]})
    ex.start()
    collect(ex, 2)
    ex.shutdown()
    reqs = {r["client_id"]: r for r in
            read_records(os.path.join(str(tmp_path), "snx.%d.reqs.bin" % os.getpid()), "reqs")}
    steps = read_records(os.path.join(str(tmp_path), "snx.%d.steps.bin" % os.getpid()), "steps")
    t0 = reqs[1]["take_step"]
    assert [s["prefill_tokens"] for s in steps[:4]] == [4, 4, 4, 1]
    # r1 prefill done in round t0+2 (4+4+2), first response one round later; r2 done in t0+3
    assert reqs[1]["first_emit_step"] == t0 + 3
    assert reqs[2]["first_emit_step"] == t0 + 4


def test_max_active_cap(run, tmp_path):
    ex = run(cfg(beat_ns=2 * MS, max_active=2, out_dir=str(tmp_path)), start=False)
    for c in range(5):
        ex.enqueue_request({"client_id": c, "max_tokens": 3})
    ex.start()
    collect(ex, 5)
    ex.shutdown()
    steps = read_records(os.path.join(str(tmp_path), "snx.%d.steps.bin" % os.getpid()), "steps")
    assert max(s["active"] for s in steps) <= 2 and steps[0]["queued_after"] == 3


def test_busy_loop_charges_step_cpu(run, tmp_path):
    ex = run(cfg(beat_ns=5 * MS, busy_ns_per_step=2 * MS, out_dir=str(tmp_path)))
    ex.enqueue_request({"client_id": 1, "max_tokens": 5})
    collect(ex, 1)
    ex.shutdown()
    steps = read_records(os.path.join(str(tmp_path), "snx.%d.steps.bin" % os.getpid()), "steps")
    assert all(s["busy_iters"] > 0 and s["wall_ns"] >= 2 * MS for s in steps)


def test_self_drain_and_arrival_hook(run):
    seen = []

    def hook(now):
        if not seen:
            seen.append(now)
            ex.enqueue_request({"client_id": 3, "max_tokens": 4})
    ex = NullExecutor(cfg(beat_ns=2 * MS), FakeAdapter(), self_drain=True, arrival_hook=hook)
    try:
        end = time.monotonic() + 5
        while ex.counts["finished"] < 1 and time.monotonic() < end:
            time.sleep(0.01)
        assert ex.counts["self_drained"] == 4 and ex.counts["finished"] == 1
        assert ex.await_responses(timeout=0) == []
    finally:
        ex.shutdown()


def test_take_lag_and_summary(run, tmp_path):
    ex = run(cfg(beat_ns=2 * MS, out_dir=str(tmp_path)))
    for c in range(3):
        ex.enqueue_request({"client_id": c, "max_tokens": 5})
    time.sleep(0.05)  # responses pile up: the first take is late
    collect(ex, 3)
    ex.shutdown()
    takes = read_records(os.path.join(str(tmp_path), "snx.%d.takes.bin" % os.getpid()), "takes")
    assert takes[0]["return_ns"] - takes[0]["oldest_planned_ns"] >= 30 * MS
    assert sum(t["responses"] for t in takes) == 15
    doc = summarize(str(tmp_path))
    w = doc["processes"][str(os.getpid())]["whole"]
    assert w["released"] == 15 and w["take_responses"] == 15 and w["reqs_finished"] == 3
    summ = doc["processes"][str(os.getpid())]["summary"]
    assert summ["counts"]["tokens"] == 15 and summ["counts"]["steps"] == w["steps"]
    with open(os.path.join(str(tmp_path), "snx.%d.meta.json" % os.getpid())) as f:
        meta = json.load(f)
    assert meta["threads"]["snx_step"] > 0 and meta["records"]["steps"]["size"] == \
        nx.STEP_REC.size


def _table(tmp_path, tiers):
    from conftest import g  # noqa: PLC0415 -- tests/grpc helpers via the capstubs conftest
    spec = [(kind, win, rate, kw) for kind, win, rate, kw in tiers]
    doc = g.table_doc(spec)
    p = os.path.join(str(tmp_path), "t.json")
    with open(p, "w") as f:
        json.dump(doc, f)
    return TierTable.load(p), p


def test_stepclock_tier_grid(tmp_path):
    tt, _p = _table(tmp_path, [("warmup", 100, 10.0, {"beat": 10 * MS}),
                               ("drain", 40, 0.0, {"beat": 10 * MS}),
                               ("up", 40, 10.0, {"beat": 4 * MS}),
                               ("drain", 40, 0.0, {"beat": 4 * MS})])
    org = 10 ** 12
    c = StepClock(1, 0, tt, org, 0)
    assert c.at_or_after(org - 25 * MS) == (org - 20 * MS, nx.PRE_SEQ, 10 * MS)
    assert c.at_or_after(org + 1) == (org + 10 * MS, 0, 10 * MS)
    assert c.at_or_after(org + 101 * MS) == (org + 110 * MS, 1, 10 * MS)
    assert c.at_or_after(org + 141 * MS) == (org + 144 * MS, 2, 4 * MS)
    assert c.at_or_after(org + 221 * MS) == (org + 224 * MS, nx.POST_SEQ, 4 * MS)
    assert c.next_after(org + 96 * MS)[0] == org + 100 * MS


def test_tier_shape_and_stream_interval_cross_check(tmp_path):
    tt, p = _table(tmp_path, [("warmup", 50, 10.0, {"k": 2, "s": 2}),
                              ("drain", 50, 0.0, {"k": 2, "s": 2}),
                              ("up", 50, 10.0, {"k": 6, "s": 2}),
                              ("drain", 50, 0.0, {"k": 2, "s": 2})])
    ks, s = tier_shape(tt, cfg())
    assert ks == {0: 1, 1: 1, 2: 3, 3: 1} and s == 2
    with pytest.raises(ConfigError):
        tier_shape(tt, cfg(stream_interval=1))
    with pytest.raises(ConfigError):
        NullExecutor(cfg(tier_table=p, origin_ns=1), FakeAdapter(), tier_table=tt,
                     llm_stream_interval=1, start=False)
    tt2, _ = _table(tmp_path, [("warmup", 50, 10.0, {"s": 1}), ("drain", 50, 0.0, {}),
                               ("up", 50, 10.0, {"k": 2, "s": 2}),
                               ("drain", 50, 0.0, {"k": 2, "s": 2})])
    with pytest.raises(ConfigError):
        tier_shape(tt2, cfg())


def test_tier_table_run_uses_tier_k(run, tmp_path):
    tt, p = _table(tmp_path, [("warmup", 2000, 10.0, {"k": 3, "beat": 2 * MS}),
                              ("drain", 100, 0.0, {"k": 3, "beat": 2 * MS})])
    ex = run(cfg(tier_table=p, origin_ns=time.monotonic_ns(), out_dir=str(tmp_path)),
             tier_table=tt)
    ex.enqueue_request({"client_id": 1, "max_tokens": 9})
    got = collect(ex, 1)[1]
    assert [len(r["tokens"]) for r in got] == [3, 3, 3]
    ex.shutdown()
    doc = summarize(str(tmp_path), p, ex.cfg.origin_ns)
    tiers = doc["processes"][str(os.getpid())]["tiers"]
    assert tiers[0]["tokens_per_step"] == 3 and tiers[0]["released"] == 3


def test_step_times_replay():
    st = StepTimes({"decode_ns": 12, "table": [[100, 50], [1000, 90]]}, 0.25)
    assert st.round_ns(0) == 12
    assert st.round_ns(80) == 12 + int(0.75 * 38)
    assert st.round_ns(0) == 12 + int(0.25 * 38)
    assert st.round_ns(5000) == 12 + int(0.75 * 78)
    with pytest.raises(ConfigError):
        StepTimes({"decode_ns": 12, "table": [[1, 5]]}, 0)


def test_config_doc_env_and_refusals():
    c = cfg(tokens_per_step=2)
    assert SnxConfig.from_doc(c.to_doc()) == c
    assert SnxConfig.from_env({ENV_VAR: json.dumps(c.to_doc())}) == c
    with pytest.raises(ConfigError):
        SnxConfig.from_env({})
    with pytest.raises(ConfigError):
        SnxConfig.from_doc(dict(c.to_doc(), bogus=1))
    for bad in ({"beat_ns": 0}, {"token_source": "x"}, {"k_mix": [[0, 1]]},
                {"idle_restart": "never"}, {"tier_table": "t.json"},
                {"prefill_spill_frac": 2.0}, {"token_ids": []}):
        with pytest.raises(ConfigError):
            SnxConfig(**bad).validate()


def test_shutdown_unblocks_awaiter_and_refuses(run):
    ex = run(cfg())
    got = []
    t = threading.Thread(target=lambda: got.append(ex.await_responses(timeout=5)))
    t.start()
    time.sleep(0.05)
    ex.shutdown()
    t.join(2)
    assert got == [[]] and not ex.can_enqueue_requests()
    with pytest.raises(RuntimeError):
        ex.enqueue_request({"client_id": 1, "max_tokens": 1})


def test_step_thread_error_is_reported_as_event_loop_error(tmp_path):
    class Bad(FakeAdapter):
        def create_response(self, llm):
            raise ValueError("boom")
    ex = NullExecutor(cfg(out_dir=str(tmp_path)), Bad())
    ex.enqueue_request({"client_id": 1, "max_tokens": 2})
    assert ex.shutdown_event.wait(5)
    assert isinstance(ex._event_loop_error, ValueError) and ex.is_shutdown
    assert ex.await_responses(timeout=1) == []
    with open(os.path.join(str(tmp_path), "snx.%d.summary.json" % os.getpid())) as f:
        assert json.load(f)["counts"]["step_thread_errors"] == 1
