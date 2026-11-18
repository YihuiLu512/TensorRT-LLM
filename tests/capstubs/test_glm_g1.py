# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.glm_g1.G1Engine (E10 G1 substitute): the engine_api contract as the glm server uses
it (one response per active request per step, finish on max_tokens, abort, get_responses
batching and timeout), the absolute step grid and the self-report."""

import threading
import time

import pytest

from capstubs import glm_g1


def make(step_ms=2.0, k=1, pattern=(1, 2, 3), **kw):
    cfg = {"mock_step_ms": step_ms, "mock_tokens_per_step": k, "mock_pattern": list(pattern)}
    cfg.update(kw)
    return glm_g1.G1Engine(cfg)


def drain(e, until, timeout_s=5.0):
    """get_responses until `until(all responses)` is true."""
    out, end = [], time.monotonic() + timeout_s
    while time.monotonic() < end:
        out += e.get_responses(50)
        if until(out):
            return out
    raise AssertionError("timed out; got %d responses" % len(out))


def test_step_semantics_without_thread():
    e = make(k=2, pattern=(7, 8, 9))
    e.add_request(1, [0] * 4, {"max_tokens": 5})
    e.add_request(2, [0], {"max_tokens": 1})
    r1 = e._step()
    assert [(r["request_id"], r["token_ids"], r["finished"]) for r in r1] == \
        [(1, [7, 8], False), (2, [7], True)]
    assert r1[1]["finish_reason"] == "length" and r1[1]["num_generated_tokens"] == 1
    r2 = e._step()
    assert [(r["token_ids"], r["finished"], r["num_generated_tokens"]) for r in r2] == \
        [([9, 7], False, 4)]
    r3 = e._step()
    assert [(r["token_ids"], r["finished"], r["num_generated_tokens"]) for r in r3] == \
        [([8], True, 5)]
    assert r3[0]["num_prompt_tokens"] == 4 and r3[0]["logprobs"] is None
    assert e._step() == [] and e.st.finished == 2


def test_abort_known_and_unknown():
    e = make()
    e.add_request(5, [0], {"max_tokens": 10})
    e._step()
    e.abort_request(5)
    e.abort_request(99)  # never added: no response (MockEngine: only known requests)
    out = e._step()
    assert [(r["request_id"], r["finished"], r["finish_reason"]) for r in out] == \
        [(5, True, "abort")]
    assert e._step() == []


def test_add_request_refusals():
    e = make()
    with pytest.raises(ValueError):
        e.add_request(1, [0], {})
    with pytest.raises(ValueError):
        e.add_request(1, [0], {"max_tokens": 3, "logprobs": 2})
    with pytest.raises(ValueError):
        e.add_request(1, [0] * 9000, {"max_tokens": 3})
    with pytest.raises(ValueError):
        make(step_ms=0)
    assert e._new == []


def test_info_matches_mock_shape():
    info = make(max_seq_len=4096).info()
    assert info["max_seq_len"] == 4096 and info["vocab_size"] == glm_g1.MOCK_VOCAB
    assert info["eos_token_ids"] == list(glm_g1.DEFAULT_EOS)
    assert make().stats()["num_active_requests"] == 0


def test_threaded_streams_and_grid():
    e = make(step_ms=2.0)
    e.start()
    try:
        for rid in range(1, 9):
            e.add_request(rid, [0], {"max_tokens": 10})
        out = drain(e, lambda o: sum(r["finished"] for r in o) == 8)
        assert len(out) == 80
        per = {}
        for r in out:
            per.setdefault(r["request_id"], []).append(r["num_generated_tokens"])
        assert all(v == list(range(1, 11)) for v in per.values())
        time.sleep(0.05)
        snap = e.stats_snapshot()
        assert snap["kind"] == "g1" and snap["responses"] == 80 and snap["finished"] == 8
        assert snap["steps"] >= 10 and snap["busy_steps"] >= 10
        assert sum(snap["lag_hist"]) == snap["steps"]
        assert snap["wake_lag_ns"] >= 0 and snap["cpu_ns"] > 0
        # the grid does not drift: steps done ~= elapsed / step (allow scheduling noise)
        elapsed = time.monotonic_ns() - snap["t0_ns"]
        assert snap["steps"] >= elapsed // e.step_ns - 5
    finally:
        e.shutdown()
    assert not e._thread.is_alive()


def test_get_responses_times_out_empty():
    e = make()
    t0 = time.monotonic()
    assert e.get_responses(30) == []
    assert time.monotonic() - t0 >= 0.025


def test_one_notify_per_step_batches():
    e = make(step_ms=50.0)
    e.start()
    try:
        for rid in range(1, 21):
            e.add_request(rid, [0], {"max_tokens": 1})
        out = []
        end = time.monotonic() + 3
        batches = []
        while len(out) < 20 and time.monotonic() < end:
            b = e.get_responses(200)
            if b:
                batches.append(len(b))
                out += b
        assert len(out) == 20 and batches == [20]
    finally:
        e.shutdown()


def test_late_step_counts_behind():
    e = make(step_ms=1.0)
    real = e._step
    hold = threading.Event()

    def slow():
        if not hold.is_set():
            hold.set()
            time.sleep(0.02)
        return real()
    e._step = slow
    e.start()
    try:
        time.sleep(0.1)
        snap = e.stats_snapshot()
        assert snap["max_behind"] >= 10
        assert snap["max_wake_lag_ns"] >= 10 * e.step_ns
    finally:
        e.shutdown()
