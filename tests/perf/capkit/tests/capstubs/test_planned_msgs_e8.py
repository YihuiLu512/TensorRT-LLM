# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capgrpc.load --planned-msgs in the shape of an E8 reader (PLAN-R2R5 3.1 B5): several worker
processes under the supervisor (not --in-process), --request-format openengine, a message sample
modulus coprime with the worker count, against S-PB on a beat (the synthetic stream). Every worker
writes drain.msgs rows, each derived from the schedule as capgrpc.msgplan says; the lag values of
LoadWorker.planned_msg on hand-made streams are exact, and a stream delayed by 50 ms carries the
50 ms in every message's lag."""

import os
import types

import pytest

from capcommon import expect
from capcommon.capbin import RecordArray
from capcommon.lifecycle import read_port_file
from capcommon.schedule import BeatSchedule
from capcommon.tiers import TierTable

from conftest import (METHOD, MS, real_pb2, spb_args, table_doc, total, write_prebaked,
                      write_reqbytes)

N, BEAT = 5, 10 * MS
WORKERS, MOD = 2, 3  # gcd(MOD, WORKERS) = 1: every worker owns sampled streams


def _spec():
    kw = {"n": N, "beat": BEAT}
    return [("warmup", 400, 20.0, kw), ("drain", 300, 0.0, {"beat": BEAT}),
            ("up", 600, 60.0, kw), ("drain", 500, 0.0, {"beat": BEAT})]


def _cpus(n):
    have = sorted(os.sched_getaffinity(0))
    return ",".join(str(c) for c in have[:max(n, 1)])


def test_supervised_workers_all_write_planned_msgs(launch):
    launch.write_table(table_doc(_spec()))
    gen, _oe = real_pb2()
    pb_path, _doc = write_prebaked(launch, pb2=gen)
    rb = write_reqbytes(launch)
    launch.set_origin(5.0)
    launch.server("spb", "capstubs.prebaked_server",
                  spb_args(pb_path, "--schedule", launch.sched_path), cpus="0")
    launch.wait_file(launch.tree.ready_file("spb"), 30, "spb")
    port = read_port_file(launch.tree.port_file("spb"))[1]
    assert port > 0
    argv = launch.common("reader", cpus=_cpus(WORKERS)) + [
        "--target-port-file", launch.tree.port_file("spb"), "--method", METHOD,
        "--workers", str(WORKERS), "--channels-per-worker", "1",
        "--schedule", launch.sched_path, "--request-format", "openengine",
        "--request-bytes", rb, "--msg-sample-mod", str(MOD), "--max-records-per-worker", "1000",
        "--planned-msgs", "--max-msg-records-per-worker", "2000"]
    launch.start("reader", "capgrpc.load", argv)
    assert launch.wait_exit("reader", 60) == 0, launch.stderr("reader")
    assert launch.stop("spb") == 0, launch.stderr("spb")

    summ = launch.summaries("reader")
    assert all(len(docs) == WORKERS for docs in summ.values())
    assert total(summ, "counts.failed") == 0 and total(summ, "counts.not_sent") == 0
    assert total(summ, "counts.ok") == total(summ, "counts.planned") > 0
    assert total(summ, "msg_plan.unplanned") == 0
    msgs, heads = launch.records("reader", "msgs")
    assert heads and all(h["meta"]["planned_source"] == "schedule" for h in heads)
    assert all(h["meta"]["overflow"] == 0 for h in heads)
    assert {h["meta"]["msg_sample_mod"] for h in heads} == {MOD}
    workers = {h["identity"]["worker"] for h in heads}
    assert workers == set(range(WORKERS)), workers

    reqs, _ = launch.records("reader", "requests")
    planned = dict(zip(reqs["req_id"], reqs["planned_rel_ns"]))
    sampled = [r for r in reqs["req_id"] if r % MOD == 0]
    per_req = expect.msgs_per_request(N, 1, 1, "product") + 1
    assert len(msgs["idx"]) == len(sampled) * per_req
    bs = BeatSchedule(launch.tt)
    offs = expect.msg_step_offsets(N, 1, 1, "product")
    offs = offs + [offs[-1]]
    for rid, j, step, p, r in zip(msgs["stream_id"], msgs["idx"], msgs["step"],
                                  msgs["planned_rel_ns"], msgs["recv_rel_ns"]):
        assert rid % MOD == 0
        assert step == bs.step_at_or_after(planned[rid]) + offs[j]
        assert p == bs.planned_rel_ns(step) and r >= p
    # the lag histogram counts every message, sampled or not
    assert total(summ, "msgs.msgs") == total(summ, "counts.ok") * per_req


def _worker(tt, origin):
    """The fields LoadWorker.planned_msg reads, on a hand-made table (no process, no channel)."""
    from capgrpc.common import TierCursor
    from capgrpc.loader import MSG_FIELDS
    from capgrpc.msgplan import MsgPlan
    lags = []
    w = types.SimpleNamespace(msgplan=MsgPlan(tt), origin=origin,
                              h_msg=types.SimpleNamespace(add=lags.append),
                              msg_rec=RecordArray(MSG_FIELDS, 1000),
                              unplanned_by_tier=[0] * len(tt), cursor=TierCursor(tt))
    return w, lags


@pytest.mark.parametrize("delay_ns", [0, 50 * MS])
def test_planned_msg_lag_values_on_a_synthetic_stream(delay_ns):
    """A request planned at p (15 ms: rounds up to step 2 = 20 ms) whose message j arrives at
    planned step time + delay: every lag is exactly delay; one message past the expected count
    is unplanned and stays out of the lag."""
    from capgrpc.loader import LoadWorker
    tt = TierTable.from_doc(table_doc([("warmup", 400, 10.0, {"n": 3, "beat": BEAT}),
                                       ("drain", 300, 0.0, {"beat": BEAT})]))
    origin = 1_000_000_000_000
    w, lags = _worker(tt, origin)
    st = w.msgplan.stream(15 * MS, 0)
    assert st[0] == 2
    n_exp = expect.msgs_per_request(3, 1, 1, "product") + 1
    for j in range(n_exp):
        step, p = w.msgplan.planned(st, j)
        LoadWorker.planned_msg(w, st, j, origin + p + delay_ns, 7, 0, True)
    LoadWorker.planned_msg(w, st, n_exp, origin + 300 * MS, 7, 0, True)
    assert lags == [delay_ns] * n_exp
    assert w.unplanned_by_tier[0] == 1
    cols = {k: list(v[:w.msg_rec.n]) for k, v in w.msg_rec.cols.items()}
    assert cols["idx"] == list(range(n_exp)) and set(cols["stream_id"]) == {7}
    assert [r - p for r, p in zip(cols["recv_rel_ns"], cols["planned_rel_ns"])] == \
        [delay_ns] * n_exp
    assert cols["step"] == [2 + o for o in expect.msg_step_offsets(3, 1, 1, "product")] + \
        [2 + expect.msg_step_offsets(3, 1, 1, "product")[-1]]


def test_delayed_half_window_shows_as_a_lag_step():
    """Positive control at the record level: streams planned in the second half of a window
    arrive 50 ms late, the first half on time; the half-window lag medians differ by 50 ms (what
    the readout's X rule compares, together with the per-second slope)."""
    from capgrpc.loader import LoadWorker
    tt = TierTable.from_doc(table_doc([("warmup", 2000, 10.0, {"n": 3, "beat": BEAT}),
                                       ("drain", 300, 0.0, {"beat": BEAT})]))
    origin = 5_000_000_000_000
    w, lags = _worker(tt, origin)
    first, second = [], []
    n_exp = expect.msgs_per_request(3, 1, 1, "product") + 1
    for k in range(20):  # one request every 100 ms over the 2 s window
        prel = k * 100 * MS
        st = w.msgplan.stream(prel, 0)
        delay = 50 * MS if prel >= 1000 * MS else 0
        for j in range(n_exp):
            _step, p = w.msgplan.planned(st, j)
            LoadWorker.planned_msg(w, st, j, origin + p + delay, k, 0, True)
        (second if delay else first).extend(lags[-n_exp:])
    first.sort()
    second.sort()
    assert second[len(second) // 2] - first[len(first) // 2] == 50 * MS
    assert w.unplanned_by_tier[0] == 0 and w.msg_rec.n == 20 * n_exp
