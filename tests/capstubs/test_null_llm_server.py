# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NL (python -m capstubs.null_llm_server) as a real process on 127.0.0.1, with the product
server replaced by the test double snl_double (--server-module): capgrpc.load (openengine
request bytes, K processes) end to end with counts, yielded counts and heartbeats reconciled;
a raw client for the metronome's step timing, the minimal mode, rejections and truncation; the
metronome-only dry run; the short switch; start-up refusals (exit 3)."""

import json
import os
import subprocess
import sys
import time

import grpc
import pytest

import oe_double
from capcommon import expect
from capcommon.lifecycle import read_port_file
from capcommon.schedule import BeatSchedule, load_schedule
from capgrpc.reqbytes import encode_pb_string_field

from conftest import HERE, METHOD, MS, TOOLS, Launch, per_tier, table_doc, total, write_reqbytes

MOD = "capstubs.null_llm_server"


class SnlLaunch(Launch):
    """The gRPC harness with tests/capstubs on PYTHONPATH (the server double lives there)."""

    def start(self, role, module, args):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([self.tools, HERE])
        env.pop("CAPKIT_CODE_SHA", None)
        out = open(str(self.tmp / ("%s.out" % role)), "wb")
        err = open(str(self.tmp / ("%s.err" % role)), "wb")
        p = subprocess.Popen([self.py, "-m", module] + args, env=env, stdout=out, stderr=err,
                             cwd=str(self.tmp))
        self.procs[role] = p
        return p


@pytest.fixture
def launch(tmp_path):
    L = SnlLaunch(tmp_path, sys.executable)
    L.tools = TOOLS
    try:
        yield L
    finally:
        L.kill_all()


def snl_args(mode, *extra, server="snl_double", token_ids="11,12,13"):
    out = ["--workers", "1", "--mode", mode, "--model", "m0", "--token-source", "fixed",
           "--token-ids", token_ids]
    if mode != "metronome-only":
        out += ["--server-module", server]
    return out + list(extra)


def start_snl(launch, role, mode, *extra, cpus="0", **kw):
    launch.server(role, MOD, snl_args(mode, *extra, **kw), cpus=cpus)
    launch.wait_file(launch.tree.ready_file(role), 30, role)
    return read_port_file(launch.tree.port_file(role))[1]


def ready_info(launch, role):
    with open(launch.tree.ready_file(role)) as f:
        return json.load(f)["info"]["workers"][0]["info"]


def request(rid, n, tokens=(1, 2, 3)):
    r = oe_double.GenerateRequest(model="m0", token_ids=oe_double.TokenIds(ids=list(tokens)))
    r.stopping.max_tokens = n
    r.stopping.ignore_eos = True
    return (encode_pb_string_field(1, rid) if rid is not None else b"") + r.SerializeToString()


def raw_call(port, body, timeout=10.0):
    """(parsed responses, receive monotonic ns of each, status code) of one call."""
    msgs, ts = [], []
    with grpc.insecure_channel("127.0.0.1:%d" % port) as ch:
        call = ch.unary_stream(METHOD)(body, timeout=timeout)
        try:
            for m in call:
                ts.append(time.monotonic_ns())
                msgs.append(oe_double.GenerateResponse.FromString(m))
            code = grpc.StatusCode.OK
        except grpc.RpcError as e:
            code = e.code()
    return msgs, ts, code


def wait_until(abs_ns):
    while time.monotonic_ns() < abs_ns:
        time.sleep(0.005)


def inject_totals(summ):
    inj = {}
    for docs in summ.values():
        for e in docs[0]["inject"]:
            h, a = inj.get(e["site"], (0, 0))
            inj[e["site"]] = (h + e["hits_delta"], a + e["applied_delta"])
    return inj


# ---- capgrpc.load (K processes, openengine request bytes) -> S-NL full ---------------------------

def test_full_mode_with_a_k_process_reader(launch):
    launch.write_table(table_doc([("warmup", 400, 25.0, {"n": 5}), ("drain", 300, 0.0, {}),
                                  ("up", 500, 40.0, {"n": 7}), ("drain", 400, 0.0, {})]))
    rb = write_reqbytes(launch)
    launch.set_origin(5.0)
    start_snl(launch, "snl", "full", "--schedule", launch.sched_path,
              "--inject", "site=send.msg,mode=busy,delta_ns=1000",
              "--inject", "site=beat.step,mode=busy,delta_ns=1000")
    launch.start("load", "capgrpc.load",
                 launch.common("load", cpus="1,2")
                 + ["--target-port-file", launch.tree.port_file("snl"), "--method", METHOD,
                    "--workers", "2", "--channels-per-worker", "1",
                    "--schedule", launch.sched_path, "--request-format", "openengine",
                    "--request-bytes", rb, "--msg-sample-mod", "1",
                    "--max-records-per-worker", "1000"])
    assert launch.wait_exit("load", 60) == 0, launch.stderr("load")
    assert launch.stop("snl") == 0, launch.stderr("snl")
    ld, sn = launch.summaries("load"), launch.summaries("snl")
    planned = per_tier(ld, "counts.planned")
    assert planned[0] > 0 and planned[2] > 0
    assert per_tier(ld, "counts.ok") == planned and total(ld, "counts.failed") == 0
    msgs_ld = per_tier(ld, "msgs.msgs")
    for seq, n in ((0, 5), (2, 7)):
        c = sn[seq][0]["counts"]
        mpr = expect.msgs_per_request(n, 1, 1, "product")
        assert c["streams_accepted"] == c["streams_completed"] == planned[seq]
        assert c["rpcs"] == planned[seq] and c["rpcs_without_stream"] == 0
        # the servicer yields one message per item plus the finished message; counted by the
        # Generate wrapper, equal to what the K reader processes received
        assert c["msgs_yielded"] == planned[seq] * (mpr + 1) == msgs_ld[seq]
        assert c["items_taken"] == planned[seq] * mpr
        assert c["tokens_produced"] == planned[seq] * n
        assert sn[seq][0]["tier_attribution"] == "schedule"
        assert sn[seq][0]["servicer_wrapped"] is True
        assert sn[seq][0]["imported"]["server_module"] == "snl_double"
        assert sn[seq][0]["imported"]["servicer_class"] == "snl_double.OpenEngineInferenceServicer"
        assert sn[seq][0]["beat"]["work_ns"] > 0
        assert 0 < sn[seq][0]["beat"]["work_share_of_loop_thread"] <= 1
    # event-time slot deltas cover the same messages (the yielded counter is the wrapper's)
    assert total(sn, "yielded_event_time") == total(sn, "counts.msgs_yielded") == \
        total(ld, "msgs.msgs")
    assert total(sn, "items_taken_event_time") == total(sn, "counts.items_taken")
    steps, _ = launch.records("snl", "steps")
    assert sum(steps["items"]) == total(sn, "beat.enqueued") == total(sn, "counts.items_taken")
    assert all(w >= 0 for w in steps["work_ns"])
    # heartbeats: send.msg once per yielded message, beat.step once per metronome step
    inj = inject_totals(sn)
    assert inj["send.msg"] == (total(sn, "counts.msgs_yielded"),) * 2
    assert inj["beat.step"] == (len(steps["step"]),) * 2 == (total(sn, "beat.steps"),) * 2
    assert inj["short.req"] == (total(sn, "counts.streams_accepted"), 0)
    streams, heads = launch.records("snl", "streams")
    reqs, _ = launch.records("load", "requests")
    assert sorted(streams["req_id"]) == sorted(reqs["req_id"])
    assert heads[0]["meta"]["mode"] == "full"
    info = ready_info(launch, "snl")
    assert info["mode"] == "full" and info["servicer_wrapped"] is True
    assert info["imported"]["server_module_file"].endswith("snl_double.py")


# ---- raw client: metronome timing, minimal mode, rejections, truncation --------------------------

def test_metronome_paces_one_item_per_planned_step(launch):
    beat = 60 * MS
    tt = launch.write_table(table_doc([("warmup", 1980, 2.0, {"n": 4, "beat": beat}),
                                       ("drain", 300, 0.0, {"beat": beat})]))
    origin = launch.set_origin(4.0)
    port = start_snl(launch, "snl", "full")
    wait_until(origin + 100 * MS)
    msgs, ts, code = raw_call(port, request("cmpl-a", 4))
    assert code == grpc.StatusCode.OK
    assert [m.WhichOneof("event") for m in msgs] == ["token"] * 4 + ["finished"]
    assert msgs[-1].usage.completion_tokens == 4 and msgs[-1].usage.prompt_tokens == 3
    assert [t.token_id for m in msgs[:4] for t in m.token.tokens] == [11, 12, 13, 11]
    assert launch.stop("snl") == 0, launch.stderr("snl")
    streams, _ = launch.records("snl", "streams")
    first = streams["first_step"][0]
    bs = BeatSchedule(tt)
    offs = expect.msg_step_offsets(4, 1, 1, "product")
    for j in range(4):  # token message j no earlier than its planned step
        planned = origin + bs.planned_rel_ns(first + offs[j])
        assert ts[j] >= planned - 2 * MS, (j, (ts[j] - planned) / MS)
    gaps = [ts[j + 1] - ts[j] for j in range(3)]
    assert min(gaps) >= 0.6 * beat, [g / MS for g in gaps]
    sn = launch.summaries("snl")
    assert sn[0][0]["counts"]["streams_completed"] == 1
    assert sn[0][0]["tier_attribution"] == "event"


def test_minimal_mode_answers_without_a_beat(launch):
    launch.write_table(table_doc([("warmup", 1500, 2.0, {"n": 6, "beat": 0}),
                                  ("drain", 300, 0.0, {"beat": 0})]))
    origin = launch.set_origin(4.0)
    port = start_snl(launch, "snl", "minimal")
    wait_until(origin + 50 * MS)
    msgs, ts, code = raw_call(port, request("x1", 6))
    assert code == grpc.StatusCode.OK
    assert [m.WhichOneof("event") for m in msgs] == ["token"] * 6 + ["finished"]
    assert ts[-1] - ts[0] < 500 * MS
    assert launch.stop("snl") == 0, launch.stderr("snl")
    sn = launch.summaries("snl")
    c = sn[0][0]["counts"]
    assert c["streams_completed"] == 1 and c["msgs_yielded"] == 7 and c["items_taken"] == 6
    assert sn[0][0]["pacing"] == "asap" and sn[0][0]["beat"]["steps"] == 0
    assert not launch.records("snl", "steps")[1]  # no metronome, no steps file
    assert ready_info(launch, "snl")["beat_steps"] == 0


def test_rejections_truncation_and_raw_requests(launch):
    beat = 20 * MS
    launch.write_table(table_doc([("warmup", 600, 5.0, {"n": 30, "beat": beat}),
                                  ("drain", 200, 0.0, {"beat": beat})]))
    origin = launch.set_origin(4.0)
    port = start_snl(launch, "snl", "full", "--schedule", launch.sched_path)
    rids, _p, tseq, _m = load_schedule(launch.sched_path, 0, 1)
    wait_until(origin + 50 * MS)
    _m1, _t1, c1 = raw_call(port, request("cmpl-x", 3))  # not a req_id of the schedule
    _m2, _t2, c2 = raw_call(port, request(None, 3))  # raw format: no request_id at all
    msgs, _t3, c3 = raw_call(port, request(str(rids[0]), 45))  # 45 steps do not fit: truncated
    assert c1 == c2 == grpc.StatusCode.INVALID_ARGUMENT
    assert c3 == grpc.StatusCode.OK and msgs[-1].WhichOneof("event") == "error"
    assert launch.stop("snl") == 0, launch.stderr("snl")
    sn = launch.summaries("snl")
    c = sn[0][0]["counts"]
    assert c["rejected"] == 2 and c["streams_truncated"] == 1 and c["rpcs"] == 3
    assert c["request_id_not_decimal"] == 0  # rejected before a stream exists
    assert total(sn, "counts.streams_accepted") == 1


def test_metronome_only_dry_run(launch):
    launch.write_table(table_doc([("warmup", 400, 25.0, {"n": 5}), ("drain", 300, 0.0, {}),
                                  ("up", 500, 40.0, {"n": 3}), ("drain", 400, 0.0, {})]))
    origin = launch.set_origin(4.0)
    start_snl(launch, "snl", "metronome-only", "--schedule", launch.sched_path)
    wait_until(origin + 1700 * MS)
    assert launch.stop("snl") == 0, launch.stderr("snl")
    rids, _p, tseq, _m = load_schedule(launch.sched_path, 0, 1)
    rows = {s: list(tseq).count(s) for s in set(tseq)}
    sn = launch.summaries("snl")
    for seq, n in ((0, 5), (2, 3)):
        c = sn[seq][0]["counts"]
        mpr = expect.msgs_per_request(n, 1, 1, "product")
        assert c["streams_accepted"] == c["streams_completed"] == rows[seq] > 0
        assert c["items_taken"] == rows[seq] * mpr and c["tokens_produced"] == rows[seq] * n
        assert c["msgs_yielded"] == 0 and sn[seq][0]["servicer"] is None
        assert sn[seq][0]["imported"] is None and sn[seq][0]["beat"]["work_ns"] > 0
    assert total(sn, "yielded_event_time") == 0
    assert "send.msg" not in {s for s, (h, _a) in inject_totals(sn).items() if h}
    info = ready_info(launch, "snl")
    assert info["mode"] == "metronome-only" and info["imported"] is None


def test_short_switch(launch):
    launch.write_table(table_doc([("warmup", 600, 20.0, {"n": 6}), ("drain", 300, 0.0, {})]))
    rb = write_reqbytes(launch)
    launch.set_origin(4.0)
    start_snl(launch, "snl", "full", "--schedule", launch.sched_path,
              "--inject", "site=short.req,mode=short,short_tokens=2,every_n=2")
    launch.start("load", "capgrpc.load",
                 launch.common("load", cpus="1")
                 + ["--target-port-file", launch.tree.port_file("snl"), "--method", METHOD,
                    "--workers", "1", "--in-process", "--channels-per-worker", "1",
                    "--schedule", launch.sched_path, "--request-format", "openengine",
                    "--request-bytes", rb, "--msg-sample-mod", "1",
                    "--max-records-per-worker", "1000"])
    assert launch.wait_exit("load", 40) == 0, launch.stderr("load")
    assert launch.stop("snl") == 0, launch.stderr("snl")
    sn = launch.summaries("snl")
    acc = total(sn, "counts.streams_accepted")
    short = total(sn, "counts.short_requests")
    assert acc > 2 and short == (acc + 1) // 2
    assert total(sn, "counts.tokens_produced") == acc * 6 - 2 * short
    assert inject_totals(sn)["short.req"] == (acc, short)


REFUSALS = {
    "workers": (["--workers", "2"], "--workers must be 1"),
    "no_product": (["--server-module", "no_such_product_module"], "needs the product server"),
    "unwrapped": (["--server-module", "snl_double_unwrapped"], "cannot count"),
    "tokenizer_needs_product": (["--tokenizer", "{tmp}", "--trust-remote-code", "false",
                                 "--tokenizer-mode", "auto"], "cannot load the tokenizer"),
    "tokenizer_flags": (["--tokenizer", "{tmp}"], "needs --trust-remote-code"),
    "stream_interval": (["--stream-interval", "2"], "--stream-interval 2"),
    "inject_site": (["--inject", "site=recv.msg,mode=busy,delta_ns=10"], "unknown to S-NL"),
    "inject_mode": (["--inject", "site=send.msg,mode=short,short_tokens=1"],
                    "does not support mode"),
    "metronome_only_schedule": (["--mode", "metronome-only"], "needs --schedule"),
    "no_beat": (["--beat0"], "no tier has beat_ns > 0"),
    "minimal_prefill": (["--mode", "minimal", "--prefill-delay-us", "10"], "no effect"),
    "token_ids_prompt": (["--token-source", "prompt", "--token-ids", "5"],
                         "--token-ids is required"),
    "token_ids_fixed": (["--token-source", "fixed"], "--token-ids is required"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_refused_at_start_up(launch, case):
    extra, msg = REFUSALS[case]
    beat = 0 if "--beat0" in extra else 10 * MS
    launch.write_table(table_doc([("warmup", 500, 5.0, {"n": 4, "beat": beat}),
                                  ("drain", 300, 0.0, {"beat": beat})]))
    launch.set_origin(20.0)
    base = snl_args("full")
    extra = [a.replace("{tmp}", str(launch.tmp)) for a in extra if a != "--beat0"]
    for flag in ("--workers", "--mode", "--server-module", "--token-source"):
        if flag in extra and flag in base:  # replace the default value
            i = base.index(flag)
            del base[i:i + 2]
    if "--token-source" in extra:
        i = base.index("--token-ids")
        del base[i:i + 2]
    launch.server("snl", MOD, base + extra)
    rc = launch.wait_exit("snl", 30)
    assert rc == 3, (rc, launch.stderr("snl"))
    assert msg in launch.stderr("snl"), launch.stderr("snl")
    assert not os.path.exists(launch.tree.ready_file("snl"))
