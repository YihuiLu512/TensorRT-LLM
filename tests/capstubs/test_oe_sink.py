# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capgrpc.oe_sink as a real process on 127.0.0.1 against S-PB, with the client double of
oe_product (the probe tree's own client files in a minimal tensorrt_llm package): both tiers and
both connect modes end to end with counts reconciled against S-PB; the weak-reader swap
(capgrpc.load with the same reader argv); --planned-msgs on both readers; timeouts, rejections,
the slow-read switch and recv.msg injection; start-up refusals; unit checks of the message plan
and the EngineRequest round trip."""

import base64
import hashlib
import json
import os
import subprocess
import sys
import types

import pytest

import oe_double
import oe_product
from capcommon import expect
from capcommon.lifecycle import read_port_file
from capcommon.schedule import BeatSchedule

from conftest import (HERE, METHOD, MS, TOOLS, Launch, per_tier, real_pb2, spb_args, table_doc,
                      total, write_json, write_prebaked, write_reqbytes)

SINK = "capgrpc.oe_sink"
SPB = "capstubs.prebaked_server"
TIER1, TIER2 = "generate-stream", "engine-event-stream"


class SinkLaunch(Launch):
    """The gRPC harness with the client double (and tests/capstubs) on PYTHONPATH."""

    product = True

    def start(self, role, module, args):
        env = dict(os.environ)
        paths = [TOOLS, HERE] + ([self.pkg] if self.product else [])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        env.pop("CAPKIT_CODE_SHA", None)
        out = open(str(self.tmp / ("%s.out" % role)), "wb")
        err = open(str(self.tmp / ("%s.err" % role)), "wb")
        p = subprocess.Popen([self.py, "-m", module] + args, env=env, stdout=out, stderr=err,
                             cwd=str(self.tmp))
        self.procs[role] = p
        return p


@pytest.fixture
def launch(tmp_path):
    L = SinkLaunch(tmp_path, sys.executable)
    L.pkg = oe_product.build(str(tmp_path / "pkg"))
    try:
        yield L
    finally:
        L.kill_all()


def spec(n=5, rate=40.0, beat=10 * MS, win=600):
    return [("warmup", 400, 20.0, {"n": n, "beat": beat}), ("drain", 300, 0.0, {"beat": beat}),
            ("up", win, rate, {"n": n, "beat": beat}), ("drain", 500, 0.0, {"beat": beat})]


def setup(launch, sp=None, timeout_ns=10_000_000_000, raw=False, origin_s=5.0):
    launch.write_table(table_doc(sp or spec(), timeout_ns=timeout_ns))
    gen, _oe = real_pb2()
    pb_path, doc = write_prebaked(launch, pb2=gen)
    if raw:
        from capgrpc.reqbytes import build_raw
        r = oe_double.GenerateRequest(model="m0", token_ids=oe_double.TokenIds(ids=[1, 2]))
        r.stopping.max_tokens = 5
        r.stopping.ignore_eos = True
        rb = write_json(launch.tmp / "raw.json", build_raw(r.SerializeToString(), METHOD))
    else:
        rb = write_reqbytes(launch)
    launch.set_origin(origin_s)
    return pb_path, rb


def start_spb(launch, pb_path, *extra, role="spb"):
    launch.server(role, SPB, spb_args(pb_path, *extra), cpus="0")
    launch.wait_file(launch.tree.ready_file(role), 30, role)
    return read_port_file(launch.tree.port_file(role))[1]


def reader_argv(launch, role, rb, fmt="openengine", *extra):
    """The reader arguments oe_sink and the weak reader share (capgrpc.loader.add_reader_args)."""
    return launch.common(role, cpus="1") + [
        "--target-port-file", launch.tree.port_file("spb"), "--method", METHOD,
        "--workers", "1", "--in-process", "--channels-per-worker", "1",
        "--schedule", launch.sched_path, "--request-format", fmt, "--request-bytes", rb,
        "--msg-sample-mod", "1", "--max-records-per-worker", "1000"] + list(extra)


PLANNED = ("--planned-msgs", "--max-msg-records-per-worker", "5000")


def sink_argv(launch, rb, tier=TIER1, connect="channel", *extra, fmt="openengine", role="sink"):
    return reader_argv(launch, role, rb, fmt, *extra) + ["--tier", tier, "--connect", connect]


def run_reader(launch, role, module, argv, timeout=40):
    launch.start(role, module, argv)
    return launch.wait_exit(role, timeout)


def inject_totals(summ, site):
    h = a = 0
    for docs in summ.values():
        for d in docs:
            for e in d["inject"]:
                if e["site"] == site:
                    h += e["hits_delta"]
                    a += e["applied_delta"]
    return h, a


def check_sink_against_spb(launch, role, type_name):
    """Every planned request ok; per tier, the reader's message count is S-PB's msgs_sent."""
    sk, sp = launch.summaries(role), launch.summaries("spb")
    planned = per_tier(sk, "counts.planned")
    assert planned[0] > 0 and planned[2] > 0
    assert per_tier(sk, "counts.ok") == planned and total(sk, "counts.failed") == 0
    m = expect.msgs_per_request(5, 1, 1, "product") + 1
    for seq in (0, 2):
        assert sk[seq][0]["msgs"]["msgs"] == sp[seq][0]["counts"]["msgs_sent"] == planned[seq] * m
        assert sk[seq][0]["recv_types"] == {type_name: planned[seq]}
    return sk, sp


# ---- end to end ----------------------------------------------------------------------------------

@pytest.mark.parametrize("tier,type_name", [(TIER1, "GenerateResponse"), (TIER2, "TokenDelta")])
def test_sink_channel_mode_reads_spb(launch, tier, type_name):
    pb_path, rb = setup(launch)
    start_spb(launch, pb_path, "--schedule", launch.sched_path)
    rc = run_reader(launch, "sink", SINK, sink_argv(launch, rb, tier, "channel", *PLANNED))
    assert rc == 0, launch.stderr("sink")
    assert launch.stop("spb") == 0, launch.stderr("spb")
    sk, sp = check_sink_against_spb(launch, "sink", type_name)
    d = sk[2][0]
    assert d["tool"] == SINK and d["in_process"] is True and d["n_workers"] == 1
    assert d["oe_sink"] == {"tier": tier, "connect": "channel",
                            "timeout_mechanism": "client_cancel" if tier == TIER2
                            else "grpc_deadline", "slow_read_ns": None, "slow_read_pauses": 0}
    assert d["msgs"]["bytes"] is None and d["local_subchannel_pool"] is None
    imp = d["imported"]
    assert imp["tier"] == tier and imp["generate_method"] == METHOD
    assert imp["modules"]["tensorrt_llm.grpc.openengine_client"].startswith(launch.pkg)
    assert ("tensorrt_llm.serve.openengine_poc.engine_client_impl" in imp["modules"]) == \
        (tier == TIER2)
    # requests rows: bytes not counted (-1); every message row has a planned step
    reqs, _ = launch.records("sink", "requests")
    assert set(reqs["bytes"]) == {-1} and set(reqs["status"]) == {0}
    msgs, heads = launch.records("sink", "msgs")
    assert heads[0]["meta"]["planned_source"] == "schedule"
    assert len(msgs["idx"]) == total(sk, "msgs.msgs") and total(sk, "msg_plan.unplanned") == 0
    # heartbeats: send.req once per request, recv.msg once per counted message
    assert inject_totals(sk, "send.req")[0] == total(sk, "counts.sent")
    assert inject_totals(sk, "recv.msg")[0] == total(sk, "msgs.msgs")
    # S-PB attributes by the schedule: the decimal request ids reached it
    assert sp[2][0]["tier_attribution"] == "schedule"
    assert total(sp, "counts.request_id_not_decimal") == 0
    # the product client (both tiers) sends the request_id oe_sink set: the decimal req_id
    sent = {str(r) for r in reqs["req_id"]}
    got = [r for d in sp.values() for r in d[0]["request_id_samples"]["first"]]
    assert got and set(got) <= sent
    with open(launch.tree.ready_file("sink")) as f:
        info = json.load(f)["info"]["workers"][0]["info"]
    assert info["connect"] == "channel" and info["imported"]["tier"] == tier
    assert info["planned_msgs"] is True


def test_planned_msg_lag_matches_spb_steps(launch):
    """--planned-msgs: message j of a request planned at p is planned on
    step_at_or_after(p) + offsets[j]; S-PB's own first step is step_at_or_after(accept), so the
    derived step is never after S-PB's, and every received message is after its derived time."""
    pb_path, rb = setup(launch)
    start_spb(launch, pb_path, "--schedule", launch.sched_path)
    assert run_reader(launch, "sink", SINK, sink_argv(launch, rb, TIER1, "channel",
                                                      *PLANNED)) == 0, launch.stderr("sink")
    assert launch.stop("spb") == 0
    msgs, _ = launch.records("sink", "msgs")
    reqs, _ = launch.records("sink", "requests")
    streams, _ = launch.records("spb", "streams")
    bs = BeatSchedule(launch.tt)
    offs = expect.msg_step_offsets(5, 1, 1, "product")
    offs = offs + [offs[-1]]
    planned = dict(zip(reqs["req_id"], reqs["planned_rel_ns"]))
    first = dict(zip(streams["req_id"], streams["first_step"]))
    for rid, j, step, p, r in zip(msgs["stream_id"], msgs["idx"], msgs["step"],
                                  msgs["planned_rel_ns"], msgs["recv_rel_ns"]):
        assert step == bs.step_at_or_after(planned[rid]) + offs[j]
        assert p == bs.planned_rel_ns(step) and r >= p
        assert step <= first[rid] + offs[j]


def test_engine_event_stream_product_connect_with_real_sfe(launch):
    """--connect product against S-PB's frontend mode running the probe tree's fake_openengine:
    the client's own connect() (Health, GetServerInfo, GetModelInfo) and both conversions."""
    pb_path, rb = setup(launch)
    start_spb(launch, pb_path, "--schedule", launch.sched_path, "--sfe-module", oe_product.SFE,
              "--sfe-profile", "trtllm-like", "--model", "m0", "--vocab-size", "64")
    rc = run_reader(launch, "sink", SINK, sink_argv(launch, rb, TIER2, "product"))
    assert rc == 0, launch.stderr("sink")
    assert launch.stop("spb") == 0, launch.stderr("spb")
    sk, _ = check_sink_against_spb(launch, "sink", "TokenDelta")
    assert sk[2][0]["oe_sink"]["connect"] == "product"
    assert sk[2][0]["msg_lag_ns"] is None and sk[2][0]["msg_plan"]["source"] is None


def test_product_connect_needs_control(launch):
    """--connect product against a Generate-only S-PB: the product connect() fails (Control is
    UNIMPLEMENTED) before the origin, exit 4 with the client's error."""
    pb_path, rb = setup(launch)
    start_spb(launch, pb_path)
    assert run_reader(launch, "sink", SINK, sink_argv(launch, rb, TIER1, "product")) == 4
    assert "connect" in launch.stderr("sink") and "UNIMPLEMENTED" in launch.stderr("sink")


def test_weak_reader_swap_is_module_and_sink_flags_only(launch):
    """The weak reader (capgrpc.load --in-process, one channel) and oe_sink read the same S-PB
    with one shared reader argv; with --planned-msgs both write the same drain.msgs records."""
    pb_path, rb = setup(launch)
    start_spb(launch, pb_path, "--schedule", launch.sched_path)
    base = reader_argv(launch, "weak", rb, "openengine", *PLANNED)
    sink = [a if a != "weak" else "sink" for a in base]
    sink = [a.replace("/weak.", "/sink.") for a in sink]
    assert sink != base and len(sink) == len(base)  # only the role name differs
    # one origin, so the two readers run side by side (S-PB serves both)
    launch.start("weak", "capgrpc.load", base)
    launch.start("sink", SINK, sink + ["--tier", TIER1, "--connect", "channel"])
    assert launch.wait_exit("weak", 40) == 0, launch.stderr("weak")
    assert launch.wait_exit("sink", 40) == 0, launch.stderr("sink")
    assert launch.stop("spb") == 0
    wk, sk = launch.summaries("weak"), launch.summaries("sink")
    for key in ("counts.planned", "counts.ok", "msgs.msgs"):
        assert per_tier(wk, key) == per_tier(sk, key)
    assert wk[2][0]["recv_types"] == {"bytes": wk[2][0]["counts"]["ok"]}
    for role in ("weak", "sink"):
        m, h = launch.records(role, "msgs")
        assert h[0]["kind"] == "drain.msgs" and h[0]["meta"]["planned_source"] == "schedule"
        assert len(m["idx"]) == total(launch.summaries(role), "msgs.msgs")
    sp = launch.summaries("spb")
    assert total(sp, "counts.streams_accepted") == 2 * total(sk, "counts.ok")


# ---- failure paths -------------------------------------------------------------------------------

@pytest.mark.parametrize("tier", [TIER1, TIER2])
def test_timeouts_are_registered(launch, tier):
    """Streams longer than the request timeout end as timeout_registered (overload): tier 1 by
    the gRPC deadline, tier 2 by the client-side cancel at the same instant."""
    sp = [("warmup", 400, 10.0, {"n": 40, "beat": 20 * MS}),
          ("drain", 1200, 0.0, {"beat": 20 * MS})]
    pb_path, rb = setup(launch, sp=sp, timeout_ns=200 * MS)
    start_spb(launch, pb_path, "--schedule", launch.sched_path)
    assert run_reader(launch, "sink", SINK, sink_argv(launch, rb, tier, "channel")) == 0, \
        launch.stderr("sink")
    assert launch.stop("spb") == 0
    sk = launch.summaries("sink")
    d = sk[0][0]
    assert d["counts"]["ok"] == 0 and d["counts"]["failed"] == d["counts"]["planned"] > 0
    assert [{k: f[k] for k in ("reason", "count", "class_hint")} for f in d["failures"]] == [
        {"reason": "timeout_registered", "count": d["counts"]["planned"],
         "class_hint": "overload"}]
    assert 0 <= d["failures"][0]["zero_msgs"] <= d["counts"]["planned"]
    assert d["failures"][0]["max_send_to_end_ns"] >= 200 * MS
    assert 0 < d["msgs"]["msgs"] < d["counts"]["planned"] * 41
    reqs, _ = launch.records("sink", "requests")
    for p, done in zip(reqs["planned_rel_ns"], reqs["done_rel_ns"]):
        assert 200 * MS <= done - p < 600 * MS


def test_rejected_request_is_a_grpc_status(launch):
    """raw request bytes carry no request id: the sink sends the product's fresh id, which an
    S-PB holding the schedule rejects (INVALID_ARGUMENT, non-overload)."""
    pb_path, rb = setup(launch, raw=True)
    start_spb(launch, pb_path, "--schedule", launch.sched_path)
    assert run_reader(launch, "sink", SINK, sink_argv(launch, rb, TIER1, "channel",
                                                      fmt="raw")) == 0, launch.stderr("sink")
    assert launch.stop("spb") == 0
    sk, sp = launch.summaries("sink"), launch.summaries("spb")
    n = total(sk, "counts.planned")
    assert total(sk, "counts.failed") == n > 0 and total(sk, "msgs.msgs") == 0
    reasons = {f["reason"]: f["class_hint"] for d in sk.values() for f in d[0]["failures"]}
    assert reasons == {"grpc_INVALID_ARGUMENT": "unclassified"}
    assert sum(d[0]["rejects"].get("request_id_not_in_schedule", 0) for d in sp.values()) == n


def test_raw_requests_without_schedule_are_read(launch):
    pb_path, rb = setup(launch, raw=True)
    start_spb(launch, pb_path)
    assert run_reader(launch, "sink", SINK, sink_argv(launch, rb, TIER2, "channel",
                                                      fmt="raw")) == 0, launch.stderr("sink")
    assert launch.stop("spb") == 0
    sk, sp = launch.summaries("sink"), launch.summaries("spb")
    assert total(sk, "counts.failed") == 0 and total(sk, "counts.ok") > 0
    assert total(sk, "msgs.msgs") == total(sp, "counts.msgs_sent")
    assert total(sp, "counts.request_id_not_decimal") == total(sk, "counts.ok")
    # raw format: the product client's own fresh ids, none of them decimal
    got = [r for d in sp.values() for r in d[0]["request_id_samples"]["first"]]
    assert got and not any(r.isdigit() for r in got)


def test_slow_read_pauses_every_message(launch):
    pb_path, rb = setup(launch, sp=spec(rate=20.0, beat=1 * MS))
    start_spb(launch, pb_path, "--schedule", launch.sched_path)
    argv = sink_argv(launch, rb, TIER1, "channel", *PLANNED) + ["--slow-read-us", "20000"]
    assert run_reader(launch, "sink", SINK, argv) == 0, launch.stderr("sink")
    assert launch.stop("spb") == 0
    sk = launch.summaries("sink")
    assert per_tier(sk, "oe_sink.slow_read_pauses") == per_tier(sk, "msgs.msgs")
    assert sk[2][0]["oe_sink"]["slow_read_ns"] == 20_000_000
    msgs, _ = launch.records("sink", "msgs")
    last = {}
    gaps = []
    for rid, r in zip(msgs["stream_id"], msgs["recv_rel_ns"]):
        if rid in last:
            gaps.append(r - last[rid])
        last[rid] = r
    # S-PB sends one message per 1 ms step; the reader takes one per >= 20 ms
    assert gaps and min(gaps) >= 20 * MS


def test_recv_msg_busy_injection_heartbeat(launch):
    pb_path, rb = setup(launch)
    start_spb(launch, pb_path, "--schedule", launch.sched_path)
    argv = sink_argv(launch, rb, TIER2, "channel") + ["--inject",
                                                      "site=recv.msg,mode=busy,delta_ns=20000"]
    assert run_reader(launch, "sink", SINK, argv) == 0, launch.stderr("sink")
    assert launch.stop("spb") == 0
    sk = launch.summaries("sink")
    assert inject_totals(sk, "recv.msg") == (total(sk, "msgs.msgs"),) * 2


# ---- start-up refusals ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["no_product", "workers", "channels", "capkit", "method",
                                  "text_prompt_tier2", "slow_zero", "planned_raw",
                                  "planned_stamped_load", "no_tier", "abbrev"])
def test_startup_refusals(launch, case):
    pb_path, rb = setup(launch)
    if case == "no_product":
        launch.product = False
        argv, want, module = sink_argv(launch, rb), "container only", SINK
    elif case == "workers":
        argv = sink_argv(launch, rb)
        argv[argv.index("--workers") + 1] = "2"
        argv.remove("--in-process")
        want, module = "--workers must be 1", SINK
    elif case == "channels":
        argv = sink_argv(launch, rb)
        argv[argv.index("--channels-per-worker") + 1] = "2"
        want, module = "--channels-per-worker must be 1", SINK
    elif case == "capkit":
        argv = sink_argv(launch, rb)
        i = argv.index("--request-format")
        argv[i + 1] = "capkit"
        j = argv.index("--request-bytes")
        argv[j:j + 2] = ["--prompts", launch.prompts_path]
        want, module = "--request-format openengine or raw", SINK
    elif case == "method":
        argv = sink_argv(launch, rb)
        argv[argv.index("--method") + 1] = "/openengine.v1.Inference/Other"
        want, module = "not the product's Generate path", SINK
    elif case == "text_prompt_tier2":
        from capgrpc.reqbytes import build_openengine
        from conftest import g
        pdoc = g.prompts_doc()
        pdoc["sets"]["p"] = pdoc["sets"]["t"]
        doc = build_openengine(oe_double, pdoc, launch.tt, "m0", METHOD)
        trb = write_json(launch.tmp / "text_rb.json", doc)
        argv, want, module = sink_argv(launch, trb, TIER2), "needs token-id prompts", SINK
    elif case == "slow_zero":
        argv = sink_argv(launch, rb) + ["--slow-read-us", "0"]
        want, module = "--slow-read-us must be >= 1", SINK
    elif case == "planned_raw":
        argv = reader_argv(launch, "sink", rb, "raw", *PLANNED)
        want, module = "raw request bytes carry their own", "capgrpc.load"
    elif case == "planned_stamped_load":
        argv = reader_argv(launch, "sink", rb, "openengine", *PLANNED) + ["--stamped"]
        want, module = "exclude each other", "capgrpc.load"
    elif case == "no_tier":
        argv = reader_argv(launch, "sink", rb) + ["--connect", "channel"]
        want, module = "--tier", SINK
    else:  # abbrev: a prefix of --slow-read-us is not accepted
        argv = sink_argv(launch, rb) + ["--slow-read", "5"]
        want, module = "unrecognized arguments", SINK
    rc = run_reader(launch, "sink", module, argv, timeout=30)
    assert rc == (2 if case in ("no_tier", "abbrev") else 3), launch.stderr("sink")
    assert want in launch.stderr("sink"), launch.stderr("sink")
    assert not os.path.exists(launch.tree.ready_file("sink"))


def test_template_with_request_id_is_refused(launch):
    """Templates are request bodies without request_id (the sink sets it per request)."""
    from capgrpc.reqbytes import build_openengine, encode_pb_string_field
    from conftest import g
    setup(launch)
    doc = build_openengine(oe_double, g.prompts_doc(), launch.tt, "m0", METHOD)
    for e in doc["entries"]:
        raw = encode_pb_string_field(1, "7") + base64.b64decode(e["b64"])
        e["b64"] = base64.b64encode(raw).decode()
        e["sha256"] = hashlib.sha256(raw).hexdigest()
    rb = write_json(launch.tmp / "rid_rb.json", doc)
    assert run_reader(launch, "sink", SINK, sink_argv(launch, rb, TIER2), timeout=30) == 3
    assert "carries a request_id" in launch.stderr("sink")


# ---- in-process unit checks (the client double imported into this process) ----------------------

@pytest.fixture
def product(tmp_path):
    pkg = oe_product.build(str(tmp_path / "pkg"))
    sys.path.insert(0, pkg)
    try:
        from capgrpc.oe_sink import Product
        yield Product(TIER2)
    finally:
        sys.path.remove(pkg)


def test_engine_request_round_trip(product):
    from capcommon.exitcodes import ConfigError
    from capgrpc.oe_sink import engine_request_kwargs
    gen = product.pb2
    ok = gen.GenerateRequest(model="m0", token_ids=gen.TokenIds(ids=[1, 2, 3]))
    ok.stopping.max_tokens = 4
    ok.stopping.ignore_eos = True
    ok.sampling.temperature = 0.5
    kw = engine_request_kwargs(product, ok)
    assert kw == {"model": "m0", "prompt_token_ids": [1, 2, 3], "max_tokens": 4,
                  "ignore_eos": True, "sampling": {"temperature": 0.5}}
    # a field the EngineRequest path is not given -> refused (no silent difference on the wire)
    bad = gen.GenerateRequest()
    bad.CopyFrom(ok)
    bad.stopping.min_tokens = 2
    with pytest.raises(ConfigError, match="does not rebuild"):
        engine_request_kwargs(product, bad)
    text = gen.GenerateRequest(model="m0", prompt="hi")
    with pytest.raises(ConfigError, match="token-id prompts"):
        engine_request_kwargs(product, text)


def test_generate_method_is_read_from_the_stub(product):
    from capgrpc.oe_sink import generate_method
    assert product.method == METHOD == generate_method(product.pb2_grpc)


def test_msgplan_offsets_and_bounds():
    from capcommon.tiers import TierTable
    from capgrpc.msgplan import MsgPlan
    tt = TierTable.from_doc(table_doc([("warmup", 100, 10.0, {"n": 3, "beat": 10 * MS}),
                                       ("drain", 50, 0.0, {"beat": 10 * MS})]))
    mp = MsgPlan(tt)
    offs = expect.msg_step_offsets(3, 1, 1, "product")
    assert mp.offsets[0] == tuple(offs) + (offs[-1],) and mp.offsets[1] is None
    st = mp.stream(15 * MS, 0)  # step_at_or_after(15 ms) = step 2 (20 ms)
    assert st[0] == 2
    assert mp.planned(st, 0) == (2, 20 * MS)
    assert mp.planned(st, len(offs)) == (2 + offs[-1], (2 + offs[-1]) * 10 * MS)
    assert mp.planned(st, len(offs) + 1) is None  # past the expected count
    assert mp.planned((14, st[1]), 1) is None  # past the schedule end (15 steps)
    assert mp.stream(5 * MS, 1) is None  # a drain tier has no offsets


class _St:
    def __init__(self, cancels=True, sent=False):
        self.cancelled, self.sent, self._cancels = False, sent, cancels

    def cancel(self):
        self.cancelled = self._cancels


@pytest.mark.parametrize("stream,want", [
    (lambda: _St(), None),
    (lambda: _St(cancels=False), "reports cancelled=False"),
    (lambda: _St(sent=True), "sent=True"),
    (lambda: (_ for _ in ()).throw(AttributeError("no _client")), "no _client"),
])
def test_self_check_builds_and_cancels_one_stream(stream, want):
    """The pre-origin self-check builds the first planned request's stream and cancels it; a
    build failure or a stream that does not end cancelled-and-unsent is exit 3."""
    from capcommon.exitcodes import ConfigError
    from capgrpc.oe_sink import OeSinkWorker
    calls = []

    def open_stream(rid, seq, timeout):
        calls.append((rid, seq, timeout))
        return stream()

    w = types.SimpleNamespace(plan=types.SimpleNamespace(n=2, req_ids=[7, 8], tseq=[0, 0]),
                              product=types.SimpleNamespace(tier=TIER1), open_stream=open_stream)
    if want is None:
        OeSinkWorker.self_check(w)
    else:
        with pytest.raises(ConfigError, match=want):
            OeSinkWorker.self_check(w)
    assert calls == [(7, 0, None)]
