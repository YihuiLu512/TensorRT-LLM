# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-PB (python -m capstubs.prebaked_server) as a real process on 127.0.0.1: a nullgrpc-style
reader (capgrpc.load --request-format openengine, request bytes built with the pb2 test double)
end to end; a raw-bytes client for exact bytes, step timing, tier attribution and rejections;
the short switch; start-up refusals (exit 3); frontend mode with a stand-in S-FE module."""

import hashlib
import json
import os
import time

import grpc
import pytest

import oe_double
from capcommon import expect
from capcommon.lifecycle import read_port_file
from capcommon.schedule import BeatSchedule, load_schedule
from capgrpc.reqbytes import encode_pb_string_field
from capstubs.prebaked import Prebaked

from conftest import (METHOD, per_tier, spb_args, table_doc, total, write_json, write_prebaked,
                      write_reqbytes)

MOD = "capstubs.prebaked_server"


def _k2(spec):
    """Every tier of a (kind, ms, rate, kw) spec with tokens_per_msg 2."""
    return [(kind, ms, rate, dict(kw, k=2)) for kind, ms, rate, kw in spec]


def _port(launch, role):
    return read_port_file(launch.tree.port_file(role))[1]


def raw_call(port, body, method=METHOD, timeout=10.0):
    """(messages, receive monotonic ns of each, status code) of one raw-bytes streaming call."""
    msgs, ts = [], []
    with grpc.insecure_channel("127.0.0.1:%d" % port) as ch:
        call = ch.unary_stream(method)(body, timeout=timeout)
        try:
            for m in call:
                ts.append(time.monotonic_ns())
                msgs.append(m)
            code = grpc.StatusCode.OK
        except grpc.RpcError as e:
            code = e.code()
    return msgs, ts, code


def request(rid, n):
    r = oe_double.GenerateRequest(model="m0", prompt="hi")
    if n is not None:
        r.stopping.max_tokens = n
    r.stopping.ignore_eos = True
    return (encode_pb_string_field(1, rid) if rid is not None else b"") + r.SerializeToString()


def wait_until(abs_ns):
    while time.monotonic_ns() < abs_ns:
        time.sleep(0.005)


def start_spb(launch, role, pb_path, *extra, cpus="0"):
    launch.server(role, MOD, spb_args(pb_path, *extra), cpus=cpus)
    launch.wait_file(launch.tree.port_file(role), 20, role)
    return _port(launch, role)


def ready_info(launch, role):
    with open(launch.tree.ready_file(role)) as f:
        return json.load(f)["info"]["workers"][0]["info"]


# ---- reader end to end -------------------------------------------------------------------------

def test_reader_end_to_end_with_schedule(launch):
    launch.write_table(table_doc(_k2([("warmup", 400, 25.0, {"n": 5}), ("drain", 300, 0.0, {}),
                                      ("up", 500, 40.0, {"n": 7}), ("drain", 400, 0.0, {})])))
    pb_path, doc = write_prebaked(launch)
    pb = Prebaked(doc)
    rb = write_reqbytes(launch)
    launch.set_origin(4.0)
    start_spb(launch, "spb", pb_path, "--schedule", launch.sched_path,
              "--inject", "site=send.msg,mode=busy,delta_ns=1000")
    launch.start("load", "capgrpc.load",
                 launch.common("load", cpus="1")
                 + ["--target-port-file", launch.tree.port_file("spb"), "--method", METHOD,
                    "--workers", "1", "--in-process", "--channels-per-worker", "1",
                    "--schedule", launch.sched_path, "--request-format", "openengine",
                    "--request-bytes", rb, "--msg-sample-mod", "1",
                    "--max-records-per-worker", "1000"])
    assert launch.wait_exit("load", 40) == 0, launch.stderr("load")
    assert launch.stop("spb") == 0, launch.stderr("spb")
    ld, sp = launch.summaries("load"), launch.summaries("spb")
    planned = per_tier(ld, "counts.planned")
    assert planned[0] > 0 and planned[2] > 0
    assert per_tier(ld, "counts.ok") == planned and total(ld, "counts.failed") == 0
    # --schedule: every stream under its planned tier, exactly the reader's
    assert {s: c for s, c in per_tier(sp, "counts.streams_accepted").items() if c} == \
        {s: c for s, c in planned.items() if c}
    n_of = {0: 5, 2: 7}
    for seq, n in n_of.items():
        sb = pb.lookup(n, 2, 1)
        m = expect.msgs_per_request(n, 2, 1, "product") + 1
        assert len(sb.msgs) == m
        assert sp[seq][0]["counts"]["msgs_sent"] == planned[seq] * m == ld[seq][0]["msgs"]["msgs"]
        assert sp[seq][0]["counts"]["bytes_sent"] == planned[seq] * sum(map(len, sb.msgs)) == \
            ld[seq][0]["msgs"]["bytes"]
        assert sp[seq][0]["counts"]["finished_sent"] == sp[seq][0]["counts"][
            "streams_completed"] == planned[seq]
        assert sp[seq][0]["counts"]["tokens_sent"] == planned[seq] * n
        assert sp[seq][0]["tier_attribution"] == "schedule"
        assert sp[seq][0]["prebaked"]["sha256"] == hashlib.sha256(
            open(pb_path, "rb").read()).hexdigest()
    streams, heads = launch.records("spb", "streams")
    reqs, _ = launch.records("load", "requests")
    by_rid = dict(zip(reqs["req_id"], reqs["tier_seq"]))
    assert sorted(streams["req_id"]) == sorted(reqs["req_id"])
    for rid, fs, nm, tok, seq in zip(streams["req_id"], streams["first_step"],
                                     streams["n_msgs"], streams["tokens"], streams["tier_seq"]):
        assert seq == by_rid[rid] and fs >= 0 and tok == n_of[seq]
        assert nm == expect.msgs_per_request(n_of[seq], 2, 1, "product") + 1
    assert heads[0]["meta"]["tier_seq"] == "schedule"
    steps, _ = launch.records("spb", "steps")
    assert sum(steps["msgs"]) == total(sp, "counts.msgs_sent") == total(sp, "beat.enqueued")
    # injection heartbeats: send.msg once per message, beat.step once per step
    inj = {}
    for docs in sp.values():
        for e in docs[0]["inject"]:
            h, a = inj.get(e["site"], (0, 0))
            inj[e["site"]] = (h + e["hits_delta"], a + e["applied_delta"])
    assert inj["send.msg"] == (total(sp, "counts.msgs_sent"),) * 2
    assert inj["beat.step"][0] == total(sp, "beat.steps") == len(steps["step"])
    assert inj["short.req"] == (total(sp, "counts.streams_accepted"), 0)


# ---- raw-bytes client -----------------------------------------------------------------------------

def test_exact_bytes_step_timing_and_tier_attribution(launch):
    tt = launch.write_table(table_doc(_k2([("warmup", 1500, 5.0, {"n": 5}),
                                           ("drain", 500, 0.0, {}),
                                           ("up", 1000, 5.0, {"n": 7}),
                                           ("drain", 500, 0.0, {})])))
    pb_path, doc = write_prebaked(launch)
    pb = Prebaked(doc)
    origin = launch.set_origin(4.0)
    p_ev = start_spb(launch, "spb", pb_path, cpus="0")
    p_sc = start_spb(launch, "spbs", pb_path, "--schedule", launch.sched_path, cpus="1")
    rids, _planned, tseq, _meta = load_schedule(launch.sched_path, 0, 1)
    up_rid = next(r for r, t in zip(rids, tseq) if t == 2)
    wait_until(origin + 200 * 1_000_000)  # inside the warmup tier (0)
    got = {}
    for name, port, rid, n in (("ev_text", p_ev, "cmpl-x", 5), ("ev_up", p_ev, str(up_rid), 7),
                               ("sc_up", p_sc, str(up_rid), 7)):
        got[name] = raw_call(port, request(rid, n))
    wait_until(origin + 3100 * 1_000_000)  # tier 2 has a summary only once it has run
    assert launch.stop("spb") == 0 and launch.stop("spbs") == 0
    for name, n in (("ev_text", 5), ("ev_up", 7), ("sc_up", 7)):
        msgs, _ts, code = got[name]
        assert code == grpc.StatusCode.OK and msgs == list(pb.lookup(n, 2, 1).msgs), name
    # without --schedule: the tier at acceptance (0); a non-decimal request_id is req_id -1
    st, _ = launch.records("spb", "streams")
    rows = sorted(zip(st["req_id"], st["tier_seq"], st["first_step"], st["tokens"]))
    assert [(r, t, tok) for r, t, _f, tok in rows] == [(-1, 0, 5), (up_rid, 0, 7)]
    ev = launch.summaries("spb")
    assert ev[0][0]["counts"]["request_id_not_decimal"] == 1
    assert ev[0][0]["counts"]["streams_accepted"] == 2 and ev[0][0]["tier_attribution"] == "event"
    # with --schedule: the planned tier of the req_id (2), whatever the arrival time
    st2, _ = launch.records("spbs", "streams")
    assert list(zip(st2["req_id"], st2["tier_seq"])) == [(up_rid, 2)]
    sc = launch.summaries("spbs")
    assert sc[2][0]["counts"]["streams_accepted"] == 1 and sc[0][0]["counts"][
        "streams_accepted"] == 0
    # no message before its planned step; the finished message comes on the last token step
    bs = BeatSchedule(tt)
    for (r, _t, first, _tok), name in zip(rows, ("ev_text", "ev_up")):
        n = 7 if r == up_rid else 5
        sb = pb.lookup(n, 2, 1)
        _msgs, ts, _code = got[name]
        assert len(ts) == len(sb.offs) and sb.offs[-1] == sb.offs[-2]
        for j, t in enumerate(ts):
            assert t >= origin + bs.planned_rel_ns(first + sb.offs[j]), (name, j)
        # every message of a step is sent within that step: the stream lasts about its steps
        assert ts[-1] < origin + bs.planned_rel_ns(first + sb.offs[-1]) + 200 * 1_000_000


def test_rejections_and_other_methods(launch):
    launch.write_table(table_doc(_k2([("warmup", 3000, 5.0, {"n": 5}),
                                      ("drain", 500, 0.0, {})])))
    pb_path, _doc = write_prebaked(launch)
    launch.set_origin(3.0)
    port = start_spb(launch, "spb", pb_path, "--schedule", launch.sched_path)
    wait_until(launch.origin + 100 * 1_000_000)
    cases = [(b"\x0b", grpc.StatusCode.INVALID_ARGUMENT),          # malformed
             (request("1", None), grpc.StatusCode.INVALID_ARGUMENT),  # no max_tokens
             (request("1", 0), grpc.StatusCode.INVALID_ARGUMENT),     # max_tokens 0
             (request("1", 6), grpc.StatusCode.INVALID_ARGUMENT),     # no prebaked stream
             (request("cmpl-1", 5), grpc.StatusCode.INVALID_ARGUMENT),
             (request("999999", 5), grpc.StatusCode.INVALID_ARGUMENT),
             (request("1", 5), grpc.StatusCode.OK)]
    for body, want in cases:
        assert raw_call(port, body)[2] == want, body
    msgs, _ts, code = raw_call(port, request("1", 5), method="/openengine.v1.Control/GetLoad")
    assert code == grpc.StatusCode.UNIMPLEMENTED and msgs == []
    assert launch.stop("spb") == 0
    sp = launch.summaries("spb")
    rej = {}
    for docs in sp.values():
        for k, v in docs[0]["rejects"].items():
            rej[k] = rej.get(k, 0) + v
    assert rej == {"malformed": 1, "no_max_tokens": 2, "no_prebaked_stream": 1,
                   "request_id_not_in_schedule": 2}
    assert total(sp, "counts.streams_rejected") == 6
    assert total(sp, "counts.streams_accepted") == total(sp, "counts.streams_completed") == 1
    inj = [e for e in sp[0][0]["inject"] if e["site"] == "short.req"]
    assert inj[0]["hits_delta"] == 1  # the short switch is consulted for accepted requests only


# ---- short switch -----------------------------------------------------------------------------

def test_short_switch(launch):
    launch.write_table(table_doc([("warmup", 600, 30.0, {"n": 6}), ("drain", 400, 0.0, {})]))
    pb_path, doc = write_prebaked(launch, short_tokens=(2,))
    pb = Prebaked(doc)
    rb = write_reqbytes(launch)
    launch.set_origin(4.0)
    start_spb(launch, "spb", pb_path, "--schedule", launch.sched_path,
              "--inject", "site=short.req,mode=short,short_tokens=2,every_n=2")
    launch.start("load", "capgrpc.load",
                 launch.common("load", cpus="1")
                 + ["--target-port-file", launch.tree.port_file("spb"), "--method", METHOD,
                    "--workers", "1", "--in-process", "--channels-per-worker", "1",
                    "--schedule", launch.sched_path, "--request-format", "openengine",
                    "--request-bytes", rb, "--msg-sample-mod", "1",
                    "--max-records-per-worker", "1000"])
    assert launch.wait_exit("load", 40) == 0, launch.stderr("load")
    assert launch.stop("spb") == 0, launch.stderr("spb")
    sp, ld = launch.summaries("spb"), launch.summaries("load")
    acc = total(sp, "counts.streams_accepted")
    short = total(sp, "counts.short_requests")
    assert acc == total(ld, "counts.ok") > 2 and short == (acc + 1) // 2
    assert total(sp, "counts.tokens_sent") == 6 * acc - 2 * short
    st, _ = launch.records("spb", "streams")
    assert sorted(st["tokens"]) == sorted([4] * short + [6] * (acc - short))
    full, cut = pb.lookup(6, 1, 1), pb.lookup(4, 1, 1)
    assert total(ld, "msgs.msgs") == total(sp, "counts.msgs_sent") == \
        len(cut.msgs) * short + len(full.msgs) * (acc - short)
    e = [x for x in sp[0][0]["inject"] if x["site"] == "short.req"][0]
    assert e["hits_delta"] == acc and e["applied_delta"] == short == e["expected_applied_delta"]


# ---- start-up refusals ------------------------------------------------------------------------

STANDIN_IMPORT_ERROR = "import tensorrt_llm_not_in_this_venv_capkit_test\n"


def _refusal_setup(launch, case):
    """(server extra args, expected stderr fragment) for one refusal case."""
    k_mixed = case == "mixed_k"
    launch.write_table(table_doc([("warmup", 300, 10.0, {"n": 5}), ("drain", 300, 0.0, {}),
                                  ("up", 300, 10.0, {"n": 7, "k": 2 if k_mixed else 1}),
                                  ("drain", 300, 0.0, {})]))
    if case in ("mixed_k", "missing_n"):
        from capstubs.prebaked import build_prebaked
        from capcommon.tiers import TierTable
        other = launch.tmp / "other.json"
        other.write_text(json.dumps(table_doc([("warmup", 300, 10.0, {"n": 5}),
                                               ("drain", 300, 0.0, {})])))
        p = write_json(launch.tmp / "pb5.json",
                       build_prebaked(oe_double, TierTable.load(str(other)), [1], 0))
        return spb_args(p), ("one (tokens_per_msg, stream_interval)" if k_mixed
                             else "has no stream for")
    pb_path, _doc = write_prebaked(launch)
    base = spb_args(pb_path)
    if case == "missing_short":
        return base + ["--inject", "site=short.req,mode=short,short_tokens=3"], \
            "has no stream for"
    if case == "unknown_site":
        return base + ["--inject", "site=recv.msg,mode=busy,delta_ns=5"], "unknown to S-PB"
    if case == "wrong_mode":
        return base + ["--inject", "site=short.req,mode=busy,delta_ns=5"], "does not support"
    if case == "workers2":
        return ["--workers", "2", "--prebaked", pb_path], "--workers must be 1"
    if case == "sfe_option_alone":
        return base + ["--vocab-size", "5"], "need --sfe-module"
    if case == "sfe_unimportable":
        f = launch.tmp / "fake_openengine.py"
        f.write_text(STANDIN_IMPORT_ERROR)
        return base + ["--sfe-module", str(f)], "cannot be imported here"
    if case == "sfe_missing":
        return base + ["--sfe-module", str(launch.tmp / "nope.py")], "cannot read --sfe-module"
    if case == "prebaked_missing":
        return spb_args(str(launch.tmp / "nope.json")), "cannot read prebaked file"
    if case == "schedule_other_table":
        from capcommon.schedule import write_schedule
        from capcommon.tiers import TierTable
        other = launch.tmp / "other.json"
        other.write_text(json.dumps(table_doc([("warmup", 300, 10.0, {"n": 5}),
                                               ("drain", 300, 0.0, {})])))
        sp = str(launch.tmp / "other.sched")
        write_schedule(sp, TierTable.load(str(other)), 1000)
        return base + ["--schedule", sp], "was generated from tier table"
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["mixed_k", "missing_n", "missing_short", "unknown_site",
                                  "wrong_mode", "workers2", "sfe_option_alone",
                                  "sfe_unimportable", "sfe_missing", "prebaked_missing",
                                  "schedule_other_table"])
def test_refused_at_start_up(launch, case):
    extra, why = _refusal_setup(launch, case)
    launch.set_origin(5.0)
    launch.server("spb", MOD, extra)
    assert launch.wait_exit("spb", 30) == 3, launch.stderr("spb")
    assert why in launch.stderr("spb"), launch.stderr("spb")
    assert not os.path.exists(launch.tree.port_file("spb"))


# ---- frontend mode (stand-in S-FE module) ------------------------------------------------------

# The names S-PB takes from fake_openengine.py (probe-tree tests/unittest/grpc/openengine_client):
# _parse_args, _profile_from_args, FakeEngine(profile, tokens_per_step=, step_delay=, record=),
# _Control(engine), _Health(engine), openengine_pb2_grpc.add_ControlServicer_to_server and
# health_pb2_grpc.add_HealthServicer_to_server. The real module needs the product package, so this
# stand-in answers Control/GetServerInfo with what S-PB handed it, as raw bytes.
STANDIN = '''
import argparse
import dataclasses
import json
import types

import grpc


@dataclasses.dataclass
class Prof:
    fields: dict


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="trtllm-like")
    p.add_argument("--model")
    p.add_argument("--max-context-length", type=int)
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--eos-token-ids", type=int, nargs="+")
    p.add_argument("--profile-json")
    return p.parse_args(argv)


def _profile_from_args(args):
    return Prof(dict(vars(args)))


class FakeEngine:
    def __init__(self, profile, tokens_per_step=1, step_delay=0.0, record=True):
        self.profile, self.knobs = profile, [tokens_per_step, step_delay, record]


class _Control:
    def __init__(self, engine):
        self.engine = engine


class _Health:
    def __init__(self, engine):
        self.engine = engine


class _Routes(grpc.GenericRpcHandler):
    def __init__(self, routes):
        self.routes = routes

    def service(self, details):
        return self.routes.get(details.method)


def _unary(fn):
    return grpc.unary_unary_rpc_method_handler(fn, request_deserializer=None,
                                               response_serializer=None)


def _add_control(servicer, server):
    async def info(request, context):
        e = servicer.engine
        return json.dumps({"profile": e.profile.fields, "knobs": e.knobs}).encode()
    server.add_generic_rpc_handlers([_Routes({"/openengine.v1.Control/GetServerInfo":
                                              _unary(info)})])


def _add_health(servicer, server):
    async def check(request, context):
        return b"SERVING"
    server.add_generic_rpc_handlers([_Routes({"/grpc.health.v1.Health/Check": _unary(check)})])


openengine_pb2_grpc = types.SimpleNamespace(add_ControlServicer_to_server=_add_control)
health_pb2_grpc = types.SimpleNamespace(add_HealthServicer_to_server=_add_health)
'''


def _unary_call(port, method):
    with grpc.insecure_channel("127.0.0.1:%d" % port) as ch:
        return ch.unary_unary(method)(b"", timeout=10)


def test_frontend_mode_registers_control_and_health(launch):
    launch.write_table(table_doc(_k2([("warmup", 2000, 5.0, {"n": 5}),
                                      ("drain", 300, 0.0, {})])))
    pb_path, doc = write_prebaked(launch)
    mod = launch.tmp / "fake_openengine.py"
    mod.write_text(STANDIN)
    launch.set_origin(3.0)
    profile_json = json.dumps({"bos_token_id": 3, "fingerprint": {"x": 1}})
    port = start_spb(launch, "spb", pb_path, "--sfe-module", str(mod), "--vocab-size", "151",
                     "--eos-token-ids", "1", "2", "--profile-json", profile_json,
                     "--model", "m1")
    wait_until(launch.origin + 100 * 1_000_000)
    got = json.loads(_unary_call(port, "/openengine.v1.Control/GetServerInfo"))
    assert got["profile"] == {"profile": "trtllm-like", "model": "m1", "max_context_length": None,
                              "vocab_size": 151, "eos_token_ids": [1, 2],
                              "profile_json": profile_json}
    assert got["knobs"] == [1, 0.0, False]  # record=False: no per-request recording
    assert _unary_call(port, "/grpc.health.v1.Health/Check") == b"SERVING"
    msgs, _ts, code = raw_call(port, request("1", 5))
    assert code == grpc.StatusCode.OK and msgs == list(Prebaked(doc).lookup(5, 2, 1).msgs)
    assert launch.stop("spb") == 0
    sha = hashlib.sha256(mod.read_bytes()).hexdigest()
    ctl = ready_info(launch, "spb")["control"]
    assert ctl["sha256"] == sha and ctl["sfe_module"] == str(mod)
    assert ctl["sfe_argv"] == ["--model", "m1", "--vocab-size", "151", "--eos-token-ids", "1",
                               "2", "--profile-json", profile_json]
    assert launch.summaries("spb")[0][0]["control"]["sha256"] == sha


def test_frontend_mode_refuses_options_the_sfe_parser_rejects(launch):
    launch.write_table(table_doc([("warmup", 300, 10.0, {"n": 5}), ("drain", 300, 0.0, {})]))
    pb_path, _doc = write_prebaked(launch)
    mod = launch.tmp / "fake_openengine.py"
    mod.write_text(STANDIN)
    launch.set_origin(5.0)
    launch.server("spb", MOD, spb_args(pb_path, "--sfe-module", str(mod), "--vocab-size", "x"))
    assert launch.wait_exit("spb", 30) == 3, launch.stderr("spb")
    assert "rejected by" in launch.stderr("spb")
