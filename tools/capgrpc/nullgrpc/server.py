# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nullgrpc: multi-process grpc.aio server answering every method with prebaked bytes.

Requests are capkit gRPC requests (capcommon.wire.GRPC_REQ header + prompt bytes); the number of
messages and their beat steps come from the request's own N, k, s through capcommon.expect with
the product gate (never from the tier table at arrival time, INTERFACES.md 4.4).

- --pacing asap: every message is yielded as fast as gRPC takes it (no beat; X2 not applicable).
- --pacing beat: the absolute-beat source. One BeatSchedule for the whole process; a beat driver
  wakes at every planned step and appends due messages to per-stream unbounded pending queues
  without waiting for any write to complete; message j of a stream is queued on step
  first_step + msg_step_offsets[j] (INTERFACES.md 3.4). Self-reports wake lag, late steps,
  pending depth and the yielded count. --stamp puts (step, planned_rel_ns) at the head of every
  message (INTERFACES.md 7.4).
"""

import argparse
import asyncio
import collections
import sys

import grpc

from capcommon import expect
from capcommon.aio import call_at_ns, run as aio_run
from capcommon.capbin import RecordArray
from capcommon.cli import add_common_args, common_from_args
from capcommon.clock import now_ns
from capcommon.exitcodes import OK, CapkitError, ConfigError
from capcommon.lifecycle import StopController, write_port_file
from capcommon.schedule import STEP_STAMP, BeatSchedule, count_schedule_rows
from capcommon.tiers import POST_SEQ
from capcommon.wire import WireError, unpack_grpc_request

from capgrpc.common import (UNREQUESTED_OPTION, AnyMethodHandler, ToolRuntime, add_server_args,
                            bind_server, child_main, make_server, reserve_port,
                            resolve_max_unrequested, run_supervisor, worker_ready_info,
                            write_inprocess_ready)

from . import SITES, SLOTS, TOOL

STREAM_FIELDS = [("req_id", "q"), ("accept_rel_ns", "q"), ("first_step", "q"), ("n_msgs", "i"),
                 ("tokens", "i"), ("tier_seq", "h")]
STEP_FIELDS = [("step", "q"), ("planned_rel_ns", "q"), ("actual_rel_ns", "q"), ("msgs", "i"),
               ("tier_seq", "h")]
GATE = "product"


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capgrpc.nullgrpc", description=__doc__.split("\n")[0])
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=True)
    g = p.add_argument_group("nullgrpc")
    g.add_argument("--host", required=True)
    g.add_argument("--port", type=int, required=True, help="0 = any free port")
    g.add_argument("--workers", type=int, required=True)
    g.add_argument("--in-process", action="store_true",
                   help="serve from this process; needs --workers 1")
    g.add_argument("--msg-bytes", type=int, required=True)
    g.add_argument("--pacing", choices=("beat", "asap"), required=True)
    g.add_argument("--stamp", action="store_true", help="beat stamp at the head of each message")
    g.add_argument("--lag-period-us", type=int, required=True)
    g.add_argument("--max-stream-records", type=int,
                   help="streams capacity per process (default: all requests the tier table "
                        "can generate, count_schedule_rows)")
    g.add_argument("--lag-records-capacity", type=int, default=0)
    add_server_args(g)
    return p


def validate_args(args, common):
    if args.workers < 1:
        raise ConfigError("--workers must be >= 1")
    if args.in_process and args.workers != 1:
        raise ConfigError("--in-process needs --workers 1")
    if args.pacing == "beat" and args.workers != 1:
        raise ConfigError("--pacing beat needs --workers 1 (one beat for the whole source)")
    if args.stamp and args.pacing != "beat":
        raise ConfigError("--stamp needs --pacing beat")
    if args.msg_bytes < (STEP_STAMP.size if args.stamp else 0):
        raise ConfigError("--msg-bytes too small%s" % (" for the stamp" if args.stamp else ""))
    if not 0 <= args.port < 65536:
        raise ConfigError("--port out of range")
    if len(common.cpus) < args.workers:
        raise ConfigError("--cpus lists %d cpus for %d workers" % (len(common.cpus), args.workers))
    args.grpc_server_options = resolve_max_unrequested(args.max_unrequested_s,
                                                       common.load_tier_table())


class _Stream:
    __slots__ = ("rid", "m", "offs", "first", "q", "ev", "dead", "truncated", "sent")

    def __init__(self, rid, m, offs):
        self.rid, self.m, self.offs = rid, m, offs
        self.first = -1
        self.q = collections.deque()
        self.ev = asyncio.Event()
        self.dead = False
        self.truncated = False
        self.sent = 0


class NullServer:
    """One serving process (a worker of the group, or the in-process server)."""

    def __init__(self, args, rt, link, bind_port):
        self.args, self.rt, self.link = args, rt, link
        self.tt = rt.tt
        self.origin = rt.origin_ns
        self.bind_port = bind_port
        self.beat = args.pacing == "beat"
        self.stamp = args.stamp
        self.msg_bytes = args.msg_bytes
        self.payload = b"x" * args.msg_bytes
        self.pad = b"x" * (args.msg_bytes - STEP_STAMP.size) if self.stamp else b""
        nt = len(self.tt)
        z = lambda: [0] * nt  # noqa: E731
        self.c_accepted, self.c_completed, self.c_cancelled = z(), z(), z()
        self.c_truncated, self.c_rejected, self.c_bumped = z(), z(), z()
        self.c_msgs, self.c_bytes = z(), z()
        self.max_pending = z()
        self.request_types = [dict() for _ in range(len(self.tt))]  # by event tier
        self.rejects = [dict() for _ in range(len(self.tt))]  # by event tier
        self._shapes = {}
        cap = args.max_stream_records
        if cap is None:
            cap = count_schedule_rows(self.tt)
        self.stream_rec = RecordArray(STREAM_FIELDS, cap)
        self.bs = BeatSchedule(self.tt) if self.beat else None
        self.step_rec = RecordArray(STEP_FIELDS, self.bs.n_steps) if self.beat else None
        if self.beat and self.bs.n_steps == 0:
            raise ConfigError("--pacing beat but no tier has beat_ns > 0")
        self.due = {}
        self.next_step = 0
        v = rt.registry.values
        self.v = v
        pos = {d["name"]: d["offset"] for d in rt.registry.slots}
        self.i_active = pos["backlog.active_streams"]
        self.i_pending = pos["backlog.pending_msgs"]
        self.i_yielded = pos["backlog.yielded"]
        self.i_acc = pos["self.accepted"]
        self.i_done = pos["self.completed"]
        self.i_canc = pos["self.cancelled"]
        self.i_bytes = pos["bytes.sent"]
        self.i_steps = pos["count.beat_steps"]
        self.i_late = pos["count.beat_late_steps"]
        self.i_enq = pos["count.beat_enqueued"]
        self.h_wake = rt.slot_refs["lag.beat_wake"]
        self.inj_send = rt.injectors["send.msg"]
        self.inj_step = rt.injectors["beat.step"]
        self.stop_ev = None
        self.stopping = False
        self._wake = None
        self.first_type_printed = False

    def shape(self, n, k, s):
        key = (n, k, s)
        sh = self._shapes.get(key)
        if sh is None:
            toks = expect.msg_tokens(n, k, s, GATE)
            cum = [0]
            for t in toks:
                cum.append(cum[-1] + t)
            sh = (len(toks), expect.msg_step_offsets(n, k, s, GATE), cum)
            self._shapes[key] = sh
        return sh

    # -- request handling ------------------------------------------------------------------------
    def behavior_for(self, method):
        async def behavior(request, context):
            tn = type(request).__name__
            ev_seq = self.rt.cur_seq  # event time: the planned tier is not parsed yet
            ev_seq = len(self.tt) - 1 if ev_seq == POST_SEQ else max(ev_seq, 0)
            types = self.request_types[ev_seq]
            types[tn] = types.get(tn, 0) + 1
            if not self.first_type_printed:
                self.first_type_printed = True
                print("nullgrpc: first request object type = %s (method %s)" % (tn, method),
                      file=sys.stderr)
            try:
                if type(request) is not bytes:
                    raise WireError("request object is %s, not bytes" % tn)
                rid, n, k, s, seq, _prompt = unpack_grpc_request(request)
                if not 0 <= seq < len(self.tt):
                    raise WireError("tier_seq %d outside the tier table" % seq)
                m, offs, cum = self.shape(n, k, s)
            except (WireError, expect.ExpectError) as e:
                why = str(e)
                rej = self.rejects[ev_seq]  # rejected requests have no planned tier: event time
                rej[why] = rej.get(why, 0) + 1
                self.c_rejected[ev_seq] += 1
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, why)
                return
            accept = now_ns() - self.origin
            v = self.v
            v[self.i_acc] += 1
            v[self.i_active] += 1
            self.c_accepted[seq] += 1
            sent = 0
            first = -1
            outcome = "cancelled"
            try:
                if self.beat:
                    st = _Stream(rid, m, offs)
                    first = self.register(st, accept, seq)
                    if first is None:
                        outcome = "truncated"
                        await context.abort(grpc.StatusCode.OUT_OF_RANGE,
                                            "beat schedule ended before the first step")
                        return
                    while sent < m:
                        q = st.q
                        if not q:
                            if st.truncated:
                                outcome = "truncated"
                                await context.abort(grpc.StatusCode.OUT_OF_RANGE,
                                                    "beat schedule ended mid-stream")
                                return
                            st.ev.clear()
                            await st.ev.wait()
                            continue
                        msg = q.popleft()
                        v[self.i_pending] -= 1
                        self.inj_send.hit()
                        v[self.i_yielded] += 1
                        v[self.i_bytes] += len(msg)
                        self.c_msgs[seq] += 1
                        self.c_bytes[seq] += len(msg)
                        sent += 1
                        st.sent = sent
                        yield msg
                else:
                    payload, nb = self.payload, self.msg_bytes
                    for _ in range(m):
                        self.inj_send.hit()
                        v[self.i_yielded] += 1
                        v[self.i_bytes] += nb
                        self.c_msgs[seq] += 1
                        self.c_bytes[seq] += nb
                        sent += 1
                        yield payload
                outcome = "completed"
            finally:
                if self.beat:
                    st.dead = True
                    v[self.i_pending] -= len(st.q)
                    st.q.clear()
                v[self.i_active] -= 1
                if outcome == "completed":
                    v[self.i_done] += 1
                    self.c_completed[seq] += 1
                elif outcome == "truncated":
                    self.c_truncated[seq] += 1
                else:
                    v[self.i_canc] += 1
                    self.c_cancelled[seq] += 1
                self.stream_rec.append(rid, accept, first, sent, cum[sent], seq)

        return behavior

    def register(self, st, accept_rel, seq):
        """Give a new beat stream its first step and queue its first message slot."""
        first = self.bs.step_at_or_after(accept_rel)
        if first is None:
            return None
        if first < self.next_step:  # the driver already passed that step (accept at its instant)
            first = self.next_step
            self.c_bumped[seq] += 1
        st.first = first
        step0 = first + st.offs[0]
        if step0 >= self.bs.n_steps:
            return None
        self.due.setdefault(step0, []).append((st, 0))
        return first

    # -- beat driver -------------------------------------------------------------------------------
    async def _sleep_until(self, loop, abs_ns):
        if self.stopping:
            return False
        if abs_ns <= now_ns():
            return True
        fut = loop.create_future()
        self._wake = fut
        h = call_at_ns(loop, abs_ns, lambda f: f.done() or f.set_result(None), fut)
        try:
            await fut
        finally:
            h.cancel()
            self._wake = None
        return not self.stopping

    async def beat_driver(self, loop):
        bs, origin, v = self.bs, self.origin, self.v
        n = bs.n_steps
        due, rec = self.due, self.step_rec
        stamp, pad = self.stamp, self.pad
        pack = STEP_STAMP.pack
        payload = self.payload
        step = 0
        while step < n:
            prel = bs.planned_rel_ns(step)
            if not await self._sleep_until(loop, origin + prel):
                break
            t = now_ns()
            rel = t - origin
            self.h_wake.add(rel - prel)
            self.inj_step.hit()
            seq = bs.tier_of_step(step)
            if step + 1 < n and rel >= bs.planned_rel_ns(step + 1):
                v[self.i_late] += 1
            nq = 0
            entries = due.pop(step, None)
            if entries:
                msg = pack(step, prel) + pad if stamp else payload
                for st, j in entries:
                    if st.dead:
                        continue
                    st.q.append(msg)
                    st.ev.set()
                    nq += 1
                    if j + 1 < st.m:
                        nxt = st.first + st.offs[j + 1]
                        if nxt >= n:
                            st.truncated = True
                        else:
                            due.setdefault(nxt, []).append((st, j + 1))
                v[self.i_pending] += nq
                v[self.i_enq] += nq
                if v[self.i_pending] > self.max_pending[seq]:
                    self.max_pending[seq] = v[self.i_pending]
            self.next_step = step + 1
            v[self.i_steps] += 1
            rec.append(step, prel, rel, nq, seq)
            step += 1
        self.next_step = max(self.next_step, step)

    # -- main --------------------------------------------------------------------------------------
    def on_stop(self, hard):
        self.stopping = True
        w = self._wake
        if w is not None and not w.done():
            w.set_result(None)
        if self.stop_ev is not None:
            self.stop_ev.set()

    async def run(self, loop):
        rt, args = self.rt, self.args
        self.stop_ev = asyncio.Event()
        stopper = StopController()
        stopper.attach_loop(loop)
        stopper.on_stop(self.on_stop)
        if self.link is not None:
            self.link.attach(loop, stopper)
        opts = args.grpc_server_options
        server = make_server(reuseport=True, max_unrequested_s=opts[UNREQUESTED_OPTION])
        server.add_generic_rpc_handlers([AnyMethodHandler(self.behavior_for)])
        port = bind_server(server, args.host, self.bind_port)
        await server.start()
        info = {"host": args.host, "port": port, "pacing": args.pacing, "stamp": self.stamp,
                "msg_bytes": self.msg_bytes, "gate": GATE,
                "stream_capacity": self.stream_rec.capacity,
                "beat_steps": self.bs.n_steps if self.beat else None,
                "serializers": "identity (request_deserializer=None, response_serializer=None)",
                "grpc_server_options": dict(opts)}
        rt.start(loop, info)
        driver = loop.create_task(self.beat_driver(loop)) if self.beat else None
        if self.link is not None:
            self.link.ready(worker_ready_info(rt, info))
        else:
            write_port_file(args.port_file, rt.identity, args.host, port)
            write_inprocess_ready(rt, args.ready_file, info)
        if not self.stopping:
            await self.stop_ev.wait()
        if driver is not None:
            driver.cancel()
            await asyncio.gather(driver, return_exceptions=True)
        await server.stop(args.stop_grace_s)
        rt.close_tiers()
        rt.read_stop()
        self.write_outputs()
        rt.stop()
        return OK

    def write_outputs(self):
        rt, args = self.rt, self.args
        rt.dump_records(self.stream_rec, "streams", "server.streams",
                        {"pacing": args.pacing, "gate": GATE,
                         "tier_seq": "planned tier carried by the request"})
        if self.step_rec is not None:
            rt.dump_records(self.step_rec, "steps", "beat.steps",
                            {"n_steps": self.bs.n_steps, "stamp": self.stamp})
        rt.dump_lag()
        for seq in sorted(rt.end_marks):
            a, b = rt.start_marks[seq], rt.end_marks[seq]
            payload = rt.tier_common(seq)
            payload.update({
                "schedule_sha256": None, "pacing": args.pacing, "stamp": self.stamp,
                "msg_bytes": self.msg_bytes, "gate": GATE, "in_process": bool(args.in_process),
                "grpc_server_options": dict(args.grpc_server_options),
                "counts": {"streams_accepted": self.c_accepted[seq],
                           "streams_completed": self.c_completed[seq],
                           "streams_cancelled": self.c_cancelled[seq],
                           "streams_truncated": self.c_truncated[seq],
                           "streams_rejected": self.c_rejected[seq],
                           "first_step_bumped": self.c_bumped[seq],
                           "msgs_sent": self.c_msgs[seq], "bytes_sent": self.c_bytes[seq],
                           "attribution": "planned tier carried by the request "
                                          "(streams_rejected: event tier)"},
                "beat": None if not self.beat else {
                    "steps": rt.slot_delta("count.beat_steps", a["vals"], b["vals"]),
                    "late_steps": rt.slot_delta("count.beat_late_steps", a["vals"], b["vals"]),
                    "enqueued": rt.slot_delta("count.beat_enqueued", a["vals"], b["vals"]),
                    "max_pending": self.max_pending[seq],
                    "wake_lag_ns": rt.hist_summary("lag.beat_wake", a["vals"], b["vals"]),
                    "attribution": "event time (step planned time)"},
                "yielded_event_time": rt.slot_delta("backlog.yielded", a["vals"], b["vals"]),
                "request_types": dict(self.request_types[seq]),
                "rejects": dict(self.rejects[seq]),
                "overflow": {"streams": self.stream_rec.overflow,
                             "steps": self.step_rec.overflow if self.step_rec else 0,
                             "lag": rt.lag_records.overflow if rt.lag_records else 0}})
            rt.write_summary(seq, payload)


# ---- entry points ------------------------------------------------------------------------------

def worker_run(args, worker, link):
    common = common_from_args(args)
    rt = ToolRuntime(common, TOOL, SLOTS, SITES, worker, args.workers, [common.cpus[worker]],
                     args.lag_period_us, args.lag_records_capacity)
    srv = NullServer(args, rt, link, args.bind_port)
    return aio_run(srv.run, common.loop)


def child_entry(args_dict, worker, conn):
    child_main(args_dict, worker, conn, worker_run)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        validate_args(args, common)
        if args.in_process:
            args.bind_port = args.port
            return worker_run(args, 0, None)
        sock, port = reserve_port(args.host, args.port)
        args.bind_port = port

        def on_all_ready(ident):
            write_port_file(args.port_file, ident, args.host, port)
            sock.close()

        try:
            return run_supervisor(args, common, args.workers, child_entry, TOOL, SLOTS,
                                  on_all_ready=on_all_ready)
        finally:
            sock.close()
    except CapkitError as e:
        print("%s: %s" % (TOOL, e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
