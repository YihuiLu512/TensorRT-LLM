# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capgrpc: multi-process grpc.aio open-loop load generator and stamped drain.

- Absolute schedule: worker k sends the schedule rows with req_id % workers == k at their planned
  times (one dispatcher per worker; late sends are sent, never skipped; latency columns start from
  the planned time and the actual send time is recorded beside it).
- Raw bytes: response_deserializer=None; every received message's type is checked against bytes
  (a non-bytes message ends the request as protocol_error), and summary.recv_types counts the
  type of each stream's first message, so "we really get raw bytes" is data, not assumption.
- Channels: --channels-per-worker N grpc.aio channels per worker, each with
  grpc.use_local_subchannel_pool=1 (own TCP connection) unless --shared-subchannel-pool (control).
- --stamped: every message starts with the beat stamp (step, planned_rel_ns); the drain records
  message-level lag = receive time - stamped planned time (INTERFACES.md 7.4).
- --planned-msgs: for unstamped streams (the OpenEngine stubs), message-level lag against the
  planned time derived from the schedule (capgrpc.msgplan; INTERFACES errata FT1-9).
- --in-process: run the single worker in this process (weak-reader mode with
  --workers 1 --channels-per-worker 1); otherwise a supervisor process spawns the workers.
  capgrpc.oe_sink subclasses LoadWorker / RequestPlan and takes the same reader arguments
  (add_reader_args), so a scenario swaps the two readers by the module name and the oe_sink-only
  flags.
No disk writes until the end: records go to preallocated RecordArrays, dumped per tier at exit.
"""

import argparse
import asyncio
import struct
import sys

import grpc

from capcommon import expect, hist as _hist
from capcommon.aio import call_at_ns, run as aio_run
from capcommon.capbin import RecordArray
from capcommon.cli import add_common_args, common_from_args
from capcommon.clock import now_ns
from capcommon.exitcodes import OK, STOPPED_EARLY, CapkitError, ConfigError
from capcommon.hashing import sha256_file
from capcommon.lifecycle import StopController
from capcommon.schedule import STEP_STAMP, load_schedule, per_worker_tier_counts, prompt_index
from capcommon.wire import GRPC_REQ

from . import SITES, SLOTS, TOOL
from .common import (ToolRuntime, TierCursor, child_main, open_channel, raw_stream_callable,
                     resolve_target, run_supervisor, status_reason, wait_channels_ready,
                     worker_ready_info, write_inprocess_ready)
from .msgplan import MsgPlan
from .reqbytes import encode_pb_string_field, load_prompts, load_reqbytes

REQUEST_FORMATS = ("capkit", "openengine", "raw")
REQ_FIELDS = [("req_id", "q"), ("planned_rel_ns", "q"), ("sent_rel_ns", "q"),
              ("first_rel_ns", "q"), ("done_rel_ns", "q"), ("frames", "i"), ("bytes", "q"),
              ("tokens", "i"), ("status", "h"), ("conn_id", "i"), ("new_conn", "b"),
              ("tier_seq", "h")]
MSG_FIELDS = [("stream_id", "q"), ("idx", "i"), ("step", "q"), ("planned_rel_ns", "q"),
              ("recv_rel_ns", "q"), ("tier_seq", "h")]
CLASS_HINTS = {"timeout_registered": "overload", "protocol_error": "non_overload",
               "engine_error": "non_overload"}


# ---- command line -----------------------------------------------------------------------------

def add_reader_args(g):
    """The reader arguments capgrpc.load and capgrpc.oe_sink share (same names and meaning)."""
    g.add_argument("--target", help="HOST:PORT (exclusive with --target-port-file)")
    g.add_argument("--target-port-file", help="port file of the server role")
    g.add_argument("--method", required=True, help="gRPC method path, e.g. /pkg.Service/Method")
    g.add_argument("--workers", type=int, required=True)
    g.add_argument("--in-process", action="store_true",
                   help="run the (single) worker in this process; needs --workers 1")
    g.add_argument("--channels-per-worker", type=int, required=True)
    g.add_argument("--schedule", required=True)
    g.add_argument("--request-format", required=True, choices=REQUEST_FORMATS)
    g.add_argument("--prompts", help="prompt pool (capkit format)")
    g.add_argument("--request-bytes", help="request-bytes file (openengine / raw format)")
    g.add_argument("--planned-msgs", action="store_true",
                   help="message-level lag against the planned time derived from the schedule "
                        "(unstamped streams; capgrpc.msgplan)")
    g.add_argument("--msg-sample-mod", type=int, required=True,
                   help="record msgs rows for streams with req_id %% M == 0")
    g.add_argument("--lag-period-us", type=int, required=True)
    g.add_argument("--max-records-per-worker", type=int, required=True)
    g.add_argument("--max-msg-records-per-worker", type=int,
                   help="msgs capacity; derived exactly from the schedule for capkit requests")
    g.add_argument("--lag-records-capacity", type=int, default=0,
                   help="per-sample loop-lag rows (0 = histogram only)")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capgrpc.load", description=__doc__.split("\n")[0])
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=False)
    g = p.add_argument_group("capgrpc.load")
    add_reader_args(g)
    g.add_argument("--shared-subchannel-pool", action="store_true",
                   help="CONTROL ONLY: do not set grpc.use_local_subchannel_pool, so the "
                        "channels of a worker share one connection")
    g.add_argument("--stamped", action="store_true",
                   help="drain mode: read the beat stamp of every message (INTERFACES.md 7.4)")
    return p


def validate_args(args, common):
    if args.workers < 1:
        raise ConfigError("--workers must be >= 1")
    if args.in_process and args.workers != 1:
        raise ConfigError("--in-process needs --workers 1")
    if args.channels_per_worker < 1:
        raise ConfigError("--channels-per-worker must be >= 1")
    if args.msg_sample_mod < 1:
        raise ConfigError("--msg-sample-mod must be >= 1")
    if args.max_records_per_worker < 1:
        raise ConfigError("--max-records-per-worker must be >= 1")
    if args.lag_records_capacity < 0:
        raise ConfigError("--lag-records-capacity must be >= 0")
    if (args.target is None) == (args.target_port_file is None):
        raise ConfigError("give exactly one of --target and --target-port-file")
    if not args.method.startswith("/") or args.method.count("/") != 2:
        raise ConfigError("--method must look like /pkg.Service/Method")
    if args.request_format == "capkit":
        if not args.prompts or args.request_bytes:
            raise ConfigError("--request-format capkit needs --prompts (and no --request-bytes)")
    elif not args.request_bytes or args.prompts:
        raise ConfigError("--request-format %s needs --request-bytes (and no --prompts)"
                          % args.request_format)
    if args.planned_msgs:
        if args.stamped:
            raise ConfigError("--planned-msgs and --stamped exclude each other")
        if args.request_format == "raw":
            raise ConfigError("--planned-msgs needs the tier's max_tokens in every request; "
                              "raw request bytes carry their own")
    if len(common.cpus) < args.workers:
        raise ConfigError("--cpus lists %d cpus for %d workers" % (len(common.cpus), args.workers))


# ---- per-worker request plan ------------------------------------------------------------------

def _prompt_bytes(kind, item):
    if kind == "text":
        return item.encode("utf-8")
    return struct.pack("<%di" % len(item), *item)  # token ids as int32 little-endian


class RequestPlan:
    """This worker's schedule rows, request builder, planned counts and record capacities."""

    def __init__(self, args, tt, worker):
        self.tt = tt
        self.n_workers = args.workers
        try:
            self.req_ids, self.planned, self.tseq, meta = load_schedule(
                args.schedule, worker, args.workers, tier_table_sha=tt.sha256)
        except OSError as e:
            raise ConfigError("cannot read --schedule: %s" % e)
        self.schedule_sha256 = sha256_file(args.schedule)
        counts = per_worker_tier_counts(meta, args.workers)
        self.planned_by_tier = [counts.get((worker, t.seq), 0) for t in tt.tiers]
        seen = [0] * len(tt)
        for s in self.tseq:
            seen[s] += 1
        if seen != self.planned_by_tier:
            raise ConfigError("schedule rows per tier %s != header per_tier %s"
                              % (seen, self.planned_by_tier))
        self.n = len(self.req_ids)
        if self.n > args.max_records_per_worker:
            raise ConfigError("worker %d has %d requests > --max-records-per-worker %d"
                              % (worker, self.n, args.max_records_per_worker))
        self.format = args.request_format
        self.expected_msgs = [None] * len(tt)
        if self.format == "capkit":
            self._init_capkit(args)
        else:
            self._init_reqbytes(args)
        self.msg_capacity = 0
        if args.stamped or args.planned_msgs:
            if args.max_msg_records_per_worker is not None:
                self.msg_capacity = args.max_msg_records_per_worker
            elif self.format == "capkit":
                mod = args.msg_sample_mod
                self.msg_capacity = sum(self.expected_msgs[s]
                                        for r, s in zip(self.req_ids, self.tseq) if r % mod == 0)
            else:
                raise ConfigError("--%s with --request-format %s needs "
                                  "--max-msg-records-per-worker"
                                  % ("stamped" if args.stamped else "planned-msgs", self.format))

    def _init_capkit(self, args):
        try:
            prompts = load_prompts(args.prompts)
        except OSError as e:
            raise ConfigError("cannot read --prompts: %s" % e)
        sets = prompts["sets"]
        self._sets = {}
        for t in self.tt.tiers:
            if not t.is_load:
                continue
            if t.prompt_set not in sets:
                raise ConfigError("tier %d prompt_set %r not in the prompts file"
                                  % (t.seq, t.prompt_set))
            if t.prompt_set not in self._sets:
                s = sets[t.prompt_set]
                self._sets[t.prompt_set] = [_prompt_bytes(s["kind"], it) for it in s["items"]]
            self.expected_msgs[t.seq] = expect.msgs_per_request(
                t.max_tokens, t.tokens_per_msg, t.stream_interval, "product")
        tiers = self.tt.tiers
        sets_by_tier = [self._sets.get(t.prompt_set) for t in tiers]
        pack = GRPC_REQ.pack

        def build(rid, seq):
            t = tiers[seq]
            items = sets_by_tier[seq]
            return pack(rid, t.max_tokens, t.tokens_per_msg, t.stream_interval, seq) + \
                items[prompt_index(rid, len(items))]

        self.build = build

    def _init_reqbytes(self, args):
        try:
            rb = load_reqbytes(args.request_bytes)
        except OSError as e:
            raise ConfigError("cannot read --request-bytes: %s" % e)
        if rb.format != self.format:
            raise ConfigError("--request-bytes file is %s, not %s" % (rb.format, self.format))
        self.reqbytes = rb
        if self.format == "raw":
            body = rb.wildcard
            self.build = lambda rid, seq: body
            return
        tiers = self.tt.tiers
        set_sizes = {}
        for key in rb.table:
            set_sizes[key[0]] = max(set_sizes.get(key[0], 0), key[1] + 1)
        self._set_sizes = set_sizes
        # check every request this worker will send has a template, before the origin
        for rid, seq in zip(self.req_ids, self.tseq):
            t = tiers[seq]
            n = set_sizes.get(t.prompt_set, 0)
            if not n or rb.lookup(t.prompt_set, prompt_index(rid, n), t.max_tokens) is None:
                raise ConfigError("request-bytes has no template for tier %d (set %r, "
                                  "max_tokens %d)" % (seq, t.prompt_set, t.max_tokens))
        num = rb.id_field_number

        def build(rid, seq):
            return encode_pb_string_field(num, str(rid)) + rb.lookup(*self.template_key(rid, seq))

        self.build = build

    def template_key(self, rid, seq):
        """(prompt_set, prompt_index, max_tokens) of request rid's openengine template."""
        t = self.tt.tiers[seq]
        return t.prompt_set, prompt_index(rid, self._set_sizes[t.prompt_set]), t.max_tokens


# ---- worker -----------------------------------------------------------------------------------

class LoadWorker:
    """One worker process (or the in-process worker): dispatcher + one task per request.

    Subclass hooks (capgrpc.oe_sink): connect / disconnect / connection_info open and close the
    connection and describe it in the ready file; one() is the per-request coroutine;
    extra_summary(seq) adds keys to each tier summary; COUNT_BYTES says whether message bytes are
    counted (records and summary carry -1 / null otherwise)."""

    COUNT_BYTES = True

    def __init__(self, args, rt, plan, link):
        self.args = args
        self.rt = rt
        self.plan = plan
        self.link = link
        self.tt = rt.tt
        self.origin = rt.origin_ns
        self.nch = args.channels_per_worker
        self.local_pool = not args.shared_subchannel_pool
        self.stamped = args.stamped
        self.msgplan = MsgPlan(rt.tt) if args.planned_msgs else None
        self.mod = args.msg_sample_mod
        T = self.tt.request_timeout_ns
        self.timeout_ns = T
        nt = len(self.tt)
        self.sent_by_tier = [0] * nt
        self.ok_by_tier = [0] * nt
        self.failed_by_tier = [0] * nt
        self.msgs_by_tier = [0] * nt
        self.bytes_by_tier = [0] * nt
        self.mismatch_by_tier = [0] * nt
        self.unplanned_by_tier = [0] * nt
        self.fail_reasons = [dict() for _ in range(nt)]
        self.recv_types = [dict() for _ in range(nt)]
        self.send_hist = [_hist.new_counts() for _ in range(nt)]
        self.send_max = [None] * nt
        self.status_ids = {}
        self.req_rec = RecordArray(REQ_FIELDS, plan.n)
        self.msg_rec = RecordArray(MSG_FIELDS, plan.msg_capacity) \
            if self.stamped or self.msgplan is not None else None
        self.cursor = TierCursor(self.tt)
        v = rt.registry.values
        self.v = v
        pos = {d["name"]: d["offset"] for d in rt.registry.slots}
        self.i_inflight = pos["backlog.inflight"]
        self.i_active = pos["backlog.active_streams"]
        self.i_read = pos["backlog.read"]
        self.i_sent = pos["self.sent"]
        self.i_ok = pos["self.ok"]
        self.i_failed = pos["self.failed"]
        self.i_bytes = pos["bytes.recv"] if self.COUNT_BYTES else None
        self.h_send = rt.slot_refs["lag.send"]
        self.h_msg = rt.slot_refs["lag.msg"]
        self.inj_send = rt.injectors["send.req"]
        self.inj_recv = rt.injectors["recv.msg"]
        self.inj_drop = rt.injectors["drop.msg"]
        self.tasks = set()
        self.stopping = False
        self.stop_hard = False
        self.stop_at_ns = None
        self._wake = None
        self.n_dispatched = 0
        self.client_error_logged = False

    # -- stop ------------------------------------------------------------------------------------
    def on_stop(self, hard):
        if not self.stopping:
            self.stop_at_ns = now_ns()
        self.stopping = True
        self.stop_hard = self.stop_hard or hard
        w = self._wake
        if w is not None and not w.done():
            w.set_result(None)
        if hard:
            for t in list(self.tasks):
                t.cancel()

    async def _sleep_until(self, loop, abs_ns):
        """Sleep until abs_ns unless a stop arrives first; returns False when stopped."""
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

    # -- main --------------------------------------------------------------------------------------
    async def run(self, loop):
        rt, args = self.rt, self.args
        stopper = StopController()
        stopper.attach_loop(loop)
        stopper.on_stop(self.on_stop)
        if self.link is not None:
            self.link.attach(loop, stopper)
        target = resolve_target(args.target, args.target_port_file)
        try:
            await self.connect(target)
            info = {"target": target, "method": args.method, "channels": self.nch,
                    "local_subchannel_pool": self.local_pool,
                    "request_format": self.plan.format, "stamped": self.stamped,
                    "planned_msgs": self.msgplan is not None,
                    "n_requests": self.plan.n, "planned_by_tier": self.plan.planned_by_tier,
                    "msg_capacity": self.plan.msg_capacity,
                    "schedule_sha256": self.plan.schedule_sha256}
            info.update(self.connection_info())
            rt.start(loop, info)
            if self.link is not None:
                self.link.ready(worker_ready_info(rt, info))
            else:
                write_inprocess_ready(rt, args.ready_file, info)
            await self.dispatch(loop)
            await self._sleep_until(loop, self.origin + self.tt.end_ns)
            if self.tasks and not self.stop_hard:
                await asyncio.wait(list(self.tasks), timeout=args.stop_grace_s)
            for t in list(self.tasks):
                t.cancel()
            if self.tasks:
                await asyncio.gather(*self.tasks, return_exceptions=True)
        finally:
            rt.close_tiers()
            await self.disconnect()
        rt.read_stop()
        self.write_outputs()
        rt.stop()
        stopped_early = self.stop_at_ns is not None and \
            self.stop_at_ns < self.origin + self.tt.end_ns
        return STOPPED_EARLY if stopped_early else OK

    # -- connection hooks ----------------------------------------------------------------------
    async def connect(self, target):
        """Open the channels and wait until every one is connected (before the origin)."""
        self.channels = [open_channel(target, self.local_pool) for _ in range(self.nch)]
        self.calls = [raw_stream_callable(c, self.args.method) for c in self.channels]
        await wait_channels_ready(self.channels, self.origin)

    async def disconnect(self):
        for c in getattr(self, "channels", ()):
            await c.close()

    def connection_info(self):
        """Extra keys of the ready-file self-report (none for raw channels)."""
        return {}

    def extra_summary(self, seq):
        """Extra keys of the tier-`seq` summary (none for capgrpc.load)."""
        return {}

    async def dispatch(self, loop):
        planned, origin, n = self.plan.planned, self.origin, self.plan.n
        i = 0
        create = loop.create_task
        tasks = self.tasks
        while i < n:
            if not await self._sleep_until(loop, origin + planned[i]):
                break
            now = now_ns()
            while i < n and origin + planned[i] <= now:
                t = create(self.one(i))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
                i += 1
        self.n_dispatched = i

    def _status_id(self, reason):
        sid = self.status_ids.get(reason)
        if sid is None:
            sid = len(self.status_ids) + 1
            self.status_ids[reason] = sid
        return sid

    def begin_request(self, seq, prel):
        """Send-side bookkeeping just before a request's call is issued: the send.req site, the
        send lag, the sent / in-flight / active counters. Returns (t0, timeout in s or None):
        the timeout runs from the planned time (>= 1 ms when sent)."""
        v, origin = self.v, self.origin
        self.inj_send.hit()
        t0 = now_ns()
        timeout = None
        if self.timeout_ns is not None:
            timeout = max((origin + prel + self.timeout_ns - t0) / 1e9, 0.001)
        lag = t0 - origin - prel
        self.h_send.add(lag)
        _hist.add(self.send_hist[seq], lag)
        if self.send_max[seq] is None or lag > self.send_max[seq]:
            self.send_max[seq] = lag
        self.sent_by_tier[seq] += 1
        v[self.i_sent] += 1
        v[self.i_inflight] += 1
        v[self.i_active] += 1
        return t0, timeout

    def end_request(self, rid, seq, prel, t0, first, nmsg, nbytes, reason, ch):
        """Outcome bookkeeping of one request (the active gauge is the caller's): counters,
        failure reasons, msgs per planned tier, and its requests row."""
        plan, v, origin = self.plan, self.v, self.origin
        done = now_ns()
        v[self.i_inflight] -= 1
        if reason is None:
            status = 0
            self.ok_by_tier[seq] += 1
            v[self.i_ok] += 1
            em = plan.expected_msgs[seq]
            if em is not None and nmsg != em:
                self.mismatch_by_tier[seq] += 1
        else:
            status = self._status_id(reason)
            self.failed_by_tier[seq] += 1
            v[self.i_failed] += 1
            fr = self.fail_reasons[seq]
            fr[reason] = fr.get(reason, 0) + 1
        self.msgs_by_tier[seq] += nmsg
        if nbytes >= 0:
            self.bytes_by_tier[seq] += nbytes
        self.req_rec.append(rid, prel, t0 - origin, first - origin if first >= 0 else -1,
                            done - origin, nmsg, nbytes, -1, status, ch, 0, seq)

    def log_client_error(self, rid, e):
        if not self.client_error_logged:
            self.client_error_logged = True
            print("%s: request %d: client-side %s: %s (first occurrence; counted as "
                  "client_error)" % (self.rt.tool, rid, type(e).__name__, e), file=sys.stderr)

    async def one(self, i):
        plan, v, origin = self.plan, self.v, self.origin
        rid, seq, prel = plan.req_ids[i], plan.tseq[i], plan.planned[i]
        req = plan.build(rid, seq)
        ch = i % self.nch
        t0, timeout = self.begin_request(seq, prel)
        active = True
        nmsg = nbytes = 0
        first = -1
        reason = None
        call = None
        stamped, sampled = self.stamped, rid % self.mod == 0
        msgplan = self.msgplan
        mst = msgplan.stream(prel, seq) if msgplan is not None else None
        inj_recv, inj_drop = self.inj_recv, self.inj_drop
        try:
            call = self.calls[ch](req, timeout=timeout)
            while True:
                msg = await call.read()
                if msg is grpc.aio.EOF:
                    break
                t = now_ns()
                inj_recv.hit()
                if inj_drop.should_drop():
                    continue
                if type(msg) is not bytes:
                    if nmsg == 0:
                        tn = type(msg).__name__
                        self.recv_types[seq][tn] = self.recv_types[seq].get(tn, 0) + 1
                    reason = "protocol_error"
                    call.cancel()
                    break
                if nmsg == 0:
                    self.recv_types[seq]["bytes"] = self.recv_types[seq].get("bytes", 0) + 1
                    first = t
                ln = len(msg)
                nbytes += ln
                v[self.i_read] += 1
                v[self.i_bytes] += ln
                if stamped:
                    if ln < 16:
                        reason = "protocol_error"
                        call.cancel()
                        break
                    step, splanned = STEP_STAMP.unpack_from(msg, 0)
                    rel = t - origin
                    self.h_msg.add(rel - splanned)
                    if sampled:
                        self.msg_rec.append(rid, nmsg, step, splanned, rel,
                                            self.cursor.seq_at(rel))
                elif msgplan is not None:
                    self.planned_msg(mst, nmsg, t, rid, seq, sampled)
                nmsg += 1
            v[self.i_active] -= 1
            active = False
            if reason is None:
                code = await call.code()
                if code != grpc.StatusCode.OK:
                    reason = self._reason(code)
        except grpc.aio.AioRpcError as e:
            reason = self._reason(e.code())
        except asyncio.CancelledError:
            reason = "grpc_CANCELLED"
            if call is not None:
                call.cancel()
        except Exception as e:  # noqa: BLE001 - a client-side fault still ends the request once
            reason = "client_error"
            if call is not None:
                call.cancel()
            self.log_client_error(rid, e)
        finally:
            if active:
                v[self.i_active] -= 1
        self.end_request(rid, seq, prel, t0, first, nmsg, nbytes, reason, ch)

    def planned_msg(self, mst, j, t, rid, seq, sampled):
        """--planned-msgs: lag of message j (received at t) against its derived planned time."""
        pr = self.msgplan.planned(mst, j) if mst is not None else None
        if pr is None:
            self.unplanned_by_tier[seq] += 1
            return
        rel = t - self.origin
        self.h_msg.add(rel - pr[1])
        if sampled:
            self.msg_rec.append(rid, j, pr[0], pr[1], rel, self.cursor.seq_at(rel))

    def _reason(self, code):
        if code == grpc.StatusCode.DEADLINE_EXCEEDED and self.timeout_ns is not None:
            return "timeout_registered"
        return status_reason(code)


    # -- outputs -------------------------------------------------------------------------------
    def write_outputs(self):
        rt, plan, args = self.rt, self.plan, self.args
        status_codes = {str(i): r for r, i in self.status_ids.items()}
        rt.dump_records(self.req_rec, "requests", "loadgen.requests",
                        {"status_codes": status_codes, "method": args.method,
                         "request_format": plan.format, "conn_id": "channel index",
                         "tokens": "-1 = not counted by the gRPC loader",
                         "schedule_sha256": plan.schedule_sha256})
        msg_src = "stamp" if self.stamped else "schedule" if self.msgplan is not None else None
        if self.msg_rec is not None:
            rt.dump_records(self.msg_rec, "msgs", "drain.msgs",
                            {"chunk_sample_mod": self.mod, "msg_sample_mod": self.mod,
                             "sample_rule": "stream_id % msg_sample_mod == 0",
                             "tier_seq": "event time (receive)", "planned_source": msg_src})
        rt.dump_lag()
        for seq in sorted(rt.end_marks):
            a, b = rt.start_marks[seq], rt.end_marks[seq]
            P, S = plan.planned_by_tier[seq], self.sent_by_tier[seq]
            payload = rt.tier_common(seq)
            payload.update({
                "schedule_sha256": plan.schedule_sha256,
                "request_timeout_ns": self.tt.request_timeout_ns,
                "method": args.method, "request_format": plan.format,
                "channels_per_worker": self.nch, "local_subchannel_pool": self.local_pool,
                "stamped": self.stamped, "in_process": bool(args.in_process),
                "counts": {"planned": P, "sent": S, "ok": self.ok_by_tier[seq],
                           "failed": self.failed_by_tier[seq], "not_sent": P - S},
                "failures": [{"reason": r, "count": c,
                              "class_hint": CLASS_HINTS.get(r, "unclassified")}
                             for r, c in sorted(self.fail_reasons[seq].items())],
                "msgs": {"msgs": self.msgs_by_tier[seq],
                         "bytes": self.bytes_by_tier[seq] if self.COUNT_BYTES else None,
                         "expected_per_request": plan.expected_msgs[seq],
                         "ok_requests_with_unexpected_msg_count":
                             self.mismatch_by_tier[seq]
                             if plan.expected_msgs[seq] is not None else None,
                         "attribution": "planned tier of the request"},
                "send_lag_ns": {"hist": {"bins": _hist.NAME, "counts": self.send_hist[seq]},
                                "max": self.send_max[seq], "n": sum(self.send_hist[seq])},
                "msg_lag_ns": (rt.hist_summary("lag.msg", a["vals"], b["vals"])
                               if msg_src is not None else None),
                "msg_plan": {"source": msg_src,
                             "unplanned": self.unplanned_by_tier[seq]
                             if self.msgplan is not None else None},
                "read_msgs_event_time": rt.slot_delta("backlog.read", a["vals"], b["vals"]),
                "recv_types": self.recv_types[seq],
                "overflow": {"requests": self.req_rec.overflow,
                             "msgs": self.msg_rec.overflow if self.msg_rec else 0,
                             "lag": rt.lag_records.overflow if rt.lag_records else 0}})
            payload.update(self.extra_summary(seq))
            rt.write_summary(seq, payload)


# ---- entry points ------------------------------------------------------------------------------

def worker_run(args, worker, link):
    common = common_from_args(args)
    cpus = [common.cpus[worker]]
    rt = ToolRuntime(common, TOOL, SLOTS, SITES, worker, args.workers, cpus, args.lag_period_us,
                     args.lag_records_capacity)
    plan = RequestPlan(args, rt.tt, worker)
    w = LoadWorker(args, rt, plan, link)
    return aio_run(w.run, common.loop)


def child_entry(args_dict, worker, conn):
    child_main(args_dict, worker, conn, worker_run)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        validate_args(args, common)
        if args.in_process:
            return worker_run(args, 0, None)
        return run_supervisor(args, common, args.workers, child_entry, TOOL, SLOTS)
    except CapkitError as e:
        print("%s: %s" % (TOOL, e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
