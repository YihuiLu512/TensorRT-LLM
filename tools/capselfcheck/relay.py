# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capselfcheck.relay: known-bottleneck gRPC stream relay (E0 (b), design 2.4 / E0).

Serves every method path as unary_stream with raw bytes. For each downstream call it opens the
same method upstream with the same request bytes, and per upstream message:
  1. counts it as read (backlog.read),
  2. passes the injection site relay.msg on the event loop -- the known extra work:
       busy inline   : spin delta on the loop (scenario 1; two chained relays = scenario 2),
       busy aux_thread: hand the spin to a plain Python thread of this process, competing for
                        the GIL (scenario 3),
       sleep          : time.sleep(delta) on the loop, blocking without CPU (scenario 4),
  3. queues it for downstream; the handler yields queued messages (backlog.yielded).
backlog.queue_msgs = messages read but not yet handed downstream. The per-message work is done in
the upstream reader task, so when the relay is the bottleneck the backlog builds upstream of it
(source yielded - relay read), which is what X / I3 must locate.
Per tier: counts by the request's planned tier (capkit header) and event-time slot deltas, the
service-time histogram (time spent in step 2 on the loop), hold time (read -> handed downstream),
loop lag, CPU of the process and of the loop thread.
"""

import argparse
import asyncio
import collections
import sys

import grpc

from capcommon.aio import run as aio_run
from capcommon.cli import add_common_args, common_from_args
from capcommon.clock import now_ns
from capcommon.exitcodes import OK, CapkitError, ConfigError
from capcommon.lifecycle import StopController, write_port_file
from capcommon.tiers import POST_SEQ
from capcommon.wire import WireError, unpack_grpc_request

from capgrpc.common import (AnyMethodHandler, ToolRuntime, bind_server, make_server,
                                 open_channel, raw_stream_callable, resolve_target,
                                 status_reason, wait_channels_ready, write_inprocess_ready)

from . import SITES, SLOTS, TOOL


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capselfcheck.relay",
                                description=__doc__.split("\n")[0])
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=True)
    g = p.add_argument_group("capselfcheck.relay")
    g.add_argument("--host", required=True)
    g.add_argument("--port", type=int, required=True, help="0 = any free port")
    g.add_argument("--upstream", help="HOST:PORT of the upstream server")
    g.add_argument("--upstream-port-file", help="port file of the upstream role")
    g.add_argument("--lag-period-us", type=int, required=True)
    g.add_argument("--lag-records-capacity", type=int, default=0)
    return p


def validate_args(args, common):
    if (args.upstream is None) == (args.upstream_port_file is None):
        raise ConfigError("give exactly one of --upstream and --upstream-port-file")
    if not 0 <= args.port < 65536:
        raise ConfigError("--port out of range")
    if len(common.cpus) != 1:
        raise ConfigError("the relay is one process: --cpus takes exactly one cpu")


class Relay:
    def __init__(self, args, rt):
        self.args, self.rt = args, rt
        self.tt = rt.tt
        self.origin = rt.origin_ns
        nt = len(self.tt)
        z = lambda: [0] * nt  # noqa: E731
        self.c_acc, self.c_done, self.c_fail, self.c_canc = z(), z(), z(), z()
        self.c_recv, self.c_fwd, self.c_bytes, self.c_unparsed = z(), z(), z(), z()
        self.upstream_failures = {}
        v = rt.registry.values
        self.v = v
        pos = {d["name"]: d["offset"] for d in rt.registry.slots}
        self.i_queue = pos["backlog.queue_msgs"]
        self.i_active = pos["backlog.active_streams"]
        self.i_read = pos["backlog.read"]
        self.i_yielded = pos["backlog.yielded"]
        self.i_acc = pos["self.accepted"]
        self.i_done = pos["self.completed"]
        self.i_fail = pos["self.failed"]
        self.i_canc = pos["self.cancelled"]
        self.i_bytes = pos["bytes.forwarded"]
        self.h_service = rt.slot_refs["lag.relay_service"]
        self.h_hold = rt.slot_refs["lag.relay_hold"]
        self.inj = rt.injectors["relay.msg"]
        self.calls = {}
        self.channel = None
        self.stop_ev = None
        self.loop = None

    def _planned_tier(self, request):
        """Planned tier from the capkit header; event tier when the request has no valid one."""
        try:
            if type(request) is not bytes:
                raise WireError("not bytes")
            seq = unpack_grpc_request(request)[4]
            if 0 <= seq < len(self.tt):
                return seq, True
        except WireError:
            pass
        ev = self.rt.cur_seq
        return (len(self.tt) - 1 if ev == POST_SEQ else max(ev, 0)), False

    def behavior_for(self, method):
        call_fn = self.calls.get(method)
        if call_fn is None:
            call_fn = raw_stream_callable(self.channel, method)
            self.calls[method] = call_fn

        async def behavior(request, context):
            v = self.v
            seq, parsed = self._planned_tier(request)
            if not parsed:
                self.c_unparsed[seq] += 1
            v[self.i_acc] += 1
            v[self.i_active] += 1
            self.c_acc[seq] += 1
            q = collections.deque()
            ev = asyncio.Event()
            state = {"done": False, "code": None, "details": None}
            call = call_fn(request)
            reader = self.loop.create_task(self._pump(call, q, ev, state, seq))
            outcome = "cancelled"
            try:
                while True:
                    if q:
                        msg, t_recv = q.popleft()
                        v[self.i_queue] -= 1
                        self.h_hold.add(now_ns() - t_recv)
                        v[self.i_yielded] += 1
                        v[self.i_bytes] += len(msg)
                        self.c_fwd[seq] += 1
                        self.c_bytes[seq] += len(msg)
                        yield msg
                        continue
                    if state["done"]:
                        break
                    ev.clear()
                    await ev.wait()
                code = state["code"]
                if code != grpc.StatusCode.OK:
                    outcome = "failed"
                    r = status_reason(code)
                    self.upstream_failures[r] = self.upstream_failures.get(r, 0) + 1
                    await context.abort(code, "upstream: %s" % (state["details"] or ""))
                    return
                outcome = "completed"
            finally:
                if not reader.done():
                    reader.cancel()
                    call.cancel()
                v[self.i_queue] -= len(q)
                q.clear()
                v[self.i_active] -= 1
                if outcome == "completed":
                    v[self.i_done] += 1
                    self.c_done[seq] += 1
                elif outcome == "failed":
                    v[self.i_fail] += 1
                    self.c_fail[seq] += 1
                else:
                    v[self.i_canc] += 1
                    self.c_canc[seq] += 1

        return behavior

    async def _pump(self, call, q, ev, state, seq):
        v, inj, h = self.v, self.inj, self.h_service
        try:
            while True:
                msg = await call.read()
                if msg is grpc.aio.EOF:
                    break
                t = now_ns()
                v[self.i_read] += 1
                self.c_recv[seq] += 1
                inj.hit()
                h.add(now_ns() - t)
                q.append((msg, t))
                v[self.i_queue] += 1
                ev.set()
            state["code"] = await call.code()
            state["details"] = await call.details()
        except grpc.aio.AioRpcError as e:
            state["code"], state["details"] = e.code(), e.details()
        except asyncio.CancelledError:
            state["code"] = grpc.StatusCode.CANCELLED
            raise
        finally:
            state["done"] = True
            ev.set()

    def on_stop(self, hard):
        if self.stop_ev is not None:
            self.stop_ev.set()

    async def run(self, loop):
        rt, args = self.rt, self.args
        self.loop = loop
        self.stop_ev = asyncio.Event()
        stopper = StopController()
        stopper.attach_loop(loop)
        stopper.on_stop(self.on_stop)
        target = resolve_target(args.upstream, args.upstream_port_file)
        self.channel = open_channel(target, local_subchannel_pool=True)
        await wait_channels_ready([self.channel], rt.origin_ns)
        server = make_server(reuseport=False)
        server.add_generic_rpc_handlers([AnyMethodHandler(self.behavior_for)])
        port = bind_server(server, args.host, args.port)
        await server.start()
        info = {"host": args.host, "port": port, "upstream": target,
                "serializers": "identity (raw bytes both ways)",
                "inject_site": SITES[0], "aux_threads": rt.aux_threads}
        rt.start(loop, info)
        write_port_file(args.port_file, rt.identity, args.host, port)
        write_inprocess_ready(rt, args.ready_file, info)
        if not stopper.requested:
            await self.stop_ev.wait()
        await server.stop(args.stop_grace_s)
        await self.channel.close()
        rt.close_tiers()
        rt.read_stop()
        self.write_outputs()
        rt.stop()
        return OK

    def write_outputs(self):
        rt = self.rt
        rt.dump_lag()
        for seq in sorted(rt.end_marks):
            a, b = rt.start_marks[seq], rt.end_marks[seq]
            payload = rt.tier_common(seq)
            payload.update({
                "schedule_sha256": None,
                "counts": {"streams_accepted": self.c_acc[seq],
                           "streams_completed": self.c_done[seq],
                           "streams_failed": self.c_fail[seq],
                           "streams_cancelled": self.c_canc[seq],
                           "msgs_received": self.c_recv[seq],
                           "msgs_forwarded": self.c_fwd[seq],
                           "bytes_forwarded": self.c_bytes[seq],
                           "requests_without_capkit_header": self.c_unparsed[seq],
                           "attribution": "planned tier from the capkit request header "
                                          "(event tier when the header is missing)"},
                "upstream_failures": dict(self.upstream_failures),
                "event_time": {
                    "read": rt.slot_delta("backlog.read", a["vals"], b["vals"]),
                    "yielded": rt.slot_delta("backlog.yielded", a["vals"], b["vals"]),
                    "service_ns": rt.hist_summary("lag.relay_service", a["vals"], b["vals"]),
                    "hold_ns": rt.hist_summary("lag.relay_hold", a["vals"], b["vals"])},
                "overflow": {"lag": rt.lag_records.overflow if rt.lag_records else 0}})
            rt.write_summary(seq, payload)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        validate_args(args, common)
        rt = ToolRuntime(common, TOOL, SLOTS, SITES, args.worker, 1, common.cpus,
                         args.lag_period_us, args.lag_records_capacity)
        return aio_run(Relay(args, rt).run, common.loop)
    except CapkitError as e:
        print("%s: %s" % (TOOL, e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
