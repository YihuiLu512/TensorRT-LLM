# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-PB: prebaked-bytes OpenEngine stub (design 2.3 "S-PB", E2 card; INTERFACES errata B2B).

One grpc.aio process. The Generate method (the prebaked file's `method`) is served with identity
(de)serializers: the request bytes are read only for request_id and stopping.max_tokens
(capstubs.prebaked.parse_generate_request) and the response bytes come from the prebaked file
(capstubs.prebaked), so no protobuf object is built or serialized on the Generate path.

Pacing is nullgrpc's beat source: one BeatSchedule for the process; a beat driver wakes at every
planned step and appends due messages to per-stream unbounded pending queues without waiting for
any write; message j of a stream is queued on step first_step + msg_step_offsets(N, k, s,
"product")[j], and the finished message (usage) on the step of the last token message. N comes
from the request; k and s are the tier table's single (k, s) over its load tiers.

Optional frontend mode (--sfe-module PATH): the S-FE module (fake_openengine.py, unchanged from the
probe-tree baseline) is loaded from that path and its Control and Health servicers are
registered next to Generate, built from the same S-FE options as `fake_openengine.py` takes
(the fingerprint's `sfe_args` splice in unchanged). Without it only Generate is served.

Entry: python -m capstubs.prebaked_server
"""

import argparse
import asyncio
import collections
import hashlib
import importlib.util
import os
import sys

import grpc

from capcommon.aio import call_at_ns, run as aio_run
from capcommon.capbin import RecordArray
from capcommon.cli import add_common_args, common_from_args
from capcommon.clock import now_ns
from capcommon.exitcodes import OK, CapkitError, ConfigError
from capcommon.hashing import sha256_file
from capcommon.lifecycle import StopController, write_port_file
from capcommon.schedule import BeatSchedule, count_schedule_rows, load_schedule
from capcommon.tiers import POST_SEQ, TierTable

from capgrpc import inject_slot_decls
from capgrpc.common import (UNREQUESTED_OPTION, ToolRuntime, add_server_args, bind_server,
                            combine_launch_specs, make_server, resolve_max_unrequested,
                            write_inprocess_ready)

from .prebaked import (SHORT_SITE, RequestIdSamples, WireFormatError, decimal_request_id,
                       load_prebaked, parse_generate_request, required_shapes, short_amounts,
                       shorted, uniform_msg_shape)

TOOL = "capstubs.prebaked_server"

# send.msg: once per message just before it is handed to gRPC (busy/sleep); beat.step: once per
# beat step in the beat driver (busy/sleep); short.req: once per accepted request (short).
SITE_MODES = {"send.msg": ("busy", "sleep"), "beat.step": ("busy", "sleep"),
              SHORT_SITE: ("short",)}
SITES = tuple(sorted(SITE_MODES))

SLOTS = {
    "backlog.active_streams": {"kind": "gauge", "site": None, "group": "backlog",
                               "per": "request"},
    "backlog.pending_msgs": {"kind": "gauge", "site": None, "group": "backlog",
                             "per": "message"},
    "backlog.yielded": {"kind": "counter", "site": "send.msg", "group": "backlog",
                        "per": "message"},
    "self.accepted": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.completed": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.cancelled": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.rejected": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "bytes.sent": {"kind": "counter", "site": "send.msg", "group": "bytes", "per": "message"},
    "count.beat_steps": {"kind": "counter", "site": "beat.step", "group": "count",
                         "per": "step"},
    "count.beat_late_steps": {"kind": "counter", "site": "beat.step", "group": "count",
                              "per": "step"},
    "count.beat_enqueued": {"kind": "counter", "site": "beat.step", "group": "count",
                            "per": "message"},
    "lag.beat_wake": {"kind": "hist", "site": "beat.step", "group": "lag", "per": "step"},
    "lag.loop": {"kind": "hist", "site": None, "group": "lag", "per": "sample"},
}
SLOTS.update(inject_slot_decls(SITES, {"send.msg": "message", "beat.step": "step",
                                       SHORT_SITE: "request"}))

STREAM_FIELDS = [("req_id", "q"), ("accept_rel_ns", "q"), ("first_step", "q"), ("n_msgs", "i"),
                 ("tokens", "i"), ("tier_seq", "h")]
STEP_FIELDS = [("step", "q"), ("planned_rel_ns", "q"), ("actual_rel_ns", "q"), ("msgs", "i"),
               ("tier_seq", "h")]
# Rejection reasons (fixed codes; the abort message carries the detail).
REJECT_REASONS = ("not_bytes", "malformed", "no_max_tokens", "no_prebaked_stream",
                  "request_id_not_in_schedule")
# The S-FE options frontend mode passes to fake_openengine's own parser (same flag names).
SFE_OPTIONS = (("--sfe-profile", "--profile", 1), ("--model", "--model", 1),
               ("--max-context-length", "--max-context-length", 1),
               ("--vocab-size", "--vocab-size", 1), ("--eos-token-ids", "--eos-token-ids", "+"),
               ("--profile-json", "--profile-json", 1))


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.prebaked_server", allow_abbrev=False,
                                description=__doc__.split("\n")[0])
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=True)
    g = p.add_argument_group("S-PB")
    g.add_argument("--host", required=True)
    g.add_argument("--port", type=int, required=True, help="0 = any free port")
    g.add_argument("--workers", type=int, required=True, help="must be 1 (single process)")
    g.add_argument("--prebaked", required=True, help="capkit.prebaked/1 file")
    g.add_argument("--lag-period-us", type=int, required=True)
    g.add_argument("--schedule",
                   help="arrival schedule: request_id must be a req_id of it and each stream is "
                        "archived under its planned tier (without it: the tier at acceptance)")
    g.add_argument("--max-stream-records", type=int,
                   help="streams capacity (default: count_schedule_rows of the tier table)")
    g.add_argument("--lag-records-capacity", type=int, default=0)
    add_server_args(g)
    f = p.add_argument_group("frontend mode (Control + Health from the S-FE module)")
    f.add_argument("--sfe-module", help="path of fake_openengine.py (probe-tree baseline copy)")
    for flag, _sfe, nargs in SFE_OPTIONS:
        if nargs == "+":
            f.add_argument(flag, nargs="+", help="passed to the S-FE parser as is")
        else:
            f.add_argument(flag, help="passed to the S-FE parser as is")
    return p


def sfe_argv(args):
    """The fake_openengine command line for the S-FE options given to this stub."""
    out = []
    for flag, sfe, nargs in SFE_OPTIONS:
        v = getattr(args, flag[2:].replace("-", "_"))
        if v is None:
            continue
        out += [sfe] + (list(v) if nargs == "+" else [v])
    return out


def validate_args(args, common):
    if args.workers != 1:
        raise ConfigError("--workers must be 1 (S-PB is one process)")
    if not 0 <= args.port < 65536:
        raise ConfigError("--port out of range")
    if len(common.cpus) < 1:
        raise ConfigError("--cpus lists no cpu")
    if args.max_stream_records is not None and args.max_stream_records < 1:
        raise ConfigError("--max-stream-records must be >= 1")
    if args.lag_records_capacity < 0:
        raise ConfigError("--lag-records-capacity must be >= 0")
    given = [flag for flag, _sfe, _n in SFE_OPTIONS
             if getattr(args, flag[2:].replace("-", "_")) is not None]
    if args.sfe_module is None and given:
        raise ConfigError("S-FE options %s need --sfe-module" % given)


def check_inject_specs(specs, site_modes=None, who="S-PB"):
    """Every spec targets a known site in a mode that site supports (else exit 3). site_modes
    defaults to S-PB's; S-NL passes its own."""
    site_modes = SITE_MODES if site_modes is None else site_modes
    for sp in specs:
        if sp.site not in site_modes:
            raise ConfigError("inject site %s unknown to %s (sites: %s)"
                              % (sp.site, who, tuple(sorted(site_modes))))
        if sp.mode not in site_modes[sp.site]:
            raise ConfigError("inject site %s does not support mode %s (supports %s)"
                              % (sp.site, sp.mode, site_modes[sp.site]))


def plan_streams(tt, prebaked, specs):
    """(k, s) of the tier table, after checking that the prebaked file covers every shape the
    table's load tiers and the short specs (command line + table, this role) can ask for."""
    k, s = uniform_msg_shape(tt)
    need = required_shapes(tt, short_amounts(specs))
    missing = [x for x in need if prebaked.lookup(*x) is None]
    if missing:
        raise ConfigError("prebaked file %s has no stream for (max_tokens, k, s) %s"
                          % (prebaked.path, missing))
    return k, s


def load_sfe_module(path):
    """Run fake_openengine.py from `path` as a module under a private name; return (module,
    sha256). The file is read once: the bytes hashed are the bytes executed. Any exception
    while compiling or running it is a ConfigError (exit 3)."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        raise ConfigError("cannot read --sfe-module %s: %s" % (path, e))
    sha = hashlib.sha256(data).hexdigest()
    name = "_capstubs_sfe_%s" % sha[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ConfigError("--sfe-module %s is not a Python file" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up while the class body runs
    try:
        exec(compile(data, path, "exec"), mod.__dict__)  # noqa: S102
    except (Exception, SystemExit) as e:  # noqa: BLE001 - whatever it raises, exit 3
        sys.modules.pop(name, None)
        raise ConfigError("--sfe-module %s cannot be imported here: %s: %s"
                          % (path, type(e).__name__, e))
    return mod, sha


def frontend_control(args):
    """Frontend mode: load the --sfe-module file and build what S-PB serves from it. Returns
    (sha256, S-FE argv, register(server)), register adding its Control and Health servicers to
    the server. Its parser exiting is a rejection of the S-FE options; any other exception from
    taking its entry points to registering the servicers is a ConfigError too (exit 3)."""
    mod, sha = load_sfe_module(args.sfe_module)
    sargv = sfe_argv(args)
    try:
        profile = mod._profile_from_args(mod._parse_args(sargv))
        engine = mod.FakeEngine(profile, tokens_per_step=1, step_delay=0.0, record=False)
        services = [(mod.openengine_pb2_grpc.add_ControlServicer_to_server, mod._Control(engine)),
                    (mod.health_pb2_grpc.add_HealthServicer_to_server, mod._Health(engine))]
    except SystemExit as e:
        raise ConfigError("S-FE options %s rejected by %s (exit %s)"
                          % (sargv, args.sfe_module, e.code))
    except Exception as e:  # noqa: BLE001 - whatever the S-FE setup raises, exit 3
        raise ConfigError("--sfe-module %s: S-FE setup with options %s failed: %s: %s"
                          % (args.sfe_module, sargv, type(e).__name__, e))

    def register(server):
        try:
            for add, servicer in services:
                add(servicer, server)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - whatever it raises, exit 3
            raise ConfigError("--sfe-module %s: registering its servicers failed: %s: %s"
                              % (args.sfe_module, type(e).__name__, e))
    return sha, sargv, register


class _Stream:
    __slots__ = ("sb", "first", "q", "ev", "dead", "truncated")

    def __init__(self, sb):
        self.sb = sb
        self.first = -1
        self.q = collections.deque()
        self.ev = asyncio.Event()
        self.dead = False
        self.truncated = False


class _MethodHandler(grpc.GenericRpcHandler):
    """Serve one method path as unary_stream with identity (de)serializers; other paths fall
    through to the remaining handlers (Control and Health in frontend mode) or UNIMPLEMENTED."""

    def __init__(self, method, behavior):
        self._method = method
        self._h = grpc.unary_stream_rpc_method_handler(behavior, request_deserializer=None,
                                                       response_serializer=None)

    def service(self, handler_call_details):
        return self._h if handler_call_details.method == self._method else None


class PrebakedServer:
    def __init__(self, args, rt, prebaked, ks, sched_tiers, sched_sha256, control):
        self.args, self.rt, self.pb = args, rt, prebaked
        self.k, self.s = ks
        self.sched = sched_tiers  # array: req_id -> planned tier, or None
        self.sched_sha256 = sched_sha256  # of the --schedule file read, or None
        self.control = control  # frontend_control(args) or None
        self.tt = rt.tt
        self.origin = rt.origin_ns
        nt = len(self.tt)
        self.last_seq = nt - 1
        z = lambda: [0] * nt  # noqa: E731
        self.c_accepted, self.c_completed, self.c_cancelled = z(), z(), z()
        self.c_truncated, self.c_rejected, self.c_bumped = z(), z(), z()
        self.c_msgs, self.c_bytes, self.c_tokens, self.c_finished = z(), z(), z(), z()
        self.c_short, self.c_nondec = z(), z()
        self.rid_samples = RequestIdSamples(nt)
        self.max_pending = z()
        self.rejects = [dict() for _ in range(nt)]  # by event tier
        cap = args.max_stream_records
        if cap is None:
            cap = count_schedule_rows(self.tt)
        self.stream_rec = RecordArray(STREAM_FIELDS, cap)
        self.bs = BeatSchedule(self.tt)
        if self.bs.n_steps == 0:
            raise ConfigError("S-PB paces by the beat, but no tier has beat_ns > 0")
        self.step_rec = RecordArray(STEP_FIELDS, self.bs.n_steps)
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
        self.i_rej = pos["self.rejected"]
        self.i_bytes = pos["bytes.sent"]
        self.i_steps = pos["count.beat_steps"]
        self.i_late = pos["count.beat_late_steps"]
        self.i_enq = pos["count.beat_enqueued"]
        self.h_wake = rt.slot_refs["lag.beat_wake"]
        self.inj_send = rt.injectors["send.msg"]
        self.inj_step = rt.injectors["beat.step"]
        self.inj_short = rt.injectors[SHORT_SITE]
        self.stop_ev = None
        self.stopping = False
        self._wake = None

    # -- request handling ------------------------------------------------------------------------
    def _admit(self, request, ev_seq):
        """(stream bytes, planned-or-event tier, req_id) or raises _Reject(reason, detail)."""
        if type(request) is not bytes:
            raise _Reject("not_bytes", "request object is %s" % type(request).__name__)
        try:
            rid_b, n = parse_generate_request(request)
        except WireFormatError as e:
            raise _Reject("malformed", "GenerateRequest: %s" % e)
        self.rid_samples.add(ev_seq, rid_b)
        if not n:
            raise _Reject("no_max_tokens", "stopping.max_tokens is unset or 0")
        rid = decimal_request_id(rid_b)
        if self.sched is not None:
            if not 0 <= rid < len(self.sched):
                raise _Reject("request_id_not_in_schedule",
                              "request_id %r is not a req_id of the schedule" % rid_b[:40])
            seq = self.sched[rid]
        else:
            seq = ev_seq
        if self.pb.lookup(n, self.k, self.s) is None:
            raise _Reject("no_prebaked_stream", "no prebaked stream for max_tokens %d" % n)
        d = self.inj_short.short_by()
        sb = self.pb.lookup(shorted(n, d), self.k, self.s)
        if sb is None:
            raise _Reject("no_prebaked_stream", "no prebaked stream for max_tokens %d "
                          "(short by %d)" % (shorted(n, d), d))
        if d:
            self.c_short[seq] += 1
        if rid < 0:
            self.c_nondec[seq] += 1
        return sb, seq, rid

    async def generate(self, request, context):
        ev_seq = self.rt.cur_seq  # event time: used for rejects and without --schedule
        ev_seq = self.last_seq if ev_seq == POST_SEQ else max(ev_seq, 0)
        try:
            sb, seq, rid = self._admit(request, ev_seq)
        except _Reject as r:
            rej = self.rejects[ev_seq]
            rej[r.reason] = rej.get(r.reason, 0) + 1
            self.c_rejected[ev_seq] += 1
            self.v[self.i_rej] += 1
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, r.detail)
            return
        accept = now_ns() - self.origin
        v = self.v
        v[self.i_acc] += 1
        v[self.i_active] += 1
        self.c_accepted[seq] += 1
        msgs = sb.msgs
        m = len(msgs)
        st = _Stream(sb)
        sent = 0
        outcome = "cancelled"
        try:
            first = self.register(st, accept, seq)
            if first is None:
                outcome = "truncated"
                await context.abort(grpc.StatusCode.OUT_OF_RANGE,
                                    "beat schedule ended before the first step")
                return
            q = st.q
            while sent < m:
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
                yield msg
            self.c_finished[seq] += 1
            outcome = "completed"
        finally:
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
            self.c_tokens[seq] += sb.cum[sent]
            self.stream_rec.append(rid, accept, st.first, sent, sb.cum[sent], seq)

    def register(self, st, accept_rel, seq):
        """Give a new stream its first step and queue its first message slot."""
        first = self.bs.step_at_or_after(accept_rel)
        if first is None:
            return None
        if first < self.next_step:  # the driver already passed that step (accept at its instant)
            first = self.next_step
            self.c_bumped[seq] += 1
        st.first = first
        step0 = first + st.sb.offs[0]
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
        step = 0
        while step < n:
            prel = bs.planned_rel_ns(step)
            if not await self._sleep_until(loop, origin + prel):
                break
            rel = now_ns() - origin
            self.h_wake.add(rel - prel)
            self.inj_step.hit()
            seq = bs.tier_of_step(step)
            if step + 1 < n and rel >= bs.planned_rel_ns(step + 1):
                v[self.i_late] += 1
            nq = 0
            entries = due.pop(step, None)
            if entries:
                for st, j in entries:
                    if st.dead:
                        continue
                    sb = st.sb
                    msgs, offs, m = sb.msgs, sb.offs, len(sb.msgs)
                    # this message and every later one planned on the same step
                    while True:
                        st.q.append(msgs[j])
                        nq += 1
                        j += 1
                        if j >= m or offs[j] != offs[j - 1]:
                            break
                    st.ev.set()
                    if j < m:
                        nxt = st.first + offs[j]
                        if nxt >= n:
                            st.truncated = True
                        else:
                            due.setdefault(nxt, []).append((st, j))
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

    def info(self, port):
        a = self.args
        return {"host": a.host, "port": port, "pacing": "beat", "gate": "product",
                "method": self.pb.method, "prebaked": {"path": os.path.abspath(a.prebaked),
                                                      "sha256": self.pb.sha256},
                "tokens_per_msg": self.k, "stream_interval": self.s,
                "tier_attribution": "schedule" if self.sched is not None else "event",
                "schedule": os.path.abspath(a.schedule) if a.schedule else None,
                "schedule_sha256": self.sched_sha256,
                "stream_capacity": self.stream_rec.capacity, "beat_steps": self.bs.n_steps,
                "grpc_server_options": dict(a.grpc_server_options),
                "control": None if self.control is None else {
                    "sfe_module": os.path.abspath(a.sfe_module), "sha256": self.control[0],
                    "sfe_argv": self.control[1]},
                "serializers": "identity (request_deserializer=None, response_serializer=None)"}

    async def run(self, loop):
        rt, args = self.rt, self.args
        self.stop_ev = asyncio.Event()
        stopper = StopController()
        stopper.attach_loop(loop)
        stopper.on_stop(self.on_stop)
        opts = args.grpc_server_options
        server = make_server(reuseport=False, max_unrequested_s=opts[UNREQUESTED_OPTION])
        server.add_generic_rpc_handlers([_MethodHandler(self.pb.method, self.generate)])
        if self.control is not None:
            self.control[2](server)
        port = bind_server(server, args.host, args.port)
        await server.start()
        info = self.info(port)
        rt.start(loop, info)
        driver = loop.create_task(self.beat_driver(loop))
        write_port_file(args.port_file, rt.identity, args.host, port)
        write_inprocess_ready(rt, args.ready_file, info)
        if not self.stopping:
            await self.stop_ev.wait()
        driver.cancel()
        await asyncio.gather(driver, return_exceptions=True)
        await server.stop(args.stop_grace_s)
        rt.close_tiers()
        rt.read_stop()
        self.write_outputs(info)
        rt.stop()
        return OK

    def write_outputs(self, info):
        rt = self.rt
        rt.dump_records(self.stream_rec, "streams", "server.streams",
                        {"pacing": "beat", "gate": "product", "tier_seq": info["tier_attribution"],
                         "n_msgs": "gRPC messages sent, the finished message included",
                         "req_id": "request_id as decimal, -1 when it is not one"})
        rt.dump_records(self.step_rec, "steps", "beat.steps",
                        {"n_steps": self.bs.n_steps, "stamp": False})
        rt.dump_lag()
        attribution = ("planned tier of the request's req_id in --schedule"
                       if self.sched is not None else "tier at acceptance (event time)")
        for seq in sorted(rt.end_marks):
            a, b = rt.start_marks[seq], rt.end_marks[seq]
            payload = rt.tier_common(seq)
            payload.update({
                "schedule_sha256": self.sched_sha256, "pacing": "beat", "gate": "product",
                "method": info["method"], "prebaked": info["prebaked"],
                "tokens_per_msg": self.k, "stream_interval": self.s,
                "tier_attribution": info["tier_attribution"], "control": info["control"],
                "grpc_server_options": dict(self.args.grpc_server_options),
                "counts": {"streams_accepted": self.c_accepted[seq],
                           "streams_completed": self.c_completed[seq],
                           "streams_cancelled": self.c_cancelled[seq],
                           "streams_truncated": self.c_truncated[seq],
                           "streams_rejected": self.c_rejected[seq],
                           "first_step_bumped": self.c_bumped[seq],
                           "msgs_sent": self.c_msgs[seq], "bytes_sent": self.c_bytes[seq],
                           "finished_sent": self.c_finished[seq],
                           "tokens_sent": self.c_tokens[seq],
                           "short_requests": self.c_short[seq],
                           "request_id_not_decimal": self.c_nondec[seq],
                           "attribution": attribution + " (streams_rejected: event time)"},
                "beat": {
                    "steps": rt.slot_delta("count.beat_steps", a["vals"], b["vals"]),
                    "late_steps": rt.slot_delta("count.beat_late_steps", a["vals"], b["vals"]),
                    "enqueued": rt.slot_delta("count.beat_enqueued", a["vals"], b["vals"]),
                    "max_pending": self.max_pending[seq],
                    "wake_lag_ns": rt.hist_summary("lag.beat_wake", a["vals"], b["vals"]),
                    "attribution": "event time (step planned time)"},
                "yielded_event_time": rt.slot_delta("backlog.yielded", a["vals"], b["vals"]),
                "rejects": dict(self.rejects[seq]),
                "request_id_samples": self.rid_samples.summary(seq),
                "overflow": {"streams": self.stream_rec.overflow,
                             "steps": self.step_rec.overflow,
                             "lag": rt.lag_records.overflow if rt.lag_records else 0}})
            rt.write_summary(seq, payload)


class _Reject(Exception):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


# ---- entry point --------------------------------------------------------------------------------

def _schedule_tiers(path, tt):
    """(req_id -> planned tier_seq, sha256 of the file) of the schedule (generated from this
    tier table); the sha is taken as capgrpc.load takes it (sha256_file)."""
    try:
        _ids, _planned, tiers, _meta = load_schedule(path, 0, 1, tier_table_sha=tt.sha256)
        sha = sha256_file(path)
    except OSError as e:
        raise ConfigError("cannot read --schedule %s: %s" % (path, e))
    return tiers, sha


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        validate_args(args, common)
        tt = TierTable.load(common.tier_table_path)
        args.grpc_server_options = resolve_max_unrequested(args.max_unrequested_s, tt)
        specs = combine_launch_specs(common.inject_specs, tt, common.identity.role) \
            + tt.all_inject_for(common.identity.role)
        check_inject_specs(specs)
        prebaked = load_prebaked(args.prebaked)
        ks = plan_streams(tt, prebaked, specs)
        sched, sched_sha = _schedule_tiers(args.schedule, tt) if args.schedule else (None, None)
        control = frontend_control(args) if args.sfe_module else None
        rt = ToolRuntime(common, TOOL, SLOTS, SITES, 0, 1, [common.cpus[0]],
                         args.lag_period_us, args.lag_records_capacity)
        srv = PrebakedServer(args, rt, prebaked, ks, sched, sched_sha, control)
        return aio_run(srv.run, common.loop)
    except CapkitError as e:
        print("%s: %s" % (TOOL, e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
