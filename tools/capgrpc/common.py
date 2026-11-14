# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared runtime of the capkit gRPC tools (capgrpc.load, capgrpc.nullgrpc, capselfcheck.relay).

- raw-bytes gRPC plumbing: a generic handler that serves every method path as unary_stream with
  identity (de)serializers, channels with an explicit subchannel-pool choice, port reservation for
  SO_REUSEPORT worker groups;
- ToolRuntime: per-process slot registry, snapshot writer, injector set and tier-boundary marks
  (event-time attribution of slot deltas, per-tier inject reports, CPU readings);
- a spawn-based supervisor for multi-process tools: children report readiness over a pipe and
  receive stop requests over the same pipe (no signal forwarding, so a child that is also
  signalled directly by the orchestrator never sees a duplicate signal as a "second" one).
"""

import array
import asyncio
import multiprocessing
import multiprocessing.connection
import os
import socket
import sys
import time

import grpc

from capcommon import hist as _hist
from capcommon.clock import now_ns
from capcommon.exitcodes import CONFIG, OK, RUNTIME, STOPPED_EARLY, CapkitError, ConfigError
from capcommon.inject import InjectorSet, tier_report
from capcommon.jsonio import write_json_stamped
from capcommon.lifecycle import StopController, pin_self, read_port_file, read_stop_file
from capcommon.lifecycle import write_ready_file
from capcommon.runtree import RunTree
from capcommon.snapshot import SlotRegistry, SnapshotWriter
from capcommon.tiers import POST_SEQ, PRE_SEQ, TierTable
from capcommon.versions import version_record

TIER_SUMMARY_SCHEMA = "capkit.tier_summary/1"

# Message-size limits off on both sides: payload size is a test parameter, not a gRPC policy.
BASE_CHANNEL_OPTIONS = (("grpc.max_receive_message_length", -1),
                        ("grpc.max_send_message_length", -1))
LOCAL_SUBCHANNEL_POOL = ("grpc.use_local_subchannel_pool", 1)


# ---- raw-bytes gRPC plumbing -------------------------------------------------------------------

def status_reason(code):
    """INTERFACES.md 5.4.1 failure reason for a non-OK gRPC status."""
    name = getattr(code, "name", None) or str(code)
    return "grpc_%s" % name


class AnyMethodHandler(grpc.GenericRpcHandler):
    """Serve every method path as unary_stream with identity serializers (raw bytes both ways).

    behavior_factory(method_path) -> async generator function behavior(request, context).
    Handlers are built once per method path and cached.
    """

    def __init__(self, behavior_factory):
        self._factory = behavior_factory
        self._cache = {}

    def service(self, handler_call_details):
        method = handler_call_details.method
        h = self._cache.get(method)
        if h is None:
            h = grpc.unary_stream_rpc_method_handler(self._factory(method),
                                                     request_deserializer=None,
                                                     response_serializer=None)
            self._cache[method] = h
        return h


def make_server(reuseport):
    """A grpc.aio server (call from inside the running loop)."""
    opts = list(BASE_CHANNEL_OPTIONS) + [("grpc.so_reuseport", 1 if reuseport else 0)]
    return grpc.aio.server(options=opts)


def bind_server(server, host, port):
    bound = server.add_insecure_port("%s:%d" % (_host_for_target(host), port))
    if not bound:
        raise CapkitError("cannot bind %s:%d" % (host, port))
    return bound


def _host_for_target(host):
    return "[%s]" % host if ":" in host and not host.startswith("[") else host


def reserve_port(host, port):
    """Bind (not listen) a SO_REUSEPORT socket so worker processes can bind the same port.

    Returns (socket, port). Keep the socket open until every worker has bound; a bound socket
    that never listens receives no connections.
    """
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    try:
        s.bind((host, port))
    except OSError as e:
        s.close()
        raise CapkitError("cannot reserve %s:%d: %s" % (host, port, e))
    return s, s.getsockname()[1]


def open_channel(target, local_subchannel_pool):
    """A grpc.aio channel. local_subchannel_pool=True gives this channel its own connection
    (grpc.use_local_subchannel_pool=1); False lets same-target channels share one."""
    opts = list(BASE_CHANNEL_OPTIONS)
    if local_subchannel_pool:
        opts.append(LOCAL_SUBCHANNEL_POOL)
    return grpc.aio.insecure_channel(target, options=opts)


def raw_stream_callable(channel, method):
    """unary_stream multicallable sending/receiving raw bytes (no protobuf objects)."""
    return channel.unary_stream(method, request_serializer=None, response_deserializer=None)


def resolve_target(target, port_file):
    """HOST:PORT from --target or from a port file written by another role."""
    if (target is None) == (port_file is None):
        raise ConfigError("give exactly one of the target and the target port file")
    if target is not None:
        host, sep, port = target.rpartition(":")
        if not sep or not host or not (port.isascii() and port.isdigit()):
            raise ConfigError("target must be HOST:PORT, got %r" % target)
        return target
    try:
        host, port, _doc = read_port_file(port_file)
    except (OSError, ValueError) as e:
        raise CapkitError("cannot read port file %s: %s" % (port_file, e))
    return "%s:%d" % (_host_for_target(host), port)


async def wait_channels_ready(channels, deadline_ns):
    """Connect every channel before deadline_ns (the launch origin); raise CapkitError if not."""
    timeout = (deadline_ns - now_ns()) / 1e9
    if timeout <= 0:
        raise CapkitError("origin already passed before connecting")
    try:
        await asyncio.wait_for(asyncio.gather(*(c.channel_ready() for c in channels)), timeout)
    except asyncio.TimeoutError:
        raise CapkitError("channels not ready %.3fs before the origin" % timeout)


# ---- per-process runtime ------------------------------------------------------------------------

class TierCursor:
    """Event-time tier lookup with a one-comparison fast path (INTERFACES.md 3.3, 5.4)."""

    __slots__ = ("tt", "lo", "hi", "seq")

    def __init__(self, tt):
        self.tt = tt
        self.lo = 1
        self.hi = 0
        self.seq = None

    def seq_at(self, rel_ns):
        if self.lo <= rel_ns < self.hi:
            return self.seq
        tt = self.tt
        s = tt.tier_seq_at(rel_ns)
        if s == PRE_SEQ:
            self.lo, self.hi = -(1 << 62), 0
        elif s == POST_SEQ:
            self.lo, self.hi = tt.end_ns, 1 << 62
        else:
            self.lo, self.hi = tt[s].start_ns, tt[s].end_ns
        self.seq = s
        return s


def combine_launch_specs(cli_specs, tt, role):
    """CLI --inject specs plus the tier table's launch_inject entries for this role; checks every
    tier's specs for this role against them (a clash is a tier-table error, INTERFACES.md 7.1)."""
    launch = list(cli_specs) + [s for s in tt.launch_inject if s.role == role]
    sites = [s.site for s in launch]
    if len(set(sites)) != len(sites):
        raise ConfigError("more than one launch-level inject spec for a site: %s" % sites)
    for t in tt.tiers:
        tier_sites = [s.site for s in t.inject if s.role == role]
        clash = sorted(set(tier_sites) & set(sites))
        if clash:
            raise ConfigError("tier %d inject clashes with launch-level spec on %s" % (t.seq, clash))
    return launch


class ToolRuntime:
    """Per-process state shared by the gRPC tools: identity, pinning, slots, snapshots,
    injectors, tier-boundary marks and output paths."""

    def __init__(self, common, tool, slots, sites, worker, n_workers, cpus, lag_period_us,
                 lag_records_capacity=0):
        self.common = common
        self.tool = tool
        self.worker = worker
        self.n_workers = n_workers
        self.identity = common.identity.with_worker(worker).with_pid()
        self.role = self.identity.role
        self.affinity = pin_self(cpus) if cpus else None
        self.tt = TierTable.load(common.tier_table_path)
        self.origin_ns = common.origin_ns
        self.tree = RunTree(common.run_dir)
        self.registry = SlotRegistry()
        self.slot_refs = self.registry.register_decls(slots)
        self._pos = {d["name"]: (d["offset"], d["length"]) for d in self.registry.slots}
        self.injectors = InjectorSet(sites, self.registry)
        launch = combine_launch_specs(common.inject_specs, self.tt, self.role)
        # helper threads first (an aux_thread spec cannot be activated without its thread)
        self.aux_threads = self.injectors.prestart_aux(launch + self.tt.all_inject_for(self.role))
        self.injectors.set_launch_specs(launch)
        if not isinstance(lag_period_us, int) or lag_period_us <= 0:
            raise ConfigError("--lag-period-us must be an int > 0")
        self.lag_period_ns = lag_period_us * 1000
        self.lag_records = None
        if lag_records_capacity:
            from capcommon.capbin import RecordArray  # noqa: PLC0415
            self.lag_records = RecordArray([("planned_rel_ns", "q"), ("late_ns", "q"),
                                            ("tier_seq", "h")], lag_records_capacity)
        self.start_marks = {}
        self.end_marks = {}
        self.cur_seq = PRE_SEQ
        self.on_boundary = []
        self.finished = False
        self.writer = None
        self.lag_probe = None
        self.loop = None
        self.versions = None
        self.stop_doc = None

    # -- start / stop --------------------------------------------------------------------------
    def start(self, loop, info=None):
        """Version record, snapshot writer, loop-lag probe, tier boundary timers."""
        from capcommon.aio import LoopLagProbe, call_at_ns  # noqa: PLC0415
        self.loop = loop
        self.versions = version_record(loop=loop, include_grpc=True)
        meta = {"identity": self.identity.to_dict(), "role_hint": self.role,
                "versions": self.versions, "tool": self.tool}
        if info:
            meta["info"] = info
        self.writer = SnapshotWriter(self.registry, self.common.shm_dir,
                                     self.common.snapshot_interval_ns, meta_extra=meta).start()
        self.lag_probe = LoopLagProbe(loop, self.registry, "loop", self.lag_period_ns,
                                      records=self.lag_records,
                                      origin_ns=self.origin_ns if self.lag_records else None,
                                      tier_seq_at=self.tt.tier_seq_at if self.lag_records else None)
        self.lag_probe.start()
        for t in self.tt.tiers:
            call_at_ns(loop, self.origin_ns + t.start_ns, self._boundary, t.seq)
        call_at_ns(loop, self.origin_ns + self.tt.end_ns, self._boundary, len(self.tt))

    def _mark(self, with_inj=True):
        return {"vals": array.array("q", self.registry.values), "t": now_ns(),
                "cpu": (time.process_time_ns(), time.thread_time_ns()),
                "inj": self.injectors.mark() if with_inj else None}

    def _boundary(self, seq):
        """Runs on the loop thread at the absolute start of tier `seq` (seq == len = table end):
        end-mark the previous tier, apply this tier's inject specs, start-mark this tier."""
        if self.finished:
            return
        m = self._mark()
        prev = seq - 1
        if prev in self.start_marks and prev not in self.end_marks:
            m["completed"] = True
            self.end_marks[prev] = m
        if seq < len(self.tt):
            self.injectors.apply_tier_specs([s for s in self.tt[seq].inject
                                             if s.role == self.role])
            s = dict(m)
            s["inj"] = self.injectors.mark()
            self.start_marks[seq] = s
            self.cur_seq = seq
        else:
            self.cur_seq = POST_SEQ
        for cb in self.on_boundary:
            cb(seq)

    def close_tiers(self):
        """End-mark a tier that is still open (stop before its end); call on the loop thread."""
        if self.finished:
            return
        m = None
        for seq in sorted(self.start_marks):
            if seq not in self.end_marks:
                if m is None:
                    m = self._mark()
                    m["completed"] = False
                self.end_marks[seq] = m
        self.finished = True

    def stop(self):
        if self.lag_probe is not None:
            self.lag_probe.stop()
        if self.writer is not None:
            self.writer.stop()
        self.injectors.close()

    # -- slot arithmetic -----------------------------------------------------------------------
    def slot_delta(self, name, a, b):
        off, ln = self._pos[name]
        if ln == 1:
            return b[off] - a[off]
        return [b[off + i] - a[off + i] for i in range(ln)]

    def hist_summary(self, name, a, b):
        counts = self.slot_delta(name, a, b)
        return {"hist": {"bins": _hist.NAME, "counts": counts},
                "n": self.slot_delta(name + ".n", a, b),
                "sum": self.slot_delta(name + ".sum", a, b),
                "q99_upper_ns": _hist.quantile(counts, 0.99) if sum(counts) else None}

    def tier_common(self, seq):
        a, b = self.start_marks[seq], self.end_marks[seq]
        return {"tool": self.tool, "n_workers": self.n_workers,
                "tier_table_sha256": self.tt.sha256, "completed": b["completed"],
                "t_start_mono_ns": a["t"], "t_end_mono_ns": b["t"],
                "inject": tier_report(a["inj"], b["inj"]),
                "loop_lag_ns": self.hist_summary("lag.loop", a["vals"], b["vals"]),
                "cpu": {"process_ns": b["cpu"][0] - a["cpu"][0],
                        "loop_thread_ns": b["cpu"][1] - a["cpu"][1]},
                "stop": self.stop_doc}

    # -- outputs -------------------------------------------------------------------------------
    def read_stop(self):
        self.stop_doc = read_stop_file(self.tree.stop_json)
        return self.stop_doc

    def write_summary(self, seq, payload):
        label = self.tt.label_of(seq)
        path = self.tree.tier_file(seq, label, self.role, self.worker, "summary", "json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_json_stamped(path, TIER_SUMMARY_SCHEMA, self.identity.with_tier(seq, label), payload)
        return path

    def dump_records(self, rec, name, kind, meta=None):
        tt, tree, ident = self.tt, self.tree, self.identity
        return rec.dump_split(
            lambda v: tree.tier_file(v, tt.label_of(v), self.role, self.worker, name, "capbin"),
            kind, lambda v: ident.with_tier(v, tt.label_of(v)), key="tier_seq", meta=meta)

    def dump_lag(self):
        if self.lag_records is not None:
            self.dump_records(self.lag_records, "lag", "loop.lag",
                              {"period_ns": self.lag_period_ns,
                               "loop_impl": self.versions.get("loop_impl")})


# ---- supervisor (multi-process tools) ---------------------------------------------------------

_CODE_PRIORITY = (CONFIG, RUNTIME, 1, STOPPED_EARLY)


def combine_exit_codes(codes):
    """One exit code for a process group: CONFIG > RUNTIME > other non-zero > STOPPED_EARLY > OK."""
    codes = [c for c in codes if c not in (None, OK)]
    for want in _CODE_PRIORITY:
        if want == 1:
            other = [c for c in codes if c not in (CONFIG, RUNTIME, STOPPED_EARLY)]
            if other:
                return 1 if 1 in other else other[0]
        elif want in codes:
            return want
    return OK


class ParentLink:
    """Child side of the supervisor pipe: report ready / failure / done, receive stop."""

    def __init__(self, conn):
        self.conn = conn
        self._stopper = None

    def send(self, *msg):
        try:
            self.conn.send(msg)
        except (OSError, EOFError, ValueError):
            pass

    def ready(self, info):
        self.send("ready", info)

    def attach(self, loop, stopper):
        """Deliver ("stop", hard) messages and parent death (EOF) to `stopper` on the loop."""
        self._stopper = stopper
        loop.add_reader(self.conn.fileno(), self._on_readable, loop)

    def _on_readable(self, loop):
        try:
            msg = self.conn.recv()
        except (EOFError, OSError):
            loop.remove_reader(self.conn.fileno())
            self._stopper.trigger(None)  # parent gone: graceful stop
            return
        if msg and msg[0] == "stop":
            if msg[1] and not self._stopper.requested:
                self._stopper.trigger(None)
            self._stopper.trigger(None)


def child_main(args_dict, worker, conn, worker_fn):
    """Body of every spawned child: run worker_fn(args, worker, link) and report its exit code."""
    import argparse  # noqa: PLC0415
    import traceback  # noqa: PLC0415
    args = argparse.Namespace(**args_dict)
    args.worker = worker
    link = ParentLink(conn)
    try:
        code = worker_fn(args, worker, link)
    except CapkitError as e:
        print("%s worker %d: %s" % (args.role, worker, e), file=sys.stderr)
        code = e.exit_code
        link.send("fail", code, str(e))
    except BaseException as e:  # noqa: BLE001  -- report, then exit non-zero
        traceback.print_exc()
        code = 1
        link.send("fail", code, repr(e))
    link.send("done", code)
    sys.exit(code)


def worker_ready_info(rt, info):
    """What a worker reports to its supervisor (and what an in-process tool writes itself)."""
    return {"pid": os.getpid(), "worker": rt.worker, "affinity": rt.affinity,
            "versions": rt.versions, "aux_threads": rt.aux_threads, "info": info}


def write_inprocess_ready(rt, ready_file, info):
    write_ready_file(ready_file, rt.identity, rt.versions, affinity=rt.affinity, children=[],
                     info={"tool": rt.tool, "workers": [worker_ready_info(rt, info)],
                           "threads": rt.aux_threads})


def run_supervisor(args, common, n_workers, child_target, tool, slots, poll_s=0.02,
                   on_all_ready=None):
    """Parent process of a multi-process tool. Spawns n_workers children
    (child_target(vars(args), k, conn)), waits for all to be ready before the origin, writes the
    ready file listing every child pid, relays stop requests, returns the combined exit code.

    on_all_ready(ident): called once every child is ready, before the ready file is written
    (servers write their port file there)."""
    from capcommon.aio import run as aio_run  # noqa: PLC0415
    cpus = common.cpus or []
    if len(cpus) < n_workers:
        raise ConfigError("--cpus lists %d cpus for %d workers" % (len(cpus), n_workers))
    affinity = pin_self(cpus[:n_workers])
    ident = common.identity.with_pid()
    registry = SlotRegistry()
    registry.register_decls(slots)
    alive_off = registry.gauge("self.children_alive")

    async def supervise(loop):
        versions = version_record(loop=loop, include_grpc=True)
        writer = SnapshotWriter(registry, common.shm_dir, common.snapshot_interval_ns,
                                meta_extra={"identity": ident.to_dict(), "role_hint": ident.role,
                                            "versions": versions, "tool": tool,
                                            "info": {"supervisor": True}}).start()
        stopper = StopController()
        ctx = multiprocessing.get_context("spawn")
        procs, conns = [], []
        for k in range(n_workers):
            parent_end, child_end = ctx.Pipe(duplex=True)
            p = ctx.Process(target=child_target, args=(dict(vars(args)), k, child_end),
                            name="%s-w%d" % (tool, k))
            p.start()
            child_end.close()
            procs.append(p)
            conns.append(parent_end)
        registry.values[alive_off] = n_workers
        ready, codes, fails = {}, {}, []
        sent_stop = [0]

        def relay_stop(hard):
            sent_stop[0] += 1
            for c in conns:
                try:
                    c.send(("stop", hard))
                except (OSError, ValueError):
                    pass

        def drain_msgs():
            for k, c in enumerate(conns):
                while True:
                    try:
                        if not c.poll():
                            break
                        msg = c.recv()
                    except (EOFError, OSError):
                        break
                    if msg[0] == "ready":
                        ready[k] = msg[1]
                    elif msg[0] == "fail":
                        fails.append((k, msg[1], msg[2]))
                    elif msg[0] == "done":
                        codes[k] = msg[1]

        last_signals = 0
        ok_ready = True
        while len(ready) < n_workers:
            drain_msgs()
            dead = [k for k, p in enumerate(procs) if p.exitcode is not None and k not in ready]
            if fails or dead or stopper.requested:
                ok_ready = False
                break
            if now_ns() > common.origin_ns:
                fails.append((-1, RUNTIME, "workers not ready before the origin"))
                ok_ready = False
                break
            await asyncio.sleep(poll_s)
        if ok_ready:
            try:
                if on_all_ready is not None:
                    on_all_ready(ident)
                write_ready_file(common.ready_file, ident, versions, affinity=affinity,
                                 children=[{"pid": ready[k]["pid"], "worker": k,
                                            "role": ident.role} for k in range(n_workers)],
                                 info={"tool": tool,
                                       "workers": [ready[k] for k in range(n_workers)]})
            except (CapkitError, OSError) as e:
                # the role never became ready: stop the workers instead of leaving them running
                fails.append((-1, getattr(e, "exit_code", RUNTIME), "port/ready file: %s" % e))
                ok_ready = False
        if not ok_ready:
            relay_stop(True)
        while any(p.exitcode is None for p in procs):
            if len(stopper.signals) > last_signals:
                last_signals = len(stopper.signals)
                relay_stop(stopper.hard)
            drain_msgs()
            registry.values[alive_off] = sum(1 for p in procs if p.exitcode is None)
            await asyncio.sleep(poll_s)
        drain_msgs()
        for p in procs:
            p.join()
        registry.values[alive_off] = 0
        writer.stop()
        for k, k_code, why in fails:
            print("%s: worker %d failed (%s): %s" % (tool, k, k_code, why), file=sys.stderr)
        all_codes = [codes.get(k, p.exitcode) for k, p in enumerate(procs)]
        all_codes += [c for k, c, _ in fails if k == -1]
        return combine_exit_codes(all_codes)

    return aio_run(supervise, common.loop)
