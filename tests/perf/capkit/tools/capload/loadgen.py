# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One capload worker process: an event loop that sends its share of the arrival schedule at the
planned instants (open loop), or in closed-loop waves, over HTTP/1.1 and counts SSE frames.

Hot-path rules (INTERFACES.md 0): time is time.monotonic_ns(); every record goes into a
preallocated RecordArray and is written only after the last tier; latency is kept from the
planned instant and from the actual write.
"""

import array
import asyncio
import fcntl
import os
import random
import socket
import termios
import time
from collections import OrderedDict

from capcommon import wire
from capcommon.aio import call_at_ns
from capcommon.capbin import RecordArray

from .httpwire import ChunkedDecoder, FullSseParser, ProtocolError, parse_head

# Request status codes (requests.capbin column `status`; meta.status_codes maps them). HTTP
# status h maps to HTTP_BASE + h.
ST_OK = 0
ST_CONN_REFUSED = 1
ST_CONN_FAIL = 2
ST_CONN_RESET = 3
ST_TIMEOUT = 4
ST_PROTOCOL = 5
ST_SHORT = 6
ST_ABORTED = 7
HTTP_BASE = 1000
STATUS_NAMES = {ST_OK: "ok", ST_CONN_REFUSED: "conn_refused", ST_CONN_FAIL: "conn_fail",
                ST_CONN_RESET: "conn_reset", ST_TIMEOUT: "timeout_registered",
                ST_PROTOCOL: "protocol_error", ST_SHORT: "short_output",
                ST_ABORTED: "aborted_at_stop"}
CLASS_HINT = {"conn_refused": "non_overload", "conn_fail": "unclassified",
              "conn_reset": "unclassified", "timeout_registered": "overload",
              "protocol_error": "non_overload", "short_output": "non_overload",
              "aborted_at_stop": "unclassified"}
CONNECT_FAILURES = ("conn_refused", "conn_fail")


def status_name(st):
    if st >= HTTP_BASE:
        return "http_%d" % (st - HTTP_BASE)
    return STATUS_NAMES[st]


# Connection close reasons (conns.capbin column `close_reason`).
CL_OPEN = 0            # still open when the records were written
CL_PEER_IDLE = 1       # peer closed while the connection was idle in the pool
CL_PEER_BUSY = 2       # peer closed / reset while a request was in flight
CL_DONE_CLOSE = 3      # closed by us after a response (new-per-request, or Connection: close)
CL_POOL_RESET = 4      # idle connection closed by --pool-reset per-tier at a load tier start
CL_ERROR = 5           # closed by us after a protocol / HTTP error
CL_TIMEOUT = 6         # aborted by us: registered request timeout
CL_STOP = 7            # aborted / closed by us at stop
CL_UNSOLICITED = 8     # peer sent bytes while no request was in flight
CLOSE_NAMES = {CL_OPEN: "open", CL_PEER_IDLE: "peer_closed_idle", CL_PEER_BUSY: "peer_closed_busy",
               CL_DONE_CLOSE: "closed_after_response", CL_POOL_RESET: "pool_reset",
               CL_ERROR: "closed_after_error", CL_TIMEOUT: "timeout_abort", CL_STOP: "stop",
               CL_UNSOLICITED: "unsolicited_bytes"}

REQ_FIELDS = [("req_id", "q"), ("planned_rel_ns", "q"), ("sent_rel_ns", "q"),
              ("first_rel_ns", "q"), ("done_rel_ns", "q"), ("frames", "i"), ("bytes", "q"),
              ("tokens", "i"), ("status", "h"), ("conn_id", "i"), ("new_conn", "b"),
              ("tier_seq", "h")]
CHUNK_FIELDS = [("req_id", "q"), ("idx", "i"), ("recv_rel_ns", "q"), ("nbytes", "i"),
                ("tier_seq", "h")]
FION_FIELDS = [("t_rel_ns", "q"), ("conn_id", "i"), ("bytes", "i"), ("tier_seq", "h")]
LAG_FIELDS = [("planned_rel_ns", "q"), ("late_ns", "q"), ("tier_seq", "h")]
CONN_FIELDS = [("conn_id", "i"), ("local_port", "i"), ("peer_port", "i"),
               ("opened_rel_ns", "q"), ("closed_rel_ns", "q"), ("n_requests", "i"),
               ("close_reason", "h"), ("tier_seq", "h")]

_FIONREAD = termios.FIONREAD


class Req:
    """State of one request from dispatch to completion."""

    __slots__ = ("rid", "planned", "tier", "sent", "first", "frames", "seen", "nbytes",
                 "conn", "new_conn", "reused", "retried", "sampled", "deadline", "counter",
                 "full", "rbuf", "http", "dec", "remain", "close_delim", "peer_close",
                 "expect_tokens", "wave")

    def __init__(self, rid, planned, tier, expect_tokens, sampled, deadline):
        self.rid = rid
        self.planned = planned  # absolute monotonic ns
        self.tier = tier
        self.expect_tokens = expect_tokens
        self.sampled = sampled
        self.deadline = deadline
        self.sent = -1
        self.first = -1
        self.frames = 0  # reported count (miscount injection can skew it on purpose)
        self.seen = 0  # true ordinal of frames received (chunks.idx)
        self.nbytes = 0
        self.conn = None
        self.new_conn = 0
        self.reused = False
        self.retried = False
        self.counter = wire.SseCounter()
        self.full = None
        self.rbuf = b""
        self.http = 0  # status code once the head is parsed
        self.dec = None
        self.remain = -1
        self.close_delim = False
        self.peer_close = False
        self.wave = False


class Conn(asyncio.Protocol):
    """One TCP connection to the target; parsing is done by the worker (one place)."""

    def __init__(self, worker, cid):
        self.w = worker
        self.cid = cid
        self.tr = None
        self.fd = -1
        self.req = None
        self.n_requests = 0
        self.opened = -1
        self.local_port = -1
        self.peer_port = -1
        self.close_reason = None  # set before we close it ourselves
        self.idle = False
        self.lost = False

    def connection_made(self, transport):
        self.tr = transport
        sock = transport.get_extra_info("socket")
        self.fd = sock.fileno() if sock is not None else -1
        if sock is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
        ln = transport.get_extra_info("sockname")
        pn = transport.get_extra_info("peername")
        self.local_port = ln[1] if ln else -1
        self.peer_port = pn[1] if pn else -1
        self.w.on_made(self)

    def data_received(self, data):
        r = self.req
        if r is None:
            self.w.on_unsolicited(self)
            return
        self.w.on_data(self, r, data)

    def eof_received(self):
        return False  # close our side too; connection_lost follows

    def connection_lost(self, exc):
        self.lost = True
        self.w.on_lost(self, exc)


class IdlePool:
    """Idle keep-alive connections. pick policy: roundrobin = least recently released first
    (FIFO over idle connections); random = uniform among idle connections (seeded)."""

    def __init__(self, policy, seed):
        self.policy = policy
        self.rng = random.Random(seed)
        self.fifo = OrderedDict()
        self.lst = []
        self.pos = {}

    def __len__(self):
        return len(self.fifo) if self.policy == "roundrobin" else len(self.lst)

    def add(self, conn):
        if self.policy == "roundrobin":
            self.fifo[conn.cid] = conn
        else:
            self.pos[conn.cid] = len(self.lst)
            self.lst.append(conn)

    def remove(self, conn):
        if self.policy == "roundrobin":
            return self.fifo.pop(conn.cid, None) is not None
        i = self.pos.pop(conn.cid, None)
        if i is None:
            return False
        last = self.lst.pop()
        if last is not conn:
            self.lst[i] = last
            self.pos[last.cid] = i
        return True

    def pick(self):
        if self.policy == "roundrobin":
            if not self.fifo:
                return None
            return self.fifo.popitem(last=False)[1]
        if not self.lst:
            return None
        conn = self.lst[self.rng.randrange(len(self.lst))]
        self.remove(conn)
        return conn

    def all(self):
        return list(self.fifo.values()) if self.policy == "roundrobin" else list(self.lst)


class Worker:
    """One load-generator worker (runs inside a spawned process)."""

    def __init__(self, cfg, worker, chan_conn):
        from capcommon.identity import Identity  # noqa: PLC0415
        from capcommon.inject import parse_inject_spec  # noqa: PLC0415
        from capcommon.lifecycle import pin_self  # noqa: PLC0415
        from capcommon.schedule import load_schedule, per_worker_tier_counts  # noqa: PLC0415
        from capcommon.tiers import TierTable  # noqa: PLC0415
        from .httpwire import RequestBuilder, load_prompts  # noqa: PLC0415

        self.cfg = cfg
        self.k = worker
        self.W = cfg["workers"]
        self.chan_conn = chan_conn
        self.ident = Identity.from_dict(cfg["identity"]).with_worker(worker).with_pid()
        self.role = self.ident.role
        cpus = cfg["cpus"]
        self.affinity = pin_self([cpus[worker]]) if cpus else None
        self.table = TierTable.load(cfg["tier_table"])
        self.origin = cfg["origin_ns"]
        rids, prel, tseq, meta = load_schedule(cfg["schedule"], worker, self.W,
                                               tier_table_sha=self.table.sha256)
        self.rids, self.prel, self.tseq, self.sched_meta = rids, prel, tseq, meta
        self.sched_sha = cfg["schedule_sha256"]
        self.n = len(rids)
        planned_all = per_worker_tier_counts(meta, self.W)
        self.planned = {seq: c for (w, seq), c in planned_all.items() if w == worker}
        needed = {}
        for i in range(self.n):
            needed.setdefault(tseq[i], set()).add(rids[i])
        for i in range(1, self.n):
            if prel[i] < prel[i - 1]:
                from capcommon.exitcodes import ConfigError  # noqa: PLC0415
                raise ConfigError("schedule planned times are not monotone at row %d" % i)
        self.prompt_sets = load_prompts(cfg["prompts"])
        self.builder = RequestBuilder(cfg["endpoint"], cfg["model"], cfg["host_header"],
                                      cfg["conn_policy"] == "new-per-request", self.table,
                                      self.prompt_sets, needed, cfg.get("body_extra"))
        self.cli_specs = [parse_inject_spec(t) for t in cfg["inject"]]
        self.keepalive = cfg["conn_policy"] == "keepalive"
        self.full_mode = cfg["recv_mode"] == "full"
        self.sample_mod = cfg["chunk_sample_mod"]
        self.tick_ns = cfg["fionread_every_ms"] * 1_000_000
        self.timeout_ns = self.table.request_timeout_ns
        self.grace_ns = int(cfg["stop_grace_s"] * 1e9)
        self.pause_s = cfg["slow_read_pause_us"] / 1e6
        self.mode = cfg["mode"]
        self.tier_max_tokens = [t.max_tokens for t in self.table.tiers]
        self.end_abs = self.origin + self.table.end_ns
        # runtime state
        self.inflight = {}
        self.open_conns = {}
        self.next_cid = 0
        self.next_i = 0
        self.dispatch_handle = None
        self.stopping = False
        self.stop_reason = None
        self.stopped_early = False
        self.hard = False
        self.end_passed = False
        self.finished = None
        self.dispatched_by_tier = {}
        self.written = 0
        self.ev = {}  # event-time tier -> connection event counters
        self.marks_start = {}
        self.marks_end = {}
        self.cur_seq = None
        self.unsolicited = 0
        self.target = None
        self.dumped = False
        self.grace_handle = None
        self.tick_handle = None
        self.wave_handles = []
        self.wave_all_dispatched = False
        self._fbuf = array.array("i", [0])

    # ---- set-up --------------------------------------------------------------------------
    def _resolve_target(self):
        from capcommon.lifecycle import wait_port_file  # noqa: PLC0415
        if self.cfg["target"]:
            host, _, port = self.cfg["target"].rpartition(":")
            return host, int(port)
        timeout = max(0.0, (self.origin - time.monotonic_ns()) / 1e9)
        host, port, _doc = wait_port_file(self.cfg["target_port_file"], timeout)
        return host, port

    def run(self):
        from capcommon.aio import run as aio_run  # noqa: PLC0415
        return aio_run(self._main, self.cfg["loop"])

    async def _main(self, loop):
        from capcommon.aio import LoopLagProbe  # noqa: PLC0415
        from capcommon.snapshot import SlotRegistry, SnapshotWriter  # noqa: PLC0415
        from capcommon.versions import version_record  # noqa: PLC0415
        from . import LOADGEN_SITES, LOADGEN_SLOTS  # noqa: PLC0415
        from ._proc import WorkerChannel  # noqa: PLC0415
        from ._util import build_injectors  # noqa: PLC0415

        self.loop = loop
        self.finished = loop.create_future()
        self.target = self._resolve_target()
        reg = SlotRegistry()
        reg.register_decls(LOADGEN_SLOTS)
        self.reg = reg
        self.vals = reg.values
        self.i_inflight = reg.gauge("backlog.inflight")  # idempotent: returns the offset
        self.i_unread = reg.gauge("backlog.unread_bytes")
        self.i_disp = reg.counter("self.dispatched")
        self.i_written = reg.counter("self.written")
        self.i_ok = reg.counter("self.ok")
        self.i_failed = reg.counter("self.failed")
        self.i_frames = reg.counter("self.frames")
        self.i_bytes = reg.counter("self.bytes")
        self.i_cnew = reg.counter("self.conn_new")
        self.i_crec = reg.counter("self.conn_reconnect")
        self.i_cidle = reg.counter("self.conn_peer_closed_idle")
        self.i_cfail = reg.counter("self.conn_fail")
        self.i_psize = reg.gauge("self.pool_size")
        self.i_pidle = reg.gauge("self.pool_idle")
        self.h_lag = reg.hist("self.send_lag")
        self.injs, self.aux_threads = build_injectors(LOADGEN_SITES, reg, self.cli_specs,
                                                      self.table, self.role)
        self.inj_send = self.injs["send.req"]
        self.inj_recv = self.injs["recv.chunk"]
        self.versions = version_record(loop=loop, include_grpc=False)
        self.writer = SnapshotWriter(reg, self.cfg["shm_dir"], self.cfg["snapshot_interval_ns"],
                                     meta_extra={"identity": self.ident.to_dict(),
                                                 "versions": self.versions,
                                                 "role_hint": self.role})
        cap = self.cfg["max_records"]
        self.rec_req = RecordArray(REQ_FIELDS, self.n)
        self.rec_chunk = RecordArray(CHUNK_FIELDS, cap)
        self.rec_fion = RecordArray(FION_FIELDS, cap)
        # One lag sample per period from now until the last tier ends (samples after the last
        # tier are not recorded); derived from the config like capload.nullsse does, so the
        # lag records cannot overflow because --max-records-per-worker was sized for requests.
        period_ns = self.cfg["lag_period_us"] * 1000
        lag_cap = -(-(self.table.end_ns + max(0, self.origin - time.monotonic_ns()))
                    // period_ns) + 2
        self.rec_lag = RecordArray(LAG_FIELDS, lag_cap)
        self.rec_conn = RecordArray(CONN_FIELDS, cap)
        self.pool = IdlePool(self.cfg["pool_pick"], self.cfg["pool_seed"] * 1_000_003 + self.k)
        self.writer.start()
        self.lag = LoopLagProbe(loop, reg, "loop", self.cfg["lag_period_us"] * 1000,
                                records=self.rec_lag, origin_ns=self.origin,
                                tier_seq_at=self.table.tier_seq_at).start()
        self.chan = WorkerChannel(self.chan_conn, loop, self.on_stop)
        self._arm_boundaries()
        self._arm_tick(time.monotonic_ns())
        if self.mode == "open":
            self._arm_dispatch()
        else:
            self._wave_init()
        self.chan.send(("ready", {
            "pid": os.getpid(), "worker": self.k, "versions": self.versions,
            "affinity": self.affinity, "n_rows": self.n, "request_bytes": self.builder.total_bytes,
            "aux_threads": self.aux_threads, "target": "%s:%d" % self.target,
            "ready_rel_ns": time.monotonic_ns() - self.origin}))
        await self.finished
        return self._teardown()

    # ---- tier boundaries -----------------------------------------------------------------
    def _ev(self, seq):
        e = self.ev.get(seq)
        if e is None:
            e = self.ev[seq] = {"new_conns": 0, "reconnects": 0, "peer_idle_closed": 0,
                                "conn_fail": 0, "size_max": len(self.open_conns),
                                "unsolicited": 0, "pool_reset_closed": 0}
        return e

    def _arm_boundaries(self):
        pts = [(t.start_ns, t.seq) for t in self.table.tiers] + [(self.table.end_ns, None)]
        for rel, seq in pts:
            call_at_ns(self.loop, self.origin + rel, self._boundary, seq)

    def _boundary(self, seq):
        from ._util import tier_specs_for  # noqa: PLC0415
        rel = time.monotonic_ns() - self.origin
        mark = self.injs.mark()
        if self.cur_seq is not None:
            self.marks_end[self.cur_seq] = (mark, rel)
        if seq is None:
            self.cur_seq = None
            self.end_passed = True
            self.lag.records = None  # after the last tier: histogram only, no per-sample rows
            self._maybe_finish()
            return
        self.injs.apply_tier_specs(tier_specs_for(self.table, self.role, seq))
        self.marks_start[seq] = (self.injs.mark(), rel)
        self.cur_seq = seq
        ev = self._ev(seq)
        if len(self.open_conns) > ev["size_max"]:
            ev["size_max"] = len(self.open_conns)
        if self.cfg["pool_reset"] == "per-tier" and self.table.tiers[seq].is_load:
            for c in self.pool.all():
                self._close(c, CL_POOL_RESET)
                ev["pool_reset_closed"] += 1
            self.vals[self.i_pidle] = len(self.pool)

    # ---- sending -------------------------------------------------------------------------
    def _arm_dispatch(self):
        if self.next_i < self.n and not self.stopping:
            self.dispatch_handle = call_at_ns(self.loop, self.origin + self.prel[self.next_i],
                                              self._dispatch)
        else:
            self.dispatch_handle = None
            self._maybe_finish()

    def _dispatch(self):
        """Send every row whose planned instant has passed (never skip, never send early)."""
        origin = self.origin
        prel = self.prel
        n = self.n
        i = self.next_i
        mono = time.monotonic_ns
        while i < n and prel[i] <= mono() - origin and not self.stopping:
            self._send_row(i, origin + prel[i])
            i += 1
        self.next_i = i
        self._arm_dispatch()

    def _send_row(self, i, planned_abs, wave=False):
        rid = self.rids[i]
        seq = self.tseq[i]
        deadline = planned_abs + self.timeout_ns if self.timeout_ns else 0
        r = Req(rid, planned_abs, seq, self.tier_max_tokens[seq], rid % self.sample_mod == 0,
                deadline)
        r.wave = wave
        if self.full_mode:
            r.full = FullSseParser(rid)
        self.inflight[rid] = r
        v = self.vals
        v[self.i_inflight] += 1
        v[self.i_disp] += 1
        d = self.dispatched_by_tier
        d[seq] = d.get(seq, 0) + 1
        self.inj_send.hit()
        if self.keepalive:
            c = self.pool.pick()
            if c is not None:
                v[self.i_pidle] = len(self.pool)
                r.reused = True
                self._write(c, r)
                return
        self.loop.create_task(self._connect(r))

    async def _connect(self, r):
        cid = self.next_cid
        self.next_cid += 1
        host, port = self.target
        try:
            _tr, conn = await self.loop.create_connection(lambda: Conn(self, cid), host, port)
        except ConnectionRefusedError:
            self._conn_failed(r, ST_CONN_REFUSED)
            return
        except OSError:
            self._conn_failed(r, ST_CONN_FAIL)
            return
        if conn.lost:
            if r.rid in self.inflight:
                self._finish(r, ST_CONN_RESET, time.monotonic_ns())
            return
        if r.rid not in self.inflight:  # timed out / aborted while connecting
            self._release(conn, reusable=True)
            return
        r.new_conn = 1
        self._write(conn, r)

    def _conn_failed(self, r, st):
        now = time.monotonic_ns()
        self._ev(self.table.tier_seq_at(now - self.origin))["conn_fail"] += 1
        self.vals[self.i_cfail] += 1
        if r.rid in self.inflight:
            self._finish(r, st, now)

    def _write(self, conn, r):
        conn.req = r
        conn.idle = False
        r.conn = conn
        conn.n_requests += 1
        data = self.builder.render(r.rid, r.tier)
        now = time.monotonic_ns()
        conn.tr.write(data)
        if r.sent < 0:
            # A keep-alive reconnect writes the same request again; the request is written,
            # and its send lag sampled, once (at the first write).
            r.sent = now
            self.vals[self.i_written] += 1
            self.written += 1
            self.h_lag.add(now - r.planned)

    # ---- receiving -----------------------------------------------------------------------
    def on_made(self, conn):
        now = time.monotonic_ns()
        conn.opened = now
        self.open_conns[conn.cid] = conn
        n_open = len(self.open_conns)
        self.vals[self.i_cnew] += 1
        self.vals[self.i_psize] = n_open
        ev = self._ev(self.table.tier_seq_at(now - self.origin))
        ev["new_conns"] += 1
        if n_open > ev["size_max"]:
            ev["size_max"] = n_open

    def on_unsolicited(self, conn):
        now = time.monotonic_ns()
        self.unsolicited += 1
        self._ev(self.table.tier_seq_at(now - self.origin))["unsolicited"] += 1
        self._close(conn, CL_UNSOLICITED)

    def on_data(self, conn, r, data):
        now = time.monotonic_ns()
        if not r.http:
            buf = r.rbuf + data if r.rbuf else data
            try:
                head = parse_head(buf)
            except ProtocolError:
                self._fail_conn(conn, r, ST_PROTOCOL, now)
                return
            if head is None:
                r.rbuf = buf
                return
            status, hdrs, hl = head
            r.rbuf = b""
            echo = hdrs.get(wire.HDR_ID)
            if echo is not None and echo != str(r.rid):
                self._fail_conn(conn, r, ST_PROTOCOL, now)
                return
            r.http = status
            if "chunked" in hdrs.get("transfer-encoding", "").lower():
                r.dec = ChunkedDecoder()
            elif "content-length" in hdrs:
                try:
                    r.remain = int(hdrs["content-length"])
                except ValueError:
                    self._fail_conn(conn, r, ST_PROTOCOL, now)
                    return
            else:
                r.close_delim = True
            r.peer_close = hdrs.get("connection", "").lower() == "close"
            data = buf[hl:]
            if not data:
                if r.remain == 0:
                    self._complete(conn, r, now)
                return
        nb = len(data)
        r.nbytes += nb
        self.vals[self.i_bytes] += nb
        if r.dec is not None:
            try:
                payload, used = r.dec.feed(data)
            except ProtocolError:
                self._fail_conn(conn, r, ST_PROTOCOL, now)
                return
            done = r.dec.done
            if done and used != nb:  # bytes after the end of the body (no pipelining here)
                self._fail_conn(conn, r, ST_PROTOCOL, now)
                return
        elif r.remain >= 0:
            if nb > r.remain:
                self._fail_conn(conn, r, ST_PROTOCOL, now)
                return
            payload = data
            r.remain -= nb
            done = r.remain == 0
        else:
            payload = data
            done = False
        if payload and 200 <= r.http < 300:
            nf = r.counter.feed(payload)
            if r.full is not None:
                r.full.feed(payload)
            if nf:
                self._frames(r, nf, now, nb)
        if done:
            self._complete(conn, r, now)
        elif self.pause_s and not conn.lost:
            conn.tr.pause_reading()  # deliberate slow reader (FIONREAD positive control)
            self.loop.call_later(self.pause_s, self._resume, conn)

    def _resume(self, conn):
        if not conn.lost and conn.close_reason is None:
            conn.tr.resume_reading()

    def _frames(self, r, nf, now, nb):
        inj = self.inj_recv
        v = self.vals
        rel = now - self.origin
        for _ in range(nf):
            idx = r.seen
            r.seen = idx + 1
            cnt = 1
            if inj.active:
                mode = inj.spec.mode
                if mode == "drop":
                    if inj.should_drop():
                        continue
                elif mode == "miscount":
                    cnt = inj.count_delta()
                else:
                    inj.hit()
            else:
                inj.hit()
            if r.first < 0:
                r.first = now
            r.frames += cnt
            v[self.i_frames] += cnt
            if r.sampled:
                self.rec_chunk.append(r.rid, idx, rel, nb, self.table.tier_seq_at(rel))

    def _complete(self, conn, r, now):
        c = r.counter
        f = r.full
        if not 200 <= r.http < 300:
            st = HTTP_BASE + r.http
        elif not c.done_seen or c.leftover:
            st = ST_PROTOCOL
        elif f is not None and (f.frames != c.frames or f.errors or f.id_mismatch
                                or not f.done_seen or f.leftover
                                or f.completion_tokens != c.completion_tokens):
            st = ST_PROTOCOL
        elif c.completion_tokens is None:
            st = ST_PROTOCOL
        elif c.completion_tokens != r.expect_tokens:
            st = ST_SHORT
        else:
            st = ST_OK
        conn.req = None
        reusable = (st != ST_PROTOCOL and not r.peer_close and not r.close_delim)
        self._finish(r, st, now)
        self._release(conn, reusable)

    def _release(self, conn, reusable):
        """Return a finished connection to the idle pool or close it."""
        if conn.lost or conn.close_reason is not None:
            return
        if reusable and self.keepalive and not self.stopping:
            conn.idle = True
            self.pool.add(conn)
            self.vals[self.i_pidle] = len(self.pool)
        else:
            self._close(conn, CL_DONE_CLOSE if reusable or not self.keepalive else CL_ERROR)

    def _fail_conn(self, conn, r, st, now):
        conn.req = None
        self._finish(r, st, now)
        self._close(conn, CL_ERROR)

    def _close(self, conn, reason, abort=False):
        if conn.lost or conn.close_reason is not None:
            return
        conn.close_reason = reason
        if self.pool.remove(conn):
            self.vals[self.i_pidle] = len(self.pool)
        if abort:
            conn.tr.abort()
        else:
            conn.tr.close()

    def on_lost(self, conn, exc):
        now = time.monotonic_ns()
        self.open_conns.pop(conn.cid, None)
        v = self.vals
        v[self.i_psize] = len(self.open_conns)
        if self.pool.remove(conn):
            v[self.i_pidle] = len(self.pool)
        reason = conn.close_reason
        r = conn.req
        conn.req = None
        ts = self.table.tier_seq_at(now - self.origin)
        if reason is None:
            reason = CL_PEER_IDLE if r is None else CL_PEER_BUSY
            conn.close_reason = reason
            if r is None:
                self._ev(ts)["peer_idle_closed"] += 1
                v[self.i_cidle] += 1
        if r is not None and r.rid in self.inflight:
            if r.http and r.close_delim:
                self._complete(conn, r, now)  # a close-delimited body ends at EOF
            elif (not r.http and not r.rbuf and r.nbytes == 0 and r.reused and not r.retried
                  and not self.stopping and reason == CL_PEER_BUSY):
                # the peer closed an idle keep-alive connection just as we reused it: one
                # reconnect, counted apart from request failures (design 2.2 item 4)
                r.retried = True
                r.reused = False
                r.conn = None
                self._ev(ts)["reconnects"] += 1
                v[self.i_crec] += 1
                self.loop.create_task(self._connect(r))
            else:
                self._finish(r, ST_CONN_RESET, now)
        if not self.dumped:
            o = self.origin
            self.rec_conn.append(conn.cid, conn.local_port, conn.peer_port, conn.opened - o,
                                 now - o, conn.n_requests, reason,
                                 self.table.tier_seq_at(conn.opened - o))

    def _finish(self, r, st, now, check=True):
        if self.inflight.pop(r.rid, None) is None:
            return
        v = self.vals
        v[self.i_inflight] -= 1
        if st == ST_OK:
            v[self.i_ok] += 1
        else:
            v[self.i_failed] += 1
        o = self.origin
        tokens = r.counter.completion_tokens
        self.rec_req.append(r.rid, r.planned - o, r.sent - o if r.sent >= 0 else -1,
                            r.first - o if r.first >= 0 else -1, now - o, r.frames, r.nbytes,
                            tokens if tokens is not None else -1, st,
                            r.conn.cid if r.conn is not None else -1, r.new_conn, r.tier)
        if r.wave and check:
            self._wave_done()
        if check:
            self._maybe_finish()

    # ---- housekeeping: FIONREAD sampling, registered timeouts, gauges --------------------
    def _arm_tick(self, planned):
        self.tick_planned = planned
        self.tick_handle = call_at_ns(self.loop, planned, self._tick)

    def _tick(self):
        now = time.monotonic_ns()
        rel = now - self.origin
        ts = self.table.tier_seq_at(rel)
        buf = self._fbuf
        tot = 0
        rec = self.rec_fion
        for c in list(self.open_conns.values()):
            if c.fd < 0 or c.lost or c.close_reason is not None:
                continue
            try:
                fcntl.ioctl(c.fd, _FIONREAD, buf, True)
            except OSError:
                continue
            b = buf[0]
            tot += b
            rec.append(rel, c.cid, b, ts)
        self.vals[self.i_unread] = tot
        if self.timeout_ns:
            late = [r for r in self.inflight.values() if r.deadline <= now]
            for r in late:
                conn = r.conn
                if conn is not None:
                    conn.req = None
                    self._close(conn, CL_TIMEOUT, abort=True)
                self._finish(r, ST_TIMEOUT, now)
        if self.finished.done():
            return
        nxt = self.tick_planned + self.tick_ns
        if nxt <= now:  # fell behind: do not burst
            nxt = now + self.tick_ns
        self._arm_tick(nxt)

    # ---- stopping ------------------------------------------------------------------------
    def _all_dispatched(self):
        if self.mode == "open":
            return self.next_i >= self.n
        return self.wave_all_dispatched

    def _maybe_finish(self):
        if self.finished is None or self.finished.done():
            return
        if self.stopping:
            if not self.inflight:
                self._finish_run()
            return
        if self.end_passed:
            if self._all_dispatched() and not self.inflight:
                self._finish_run()
            elif self.grace_handle is None:
                self.grace_handle = self.loop.call_later(self.grace_ns / 1e9, self._grace_expired)

    def on_stop(self, hard, reason):
        if self.finished.done():
            return
        if not self.end_passed:
            self.stopped_early = True
        if self.stop_reason is None:
            self.stop_reason = reason
        self._halt_dispatch()
        if hard:
            self.hard = True
            self._abort_all()
            self._finish_run()
            return
        if not self.inflight:
            self._finish_run()
        elif self.grace_handle is None:
            self.grace_handle = self.loop.call_later(self.grace_ns / 1e9, self._grace_expired)

    def _halt_dispatch(self):
        self.stopping = True
        if self.dispatch_handle is not None:
            self.dispatch_handle.cancel()
            self.dispatch_handle = None
        for h in self.wave_handles:
            h.cancel()
        self.wave_handles = []

    def _grace_expired(self):
        self.grace_handle = None
        if self.finished.done():
            return
        self._halt_dispatch()
        self._abort_all()
        self._finish_run()

    def _abort_all(self):
        now = time.monotonic_ns()
        for r in list(self.inflight.values()):
            conn = r.conn
            if conn is not None:
                conn.req = None
                self._close(conn, CL_STOP, abort=True)
            self._finish(r, ST_ABORTED, now, check=False)

    def _finish_run(self):
        if not self.finished.done():
            self.finished.set_result(None)

    # ---- closed-loop wave mode -----------------------------------------------------------
    def _wave_init(self):
        groups = []
        for i in range(self.n):
            seq = self.tseq[i]
            if not groups or groups[-1][0] != seq:
                groups.append((seq, []))
            groups[-1][1].append(i)
        self.wave_groups = groups
        self.wave_gi = 0
        self.wave_ri = 0
        self.wave_pending = 0
        self.wave_all_dispatched = not groups
        self.wave_size = self.cfg["wave_size"]
        self.wave_delays = self.cfg["wave_delays_ns"] or [0] * self.wave_size
        self.wave_gap = self.cfg["wave_gap_ns"]
        self.wave_count = 0
        self._wave_next(0)

    def _wave_next(self, earliest_abs):
        if self.stopping:
            return
        groups = self.wave_groups
        while self.wave_gi < len(groups) and self.wave_ri >= len(groups[self.wave_gi][1]):
            self.wave_gi += 1
            self.wave_ri = 0
        if self.wave_gi >= len(groups):
            self.wave_all_dispatched = True
            self._maybe_finish()
            return
        seq, rows = groups[self.wave_gi]
        t0 = max(self.origin + self.table.tiers[seq].start_ns, earliest_abs)
        wave = rows[self.wave_ri:self.wave_ri + self.wave_size]
        self.wave_ri += len(wave)
        self.wave_pending = len(wave)
        self.wave_count += 1
        self.wave_handles = []
        for j, i in enumerate(wave):
            planned = t0 + self.wave_delays[j]
            self.wave_handles.append(call_at_ns(self.loop, planned, self._wave_send, i, planned))

    def _wave_send(self, i, planned):
        if not self.stopping:
            self._send_row(i, planned, wave=True)

    def _wave_done(self):
        self.wave_pending -= 1
        if self.wave_pending == 0:
            self._wave_next(time.monotonic_ns() + self.wave_gap)

    # ---- end of run ----------------------------------------------------------------------
    def _teardown(self):
        from capcommon import exitcodes  # noqa: PLC0415
        from capcommon.lifecycle import read_stop_file  # noqa: PLC0415
        from capcommon.runtree import RunTree  # noqa: PLC0415
        from . import report  # noqa: PLC0415
        now = time.monotonic_ns()
        o = self.origin
        self.lag.stop()
        if self.tick_handle is not None:
            self.tick_handle.cancel()
        if self.cur_seq is not None and self.cur_seq not in self.marks_end:
            self.marks_end[self.cur_seq] = (self.injs.mark(), now - o)
        for c in list(self.open_conns.values()):
            self.rec_conn.append(c.cid, c.local_port, c.peer_port, c.opened - o, -1,
                                 c.n_requests, CL_OPEN, self.table.tier_seq_at(c.opened - o))
        self.dumped = True
        for c in list(self.open_conns.values()):
            if c.close_reason is None:
                c.close_reason = CL_STOP
                c.tr.close()
        self.writer.stop()
        stop_doc = read_stop_file(RunTree(self.cfg["run_dir"]).stop_json)
        report.write_loadgen(self, stop_doc)
        self.injs.close()
        self.chan.close()
        return exitcodes.STOPPED_EARLY if self.stopped_early else exitcodes.OK


def worker_entry(cfg, worker, conn):
    """multiprocessing target (via capload._proc.worker_bootstrap)."""
    w = Worker(cfg, worker, conn)
    return w.run()
