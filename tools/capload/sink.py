# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One nullsse worker process: an asyncio.Protocol HTTP/1.1 server on a SO_REUSEPORT socket that
answers every completion request with pre-rendered SSE frames (INTERFACES.md 4.4, 4.5, 11).

Per-request parameters come only from the request: max_tokens and stream_options.include_usage
from the JSON body, k / s / planned tier / request id from the x-capload-* headers. The number
of content frames is expect.msgs_per_request(N, k, s, "product"); with --pacing beat, frame j of
a stream is written at the planned instant of beat step first_step + msg_step_offsets(...)[j]
(one process-wide BeatSchedule; one pending queue per stream; writes never wait for the reader).
"""

import asyncio
import heapq
import re
import socket
import time

from capcommon import expect, wire
from capcommon.aio import call_at_ns
from capcommon.capbin import RecordArray
from capcommon.exitcodes import ConfigError

STREAM_FIELDS = [("req_id", "q"), ("accept_rel_ns", "q"), ("first_step", "q"), ("n_msgs", "i"),
                 ("tokens", "i"), ("tier_seq", "h")]
LAG_FIELDS = [("planned_rel_ns", "q"), ("late_ns", "q"), ("tier_seq", "h")]
ENDPOINTS = (b"/v1/completions", b"/v1/chat/completions")
TERMINAL = b"0\r\n\r\n"
# JSON object keys cannot occur unescaped inside a JSON string value, so a search for the
# quoted key is safe against prompt text; the first occurrence is the top-level key capload puts
# before the prompt.
_RE_MAX = re.compile(rb'"(?:max_tokens|max_completion_tokens)"\s*:\s*(\d+)')
_RE_OPTS = re.compile(rb'"stream_options"\s*:\s*\{([^{}]*)\}')
_RE_USAGE = re.compile(rb'"include_usage"\s*:\s*(true|false)')
MAX_HEAD = 65536


def set_nodelay(transport):
    """TCP_NODELAY on a transport's socket (both loop implementations)."""
    sock = transport.get_extra_info("socket")
    if sock is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass


def chunk(data):
    """One HTTP/1.1 chunk."""
    return b"%x\r\n" % len(data) + data + b"\r\n"


def min_chunk_bytes():
    """Smallest --chunk-bytes that fits every request id of a launch (int64 ids)."""
    return len(wire.content_frame(-(2 ** 63), ""))


class Stream:
    __slots__ = ("rid", "tier", "conn", "frame", "tail", "m", "j", "offsets", "first_step",
                 "accept_rel", "tokens", "written", "dead", "usage_frames", "beyond")

    def __init__(self, rid, tier, conn, frame, tail, m, tokens, accept_rel, usage_frames):
        self.rid = rid
        self.tier = tier
        self.conn = conn
        self.frame = frame
        self.tail = tail
        self.m = m
        self.j = 0
        self.offsets = None
        self.first_step = -1
        self.accept_rel = accept_rel
        self.tokens = tokens
        self.written = 0
        self.dead = False
        self.usage_frames = usage_frames
        self.beyond = False


class SinkConn(asyncio.Protocol):
    """One accepted connection."""

    def __init__(self, sink):
        self.s = sink
        self.tr = None
        self.buf = b""
        self.stream = None
        self.idle_handle = None
        self.lost = False
        self.closing = False

    def connection_made(self, transport):
        self.tr = transport
        set_nodelay(transport)
        self.s.on_accept(self)

    def data_received(self, data):
        self.buf += data
        if self.stream is None:
            self.s.on_data(self)

    def eof_received(self):
        return False

    def connection_lost(self, exc):
        self.lost = True
        self.s.on_lost(self)

    def pause_writing(self):
        pass

    def resume_writing(self):
        pass


class Sink:
    """One nullsse worker."""

    COUNT_KEYS = ("requests", "frames_sent", "bytes_sent", "accepts", "idle_closed",
                  "streams_completed", "streams_aborted", "streams_unfinished_at_stop",
                  "bad_requests", "foreign_requests", "beyond_beat_schedule", "short_requests")
    COUNT_RULES = {
        "requests": "accepted completion requests, by the request's planned tier (x-capload-tier)",
        "frames_sent": "content frames written (drop/miscount injection applied) + 1 usage frame "
                       "per request that asked include_usage; [DONE] not counted; by planned tier",
        "bytes_sent": "bytes handed to the transport for completion responses (head + chunks), "
                      "by planned tier",
        "accepts": "TCP connections accepted, by event-time tier",
        "idle_closed": "keep-alive connections closed by --idle-close-s, by event-time tier",
        "bad_requests": "non-completion / malformed requests answered with 4xx, by event tier",
        "foreign_requests": "requests without x-capload-* headers (counted under the arrival "
                            "tier; k = s = 1)",
    }

    def __init__(self, cfg, worker, chan_conn):
        from capcommon.identity import Identity  # noqa: PLC0415
        from capcommon.inject import parse_inject_spec  # noqa: PLC0415
        from capcommon.lifecycle import pin_self  # noqa: PLC0415
        from capcommon.schedule import BeatSchedule  # noqa: PLC0415
        from capcommon.tiers import TierTable  # noqa: PLC0415
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
        self.beat = BeatSchedule(self.table)
        self.pacing_beat = cfg["pacing"] == "beat"
        self.chunk_bytes = cfg["chunk_bytes"]
        self.idle_s = cfg["idle_close_s"]
        self.grace_s = cfg["stop_grace_s"]
        self.cli_specs = [parse_inject_spec(t) for t in cfg["inject"]]
        self.counts = {}
        self.conns = set()
        self.streams = set()
        self.heap = []
        self.heap_n = 0
        self.beat_handle = None
        self.beat_due = None
        self.stopping = False
        self.stop_reason = None
        self.finished = None
        self.server = None
        self.marks_start = {}
        self.marks_end = {}
        self.cur_seq = None
        self.grace_handle = None
        self.in_beat = False
        self.dumped = False
        if self.chunk_bytes < min_chunk_bytes():
            raise ConfigError("--chunk-bytes %d < %d (smallest frame that fits any request id)"
                              % (self.chunk_bytes, min_chunk_bytes()))

    def _c(self, seq):
        c = self.counts.get(seq)
        if c is None:
            c = self.counts[seq] = dict.fromkeys(self.COUNT_KEYS, 0)
        return c

    def run(self):
        from capcommon.aio import run as aio_run  # noqa: PLC0415
        return aio_run(self._main, self.cfg["loop"])

    async def _main(self, loop):
        from capcommon.aio import LoopLagProbe  # noqa: PLC0415
        from capcommon.snapshot import SlotRegistry, SnapshotWriter  # noqa: PLC0415
        from capcommon.versions import version_record  # noqa: PLC0415
        from . import SINK_SITES, SINK_SLOTS  # noqa: PLC0415
        from ._proc import WorkerChannel  # noqa: PLC0415
        from ._util import build_injectors  # noqa: PLC0415
        self.loop = loop
        self.finished = loop.create_future()
        reg = SlotRegistry()
        reg.register_decls(SINK_SLOTS)
        self.vals = reg.values
        self.i_active = reg.gauge("backlog.active_streams")
        self.i_req = reg.counter("self.requests")
        self.i_frames = reg.counter("self.frames_sent")
        self.i_bytes = reg.counter("self.bytes_sent")
        self.i_acc = reg.counter("self.accepts")
        self.i_idle = reg.counter("self.idle_closed")
        self.i_open = reg.gauge("self.conns_open")
        self.i_pend = reg.gauge("self.pending_msgs")
        self.injs, self.aux_threads = build_injectors(SINK_SITES, reg, self.cli_specs,
                                                      self.table, self.role)
        self.inj_chunk = self.injs["send.chunk"]
        self.inj_short = self.injs["short.req"]
        self.versions = version_record(loop=loop, include_grpc=False)
        self.writer = SnapshotWriter(reg, self.cfg["shm_dir"], self.cfg["snapshot_interval_ns"],
                                     meta_extra={"identity": self.ident.to_dict(),
                                                 "versions": self.versions,
                                                 "role_hint": self.role})
        cap = self.cfg["max_records"]
        self.rec_streams = RecordArray(STREAM_FIELDS, cap)
        self.rec_lag = RecordArray(LAG_FIELDS, self.cfg["lag_records"])
        # proto must be IPPROTO_TCP: asyncio only sets TCP_NODELAY on accepted sockets whose
        # proto is TCP (proto 0 silently keeps Nagle on -> ~40 ms delayed-ACK stalls)
        sock = socket.socket(self.cfg["family"], socket.SOCK_STREAM, socket.IPPROTO_TCP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        try:
            sock.bind((self.cfg["host"], self.cfg["port"]))
        except OSError as e:
            from capcommon.exitcodes import CapkitError  # noqa: PLC0415
            raise CapkitError("worker %d cannot bind %s:%d: %s"
                              % (self.k, self.cfg["host"], self.cfg["port"], e))
        sock.listen(socket.SOMAXCONN)
        sock.setblocking(False)
        self.server = await loop.create_server(lambda: SinkConn(self), sock=sock)
        self.writer.start()
        self.lag = LoopLagProbe(loop, reg, "loop", self.cfg["lag_period_us"] * 1000,
                                records=self.rec_lag, origin_ns=self.origin,
                                tier_seq_at=self.table.tier_seq_at).start()
        self.chan = WorkerChannel(self.chan_conn, loop, self.on_stop)
        self._arm_boundaries()
        self.chan.send(("ready", {
            "pid": __import__("os").getpid(), "worker": self.k, "versions": self.versions,
            "affinity": self.affinity, "aux_threads": self.aux_threads,
            "listen": list(sock.getsockname()[:2]),
            "ready_rel_ns": time.monotonic_ns() - self.origin}))
        await self.finished
        return self._teardown()

    # ---- tier boundaries (only injection specs switch; per-request parameters ride with the
    # request, never looked up by arrival time) -------------------------------------------
    def _arm_boundaries(self):
        for t in self.table.tiers:
            call_at_ns(self.loop, self.origin + t.start_ns, self._boundary, t.seq)
        call_at_ns(self.loop, self.origin + self.table.end_ns, self._boundary, None)

    def _boundary(self, seq):
        from ._util import tier_specs_for  # noqa: PLC0415
        rel = time.monotonic_ns() - self.origin
        mark = self.injs.mark()
        if self.cur_seq is not None:
            self.marks_end[self.cur_seq] = (mark, rel)
        self.cur_seq = seq
        if seq is None:
            self.lag.records = None  # after the last tier: histogram only, no per-sample rows
            return
        self.injs.apply_tier_specs(tier_specs_for(self.table, self.role, seq))
        self.marks_start[seq] = (self.injs.mark(), rel)

    # ---- connections and requests --------------------------------------------------------
    def on_accept(self, conn):
        now = time.monotonic_ns()
        self.conns.add(conn)
        v = self.vals
        v[self.i_open] = len(self.conns)
        v[self.i_acc] += 1
        self._c(self.table.tier_seq_at(now - self.origin))["accepts"] += 1
        self._arm_idle(conn)

    def on_data(self, conn):
        while conn.stream is None and conn.buf and not conn.lost and not conn.closing:
            buf = conn.buf
            he = buf.find(b"\r\n\r\n")
            if he < 0:
                if len(buf) > MAX_HEAD:
                    self._simple(conn, 431, b"Request Header Fields Too Large", close=True)
                return
            lines = buf[:he].split(b"\r\n")
            rl = lines[0].split(b" ")
            hdrs = {}
            for ln in lines[1:]:
                key, _sep, val = ln.partition(b":")
                hdrs[key.strip().lower()] = val.strip()
            try:
                clen = int(hdrs.get(b"content-length", b"0"))
            except ValueError:
                clen = -1
            if len(rl) != 3 or clen < 0 or b"chunked" in hdrs.get(b"transfer-encoding", b""):
                self._simple(conn, 400, b"Bad Request", close=True)
                return
            total = he + 4 + clen
            if len(buf) < total:
                return
            body = buf[he + 4:total]
            conn.buf = buf[total:]
            if conn.idle_handle is not None:
                conn.idle_handle.cancel()
                conn.idle_handle = None
            self._request(conn, rl[0], rl[1], hdrs, body)

    def _simple(self, conn, code, reason, body=b"", close=False, count_bad=True):
        if count_bad:
            self._c(self.table.tier_seq_at(time.monotonic_ns() - self.origin))["bad_requests"] += 1
        conn.tr.write(b"HTTP/1.1 %d %s\r\ncontent-type: text/plain\r\ncontent-length: %d\r\n%s\r\n"
                      % (code, reason, len(body), b"connection: close\r\n" if close else b"")
                      + body)
        if close:
            conn.closing = True
            conn.tr.close()
        else:
            self._arm_idle(conn)

    def _request(self, conn, method, path, hdrs, body):
        now = time.monotonic_ns()
        rel = now - self.origin
        path = path.split(b"?", 1)[0]
        if method == b"GET" and path == b"/health":
            self._simple(conn, 200, b"OK", b"ok", count_bad=False)
            return
        if method != b"POST" or path not in ENDPOINTS:
            self._simple(conn, 404, b"Not Found")
            return
        m = _RE_MAX.search(body)
        if m is None or int(m.group(1)) < 1:
            self._simple(conn, 400, b"Bad Request", b"max_tokens >= 1 required")
            return
        n_req = int(m.group(1))
        mo = _RE_OPTS.search(body)
        include_usage = False
        if mo is not None:
            u = _RE_USAGE.search(mo.group(1))
            include_usage = u is None or u.group(1) == b"true"  # OpenAI default: true
        rid_h = hdrs.get(b"x-capload-id")
        foreign = rid_h is None
        try:
            if foreign:
                rid, k, s, tier = -1, 1, 1, self.table.tier_seq_at(rel)
            else:
                rid = int(rid_h)
                k = int(hdrs[b"x-capload-k"])
                s = int(hdrs[b"x-capload-s"])
                tier = int(hdrs[b"x-capload-tier"])
                if not 0 <= tier < len(self.table.tiers):
                    raise ValueError("tier %d outside the table" % tier)
            short = self.inj_short.short_by()
            n_tok = n_req - short if short < n_req else 1
            tokens = expect.msg_tokens(n_tok, k, s, "product")
        except (KeyError, ValueError, ConfigError) as e:
            self._simple(conn, 400, b"Bad Request", str(e).encode("utf-8", "replace"))
            return
        c = self._c(tier)
        c["requests"] += 1
        if foreign:
            c["foreign_requests"] += 1
        if short:
            c["short_requests"] += 1
        self.vals[self.i_req] += 1
        frame = chunk(wire.padded_content_frame(max(rid, 0), self.chunk_bytes))
        tail = ((chunk(wire.usage_frame(max(rid, 0), 0, n_tok)) if include_usage else b"")
                + chunk(wire.DONE_FRAME) + TERMINAL)
        st = Stream(rid, tier, conn, frame, tail, len(tokens), n_tok, rel,
                    1 if include_usage else 0)
        conn.stream = st
        self.streams.add(st)
        v = self.vals
        v[self.i_active] += 1
        v[self.i_pend] += st.m
        head = (b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache"
                b"\r\nconnection: keep-alive\r\ntransfer-encoding: chunked\r\n"
                + (b"" if foreign else b"x-capload-id: %d\r\n" % rid) + b"\r\n")
        self._write(st, head)
        if self.pacing_beat:
            fs = self.beat.step_at_or_after(rel)
            if fs is not None:
                st.first_step = fs
                st.offsets = expect.msg_step_offsets(n_tok, k, s, "product")
                self._push(st, fs + st.offsets[0])
                return
            st.beyond = True
            c["beyond_beat_schedule"] += 1
        while st.j < st.m and not st.dead:
            self._emit(st)

    def _write(self, st, data):
        st.conn.tr.write(data)
        n = len(data)
        self.vals[self.i_bytes] += n
        self._c(st.tier)["bytes_sent"] += n

    def _emit(self, st):
        inj = self.inj_chunk
        cnt = 1
        drop = False
        if inj.active:
            mode = inj.spec.mode
            if mode == "drop":
                drop = inj.should_drop()
            elif mode == "miscount":
                cnt = inj.count_delta()
            else:
                inj.hit()
        else:
            inj.hit()
        st.j += 1
        v = self.vals
        v[self.i_pend] -= 1
        c = self._c(st.tier)
        data = b""
        if not drop:
            data = st.frame
            st.written += 1
            c["frames_sent"] += cnt
            v[self.i_frames] += cnt
        done = st.j >= st.m
        if done:
            data += st.tail
            c["frames_sent"] += st.usage_frames
            v[self.i_frames] += st.usage_frames
        if data:
            self._write(st, data)
        if done:
            self._end_stream(st, "streams_completed")
        elif self.pacing_beat and not st.beyond:
            self._push(st, st.first_step + st.offsets[st.j])

    # ---- beat pacing: one process-wide absolute beat; a heap of (next step, stream) ------
    def _push(self, st, step):
        self.heap_n += 1
        heapq.heappush(self.heap, (step, self.heap_n, st))
        if not self.in_beat and (self.beat_due is None or step < self.beat_due):
            self._arm_beat()

    def _arm_beat(self):
        if self.beat_handle is not None:
            self.beat_handle.cancel()
            self.beat_handle = None
        self.beat_due = None
        if not self.heap:
            return
        step = self.heap[0][0]
        try:
            at = self.origin + self.beat.planned_rel_ns(step)
        except IndexError:  # past the last step of the table: due now
            at = time.monotonic_ns()
        self.beat_due = step
        self.beat_handle = call_at_ns(self.loop, at, self._on_beat)

    def _on_beat(self):
        self.beat_handle = None
        self.beat_due = None
        rel = time.monotonic_ns() - self.origin
        heap = self.heap
        self.in_beat = True
        try:
            while heap:
                step = heap[0][0]
                try:
                    p = self.beat.planned_rel_ns(step)
                except IndexError:
                    p = None
                if p is not None and p > rel:
                    break
                st = heapq.heappop(heap)[2]
                if st.dead:
                    continue
                if p is None and not st.beyond:
                    st.beyond = True
                    self._c(st.tier)["beyond_beat_schedule"] += 1
                self._emit(st)
                if st.beyond and not st.dead:
                    while st.j < st.m and not st.dead:
                        self._emit(st)
        finally:
            self.in_beat = False
        self._arm_beat()

    def _end_stream(self, st, how, teardown=False):
        if st not in self.streams:
            return
        self.streams.discard(st)
        st.dead = True
        v = self.vals
        v[self.i_active] -= 1
        v[self.i_pend] -= st.m - st.j
        self._c(st.tier)[how] += 1
        self.rec_streams.append(st.rid, st.accept_rel, st.first_step, st.written, st.tokens,
                                st.tier)
        conn = st.conn
        if conn.stream is st:
            conn.stream = None
        if teardown:
            return
        if self.stopping and not self.streams:
            self._finish_run()
            return
        if not conn.lost and not conn.closing:
            if self.stopping:
                conn.closing = True
                conn.tr.close()
            elif conn.buf:
                self.on_data(conn)
            else:
                self._arm_idle(conn)

    # ---- idle close, loss, stop ----------------------------------------------------------
    def _arm_idle(self, conn):
        if self.idle_s > 0 and not conn.lost and conn.idle_handle is None:
            conn.idle_handle = self.loop.call_later(self.idle_s, self._idle_close, conn)

    def _idle_close(self, conn):
        conn.idle_handle = None
        if conn.lost or conn.closing or conn.stream is not None:
            return
        now = time.monotonic_ns()
        self._c(self.table.tier_seq_at(now - self.origin))["idle_closed"] += 1
        self.vals[self.i_idle] += 1
        conn.closing = True
        conn.tr.close()

    def on_lost(self, conn):
        self.conns.discard(conn)
        self.vals[self.i_open] = len(self.conns)
        if conn.idle_handle is not None:
            conn.idle_handle.cancel()
            conn.idle_handle = None
        st = conn.stream
        if st is not None and not self.dumped:
            self._end_stream(st, "streams_aborted")

    def on_stop(self, hard, reason):
        if self.finished.done():
            return
        if self.stop_reason is None:
            self.stop_reason = reason
        self.stopping = True
        if self.server is not None:
            self.server.close()
        for conn in list(self.conns):
            if conn.stream is None and not conn.closing:
                conn.closing = True
                conn.tr.close()
        if hard or not self.streams:
            self._finish_run()
        elif self.grace_handle is None:
            self.grace_handle = self.loop.call_later(self.grace_s, self._finish_run)

    def _finish_run(self):
        if not self.finished.done():
            self.finished.set_result(None)

    def _teardown(self):
        from capcommon import exitcodes  # noqa: PLC0415
        from capcommon.lifecycle import read_stop_file  # noqa: PLC0415
        from capcommon.runtree import RunTree  # noqa: PLC0415
        from . import report  # noqa: PLC0415
        now = time.monotonic_ns()
        self.lag.stop()
        if self.beat_handle is not None:
            self.beat_handle.cancel()
        if self.grace_handle is not None:
            self.grace_handle.cancel()
        if self.cur_seq is not None and self.cur_seq not in self.marks_end:
            self.marks_end[self.cur_seq] = (self.injs.mark(), now - self.origin)
        for st in list(self.streams):
            self._end_stream(st, "streams_unfinished_at_stop", teardown=True)
        self.dumped = True
        for conn in list(self.conns):
            if not conn.closing:
                conn.closing = True
                conn.tr.abort()
        self.writer.stop()
        stop_doc = read_stop_file(RunTree(self.cfg["run_dir"]).stop_json)
        report.write_sink(self, stop_doc)
        self.injs.close()
        self.chan.close()
        return exitcodes.OK


def sink_entry(cfg, worker, conn):
    """multiprocessing target (via capload._proc.worker_bootstrap)."""
    return Sink(cfg, worker, conn).run()
