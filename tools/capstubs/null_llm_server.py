# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NL: the product OpenEngineServer (real servicer, real _format_result) driven by NullLlm
(design 2.3 "S-NL", E2 / E3 cards; INTERFACES errata FT1).

One process, one event loop. Modes (--mode):
- minimal: the product server on NullLlm; every step of a request is queued as soon as the
  request arrives (no beat). Stage-0 smoke (design 2.3, D:203): one request answers.
- full: the product server on NullLlm, paced by one metronome for the process: a BeatSchedule
  over the tier table; at every planned step the metronome puts that step's item into the
  unbounded queue of each stream due on it, without waiting for the servicer (the queue is
  upstream of the servicer's yield). Item j of a stream is queued on step
  first_step + msg_step_offsets(N, k, s, "product")[j] and carries msg_tokens(...)[j] tokens;
  NullHandle.__anext__ takes ONE item per call. The servicer yields one token message per item
  and the finished message (with usage) after the last one.
- metronome-only: the same metronome, NullLlm and queues, with no servicer and no gRPC traffic:
  streams are opened at the planned times of --schedule (every row, N = the row's tier
  max_tokens) and drained by one in-process consumer task each. The E3-3a same-process dry run
  (the metronome's own cost at the same C and step period). A gRPC server with no method is
  still bound so the role has a port.

The yielded count (D17) comes from wrapping the servicer's Generate: S-NL builds the product
server with a subclass of its OpenEngineInferenceServicer whose Generate iterates the original
generator and counts (and injects at send.msg) every response before passing it on; no product
file is changed. Start-up checks that the server really built the subclass.

The product is imported only in minimal / full mode (--server-module, default
tensorrt_llm.grpc.openengine.server); any import or construction failure exits 3 with the reason.
What was imported is reported in the ready file and every tier summary.

Entry: python -m capstubs.null_llm_server
"""

import argparse
import asyncio
import importlib
import os
import sys
import time
from contextlib import aclosing

import grpc

from capcommon import expect
from capcommon.aio import call_at_ns, run as aio_run
from capcommon.capbin import RecordArray
from capcommon.cli import add_common_args, common_from_args
from capcommon.clock import now_ns
from capcommon.exitcodes import OK, CapkitError, ConfigError
from capcommon.lifecycle import StopController, write_port_file
from capcommon.schedule import BeatSchedule, count_schedule_rows, load_schedule
from capcommon.tiers import POST_SEQ, TierTable

from capgrpc import inject_slot_decls
from capgrpc.common import (ToolRuntime, bind_server, combine_launch_specs, make_server,
                            write_inprocess_ready)

from .nullllm import CURRENT_RPC, TOKEN_SOURCES, NullArgs, NullLlm
from .prebaked import SHORT_SITE, decimal_request_id, shorted, uniform_msg_shape
from .prebaked_server import _schedule_tiers, check_inject_specs

TOOL = "capstubs.null_llm_server"
MODES = ("minimal", "full", "metronome-only")
DEFAULT_SERVER_MODULE = "tensorrt_llm.grpc.openengine.server"
SERVICER_CLASS = "OpenEngineInferenceServicer"
SERVER_CLASS = "OpenEngineServer"

# send.msg: once per servicer response, in the Generate wrapper just before it is passed on to
# gRPC (busy/sleep); beat.step: once per beat step in the metronome (busy/sleep); short.req: once
# per stream NullLlm opens (short).
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
    "self.rpcs": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.accepted": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.completed": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.cancelled": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.rejected": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "count.items_taken": {"kind": "counter", "site": None, "group": "count",
                          "per": "message"},
    "count.beat_steps": {"kind": "counter", "site": "beat.step", "group": "count",
                         "per": "step"},
    "count.beat_late_steps": {"kind": "counter", "site": "beat.step", "group": "count",
                              "per": "step"},
    "count.beat_enqueued": {"kind": "counter", "site": "beat.step", "group": "count",
                            "per": "message"},
    "count.beat_work_ns": {"kind": "counter", "site": "beat.step", "group": "count",
                           "per": "step"},
    "lag.beat_wake": {"kind": "hist", "site": "beat.step", "group": "lag", "per": "step"},
    "lag.loop": {"kind": "hist", "site": None, "group": "lag", "per": "sample"},
}
SLOTS.update(inject_slot_decls(SITES, {"send.msg": "message", "beat.step": "step",
                                       SHORT_SITE: "request"}))

STREAM_FIELDS = [("req_id", "q"), ("accept_rel_ns", "q"), ("first_step", "q"), ("n_msgs", "i"),
                 ("items", "i"), ("tokens", "i"), ("tier_seq", "h")]
STEP_FIELDS = [("step", "q"), ("planned_rel_ns", "q"), ("actual_rel_ns", "q"), ("items", "i"),
               ("active", "i"), ("work_ns", "q"), ("tier_seq", "h")]
OUTCOMES = ("completed", "cancelled", "truncated")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.null_llm_server", allow_abbrev=False,
                                description=__doc__.split("\n")[0])
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=True)
    g = p.add_argument_group("S-NL")
    g.add_argument("--host", required=True)
    g.add_argument("--port", type=int, required=True, help="0 = any free port")
    g.add_argument("--workers", type=int, required=True, help="must be 1 (single process)")
    g.add_argument("--mode", required=True, choices=MODES)
    g.add_argument("--lag-period-us", type=int, required=True)
    g.add_argument("--model", required=True, help="model name the server advertises")
    g.add_argument("--token-source", required=True, choices=TOKEN_SOURCES,
                   help="fixed: --token-ids cycled over the stream; prompt: the request's "
                        "prompt token ids cycled")
    g.add_argument("--token-ids", help="comma-separated token ids (token source fixed)")
    g.add_argument("--schedule",
                   help="arrival schedule: in full / minimal mode request_id must be a req_id of "
                        "it and each stream is archived under its planned tier (without it: the "
                        "tier at arrival); required in metronome-only mode (the arrivals)")
    g.add_argument("--prefill-delay-us", type=int, default=0,
                   help="beat modes: a stream's first step is the first step at or after "
                        "arrival + this delay")
    g.add_argument("--stream-interval", type=int,
                   help="cross-check only: must equal the tier table's stream_interval")
    g.add_argument("--max-stream-records", type=int,
                   help="streams capacity (default: count_schedule_rows of the tier table)")
    g.add_argument("--lag-records-capacity", type=int, default=0)
    s = p.add_argument_group("product (minimal / full mode)")
    s.add_argument("--server-module", default=DEFAULT_SERVER_MODULE,
                   help="module providing %s and %s" % (SERVER_CLASS, SERVICER_CLASS))
    s.add_argument("--tokenizer", help="tokenizer directory (loaded with the product's "
                                       "ModelLoader.load_hf_tokenizer, as the frontend does)")
    s.add_argument("--trust-remote-code", choices=("true", "false"),
                   help="with --tokenizer: must equal the frontend's")
    s.add_argument("--tokenizer-mode", choices=("auto", "slow"),
                   help="with --tokenizer: must equal the frontend's")
    s.add_argument("--max-seq-len", type=int, default=32768)
    s.add_argument("--max-batch-size", type=int, default=4096)
    s.add_argument("--max-num-tokens", type=int, default=32768)
    return p


def parse_token_ids(text):
    try:
        ids = [int(x) for x in text.split(",")]
    except ValueError:
        raise ConfigError("--token-ids must be comma-separated integers, got %r" % text)
    if not ids or any(i < 0 for i in ids):
        raise ConfigError("--token-ids must list token ids >= 0")
    return ids


def validate_args(args, common):
    if args.workers != 1:
        raise ConfigError("--workers must be 1 (S-NL is one process)")
    if not 0 <= args.port < 65536:
        raise ConfigError("--port out of range")
    if len(common.cpus) < 1:
        raise ConfigError("--cpus lists no cpu")
    if args.max_stream_records is not None and args.max_stream_records < 1:
        raise ConfigError("--max-stream-records must be >= 1")
    if args.lag_records_capacity < 0:
        raise ConfigError("--lag-records-capacity must be >= 0")
    if args.prefill_delay_us < 0:
        raise ConfigError("--prefill-delay-us must be >= 0")
    if (args.token_source == "fixed") != (args.token_ids is not None):
        raise ConfigError("--token-ids is required with --token-source fixed and only then")
    for name in ("max_seq_len", "max_batch_size", "max_num_tokens"):
        if getattr(args, name) < 1:
            raise ConfigError("--%s must be >= 1" % name.replace("_", "-"))
    tok = [args.trust_remote_code, args.tokenizer_mode]
    if args.tokenizer is None and any(v is not None for v in tok):
        raise ConfigError("--trust-remote-code / --tokenizer-mode need --tokenizer")
    if args.tokenizer is not None and any(v is None for v in tok):
        raise ConfigError("--tokenizer needs --trust-remote-code and --tokenizer-mode")
    if args.mode == "metronome-only":
        if args.schedule is None:
            raise ConfigError("--mode metronome-only needs --schedule (the arrivals)")
        if args.token_source != "fixed":
            raise ConfigError("--mode metronome-only has no prompts: use --token-source fixed")
        if args.tokenizer is not None or args.server_module != DEFAULT_SERVER_MODULE:
            raise ConfigError("--mode metronome-only runs no product code: no --tokenizer or "
                              "--server-module")
    if args.mode == "minimal" and args.prefill_delay_us:
        raise ConfigError("--prefill-delay-us has no effect in minimal mode (no beat)")


def message_shape(tt, stream_interval):
    """(k, s) of the streams: the load tiers' common (tokens_per_msg, stream_interval)."""
    try:
        k, s = uniform_msg_shape(tt)
    except ConfigError as e:
        raise ConfigError("S-NL (like S-PB): %s" % e)
    if stream_interval is not None and stream_interval != s:
        raise ConfigError("--stream-interval %d != the tier table's stream_interval %d"
                          % (stream_interval, s))
    return k, s


# ---- product ------------------------------------------------------------------------------------

def load_product(module_name):
    """Import the product server module; return (module, report). Any failure exits 3."""
    import inspect  # noqa: PLC0415
    try:
        mod = importlib.import_module(module_name)
    except (Exception, SystemExit) as e:  # noqa: BLE001 - whatever the import raises, exit 3
        raise ConfigError("S-NL needs the product server module %s (container only): %s: %s"
                          % (module_name, type(e).__name__, e))
    server_cls = getattr(mod, SERVER_CLASS, None)
    servicer_cls = getattr(mod, SERVICER_CLASS, None)
    if not isinstance(server_cls, type) or not isinstance(servicer_cls, type):
        raise ConfigError("%s has no class %s and %s" % (module_name, SERVER_CLASS,
                                                          SERVICER_CLASS))
    if not inspect.isasyncgenfunction(getattr(servicer_cls, "Generate", None)):
        raise ConfigError("%s.%s.Generate is not an async generator function; S-NL cannot wrap "
                          "it" % (module_name, SERVICER_CLASS))
    trt = sys.modules.get("tensorrt_llm")
    try:
        from google.protobuf.internal import api_implementation  # noqa: PLC0415
        pb_impl = api_implementation.Type()
    except ImportError:
        pb_impl = None
    report = {"server_module": module_name, "server_module_file": getattr(mod, "__file__", None),
              "server_class": "%s.%s" % (server_cls.__module__, server_cls.__qualname__),
              "servicer_class": "%s.%s" % (servicer_cls.__module__, servicer_cls.__qualname__),
              "tensorrt_llm": None if trt is None else {
                  "version": getattr(trt, "__version__", None),
                  "file": getattr(trt, "__file__", None)},
              "protobuf_impl": pb_impl}
    return mod, report


def load_tokenizer(path, trust_remote_code, mode):
    """The tokenizer as the frontend loads it (ModelLoader.load_hf_tokenizer). Exit 3 on any
    failure: a tokenizer-less S-NL must be asked for explicitly (no --tokenizer)."""
    if not os.path.isdir(path):
        raise ConfigError("--tokenizer %s is not a directory" % path)
    try:
        from tensorrt_llm.llmapi.llm_utils import ModelLoader  # noqa: PLC0415
        tok = ModelLoader.load_hf_tokenizer(path, trust_remote_code=trust_remote_code == "true",
                                            use_fast=mode != "slow")
    except (Exception, SystemExit) as e:  # noqa: BLE001 - whatever loading raises, exit 3
        raise ConfigError("cannot load the tokenizer from %s with the product's ModelLoader: "
                          "%s: %s" % (path, type(e).__name__, e))
    if tok is None:
        raise ConfigError("the product's ModelLoader found no tokenizer in %s" % path)
    return tok


def counting_servicer_class(base, srv):
    """A subclass of the product servicer whose Generate passes every response of the original
    Generate on unchanged, counting it (backlog.yielded) and hitting send.msg before it."""
    orig = base.Generate

    class SnlCountingServicer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            srv.servicers_built += 1

        async def Generate(self, request, context):  # noqa: N802 - the product's name
            rec = srv.rpc_begin(request)
            if rec.reject is not None:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, rec.reject)
                return
            token = CURRENT_RPC.set(rec)
            inj, v, iy = srv.inj_send, srv.v, srv.i_yielded
            try:
                async with aclosing(orig(self, request, context)) as responses:
                    async for resp in responses:
                        inj.hit()
                        v[iy] += 1
                        rec.msgs += 1
                        yield resp
            finally:
                try:
                    CURRENT_RPC.reset(token)
                except (ValueError, RuntimeError):
                    pass  # closed from another context: the RPC task's context dies with it
                srv.rpc_end(rec)

    SnlCountingServicer.__qualname__ = "SnlCountingServicer(%s)" % base.__qualname__
    return SnlCountingServicer


class _Rpc:
    """One Generate RPC (or one metronome-only stream): what the wrapper and NullLlm record."""
    __slots__ = ("rid", "accept", "seq", "handle", "msgs", "reject", "nondec")

    def __init__(self, rid, accept, seq, nondec=False):
        self.rid, self.accept, self.seq = rid, accept, seq
        self.handle = None
        self.msgs = 0
        self.reject = None
        self.nondec = nondec


class _SP:
    """The sampling params metronome-only mode hands NullLlm (only max_tokens is read)."""
    __slots__ = ("max_tokens", "n")

    def __init__(self, max_tokens):
        self.max_tokens, self.n = max_tokens, 1


# ---- the stub -----------------------------------------------------------------------------------

class NullLlmServer:
    """The source NullLlm draws on (open / took / close), the metronome, the RPC bookkeeping of
    the Generate wrapper and the process lifecycle."""

    def __init__(self, args, rt, ks, sched, product, tokenizer):
        self.args, self.rt = args, rt
        self.mode = args.mode
        self.beat = args.mode != "minimal"
        self.k, self.s = ks
        self.sched_tiers, self.sched_sha256 = sched
        self.product = product  # (module, report) or None
        self.tt = rt.tt
        self.origin = rt.origin_ns
        self.prefill_ns = args.prefill_delay_us * 1000
        nt = len(self.tt)
        self.last_seq = nt - 1
        z = lambda: [0] * nt  # noqa: E731
        self.c = {name: z() for name in (
            "rpcs", "rpcs_without_stream", "rejected", "streams_accepted", "streams_completed",
            "streams_cancelled", "streams_truncated", "first_step_bumped", "msgs_yielded",
            "items_taken", "tokens_produced", "short_requests", "request_id_not_decimal",
            "streams_outside_rpc")}
        self.max_pending = z()
        cap = args.max_stream_records
        if cap is None:
            cap = count_schedule_rows(self.tt)
        self.stream_rec = RecordArray(STREAM_FIELDS, cap)
        self.bs = None
        self.step_rec = None
        if self.beat:
            self.bs = BeatSchedule(self.tt)
            if self.bs.n_steps == 0:
                raise ConfigError("S-NL mode %s paces by the beat, but no tier has beat_ns > 0"
                                  % self.mode)
            self.step_rec = RecordArray(STEP_FIELDS, self.bs.n_steps)
        self.due = {}
        self.next_step = 0
        self.plans = {}
        v = rt.registry.values
        self.v = v
        pos = {d["name"]: d["offset"] for d in rt.registry.slots}
        for name in ("backlog.active_streams", "backlog.pending_msgs", "backlog.yielded",
                     "self.rpcs", "self.accepted", "self.completed", "self.cancelled",
                     "self.rejected", "count.items_taken", "count.beat_steps",
                     "count.beat_late_steps", "count.beat_enqueued", "count.beat_work_ns"):
            setattr(self, "i_" + name.split(".", 1)[1], pos[name])
        self.h_wake = rt.slot_refs["lag.beat_wake"]
        self.inj_send = rt.injectors["send.msg"]
        self.inj_step = rt.injectors["beat.step"]
        self.inj_short = rt.injectors[SHORT_SITE]
        self.servicers_built = 0
        self.llm = None
        self.stop_ev = None
        self.stopping = False
        self._wake = None
        self.consumers = set()

    # -- the source NullLlm draws on --------------------------------------------------------------
    def plan(self, n):
        p = self.plans.get(n)
        if p is None:
            p = (tuple(expect.msg_tokens(n, self.k, self.s, "product")),
                 tuple(expect.msg_step_offsets(n, self.k, self.s, "product")))
            self.plans[n] = p
        return p

    def open(self, h, n):
        d = self.inj_short.short_by()
        n2 = shorted(n, d)
        h.n, h.shorted_by = n2, d
        h.toks, h.offs = self.plan(n2)
        rec = CURRENT_RPC.get()
        if rec is None:  # not from a Generate RPC: the product Control's inference probe
            self.c["streams_outside_rpc"][self._event_seq()] += 1
        v = self.v
        v[self.i_accepted] += 1
        v[self.i_active_streams] += 1
        if not self.beat:  # minimal: every item now
            m = len(h.toks)
            for j in range(m):
                h.push(j)
            v[self.i_pending_msgs] += m
            v[self.i_beat_enqueued] += m
            seq = self._event_seq()
            if v[self.i_pending_msgs] > self.max_pending[seq]:
                self.max_pending[seq] = v[self.i_pending_msgs]
            return
        rel = now_ns() - self.origin + self.prefill_ns
        first = self.bs.step_at_or_after(rel)
        if first is not None and first < self.next_step:  # the metronome already passed it
            first = self.next_step
            if rec is not None:
                self.c["first_step_bumped"][rec.seq] += 1
        if first is None or first + h.offs[0] >= self.bs.n_steps:
            h.truncate()
            return
        h.first = first
        self.due.setdefault(first + h.offs[0], []).append((h, 0))

    def took(self, h):
        self.v[self.i_pending_msgs] -= 1
        self.v[self.i_items_taken] += 1

    def close(self, h, outcome):
        if h.closed:
            return
        h.closed = True
        h.outcome = outcome
        v = self.v
        v[self.i_pending_msgs] -= len(h.q)
        h.q.clear()
        v[self.i_active_streams] -= 1
        if outcome == "completed":
            v[self.i_completed] += 1
        elif outcome == "cancelled":
            v[self.i_cancelled] += 1

    def _event_seq(self):
        seq = self.rt.cur_seq
        return self.last_seq if seq == POST_SEQ else max(seq, 0)

    # -- the metronome ----------------------------------------------------------------------------
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

    async def metronome(self, loop):
        bs, origin, v = self.bs, self.origin, self.v
        n = bs.n_steps
        due, rec = self.due, self.step_rec
        tnow = time.thread_time_ns
        step = 0
        while step < n:
            prel = bs.planned_rel_ns(step)
            if not await self._sleep_until(loop, origin + prel):
                break
            rel = now_ns() - origin
            self.h_wake.add(rel - prel)
            self.inj_step.hit()
            t0 = tnow()
            seq = bs.tier_of_step(step)
            if step + 1 < n and rel >= bs.planned_rel_ns(step + 1):
                v[self.i_beat_late_steps] += 1
            nq = 0
            entries = due.pop(step, None)
            if entries:
                for h, j in entries:
                    if h.closed:
                        continue
                    h.push(j)
                    nq += 1
                    j += 1
                    if j < len(h.toks):
                        nxt = h.first + h.offs[j]
                        if nxt >= n:
                            h.truncate()
                        else:
                            due.setdefault(nxt, []).append((h, j))
                v[self.i_pending_msgs] += nq
                v[self.i_beat_enqueued] += nq
                if v[self.i_pending_msgs] > self.max_pending[seq]:
                    self.max_pending[seq] = v[self.i_pending_msgs]
            self.next_step = step + 1
            v[self.i_beat_steps] += 1
            work = tnow() - t0
            v[self.i_beat_work_ns] += work
            rec.append(step, prel, rel, nq, v[self.i_active_streams], work, seq)
            step += 1
        self.next_step = max(self.next_step, step)

    # -- RPC bookkeeping (the Generate wrapper; metronome-only consumers) ------------------------
    def rpc_begin(self, request):
        accept = now_ns() - self.origin
        ev_seq = self._event_seq()
        self.v[self.i_rpcs] += 1
        rid_s = request.request_id
        rid = decimal_request_id(rid_s.encode("utf-8")) if isinstance(rid_s, str) else -1
        seq = ev_seq
        rec = _Rpc(rid, accept, seq, nondec=rid < 0)
        if self.sched_tiers is not None:
            if not 0 <= rid < len(self.sched_tiers):
                rec.reject = "request_id %r is not a req_id of the schedule" % rid_s[:40]
                self.c["rejected"][ev_seq] += 1
                self.c["rpcs"][ev_seq] += 1
                self.v[self.i_rejected] += 1
                return rec
            rec.seq = self.sched_tiers[rid]
        return rec

    def rpc_end(self, rec):
        h, seq, c = rec.handle, rec.seq, self.c
        c["rpcs"][seq] += 1
        if rec.nondec:
            c["request_id_not_decimal"][seq] += 1
        if h is None:
            c["rpcs_without_stream"][seq] += 1
            return
        if not h.closed:  # the RPC ended before the engine side did (the product aborts too)
            self.close(h, "cancelled")
        c["streams_accepted"][seq] += 1
        c["streams_" + h.outcome][seq] += 1
        c["msgs_yielded"][seq] += rec.msgs
        c["items_taken"][seq] += h.taken
        c["tokens_produced"][seq] += h.produced
        if h.shorted_by:
            c["short_requests"][seq] += 1
        self.stream_rec.append(rec.rid, rec.accept, h.first, rec.msgs, h.taken, h.produced, seq)

    async def _consume(self, rec, n):
        token = CURRENT_RPC.set(rec)
        try:
            h = self.llm.generate_async({"prompt_token_ids": []}, _SP(n), streaming=True)
            async for _ in h:
                pass
        except asyncio.CancelledError:
            if rec.handle is not None and not rec.handle.closed:
                rec.handle.abort()
            raise
        finally:
            CURRENT_RPC.reset(token)
            self.rpc_end(rec)

    async def dispatcher(self, loop):
        """metronome-only: open every schedule row's stream at its planned time."""
        req_ids, planned, tiers, _meta = load_schedule(self.args.schedule, 0, 1,
                                                       tier_table_sha=self.tt.sha256)
        tt = self.tt
        for i in range(len(req_ids)):
            if not await self._sleep_until(loop, self.origin + planned[i]):
                break
            seq = tiers[i]
            self.v[self.i_rpcs] += 1
            rec = _Rpc(req_ids[i], now_ns() - self.origin, seq)
            t = loop.create_task(self._consume(rec, tt[seq].max_tokens))
            self.consumers.add(t)
            t.add_done_callback(self.consumers.discard)

    # -- lifecycle --------------------------------------------------------------------------------
    def on_stop(self, hard):
        self.stopping = True
        w = self._wake
        if w is not None and not w.done():
            w.set_result(None)
        if self.stop_ev is not None:
            self.stop_ev.set()

    def build_llm(self, tokenizer):
        a = self.args
        null_args = NullArgs(a.max_seq_len, a.max_batch_size, a.max_num_tokens)
        ids = parse_token_ids(a.token_ids) if a.token_ids is not None else ()
        self.llm = NullLlm(self, null_args, token_source=a.token_source, token_ids=ids,
                           tokenizer=tokenizer,
                           hf_model_dir=os.path.abspath(a.tokenizer) if a.tokenizer else None)
        return self.llm

    async def start_product(self):
        """Build and start the product server on NullLlm with the counting servicer; return
        (server, port). Any failure exits 3."""
        a = self.args
        mod = self.product[0]
        orig = getattr(mod, SERVICER_CLASS)
        setattr(mod, SERVICER_CLASS, counting_servicer_class(orig, self))
        try:
            server = getattr(mod, SERVER_CLASS)(a.host, a.port, self.llm, a.model)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - whatever it raises, exit 3
            raise ConfigError("the product %s could not be built on NullLlm: %s: %s"
                              % (SERVER_CLASS, type(e).__name__, e))
        finally:
            setattr(mod, SERVICER_CLASS, orig)
        if self.servicers_built != 1:
            raise ConfigError("the product %s built %d instances of the wrapped %s (expected 1): "
                              "S-NL cannot count what the servicer yields"
                              % (SERVER_CLASS, self.servicers_built, SERVICER_CLASS))
        try:
            await server.start()
        except (Exception, SystemExit) as e:  # noqa: BLE001 - whatever it raises, exit 3
            try:
                await server.stop(0)
            finally:
                raise ConfigError("the product %s on NullLlm failed to start: %s: %s"
                                  % (SERVER_CLASS, type(e).__name__, e))
        return server, server.port

    def info(self, port):
        a = self.args
        return {"host": a.host, "port": port, "mode": self.mode,
                "pacing": "beat" if self.beat else "asap", "gate": "product",
                "servicer": None if self.product is None else "product",
                "imported": None if self.product is None else self.product[1],
                "servicer_wrapped": None if self.product is None else self.servicers_built == 1,
                "model": a.model, "token_source": a.token_source,
                "token_ids": list(self.llm.token_ids),
                "tokenizer": None if a.tokenizer is None else {
                    "path": os.path.abspath(a.tokenizer),
                    "trust_remote_code": a.trust_remote_code == "true",
                    "tokenizer_mode": a.tokenizer_mode},
                "null_args": self.llm.args.to_dict(), "llm_id": self.llm.llm_id,
                "tokens_per_msg": self.k, "stream_interval": self.s,
                "prefill_delay_ns": self.prefill_ns,
                "tier_attribution": "schedule" if self.sched_tiers is not None else "event",
                "schedule": os.path.abspath(a.schedule) if a.schedule else None,
                "schedule_sha256": self.sched_sha256,
                "stream_capacity": self.stream_rec.capacity,
                "beat_steps": self.bs.n_steps if self.bs else 0}

    async def run(self, loop):
        rt, args = self.rt, self.args
        self.stop_ev = asyncio.Event()
        stopper = StopController()
        stopper.attach_loop(loop)
        stopper.on_stop(self.on_stop)
        if self.mode == "metronome-only":
            server = make_server(reuseport=False)  # no method: the role only needs a port
            port = bind_server(server, args.host, args.port)
            await server.start()
        else:
            server, port = await self.start_product()
        info = self.info(port)
        rt.start(loop, info)
        tasks = []
        if self.beat:
            tasks.append(loop.create_task(self.metronome(loop)))
        if self.mode == "metronome-only":
            tasks.append(loop.create_task(self.dispatcher(loop)))
        write_port_file(args.port_file, rt.identity, args.host, port)
        write_inprocess_ready(rt, args.ready_file, info)
        if not self.stopping:
            await self.stop_ev.wait()
        for t in tasks + list(self.consumers):
            t.cancel()
        await asyncio.gather(*tasks, *list(self.consumers), return_exceptions=True)
        await server.stop(args.stop_grace_s)
        rt.close_tiers()
        rt.read_stop()
        self.write_outputs(info)
        rt.stop()
        return OK

    def write_outputs(self, info):
        rt = self.rt
        rt.dump_records(self.stream_rec, "streams", "server.streams",
                        {"pacing": info["pacing"], "gate": "product", "mode": self.mode,
                         "tier_seq": info["tier_attribution"],
                         "n_msgs": "responses the servicer yielded (the finished message "
                                   "included); 0 in metronome-only mode",
                         "items": "steps NullLlm handed to the servicer (one per __anext__)",
                         "req_id": "request_id as decimal, -1 when it is not one"})
        if self.step_rec is not None:
            rt.dump_records(self.step_rec, "steps", "beat.steps",
                            {"n_steps": self.bs.n_steps, "stamp": False,
                             "work_ns": "thread CPU of the metronome's step body"})
        rt.dump_lag()
        attribution = ("planned tier of the request's req_id in --schedule"
                       if self.sched_tiers is not None else "tier at arrival (event time)")
        for seq in sorted(rt.end_marks):
            a, b = rt.start_marks[seq], rt.end_marks[seq]
            payload = rt.tier_common(seq)
            work = rt.slot_delta("count.beat_work_ns", a["vals"], b["vals"])
            loop_cpu = payload["cpu"]["loop_thread_ns"]
            counts = {name: vals[seq] for name, vals in self.c.items()}
            counts["attribution"] = attribution + " (rejected: event time)"
            payload.update({
                "mode": self.mode, "pacing": info["pacing"], "gate": "product",
                "servicer": info["servicer"], "imported": info["imported"],
                "servicer_wrapped": info["servicer_wrapped"], "model": info["model"],
                "tokens_per_msg": self.k, "stream_interval": self.s,
                "prefill_delay_ns": self.prefill_ns, "schedule_sha256": self.sched_sha256,
                "tier_attribution": info["tier_attribution"], "counts": counts,
                "beat": {
                    "steps": rt.slot_delta("count.beat_steps", a["vals"], b["vals"]),
                    "late_steps": rt.slot_delta("count.beat_late_steps", a["vals"], b["vals"]),
                    "enqueued": rt.slot_delta("count.beat_enqueued", a["vals"], b["vals"]),
                    "work_ns": work,
                    "work_share_of_loop_thread": (work / loop_cpu) if loop_cpu > 0 else None,
                    "max_pending": self.max_pending[seq],
                    "wake_lag_ns": rt.hist_summary("lag.beat_wake", a["vals"], b["vals"]),
                    "attribution": "event time (step planned time)"},
                "yielded_event_time": rt.slot_delta("backlog.yielded", a["vals"], b["vals"]),
                "items_taken_event_time": rt.slot_delta("count.items_taken", a["vals"],
                                                        b["vals"]),
                "overflow": {"streams": self.stream_rec.overflow,
                             "steps": self.step_rec.overflow if self.step_rec else 0,
                             "lag": rt.lag_records.overflow if rt.lag_records else 0}})
            rt.write_summary(seq, payload)


# ---- entry point --------------------------------------------------------------------------------

def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        validate_args(args, common)
        tt = TierTable.load(common.tier_table_path)
        specs = combine_launch_specs(common.inject_specs, tt, common.identity.role) \
            + tt.all_inject_for(common.identity.role)
        check_inject_specs(specs, SITE_MODES, "S-NL")
        ks = message_shape(tt, args.stream_interval)
        sched = _schedule_tiers(args.schedule, tt) if args.schedule else (None, None)
        product = tokenizer = None
        if args.mode != "metronome-only":
            product = load_product(args.server_module)
            if args.tokenizer is not None:
                tokenizer = load_tokenizer(args.tokenizer, args.trust_remote_code,
                                           args.tokenizer_mode)
        rt = ToolRuntime(common, TOOL, SLOTS, SITES, 0, 1, [common.cpus[0]],
                         args.lag_period_us, args.lag_records_capacity)
        srv = NullLlmServer(args, rt, ks, sched, product, tokenizer)
        srv.build_llm(tokenizer)
        return aio_run(srv.run, common.loop)
    except CapkitError as e:
        print("%s: %s" % (TOOL, e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
