# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX: NullExecutor, the empty engine that stands in for PyExecutor inside the rank 0 worker
(design 2.3 "S-NX", 7, 12.5(a), 12.7; E2b, E7, E8, E9 cards).

What it keeps from the real engine (everything else -- scheduler, KV, sampler, model -- is gone):
- enqueue_request only queues the executor request and returns a request id (PyExecutor ->
  ExecutorRequestQueue.enqueue_request); the step thread takes everything queued at the start of
  each round and converts it there (executor_request_to_llm_request), so the per-request
  conversion cost lands on the step thread, as in the real engine (12.5(a) item 1).
- One step thread advances on an absolute timetable (grid mode: the tier table's beats from the
  launch origin, or a fixed beat); idle (nothing active, nothing queued) it blocks until a request
  arrives and restarts on the next grid point (or at once, idle_restart "immediate").
- Every round each active request advances k tokens (the first decoding iteration may carry a
  different count, later ones a seeded k mixture: the MTP 1 / {3, 4} shape of 12.7) and is
  released with the product gate on its OWN decoding iteration: iteration 1, finished, or
  iteration % stream_interval == 0 (py_executor.py _handle_responses). Released requests get a real
  create_response(); responses are queued under a condition variable as {req_id: [resp...]} with
  notify_all, and await_responses takes all of them at once (_enqueue_responses /
  _await_any_response).
- Optional 12.5(a) behaviours (all off by default): first response delayed by N rounds (R1), a
  per-round prefill token budget with FCFS chunking (R2), round durations replayed from a
  prefill-token -> duration table with a spill fraction into the next round (items 5, 6), and a
  cap on active requests.
- Optional pure-Python busy loop per step (E7 GIL duty-cycle scan).

Self-report (files under out_dir, one set per process: snx.<pid>.*): meta.json at start (config,
thread ids), fixed-size binary records appended in chunks -- steps (wake lag = start - planned,
wall and thread CPU of the step, takes, releases), takes (each await_responses call: how many
responses, and the take lag = now - planned time of the oldest pending response), reqs (one per
finished request; *.rec) -- and summary.json at shutdown.
`python -m capstubs.null_executor summarize` turns them into per-tier summaries.

The product is reached only through an adapter (capstubs.snx_product.ProductAdapter); this module
imports nothing from tensorrt_llm, so its logic is testable on the VM with a fake adapter.
"""

import argparse
import bisect
import collections
import dataclasses
import json
import mmap
import os
import random
import struct
import sys
import threading
import time

from capcommon.exitcodes import ConfigError

from .affinity import thread_names

SCHEMA_CONFIG = "capkit.snx_config/1"
SCHEMA_META = "capkit.snx_meta/1"
SCHEMA_SUMMARY = "capkit.snx_summary/1"
SCHEMA_TIERS = "capkit.snx_tiers/1"
ENV_VAR = "TRTLLM_CAPKIT_SNX"  # TRTLLM* is forwarded by MpiPoolSession to the workers
PRE_SEQ = -1
POST_SEQ = -2
NS = 1_000_000_000

STEP_REC = struct.Struct("<qqqqqqqhiiiiiiii")
STEP_FIELDS = ("step", "planned_ns", "start_ns", "wake_lag_ns", "wall_ns", "cpu_ns", "take_ns",
               "tier_seq", "active", "taken", "released", "finished", "tokens", "prefill_tokens",
               "queued_after", "busy_iters")
TAKE_REC = struct.Struct("<qqqii")
TAKE_FIELDS = ("call_ns", "return_ns", "oldest_planned_ns", "responses", "keys")
REQ_REC = struct.Struct("<qqqqqqqqiiib")
REQ_FIELDS = ("req_id", "client_id", "enqueue_ns", "take_step", "take_planned_ns",
              "first_emit_step", "finish_step", "finish_planned_ns", "tokens", "responses",
              "prompt_len", "cancelled")
# Record files end in .rec, not .bin: capjudge reads every raw/**/*.bin as a capkit snapshot.
REC_EXT = ".rec"
RECORDS = {"steps": (STEP_REC, STEP_FIELDS), "takes": (TAKE_REC, TAKE_FIELDS),
           "reqs": (REQ_REC, REQ_FIELDS)}
# snx.<pid>.live: one record rewritten in place (mmap) after every enqueue, step and take, for a
# driver in another process (capstubs.nx_role's drain gate). `gen` is a seqlock: odd while a
# write is in progress; a reader retries until it reads the same even gen before and after.
LIVE_REC = struct.Struct("<qqqqqqqq")
LIVE_FIELDS = ("gen", "t_ns", "enqueued", "taken", "finished", "active", "queued",
               "pending_keys")


@dataclasses.dataclass
class SnxConfig:
    """S-NX configuration (JSON object, schema capkit.snx_config/1). Times in ns."""
    out_dir: str = ""
    beat_ns: int = 10_000_000
    tier_table: str = ""
    origin_ns: int = 0
    outside_beat_ns: int = 0
    tokens_per_step: int = 1
    first_step_tokens: int = 0
    k_mix: list = dataclasses.field(default_factory=list)
    k_mix_seed: int = 0
    stream_interval: int = 0
    token_source: str = "fixed"
    token_ids: list = dataclasses.field(default_factory=lambda: [1000, 2000, 3000, 4000])
    busy_ns_per_step: int = 0
    idle_restart: str = "grid"
    first_response_delay_steps: int = 0
    prefill_token_budget: int = 0
    step_time_replay: str = ""
    prefill_spill_frac: float = 0.0
    max_active: int = 0
    max_seq_len: int = 131072
    worker_cpus: list = dataclasses.field(default_factory=list)
    flush_bytes: int = 1 << 16
    flush_interval_ms: int = 500
    worker_inject: str = ""  # capstubs.snx_inject worker-site specs ("Wr:busy:NS"), rank 0 only
    note: str = ""

    def validate(self):
        def _int(name, lo):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v < lo:
                raise ConfigError("snx config: %s must be an int >= %d, got %r" % (name, lo, v))
        for name, lo in (("beat_ns", 1), ("origin_ns", 0), ("outside_beat_ns", 0),
                         ("tokens_per_step", 1), ("first_step_tokens", 0), ("k_mix_seed", 0),
                         ("stream_interval", 0), ("busy_ns_per_step", 0),
                         ("first_response_delay_steps", 0), ("prefill_token_budget", 0),
                         ("max_active", 0), ("max_seq_len", 1), ("flush_bytes", 1),
                         ("flush_interval_ms", 1)):
            _int(name, lo)
        if self.token_source not in ("fixed", "prompt"):
            raise ConfigError("snx config: token_source must be fixed or prompt")
        if not self.token_ids or any(not isinstance(t, int) or isinstance(t, bool) or t < 0
                                     for t in self.token_ids):
            raise ConfigError("snx config: token_ids must be a non-empty list of ints >= 0")
        if any(not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in self.worker_cpus):
            raise ConfigError("snx config: worker_cpus must be a list of cpu numbers >= 0")
        if not isinstance(self.worker_inject, str):
            raise ConfigError("snx config: worker_inject must be a string")
        if self.idle_restart not in ("grid", "immediate"):
            raise ConfigError("snx config: idle_restart must be grid or immediate")
        for kw in self.k_mix:
            if (not isinstance(kw, list) or len(kw) != 2 or not isinstance(kw[0], int)
                    or kw[0] < 1 or not isinstance(kw[1], (int, float)) or kw[1] <= 0):
                raise ConfigError("snx config: k_mix entries are [k >= 1, weight > 0]")
        if not 0.0 <= float(self.prefill_spill_frac) <= 1.0:
            raise ConfigError("snx config: prefill_spill_frac must be in [0, 1]")
        if self.tier_table and not self.origin_ns:
            raise ConfigError("snx config: a tier table needs origin_ns")
        return self

    @classmethod
    def from_doc(cls, doc):
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_CONFIG:
            raise ConfigError("snx config: schema must be %s" % SCHEMA_CONFIG)
        names = {f.name for f in dataclasses.fields(cls)}
        extra = sorted(set(doc) - names - {"schema"})
        if extra:
            raise ConfigError("snx config: unknown keys %s" % extra)
        return cls(**{k: v for k, v in doc.items() if k != "schema"}).validate()

    def to_doc(self):
        d = dataclasses.asdict(self)
        d["schema"] = SCHEMA_CONFIG
        return d

    @classmethod
    def from_env(cls, environ=None):
        """The config serve_nx / nx_driver put into TRTLLM_CAPKIT_SNX (inline JSON)."""
        text = (environ if environ is not None else os.environ).get(ENV_VAR)
        if not text:
            raise ConfigError("%s is not set: the NullWorker has no S-NX config" % ENV_VAR)
        try:
            doc = json.loads(text)
        except ValueError as e:
            raise ConfigError("%s is not JSON: %s" % (ENV_VAR, e)) from e
        return cls.from_doc(doc)


class StepClock:
    """Planned step times (absolute monotonic ns) on a grid.

    Without a tier table: anchor + n * beat_ns. With one: the tier table's BeatSchedule from the
    origin (step j of a tier at start + j * beat_ns); before the origin and after the last tier
    the grid continues with outside_beat_ns (default: the first / last tier's beat) anchored at the
    origin / the table end. Every tier must have a beat (beat_ns > 0)."""

    def __init__(self, beat_ns, anchor_ns, tier_table=None, origin_ns=0, outside_beat_ns=0):
        self.tt = tier_table
        self.origin = origin_ns
        if tier_table is None:
            self.beat = beat_ns
            self.anchor = anchor_ns
            return
        from capcommon.schedule import BeatSchedule  # noqa: PLC0415
        tiers = list(tier_table.tiers)
        if not tiers or any(t.beat_ns <= 0 for t in tiers):
            raise ConfigError("snx: every tier needs beat_ns > 0 (the step thread's period)")
        self.bs = BeatSchedule(tier_table)
        self.pre_beat = outside_beat_ns or tiers[0].beat_ns
        self.post_beat = outside_beat_ns or tiers[-1].beat_ns
        self.end_rel = tier_table.end_ns
        self._beat_of = {t.seq: t.beat_ns for t in tiers}

    def at_or_after(self, t_ns):
        """(planned_abs_ns, tier_seq, beat_ns) of the first grid point at or after t_ns."""
        if self.tt is None:
            n = -(-(t_ns - self.anchor) // self.beat)
            return self.anchor + n * self.beat, PRE_SEQ, self.beat
        rel = t_ns - self.origin
        if rel < 0:
            cand = -((-rel) // self.pre_beat) * self.pre_beat
            if cand < 0:
                return self.origin + cand, PRE_SEQ, self.pre_beat
            rel = 0
        if rel < self.end_rel:
            s = self.bs.step_at_or_after(rel)
            if s is not None:
                seq = self.bs.tier_of_step(s)
                return self.origin + self.bs.planned_rel_ns(s), seq, self._beat_of[seq]
        n = -(-(rel - self.end_rel) // self.post_beat) if rel > self.end_rel else 0
        return self.origin + self.end_rel + n * self.post_beat, POST_SEQ, self.post_beat

    def next_after(self, planned_ns):
        return self.at_or_after(planned_ns + 1)


class StepTimes:
    """Replayed round durations (12.5(a) items 5, 6): {"decode_ns": d, "table": [[prefill_tokens,
    duration_ns], ...]} (sorted by tokens; a round's prefill duration is that of the first entry
    with tokens >= the round's prefill tokens, the last entry beyond the table). A round that
    prefills p tokens lasts decode_ns + (1 - f) * extra(p) + f * extra(previous round), where
    extra(p) = duration(p) - decode_ns and f = prefill_spill_frac."""

    def __init__(self, doc, spill_frac):
        if not isinstance(doc, dict) or not isinstance(doc.get("decode_ns"), int) \
                or doc["decode_ns"] <= 0 or not isinstance(doc.get("table"), list):
            raise ConfigError("snx step_time_replay: needs decode_ns > 0 and a table")
        rows = []
        for r in doc["table"]:
            if (not isinstance(r, list) or len(r) != 2 or not isinstance(r[0], int)
                    or not isinstance(r[1], int) or r[0] < 1 or r[1] < doc["decode_ns"]):
                raise ConfigError("snx step_time_replay: rows are [tokens >= 1, ns >= decode_ns]")
            rows.append((r[0], r[1]))
        rows.sort()
        self.decode_ns = doc["decode_ns"]
        self.keys = [r[0] for r in rows]
        self.vals = [r[1] for r in rows]
        self.f = float(spill_frac)
        self.prev_extra = 0

    def extra(self, prefill_tokens):
        if prefill_tokens <= 0 or not self.keys:
            return 0
        i = min(bisect.bisect_left(self.keys, prefill_tokens), len(self.keys) - 1)
        return self.vals[i] - self.decode_ns

    def round_ns(self, prefill_tokens):
        e = self.extra(prefill_tokens)
        d = self.decode_ns + int((1.0 - self.f) * e + self.f * self.prev_extra)
        self.prev_extra = e
        return d


def tier_shape(tier_table, cfg):
    """{tier_seq: tokens_per_step} and the one stream_interval of a tier table (all tiers must
    share it); without a table, the config's k and s."""
    if tier_table is None:
        return {}, cfg.stream_interval
    svals = {t.stream_interval for t in tier_table.tiers}
    if len(svals) != 1:
        raise ConfigError("snx: the tier table mixes stream_interval values %s" % sorted(svals))
    s = svals.pop()
    if cfg.stream_interval and cfg.stream_interval != s:
        raise ConfigError("snx: config stream_interval %d != the tier table's %d"
                          % (cfg.stream_interval, s))
    return {t.seq: t.tokens_per_step for t in tier_table.tiers}, s


class RecordFile:
    """Fixed-size binary records appended to `path` in chunks of about flush_bytes (or when
    flush() is called); path None only counts."""

    def __init__(self, path, st, flush_bytes):
        self.path = path
        self.st = st
        self.buf = bytearray()
        self.flush_bytes = flush_bytes
        self.n = 0
        self.lock = threading.Lock()
        if path is not None:
            with open(path, "wb"):
                pass

    def add(self, *vals):
        with self.lock:
            self.n += 1
            if self.path is None:
                return
            self.buf += self.st.pack(*vals)
            if len(self.buf) >= self.flush_bytes:
                self._flush_locked()

    def _flush_locked(self):
        if self.path is None or not self.buf:
            return
        with open(self.path, "ab") as f:
            f.write(self.buf)
        self.buf = bytearray()

    def flush(self):
        with self.lock:
            self._flush_locked()


def read_records(path, kind):
    """[{field: value}] of a record file written by RecordFile (kind: steps / takes / reqs)."""
    st, fields = RECORDS[kind]
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // st.size
    return [dict(zip(fields, st.unpack_from(data, i * st.size))) for i in range(n)]


class LiveReader:
    """Reads another process's snx.<pid>.live record (LIVE_FIELDS dict, or None while the file is
    missing or every retry saw a write in progress)."""

    def __init__(self, path):
        self.path = path
        self._m = None

    def read(self, retries=8):
        if self._m is None:
            try:
                with open(self.path, "rb") as f:
                    self._m = mmap.mmap(f.fileno(), LIVE_REC.size, access=mmap.ACCESS_READ)
            except (OSError, ValueError):
                return None
        for _ in range(retries):
            a = LIVE_REC.unpack_from(self._m, 0)
            if a[0] % 2 == 0 and a[0] > 0 and LIVE_REC.unpack_from(self._m, 0)[0] == a[0]:
                return dict(zip(LIVE_FIELDS, a))
        return None

    def close(self):
        if self._m is not None:
            self._m.close()
            self._m = None


class _Req:
    __slots__ = ("req_id", "llm", "client_id", "max_new", "prompt_len", "prompt_ids", "enq_ns",
                 "take_step", "take_planned", "produced", "iter", "k", "first_k", "rng",
                 "prefill_left", "ready_step", "first_emit_step", "n_resp", "cancelled", "tok_i")


class NullExecutor:
    """The PyExecutor stand-in (module docstring). `adapter` does every product-object operation:
    convert(req_id, executor_request) -> request; info(request, want_prompt) -> (client_id,
    max_new_tokens, prompt_len, prompt token ids or None); add_tokens(request, ids, decoding_iter);
    finish(request, cancelled); create_response(request) -> response or None.

    tier_table: a loaded capcommon TierTable (or None); llm_stream_interval: the server's
    stream_interval (llm_args), cross-checked against the table / config. self_drain: the step
    thread itself takes and drops the responses after each step (E2b's no-consumer calibration);
    arrival_hook(now_ns): called at the start of each round, before the step is timed (E2b's
    in-thread arrivals)."""

    def __init__(self, cfg, adapter, *, tier_table=None, llm_stream_interval=None,
                 max_seq_len=None, clock=time.monotonic_ns, self_drain=False,
                 arrival_hook=None, start=True, pin_info=None):
        self.cfg = cfg
        self.pin_info = pin_info
        self.adapter = adapter
        self.clock = clock
        self.tt = tier_table
        self.k_of_tier, s = tier_shape(tier_table, cfg)
        if not s:
            s = llm_stream_interval or 1
        if llm_stream_interval and llm_stream_interval != s:
            raise ConfigError("snx: the server's stream_interval %d != S-NX's %d (tier table or "
                              "config): pass the same value to both" % (llm_stream_interval, s))
        self.stream_interval = s
        self.max_seq_len = max_seq_len or cfg.max_seq_len
        self.k_mix = [(k, float(w)) for k, w in cfg.k_mix]
        self.replay = None
        if cfg.step_time_replay:
            with open(cfg.step_time_replay, "r", encoding="utf-8") as f:
                self.replay = StepTimes(json.load(f), cfg.prefill_spill_frac)
        self.steps_clock = StepClock(cfg.beat_ns, clock(), tier_table, cfg.origin_ns,
                                     cfg.outside_beat_ns)
        self.self_drain = self_drain
        self.arrival_hook = arrival_hook
        # PyExecutor-shaped state read by the worker
        self.is_shutdown = False
        self._fatal_error = None
        self._event_loop_error = None
        self.kv_cache_transceiver = None
        self.enable_iter_perf_stats = False
        self.responses = {}
        self.response_cv = threading.Condition()
        self._oldest_planned = 0
        self._req_cv = threading.Condition()
        self._queue = collections.deque()
        self._cancels = set()
        self._next_id = 1
        self._active = {}
        self._resource_governor_queue = None
        self.shutdown_event = threading.Event()
        self.counts = collections.Counter()
        self.max_wake_lag_ns = 0
        self.thread_ids = {}
        self._names_seen = {}  # native_id -> Python thread name, every name ever seen
        self._last_flush = clock()
        self._live = None
        self._live_lock = threading.Lock()
        self._live_gen = 0
        self._files_open(cfg.out_dir)
        self._thread = threading.Thread(target=self._run, name="snx_step", daemon=True)
        if start:
            self.start()

    # ---- files ---------------------------------------------------------------------------
    def _files_open(self, out_dir):
        self.out_dir = out_dir or None
        pid = os.getpid()
        self.prefix = os.path.join(out_dir, "snx.%d" % pid) if out_dir else None
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fb = self.cfg.flush_bytes
        self.rec = {kind: RecordFile(self.prefix + "." + kind + REC_EXT if out_dir else None, st,
                                     fb)
                    for kind, (st, _f) in RECORDS.items()}
        if self.prefix:
            with open(self.prefix + ".live", "w+b") as f:
                f.write(bytes(LIVE_REC.size))
                f.flush()
                self._live = mmap.mmap(f.fileno(), LIVE_REC.size)

    def _live_update(self):
        """Rewrite the live record (counters are read without the step lock: a snapshot)."""
        if self._live is None:
            return
        c = self.counts
        with self._live_lock:
            g = self._live_gen + 1
            LIVE_REC.pack_into(self._live, 0, g, 0, 0, 0, 0, 0, 0, 0)
            g += 1
            self._live_gen = g
            LIVE_REC.pack_into(self._live, 0, g, self.clock(), self._next_id - 1, c["taken"],
                               c["finished"], len(self._active), len(self._queue),
                               len(self.responses))

    def _write_json(self, name, doc):
        if not self.prefix:
            return
        path = self.prefix + "." + name
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)

    def meta_doc(self):
        self._names_seen.update(thread_names())
        return {"schema": SCHEMA_META, "pid": os.getpid(), "config": self.cfg.to_doc(),
                "stream_interval": self.stream_interval, "max_seq_len": self.max_seq_len,
                "k_of_tier": {str(k): v for k, v in sorted(self.k_of_tier.items())},
                "tier_table_sha256": getattr(self.tt, "sha256", None),
                "records": {k: {"format": st.format, "size": st.size, "fields": list(f)}
                            for k, (st, f) in RECORDS.items()},
                "threads": self.thread_ids, "self_drain": self.self_drain,
                "affinity": {"now": sorted(os.sched_getaffinity(0)), "pin": self.pin_info},
                "adapter": getattr(self.adapter, "name", type(self.adapter).__name__),
                "clock": "time.monotonic_ns"}

    def start(self):
        self.thread_ids["init"] = threading.get_native_id()
        self._thread.start()

    # ---- the PyExecutor interface the worker uses ----------------------------------------
    def enqueue_request(self, request, result_wait_queue=None):
        """Queue only (ExecutorRequestQueue.enqueue_request); conversion happens on the step
        thread at the start of the next round."""
        if result_wait_queue is not None:
            raise NotImplementedError("S-NX: result_wait_queue (Ray) is not supported")
        now = self.clock()
        with self._req_cv:
            if self.is_shutdown:
                raise RuntimeError("S-NX is shut down")
            req_id = self._next_id
            self._next_id += 1
            self._queue.append((req_id, request, now))
            self._req_cv.notify()
        self._live_update()
        return req_id

    def enqueue_requests(self, requests):
        return [self.enqueue_request(r) for r in requests]

    def cancel_request(self, id):  # noqa: A002 (PyExecutor's signature)
        with self._req_cv:
            self._cancels.add(id)
            self._req_cv.notify()

    def can_enqueue_requests(self):
        return not self.is_shutdown

    def await_responses(self, id=None, timeout=None):  # noqa: A002
        """PyExecutor.await_responses: id None takes every pending response at once."""
        t = timeout.total_seconds() if hasattr(timeout, "total_seconds") else timeout
        call = self.clock()
        with self.response_cv:
            if id is None:
                self.response_cv.wait_for(lambda: self.responses or self.is_shutdown, timeout=t)
                out = []
                keys = len(self.responses)
                for resps in self.responses.values():
                    out += resps
                self.responses = {}
            else:
                self.response_cv.wait_for(lambda: id in self.responses or self.is_shutdown,
                                          timeout=t)
                out = self.responses.pop(id, [])
                keys = 1 if out else 0
            oldest = self._oldest_planned
            if not self.responses:
                self._oldest_planned = 0
        if out:
            self.rec["takes"].add(call, self.clock(), oldest, len(out), keys)
            self._live_update()
        return out

    def get_latest_iteration_stats(self):
        return []

    def get_latest_kv_cache_events(self):
        return []

    def set_resource_governor_queue(self, queue):
        self._resource_governor_queue = queue

    def set_gather_responses(self, gather_all_responses):
        pass

    def wait_shutdown(self):
        self.shutdown_event.wait()

    def shutdown(self):
        with self._req_cv:
            if self.is_shutdown:
                return
            self.is_shutdown = True
            self._req_cv.notify_all()
        with self.response_cv:
            self.response_cv.notify_all()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=10.0)
        self.finalize()
        self.shutdown_event.set()

    def finalize(self):
        for r in self.rec.values():
            r.flush()
        self._write_json("summary.json", self.summary_doc())

    def summary_doc(self):
        self._names_seen.update(thread_names())
        c = dict(self.counts)
        c.update({"step_records": self.rec["steps"].n, "take_records": self.rec["takes"].n,
                  "req_records": self.rec["reqs"].n})
        return {"schema": SCHEMA_SUMMARY, "pid": os.getpid(), "counts": c,
                "max_wake_lag_ns": self.max_wake_lag_ns, "active_at_end": len(self._active),
                "queued_at_end": len(self._queue), "pending_keys_at_end": len(self.responses),
                "fatal_error": None if self._event_loop_error is None
                else repr(self._event_loop_error),
                "stream_interval": self.stream_interval,
                "thread_names": dict(self._names_seen),
                "affinity_end": sorted(os.sched_getaffinity(0))}

    # ---- the step thread -----------------------------------------------------------------
    def _run(self):
        self.thread_ids["snx_step"] = threading.get_native_id()
        self._write_json("meta.json", self.meta_doc())
        try:
            self._loop()
        except BaseException as e:  # noqa: BLE001 -- reported like PyExecutor's event loop
            self._event_loop_error = e
            self.counts["step_thread_errors"] += 1
            with self._req_cv:
                self.is_shutdown = True
            with self.response_cv:
                self.response_cv.notify_all()
            try:
                self.finalize()
            finally:
                self.shutdown_event.set()
            raise

    def _idle(self):
        with self._req_cv:
            return not self._active and not self._queue and not self._cancels

    def _wait_for_work(self):
        """Block while idle; True once there is work, False on shutdown."""
        with self._req_cv:
            while not self.is_shutdown and not self._active and not self._queue \
                    and not self._cancels:
                self._req_cv.wait(0.5)
            return not self.is_shutdown

    def _loop(self):
        clock = self.clock
        planned, seq, beat = self.steps_clock.at_or_after(clock())
        step = 0
        while True:
            if self.arrival_hook is None and self._idle():
                if not self._wait_for_work():
                    return
                now = clock()
                self.counts["idle_restarts"] += 1
                if self.cfg.idle_restart == "grid" and self.replay is None:
                    planned, seq, beat = self.steps_clock.at_or_after(now)
                else:
                    planned = now
                    if self.replay is None:
                        _p, seq, beat = self.steps_clock.at_or_after(now)
            if self.is_shutdown:
                return
            while True:
                now = clock()
                if now >= planned:
                    break
                time.sleep((planned - now) / NS)
            if self.arrival_hook is not None:
                self.arrival_hook(clock())
            prefill = self._step(step, planned, seq, beat)
            step += 1
            if self.replay is not None:
                planned = planned + self.replay.round_ns(prefill)
                _p, seq, beat = self.steps_clock.at_or_after(planned)
            else:
                planned, seq, beat = self.steps_clock.next_after(planned)
            if clock() - self._last_flush >= self.cfg.flush_interval_ms * 1_000_000:
                self._last_flush = clock()
                for r in self.rec.values():
                    r.flush()
                self._write_json("summary.json", self.summary_doc())

    def _k(self, seq):
        return self.k_of_tier.get(seq, self.cfg.tokens_per_step) if self.k_of_tier \
            else self.cfg.tokens_per_step

    def _take(self, step, planned, seq):
        cap = self.cfg.max_active
        with self._req_cv:
            if cap:
                room = max(0, cap - len(self._active))
                taken = [self._queue.popleft() for _ in range(min(room, len(self._queue)))]
            else:
                taken = list(self._queue)
                self._queue.clear()
            cancels = self._cancels
            self._cancels = set() if cancels else cancels
            queued_after = len(self._queue)
        ad = self.adapter
        k = self._k(seq)
        for req_id, ereq, enq_ns in taken:
            r = _Req()
            r.llm = ad.convert(req_id, ereq)
            client_id, max_new, prompt_len, prompt_ids = ad.info(
                r.llm, self.cfg.token_source == "prompt")
            r.req_id, r.client_id, r.enq_ns = req_id, client_id, enq_ns
            r.max_new = max_new
            r.prompt_len = prompt_len
            r.prompt_ids = prompt_ids
            r.take_step, r.take_planned = step, planned
            r.produced = r.iter = r.n_resp = r.tok_i = 0
            r.k = k
            r.first_k = self.cfg.first_step_tokens or k
            r.rng = random.Random(self.cfg.k_mix_seed * 1_000_003 + req_id) if self.k_mix \
                else None
            r.prefill_left = prompt_len if self.cfg.prefill_token_budget else 0
            r.ready_step = None if r.prefill_left else step + self.cfg.first_response_delay_steps
            r.first_emit_step = -1
            r.cancelled = False
            self._active[req_id] = r
        return len(taken), cancels, queued_after

    def _tokens(self, r, n):
        if r.prompt_ids:
            src = r.prompt_ids
        else:
            src = self.cfg.token_ids
        out = []
        for _ in range(n):
            out.append(src[r.tok_i % len(src)])
            r.tok_i += 1
        return out

    def _next_k(self, r):
        if r.iter == 0:
            return r.first_k
        if r.rng is not None:
            x = r.rng.random() * sum(w for _k, w in self.k_mix)
            for k, w in self.k_mix:
                x -= w
                if x < 0:
                    return k
            return self.k_mix[-1][0]
        return r.k

    def _step(self, step, planned, seq, beat):
        clock = self.clock
        t0 = clock()
        c0 = time.thread_time_ns()
        lag = t0 - planned
        if lag > self.max_wake_lag_ns:
            self.max_wake_lag_ns = lag
        tk0 = clock()
        n_taken, cancels, queued_after = self._take(step, planned, seq)
        take_ns = clock() - tk0
        ad = self.adapter
        s = self.stream_interval
        out = []
        done = []
        n_tokens = 0
        if cancels:
            for req_id in cancels:
                r = self._active.get(req_id)
                if r is None:
                    continue
                r.cancelled = True
                ad.finish(r.llm, True)
                resp = ad.create_response(r.llm)
                if resp is not None:
                    out.append((req_id, resp))
                    r.n_resp += 1
                done.append(r)
                self.counts["cancelled"] += 1
        budget = self.cfg.prefill_token_budget
        prefill = 0
        if budget:
            left = budget
            for r in self._active.values():
                if r.prefill_left <= 0 or r.cancelled:
                    continue
                if left <= 0:
                    break
                n = min(r.prefill_left, left)
                r.prefill_left -= n
                left -= n
                prefill += n
                if r.prefill_left == 0:
                    r.ready_step = step + self.cfg.first_response_delay_steps
        for r in self._active.values():
            if r.cancelled or r.ready_step is None or step < r.ready_step:
                continue
            n = min(self._next_k(r), r.max_new - r.produced)
            r.iter += 1
            if n > 0:
                ad.add_tokens(r.llm, self._tokens(r, n), r.iter)
                r.produced += n
                n_tokens += n
            finished = r.produced >= r.max_new
            if finished:
                ad.finish(r.llm, False)
            if r.iter == 1 or finished or r.iter % s == 0:
                resp = ad.create_response(r.llm)
                if resp is not None:
                    out.append((r.req_id, resp))
                    r.n_resp += 1
                    if r.first_emit_step < 0:
                        r.first_emit_step = step
            if finished:
                done.append(r)
        for r in done:
            self._active.pop(r.req_id, None)
            self.rec["reqs"].add(r.req_id, r.client_id or 0, r.enq_ns, r.take_step,
                                 r.take_planned, r.first_emit_step, step, planned, r.produced,
                                 r.n_resp, r.prompt_len, 1 if r.cancelled else 0)
        busy_iters = 0
        if self.cfg.busy_ns_per_step:
            end = time.perf_counter_ns() + self.cfg.busy_ns_per_step
            while time.perf_counter_ns() < end:
                busy_iters += 1
        if out:
            with self.response_cv:
                if not self.responses:
                    self._oldest_planned = planned
                for req_id, resp in out:
                    lst = self.responses.get(req_id)
                    if lst is None:
                        self.responses[req_id] = [resp]
                    else:
                        lst.append(resp)
                self.response_cv.notify_all()
        c1 = time.thread_time_ns()
        t1 = clock()
        self.rec["steps"].add(step, planned, t0, lag, t1 - t0, c1 - c0, take_ns, seq,
                              len(self._active) + len(done), n_taken, len(out), len(done),
                              n_tokens, prefill, queued_after, busy_iters)
        cnt = self.counts
        cnt["steps"] += 1
        if lag >= beat:
            cnt["late_steps"] += 1
        cnt["taken"] += n_taken
        cnt["released"] += len(out)
        cnt["finished"] += len(done)
        cnt["tokens"] += n_tokens
        if self.self_drain and out:
            got = self.await_responses(timeout=0)
            cnt["self_drained"] += len(got)
        self._live_update()
        return prefill


# ---- offline summary ------------------------------------------------------------------------
def _q(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, int(q * len(v)))]


def _stats(vals):
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": sum(vals) / len(vals), "p50": _q(vals, 0.5),
            "p99": _q(vals, 0.99), "max": max(vals)}


def summarize_window(steps, takes, reqs, lo, hi):
    """Per-window self-report of S-NX: steps planned in [lo, hi), takes returned in it, requests
    finished in it (absolute monotonic ns)."""
    st = [r for r in steps if lo <= r["planned_ns"] < hi]
    tk = [r for r in takes if lo <= r["return_ns"] < hi]
    rq = [r for r in reqs if lo <= r["finish_planned_ns"] < hi]
    dur = (hi - lo) / NS
    out = {"window_s": dur, "steps": len(st),
           "late_steps": sum(1 for r in st if r["wake_lag_ns"] >= r.get("beat_ns", 1 << 62)),
           "wake_lag_ns": _stats([r["wake_lag_ns"] for r in st]),
           "step_cpu_ns": _stats([r["cpu_ns"] for r in st]),
           "step_wall_ns": _stats([r["wall_ns"] for r in st]),
           "take_ns": _stats([r["take_ns"] for r in st]),
           "step_cpu_share": (sum(r["cpu_ns"] for r in st) / NS / dur) if dur > 0 else None,
           "active": _stats([r["active"] for r in st]),
           "released_per_step": _stats([r["released"] for r in st]),
           "taken": sum(r["taken"] for r in st), "released": sum(r["released"] for r in st),
           "tokens": sum(r["tokens"] for r in st), "finished": sum(r["finished"] for r in st),
           "queued_after_max": max((r["queued_after"] for r in st), default=0),
           "takes": len(tk), "take_responses": sum(r["responses"] for r in tk),
           "take_lag_ns": _stats([r["return_ns"] - r["oldest_planned_ns"] for r in tk
                                  if r["oldest_planned_ns"]]),
           "responses_per_take": _stats([r["responses"] for r in tk]),
           "reqs_finished": len(rq), "reqs_cancelled": sum(r["cancelled"] for r in rq)}
    if dur > 0:
        for k in ("taken", "released", "tokens", "take_responses"):
            out[k + "_per_s"] = out[k] / dur
    return out


def load_dir(d):
    """{pid: {"meta", "summary", "steps", "takes", "reqs"}} of every S-NX process in d."""
    out = {}
    for name in sorted(os.listdir(d)):
        if not (name.startswith("snx.") and name.endswith(".meta.json")):
            continue
        pid = name.split(".")[1]
        p = os.path.join(d, "snx." + pid)
        e = {}
        with open(p + ".meta.json", "r", encoding="utf-8") as f:
            e["meta"] = json.load(f)
        e["summary"] = None
        if os.path.exists(p + ".summary.json"):
            with open(p + ".summary.json", "r", encoding="utf-8") as f:
                e["summary"] = json.load(f)
        for kind in RECORDS:
            e[kind] = read_records(p + "." + kind + REC_EXT, kind) \
                if os.path.exists(p + "." + kind + REC_EXT) else []
        out[pid] = e
    return out


def summarize(d, tier_table_path=None, origin_ns=None):
    data = load_dir(d)
    doc = {"schema": SCHEMA_TIERS, "dir": os.path.abspath(d), "processes": {}}
    tt = None
    if tier_table_path:
        from capcommon.tiers import TierTable  # noqa: PLC0415
        tt = TierTable.load(tier_table_path)
    for pid, e in data.items():
        org = origin_ns if origin_ns is not None else e["meta"]["config"].get("origin_ns") or 0
        beats = {}
        if tt is not None:
            beats = {t.seq: t.beat_ns for t in tt.tiers}
        for r in e["steps"]:
            r["beat_ns"] = beats.get(r["tier_seq"], e["meta"]["config"]["beat_ns"])
        p = {"summary": e["summary"], "meta_threads": e["meta"].get("threads"),
             "whole": summarize_window(e["steps"], e["takes"], e["reqs"], -(1 << 62), 1 << 62),
             "tiers": []}
        if tt is not None:
            for t in tt.tiers:
                lo, hi = org + t.window_start_ns, org + t.window_end_ns
                w = summarize_window(e["steps"], e["takes"], e["reqs"], lo, hi)
                w.update({"seq": t.seq, "label": t.label, "kind": t.kind, "beat_ns": t.beat_ns,
                          "tokens_per_step": t.tokens_per_step,
                          "stream_interval": t.stream_interval})
                p["tiers"].append(w)
        doc["processes"][pid] = p
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m capstubs.null_executor", allow_abbrev=False,
                                description="S-NX self-report tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize", help="per-tier summary of an S-NX out_dir")
    s.add_argument("--dir", required=True)
    s.add_argument("--tier-table")
    s.add_argument("--origin-ns", type=int)
    s.add_argument("--out", required=True)
    c = sub.add_parser("check-config", help="validate a capkit.snx_config/1 JSON file")
    c.add_argument("--config", required=True)
    a = p.parse_args(argv)
    try:
        if a.cmd == "summarize":
            doc = summarize(a.dir, a.tier_table, a.origin_ns)
            with open(a.out, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1, sort_keys=True)
                f.write("\n")
            return 0
        with open(a.config, "r", encoding="utf-8") as f:
            SnxConfig.from_doc(json.load(f))
        return 0
    except (ConfigError, OSError, ValueError) as e:
        print("null_executor: %s" % e, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
