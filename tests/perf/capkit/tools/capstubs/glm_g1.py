# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E10 G1: low-occupancy engine substitute for glm5-serve (design card E10, "G1（新写）").

Implements the engine interface of glm5_serve/engine_api.py (the one MockEngine implements):
start, info, add_request, abort_request, get_responses, stats, shutdown. capstubs.serve_glm puts
it in place of glm5_serve.engine_api.MockEngine for the G1 arm; nothing in the glm checkout is
changed.

Differences from MockEngine (G0), each one on purpose:
- The step thread ("capg1-step") wakes on an absolute grid: step j is due at t0 + j * step_ns;
  it sleeps (time.sleep, GIL released) until the due time and never accumulates drift (MockEngine
  sleeps step_s after its work, so its period is step_s + work). A late step runs at once and the
  grid does not move: the number of steps it is behind is reported.
- add_request / abort_request (event-loop thread) only append to a list under a lock; the step
  thread takes the lists once per step. Token chunks are built once at construction (the pattern
  cycled, tokens_per_step per response); per step and request the thread builds one response dict.
- One enqueue per step: the whole step's responses are appended to the output list and the
  waiting get_responses() is notified once.
- No stop tokens, logprobs, scripts or min_tokens: every request ends with finish_reason "length"
  after max_tokens tokens. A request asking for logprobs is refused (ValueError -> the servicer
  answers INVALID_ARGUMENT), so the arm cannot silently differ from what was asked.
- Self-report: per step the wake lag (actual wake - due time), the thread CPU of the step's work
  (time.thread_time_ns), responses, and the steps behind; read by serve_glm's reporter through
  stats_snapshot().
"""

import threading
import time

DEFAULT_EOS = (154820, 154827, 154829)  # MockEngine's default (GLM-5.3)
MOCK_VOCAB = 154880  # MockEngine.info()["vocab_size"]
LAG_BUCKETS_US = (10, 50, 100, 250, 500, 1000, 2000, 5000, 10000, 50000)
INSTANCES = []


def bucket(us):
    for i, b in enumerate(LAG_BUCKETS_US):
        if us <= b:
            return i
    return len(LAG_BUCKETS_US)


class StepStats:
    """Written by the step thread only; readers take snapshot() (int reads, no lock)."""

    def __init__(self):
        self.steps = 0
        self.busy_steps = 0
        self.responses = 0
        self.finished = 0
        self.wake_lag_ns = 0
        self.cpu_ns = 0
        self.lag_hist = [0] * (len(LAG_BUCKETS_US) + 1)
        self.max_wake_lag_ns = 0
        self.max_cpu_ns = 0
        self.max_behind = 0

    def snapshot(self, reset_max=True):
        d = {k: v for k, v in vars(self).items() if k != "lag_hist"}
        d["lag_hist"] = list(self.lag_hist)
        d["lag_buckets_us"] = list(LAG_BUCKETS_US)
        if reset_max:
            self.max_wake_lag_ns = self.max_cpu_ns = self.max_behind = 0
        return d


class G1Engine:
    KIND = "g1"

    def __init__(self, config):
        self.config = dict(config)
        self.eos = list(config.get("eos_token_ids", DEFAULT_EOS))
        pattern = [int(t) for t in config.get("mock_pattern", [1, 2, 3])]
        if not pattern:
            raise ValueError("mock_pattern is empty")
        self.k = max(1, int(config.get("mock_tokens_per_step", 1)))
        self.step_ns = int(round(float(config.get("mock_step_ms", 2.0)) * 1e6))
        if self.step_ns <= 0:
            raise ValueError("G1 needs mock_step_ms > 0 (the absolute step grid)")
        p = len(pattern)
        # chunk for a response whose first token is at position pos: chunks[pos % p]
        self.chunks = [[pattern[(s + i) % p] for i in range(self.k)] for s in range(p)]
        self.p = p
        self._lock = threading.Lock()
        self._new = []
        self._aborts = []
        self._active = {}
        self._cond = threading.Condition(threading.Lock())
        self._out = []
        self._stop = threading.Event()
        self._thread = None
        self.st = StepStats()
        self.t0_ns = None
        INSTANCES.append(self)

    def start(self):
        self._thread = threading.Thread(target=self._loop, name="capg1-step", daemon=True)
        self._thread.start()

    def info(self):
        return {"max_seq_len": self.config.get("max_seq_len") or 8192,
                "max_batch_size": self.config.get("max_batch_size", 256),
                "max_num_tokens": self.config.get("max_num_tokens", 8192),
                "vocab_size": MOCK_VOCAB, "eos_token_ids": self.eos,
                "tp_size": self.config.get("tp_size", 1), "kv_cache_tokens": 1 << 20,
                "version": "capkit-g1"}

    def add_request(self, request_id, prompt_token_ids, params):
        if "max_tokens" not in params:
            raise ValueError("max_tokens is required")
        if params.get("logprobs") is not None:
            raise ValueError("the G1 substitute does not produce logprobs")
        n = int(params["max_tokens"])
        if n < 1:
            raise ValueError("max_tokens must be >= 1")
        if len(prompt_token_ids) >= self.info()["max_seq_len"]:
            raise ValueError("prompt length %d exceeds max_seq_len" % len(prompt_token_ids))
        with self._lock:
            self._new.append((request_id, n, len(prompt_token_ids)))

    def abort_request(self, request_id):
        with self._lock:
            self._aborts.append(request_id)

    def get_responses(self, timeout_ms):
        with self._cond:
            if not self._out:
                self._cond.wait(timeout_ms / 1000.0)
            out, self._out = self._out, []
        return out

    def stats(self):
        n = len(self._active)
        return {"num_active_requests": n, "iter": self.st.busy_steps,
                "iter_latency_ms": self.step_ns / 1e6,
                "kv_cache_stats": {"max_num_blocks": 32768, "free_num_blocks": 32768 - n,
                                   "used_num_blocks": n, "tokens_per_block": 32},
                "inflight_batching_stats": {"num_scheduled_requests": n, "num_gen_requests": n}}

    def stats_snapshot(self, reset_max=True):
        d = self.st.snapshot(reset_max)
        d.update(kind=self.KIND, step_ns=self.step_ns, tokens_per_step=self.k,
                 active=len(self._active), t0_ns=self.t0_ns)
        return d

    def _resp(self, rid, toks, n, nprompt, finished, reason):
        return {"request_id": rid, "token_ids": toks, "logprobs": None, "finished": finished,
                "finish_reason": reason, "stop_token_id": None, "num_prompt_tokens": nprompt,
                "num_generated_tokens": n}

    def _step(self):
        """One step's work; returns the responses (step thread only)."""
        with self._lock:
            new, self._new = self._new, []
            aborts, self._aborts = self._aborts, []
        act = self._active
        for rid, n, nprompt in new:
            act[rid] = [0, n, nprompt]
        out = []
        for rid in aborts:
            r = act.pop(rid, None)
            if r is not None:
                out.append(self._resp(rid, [], r[0], r[2], True, "abort"))
        done = []
        k, p, chunks = self.k, self.p, self.chunks
        for rid, r in act.items():
            pos, n = r[0], r[1]
            toks = chunks[pos % p]
            left = n - pos
            if left <= k:
                if left < k:
                    toks = toks[:left]
                r[0] = n
                out.append(self._resp(rid, toks, n, r[2], True, "length"))
                done.append(rid)
            else:
                r[0] = pos + k
                out.append(self._resp(rid, toks, pos + k, r[2], False, None))
        for rid in done:
            del act[rid]
        st = self.st
        st.finished += len(done)
        return out

    def _loop(self):
        st, step, mono, cpu = self.st, self.step_ns, time.monotonic_ns, time.thread_time_ns
        due = self.t0_ns = mono() + step
        while not self._stop.is_set():
            now = mono()
            if now < due:
                time.sleep((due - now) / 1e9)
                now = mono()
            lag = now - due
            c0 = cpu()
            out = self._step()
            if out:
                with self._cond:
                    self._out.extend(out)
                    self._cond.notify()
                st.busy_steps += 1
                st.responses += len(out)
            c = cpu() - c0
            st.steps += 1
            st.wake_lag_ns += lag
            st.cpu_ns += c
            st.lag_hist[bucket(lag // 1000)] += 1
            if lag > st.max_wake_lag_ns:
                st.max_wake_lag_ns = lag
            if c > st.max_cpu_ns:
                st.max_cpu_ns = c
            behind = lag // step
            if behind > st.max_behind:
                st.max_behind = behind
            due += step

    def shutdown(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
