# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Probe group `stack`: an in-process stack sampler for the frontend's event-loop thread.

A daemon thread (capkit-stack) wakes every PERIOD_NS, reads sys._current_frames() and classifies
the stack of the main thread (where the frontend runs its event loop) into one CATEGORIES entry:
frames are visited from the innermost outwards and the first frame with a definite category
decides. Frames of the loop machinery (asyncio, uvloop, anyio, contextlib, ...) are transparent;
serialisation frames (pydantic, json, openai_protocol) take the category of the first definite
frame outside them, except that serialisation inside the streaming generator counts as `encode`.
When the innermost Python frame is the loop runner itself (or, on a pure-Python asyncio loop, the
selector wait), no Python callback is running: the
thread is waiting for I/O or busy inside C code of the loop (`idle`; its CPU share tells the two
apart).

Each sample is weighted by the wall time since the previous sample (stack.main.wall_ns.<cat>)
and by the main thread's CPU time over the same interval (stack.main.cpu_ns.<cat>); the plain
sample count is stack.main.n.<cat>. The sampler needs the GIL to read the frames, so a sample is
taken at the next point where the main thread releases it: Python code is sampled close to where
it runs, a long C call holding the GIL is attributed to the Python frame that called it, and the
wall weighting gives a delayed sample the time it waited.

Other Python threads (tokenizer pool, executor threads) are classified the same way unless their
innermost frame is a known waiting point; their wall-weighted samples go to
stack.thr.wall_ns.<cat> (thread-seconds, summed over threads). capkit's own threads are skipped.

Child processes (postprocessing pool) are read from /proc every CHILD_POLL_NS:
stack.children_cpu_ns is the sum over every child ever seen of its utime + stime (+ waited-for
grandchildren), stack.children the number of live children at the last poll.

Totals: stack.samples, stack.wall_ns (sum of the weights), stack.main_cpu_ns, stack.proc_cpu_ns
(all threads of this process), stack.sampler_cpu_ns (the sampler thread's own CPU: the direct
cost), stack.errors. Per-function detail (the deciding frame, file:function, wall ns) is kept in
the snapshot meta under `stack_funcs` (cumulative, refreshed every FUNCS_REFRESH_NS, at most
FUNCS_MAX entries by weight).
"""

import os
import sys
import threading
import time

from . import sites as _sites

PERIOD_NS = 4_000_000
CHILD_POLL_NS = 500_000_000
FUNCS_REFRESH_NS = 2_000_000_000
FUNCS_MAX = 200
THREAD_NAME = "capkit-stack"

CATEGORIES = _sites.STACK_CATEGORIES
TOTALS = _sites.STACK_TOTALS

# Frame tags that are not categories.
RUNNER = "<runner>"   # the loop runner: innermost => idle; reached from below => loop
PASS = "<pass>"       # transparent loop machinery
SERDE = "<serde>"     # serialisation: category of the first definite frame outside

# (substring of the normalised file name, tag); first match wins. Function-dependent files are
# handled in _tag_of.
_FILE_RULES = (
    ("/capprobe/", "probe"), ("/capcommon/", "probe"),
    ("/tensorrt_llm/grpc/", "grpc"), ("/grpc/", "grpc"), ("/google/protobuf/", "grpc"),
    ("/openengine_poc/postproc_pool.py", "pool_ipc"), ("/tensorrt_llm/executor/ipc.py",
                                                      "pool_ipc"),
    ("/tensorrt_llm/executor/serialization.py", "pool_ipc"), ("/zmq/", "pool_ipc"),
    ("/pickle.py", "pool_ipc"), ("/copyreg.py", "pool_ipc"), ("/hmac.py", "pool_ipc"),
    ("/multiprocessing/", "pool_ipc"),
    ("/tokenizers/", "tok"), ("/transformers/", "tok"), ("/tensorrt_llm/serve/detokenize.py",
                                                        "tok"),
    ("/tensorrt_llm/llmapi/tokenizer.py", "tok"), ("/tensorrt_llm/tokenizer/", "tok"),
    ("/openengine_poc/response.py", "assemble"),
    ("/openengine_poc/launch.py", PASS),
    ("/openengine_poc/", "facade"),
    ("/tensorrt_llm/serve/postprocess_handlers.py", "openai_stream"),
    ("/tensorrt_llm/serve/openai_protocol.py", SERDE),
    ("/tensorrt_llm/serve/responses_utils.py", PASS),
    ("/pydantic/", SERDE), ("/pydantic_core/", SERDE), ("/json/", SERDE), ("/orjson", SERDE),
    ("/starlette/responses.py", "http_out"),
    ("/h11/", "http_in"), ("/httptools/", "http_in"), ("/fastapi/", "http_in"),
    ("/asyncio/runners.py", RUNNER), ("/uvloop/__init__.py", RUNNER), ("/selectors.py", RUNNER),
    ("/asyncio/", PASS), ("/uvloop/", PASS), ("/anyio/", PASS), ("/contextlib.py", PASS),
    ("/functools.py", PASS), ("/typing.py", PASS), ("/abc.py", PASS), ("/weakref.py", PASS),
    ("/enum.py", PASS), ("/dataclasses.py", PASS), ("/starlette/", PASS),
    ("/tensorrt_llm/", "trtllm_other"),
)
_UVICORN_OUT = frozenset(("send", "_send", "write", "flush", "transport_write", "send_wrapper"))
# Innermost (file substring, function) of a non-main thread that is waiting, not working.
_WAIT_POINTS = (
    ("/threading.py", None), ("/queue.py", None), ("/selectors.py", None),
    ("/concurrent/futures/thread.py", "_worker"), ("/socket.py", None), ("/ssl.py", None),
    ("/multiprocessing/connection.py", None), ("/multiprocessing/queues.py", None),
    ("/zmq/", None), ("/asyncio/runners.py", None), ("/uvloop/__init__.py", None),
)


# A sample whose GIL wait is at least this long was most likely taken at a forced switch (the
# interpreter's switch interval is 5 ms): the main thread was running Python code, not waiting.
FORCED_NS = 2_000_000
# /proc/<pid>/task/<tid>/syscall first field (x86_64 numbers) -> class
_SYSCALL_CLASS = {"232": "epoll", "281": "epoll", "441": "epoll", "7": "epoll", "271": "epoll",
                  "202": "futex", "0": "io", "1": "io", "19": "io", "20": "io", "44": "io",
                  "45": "io", "46": "io", "47": "io", "running": "other"}


def _short(path):
    """Last two components of a file name (logging/__init__.py, serve/openai_server.py)."""
    parts = path.replace("\\", "/").rsplit("/", 2)
    return "/".join(parts[-2:])


def _norm(path):
    return "/" + path.replace("\\", "/").lstrip("/")


def _tag_of(code):
    """Tag of one code object: a CATEGORIES entry, RUNNER, PASS or SERDE."""
    f = _norm(code.co_filename)
    name = code.co_name
    if "/uvicorn/" in f:
        return "http_out" if name in _UVICORN_OUT else "http_in"
    if "/tensorrt_llm/serve/openai_server.py" in f:
        return "openai_stream" if "generator" in name or "stream" in name else "openai_req"
    if "/openengine_poc/launch.py" in f and name in ("main", "<module>"):
        return RUNNER
    for sub, tag in _FILE_RULES:
        if sub in f:
            return tag
    return "other"


def _is_wait_point(code):
    f = _norm(code.co_filename)
    for sub, fn in _WAIT_POINTS:
        if sub in f and (fn is None or code.co_name == fn):
            return True
    return False


class Classifier:
    """Stack -> (category, deciding code) with a per-code-object tag cache. The cache is keyed
    by id() and keeps the code object: code objects compare equal across files when their
    bytecode and names match, so they cannot be dict keys themselves."""

    def __init__(self):
        self._tags = {}
        self._wait = {}

    def tag(self, code):
        e = self._tags.get(id(code))
        if e is None or e[0] is not code:
            e = self._tags[id(code)] = (code, _tag_of(code))
        return e[1]

    def waiting(self, code):
        e = self._wait.get(id(code))
        if e is None or e[0] is not code:
            e = self._wait[id(code)] = (code, _is_wait_point(code))
        return e[1]

    def classify(self, frame):
        """(category, code of the deciding frame or None) for the stack whose innermost frame
        is `frame`."""
        tag = self.tag
        first = True
        serde = None
        while frame is not None:
            code = frame.f_code
            t = tag(code)
            if t is RUNNER:
                if first:
                    return "idle", code
                return ("encode" if serde is not None else "loop"), serde
            if t is SERDE:
                if serde is None:
                    serde = code
            elif t is not PASS:
                if serde is not None:
                    return ("encode" if t == "openai_stream" else t), serde
                return t, code
            first = False
            frame = frame.f_back
        return ("encode" if serde is not None else "loop"), serde


def _clk_tck():
    try:
        return os.sysconf("SC_CLK_TCK")
    except (ValueError, OSError):
        return 100


def _child_pids(proc_root="/proc"):
    """Direct children of every thread of this process (Linux /proc/<pid>/task/*/children)."""
    pid = os.getpid()
    out = set()
    base = os.path.join(proc_root, str(pid), "task")
    try:
        tids = os.listdir(base)
    except OSError:
        return out
    for tid in tids:
        try:
            with open(os.path.join(base, tid, "children")) as f:
                out.update(int(x) for x in f.read().split())
        except (OSError, ValueError):
            continue
    return out


def _proc_cpu_ticks(pid, proc_root="/proc"):
    """(starttime, utime + stime + cutime + cstime) in clock ticks, or None when gone."""
    try:
        with open(os.path.join(proc_root, str(pid), "stat")) as f:
            s = f.read()
    except OSError:
        return None
    rest = s[s.rfind(")") + 2:].split()
    try:
        return int(rest[19]), sum(int(rest[i]) for i in (11, 12, 13, 14))
    except (IndexError, ValueError):
        return None


class StackSampler:
    """The sampler thread. `registry` holds the stack.* slots (registered by capprobe._setup);
    `writer` (optional) gets the per-function table through set_extra("stack_funcs", ...)."""

    def __init__(self, registry, writer=None, period_ns=PERIOD_NS, main_ident=None,
                 proc_root="/proc"):
        self.reg = registry
        self.writer = writer
        self.period_ns = period_ns
        self.main_ident = (threading.main_thread().ident if main_ident is None
                           else main_ident)
        self.proc_root = proc_root
        self.cls = Classifier()
        self.funcs = {}
        self._children = {}
        self._tck_ns = 1_000_000_000 // _clk_tck()
        self._stop = threading.Event()
        self._thread = None
        v = registry.values
        off = registry.counter
        self._v = v
        self._tot = {t: off("stack.%s" % t) for t in TOTALS}
        self._nchild = registry.gauge("stack.children")
        self._n = {c: off("stack.main.n.%s" % c) for c in CATEGORIES}
        self._wall = {c: off("stack.main.wall_ns.%s" % c) for c in CATEGORIES}
        self._cpu = {c: off("stack.main.cpu_ns.%s" % c) for c in CATEGORIES}
        self._thr = {c: off("stack.thr.wall_ns.%s" % c) for c in CATEGORIES}
        self._gil = {c: off("stack.main.gil_ns.%s" % c) for c in CATEGORIES}
        self._forced = {c: off("stack.main.forced.%s" % c) for c in CATEGORIES}
        self._idle_sys = {k: off("stack.main.idle_sys.%s" % k) for k in _sites.STACK_SYSCALLS}
        nid = threading.main_thread().native_id if main_ident is None else None
        self._sys_path = (None if nid is None else
                          os.path.join(proc_root, "self", "task", str(nid), "syscall"))
        try:
            self._main_clock = time.pthread_getcpuclockid(self.main_ident)
        except (OSError, AttributeError, OverflowError):
            self._main_clock = None

    def _main_cpu(self):
        if self._main_clock is None:
            return 0
        try:
            return time.clock_gettime_ns(self._main_clock)
        except OSError:
            return 0

    def _main_syscall(self):
        """Class of the main thread's current system call: epoll, futex, io or other."""
        try:
            with open(self._sys_path) as f:
                tok = f.read(16).split(" ", 1)[0]
        except (OSError, TypeError):
            return "other"
        return _SYSCALL_CLASS.get(tok, "other")

    def sample_once(self, dt_ns, dcpu_ns, skip_idents, gil_ns=0):
        """Take one sample weighted by dt_ns of wall and dcpu_ns of main-thread CPU. gil_ns: how
        long the sampler waited for the GIL after its sleep (the main thread or another thread
        held it); the sample's category gets that busy time too (stack.main.gil_ns.<cat>)."""
        v = self._v
        frames = sys._current_frames()
        main = frames.get(self.main_ident)
        cls = self.cls
        if main is not None:
            cat, code = cls.classify(main)
            v[self._n[cat]] += 1
            v[self._wall[cat]] += dt_ns
            v[self._cpu[cat]] += dcpu_ns
            v[self._gil[cat]] += gil_ns
            if gil_ns >= FORCED_NS:
                v[self._forced[cat]] += 1
            if cat == "idle":
                v[self._idle_sys[self._main_syscall()]] += 1
            else:
                key = "%s|%s:%s" % (cat, _short(code.co_filename), code.co_name) \
                    if code is not None else "%s|-" % cat
                self.funcs[key] = self.funcs.get(key, 0) + dt_ns
        for ident, fr in frames.items():
            if ident == self.main_ident or ident in skip_idents:
                continue
            if cls.waiting(fr.f_code):
                continue
            cat, _code = cls.classify(fr)
            if cat != "idle":
                v[self._thr[cat]] += dt_ns
        del frames, main
        v[self._tot["samples"]] += 1
        v[self._tot["wall_ns"]] += dt_ns
        v[self._tot["main_cpu_ns"]] += dcpu_ns
        v[self._tot["gil_ns"]] += gil_ns

    def poll_children(self):
        seen = self._children
        live = 0
        for pid in _child_pids(self.proc_root):
            r = _proc_cpu_ticks(pid, self.proc_root)
            if r is None:
                continue
            live += 1
            seen[(pid, r[0])] = r[1]
        self._v[self._tot["children_cpu_ns"]] = sum(seen.values()) * self._tck_ns
        self._v[self._nchild] = live

    def funcs_table(self):
        items = sorted(self.funcs.items(), key=lambda kv: -kv[1])[:FUNCS_MAX]
        return {k: w for k, w in items}

    def _skip_idents(self):
        me = threading.get_ident()
        return frozenset([me] + [t.ident for t in threading.enumerate()
                                 if t.name.startswith("capkit-") and t.ident is not None])

    def _run(self):
        v, tot = self._v, self._tot
        t_prev = time.monotonic_ns()
        c_prev = self._main_cpu()
        next_child = t_prev
        next_funcs = t_prev + FUNCS_REFRESH_NS
        skip = self._skip_idents()
        next_skip = t_prev + CHILD_POLL_NS
        period_s = self.period_ns / 1e9
        t_sleep = t_prev
        while not self._stop.wait(period_s):
            try:
                now = time.monotonic_ns()
                c_now = self._main_cpu()
                self.sample_once(now - t_prev, max(0, c_now - c_prev), skip,
                                 max(0, now - t_sleep - self.period_ns))
                t_prev, c_prev = now, c_now
                if now >= next_child:
                    self.poll_children()
                    next_child = now + CHILD_POLL_NS
                if now >= next_skip:
                    skip = self._skip_idents()
                    next_skip = now + CHILD_POLL_NS
                if self.writer is not None and now >= next_funcs:
                    self.writer.set_extra("stack_funcs", self.funcs_table())
                    next_funcs = now + FUNCS_REFRESH_NS
            except Exception:  # noqa: BLE001  a sampler error must never stop the product
                v[tot["errors"]] += 1
            v[tot["sampler_cpu_ns"]] = time.thread_time_ns()
            v[tot["proc_cpu_ns"]] = time.process_time_ns()
            t_sleep = time.monotonic_ns()

    def start(self):
        self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop the thread and publish the final children poll and function table."""
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        self._thread = None
        try:
            self.poll_children()
        except OSError:
            pass
        if self.writer is not None:
            self.writer.set_extra("stack_funcs", self.funcs_table())
