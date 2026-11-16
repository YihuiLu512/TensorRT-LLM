# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX minimal driver without HTTP (design 7 `nx_driver.py`; E2b, E7).

    python -m capstubs.nx_driver standalone ...   # E2b part 1: no IPC, no consumer
    python -m capstubs.nx_driver proxy ...        # E2b part 2 / E7: proxy <-> ZMQ <-> NullWorker
    python -m capstubs.nx_driver drain ...        # (internal) one attached result lane, K >= 2

Arrivals come from a capkit tier table + arrival schedule (capcommon.schedule; one row = one
request, max_tokens = its tier's max_tokens, prompt = --prompt-len token ids), sent at their planned
times from the launch origin (open loop: a late row is sent at once and its send lag recorded).

standalone: one process, one NullExecutor (product adapter: real executor-request conversion and
create_response), self-drained by its own step thread after each step; arrivals are built and
enqueued at the start of each round by the same thread, outside the timed step (design E2 card:
"脱离 IPC、没有消费端、没有别的线程抢锁").

proxy: GenerationExecutorProxy(worker_kwargs, worker_cls=NullWorker) -- the product proxy, ZMQ,
the product worker with a NullExecutor engine. The result side is a drain chosen by --consumer:
  raw          a subclass whose dispatch thread only recv()s the ZMQ frames and counts messages and
               bytes (no HMAC check, no unpickling; design E7 "原始字节的排空端");
  deserialize  the same drain, also HMAC-checking and unpickling every --sample-every'th message to
               count responses and finals (the "反序列化" reader of E7 isolation check (b));
  product      the product dispatch thread itself (proxy.submit + GenerationResult routing).
--lanes K >= 2 starts the proxy with K + 1 serving lanes (multi-frontend IPC: ipc:// PUSH/PULL)
and K drain subprocesses, each an attached GenerationExecutorFrontendProxy subclass that submits its
share of the rows (req_id % K) and drains its own lane; the launcher's lane 0 only starts the
worker. --lanes 1 drains lane 0 of a single-frontend proxy (tcp PAIR).

Output (--out-dir): nxd.<role>.ready.json once the worker is ready; per lane / process a summary
(nxd.<lane>.summary.json: per tier planned / sent / send lag / messages / bytes / responses
received by receive time) and binary records (sends, recvs). The NullExecutor writes its own
snx.<pid>.* files (config out_dir). After the table ends the driver writes its summaries and holds
until SIGTERM (orchestrator stop), then shuts the proxy down.
"""

import argparse
import json
import os
import pickle  # nosec B403 -- reads the product's own result lane
import signal
import struct
import subprocess
import sys
import threading
import time

from capcommon.exitcodes import ConfigError

from .null_executor import (ENV_VAR, NS, RecordFile, SCHEMA_CONFIG, SnxConfig, tier_shape)

SCHEMA_SUMMARY = "capkit.nxd_summary/1"
SCHEMA_READY = "capkit.nxd_ready/1"
SEND_REC = struct.Struct("<qqqqh")
SEND_FIELDS = ("req_id", "client_id", "planned_ns", "sent_ns", "tier_seq")
RECV_REC = struct.Struct("<qii")
RECV_FIELDS = ("recv_ns", "bytes", "responses")
CONSUMERS = ("raw", "deserialize", "product")


def add_snx_args(p):
    g = p.add_argument_group("S-NX config")
    g.add_argument("--config-json")
    g.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    g.add_argument("--snx-out-dir", help="NullExecutor out_dir")


def add_load_args(p):
    g = p.add_argument_group("arrivals")
    g.add_argument("--tier-table", required=True)
    g.add_argument("--schedule", required=True)
    g.add_argument("--origin-ns", type=int, required=True)
    g.add_argument("--prompt-len", type=int, default=8)
    g.add_argument("--prompt-first-id", type=int, default=1000)
    g.add_argument("--end-id", type=int, default=2,
                   help="SamplingParams end_id (requests are ignore_eos)")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--tail-s", type=float, default=5.0,
                   help="after the table ends: wait this long, then write the summaries")
    g.add_argument("--no-hold", action="store_true",
                   help="exit after the summaries instead of waiting for SIGTERM")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.nx_driver", allow_abbrev=False,
                                description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("standalone")
    add_snx_args(s)
    add_load_args(s)
    s.add_argument("--stream-interval", type=int, default=0)
    x = sub.add_parser("proxy")
    add_snx_args(x)
    add_load_args(x)
    x.add_argument("--model", required=True, help="model dir (tokenizer copy; no weights)")
    x.add_argument("--max-seq-len", type=int, default=8192)
    x.add_argument("--consumer", choices=CONSUMERS, default="raw")
    x.add_argument("--sample-every", type=int, default=0,
                   help="deserialize: parse every n-th message (0 = every message)")
    x.add_argument("--lanes", type=int, default=1)
    x.add_argument("--drain-cpus", default="",
                   help="K comma-separated cpu lists (taskset) for the drain subprocesses")
    x.add_argument("--python", default=sys.executable)
    d = sub.add_parser("drain")
    add_load_args(d)
    d.add_argument("--attach-info", required=True)
    d.add_argument("--frontend-id", type=int, required=True)
    d.add_argument("--lanes", type=int, required=True)
    d.add_argument("--consumer", choices=CONSUMERS, default="raw")
    d.add_argument("--sample-every", type=int, default=0)
    return p


def snx_config(a):
    doc = {"schema": SCHEMA_CONFIG}
    if a.config_json:
        with open(a.config_json, "r", encoding="utf-8") as f:
            doc.update(json.load(f))
    for kv in a.set:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise ConfigError("--set needs KEY=JSON, got %r" % kv)
        try:
            doc[k] = json.loads(v)
        except ValueError:
            doc[k] = v
    doc["tier_table"] = a.tier_table
    doc["origin_ns"] = a.origin_ns
    if a.snx_out_dir:
        doc["out_dir"] = a.snx_out_dir
    return SnxConfig.from_doc(doc)


def write_json(path, doc):
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


class Plan:
    """The rows of the arrival schedule this process sends (req_id % lanes == share), with their
    absolute planned times and max_tokens."""

    def __init__(self, a, share=0, n_shares=1):
        from capcommon.schedule import load_schedule  # noqa: PLC0415
        from capcommon.tiers import TierTable  # noqa: PLC0415
        self.tt = TierTable.load(a.tier_table)
        self.origin = a.origin_ns
        ids, rel, seqs, meta = load_schedule(a.schedule, share, n_shares, self.tt.sha256)
        self.rows = [(int(i), self.origin + int(r), int(s)) for i, r, s in zip(ids, rel, seqs)]
        self.max_tokens = {t.seq: t.max_tokens for t in self.tt.tiers}
        self.end_abs = self.origin + self.tt.end_ns
        self.prompt = list(range(a.prompt_first_id, a.prompt_first_id + a.prompt_len))
        self.schedule_meta = meta

    def tier_seq_at(self, t_abs):
        from capcommon.tiers import POST_SEQ, PRE_SEQ  # noqa: PLC0415
        rel = t_abs - self.origin
        if rel < 0:
            return PRE_SEQ
        s = self.tt.tier_seq_at(rel)
        return POST_SEQ if s is None else s


class LaneStats:
    """Send / receive accounting of one lane (one process), per tier by event time."""

    def __init__(self, plan, out_dir, name, consumer, sample_every):
        self.plan = plan
        self.name = name
        self.consumer = consumer
        self.sample_every = sample_every
        self.lock = threading.Lock()
        self.tiers = {}
        self.sends = RecordFile(os.path.join(out_dir, "nxd.%s.sends.bin" % name), SEND_REC,
                                1 << 16)
        self.recvs = RecordFile(os.path.join(out_dir, "nxd.%s.recvs.bin" % name), RECV_REC,
                                1 << 16)
        self.errors = []
        self.sentinel = False

    def _t(self, seq):
        t = self.tiers.get(seq)
        if t is None:
            t = self.tiers[seq] = {"planned": 0, "sent": 0, "send_lag_ns_max": 0,
                                   "send_lag_ns_sum": 0, "msgs": 0, "bytes": 0,
                                   "parsed_msgs": 0, "responses": 0, "finals": 0,
                                   "error_responses": 0}
        return t

    def planned(self, rows):
        for _i, _p, seq in rows:
            self._t(seq)["planned"] += 1

    def sent(self, req_id, planned_ns, sent_ns, seq, client_id=-1):
        lag = sent_ns - planned_ns
        with self.lock:
            t = self._t(seq)
            t["sent"] += 1
            t["send_lag_ns_sum"] += lag
            if lag > t["send_lag_ns_max"]:
                t["send_lag_ns_max"] = lag
        self.sends.add(req_id, client_id, planned_ns, sent_ns, seq)

    def received(self, now, nbytes, responses=-1, finals=0, errors=0):
        seq = self.plan.tier_seq_at(now)
        with self.lock:
            t = self._t(seq)
            t["msgs"] += 1
            t["bytes"] += nbytes
            if responses >= 0:
                t["parsed_msgs"] += 1
                t["responses"] += responses
                t["finals"] += finals
                t["error_responses"] += errors
        self.recvs.add(now, nbytes, responses)

    def summary(self, extra=None):
        self.sends.flush()
        self.recvs.flush()
        with self.lock:
            tiers = {str(k): dict(v) for k, v in sorted(self.tiers.items())}
        doc = {"schema": SCHEMA_SUMMARY, "lane": self.name, "pid": os.getpid(),
               "consumer": self.consumer, "sample_every": self.sample_every,
               "tiers": tiers, "errors": self.errors[:20], "sentinel_seen": self.sentinel,
               "rows": len(self.plan.rows), "origin_ns": self.plan.origin,
               "tier_table_sha256": self.plan.tt.sha256,
               "records": {"sends": list(SEND_FIELDS), "recvs": list(RECV_FIELDS),
                           "sends_format": SEND_REC.format, "recvs_format": RECV_REC.format}}
        doc.update(extra or {})
        return doc


def parse_batch(obj):
    """(responses, finals, errors) of one unpickled result-lane message (a list of responses;
    None is the worker's shutdown sentinel)."""
    if obj is None:
        return None
    items = obj if isinstance(obj, list) else [obj]
    n = fin = err = 0
    for r in items:
        if r is None:
            continue
        n += 1
        if getattr(r, "error_msg", None) is not None or type(r).__name__ == "ErrorResponse":
            err += 1
            fin += 1
            continue
        res = getattr(r, "result", None)
        if res is not None and getattr(res, "is_final", False):
            fin += 1
    return n, fin, err


def drain_loop(zq, stats, consumer, sample_every, stop):
    """The raw / deserialize result drain: recv() every frame of the lane's ZMQ socket."""
    zq.setup_lazily()
    sock = zq.socket
    i = 0
    clock = time.monotonic_ns
    while not stop.is_set():
        if not sock.poll(200):
            continue
        data = sock.recv()
        now = clock()
        i += 1
        if consumer == "deserialize" and (sample_every <= 1 or i % sample_every == 0):
            try:
                obj = zq._parse_data(data) if zq.use_hmac_encryption else \
                    pickle.loads(data)  # nosec B301 -- the product's own lane
            except Exception as e:  # noqa: BLE001 -- counted, not fatal
                stats.errors.append(repr(e))
                stats.received(now, len(data))
                continue
            got = parse_batch(obj)
            if got is None:
                stats.sentinel = True
                return
            stats.received(now, len(data), *got)
        else:
            if len(data) < 64 and pickle.loads(  # nosec B301
                    data[:-32] if zq.use_hmac_encryption else data) is None:
                stats.sentinel = True
                return
            stats.received(now, len(data))


def make_request(plan, req_id, seq, end_id):
    from tensorrt_llm.executor.request import GenerationRequest  # noqa: PLC0415
    from tensorrt_llm.sampling_params import SamplingParams  # noqa: PLC0415
    sp = SamplingParams(max_tokens=plan.max_tokens[seq], ignore_eos=True, detokenize=False,
                        end_id=end_id, pad_id=end_id)
    return GenerationRequest(list(plan.prompt), sampling_params=sp, streaming=True)


def send_loop(plan, rows, submit, stats, stop, end_id):
    """Open-loop sender: each row at its planned time (late rows at once, lag recorded)."""
    clock = time.monotonic_ns
    for req_id, planned, seq in rows:
        while not stop.is_set():
            now = clock()
            if now >= planned:
                break
            time.sleep(min((planned - now) / NS, 0.2))
        if stop.is_set():
            return
        try:
            cid = submit(make_request(plan, req_id, seq, end_id))
        except Exception as e:  # noqa: BLE001 -- recorded; the run continues
            stats.errors.append("submit %d: %r" % (req_id, e))
            continue
        stats.sent(req_id, planned, clock(), seq, -1 if cid is None else cid)


def raw_class(base, stats, consumer, sample_every, stop):
    """A subclass of a product proxy class whose result dispatch is the capkit drain."""

    class _RawDrain(base):
        def dispatch_result_task(self):
            drain_loop(self.result_queue.queue, stats, consumer, sample_every, stop)
            return False

    _RawDrain.__name__ = "Raw" + base.__name__
    return _RawDrain


def submit_raw(proxy):
    """Send a request the way proxy.submit does, without a GenerationResult (the raw / deserialize
    drains never route responses to one)."""
    def submit(req):
        req.set_id(proxy._get_next_client_id())
        proxy._send_request(req)
        return req.id
    return submit


class Holder:
    """SIGTERM / SIGINT -> stop event."""

    def __init__(self):
        self.stop = threading.Event()
        self.signals = []
        signal.signal(signal.SIGTERM, self._on)
        signal.signal(signal.SIGINT, self._on)

    def _on(self, signum, _frame):
        self.signals.append(signum)
        self.stop.set()

    def wait_until(self, t_abs):
        while not self.stop.is_set():
            now = time.monotonic_ns()
            if now >= t_abs:
                return True
            self.stop.wait(min((t_abs - now) / NS, 0.5))
        return False


def _finish(a, holder, stats_list, extra):
    if holder.wait_until(stats_list[0].plan.end_abs + int(a.tail_s * NS)):
        pass
    for st in stats_list:
        write_json(os.path.join(a.out_dir, "nxd.%s.summary.json" % st.name), st.summary(extra))
    if not a.no_hold:
        while not holder.stop.is_set():
            holder.stop.wait(0.5)


def run_standalone(a, holder):
    from .snx_product import ProductAdapter  # noqa: PLC0415
    from .null_executor import NullExecutor  # noqa: PLC0415
    cfg = snx_config(a)
    plan = Plan(a)
    stats = LaneStats(plan, a.out_dir, "standalone", "self_drain", 0)
    stats.planned(plan.rows)
    from tensorrt_llm.bindings import executor as tllm  # noqa: PLC0415
    rows = list(plan.rows)
    pos = [0]
    ex = []

    def arrivals(now):
        while pos[0] < len(rows) and rows[pos[0]][1] <= now and not ex[0].is_shutdown:
            req_id, planned, seq = rows[pos[0]]
            pos[0] += 1
            r = tllm.Request(input_token_ids=plan.prompt, max_tokens=plan.max_tokens[seq],
                             streaming=True, end_id=-1, client_id=req_id)
            ex[0].enqueue_request(r)
            stats.sent(req_id, planned, time.monotonic_ns(), seq, req_id)

    ex.append(NullExecutor(cfg, ProductAdapter(), tier_table=plan.tt,
                           llm_stream_interval=a.stream_interval or None, self_drain=True,
                           arrival_hook=arrivals, start=False))
    write_json(os.path.join(a.out_dir, "nxd.standalone.ready.json"),
               {"schema": SCHEMA_READY, "pid": os.getpid(), "mode": "standalone"})
    late = time.monotonic_ns() - a.origin_ns
    ex[0].start()
    _finish(a, holder, [stats], {"mode": "standalone", "ready_after_origin_ns": late})
    ex[0].shutdown()
    return 0


def _lane_submit(proxy, consumer):
    if consumer == "product":
        def submit(req):
            proxy.submit(req)  # the GenerationResult is dropped
            return req.id
        return submit
    proxy._start_dispatch_threads()
    return submit_raw(proxy)


def start_proxy(a, plan, stats, stop):
    """The product proxy with a NullWorker (SNX config into the environment first). Returns
    (proxy, extra) where extra describes it; lanes >= 2 starts K + 1 serving lanes."""
    from pathlib import Path  # noqa: PLC0415
    cfg = snx_config(a)
    _ks, s = tier_shape(plan.tt, cfg)
    os.environ[ENV_VAR] = json.dumps(cfg.to_doc(), sort_keys=True)
    from tensorrt_llm.executor.postproc_worker import PostprocWorkerConfig  # noqa: PLC0415
    from tensorrt_llm.executor.proxy import GenerationExecutorProxy  # noqa: PLC0415
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs  # noqa: PLC0415

    from .null_worker import NullWorker  # noqa: PLC0415
    from .serve_nx import patch_cuda_stream_if_no_cuda  # noqa: PLC0415
    cuda_patched = patch_cuda_stream_if_no_cuda()
    lanes = a.lanes
    if lanes < 1:
        raise ConfigError("--lanes must be >= 1")
    llm_args = TorchLlmArgs(model=a.model, max_seq_len=a.max_seq_len, stream_interval=s,
                            num_serve_frontends=lanes + 1 if lanes >= 2 else 1)
    base = GenerationExecutorProxy
    cls = base if a.consumer == "product" or lanes >= 2 else \
        raw_class(base, stats, a.consumer, a.sample_every, stop)
    t0 = time.monotonic_ns()
    proxy = cls({"hf_model_dir": Path(a.model), "tokenizer": None, "llm_args": llm_args},
                model_world_size=1, worker_cls=NullWorker,
                postproc_worker_config=PostprocWorkerConfig(), is_llm_executor=True)
    extra = {"mode": "proxy", "lanes": lanes, "consumer": a.consumer,
             "proxy_class": cls.__name__, "proxy_start_s": (time.monotonic_ns() - t0) / NS,
             "cuda_stream_patched": cuda_patched, "stream_interval": s,
             "ready_after_origin_ns": time.monotonic_ns() - a.origin_ns}
    return proxy, extra


def spawn_drains(a, proxy, out_dir, stopped):
    """K attached drain subprocesses (lanes >= 2), each sending its share of the rows; returns
    the Popen list once every one wrote its ready file. stopped(): poll for an abort."""
    info_path = os.path.join(out_dir, "nxd.attach_info.json")
    write_json(info_path, proxy.multi_frontend_attach_info())
    cpus = a.drain_cpus.split(";") if a.drain_cpus else []
    children = []
    for fid in range(1, a.lanes + 1):
        argv = [a.python, "-m", "capstubs.nx_driver", "drain",
                "--attach-info", info_path, "--frontend-id", str(fid),
                "--lanes", str(a.lanes), "--consumer", a.consumer,
                "--sample-every", str(a.sample_every), "--tier-table", a.tier_table,
                "--schedule", a.schedule, "--origin-ns", str(a.origin_ns),
                "--prompt-len", str(a.prompt_len),
                "--prompt-first-id", str(a.prompt_first_id), "--end-id", str(a.end_id),
                "--out-dir", out_dir, "--tail-s", str(a.tail_s)]
        if len(cpus) >= fid and cpus[fid - 1]:
            argv = ["taskset", "-c", cpus[fid - 1]] + argv
        children.append(subprocess.Popen(argv))  # nosec B603
    for fid in range(1, a.lanes + 1):
        p = os.path.join(out_dir, "nxd.lane%d.ready.json" % fid)
        while not os.path.exists(p):
            time.sleep(0.2)
            if stopped() or any(c.poll() is not None for c in children):
                stop_children(children)
                raise RuntimeError("drain lane %d did not get ready" % fid)
    return children


def stop_children(children, timeout=30):
    for c in children:
        if c.poll() is None:
            c.send_signal(signal.SIGTERM)
    for c in children:
        try:
            c.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            c.kill()


def run_proxy(a, holder):
    plan = Plan(a)
    stop = threading.Event()
    stats = LaneStats(plan, a.out_dir, "lane0", a.consumer, a.sample_every)
    proxy, extra = start_proxy(a, plan, stats, stop)
    children = []
    try:
        if a.lanes >= 2:
            children = spawn_drains(a, proxy, a.out_dir, holder.stop.is_set)
            lane_stats = []
        else:
            submit = _lane_submit(proxy, a.consumer)
            stats.planned(plan.rows)
            threading.Thread(target=send_loop, name="nxd_send", daemon=True,
                             args=(plan, plan.rows, submit, stats, stop, a.end_id)).start()
            lane_stats = [stats]
        write_json(os.path.join(a.out_dir, "nxd.ready.json"),
                   {"schema": SCHEMA_READY, "pid": os.getpid(),
                    "children": [c.pid for c in children], **extra})
        if lane_stats:
            _finish(a, holder, lane_stats, extra)
        else:  # the drains write their own summaries; wait as _finish does
            holder.wait_until(plan.end_abs + int(a.tail_s * NS))
            while not a.no_hold and not holder.stop.is_set():
                holder.stop.wait(0.5)
    finally:
        stop.set()
        stop_children(children)
        proxy.shutdown()
    return 0


def run_drain(a, holder):
    from tensorrt_llm.executor.postproc_worker import PostprocWorkerConfig  # noqa: PLC0415
    from tensorrt_llm.executor.proxy import GenerationExecutorFrontendProxy  # noqa: PLC0415
    with open(a.attach_info, "r", encoding="utf-8") as f:
        info = json.load(f)
    fid = a.frontend_id
    plan = Plan(a, share=fid - 1, n_shares=a.lanes)
    stop = threading.Event()
    stats = LaneStats(plan, a.out_dir, "lane%d" % fid, a.consumer, a.sample_every)
    base = GenerationExecutorFrontendProxy
    cls = base if a.consumer == "product" else \
        raw_class(base, stats, a.consumer, a.sample_every, stop)
    proxy = cls(info, frontend_id=fid, postproc_worker_config=PostprocWorkerConfig(),
                is_llm_executor=True)
    try:
        submit = _lane_submit(proxy, a.consumer)
        stats.planned(plan.rows)
        write_json(os.path.join(a.out_dir, "nxd.lane%d.ready.json" % fid),
                   {"schema": SCHEMA_READY, "pid": os.getpid(), "mode": "drain",
                    "frontend_id": fid, "proxy_class": cls.__name__})
        threading.Thread(target=send_loop, name="nxd_send", daemon=True,
                         args=(plan, plan.rows, submit, stats, stop, a.end_id)).start()
        _finish(a, holder, [stats], {"mode": "drain", "frontend_id": fid})
    finally:
        stop.set()
        proxy.shutdown()
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    holder = Holder()
    try:
        if a.mode == "standalone":
            return run_standalone(a, holder)
        if a.mode == "proxy":
            return run_proxy(a, holder)
        return run_drain(a, holder)
    except ConfigError as e:
        print("nx_driver: %s" % e, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
