# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.nx_role: capstubs.nx_driver as a capkit tool (caplaunch TOOL_TABLE entry), so an
E2b / E7 launch has a role that writes snapshots, per-tier summaries and a ready file.

    python -m capstubs.nx_role <capkit common arguments> --mode standalone|proxy ...

- standalone (E2b part 1): one NullExecutor in this process, arrivals enqueued by its step thread,
  self-drained (nx_driver run_standalone's engine).
- proxy (E2b part 2, E7): the product GenerationExecutorProxy with a NullWorker; the result side
  is nx_driver's drain (--consumer raw | deserialize | product); --lanes K >= 2 starts K attached
  drain subprocesses (python -m capstubs.nx_driver drain), each sending its share of the rows.

Snapshot slots (drain gate): backlog.inflight = requests sent - requests the engine finished
(engine side: the NullExecutor's own counters, read in place in standalone mode and from the
worker's snx.<pid>.live record in proxy mode; K >= 2: requests the engine took in - finished,
the drains' sends being in other processes), backlog.engine_open = active + queued requests in
the engine, backlog.pending_keys = request keys with responses not yet taken by
await_responses. They are polled every --poll-ms on the loop thread.

Per-tier summaries (written after the table end, like capgrpc.load): counts planned / sent / ok /
failed / not_sent by the planned tier of each request; ok = the engine finished it (snx reqs
records, matched by client id), failed = sent but not finished when the role stopped (reason
not_finished). Messages and bytes drained are reported by receive-time tier under `zmq` (raw
drains cannot tell which request a message belongs to). The nx_driver files (nxd.*) go to
--nxd-dir, the NullExecutor's (snx.*) to --snx-out-dir.
"""

import argparse
import asyncio
import glob
import os
import sys
import threading
import time
import types

from capcommon.cli import add_common_args, common_from_args
from capcommon.exitcodes import OK, STOPPED_EARLY, CapkitError, ConfigError

from . import nx_driver as nxd
from .null_executor import LiveReader, read_records

TOOL = "capstubs.nx_role"
SLOTS = {
    "backlog.inflight": {"kind": "gauge", "site": None, "group": "backlog", "per": "request"},
    "backlog.engine_open": {"kind": "gauge", "site": None, "group": "backlog",
                            "per": "request"},
    "backlog.pending_keys": {"kind": "gauge", "site": None, "group": "backlog",
                             "per": "request"},
    "self.sent": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.finished": {"kind": "counter", "site": None, "group": "self", "per": "request"},
    "self.msgs": {"kind": "counter", "site": None, "group": "self", "per": "message"},
    "self.live_age_ns": {"kind": "gauge", "site": None, "group": "self", "per": "none"},
    "lag.loop": {"kind": "hist", "site": None, "group": "lag", "per": "sample"},
}
MODES = ("standalone", "proxy")


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.nx_role", allow_abbrev=False,
                                description=__doc__.split("\n")[0])
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=False)
    g = p.add_argument_group("capstubs.nx_role")
    g.add_argument("--mode", required=True, choices=MODES)
    g.add_argument("--schedule", required=True)
    g.add_argument("--nxd-dir", required=True, help="nx_driver files (nxd.*)")
    g.add_argument("--snx-out-dir", required=True, help="NullExecutor files (snx.*)")
    g.add_argument("--config-json")
    g.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    g.add_argument("--prompt-len", type=int, default=8)
    g.add_argument("--prompt-first-id", type=int, default=1000)
    g.add_argument("--end-id", type=int, default=2)
    g.add_argument("--tail-s", type=float, default=5.0,
                   help="after the table ends: wait this long before stopping the engine")
    g.add_argument("--lag-period-us", type=int, required=True)
    g.add_argument("--poll-ms", type=int, default=10, help="engine counters -> slots period")
    g.add_argument("--stream-interval", type=int, default=0, help="standalone only")
    g.add_argument("--model", help="proxy: model dir (tokenizer copy; no weights)")
    g.add_argument("--max-seq-len", type=int, default=8192)
    g.add_argument("--consumer", choices=nxd.CONSUMERS, default="raw")
    g.add_argument("--sample-every", type=int, default=0)
    g.add_argument("--lanes", type=int, default=1)
    g.add_argument("--drain-cpus", default="")
    g.add_argument("--python", default=sys.executable)
    return p


def driver_args(args, common):
    """The nx_driver namespace (Plan, snx_config, start_proxy, spawn_drains read these)."""
    return types.SimpleNamespace(
        tier_table=common.tier_table_path, origin_ns=common.origin_ns, schedule=args.schedule,
        prompt_len=args.prompt_len, prompt_first_id=args.prompt_first_id, end_id=args.end_id,
        out_dir=args.nxd_dir, tail_s=args.tail_s, config_json=args.config_json, set=args.set,
        snx_out_dir=args.snx_out_dir, model=args.model, max_seq_len=args.max_seq_len,
        consumer=args.consumer, sample_every=args.sample_every, lanes=args.lanes,
        drain_cpus=args.drain_cpus, python=args.python, stream_interval=args.stream_interval)


def validate_args(args):
    if args.mode == "proxy" and not args.model:
        raise ConfigError("--mode proxy needs --model")
    if args.mode == "standalone" and (args.lanes != 1 or args.consumer != "raw"):
        raise ConfigError("--lanes / --consumer are proxy options")
    if args.lanes < 1 or args.poll_ms < 1 or args.tail_s < 0:
        raise ConfigError("--lanes and --poll-ms must be >= 1, --tail-s >= 0")


def _engine_files(snx_dir, kind, exclude_pid=None):
    out = []
    for p in sorted(glob.glob(os.path.join(snx_dir, "snx.*.%s" % kind))):
        try:
            pid = int(os.path.basename(p).split(".")[1])
        except ValueError:
            continue
        if pid != exclude_pid:
            out.append((pid, p))
    return out


def standalone_parts():
    """(adapter, make_request(plan, req_id, seq)) of standalone mode: the product adapter and a
    product executor Request with client_id = the schedule's req_id (tests replace this)."""
    from tensorrt_llm.bindings import executor as tllm  # noqa: PLC0415

    from .snx_product import ProductAdapter  # noqa: PLC0415

    def make_request(plan, req_id, seq):
        return tllm.Request(input_token_ids=plan.prompt, max_tokens=plan.max_tokens[seq],
                            streaming=True, end_id=-1, client_id=req_id)
    return ProductAdapter(), make_request


class NxRole:
    """One nx_role process: the engine (in process or behind the proxy), the senders, the slots
    and the outputs."""

    def __init__(self, args, common, stopper):
        from capgrpc.common import ToolRuntime  # noqa: PLC0415
        self.args, self.common, self.stopper = args, common, stopper
        self.da = driver_args(args, common)
        os.makedirs(args.nxd_dir, exist_ok=True)
        os.makedirs(args.snx_out_dir, exist_ok=True)
        self.rt = ToolRuntime(common, TOOL, SLOTS, (), 0, 1, list(common.cpus),
                              args.lag_period_us)
        self.pos = {k: self.rt._pos[k][0] for k in SLOTS if k != "lag.loop"}
        self.plan = nxd.Plan(self.da)
        self.stop = threading.Event()
        self.stats = nxd.LaneStats(self.plan, args.nxd_dir,
                                   "standalone" if args.mode == "standalone" else "lane0",
                                   "self_drain" if args.mode == "standalone" else args.consumer,
                                   args.sample_every)
        self.ex = self.proxy = self.live = None
        self.children = []
        self.engine_pid = None
        self.info = {"mode": args.mode, "lanes": args.lanes, "consumer": self.stats.consumer,
                     "nxd_dir": args.nxd_dir, "snx_out_dir": args.snx_out_dir,
                     "rows": len(self.plan.rows)}

    # -- engine --------------------------------------------------------------------------------
    def build(self):
        """Blocking: construct the engine (and the drains) before the event loop runs."""
        if self.args.mode == "standalone":
            self._build_standalone()
        else:
            self.proxy, extra = nxd.start_proxy(self.da, self.plan, self.stats, self.stop)
            self.info.update(extra)
            lives = _engine_files(self.args.snx_out_dir, "live")
            if len(lives) != 1:
                raise ConfigError("expected one worker live file in %s, found %d"
                                  % (self.args.snx_out_dir, len(lives)))
            self.engine_pid, path = lives[0]
            self.live = LiveReader(path)
            if self.args.lanes >= 2:
                self.children = nxd.spawn_drains(self.da, self.proxy, self.args.nxd_dir,
                                                 lambda: self.stopper.requested)
                self.info["drain_pids"] = [c.pid for c in self.children]
        self.info["engine_pid"] = self.engine_pid

    def _build_standalone(self):
        from .null_executor import NullExecutor  # noqa: PLC0415
        adapter, make_request = standalone_parts()
        plan, stats = self.plan, self.stats
        rows, pos = list(plan.rows), [0]

        def arrivals(now):
            ex = self.ex
            while pos[0] < len(rows) and rows[pos[0]][1] <= now and not ex.is_shutdown:
                req_id, planned, seq = rows[pos[0]]
                pos[0] += 1
                ex.enqueue_request(make_request(plan, req_id, seq))
                stats.sent(req_id, planned, time.monotonic_ns(), seq, req_id)

        self.ex = NullExecutor(nxd.snx_config(self.da), adapter, tier_table=plan.tt,
                               llm_stream_interval=self.args.stream_interval or None,
                               self_drain=True, arrival_hook=arrivals, start=False)
        self.engine_pid = os.getpid()

    def start_traffic(self):
        self.stats.planned(self.plan.rows)
        if self.ex is not None:
            self.ex.start()
        elif self.args.lanes == 1:
            submit = nxd._lane_submit(self.proxy, self.args.consumer)
            threading.Thread(target=nxd.send_loop, name="nxd_send", daemon=True,
                             args=(self.plan, self.plan.rows, submit, self.stats, self.stop,
                                   self.args.end_id)).start()

    def engine_counters(self):
        """LIVE_FIELDS-shaped dict of the engine's counters, or None (not readable yet)."""
        if self.ex is not None:
            ex = self.ex
            return {"t_ns": time.monotonic_ns(), "enqueued": ex._next_id - 1,
                    "taken": ex.counts["taken"], "finished": ex.counts["finished"],
                    "active": len(ex._active), "queued": len(ex._queue),
                    "pending_keys": len(ex.responses)}
        return self.live.read() if self.live is not None else None

    # -- slots ---------------------------------------------------------------------------------
    def poll(self):
        v, p = self.rt.registry.values, self.pos
        with self.stats.lock:
            sent = sum(t["sent"] for t in self.stats.tiers.values())
            msgs = sum(t["msgs"] for t in self.stats.tiers.values())
        c = self.engine_counters()
        if self.args.lanes >= 2:
            sent = c["enqueued"] if c else 0
        v[p["self.sent"]] = sent
        v[p["self.msgs"]] = msgs
        if c is None:
            v[p["backlog.inflight"]] = sent
            return
        v[p["backlog.inflight"]] = sent - c["finished"]
        v[p["backlog.engine_open"]] = c["active"] + c["queued"]
        v[p["backlog.pending_keys"]] = c["pending_keys"]
        v[p["self.finished"]] = c["finished"]
        v[p["self.live_age_ns"]] = time.monotonic_ns() - c["t_ns"]

    async def _poll_task(self):
        period = self.args.poll_ms / 1000.0
        while not self.stop.is_set():
            self.poll()
            await asyncio.sleep(period)

    # -- run -----------------------------------------------------------------------------------
    async def run(self, loop):
        rt = self.rt
        wake = asyncio.Event()
        self.stopper.attach_loop(loop)
        self.stopper.on_stop(lambda hard: wake.set())
        rt.start(loop, self.info)
        from capgrpc.common import write_inprocess_ready  # noqa: PLC0415
        write_inprocess_ready(rt, self.args.ready_file, self.info)
        self.start_traffic()
        poller = loop.create_task(self._poll_task())
        end_abs = self.common.origin_ns + rt.tt.end_ns
        until = end_abs + int(self.args.tail_s * nxd.NS)
        while not wake.is_set():
            now = time.monotonic_ns()
            if now >= until:
                break
            try:
                await asyncio.wait_for(wake.wait(), min((until - now) / nxd.NS, 1.0))
            except asyncio.TimeoutError:
                pass
        stopped_early = time.monotonic_ns() < end_abs
        self.poll()
        rt.close_tiers()
        self.stop.set()
        await poller
        await loop.run_in_executor(None, self.shutdown)
        rt.read_stop()
        self.write_outputs()
        rt.stop()
        return STOPPED_EARLY if stopped_early else OK

    def shutdown(self):
        nxd.stop_children(self.children)
        if self.ex is not None:
            self.ex.shutdown()
        if self.proxy is not None:
            self.proxy.shutdown()
        if self.live is not None:
            self.live.close()

    # -- outputs -------------------------------------------------------------------------------
    def _sends(self):
        """[(req_id, client_id, tier_seq)] of every send of this launch (own lane + drains)."""
        st = nxd.SEND_REC
        out = []
        for p in sorted(glob.glob(os.path.join(self.args.nxd_dir, "nxd.*.sends.bin"))):
            with open(p, "rb") as f:
                data = f.read()
            for i in range(len(data) // st.size):
                r = st.unpack_from(data, i * st.size)
                out.append((r[0], r[1], r[4]))
        return out

    def write_outputs(self):
        rt, a = self.rt, self.args
        extra = dict(self.info)
        extra["engine_counters_at_stop"] = self.engine_counters() if self.ex else None
        nxd.write_json(os.path.join(a.nxd_dir, "nxd.%s.summary.json" % self.stats.name),
                       self.stats.summary(extra))
        done, cancelled = set(), set()
        for _pid, path in _engine_files(a.snx_out_dir, "reqs.bin"):
            for r in read_records(path, "reqs"):
                (cancelled if r["cancelled"] else done).add(r["client_id"])
        planned, sent, ok, cancel = {}, {}, {}, {}
        for _i, _p, seq in self.plan.rows:
            planned[seq] = planned.get(seq, 0) + 1
        for _rid, cid, seq in self._sends():
            sent[seq] = sent.get(seq, 0) + 1
            if cid in done:
                ok[seq] = ok.get(seq, 0) + 1
            elif cid in cancelled:
                cancel[seq] = cancel.get(seq, 0) + 1
        from capcommon.hashing import sha256_file  # noqa: PLC0415
        sched_sha = sha256_file(a.schedule)
        with self.stats.lock:
            zmq = {k: {"msgs": t["msgs"], "bytes": t["bytes"], "parsed_msgs": t["parsed_msgs"],
                       "responses": t["responses"], "finals": t["finals"]}
                   for k, t in self.stats.tiers.items()}
        for seq in sorted(rt.end_marks):
            P, S, O, X = (d.get(seq, 0) for d in (planned, sent, ok, cancel))
            fails = []
            if X:
                fails.append({"reason": "cancelled", "count": X, "class_hint": "unclassified"})
            if S - O - X:
                fails.append({"reason": "not_finished", "count": S - O - X,
                              "class_hint": "unclassified"})
            payload = rt.tier_common(seq)
            payload.update({
                "schedule_sha256": sched_sha, "request_timeout_ns": rt.tt.request_timeout_ns,
                "mode": a.mode, "consumer": self.stats.consumer, "lanes": a.lanes,
                "counts": {"planned": P, "sent": S, "ok": O, "failed": S - O,
                           "not_sent": P - S},
                "failures": fails,
                "zmq": zmq.get(seq, {"msgs": 0, "bytes": 0, "parsed_msgs": 0, "responses": 0,
                                     "finals": 0}),
                "attribution": "counts: planned tier of the request; zmq: receive time"})
            rt.write_summary(seq, payload)


def main(argv=None):
    from capcommon.aio import run as aio_run  # noqa: PLC0415
    from capcommon.lifecycle import StopController  # noqa: PLC0415
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        validate_args(args)
        stopper = StopController()
        role = NxRole(args, common, stopper)
        role.build()
        if stopper.requested:
            role.shutdown()
            return STOPPED_EARLY
        return aio_run(role.run, common.loop)
    except CapkitError as e:
        print("%s: %s" % (TOOL, e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
