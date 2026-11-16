# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failure paths: a client-side fault inside a loader request is still counted exactly once, and
a tool whose port / ready file cannot be written stops its workers and exits non-zero."""

import os

import pytest

from capcommon.snapshot import list_snapshot_files, read_snapshot_file

from conftest import table_doc, total

METHOD = "/capkit.Test/Stream"

# Runs capgrpc.load in this process with a call wrapper whose second read() either raises a plain
# RuntimeError (FAULT=raise: a client-side fault, not an AioRpcError) or returns a str
# (FAULT=nonbytes: a non-bytes message after a bytes one).
_DRIVER = '''
import os
import sys
import capgrpc.loader as L

_MODE = os.environ["FAULT"]

_real = L.raw_stream_callable


class _Call:
    def __init__(self, call):
        self._c = call
        self._n = 0

    async def read(self):
        msg = await self._c.read()
        self._n += 1
        if self._n == 2:
            if _MODE == "raise":
                raise RuntimeError("injected client fault")
            return "not bytes"
        return msg

    def __getattr__(self, name):
        return getattr(self._c, name)


def _fake(channel, method):
    fn = _real(channel, method)
    return lambda req, timeout=None: _Call(fn(req, timeout=timeout))


L.raw_stream_callable = _fake
sys.exit(L.main(sys.argv[1:]))
'''


def _sink_in_process(launch):
    launch.server("sink", "capgrpc.nullgrpc", ["--workers", "1", "--in-process",
                                               "--msg-bytes", "8", "--pacing", "asap"])
    launch.wait_file(launch.tree.port_file("sink"), 15, "sink")


def _final_slots(launch, role, names):
    tot = dict.fromkeys(names, 0)
    for path in list_snapshot_files(launch.shm).values():
        snap = read_snapshot_file(path)
        if (snap.meta.get("identity") or {}).get("role") == role:
            for n in names:
                tot[n] += snap.values.get(n, 0)
    return tot


def _run_fault(launch, mode):
    launch.write_table(table_doc([("warmup", 500, 20.0, {}), ("drain", 500, 0.0, {})]))
    launch.set_origin(4.0)
    _sink_in_process(launch)
    (launch.tmp / "faultdriver.py").write_text(_DRIVER)
    args = (launch.common("load", cpus="3")
            + ["--target-port-file", launch.tree.port_file("sink"), "--method", METHOD,
               "--workers", "1", "--in-process", "--channels-per-worker", "1",
               "--schedule", launch.sched_path, "--request-format", "capkit",
               "--prompts", launch.prompts_path, "--msg-sample-mod", "1",
               "--max-records-per-worker", "1000"])
    os.environ["FAULT"] = mode
    try:
        launch.start("load", "faultdriver", args)
    finally:
        del os.environ["FAULT"]
    assert launch.wait_exit("load", 30) == 0, launch.stderr("load")
    assert launch.stop("sink") == 0
    ld = launch.summaries("load")
    sent = total(ld, "counts.sent")
    assert sent == total(ld, "counts.planned") > 0
    sk = launch.summaries("sink")
    req_types = {}
    for docs in sk.values():
        for d in docs:
            for t, c in d["request_types"].items():
                req_types[t] = req_types.get(t, 0) + c
    # per-tier summaries hold only their own tier's requests: the sum over tiers is the total
    assert req_types == {"bytes": total(sk, "counts.streams_accepted")} == {"bytes": sent}
    assert total(ld, "counts.ok") == 0
    assert total(ld, "counts.failed") == sent
    fails, types = {}, {}
    for docs in ld.values():
        for d in docs:
            for f in d["failures"]:
                fails[f["reason"]] = fails.get(f["reason"], 0) + f["count"]
            for t, c in d["recv_types"].items():
                types[t] = types.get(t, 0) + c
    rows, _ = launch.records("load", "requests")
    assert len(rows["req_id"]) == sent
    s = _final_slots(launch, "load", ["backlog.inflight", "backlog.active_streams",
                                      "self.sent", "self.failed"])
    assert s == {"backlog.inflight": 0, "backlog.active_streams": 0, "self.sent": sent,
                 "self.failed": sent}
    return sent, fails, types


def test_loader_client_fault_counted_once(launch):
    sent, fails, _types = _run_fault(launch, "raise")
    assert fails == {"client_error": sent}
    assert "injected client fault" in launch.stderr("load")


def test_loader_recv_types_counts_first_message_per_stream(launch):
    sent, fails, types = _run_fault(launch, "nonbytes")
    assert fails == {"protocol_error": sent}
    assert types == {"bytes": sent}


def _blocked_port_file(launch):
    blocker = launch.tmp / "not-a-dir"
    blocker.write_text("x")
    return str(blocker / "port.json")


def _with_port_file(args, path):
    args = list(args)
    args[args.index("--port-file") + 1] = path
    return args


@pytest.mark.parametrize("mode", ["supervisor", "in-process"])
def test_nullgrpc_port_file_failure_exits_nonzero(launch, mode):
    launch.write_table(table_doc([("warmup", 300, 10.0, {}), ("drain", 300, 0.0, {})]))
    launch.set_origin(4.0)
    extra = ["--workers", "1", "--msg-bytes", "8", "--pacing", "asap"]
    if mode == "in-process":
        extra.append("--in-process")
    args = _with_port_file(launch.common("sink") + ["--port-file", "", "--host", "127.0.0.1",
                                                    "--port", "0"] + extra,
                           _blocked_port_file(launch))
    launch.start("sink", "capgrpc.nullgrpc", args)
    rc = launch.wait_exit("sink", 30)
    assert rc not in (0, None), launch.stderr("sink")
    assert "port.json" in launch.stderr("sink")


def test_relay_port_file_failure_exits_nonzero(launch):
    launch.write_table(table_doc([("warmup", 300, 10.0, {}), ("drain", 300, 0.0, {})]))
    launch.set_origin(4.0)
    _sink_in_process(launch)
    args = _with_port_file(launch.common("relay")
                           + ["--port-file", "", "--host", "127.0.0.1", "--port", "0",
                              "--upstream-port-file", launch.tree.port_file("sink")],
                           _blocked_port_file(launch))
    launch.start("relay", "capselfcheck.relay", args)
    rc = launch.wait_exit("relay", 30)
    assert rc not in (0, None), launch.stderr("relay")
    assert "port.json" in launch.stderr("relay")
    assert launch.stop("sink") == 0
    assert os.path.exists(launch.tree.port_file("sink"))


# Runs capselfcheck.relay with a wrapped upstream callable: RELAY_FAULT=call -> creating the
# upstream call raises; RELAY_FAULT=read -> the second upstream read raises (neither is an
# AioRpcError).
_RELAY_DRIVER = '''
import os
import sys
import capselfcheck.relay as R

_MODE = os.environ["RELAY_FAULT"]
_real = R.raw_stream_callable


class _Call:
    def __init__(self, call):
        self._c = call
        self._n = 0

    async def read(self):
        msg = await self._c.read()
        self._n += 1
        if self._n == 2:
            raise RuntimeError("injected relay fault")
        return msg

    def __getattr__(self, name):
        return getattr(self._c, name)


def _fake(channel, method):
    fn = _real(channel, method)

    def make(req, timeout=None):
        if _MODE == "call":
            raise RuntimeError("injected relay call fault")
        return _Call(fn(req, timeout=timeout))
    return make


R.raw_stream_callable = _fake
sys.exit(R.main(sys.argv[1:]))
'''


@pytest.mark.parametrize("mode", ["call", "read"])
def test_relay_fault_counts_each_stream_once(launch, mode):
    launch.write_table(table_doc([("warmup", 500, 10.0, {}), ("drain", 500, 0.0, {})]))
    launch.set_origin(5.0)
    _sink_in_process(launch)
    (launch.tmp / "relaydriver.py").write_text(_RELAY_DRIVER)
    os.environ["RELAY_FAULT"] = mode
    try:
        launch.start("relay", "relaydriver",
                     launch.common("relay", cpus="1")
                     + ["--port-file", launch.tree.port_file("relay"), "--host", "127.0.0.1",
                        "--port", "0", "--upstream-port-file", launch.tree.port_file("sink")])
    finally:
        del os.environ["RELAY_FAULT"]
    launch.wait_file(launch.tree.port_file("relay"), 15, "relay")
    launch.start("load", "capgrpc.load",
                 launch.common("load", cpus="3")
                 + ["--target-port-file", launch.tree.port_file("relay"), "--method", METHOD,
                    "--workers", "1", "--in-process", "--channels-per-worker", "1",
                    "--schedule", launch.sched_path, "--request-format", "capkit",
                    "--prompts", launch.prompts_path, "--msg-sample-mod", "1",
                    "--max-records-per-worker", "1000"])
    assert launch.wait_exit("load", 30) == 0, launch.stderr("load")
    assert launch.stop("relay") == 0, launch.stderr("relay")
    assert launch.stop("sink") == 0
    ld = launch.summaries("load")
    sent = total(ld, "counts.sent")
    assert sent > 0 and total(ld, "counts.failed") == sent
    rl = launch.summaries("relay")
    assert total(rl, "counts.streams_accepted") == sent
    assert total(rl, "counts.streams_failed") == sent
    assert total(rl, "counts.streams_completed") + total(rl, "counts.streams_cancelled") == 0
    s = _final_slots(launch, "relay", ["backlog.active_streams", "backlog.queue_msgs",
                                       "self.accepted", "self.failed"])
    assert s == {"backlog.active_streams": 0, "backlog.queue_msgs": 0, "self.accepted": sent,
                 "self.failed": sent}
    ups = {}
    for docs in rl.values():
        for d in docs:
            for k, c in d["upstream_failures"].items():
                ups[k] = ups.get(k, 0) + c
    # per-tier summaries hold only their own tier's failures: the sum over tiers is the total
    assert ups == ({"grpc_UNKNOWN": sent} if mode == "read" else {})
