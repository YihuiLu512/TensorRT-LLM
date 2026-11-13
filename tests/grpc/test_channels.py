# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate channels => separate TCP connections (grpc.use_local_subchannel_pool=1), verified
from the process connection tables with psutil, against a deliberate shared-pool control that
must read differently (so the reading is shown to be able to change)."""

import psutil

from capcommon.jsonio import read_json_stamped
from capcommon.lifecycle import read_port_file

from conftest import per_tier, table_doc

METHOD = "/capkit.Test/Stream"


def established_to(pid, port, local=False):
    p = psutil.Process(pid)
    conns = p.net_connections(kind="tcp") if hasattr(p, "net_connections") \
        else p.connections(kind="tcp")
    n = 0
    for c in conns:
        if c.status != psutil.CONN_ESTABLISHED:
            continue
        addr = c.laddr if local else c.raddr
        if addr and addr.port == port:
            n += 1
    return n


def start_pair(launch, loader_extra, server_workers=1, loader_workers=1, channels=3):
    launch.write_table(table_doc([("warmup", 300, 10.0, {}), ("drain", 300, 0.0, {})]))
    launch.set_origin(4.0)
    srv = ["--workers", str(server_workers), "--msg-bytes", "16", "--pacing", "asap"]
    if server_workers == 1:
        srv.append("--in-process")
    launch.server("sink", "capgrpc.nullgrpc", srv,
                  cpus=",".join(str(i) for i in range(server_workers)))
    launch.wait_file(launch.tree.port_file("sink"), 15, "sink")
    _host, port, _ = read_port_file(launch.tree.port_file("sink"))
    cpus = ",".join(str(4 + i) for i in range(loader_workers))
    launch.start("load", "capgrpc.load",
                 launch.common("load", cpus=cpus)
                 + ["--target-port-file", launch.tree.port_file("sink"), "--method", METHOD,
                    "--workers", str(loader_workers), "--channels-per-worker", str(channels),
                    "--schedule", launch.sched_path, "--request-format", "capkit",
                    "--prompts", launch.prompts_path, "--msg-sample-mod", "1",
                    "--max-records-per-worker", "100"] + loader_extra)
    launch.wait_file(launch.tree.ready_file("load"), 15, "load")
    return port


def loader_pids(launch):
    _i, ready = read_json_stamped(launch.tree.ready_file("load"), "capkit.ready/1")
    kids = [c["pid"] for c in ready["children"]]
    return kids or [launch.procs["load"].pid]


def server_pids(launch):
    _i, ready = read_json_stamped(launch.tree.ready_file("sink"), "capkit.ready/1")
    kids = [c["pid"] for c in ready["children"]]
    return kids or [launch.procs["sink"].pid]


def finish(launch):
    assert launch.wait_exit("load", 20) == 0, launch.stderr("load")
    assert launch.stop("sink") == 0, launch.stderr("sink")
    ld, sk = launch.summaries("load"), launch.summaries("sink")
    assert per_tier(ld, "counts.ok") == per_tier(ld, "counts.planned")
    assert sum(per_tier(ld, "counts.ok").values()) > 0
    assert per_tier(ld, "msgs.msgs") == per_tier(sk, "counts.msgs_sent")


def test_local_pool_gives_one_connection_per_channel(launch):
    port = start_pair(launch, ["--in-process"], channels=3)
    client = sum(established_to(p, port) for p in loader_pids(launch))
    server = sum(established_to(p, port, local=True) for p in server_pids(launch))
    finish(launch)
    assert client == 3 and server == 3


def test_shared_pool_control_collapses_to_one_connection(launch):
    port = start_pair(launch, ["--in-process", "--shared-subchannel-pool"], channels=3)
    client = sum(established_to(p, port) for p in loader_pids(launch))
    server = sum(established_to(p, port, local=True) for p in server_pids(launch))
    finish(launch)
    assert client == 1 and server == 1   # differs from the 3 above: the reading can move


def test_multiprocess_loader_and_reuseport_sink(launch):
    port = start_pair(launch, [], server_workers=2, loader_workers=2, channels=2)
    lp, sp = loader_pids(launch), server_pids(launch)
    assert len(lp) == 2 and len(sp) == 2 and launch.procs["load"].pid not in lp
    per_child = [established_to(p, port) for p in lp]
    server = sum(established_to(p, port, local=True) for p in sp)
    finish(launch)
    assert per_child == [2, 2] and server == 4
    sk = launch.summaries("sink")
    assert {d["n_workers"] for docs in sk.values() for d in docs} == {2}
