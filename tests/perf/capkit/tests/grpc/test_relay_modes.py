# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Known-answer relay modes (E0 (b) scenarios 1-4): each mode must move the measured
per-message service time in the direction its construction dictates. Functional assertions only
(direction and order of magnitude on this VM); no number here is a performance result.

Setup: asap source (one request, M messages, sent as fast as gRPC takes them) -> relay(s) ->
in-process drain. The relay is the slowest hop by construction once delta is injected.
"""

import pytest

from conftest import table_doc

METHOD = "/capkit.Test/Stream"
M = 300            # messages in the one request
DELTA = 2_000_000  # 2 ms


def run_case(launch, relays, launch_inject=None, relay_cli_inject=None):
    """relays: list of role names, upstream first. Returns per-role summaries + drain records."""
    launch.write_table(table_doc([("warmup", 1000, 1.0, {"n": M}), ("drain", 2500, 0.0, {})],
                                 launch_inject=launch_inject))
    launch.set_origin(4.0)
    launch.server("source", "capgrpc.nullgrpc", ["--workers", "1", "--in-process", "--msg-bytes", "64",
                                         "--pacing", "asap"], cpus="0")
    launch.wait_file(launch.tree.port_file("source"), 15, "source")
    up = "source"
    for i, role in enumerate(relays):
        extra = ["--upstream-port-file", launch.tree.port_file(up)]
        for spec in (relay_cli_inject or {}).get(role, []):
            extra += ["--inject", spec]
        launch.server(role, "capselfcheck.relay", extra, cpus=str(1 + i))
        launch.wait_file(launch.tree.port_file(role), 15, role)
        up = role
    launch.start("drain", "capgrpc.load",
                 launch.common("drain", cpus="5")
                 + ["--target-port-file", launch.tree.port_file(up), "--method", METHOD,
                    "--workers", "1", "--in-process", "--channels-per-worker", "1",
                    "--schedule", launch.sched_path, "--request-format", "capkit",
                    "--prompts", launch.prompts_path, "--msg-sample-mod", "1",
                    "--max-records-per-worker", "100"])
    assert launch.wait_exit("drain", 30) == 0, launch.stderr("drain")
    for role in reversed(relays):
        assert launch.stop(role) == 0, launch.stderr(role)
    assert launch.stop("source") == 0, launch.stderr("source")
    reqs, _ = launch.records("drain", "requests")
    assert len(reqs["req_id"]) == 1 and reqs["status"] == [0] and reqs["frames"] == [M]
    per_msg = (reqs["done_rel_ns"][0] - reqs["first_rel_ns"][0]) / (M - 1)
    out = {"per_msg_ns": per_msg}
    for role in relays:
        summ = launch.summaries(role)
        docs = [d for ds in summ.values() for d in ds]
        svc_n = sum(d["event_time"]["service_ns"]["n"] for d in docs)
        svc_sum = sum(d["event_time"]["service_ns"]["sum"] for d in docs)
        inj = [i for d in docs for i in d["inject"] if i["site"] == "relay.msg"]
        out[role] = {
            "received": sum(d["counts"]["msgs_received"] for d in docs),
            "forwarded": sum(d["counts"]["msgs_forwarded"] for d in docs),
            "service_mean_ns": svc_sum / svc_n if svc_n else None,
            "loop_cpu_ns": sum(d["cpu"]["loop_thread_ns"] for d in docs),
            "proc_cpu_ns": sum(d["cpu"]["process_ns"] for d in docs),
            "hits": sum(i["hits_delta"] for i in inj),
            "applied": sum(i["applied_delta"] for i in inj),
            "applied_ns": sum(i["applied_ns_delta"] for i in inj),
            "aux_pending_end": [i["aux_pending_at_end"] for i in inj][-1] if inj else None,
            "loop_lag_q99": max((d["loop_lag_ns"]["q99_upper_ns"] or 0) for d in docs),
        }
        assert out[role]["received"] == out[role]["forwarded"] == M
        assert out[role]["hits"] == M
    print("CASE", relays, launch_inject, relay_cli_inject, out)
    return out


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    import sys
    from conftest import Launch
    L = Launch(tmp_path_factory.mktemp("base"), sys.executable)
    try:
        return run_case(L, ["relay"])
    finally:
        L.kill_all()


def test_mode1_busy_inline(launch, baseline):
    r = run_case(launch, ["relay"], relay_cli_inject={
        "relay": ["site=relay.msg,mode=busy,delta_ns=%d" % DELTA]})
    x = r["relay"]
    assert x["applied"] == M and x["applied_ns"] >= M * DELTA
    assert x["service_mean_ns"] >= 0.95 * DELTA > 10 * baseline["relay"]["service_mean_ns"]
    assert x["loop_cpu_ns"] >= 0.8 * M * DELTA          # the spin is on the loop thread
    assert r["per_msg_ns"] >= 0.9 * DELTA > 5 * baseline["per_msg_ns"]


def test_mode4_sleep_blocks_without_cpu(launch, baseline):
    r = run_case(launch, ["relay"], relay_cli_inject={
        "relay": ["site=relay.msg,mode=sleep,delta_ns=%d" % DELTA]})
    x = r["relay"]
    assert x["applied"] == M and x["applied_ns"] >= M * DELTA
    assert x["service_mean_ns"] >= 0.95 * DELTA            # the loop is blocked delta per msg
    assert x["loop_cpu_ns"] < 0.3 * M * DELTA               # ... but does not burn CPU
    assert r["per_msg_ns"] >= 0.9 * DELTA > 5 * baseline["per_msg_ns"]


def test_mode3_busy_on_aux_thread(launch, baseline):
    r = run_case(launch, ["relay"], relay_cli_inject={
        "relay": ["site=relay.msg,mode=busy,delta_ns=%d,where=aux_thread" % DELTA]})
    x = r["relay"]
    assert x["applied"] == M and x["aux_pending_end"] == 0  # every hand-off was spun
    assert x["applied_ns"] >= M * DELTA
    assert x["service_mean_ns"] < 0.2 * DELTA               # not on the loop's own path ...
    assert x["proc_cpu_ns"] - x["loop_cpu_ns"] >= 0.8 * M * DELTA  # ... but in this process
    # ... and it competes with the loop for the GIL: loop callbacks wait for the GIL switch
    # interval and the drain sees the relay slow down (the stream is shorter than the queued
    # aux work, so the slowdown is modest; the loop-lag shift is the strong signal)
    assert x["loop_lag_q99"] > 3 * baseline["relay"]["loop_lag_q99"]
    assert r["per_msg_ns"] > 1.2 * baseline["per_msg_ns"]


@pytest.mark.parametrize("d1,d2", [(DELTA // 4, DELTA), (DELTA, DELTA // 4)])
def test_mode2_two_chained_relays(launch, baseline, d1, d2):
    """Per-role delta from the tier table's launch_inject; swapping the deltas swaps which relay
    shows the larger service time."""
    inj = [{"role": "relay1", "site": "relay.msg", "mode": "busy", "delta_ns": d1},
           {"role": "relay2", "site": "relay.msg", "mode": "busy", "delta_ns": d2}]
    r = run_case(launch, ["relay1", "relay2"], launch_inject=inj)
    a, b = r["relay1"], r["relay2"]
    assert a["applied"] == b["applied"] == M
    assert a["applied_ns"] >= M * d1 and b["applied_ns"] >= M * d2
    slow, fast = (b, a) if d2 > d1 else (a, b)
    assert slow["service_mean_ns"] > 2 * fast["service_mean_ns"]
    assert slow["service_mean_ns"] >= 0.95 * max(d1, d2)
    # the chain runs at the pace of the slower relay, not the sum of both
    assert 0.9 * max(d1, d2) <= r["per_msg_ns"] < 1.5 * max(d1, d2) + baseline["per_msg_ns"]
