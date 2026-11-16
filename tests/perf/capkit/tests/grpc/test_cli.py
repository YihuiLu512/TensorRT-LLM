# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exit codes and stop behaviour (INTERFACES.md 2.3, 2.4)."""

import importlib.util
import json
import time

import pytest

from capcommon.jsonio import write_json_stamped

from conftest import per_tier, table_doc

METHOD = "/capkit.Test/Stream"


def loader_args(launch, target_port_file, extra, loop="asyncio", cpus="3"):
    return (launch.common("load", cpus=cpus, loop=loop)
            + ["--target-port-file", target_port_file, "--method", METHOD,
               "--channels-per-worker", "1", "--schedule", launch.sched_path,
               "--request-format", "capkit", "--prompts", launch.prompts_path,
               "--msg-sample-mod", "1", "--max-records-per-worker", "1000"] + extra)


def small_table(launch):
    return launch.write_table(table_doc([("warmup", 300, 10.0, {}), ("drain", 300, 0.0, {})]))


@pytest.mark.skipif(importlib.util.find_spec("uvloop") is not None,
                    reason="uvloop installed here; the missing-uvloop path cannot be shown")
@pytest.mark.parametrize("tool", ["loader-in", "loader-sup", "nullgrpc", "relay"])
def test_uvloop_requested_but_missing_exits_3(launch, tool):
    small_table(launch)
    launch.set_origin(5.0)
    if tool.startswith("loader"):
        extra = ["--workers", "1"] + (["--in-process"] if tool == "loader-in" else [])
        fake_port = launch.tmp / "port.json"
        fake_port.write_text("{}")
        launch.start("load", "capgrpc.load", loader_args(launch, str(fake_port), extra,
                                                         loop="uvloop"))
        role = "load"
    elif tool == "nullgrpc":
        launch.server("sink", "capgrpc.nullgrpc", ["--workers", "1", "--in-process", "--msg-bytes", "8",
                                           "--pacing", "asap"], loop="uvloop")
        role = "sink"
    else:
        launch.server("relay", "capselfcheck.relay", ["--upstream", "127.0.0.1:1"],
                      loop="uvloop")
        role = "relay"
    assert launch.wait_exit(role, 30) == 3
    assert "uvloop" in launch.stderr(role)


@pytest.mark.parametrize("extra,why", [
    (["--workers", "1", "--in-process", "--msg-bytes", "8", "--pacing", "asap", "--stamp"],
     "--stamp needs --pacing beat"),
    (["--workers", "2", "--msg-bytes", "8", "--pacing", "beat"], "--pacing beat needs"),
    (["--workers", "1", "--in-process", "--msg-bytes", "8", "--pacing", "beat", "--stamp"],
     "too small"),
    (["--workers", "1", "--in-process", "--msg-bytes", "32", "--pacing", "asap",
      "--inject", "site=no.such.site,mode=busy,delta_ns=5"], "unknown inject site"),
])
def test_nullgrpc_config_errors_exit_3(launch, extra, why):
    small_table(launch)
    launch.set_origin(5.0)
    launch.server("sink", "capgrpc.nullgrpc", extra, cpus="0,1")
    assert launch.wait_exit("sink", 30) == 3
    assert why in launch.stderr("sink")


def test_loader_config_errors_exit_3(launch):
    small_table(launch)
    launch.set_origin(5.0)
    args = loader_args(launch, "/nonexistent", ["--workers", "1"])
    i = args.index("--prompts")
    del args[i:i + 2]
    launch.start("load", "capgrpc.load", args)
    assert launch.wait_exit("load", 30) == 3
    assert "needs --prompts" in launch.stderr("load")


def test_unreachable_target_exits_4(launch):
    small_table(launch)
    launch.set_origin(2.0)
    port_file = launch.tree.port_file("ghost")
    from capcommon.identity import Identity
    from capcommon.lifecycle import write_port_file
    from conftest import SHA
    ident = Identity(experiment="E0", arm="unit-a", launch_seq=1, job_id="local-grpctest",
                     restart_count=0, register_sha=SHA, role="ghost")
    write_port_file(port_file, ident, "127.0.0.1", 1)  # nothing listens on port 1
    launch.start("load", "capgrpc.load", loader_args(launch, port_file,
                                                     ["--workers", "1", "--in-process"]))
    assert launch.wait_exit("load", 30) == 4
    assert "not ready" in launch.stderr("load")


def test_stop_mid_schedule_exits_5_with_not_sent(launch):
    """SIGTERM to the supervisor of a 2-worker loader during a long load tier: children stop
    via the pipe, unsent requests are counted as not_sent, stop.json is copied, exit 5."""
    tt = launch.write_table(table_doc([("warmup", 4000, 20.0, {}), ("drain", 300, 0.0, {})]))
    launch.set_origin(4.0)
    launch.server("sink", "capgrpc.nullgrpc", ["--workers", "1", "--in-process", "--msg-bytes", "8",
                                       "--pacing", "asap"])
    launch.wait_file(launch.tree.port_file("sink"), 15, "sink")
    launch.start("load", "capgrpc.load",
                 loader_args(launch, launch.tree.port_file("sink"), ["--workers", "2"],
                             cpus="3,4"))
    launch.wait_file(launch.tree.ready_file("load"), 15, "load")
    while time.monotonic_ns() < launch.origin + 1_000_000_000:
        time.sleep(0.05)
    write_json_stamped(launch.tree.stop_json, "capkit.stop/1",
                       json.loads(json.dumps(_ident_dict())),
                       {"reason": "test_stop", "t_mono_ns": time.monotonic_ns(),
                        "after_tier_seq": 0})
    assert launch.stop("load", 20) == 5, launch.stderr("load")
    assert launch.stop("sink") == 0
    ld = launch.summaries("load")
    assert sorted(ld) == [0] and len(ld[0]) == 2       # only the tier that ran, both workers
    planned = sum(per_tier(ld, "counts.planned").values())
    sent = sum(per_tier(ld, "counts.sent").values())
    not_sent = sum(per_tier(ld, "counts.not_sent").values())
    assert planned == launch.sched_meta["per_tier"]["0"] and 0 < sent < planned
    assert not_sent == planned - sent
    assert all(d["completed"] is False and d["stop"]["reason"] == "test_stop" for d in ld[0])
    assert tt.end_ns > 0


def _ident_dict():
    from conftest import SHA
    return {"experiment": "E0", "arm": "unit-a", "launch_seq": 1, "job_id": "local-grpctest",
            "restart_count": 0, "register_sha": SHA, "role": "launch", "worker": 0, "pid": None,
            "tier_seq": None, "tier_label": None}


@pytest.mark.parametrize("target,ok", [("h:50051", True), ("h:\u00b2", False),
                                       ("h:\u0661", False), ("h:", False)])
def test_resolve_target_port_is_ascii_digits(target, ok):
    from capcommon.exitcodes import ConfigError
    from capgrpc.common import resolve_target
    if ok:
        assert resolve_target(target, None) == target
    else:
        with pytest.raises(ConfigError, match="HOST:PORT"):
            resolve_target(target, None)
