# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""grpc.server_max_unrequested_time_in_server on every capkit gRPC server.

gRPC core cancels a call that waits longer than this in the server's queue of calls the
application has not picked up yet (grpcio default 30 s). A server whose event loop falls behind
under overload then cancels streams that the load generator would have counted against its
registered timeout. The local reproduction blocks the server's event loop with a sleep injection
while the load generator keeps opening streams.
"""

import json
import types

import pytest

from capgrpc.common import (GRPC_INT_MAX, UNREQUESTED_OPTION, make_server,
                            resolve_max_unrequested)
from capcommon.exitcodes import ConfigError

from conftest import table_doc

METHOD = "/capkit.Test/Stream"
S = 1_000_000_000
# only the first pass through the site sleeps (every_n larger than any count here)
FIRST_ONLY = "every_n=100000,phase=0"


def _tt(timeout_ns, end_ns):
    return types.SimpleNamespace(request_timeout_ns=timeout_ns, end_ns=end_ns)


# ---- resolve_max_unrequested -------------------------------------------------------------------

def test_default_is_the_longer_of_timeout_and_table_rounded_up():
    got = resolve_max_unrequested(None, _tt(60 * S, 900 * S + 1))
    assert got[UNREQUESTED_OPTION] == 901 and got["source"].startswith("default")
    assert resolve_max_unrequested(None, _tt(60 * S + 1, 10 * S))[UNREQUESTED_OPTION] == 61
    assert resolve_max_unrequested(None, _tt(60 * S, 60 * S))[UNREQUESTED_OPTION] == 60


def test_default_without_a_registered_timeout_is_the_table_length():
    assert resolve_max_unrequested(None, _tt(None, 35 * S))[UNREQUESTED_OPTION] == 35
    # a table shorter than a second still gives a valid (>= 1 s) value
    assert resolve_max_unrequested(None, _tt(None, 1))[UNREQUESTED_OPTION] == 1


def test_explicit_value_at_or_above_the_timeout_is_used():
    got = resolve_max_unrequested(60, _tt(60 * S, 900 * S))
    assert got == {UNREQUESTED_OPTION: 60, "source": "--max-unrequested-s"}
    assert resolve_max_unrequested(1, _tt(None, 900 * S))[UNREQUESTED_OPTION] == 1
    assert resolve_max_unrequested(GRPC_INT_MAX, _tt(None, S))[UNREQUESTED_OPTION] == GRPC_INT_MAX


@pytest.mark.parametrize("value,timeout_ns", [
    (59, 60 * S),            # shorter than the registered timeout
    (60, 60 * S + 1),        # shorter by 1 ns
    (0, None),               # 0 cancels every queued call in gRPC core
    (-1, None),
    (GRPC_INT_MAX + 1, None),
])
def test_explicit_value_refused(value, timeout_ns):
    with pytest.raises(ConfigError):
        resolve_max_unrequested(value, _tt(timeout_ns, 900 * S))


def test_make_server_has_no_default_for_the_option():
    with pytest.raises(TypeError):
        make_server(reuseport=False)  # pylint: disable=no-value-for-parameter


# ---- the tools refuse a bad value at start (exit 3, before binding) ----------------------------

@pytest.mark.parametrize("module,extra", [
    ("capgrpc.nullgrpc", ["--workers", "1", "--in-process", "--msg-bytes", "8",
                          "--pacing", "asap"]),
    ("capselfcheck.relay", ["--upstream", "127.0.0.1:1"]),
])
@pytest.mark.parametrize("value", ["0", "9"])
def test_tools_refuse_a_value_below_the_timeout(launch, module, extra, value):
    launch.write_table(table_doc([("warmup", 500, 4.0, {}), ("drain", 500, 0.0, {})],
                                 timeout_ns=10 * S))
    launch.set_origin(4.0)
    launch.server("srv", module, extra + ["--max-unrequested-s", value])
    assert launch.wait_exit("srv", 30) == 3, launch.stderr("srv")
    assert "max_unrequested" in launch.stderr("srv") or "max-unrequested" in launch.stderr("srv")


# ---- local reproduction ------------------------------------------------------------------------

def _load(launch, target_role, cpus):
    launch.start("load", "capgrpc.load",
                 launch.common("load", cpus=cpus)
                 + ["--target-port-file", launch.tree.port_file(target_role), "--method", METHOD,
                    "--workers", "1", "--in-process", "--channels-per-worker", "1",
                    "--schedule", launch.sched_path, "--request-format", "capkit",
                    "--prompts", launch.prompts_path, "--msg-sample-mod", "1",
                    "--max-records-per-worker", "100"])


def _failures(launch, role):
    out = {}
    for docs in launch.summaries(role).values():
        for d in docs:
            for f in d["failures"]:
                out[f["reason"]] = out.get(f["reason"], 0) + f["count"]
    return out


def _counts(launch, role, key):
    return sum(d["counts"][key] for docs in launch.summaries(role).values() for d in docs)


def _recorded_option(launch, role):
    """The value written to the role's ready file and to every tier summary (one value)."""
    with open(launch.tree.ready_file(role)) as f:
        ready = json.load(f)
    vals = {w["info"]["grpc_server_options"][UNREQUESTED_OPTION]
            for w in ready["info"]["workers"]}
    vals |= {d["grpc_server_options"][UNREQUESTED_OPTION]
             for docs in launch.summaries(role).values() for d in docs}
    assert len(vals) == 1, vals
    return vals.pop()


def _sink_stall(launch, value_s):
    """nullgrpc whose loop sleeps 3 s on its first message while 3 more streams arrive
    (fixed arrivals at 0, 0.25, 0.5, 0.75 s; no registered timeout)."""
    launch.write_table(table_doc([("warmup", 1000, 4.0, {"n": 1}), ("drain", 5000, 0.0, {})],
                                 timeout_ns=None))
    launch.set_origin(4.0)
    launch.server("sink", "capgrpc.nullgrpc",
                  ["--workers", "1", "--in-process", "--msg-bytes", "8", "--pacing", "asap",
                   "--inject", "site=send.msg,mode=sleep,delta_ns=%d,%s" % (3 * S, FIRST_ONLY),
                   "--max-unrequested-s", str(value_s)], cpus="0")
    launch.wait_file(launch.tree.port_file("sink"), 15, "sink")
    _load(launch, "sink", "1")
    assert launch.wait_exit("load", 60) == 0, launch.stderr("load")
    assert launch.stop("sink") == 0, launch.stderr("sink")
    assert _recorded_option(launch, "sink") == value_s
    return _failures(launch, "load")


def test_short_value_cancels_streams_queued_behind_a_stalled_loop(launch):
    """The option reaches gRPC core: with 1 s, streams queued ~2.3-2.8 s are cancelled."""
    fails = _sink_stall(launch, 1)
    assert fails.get("grpc_CANCELLED", 0) >= 1, fails
    assert _counts(launch, "load", "ok") + fails["grpc_CANCELLED"] == 4
    assert _counts(launch, "sink", "streams_accepted") == _counts(launch, "load", "ok")


def test_long_value_keeps_streams_queued_behind_a_stalled_loop(launch):
    fails = _sink_stall(launch, 10)
    assert fails == {}, fails
    assert _counts(launch, "load", "ok") == _counts(launch, "sink", "streams_accepted") == 4


def test_relay_default_outlasts_the_grpc_30s_default(launch):
    """SEAMS SM-18 reproduction: the relay's loop is blocked 32 s while 3 more streams arrive;
    with gRPC's own 30 s limit they are cancelled (grpc_CANCELLED at the load generator, missing
    from the relay's streams_accepted). The relay's default (no --max-unrequested-s) is the
    longer of the registered timeout (45 s) and the table length, so they complete. ~45 s."""
    launch.write_table(table_doc([("warmup", 1000, 4.0, {"n": 1}), ("drain", 36000, 0.0, {})],
                                 timeout_ns=45 * S))
    launch.set_origin(4.0)
    launch.server("source", "capgrpc.nullgrpc",
                  ["--workers", "1", "--in-process", "--msg-bytes", "8", "--pacing", "asap"],
                  cpus="0")
    launch.wait_file(launch.tree.port_file("source"), 15, "source")
    launch.server("relay1", "capselfcheck.relay",
                  ["--upstream-port-file", launch.tree.port_file("source"), "--inject",
                   "site=relay.msg,mode=sleep,delta_ns=%d,%s" % (32 * S, FIRST_ONLY)], cpus="1")
    launch.wait_file(launch.tree.port_file("relay1"), 15, "relay1")
    _load(launch, "relay1", "2")
    assert launch.wait_exit("load", 90) == 0, launch.stderr("load")
    assert launch.stop("relay1") == 0, launch.stderr("relay1")
    assert launch.stop("source") == 0, launch.stderr("source")
    assert _recorded_option(launch, "relay1") == 45
    fails = _failures(launch, "load")
    assert fails == {}, fails
    assert _counts(launch, "load", "ok") == 4
    assert _counts(launch, "relay1", "streams_accepted") == 4
    assert _counts(launch, "relay1", "streams_completed") == 4
