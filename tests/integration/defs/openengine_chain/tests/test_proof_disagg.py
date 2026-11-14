# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chain proof of disaggregated tests, leg by leg: CH-1 to CH-6, CH-9 and cross_node."""

import dataclasses
import json
import os
import pathlib
import xml.etree.ElementTree as ET

import pytest
from openengine_chain import expectations as ex
from openengine_chain import proof as P
from openengine_chain import runner

pytestmark = pytest.mark.cpu_only

CTX, GEN = P.CTX, P.GEN
ACCEPTED = P.ENGINE_ACCEPTED_BUCKETS
SENT = P.SERVE_SENT_BUCKETS
HTTP = P.SERVE_HTTP_RECEIVED
_FAKE_ENGINE_DIR = str(
    pathlib.Path(__file__).resolve().parents[4] / "unittest" / "grpc" / "openengine_client"
)


class Worker:
    """The counters of one worker (front end and engine) as a test moves them."""

    def __init__(self, role, port, instance=None, host="node0"):
        self.role = role
        self.address = f"{host}:{port}"
        self.sent = 0
        self.received = 0
        self.serve = dict.fromkeys(P.SERVE_V4_COUNTERS, 0)
        self.engine = dict.fromkeys((P.ENGINE_ACCEPTED, *P.ENGINE_V4_COUNTERS), 0)
        self.info = P.EngineInfo(instance or f"{role}-{port}", True)
        self.topology = P.Check(P.OK, "fake topology")
        self.start = self.counters()

    def counters(self):
        return P.Counters(
            self.sent, self.received, serve=dict(self.serve), engine_v4=dict(self.engine)
        )

    def take(self, n=1, bucket=None, *, reach=True, handoff=True):
        """``n`` HTTP requests of ``bucket`` (the role's by default), each sent on as one call."""
        bucket = bucket or P.ROLE_BUCKET[self.role]
        self.serve[HTTP[bucket]] += n
        self.serve[SENT[bucket]] += n
        self.sent += n
        if not reach:
            return
        self.received += n
        self.engine[P.ENGINE_ACCEPTED] += n
        self.engine[ACCEPTED[bucket]] += n
        if bucket == P.BUCKET_CONTEXT_ONLY and handoff:
            self.engine[P.ENGINE_PREFILL_READY_SENT] += n
            self.serve[P.SERVE_PREFILL_READY_RECEIVED] += n

    def leg(self):
        return P.Leg(self.role, self.address, self.start, self.counters(), self.info, self.topology)


def _orchestrator(ctx=0, gen=0, *, retries=0, workers=(), extra=(), stale=False):
    counts = {f"{r}.{n}": 0 for r in P.ROLES for n in P.ORCH_METRICS}
    counts.update({"ctx.total_requests": ctx, "gen.total_requests": gen})
    counts["ctx.retry_requests"] = retries
    members = {r: tuple(w.address for w in workers if w.role == r) for r in P.ROLES}
    members[GEN] += tuple(extra)
    return P.OrchestratorReading(counts, members, stale)


_DEFAULT = object()


def _judge(workers, orchestrator=_DEFAULT, inprocess=(), rule=P.LegRule()):
    """judge_disagg; by default the disaggregated server sent what the workers received."""
    if orchestrator is _DEFAULT:
        n = {
            r: sum(w.serve[HTTP[P.ROLE_BUCKET[r]]] for w in workers if w.role == r) for r in P.ROLES
        }
        orchestrator = _orchestrator(n[CTX], n[GEN], workers=workers)
    return P.judge_disagg([w.leg() for w in workers], orchestrator, inprocess, rule)


def _two_legs(n=2):
    ctx, gen = Worker(CTX, 8001), Worker(GEN, 8002)
    ctx.take(n)
    gen.take(n)
    return ctx, gen


def _failed(result):
    return {(name, role) for name, role, check in result.checks if check.status == P.FAIL}


def test_two_legs_are_proven():
    result = _judge(_two_legs())
    assert result.status == P.OK, result.details()
    assert result.roles == {CTX: P.OK, GEN: P.OK}
    assert {name for name, _, _ in result.checks} == {f"CH-{i}" for i in range(1, 7)}
    sums = result.counts["roles"]
    assert (sums[CTX]["counts"]["received"], sums[GEN]["counts"]["received"]) == (2, 2)
    assert sums[CTX]["counts"][P.ENGINE_PREFILL_READY_SENT] == 2


def _mutations():
    def no_connector(ctx, gen):
        ctx.info = P.EngineInfo("ctx-x", False)

    def same_instance(ctx, gen):
        gen.info = P.EngineInfo(ctx.info.instance_id, True)

    def in_process(ctx, gen):
        gen.topology = P.Check(P.FAIL, "the front end holds a GPU")

    def call_lost(ctx, gen):
        ctx.take(1, reach=False, handoff=False)
        gen.take(1)

    def aggregated_on_gen(ctx, gen):
        gen.take(1, P.BUCKET_AGGREGATED)

    def prefill_ready_lost(ctx, gen):
        ctx.engine[P.ENGINE_PREFILL_READY_SENT] += 1

    def request_dropped(ctx, gen):
        ctx.serve[HTTP[P.BUCKET_CONTEXT_ONLY]] += 1

    return [
        ("CH-1", CTX, no_connector),
        ("CH-1", GEN, same_instance),
        ("CH-1", GEN, in_process),
        ("CH-2", CTX, call_lost),
        ("CH-3", GEN, aggregated_on_gen),
        ("CH-4", CTX, prefill_ready_lost),
        ("CH-5", CTX, request_dropped),
    ]


@pytest.mark.parametrize(
    "check, role, mutate", _mutations(), ids=lambda v: getattr(v, "__name__", v)
)
def test_a_broken_leg_fails_its_check(check, role, mutate):
    ctx, gen = _two_legs()
    mutate(ctx, gen)
    result = _judge([ctx, gen])
    assert (check, role) in _failed(result), result.details()
    assert result.status == P.FAIL
    assert result.roles[role] == P.FAIL


def test_unread_counters_leave_the_proof_incomplete():
    ctx, gen = _two_legs()
    gen.info = None
    result = _judge([ctx, gen], inprocess=None)
    assert {(n, r) for n, r, c in result.checks if c.status == P.INCOMPLETE} == {
        ("CH-1", GEN),
        ("CH-6", None),
    }
    assert result.status == P.INCOMPLETE
    result = _judge(list(_two_legs()), orchestrator=None)
    assert {(n, r) for n, r, c in result.checks if c.status == P.INCOMPLETE} == {
        ("CH-5", CTX),
        ("CH-5", GEN),
        ("CH-6", None),
    }


def test_a_worker_outside_the_chain_fails_ch6():
    ctx, gen = _two_legs()
    outside = _orchestrator(2, 2, workers=[ctx, gen], extra=["other-host:8003"])
    assert ("CH-6", None) in _failed(_judge([ctx, gen], outside))
    assert ("CH-6", None) in _failed(
        _judge([ctx, gen], inprocess=["pid 7 (trtllm-serve m) holds a GPU"])
    )


@pytest.mark.parametrize(
    "retries, cancels, status",
    [(1, False, P.OK), (0, False, P.FAIL), (0, True, P.OK)],
)
def test_orphan_hand_offs_are_allowed_up_to_the_retries(retries, cancels, status):
    """A retried or cancelled context leg leaves a hand-off no generation leg takes."""
    ctx, gen = _two_legs()
    ctx.take(1)
    orchestrator = _orchestrator(3 - retries, 2, retries=retries, workers=[ctx, gen])
    rule = P.LegRule(cancels_calls=cancels)
    result = _judge([ctx, gen], orchestrator, rule=rule)
    statuses = {(n, r): c.status for n, r, c in result.checks}
    assert statuses[("CH-4", None)] == status, result.details()


def test_a_stopped_disaggregated_server_counts_as_a_lower_bound():
    ctx, gen = _two_legs()
    stale = _orchestrator(1, 2, workers=[ctx, gen], stale=True)
    statuses = {(n, r): c.status for n, r, c in _judge([ctx, gen], stale).checks}
    assert statuses[("CH-5", CTX)] == P.INCOMPLETE
    fresh = dataclasses.replace(stale, stale=False)
    assert ("CH-5", CTX) in _failed(_judge([ctx, gen], fresh))


def _gen_zero_ok():
    ctx, gen = Worker(CTX, 8001), Worker(GEN, 8002)
    ctx.take(2, handoff=False)
    return [ctx, gen], _DEFAULT, ("CH-3", GEN)


def _ctx_zero_ok():
    ctx, gen = Worker(CTX, 8001), Worker(GEN, 8002)
    gen.take(2, P.BUCKET_AGGREGATED)
    return [ctx, gen], _orchestrator(0, 2, workers=[ctx, gen]), ("CH-3", CTX)


def _single_side():
    ctx = Worker(CTX, 8001)
    ctx.take(2)
    return [ctx], _orchestrator(2, 0, workers=[ctx]), ("CH-3", GEN)


def _direct_worker():
    ctx, gen = Worker(CTX, 8001), Worker(GEN, 8002)
    gen.take(3, P.BUCKET_AGGREGATED)
    return [ctx, gen], None, ("CH-3", GEN)


def _direct_two_leg():
    return list(_two_legs()), None, ("CH-5", CTX)


def _direct_ctx_leg():
    ctx, gen = Worker(CTX, 8001), Worker(GEN, 8002)
    ctx.take(1)
    return [ctx, gen], None, ("CH-3", GEN)


def _killed():
    ctx, gen = Worker(CTX, 8001), Worker(GEN, 8002)
    ctx.take(2)
    gen.start = P.Counters(None, None)
    gen.info = P.EngineInfo("gen-8002", True)
    return [ctx, gen], _orchestrator(2, 2, workers=[ctx, gen]), ("CH-3", GEN)


_LEGS = {
    ex.GEN_ZERO_OK: (_gen_zero_ok, {}),
    ex.CTX_ZERO_OK: (_ctx_zero_ok, {}),
    ex.SINGLE_SIDE: (_single_side, {}),
    ex.DIRECT_WORKER: (_direct_worker, {"direct_requests": 3}),
    ex.DIRECT_TWO_LEG: (_direct_two_leg, {}),
    ex.DIRECT_CTX_LEG: (_direct_ctx_leg, {}),
    ex.KILLED: (_killed, {"killed": (GEN,)}),
}


@pytest.mark.parametrize("leg", sorted(_LEGS))
def test_each_leg_expectation_allows_what_it_names(leg):
    make, fields = _LEGS[leg]
    workers, orchestrator, fails_without = make()
    result = _judge(workers, orchestrator, rule=P.LegRule(leg=leg, **fields))
    assert result.status == P.OK, result.details()
    assert fails_without in {
        (n, r) for n, r, c in _judge(workers, orchestrator).checks if c.status != P.OK
    }


def test_every_leg_value_is_covered():
    assert set(_LEGS) | {ex.CROSS_NODE} == set(ex.LEG_VALUES)


def test_killed_names_the_unproven_role():
    workers, orchestrator, _ = _killed()
    result = _judge(workers, orchestrator, rule=P.LegRule(leg=ex.KILLED, killed=(GEN,)))
    assert result.roles == {CTX: P.OK, GEN: P.KILLED_INCOMPLETE}


@pytest.mark.parametrize(
    "leg, fields, check",
    [
        (ex.DIRECT_WORKER, {"direct_requests": 2}, ("CH-3", GEN)),
        (ex.DIRECT_CTX_LEG, {"orphans": 0}, ("CH-4", None)),
    ],
)
def test_a_leg_expectation_with_a_wrong_count_fails(leg, fields, check):
    workers, orchestrator, _ = _LEGS[leg][0]()
    result = _judge(workers, orchestrator, rule=P.LegRule(leg=leg, **fields))
    assert check in _failed(result), result.details()


def _node(workers, orchestrator, names):
    """One node's ``chain_leg_counts`` for a cross_node test."""
    result = _judge(workers, orchestrator, rule=P.LegRule(leg=ex.CROSS_NODE))
    assert result.status == P.OK and result.deferred, result.details()
    counts = json.loads(json.dumps(result.counts))
    counts["node_names"] = list(names)
    return counts


def _cross_node(gen_takes=2, gen_listed_as="node1"):
    ctx, gen = Worker(CTX, 8001, host="node0"), Worker(GEN, 8002, host="node1")
    ctx.take(2)
    gen.take(gen_takes)
    members = {CTX: ("node0:8001",), GEN: (f"{gen_listed_as}:8002",)}
    counts = {f"{r}.{n}": 0 for r in P.ROLES for n in P.ORCH_METRICS}
    counts.update({"ctx.total_requests": 2, "gen.total_requests": gen_takes})
    node0 = _node([ctx], P.OrchestratorReading(counts, members), ["node0", "10.0.0.1"])
    node1 = _node([gen], None, ["node1", "10.0.0.2"])
    return node0, node1


def test_a_cross_node_test_is_proven_over_both_nodes():
    result = P.judge_cross_node(_cross_node())
    assert result.status == P.OK, result.details()
    props = P.disagg_props(result.counts, result.roles)
    assert (props["handoffs"], props["orphan_handoffs"], props["engine_calls_gen"]) == (
        "2",
        "0",
        "2",
    )
    # The disaggregated server may list a worker by any name of its node.
    assert P.judge_cross_node(_cross_node(gen_listed_as="10.0.0.2")).status == P.OK


def test_a_cross_node_mismatch_or_a_missing_node_is_not_proven():
    result = P.judge_cross_node(_cross_node(gen_takes=3))
    assert result.status == P.FAIL
    assert any(d.startswith("CH-4: fail") for d in result.details()), result.details()
    node0, node1 = _cross_node(gen_listed_as="10.0.0.9")
    assert P.judge_cross_node([node0, node1]).status == P.FAIL
    assert P.judge_cross_node([node0, None]).status == P.INCOMPLETE
    assert P.judge_cross_node([node0]).status == P.INCOMPLETE


def test_a_cross_node_test_keeps_its_expectations():
    """Orphan hand-offs of a test that cancels calls are allowed after the merge too."""
    ctx, gen = Worker(CTX, 8001, host="node0"), Worker(GEN, 8002, host="node1")
    ctx.take(3)
    gen.take(2)
    members = {CTX: ("node0:8001",), GEN: ("node1:8002",)}
    counts = {f"{r}.{n}": 0 for r in P.ROLES for n in P.ORCH_METRICS}
    counts.update({"ctx.total_requests": 3, "gen.total_requests": 2})
    nodes = []
    for workers, orchestrator, names in (
        ([ctx], P.OrchestratorReading(counts, members), ["node0"]),
        ([gen], None, ["node1"]),
    ):
        rule = P.LegRule(leg=ex.CROSS_NODE, cancels_calls=True)
        result = _judge(workers, orchestrator, rule=rule)
        node = json.loads(json.dumps(result.counts))
        node["node_names"] = names
        nodes.append(node)
        props = P.disagg_props(result.counts, result.roles, partial=True)
        assert props["orphan_handoffs"] == P.NOT_HERE
    assert P.judge_cross_node(nodes).status == P.OK
    for node in nodes:
        node["rule"]["cancels_calls"] = False
    assert P.judge_cross_node(nodes).status == P.FAIL


def _junit(path, name, outcome, counts):
    failure = "<failure message='x'/>" if outcome == "failed" else ""
    props = {"chain_proof": "ok", "leg_expectation": "cross_node:met on this node"}
    props["chain_leg_counts"] = json.dumps(counts)
    xml = "".join(f"<property name='{k}' value='{_attr(v)}'/>" for k, v in props.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<testsuites><testsuite><testcase classname='t' name='{name}' time='1'>"
        f"<properties>{xml}</properties>{failure}</testcase></testsuite></testsuites>"
    )


def _attr(value):
    return (
        value.replace("&", "&amp;")
        .replace("'", "&apos;")
        .replace("<", "&lt;")
        .replace('"', "&quot;")
    )


@pytest.mark.parametrize(
    "gen_takes, outcome1, expected",
    [
        (2, "passed", "passed"),
        (3, "passed", "failed"),
        (2, "failed", "failed"),
        (2, None, "incomplete"),
    ],
)
def test_the_runner_merges_the_nodes_of_a_cross_node_test(tmp_path, gen_takes, outcome1, expected):
    node0, node1 = _cross_node(gen_takes=gen_takes)
    _junit(tmp_path / "node-0" / "junit" / "a.xml", "test_completion[x]", "passed", node0)
    if outcome1 is not None:
        _junit(tmp_path / "node-1" / "junit" / "a.xml", "test_completion[x]", outcome1, node1)
    rows = runner.summarize(str(tmp_path))
    merged = [r for r in rows if r["file"] == "cross-node"]
    assert [r["outcome"] for r in merged] == [expected], merged
    assert sorted(r["file"].split("/")[0] for r in rows if r["file"] != "cross-node") == sorted(
        ["node-0"] + (["node-1"] if outcome1 else [])
    )
    # The test counts once, and a failed merge fails the summary run.
    counts = (tmp_path / "summary.md").read_text().splitlines()[2]
    assert counts == f"Counts: {expected} 1", counts
    rc = runner.main(["--chain", "trtllm", "--out", str(tmp_path), "--summarize-only"])
    assert rc == (1 if expected == "failed" else 0)
    if expected == "passed":
        assert (merged[0]["handoffs"], merged[0]["orphan_handoffs"]) == ("2", "0"), merged


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"SLURM_NODEID": "1", "SLURM_NNODES": "2"}, "out/node-1"),
        ({"SLURM_NODEID": "0", "SLURM_NNODES": "1"}, "out"),
        ({}, "out"),
    ],
)
def test_each_node_writes_its_own_results(env, expected):
    assert runner.node_dir("out", env) == expected


def test_a_serving_process_holding_a_gpu_without_grpc_is_found():
    table = {
        10: (1, 0, 0),  # an in-process worker
        11: (10, 0, 0),  # its executor process, on a GPU
        20: (1, 0, 0),  # a chain engine
        21: (20, 0, 0),
        30: (2, 0, 0),  # outside the test's processes
        40: (1, 0, 0),  # a ds chain engine
        41: (40, 0, 0),
        50: (1, 0, 0),  # a ds server outside the chain
    }
    cmdlines = {
        10: ["/usr/bin/python3", "/usr/local/bin/trtllm-serve", "m", "--server_role", "context"],
        11: ["python3", "-c", "worker"],
        20: ["trtllm-serve", "m", "--grpc", "--port", "50051"],
        21: ["python3", "-c", "worker"],
        30: ["trtllm-serve", "m"],
        40: ["python3", "-m", "dsv4_serve", "/ckpt", "--grpc", "--grpc-protocol", "openengine"],
        41: ["python3", "-c", "worker"],
        50: ["/opt/bin/dsv4-serve", "/ckpt"],
    }
    found = P.inprocess_workers(
        1,
        gpu_pid_reader=lambda: {11, 21, 30, 41, 50},
        table_reader=lambda: table,
        cmdline_reader=cmdlines.get,
    )
    assert [f.split(" (")[0] for f in found] == ["pid 10", "pid 50"], found
    assert P.inprocess_workers(1, gpu_pid_reader=lambda: None) is None


def test_disaggregated_servers_are_found_and_read(tmp_path, monkeypatch):
    config = tmp_path / "disagg.yaml"
    config.write_text("hostname: 0.0.0.0\nport: 8123\n")
    cmdlines = {
        5: ["trtllm-serve", "disaggregated", "-c", str(config)],
        6: [
            "python3",
            "-m",
            "tensorrt_llm.commands.serve",
            "disaggregated",
            f"--config_file={config}",
        ],
        7: ["trtllm-serve", "m", "--server_role", "generation"],
    }
    monkeypatch.setattr(P, "live_descendants", lambda root: set(cmdlines))
    found = P.find_orchestrators(1, cmdline_reader=cmdlines.get)
    assert found == [P.Orchestrator(5, "http://127.0.0.1:8123", str(config))]
    pages = {
        "/prometheus/metrics": b"# TYPE ctx_total_requests_total counter\n"
        b"ctx_total_requests_total 3.0\ngen_total_requests_total 2.0\n"
        b'ctx_retry_requests_total{server="a"} 1.0\ngen_error_requests_created 1.7e9\n',
        "/cluster_info": json.dumps(
            {"server_lists": {"context": ["http://node0:8001"], "generation": ["node1:8002"]}}
        ).encode(),
    }
    reading = P.read_orchestrator(found[0], getter=lambda url: pages[url[len(found[0].url) :]])
    assert {k: v for k, v in reading.counts.items() if v} == {
        "ctx.total_requests": 3,
        "gen.total_requests": 2,
        "ctx.retry_requests": 1,
    }
    assert reading.members == {CTX: ("http://node0:8001",), GEN: ("node1:8002",)}


def test_worker_addresses_compare_across_spellings():
    local = P.normalize_address("localhost:8001")
    assert P.normalize_address("http://127.0.0.1:8001/") == local
    assert P.normalize_address("node1.example.com:8002") == "node1:8002"
    assert P.normalize_address("10.0.0.2:8002") == "10.0.0.2:8002"
    assert local.split(":")[0] in P.node_names()


def test_the_counter_names_match_the_conventions():
    conventions = pytest.importorskip("tensorrt_llm.grpc.openengine_conventions")
    assert dict(conventions.COUNTER_GENERATE_ACCEPTED_BUCKETS) == P.ENGINE_ACCEPTED_BUCKETS
    assert conventions.COUNTER_GENERATE_ACCEPTED == P.ENGINE_ACCEPTED
    assert conventions.COUNTER_PREFILL_READY_SENT == P.ENGINE_PREFILL_READY_SENT
    assert tuple(conventions.VERSION_4_ENGINE_COUNTERS) == P.ENGINE_V4_COUNTERS
    assert conventions.SERVE_COUNTERS[1:] == P.SERVE_V4_COUNTERS
    assert conventions.SERVE_COUNTERS[0] == P.cv.FRONTEND_COUNTER_FIELD


class _Handle:
    """What read_counters and engine_info read of a chain worker."""

    def __init__(self, role, counter_dir, endpoint):
        self.role = role
        self.alive = True
        self.counter_dir = counter_dir
        self.engine_endpoint = endpoint
        self._engine_name = None


def test_a_worker_s_counters_are_read_from_its_engine_and_front_end(tmp_path, monkeypatch):
    pytest.importorskip("grpc")
    contracts = pytest.importorskip("tensorrt_llm.serve.openengine_poc.contracts")
    monkeypatch.syspath_prepend(_FAKE_ENGINE_DIR)
    fake = pytest.importorskip("fake_openengine")
    counter = contracts.GenerateCounter(str(tmp_path))
    counter.start()
    try:
        engine = fake.FakeEngine(fake.disagg_profile("trtllm-like"))
        with fake.running_in_thread(engine) as engine:
            handle = _Handle(CTX, str(tmp_path), f"127.0.0.1:{engine.port}")
            first = P.read_counters(handle)
            assert first.serve == dict.fromkeys(P.SERVE_V4_COUNTERS, 0)
            assert first.engine_v4 == dict.fromkeys((P.ENGINE_ACCEPTED, *P.ENGINE_V4_COUNTERS), 0)
            counter.increment(2, P.BUCKET_CONTEXT_ONLY)
            counter.add(HTTP[P.BUCKET_CONTEXT_ONLY], 2)
            later = P.read_counters(handle)
            assert later.frontend == 2
            assert later.serve[SENT[P.BUCKET_CONTEXT_ONLY]] == 2
            assert later.serve[HTTP[P.BUCKET_CONTEXT_ONLY]] == 2
            info = P.engine_info(handle)
            assert info == P.EngineInfo(engine.instance_id, True)
    finally:
        counter.close()


GENERATED = """
import time

import pytest
from openengine_chain import proof as P
from plugin_support import FakeHandle

STATE = P._disagg_test_state


class Worker(FakeHandle):
    def __init__(self, role, port):
        super().__init__()
        self.role = role
        self.host = "localhost"
        self.port = port
        self.info = P.EngineInfo(f"{role}-{port}", True)
        self.serve = dict.fromkeys(P.SERVE_V4_COUNTERS, 0)
        self.v4 = dict.fromkeys((P.ENGINE_ACCEPTED, *P.ENGINE_V4_COUNTERS), 0)
        self.take(0)

    def take(self, n=1, handoff=True):
        bucket = P.ROLE_BUCKET[self.role]
        for key in (P.SERVE_HTTP_RECEIVED[bucket], P.SERVE_SENT_BUCKETS[bucket]):
            self.serve[key] += n
        for key in (P.ENGINE_ACCEPTED, P.ENGINE_ACCEPTED_BUCKETS[bucket]):
            self.v4[key] += n
        if self.role == P.CTX and handoff:
            self.v4[P.ENGINE_PREFILL_READY_SENT] += n
            self.serve[P.SERVE_PREFILL_READY_RECEIVED] += n
        c = self.counters
        self.counters = P.Counters(
            c.frontend + n, c.engine + n, serve=dict(self.serve), engine_v4=dict(self.v4)
        )
        STATE["counts"][f"{self.role}.total_requests"] += n


@pytest.fixture(scope="module")
def legs():
    ctx, gen = Worker("ctx", 8001), Worker("gen", 8002)
    STATE["members"] = {"ctx": ("localhost:8001",), "gen": ("http://127.0.0.1:8002",)}
    yield ctx, gen
    ctx.end()
    gen.end()


@pytest.fixture(autouse=True)
def no_worker_outside():
    yield
    STATE["inprocess"] = []


def test_two_legs(legs):
    legs[0].take(2)
    legs[1].take(2)


def test_normal_request(legs):
    legs[0].take(1)
    legs[1].take(1)


def test_orphan(legs):
    legs[0].take(2)
    legs[1].take(1)


def test_gen_zero_ok(legs):
    legs[0].take(1, handoff=False)


def test_gen_zero_unregistered(legs):
    legs[0].take(1, handoff=False)


def test_worker_outside(legs):
    STATE["inprocess"] = ["pid 7 (trtllm-serve m --server_role generation) holds a GPU"]
    legs[0].take(1)
    legs[1].take(1)


@pytest.fixture
def pair(legs):
    ctx, gen = Worker("ctx", 8003), Worker("gen", 8004)
    STATE["members"] = {
        "ctx": ("localhost:8001", "localhost:8003"),
        "gen": ("localhost:8002", "localhost:8004"),
    }
    yield ctx, gen
    ctx.end()
    gen.end()


def test_killed_gen(pair):
    time.sleep(0.5)  # the watch takes the workers' topology while they run
    pair[0].take(1)
    pair[1].topology = P.Check(P.INCOMPLETE, "the front end has exited")
"""

AGG = """
from plugin_support import FakeHandle

def test_agg():
    h = FakeHandle()
    h.send(1)
    h.end()
"""


@pytest.fixture
def run_disagg(pytester, monkeypatch, tmp_path):
    import plugin_support
    from openengine_chain import conventions as cv

    plugin_support.install(monkeypatch)
    state = {
        "counts": dict.fromkeys(_orchestrator().counts, 0),
        "members": {CTX: (), GEN: ()},
        "inprocess": [],
    }
    monkeypatch.setattr(P, "_disagg_test_state", state, raising=False)
    monkeypatch.setattr(
        P, "find_orchestrators", lambda root: [P.Orchestrator(1, "http://o:1", "c")]
    )
    monkeypatch.setattr(
        P,
        "read_orchestrator",
        lambda o: P.OrchestratorReading(dict(state["counts"]), dict(state["members"])),
    )
    monkeypatch.setattr(P, "inprocess_workers", lambda root: list(state["inprocess"]))
    monkeypatch.setattr(P, "engine_info", lambda h: h.info)
    table = [
        ex.Expectation(
            "WT-38", "test_disagg.py::test_gen_zero_ok", ex.LEG_EXPECT, "r", leg=ex.GEN_ZERO_OK
        ),
        ex.Expectation(
            "WT-38",
            "test_disagg.py::test_killed_gen",
            ex.LEG_EXPECT,
            "r",
            leg=ex.KILLED,
            killed=(GEN,),
        ),
    ]
    from openengine_chain import pytest_plugin as plugin

    monkeypatch.setattr(plugin, "_WATCH_INTERVAL_S", 0.05)
    monkeypatch.setattr(plugin, "_GPU_CHECK_INTERVAL_S", 0.0)
    monkeypatch.setattr(ex, "load", lambda directory=None: list(table))
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    monkeypatch.delenv(cv.ENV_RESULTS_DIR, raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.syspath_prepend(os.path.dirname(os.path.abspath(plugin_support.__file__)))
    pytester.makepyfile(test_disagg=GENERATED, test_disagg_z_agg=AGG)

    def run(negative_control=None):
        if negative_control:
            monkeypatch.setenv("TRTLLM_TEST_OPENENGINE_NEGATIVE_CONTROL", negative_control)
        else:
            monkeypatch.delenv("TRTLLM_TEST_OPENENGINE_NEGATIVE_CONTROL", raising=False)
        xml = tmp_path / f"junit-{len(os.listdir(tmp_path))}.xml"
        pytester.runpytest_inprocess("-p", cv.PLUGIN_MODULE, f"--junitxml={xml}")
        return _cases(str(xml))

    return run


def _cases(path):
    """``{test name: (outcome, junit properties)}``."""
    cases = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        outcome = "passed"
        for tag in ("failure", "error", "skipped"):
            node = case.find(tag)
            if node is not None:
                outcome = tag
                props["_text"] = (node.get("message") or "") + (node.text or "")
                break
        cases[case.get("name")] = (outcome, props)
    return cases


def test_the_plugin_proves_each_leg(run_disagg):
    cases = run_disagg()
    outcome, props = cases["test_two_legs"]
    assert outcome == "passed", props
    shown = {k: v for k, v in props.items() if k not in ("chain_leg_counts", "chain_proof_detail")}
    assert shown == {
        "openengine_chain": "trtllm",
        "engine_calls": "4",
        "engine_name": "fake-engine",
        "chain_proof": "ok",
        "chain_topology": "disagg",
        "chain_roles": "ctx:1,gen:1",
        "engine_calls_ctx": "2",
        "engine_calls_gen": "2",
        "handoffs": "2",
        "orphan_handoffs": "0",
        "orch_requests_ctx": "2",
        "orch_requests_gen": "2",
        "chain_proof_ctx": "ok",
        "chain_proof_gen": "ok",
        "leg_expectation": "two_legs:met",
    }
    assert json.loads(props["chain_leg_counts"])["node_names"] == P.node_names()
    assert cases["test_gen_zero_ok"][0] == "passed"
    assert cases["test_gen_zero_ok"][1]["leg_expectation"] == "gen_zero_ok:met"
    outcome, props = cases["test_gen_zero_unregistered"]
    assert (outcome, props["chain_proof_gen"], props["chain_proof_ctx"]) == (
        "failure",
        "fail",
        "ok",
    )
    assert props["leg_expectation"] == "two_legs:not met"
    outcome, props = cases["test_worker_outside"]
    assert outcome == "failure" and "CH-6: fail" in props["chain_proof_detail"], props
    outcome, props = cases["test_killed_gen"]
    assert (outcome, props["chain_proof"], props["chain_proof_gen"]) == (
        "passed",
        "ok",
        P.KILLED_INCOMPLETE,
    ), props
    # An aggregated server next to a disaggregated one is not proven.
    outcome, props = cases["test_agg"]
    assert (props["chain_topology"], props["chain_proof"]) == ("agg", "incomplete"), props


def test_orphan_handoffs_are_counted(run_disagg):
    """NT-024: a hand-off the context engine sent and no generation engine took is an orphan."""
    cases = run_disagg()
    assert cases["test_normal_request"][1]["orphan_handoffs"] == "0"
    outcome, props = cases["test_orphan"]
    assert (outcome, props["handoffs"], props["orphan_handoffs"]) == ("failure", "2", "1")
    assert (
        "CH-4: fail: 2 hand-offs sent, 1 taken: 1 orphan hand-offs" in props["chain_proof_detail"]
    )


def test_the_negative_control_passes_only_when_ch6_fails(run_disagg):
    """CH-9: with a worker started outside the chain, only a test whose CH-6 fails passes."""
    cases = run_disagg(negative_control=GEN)
    outcome, props = cases["test_worker_outside"]
    assert (outcome, props["negative_control"]) == ("passed", "gen:met"), props
    outcome, props = cases["test_two_legs"]
    assert (outcome, props["negative_control"]) == ("failure", "gen:not met"), props


def test_direct_two_leg_checks_each_worker_s_own_counts():
    ctx, gen = _two_legs()
    ctx.engine[P.ENGINE_PREFILL_READY_SENT] += 1
    result = _judge([ctx, gen], None, rule=P.LegRule(leg=ex.DIRECT_TWO_LEG))
    assert _failed(result) == {("CH-4", CTX)}, result.details()
    assert not [c for n, _, c in result.checks if n == "CH-6"]


def test_direct_ctx_leg_without_a_reading_is_incomplete():
    workers, _, _ = _direct_ctx_leg()
    workers[0].start = dataclasses.replace(workers[0].start, engine_v4=None)
    result = _judge(workers, None, rule=P.LegRule(leg=ex.DIRECT_CTX_LEG))
    assert result.status == P.INCOMPLETE, result.details()


def test_an_aggregated_request_split_into_calls_is_not_a_mismatch():
    """UV-52: a request with n > 1 may reach the engine as n calls."""
    ctx, gen = Worker(CTX, 8001), Worker(GEN, 8002)
    gen.take(2, P.BUCKET_AGGREGATED)
    gen.serve[HTTP[P.BUCKET_AGGREGATED]] -= 1
    rule = P.LegRule(leg=ex.DIRECT_WORKER)
    statuses = {(n, r): c.status for n, r, c in _judge([ctx, gen], None, rule=rule).checks}
    assert statuses[("CH-5", GEN)] == P.OK
    rule = P.LegRule(leg=ex.DIRECT_WORKER, cancels_calls=True)
    statuses = {(n, r): c.status for n, r, c in _judge([ctx, gen], None, rule=rule).checks}
    assert statuses[("CH-5", GEN)] == P.INCOMPLETE


def test_the_runner_keeps_the_per_role_properties():
    names = (
        "chain_topology chain_roles engine_calls_ctx engine_calls_gen handoffs orphan_handoffs "
        "orch_requests_ctx orch_requests_gen chain_proof_ctx chain_proof_gen leg_expectation"
    ).split()
    assert set(names) <= set(runner._PROPS)


def test_the_negative_control_starts_the_first_worker_of_its_role_outside_the_chain(monkeypatch):
    import sys

    from openengine_chain import launch
    from openengine_chain import pytest_plugin as plugin

    started = []
    monkeypatch.setattr(launch, "_HANDLES", [])
    monkeypatch.setattr(launch, "launch_serve_command", lambda cmd, **kw: started.append(cmd))
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    undo = plugin._start_one_worker_outside_the_chain(GEN)
    try:
        context = launch.launch_serve_command(sleeper + ["--server_role", "CONTEXT"])
        outside = launch.launch_serve_command(
            sleeper + ["--server_role", "GENERATION", "--port", "8"]
        )
        again = launch.launch_serve_command(sleeper + ["--server_role", "GENERATION"])
        assert context is None and again is None and len(started) == 2
        try:
            assert (outside.chain, outside.role, outside.port) == ("none", GEN, 8)
            assert outside.frontend.poll() is None
        finally:
            outside.frontend.kill()
            outside.frontend.wait()
    finally:
        undo()
    assert launch.launch_serve_command is not None and len(started) == 2
    wrapper_after_undo = plugin._start_one_worker_outside_the_chain(GEN)
    held = launch.launch_serve_command
    wrapper_after_undo()
    # A module that took the replacement keeps it, but it no longer replaces anything.
    assert held(sleeper + ["--server_role", "GENERATION"]) is None and len(started) == 3


def test_a_test_that_cancels_calls_is_held_to_equal_counts_until_the_last_read():
    ctx, gen = _two_legs()
    gen.take(1, reach=False)
    rule = P.LegRule(cancels_calls=True)
    legs = [ctx.leg(), gen.leg()]
    orchestrator = _orchestrator(2, 3, workers=[ctx, gen])
    early = P.judge_disagg(legs, orchestrator, (), rule, last=False)
    assert ("CH-2", GEN) in _failed(early)
    assert ("CH-2", GEN) not in _failed(P.judge_disagg(legs, orchestrator, (), rule, last=True))


def test_a_split_request_does_not_hide_extra_calls_of_another_bucket():
    ctx, gen = Worker(CTX, 8001), Worker(GEN, 8002)
    gen.take(2, P.BUCKET_AGGREGATED)
    gen.serve[HTTP[P.BUCKET_AGGREGATED]] -= 1
    gen.take(1, reach=False)
    gen.serve[HTTP[P.BUCKET_GENERATION_ONLY]] -= 1
    rule = P.LegRule(leg=ex.DIRECT_WORKER)
    assert ("CH-5", GEN) in _failed(_judge([ctx, gen], None, rule=rule))


def test_direct_worker_without_a_registered_count_needs_a_call_per_request():
    ctx, gen = Worker(CTX, 8001), Worker(GEN, 8002)
    gen.take(2, P.BUCKET_AGGREGATED)
    gen.serve[HTTP[P.BUCKET_AGGREGATED]] += 1
    rule = P.LegRule(leg=ex.DIRECT_WORKER)
    assert ("CH-3", GEN) in _failed(_judge([ctx, gen], None, rule=rule))


def test_single_side_without_a_disaggregated_server_judges_each_worker_alone():
    ctx = Worker(CTX, 8001)
    ctx.take(2, P.BUCKET_AGGREGATED)
    result = _judge([ctx], None, rule=P.LegRule(leg=ex.SINGLE_SIDE))
    assert result.status == P.OK, result.details()
    gen = Worker(GEN, 8002)
    gen.take(2, P.BUCKET_AGGREGATED)
    gen_only = _judge([gen], None, rule=P.LegRule(leg=ex.SINGLE_SIDE))
    assert ("CH-3", GEN) in _failed(gen_only), gen_only.details()


def test_a_worker_of_unknown_address_leaves_ch6_incomplete():
    ctx, gen = _two_legs()
    gen.address = None
    orchestrator = _orchestrator(2, 2, workers=[ctx], extra=["node0:8002"])
    statuses = {(n, r): c.status for n, r, c in _judge([ctx, gen], orchestrator).checks}
    assert statuses[("CH-6", None)] == P.INCOMPLETE


def test_the_listening_ports_of_a_process_are_found():
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert port in P.listening_ports({os.getpid()})
    assert port not in P.listening_ports({os.getpid()})


def test_gpu_use_that_no_process_of_the_test_shows_is_not_checked():
    table = {10: (1, 0, 0), 30: (2, 0, 0)}
    found = P.inprocess_workers(
        1,
        gpu_pid_reader=lambda: {30},
        table_reader=lambda: table,
        cmdline_reader=lambda pid: ["trtllm-serve", "m"],
    )
    assert found is None


@pytest.mark.parametrize(
    "fields",
    [
        {"leg": ex.KILLED, "killed": ["gen"]},
        {"leg": ex.KILLED},
        {"leg": ex.GEN_ZERO_OK, "killed": ("gen",)},
        {"leg": "other"},
        {"leg": ex.DIRECT_WORKER, "orphans": 1},
        {"leg": ex.DIRECT_CTX_LEG, "orphans": -1},
        {},
    ],
)
def test_a_wrong_leg_expectation_is_refused(fields):
    with pytest.raises(ValueError):
        ex.Expectation("WT-38", "t.py::t", ex.LEG_EXPECT, "r", **fields)


def test_only_one_leg_expectation_applies_to_a_test():
    table = [
        ex.Expectation("WT-38", "t.py::t*", ex.LEG_EXPECT, "r", leg=ex.GEN_ZERO_OK),
        ex.Expectation("WT-51", "t.py::test", ex.LEG_EXPECT, "r", leg=ex.CTX_ZERO_OK),
    ]
    with pytest.raises(ValueError, match="conflicting"):
        ex.applicable(table, "a/t.py::test", "trtllm")
