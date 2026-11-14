# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: chain proof, junit markers and expectation tables for every test.

Load it with ``-p openengine_chain.pytest_plugin`` (the runner sets
``PYTEST_ADDOPTS`` so that pytest processes started by tests load it too).

Per test it records ``openengine_chain``, ``engine_calls``, ``engine_name`` and
``chain_proof`` and fails a passing test whose proof fails, or whose servers
leave processes running when they stop. A test whose chain servers are workers
of a disaggregated server is proven leg by leg (``proof.judge_disagg``) and
gets the per-role markers too; while it runs, a thread follows the
disaggregated servers and the GPU use of the processes the test started. A
test that runs a disaggregated server whose workers have no role is not
proven (``incomplete``). With
``TRTLLM_TEST_OPENENGINE_NEGATIVE_CONTROL=<ctx|gen>`` (CH-9), the first worker
of that role starts outside the chain and the test passes only if CH-6 fails.
When ``TRTLLM_OPENENGINE_RESULTS_DIR`` is set and the pytest run writes no
junit file, it writes one there, so that the pytest a wrapper test starts still
leaves per-test results.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

import pytest

from . import conventions as cv
from . import expectations as ex
from . import launch, proof
from .launch import (
    ChainError,
    ChainStartupError,
    handles_ended_after,
    handles_since,
    next_seq,
    resolve_chain,
    tail,
)

_CHAIN = pytest.StashKey[str]()
_EXPECTATIONS = pytest.StashKey[list]()
_FOUND = pytest.StashKey[list]()
_SETUP_SEQ = pytest.StashKey[int]()
_TEARDOWN_SEQ = pytest.StashKey[int]()
_RESULT = pytest.StashKey["ProofResult"]()
_OUTER = pytest.StashKey[str]()
# Startup verdicts by the exception a failed fixture raised; pytest re-raises the same
# object for every test that uses the fixture.
_STARTUP_VERDICTS = pytest.StashKey[dict]()
_STARTUP_DECIDED = pytest.StashKey[bool]()

# Re-reads before a count mismatch is reported: the front end counts a call that was
# cancelled before its first event only when it finishes reading it, which can be late.
_SETTLE_TRIES = 4
_SETTLE_INTERVAL_S = 0.5
_LOG_TAIL_BYTES = 8192
_WATCH = pytest.StashKey["_Watch"]()
# How often the watch reads the disaggregated servers and the GPU use during a test.
_WATCH_INTERVAL_S = 1.0
_GPU_CHECK_INTERVAL_S = 5.0
# Names the role whose first worker starts outside the chain, for the CH-9 negative control.
ENV_NEGATIVE_CONTROL = "TRTLLM_TEST_OPENENGINE_NEGATIVE_CONTROL"
_NEGATIVE_CONTROL_UNDO = pytest.StashKey[object]()
# The default leg expectation, written in leg_expectation when a test registers none.
TWO_LEGS = "two_legs"


@dataclass
class ProofResult:
    status: str
    engine_calls: int | None = None
    engine_name: str | None = None
    details: list[str] = field(default_factory=list)
    handles: list = field(default_factory=list)
    # Processes left running by servers that stopped during the test.
    teardown_errors: list[str] = field(default_factory=list)
    # junit properties of a disaggregated test's proof, after the common ones.
    props: list[tuple[str, str]] = field(default_factory=list)
    # The proof of a disaggregated test.
    disagg: proof.DisaggProof | None = None


def _worst(statuses) -> str:
    statuses = list(statuses)
    if proof.FAIL in statuses:
        return proof.FAIL
    if proof.INCOMPLETE in statuses or not statuses:
        return proof.INCOMPLETE
    return proof.OK


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    chain = resolve_chain()
    config.stash[_CHAIN] = chain
    config.stash[_EXPECTATIONS] = ex.load() if chain != cv.CHAIN_NONE else []
    config.stash[_STARTUP_VERDICTS] = {}
    config.stash[_NEGATIVE_CONTROL_UNDO] = (
        _start_one_worker_outside_the_chain(negative_control_role())
        if chain != cv.CHAIN_NONE and negative_control_role()
        else None
    )
    # The test of an outer pytest that started this one; read before this run overwrites it.
    # "<node id> (<phase>)"; node ids can hold spaces (``-k "..."`` in a parameter).
    config.stash[_OUTER] = re.sub(
        r" \((?:setup|call|teardown)\)$", "", os.environ.get("PYTEST_CURRENT_TEST", "")
    )
    results = os.environ.get(cv.ENV_RESULTS_DIR)
    if results and getattr(config.option, "xmlpath", None) is None:
        directory = os.path.join(results, "junit")
        os.makedirs(directory, exist_ok=True)
        outer = re.sub(r"[^A-Za-z0-9_.-]+", "_", config.stash[_OUTER])[-120:]
        config.option.xmlpath = os.path.join(
            # The nanosecond clock orders a wrapper's rerun after its first run.
            directory,
            f"inner-{time.strftime('%Y%m%dT%H%M%S')}-{time.time_ns()}-{os.getpid()}-{outer}.xml",
        )


def pytest_unconfigure(config: pytest.Config) -> None:
    undo = config.stash.get(_NEGATIVE_CONTROL_UNDO, None)
    if undo is not None:
        undo()


def pytest_report_header(config: pytest.Config) -> str:
    return f"openengine_chain: {config.stash[_CHAIN]}"


def negative_control_role() -> str | None:
    """The role named by ``TRTLLM_TEST_OPENENGINE_NEGATIVE_CONTROL``; raises for another value."""
    role = os.environ.get(ENV_NEGATIVE_CONTROL, "")
    if role and role not in proof.ROLES:
        raise pytest.UsageError(
            f"{ENV_NEGATIVE_CONTROL} must be one of {proof.ROLES}, not {role!r}"
        )
    return role or None


def _start_outside_the_chain(cmd, env=None, sink=None) -> launch.ServeHandle:
    """Start ``cmd`` as it is, an in-process worker, behind a handle outside the chain."""
    cmd = [str(a) for a in cmd]
    proc = subprocess.Popen(
        cmd,
        env=None if env is None else dict(env),
        stdout=sink,
        stderr=None if sink is None else subprocess.STDOUT,
    )
    host = proof._option(cmd, "--host") or "localhost"
    port = proof._option(cmd, "--port")
    return launch.ServeHandle(
        chain=cv.CHAIN_NONE,
        model=cmd[1] if len(cmd) > 1 else "",
        host=host,
        port=int(port) if port and port.isdigit() else None,
        frontend=proc,
        engine=None,
        engine_endpoint=None,
        workdir=None,
        counter_dir=None,
        role=launch.ServeHandle.role_in(cmd),
    )


def _start_one_worker_outside_the_chain(role: str):
    """Make the harness start the first ``role`` worker as an in-process worker (CH-9).

    Replaces ``launch_serve_command`` in ``launch`` and in every module of this
    package already holding it; returns the function that undoes this. A module
    that imports it later keeps the replacement, which does nothing once undone.
    """
    original = launch.launch_serve_command
    used = []
    active = [True]

    def launch_serve_command(cmd, *args, **kwargs):
        if active[0] and not used and launch.ServeHandle.role_in(cmd) == role:
            used.append(cmd)
            return _start_outside_the_chain(cmd, kwargs.get("env"), kwargs.get("sink"))
        return original(cmd, *args, **kwargs)

    package = launch.__name__.rpartition(".")[0]
    holders = [launch] + [
        module
        for name, module in list(sys.modules.items())
        if module is not None
        and module is not launch
        and name.startswith(package + ".")
        and vars(module).get("launch_serve_command") is original
    ]
    for module in holders:
        module.launch_serve_command = launch_serve_command

    def undo() -> None:
        active[0] = False
        for module in holders:
            if vars(module).get("launch_serve_command") is launch_serve_command:
                module.launch_serve_command = original

    return undo


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    chain = config.stash[_CHAIN]
    expectations = config.stash[_EXPECTATIONS]
    for item in items:
        found = ex.applicable(expectations, item.nodeid, chain) if chain != cv.CHAIN_NONE else []
        item.stash[_FOUND] = found
        decide = ex.outcome_expectation(found)
        if decide is not None and decide.kind == ex.SKIP:
            item.add_marker(pytest.mark.skip(reason=f"{decide.wt}: {decide.reason}"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item: pytest.Item):
    item.stash[_SETUP_SEQ] = next_seq()
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item: pytest.Item):
    item.stash[_TEARDOWN_SEQ] = next_seq()
    yield


def _safe_counters(handle, details: list[str]) -> proof.Counters:
    try:
        return proof.read_counters(handle)
    except Exception as error:  # noqa: BLE001 - an unreadable counter only makes the proof incomplete
        details.append(f"reading counters failed: {error!r}")
        return proof.Counters(None, None)


def _chain_handles(seq: int) -> list:
    return [h for h in handles_since(seq) if h.chain != cv.CHAIN_NONE]


def _started_since(seq: int) -> list:
    """Chain servers this test's own setup or body started (not earlier tests' servers)."""
    return [h for h in _chain_handles(seq) if h.started_seq > seq]


class _Watch:
    """Follows the disaggregated servers and the workers below this pytest during a test's call.

    It reads the disaggregated servers when started, when stopped and every
    ``_WATCH_INTERVAL_S`` in between from a thread, so that one the test starts
    and stops is still seen. For a test with worker roles it also looks for
    in-process workers holding a GPU, and takes each worker's topology and
    engine information while it runs (for a test that kills a worker), every
    ``_GPU_CHECK_INTERVAL_S`` and when stopped.
    """

    def __init__(self, seq: int, root: int | None = None, disagg: bool = False) -> None:
        self._seq = seq
        # The test is known to run a disaggregated server even if its workers have no role.
        self._disagg = disagg
        self._root = os.getpid() if root is None else root
        # Per server address: the counters the increase is taken from, and the last reading.
        self._base: dict[str, dict[str, int]] = {}
        self._last: dict[str, proof.OrchestratorReading] = {}
        # Servers found when the test started, those whose increase is only a lower bound
        # (not read when the test started, or restarted during it), and those alive at the end.
        self._present_at_start: set[str] = set()
        self._lower_bound: set[str] = set()
        self._alive_at_end: set[str] = set()
        self._inprocess: set[str] | None = None
        self._last_gpu_check = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._poll(at_start=True)
        self._thread = threading.Thread(
            target=self._run, name="openengine-chain-watch", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(_WATCH_INTERVAL_S):
            self._poll()

    def _poll(self, at_start: bool = False, at_end: bool = False) -> None:
        try:
            orchestrators = proof.find_orchestrators(self._root)
        except Exception:  # noqa: BLE001 - a failed read only leaves the proof incomplete
            orchestrators = []
        for orchestrator in orchestrators:
            url = orchestrator.url
            if at_start:
                self._present_at_start.add(url)
            try:
                reading = proof.read_orchestrator(orchestrator)
            except Exception:  # noqa: BLE001 - not serving yet, or stopping
                continue
            with self._lock:
                self._record(url, reading, at_start)
                if at_end:
                    self._alive_at_end.add(url)
        handles = _chain_handles(self._seq)
        disagg = self._disagg or any(getattr(h, "role", None) for h in handles)
        now = time.monotonic()
        if disagg and (at_end or now - self._last_gpu_check >= _GPU_CHECK_INTERVAL_S):
            self._last_gpu_check = now
            for handle in handles:
                if getattr(handle, "role", None) and handle.alive:
                    _topology(handle)
                    _engine_info(handle, [])
            try:
                found = proof.inprocess_workers(self._root)
            except Exception:  # noqa: BLE001
                found = None
            if found is not None:
                with self._lock:
                    self._inprocess = (self._inprocess or set()) | set(found)

    def _record(self, url: str, reading: proof.OrchestratorReading, at_start: bool) -> None:
        last = self._last.get(url)
        if url not in self._base:
            if at_start or url not in self._present_at_start:
                # Read when the test started, or started during the test (counting from 0).
                self._base[url] = dict(reading.counts) if at_start else {}
            else:
                # Ran before the test but was not read then: count from this reading on.
                self._base[url] = dict(reading.counts)
                self._lower_bound.add(url)
        elif last is not None and any(reading.counts[k] < v for k, v in last.counts.items()):
            # Restarted: its counters start again from 0, and what it sent before is lost.
            self._base[url] = {}
            self._lower_bound.add(url)
        self._last[url] = reading

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._poll(at_end=True)

    def orchestrator(self) -> proof.OrchestratorReading | None:
        """The increase of every disaggregated server's counters during the test; ``None`` if none ran."""
        with self._lock:
            if not self._last:
                return None
            readings = []
            for url, last in self._last.items():
                base = self._base[url]
                members = None
                if last.members is not None:
                    members = {
                        r: tuple(proof.normalize_address(a) for a in last.members[r])
                        for r in proof.ROLES
                    }
                readings.append(
                    proof.OrchestratorReading(
                        {k: v - base.get(k, 0) for k, v in last.counts.items()},
                        members,
                        stale=url in self._lower_bound or url not in self._alive_at_end,
                    )
                )
            return proof.add_orchestrator_readings(readings)

    def inprocess(self) -> list[str] | None:
        with self._lock:
            return None if self._inprocess is None else sorted(self._inprocess)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    if item.config.stash[_CHAIN] == cv.CHAIN_NONE or item.stash.get(_STARTUP_DECIDED, False):
        yield
        return
    seq = next_seq()
    details: list[str] = []
    before = {id(h): _safe_counters(h, details) for h in _chain_handles(seq)}
    watch = _Watch(seq, disagg=ex.leg_expectation(item.stash.get(_FOUND, [])) is not None)
    try:
        yield
    finally:
        watch.stop()
    item.stash[_RESULT] = _prove(item, seq, before, details, watch)


def _topology(handle) -> proof.Check:
    """The topology of a running server, checked until it is once ``ok``; else the one at shutdown."""
    if not handle.alive:
        return handle.final_topology or proof.Check(proof.INCOMPLETE, "not checked before shutdown")
    if handle.proven_topology is not None:
        return handle.proven_topology
    try:
        topology = proof.check_topology(handle)
    except Exception as error:  # noqa: BLE001 - a failed check only makes the proof incomplete
        return proof.Check(proof.INCOMPLETE, f"topology check failed: {error!r}")
    if topology.status == proof.OK:
        handle.proven_topology = topology
    return topology


def _prove(
    item: pytest.Item, seq: int, before: dict, details: list[str], watch: _Watch | None = None
) -> ProofResult:
    handles = _chain_handles(seq)
    # Tests expected to be rejected by the server send nothing to the engine either.
    found = item.stash.get(_FOUND, [])
    whitelisted = any(
        e.kind in (ex.NO_ENGINE_TRAFFIC, ex.HTTP_STATUS, ex.STARTUP_REJECTED) for e in found
    )
    cancels_calls = any(e.kind == ex.CANCELS_CALLS for e in found)
    if not handles:
        if item.config.stash[_OUTER] and not whitelisted:
            # A pytest started by a wrapper test runs tests against a chain server; one
            # that sent nothing anywhere is not proven to have used the chain.
            return ProofResult(
                proof.FAIL,
                details=[
                    "no chain server ran in this process during the test, and the test's "
                    "expectations do not say it sends no Generate call"
                ],
            )
        return ProofResult(
            "no_server", details=["no chain server ran in this process during the test"]
        )
    rule = _leg_rule(found, whitelisted, cancels_calls)
    roles = _roles(handles, rule)
    if any(roles.values()):
        return _prove_disagg(handles, roles, before, details, watch, rule)
    # Counters start at 0 when a process starts; a server started during the test starts there.
    # A test that cancels calls is held to equal counts until the last read, so an engine
    # count that arrives during the re-reads is still this test's.
    for attempt in range(_SETTLE_TRIES):
        last = attempt == _SETTLE_TRIES - 1
        ends = [_safe_counters(h, details) for h in handles]
        starts = [
            proof.counting_start(before.get(id(h), proof.ZERO), end)
            for h, end in zip(handles, ends)
        ]
        counts = proof.check_server_counts(
            starts, ends, no_engine_traffic=whitelisted, cancels_calls=cancels_calls and last
        )
        if not counts.mismatch or last:
            break
        time.sleep(_SETTLE_INTERVAL_S)
    statuses = [counts.status]
    details.append(f"counts: {counts.status}: {counts.detail}")
    names = set()
    teardown_errors: list[str] = []
    for h in handles:
        topology = _topology(h)
        if h.alive:
            try:
                names.add(proof.read_engine_name(h))
            except Exception as error:  # noqa: BLE001
                details.append(f"reading engine name failed: {error!r}")
        else:
            names.add(h._engine_name)
            teardown_errors.extend(h.teardown_errors)
            details.extend(f"teardown: {n}" for n in h.teardown_notes)
        statuses.append(topology.status)
        details.append(f"topology: {topology.status}: {topology.detail}")
    if watch is not None and watch.orchestrator() is not None:
        # Its workers are chain servers without a worker role: neither proof applies.
        statuses.append(proof.INCOMPLETE)
        details.append(
            "a disaggregated server ran, but no chain server is one of its workers by role "
            "(--server_role); its legs are not proven"
        )
    names.discard(None)
    return ProofResult(
        _worst(statuses),
        engine_calls=counts.engine_calls,
        engine_name=",".join(sorted(names)) or None,
        # A counter that fails to read fails again on each settle attempt: report it once.
        details=list(dict.fromkeys(details)),
        handles=handles,
        teardown_errors=teardown_errors,
    )


def _leg_rule(found, whitelisted: bool, cancels_calls: bool) -> proof.LegRule:
    entry = ex.leg_expectation(found)
    if entry is None:
        return proof.LegRule(no_engine_traffic=whitelisted, cancels_calls=cancels_calls)
    return proof.LegRule(
        leg=entry.leg,
        killed=entry.killed,
        orphans=entry.orphans,
        direct_requests=entry.direct_requests,
        no_engine_traffic=whitelisted,
        cancels_calls=cancels_calls,
    )


def _roles(handles, rule: proof.LegRule) -> dict[int, str | None]:
    """Each handle's worker role.

    A cross_node test's worker without one takes its node's: the context worker
    runs on node 0 and the generation worker on node 1.
    """
    node_role = None
    if rule.leg == ex.CROSS_NODE:
        node_role = {"0": proof.CTX, "1": proof.GEN}.get(os.environ.get("SLURM_NODEID", ""))
    return {id(h): getattr(h, "role", None) or node_role for h in handles}


def _engine_info(handle, details: list[str]) -> proof.EngineInfo | None:
    try:
        return proof.engine_info(handle)
    except Exception as error:  # noqa: BLE001 - an unreadable engine only makes the proof incomplete
        details.append(f"reading the engine's server info failed: {error!r}")
        return None


def _address(handle) -> str | None:
    """The worker front end's ``host:port``; for one started on port 0, the port it listens on.

    ``None`` if that port is not known (CH-6's member check is then incomplete).
    """
    port = handle.port
    if not port:
        frontend = getattr(handle, "frontend", None)
        if frontend is None or not handle.alive:
            return None
        try:
            ports = proof.listening_ports(launch.live_process_tree(frontend.pid))
        except OSError:
            return None
        if len(ports) != 1:
            return None
        port = ports.pop()
    return proof.normalize_address(f"{handle.host}:{port}")


def _settles(result: proof.DisaggProof) -> bool:
    """Whether no count of a side is behind the other's: the counts CH-2 and CH-4 compare per worker."""
    return not any(
        name in ("CH-2", "CH-4") and role is not None and c.status == proof.FAIL
        for name, role, c in result.checks
    )


def _prove_disagg(
    handles, roles: dict, before: dict, details: list[str], watch, rule
) -> ProofResult:
    """The proof of a test whose chain servers are workers of a disaggregated server."""
    orchestrator = watch.orchestrator() if watch is not None else None
    inprocess = watch.inprocess() if watch is not None else None
    names = set()
    teardown_errors: list[str] = []
    for h in handles:
        if h.alive:
            try:
                names.add(proof.read_engine_name(h))
            except Exception as error:  # noqa: BLE001
                details.append(f"reading engine name failed: {error!r}")
        else:
            names.add(h._engine_name)
            teardown_errors.extend(h.teardown_errors)
            details.extend(f"teardown: {n}" for n in h.teardown_notes)
    topologies = {id(h): _topology(h) for h in handles}
    infos = {id(h): _engine_info(h, details) for h in handles}
    for attempt in range(_SETTLE_TRIES):
        legs = []
        for h in handles:
            end = _safe_counters(h, details)
            start = before.get(id(h))
            start = proof.zero_like(end) if start is None else proof.counting_start(start, end)
            role = roles[id(h)] or "agg"
            legs.append(proof.Leg(role, _address(h), start, end, infos[id(h)], topologies[id(h)]))
        unknown = sorted({leg.role for leg in legs} - set(proof.ROLES))
        if unknown:
            # An aggregated server next to the workers is not part of a two-leg proof.
            result = proof.DisaggProof(
                proof.FAIL,
                {r: proof.FAIL for r in proof.ROLES},
                [
                    (
                        "CH-1",
                        None,
                        proof.Check(
                            proof.FAIL,
                            "a chain server without a worker role "
                            "runs next to the workers of a disaggregated server",
                        ),
                    )
                ],
                {"roles": proof.role_sums(legs), "orchestrator": None},
            )
            break
        last = attempt == _SETTLE_TRIES - 1
        result = proof.judge_disagg(legs, orchestrator, inprocess, rule, last=last)
        if _settles(result) or last:
            break
        time.sleep(_SETTLE_INTERVAL_S)
    names.discard(None)
    leg = rule.leg or TWO_LEGS
    met = {proof.OK: "met", proof.FAIL: "not met"}.get(result.status, result.status)
    if result.deferred and result.status == proof.OK:
        met = "met on this node"
    sums = result.counts["roles"]
    received = [sums[r]["counts"]["received"] for r in proof.ROLES]
    result.counts["node_names"] = proof.node_names()
    props = list(proof.disagg_props(result.counts, result.roles, partial=result.deferred).items())
    props += [
        ("leg_expectation", f"{leg}:{met}"),
        ("chain_leg_counts", json.dumps(result.counts, sort_keys=True)),
    ]
    return ProofResult(
        result.status,
        engine_calls=None if None in received else sum(received),
        engine_name=",".join(sorted(names)) or None,
        details=list(dict.fromkeys(details + result.details())),
        handles=handles,
        teardown_errors=teardown_errors,
        props=props,
        disagg=result,
    )


def _judge_negative_control(report: pytest.TestReport, props: list, result: ProofResult) -> None:
    """CH-9: with a worker outside the chain, the test passes only if CH-6 fails.

    The test's own outcome does not count, as the worker outside the chain may
    change what it sees; processes its servers left running still fail it.
    """
    failed = result.disagg is not None and any(
        name == "CH-6" and check.status == proof.FAIL for name, _, check in result.disagg.checks
    )
    props.append(
        ("negative_control", f"{negative_control_role()}:{'met' if failed else 'not met'}")
    )
    if failed and result.teardown_errors:
        _fail(
            report,
            "OpenEngine chain shutdown left processes running:\n"
            + "\n".join(result.teardown_errors),
        )
    elif failed:
        _pass(report)
    else:
        _fail(
            report,
            f"negative control ({ENV_NEGATIVE_CONTROL}): a worker ran outside the chain, but CH-6 "
            f"did not fail (chain_proof {result.status}):\n" + "\n".join(result.details),
        )
        _attach_logs(report, result.handles)


def _append_text(report: pytest.TestReport, title: str, text: str) -> None:
    longrepr = report.longrepr
    if hasattr(longrepr, "addsection"):
        longrepr.addsection(title, text)
    else:
        report.longrepr = f"{longrepr or ''}\n\n----- {title} -----\n{text}"


def _fail(report: pytest.TestReport, message: str) -> None:
    report.outcome = "failed"
    report.longrepr = message


def _pass(report: pytest.TestReport) -> None:
    report.outcome = "passed"
    report.longrepr = None


def _attach_logs(report: pytest.TestReport, handles) -> None:
    for h in handles:
        for role, path in (("engine", h.engine_log), ("frontend", h.frontend_log)):
            if path:
                _append_text(
                    report,
                    f"openengine_chain {role} log {path} (tail)",
                    tail(path, _LOG_TAIL_BYTES),
                )


def _is_startup_failure(error: BaseException) -> bool:
    """The harness's startup error, or the RuntimeError a test raises when its server exits."""
    return isinstance(error, ChainStartupError) or (
        isinstance(error, RuntimeError) and not isinstance(error, ChainError)
    )


def _startup_rejected(
    expectation: ex.Expectation, error: BaseException | None, handles
) -> tuple[bool, str]:
    if error is None:
        return False, "the server started"
    if not _is_startup_failure(error):
        return False, f"failed with {error!r}, not a server startup failure"
    logs = [path for h in handles for path in (h.engine_log, h.frontend_log) if path]
    if not any(expectation.log_contains in tail(path, 1 << 20) for path in logs):
        return False, f"no server log contains {expectation.log_contains!r} ({logs})"
    return True, ""


def _report_teardown(item: pytest.Item, report: pytest.TestReport) -> None:
    """Problems of servers that stopped while fixtures were torn down."""
    handles = [
        h for h in handles_ended_after(item.stash[_TEARDOWN_SEQ]) if h.chain != cv.CHAIN_NONE
    ]
    errors = [e for h in handles for e in h.teardown_errors]
    notes = [n for h in handles for n in h.teardown_notes]
    if errors or notes:
        prop = ("chain_teardown_problems", "; ".join(errors + notes))
        # junit takes the properties of the teardown report, copied when it was made.
        item.user_properties.append(prop)
        report.user_properties.append(prop)
    if errors and report.passed:
        _fail(report, "OpenEngine chain shutdown left processes running:\n" + "\n".join(errors))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report: pytest.TestReport = outcome.get_result()
    chain = item.config.stash[_CHAIN]
    props = item.user_properties
    if report.when == "setup":
        props.append(("openengine_chain", chain))
        outer = item.config.stash[_OUTER]
        if outer and cv.ENV_RESULTS_DIR in os.environ:
            props.append(("outer_test", outer))
    if chain == cv.CHAIN_NONE or report.skipped:
        return
    if report.when == "teardown":
        _report_teardown(item, report)
        return
    decide = ex.outcome_expectation(item.stash.get(_FOUND, []))
    error = call.excinfo.value if call.excinfo is not None else None

    if report.when == "setup":
        if not report.failed:
            return
        handles = _started_since(item.stash[_SETUP_SEQ])
        errors = [e for h in handles for e in h.teardown_errors]
        if decide is None or decide.kind != ex.STARTUP_REJECTED:
            # The setup already failed; the problems only need to be reported.
            if errors:
                props.append(("chain_teardown_problems", "; ".join(errors)))
                _append_text(report, "openengine_chain teardown", "\n".join(errors))
            _attach_logs(report, handles)
            return
        # A fixture that failed to start the server.
        verdicts = item.config.stash[_STARTUP_VERDICTS]
        if id(error) in verdicts:
            ok, why = verdicts[id(error)][1:]
        else:
            ok, why = _startup_rejected(decide, error, handles)
            # Keeps the exception alive so that its id is not reused.
            verdicts[id(error)] = (error, ok, why)
        props.append(("expectation", f"{decide.wt}:{decide.kind}:{'met' if ok else 'not met'}"))
        if errors:
            props.append(("chain_teardown_problems", "; ".join(errors)))
        if ok and errors:
            _fail(
                report,
                "the server failed to start as expected, but its shutdown left processes "
                "running:\n" + "\n".join(errors),
            )
        elif ok:
            item.stash[_STARTUP_DECIDED] = True
            _pass(report)
            # The server never started, so there is nothing to run.
            item.runtest = lambda: None
        else:
            _append_text(
                report,
                "openengine_chain expectation",
                f"expected startup rejection ({decide.wt}): {why}",
            )
            _attach_logs(report, handles)
        return

    result: ProofResult | None = item.stash.get(_RESULT, None)
    if result is not None:
        props.append(("chain_proof", result.status))
        props.append(
            (
                "engine_calls",
                "incomplete" if result.engine_calls is None else str(result.engine_calls),
            )
        )
        props.append(("engine_name", result.engine_name or "unknown"))
        if result.details:
            props.append(("chain_proof_detail", "; ".join(result.details)))
        if result.teardown_errors:
            props.append(("chain_teardown_problems", "; ".join(result.teardown_errors)))
        if result.props:
            props.extend(result.props)
        elif result.handles:
            props.append(("chain_topology", "agg"))

    if result is not None and negative_control_role():
        _judge_negative_control(report, props, result)
        return

    if decide is not None and decide.kind == ex.HTTP_STATUS:
        status = ex.http_status_of(error) if error is not None else None
        met = status == decide.status
        props.append(("expectation", f"{decide.wt}:{decide.kind}:{'met' if met else 'not met'}"))
        if met:
            _pass(report)
        elif report.passed:
            _fail(
                report,
                f"expected HTTP {decide.status} ({decide.wt}: {decide.reason}) but the test passed",
            )
        else:
            _append_text(
                report,
                "openengine_chain expectation",
                f"expected HTTP {decide.status} ({decide.wt}); got {error!r}",
            )
    elif (
        decide is not None
        and decide.kind == ex.STARTUP_REJECTED
        and not item.stash.get(_STARTUP_DECIDED, False)
    ):
        # A server started by the test itself, or a fixture's server that did start.
        handles = _started_since(item.stash[_SETUP_SEQ])
        if any(h.alive for h in handles):
            ok, why = False, "a chain server is running"
        else:
            ok, why = _startup_rejected(decide, error, handles)
        props.append(("expectation", f"{decide.wt}:{decide.kind}:{'met' if ok else 'not met'}"))
        if ok:
            _pass(report)
        elif report.passed:
            _fail(
                report,
                f"expected the server to fail to start ({decide.wt}: {decide.reason}); {why}",
            )

    if result is not None and report.passed:
        if result.status == proof.FAIL:
            _fail(report, "OpenEngine chain proof failed:\n" + "\n".join(result.details))
        elif result.teardown_errors:
            _fail(
                report,
                "OpenEngine chain shutdown left processes running:\n"
                + "\n".join(result.teardown_errors),
            )
    if report.failed and result is not None:
        _attach_logs(report, result.handles)
