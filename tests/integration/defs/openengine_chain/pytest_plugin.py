# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin: chain proof, junit markers and expectation tables for every test.

Load it with ``-p openengine_chain.pytest_plugin`` (the runner sets
``PYTEST_ADDOPTS`` so that pytest processes started by tests load it too).

Per test it records ``openengine_chain``, ``engine_calls``, ``engine_name`` and
``chain_proof`` and fails a passing test whose proof fails, or whose servers
leave processes running when they stop. When
``TRTLLM_OPENENGINE_RESULTS_DIR`` is set and the pytest run writes no junit
file, it writes one there, so that the pytest a wrapper test starts still
leaves per-test results.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import pytest

from . import conventions as cv
from . import expectations as ex
from . import proof
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


@dataclass
class ProofResult:
    status: str
    engine_calls: int | None = None
    engine_name: str | None = None
    details: list[str] = field(default_factory=list)
    handles: list = field(default_factory=list)
    # Processes left running by servers that stopped during the test.
    teardown_errors: list[str] = field(default_factory=list)


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


def pytest_report_header(config: pytest.Config) -> str:
    return f"openengine_chain: {config.stash[_CHAIN]}"


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    if item.config.stash[_CHAIN] == cv.CHAIN_NONE or item.stash.get(_STARTUP_DECIDED, False):
        yield
        return
    seq = next_seq()
    details: list[str] = []
    before = {id(h): _safe_counters(h, details) for h in _chain_handles(seq)}
    yield
    item.stash[_RESULT] = _prove(item, seq, before, details)


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


def _prove(item: pytest.Item, seq: int, before: dict, details: list[str]) -> ProofResult:
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
