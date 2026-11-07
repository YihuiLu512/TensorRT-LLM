# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pytest plugin end to end: proof, junit markers and expectation tables on generated tests."""

import glob
import os
import xml.etree.ElementTree as ET

import plugin_support
import pytest
from openengine_chain import conventions as cv
from openengine_chain import expectations as ex

pytestmark = pytest.mark.cpu_only

E = ex.Expectation
TABLE = [
    E("WT-10", "test_gen.py::test_whitelisted", ex.NO_ENGINE_TRAFFIC, "sends nothing"),
    E("WT-11", "test_gen.py::test_400*", ex.HTTP_STATUS, "X-a", status=400),
    E("WT-11", "test_gen.py::test_not_for_chain", ex.SKIP, "X-b"),
    E(
        "WT-11",
        "test_rejected.py::test_*",
        ex.STARTUP_REJECTED,
        "X-a",
        log_contains="max_beam_width",
    ),
    E(
        "WT-11",
        "test_mixed.py::test_rejected",
        ex.STARTUP_REJECTED,
        "X-a",
        log_contains="max_beam_width",
    ),
]

GENERATED = """
import pytest
from plugin_support import FakeHandle

class HttpError(Exception):
    def __init__(self, status):
        self.status_code = status

@pytest.fixture(scope="module")
def server():
    h = FakeHandle()
    yield h
    h.end()

def test_ok(server):
    server.send(2)

def test_mismatch(server):
    server.send(2, 1)

def test_zero(server):
    pass

def test_whitelisted(server):
    pass

def test_400_met(server):
    raise HttpError(400)

def test_400_but_passed(server):
    server.send(1)

def test_400_other_error(server):
    raise HttpError(500)

def test_not_for_chain(server):
    raise AssertionError("must not run")

def test_server_started_in_test():
    h = FakeHandle()
    h.send(3)
    h.end()

def test_incomplete():
    h = FakeHandle(frontend=None)
    h.send(1)
    h.end()

@pytest.fixture(scope="module")
def fresh_server():
    h = FakeHandle(unwritten=True)
    yield h
    h.end()

def test_first_counter_file(server, fresh_server):
    server.send(1)
    fresh_server.send(2)
"""

REJECTED = """
import pytest
from plugin_support import FakeHandle

@pytest.fixture(scope="module")
def server(tmp_path_factory):
    log = tmp_path_factory.mktemp("logs") / "frontend.log"
    log.write_text("ValueError: {word} > 1 is not supported")
    h = FakeHandle(frontend_log=str(log))
    h.end()
    raise RuntimeError("Server exited unexpectedly.")

def test_a(server):
    raise AssertionError("must not run")

def test_b(server):
    raise AssertionError("must not run")
"""


@pytest.fixture
def run(pytester, monkeypatch, tmp_path):
    plugin_support.install(monkeypatch)
    monkeypatch.setattr(ex, "load", lambda directory=None: list(TABLE))
    monkeypatch.setenv(cv.ENV_CHAIN, "trtllm")
    monkeypatch.delenv(cv.ENV_RESULTS_DIR, raising=False)
    monkeypatch.syspath_prepend(os.path.dirname(os.path.abspath(plugin_support.__file__)))

    def run(*args, outer_test=None):
        # The generated tests run as an outer pytest unless they are given the test that
        # started them; the variable pytest sets for the running test is not that.
        if outer_test is None:
            monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        else:
            monkeypatch.setenv("PYTEST_CURRENT_TEST", f"{outer_test} (call)")
        xml = tmp_path / f"junit-{len(os.listdir(tmp_path))}.xml"
        result = pytester.runpytest_inprocess("-p", cv.PLUGIN_MODULE, f"--junitxml={xml}", *args)
        return result, _cases(str(xml))

    return run


def _cases(path):
    cases = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        outcome = "passed"
        for tag, name in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
            node = case.find(tag)
            if node is not None:
                outcome = name
                props["_text"] = (node.get("message") or "") + (node.text or "")
                break
        cases[case.get("name")] = (outcome, props)
    return cases


def test_proof_and_expectations(pytester, run):
    pytester.makepyfile(test_gen=GENERATED, test_gen_z="def test_no_server():\n    pass\n")
    _, cases = run()
    expected = {
        "test_ok": ("passed", "ok", "2"),
        "test_mismatch": ("failed", "fail", "1"),
        "test_zero": ("failed", "fail", "0"),
        "test_whitelisted": ("passed", "ok", "0"),
        "test_400_met": ("passed", "ok", "0"),
        "test_400_but_passed": ("failed", "ok", "1"),
        "test_400_other_error": ("failed", "ok", "0"),
        "test_server_started_in_test": ("passed", "ok", "3"),
        "test_incomplete": ("passed", "incomplete", "1"),
        # The counter file a server writes with its first call shows it had sent none before.
        "test_first_counter_file": ("passed", "ok", "3"),
        "test_no_server": ("passed", "no_server", None),
    }
    for name, (outcome, status, calls) in expected.items():
        got_outcome, props = cases[name]
        assert got_outcome == outcome, (name, props)
        assert props["openengine_chain"] == "trtllm"
        assert props.get("chain_proof") == status, (name, props)
        if calls is not None:
            assert props["engine_calls"] == calls, (name, props)
    assert cases["test_not_for_chain"][0] == "skipped"
    assert "WT-11: X-b" in cases["test_not_for_chain"][1]["_text"]
    assert cases["test_400_met"][1]["expectation"] == "WT-11:http_status:met"
    assert "expected HTTP 400" in cases["test_400_but_passed"][1]["_text"]
    assert "chain proof failed" in cases["test_mismatch"][1]["_text"]
    assert cases["test_ok"][1]["engine_name"] == "fake-engine"


@pytest.mark.parametrize("word, outcome", [("max_beam_width", "passed"), ("other", "error")])
def test_startup_rejection(pytester, run, word, outcome):
    pytester.makepyfile(test_rejected=REJECTED.replace("{word}", word))
    _, cases = run()
    for name in ("test_a", "test_b"):
        assert cases[name][0] == outcome, cases[name]
        assert cases[name][1]["expectation"].endswith("met" if outcome == "passed" else "not met")


STARTED = """
import pytest
from plugin_support import FakeHandle

@pytest.fixture(scope="module")
def server():
    h = FakeHandle()
    yield h
    h.end()

def test_passes(server):
    server.send(1)

def test_skips(server):
    pytest.skip("not here")
"""


def test_startup_rejection_of_a_server_that_started_is_not_met(pytester, run):
    pytester.makepyfile(test_rejected=STARTED)
    _, cases = run()
    outcome, props = cases["test_passes"]
    assert outcome == "failed"
    assert props["expectation"] == "WT-11:startup_rejected:not met"
    assert "a chain server is running" in props["_text"]
    # A skip stays a skip whatever the expectation.
    assert cases["test_skips"][0] == "skipped"


TEARDOWN = """
import pytest
from plugin_support import FakeHandle

@pytest.fixture(scope="module")
def server():
    h = FakeHandle()
    yield h
    h.end(errors=["engine pid 7 still running after SIGKILL"])

def test_stops_its_own_server():
    h = FakeHandle()
    h.send(1)
    h.end(errors=["frontend pid 8 still running after SIGKILL"])

def test_first(server):
    server.send(1)

def test_last(server):
    server.send(1)
"""


def test_processes_left_at_shutdown_fail_the_test_that_stopped_the_server(pytester, run):
    pytester.makepyfile(test_gen=TEARDOWN)
    result, cases = run()
    assert cases["test_first"][0] == "passed"
    # The module fixture stops after the last test: its teardown errors.
    result.assert_outcomes(passed=2, failed=1, errors=1)
    assert "pid 7 still running" in cases["test_last"][1]["chain_teardown_problems"]
    outcome, props = cases["test_stops_its_own_server"]
    assert outcome == "failed"
    assert "pid 8 still running" in props["chain_teardown_problems"]


TOPOLOGY = """
import pytest
from plugin_support import FakeHandle

@pytest.fixture(scope="module")
def server():
    h = FakeHandle()
    yield h
    h.end()

def test_first(server):
    server.send(1)

def test_second(server):
    server.topology = None  # A second check would fail on this.
    server.send(1)
"""


def test_topology_is_checked_until_it_is_proven(pytester, run):
    pytester.makepyfile(test_gen=TOPOLOGY)
    _, cases = run()
    assert cases["test_second"][0] == "passed"
    assert cases["test_second"][1]["chain_proof"] == "ok"


def test_inner_run_test_without_a_server_fails(pytester, run):
    pytester.makepyfile(
        test_gen="def test_whitelisted():\n    pass\n\ndef test_no_server():\n    pass\n"
    )
    _, cases = run(outer_test="test_e2e.py::test_outer")
    assert cases["test_no_server"][0] == "failed"
    assert cases["test_no_server"][1]["chain_proof"] == "fail"
    assert cases["test_whitelisted"][0] == "passed"
    assert cases["test_whitelisted"][1]["chain_proof"] == "no_server"


MIXED = """
import pytest
from plugin_support import FakeHandle

@pytest.fixture(scope="module")
def running():
    h = FakeHandle()
    yield h
    h.end()

@pytest.fixture
def rejected(tmp_path):
    log = tmp_path / "frontend.log"
    log.write_text("ValueError: max_beam_width > 1 is not supported")
    h = FakeHandle(frontend_log=str(log))
    h.end(errors={errors})
    raise RuntimeError("Server exited unexpectedly.")

def test_uses_running(running):
    running.send(1)

def test_rejected(running, rejected):
    raise AssertionError("must not run")
"""


@pytest.mark.parametrize(
    "errors, outcome",
    [("[]", "passed"), ('["engine pid 9 still running after SIGKILL"]', "error")],
)
def test_startup_rejection_ignores_servers_of_earlier_tests(pytester, run, errors, outcome):
    pytester.makepyfile(test_mixed=MIXED.replace("{errors}", errors))
    _, cases = run()
    assert cases["test_uses_running"][0] == "passed"
    got, props = cases["test_rejected"]
    assert got == outcome, props
    assert props["expectation"] == "WT-11:startup_rejected:met"
    if outcome == "error":
        assert "pid 9 still running" in props["chain_teardown_problems"]


def test_failed_start_without_an_expectation_reports_its_shutdown_problems(pytester, run):
    # test_unexpected.py has no expectation entry: the setup error stands.
    body = MIXED.replace("{errors}", '["engine pid 9 still running after SIGKILL"]')
    pytester.makepyfile(test_unexpected=body)
    _, cases = run()
    got, props = cases["test_rejected"]
    assert got == "error" and "expectation" not in props
    assert "pid 9 still running" in props["chain_teardown_problems"]


def test_without_a_chain_only_the_marker_is_recorded(pytester, run, monkeypatch):
    monkeypatch.delenv(cv.ENV_CHAIN)
    pytester.makepyfile(test_gen="def test_a():\n    pass\n")
    _, cases = run()
    outcome, props = cases["test_a"]
    assert outcome == "passed"
    assert props == {"openengine_chain": "none"}


def test_results_dir_gets_a_junit_file_when_pytest_writes_none(pytester, monkeypatch, tmp_path):
    plugin_support.install(monkeypatch)
    results = tmp_path / "results"
    monkeypatch.setenv(cv.ENV_RESULTS_DIR, str(results))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "test_e2e.py::test_outer (call)")
    monkeypatch.delenv(cv.ENV_CHAIN, raising=False)
    pytester.makepyfile(test_inner="def test_a():\n    pass\n")
    pytester.runpytest_inprocess("-p", cv.PLUGIN_MODULE).assert_outcomes(passed=1)
    files = glob.glob(str(results / "junit" / "inner-*-test_e2e.py_test_outer.xml"))
    assert len(files) == 1
    props = _cases(files[0])["test_a"][1]
    assert props["outer_test"] == "test_e2e.py::test_outer"


def test_outer_test_keeps_spaces_of_its_node_id(pytester, monkeypatch, tmp_path):
    plugin_support.install(monkeypatch)
    results = tmp_path / "results"
    outer = 'test_unittests.py::test_unittests_v2[unittest/a.py -k "b and c"]'
    monkeypatch.setenv(cv.ENV_RESULTS_DIR, str(results))
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"{outer} (call)")
    monkeypatch.delenv(cv.ENV_CHAIN, raising=False)
    pytester.makepyfile(test_inner="def test_a():\n    pass\n")
    pytester.runpytest_inprocess("-p", cv.PLUGIN_MODULE).assert_outcomes(passed=1)
    (path,) = glob.glob(str(results / "junit" / "inner-*.xml"))
    assert _cases(path)["test_a"][1]["outer_test"] == outer
