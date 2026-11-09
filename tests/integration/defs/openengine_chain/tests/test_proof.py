# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Topology and count decisions, with the process and RPC readers replaced."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import types
import uuid

import pytest
from openengine_chain import proof

pytestmark = pytest.mark.cpu_only

C = proof.Counters


@pytest.mark.parametrize(
    "before, after, whitelisted, status, calls",
    [
        (C(0, 0), C(3, 3), False, proof.OK, 3),
        (C(2, 5), C(4, 7), False, proof.OK, 2),
        (C(0, 0), C(3, 2), False, proof.FAIL, 2),
        (C(0, 0), C(0, 0), False, proof.FAIL, 0),
        (C(0, 0), C(0, 0), True, proof.OK, 0),
        (C(None, 0), C(None, 4), False, proof.INCOMPLETE, 4),
        # Only the engine's counter gives the engine's calls.
        (C(0, None), C(4, None), False, proof.INCOMPLETE, None),
        (C(None, None), C(None, None), False, proof.INCOMPLETE, None),
    ],
)
def test_check_counts(before, after, whitelisted, status, calls):
    result = proof.check_counts(before, after, no_engine_traffic=whitelisted)
    assert (result.status, result.engine_calls) == (status, calls)


@pytest.mark.parametrize(
    "after, cancels_calls, status, mismatch",
    [
        # Every test not listed as cancelling calls needs equal increases.
        (C(5, 3), False, proof.FAIL, True),
        (C(3, 3), False, proof.OK, False),
        # A test that cancels calls may count more calls on the front end ...
        (C(5, 3), True, proof.OK, False),
        (C(3, 3), True, proof.OK, False),
        # ... but not more on the engine, and still needs a call to reach the engine.
        (C(3, 5), True, proof.FAIL, True),
        (C(2, 0), True, proof.FAIL, False),
    ],
)
def test_only_tests_that_cancel_calls_may_count_more_on_the_front_end(
    after, cancels_calls, status, mismatch
):
    result = proof.check_counts(
        C(0, 0), after, no_engine_traffic=False, cancels_calls=cancels_calls
    )
    assert (result.status, result.mismatch, result.engine_calls) == (
        status,
        mismatch,
        after.engine,
    )


def test_a_cancelling_test_is_checked_on_each_server():
    # The total has more calls on the front end, but the second server's engine
    # received a call its front end never sent.
    starts, ends = [C(0, 0), C(0, 0)], [C(5, 2), C(1, 2)]
    relaxed = proof.check_server_counts(starts, ends, no_engine_traffic=False, cancels_calls=True)
    assert (relaxed.status, relaxed.mismatch) == (proof.FAIL, True)
    assert "front end sent 1 Generate calls, engine received 2" in relaxed.detail
    ok = proof.check_server_counts(
        starts, [C(5, 2), C(2, 2)], no_engine_traffic=False, cancels_calls=True
    )
    assert (ok.status, ok.engine_calls) == (proof.OK, 4)
    strict = proof.check_server_counts(starts, [C(5, 2), C(2, 2)], no_engine_traffic=False)
    assert (strict.status, strict.mismatch) == (proof.FAIL, True)


def test_add_counters_is_unknown_if_any_side_is_unknown():
    assert proof.add_counters([C(1, 2), C(3, 4)]) == C(4, 6)
    assert proof.add_counters([C(1, 2), C(None, 4)]) == C(None, 6)
    assert proof.add_counters([]) == C(0, 0)


def _handle(tmp_path):
    counter_dir = tmp_path / "counters"
    counter_dir.mkdir(exist_ok=True)
    return types.SimpleNamespace(counter_dir=str(counter_dir), alive=True)


def test_frontend_counter_sums_every_frontend_file(tmp_path):
    handle = _handle(tmp_path)
    for pid, sent in ((11, 3), (12, 4)):
        (tmp_path / "counters" / f"{pid}.json").write_text(
            json.dumps({"pid": pid, "generate_sent": sent})
        )
    assert proof.read_frontend_counter(handle) == 7


def test_frontend_counter_rereads_a_file_being_rewritten(tmp_path, monkeypatch):
    handle = _handle(tmp_path)
    path = tmp_path / "counters" / "11.json"
    path.write_text('{"pid": 11, "generate')

    def finish(_):
        path.write_text(json.dumps({"pid": 11, "generate_sent": 5}))

    monkeypatch.setattr(proof.time, "sleep", finish)
    assert proof.read_frontend_counter(handle) == 5


class _RunningFrontend:
    """Answers each connection with ``{"pid": ..., "generate_sent": ...}``, as a running front end does."""

    def __init__(self, pid, sent):
        self.address = f"openengine-chain-test-{os.getpid()}-{uuid.uuid4().hex}"
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind("\0" + self.address)
        self._listener.listen()
        self.answer = json.dumps({"pid": pid, "generate_sent": sent}).encode()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while True:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                return
            with connection:
                connection.sendall(self.answer)

    def close(self):
        self._listener.shutdown(socket.SHUT_RDWR)
        self._listener.close()
        self._thread.join()


def _write_record(path, pid, sent, address=None):
    record = {"pid": pid, "generate_sent": sent}
    if address is not None:
        record["address"] = address
    path.write_text(json.dumps(record))


def test_a_running_frontend_is_asked_for_its_count(tmp_path):
    handle = _handle(tmp_path)
    running = _RunningFrontend(11, 9)
    try:
        # The file holds the count of when the front end started.
        _write_record(tmp_path / "counters" / "11.json", 11, 0, running.address)
        _write_record(tmp_path / "counters" / "12.json", 12, 4)
        assert proof.read_frontend_counter(handle) == 13
    finally:
        running.close()


def test_a_running_frontend_that_does_not_answer_raises(tmp_path):
    handle = _handle(tmp_path)
    _write_record(tmp_path / "counters" / "11.json", 11, 0, f"nobody-{uuid.uuid4().hex}")
    with pytest.raises(OSError, match="pid 11 did not answer"):
        proof.read_frontend_counter(handle)


def test_a_running_frontend_that_hangs_up_without_answering_raises(tmp_path):
    handle = _handle(tmp_path)
    running = _RunningFrontend(11, 9)
    running.answer = b""
    try:
        _write_record(tmp_path / "counters" / "11.json", 11, 0, running.address)
        with pytest.raises(OSError, match="pid 11 did not answer"):
            proof.read_frontend_counter(handle)
    finally:
        running.close()


def test_a_frontend_that_stopped_while_being_read_gives_its_final_count(tmp_path, monkeypatch):
    handle = _handle(tmp_path)
    path = tmp_path / "counters" / "11.json"
    _write_record(path, 11, 0, "stopping")

    def stopped(address):
        # The front end wrote its final count, then stopped answering.
        _write_record(path, 11, 6)
        raise ConnectionRefusedError(address)

    monkeypatch.setattr(proof, "_ask_frontend", stopped)
    assert proof.read_frontend_counter(handle) == 6


@pytest.mark.parametrize(
    "content",
    [
        "null",
        '{"pid": 11}',
        '{"pid": 11, "generate_sent": null}',
        '{"pid": 11, "generate_sent": true}',
        '{"pid": 11, "generate_sent": 1, "address": 7}',
    ],
)
def test_a_file_without_a_count_raises_value_error(tmp_path, monkeypatch, content):
    handle = _handle(tmp_path)
    (tmp_path / "counters" / "11.json").write_text(content)
    monkeypatch.setattr(proof.time, "sleep", lambda _: None)
    with pytest.raises(ValueError):
        proof.read_frontend_counter(handle)


def test_frontend_counter_is_unknown_without_files(tmp_path, monkeypatch):
    handle = _handle(tmp_path)
    monkeypatch.setattr(proof, "read_engine_counter", lambda h: 7)
    assert proof.read_frontend_counter(handle) is None
    assert proof.read_counters(handle) == C(None, 7, frontend_unwritten=True)


@pytest.mark.parametrize(
    "before, after, start",
    [
        # A file appeared during the test: the front end counts and had sent nothing.
        (C(None, 3, frontend_unwritten=True), C(2, 5), C(0, 3)),
        # Still no file: the front end may not count at all.
        (
            C(None, 3, frontend_unwritten=True),
            C(None, 5, frontend_unwritten=True),
            C(None, 3, frontend_unwritten=True),
        ),
        # A counter that could not be read is not taken for an unwritten one.
        (C(None, None), C(2, 5), C(None, None)),
        (C(4, 3), C(6, 5), C(4, 3)),
    ],
)
def test_counting_start(before, after, start):
    assert proof.counting_start(before, after) == start


def _topology_handle(frontend_pid, engine_pid):
    return types.SimpleNamespace(
        frontend=types.SimpleNamespace(pid=frontend_pid),
        engine=types.SimpleNamespace(pid=engine_pid),
    )


@pytest.fixture
def cmdline(monkeypatch):
    lines = {}
    monkeypatch.setattr(proof, "_cmdline", lambda pid: lines.get(pid))
    return lines


SESSIONS = {10: {10, 11}, 20: {20, 21}}


@pytest.mark.parametrize(
    "gpu, status",
    [
        ({20, 21}, proof.OK),
        ({11, 21}, proof.FAIL),
        (None, proof.INCOMPLETE),
        # Positive control: nvidia-smi showing no engine process proves nothing.
        (set(), proof.INCOMPLETE),
    ],
)
def test_topology(cmdline, gpu, status):
    cmdline[10] = ["trtllm-serve", "/m", "--openengine-endpoint", "127.0.0.1:1"]
    result = proof.check_topology(
        _topology_handle(10, 20), gpu_pid_reader=lambda: gpu, tree_reader=SESSIONS.get
    )
    assert result.status == status, result.detail


def test_topology_fails_without_the_endpoint_flag(cmdline):
    cmdline[10] = ["trtllm-serve", "/m"]
    result = proof.check_topology(
        _topology_handle(10, 20), gpu_pid_reader=lambda: {20}, tree_reader=SESSIONS.get
    )
    assert result.status == proof.FAIL


def test_topology_of_an_exited_frontend_is_incomplete(cmdline):
    result = proof.check_topology(
        _topology_handle(10, 20), gpu_pid_reader=lambda: {20}, tree_reader=SESSIONS.get
    )
    assert result.status == proof.INCOMPLETE


def test_topology_of_an_unstarted_frontend_is_incomplete():
    handle = types.SimpleNamespace(frontend=None, engine=types.SimpleNamespace(pid=20))
    assert proof.check_topology(handle).status == proof.INCOMPLETE


def test_cmdline_reads_this_process():
    assert proof._cmdline(os.getpid())


def test_cmdline_of_a_zombie_is_unknown(monkeypatch, tmp_path):
    empty = tmp_path / "cmdline"
    empty.write_bytes(b"")
    real_open = open
    monkeypatch.setattr(
        "builtins.open",
        lambda path, *a, **k: real_open(empty, *a, **k)
        if path == "/proc/42/cmdline"
        else real_open(path, *a, **k),
    )
    assert proof._cmdline(42) is None


def test_only_disagreeing_sides_are_a_mismatch():
    assert proof.check_counts(C(0, 0), C(3, 2), no_engine_traffic=False).mismatch
    assert not proof.check_counts(C(0, 0), C(0, 0), no_engine_traffic=False).mismatch


def test_unlistable_counter_directory_is_not_read_as_unwritten(tmp_path):
    handle = _handle(tmp_path)
    os.chmod(handle.counter_dir, 0)
    try:
        if os.access(handle.counter_dir, os.R_OK):
            pytest.skip("running as a user that can read any directory")
        with pytest.raises(PermissionError):
            proof.read_frontend_counter(handle)
    finally:
        os.chmod(handle.counter_dir, 0o755)


def test_missing_counter_directory_reads_as_unwritten(tmp_path):
    handle = types.SimpleNamespace(counter_dir=str(tmp_path / "absent"), alive=True)
    assert proof.read_frontend_counter(handle) is None


def test_one_server_without_counters_does_not_hide_a_disagreement_on_another():
    starts, ends = [C(0, 0), C(None, 0)], [C(3, 2), C(None, 5)]
    counts = proof.check_server_counts(starts, ends, no_engine_traffic=False)
    assert (counts.status, counts.mismatch) == (proof.FAIL, True)
    assert "front end sent 3 Generate calls, engine received 2" in counts.detail
    agreeing = proof.check_server_counts(starts, [C(2, 2), C(None, 5)], no_engine_traffic=False)
    assert agreeing.status == proof.INCOMPLETE and not agreeing.mismatch
    complete = proof.check_server_counts(
        [C(0, 0), C(1, 1)], [C(2, 2), C(4, 4)], no_engine_traffic=False
    )
    assert (complete.status, complete.engine_calls) == (proof.OK, 5)


# A launcher with two further front ends (one with a child of its own) and a pool
# process; it writes its argument once all of them have started.
_FRONTEND_TREE = """
import os, subprocess, sys
sleep = [sys.executable, "-c", "import time; time.sleep(60)"]
def start(cmd, frontend_id=None):
    env = dict(os.environ)
    if frontend_id is not None:
        env["TLLM_EXECUTOR_FRONTEND_ID"] = frontend_id
    return subprocess.Popen(cmd, env=env)
grandchild = [sys.executable, "-c", "import subprocess, sys, time; "
              "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
              "time.sleep(60)"]
children = [start(grandchild, "1"), start(sleep, "2"), start(sleep)]
open(sys.argv[1], "w").close()
for child in children:
    child.wait()
"""


def _front_end(tmp_path, script, env=None):
    ready = tmp_path / "ready"
    proc = subprocess.Popen(
        [sys.executable, "-c", script, str(ready)], start_new_session=True, env=env
    )
    deadline = time.monotonic() + 30
    while not ready.exists():
        assert proc.poll() is None and time.monotonic() < deadline
        time.sleep(0.05)
    return proc


# With an id in the launcher's own environment, the pool process inherits it.
@pytest.mark.parametrize("inherited", [None, "7"])
def test_frontend_count_counts_the_front_ends_the_launcher_started(tmp_path, inherited):
    env = {k: v for k, v in os.environ.items() if k != "TLLM_EXECUTOR_FRONTEND_ID"}
    if inherited is not None:
        env["TLLM_EXECUTOR_FRONTEND_ID"] = inherited
    proc = _front_end(tmp_path, _FRONTEND_TREE, env)
    try:
        # The grandchild must have started before the count.
        deadline = time.monotonic() + 30
        while len(proof.live_descendants(proc.pid)) < 4:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert proof.frontend_count(types.SimpleNamespace(frontend=proc)) == 3
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def test_frontend_count_is_one_without_further_front_ends(tmp_path):
    proc = _front_end(tmp_path, "import sys, time; open(sys.argv[1], 'w').close(); time.sleep(60)")
    try:
        assert proof.frontend_count(types.SimpleNamespace(frontend=proc)) == 1
    finally:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    with pytest.raises(AssertionError, match="not running"):
        proof.frontend_count(types.SimpleNamespace(frontend=proc))


def test_engine_received_fails_when_the_engine_does_not_report_it(monkeypatch):
    monkeypatch.setattr(proof, "read_engine_counter", lambda h: 4)
    assert proof.engine_received(object()) == 4
    monkeypatch.setattr(proof, "read_engine_counter", lambda h: None)
    with pytest.raises(AssertionError, match="does not report"):
        proof.engine_received(object())
