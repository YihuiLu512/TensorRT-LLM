# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capgrpc.e15probe: the E15-B admission checks against a fake engine client (no product import),
plus the CLI's failure path (product import or connect fails: recorded, not raised) and the
scenario tool-table registration."""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import types

import pytest

from capgrpc import e15probe as ep
from caplaunch import scenario as sc

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))


class FakeStream:
    def __init__(self, eng, rid, n_tokens):
        self.eng, self.request_id, self.n = eng, rid, n_tokens
        self.i, self._cancelled, self._aborted = 0, False, False

    @property
    def cancelled(self):  # a property, as on the product's GenerateStream
        return self._cancelled

    def cancel(self):
        self._cancelled = True
        self.eng.running.discard(self.request_id)

    async def __aenter__(self):
        self.eng.running.add(self.request_id)
        return self

    async def __aexit__(self, *a):
        if self.i < self.n and not self._aborted:
            self.cancel()
        self.eng.running.discard(self.request_id)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)
        if self._cancelled or self.i > self.n:
            raise StopAsyncIteration
        if self.request_id in self.eng.aborted:
            self.i = self.n + 1
            self.eng.running.discard(self.request_id)
            return ("finished", "FINISH_REASON_ABORT")
        self.i += 1
        if self.i > self.n:
            self.eng.running.discard(self.request_id)
            return ("finished", "FINISH_REASON_LENGTH")
        return ("token", [self.eng.token(self.i)])


class FakeClient:
    def __init__(self, cfg):
        self.cfg, self.running, self.aborted, self.k = cfg, set(), set(), 0
        self.closed = False

    def token(self, i):
        return 1000 + i if not self.cfg.get("nondeterministic") else 1000 + i + self.k

    async def health(self, include_inference_probe=False, timeout=None):
        checks = [("grpc", "HEALTH_STATE_READY", "")]
        if include_inference_probe and not self.cfg.get("no_probe_check"):
            checks.append(("inference_probe", "HEALTH_STATE_READY", ""))
        return ("HEALTH_STATE_READY", checks)

    async def get_load(self, timeout=None):
        if self.cfg.get("no_getload"):
            return None
        return {"running": len(self.running), "queued": 0}

    def generate(self, req, timeout=None):
        self.k += 1
        return FakeStream(self, "r%d" % self.k, req["max_tokens"])

    async def abort(self, rid, timeout=None):
        if self.cfg.get("abort_not_found"):
            return "NOT_FOUND"
        self.aborted.add(rid)
        return "ABORTED"

    async def close(self):
        self.closed = True


class FakeApi:
    def __init__(self, **cfg):
        self.cfg = cfg
        self.client = None

    async def connect(self, target, ready_timeout_s):
        if self.cfg.get("connect_fails"):
            raise ConnectionError("no engine at %s" % target)
        self.client = FakeClient(self.cfg)
        return self.client

    def health_state(self, resp):
        return resp[0]

    def health_checks(self, resp):
        return resp[1]

    def running(self, load):
        return None if load is None else load["running"]

    def queued(self, load):
        return None if load is None else load["queued"]

    def request(self, model, token_ids, max_tokens, greedy):
        return {"model": model, "ids": token_ids, "max_tokens": max_tokens, "greedy": greedy}

    def event(self, resp):
        return resp

    def abort_outcome(self, outcome):
        return outcome


def cfg(**over):
    c = {"target": "127.0.0.1:1", "model": "m", "prompt": [5] * 16, "max_tokens": 64,
         "after_chunks": 4, "greedy_tokens": 8, "ready_timeout_s": 1.0,
         "control_timeout_s": 1.0, "probe_timeout_s": 1.0, "clear_timeout_s": 1.0,
         "stream_timeout_s": 5.0, "check_timeout_s": 5.0}
    c.update(over)
    return c


def by_name(recs):
    return {r["check"]: r for r in recs}


def test_all_checks_pass_on_a_conforming_engine():
    api = FakeApi()
    recs = asyncio.run(ep.run_checks(api, cfg()))
    assert [r["check"] for r in recs] == list(ep.CHECKS)
    assert all(r["ok"] for r in recs), recs
    r = by_name(recs)
    assert r["cancel"]["token_events_before_stop"] == 4 and r["cancel"]["running_mid"] == 1
    assert r["cancel"]["clear_s"] is not None and r["cancel"]["stream_cancelled"] is True
    assert r["abort"]["abort_outcome"] == "ABORTED"
    assert r["abort"]["finished_after_abort"] == "FINISH_REASON_ABORT"
    g = r["greedy_aa"]["runs"]
    assert g[0]["n"] == 8 and g[0]["token_ids"] == g[1]["token_ids"]
    assert g[0]["sha256"] == g[1]["sha256"]
    assert api.client.closed


@pytest.mark.parametrize("flaw, failing", [
    ("no_probe_check", {"health_probe"}),
    ("abort_not_found", {"abort"}),
    ("nondeterministic", {"greedy_aa"}),
    ("no_getload", {"getload", "cancel", "abort"}),
])
def test_each_flaw_fails_only_its_checks(flaw, failing):
    recs = asyncio.run(ep.run_checks(FakeApi(**{flaw: True}), cfg()))
    assert len(recs) == len(ep.CHECKS)  # one failing check does not stop the others
    assert {r["check"] for r in recs if not r["ok"]} == failing


def test_a_check_that_raises_is_recorded_and_the_rest_still_run():
    api = FakeApi()

    async def boom(*a, **k):
        raise RuntimeError("health exploded")

    async def go():
        client = await FakeApi.connect(api, "t", 1.0)
        client.health = boom
        api.connect = lambda *a: _ret(client)
        return await ep.run_checks(api, cfg())

    recs = by_name(asyncio.run(go()))
    assert "health exploded" in recs["health_plain"]["error"]
    assert not recs["health_plain"]["ok"] and not recs["health_probe"]["ok"]
    assert recs["getload"]["ok"] and recs["cancel"]["ok"] and recs["greedy_aa"]["ok"]


async def _ret(x):
    return x


def test_a_hung_check_hits_its_timeout():
    api = FakeApi()

    async def hang(*a, **k):
        await asyncio.sleep(30)

    async def go():
        client = await FakeApi.connect(api, "t", 1.0)
        client.health = hang
        api.connect = lambda *a: _ret(client)
        return await ep.run_checks(api, cfg(check_timeout_s=0.2))

    t0 = time.monotonic()
    recs = by_name(asyncio.run(go()))
    assert time.monotonic() - t0 < 5
    assert "TimeoutError" in recs["health_plain"]["error"]
    assert recs["getload"]["ok"]


def _prompts(tmp_path, kind="token_ids"):
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps({"schema": "capkit.prompts/1",
                             "sets": {"tid": {"kind": kind, "items": [[7] * 32]}}}))
    return str(p)


def _argv(tmp_path, at_s="0", extra=()):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    return (["--experiment", "E15-B", "--arm", "e15b-ipc-s20", "--launch-seq", "0",
             "--job-id", "local-x", "--restart-count", "0", "--register-sha", "a" * 64,
             "--role", "e15probe", "--run-dir", str(run), "--stop-grace-s", "1.0",
             "--cpus", "0", "--origin-ns", str(time.monotonic_ns()),
             "--ready-file", str(tmp_path / "ready.json"), "--target", "127.0.0.1:9",
             "--model", "m", "--prompts", _prompts(tmp_path), "--prompt-set", "tid",
             "--at-s", at_s, "--result", str(tmp_path / "result.json")] + list(extra))


def test_config_refusals(tmp_path):
    a = ep.build_parser().parse_args(_argv(tmp_path, extra=["--cancel-after-chunks", "2000"]))
    with pytest.raises(ep.ConfigError):
        ep.config_from_args(a)
    a = ep.build_parser().parse_args(_argv(tmp_path, extra=["--prompt-set", "nope"]))
    with pytest.raises(ep.ConfigError):
        ep.config_from_args(a)
    a = ep.build_parser().parse_args(_argv(tmp_path))
    c = ep.config_from_args(a)
    assert c["prompt"] == [7] * 32 and c["after_chunks"] == 8 and c["max_tokens"] == 1024


def test_cli_records_a_connect_failure_writes_ready_and_result_then_holds(tmp_path):
    """Without the product (no tensorrt_llm here) the import/connect failure is recorded in the
    result, the tool still writes its ready file and result, and exits 0 on SIGTERM."""
    env = dict(os.environ, PYTHONPATH=TOOLS)
    p = subprocess.Popen([sys.executable, "-B", "-m", "capgrpc.e15probe"] + _argv(tmp_path),
                         env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=str(tmp_path))
    res = tmp_path / "result.json"
    t0 = time.monotonic()
    while not res.exists() and time.monotonic() - t0 < 60 and p.poll() is None:
        time.sleep(0.1)
    assert res.exists(), p.communicate(timeout=5)
    assert (tmp_path / "ready.json").exists()
    doc = json.loads(res.read_text())
    assert doc["schema"] == ep.SCHEMA and doc["all_ok"] is False
    assert doc["error"] and doc["checks"] == []
    assert p.poll() is None  # holds until SIGTERM
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) == 0


def test_main_with_a_fake_api_writes_all_ok(tmp_path):
    argv = _argv(tmp_path, extra=["--max-tokens", "32", "--cancel-after-chunks", "4",
                                  "--greedy-tokens", "8"])
    old = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGALRM, lambda *a: os.kill(os.getpid(), signal.SIGTERM))
        signal.setitimer(signal.ITIMER_REAL, 2.0)
        assert ep.main(argv, api=FakeApi()) == 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, old[0])
        signal.signal(signal.SIGINT, old[1])
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["all_ok"] is True and [r["check"] for r in doc["checks"]] == list(ep.CHECKS)
    assert "prompt" not in doc["config"] and doc["prompt_len"] == 32


def test_registered_as_an_aux_observer_tool():
    t = sc.TOOL_TABLE["capgrpc.e15probe"]
    assert t["origin"] and t["ready"] and not t.get("tiers") and not t.get("snapshot")
    assert t["role_fixed"]["kind"] == "aux" and t["role_fixed"]["measure_class"] == "observer"
    assert t["role_fixed"]["uses_grpc"] is True and t["role_fixed"]["expected_procs"] == 1


def test_checks_subset_runs_only_those(tmp_path):
    recs = asyncio.run(ep.run_checks(FakeApi(), cfg(checks=["cancel"])))
    assert [r["check"] for r in recs] == ["cancel"] and recs[0]["ok"]
    assert recs[0]["stream_cancelled"] is True
    a = ep.build_parser().parse_args(_argv(tmp_path, extra=["--checks", "greedy_aa,cancel"]))
    assert ep.config_from_args(a)["checks"] == ["cancel", "greedy_aa"]  # CHECKS order
    a = ep.build_parser().parse_args(_argv(tmp_path))
    assert ep.config_from_args(a)["checks"] == list(ep.CHECKS)
    for bad in ("cancel,nope", "", "cancel,cancel"):
        a = ep.build_parser().parse_args(_argv(tmp_path, extra=["--checks", bad]))
        with pytest.raises(ep.ConfigError):
            ep.config_from_args(a)
    doc = ep.write_result(str(tmp_path / "r.json"), types.SimpleNamespace(to_dict=dict),
                          cfg(checks=["cancel"]), recs)
    assert doc["all_ok"] is True
