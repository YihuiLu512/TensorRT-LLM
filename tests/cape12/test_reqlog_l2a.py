# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cape12.reqlog_l2a (E12-L2a W1-W6) on a fake clock and a fake request function: the applied
delay, the open-loop schedule, the argv rewrite, warm-up recognition, the header and the hashes.
Every check asserts it saw a non-zero number of request lines (no empty pass)."""

import asyncio
import io
import json
import types

import pytest

from cape12 import reqlog_l2a as rl

ARGV = ["--model", "m", "--max-concurrency", "8", "--num-prompts", "16", "--seed", "7"]


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    async def sleep(self, s):
        self.t += s


def fake_fn(clock, serve_s=0.5):
    async def fn(inp):
        clock.t += serve_s
        return types.SimpleNamespace(ttft=0.1, latency=serve_s, success=True,
                                     output_tokens=3, itl=[0.01, 0.01],
                                     generated_text="out-%s" % inp.prompt, error="")
    return fn


def run(cfg, n, concurrent=False):
    clock, log = Clock(), io.StringIO()
    w = rl.Wrapper(cfg, log, clock=clock, wall=lambda: 1e9, sleep=clock.sleep)
    f = w.wrap(fake_fn(clock))
    inputs = [types.SimpleNamespace(prompt="p%d" % (i % 2), prompt_len=5, output_len=3)
              for i in range(n)]

    async def go():
        for x in inputs:
            await f(x)
    asyncio.run(go())
    lines = [json.loads(x) for x in log.getvalue().splitlines()]
    assert lines[0]["hdr"] == 1 and len(lines) == n + 1
    return lines[0], lines[1:]


def test_closed_logs_like_the_original_plus_w5_w6(tmp_path):
    cfg = rl.load_config({"E12_REQLOG_DIR": str(tmp_path)}, ARGV)
    hdr, reqs = run(cfg, 4)
    assert hdr["mode"] == "closed" and hdr["argv"] == ARGV and hdr["warmup"] is False
    assert all(r["t_call"] == r["t_enter"] and not r["delayed"] for r in reqs)
    assert reqs[0]["output_sha1"] == reqs[2]["output_sha1"] != reqs[1]["output_sha1"]
    assert reqs[0]["prompt_sha1"] == reqs[2]["prompt_sha1"]


def test_header_records_run_step_and_arm(tmp_path):
    env = {"E12_REQLOG_DIR": str(tmp_path), "E12_RUN": "c8_r1_text", "TRTLLM_PERF_RUN": "3",
           "TRTLLM_PERF_ARM": "B-s1"}
    hdr, reqs = run(rl.load_config(env, ARGV), 2)
    assert len(reqs) == 2
    assert (hdr["run"], hdr["step"], hdr["arm"]) == ("c8_r1_text", "3", "B-s1")
    hdr, _reqs = run(rl.load_config({"E12_REQLOG_DIR": str(tmp_path)}, ARGV), 1)
    assert (hdr["run"], hdr["step"], hdr["arm"]) == (None, None, None)


def test_w1_closed_delta_waits_delta_before_the_call(tmp_path):
    cfg = rl.load_config({"E12_REQLOG_DIR": str(tmp_path), "E12_MODE": "closed_delta",
                          "E12_DELTA_MS": "12.5"}, ARGV)
    hdr, reqs = run(cfg, 5)
    assert hdr["delta_s"] == 0.0125
    waits = [r["t_call"] - r["t_enter"] for r in reqs]
    assert len(waits) == 5 and all(abs(w - 0.0125) < 1e-9 for w in waits)


def test_partial_delta_delays_a_seeded_subset(tmp_path):
    env = {"E12_REQLOG_DIR": str(tmp_path), "E12_MODE": "partial_delta", "E12_DELTA_MS": "10",
           "E12_MASK_SEED": "3"}
    _h, a = run(rl.load_config(env, ARGV), 200)
    _h, b = run(rl.load_config(env, ARGV), 200)
    flags = [r["delayed"] for r in a]
    assert flags == [r["delayed"] for r in b]  # same seed, same mask
    assert 60 < sum(flags) < 140
    assert all((r["t_call"] - r["t_enter"] > 0) == r["delayed"] for r in a)


def _sched(tmp_path, offs):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"schema": rl.SCHEDULE_SCHEMA, "offsets_s": offs}))
    return str(p)


def test_w2_w3_open_follows_the_schedule_and_drops_max_concurrency(tmp_path):
    offs = [0.0, 1.0, 1.2, 5.0]
    cfg = rl.load_config({"E12_REQLOG_DIR": str(tmp_path), "E12_MODE": "open",
                          "E12_SCHEDULE": _sched(tmp_path, offs)}, ARGV)
    assert "--max-concurrency" not in cfg["argv"] and "--num-prompts" in cfg["argv"]
    hdr, reqs = run(cfg, 4)
    assert hdr["schedule_sha256"] and hdr["n_offsets"] == 4
    t0 = reqs[0]["t_enter"]
    for r, o in zip(reqs, offs):
        assert r["planned"] == pytest.approx(t0 + o)
        # serial fake: a request whose planned time already passed is called at once
        assert r["t_call"] == pytest.approx(max(r["planned"], r["t_enter"]))
    assert reqs[2]["t_call"] > reqs[2]["planned"]  # late (the previous one took 0.5 s)


def test_w4_warmup_client_runs_closed_without_delay(tmp_path):
    env = {"E12_REQLOG_DIR": str(tmp_path), "E12_MODE": "open", "E12_WARMUP_SEED": "7",
           "E12_SCHEDULE": "/nonexistent"}
    cfg = rl.load_config(env, ARGV)  # the schedule is not even opened for the warm-up client
    assert cfg["warmup"] and cfg["mode"] == "closed" and cfg["requested_mode"] == "open"
    assert cfg["argv"] == ARGV
    hdr, reqs = run(cfg, 3)
    assert hdr["warmup"] is True and all(r["t_call"] == r["t_enter"] for r in reqs)
    env["E12_WARMUP_SEED"] = "8"
    with pytest.raises(FileNotFoundError):
        rl.load_config(env, ARGV)


@pytest.mark.parametrize("env, msg", [
    ({"E12_MODE": "bogus"}, "E12_MODE"),
    ({}, "E12_REQLOG_DIR"),
    ({"E12_MODE": "closed_delta"}, "E12_DELTA_MS"),
    ({"E12_MODE": "partial_delta", "E12_DELTA_MS": "1"}, "E12_MASK_SEED"),
    ({"E12_MODE": "open"}, "E12_SCHEDULE"),
])
def test_refusals(tmp_path, env, msg):
    if env != {}:
        env = dict(env, E12_REQLOG_DIR=str(tmp_path))
    with pytest.raises(rl.WrapperConfigError, match=msg):
        rl.load_config(env, ARGV)


def test_bad_schedule_refused(tmp_path):
    for offs in ([], [1.0, 0.5], [-1.0]):
        with pytest.raises(rl.WrapperConfigError):
            rl.load_config({"E12_REQLOG_DIR": str(tmp_path), "E12_MODE": "replay",
                            "E12_SCHEDULE": _sched(tmp_path, offs)}, ARGV)


def test_strip_and_seed_helpers():
    assert rl.strip_max_concurrency(["a", "--max-concurrency=8", "b"]) == ["a", "b"]
    assert rl.argv_seed(["--seed=9"]) == "9" and rl.argv_seed(["x"]) is None
