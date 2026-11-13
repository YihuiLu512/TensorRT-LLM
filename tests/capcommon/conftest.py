# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Put <C>/tools on sys.path so `import capcommon` works when running from the repo."""

import copy
import json
import os
import sys

import pytest

TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

SHA = "a" * 64

GATE = {"mode": "enforce", "rule": "abs_max",
        "readings": [{"name": "inflight",
                      "terms": [{"role": "capload", "slot": "backlog.inflight", "sign": 1,
                                 "agg": "sum"}],
                      "abs_max": 0}],
        "poll_ns": 100_000_000, "settle_polls": 3, "deadline_guard_ns": 1_000_000_000,
        "max_snapshot_age_ns": 200_000_000}


def make_tier(seq, label, kind, start, ramp, window, tail, rate, max_tokens=8, beat=10_000_000,
              k=1, s=1, prompt_set="p", inject=None, gate=None, baseline=False, streams_c=None,
              load_value="auto"):
    if load_value == "auto":  # R axis: load_value = rate_rps on load tiers, null on drain tiers
        load_value = None if kind == "drain" else rate
    return {"seq": seq, "label": label, "kind": kind, "start_ns": start, "ramp_ns": ramp,
            "window_ns": window, "tail_ns": tail, "rate_rps": rate, "streams_c": streams_c,
            "max_tokens": max_tokens, "tokens_per_msg": k, "stream_interval": s,
            "beat_ns": beat, "prompt_set": prompt_set, "inject": inject or [], "gate": gate,
            "baseline": baseline, "note": "", "load_value": load_value}


def good_table_doc(dist="fixed"):
    """warmup(1s) drain(0.5s) up(2s, baseline) drain(0.5s) up(1s) drain(0.5s)."""
    S = 1_000_000_000
    tiers = [
        make_tier(0, "warmup", "warmup", 0, 0, S, 0, 100.0),
        make_tier(1, "drain-1", "drain", S, 0, S // 2, 0, 0.0, max_tokens=0, prompt_set=None),
        make_tier(2, "up-r200", "up", S + S // 2, S // 2, S, S // 2, 200.0, baseline=True),
        make_tier(3, "drain-2", "drain", 3 * S + S // 2, 0, S // 2, 0, 0.0, max_tokens=0,
                  prompt_set=None),
        make_tier(4, "up-r400", "up", 4 * S, 0, S, 0, 400.0,
                  inject=[{"role": "capload", "site": "recv.chunk", "mode": "busy",
                           "delta_ns": 1000}]),
        make_tier(5, "drain-3", "drain", 5 * S, 0, S // 2, 0, 0.0, max_tokens=0,
                  prompt_set=None),
    ]
    return {"schema": "capkit.tiers/1", "experiment": "E0", "name": "unit",
            "arrival": {"dist": dist, "seed": 7},
            "axis": {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"},
            "request_timeout_ns": 30_000_000_000, "beat_origin_rel_ns": 0,
            "default_gate": copy.deepcopy(GATE), "launch_inject": [], "tiers": tiers}


@pytest.fixture
def table_path(tmp_path):
    def _write(doc=None, name="tiers.json"):
        p = tmp_path / name
        p.write_text(json.dumps(doc if doc is not None else good_table_doc()))
        return str(p)
    return _write


@pytest.fixture
def ident():
    from capcommon.identity import Identity
    return Identity(experiment="E0", arm="unit-a", launch_seq=1, job_id="local-test",
                    restart_count=0, register_sha=SHA, role="capload", worker=0, pid=os.getpid())
