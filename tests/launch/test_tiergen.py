# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""caplaunch.tiergen against the shipped configs/e0 templates (no processes)."""

import fractions
import json
import os

import pytest

import conftest as cf
from capcommon.exitcodes import ConfigError
from capcommon.tiers import TierTable
from caplaunch import tiergen

CONFIGS = os.path.join(cf.C, "configs", "e0")
BASE = {"gate_poll_ns": 50 * cf.MS, "gate_settle_polls": 3, "gate_deadline_guard_ns": 100 * cf.MS,
        "gate_max_snapshot_age_ns": 200 * cf.MS, "arrival_seed": 7,
        "request_timeout_ns": 30_000 * cf.MS, "beat_ns": 10 * cf.MS}


def _templates():
    return sorted(n for n in os.listdir(CONFIGS) if n.endswith(".tiers.template.json"))


def _load(name):
    with open(os.path.join(CONFIGS, name)) as f:
        return json.load(f)


def _sample(names):
    return {n: BASE.get(n, 2.0 if n.startswith("rate") else
                        (1000 * cf.MS if n.endswith("_ns") else 4)) for n in names}


def test_every_template_has_a_matching_todo_checklist():
    assert len(_templates()) >= 7  # (a) x2 + (b) x4 + (c) x1
    for n in _templates():
        todo = _load(n.replace(".tiers.template.json", ".values.todo.json"))
        assert todo["schema"] == tiergen.VALUES_SCHEMA
        assert set(todo["values"]) == tiergen.tbd_names(_load(n))
        assert all(v is None for v in todo["values"].values())


@pytest.mark.parametrize("name", _templates())
def test_templates_expand_with_sample_values(name):
    t = _load(name)
    doc = tiergen.expand(t, _sample(tiergen.tbd_names(t)))
    tt = TierTable.from_doc(doc)
    assert tt.tiers[-1].kind == "drain"
    assert [x.seq for x in tt.tiers] == list(range(len(tt.tiers)))


def test_rate_from_streams_c_is_exact():
    t = _load("e0b-s1-single.tiers.template.json")
    vals = _sample(tiergen.tbd_names(t))
    vals.update(n_tokens=7, beat_ns=4 * cf.MS, c_sub=5)
    doc = tiergen.expand(t, vals)
    up = [x for x in doc["tiers"] if x["label"] == "up-sub"][0]
    assert up["rate_rps"] == float(fractions.Fraction(5 * 10 ** 9, 7 * 4 * cf.MS))
    assert up["load_value"] == float(fractions.Fraction(5 * 10 ** 9, 4 * cf.MS))


def test_missing_unused_and_null_values_are_refused(tmp_path):
    t = _load("e0a-nullsse.tiers.template.json")
    vals = _sample(tiergen.tbd_names(t))
    short = dict(vals)
    short.pop("beat_ns")
    with pytest.raises(ConfigError, match="missing"):
        tiergen.expand(t, short)
    with pytest.raises(ConfigError, match="not used"):
        tiergen.expand(t, dict(vals, bogus=1))
    tpl = os.path.join(CONFIGS, "e0a-nullsse.tiers.template.json")
    todo = os.path.join(CONFIGS, "e0a-nullsse.values.todo.json")
    out = str(tmp_path / "t.json")
    assert tiergen.main(["--template", tpl, "--values", todo, "--out", out]) == 3
    assert not os.path.exists(out)


def test_main_writes_once_and_refuses_overwrite(tmp_path):
    tpl = os.path.join(CONFIGS, "e0a-nullsse.tiers.template.json")
    vals = tmp_path / "v.json"
    vals.write_text(json.dumps({"schema": tiergen.VALUES_SCHEMA,
                                "values": _sample(tiergen.tbd_names(_load(os.path.basename(
                                    tpl))))}))
    out, sched = str(tmp_path / "t.json"), str(tmp_path / "s.capbin")
    argv = ["--template", tpl, "--values", str(vals), "--out", out, "--schedule-out", sched,
            "--max-rows", "100000"]
    assert tiergen.main(argv) == 0
    assert os.path.getsize(sched) > 0
    TierTable.load(out)
    assert tiergen.main(argv) == 3  # out exists
