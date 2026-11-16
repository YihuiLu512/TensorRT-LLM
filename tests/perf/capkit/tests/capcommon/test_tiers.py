# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pytest

from capcommon.hashing import sha256_file
from capcommon.tiers import POST_SEQ, PRE_SEQ, TierTable, TierTableError

from conftest import good_table_doc

S = 1_000_000_000


def test_good_table_loads(table_path):
    p = table_path()
    tt = TierTable.load(p)
    assert tt.sha256 == sha256_file(p)
    assert len(tt) == 6 and tt.end_ns == 5 * S + S // 2
    assert tt.baseline_seq() == 2
    assert tt.window_of(2) == (2 * S, 3 * S)
    assert tt.tier_at(-1) is None and tt.tier_at(tt.end_ns) is None
    assert tt.tier_at(0).seq == 0 and tt.tier_at(S).seq == 1 and tt.tier_at(S - 1).seq == 0
    assert tt.tier_seq_at(-5) == PRE_SEQ and tt.tier_seq_at(10 ** 12) == POST_SEQ
    assert tt.label_of(PRE_SEQ) == "pre" and tt.label_of(POST_SEQ) == "post"
    gate, src = tt.gate_of(1)
    assert src == "default" and gate["rule"] == "abs_max"
    with pytest.raises(TierTableError):
        tt.gate_of(2)
    assert [s.site for s in tt.inject_for("capload", 4)] == ["recv.chunk"]
    assert tt.inject_for("capload", 2) == [] and tt.inject_for("nullsse", 4) == []
    assert tt[2].tokens_per_step == 1 and tt[2].steps_per_stream() == 8


def _mut(fn):
    doc = good_table_doc()
    before = copy.deepcopy(doc)
    fn(doc)
    # compare serialized forms: 1e9 == 1000000000 in Python, but not in the file
    assert json.dumps(doc, sort_keys=True) != json.dumps(before, sort_keys=True), \
        "sabotage did not change the document"
    return doc


SABOTAGES = {
    "gap_between_tiers": lambda d: d["tiers"][2].__setitem__("start_ns", d["tiers"][2]["start_ns"] + 1),
    "first_not_warmup": lambda d: d["tiers"][0].__setitem__("kind", "up"),
    "last_not_drain": lambda d: d["tiers"].pop(),
    "up_without_drain_before": lambda d: d["tiers"][1].update(
        {"kind": "up", "rate_rps": 10.0, "max_tokens": 8, "prompt_set": "p"}),
    "drain_with_rate": lambda d: d["tiers"][1].__setitem__("rate_rps", 1.0),
    "drain_with_ramp": lambda d: d["tiers"][1].update({"ramp_ns": 1, "window_ns": S // 2 - 1}),
    "load_zero_rate": lambda d: d["tiers"][2].__setitem__("rate_rps", 0.0),
    "load_with_gate": lambda d: d["tiers"][2].__setitem__("gate", d["default_gate"]),
    "k_not_multiple_of_s": lambda d: d["tiers"][2].update({"tokens_per_msg": 3, "stream_interval": 2}),
    "duration_not_multiple_of_beat": lambda d: d["tiers"][0].__setitem__("beat_ns", 3_000_000),
    "drain_beat_differs": lambda d: d["tiers"][1].__setitem__("beat_ns", 5_000_000),
    "two_baselines": lambda d: d["tiers"][4].__setitem__("baseline", True),
    "warmup_baseline": lambda d: (d["tiers"][2].__setitem__("baseline", False),
                                  d["tiers"][0].__setitem__("baseline", True)),
    "drain_baseline": lambda d: (d["tiers"][2].__setitem__("baseline", False),
                                 d["tiers"][1].__setitem__("baseline", True)),
    "float_int_field": lambda d: d["tiers"][2].__setitem__("window_ns", float(S)),
    "bool_as_int": lambda d: d["tiers"][2].__setitem__("max_tokens", True),
    "dup_label": lambda d: d["tiers"][4].__setitem__("label", "up-r200"),
    "reserved_label": lambda d: d["tiers"][4].__setitem__("label", "pre"),
    "bad_seq": lambda d: d["tiers"][3].__setitem__("seq", 7),
    "unknown_tier_key": lambda d: d["tiers"][3].__setitem__("foo", 1),
    "unknown_top_key": lambda d: d.__setitem__("foo", 1),
    "no_gate_anywhere": lambda d: d.__setitem__("default_gate", None),
    "gate_slot_not_backlog": lambda d: d["default_gate"]["readings"][0]["terms"][0].__setitem__(
        "slot", "self.sent"),
    "gate_bad_sign": lambda d: d["default_gate"]["readings"][0]["terms"][0].__setitem__("sign", 2),
    "inject_unknown_mode": lambda d: d["tiers"][4]["inject"][0].__setitem__("mode", "spin"),
    "inject_without_role": lambda d: d["tiers"][4]["inject"][0].pop("role"),
    "inject_clash_launch": lambda d: d.__setitem__("launch_inject", [dict(d["tiers"][4]["inject"][0])]),
    "streams_c_inconsistent": lambda d: d["tiers"][2].__setitem__("streams_c", 5),
    "beat_origin_nonzero": lambda d: d.__setitem__("beat_origin_rel_ns", 1),
    "poisson_typo": lambda d: d["arrival"].__setitem__("dist", "Poisson"),
}


@pytest.mark.parametrize("name", sorted(SABOTAGES))
def test_validation_rejects(name, table_path):
    doc = _mut(SABOTAGES[name])
    with pytest.raises(TierTableError):
        TierTable.load(table_path(doc))


def test_streams_c_consistent_accepted(table_path):
    doc = good_table_doc()
    t = doc["tiers"][2]
    # 8 steps * 10 ms = 80 ms per stream; C = 16 -> 200 req/s
    t["streams_c"] = 16
    assert TierTable.load(table_path(doc))[2].stream_duration_ns() == 80_000_000


def test_unreadable(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json")
    with pytest.raises(TierTableError):
        TierTable.load(str(p))
    with pytest.raises(TierTableError):
        TierTable.load(str(tmp_path / "missing.json"))
