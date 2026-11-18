# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cape12.schedgen: Poisson tables (seeded shape, rate scaling), trajectory replay from a reqlog,
dose sequences; and that cape12.reqlog_l2a accepts what it writes."""

import json

import pytest

from cape12 import reqlog_l2a as rl
from cape12 import schedgen as sg


def test_poisson_shape_depends_on_seed_only_and_scales_with_rate():
    a, ua = sg.poisson_offsets(5, 160, 2.0)
    b, ub = sg.poisson_offsets(5, 160, 4.0)
    c, _ = sg.poisson_offsets(6, 160, 2.0)
    assert len(a) == 160 and a[0] == 0.0 and ua == ub
    assert all(x == pytest.approx(2 * y) for x, y in zip(a, b))
    assert a != c and all(q >= p for p, q in zip(a, a[1:]))
    # mean inter-arrival close to 1 / rate (159 gaps; loose bound)
    assert 0.35 < a[-1] / 159 < 0.7


def test_poisson_cli_output_is_accepted_by_the_wrapper(tmp_path):
    out = tmp_path / "s.json"
    assert sg.main(["poisson", "--seed", "1", "--n", "20", "--rate", "3.6", "--out",
                    str(out)]) == 0
    cfg = rl.load_config({"E12_REQLOG_DIR": str(tmp_path), "E12_MODE": "open",
                          "E12_SCHEDULE": str(out)}, ["--max-concurrency", "8"])
    assert len(cfg["offsets_s"]) == 20 and cfg["argv"] == []
    assert sg.main(["poisson", "--seed", "1", "--n", "20", "--rate", "3.6", "--out",
                    str(out)]) == 3  # refuses to overwrite


def _reqlog(tmp_path, hdr, ts):
    p = tmp_path / "r.jsonl"
    lines = [dict(hdr, hdr=1)] + [{"i": i, "t_call": t} for i, t in enumerate(ts)]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return str(p)


def test_replay_takes_call_times_relative_to_the_first(tmp_path):
    src = _reqlog(tmp_path, {"mode": "closed", "warmup": False}, [10.0, 10.5, 10.5, 12.25])
    out = tmp_path / "rep.json"
    assert sg.main(["replay", "--reqlog", src, "--n", "4", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["offsets_s"] == [0.0, 0.5, 0.5, 2.25] and doc["kind"] == "replay"
    rl.load_config({"E12_REQLOG_DIR": str(tmp_path), "E12_MODE": "replay",
                    "E12_SCHEDULE": str(out)}, [])


@pytest.mark.parametrize("hdr, n", [({"mode": "closed", "warmup": True}, 2),
                                    ({"mode": "open", "warmup": False}, 2),
                                    ({"mode": "closed", "warmup": False}, 3)])
def test_replay_refusals(tmp_path, hdr, n):
    src = _reqlog(tmp_path, hdr, [1.0, 2.0])
    assert sg.main(["replay", "--reqlog", src, "--n", str(n), "--out",
                    str(tmp_path / "x.json")]) == 3


def test_dose_sequences_are_seeded_permutations_with_reverses(tmp_path):
    out = tmp_path / "d.json"
    assert sg.main(["dose", "--levels", "0,2,3,4,8", "--sequences", "4", "--seed", "11",
                    "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    seqs = doc["sequences"]
    assert len(seqs) == 4
    for s in seqs:
        assert sorted(s["forward"]) == [0.0, 2.0, 3.0, 4.0, 8.0]
        assert s["reverse"] == s["forward"][::-1]
    assert sg.dose_sequences([0.0, 2.0, 3.0, 4.0, 8.0], 4, 11) == seqs
    assert len({tuple(s["forward"]) for s in seqs}) > 1
