# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from capcommon import jsonio
from capcommon.exitcodes import ConfigError
from capcommon.identity import Identity, check_same_launch

from conftest import SHA


def test_identity_validation(ident):
    d = ident.to_dict()
    assert Identity.from_dict(d) == ident
    for field, bad in [("experiment", "X0"), ("arm", "-bad"), ("launch_seq", -1),
                       ("launch_seq", True), ("register_sha", "A" * 64), ("role", "Cap"),
                       ("job_id", "a b"), ("restart_count", 1.0)]:
        dd = dict(d)
        dd[field] = bad
        with pytest.raises(ConfigError):
            Identity.from_dict(dd)
    dd = dict(d)
    dd["tier_seq"] = 3
    with pytest.raises(ConfigError):  # tier_seq without label
        Identity.from_dict(dd)
    dd["tier_label"] = "up-1"
    assert Identity.from_dict(dd).tier_seq == 3
    with pytest.raises(ConfigError):
        Identity.from_dict({**d, "extra": 1})
    with pytest.raises(ConfigError):
        Identity.from_dict({**d, "role": None})
    assert Identity.from_dict({**d, "role": None}, allow_null_role=True).role is None


def test_check_same_launch(ident):
    check_same_launch([ident, ident.with_role("nullsse").with_worker(2)])
    with pytest.raises(ConfigError):
        check_same_launch([ident, Identity(**{**ident.to_dict(), "launch_seq": 9})])


def test_stamped_roundtrip(tmp_path, ident):
    p = str(tmp_path / "x.json")
    jsonio.write_json_stamped(p, "capkit.test/1", ident, {"a": 1})
    got_ident, doc = jsonio.read_json_stamped(p, "capkit.test/1")
    assert got_ident == ident and doc["a"] == 1 and doc["written_wall"].endswith("Z")
    with pytest.raises(ConfigError):
        jsonio.read_json_stamped(p, "capkit.other/1")
    with pytest.raises(ConfigError):
        jsonio.write_json_stamped(p, "capkit.test/1", ident, {"identity": 1})
    with pytest.raises(ConfigError):
        jsonio.write_json_stamped(p, "capkit.test/1", None, {})
    assert not [f for f in tmp_path.iterdir() if f.name.startswith(".")]


def test_jsonl_roundtrip_and_truncation(tmp_path, ident):
    p = str(tmp_path / "s.jsonl")
    with jsonio.JsonlWriter(p, "capkit.t/1", ident, ["t", "v"]) as w:
        w.write_row([1, 2])
        w.write_row([3, 4])
        with pytest.raises(ValueError):
            w.write_row([1])
    h, rows = jsonio.read_jsonl(p, "capkit.t/1")
    assert rows == [[1, 2], [3, 4]] and h["_truncated_tail"] is False
    assert h["identity"]["register_sha"] == SHA
    with open(p, "a") as f:
        f.write("[5, ")
    h, rows = jsonio.read_jsonl(p)
    assert rows == [[1, 2], [3, 4]] and h["_truncated_tail"] is True
    bad = tmp_path / "b.jsonl"
    bad.write_text(json.dumps([1]) + "\n")
    with pytest.raises(ConfigError):
        jsonio.read_jsonl(str(bad))
