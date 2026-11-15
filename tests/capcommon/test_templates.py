# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capcommon.templates (placeholders, --param values) and capcommon.jobinputs.load_verified: the
one implementation caplaunch, capjobs and capjudge share (RS15)."""

import json

import pytest

from capcommon import jobinputs as JI
from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_bytes
from capcommon.templates import parse_params, placeholders


def test_placeholders():
    assert placeholders("{run_dir}/x/{param:a_b.c-1}{input:prebaked}") == [
        ("run_dir", None), ("param", "a_b.c-1"), ("input", "prebaked")]
    assert placeholders("{Bad} {a:b c} plain") == []


def test_parse_params():
    assert parse_params(["a=1", "b=c=d", "e="]) == {"a": "1", "b": "c=d", "e": ""}
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        parse_params(["noeq"])
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        parse_params(["=v"])
    with pytest.raises(ConfigError, match="given twice"):
        parse_params(["a=1", "a=2"])


def _pair(doc, **rec):
    data = json.dumps(doc).encode()
    record = dict({"schema": JI.RECORD_SCHEMA, "kind": "prebaked", "module": "capstubs.prebaked",
                   "rc": 0, "out_sha256": sha256_bytes(data),
                   "inputs": {"tier_table": "1" * 64}}, **rec)
    return data, json.dumps(record).encode()


def test_load_verified():
    reg = {"tier_table": "1" * 64}
    data, rec = _pair({"schema": "capkit.prebaked/1"})
    assert JI.load_verified("prebaked", data, rec, reg) == sha256_bytes(data)
    for args, msg in (
            ((b"{not json", rec), "not JSON"),
            ((data, b"\xff"), "not JSON"),
            (_pair({"schema": "capkit.reqbytes/1"}), "not a capkit.prebaked/1 file"),
            (_pair({"schema": "capkit.prebaked/1"}, rc=1), "record rc 1"),
            (_pair({"schema": "capkit.prebaked/1"}, out_sha256="0" * 64), "out_sha256"),
            (_pair({"schema": "capkit.prebaked/1"}, inputs={"tier_table": "2" * 64}),
             "built from tier_table")):
        with pytest.raises(ConfigError, match=msg):
            JI.load_verified("prebaked", *args, reg)
