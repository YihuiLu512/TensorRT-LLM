# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import array

import pytest

from capcommon import capbin


def test_roundtrip_all_typecodes(tmp_path, ident):
    cols = [(tc, array.array(tc, [1, 2, 3, 0, 5] if tc not in "fd" else [1.5, -2, 3, 0, 5]))
            for tc in capbin.TYPECODES]
    p = str(tmp_path / "x.capbin")
    capbin.write_capbin(p, "test", ident, cols, {"k": 1})
    hdr, got = capbin.read_capbin(p)
    assert hdr["n_rows"] == 5 and hdr["meta"] == {"k": 1} and hdr["identity"]["role"] == "capload"
    for name, arr in cols:
        assert got[name].tolist() == arr.tolist() and got[name].typecode == arr.typecode
    _h, sub = capbin.read_capbin(p, columns=["q"], stride=(1, 2))
    assert list(sub) == ["q"] and sub["q"].tolist() == [2, 0]


def test_errors(tmp_path):
    p = tmp_path / "bad.capbin"
    p.write_bytes(b"NOTCAPBN" + b"\0" * 16)
    with pytest.raises(capbin.CapbinError):
        capbin.read_capbin(str(p))
    good = capbin.encode_capbin("t", None, [("a", array.array("q", range(10)))])
    p.write_bytes(good[:-8])
    with pytest.raises(capbin.CapbinError):
        capbin.read_capbin(str(p))
    with pytest.raises(capbin.CapbinError):
        capbin.encode_capbin("t", None, [("a", array.array("q", [1])), ("b", array.array("q"))])
    with pytest.raises(capbin.CapbinError):
        capbin.encode_capbin("t", None, [("a", array.array("l", [1]))])
    p.write_bytes(good)
    with pytest.raises(capbin.CapbinError):
        capbin.read_capbin(str(p), columns=["zz"])


def test_record_array_overflow_and_split(tmp_path, ident):
    ra = capbin.RecordArray([("req_id", "q"), ("t", "q"), ("tier_seq", "h")], capacity=5)
    rows = [(0, 10, 0), (1, 11, 0), (2, 12, 2), (3, 13, 2), (4, 14, 0)]
    for r in rows:
        assert ra.append(*r)
    assert not ra.append(5, 15, 2) and ra.overflow == 1 and len(ra) == 5
    with pytest.raises(capbin.CapbinError):
        ra.append(1, 2)
    written = ra.dump_split(lambda v: str(tmp_path / ("t%d" % v) / "r.capbin"), "test.rows",
                            lambda v: ident.with_tier(v, "x%d" % v))
    assert sorted(written) == [0, 2]
    h0, c0 = capbin.read_capbin(written[0])
    assert c0["req_id"].tolist() == [0, 1, 4] and h0["meta"]["overflow"] == 1
    assert h0["identity"]["tier_seq"] == 0
    _h2, c2 = capbin.read_capbin(written[2])
    assert c2["t"].tolist() == [12, 13]
    p = str(tmp_path / "all.capbin")
    ra.dump(p, "test.rows", ident)
    _h, c = capbin.read_capbin(p)
    assert c["req_id"].tolist() == [0, 1, 2, 3, 4]
    empty = capbin.RecordArray([("a", "q")], capacity=0)
    assert not empty.append(1) and empty.overflow == 1
