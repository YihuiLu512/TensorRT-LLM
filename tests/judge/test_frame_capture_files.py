# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The frame files of capstubs.frame_capture (frames.{result,request}.bin, nx_role
--frame-capture-dir somewhere under raw/) are .bin but not snapshots: the judge does not open
them, neither in the identity scan of raw/ nor in a tier directory. Only those two exact names
are skipped: any other .bin that is not a readable snapshot (a damaged snapshot, a foreign .bin
under another name) still makes the tree INPUT_INVALID."""

import json
import os
import struct
import sys

import pytest
from conftest import SAMPLES, TOOLS, judge_mod

J = judge_mod("judge")
TD = judge_mod("treedata")


@pytest.fixture
def good(tmp_path, monkeypatch):
    monkeypatch.setenv("CAPKIT_CODE_SHA", "b" * 40)
    sys.path.insert(0, SAMPLES)
    try:
        import make_samples
    finally:
        sys.path.remove(SAMPLES)
    W = make_samples.build(str(tmp_path), "good", {}, make_samples.SAMPLES["good"][1])
    with open(os.path.join(W, "expect.json")) as f:
        tree = os.path.join(W, json.load(f)["tree"])
    return W, tree


def _frames(path, n=3):
    """A frame file in the frame_capture layout (FRAME_HDR <qqhI + payload), not a snapshot."""
    with open(path, "wb") as f:
        for seq in range(1, n + 1):
            payload = b"\x80\x04" + os.urandom(40)
            f.write(struct.pack("<qqhI", seq, 1000 * seq, 2, len(payload)) + payload)


def _capture_dir(tree, rel="raw/e15a", meta=True):
    d = os.path.join(tree, rel)
    os.makedirs(d, exist_ok=True)
    for name in sorted(TD.FRAME_CAPTURE_FILES):
        _frames(os.path.join(d, name))
    if meta:
        with open(os.path.join(d, "frames.meta.json"), "w") as f:
            json.dump({"schema": "capkit.frame_capture/1", "role": "nx"}, f)
    return d


def _judge(W, tree):
    rc = J.main(["--tree", tree, "--workdir", W])
    jdir = os.path.join(tree, "judge")
    if not os.path.isdir(jdir):
        (a,) = os.listdir(os.path.join(tree, J.ATTEMPTS_DIR))
        jdir = os.path.join(tree, J.ATTEMPTS_DIR, a)
    with open(os.path.join(jdir, "inputs.json")) as f:
        return rc, json.load(f), jdir


def test_names_are_the_ones_frame_capture_writes():
    sys.path.insert(0, TOOLS)
    from capstubs import frame_capture
    assert TD.FRAME_CAPTURE_FILES == {"frames.%s.bin" % d for d in frame_capture.DIRECTIONS}


def test_frame_files_in_raw_are_ignored(good):
    W, tree = good
    _capture_dir(tree)
    rc, inputs, jdir = _judge(W, tree)
    assert rc == 0
    assert inputs["identity_scan"]["frame_capture_files_skipped"] == 2
    assert os.path.isfile(os.path.join(jdir, "verdicts.json"))
    assert not any("frames." in k for k in inputs["files_read"])


def test_without_frame_files_the_identity_scan_is_unchanged(good):
    W, tree = good
    rc, inputs, _ = _judge(W, tree)
    assert rc == 0
    assert set(inputs["identity_scan"]) == {"files_with_identity"}


def test_frame_files_in_a_tier_directory_are_ignored(good):
    W, tree = good
    _capture_dir(tree, "raw/tiers/002_up-r20/snapshots", meta=False)
    rc, inputs, _ = _judge(W, tree)
    assert rc == 0
    assert inputs["identity_scan"]["frame_capture_files_skipped"] == 2


def test_damaged_snapshot_is_still_input_invalid(good):
    """Positive control: a real snapshot whose magic is overwritten."""
    W, tree = good
    _capture_dir(tree)
    p = os.path.join(tree, "raw/tiers/002_up-r20/snapshots/capload.1001.wend.bin")
    with open(p, "r+b") as f:
        f.write(b"XXXXXXXX")
    rc, inputs, _ = _judge(W, tree)
    assert rc == 13
    assert inputs["status"] == "INPUT_INVALID"
    assert "capload.1001.wend.bin" in json.dumps(inputs)
    assert not os.path.lexists(os.path.join(tree, "judge"))


def test_other_non_snapshot_bin_is_still_input_invalid(good):
    """Only the two exact names are skipped: the same bytes under another name are read."""
    W, tree = good
    d = _capture_dir(tree)
    _frames(os.path.join(d, "frames.other.bin"))
    rc, inputs, _ = _judge(W, tree)
    assert rc == 13
    assert "frames.other.bin" in json.dumps(inputs)
