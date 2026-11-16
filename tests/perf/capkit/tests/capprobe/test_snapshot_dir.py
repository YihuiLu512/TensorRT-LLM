# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capcommon.snapshot.SnapshotWriter only writes into a real directory owned by this user that
group and others cannot write, and never follows a symlink at its temporary file name (the
writer behind every capprobe snapshot)."""

import os
import stat

import pytest

from capcommon import snapshot
from capcommon.snapshot import SlotRegistry, SnapshotError, SnapshotWriter, read_snapshot_file

INTERVAL_NS = 10 ** 9


def writer(shm_dir):
    reg = SlotRegistry()
    reg.counter("count.H1.arrivals")
    return SnapshotWriter(reg, str(shm_dir), INTERVAL_NS)


def test_new_dir_is_private_and_writes(tmp_path):
    d = tmp_path / "a" / "b" / "snap"
    w = writer(d)
    assert stat.S_IMODE(os.lstat(d).st_mode) & 0o077 == 0
    w.write_now()
    assert read_snapshot_file(w.path).seq == 1


def test_existing_own_dir_is_accepted(tmp_path):
    d = tmp_path / "snap"
    d.mkdir(mode=0o755)
    writer(d).write_now()
    assert os.listdir(d) == ["%d.bin" % os.getpid()]


@pytest.mark.parametrize("suffix", ["", "/"])
def test_symlinked_dir_refused(tmp_path, suffix):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(SnapshotError, match="symlink"):
        writer(str(link) + suffix)
    assert os.listdir(real) == []


def test_non_directory_refused(tmp_path):
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(SnapshotError, match="not a directory"):
        writer(f)


def test_other_users_dir_refused(tmp_path, monkeypatch):
    d = tmp_path / "snap"
    d.mkdir()
    owner = os.lstat(d).st_uid
    monkeypatch.setattr(snapshot.os, "geteuid", lambda: owner + 1)
    with pytest.raises(SnapshotError, match="owned by uid"):
        writer(d)


@pytest.mark.parametrize("mode", [0o770, 0o707, 0o722, 0o1777])
def test_group_or_other_writable_dir_refused(tmp_path, mode):
    d = tmp_path / "snap"
    d.mkdir()
    os.chmod(d, mode)
    with pytest.raises(SnapshotError, match="must not be writable by group or others"):
        writer(d)
    assert os.listdir(d) == []


def test_symlink_at_tmp_name_not_followed(tmp_path):
    d = tmp_path / "snap"
    w = writer(d)
    victim = tmp_path / "victim"
    victim.write_text("keep me")
    os.symlink(victim, w._tmp)
    with pytest.raises(OSError):
        w.write_now()
    assert victim.read_text() == "keep me"
    assert not os.path.exists(w.path)
