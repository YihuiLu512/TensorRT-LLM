# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import time

import pytest

from capcommon import hist, snapshot


def test_registry_and_decode(tmp_path, ident):
    reg = snapshot.SlotRegistry()
    i_sent = reg.counter("self.sent")
    assert reg.counter("self.sent") == i_sent  # idempotent
    i_inf = reg.gauge("backlog.inflight")
    h = reg.hist("lag.loop")
    with pytest.raises(snapshot.SnapshotError):
        reg.gauge("self.sent")
    with pytest.raises(snapshot.SnapshotError):
        reg.counter("Bad.name")
    with pytest.raises(snapshot.SnapshotError):
        reg.counter("nodot")
    reg.values[i_sent] += 5
    reg.values[i_inf] = 3
    for v in (10, 2000, 3_000_000, -1):
        h.add(v)
    w = snapshot.SnapshotWriter(reg, str(tmp_path), 50_000_000,
                                meta_extra={"identity": ident.to_dict(), "role_hint": "x"})
    w.write_now()
    snap = snapshot.read_snapshot_file(w.path)
    assert snap.seq == 1 and snap.pid == os.getpid()
    assert snap.values["self.sent"] == 5 and snap.values["backlog.inflight"] == 3
    counts = snap.values["lag.loop"]
    assert len(counts) == hist.NBINS and sum(counts) == 4
    assert counts[hist.bucket_index(3_000_000)] == 1
    assert snap.values["lag.loop.n"] == 4 and snap.values["lag.loop.neg"] == 1
    assert snap.values["lag.loop.max"] == 3_000_000 and snap.values["lag.loop.sum"] == 3_002_010
    assert snap.meta["identity"]["role"] == "capload" and snap.meta["role_hint"] == "x"
    assert snap.meta["starttime_ticks"] == snapshot.proc_starttime_ticks()
    assert snapshot.list_snapshot_files(str(tmp_path)) == {os.getpid(): w.path}
    assert snapshot.list_snapshot_files(str(tmp_path / "nope")) == {}


def test_writer_thread_and_late_registration(tmp_path):
    reg = snapshot.SlotRegistry()
    i = reg.counter("self.ticks")
    w = snapshot.SnapshotWriter(reg, str(tmp_path), 20_000_000).start()
    try:
        reg.values[i] = 1
        time.sleep(0.1)
        j = reg.gauge("backlog.late")  # registered after start
        reg.values[j] = 9
        time.sleep(0.1)
    finally:
        w.stop()
    snap = snapshot.read_snapshot_file(w.path)
    assert snap.seq >= 5, snap.seq  # initial + ~10 periodic + final; lenient on a busy VM
    assert snap.values["self.ticks"] == 1 and snap.values["backlog.late"] == 9
    assert w.errors == 0
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_corrupt_rejected(tmp_path):
    reg = snapshot.SlotRegistry()
    reg.counter("self.x")
    w = snapshot.SnapshotWriter(reg, str(tmp_path), 1_000_000)
    w.write_now()
    data = open(w.path, "rb").read()
    with pytest.raises(snapshot.SnapshotError):
        snapshot.decode_snapshot(data[:-8])
    with pytest.raises(snapshot.SnapshotError):
        snapshot.decode_snapshot(b"XXXXXXXX" + data[8:])
    with pytest.raises(snapshot.SnapshotError):
        snapshot.SnapshotWriter(reg, str(tmp_path), 0)
