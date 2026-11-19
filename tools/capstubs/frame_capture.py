# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.frame_capture: ZeroMQ frame capture of the proxy <-> rank 0 hop (design 13.3 E15-A2).

Off unless capstubs.nx_role is given --frame-capture-dir (no environment switch). Two directions,
both in the nx_role process (the driver; nothing runs in the worker, no product file changes):

  result   every frame the nx_driver raw / deserialize drain receives on the proxy's result lane
           (K = 1, tcp PAIR): drain_loop hands it to record("result", ...) next to the drain's own
           LaneStats.received, so result frame seq k is the drain's recvs record k and the worker's
           k-th non-empty await_responses take (snx.<pid>.takes.rec).
  request  install_request_capture wraps ZeroMqQueue._prepare_data (class attribute, this process
           only) and records the returned bytes when the queue is the proxy's request_queue and the
           object is a request (default: type name GenerationRequest); other objects on that queue
           (shutdown sentinel, cancels) are only counted. Request frame seq k is the drain's sends
           record k (nx_driver send_loop, one sender thread).

Every frame of a direction gets a sequence number (1, 2, ...) and its tier (plan.tier_seq_at of the
event time). A frame is written when seq % every == 0, its tier has fewer than per_tier captured
frames of that direction, and the bytes captured so far are below max_bytes. File
frames.<direction>.bin = records of FRAME_HDR (seq, t_ns monotonic, tier_seq, nbytes) followed by
the frame bytes (pickle + 32-byte HMAC, as sent). close() writes frames.meta.json (counts per tier)
and frames.keys.json (mode 0600: the HMAC keys of the two queues; they stay in this launch's raw
directory, which is the experiment directory).
"""

import json
import os
import struct
import threading
import time

SCHEMA_META = "capkit.frame_capture/1"
SCHEMA_KEYS = "capkit.frame_capture_keys/1"
FRAME_HDR = struct.Struct("<qqhI")
FRAME_FIELDS = ("seq", "t_ns", "tier_seq", "nbytes")
DIRECTIONS = ("result", "request")
REQUEST_TYPE = "GenerationRequest"


class _Dir:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "xb")  # noqa: SIM115 -- closed in FrameCapture.close
        self.seq = 0
        self.captured = 0
        self.bytes_seen = 0
        self.bytes_captured = 0
        self.by_tier = {}
        self.other = {}
        self.lock = threading.Lock()


class FrameCapture:
    """Frame writer of both directions (module docstring); record() is thread-safe."""

    def __init__(self, out_dir, plan, every=1, per_tier=1000, max_bytes=512 << 20,
                 clock=None):
        if every < 1 or per_tier < 1 or max_bytes < 1:
            raise ValueError("every, per_tier and max_bytes must be >= 1")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.plan = out_dir, plan
        self.every, self.per_tier, self.max_bytes = int(every), int(per_tier), int(max_bytes)
        self.dirs = {d: _Dir(os.path.join(out_dir, "frames.%s.bin" % d)) for d in DIRECTIONS}
        self.keys = {}
        self.queues = {}
        self.closed = False
        self._restore = None
        self.clock = clock or time.monotonic_ns

    def record(self, direction, data, t_ns):
        """One frame of `direction` (bytes as on the wire) at monotonic time t_ns."""
        d = self.dirs[direction]
        n = len(data)
        with d.lock:
            if d.f.closed:
                return False
            d.seq += 1
            seq = d.seq
            tier = self.plan.tier_seq_at(t_ns)
            t = d.by_tier.setdefault(tier, [0, 0])
            t[0] += 1
            d.bytes_seen += n
            take = (seq % self.every == 0 and t[1] < self.per_tier
                    and self._total_captured() < self.max_bytes)
            if not take:
                return False
            d.f.write(FRAME_HDR.pack(seq, t_ns, tier, n))
            d.f.write(data)
            t[1] += 1
            d.captured += 1
            d.bytes_captured += n
            return True

    def _total_captured(self):
        return sum(x.bytes_captured for x in self.dirs.values())

    def count_other(self, direction, kind):
        d = self.dirs[direction]
        with d.lock:
            d.other[kind] = d.other.get(kind, 0) + 1

    def set_queue(self, direction, queue, name):
        """Record the HMAC key and the name of the queue a direction belongs to."""
        key = getattr(queue, "hmac_key", None)
        self.keys[direction] = key.hex() if key else None
        self.queues[direction] = {"name": name,
                                  "use_hmac_encryption": bool(getattr(
                                      queue, "use_hmac_encryption", False))}

    def close(self, extra=None):
        """Stop recording (later frames are ignored), restore the patch, write meta and keys."""
        if self.closed:
            return None
        for d in self.dirs.values():
            with d.lock:
                d.f.flush()
                d.f.close()
        self.closed = True
        if self._restore is not None:
            self._restore()
            self._restore = None
        meta = {"schema": SCHEMA_META, "pid": os.getpid(), "every": self.every,
                "per_tier": self.per_tier, "max_bytes": self.max_bytes,
                "header": {"format": FRAME_HDR.format, "fields": list(FRAME_FIELDS)},
                "queues": self.queues, "keys_file": "frames.keys.json",
                "directions": {}}
        for name, d in self.dirs.items():
            meta["directions"][name] = {
                "file": os.path.basename(d.path), "frames_seen": d.seq,
                "frames_captured": d.captured, "bytes_seen": d.bytes_seen,
                "bytes_captured": d.bytes_captured, "other_objects": d.other,
                "by_tier": {str(k): {"seen": v[0], "captured": v[1]}
                            for k, v in sorted(d.by_tier.items())}}
        meta.update(extra or {})
        _write_json(os.path.join(self.out_dir, "frames.meta.json"), meta)
        kp = os.path.join(self.out_dir, "frames.keys.json")
        fd = os.open(kp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"schema": SCHEMA_KEYS, "hmac_key_hex": self.keys,
                       "note": "HMAC keys of this launch's proxy queues; experiment directory "
                               "only"}, f, indent=1, sort_keys=True)
            f.write("\n")
        return meta


def _write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def install_request_capture(cap, queue_cls, target, is_request=None):
    """Wrap queue_cls._prepare_data (the class attribute, this process) so that frames prepared
    by `target` (the proxy's request_queue) go to cap.record("request", ...); objects that are
    not requests (is_request false; default: type name GenerationRequest) are only counted.
    cap.close() restores the original. Returns the wrapper."""
    orig = queue_cls.__dict__["_prepare_data"]
    is_request = is_request or (lambda obj: type(obj).__name__ == REQUEST_TYPE)
    clock = cap.clock

    def _prepare_data(self, obj):
        data = orig(self, obj)
        if self is target:
            if is_request(obj):
                cap.record("request", data, clock())
            else:
                cap.count_other("request", type(obj).__name__)
        return data
    _prepare_data.__wrapped__ = orig
    queue_cls._prepare_data = _prepare_data

    def restore():
        if queue_cls.__dict__.get("_prepare_data") is _prepare_data:
            queue_cls._prepare_data = orig
    cap._restore = restore
    return _prepare_data


def attach_proxy(cap, proxy, queue_cls=None):
    """Capture both directions of a single-frontend product proxy (K = 1): keys and names of
    proxy.request_queue and proxy.result_queue.queue (FusedIpcQueue's ZeroMqQueue), and the
    request wrapper on ZeroMqQueue._prepare_data."""
    if queue_cls is None:
        from tensorrt_llm.executor.ipc import ZeroMqQueue as queue_cls  # noqa: PLC0415,N813
    res = getattr(proxy.result_queue, "queue", proxy.result_queue)
    cap.set_queue("result", res, getattr(res, "name", None) or "proxy_result_queue")
    cap.set_queue("request", proxy.request_queue,
                  getattr(proxy.request_queue, "name", None) or "proxy_request_queue")
    install_request_capture(cap, queue_cls, proxy.request_queue)


def read_frames(path):
    """[(seq, t_ns, tier_seq, data)] of one frames.<direction>.bin file; a truncated last record
    raises ValueError."""
    with open(path, "rb") as f:
        buf = f.read()
    out, i, h = [], 0, FRAME_HDR.size
    while i < len(buf):
        if i + h > len(buf):
            raise ValueError("%s: truncated header at byte %d" % (path, i))
        seq, t_ns, tier, n = FRAME_HDR.unpack_from(buf, i)
        i += h
        if i + n > len(buf):
            raise ValueError("%s: truncated frame seq %d" % (path, seq))
        out.append((seq, t_ns, tier, bytes(buf[i:i + n])))
        i += n
    return out


def read_capture(cap_dir):
    """(meta, keys {direction: bytes or None}, frames {direction: read_frames(...)})."""
    with open(os.path.join(cap_dir, "frames.meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(cap_dir, meta.get("keys_file", "frames.keys.json")), "r",
              encoding="utf-8") as f:
        kd = json.load(f)["hmac_key_hex"]
    keys = {d: (bytes.fromhex(v) if v else None) for d, v in kd.items()}
    frames = {d: read_frames(os.path.join(cap_dir, meta["directions"][d]["file"]))
              for d in DIRECTIONS}
    return meta, keys, frames
