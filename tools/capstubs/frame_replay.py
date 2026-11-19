# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m capstubs.frame_replay: offline single-thread replay of captured ZeroMQ frames
(design 13.3 E15-A2; frames from capstubs.nx_role --frame-capture-dir).

    python -m capstubs.frame_replay --capture-dir <raw>/e15a --nxd-dir <raw>/nxd \\
        --snx-dir <raw>/snx --out <file.json> [--reps 5] [--impl auto|product|local]

The four operations, each timed per frame on this one thread (thread CPU and wall clock, the mean
of --reps back-to-back calls after one untimed call, i.e. a hot cache):
  result_dumps   result batch: pickle.dumps + HMAC sign   (ZeroMqQueue._prepare_data)
  result_loads   result batch: HMAC check + pickle.loads  (ZeroMqQueue._parse_data)
  request_dumps  request: the same _prepare_data
  request_loads  request: the same _parse_data
--impl product runs the product's own ZeroMqQueue methods (tensorrt_llm.executor.ipc) on an
object that only holds the captured key; local runs the same steps written out here (pickle +
hmac sha256, compare_digest); auto = product when importable. The frames' objects are product
classes, so a real capture is replayed where tensorrt_llm imports (the container).

Self-checks (design 13.3 "A 的自检"); any failure means no timing in the output:
  sampled frames   either direction with 0 captured frames -> rc 2.
  reconciliation   the expected values come from records the replay does not write: result frame
                   seq k must have the bytes and the receive time of the drain's recvs record k and
                   the response count of the worker's k-th non-empty take (snx.<pid>.takes.rec);
                   request frame seq k must carry the client id of the drain's sends record k and
                   the prompt length the worker recorded for that client id (snx reqs); every
                   untampered frame must pass the HMAC check. Any mismatch -> rc 3.
  tamper control   per frame, one byte flipped in the signature (the last 32 bytes) and one byte
                   flipped in the middle of the payload: both must fail the HMAC check -> else rc 4.
  equivalence      re-dumping the parsed object must give the frame byte for byte; only equal
                   frames are timed for *_dumps; when the unequal share of a direction exceeds
                   --max-unequal-share the *_dumps item of that direction is not reported.

Strata: result frames by tier and by responses per frame (power-of-two buckets), request frames
by tier and by prompt length L. Units: us per response (requests: per request), us per frame,
bytes per response. A lower-bound reading (one thread, hot cache, no GIL contention).
"""

import argparse
import gc
import glob
import hashlib
import hmac
import json
import os
import pickle  # nosec B403 -- replays this launch's own captured frames
import platform
import sys
import time

from .frame_capture import DIRECTIONS, read_capture

SCHEMA = "capkit.e15a_replay/1"
RC_OK, RC_NO_FRAMES, RC_RECONCILE, RC_TAMPER, RC_INPUT = 0, 2, 3, 4, 5
SIG = 32
OPS = ("dumps", "loads")


class InputError(Exception):
    pass


def local_codec(key):
    def prepare(obj):
        d = pickle.dumps(obj)  # nosec B301
        return d + hmac.new(key, d, hashlib.sha256).digest()

    def parse(data):
        m, sig = data[:-SIG], data[-SIG:]
        if not hmac.compare_digest(hmac.new(key, m, hashlib.sha256).digest(), sig):
            raise ValueError("HMAC verification failed")
        return pickle.loads(m)  # nosec B301
    return prepare, parse


def product_codec(key):
    from tensorrt_llm.executor.ipc import ZeroMqQueue  # noqa: PLC0415

    class _KeyOnly(ZeroMqQueue):
        def __init__(self, k):  # noqa: D107 -- no socket: only the key and the HMAC flag
            self.use_hmac_encryption = True
            self.hmac_key = k

        def __del__(self):
            pass
    q = _KeyOnly(key)
    return q._prepare_data, q._parse_data


def make_codec(impl, key):
    """(prepare, parse, impl used)."""
    if impl in ("auto", "product"):
        try:
            return product_codec(key) + ("product",)
        except ImportError:
            if impl == "product":
                raise
    return local_codec(key) + ("local",)


def n_responses(obj):
    if isinstance(obj, list):
        return sum(1 for r in obj if r is not None)
    return 0 if obj is None else 1


def rpf_bucket(n):
    if n <= 0:
        return "0"
    lo = 1 << (n.bit_length() - 1)
    return str(lo) if lo == 1 else "%d-%d" % (lo, 2 * lo - 1)


def flip(data, i):
    b = bytearray(data)
    b[i] ^= 0x01
    return bytes(b)


def tamper_check(frames, parse):
    """Per frame: a flipped signature byte and a flipped payload byte must both fail."""
    out = {"frames": 0, "sig_rejected": 0, "payload_rejected": 0, "failures": []}
    for seq, _t, _tier, data in frames:
        out["frames"] += 1
        for kind, pos in (("sig", len(data) - SIG // 2), ("payload", (len(data) - SIG) // 2)):
            try:
                parse(flip(data, pos))
            except Exception:  # noqa: BLE001 -- any rejection counts (product: RuntimeError)
                out[kind + "_rejected"] += 1
                continue
            if len(out["failures"]) < 20:
                out["failures"].append({"seq": seq, "tampered": kind, "accepted": True})
    out["ok"] = out["frames"] > 0 and out["sig_rejected"] == out["payload_rejected"] == \
        out["frames"]
    return out


def load_expected(meta, nxd_dir, snx_dir):
    """recvs [(recv_ns, bytes, responses)], sends [client_id], takes [responses], reqs
    {client_id: prompt_len} from the drain's and the worker's own record files."""
    from .null_executor import read_records  # noqa: PLC0415
    from .nx_driver import RECV_REC, SEND_REC  # noqa: PLC0415

    def recs(path, st):
        if not os.path.exists(path):
            raise InputError("missing %s" % path)
        with open(path, "rb") as f:
            data = f.read()
        return [st.unpack_from(data, i * st.size) for i in range(len(data) // st.size)]
    recvs = recs(os.path.join(nxd_dir, meta.get("recvs", "nxd.lane0.recvs.rec")), RECV_REC)
    sends = [r[1] for r in recs(os.path.join(nxd_dir, meta.get("sends", "nxd.lane0.sends.rec")),
                                SEND_REC)]
    pid = meta.get("engine_pid")
    takes_p = sorted(glob.glob(os.path.join(snx_dir, "snx.%s.takes.rec" % (pid or "*"))))
    if len(takes_p) != 1:
        raise InputError("expected one worker takes record in %s, found %d"
                         % (snx_dir, len(takes_p)))
    takes = [r["responses"] for r in read_records(takes_p[0], "takes")]
    reqs = {}
    for p in glob.glob(os.path.join(snx_dir, "snx.%s.reqs.rec" % (pid or "*"))):
        for r in read_records(p, "reqs"):
            reqs[r["client_id"]] = r["prompt_len"]
    return {"recvs": recvs, "sends": sends, "takes": takes, "reqs": reqs,
            "files": {"takes": takes_p[0]}}


def reconcile(direction, frames, objs, exp):
    """Frame-by-frame check against exp (load_expected); objs[i] is the parsed object or the
    exception string of an untampered frame that failed the check."""
    out = {"checked": 0, "mismatches": 0, "verify_failed": 0, "examples": []}

    def bad(seq, what, got, want):
        out["mismatches"] += 1
        if len(out["examples"]) < 20:
            out["examples"].append({"seq": seq, "item": what, "frame": got, "expected": want})
    for (seq, t_ns, _tier, data), obj in zip(frames, objs):
        out["checked"] += 1
        if isinstance(obj, Exception):
            out["verify_failed"] += 1
            bad(seq, "hmac_or_unpickle", repr(obj), "ok")
            continue
        if direction == "result":
            if seq > len(exp["recvs"]) or seq > len(exp["takes"]):
                bad(seq, "seq", seq, [len(exp["recvs"]), len(exp["takes"])])
                continue
            recv_ns, nbytes, _r = exp["recvs"][seq - 1]
            if nbytes != len(data):
                bad(seq, "bytes", len(data), nbytes)
            if recv_ns != t_ns:
                bad(seq, "recv_ns", t_ns, recv_ns)
            if n_responses(obj) != exp["takes"][seq - 1]:
                bad(seq, "responses", n_responses(obj), exp["takes"][seq - 1])
        else:
            if seq > len(exp["sends"]):
                bad(seq, "seq", seq, len(exp["sends"]))
                continue
            cid = getattr(obj, "id", None)
            if cid != exp["sends"][seq - 1]:
                bad(seq, "client_id", cid, exp["sends"][seq - 1])
            L = len(getattr(obj, "prompt_token_ids", None) or ())
            if exp["reqs"].get(cid) != L:
                bad(seq, "prompt_len", L, exp["reqs"].get(cid))
    out["ok"] = out["checked"] > 0 and out["mismatches"] == 0
    return out


def timed(fn, arg, reps):
    tt, pc = time.thread_time_ns, time.perf_counter_ns
    c0, w0 = tt(), pc()
    for _ in range(reps):
        fn(arg)
    w1, c1 = pc(), tt()
    return (c1 - c0) / reps, (w1 - w0) / reps


def time_frames(direction, frames, objs, eq, prepare, parse, reps):
    rows = []
    for (seq, _t, tier, data), obj, same in zip(frames, objs, eq):
        n = n_responses(obj) if direction == "result" else 1
        row = {"seq": seq, "tier": tier, "n": n, "bytes": len(data),
               "L": len(getattr(obj, "prompt_token_ids", None) or ()) if direction == "request"
               else None, "eq": same}
        row["loads"] = timed(parse, data, reps)
        row["dumps"] = timed(prepare, obj, reps) if same else None
        rows.append(row)
    return rows


def aggregate(rows):
    n = sum(r["n"] for r in rows)
    b = sum(r["bytes"] for r in rows)
    out = {"frames": len(rows), "responses": n, "bytes": b,
           "bytes_per_response": b / n if n else None, "bytes_per_frame": b / len(rows)}
    for op in OPS:
        rr = [r for r in rows if r[op] is not None]
        m = sum(r["n"] for r in rr)
        out[op] = None if not rr or not m else {
            "frames": len(rr), "responses": m,
            "cpu_us_per_response": sum(r[op][0] for r in rr) / m / 1e3,
            "wall_us_per_response": sum(r[op][1] for r in rr) / m / 1e3,
            "cpu_us_per_frame": sum(r[op][0] for r in rr) / len(rr) / 1e3,
            "wall_us_per_frame": sum(r[op][1] for r in rr) / len(rr) / 1e3}
    return out


def strata(direction, rows):
    keys = [("tier", lambda r: str(r["tier"]))]
    keys.append(("responses_per_frame", lambda r: rpf_bucket(r["n"])) if direction == "result"
                else ("prompt_len", lambda r: str(r["L"])))
    out = {"all": aggregate(rows)}
    for name, fn in keys:
        groups = {}
        for r in rows:
            groups.setdefault(fn(r), []).append(r)
        out["by_" + name] = {k: aggregate(v) for k, v in sorted(groups.items())}
    return out


def replay(a):
    """(doc, rc). Timing only when every self-check passes."""
    from capcommon.hashing import sha256_file  # noqa: PLC0415
    meta, keys, frames = read_capture(a.capture_dir)
    doc = {"schema": SCHEMA, "capture_dir": os.path.abspath(a.capture_dir),
           "inputs": {os.path.basename(p): sha256_file(p) for p in sorted(glob.glob(
               os.path.join(a.capture_dir, "frames.*")))},
           "capture_meta": {k: meta.get(k) for k in ("every", "per_tier", "max_bytes",
                                                     "prompt_len", "queues")},
           "reps": a.reps, "max_unequal_share": a.max_unequal_share,
           "python": platform.python_version(), "gc_enabled": gc.isenabled(),
           "directions": {}, "ok": False}
    counts = {d: len(frames[d]) for d in DIRECTIONS}
    doc["sampled_frames"] = counts
    if min(counts.values()) == 0:
        doc["error"] = "sampled frames: %s (a direction with 0 frames gives no numbers)" % counts
        return doc, RC_NO_FRAMES
    if any(keys.get(d) is None for d in DIRECTIONS):
        raise InputError("capture has no HMAC key for %s" % [d for d in DIRECTIONS
                                                            if keys.get(d) is None])
    exp = load_expected(meta, a.nxd_dir, a.snx_dir)
    doc["expected_from"] = {"recvs": meta.get("recvs"), "sends": meta.get("sends"),
                            "takes": exp["files"]["takes"], "reqs": len(exp["reqs"])}
    codecs, objs, rc = {}, {}, RC_OK
    for d in DIRECTIONS:
        fr = frames[d][:a.max_frames] if a.max_frames else frames[d]
        frames[d] = fr
        prepare, parse, used = make_codec(a.impl, keys[d])
        codecs[d] = (prepare, parse)
        doc["impl"] = used
        ob = []
        for _s, _t, _tier, data in fr:
            try:
                ob.append(parse(data))
            except Exception as e:  # noqa: BLE001 -- counted by reconcile as verify_failed
                ob.append(e)
        objs[d] = ob
        rec = reconcile(d, fr, ob, exp)
        tam = tamper_check(fr, parse)
        doc["directions"][d] = {"reconcile": rec, "tamper": tam}
        if not rec["ok"]:
            rc = rc or RC_RECONCILE
        elif not tam["ok"]:
            rc = rc or RC_TAMPER
    if rc != RC_OK:
        doc["error"] = "self-check failed (rc %d): no timing" % rc
        return doc, rc
    for d in DIRECTIONS:
        prepare, parse = codecs[d]
        eq = [prepare(o) == f[3] for o, f in zip(objs[d], frames[d])]
        share = 1.0 - sum(eq) / len(eq)
        dd = doc["directions"][d]
        dd["equivalence"] = {"frames": len(eq), "unequal": len(eq) - sum(eq),
                             "unequal_share": share,
                             "dumps_reported": share <= a.max_unequal_share}
        rows = time_frames(d, frames[d], objs[d], eq, prepare, parse, a.reps)
        st = strata(d, rows)
        if not dd["equivalence"]["dumps_reported"]:
            for agg in [st["all"]] + [x for k, v in st.items() if k != "all"
                                      for x in v.values()]:
                agg["dumps"] = None
        dd["strata"] = st
    doc["ok"] = True
    return doc, RC_OK


def summary_lines(doc):
    out = ["frame_replay impl=%s sampled=%s ok=%s" % (doc.get("impl"), doc.get("sampled_frames"),
                                                       doc["ok"])]
    for d, dd in doc["directions"].items():
        a = (dd.get("strata") or {}).get("all")
        if a is None:
            out.append("  %s: reconcile %s tamper %s (no timing)" % (
                d, dd["reconcile"].get("ok"), dd["tamper"].get("ok")))
            continue
        for op in OPS:
            x = a[op]
            out.append("  %s_%s: %s" % (d, op, "not reported" if x is None else
                                         "%.2f us cpu / %.2f us wall per response, %d frames"
                                         % (x["cpu_us_per_response"],
                                            x["wall_us_per_response"], x["frames"])))
    return out


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.frame_replay", allow_abbrev=False,
                                description=__doc__.split("\n")[0])
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--nxd-dir", required=True, help="the launch's nx_driver files (recvs, sends)")
    p.add_argument("--snx-dir", required=True, help="the launch's NullExecutor files (takes, reqs)")
    p.add_argument("--out", required=True, help="JSON written once (never overwritten)")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--impl", choices=("auto", "product", "local"), default="auto")
    p.add_argument("--max-unequal-share", type=float, default=0.01)
    p.add_argument("--max-frames", type=int, default=0, help="per direction (0 = all)")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.reps < 1 or a.max_frames < 0 or not 0 <= a.max_unequal_share <= 1:
        print("frame_replay: --reps >= 1, --max-frames >= 0, 0 <= --max-unequal-share <= 1",
              file=sys.stderr)
        return RC_INPUT
    if os.path.exists(a.out):
        print("frame_replay: %s exists (never overwritten)" % a.out, file=sys.stderr)
        return RC_INPUT
    try:
        doc, rc = replay(a)
    except (InputError, OSError, ValueError, KeyError) as e:
        print("frame_replay: %s" % e, file=sys.stderr)
        return RC_INPUT
    doc["rc"] = rc
    with open(a.out, "x", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True, default=str)
        f.write("\n")
    print("\n".join(summary_lines(doc)))
    return rc


if __name__ == "__main__":
    sys.exit(main())
