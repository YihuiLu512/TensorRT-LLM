# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m caplaunch.httpprobe: E0(c) HTTP smoke against a running frontend (stdlib only).

Sends each request of --requests (a JSON list of {"name", "endpoint", "body"}) once, streams the
SSE body, and records per frame: byte length, the text delta (completions `choices[0].text`,
chat `choices[0].delta.content`), whether the delta is empty, and finish_reason / usage. This
answers "does the frontend send a chunk when the text delta is empty" (X2 premise) and "do
/v1/completions and /v1/chat/completions each answer" (S-FE + real tokenizer + frontend).
Frames are also counted with capcommon.wire.SseCounter so the two counts can be compared.

It writes --result, then (with --hold) stays alive until SIGTERM so the orchestrator can run it as
an aux role (readiness file_exists on --result). No timing here is a measurement result.
"""

import argparse
import http.client
import json
import os
import signal
import sys
import threading
import time

from capcommon.cli import add_identity_args
from capcommon.jsonio import atomic_write_bytes, dumps
from capcommon.wire import SseCounter


def parse_frames(body):
    """Split an SSE body into frames; returns (frames, crlf_seen, leftover_bytes)."""
    crlf = b"\r\n" in body
    norm = body.replace(b"\r\n", b"\n")
    parts = norm.split(b"\n\n")
    leftover = parts.pop() if parts else b""
    return [p for p in parts if p], crlf, len(leftover)


def frame_info(frame):
    out = {"bytes": len(frame), "done": frame == b"data: [DONE]"}
    if out["done"] or not frame.startswith(b"data: "):
        return out
    try:
        obj = json.loads(frame[6:].decode("utf-8"))
    except ValueError:
        out["unparsable"] = True
        return out
    ch = obj.get("choices") or []
    if ch:
        c0 = ch[0]
        text = c0.get("text")
        if text is None:
            text = (c0.get("delta") or {}).get("content")
        out["text_len"] = None if text is None else len(text)
        out["empty_text"] = text in ("", None)
        out["finish_reason"] = c0.get("finish_reason")
    if obj.get("usage") is not None:
        out["usage"] = obj["usage"]
    return out


def one_request(host, port, req, timeout_s):
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    t0 = time.monotonic_ns()
    try:
        conn.request("POST", req["endpoint"], body=json.dumps(req["body"]).encode(),
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        body = resp.read()
        status = resp.status
    except OSError as e:
        return {"name": req["name"], "error": repr(e)}
    finally:
        conn.close()
    frames, crlf, leftover = parse_frames(body)
    sc = SseCounter()
    sc.feed(body)
    infos = [frame_info(f) for f in frames]
    data = [i for i in infos if not i["done"]]
    return {"name": req["name"], "endpoint": req["endpoint"], "status": status,
            "elapsed_ns": time.monotonic_ns() - t0, "body_bytes": len(body),
            "crlf_seen": crlf, "leftover_bytes": leftover, "frames": infos,
            "n_frames_excl_done": len(data),
            "n_empty_text_frames": sum(1 for i in data if i.get("empty_text")),
            "done_seen": any(i["done"] for i in infos),
            "ssecounter": {"frames": sc.frames, "done_seen": sc.done_seen}}


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m caplaunch.httpprobe", description=__doc__)
    add_identity_args(p)  # passed by caplaunch for every capkit role; recorded, not used
    p.add_argument("--run-dir", required=True)
    p.add_argument("--stop-grace-s", type=float, required=True)
    p.add_argument("--cpus", required=True)
    p.add_argument("--addr", required=True, help="host:port of the frontend")
    p.add_argument("--requests", required=True, help="JSON file: list of requests")
    p.add_argument("--timeout-s", type=float, required=True)
    p.add_argument("--result", required=True, help="output JSON (readiness: file_exists)")
    p.add_argument("--hold", action="store_true", help="stay alive until SIGTERM after writing")
    a = p.parse_args(argv)
    host, _, port = a.addr.rpartition(":")
    with open(a.requests) as f:
        reqs = json.load(f)
    res = [one_request(host.strip("[]"), int(port), r, a.timeout_s) for r in reqs]
    atomic_write_bytes(a.result, (dumps({"schema": "capkit.httpprobe/1", "addr": a.addr,
                                         "role": a.role, "launch_seq": a.launch_seq,
                                         "results": res}) + "\n").encode())
    if a.hold:
        done = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: done.set())
        while not done.wait(0.2):
            pass
    bad = [r["name"] for r in res if r.get("error") or r.get("status") != 200]
    return 4 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
