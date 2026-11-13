# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""caplaunch.httpprobe against a local SSE server, and caplaunch.smoke on a fake tree."""

import http.server
import json
import os
import sys
import threading

import conftest as cf
from caplaunch import httpprobe, smoke

CONFIGS = os.path.join(cf.C, "configs", "e0")
SID = ["--experiment", "E0", "--arm", "smoke", "--launch-seq", "0", "--job-id", "local-x",
       "--restart-count", "0", "--register-sha", "a" * 64, "--role", "smoke"]
FRAMES = [{"choices": [{"text": "a", "finish_reason": None}]},
          {"choices": [{"text": "", "finish_reason": None}]},  # an empty text delta
          {"choices": [{"delta": {"content": "b"}, "finish_reason": "stop"}]},
          {"choices": [], "usage": {"completion_tokens": 3}}]


class _Sse(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        body = b"".join(b"data: " + json.dumps(f).encode() + b"\n\n" for f in FRAMES)
        body += b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def _ident(role="httpprobe"):
    return ["--experiment", "E0", "--arm", "a", "--launch-seq", "0", "--job-id", "local-x",
            "--restart-count", "0", "--register-sha", "a" * 64, "--role", role,
            "--run-dir", "/nonexistent", "--stop-grace-s", "1.0", "--cpus", "0"]


def _probe(tmp_path, port):
    reqs = tmp_path / "reqs.json"
    reqs.write_text(json.dumps([{"name": "c", "endpoint": "/v1/completions", "body": {}},
                                {"name": "chat", "endpoint": "/v1/chat/completions",
                                 "body": {}}]))
    out = tmp_path / "tree" / "raw" / "httpprobe.json"
    out.parent.mkdir(parents=True)
    rc = httpprobe.main(_ident() + ["--addr", "127.0.0.1:%d" % port, "--requests", str(reqs),
                                    "--timeout-s", "5", "--result", str(out)])
    return rc, json.loads(out.read_text())


def test_httpprobe_counts_frames_and_empty_deltas(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Sse)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        rc, doc = _probe(tmp_path, srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()
    assert rc == 0 and len(doc["results"]) == 2
    r = doc["results"][0]
    assert r["status"] == 200 and r["done_seen"] and not r["crlf_seen"]
    assert r["n_frames_excl_done"] == 4 == r["ssecounter"]["frames"]
    assert r["n_empty_text_frames"] == 1  # the "" delta; the usage-only frame has no choices
    assert [f.get("text_len") for f in r["frames"][:3]] == [1, 0, 1]


def test_httpprobe_connection_refused_is_an_error(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Sse)
    port = srv.server_address[1]
    srv.server_close()  # nothing listens on port now
    rc, doc = _probe(tmp_path, port)
    assert rc == 4 and all("error" in r for r in doc["results"])


def _cfg(tmp_path, **over):
    with open(os.path.join(CONFIGS, "e0c-smoke.json")) as f:
        cfg = json.load(f)
    cfg.update(over)
    p = tmp_path / "smoke.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def _items(out):
    with open(os.path.join(out, "smoke.json")) as f:
        return {i["id"]: i for i in json.load(f)["items"]}


def test_smoke_without_params_reports_not_run(tmp_path):
    out = str(tmp_path / "o")
    smoke.main(SID + ["--config", _cfg(tmp_path), "--out", out])
    it = _items(out)
    for k in ("env.compat_path", "env.import_probe", "env.mpi_launch_probe",
              "d13.switchinterval", "d14.loopback", "tree.access_log", "tree.empty_delta",
              "budget.E4-old-arch", "budget.E5-new-arch-frontend", "budget.E6-E9-max"):
        assert it[k]["status"] == "NOT_RUN", k
    assert it["budget.E4-old-arch"]["missing_params"]


def test_smoke_budget_switchinterval_and_tree(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Sse)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        _probe(tmp_path, srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()
    tree = tmp_path / "tree"
    (tree / "raw" / "logs").mkdir()
    (tree / "raw" / "logs" / "frontend.w0.1.err").write_bytes(
        b'INFO:     127.0.0.1:5 - "POST /v1/completions HTTP/1.1" 200 OK\n'
        b'INFO:     127.0.0.1:5 - "GET /health HTTP/1.1" 200 OK\nother line\n')
    n = len(os.sched_getaffinity(0))
    params = ["python=%s" % sys.executable, "cores_loadgen=1", "cores_server=1",
              "cores_postproc=1", "cores_observers=1", "cores_frontend=1", "cores_sfe=1",
              "cores_engine_side=%d" % n, "cores_stubs=1"]
    out = str(tmp_path / "o")
    argv = SID + ["--config", _cfg(tmp_path), "--out", out, "--tree", str(tree)]
    for p in params:
        argv += ["--param", p]
    rc = smoke.main(argv)
    it = _items(out)
    assert it["budget.E4-old-arch"]["status"] == ("PASS" if n >= 4 else "FAIL")
    assert it["budget.E6-E9-max"]["status"] == "FAIL"  # needs n + 4 cores
    assert it["d13.switchinterval"]["status"] == "RECORDED"
    assert it["tree.access_log"]["total"] == 2
    assert it["tree.empty_delta"]["status"] == "PASS"
    assert it["tree.empty_delta"]["seen_requests"] == 2
    assert rc == 4  # at least one FAIL (E6-E9 budget)


def test_smoke_empty_delta_fails_on_request_count_mismatch(tmp_path):
    tree = tmp_path / "tree" / "raw"
    tree.mkdir(parents=True)
    (tree / "httpprobe.json").write_text(json.dumps({"results": [
        {"name": "c", "status": 200, "done_seen": True, "n_frames_excl_done": 1,
         "ssecounter": {"frames": 1}}]}))
    out = str(tmp_path / "o")
    smoke.main(SID + ["--config", _cfg(tmp_path), "--out", out, "--tree", str(tmp_path / "tree")])
    assert _items(out)["tree.empty_delta"]["status"] == "FAIL"  # config expects 2 requests
