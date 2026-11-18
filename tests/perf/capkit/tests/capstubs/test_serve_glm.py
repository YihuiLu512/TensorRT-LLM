# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.serve_glm (E10) on a fake glm5_serve package written to tmp (no glm environment on
the VM): option refusals, the probe layer (M1 count / busy, dispatch batches, loop lag, server
state, engine kind check), the readiness file and the JSONL rows; plus a static check that the
real glm checkout still has every target the probe layer wraps."""

import ast
import json
import os
import subprocess
import sys
import textwrap

import pytest

from capstubs import serve_glm

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(HERE, "..", "..", "tools"))
GLM_SRC = "/home/scratch.yihlu_coreai_1/repro-env/j17740-oe-eda0ca6/src/python/glm5_serve"

FAKE = {
    "__init__.py": "__version__ = 'fake'\n",
    "engine_api.py": """
        class MockEngine:
            def __init__(self, config):
                self.config = dict(config); self._iter = 0; self._reqs = {}
                self.step_s = config.get('mock_step_ms', 2.0) / 1000.0
            def start(self): pass
        """,
    "engine_client.py": """
        class EngineClient:
            def __init__(self, engine): self.engine = engine; self.got = []
            def start_dispatcher(self, loop): self.loop = loop
            def _dispatch(self, responses): self.got += responses
        """,
    "openengine/__init__.py": "",
    "openengine/servicer.py": """
        class ServerState:
            STARTING, READY = 'starting', 'ready'
            def __init__(self):
                self.phase = self.STARTING; self.requests = {}; self.generate_accepted = 0
        def _token_output(rid, tokens, logprobs, want):
            return ('tok', rid, tuple(tokens))
        def stream(n):  # the servicer looks the name up at call time
            return [_token_output('r', [i], None, False) for i in range(n)]
        """,
    "cli.py": """
        import asyncio, json, os, sys, time
        def main():
            from .config import load_mock_config
            from .engine_api import MockEngine
            from .engine_client import EngineClient
            from .openengine import servicer
            args = sys.argv[1:]
            eng = MockEngine(dict(load_mock_config(), model_dir=args[0]))
            eng.start()
            st = servicer.ServerState()
            async def serve():
                cl = EngineClient(eng)
                cl.start_dispatcher(asyncio.get_running_loop())
                for _ in range(3):
                    cl._dispatch([{'request_id': 1}] * 4)
                servicer.stream(5)
                st.phase = st.READY
                st.generate_accepted = 7
                await asyncio.sleep(float(os.environ.get('FAKE_SERVE_S', '1.5')))
            asyncio.run(serve())
        """,
    "config.py": """
        import json, os
        def load_mock_config():
            return json.loads(os.environ.get('GLM5_MOCK_CONFIG') or '{}')
        """,
}
SERVE = ["tok", "--host", "127.0.0.1", "--grpc", "--grpc-protocol", "openengine"]


@pytest.fixture
def fake_glm(tmp_path):
    root = tmp_path / "fakeglm"
    for rel, src in FAKE.items():
        p = root / "glm5_serve" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(src))
    return str(root)


def run(fake, own, serve=SERVE, env=None, timeout=60):
    e = dict(os.environ, PYTHONPATH=os.pathsep.join([TOOLS, fake]), PYTHONDONTWRITEBYTECODE="1")
    e.update(env or {})
    return subprocess.run([sys.executable, "-m", "capstubs.serve_glm"] + own + ["--"] + serve,
                          env=e, capture_output=True, text=True, timeout=timeout, cwd="/tmp")


def rows(out_dir):
    (f,) = [x for x in os.listdir(out_dir) if x.startswith("serve_glm.") and
            x.endswith(".jsonl")]
    with open(os.path.join(out_dir, f)) as fh:
        lines = [json.loads(x) for x in fh]
    return lines[0], lines[1:]


@pytest.mark.parametrize("own,serve,why", [
    (["--engine", "g1", "--step-ms", "1"], ["tok", "--grpc"], "--grpc-protocol openengine"),
    (["--engine", "g1", "--step-ms", "1"], SERVE + ["--mock_engine"], "--mock_engine"),
    (["--engine", "g1", "--step-ms", "0"], SERVE, "G1 needs"),
    (["--engine", "mock", "--step-ms", "1", "--inject", "M2:count"], SERVE, "unknown site"),
    (["--engine", "mock", "--step-ms", "1", "--diag", "bogus"], SERVE, "unknown diagnostics"),
    (["--engine", "mock", "--step-ms", "1", "--report-addr", "x"], SERVE, "fixed --port"),
    (["--engine", "mock", "--step-ms", "1", "--pick-port"], SERVE + ["--port", "5"], "--pick-port"),
])
def test_refusals(fake_glm, tmp_path, own, serve, why):
    r = run(fake_glm, own + ["--out-dir", str(tmp_path / "o")], serve)
    assert r.returncode == 3 and why in r.stderr, r.stderr


@pytest.mark.parametrize("engine", ["mock", "g1"])
def test_probe_rows_and_readiness(fake_glm, tmp_path, engine):
    out, rec, addr = tmp_path / "o", tmp_path / "rec.json", tmp_path / "addr"
    own = ["--engine", engine, "--step-ms", "2", "--tokens-per-step", "2", "--out-dir", str(out),
           "--record", str(rec), "--inject", "M1:busy:200000", "--report-addr", str(addr),
           "--pick-port", "--report-interval-ms", "200", "--loop-period-ms", "5"]
    r = run(fake_glm, own)
    assert r.returncode == 0, r.stderr
    hdr, body = rows(str(out))
    assert hdr["engine"] == engine and hdr["inject"] == ["M1:busy:200000"]
    last = body[-1]
    assert last["M1"]["hits"] == 5 and last["M1"]["applied"] == 5
    assert last["M1"]["applied_ns"] >= 5 * 200000
    assert last["dispatch"]["calls"] == 3 and last["dispatch"]["responses"] == 12
    assert last["loop"]["n"] > 10 and sum(last["loop"]["hist"]) == last["loop"]["n"]
    assert last["server"] == {"phase": "ready", "requests": 0, "generate_accepted": 7}
    assert last["engine"]["kind"] == ("g1" if engine == "g1" else "mock")
    assert any(name == "MainThread" for name, _ in last["threads"])
    d = json.load(open(rec))
    assert d["mock_config"] == {"mock_step_ms": 2.0, "mock_tokens_per_step": 2,
                                "mock_pattern": [1, 2, 3]}
    assert d["serve_args"][-2] == "--port" and d["glm5_serve_version"] == "fake"
    # readiness: written only after READY + TCP connect; the fake never listens -> no file
    assert not addr.exists()


def test_engine_mismatch_logic():
    p = serve_glm.Probe(["engine"], 10.0, "g1")
    assert "0 engines" in p.engine_mismatch()

    class Mock:  # a MockEngine where a G1 was asked for
        pass
    p.engines = [Mock()]
    assert "asked for g1" in p.engine_mismatch()
    p2 = serve_glm.Probe(["engine"], 10.0, "mock")
    p2.engines = [type("MockEngine", (), {})()]
    assert p2.engine_mismatch() is None


def test_tier_beat_check(tmp_path):
    import conftest as cf  # noqa: PLC0415
    from capcommon.exitcodes import ConfigError  # noqa: PLC0415
    b = 5 * cf.MS
    doc = cf.table_doc([("warmup", 100, 10.0, {"beat": b}), ("drain", 100, 0, {"beat": b}),
                        ("up", 100, 10.0, {"beat": b}), ("drain", 100, 0, {"beat": b})])
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    assert serve_glm.check_tier_beats(str(p), 5.0) == [b]
    with pytest.raises(ValueError):
        serve_glm.check_tier_beats(str(p), 4.0)
    with pytest.raises(ConfigError):
        serve_glm.check_tier_beats(str(tmp_path / "missing.json"), 5.0)


@pytest.mark.skipif(not os.path.isdir(GLM_SRC), reason="glm checkout not mounted")
def test_real_glm_targets_exist():
    """Static (AST) check of the eda0ca6 checkout: every name the probe layer wraps exists, and
    cli._create_engine imports MockEngine inside the function (so the G1 swap takes)."""
    def tree(rel):
        with open(os.path.join(GLM_SRC, rel)) as f:
            return ast.parse(f.read())

    def names(t):
        out = {}
        for n in t.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                out[n.name] = n
        return out
    sv = names(tree("openengine/servicer.py"))
    assert "_token_output" in sv and "ServerState" in sv
    ec = names(tree("engine_client.py"))["EngineClient"]
    meths = {n.name for n in ec.body if isinstance(n, ast.FunctionDef)}
    assert {"_dispatch", "start_dispatcher"} <= meths
    assert "MockEngine" in names(tree("engine_api.py"))
    ce = names(tree("cli.py"))["_create_engine"]
    inner = [n for n in ast.walk(ce) if isinstance(n, ast.ImportFrom) and n.module == "engine_api"]
    assert inner and any(a.name == "MockEngine" for a in inner[0].names)


def test_pick_free_port_stays_below_the_ephemeral_range(tmp_path):
    from capstubs import serve_nx
    assert serve_glm.pick_free_port is serve_nx.pick_free_port
    rf = tmp_path / "range"
    rf.write_text("32768\t60999\n")
    ports = {serve_glm.pick_free_port("127.0.0.1", range_file=str(rf)) for _ in range(20)}
    assert all(serve_nx.PORT_FLOOR <= p < 32768 for p in ports)


def test_addr_reporter_ignores_a_self_connect(tmp_path, monkeypatch):
    """The reporter writes the address only after a connect that is not a self-connect."""
    import socket
    import time

    class St:
        READY = "ready"
        phase = "ready"

    class Probe:
        state = St()

        def engine_mismatch(self):
            return None
    seen = []

    def fake_self_connected(sock):
        seen.append(1)
        return len(seen) <= 2  # first two connects look like self-connects
    monkeypatch.setattr(serve_glm, "self_connected", fake_self_connected)
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]
    out = tmp_path / "addr"
    try:
        t = serve_glm.start_addr_reporter(Probe(), str(out), "127.0.0.1", port, 10)
        t.join(10)
        assert out.read_text() == "127.0.0.1:%d\n" % port
        assert len(seen) == 3
    finally:
        srv.close()
