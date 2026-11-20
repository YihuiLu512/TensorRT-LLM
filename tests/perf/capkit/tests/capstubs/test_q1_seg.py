# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.q1seg + capstubs/q1_seg/sitecustomize.py (front-end item 10, design card 10'):
positive control with known CPU / wall, per-call overhead, coroutine refusal, sitecustomize
chaining, missing sites stop the process, exact hit counts on a stand-in product tree, the H1
busy injection, and the report reader's window interpolation."""

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.abspath(os.path.join(HERE, "..", ".."))
TOOLS = os.path.join(C, "tools")
MOD = os.path.join(TOOLS, "capstubs", "q1_seg")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

from capstubs import q1seg  # noqa: E402

FAKE = {
    "starlette/__init__.py": "",
    "starlette/requests.py": """
        import json
        class Request:
            def __init__(self, body):
                self._body = body
            async def json(self):
                if not hasattr(self, "_json"):
                    self._json = json.loads(self._body)
                return self._json
        """,
    "fastapi/__init__.py": "",
    "fastapi/_compat/__init__.py": "from .v2 import ModelField\n",
    "fastapi/_compat/v2.py": """
        class ModelField:
            def validate(self, value, values={}, *, loc=()):
                return dict(value), []
        """,
    "tensorrt_llm/__init__.py": "",
    "tensorrt_llm/serve/__init__.py": "",
    "tensorrt_llm/serve/postprocess_handlers.py": """
        def completion_stream_post_processor(rsp, args):
            return ["data: %s\\n\\n" % (rsp,)]
        """,
    "tensorrt_llm/serve/openai_server.py": """
        from .postprocess_handlers import completion_stream_post_processor
        class OpenAIServer:
            async def await_disconnected(self, raw_request, promise):
                return None
            def post(self, out):
                return completion_stream_post_processor(out, None)
        """,
    "tensorrt_llm/serve/responses_utils.py": """
        class ServerArrivalTimeMiddleware:
            def __init__(self, app):
                self.app = app
            async def __call__(self, scope, receive, send):
                await self.app(scope, receive, send)
        """,
    "tensorrt_llm/serve/openengine_poc/__init__.py": "",
    "tensorrt_llm/serve/openengine_poc/facade.py": """
        class OpenEngineLLM:
            def generate_async(self, inputs, sampling_params=None, **kw):
                return ("req", inputs)
        """,
    "tensorrt_llm/grpc/__init__.py": "",
    "tensorrt_llm/grpc/openengine_client/__init__.py": "",
    "tensorrt_llm/grpc/openengine_client/stream.py": """
        class GenerateStream:
            def _validate(self, response):
                return True
        """,
}

DRIVER = """
import asyncio, sys, threading
import starlette.requests as sr
from fastapi._compat.v2 import ModelField
from tensorrt_llm.serve import openai_server as osv, postprocess_handlers as ph
from tensorrt_llm.serve import responses_utils as ru
from tensorrt_llm.serve.openengine_poc.facade import OpenEngineLLM
from tensorrt_llm.grpc.openengine_client.stream import GenerateStream
N = int(sys.argv[1])

async def app(scope, receive, send):
    return None

mw = ru.ServerArrivalTimeMiddleware(app)
srv = osv.OpenAIServer()

async def one():
    await mw({"type": "http", "path": "/v1/completions"}, None, None)
    body = await sr.Request(b'{"prompt": [1, 2]}').json()
    ModelField().validate(body)
    OpenEngineLLM().generate_async([1, 2])
    await asyncio.ensure_future(srv.await_disconnected(None, None))
    for k in range(2):
        GenerateStream()._validate(k)
        srv.post(k)

async def main():
    for _ in range(N):
        await one()
    await mw({"type": "http", "path": "/health"}, None, None)
    await mw({"type": "lifespan"}, None, None)

asyncio.run(main())
th = threading.Thread(target=lambda: GenerateStream()._validate(0))
th.start()
th.join()
assert ph.completion_stream_post_processor is osv.completion_stream_post_processor
assert getattr(osv.completion_stream_post_processor, "_capkit_q1seg", False)
print("driver ok")
"""


def fake_tree(root, override=None):
    files = dict(FAKE, **(override or {}))
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    return root


def run(tmp_path, env_extra, code=DRIVER, args=("5",), pp_extra=(), override=None):
    fake = fake_tree(tmp_path / "pkgs", override)
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH",) and not k.startswith("CAPKIT_Q1SEG")}
    env["PYTHONPATH"] = os.pathsep.join([MOD, TOOLS, str(fake)] + list(pp_extra))
    env.update(env_extra)
    script = tmp_path / "driver.py"
    script.write_text(code)
    return subprocess.run([sys.executable, str(script)] + list(args), env=env,
                          capture_output=True, text=True, timeout=120, cwd=str(tmp_path))


def reports(stderr):
    return q1seg.parse_reports(stderr.splitlines())


ALL = {"CAPKIT_Q1SEG": "all"}


# ---------------------------------------------------------------- in-process wrapper tests
def _timed(fn, c=None, depth=None):
    c = c if c is not None else [0] * len(q1seg.SEG_FIELDS)
    return q1seg.make_timed(fn, c, threading.get_ident(), depth or [0], [None]), c


def test_positive_control_known_cpu_and_wall():
    """Known thread CPU (spin on thread_time) and known wall (sleep, ~0 CPU), +-5 %."""
    cpu_target, sleep_ns, calls = 2_000_000, 3_000_000, 20

    def burn():
        t0 = time.thread_time_ns()
        while time.thread_time_ns() - t0 < cpu_target:
            pass

    def nap():
        time.sleep(sleep_ns / 1e9)

    wb, cb = _timed(burn)
    wn, cn = _timed(nap)
    for _ in range(calls):
        wb()
        wn()
    assert cb[0] == calls and cn[0] == calls
    assert abs(cb[1] / calls - cpu_target) <= 0.05 * cpu_target, cb
    assert cb[2] >= cb[1] * 0.95
    assert abs(cn[2] / calls - sleep_ns) <= 0.05 * sleep_ns + 200_000, cn
    assert cn[1] / calls < 0.05 * sleep_ns, cn


def test_wrapper_overhead_below_1us():
    def noop(x):
        return x

    w, c = _timed(noop)
    n, best_bare, best_wrap = 20000, None, None
    for _ in range(7):
        t0 = time.perf_counter_ns()
        for i in range(n):
            noop(i)
        t1 = time.perf_counter_ns()
        for i in range(n):
            w(i)
        t2 = time.perf_counter_ns()
        b, x = (t1 - t0) / n, (t2 - t1) / n
        best_bare = b if best_bare is None else min(best_bare, b)
        best_wrap = x if best_wrap is None else min(best_wrap, x)
    over = best_wrap - best_bare
    print("q1seg wrapper overhead %.0f ns/call (bare %.0f, wrapped %.0f)"
          % (over, best_bare, best_wrap))
    assert c[0] == 7 * n
    assert over <= 1000, over


def test_refuses_coroutine_and_async_generator():
    async def co():
        return 1

    async def agen():
        yield 1

    for fn in (co, agen):
        with pytest.raises(q1seg.SiteError):
            _timed(fn)
    with pytest.raises(q1seg.SiteError):
        q1seg.make_count_coro(lambda: None, [0] * 5, 0)


def test_nested_calls_are_counted_not_timed_twice():
    depth = [0]

    def inner():
        t0 = time.thread_time_ns()
        while time.thread_time_ns() - t0 < 1_000_000:
            pass

    wi, ci = _timed(inner, depth=depth)

    def outer():
        wi()

    wo, co = _timed(outer, depth=depth)
    wo()
    assert co[0] == 1 and co[1] >= 1_000_000
    assert ci[0] == 1 and ci[4] == 1 and ci[1] == 0


def test_other_threads_counted_apart():
    w, c = _timed(lambda: None)
    th = threading.Thread(target=w)
    th.start()
    th.join()
    w()
    assert c[0] == 1 and c[3] == 1


def test_spec_parsing():
    assert q1seg.parse_segments("") == ()
    assert q1seg.parse_segments("all") == tuple(q1seg.SEGMENTS)
    assert q1seg.parse_segments("S2,S5") == ("S2", "S5")
    for bad in ("S9", "S2,S2"):
        with pytest.raises(q1seg.SpecError):
            q1seg.parse_segments(bad)
    assert q1seg.parse_inject("H1:count") == {"site": "H1", "mode": "count", "delta_ns": 0}
    assert q1seg.parse_inject("H1:busy:100000")["delta_ns"] == 100000
    for bad in ("H2:count", "H1:busy", "H1:busy:0", "H1:sleep:5"):
        with pytest.raises(q1seg.SpecError):
            q1seg.parse_inject(bad)


def test_window_delta_interpolates_and_refuses_outside():
    def rep(t, n):
        return {"v": 1, "pid": 7, "t_ns": t, "seg": {"S2": [n, 10 * n, 20 * n, 0, 0]},
                "inject": {"H1": [n, n, 100 * n, 0]}}
    rs = [rep(0, 0), rep(1000, 100), rep(2000, 300)]
    d = q1seg.window_delta(rs, 500, 1500)
    assert d["seg"]["S2"]["n"] == pytest.approx(150)  # 50 + 100
    assert d["seg"]["S2"]["cpu_ns"] == pytest.approx(1500)
    assert d["inject"]["H1"]["applied_ns"] == pytest.approx(15000)
    assert d["gap_ns"] == 1000 and d["n_reports"] == 1
    assert q1seg.window_delta(rs, -1, 1500) is None
    assert q1seg.window_delta(rs, 500, 2001) is None
    lines = ["x", q1seg.REPORT + json.dumps(rs[0]), "capkit-q1seg-report " + json.dumps(
        dict(rs[1], pid=8))]
    assert len(q1seg.parse_reports(lines)) == 2 and len(q1seg.parse_reports(lines, pid=7)) == 1
    with pytest.raises(ValueError):
        q1seg.parse_reports([q1seg.REPORT + json.dumps({"v": 2})])


# ---------------------------------------------------------------- sitecustomize, stand-in tree
def test_end_to_end_hit_counts_and_h1_count(tmp_path):
    n = 7
    r = run(tmp_path, dict(ALL, CAPKIT_Q1SEG_INJECT="H1:count"), args=(str(n),))
    assert r.returncode == 0, r.stderr
    assert "driver ok" in r.stdout
    assert r.stderr.count("capkit-q1seg: loaded pid=") == 1
    for s in list(q1seg.SEGMENTS) + ["H1"]:
        assert r.stderr.count("capkit-q1seg: site %s installed" % s) == 1, s
    rs = reports(r.stderr)
    assert len(rs) >= 2  # the first timed call and the exit report
    last = rs[-1]
    assert sorted(last["installed"]) == sorted(list(q1seg.SEGMENTS) + ["H1"])
    seg = {k: dict(zip(q1seg.SEG_FIELDS, v)) for k, v in last["seg"].items()}
    want = {"S1j": n, "S1v": n, "S2": n, "S3": 2 * n, "S4": n, "S5": 2 * n}
    assert {k: v["n"] for k, v in seg.items()} == want
    assert seg["S5"]["n_other"] == 1
    assert all(seg[k]["n_nested"] == 0 for k in seg)
    assert all(seg[k]["cpu_ns"] > 0 and seg[k]["wall_ns"] > 0 for k in q1seg.TIMED)
    assert seg["S4"]["cpu_ns"] == 0  # count only
    h1 = dict(zip(q1seg.INJ_FIELDS, last["inject"]["H1"]))
    assert h1 == {"hits": n, "applied": 0, "applied_ns": 0, "hits_other": 0}
    ts = [x["t_ns"] for x in rs]
    assert ts == sorted(ts) and [x["seq"] for x in rs] == list(range(len(rs)))


def test_h1_busy_positive_control(tmp_path):
    n, delta = 6, 2_000_000
    r = run(tmp_path, dict(ALL, CAPKIT_Q1SEG_INJECT="H1:busy:%d" % delta), args=(str(n),))
    assert r.returncode == 0, r.stderr
    h1 = dict(zip(q1seg.INJ_FIELDS, reports(r.stderr)[-1]["inject"]["H1"]))
    assert h1["hits"] == n and h1["applied"] == n
    assert n * delta <= h1["applied_ns"] <= n * delta * 1.2, h1
    seg = reports(r.stderr)[-1]["seg"]
    assert sum(seg[s][1] for s in q1seg.SUM_SEGMENTS) < n * delta / 4  # spin is not in a segment


def test_segment_subset_and_report_period(tmp_path):
    code = (DRIVER.replace("assert ph.completion_stream_post_processor is", "assert True or")
            .replace('assert getattr(osv.completion_stream_post_processor, "_capkit_q1seg", '
                     'False)', "")
            .replace("        await one()\n", "        await one()\n"
                     "        await asyncio.sleep(0.003)\n"))
    assert code.count("asyncio.sleep(0.003)") == 1 and "_capkit_q1seg" not in code
    r = run(tmp_path, {"CAPKIT_Q1SEG": "S2,S5", "CAPKIT_Q1SEG_REPORT_MS": "1"}, code=code,
            args=("40",))
    assert r.returncode == 0, r.stderr
    rs = reports(r.stderr)
    assert set(rs[-1]["seg"]) == {"S2", "S5"} and rs[-1]["inject"] == {}
    assert rs[-1]["seg"]["S2"][0] == 40 and rs[-1]["seg"]["S5"][0] == 80
    assert len(rs) >= 20  # every timed call after >= 1 ms reports, plus the exit report


def test_off_without_switches_only_chains(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("import builtins\nbuiltins._q1_chained = True\n")
    code = ("import builtins, sys, sitecustomize as s\n"
            "print(getattr(builtins, '_q1_chained', False), s.STATE, 'capstubs' in sys.modules)\n")
    r = run(tmp_path, {}, code=code, args=(), pp_extra=(str(other),))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["True", "None", "False"]
    assert "capkit-q1seg: loaded pid=" in r.stderr and q1seg.REPORT not in r.stderr


def test_chains_next_sitecustomize_when_on(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("import builtins\nbuiltins._q1_chained = True\n")
    code = "import builtins\nprint(getattr(builtins, '_q1_chained', False))\n" + DRIVER
    r = run(tmp_path, ALL, code=code, pp_extra=(str(other),))
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[0] == "True"


@pytest.mark.parametrize("env", [{"CAPKIT_Q1SEG": "S9"}, {"CAPKIT_Q1SEG": "S2,S2"},
                                 {"CAPKIT_Q1SEG_INJECT": "H1:busy:0"},
                                 {"CAPKIT_Q1SEG": "S2", "CAPKIT_Q1SEG_REPORT_MS": "x"}])
def test_bad_switches_exit_2(tmp_path, env):
    r = run(tmp_path, env)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "capkit-q1seg: refusing to run" in r.stderr


@pytest.mark.parametrize("override,what", [
    ({"tensorrt_llm/serve/openengine_poc/facade.py": "class OpenEngineLLM:\n    pass\n"},
     "site S2"),
    ({"tensorrt_llm/grpc/openengine_client/stream.py":
      "class GenerateStream:\n    async def _validate(self, r):\n        return True\n"},
     "site S5"),
    ({"tensorrt_llm/serve/openai_server.py":
      "class OpenAIServer:\n    def await_disconnected(self, a, b):\n        return None\n"},
     "site S3"),
    ({"starlette/requests.py": "class Request:\n    pass\n"}, "site S1j"),
    ({"fastapi/_compat/v2.py": "class ModelField:\n    validate = staticmethod(lambda v: v)\n"},
     "site S1v"),
])
def test_missing_or_wrong_site_exits_3(tmp_path, override, what):
    r = run(tmp_path, ALL, override=override)
    assert r.returncode == 3, (r.returncode, r.stderr[-2000:])
    assert "capkit-q1seg: refusing to run: %s" % what in r.stderr
    assert "driver ok" not in r.stdout


def test_wrong_await_disconnected_kind_exits_3(tmp_path):
    r = run(tmp_path, {"CAPKIT_Q1SEG": "S4"}, override={
        "tensorrt_llm/serve/openai_server.py":
            "class OpenAIServer:\n    def await_disconnected(self, a, b):\n        return None\n"})
    assert r.returncode == 3 and "site S4" in r.stderr


def test_site_never_loaded_exits_3_at_first_timed_call(tmp_path):
    code = ("import asyncio, starlette.requests as sr\n"
            "asyncio.run(sr.Request(b'{}').json())\nprint('driver ok')\n")
    r = run(tmp_path, ALL, code=code, args=())
    assert r.returncode == 3, r.stderr
    assert "were never installed" in r.stderr and "driver ok" not in r.stdout


def test_product_sites_exist_with_their_kind_in_the_probe_tree():
    """The tensorrt_llm sites in <C>/probe-tree (same classes and functions as the product tree
    the E4 frontend runs): timed ones are plain functions, S4 and H1 coroutine functions, and
    openai_server passes the module global completion_stream_post_processor."""
    import ast  # noqa: PLC0415
    root = os.path.join(C, "probe-tree")
    assert os.path.isdir(os.path.join(root, "tensorrt_llm")), "needs the probe-tree worktree"
    for site, (mod, target, kind) in list(q1seg.SEGMENTS.items()) + list(
            q1seg.INJECT_SITES.items()):
        if not mod.startswith("tensorrt_llm."):
            continue
        path = os.path.join(root, *mod.split(".")) + ".py"
        tree = ast.parse(open(path).read())
        if site == "S3":
            path = os.path.join(root, *q1seg.S3_ALSO.split(".")) + ".py"
            defs = [n for n in ast.parse(open(path).read()).body
                    if isinstance(n, ast.FunctionDef) and n.name == target]
            assert len(defs) == 1
            assert "post_processor=completion_stream_post_processor" in open(
                os.path.join(root, *mod.split(".")) + ".py").read()
            continue
        cls, fn = target.split(".")
        [c] = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls]
        [f] = [n for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == fn]
        want = ast.FunctionDef if kind == "time" else ast.AsyncFunctionDef
        assert isinstance(f, want) and not f.decorator_list, (site, type(f).__name__)
