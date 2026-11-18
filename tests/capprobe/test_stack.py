# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capprobe group `stack`: stack classification rules, slot table, and the sampler thread in a
fresh subprocess (main-thread categories, other threads, child-process CPU, own cost)."""

from probe_testutil import DECODE_SELF, run_json

from capcommon.snapshot import SlotRegistry
from capprobe import config, sites, stack


class _Code:
    def __init__(self, filename, name):
        self.co_filename, self.co_name = filename, name


class _Frame:
    def __init__(self, code, back):
        self.f_code, self.f_back = code, back


def _stack(*outer_to_inner):
    """Frames from (file, function) pairs listed outermost first; returns the innermost."""
    fr = None
    for f, n in outer_to_inner:
        fr = _Frame(_Code(f, n), fr)
    return fr


SP = "/usr/local/lib/python3.12/dist-packages/"
LIB = "/usr/lib/python3.12/"
RUN = [(LIB + "asyncio/runners.py", "run"), (SP + "uvloop/__init__.py", "run")]
GEN = (SP + "tensorrt_llm/serve/openai_server.py", "completion_generator")


def _cat(*frames):
    return stack.Classifier().classify(_stack(*frames))[0]


def test_classification_rules():
    assert _cat(*RUN) == "idle"
    assert _cat(*RUN, GEN, (SP + "tensorrt_llm/grpc/openengine_client/stream.py", "_read"),
                (SP + "grpc/aio/_call.py", "read")) == "grpc"
    assert _cat(*RUN, GEN, (SP + "pydantic/main.py", "model_dump_json")) == "encode"
    assert _cat(*RUN, (SP + "fastapi/routing.py", "app"),
                (SP + "pydantic/main.py", "model_validate")) == "http_in"
    assert _cat(*RUN, GEN, (SP + "tensorrt_llm/serve/openengine_poc/facade.py", "__anext__"),
                (LIB + "asyncio/queues.py", "get")) == "facade"
    assert _cat(*RUN, (LIB + "asyncio/base_events.py", "_run_once")) == "loop"
    assert _cat(*RUN, (SP + "starlette/responses.py", "stream_response"),
                (SP + "uvicorn/protocols/http/httptools_impl.py", "send")) == "http_out"
    assert _cat(*RUN, (SP + "uvicorn/protocols/http/httptools_impl.py", "on_url")) == "http_in"
    assert _cat(*RUN, GEN, (SP + "capcommon/inject.py", "hit")) == "probe"
    assert _cat(*RUN, GEN, (SP + "tensorrt_llm/serve/openengine_poc/response.py", "apply"),
                (SP + "tensorrt_llm/serve/detokenize.py", "step"),
                (SP + "transformers/tokenization_utils_fast.py", "decode")) == "tok"
    assert _cat(*RUN, GEN, (SP + "tensorrt_llm/serve/openengine_poc/postproc_pool.py", "put"),
                (LIB + "pickle.py", "dumps")) == "pool_ipc"
    assert _cat(*RUN, (SP + "tensorrt_llm/serve/openai_server.py", "openai_completion")) \
        == "openai_req"
    assert _cat(*RUN, GEN) == "openai_stream"
    assert _cat(("/x/selectors.py", "select")) == "idle"
    assert _cat(*RUN, ("/x/elsewhere.py", "f")) == "other"


def test_slot_table_and_group_switch():
    assert "stack" in sites.IMPLEMENTED_GROUPS
    reg = SlotRegistry()
    reg.register_decls(sites.SLOTS, groups={"stack"})
    names = set(reg.names())
    for c in sites.STACK_CATEGORIES:
        assert {"stack.main.n.%s" % c, "stack.main.wall_ns.%s" % c, "stack.main.cpu_ns.%s" % c,
                "stack.thr.wall_ns.%s" % c} <= names
    assert {"stack.%s" % t for t in sites.STACK_TOTALS} | {"stack.children"} <= names
    assert not any(n.startswith(("count.", "lag.")) for n in names)
    assert config._parse_groups("stack") == ("stack",)
    assert sites.sites_for_groups(["stack"]) == {}


SNIPPET = DECODE_SELF + r"""
import subprocess, sys, threading, time
ns = {}
exec(compile("def burn(s):\n    t = time.monotonic() + s\n    while time.monotonic() < t:\n"
             "        pass\n", "/x/tensorrt_llm/grpc/fake.py", "exec"), {"time": time}, ns)
tk = {}
exec(compile("def burn(s):\n    t = time.monotonic() + s\n    while time.monotonic() < t:\n"
             "        pass\n", "/x/tokenizers/fake.py", "exec"), {"time": time}, tk)
th = threading.Thread(target=tk["burn"], args=(0.6,), name="tok-worker")
child = subprocess.Popen([sys.executable, "-c",
                          "import time\nt=time.monotonic()+0.8\nwhile time.monotonic()<t: pass"])
th.start()
t0 = time.monotonic_ns()
ns["burn"](0.8)
time.sleep(0.3)
th.join(); child.wait()
time.sleep(0.7)  # one more children poll after the child exited
el = time.monotonic_ns() - t0
live = sorted(snap()["meta"]["threads"].values())
capprobe._state.sampler.stop()
s = snap()
s["elapsed_ns"], s["live_threads"] = el, live
print(json.dumps(s))
"""


def test_sampler_in_process(probe_env):
    out = run_json(SNIPPET, probe_env("stack"))
    v, meta = out["values"], out["meta"]
    assert v["stack.errors"] == 0
    assert v["stack.samples"] >= 100
    # the weights cover the run (sampler starts at import, before t0)
    assert v["stack.wall_ns"] >= 0.9 * out["elapsed_ns"]
    # 0.8 s burning in a grpc-named file on the main thread
    assert 0.6e9 <= v["stack.main.wall_ns.grpc"] <= 1.0e9
    assert v["stack.main.cpu_ns.grpc"] >= 0.3e9  # shares the GIL with the thread
    assert v["stack.main.n.grpc"] > 0
    # 0.6 s burning in a tokenizers-named file on another thread
    assert 0.4e9 <= v["stack.thr.wall_ns.tok"] <= 0.8e9
    # the child burned about 0.8 s of CPU
    assert v["stack.children_cpu_ns"] >= 0.5e9
    assert 0 < v["stack.sampler_cpu_ns"] < 0.25 * v["stack.wall_ns"]
    assert v["stack.main_cpu_ns"] >= 0.3e9 and v["stack.proc_cpu_ns"] >= v["stack.main_cpu_ns"]
    assert meta["probe"]["groups"] == ["stack"]
    assert meta["probe"]["stack_period_ns"] == stack.PERIOD_NS
    funcs = meta["stack_funcs"]
    assert funcs["grpc|fake.py:burn"] == v["stack.main.wall_ns.grpc"]
    assert "other|<string>:<module>" in funcs and not any(k.startswith("idle|") for k in funcs)
    assert out["live_threads"] == ["MainThread", "capkit-snapshot", "capkit-stack"]
