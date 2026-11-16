# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: what the worker front end does.

Chat rendering with the DeepSeek-V4.1 tokenizer, and the reasoning and tool-call parsers, run in
the worker front end; the ds engine only sees token ids. The forms, fixtures and environment are
those of ``_test_openengine_ds_disagg_basic.py``.
"""

from __future__ import annotations

import json

import pytest

from . import _test_openengine_ds_disagg_basic as basic

MODEL_NAME = basic.MODEL_NAME
# The module's fixtures.
model_path = basic.model_path
form = basic.form
groups = basic.groups
served = basic.served

pytestmark = [pytest.mark.threadleak(enabled=False)]

MESSAGES = [{"role": "user", "content": "What is 17 times 23? Answer with the number."}]
THINK_PROMPT = "Think step by step about how many primes are below 50, then answer."
# Room for the default reasoning effort's chain of thought before the answer.
THINK_MAX_TOKENS = 8192
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _no_think_tags(text):
    return "<think>" not in (text or "") and "</think>" not in (text or "")


def test_chat_template_and_kwargs(served, model_path):
    """NT-124 (F-07): the prompt is the DeepSeek-V4.1 renderer's, and its thinking switch reaches it."""
    basic.only_chain(served, "the ds proxy renders with the ds-engine's own renderer")
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    tokenizer = DeepseekV41Tokenizer.from_pretrained(model_path)
    c = served.client()
    prompt_tokens = {}
    for thinking in (False, True):
        out = c.chat.completions.create(
            model=MODEL_NAME,
            messages=MESSAGES,
            max_tokens=1,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"thinking": thinking}},
        )
        expected = tokenizer.apply_chat_template(MESSAGES, tokenize=True, thinking=thinking)
        assert out.usage.prompt_tokens == len(expected), (thinking, out.usage)
        prompt_tokens[thinking] = out.usage.prompt_tokens
    # Thinking mode adds the reasoning-effort preamble and the opening <think>.
    assert prompt_tokens[True] > prompt_tokens[False], prompt_tokens


def _chat(client, stream: bool, **request):
    """One chat request: reasoning, content, tool calls as ``(name, arguments)``, finish reason."""
    if not stream:
        choice = client.chat.completions.create(**request).choices[0]
        message = choice.message
        calls = [(c.function.name, c.function.arguments) for c in message.tool_calls or []]
        reasoning = getattr(message, "reasoning_content", None) or ""
        return reasoning, message.content or "", calls, choice.finish_reason
    reasoning, content, finish_reason = "", "", None
    calls: dict[int, list[str]] = {}
    for chunk in client.chat.completions.create(stream=True, **request):
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        reasoning += getattr(delta, "reasoning_content", None) or ""
        content += delta.content or ""
        for call in delta.tool_calls or []:
            name_and_arguments = calls.setdefault(call.index, ["", ""])
            if call.function is not None:
                name_and_arguments[0] += call.function.name or ""
                name_and_arguments[1] += call.function.arguments or ""
        finish_reason = choice.finish_reason or finish_reason
    return reasoning, content, [tuple(calls[i]) for i in sorted(calls)], finish_reason


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_reasoning_and_tools(served, stream):
    """NT-125 (F-27, F-28, DG-65): the reasoning parser splits the reasoning out; the tool parser
    returns the call. The generation leg does not render: with thinking off, its parser still
    knows thinking is off (``resolved_thinking`` travels with the handoff).
    """
    basic.only_chain(served, "the parsers under test are the worker front end's")
    c = served.client()
    reasoning, content, _, finish_reason = _chat(
        c,
        stream,
        model=MODEL_NAME,
        messages=[{"role": "user", "content": THINK_PROMPT}],
        max_tokens=THINK_MAX_TOKENS,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"thinking": True}},
    )
    assert finish_reason == "stop", (finish_reason, reasoning[-2000:], content)
    assert reasoning, "the reasoning parser split nothing out"
    assert "15" in content, content
    assert _no_think_tags(reasoning) and _no_think_tags(content)

    # Without thinking the answer carries no reasoning.
    reasoning, content, _, _ = _chat(
        c,
        stream,
        model=MODEL_NAME,
        messages=MESSAGES,
        max_tokens=64,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )
    assert not reasoning and "391" in content, (reasoning, content)

    _, content, calls, finish_reason = _chat(
        c,
        stream,
        model=MODEL_NAME,
        max_tokens=1024,
        temperature=0.0,
        tools=[WEATHER_TOOL],
        messages=[
            {"role": "user", "content": "What is the weather in Paris right now? Use the tool."}
        ],
    )
    assert finish_reason == "tool_calls" and calls, (finish_reason, content, calls)
    name, arguments = calls[0]
    assert name == "get_weather" and "paris" in json.loads(arguments)["city"].lower()
    assert _no_think_tags(content)


# The chain harness, imported where a test reads an engine or starts a chain of its own.
__extra_import_path__ = ["~/tests/integration/defs"]

# Long enough to fill several KV-cache pages of the ds engine (128 tokens each).
CACHE_TEXT = " ".join(
    f"Fact {i}: river number {i} flows into sea number {i % 7}." for i in range(80)
)
LEG_TIMEOUT_S = 600


def _engine_args(**values) -> tuple[str, ...]:
    """:data:`basic.ENGINE_ARGS` with the options named in ``values`` set to them."""
    args = list(basic.ENGINE_ARGS)
    for name, value in values.items():
        args[args.index(f"--{name}") + 1] = str(value)
    return tuple(args)


def _prompt_ids(model_path: str, text: str) -> list[int]:
    """``text`` as DeepSeek-V4.1 token ids, without special tokens."""
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    return list(
        DeepseekV41Tokenizer.from_pretrained(model_path).encode(text, add_special_tokens=False)
    )


def _disagg_request_id() -> int:
    """A ``disagg_request_id`` for a leg the test sends itself, apart from the orchestrator's."""
    import random

    from tensorrt_llm.grpc.openengine_conventions import DISAGG_REQUEST_ID_MIN

    return DISAGG_REQUEST_ID_MIN + random.randrange(1 << 40)


def _context_leg(served: basic.Served, prompt_ids: list[int], **body) -> dict:
    """A context_only leg sent to the context worker itself; its answer, which holds the handoff."""
    import requests

    request = {
        "model": MODEL_NAME,
        "prompt": prompt_ids,
        "ignore_eos": True,
        **body,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": _disagg_request_id(),
        },
    }
    url = f"{served.workers['context']}/v1/completions"
    response = requests.post(url, json=request, timeout=LEG_TIMEOUT_S)
    assert response.status_code == 200, response.text
    return response.json()


def _generation_leg(served: basic.Served, prompt_ids: list[int], context: dict, **body):
    """The generation_only leg of the context leg answer ``context``, sent to the generation worker."""
    import requests

    handoff = {**context["choices"][0]["disaggregated_params"], "request_type": "generation_only"}
    request = {"model": MODEL_NAME, "prompt": prompt_ids, **body, "disaggregated_params": handoff}
    url = f"{served.workers['generation']}/v1/completions"
    return requests.post(url, json=request, timeout=LEG_TIMEOUT_S)


def _cached_tokens(usage) -> int:
    """``usage``'s cached prompt tokens; a usage without them fails the test."""
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
    assert cached is not None, f"no prompt_tokens_details.cached_tokens: {usage}"
    return int(cached)


@pytest.mark.parametrize("api", ["completions", "chat"])
def test_usage_and_cached_tokens(served, api, record_property):
    """NT-126 (F-30, F-34, DG-86, DS-S14, DS-S15, DS-I10): usage, and the context leg's cached tokens.

    The second turn of a conversation finds the first turn's prompt in reused blocks (only the
    context engine reuses blocks): its ``cached_tokens`` is above 0 and below the prompt.
    Streamed with ``include_usage`` it has the prompt and completion token counts of the
    unstreamed answer, and ``completion_tokens`` counts the context leg's first token once.
    The second turn's prompt and cached tokens are recorded, to compare the two forms' runs.
    """
    max_tokens = 8
    question = "Which sea does river 3 flow into?"
    if api == "completions":
        first = {"prompt": f"{CACHE_TEXT}\nQ: {question}\nA:"}
    else:
        first = {"messages": [{"role": "user", "content": f"{CACHE_TEXT}\n{question}"}]}
    answer, _, _ = basic.generate(served, api, False, max_tokens=max_tokens, **first)
    if api == "completions":
        second = {"prompt": f"{first['prompt']}{answer}\nQ: And river 4?\nA:"}
    else:
        second = {
            "messages": [
                *first["messages"],
                {"role": "assistant", "content": answer},
                {"role": "user", "content": "And river 4?"},
            ]
        }
    usages = {}
    for stream in (False, True):
        _, finish, usage = basic.generate(
            served, api, stream, max_tokens=max_tokens, ignore_eos=True, **second
        )
        assert finish == "length" and usage.completion_tokens == max_tokens, (stream, finish, usage)
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens, (stream, usage)
        assert 0 < _cached_tokens(usage) < usage.prompt_tokens, (stream, usage)
        usages[stream] = usage
    record_property(
        "second_turn_usage",
        json.dumps(
            {
                "stream" if stream else "non_stream": {
                    "prompt_tokens": usage.prompt_tokens,
                    "cached_tokens": _cached_tokens(usage),
                }
                for stream, usage in usages.items()
            }
        ),
    )
    assert usages[True].prompt_tokens == usages[False].prompt_tokens, usages


# Decoded tokens of the DSpark test: several draft-and-verify steps.
DRAFT_DECODE = 64
STATS_WAIT_S = 30


def _accepted_drafts(url: str) -> tuple[int, int]:
    """Drafts proposed and accepted in the iteration stats a ds worker's ``/metrics`` hands out.

    The worker hands out each iteration's stats once, so a call counts the iterations since the
    previous one. An iteration without the counts fails the test.
    """
    import requests

    stats = requests.get(f"{url}/metrics", timeout=60).json()
    keys = ("numDraftTokens", "numAcceptedDraftTokens")
    for it in stats:
        assert all(key in it for key in keys), f"an iteration without {keys}: {sorted(it)}"
    drafted = sum(int(it["numDraftTokens"]) for it in stats)
    accepted = sum(int(it["numAcceptedDraftTokens"]) for it in stats)
    return drafted, accepted


def test_dspark_draft_handoff(served, model_path, record_property):
    """NT-130 (F-31, DS-S16): the context engine's DSpark drafts travel with the handoff.

    The test sends both legs to the workers itself: the context leg's handoff carries draft
    tokens, and the generation leg decodes from it. In form K the generation worker's iteration
    stats count accepted drafts; the chain's ds engine reports no acceptance through its
    OpenEngine server (UV-10), which the test records.
    """
    import time

    ids = _prompt_ids(model_path, basic.COMPLETION_PROMPT)
    gen_url = served.workers["generation"]
    if not served.chain:
        _accepted_drafts(gen_url)  # hand out the stats of earlier iterations
    context = _context_leg(served, ids, max_tokens=DRAFT_DECODE)
    params = context["choices"][0].get("disaggregated_params") or {}
    assert params.get("draft_tokens"), params
    response = _generation_leg(served, ids, context, max_tokens=DRAFT_DECODE, ignore_eos=True)
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["choices"][0]["text"], out
    assert out["usage"]["completion_tokens"] == DRAFT_DECODE, out["usage"]
    if served.chain:
        record_property("uv10", "the ds OpenEngine server reports no accepted drafts")
        return
    drafted = accepted = 0
    deadline = time.monotonic() + STATS_WAIT_S
    while accepted == 0 and time.monotonic() < deadline:
        more = _accepted_drafts(gen_url)
        drafted, accepted = drafted + more[0], accepted + more[1]
        time.sleep(1)
    record_property("drafts", f"{accepted} of {drafted} accepted")
    assert accepted > 0, (drafted, accepted)


# A sample hook of _test_openai_post_processor.py: it upper-cases every chunk of text.
UPPERCASE_HOOK = "_postproc_hook_samples.UppercaseHook"
POSTPROC_WORKERS = 2


def _postproc_group() -> basic.Group:
    """Both worker front ends postprocess in a pool and run :data:`UPPERCASE_HOOK`.

    The hook's module is next to this file, which goes first on the workers' ``PYTHONPATH``.
    """
    import os

    apps = os.path.dirname(os.path.abspath(__file__))
    path = os.pathsep.join(p for p in (apps, os.environ.get("PYTHONPATH")) if p)
    front = {"num_postprocess_workers": POSTPROC_WORKERS, "post_processor_hook": UPPERCASE_HOOK}
    env = {role: {"PYTHONPATH": path} for role in basic.ROLES}
    return basic.Group(context=front, generation=front, env=env)


POSTPROC_CASES = [(api, stream) for api in ("completions", "chat") for stream in (False, True)]


@pytest.fixture
def postproc_served(groups, form):
    """The default group's answers to :data:`POSTPROC_CASES`, then the group of :func:`_postproc_group`."""
    if form != basic.FORM_CHAIN:
        pytest.skip("form K: the ds proxy has no postprocessing pool or hook")
    served = groups.get()
    baseline = {case: basic.generate(served, *case, max_tokens=8)[0] for case in POSTPROC_CASES}
    return groups.get(_postproc_group()), baseline


def test_postproc_workers_and_hook(postproc_served):
    """NT-127 (F-38, F-40): with a postprocessing pool and a hook, the output is the hook's rewrite.

    Each answer of the workers with a postprocessing pool and the upper-casing hook is the
    module's default group's answer, upper-cased.
    """
    served, baseline = postproc_served
    for case in POSTPROC_CASES:
        text, _, _ = basic.generate(served, *case, max_tokens=8)
        assert any(c.islower() for c in baseline[case]), (case, baseline[case])
        assert text == baseline[case].upper(), (case, text, baseline[case])


# The context engine's log line of its conversation statistics.
CONVERSATION_STATS = (
    r"attention-DP conversations: (\d+) admissions of a conversation seen before, (\d+) of them "
    r"on another rank; (\d+) pinned"
)
# The context engine places a conversation's requests on its rank, and logs its statistics
# every scheduling step.
CONVERSATION_GROUP = basic.Group(
    env={"context": {"DSV4_ADP_CONV_PIN": "1", "DSV4_SCHED_PROF": "1"}}
)
CONVERSATION_PROMPTS = (
    "The capital of France is",
    "Two plus two equals",
    "The largest planet of the solar system is",
)


def _engine_log(served: basic.Served, role: str):
    """The log of the ``role`` ds engine (form K: the ds worker)."""
    from pathlib import Path

    if served.chain:
        return Path(served.servers[role]._chain_handle.engine_log)
    return served.native.log(role)


def _conversation_stats(path) -> tuple[int, int, int]:
    """The context engine's last logged conversation statistics: seen again, moved, pinned.

    All 0 while it has logged none, or no log yet.
    """
    import re

    text = path.read_text(errors="replace") if path.exists() else ""
    found = re.findall(CONVERSATION_STATS, text)
    return tuple(int(n) for n in found[-1]) if found else (0, 0, 0)


@pytest.fixture
def conversation_served(groups) -> basic.Served:
    return groups.get(CONVERSATION_GROUP)


def test_conversation_id_reaches_engine(conversation_served, record_property):
    """NT-129 (F-44c, DS-S18b, DS-O10): a request's conversation reaches the context engine.

    The requests of one conversation carry its id (the chain: ``conversation_params``, which
    becomes the engine request's ``conversation_id``; form K: the session header the ds worker
    hashes). Their prompts share no KV-cache page with any request before them, so no rank
    holds a prefix of them. The context engine counts each request after the first as an
    admission of a conversation seen before, and, with no prefix to follow, pins it to the
    conversation's rank (``DSV4_ADP_CONV_PIN``); none moves to another rank.
    """
    import time
    import uuid

    served = conversation_served
    log = _engine_log(served, "context")
    before = _conversation_stats(log)
    conversation = f"nt129-{uuid.uuid4().hex}"
    for prompt in CONVERSATION_PROMPTS:
        text, _, _ = basic.generate(
            served,
            "completions",
            False,
            # The conversation id first: no earlier request shares the prompt's first page.
            prompt=f"[{conversation}] {prompt}",
            max_tokens=4,
            conversation_params={"conversation_id": conversation},
            extra_headers={"x-correlation-id": conversation},
        )
        assert text, prompt
    again = len(CONVERSATION_PROMPTS) - 1
    deadline = time.monotonic() + STATS_WAIT_S
    while True:
        after = _conversation_stats(log)
        seen, moved, pinned = (a - b for a, b in zip(after, before))
        if seen >= again or time.monotonic() > deadline:
            break
        time.sleep(1)
    record_property("conversation_stats", f"seen {seen}, moved {moved}, pinned {pinned}")
    assert seen >= again and pinned >= again and moved == 0, (before, after)


LONG_CONTEXT_TOKENS = 128000
LONG_CONTEXT_OUTPUT = 100
LONG_CONTEXT_GROUP = basic.Group(engine_args=_engine_args(max_seq_len=131072))


@pytest.fixture
def long_context_served(groups) -> basic.Served:
    return groups.get(LONG_CONTEXT_GROUP)


def test_long_context_smoke(long_context_served, model_path):
    """NT-131 (F-37): a 128000-token prompt and 100 output tokens, the values of XD-056.

    The engines' ``--max_seq_len`` is 131072 instead of the default group's 32768.
    """
    served = long_context_served
    ids = _prompt_ids(model_path, CACHE_TEXT)
    prompt = (ids * (LONG_CONTEXT_TOKENS // len(ids) + 1))[:LONG_CONTEXT_TOKENS]
    text, finish, usage = basic.generate(
        served,
        "completions",
        False,
        prompt=prompt,
        max_tokens=LONG_CONTEXT_OUTPUT,
        ignore_eos=True,
    )
    assert text and finish == "length", (text, finish)
    assert usage.prompt_tokens == LONG_CONTEXT_TOKENS, usage
    assert usage.completion_tokens == LONG_CONTEXT_OUTPUT, usage


# Each worker runs two serving front ends; the context worker's front end i also listens on
# its port + AFFINITY_STRIDE * (i + 1), where the orchestrator sends a conversation's requests.
FRONTENDS = 2
AFFINITY_STRIDE = 7
_FRONTENDS = {"num_serve_frontends": FRONTENDS}
_AFFINITY = {**_FRONTENDS, "serve_frontend_affinity_port_stride": AFFINITY_STRIDE}
MULTI_FRONTEND_GROUP = basic.Group(
    context=_AFFINITY,
    generation=_FRONTENDS,
    front={"context_servers": {"num_instances": 0, "urls": [], **_AFFINITY}},
)
COUNTER_WAIT_S = 30


def _port_with_affinity_ports() -> tuple[int, list[int]]:
    """A free port P with P + AFFINITY_STRIDE * (i + 1) free for each front end i; checked only."""
    import contextlib
    import socket

    for _ in range(100):
        with socket.socket() as probe:
            probe.bind(("", 0))
            port = probe.getsockname()[1]
        ports = [port + AFFINITY_STRIDE * (i + 1) for i in range(FRONTENDS)]
        if ports[-1] >= 65536:
            continue
        with contextlib.ExitStack() as stack:
            try:
                for candidate in (port, *ports):
                    stack.enter_context(socket.socket()).bind(("", candidate))
            except OSError:
                continue
        return port, ports
    raise RuntimeError("no port with free affinity ports")


def _start_multi_frontend_chain(stack, model: str, workdir, group: basic.Group) -> basic.Served:
    """:func:`basic.ds_disagg_chain` with the context worker on a port whose affinity ports are free.

    The affinity ports are reserved with the worker ports, so the harness does not give them
    to an engine. The servers stop when the ``contextlib.ExitStack`` ``stack`` closes.
    """
    import os
    from pathlib import Path

    from openengine_chain import launch

    from tensorrt_llm._utils import get_free_ports

    from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    ctx_port, affinity_ports = _port_with_affinity_ports()
    disagg_port, gen_port = get_free_ports(2)
    launch.reserve_ports(disagg_port, ctx_port, gen_port, *affinity_ports)
    cluster = {"cluster_uri": f"http://localhost:{disagg_port}", "cluster_name": ""}

    def worker(role: str, port: int) -> RemoteOpenAIServer:
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": basic.ROLE_GPUS[role],
            **group.env.get(role, {}),
        }
        return RemoteOpenAIServer(
            model,
            port=port,
            env=env,
            cli_args=basic.worker_serve_args(role, group),
            extra_config={**group.config(role), "disagg_cluster": cluster},
            log_path=str(workdir / f"output_{role}.log"),
            wait=False,
        )

    front = {"hostname": "localhost", "port": disagg_port, "disagg_cluster": cluster, **group.front}
    ctx = stack.enter_context(worker("context", ctx_port))
    gen = stack.enter_context(worker("generation", gen_port))
    orchestrator = stack.enter_context(
        RemoteDisaggOpenAIServer(
            ctx_servers=[],
            gen_servers=[],
            port=disagg_port,
            disagg_config=front,
            log_path=str(workdir / "output_orchestrator.log"),
            # Its /health answers 503 until both workers register; RemoteOpenAIServer's own wait
            # has no time limit for a non-200 answer.
            wait_ready=False,
        )
    )
    for server in (ctx, gen):
        server.wait_for_server(timeout=basic.NATIVE_READY_TIMEOUT_S)
    url = f"http://localhost:{disagg_port}"
    basic.wait_registered(url, {"context": 1, "generation": 1})
    basic.wait_healthy(url)
    return basic.Served(
        url,
        ctx._chain_handle.engine_endpoint,
        basic.FORM_CHAIN,
        workers={"context": ctx.url_root, "generation": gen.url_root},
        engines={
            "context": ctx._chain_handle.engine_endpoint,
            "generation": gen._chain_handle.engine_endpoint,
        },
        servers={"context": ctx, "generation": gen, "orchestrator": orchestrator},
    )


def _conversations() -> dict[int, str]:
    """A conversation id for each context front end, by the front end the orchestrator picks."""
    import zlib

    found: dict[int, str] = {}
    for n in range(1000):
        conversation = f"nt128-conversation-{n}"
        found.setdefault(zlib.crc32(conversation.encode()) % FRONTENDS, conversation)
        if len(found) == FRONTENDS:
            return found
    raise AssertionError("no conversation id for some front end")


def _sent_by_pid(served: basic.Served, role: str, bucket: str) -> dict[int, int]:
    """The ``role`` worker's front ends' Generate calls of ``bucket`` sent, by process id."""
    from tensorrt_llm.grpc.openengine_conventions import SERVE_COUNTER_GENERATE_SENT_BUCKETS
    from tensorrt_llm.serve.openengine_poc import contracts

    key = SERVE_COUNTER_GENERATE_SENT_BUCKETS[bucket]
    counters = contracts.read_serve_counters_by_pid(served.servers[role]._chain_handle.counter_dir)
    return {pid: counts[key] for pid, counts in counters.items()}


@pytest.fixture
def multi_frontend_served(groups, model_path):
    """The chain of :data:`MULTI_FRONTEND_GROUP`, started here instead of by ``groups``.

    :func:`basic.ds_disagg_chain` does not keep the affinity ports free, so the test's chain is
    its own; the group that is up stops first.
    """
    import contextlib

    if groups.form != basic.FORM_CHAIN:
        pytest.skip("form K: the serving front ends under test are the worker front ends")
    groups.close()
    workdir = groups.workdir / "multi_frontend"
    with contextlib.ExitStack() as stack:
        yield _start_multi_frontend_chain(stack, model_path, workdir, MULTI_FRONTEND_GROUP)


def test_multi_frontend_and_affinity_port(multi_frontend_served):
    """NT-128 (F-39, DG-23, DS-O04, DS-O09): workers of two front ends serve two-stage requests.

    Both workers run two serving front ends on one ds engine each, and the orchestrator knows the
    context worker's affinity ports. Every request of a conversation reaches the same context
    front end, and the two conversations, picked for different front ends, reach different ones.
    """
    import time

    from tensorrt_llm.grpc.openengine_conventions import BUCKET_CONTEXT_ONLY, BUCKET_GENERATION_ONLY

    served = multi_frontend_served
    conversations = _conversations()
    # By chance alone, two conversations of four requests each keep to one front end each,
    # and different ones, once in 128 runs.
    rounds = 4
    gen_before = sum(_sent_by_pid(served, "generation", BUCKET_GENERATION_ONLY).values())
    reached: dict[str, set[int]] = {c: set() for c in conversations.values()}
    for _ in range(rounds):
        for conversation in conversations.values():
            before = _sent_by_pid(served, "context", BUCKET_CONTEXT_ONLY)
            text, _, _ = basic.generate(
                served,
                "completions",
                False,
                max_tokens=4,
                conversation_params={"conversation_id": conversation},
            )
            assert text, conversation
            deadline = time.monotonic() + COUNTER_WAIT_S
            while True:
                after = _sent_by_pid(served, "context", BUCKET_CONTEXT_ONLY)
                grown = {
                    pid: n - before.get(pid, 0)
                    for pid, n in after.items()
                    if n != before.get(pid, 0)
                }
                if grown or time.monotonic() > deadline:
                    break
                time.sleep(0.5)
            assert list(grown.values()) == [1], (conversation, before, after)
            reached[conversation] |= set(grown)
    assert all(len(pids) == 1 for pids in reached.values()), reached
    assert len(set.union(*reached.values())) == FRONTENDS, reached
    want = gen_before + rounds * FRONTENDS
    deadline = time.monotonic() + COUNTER_WAIT_S
    while True:
        gen_after = sum(_sent_by_pid(served, "generation", BUCKET_GENERATION_ONLY).values())
        if gen_after >= want or time.monotonic() > deadline:
            break
        time.sleep(0.5)
    assert gen_after == want, (gen_before, gen_after)
