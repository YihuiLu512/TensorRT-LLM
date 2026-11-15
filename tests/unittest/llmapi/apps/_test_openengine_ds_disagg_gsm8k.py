# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: accuracy and output parity.

GSM8K-200 runs with the DeepSeek-V4.1 recipe of ``tests/integration/defs/deepseek_v41_gsm8k.py``
(32-shot, chat template without thinking, greedy, up to 1024 new tokens, the rendered prompt
sent to ``/v1/completions`` without special tokens). The forms, fixtures and environment are
those of ``_test_openengine_ds_disagg_basic.py``; needs ``${LLM_MODELS_ROOT}/datasets/openai/gsm8k``.

:func:`test_gsm8k_200_paired_native_vs_chain` compares the two forms, which run in two pytest
sessions: each session appends its rounds to a file in ``TRTLLM_TEST_DS_GSM8K_PAIR_DIR``, and the
chain's session judges its rounds against form K's.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import _test_openengine_ds_disagg_basic as basic

# deepseek_v41_serving: the smoke prompts of the DeepSeek-V4.1 disaggregated tests.
__extra_import_path__ = ["~/tests/integration/defs"]

MODEL_NAME = basic.MODEL_NAME
# The module's fixtures.
model_path = basic.model_path
form = basic.form
groups = basic.groups
served = basic.served

pytestmark = [pytest.mark.threadleak(enabled=False)]

GSM8K_NUM_SAMPLES = 200
GSM8K_MAX_OUTPUT_TOKENS = 1024
GSM8K_NUM_FEWSHOT = 32
GSM8K_SYSTEM_PROMPT = (
    "Solve the problem carefully. "
    "End your response with a final line exactly in the form #### <answer>."
)
GSM8K_FILTERS = ("strict-match", "flexible-extract")
# Every run of GSM8K-200 must score above this, on both filters.
GSM8K_GATE = 95.0
# deepseek_v41_gsm8k.py's gate (reference 97.0 for 200 samples): computed and reported only.
ORIGINAL_REFERENCE = 97.0
ORIGINAL_SIGMA, ORIGINAL_ALPHA = 50.0, 0.05
# Requests in flight: the ds-engine's dev/test_disagg.sh, with which form K's baseline was
# read, and deepseek_v41_gsm8k.py's.
PAIRED_CONCURRENCY = 32
SMOKE_CONCURRENCY = 8

ENV_PAIR_DIR = "TRTLLM_TEST_DS_GSM8K_PAIR_DIR"
ENV_ROUNDS = "TRTLLM_TEST_DS_GSM8K_ROUNDS"
# The paired comparison: three rounds per form by default, and the chain's mean strict-match
# score at least form K's minus PAIRED_MARGIN points (1.0: two of the 200 questions).
DEFAULT_ROUNDS = 3
PAIRED_MARGIN = 1.0


@pytest.fixture(scope="module")
def tokenizer(model_path):
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    return DeepseekV41Tokenizer.from_pretrained(model_path)


def original_threshold() -> float:
    """``HypothesisTestingParams(ref_accuracy=97.0, num_samples=200).threshold``."""
    from scipy.stats import norm

    scale = (2 * ORIGINAL_SIGMA**2 / GSM8K_NUM_SAMPLES) ** 0.5
    return ORIGINAL_REFERENCE + norm.ppf(ORIGINAL_ALPHA) * scale


@dataclass
class Gsm8kRun:
    """One GSM8K-200 run: the score per filter, the documents strict-match got right."""

    scores: dict[str, float]
    correct: list[int]
    seconds: float


def _dataset_path() -> str:
    """The GSM8K dataset; skipped without a models root, failed when the root lacks it."""
    from ..test_llm import llm_models_root

    root = llm_models_root()
    if root is None:
        pytest.skip("no models root: LLM_MODELS_ROOT is not set or does not exist")
    path = os.path.join(root, "datasets/openai/gsm8k")
    assert os.path.isdir(path), f"no GSM8K dataset at {path} under the models root"
    return path


def _results(output_dir: Path) -> tuple[dict[str, float], list[int]]:
    """The scores and strict-match's correct documents, after checking every sample is there."""
    with (output_dir / "samples_gsm8k.json").open() as f:
        results = json.load(f)
    assert results["n-samples"]["gsm8k"]["effective"] == GSM8K_NUM_SAMPLES
    samples = results["samples"]["gsm8k"]
    scores, correct = {}, []
    for name in GSM8K_FILTERS:
        rows = [row for row in samples if row["filter"] == name]
        assert sorted(row["doc_id"] for row in rows) == list(range(GSM8K_NUM_SAMPLES)), name
        assert all(row["exact_match"] in (0, 1) for row in rows), name
        score = results["results"]["gsm8k"][f"exact_match,{name}"]
        expected = 100 * sum(row["exact_match"] for row in rows) / GSM8K_NUM_SAMPLES
        assert math.isfinite(score) and math.isclose(score, expected, abs_tol=1e-9), name
        scores[name] = score
        if name == "strict-match":
            correct = sorted(row["doc_id"] for row in rows if row["exact_match"] == 1)
    return scores, correct


def run_gsm8k(
    url: str, tokenizer, output_dir: Path, concurrency: int, monkeypatch: pytest.MonkeyPatch
) -> Gsm8kRun:
    """GSM8K-200 against the server at ``url`` (deepseek_v41_gsm8k.py's recipe, over HTTP).

    Each request is the rendered prompt without special tokens: its ``prompt_tokens`` must be
    the client's own count of it (F-09).
    """
    from openai import OpenAI

    from tensorrt_llm.evaluate import GSM8K
    from tensorrt_llm.llmapi import SamplingParams

    evaluator = GSM8K(
        dataset_path=_dataset_path(),
        num_samples=GSM8K_NUM_SAMPLES,
        random_seed=0,
        num_fewshot=GSM8K_NUM_FEWSHOT,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        system_prompt=GSM8K_SYSTEM_PROMPT,
        chat_template_kwargs={"enable_thinking": False},
        stop_strings=[],
        log_samples=True,
        output_path=str(output_dir),
    )
    task = evaluator.task_dict["gsm8k"]
    # Task generation kwargs override SamplingParams in LmEvalWrapper.
    task.set_config(
        key="generation_kwargs",
        value={**task.config.generation_kwargs, "max_gen_toks": GSM8K_MAX_OUTPUT_TOKENS},
    )
    sampling = SamplingParams(
        max_tokens=GSM8K_MAX_OUTPUT_TOKENS,
        temperature=0,
        top_p=1,
        seed=0,
        stop=[],
        add_special_tokens=False,
    )
    client = OpenAI(api_key=basic.DUMMY_API_KEY, base_url=f"{url}/v1", timeout=600, max_retries=0)
    pool = ThreadPoolExecutor(max_workers=concurrency)
    submitted = 0

    def complete(prompt: str, params) -> SimpleNamespace:
        response = client.completions.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            seed=params.seed,
            stop=params.stop,
            n=1,
            stream=False,
            extra_body={"add_special_tokens": False},
        )
        assert len(response.choices) == 1, response
        choice = response.choices[0]
        assert choice.finish_reason in ("stop", "length"), choice
        assert response.usage.prompt_tokens == len(
            tokenizer.encode(prompt, add_special_tokens=False)
        ), response.usage
        assert 0 < response.usage.completion_tokens <= GSM8K_MAX_OUTPUT_TOKENS, response.usage
        return SimpleNamespace(outputs=[SimpleNamespace(text=choice.text)])

    def generate_async(prompt: str, sampling_params, streaming: bool = False) -> Future:
        nonlocal submitted
        assert not streaming
        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        assert prompt_tokens + GSM8K_MAX_OUTPUT_TOKENS <= basic_max_seq_len(), prompt_tokens
        submitted += 1
        return pool.submit(complete, prompt, sampling_params)

    monkeypatch.setenv("TLLM_EVAL_MAX_IN_FLIGHT", str(concurrency))
    monkeypatch.delenv("TLLM_EVAL_SPEC_STATS", raising=False)
    started = time.monotonic()
    try:
        evaluator.evaluate(
            SimpleNamespace(tokenizer=tokenizer, generate_async=generate_async),
            sampling,
            scores_filter="exact_match,strict-match",
            sampling_override=True,
        )
    finally:
        # After a failure, do not wait for the questions still queued.
        pool.shutdown(wait=False, cancel_futures=True)
        client.close()
    seconds = time.monotonic() - started
    assert submitted == GSM8K_NUM_SAMPLES, submitted
    scores, correct = _results(output_dir)
    return Gsm8kRun(scores, correct, seconds)


def basic_max_seq_len() -> int:
    """The engines' ``--max_seq_len``."""
    return int(basic.ENGINE_ARGS[basic.ENGINE_ARGS.index("--max_seq_len") + 1])


def check_gsm8k(run: Gsm8kRun, what: str) -> list[str]:
    """Filters not above :data:`GSM8K_GATE`; the original gate is reported, not applied."""
    threshold = original_threshold()
    print(
        f"[gsm8k] {what}: {json.dumps(run.scores)} in {run.seconds:.1f} s; the original gate "
        f"(reference {ORIGINAL_REFERENCE}) would need >= {threshold:.2f}: "
        + ("met" if min(run.scores.values()) >= threshold else "NOT met")
    )
    return [
        f"{what}: GSM8K-200 {name} {score:.2f} <= {GSM8K_GATE}"
        for name, score in run.scores.items()
        if not score > GSM8K_GATE
    ]


def pair_settings(model: str) -> dict:
    """What two forms' rounds must share to be compared."""
    return {
        "model": os.path.basename(os.path.normpath(model)),
        "samples": GSM8K_NUM_SAMPLES,
        "fewshot": GSM8K_NUM_FEWSHOT,
        "max_output_tokens": GSM8K_MAX_OUTPUT_TOKENS,
        "concurrency": PAIRED_CONCURRENCY,
        "engine_args": list(basic.ENGINE_ARGS),
        "job": os.environ.get("SLURM_JOB_ID", ""),
    }


def rounds_file(directory: str, form: str) -> Path:
    return Path(directory) / f"gsm8k_{form}.jsonl"


def append_round(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_rounds(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_pairs(native: list[dict], chain: list[dict]) -> list[str]:
    """Problems of the chain's rounds against form K's; the per-question comparison is printed.

    Round ``i`` of the chain is paired with round ``i`` of form K. The chain's mean strict-match
    score must be at least form K's minus :data:`PAIRED_MARGIN`.
    """
    problems = []
    if len(chain) != len(native) or not chain:
        return [
            f"{len(native)} form K rounds and {len(chain)} chain rounds: run each K round first"
        ]
    for k, c in zip(native, chain):
        if k["settings"] != c["settings"]:
            problems.append(f"round {c['round']} ran with other settings: {k['settings']}")
    native_mean = statistics.mean(r["scores"]["strict-match"] for r in native)
    chain_mean = statistics.mean(r["scores"]["strict-match"] for r in chain)
    for k, c in zip(native, chain):
        k_right, c_right = set(k["correct"]), set(c["correct"])
        print(
            f"[gsm8k] round {c['round']}: K {k['scores']['strict-match']:.1f}, chain "
            f"{c['scores']['strict-match']:.1f}; right on the chain only {len(c_right - k_right)}, "
            f"on K only {len(k_right - c_right)}"
        )
    print(f"[gsm8k] mean strict-match: K {native_mean:.2f}, chain {chain_mean:.2f}")
    if chain_mean < native_mean - PAIRED_MARGIN:
        problems.append(
            f"the chain's mean {chain_mean:.2f} is below form K's {native_mean:.2f} by more "
            f"than {PAIRED_MARGIN}"
        )
    return problems


@pytest.mark.timeout(7200)
def test_gsm8k_200_paired_native_vs_chain(
    served, form, model_path, tokenizer, tmp_path, monkeypatch
):
    """NT-108 (F-36): GSM8K-200 in rounds on both forms, every round > 95, the chain paired with K.

    This session runs ``TRTLLM_TEST_DS_GSM8K_ROUNDS`` rounds (default 3) on its form and appends
    each to ``gsm8k_<form>.jsonl`` in ``TRTLLM_TEST_DS_GSM8K_PAIR_DIR`` (without it, form K's
    session only checks its rounds and the chain's session fails). Form K's session runs first;
    the chain's session then pairs its rounds with form K's, round for round, and requires the
    chain's mean strict-match score to be at least form K's minus one point (two questions).
    The questions only one form answered right are reported per round, not judged. Sessions may
    alternate (K, chain, K, chain, ...): the chain's session judges every round recorded so far.
    """
    _dataset_path()
    directory = os.environ.get(ENV_PAIR_DIR)
    if served.chain and not directory:
        pytest.fail(f"the chain's rounds are judged against form K's: set {ENV_PAIR_DIR}")
    rounds = int(os.environ.get(ENV_ROUNDS, DEFAULT_ROUNDS))
    settings = pair_settings(model_path)
    path = rounds_file(directory, form) if directory else None
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
    done = len(read_rounds(path)) if path else 0
    problems = []
    for i in range(rounds):
        run = run_gsm8k(
            served.url, tokenizer, tmp_path / f"round{i}", PAIRED_CONCURRENCY, monkeypatch
        )
        if path:
            record = {"form": form, "round": done + i, "settings": settings, "time": time.time()}
            record.update(scores=run.scores, correct=run.correct, seconds=run.seconds)
            append_round(path, record)
        problems += check_gsm8k(run, f"form {form}, round {done + i}")
    if served.chain:
        native = read_rounds(rounds_file(directory, basic.FORM_K))
        problems += judge_pairs(native, read_rounds(path))
    assert not problems, problems


PARITY_PROMPTS = (
    basic.COMPLETION_PROMPT,
    "The three primary colors are",
    "def fibonacci(n):\n",
    "Water boils at sea level at a temperature of",
    "Once upon a time, in a village by the sea,",
    "The largest planet in the solar system is",
    "Translate to French: Good morning, how are you?\n",
    "1, 1, 2, 3, 5, 8,",
)
PARITY_MAX_TOKENS = 32


def _greedy(client, prompt: str) -> tuple[list[int] | None, str]:
    """The greedy output's token ids (``None`` when the endpoint returns none) and text."""
    out = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt,
        max_tokens=PARITY_MAX_TOKENS,
        temperature=0.0,
        extra_body={"detokenize": False},
    )
    ids = getattr(out.choices[0], "token_ids", None)
    text = out.choices[0].text
    if ids:
        return ids, text
    if not text:
        out = client.completions.create(
            model=MODEL_NAME, prompt=prompt, max_tokens=PARITY_MAX_TOKENS, temperature=0.0
        )
        text = out.choices[0].text
    return None, text


def _common_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_two_endpoint_parity(served, record_property):
    """NT-109 (F-01, F-02): the disaggregated server against one worker serving alone.

    Each prompt goes, one request at a time, to the context worker as an aggregated request
    (the reference) and to the disaggregated server (the candidate). Reported per prompt: whether
    the first tokens agree, and the longest common prefix of the greedy outputs, in tokens
    (where an endpoint returns no token ids, in characters, and the first character stands for
    the first token). Only reported: serving output is not deterministic from run to run, so no
    threshold is set; the test fails only when an endpoint returns no output.
    """
    reference = served.client(served.workers["context"])
    candidate = served.client()
    rows = []
    for prompt in PARITY_PROMPTS:
        ref_ids, ref_text = _greedy(reference, prompt)
        cand_ids, cand_text = _greedy(candidate, prompt)
        assert (ref_ids or ref_text) and (cand_ids or cand_text), (prompt, ref_text, cand_text)
        tokens = ref_ids is not None and cand_ids is not None
        a, b = (ref_ids, cand_ids) if tokens else (ref_text, cand_text)
        rows.append(
            {
                "prompt": prompt,
                "unit": "token" if tokens else "char",
                "first_token_match": a[:1] == b[:1],
                "common_prefix": _common_prefix(a, b),
                "reference_len": len(a),
                "candidate_len": len(b),
            }
        )
    summary = {
        "form": served.form,
        "prompts": len(rows),
        "first_token_matches": sum(r["first_token_match"] for r in rows),
        "identical": sum(
            r["common_prefix"] == r["reference_len"] == r["candidate_len"] for r in rows
        ),
        "rows": rows,
    }
    print("[parity] " + json.dumps(summary))
    record_property("two_endpoint_parity", json.dumps(summary))


# The tests below run on engine groups of their own, after the default group's tests. A test
# file's group changes only in a fixture, during setup.

# deepseek_v41_serving.MAX_NUM_TOKENS: the long smoke prompt is prefilled in several chunks.
SMOKE_MAX_NUM_TOKENS = "2048"


def _smoke_args() -> tuple[str, ...]:
    args = list(basic.ENGINE_ARGS)
    args[args.index("--max_num_tokens") + 1] = SMOKE_MAX_NUM_TOKENS
    return tuple(args)


# N-0: context DEP2, generation TP2.
GEN_TP2_GROUP = basic.Group(engine_args=_smoke_args())
# Context DEP2, generation DEP2.
GEN_DEP2_GROUP = basic.Group(generation={"enable_attention_dp": True}, engine_args=_smoke_args())


def _smoke_completion(url: str, prompt: str, stream: bool) -> tuple[str, dict]:
    """deepseek_v41's smoke request: greedy, ``stop`` expected; the text and the usage."""
    import requests
    from deepseek_v41_serving import MAX_OUTPUT_TOKENS

    body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    with requests.post(f"{url}/v1/completions", json=body, stream=stream, timeout=600) as response:
        response.raise_for_status()
        if not stream:
            payload = response.json()
            assert payload["choices"][0]["finish_reason"] == "stop", payload
            return payload["choices"][0]["text"], payload["usage"]
        text, usage, finished, done = "", {}, False, False
        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                continue
            payload = json.loads(data)
            assert "error" not in payload, payload
            usage = payload.get("usage") or usage
            for choice in payload.get("choices", []):
                text += choice["text"]
                if choice.get("finish_reason") is not None:
                    assert choice["finish_reason"] == "stop", choice
                    finished = True
        assert done and finished, "the streamed response did not finish cleanly"
        return text, usage


# How long the engines' counters may take to show a request after its HTTP response, and how
# long they are watched for more once they do.
COUNTER_TIMEOUT_S = 30.0
COUNTER_SETTLE_S = 2.0


def _handoffs(served) -> tuple[int, int]:
    """PrefillReady events the context engine sent, generation-only calls the generation took."""
    from tensorrt_llm.grpc.openengine_conventions import (
        BUCKET_GENERATION_ONLY,
        COUNTER_GENERATE_ACCEPTED_BUCKETS,
        COUNTER_PREFILL_READY_SENT,
    )

    return (
        basic.engine_counter(served.engines["context"], COUNTER_PREFILL_READY_SENT),
        basic.engine_counter(
            served.engines["generation"], COUNTER_GENERATE_ACCEPTED_BUCKETS[BUCKET_GENERATION_ONLY]
        ),
    )


def _one_handoff(served, before: tuple[int, int]) -> tuple[int, int]:
    """The hand-offs since ``before``: polled until both counters moved (at most
    ``COUNTER_TIMEOUT_S``), then read again ``COUNTER_SETTLE_S`` later."""
    deadline = time.monotonic() + COUNTER_TIMEOUT_S
    while True:
        now = _handoffs(served)
        if all(n > b for n, b in zip(now, before)) or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    time.sleep(COUNTER_SETTLE_S)
    after = _handoffs(served)
    return after[0] - before[0], after[1] - before[1]


def _chain_group(groups, group: basic.Group) -> basic.Served:
    """Engine group ``group`` of the chain; skipped in form K without starting it."""
    if groups.form != basic.FORM_CHAIN:
        pytest.skip("form K: the layouts under test are the chain's")
    return groups.get(group)


@pytest.fixture
def gen_tp2_served(groups) -> basic.Served:
    """The chain on N-0 (context DEP2, generation TP2) with the smoke prompts' token budget."""
    return _chain_group(groups, GEN_TP2_GROUP)


@pytest.fixture
def gen_dep2_served(groups) -> basic.Served:
    """The chain with context DEP2 and generation DEP2, with the smoke prompts' token budget."""
    return _chain_group(groups, GEN_DEP2_GROUP)


def _smoke_and_gsm8k(served, tokenizer, tmp_path, monkeypatch, layout: str) -> None:
    """deepseek_v41's smoke prompts and GSM8K-200 on the chain's engine group ``served``.

    Each smoke prompt, non-streamed and streamed, gives the expected answer and stops with more
    than one output token, and makes one hand-off: the context engine sent one PrefillReady and
    the generation engine took one generation-only call. The
    long prompt is longer than twice the engines' ``--max_num_tokens``. GSM8K-200 must score
    above 95 on both filters. The DSpark acceptance length is not read: the chain carries no
    speculative-decoding statistics.
    """
    from deepseek_v41_serving import MAX_NUM_TOKENS, build_smoke_cases

    assert int(SMOKE_MAX_NUM_TOKENS) == MAX_NUM_TOKENS
    for case in build_smoke_cases(tokenizer):
        outputs = []
        for stream in (False, True):
            before = _handoffs(served)
            text, usage = _smoke_completion(served.url, case.prompt, stream)
            handoffs = _one_handoff(served, before)
            assert handoffs == (1, 1), (case.name, stream, handoffs)
            assert text.strip() == case.expected, (case.name, stream, text)
            assert usage["completion_tokens"] > 1, (case.name, usage)
            if case.name == "long_recall":
                assert usage["prompt_tokens"] > 4096 > MAX_NUM_TOKENS, usage
            outputs.append(text)
        assert outputs[0] == outputs[1], (case.name, outputs)
    run = run_gsm8k(served.url, tokenizer, tmp_path / "gsm8k", SMOKE_CONCURRENCY, monkeypatch)
    problems = check_gsm8k(run, f"layout {layout}")
    assert not problems, problems


@pytest.mark.timeout(7200)
def test_deepseek_v41_smoke_and_gsm8k_ctx_dep2_gen_tp2(
    gen_tp2_served, tokenizer, tmp_path, monkeypatch
):
    """NT-217 (F-02, F-09, F-31, F-32, F-33, F-36, DG-12, DG-13, DS-S16, DS-S19c): N-0's smoke and GSM8K.

    ``test_disaggregated_deepseek_v41_gsm8k_ctxtp4_gentp4``'s smoke prompts and GSM8K-200, rewritten
    for the chain with context DEP2 and generation TP2 (:func:`_smoke_and_gsm8k`).
    """
    _smoke_and_gsm8k(gen_tp2_served, tokenizer, tmp_path, monkeypatch, "ctx DEP2 + gen TP2")


@pytest.mark.timeout(7200)
def test_deepseek_v41_smoke_and_gsm8k_ctx_dep2_gen_dep2(
    gen_dep2_served, tokenizer, tmp_path, monkeypatch
):
    """NT-218 (F-32, DS-S19c): NT-217 with generation DEP2 as well.

    The one-node counterpart of ``test_disaggregated_deepseek_v41_gsm8k_ctxdp4_gendp4``.
    """
    _smoke_and_gsm8k(gen_dep2_served, tokenizer, tmp_path, monkeypatch, "ctx DEP2 + gen DEP2")
