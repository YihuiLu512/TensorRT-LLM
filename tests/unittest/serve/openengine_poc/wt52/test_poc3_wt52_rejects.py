# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a worker frontend refuses of a disaggregated request, and how (SV-DSG-8, SV-DSG-9), on CPU.

Requests go to the worker frontends of the CPU chain (``cpu_disagg_chain``)
directly, as the orchestrator sends the two legs, or to the orchestrator.
NT-023 (DG-88): each refusal is HTTP 400 before any Generate call, and its
message names the leg. NT-025 (EG-05b): ``trtllm_decode_preparation`` in a
generation leg's ``openengine_handoff_attributes`` is refused. NT-026 (EG-06)
and NT-027 (EG-19): a worker frontend does not start with ``max_beam_width >
1`` or with the remote tail. NT-028 (EG-07): ``n > 1`` with a disaggregated
request is refused. The ``min_tokens`` capability gate (DQ-48) and the leg
named in an engine's own refusal are tested on a facade started in-process.
"""

from __future__ import annotations

import asyncio
import json
import subprocess  # nosec B404
from pathlib import Path

import grpc
import pytest
import yaml

from tensorrt_llm.grpc.openengine_conventions import (
    CAP_STOPPING_MIN_TOKENS,
    CONVENTIONS_VERSION_KEY,
    DISAGG_REQUEST_ID_MIN,
    HANDOFF_DECODE_PREPARATION,
    SERVE_COUNTER_GENERATE_SENT,
    SERVE_COUNTER_HTTP_REJECTED,
)
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc.engine_client import MediaSpec
from tensorrt_llm.serve.openengine_poc.facade import _MAPPED_MEDIA_KEY, OpenEngineLLM, _MappedMedia

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain import (  # noqa: E402
    MODEL,
    CpuDisaggChain,
    _chain_profile,
    _fingerprint,
    _process_env,
    _serve_command,
    make_model_dir,
)
from fake_openengine import Fail, FakeEngine, disagg_profile, running_in_thread  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

CTX_LEG = "the context leg (context_only) of a disaggregated request"
GEN_LEG = "the generation leg (generation_only) of a disaggregated request"
PROMPT = "He llo"


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    directory = tmp_path_factory.mktemp("wt52-rejects")
    # SV-DSG-1: without cache_transceiver_config OpenAIServer refuses the handoff fields itself.
    config = directory / "frontend.yaml"
    config.write_text(yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}}))
    args = ["--config", str(config)]
    with CpuDisaggChain(
        directory, "trtllm", frontend_args={"context": args, "generation": args}
    ) as (chain):
        yield chain


def _delta(before: dict, after: dict, role: str, key: str) -> int:
    return after[role]["frontend"].get(key, 0) - before[role]["frontend"].get(key, 0)


def _context_leg(chain, disagg_request_id: int, **fields) -> dict:
    body = {
        "prompt": PROMPT,
        "max_tokens": 4,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": disagg_request_id,
        },
        **fields,
    }
    response = chain.post("/v1/completions", body, url=chain.ctx.url)
    assert response.status_code == 200, response.text
    return response.json()


def _generation_leg_body(ctx: dict, params: dict | None = None, **fields) -> dict:
    handoff = {**ctx["choices"][0]["disaggregated_params"], "request_type": "generation_only"}
    handoff.update(params or {})
    return {
        "prompt": ctx["prompt_token_ids"],
        "max_tokens": 4,
        "disaggregated_params": handoff,
        **fields,
    }


def _refused(chain, role: str, body: dict, leg: str, *named: str) -> None:
    """``body`` sent to the ``role`` worker is refused with 400 naming ``leg`` and ``named``."""
    worker = chain.ctx if role == "context" else chain.gen
    before = chain.snapshot()
    response = chain.post("/v1/completions", body, url=worker.url)
    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]
    assert message.startswith(f"{leg}: "), message
    for text in named:
        assert text in message, message
    after = chain.snapshot()
    assert _delta(before, after, role, SERVE_COUNTER_GENERATE_SENT) == 0
    assert _delta(before, after, role, SERVE_COUNTER_HTTP_REJECTED) == 1


@pytest.mark.parametrize(
    "fields, named",
    [
        ({"n": 2}, ["n=2", "has one output"]),
        ({"n": 2, "best_of": 2}, ["n=2", "has one output"]),
        ({"best_of": 2}, ["best_of=2", "has one output"]),
    ],
    ids=["n", "n-best_of", "best_of"],
)
def test_two_phase_request_with_n_above_one_is_rejected(chain, fields, named):
    """NT-028 (EG-07, F-19): either leg with more than one output is refused before Generate."""
    ctx_body = {
        "prompt": PROMPT,
        "max_tokens": 4,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": DISAGG_REQUEST_ID_MIN + 11,
        },
        **fields,
    }
    _refused(chain, "context", ctx_body, CTX_LEG, *named)
    ctx = _context_leg(chain, DISAGG_REQUEST_ID_MIN + 12)
    _refused(chain, "generation", _generation_leg_body(ctx, **fields), GEN_LEG, *named)


def test_n_above_one_through_the_orchestrator_is_refused(chain):
    """NT-028 end to end: the orchestrator passes on the context worker's 400."""
    before = chain.snapshot()
    response = chain.complete(prompt=PROMPT, max_tokens=4, n=2)
    assert response.status_code == 400, response.text
    assert CTX_LEG in response.text, response.text
    after = chain.snapshot()
    for role in ("context", "generation"):
        assert _delta(before, after, role, SERVE_COUNTER_GENERATE_SENT) == 0
    assert _delta(before, after, "context", SERVE_COUNTER_HTTP_REJECTED) == 1


@pytest.mark.parametrize(
    "params, named",
    [
        ({"first_gen_logits": [[0.5]]}, ["first_gen_logits"]),
        ({"remote_tail_start": 1}, ["remote_tail_start"]),
        ({"multimodal_positions": [0], "multimodal_lengths": [1]}, ["multimodal"]),
        (
            {"openengine_handoff_attributes": json.dumps({"first_gen_tokens": [1]})},
            ["first_gen_tokens"],
        ),
    ],
    ids=["first_gen_logits", "remote_tail_start", "multimodal", "mapped-key"],
)
def test_two_leg_rejections_name_the_leg(chain, params, named):
    """NT-023 (DG-88): what a handoff cannot carry is refused, naming the generation leg."""
    ctx = _context_leg(chain, DISAGG_REQUEST_ID_MIN + 21)
    _refused(chain, "generation", _generation_leg_body(ctx, params), GEN_LEG, *named)


def test_decode_preparation_attribute_is_not_silently_used(chain):
    """NT-025 (EG-05b): ``trtllm_decode_preparation`` is refused; another attribute reaches the engine."""
    ctx = _context_leg(chain, DISAGG_REQUEST_ID_MIN + 31)
    preparation = {HANDOFF_DECODE_PREPARATION: {"mrope": [1, 2]}}
    body = _generation_leg_body(ctx, {"openengine_handoff_attributes": json.dumps(preparation)})
    _refused(chain, "generation", body, GEN_LEG, HANDOFF_DECODE_PREPARATION)
    vendor = {"vendor_hint": "x"}
    body = _generation_leg_body(ctx, {"openengine_handoff_attributes": json.dumps(vendor)})
    response = chain.post("/v1/completions", body, url=chain.gen.url)
    assert response.status_code == 200, response.text
    assert chain.gen.engine.requests[-1].kv.session.attributes_struct["vendor_hint"] == "x"


@pytest.mark.parametrize(
    "code, status",
    [(grpc.StatusCode.INVALID_ARGUMENT, 400), (grpc.StatusCode.INTERNAL, 500)],
    ids=["refused", "failed"],
)
def test_an_engine_error_names_the_leg(chain, code, status):
    """SV-DSG-8: the engine's own refusal or failure of a leg names the leg, the engine and its message."""
    ctx = _context_leg(chain, DISAGG_REQUEST_ID_MIN + 41)
    chain.gen.engine.push_script(Fail(code, "scripted leg failure"))
    response = chain.post("/v1/completions", _generation_leg_body(ctx), url=chain.gen.url)
    assert response.status_code == status, response.text
    message = response.json()["error"]["message"]
    assert message.startswith(f"{GEN_LEG}: "), message
    assert "OpenEngine engine 'tensorrt_llm'" in message and "scripted leg failure" in message
    assert f":{chain.gen.engine.port}" not in message


def test_an_aggregated_request_error_names_no_leg(chain):
    chain.gen.engine.push_script(Fail(grpc.StatusCode.INVALID_ARGUMENT, "scripted failure"))
    response = chain.complete(prompt=PROMPT, url=chain.gen.url)
    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]
    assert "leg" not in message and "scripted failure" in message


# -- In-process facade ---------------------------------------------------------------------------


def _with_facade(tmp_path: Path, profile_changes: dict, check) -> None:
    """Run ``check(facade)`` on a facade started in-process on a disaggregated trtllm-like fake."""
    model = make_model_dir(tmp_path / "model")
    profile = _chain_profile(disagg_profile("trtllm-like"), _fingerprint(model))
    if profile_changes:
        profile = profile.replace(**profile_changes)

    async def main() -> None:
        async with FakeEngine(profile) as fake:
            facade = OpenEngineLLM.from_llm_args(
                {"model": str(model)}, fake.target, ready_timeout=30
            )
            await facade.start()
            try:
                await check(facade, fake)
            finally:
                await facade.aclose()

    asyncio.run(main())


def _capabilities(min_tokens: bool) -> dict:
    profile = disagg_profile("trtllm-like")
    return {"capabilities": {**profile.capabilities, CAP_STOPPING_MIN_TOKENS: min_tokens}}


@pytest.mark.parametrize(
    "changes, refused",
    [
        (_capabilities(min_tokens=False), True),
        (_capabilities(min_tokens=True), False),
        (
            {**_capabilities(min_tokens=False), "server_info_extra": {CONVENTIONS_VERSION_KEY: 3}},
            False,
        ),
    ],
    ids=["v4-without-key", "v4-with-key", "v3"],
)
def test_min_tokens_needs_the_capability_on_version_4(tmp_path, changes, refused):
    """DQ-48: ``min_tokens >= 1`` is refused only by an engine on version 4 that does not declare it."""

    async def check(facade, fake) -> None:
        sent = len(fake.requests)
        params = SamplingParams(max_tokens=2, min_tokens=1)
        if refused:
            with pytest.raises(ValueError, match="min_tokens=1 is not supported"):
                facade.generate_async({"prompt_token_ids": [5, 6]}, params)
            assert len(fake.requests) == sent
        else:
            output = await facade.generate_async({"prompt_token_ids": [5, 6]}, params)
            assert output.outputs[0].finish_reason is not None
            assert fake.requests[-1].stopping.min_tokens == 1
        # min_tokens 0 is not a request for the capability.
        await facade.generate_async({"prompt_token_ids": [5, 6]}, SamplingParams(max_tokens=2))

    _with_facade(tmp_path, changes, check)


def test_a_disaggregated_request_with_media_is_refused(tmp_path):
    """SV-DSG-8 (F-10, EG-05a): media and a leg do not go together, whatever maps the media."""
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams, to_llm_disaggregated_params

    media = _MappedMedia([MediaSpec(modality="image", url="https://example.invalid/a.png")])
    params = to_llm_disaggregated_params(
        DisaggregatedParams(request_type="context_only", disagg_request_id=DISAGG_REQUEST_ID_MIN)
    )

    async def check(facade, fake) -> None:
        with pytest.raises(ValueError) as error:
            facade.generate_async(
                {"prompt_token_ids": [5, 6], _MAPPED_MEDIA_KEY: media},
                SamplingParams(max_tokens=2),
                disaggregated_params=params,
            )
        message = str(error.value)
        assert message.startswith(f"{CTX_LEG}: ") and "with media is not supported" in message
        assert not fake.requests

    _with_facade(tmp_path, {}, check)


# -- Startup refusals of a worker frontend ---------------------------------------------------------


def _start_worker(tmp_path: Path, config: dict) -> subprocess.CompletedProcess:
    """Run ``trtllm-serve --server_role generation`` with ``config`` in its YAML until it exits."""
    model = make_model_dir(tmp_path / "model")
    config_path = tmp_path / "worker.yaml"
    config_path.write_text(yaml.safe_dump(config))
    profile = _chain_profile(disagg_profile("trtllm-like"), _fingerprint(model))
    with running_in_thread(FakeEngine(profile)) as fake:
        return subprocess.run(  # nosec B603
            [
                *_serve_command(),
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--served_model_name",
                MODEL,
                "--openengine-endpoint",
                fake.target,
                "--openengine-ready-timeout",
                "60",
                "--server_role",
                ServerRole.GENERATION.name.lower(),
                "--config",
                str(config_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=_process_env({}),
        )


def test_worker_role_frontend_refuses_beam_width_above_one(tmp_path):
    """NT-026 (EG-06, F-25): ``max_beam_width > 1`` stops a worker frontend at startup."""
    result = _start_worker(tmp_path, {"max_beam_width": 2})
    output = result.stdout + result.stderr
    assert result.returncode != 0, output[-4000:]
    assert "max_beam_width > 1 is not supported" in output, output[-4000:]


def test_worker_role_frontend_refuses_remote_tail_config(tmp_path):
    """NT-027 (EG-19): the remote tail (``bounded_replay_on_generation``) stops a worker frontend."""
    result = _start_worker(tmp_path, {"bounded_replay_on_generation": True})
    output = result.stdout + result.stderr
    assert result.returncode != 0, output[-4000:]
    assert "bounded_replay_on_generation is not supported" in output, output[-4000:]
