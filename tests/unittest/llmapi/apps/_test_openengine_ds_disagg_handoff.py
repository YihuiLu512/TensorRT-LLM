# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: handoffs a test builds itself, and how they fail.

Most tests here act as the orchestrator: they send a ``context_only`` leg to the context worker,
build the ``generation_only`` leg from its answer, possibly altered, and send it to the
generation worker (on the chain, the worker front ends; in form K, the ds workers). Each test
checks the form its session serves. The forms, fixtures and environment are those of
``_test_openengine_ds_disagg_basic.py``.

The tests that need workers of another kind run last, in this order: the chain with signed
handoffs (:data:`AUTH_GROUP`), the generation worker with another speculative configuration
(:data:`SPEC_MISMATCH_GROUP`), a ds engine without KV transfer behind worker front ends with
roles, and workers started without ``--server_role`` (:func:`static_workers`).
"""

from __future__ import annotations

import base64
import contextlib
import itertools
import json
import os
import re
import secrets
import signal
import struct
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import pytest
import requests

from tensorrt_llm.grpc.openengine_conventions import (
    COUNTER_GENERATE_RECEIVED,
    DISAGG_REQUEST_ID_MIN,
    HANDOFF_DECODE_PREPARATION,
)

from . import _test_openengine_ds_disagg_basic as basic

# The chain harness's engine readings.
__extra_import_path__ = ["~/tests/integration/defs"]

MODEL_NAME = basic.MODEL_NAME
# The module's fixtures.
model_path = basic.model_path
form = basic.form
groups = basic.groups
served = basic.served

pytestmark = [pytest.mark.threadleak(enabled=False)]

COMPLETIONS = "/v1/completions"
REQUEST_TIMEOUT_S = 300
FRONTEND_EXIT_TIMEOUT_S = 600
COUNTER_WAIT_S = 30
# An unfinished sentence: its first token is not the end of the sequence.
PROMPT = "The sky looks blue on a clear day because"
LEG_MAX_TOKENS = 16
# The engines' --max_seq_len (basic.ENGINE_ARGS) is 32768.
TOO_LONG_PROMPT_TOKENS = 40000
# What the ds engine answers a pull of a handoff it does not hold ("no parked KV for disagg id N";
# its MockEngine says "no KV parked under disagg_id N").
NO_PARKED_KV = re.compile(r"no (parked KV|KV parked)")
# A token id beyond any vocabulary the engine loads.
OUT_OF_VOCAB_ID = 1 << 24
# The ids of the context legs a test sends to a worker itself.
_DISAGG_REQUEST_IDS = itertools.count(DISAGG_REQUEST_ID_MIN + secrets.randbelow(1 << 30) * 1024)


def _generate_received(endpoint: str) -> int:
    """The Generate calls the engine at ``endpoint`` has received; it must report the count."""
    from openengine_chain import direct_replay

    attributes = direct_replay.get_load(endpoint).get("attributes", {})
    assert COUNTER_GENERATE_RECEIVED in attributes, (endpoint, attributes)
    return int(float(attributes[COUNTER_GENERATE_RECEIVED]))


def _post(url: str, body: dict, headers: dict | None = None) -> requests.Response:
    return requests.post(
        f"{url}{COMPLETIONS}", json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
    )


def _message(response: requests.Response) -> str:
    """The error message of a worker front end, the orchestrator or a ds server."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    if not isinstance(body, dict):
        return response.text
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")
    return str(body.get("message") or body.get("detail") or error or "")


def _context_leg(served: basic.Served, url: str | None = None, **fields) -> requests.Response:
    """A ``context_only`` leg of PROMPT sent to the context worker (or ``url``).

    ``fields`` add to the request; ``disagg_request_id`` defaults to a new id.
    """
    disagg_request_id = fields.pop("disagg_request_id", None) or next(_DISAGG_REQUEST_IDS)
    body = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "max_tokens": LEG_MAX_TOKENS,
        "temperature": 0.0,
        **fields,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": disagg_request_id,
        },
    }
    return _post(url or served.workers["context"], body)


def _handed_off(response: requests.Response) -> dict:
    """The answer of a context leg that handed its KV cache off."""
    assert response.status_code == 200, response.text[:2000]
    answer = response.json()
    params = answer["choices"][0].get("disaggregated_params")
    assert params and params.get("first_gen_tokens"), answer
    return answer


def _prompt_ids(answer: dict) -> list[int]:
    """The prompt's token ids a context leg's answer carries."""
    ids = answer.get("prompt_token_ids")
    if ids is None and answer.get("prompt_token_ids_b64"):
        blob = base64.b64decode(answer["prompt_token_ids_b64"])
        ids = list(struct.unpack(f"<{len(blob) // 4}i", blob))
    assert ids, answer
    return list(ids)


def _generation_leg(answer: dict, *, drop=(), **changes) -> dict:
    """The ``generation_only`` leg of the context leg answer ``answer``.

    ``changes`` replace fields of its handoff; ``drop`` names fields to leave out.
    """
    params = {**answer["choices"][0]["disaggregated_params"], "request_type": "generation_only"}
    params.update(changes)
    for name in drop:
        params.pop(name, None)
    return {
        "model": MODEL_NAME,
        "prompt": _prompt_ids(answer),
        "max_tokens": LEG_MAX_TOKENS,
        "temperature": 0.0,
        "disaggregated_params": params,
    }


def _generate(served: basic.Served, body: dict, headers: dict | None = None) -> requests.Response:
    return _post(served.workers["generation"], body, headers)


def _text(response: requests.Response) -> str:
    """The text of a request that succeeded."""
    assert response.status_code == 200, response.text[:2000]
    text = response.json()["choices"][0]["text"]
    assert text, response.text[:2000]
    return text


def _no_parked_kv(response: requests.Response) -> None:
    """``response`` is the ds engine's error for a pull of a handoff it does not hold."""
    assert response.status_code >= 400, (response.status_code, response.text[:2000])
    assert NO_PARKED_KV.search(_message(response)), (response.status_code, response.text[:2000])


def _stopped_context_leg(served: basic.Served, first_token: int) -> int:
    """The disagg_request_id of a context leg that ended with ``stop``, parking nothing.

    It stops on its first token. Greedy is not bitwise repeatable: a leg that hands off another
    first token instead has its handoff served, and the next attempt stops on that token.
    """
    stop = first_token
    for _ in range(3):
        disagg_request_id = next(_DISAGG_REQUEST_IDS)
        response = _context_leg(served, disagg_request_id=disagg_request_id, stop_token_ids=[stop])
        assert response.status_code == 200, response.text[:2000]
        choice = response.json()["choices"][0]
        if choice["finish_reason"] == "stop":
            return disagg_request_id
        answer = _handed_off(response)
        stop = int(choice["disaggregated_params"]["first_gen_tokens"][0])
        _text(_generate(served, _generation_leg(answer)))
    pytest.fail("no context leg stopped on its first token in 3 attempts")


ENGINE_ERRORS = ("missing_first_tokens", "unknown_id", "pulled_twice", "ctx_stopped")


@pytest.mark.parametrize("case", ENGINE_ERRORS)
def test_direct_handoff_engine_errors(served, case):
    """NT-135 (DS-R02, DS-S11, DS-S13, DS-U02, DS-U03): a generation leg the ds engine cannot serve.

    Both forms: a handoff without ``first_gen_tokens`` is answered 400 naming them; a pull of a
    disagg_request_id no context leg parked, of a handoff already pulled, or of a context leg
    that ended with ``stop`` gets the engine's error that it holds no parked KV. The handoff the
    test parked is still served afterwards (except the one pulled twice).
    """
    answer = _handed_off(_context_leg(served))
    if case == "pulled_twice":
        _text(_generate(served, _generation_leg(answer)))
        _no_parked_kv(_generate(served, _generation_leg(answer)))
        return
    if case == "missing_first_tokens":
        response = _generate(served, _generation_leg(answer, drop=("first_gen_tokens",)))
        assert response.status_code == 400, response.text[:2000]
        assert "first_gen_tokens" in _message(response), response.text[:2000]
    else:
        if case == "unknown_id":
            other = next(_DISAGG_REQUEST_IDS)
        else:
            first = answer["choices"][0]["disaggregated_params"]["first_gen_tokens"][0]
            other = _stopped_context_leg(served, int(first))
        body = _generation_leg(answer, ctx_request_id=other, disagg_request_id=other)
        _no_parked_kv(_generate(served, body))
    _text(_generate(served, _generation_leg(answer)))


SILENT = ("short_draft", "dp_rank_changed", "duplicate_id")


@pytest.mark.parametrize("case", SILENT)
def test_direct_handoff_silent_behaviors(served, case):
    """NT-136 (DS-I07, DS-I08, DS-I09): handoffs the ds engine serves although they are altered.

    Both forms: a handoff whose ``draft_tokens`` are cut short is served; one whose
    ``ctx_dp_rank`` names the other context rank is served with the output of the unaltered
    handoff (serial greedy requests); of two context legs sent with the same disagg_request_id,
    the handoff of the second is served.
    """
    if case == "duplicate_id":
        disagg_request_id = next(_DISAGG_REQUEST_IDS)
        _handed_off(_context_leg(served, disagg_request_id=disagg_request_id))
        answer = _handed_off(_context_leg(served, disagg_request_id=disagg_request_id))
        _text(_generate(served, _generation_leg(answer)))
        return
    answer = _handed_off(_context_leg(served))
    params = answer["choices"][0]["disaggregated_params"]
    if case == "short_draft":
        drafts = params.get("draft_tokens") or []
        assert drafts, f"the handoff carries no draft tokens to cut: {params}"
        _text(_generate(served, _generation_leg(answer, draft_tokens=drafts[: len(drafts) // 2])))
        return
    reference = _text(_generate(served, _generation_leg(answer)))
    answer = _handed_off(_context_leg(served))
    rank = int(answer["choices"][0]["disaggregated_params"].get("ctx_dp_rank") or 0)
    text = _text(_generate(served, _generation_leg(answer, ctx_dp_rank=1 - rank)))
    assert text == reference, (text, reference)


INVALID_PROMPTS = ("empty", "too_long", "out_of_vocab")
# What the refusal of each invalid prompt names.
INVALID_PROMPT_NAMED = {
    "empty": re.compile(r"(?i)empty|at least one token"),
    "too_long": re.compile(r"(?i)max_seq_len|maximum context length"),
    "out_of_vocab": re.compile(r"(?i)vocab"),
}


def _invalid_prompt(case: str):
    if case == "empty":
        return ""
    if case == "too_long":
        return [1000 + i % 1000 for i in range(TOO_LONG_PROMPT_TOKENS)]
    return [1000, OUT_OF_VOCAB_ID, 1001]


@pytest.mark.parametrize("case", INVALID_PROMPTS)
def test_invalid_prompt_rejected(served, case):
    """NT-140 (DS-R04, DG-41): an empty prompt, one longer than the engines take, an id out of vocabulary.

    The context worker answers the context leg 400 naming what is wrong. On the chain the client
    sees, through the orchestrator, that status and message unchanged; form K's ds proxy wraps
    the 400 in a 500 quoting it. Only that request fails: the next one is served.
    """
    body = {"model": MODEL_NAME, "prompt": _invalid_prompt(case), "max_tokens": 4}
    direct = _context_leg(served, prompt=body["prompt"], max_tokens=4)
    assert direct.status_code == 400, (direct.status_code, direct.text[:2000])
    assert INVALID_PROMPT_NAMED[case].search(_message(direct)), direct.text[:2000]
    response = _post(served.url, body)
    if served.chain:
        assert (response.status_code, _message(response)) == (400, _message(direct)), (
            response.status_code,
            response.text[:2000],
            direct.text[:2000],
        )
    else:
        # The ds proxy wraps the worker's 400 in a 500.
        assert response.status_code == 500, (response.status_code, response.text[:2000])
        message = _message(response)
        assert "HTTP 400" in message, response.text[:2000]
        assert INVALID_PROMPT_NAMED[case].search(message), response.text[:2000]
    text, _, _ = basic.generate(served, "completions", False, max_tokens=4)
    assert text, text


AUTH_KEY = secrets.token_hex(32)
AUTH_REFUSED = "Invalid internal disaggregated request authentication"
# The chain with the internal request authentication: the orchestrator signs each handoff.
AUTH_GROUP = basic.Group(
    context={"internal_request_auth_key": AUTH_KEY},
    generation={"internal_request_auth_key": AUTH_KEY},
    front={"internal_request_auth_key": AUTH_KEY},
)


def _signed(body: dict) -> dict:
    """The headers that sign the handoff of ``body`` with AUTH_KEY, as the orchestrator does."""
    from tensorrt_llm.serve.disagg_auth import build_internal_disagg_auth_headers
    from tensorrt_llm.serve.openai_protocol import CompletionRequest

    headers = build_internal_disagg_auth_headers(AUTH_KEY, CompletionRequest(**body))
    assert headers, "the handoff has no field to sign"
    return headers


def _with_attributes(answer: dict, **attributes) -> dict:
    """The generation leg of ``answer`` with ``attributes`` added to openengine_handoff_attributes."""
    params = answer["choices"][0]["disaggregated_params"]
    held = json.loads(params.get("openengine_handoff_attributes") or "{}")
    return _generation_leg(answer, openengine_handoff_attributes=json.dumps({**held, **attributes}))


@pytest.fixture
def auth_served(groups) -> basic.Served:
    """The chain of :data:`AUTH_GROUP`; skipped in form K, whose ds workers sign nothing."""
    if groups.form != basic.FORM_CHAIN:
        pytest.skip("form K: the ds workers neither sign nor check handoffs")
    return groups.get(AUTH_GROUP)


def test_direct_handoff_auth_and_multimodal_attributes(auth_served):
    """NT-137 (DG-34, DG-37, DG-83): the generation worker front end checks the signed handoff.

    With the internal request authentication on, a generation leg sent to the generation worker
    without the orchestrator's signature, with a wrong one, or with handoff attributes changed
    after signing is answered 4xx naming the failed authentication; a signed one whose handoff
    attributes hold ``trtllm_decode_preparation`` (a multimodal request) is answered 400 naming
    it. The generation engine's Generate count is unchanged after them. The handoff signed
    unchanged is then served.
    """
    from tensorrt_llm.serve.disagg_auth import INTERNAL_DISAGG_AUTH_HEADER

    served = auth_served
    answer = _handed_off(_context_leg(served))
    body = _generation_leg(answer)
    before = _generate_received(served.engines["generation"])

    wrong = {INTERNAL_DISAGG_AUTH_HEADER: "sha256=" + "0" * 64}
    noted = _with_attributes(answer, test_note="signed")
    changed = _with_attributes(answer, test_note="changed")
    for request, headers in ((body, None), (body, wrong), (changed, _signed(noted))):
        response = _generate(served, request, headers)
        assert 400 <= response.status_code < 500, (response.status_code, response.text[:2000])
        assert AUTH_REFUSED in _message(response), response.text[:2000]
    multimodal = _with_attributes(answer, **{HANDOFF_DECODE_PREPARATION: {"positions": [0]}})
    response = _generate(served, multimodal, _signed(multimodal))
    assert response.status_code == 400, response.text[:2000]
    assert HANDOFF_DECODE_PREPARATION in _message(response), response.text[:2000]
    # Each refusal is answered before any Generate call it would make, so one reading is final.
    assert _generate_received(served.engines["generation"]) == before

    _text(_generate(served, body, _signed(body)))


# The generation worker without the speculative decoding the context worker runs.
SPEC_MISMATCH_GROUP = basic.Group(generation={"speculative_config": None})
LAYOUTS_DIFFER = "KV layouts differ"


@pytest.fixture
def spec_mismatch_served(groups):
    """:data:`SPEC_MISMATCH_GROUP` in this session's form, stopped after the test."""
    yield groups.get(SPEC_MISMATCH_GROUP)
    groups.close()


def test_mismatched_spec_config_pull_fails(spec_mismatch_served):
    """NT-141 (DS-U04, DG-41): a generation worker with another speculative configuration.

    Both forms: its pull of the context worker's handoff fails, as the two KV layouts differ;
    the request ends with HTTP 5xx and the engine's reason. On the chain, both engines have
    received the request's leg.
    """
    served = spec_mismatch_served
    before = {role: _generate_received(e) for role, e in served.engines.items()}
    body = {"model": MODEL_NAME, "prompt": PROMPT, "max_tokens": 8, "temperature": 0.0}
    response = _post(served.url, body)
    # Form K: the ds proxy answers a worker's failure with a 500 quoting it.
    assert response.status_code >= 500, (response.status_code, response.text[:2000])
    assert LAYOUTS_DIFFER in response.text, response.text[:2000]
    if not served.chain:
        return
    assert set(before) == set(basic.ROLES), served.engines
    deadline = time.monotonic() + COUNTER_WAIT_S
    while True:
        now = {role: _generate_received(e) for role, e in served.engines.items()}
        if all(now[role] > before[role] for role in basic.ROLES):
            break
        assert time.monotonic() < deadline, (before, now)
        time.sleep(1)


def _serve_command() -> list[str]:
    entry = Path(sys.executable).parent / "trtllm-serve"
    if entry.is_file():
        return [str(entry)]
    return [sys.executable, "-m", "tensorrt_llm.commands.serve"]


def _serving(endpoint: str) -> bool:
    import grpc
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    try:
        with grpc.insecure_channel(endpoint) as channel:
            reply = health_pb2_grpc.HealthStub(channel).Check(
                health_pb2.HealthCheckRequest(service="inference"), timeout=5
            )
    except grpc.RpcError:
        return False
    return reply.status == health_pb2.HealthCheckResponse.SERVING


@contextlib.contextmanager
def _engine_without_kv_transfer(model: str, workdir: Path):
    """A ds engine (OpenEngine server) on the context GPUs, without a role or KV transfer.

    Yields its gRPC address once it serves; stops its process group at the end.
    """
    ds = basic.ds_command()
    if not ds:
        pytest.skip(f"needs {basic.ENV_DS_SERVE_CMD}")
    port = basic.free_ports(1)[0]
    endpoint = f"127.0.0.1:{port}"
    argv = [*ds, model, "--grpc", "--grpc-protocol", "openengine", "--host", "127.0.0.1"]
    argv += ["--port", str(port), "--tp_size", "2", *basic.ENGINE_ARGS, *basic.TOKENIZER_ARGS]
    env = {k: v for k, v in os.environ.items() if k != "TRTLLM_DISAGG_ROLE"}
    env["CUDA_VISIBLE_DEVICES"] = basic.ROLE_GPUS["context"]
    log_path = workdir / "engine.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(  # nosec B603
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    try:
        deadline = time.monotonic() + basic.NATIVE_READY_TIMEOUT_S
        while not _serving(endpoint):
            assert proc.poll() is None, f"the engine exited with {proc.returncode}; log {log_path}"
            assert time.monotonic() < deadline, f"the engine does not serve; log {log_path}"
            time.sleep(2)
        yield endpoint, proc
    finally:
        for signum in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signum)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=basic.NATIVE_STOP_TIMEOUT_S)


@pytest.fixture
def no_group(groups) -> None:
    """The module's engine group stopped, so that the test's own engine has the GPUs.

    Skipped in form K, which has no worker front end.
    """
    if groups.form != basic.FORM_CHAIN:
        pytest.skip("form K: the ds workers have no worker front end")
    groups.close()


def test_role_startup_check(no_group, model_path, tmp_path):
    """NT-139 (DG-84): a worker front end with a role refuses a ds engine without KV transfer.

    Started with ``--server_role context`` or ``--server_role generation`` in front of a ds
    engine started without a role, the front end exits at its startup check of the engine,
    naming what the role needs; the engine keeps serving and reports no Generate call. The test
    starts the engine and the front ends itself, after the fixture stopped the module's engine
    group.
    """
    with _engine_without_kv_transfer(model_path, tmp_path) as (endpoint, engine):
        for role in basic.ROLES:
            result = subprocess.run(  # nosec B603
                [
                    *_serve_command(),
                    model_path,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "0",
                    "--report_addr",
                    str(tmp_path / f"addr-{role}"),
                    "--openengine-endpoint",
                    endpoint,
                    "--server_role",
                    role,
                    *basic.TOKENIZER_ARGS,
                ],
                capture_output=True,
                text=True,
                timeout=FRONTEND_EXIT_TIMEOUT_S,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
            output = result.stdout + result.stderr
            assert result.returncode != 0, output[-4000:]
            assert "OpenEngine startup failed at step" in output, output[-4000:]
            assert f"a {role.upper()} worker needs" in output, output[-4000:]
        assert engine.poll() is None
        assert _serving(endpoint)
        # Read after both front ends exited, so one reading is final.
        assert _generate_received(endpoint) == 0


@pytest.fixture(scope="module")
def static_workers(groups, form, model_path, tmp_path_factory):
    """The two workers started without ``--server_role``: each engine has its role, the front ends none.

    Form K: the module's default group, whose ds proxy lists its workers by URL.
    """
    if form != basic.FORM_CHAIN:
        yield groups.get()
        return
    from .openai_server import RemoteOpenAIServer

    groups.close()
    workdir = tmp_path_factory.mktemp("ds-static")

    def worker(role: str) -> RemoteOpenAIServer:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": basic.ROLE_GPUS[role]}
        # The engine takes its role from the variable --server_role would set.
        env["TRTLLM_DISAGG_ROLE"] = role
        args = basic.worker_serve_args(role)
        assert args[:2] == ["--server_role", role], args
        return RemoteOpenAIServer(
            model_path,
            env=env,
            cli_args=args[2:],
            extra_config=basic.DEFAULT_GROUP.config(role),
            log_path=str(workdir / f"output_{role}.log"),
            wait=False,
        )

    with worker("context") as ctx, worker("generation") as gen:
        for server in (ctx, gen):
            server.wait_for_server(timeout=basic.NATIVE_READY_TIMEOUT_S)
        servers = {"context": ctx, "generation": gen}
        yield basic.Served(
            "",
            form=basic.FORM_CHAIN,
            workers={role: server.url_root for role, server in servers.items()},
            engines={
                role: server._chain_handle.engine_endpoint
                for role, server in servers.items()
                if server._chain_handle is not None
            },
            servers=servers,
        )


# What the refusal of a leg sent to the other role says: the ds server's, its OpenEngine
# validation's, or its engine's wording.
ROLE_MISMATCH = {
    "context_only": re.compile(
        r"without a context role|context_only request sent to a generation instance"
        r"|context_only request on a non-context instance"
    ),
    "generation_only": re.compile(
        r"without a generation role|generation_only request\) sent to a context instance"
        r"|generation_only request on a non-generation instance"
    ),
}


@pytest.mark.parametrize("case", ("gen_to_ctx", "ctx_to_gen"))
def test_role_mismatch_static_urls(static_workers, case):
    """NT-138 (DS-R01, DS-R03): a leg sent directly to the worker of the other role is answered 400.

    No orchestrator takes part: the test sends the leg itself. On the chain the worker front
    ends run without ``--server_role`` and each engine has its role; in form K the ds workers
    have theirs. A ``generation_only`` leg sent to the context worker, and a ``context_only``
    leg sent to the generation worker, are each answered 400 naming the role mismatch.
    """
    served = static_workers
    if case == "ctx_to_gen":
        response = _context_leg(served, url=served.workers["generation"])
        named = ROLE_MISMATCH["context_only"]
    else:
        # A handoff of the right shape that no context leg made: the role is checked first.
        unknown = next(_DISAGG_REQUEST_IDS)
        params = {
            "request_type": "generation_only",
            "first_gen_tokens": [1001],
            "ctx_request_id": unknown,
            "disagg_request_id": unknown,
            "ctx_info_endpoint": f"tcp://127.0.0.1:{basic.free_ports(1)[0]}",
            "ctx_dp_rank": 0,
        }
        body = {"model": MODEL_NAME, "prompt": [1000, 1001], "max_tokens": 4}
        response = _post(served.workers["context"], {**body, "disaggregated_params": params})
        named = ROLE_MISMATCH["generation_only"]
    assert response.status_code == 400, (response.status_code, response.text[:2000])
    assert named.search(_message(response)), response.text[:2000]
