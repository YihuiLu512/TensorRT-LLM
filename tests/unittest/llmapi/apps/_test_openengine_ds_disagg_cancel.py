# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: requests that end early, and engines that go away.

The forms, fixtures and environment are those of ``_test_openengine_ds_disagg_basic.py``. Each
test runs on an engine group of its own, since it cancels requests, lets parked KV time out or
kills an engine:

- NT-132: a request left while it waits, while its context leg prefills, or while its
  generation leg decodes, ends on its engine, and the engines serve the next request.
- NT-133: a handoff no generation leg takes is dropped after the context engine's parking
  timeout, and pulling it afterwards fails with an error.
- NT-134 (the chain only): a worker whose ds engine is killed stops its front end and leaves the
  orchestrator's cluster.

A test reads the ds engines' load and counters: on the chain over ``Control.GetLoad``, in form K
from the ds workers' ``/load`` and logs.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import random
import time
import urllib.parse
from pathlib import Path

import pytest
import requests

from . import _test_openengine_ds_disagg_basic as basic

# The chain harness, imported where a test reads an engine.
__extra_import_path__ = ["~/tests/integration/defs"]

MODEL_NAME = basic.MODEL_NAME
# The module's fixtures.
model_path = basic.model_path
form = basic.form
groups = basic.groups

pytestmark = [pytest.mark.threadleak(enabled=False)]

REQUEST_TIMEOUT_S = 600
# How long an engine may take to end a call whose client left (a worker front end checks for a
# gone client about once a second).
CANCEL_ARRIVES_S = 15.0
# Token ids inside the DeepSeek-V4.1 vocabulary, away from its special tokens.
SHORT_PROMPT = [1000 + i for i in range(64)]


def _disagg_request_id() -> int:
    """A ``disagg_request_id`` for a leg the test sends itself, apart from the orchestrator's."""
    from tensorrt_llm.grpc.openengine_conventions import DISAGG_REQUEST_ID_MIN

    return DISAGG_REQUEST_ID_MIN + random.randrange(1 << 40)


def _context_body(prompt: list[int]) -> dict:
    return {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 4,
        # A context leg whose first token is EOS would end without a handoff.
        "ignore_eos": True,
        "disaggregated_params": {
            "request_type": "context_only",
            "disagg_request_id": _disagg_request_id(),
        },
    }


def _context_url(served: basic.Served) -> str:
    return f"{served.workers['context']}/v1/completions"


def _context_leg(served: basic.Served, prompt: list[int]) -> tuple[int, dict]:
    """A context_only leg sent to the context worker itself.

    Returns:
        Its ``disagg_request_id``, and its answer, which holds the handoff.
    """
    body = _context_body(prompt)
    response = requests.post(_context_url(served), json=body, timeout=REQUEST_TIMEOUT_S)
    assert response.status_code == 200, response.text
    return body["disaggregated_params"]["disagg_request_id"], response.json()


def _generation_leg(
    served: basic.Served, prompt: list[int], context: dict, timeout: float = REQUEST_TIMEOUT_S
) -> requests.Response:
    """The generation_only leg of the context leg answer ``context``, sent to the generation worker."""
    handoff = {**context["choices"][0]["disaggregated_params"], "request_type": "generation_only"}
    body = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 4,
        "disaggregated_params": handoff,
    }
    url = f"{served.workers['generation']}/v1/completions"
    return requests.post(url, json=body, timeout=timeout)


@contextlib.contextmanager
def _sent(url: str, body: dict):
    """Send ``body`` to ``url`` and give the connection, unread; closing it leaves the request.

    The connection is closed at the end in any case, so that a failed test leaves no request
    behind.
    """
    parts = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=REQUEST_TIMEOUT_S)
    try:
        connection.request(
            "POST", parts.path, body=json.dumps(body), headers={"Content-Type": "application/json"}
        )
        yield connection
    finally:
        connection.close()


def _wait(condition, what: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out after {timeout} s waiting for {what}"
        time.sleep(0.2)


def _pending(served: basic.Served, role: str) -> tuple[int, int]:
    """The ``role`` ds engine's requests running and waiting (the chain: received and not ended)."""
    if served.chain:
        from openengine_chain import direct_replay

        load = direct_replay.get_load(served.engines[role])
        return int(load.get("running_requests", 0)), int(load.get("queued_requests", 0))
    load = requests.get(f"{served.workers[role]}/load", timeout=30).json()
    return int(load.get("running", 0)), int(load.get("waiting", 0))


def _handoffs(served: basic.Served) -> int | None:
    """The PrefillReady events the context engine sent (the chain only)."""
    from tensorrt_llm.grpc.openengine_conventions import COUNTER_PREFILL_READY_SENT

    if not served.chain:
        return None
    return basic.engine_counter(served.engines["context"], COUNTER_PREFILL_READY_SENT)


def _engine_log(served: basic.Served, role: str) -> Path:
    """The log of the ``role`` ds engine (form K: the ds worker)."""
    if served.chain:
        return Path(served.servers[role]._chain_handle.engine_log)
    return served.native.log(role)


def _log_count(served: basic.Served, role: str, text: str) -> int:
    return _engine_log(served, role).read_text(errors="replace").count(text)


def _serves(served: basic.Served) -> None:
    """A request through the orchestrator (form K: the ds proxy) still completes."""
    text, finish, _ = basic.generate(served, "completions", False, max_tokens=4)
    assert text and finish in ("stop", "length"), (text, finish)


def _engine_args(**values) -> tuple[str, ...]:
    """:data:`basic.ENGINE_ARGS` with the options named in ``values`` set to them."""
    args = list(basic.ENGINE_ARGS)
    for name, value in values.items():
        args[args.index(f"--{name}") + 1] = str(value)
    return tuple(args)


# Each attention-DP rank of the context engine runs one request at a time and prefills in
# chunks of one KV-cache page, so that a LONG_PROMPT keeps it busy for seconds.
CANCEL_GROUP = basic.Group(engine_args=_engine_args(max_batch_size=1, max_num_tokens=128))
LONG_PROMPT_TOKENS = 30000
# A generation leg this long decodes for far longer than a cancel takes to arrive.
LONG_DECODE = 4000
# Long context legs sent at most until one waits: the context engine runs one per rank.
MAX_LONG_LEGS = 4
PREFILL_STARTS_S = 120
DECODE_STARTS_S = 120
# What a ds worker of form K logs when the client of a request it runs has left.
LEFT_LOG = "is disconnected, abort"


def _long_prompt(n: int) -> list[int]:
    """The ``n``-th of distinct long prompts, which share no prefix the engine could reuse."""
    return [1000 + (n * 7919 + i) % 50000 for i in range(LONG_PROMPT_TOKENS)]


def _total(served: basic.Served, role: str) -> int:
    return sum(_pending(served, role))


class _Left:
    """Checks that the context legs the test leaves end without a handoff.

    On the chain the context engine's PrefillReady count stays where it was; in form K the ds
    context worker logs each request whose client left.
    """

    def __init__(self, served: basic.Served) -> None:
        self.served = served
        self.handoffs = _handoffs(served)
        self.logged = None if served.chain else _log_count(served, "context", LEFT_LOG)

    def handed_off(self) -> bool:
        return self.served.chain and _handoffs(self.served) != self.handoffs

    def check(self, what: str) -> None:
        assert not self.handed_off(), f"a long prefill ended {what}; the prompt is too short here"

    def ended(self, left: int) -> None:
        """``left`` legs have been left; each ended on the context engine without a handoff."""
        _wait(
            lambda: _total(self.served, "context") == 0,
            "the left context calls to end",
            CANCEL_ARRIVES_S,
        )
        self.check("after its client left")
        if not self.served.chain:
            _wait(
                lambda: _log_count(self.served, "context", LEFT_LOG) >= self.logged + left,
                f"the context worker to log {left} left requests",
                CANCEL_ARRIVES_S,
            )


def _leave_waiting(served: basic.Served) -> None:
    """Leave a context leg that waits behind long prefills, then the prefilling legs too."""
    left = _Left(served)
    with contextlib.ExitStack() as stack:
        legs = []
        while True:
            assert len(legs) < MAX_LONG_LEGS, "no context leg waited behind the long prefills"
            body = _context_body(_long_prompt(len(legs)))
            legs.append(stack.enter_context(_sent(_context_url(served), body)))
            _wait(
                lambda: _total(served, "context") >= len(legs) or left.handed_off(),
                f"{len(legs)} context calls",
                PREFILL_STARTS_S,
            )
            left.check("before a leg waited")
            if _pending(served, "context")[1]:
                break
        legs[-1].close()
        _wait(
            lambda: _total(served, "context") == len(legs) - 1,
            "the waiting call to end",
            CANCEL_ARRIVES_S,
        )
        left.check("before its client left")
    left.ended(len(legs))


def _leave_prefilling(served: basic.Served) -> None:
    """Leave a context leg while it prefills."""
    left = _Left(served)
    with _sent(_context_url(served), _context_body(_long_prompt(0))) as leg:
        _wait(
            lambda: _total(served, "context") >= 1 or left.handed_off(),
            "the prefill to start",
            PREFILL_STARTS_S,
        )
        left.check("before its client left")
        leg.close()
    left.ended(1)


def _leave_decoding(served: basic.Served) -> None:
    """Stream a request through the orchestrator and leave once its generation leg decodes."""
    body = {
        "model": MODEL_NAME,
        "prompt": basic.COMPLETION_PROMPT,
        "max_tokens": LONG_DECODE,
        "ignore_eos": True,
        "stream": True,
    }
    with requests.post(
        f"{served.url}/v1/completions",
        json=body,
        headers={"Connection": "close"},
        timeout=REQUEST_TIMEOUT_S,
        stream=True,
    ) as response:
        assert response.status_code == 200, response.text
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: ") and line != "data: [DONE]":
                choices = json.loads(line[len("data: ") :]).get("choices") or [{}]
                if choices[0].get("text"):
                    break
        else:
            raise AssertionError("the stream ended before any text")
        _wait(lambda: _total(served, "generation") == 1, "the decode to run", DECODE_STARTS_S)
    _wait(lambda: _total(served, "generation") == 0, "the decode call to end", CANCEL_ARRIVES_S)


@pytest.fixture
def cancel_served(groups) -> basic.Served:
    return groups.get(CANCEL_GROUP)


@pytest.mark.parametrize("stage", ["queued", "prefill", "decode"])
def test_abort_stages(cancel_served, stage):
    """NT-132 (DG-38, DG-39c, DS-S11, DS-S12): a request left in any stage ends on its engine.

    ``queued`` and ``prefill`` send context legs to the context worker itself and leave one
    while it waits behind long prefills or while it prefills: its call ends on the context
    engine before a handoff. ``decode`` streams a request through the orchestrator (form K: the
    ds proxy) and leaves once its generation leg decodes: the generation call ends long before
    its last token. Then the engines serve a request through the orchestrator.
    """
    served = cancel_served
    {"queued": _leave_waiting, "prefill": _leave_prefilling, "decode": _leave_decoding}[stage](
        served
    )
    _serves(served)


# The context engine drops a parked KV no generation leg pulled within this many seconds.
PARK_TIMEOUT_S = 15
PARKED_LEGS = 2
DROP_LOGGED_S = PARK_TIMEOUT_S + 60


def _dropped_log(disagg_request_id: int) -> str:
    """What the context engine logs when it drops the parked KV of ``disagg_request_id``."""
    return f"context_only request {disagg_request_id}: KV not pulled within"


def _orphans(served: basic.Served) -> int:
    """Handoffs the context engine sent that the generation engine has not taken (the chain)."""
    from tensorrt_llm.grpc.openengine_conventions import (
        BUCKET_GENERATION_ONLY,
        COUNTER_GENERATE_ACCEPTED_BUCKETS,
    )

    taken = COUNTER_GENERATE_ACCEPTED_BUCKETS[BUCKET_GENERATION_ONLY]
    return _handoffs(served) - basic.engine_counter(served.engines["generation"], taken)


def _abort(endpoint: str, request_id: str) -> str:
    """``Control.Abort(request_id)`` on the engine at ``endpoint``: the status's name."""
    import grpc

    from tensorrt_llm.grpc.openengine.bindings import lifecycle_pb2, openengine_pb2_grpc

    with grpc.insecure_channel(endpoint) as channel:
        reply = openengine_pb2_grpc.ControlStub(channel).Abort(
            lifecycle_pb2.AbortRequest(request_id=request_id), timeout=60
        )
    return lifecycle_pb2.AbortStatus.Name(reply.status)


def _context_request_ids(directory: Path) -> dict[str, str]:
    """The GenerateRequest ids of the context legs a front end wrote to ``directory``, by
    ``disagg_request_id``."""
    found = {}
    for path in directory.glob("*.json"):
        extra = json.loads(path.read_text()).get("extra", {})
        if extra.get("request_type") == "context_only":
            found[str(extra.get("disagg_request_id"))] = json.loads(path.read_text())["request_id"]
    return found


@pytest.fixture
def parked_served(groups) -> tuple[basic.Served, Path]:
    """The group whose context engine parks KV for PARK_TIMEOUT_S, and where its front end
    writes the GenerateRequests it sends."""
    from tensorrt_llm.grpc.openengine_conventions import SERVE_REQUEST_DUMP_DIR_ENV

    dumps = groups.workdir / "parked_kv_requests"
    dumps.mkdir(parents=True, exist_ok=True)
    env = {"DSV4_KV_PARK_TIMEOUT": str(PARK_TIMEOUT_S), SERVE_REQUEST_DUMP_DIR_ENV: str(dumps)}
    return groups.get(basic.Group(env={"context": env})), dumps


def test_parked_kv_timeout_and_orphans(parked_served, record_property):
    """NT-133 (DG-90, DS-S13, DS-S20a, DS-I05): an unclaimed handoff is dropped after its timeout.

    The test sends context legs to the context worker itself and no generation leg; their KV
    stays parked on the context engine, whose parking timeout is short here. On the chain the
    engines' counters show one more orphan handoff per leg, and aborting a parked request, or a
    request id the engines do not know, is no error. Once the timeout passed, the context
    engine logs each KV as dropped, and a generation leg that pulls one fails naming it. Then
    the engines serve a request through the orchestrator.
    """
    served, dumps = parked_served
    if served.chain:
        handed, orphans = _handoffs(served), _orphans(served)
    legs = dict(_context_leg(served, SHORT_PROMPT) for _ in range(PARKED_LEGS))

    if served.chain:
        assert _handoffs(served) - handed == PARKED_LEGS
        assert _orphans(served) - orphans == PARKED_LEGS
        request_ids = _context_request_ids(dumps)
        assert {str(did) for did in legs} <= set(request_ids), (legs, request_ids)
        for did in legs:
            status = _abort(served.engines["context"], request_ids[str(did)])
            assert status == "ABORT_STATUS_ALREADY_FINISHED", (did, status)
        unknown = f"nt133-unknown-{random.randrange(1 << 32)}"
        for role in basic.ROLES:
            status = _abort(served.engines[role], unknown)
            assert status == "ABORT_STATUS_ALREADY_FINISHED", (role, status)
    else:
        record_property("ds_i05", "form K cancels only by leaving: a parked request has no client")

    _wait(
        lambda: all(_log_count(served, "context", _dropped_log(did)) for did in legs),
        "the context engine to drop the parked KV",
        DROP_LOGGED_S,
    )
    response = _generation_leg(served, SHORT_PROMPT, next(iter(legs.values())))
    assert response.status_code >= 400, (response.status_code, response.text)
    assert "parked" in response.text.lower(), response.text
    _serves(served)


# A front end gives up on its engine after its engine-lost timeout (30 s by default).
FRONTEND_STOP_S = 180
# Workers beat every 5 s and leave the cluster after 10 s without one (the defaults).
DROP_S = 60
# How long the log helper may take to write a front end's last lines once it exits.
LOG_FLUSH_S = 30
# What a front end logs when it gives up on its engine.
LOST_LOG = "stopping this frontend"
# A generation engine retries connecting to a context engine for 30 s, twice.
PULL_FAILS_S = 180


def _listed(served: basic.Served, role: str) -> bool | None:
    """Whether the orchestrator lists the ``role`` worker; None if it did not answer."""
    port = served.servers[role].port
    try:
        lists = requests.get(f"{served.url}/cluster_info", timeout=10).json()
        addresses = lists["server_lists"][role]
    except (requests.RequestException, ValueError, KeyError):
        return None
    return any(str(address).endswith(f":{port}") for address in addresses)


@pytest.fixture
def loss_served(groups) -> basic.Served:
    """A default group of its own: the group that is up stops first, whatever it is."""
    if groups.form != basic.FORM_CHAIN:
        pytest.skip("form K: the ds worker serves HTTP from the engine's own process")
    groups.close()
    return groups.get(basic.Group())


@pytest.mark.parametrize("role", ["context", "generation"], ids=["ctx", "gen"])
def test_engine_loss(loss_served, role):
    """NT-134 (DG-87, DS-S13): a worker whose ds engine exits stops, and the orchestrator drops it.

    The test kills the ``role`` ds engine (SIGKILL to the process that serves its gRPC): its
    worker front end reports the engine lost and exits, and the orchestrator no longer lists the
    worker, while the other worker still runs and is listed. Each run starts an engine group of
    its own.
    """
    served = loss_served
    other = "generation" if role == "context" else "context"
    _serves(served)
    assert _listed(served, role) is True
    handle = served.servers[role]._chain_handle
    handle.engine.kill()
    _wait(lambda: handle.frontend.poll() is not None, "the front end to stop", FRONTEND_STOP_S)
    log = Path(handle.frontend_log)
    _wait(lambda: LOST_LOG in log.read_text(errors="replace"), "the front end's log", LOG_FLUSH_S)
    _wait(lambda: _listed(served, role) is False, "the worker to be dropped", DROP_S)
    # The other worker is not affected.
    assert served.servers[other]._chain_handle.frontend.poll() is None
    assert requests.get(f"{served.workers[other]}/health", timeout=10).status_code == 200
    assert _listed(served, other) is True


def test_engine_loss_pull_from_lost_context(loss_served, record_property):
    """NT-134 (DS-S13): a generation leg whose KV is parked on a killed context engine fails.

    The test sends both legs to the worker front ends itself: a context leg parks its KV, the
    context ds engine is killed, and the generation leg that pulls that KV ends with an error
    instead of hanging.
    """
    served = loss_served
    parked = _context_leg(served, SHORT_PROMPT)[1]
    served.servers["context"]._chain_handle.engine.kill()
    start = time.monotonic()
    response = _generation_leg(served, SHORT_PROMPT, parked, timeout=PULL_FAILS_S)
    record_property("pull_from_lost_context_s", f"{time.monotonic() - start:.1f}")
    assert response.status_code >= 400, (response.status_code, response.text)
