# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on case-03, the ds-engine: the orchestrator and its workers.

Service discovery, sub-agent affinity, perf records and request ids, the two legs' requests, the
orchestrator's own switches, and a context leg retried after a worker front end dies. The forms,
fixtures and environment are those of ``_test_openengine_ds_disagg_basic.py``. Every test here
runs on the chain only: form K's ds proxy has no worker front ends to register, sign or record.

A test that changes the configuration runs on an engine group of its own, which a fixture
starts: groups change during a test's setup only, so that the chain proof of the test reads the
servers the test ran on from start to end. The two legs' requests are read from what each worker
front end sends its engine (``TRTLLM_OPENENGINE_REQUEST_DUMP_DIR``): the ds engine writes no
per-request record itself. NT-215 kills a worker front end, so it goes last; the module's
``groups`` fixture stops its group.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests

from . import _test_openengine_ds_disagg_basic as basic

# The chain harness, imported where a chain starts.
__extra_import_path__ = ["~/tests/integration/defs"]

MODEL_NAME = basic.MODEL_NAME
# The module's fixtures.
model_path = basic.model_path
form = basic.form
groups = basic.groups

pytestmark = [pytest.mark.threadleak(enabled=False)]

ROLES = basic.ROLES
REQUEST_TIMEOUT_S = 300
RECORD_WAIT_S = 30
# Logged by a worker front end when the orchestrator sets its clock offset.
OFFSET_LOG = "The steady clock offset between local and disagg server"
FORM_K_REASON = "the ds proxy has no worker front ends"


def _chain(groups, group: basic.Group = basic.DEFAULT_GROUP) -> basic.Served:
    """``group`` up on the chain; skipped in form K before any group starts. For fixtures only."""
    if groups.form != basic.FORM_CHAIN:
        pytest.skip(f"form K: {FORM_K_REASON}")
    return groups.get(group)


@contextlib.contextmanager
def _ds_disagg_chain(
    model: str,
    workdir: Path,
    group: basic.Group,
    cluster: dict,
    orchestrator_env: dict | None = None,
    reserved: tuple[int, ...] = (),
):
    """:func:`basic.ds_disagg_chain` with the workers registering at ``cluster``.

    ``orchestrator_env`` adds to the orchestrator's environment; ``reserved`` are further ports the
    harness must not give an engine.
    """
    from openengine_chain import launch

    from .openai_server import RemoteDisaggOpenAIServer, RemoteOpenAIServer

    workdir.mkdir(parents=True, exist_ok=True)
    disagg_port, ctx_port, gen_port = basic.free_ports_besides(3, reserved)
    launch.reserve_ports(disagg_port, ctx_port, gen_port, *reserved)
    cluster = {"cluster_name": "", **cluster}
    cluster.setdefault("cluster_uri", f"http://localhost:{disagg_port}")

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
    with worker("context", ctx_port) as ctx, worker("generation", gen_port) as gen:
        with RemoteDisaggOpenAIServer(
            ctx_servers=[],
            gen_servers=[],
            port=disagg_port,
            env={**os.environ, **(orchestrator_env or {})},
            disagg_config=front,
            log_path=str(workdir / "output_orchestrator.log"),
            # Its /health answers 503 until both workers register, and the server's own wait for
            # it has no time limit: the workers are waited for first, then its /health.
            wait_ready=False,
        ) as orchestrator:
            for server in (ctx, gen):
                server.wait_for_server(timeout=basic.NATIVE_READY_TIMEOUT_S)
            url = f"http://localhost:{disagg_port}"
            basic.wait_registered(url, {"context": 1, "generation": 1})
            basic.wait_healthy(url)
            engines = {
                role: server._chain_handle.engine_endpoint
                for role, server in (("context", ctx), ("generation", gen))
                if server._chain_handle is not None
            }
            yield basic.Served(
                url,
                engines.get("context"),
                basic.FORM_CHAIN,
                workers={"context": ctx.url_root, "generation": gen.url_root},
                engines=engines,
                servers={"context": ctx, "generation": gen, "orchestrator": orchestrator},
            )


def _frontend_log_text(server) -> str:
    """What a worker front end has logged: the harness's front-end log and the server's own."""
    handle = getattr(server, "_chain_handle", None)
    paths = [handle.frontend_log if handle is not None else None, server.log_path]
    text = ""
    for path in paths:
        if path and os.path.exists(path):
            text += Path(path).read_text(errors="replace")
    return text


def _completion(served: basic.Served, prompt: str = basic.COMPLETION_PROMPT, **fields):
    body = {"model": MODEL_NAME, "prompt": prompt, "max_tokens": 8, "temperature": 0.0, **fields}
    return requests.post(f"{served.url}/v1/completions", json=body, timeout=REQUEST_TIMEOUT_S)


def _chat(served: basic.Served, messages, headers=None, **fields):
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 16,
        "temperature": 0.0,
        **basic.NO_THINKING,
        **fields,
    }
    return requests.post(
        f"{served.url}/v1/chat/completions", json=body, headers=headers, timeout=REQUEST_TIMEOUT_S
    )


def _orchestrator_counter(served: basic.Served, role: str, name: str) -> int:
    """The orchestrator's ``<ctx|gen>_<name>_total`` counter, summed over its labels.

    The orchestrator creates its counters with its clients, at zero: fails when it reports none.
    """
    text = requests.get(f"{served.url}/prometheus/metrics", timeout=30).text
    pattern = re.compile(rf"^{role}_{name}_total(?:\{{[^}}]*\}})?\s+(\S+)\s*$", re.MULTILINE)
    values = pattern.findall(text)
    assert values, f"the orchestrator reports no {role}_{name}_total"
    return int(sum(float(value) for value in values))


def _marked_roots(root_pid: int, marker: str) -> list:
    """The processes below ``root_pid`` that a parent started with ``marker`` in their environment.

    A process's children inherit the variable, so only the processes whose parent lacks it count:
    the roots of the marked subtrees.
    """
    import psutil

    def marked(process) -> bool:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return marker in process.environ()
        return False

    roots = []
    for child in psutil.Process(root_pid).children(recursive=True):
        try:
            parent = child.parent()
        except psutil.NoSuchProcess:
            continue
        if marked(child) and (parent is None or not marked(parent)):
            roots.append(child)
    return roots


def _describe(processes) -> list[str]:
    """``pid: cmdline`` of each process, for an assertion's message."""
    import psutil

    described = []
    for process in processes:
        try:
            described.append(f"{process.pid}: {' '.join(process.cmdline())}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            described.append(f"{process.pid}: (gone)")
    return described


# -- GenerateRequests the worker front ends sent (TRTLLM_OPENENGINE_REQUEST_DUMP_DIR) --------


def _dump_env(root: Path) -> dict[str, dict[str, str]]:
    """Each role's front-end environment that writes its GenerateRequests under ``root``."""
    from tensorrt_llm.grpc.openengine_conventions import SERVE_REQUEST_DUMP_DIR_ENV

    return {role: {SERVE_REQUEST_DUMP_DIR_ENV: str(root / role)} for role in ROLES}


def _dumps(directory: Path) -> dict[str, dict]:
    """The GenerateRequests written to ``directory``, by file name."""
    if not directory.is_dir():
        return {}
    return {path.name: json.loads(path.read_text()) for path in directory.glob("*.json")}


def _new_dumps(directory: Path, seen: dict[str, dict], count: int) -> list[dict]:
    """The GenerateRequests written to ``directory`` since ``seen``, once there are ``count``."""
    deadline = time.monotonic() + RECORD_WAIT_S
    while True:
        new = [dump for name, dump in _dumps(directory).items() if name not in seen]
        if len(new) >= count or time.monotonic() > deadline:
            return new
        time.sleep(0.2)


def _ctx_id(dump: dict) -> str | None:
    """The disagg_request_id a context leg's GenerateRequest carries."""
    from tensorrt_llm.grpc.openengine_conventions import REQ_DISAGG_REQUEST_ID

    value = dump.get("extra", {}).get(REQ_DISAGG_REQUEST_ID)
    return None if value is None else str(value)


def _gen_session(dump: dict) -> dict:
    """The kv.session a generation leg's GenerateRequest carries."""
    return dump.get("kv", {}).get("session", {})


def _gen_id(dump: dict) -> str | None:
    """The disagg_request_id of the hand-off a generation leg's GenerateRequest takes."""
    value = _gen_session(dump).get("attributes_struct", {}).get("disagg_request_id")
    return None if value is None else str(value)


# -- NT-142: service discovery --------------------------------------------------------------

ETCD_READY_TIMEOUT_S = 60


@contextlib.contextmanager
def _etcd(etcd: str, workdir: Path):
    """A single-member etcd on free loopback ports, as its ``etcd://`` URI."""
    client, peer = basic.free_ports(2)
    workdir.mkdir(parents=True, exist_ok=True)
    client_url, peer_url = f"http://127.0.0.1:{client}", f"http://127.0.0.1:{peer}"
    cmd = [
        etcd,
        "--name",
        "ds-cluster",
        "--data-dir",
        str(workdir / "etcd-data"),
        "--listen-client-urls",
        client_url,
        "--advertise-client-urls",
        client_url,
        "--listen-peer-urls",
        peer_url,
        "--initial-advertise-peer-urls",
        peer_url,
        "--initial-cluster",
        f"ds-cluster={peer_url}",
    ]
    with open(workdir / "etcd.log", "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + ETCD_READY_TIMEOUT_S
        while True:
            with contextlib.suppress(requests.RequestException):
                if requests.get(f"{client_url}/health", timeout=5).status_code == 200:
                    break
            assert proc.poll() is None, f"etcd exited with {proc.returncode}; log in {workdir}"
            assert time.monotonic() < deadline, f"etcd is not healthy; log in {workdir}"
            time.sleep(0.5)
        yield f"etcd://127.0.0.1:{client}", (client, peer)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _clock_offset_evidence(served: basic.Served, role: str) -> bool:
    """Whether the ``role`` worker took the orchestrator's clock offset (``True``), or the
    orchestrator logged that it discarded the offset as too uncertain to apply (``False``).

    Fails when neither shows within :data:`RECORD_WAIT_S`.
    """
    port = served.servers[role].port
    discarded = re.compile(rf"Steady clock handshake with \S*:{port}/\S* was too slow")
    orchestrator_log = Path(served.servers["orchestrator"].log_path)
    deadline = time.monotonic() + RECORD_WAIT_S
    while True:
        if OFFSET_LOG in _frontend_log_text(served.servers[role]):
            return True
        if orchestrator_log.exists() and discarded.search(
            orchestrator_log.read_text(errors="replace")
        ):
            return False
        assert time.monotonic() < deadline, f"the {role} front end logged no clock offset"
        time.sleep(0.5)


def _check_discovery(served: basic.Served, record_property) -> None:
    """Both workers are listed and serve a request; the clock offset of each (DG-54)."""
    info = requests.get(f"{served.url}/cluster_info", timeout=30).json()
    assert info.get("is_ready") is True, info
    for role in ROLES:
        (address,) = info["server_lists"][role]
        assert address.endswith(f":{served.servers[role].port}"), (role, info)
    text, finish, _ = basic.generate(served, "completions", False, max_tokens=8)
    assert "Paris" in text and finish in ("stop", "length"), (text, finish)
    # DG-54: the orchestrator probed each worker's clock and wrote the offset to it, unless the
    # best round trip was too slow for the offset to be applied.
    for role in ROLES:
        response = requests.get(f"{served.workers[role]}/steady_clock_offset", timeout=60)
        assert response.status_code == 200, (role, response.text)
        body = response.json()
        assert 0 < body["receive_ts"] <= body["transmit_ts"], (role, body)
        if not _clock_offset_evidence(served, role):
            record_property(
                f"dg54_no_offset_evidence_{role}",
                "the orchestrator discarded the clock offset as too slow; none was written",
            )


@pytest.fixture
def discovery_served(request, groups):
    """The chain registering through ``http`` (the module's default group) or an ``etcd``."""
    if groups.form != basic.FORM_CHAIN:
        pytest.skip(f"form K: {FORM_K_REASON}")
    if request.param == "http":
        yield groups.get()
        return
    etcd = shutil.which("etcd")
    assert etcd is not None, "no etcd executable on PATH: DG-29 needs one"
    groups.close()
    workdir = groups.workdir / "etcd"
    with _etcd(etcd, workdir) as (uri, ports):
        cluster = {"cluster_uri": uri, "cluster_name": "ds-cluster"}
        with _ds_disagg_chain(
            groups.model, workdir / "chain", basic.DEFAULT_GROUP, cluster, reserved=ports
        ) as served:
            yield served


@pytest.mark.parametrize("discovery_served", ["etcd", "http"], indirect=True)
def test_service_discovery(discovery_served, record_property):
    """NT-142 (DG-28, DG-29, DG-32, DG-54): the worker front ends register through the cluster.

    ``http``: the orchestrator's own HTTP cluster storage, the module's default (every other test
    starts its workers this way too); ``etcd``: an etcd the test starts. Both workers are listed,
    a request through the orchestrator succeeds, and each worker answers ``/steady_clock_offset``.
    Each worker logged the offset the orchestrator wrote to it, except when the orchestrator
    logged that it discarded the offset (best round trip too slow): then the test records
    ``dg54_no_offset_evidence_<role>`` instead.
    """
    _check_discovery(discovery_served, record_property)


# -- NT-146: the orchestrator's switches -----------------------------------------------------

# The orchestrator's counterparts of the ds proxy's options: gen_strip_message_history,
# gen_tokids_ctxbytes and its number of proxy processes.
SWITCHES = {
    "gen_strip_history": {"gen_strip_message_history": True},
    "tokids_ctxbytes": {"gen_tokids_ctxbytes": True},
    "num_workers": {"num_workers": 2},
}
# A conversation of several turns: gen_strip_message_history has a history to strip.
CONVERSATION = [
    {"role": "user", "content": "Remember the number 17."},
    {"role": "assistant", "content": "I will remember 17."},
    {"role": "user", "content": "What is the capital of France? Answer with one word."},
]


def _alignment_outputs(served: basic.Served) -> dict[str, str]:
    """Greedy outputs of completions and of a multi-turn chat, streamed and not, sent one by one."""
    outputs = {}
    for stream in (False, True):
        key = "stream" if stream else "non_stream"
        outputs[f"completions-{key}"] = basic.generate(
            served, "completions", stream, max_tokens=16
        )[0]
        outputs[f"chat-{key}"] = basic.generate(
            served, "chat", stream, messages=CONVERSATION, max_tokens=16
        )[0]
    return outputs


@pytest.fixture(scope="module")
def alignment_baseline(groups) -> dict[str, str]:
    """The outputs on the module's default engine group, the switches off."""
    return _alignment_outputs(_chain(groups))


@pytest.fixture
def switch_served(request, groups, alignment_baseline):
    """The chain with one switch on, and the switch; the baseline is read first, on the default
    group."""
    if groups.form != basic.FORM_CHAIN:
        pytest.skip(f"form K: {FORM_K_REASON}")
    front = SWITCHES[request.param]
    if request.param != "num_workers":
        yield _chain(groups, basic.Group(front=front)), request.param
        return
    # The fleet's coordinator serves the cluster storage on a port of its own.
    groups.close()
    coordinator_port = basic.free_ports(1)[0]
    cluster = {"cluster_uri": f"http://localhost:{coordinator_port}"}
    env = {"TRTLLM_DISAGG_COORDINATOR_PORT": str(coordinator_port)}
    with _ds_disagg_chain(
        groups.model,
        groups.workdir / "num_workers",
        basic.Group(front=front),
        cluster,
        orchestrator_env=env,
        reserved=(coordinator_port,),
    ) as served:
        yield served, request.param


def _fleet_processes(orchestrator_pid: int) -> list:
    """The orchestrator's fleet processes (``num_workers``), by their per-process index."""
    # serve.DisaggWorkerEnvs.TLLM_DISAGG_WORKER_PROCESS_ID
    return _marked_roots(orchestrator_pid, "TRTLLM_DISAGG_WORKER_PROCESS_ID")


@pytest.mark.parametrize("switch_served", list(SWITCHES), indirect=True)
def test_orchestrator_alignment_switches(alignment_baseline, switch_served):
    """NT-146 (DS-O01, DS-O02, DS-O08): the orchestrator serves with each ds proxy option's twin.

    The orchestrator is configured with ``gen_strip_message_history``, ``gen_tokids_ctxbytes`` or
    ``num_workers: 2``. Requests succeed, and their greedy outputs, sent one at a time, are those
    of the default group, where the options are off. With ``num_workers`` the orchestrator runs
    two fleet processes. The test does not observe what the first two options change on the way
    to the generation worker.
    """
    served, switch = switch_served
    outputs = _alignment_outputs(served)
    assert outputs == alignment_baseline, (outputs, alignment_baseline)
    if switch == "num_workers":
        fleet = _fleet_processes(served.servers["orchestrator"].proc.pid)
        assert len(fleet) == 2, _describe(fleet)


# -- NT-143: sub-agent affinity -------------------------------------------------------------

# The orchestrator signs the parent-session hint with it, each worker front end re-signs it for
# its engine, and the context engine verifies it.
AFFINITY_KEY = secrets.token_hex(32)
PARENT_HEADER = "X-Dynamo-Parent-Session-ID"
SESSION_HEADER = "x-session-id"
SUBAGENTS = 6
# The context engine's log line of its conversation statistics (DSV4_SCHED_PROF).
CONVERSATION_STATS = re.compile(
    r"attention-DP conversations: (\d+) admissions of a conversation seen before, (\d+) of them "
    r"on another rank; (\d+) pinned"
)


@pytest.fixture(scope="module")
def dump_root(groups) -> Path:
    """Where this file's worker front ends write their GenerateRequests, one directory per group."""
    root = groups.workdir / "requests"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _affinity_group(scope: str) -> basic.Group:
    """Sub-agent affinity on the context leg (``scope`` ``context``) or on both legs (``both``).

    The context engine places a request by its conversation (``DSV4_ADP_CONV_PIN``, with the
    module's attention DP, conversation affinity and block reuse), so it declares sub-agent
    affinity, and logs its conversation statistics every scheduling step (``DSV4_SCHED_PROF``).
    """
    conversation = {"router": {"type": "conversation"}}
    front = {
        "internal_request_auth_key": AFFINITY_KEY,
        "conversation_affinity_header_for_subagents": PARENT_HEADER,
        "subagent_affinity_scope": scope,
        "context_servers": conversation,
    }
    if scope == "both":
        front["generation_servers"] = conversation
    worker = {"internal_request_auth_key": AFFINITY_KEY}
    env = {"context": {"DSV4_ADP_CONV_PIN": "1", "DSV4_SCHED_PROF": "1"}}
    return basic.Group(context=worker, generation=worker, env=env, front=front)


@pytest.fixture
def affinity_served(request, groups) -> basic.Served:
    """The chain with sub-agent affinity of scope ``request.param``."""
    return _chain(groups, _affinity_group(request.param))


def _conversation_stats(path: Path) -> tuple[int, int, int] | None:
    """The context engine's last logged conversation statistics: seen again, moved, pinned.

    ``None`` when it logged none yet; it logs them once it has seen a conversation again.
    """
    found = CONVERSATION_STATS.findall(path.read_text(errors="replace")) if path.exists() else []
    return tuple(int(n) for n in found[-1]) if found else None


@pytest.mark.parametrize("affinity_served", ["context", "both"], indirect=True)
def test_subagent_affinity(affinity_served, record_property):
    """NT-143 (DG-22, DG-35, DS-O06, DS-O07): a sub-agent's context leg joins its parent's.

    The context engine declares sub-agent affinity. A parent and then sub-agents, each with a
    conversation of its own and the parent's session in the parent header, all succeed. The
    context engine admits every sub-agent's request as one of a conversation it has seen before,
    on that conversation's rank: the parent's session, verified and taken as the request's
    conversation (``params["conv"]``); without it each sub-agent's conversation is new. With
    ``both`` the generation leg carries the hint as well and the requests still succeed; the ds
    engine refuses the hint on a generation request, and the generation engine does not declare
    sub-agent affinity (recorded), so this does not tell which of the two keeps it away.
    """
    import uuid

    served = affinity_served
    server_extra, _ = basic.engine_extras(served.engines["context"])
    assert server_extra.get("trtllm_supports_subagent_affinity") is True, server_extra
    gen_extra, _ = basic.engine_extras(served.engines["generation"])
    record_property(
        "gen_declares_subagent_affinity", str(gen_extra.get("trtllm_supports_subagent_affinity"))
    )
    log = Path(served.servers["context"]._chain_handle.engine_log)
    before = _conversation_stats(log)

    parent = f"nt143-parent-{uuid.uuid4().hex}"
    sessions = [(parent, None)] + [(f"{parent}-child-{i}", parent) for i in range(SUBAGENTS)]
    for session, parent_session in sessions:
        headers = {SESSION_HEADER: session}
        if parent_session is not None:
            headers[PARENT_HEADER] = parent_session
        question = f"Agent {session[-8:]}: name one prime number above {len(session) * 7}."
        response = _chat(
            served,
            [{"role": "user", "content": question}],
            headers=headers,
            conversation_params={"conversation_id": session},
        )
        assert response.status_code == 200, (session, response.text)
        (choice,) = response.json()["choices"]
        assert choice["message"]["content"], (session, choice)

    # Statistics not logged before these requests count from zero; after them they must be.
    start = before or (0, 0, 0)
    deadline = time.monotonic() + RECORD_WAIT_S
    while True:
        after = _conversation_stats(log)
        if after is not None:
            seen, moved, pinned = (a - b for a, b in zip(after, start))
            if seen >= SUBAGENTS:
                break
        assert time.monotonic() < deadline, f"conversation statistics {before} -> {after} in {log}"
        time.sleep(1)
    record_property("conversation_stats", f"seen {seen}, moved {moved}, pinned {pinned}")
    assert moved == 0, (before, after)


# -- NT-144, NT-145: perf records, request ids and the two legs' requests --------------------

# The kinds of perf record files: perf_metrics-<kind>-*.jsonl.
RECORD_KINDS = ("disagg", "context", "generation")


def _observed_group(dump_root: Path) -> basic.Group:
    """Perf records from the orchestrator and both worker front ends, and the legs' requests."""
    perf = {
        "return_perf_metrics": True,
        "perf_metrics_output_dir": str(dump_root / "perf"),
        "perf_metrics_max_requests": 1000,
    }
    return basic.Group(
        context=perf, generation=perf, env=_dump_env(dump_root / "observed"), front=perf
    )


def _jsonl(path: Path) -> list[dict]:
    """The complete records of a JSONL file another process may still be appending to."""
    text = path.read_text()
    lines = text.splitlines()
    if lines and not text.endswith("\n"):
        lines.pop()
    return [json.loads(line) for line in lines if line]


def _records(perf_dir: Path) -> dict[str, list[dict]]:
    """The perf records in ``perf_dir`` by the kind of server that wrote them."""
    return {
        kind: [r for p in sorted(perf_dir.glob(f"perf_metrics-{kind}-*.jsonl")) for r in _jsonl(p)]
        for kind in RECORD_KINDS
    }


def _new_requests(perf_dir: Path, seen: set, count: int) -> dict[str, list[dict]]:
    """The records of the ``count`` requests the orchestrator recorded since ``seen`` (ids)."""
    deadline = time.monotonic() + RECORD_WAIT_S
    while True:
        records = _records(perf_dir)
        new = {r.get("disagg_request_id") for r in records["disagg"]} - seen
        mine = {
            kind: [r for r in found if r.get("disagg_request_id") in new]
            for kind, found in records.items()
        }
        if all(len(mine[kind]) >= count for kind in RECORD_KINDS):
            return mine
        assert time.monotonic() < deadline, {kind: len(found) for kind, found in mine.items()}
        time.sleep(0.5)


@pytest.fixture
def observed_served(groups, dump_root) -> basic.Served:
    """The chain with perf records and the legs' requests written under ``dump_root``."""
    return _chain(groups, _observed_group(dump_root))


def test_perf_metrics_and_request_id(observed_served, dump_root):
    """NT-144 (DG-40, DG-82): front-end perf metrics, and one disagg_request_id end to end.

    A request that asks for its metrics gets them as response headers (not streamed) or as a
    metrics event of the stream; the orchestrator and both worker front ends write one JSONL
    record of it each, all ``complete`` and with the orchestrator's ``disagg_request_id``. The
    ds engine writes no record (it ignores ``return_perf_metrics``); what each engine was sent
    carries the id: the context leg's GenerateRequest, as a decimal string, and the generation
    leg's hand-off, whose session id the context engine set to it.
    """
    from tensorrt_llm.serve.perf_metrics import (
        RETURN_METRICS_HEADER,
        SERVER_TIMING_HEADER,
        SSE_METRICS_EVENT,
        START_END_TIME_HEADER,
    )

    served = observed_served
    perf_dir = dump_root / "perf"
    dirs = {role: dump_root / "observed" / role for role in ROLES}
    seen_ids = {r.get("disagg_request_id") for r in _records(perf_dir)["disagg"]}
    seen_dumps = {role: _dumps(path) for role, path in dirs.items()}
    asked = {RETURN_METRICS_HEADER: "1"}

    body = {"model": MODEL_NAME, "prompt": basic.COMPLETION_PROMPT, "max_tokens": 8}
    body["temperature"] = 0.0
    response = requests.post(
        f"{served.url}/v1/completions", json=body, headers=asked, timeout=REQUEST_TIMEOUT_S
    )
    assert response.status_code == 200, response.text
    for header in (SERVER_TIMING_HEADER, START_END_TIME_HEADER):
        assert response.headers.get(header), (header, dict(response.headers))

    streamed = requests.post(
        f"{served.url}/v1/completions",
        json={**body, "stream": True},
        headers=asked,
        timeout=REQUEST_TIMEOUT_S,
    )
    assert streamed.status_code == 200, streamed.text
    assert "data: [DONE]" in streamed.text, streamed.text
    event = streamed.text.split(f"event: {SSE_METRICS_EVENT}\n", 1)
    assert len(event) == 2, streamed.text[-2000:]
    metrics = json.loads(event[1].split("\n", 1)[0].removeprefix("data: "))
    assert metrics, streamed.text[-2000:]

    mine = _new_requests(perf_dir, seen_ids, 2)
    ids = {r["disagg_request_id"] for r in mine["disagg"]}
    assert len(ids) == 2 and all(isinstance(i, int) for i in ids), mine["disagg"]
    for kind in RECORD_KINDS:
        assert sorted(r["disagg_request_id"] for r in mine[kind]) == sorted(ids), (kind, mine)
        assert all(r.get("status") == "complete" for r in mine[kind]), (kind, mine[kind])
    assert all({"ctx_perf_metrics", "gen_perf_metrics"} <= r.keys() for r in mine["disagg"])

    # What each engine was sent names the same id.
    from tensorrt_llm.grpc.openengine_conventions import REQ_DISAGG_REQUEST_ID

    legs = {role: _new_dumps(path, seen_dumps[role], 2) for role, path in dirs.items()}
    assert all(len(found) == 2 for found in legs.values()), legs
    expected = sorted(str(i) for i in ids)
    for dump in legs["context"]:
        assert isinstance(dump.get("extra", {}).get(REQ_DISAGG_REQUEST_ID), str), dump
    assert sorted(_ctx_id(d) for d in legs["context"]) == expected, legs["context"]
    assert sorted(_gen_id(d) for d in legs["generation"]) == expected, legs["generation"]
    for dump in legs["generation"]:
        assert _gen_session(dump).get("session_id") == _gen_id(dump), _gen_session(dump)


def _comparable(dump: dict, leg_keys: tuple[str, ...]) -> dict:
    """``dump`` without ``request_id``, ``kv.session`` and the keys only its leg carries."""
    dump = json.loads(json.dumps(dump))
    dump.pop("request_id")
    extra = dump.get("extra", {})
    for key in leg_keys:
        extra.pop(key, None)
    if not extra:
        dump.pop("extra", None)
    kv = dump.get("kv", {})
    kv.pop("session", None)
    if not kv:
        dump.pop("kv", None)
    return dump


# The sampling fields the bodies below set, each under its own name in a GenerateRequest's
# ``sampling``; the ds engine takes no penalty other than its default and returns no logprobs.
_SAMPLING = ("temperature", "top_p", "top_k", "min_p", "seed")
LEG_BODIES = [
    ("v1/completions", {"prompt": basic.COMPLETION_PROMPT, "max_tokens": 3}),
    (
        "v1/completions",
        {
            "prompt": basic.COMPLETION_PROMPT,
            "max_tokens": 5,
            "temperature": 0.5,
            "top_p": 0.9,
            "seed": 7,
            "stop": ["!"],
        },
    ),
    (
        "v1/completions",
        {"prompt": basic.COMPLETION_PROMPT, "max_tokens": 4, "stop": ["x"], "ignore_eos": True},
    ),
    (
        "v1/completions",
        {"prompt": basic.COMPLETION_PROMPT, "max_tokens": 4, "top_k": 3, "min_p": 0.1},
    ),
    (
        "v1/chat/completions",
        {"messages": basic.CHAT_MESSAGES, "max_tokens": 6, **basic.NO_THINKING},
    ),
]


def _carries(dump: dict, body: dict) -> None:
    """``dump`` holds each field ``body`` sets: nothing the client gave is dropped."""
    checked = {"prompt", "messages", "chat_template_kwargs", "max_tokens", "stop", "ignore_eos"}
    assert set(body) <= checked | set(_SAMPLING), sorted(set(body) - checked - set(_SAMPLING))
    sampling, stopping = dump.get("sampling", {}), dump.get("stopping", {})
    for name in _SAMPLING:
        if name in body:
            # The JSON form of a uint64 is a string.
            assert float(sampling.get(name, "nan")) == pytest.approx(body[name]), (name, dump)
    stops = [condition.get("stop_text") for condition in stopping.get("conditions", [])]
    for text in body.get("stop", []):
        assert text in stops, (text, dump)
    if "ignore_eos" in body:
        assert stopping.get("ignore_eos", False) is body["ignore_eos"], dump


def test_two_leg_request_consistency(observed_served, dump_root):
    """NT-145 (DG-85): the two legs' GenerateRequests differ only in the single-leg keys.

    For each request through the orchestrator, the context and the generation leg's
    GenerateRequests pair up by ``disagg_request_id``; each holds every sampling and stopping
    field the client set; they are the same but for ``request_id``, the hand-off and the keys
    only one leg carries (EX-DSG-7). The context leg's ``max_tokens`` is the client's.
    """
    from tensorrt_llm.grpc.openengine_conventions import (
        REQ_DISAGG_REQUEST_ID,
        REQ_ORIG_ELAPSED_S,
        REQ_REQUEST_TYPE,
        REQ_SCHEDULE_STYLE,
    )

    prefill_only = (REQ_REQUEST_TYPE, REQ_DISAGG_REQUEST_ID, REQ_SCHEDULE_STYLE)
    decode_only = (REQ_ORIG_ELAPSED_S,)
    served = observed_served
    dirs = {role: dump_root / "observed" / role for role in ROLES}
    for path, body in LEG_BODIES:
        seen = {role: _dumps(directory) for role, directory in dirs.items()}
        response = requests.post(
            f"{served.url}/{path}", json={"model": MODEL_NAME, **body}, timeout=REQUEST_TIMEOUT_S
        )
        assert response.status_code == 200, (body, response.text)
        (prefill,) = _new_dumps(dirs["context"], seen["context"], 1)
        (decode,) = _new_dumps(dirs["generation"], seen["generation"], 1)
        assert prefill["extra"][REQ_REQUEST_TYPE] == "context_only", prefill
        assert _ctx_id(prefill) is not None and _gen_id(decode) == _ctx_id(prefill), (
            prefill,
            decode,
        )
        assert prefill["request_id"] != decode["request_id"]
        for dump in (prefill, decode):
            _carries(dump, body)
        assert int(prefill["stopping"]["max_tokens"]) == body["max_tokens"], prefill
        assert _comparable(prefill, prefill_only) == _comparable(decode, decode_only)


# -- NT-215: a context leg retried after a context front end dies (last: it kills one) --------

RETRY_REQUESTS = 32
RETRY_MAX_TOKENS = 4
# Batches to send until the attached front end has context legs in flight.
RETRY_BATCHES = 3
# Context legs in flight on the attached front end when it is killed: more than one, so that one
# finishing between the reading and the kill still leaves a lost leg.
LOST_LEGS = 2
# Long prompts keep the context engine busy, so that requests stay in flight on both front ends;
# distinct, so that block reuse does not shorten their prefill.
RETRY_PROMPT_WORDS = 1000
ATTACHED_FRONTEND_ENV = "TLLM_EXECUTOR_FRONTEND_ID"


def _retry_group(dump_root: Path) -> basic.Group:
    """Two front ends for the context worker, sharing its port and its ds engine."""
    return basic.Group(
        context={"num_serve_frontends": 2},
        env=_dump_env(dump_root / "retry"),
        front={"max_retries": 3},
    )


def _attached_frontends(launcher_pid: int) -> list:
    """The processes a multi-frontend worker front end started to serve beside it."""
    return _marked_roots(launcher_pid, ATTACHED_FRONTEND_ENV)


def _retry_prompt(index: int) -> str:
    words = " ".join(f"w{index}x{i}" for i in range(RETRY_PROMPT_WORDS))
    return f"Request {index}. {words}. Summarize the list in one word:"


@pytest.fixture
def retry_served(groups, dump_root) -> basic.Served:
    """The chain whose context worker has two front ends."""
    return _chain(groups, _retry_group(dump_root))


def _ctx_legs_in_flight(counter_dir: str, pid: int) -> int:
    """Context legs front end ``pid`` sent its engine that have not yet had their PrefillReady."""
    from tensorrt_llm.grpc.openengine_conventions import (
        BUCKET_CONTEXT_ONLY,
        SERVE_COUNTER_GENERATE_SENT_BUCKETS,
        SERVE_COUNTER_PREFILL_READY_RECEIVED,
    )
    from tensorrt_llm.serve.openengine_poc import contracts

    by_pid = contracts.read_serve_counters_by_pid(counter_dir)
    # A front end publishes its counters once connected to its engine, before serving.
    assert pid in by_pid, (pid, sorted(by_pid), counter_dir)
    counts = by_pid[pid]
    sent = counts[SERVE_COUNTER_GENERATE_SENT_BUCKETS[BUCKET_CONTEXT_ONLY]]
    return sent - counts[SERVE_COUNTER_PREFILL_READY_RECEIVED]


def test_ctx_leg_retry_after_ctx_frontend_killed(retry_served, dump_root):
    """NT-215 (DG-45): a context leg lost with its front end is retried under a new id.

    The context worker runs two front ends on one port and one ds engine. Requests are sent until
    the attached front end has context legs in flight (its own counters), and it is killed then.
    Every request succeeds with all its tokens, and the orchestrator counted at least one context
    retry. The context front ends sent more attempts than there are requests, each with a
    ``disagg_request_id`` of its own, and each request's generation leg takes one of them.
    Requests sent afterwards succeed too.
    """
    served = retry_served
    # This is the file's last test: the module's groups fixture stops the group after it.
    handle = served.servers["context"]._chain_handle
    found = _attached_frontends(handle.frontend.pid)
    assert len(found) == 1, _describe(found)
    (attached,) = found
    dirs = {role: dump_root / "retry" / role for role in ROLES}
    seen = {role: _dumps(path) for role, path in dirs.items()}
    retries = _orchestrator_counter(served, "ctx", "retry_requests")

    sent, killed, peak = 0, False, 0
    with concurrent.futures.ThreadPoolExecutor(RETRY_REQUESTS) as pool:
        for _ in range(RETRY_BATCHES):
            # Count from here: a leg that ended without a PrefillReady stays in the count.
            idle = _ctx_legs_in_flight(handle.counter_dir, attached.pid)
            # ignore_eos: a request whose first token is EOS would end on the context leg,
            # with no generation leg to take a handoff.
            futures = [
                pool.submit(
                    _completion,
                    served,
                    _retry_prompt(sent + i),
                    max_tokens=RETRY_MAX_TOKENS,
                    ignore_eos=True,
                )
                for i in range(RETRY_REQUESTS)
            ]
            sent += RETRY_REQUESTS
            while not all(future.done() for future in futures):
                in_flight = _ctx_legs_in_flight(handle.counter_dir, attached.pid) - idle
                peak = max(peak, in_flight)
                if in_flight >= LOST_LEGS:
                    # psutil checks that the pid is still this process before it signals.
                    attached.kill()
                    killed = True
                    break
                time.sleep(0.02)
            for future in futures:
                response = future.result()
                assert response.status_code == 200, response.text
                # Every token generated; not the text, which special tokens can leave empty.
                body = response.json()
                assert body["usage"]["completion_tokens"] == RETRY_MAX_TOKENS, response.text
                assert body["choices"][0]["finish_reason"] == "length", response.text
            if killed:
                break
    assert killed, (
        f"the attached front end never had {LOST_LEGS} context legs in flight in "
        f"{RETRY_BATCHES} batches of {RETRY_REQUESTS} requests (at most {peak})"
    )
    deadline = time.monotonic() + RECORD_WAIT_S
    while _orchestrator_counter(served, "ctx", "retry_requests") == retries:
        assert time.monotonic() < deadline, "the orchestrator counted no context retry"
        time.sleep(0.5)

    # A lost attempt and its retry are both among what the context front ends sent.
    attempts = [_ctx_id(d) for d in _new_dumps(dirs["context"], seen["context"], sent + 1)]
    handoffs = [_gen_id(d) for d in _new_dumps(dirs["generation"], seen["generation"], sent)]
    assert None not in attempts and len(attempts) == len(set(attempts)), sorted(map(str, attempts))
    assert len(attempts) > sent, (len(attempts), sent)
    assert len(handoffs) == sent and len(set(handoffs)) == sent, handoffs
    assert set(handoffs) <= set(attempts), (handoffs, attempts)

    for i in range(4):
        response = _completion(served, f"After the loss {i}: the capital of France is")
        assert response.status_code == 200, response.text
