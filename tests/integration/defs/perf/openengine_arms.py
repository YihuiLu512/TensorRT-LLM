# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comparison arms for running the perf tests through the OpenEngine chain.

``TRTLLM_PERF_ARM`` selects an arm. With neither it nor
``TRTLLM_TEST_OPENENGINE_CHAIN`` set, the perf tests start their server as
the test configures it and write no arm result; the perf tests that read the
arm refuse a chain without one. An arm changes
the number of serving front ends, the number of post-processing workers,
whether the server runs as an OpenEngine engine plus a
``trtllm-serve --openengine-endpoint`` front end, and which engine serves;
with the TensorRT-LLM chain, the harness also adds engine options of its own
(``openengine_chain.launch`` says which). Every other setting of the test is kept; a server that needs a fixed port gets
one chosen in advance. The arms of :data:`ARMS` take the plain aggregated
perf sanity cases (no benchmark mode such as ``ctx_only``, no
``time_breakdown``) and the legacy serve perf tests; the legacy tests still
write their usual perf records under the test's own name. The arms of
:data:`DISAGG_ARMS` take the disaggregated perf sanity cases that
:func:`check_disagg_case` accepts, which include the one-server modes
``ctx_only`` and ``gen_only_no_context`` of a disaggregated YAML: their one
server starts as the arm's settings of its role say, and its results name
that role view (``E4-chain:ctx``, :func:`result_arm`). The other perf sanity cases, perf sanity
test ids that upload results, clients other than benchmark_serving (accuracy
runs included) and the legacy trtllm-bench tests refuse an arm, and the
arms with a chain refuse the options that the OpenEngine front end's option
check refuses; perf tests that do not read these variables ignore them.

Each measured benchmark run then writes one result JSON with its metrics, the
front end's CPU time per output token, the settings that must match across arms
and, for the arms with a chain, the Generate calls the front end sent and the
engine received.

When an arm is set, the harness package ``openengine_chain`` must be importable
under that top-level name (``openengine_chain/runner.py`` puts it on
``PYTHONPATH``), and ``TRTLLM_TEST_OPENENGINE_CHAIN`` must name the arm's chain,
or be unset for the arms that do not use one, so that the harness's pytest
plugin proves each test went through the chain.

Environment variables:

- ``TRTLLM_PERF_ARM``: the arm, one of :data:`ARMS` or :data:`DISAGG_ARMS`.
- ``TRTLLM_PERF_RUN``: the run number written to the result (default 1).
- ``TRTLLM_PERF_GLM_SHA``: the glm engine's commit written to the result.
- ``TRTLLM_OPENENGINE_RESULTS_DIR``: result files go to its ``perf``
  subdirectory; unset, they go next to the test's own logs.
- ``TRTLLM_TEST_GLM_SERVE_CMD``: the glm server's command prefix, for the
  case-01 arms.
- ``TRTLLM_TEST_DS_SERVE_CMD``: dsv4-serve's command prefix, for the case-03 arms
  (``E3-*`` of :data:`DISAGG_ARMS`).
- ``TRTLLM_TEST_DS_ROOT``: the ds-engine checkout, whose commit the case-03 results
  record and whose ``dev/run_agentx.sh`` the case-03 arms' AgentX client runs;
  ``TRTLLM_PERF_DS_SHA`` gives the commit instead.

The case-03 arms also take perf sanity's AgentX client, which they run as ds's
(``python openengine_arms.py ds-agentx``, :func:`run_ds_agentx`).
"""

from __future__ import annotations

import contextlib
import copy
import functools
import importlib.util
import json
import os
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import IO, Any

import yaml

ENV_ARM = "TRTLLM_PERF_ARM"
ENV_RUN = "TRTLLM_PERF_RUN"
ENV_GLM_SHA = "TRTLLM_PERF_GLM_SHA"
# The harness's names (openengine_chain/conventions.py and split.py);
# test_poc2_wt09_perf_arms.py checks they match.
ENV_CHAIN = "TRTLLM_TEST_OPENENGINE_CHAIN"
ENV_GLM_SERVE_CMD = "TRTLLM_TEST_GLM_SERVE_CMD"
ENV_RESULTS_DIR = "TRTLLM_OPENENGINE_RESULTS_DIR"
ENV_SPEC_DECODE_ACCEPTED = "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS"

HARNESS_PACKAGE = "openengine_chain"
# GPU memory (MiB) from which a process counts as an engine process.
ENGINE_GPU_MIB = 2048
CHAIN_NONE = "none"
CHAIN_TRTLLM = "trtllm"
CHAIN_GLM = "glm"

CASE_GLM = "case-01"
CASE_TRTLLM = "case-02"

# trtllm-serve's default front-end count with an explicit port.
DEFAULT_SERVE_FRONTENDS = 8
# The configuration keys an arm may change.
ARM_SETTINGS = ("num_serve_frontends", "num_postprocess_workers")
CONFIG_OPTIONS = ("--config", "--extra_llm_api_options")

# Reads of the counters after a run, repeated while the front end's and the engine's increases differ.
_SETTLE_TRIES = 4
_SETTLE_INTERVAL_S = 0.5


class ArmError(RuntimeError):
    """The arm cannot be applied to this test."""


@dataclass(frozen=True)
class RoleArm:
    """What a disaggregated arm sets for the workers of one role.

    Attributes:
        chain: ``none`` (an in-process ``trtllm-serve`` worker), ``trtllm`` or ``ds``.
        frontends: ``num_serve_frontends`` the arm sets, or ``None`` to keep the test's.
        postprocess_workers: ``num_postprocess_workers`` the arm sets, or ``None`` to keep the test's.
        ds_http: With chain ``none``: the worker is ``dsv4-serve`` serving HTTP itself
            (``TRTLLM_TEST_DS_SERVE_CMD`` replaces ``trtllm-serve``).
    """

    chain: str
    frontends: int | None
    postprocess_workers: int | None
    ds_http: bool = False

    @property
    def in_process(self) -> bool:
        return self.chain == "none"

    def settings(self) -> dict[str, int]:
        values = {
            "num_serve_frontends": self.frontends,
            "num_postprocess_workers": self.postprocess_workers,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class Arm:
    """One comparison arm.

    Attributes:
        name: The value of ``TRTLLM_PERF_ARM``.
        case: ``case-02`` (TRT-LLM engine) or ``case-01`` (glm engine) for one server;
            ``case-04`` (TRT-LLM engine) or ``case-03`` (ds engine) for a disaggregated test.
        chain: ``none``, ``trtllm``, ``glm`` or ``ds``.
        glm_http: Serve with the glm server's own HTTP front end instead of trtllm-serve.
        frontends: ``num_serve_frontends`` the arm sets, or ``None`` to keep the test's.
        postprocess_workers: ``num_postprocess_workers`` the arm sets, or ``None`` to keep the test's.
        topology: ``agg`` for one server; ``disagg`` for the arms of a disaggregated
            perf test, whose worker roles ``ctx`` and ``gen`` are set per role; ``direct``
            for the direct replay tool, which perf sanity does not run.
        ctx: The context workers' settings of a ``disagg`` arm.
        gen: The generation workers' settings of a ``disagg`` arm.
        orchestrator: The disaggregated server of a ``disagg`` arm: ``trtllm``
            (``trtllm-serve disaggregated``) or ``ds`` (``dsv4-serve disaggregated``).
        aligned: A case-03 arm whose disaggregated server gets :data:`DS_ALIGNED_ORCHESTRATOR`.
    """

    name: str
    case: str
    chain: str
    glm_http: bool
    frontends: int | None
    postprocess_workers: int | None
    topology: str = "agg"
    ctx: RoleArm | None = None
    gen: RoleArm | None = None
    orchestrator: str = "trtllm"
    aligned: bool = False

    @property
    def in_process(self) -> bool:
        """Whether the front end and the engine run in one server process tree."""
        return self.chain == CHAIN_NONE

    def settings(self) -> dict[str, int]:
        """The configuration values the arm sets."""
        values = {
            "num_serve_frontends": self.frontends,
            "num_postprocess_workers": self.postprocess_workers,
        }
        return {key: value for key, value in values.items() if value is not None}

    @property
    def explicit_port(self) -> bool:
        """Whether the server needs a port chosen in advance (see :func:`free_port`).

        The glm server does not take ``--report_addr``, and trtllm-serve
        refuses ``--port 0`` with more than one front end; the arms with a
        chain use a port chosen in advance as well.
        """
        return not self.in_process or self.glm_http or (self.frontends or 1) > 1


_PORT_FLOOR = 10000
_PORT_TRIES = 64
# The size of the window just below the ephemeral range (get_static_port_range) from
# which get_free_port_in_ci in tests/integration/defs/common.py hands ports to other
# test servers; test_poc2_wt09_perf_arms.py checks it matches STATIC_PORT_RANGE_SIZE there.
_STATIC_PORT_RANGE_SIZE = 4096
_EPHEMERAL_RANGE_FILE = "/proc/sys/net/ipv4/ip_local_port_range"


def _ephemeral_floor() -> int:
    """The lowest port the kernel numbers sockets from by itself, or 0 when unknown.

    A range outside 1-65535, or with its bounds reversed, counts as unknown.
    """
    try:
        with open(_EPHEMERAL_RANGE_FILE, encoding="ascii") as f:
            low, high = (int(value) for value in f.read().split())
    except (OSError, ValueError):
        return 0
    return low if 1 <= low <= high <= 65535 else 0


def _family(host: str) -> socket.AddressFamily:
    """IPv6 when every address of ``host`` is IPv6, else IPv4, the family trtllm-serve binds."""
    try:
        infos = socket.getaddrinfo(host, 0, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        return socket.AF_INET
    if infos and all(info[0] == socket.AF_INET6 for info in infos):
        return socket.AF_INET6
    return socket.AF_INET


def _container_ports() -> range:
    """The CI container's port range (``CONTAINER_PORT_START``, ``CONTAINER_PORT_NUM``).

    get_free_port_in_ci in tests/integration/defs/common.py hands other test
    servers ports from it first. Empty when unset or not a number.
    """
    try:
        start = int(os.environ.get("CONTAINER_PORT_START", -1))
        count = int(os.environ.get("CONTAINER_PORT_NUM", -1))
    except ValueError:
        return range(0)
    if start == -1 or count == -1:
        return range(0)
    return range(start, start + count)


def _candidate_ports(top: int, handed_out: range) -> Iterator[int]:
    """Up to ``_PORT_TRIES`` random ports in ``[_PORT_FLOOR, top)`` outside ``handed_out``.

    The ports are drawn from what is left once ``handed_out`` is cut out, so
    that a container range covering most of the window costs no tries.
    """
    cut = range(max(_PORT_FLOOR, handed_out.start), min(top, handed_out.stop))
    stop = top - len(cut)
    if stop <= _PORT_FLOOR:
        return
    pick = random.SystemRandom()
    for _ in range(_PORT_TRIES):
        port = pick.randrange(_PORT_FLOOR, stop)
        yield port + len(cut) if cut and port >= cut.start else port


def free_port(host: str) -> int:
    """A port of ``host`` that was free a moment ago, for an arm's server.

    When there is room, the port is taken below the kernel's ephemeral range,
    so that no socket the kernel numbers itself (the engine's own connections
    while it loads, for one) can take it before the server binds. Ports below
    the static window under that range and outside the CI container's port
    range, which get_free_port_in_ci hands to no other test server, are tried
    first. The socket is closed before the server starts, so that every bind
    of the port, the server's own and the probe of trtllm-serve's
    multi-front-end launcher included, finds it free whatever socket options it
    sets.

    Raises:
        OSError: No port of ``host`` can be bound.
    """
    family = _family(host)
    floor = _ephemeral_floor()
    draws = (
        ((floor - _STATIC_PORT_RANGE_SIZE, _container_ports()), (floor, range(0))) if floor else ()
    )
    for top, handed_out in draws:
        for port in _candidate_ports(top, handed_out):
            with socket.socket(family, socket.SOCK_STREAM) as s:
                try:
                    s.bind((host, port))
                except OSError:
                    continue
                return port
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm("A-def", CASE_TRTLLM, CHAIN_NONE, False, DEFAULT_SERVE_FRONTENDS, None),
        Arm("A-1f", CASE_TRTLLM, CHAIN_NONE, False, 1, None),
        Arm("A-1f0", CASE_TRTLLM, CHAIN_NONE, False, 1, 0),
        Arm("B-s1", CASE_TRTLLM, CHAIN_TRTLLM, False, 1, 0),
        Arm("B-full", CASE_TRTLLM, CHAIN_TRTLLM, False, DEFAULT_SERVE_FRONTENDS, None),
        Arm("C-http", CASE_GLM, CHAIN_NONE, True, None, None),
        Arm("C-s1", CASE_GLM, CHAIN_GLM, False, 1, 0),
        Arm("C-full", CASE_GLM, CHAIN_GLM, False, DEFAULT_SERVE_FRONTENDS, None),
    )
}

# The case-04 arms (TensorRT-LLM engines behind a disaggregated server). Each worker
# runs one front end; the lean arms also run no post-processing workers.
CASE_DISAGG = "case-04"
TOPOLOGY_AGG = "agg"
TOPOLOGY_DISAGG = "disagg"
TOPOLOGY_DIRECT = "direct"
ROLE_CTX = "ctx"
ROLE_GEN = "gen"
WORKER_ROLES = (ROLE_CTX, ROLE_GEN)
_IN_PROCESS_1F = RoleArm(CHAIN_NONE, 1, None)
_CHAIN_1F = RoleArm(CHAIN_TRTLLM, 1, None)
_IN_PROCESS_LEAN = RoleArm(CHAIN_NONE, 1, 0)
_CHAIN_LEAN = RoleArm(CHAIN_TRTLLM, 1, 0)
# Kept apart from ARMS, the single-server arms, which they do not change.
DISAGG_ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm(
            "E4-base",
            CASE_DISAGG,
            CHAIN_NONE,
            False,
            None,
            None,
            TOPOLOGY_DISAGG,
            _IN_PROCESS_1F,
            _IN_PROCESS_1F,
        ),
        Arm(
            "E4-chain",
            CASE_DISAGG,
            CHAIN_TRTLLM,
            False,
            None,
            None,
            TOPOLOGY_DISAGG,
            _CHAIN_1F,
            _CHAIN_1F,
        ),
        Arm(
            "E4-ctx",
            CASE_DISAGG,
            CHAIN_TRTLLM,
            False,
            None,
            None,
            TOPOLOGY_DISAGG,
            _CHAIN_1F,
            _IN_PROCESS_1F,
        ),
        Arm(
            "E4-gen",
            CASE_DISAGG,
            CHAIN_TRTLLM,
            False,
            None,
            None,
            TOPOLOGY_DISAGG,
            _IN_PROCESS_1F,
            _CHAIN_1F,
        ),
        Arm(
            "E4-base-lean",
            CASE_DISAGG,
            CHAIN_NONE,
            False,
            None,
            None,
            TOPOLOGY_DISAGG,
            _IN_PROCESS_LEAN,
            _IN_PROCESS_LEAN,
        ),
        Arm(
            "E4-chain-lean",
            CASE_DISAGG,
            CHAIN_TRTLLM,
            False,
            None,
            None,
            TOPOLOGY_DISAGG,
            _CHAIN_LEAN,
            _CHAIN_LEAN,
        ),
        # The direct replay tool (openengine_chain/direct_replay_load.py) against two
        # engines; perf sanity refuses it.
        Arm("E4-direct", CASE_DISAGG, CHAIN_TRTLLM, False, None, None, TOPOLOGY_DIRECT),
    )
}

# case-03: the ds-engine (TH-DSG-6). The native arms run dsv4-serve's own workers behind
# its own disaggregated server (form K). E3-native keeps the deployment's front-end count and
# disaggregated-server variables; E3-native-al runs one front end per worker and one ds proxy
# process without load routing, and writes the orchestrator values of DS_ALIGNED_ORCHESTRATOR,
# as do the arms with trtllm-serve's disaggregated server. On layout N-0 (PU-06,
# $DS/dev/test_disagg.sh) the deployment sets none of the switches E3-native-al changes, so
# there the two native arms are one group.
CASE_DS = "case-03"
CHAIN_DS = "ds"
ORCHESTRATOR_TRTLLM = "trtllm"
ORCHESTRATOR_DS = "ds"
_DS_HTTP = RoleArm(CHAIN_NONE, None, None, True)
_DS_HTTP_1F = RoleArm(CHAIN_NONE, 1, None, True)
_DS_CHAIN_1F = RoleArm(CHAIN_DS, 1, None)
_DS_CHAIN_LEAN = RoleArm(CHAIN_DS, 1, 0)
DISAGG_ARMS.update(
    {
        arm.name: arm
        for arm in (
            Arm(
                "E3-native",
                CASE_DS,
                CHAIN_NONE,
                False,
                None,
                None,
                TOPOLOGY_DISAGG,
                _DS_HTTP,
                _DS_HTTP,
                ORCHESTRATOR_DS,
            ),
            Arm(
                "E3-native-al",
                CASE_DS,
                CHAIN_NONE,
                False,
                None,
                None,
                TOPOLOGY_DISAGG,
                _DS_HTTP_1F,
                _DS_HTTP_1F,
                ORCHESTRATOR_DS,
                True,
            ),
            Arm(
                "E3-mid",
                CASE_DS,
                CHAIN_NONE,
                False,
                None,
                None,
                TOPOLOGY_DISAGG,
                _DS_HTTP_1F,
                _DS_HTTP_1F,
                ORCHESTRATOR_TRTLLM,
                True,
            ),
            Arm(
                "E3-chain",
                CASE_DS,
                CHAIN_DS,
                False,
                None,
                None,
                TOPOLOGY_DISAGG,
                _DS_CHAIN_1F,
                _DS_CHAIN_1F,
                ORCHESTRATOR_TRTLLM,
                True,
            ),
            Arm(
                "E3-chain-lean",
                CASE_DS,
                CHAIN_DS,
                False,
                None,
                None,
                TOPOLOGY_DISAGG,
                _DS_CHAIN_LEAN,
                _DS_CHAIN_LEAN,
                ORCHESTRATOR_TRTLLM,
                True,
            ),
            # The direct replay tool against two ds OpenEngine servers; perf sanity refuses it.
            Arm("E3-direct", CASE_DS, CHAIN_DS, False, None, None, TOPOLOGY_DIRECT),
        )
    }
)


def all_arms() -> dict[str, Arm]:
    """Every arm ``TRTLLM_PERF_ARM`` may name: :data:`ARMS` and :data:`DISAGG_ARMS`."""
    return {**ARMS, **DISAGG_ARMS}


def selected_arm(environ: Mapping[str, str] | None = None) -> Arm | None:
    """The arm ``TRTLLM_PERF_ARM`` selects, or ``None`` when neither it nor a chain is set.

    Raises:
        ArmError: The arm is unknown, ``TRTLLM_TEST_OPENENGINE_CHAIN`` does
            not match it (a chain without an arm included),
            ``TRTLLM_PERF_RUN`` is not a positive integer, or the harness
            package cannot be found.
    """
    env = os.environ if environ is None else environ
    name = env.get(ENV_ARM, "")
    chain = env.get(ENV_CHAIN, "")
    known = all_arms()
    if not name:
        if chain:
            raise ArmError(
                f"{ENV_CHAIN}={chain!r} is set for a perf test; set {ENV_ARM} to one of "
                f"{', '.join(known)} to choose the comparison arm"
            )
        return None
    arm = known.get(name)
    if arm is None:
        raise ArmError(f"{ENV_ARM}={name!r}; expected one of {', '.join(known)}")
    expected = "" if arm.in_process else arm.chain
    if chain != expected:
        want = f"set to {expected!r}" if expected else "unset"
        raise ArmError(f"{ENV_ARM}={name} needs {ENV_CHAIN} {want}; it is {chain!r}")
    run_number(env)
    if importlib.util.find_spec(HARNESS_PACKAGE) is None:
        raise ArmError(
            f"{ENV_ARM}={name} needs the harness package {HARNESS_PACKAGE} importable; "
            "run the tests through openengine_chain/runner.py"
        )
    return arm


# The benchmark line perf sanity requires of speculative-decoding cases. The
# benchmark client averages the ``avg_decoded_tokens_per_iter`` field of each
# response, which the OpenEngine front end does not fill in: Generate's events
# carry no such count. So against the arms with a chain the client prints no such line.
DECODED_TOKENS_LINE = "Mean Avg Decoded Tokens per Iter"
_DECODED_TOKENS_NOTE = (
    f"speculative decoding: perf sanity does not check '{DECODED_TOKENS_LINE}' of the arms "
    "with a chain, whose OpenEngine responses carry no avg_decoded_tokens_per_iter"
)


def checks_decoded_tokens(arm: Arm | None) -> bool:
    """Whether perf sanity requires :data:`DECODED_TOKENS_LINE` of a speculative-decoding case.

    Not of the arms with a chain, which cannot report it; their results of
    such cases say so instead. Under a disaggregated arm the generation workers
    send the responses, so their chain decides.
    """
    if arm is None:
        return True
    if arm.topology == TOPOLOGY_DISAGG and arm.gen is not None:
        return arm.gen.in_process
    return arm.in_process


def _speculative(config: Mapping[str, Any]) -> bool:
    """Whether perf sanity counts this server as speculative decoding: a ``decoding_type`` is set."""
    speculative = config.get("speculative_config")
    return isinstance(speculative, Mapping) and bool(speculative.get("decoding_type"))


def run_number(environ: Mapping[str, str]) -> int:
    """``TRTLLM_PERF_RUN`` (default 1).

    Raises:
        ArmError: It is not a positive integer.
    """
    run = _int_or_none(environ.get(ENV_RUN, "1"))
    if run is None or run < 1:
        raise ArmError(f"{ENV_RUN}={environ.get(ENV_RUN)!r}; expected a positive integer")
    return run


# ----------------------------------------------------------------------------- command line


def _option_values(tokens: Sequence[str], name: str) -> list[tuple[int, str | None]]:
    """``(index, value)`` of every ``name VALUE`` or ``name=VALUE`` in ``tokens``."""
    found = []
    for i, token in enumerate(tokens):
        if token == name:
            found.append((i, tokens[i + 1] if i + 1 < len(tokens) else None))
        elif token.startswith(name + "="):
            found.append((i, token.split("=", 1)[1]))
    return found


def option_value(tokens: Sequence[str], name: str) -> str | None:
    """The last value given to option ``name`` in ``tokens``, or ``None``."""
    found = _option_values(tokens, name)
    return found[-1][1] if found else None


@dataclass(frozen=True)
class ServerPlan:
    """The server command line of one arm and the configuration it gives the server.

    Attributes:
        cmd: The command line to start.
        config: The YAML configuration the command line passes, after the arm's changes.
    """

    cmd: tuple[str, ...]
    config: dict[str, Any]


def apply_arm(config: Mapping[str, Any], arm: Arm) -> dict[str, Any]:
    """A copy of ``config`` with the arm's settings."""
    out = copy.deepcopy(dict(config))
    out.update(arm.settings())
    return out


# Top-level configuration keys the result's alignment block reads.
_ALIGNMENT_ROOTS = (
    "kv_cache_config",
    "reasoning_parser",
    "tool_parser",
    "disable_mm_encoder",
    "stream_interval",
    "num_serve_frontends",
    "num_postprocess_workers",
)


def _check_no_cli_override(tokens: Sequence[str], arm: Arm) -> None:
    """Refuse command-line values the arm would change, and ``--set`` of compared settings.

    Command-line values take precedence over the YAML the arm writes, and the
    result's alignment block does not read ``--set``.
    """
    for key in arm.settings():
        if _option_values(tokens, f"--{key}"):
            raise ArmError(f"arm {arm.name} sets {key}, which the command line passes as --{key}")
    for _, value in _option_values(tokens, "--set"):
        root = (value or "").split("=", 1)[0].strip().split(".", 1)[0].split("[", 1)[0]
        if root in _ALIGNMENT_ROOTS:
            raise ArmError(
                f"arm {arm.name}: the command line sets {root} with --set; arms take the "
                "compared settings from the YAML or their own options only"
            )


# trtllm-serve options with a value that the harness gives the chain's front end
# and that the front end's option check reads; the configuration keys it reads
# come from the YAML itself.
_FRONTEND_CLI_OPTIONS = (
    "num_postprocess_workers",
    "otlp_traces_endpoint",
    "post_processor_hook",
)


def _frontend_check() -> Callable[[Any], None]:
    """The OpenEngine front end's check of its options; tests replace this function."""
    from tensorrt_llm.serve.openengine_poc.facade import check_frontend_options

    return check_frontend_options


def _check_frontend_options(tokens: Sequence[str], config: Mapping[str, Any], arm: Arm) -> None:
    """Refuse, before any server starts, the options the front end's option check refuses.

    The harness starts the front end only once the engine serves; the front
    end runs ``facade.check_frontend_options`` when it starts.
    """
    values = dict(config)
    for name in _FRONTEND_CLI_OPTIONS:
        value = option_value(tokens, f"--{name}")
        if value is not None:
            values[name] = yaml.safe_load(value)
    for _, setting in _option_values(tokens, "--set"):
        key, _, value = (setting or "").partition("=")
        key = key.strip()
        if key and "." not in key and "[" not in key:
            values[key] = yaml.safe_load(value)
    try:
        _frontend_check()(types.SimpleNamespace(**values))
    except ValueError as error:
        raise ArmError(f"arm {arm.name}: {error}") from error
    except TypeError as error:
        # Pydantic has not validated the YAML yet, so a value of the wrong type,
        # such as a number given as a string, gets here; so would a defect of the
        # check itself, which the exception's name and the chained traceback show.
        raise ArmError(f"arm {arm.name}: {error!r}") from error


def prepare_server(
    cmd: Sequence[str],
    arm: Arm,
    out_dir: str,
    environ: Mapping[str, str] | None = None,
) -> ServerPlan:
    """The arm's version of a ``trtllm-serve`` command line.

    The arm's settings go into a copy of the command line's ``--config`` YAML
    written to ``out_dir`` (a new file when there is none). For the glm HTTP
    arm, the glm server's command prefix replaces ``trtllm-serve`` and gets
    every option of the command line.

    Raises:
        ArmError: The command line passes a value the arm sets, a ``--set`` of
            a compared setting, more than one YAML configuration, a
            configuration option without its value or a YAML that is not a
            mapping, an arm with a chain gets options the OpenEngine front end's
            option check refuses, or the glm HTTP arm has no
            ``TRTLLM_TEST_GLM_SERVE_CMD``, or ``arm`` is not a one-server arm.
        ValueError: The glm HTTP arm cannot split the command line.
        OSError: The YAML cannot be read or its copy written.
        yaml.YAMLError: The YAML or an option value cannot be parsed.
        ImportError: An arm with a chain cannot import TensorRT-LLM's
            OpenEngine front end, or the glm HTTP arm the harness.
    """
    if arm.topology != TOPOLOGY_AGG:
        # The legacy serve perf tests start one server for any arm; a disaggregated one is not theirs.
        raise ArmError(f"arm {arm.name} runs disaggregated perf sanity cases, not one server")
    tokens = list(cmd)
    _check_no_cli_override(tokens, arm)
    found = [(i, v) for name in CONFIG_OPTIONS for i, v in _option_values(tokens, name)]
    if len(found) > 1:
        raise ArmError(f"the command line passes {len(found)} YAML configurations; expected one")
    config: dict[str, Any] = {}
    if found:
        index, path = found[0]
        if path is None:
            raise ArmError(f"{tokens[index]} is missing its value")
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if not isinstance(config, dict):
            raise ArmError(f"{path} must hold a mapping, got {type(config).__name__}")
    settings = arm.settings()
    if settings:
        config = apply_arm(config, arm)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.basename(found[0][1]) if found else "config.yml"
        fd, arm_path = tempfile.mkstemp(prefix=f"arm-{arm.name}-", suffix=f"-{base}", dir=out_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        if found:
            index = found[0][0]
            if tokens[index] in CONFIG_OPTIONS:
                tokens[index + 1] = arm_path
            else:
                tokens[index] = f"{tokens[index].split('=', 1)[0]}={arm_path}"
        else:
            tokens += ["--config", arm_path]
    if not arm.in_process:
        _check_frontend_options(tokens, config, arm)
    if arm.glm_http:
        env = os.environ if environ is None else environ
        prefix = env.get(ENV_GLM_SERVE_CMD, "")
        if not prefix:
            raise ArmError(f"arm {arm.name} needs {ENV_GLM_SERVE_CMD}")
        from openengine_chain.split import parse_serve_command

        command = parse_serve_command(tokens)
        tokens = shlex.split(prefix) + [command.model] + list(command.args)
    return ServerPlan(cmd=tuple(tokens), config=config)


# ----------------------------------------------------------------------------- server


class ArmServer:
    """A started server of one arm; stands in for its ``subprocess.Popen``.

    For a chain arm, stopping it stops the front end, then the engine.
    """

    def __init__(
        self,
        arm: Arm,
        plan: ServerPlan,
        proc: Any,
        handle: Any,
        env: Mapping[str, str],
    ) -> None:
        self.arm = arm
        self.plan = plan
        self.proc = proc
        # The harness's ServeHandle for a chain arm, else None.
        self.handle = handle
        self.env = dict(env)

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def returncode(self) -> int | None:
        return self.proc.poll()

    @property
    def args(self) -> Any:
        return self.proc.args

    def poll(self) -> int | None:
        return self.proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        return self.proc.wait(timeout=timeout)

    def terminate(self) -> None:
        self.proc.terminate()

    def kill(self) -> None:
        self.proc.kill()

    @property
    def frontend_pid(self) -> int:
        """The process the front end's process tree hangs from."""
        if self.handle is not None:
            return self.handle.frontend.pid
        return self.proc.pid

    def start_run(self) -> "RunProbe":
        """Read the front end's CPU time and the chain's counters before a measured run."""
        return RunProbe.start(self)


def launch(
    plan: ServerPlan,
    arm: Arm,
    *,
    env: Mapping[str, str],
    sink: IO[Any] | int | None,
    ready_timeout: float,
) -> ArmServer:
    """Start the arm's server; the caller then waits for its ``/health``.

    A chain arm starts through the harness, which returns once the engine
    reports SERVING (within ``ready_timeout``) and the front end is started.

    Raises:
        OSError: The server cannot be started.
        Exception: For a chain arm, whatever the harness's
            ``launch_serve_command`` raises when the chain does not start.
    """
    if not arm.in_process:
        from openengine_chain.launch import ChainProcess, launch_serve_command

        handle = launch_serve_command(
            plan.cmd, chain=arm.chain, env=env, sink=sink, engine_ready_timeout=ready_timeout
        )
        return ArmServer(arm, plan, ChainProcess(handle), handle, env)
    proc = subprocess.Popen(list(plan.cmd), env=dict(env), stdout=sink, stderr=subprocess.STDOUT)
    return ArmServer(arm, plan, proc, None, env)


# ----------------------------------------------------------------------------- front-end CPU


def _process_tree(root: int) -> set[int]:
    from openengine_chain.launch import live_descendants

    return {root} | live_descendants(root)


def _engine_gpu_pids() -> set[int] | None:
    """PIDs nvidia-smi lists with at least ``ENGINE_GPU_MIB`` of GPU memory, or ``None``.

    A front end that initialises CUDA holds a context of a few hundred MiB; an
    engine holds its weights and KV cache.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pids = set()
    for line in out.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            if int(fields[1]) >= ENGINE_GPU_MIB:
                pids.add(int(fields[0]))
    return pids


def _uptime_ticks() -> int | None:
    """Clock ticks since boot, the clock of a process's start time in ``/proc/<pid>/stat``."""
    try:
        with open("/proc/uptime", encoding="ascii") as f:
            return int(float(f.read().split()[0]) * os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return None


def read_cpu_ticks(pid: int) -> tuple[int, int] | None:
    """``(start time, utime + stime)`` of ``pid`` in clock ticks, or ``None`` if it is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read()
    except OSError:
        return None
    # Fields after the parenthesised command, from field 3 (state) of proc(5).
    fields = stat[stat.rfind(b")") + 2 :].split()
    if len(fields) < 20:
        return None
    return int(fields[19]), int(fields[11]) + int(fields[12])


@dataclass
class CpuReader:
    """How the front end's processes and their CPU time are read; tests replace the readers."""

    tree: Callable[[int], set[int]] = _process_tree
    gpu: Callable[[], set[int] | None] = _engine_gpu_pids
    ticks: Callable[[int], tuple[int, int] | None] = read_cpu_ticks
    now: Callable[[], int | None] = _uptime_ticks
    ticks_per_s: float = field(default_factory=lambda: float(os.sysconf("SC_CLK_TCK")))

    def frontend_pids(self, root: int, *, in_process: bool) -> tuple[set[int] | None, str]:
        """The front end's processes: ``root``'s tree without the engine processes' trees.

        Engine processes are the ones :attr:`gpu` lists (by default those
        holding engine-sized GPU memory, see :func:`_engine_gpu_pids`), so a
        front end with a CUDA context of its own still counts as front end.
        When ``root`` itself is an engine process, the engine runs inside the
        front-end process and the two cannot be told apart.

        Returns the PIDs (``None`` when they cannot be told apart from the
        engine's) and a note on how they were found.
        """
        pids = self.tree(root)
        on_gpu = self.gpu()
        if on_gpu is None:
            if in_process:
                return None, "nvidia-smi unavailable: cannot tell the engine's processes apart"
            return pids, "nvidia-smi unavailable: the whole front-end tree is counted"
        if root in on_gpu:
            return None, (
                f"the server's root process {root} holds the engine's GPU memory: cannot tell "
                "the front end apart from the engine"
            )
        engine_roots = pids & on_gpu
        if in_process and not engine_roots:
            return None, (
                "nvidia-smi lists no engine process in the server's tree: cannot tell the "
                "engine's processes apart"
            )
        engine = set().union(*(self.tree(pid) for pid in engine_roots)) if engine_roots else set()
        note = f"{len(pids - engine)} front-end processes, {len(engine)} engine processes excluded"
        return pids - engine, note

    def snapshot(self, pids: Iterable[int]) -> dict[int, tuple[int, int]]:
        found = {}
        for pid in pids:
            value = self.ticks(pid)
            if value is not None:
                found[pid] = value
        return found


# How many front-end processes the note on the CPU time names, the busiest first.
CPU_NOTE_PROCESSES = 8
# How much of each one's command line it shows.
CPU_NOTE_CMDLINE_CHARS = 120


def cpu_by_process_note(
    by_pid: Mapping[int, int],
    ticks_per_s: float,
    cmdline: Callable[[int], str | None] | None = None,
) -> str:
    """A note on how the front end's CPU time of one run splits over its processes.

    Names the :data:`CPU_NOTE_PROCESSES` busiest processes, each with its CPU
    seconds and the start of its command line, so a front end that is busy can
    be told apart into its main process and its post-processing workers.
    """
    read = cmdline or _read_cmdline
    busiest = sorted(by_pid.items(), key=lambda item: (-item[1], item[0]))
    parts = [
        f"{pid} {ticks / ticks_per_s:.2f}s {(read(pid) or '?')[:CPU_NOTE_CMDLINE_CHARS].strip()}"
        for pid, ticks in busiest[:CPU_NOTE_PROCESSES]
    ]
    rest = busiest[CPU_NOTE_PROCESSES:]
    if rest:
        parts.append(f"{len(rest)} more {sum(t for _, t in rest) / ticks_per_s:.2f}s")
    return "front-end CPU by process: " + ("; ".join(parts) if parts else "none")


# ----------------------------------------------------------------------------- one run


@dataclass
class RunProbe:
    """The front end's CPU time and the chain's counters read before one measured run."""

    server: ArmServer
    reader: CpuReader
    cpu_before: dict[int, tuple[int, int]] | None
    counters_before: Any
    notes: list[str]
    started: int | None = None

    @classmethod
    def start(cls, server: ArmServer, reader: CpuReader | None = None) -> "RunProbe":
        reader = reader or CpuReader()
        notes: list[str] = []
        started = reader.now()
        pids, note = reader.frontend_pids(server.frontend_pid, in_process=server.arm.in_process)
        notes.append(f"before: {note}")
        cpu_before = reader.snapshot(pids) if pids is not None else None
        counters = _read_counters(server.handle, notes) if server.handle is not None else None
        return cls(server, reader, cpu_before, counters, notes, started)

    def frontend_cpu_s(self) -> float | None:
        """CPU seconds the front end used since :meth:`start`.

        Processes that start and exit between the two reads, and children the
        front end's processes reap, are not counted.
        """
        if self.cpu_before is None:
            return None
        pids, note = self.reader.frontend_pids(
            self.server.frontend_pid, in_process=self.server.arm.in_process
        )
        self.notes.append(f"after: {note}")
        if pids is None:
            return None
        after = self.reader.snapshot(pids | set(self.cpu_before))
        total = 0
        older = 0
        by_pid: dict[int, int] = {}
        for pid, (start, ticks) in after.items():
            before = self.cpu_before.get(pid)
            if before is not None and before[0] == start:
                total += ticks - before[1]
                by_pid[pid] = ticks - before[1]
            elif pid not in pids:
                # The PID of a front-end process that exited, now another process's.
                continue
            elif self.started is None or start >= self.started:
                total += ticks
                by_pid[pid] = ticks
            else:
                older += 1
        self.notes.append(cpu_by_process_note(by_pid, self.reader.ticks_per_s))
        if older:
            self.notes.append(
                f"{older} front-end processes were not read at the start of the run and "
                "started before it; their time is not counted"
            )
        gone = [
            pid
            for pid, (start, _) in self.cpu_before.items()
            if after.get(pid, (None,))[0] != start
        ]
        if gone:
            self.notes.append(
                f"{len(gone)} front-end processes exited during the run; "
                "their time after the first read is lost"
            )
        return total / self.reader.ticks_per_s

    def chain_counts(self) -> dict[str, int | None] | None:
        """Generate calls the front end sent and the engine received since :meth:`start`."""
        if self.server.handle is None:
            return None
        from openengine_chain import proof

        for attempt in range(_SETTLE_TRIES):
            after = _read_counters(self.server.handle, self.notes)
            start = proof.counting_start(self.counters_before, after)
            sent = _delta(start.frontend, after.frontend)
            received = _delta(start.engine, after.engine)
            if None in (sent, received) or sent == received:
                break
            if attempt + 1 < _SETTLE_TRIES:
                time.sleep(_SETTLE_INTERVAL_S)
        if sent is None:
            self.notes.append("no Generate count of the front end for this run")
        if received is None:
            self.notes.append("no Generate count of the engine for this run")
        if None not in (sent, received) and sent != received:
            self.notes.append(
                f"the front end sent {sent} Generate calls, the engine received {received}"
            )
        return {"generate_sent": sent, "generate_received": received}

    def finish(
        self,
        *,
        output: str,
        client_cmd: Sequence[str],
        test_id: str,
        default_dir: str,
        environ: Mapping[str, str] | None = None,
    ) -> str:
        """Write the run's result JSON and return its path.

        Args:
            output: The benchmark client's output.
            client_cmd: The client's command line.
            test_id: The pytest node id plus what tells this run apart within the test.
            default_dir: Where the file goes when ``TRTLLM_OPENENGINE_RESULTS_DIR`` is unset.
            environ: The environment the result reads (default ``os.environ``).

        Raises:
            ArmError: ``output`` is refused by :func:`check_benchmark_output`,
                ``TRTLLM_PERF_RUN`` is not a positive integer, the result is
                invalid, or no free file name is left; no file is written.
            OSError: The result file cannot be written.
        """
        check_benchmark_output(output)
        env = os.environ if environ is None else environ
        if not checks_decoded_tokens(self.server.arm) and _speculative(self.server.plan.config):
            self.notes.append(_DECODED_TOKENS_NOTE)
        cpu_s = self.frontend_cpu_s()
        chain = self.chain_counts()
        doc = build_result(
            arm=self.server.arm,
            plan=self.server.plan,
            server_env=self.server.env,
            output=output,
            client_cmd=client_cmd,
            test_id=test_id,
            frontend_cpu_s=cpu_s,
            chain_proof=chain,
            notes=self.notes,
            environ=env,
        )
        results = env.get(ENV_RESULTS_DIR)
        directory = os.path.join(results, "perf") if results else default_dir
        return write_result(doc, directory)


def _rpc_errors() -> tuple[type[BaseException], ...]:
    """GRPC's error type when gRPC is installed; the engine counter is read over gRPC."""
    try:
        import grpc
    except ImportError:
        return ()
    return (grpc.RpcError,)


def _read_counters(handle: Any, notes: list[str]) -> Any:
    """The chain's counters; a read that fails counts as no counter on either side."""
    from openengine_chain import proof

    try:
        return proof.read_counters(handle)
    except (OSError, ValueError, *_rpc_errors()) as error:
        notes.append(f"reading the chain's counters failed: {error!r}")
        return proof.Counters(None, None)


def _delta(before: int | None, after: int | None) -> int | None:
    return None if before is None or after is None else after - before


# ----------------------------------------------------------------------------- result file

_METRIC_LINES = {
    (metric, stat): re.compile(rf"^{label} {name} \(ms\):\s+(-?[\d.]+)", re.MULTILINE)
    for metric, name in (
        ("ttft_ms", "TTFT"),
        ("tpot_ms", "TPOT"),
        ("itl_ms", "ITL"),
        ("e2el_ms", "E2EL"),
    )
    for stat, label in (("mean", "Mean"), ("median", "Median"), ("p99", "P99"))
}
_OUTPUT_THROUGHPUT = re.compile(r"^Output token throughput \(tok/s\):\s+(-?[\d.]+)", re.MULTILINE)
_OUTPUT_TOKENS = re.compile(r"^Total generated tokens:\s+(\d+)", re.MULTILINE)
_FAILED_REQUESTS = re.compile(r"^Failed requests:\s+(\d+)", re.MULTILINE)
_ERROR_MARKERS = ("!FAILED REQUESTS!", "!CHECK LOG FOR ERRORS!")
_TOKEN_ID_DATASETS = ("random", "random_image")


def parse_metrics(output: str) -> tuple[dict[str, Any], int | None]:
    """The result's ``metrics`` block and the output token count from a benchmark's output."""

    def number(pattern: re.Pattern) -> float | None:
        match = pattern.search(output)
        return float(match.group(1)) if match else None

    metrics: dict[str, Any] = {}
    for (metric, stat), pattern in _METRIC_LINES.items():
        metrics.setdefault(metric, {})[stat] = number(pattern)
    metrics["output_tok_per_s"] = number(_OUTPUT_THROUGHPUT)
    tokens = _OUTPUT_TOKENS.search(output)
    return metrics, int(tokens.group(1)) if tokens else None


def check_benchmark_output(output: str) -> None:
    """Refuse a client output that is not a complete benchmark_serving result.

    Raises:
        ArmError: The output has no ``Total generated tokens`` line or reports
            0 tokens, some failed requests, or one of the error markers
            benchmark_serving prints.
    """
    tokens = _OUTPUT_TOKENS.search(output)
    if not tokens:
        raise ArmError(
            "the benchmark output has no 'Total generated tokens' line: arms measure "
            "benchmark_serving clients only"
        )
    if not int(tokens.group(1)):
        raise ArmError("the benchmark generated no tokens")
    failed = _FAILED_REQUESTS.search(output)
    if failed and int(failed.group(1)):
        raise ArmError(f"the benchmark reports {failed.group(1)} failed requests")
    for marker in _ERROR_MARKERS:
        if marker in output:
            raise ArmError(f"the benchmark output carries {marker}")


def dataset_kind(client_cmd: Sequence[str]) -> str:
    """``token_ids`` when the client sends the prompts as token ids, else ``text``."""
    name = option_value(client_cmd, "--dataset-name") or "sharegpt"
    if name in _TOKEN_ID_DATASETS and "--tokenize-on-client" in client_cmd:
        return "token_ids"
    return "text"


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def alignment(plan: ServerPlan, server_env: Mapping[str, str]) -> dict[str, Any]:
    """The settings that must match across arms, as the server gets them.

    A command-line option wins over the YAML, as in trtllm-serve
    (:func:`prepare_server` refuses ``--set`` of these settings); ``None``
    means the server's default.
    """
    cmd, config = plan.cmd, plan.config
    kv = config.get("kv_cache_config") or {}

    def setting(key: str) -> Any:
        value = option_value(cmd, f"--{key}")
        return value if value is not None else config.get(key)

    accepted = server_env.get(ENV_SPEC_DECODE_ACCEPTED)
    return {
        "enable_block_reuse": kv.get("enable_block_reuse") if isinstance(kv, dict) else None,
        "reasoning_parser": setting("reasoning_parser"),
        "tool_parser": setting("tool_parser"),
        "disable_mm_encoder": config.get("disable_mm_encoder"),
        "stream_interval": config.get("stream_interval"),
        "num_serve_frontends": _as_int(setting("num_serve_frontends")),
        "num_postprocess_workers": _as_int(setting("num_postprocess_workers")),
        "spec_decode_force_accepted": _float_or_none(accepted),
    }


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _as_int(value: Any) -> Any:
    return _int_or_none(value) if isinstance(value, str) else value


def trtllm_sha(environ: Mapping[str, str] | None = None) -> str | None:
    """The commit of the checkout the tests run from, or ``None``."""
    env = os.environ if environ is None else environ
    root = env.get("LLM_ROOT") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    return _git_head(root)


@functools.lru_cache(maxsize=None)
def _git_head(root: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


def build_result(
    *,
    arm: Arm,
    plan: ServerPlan,
    server_env: Mapping[str, str],
    output: str,
    client_cmd: Sequence[str],
    test_id: str,
    frontend_cpu_s: float | None,
    chain_proof: dict[str, int | None] | None,
    notes: Sequence[str] = (),
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One run's result document.

    Raises:
        ArmError: ``TRTLLM_PERF_RUN`` is not a positive integer.
    """
    env = os.environ if environ is None else environ
    metrics, output_tokens = parse_metrics(output)
    per_token = None
    if frontend_cpu_s is not None and output_tokens:
        per_token = frontend_cpu_s * 1e6 / output_tokens
    run = run_number(env)
    aligned = alignment(plan, server_env)
    notes = list(notes)
    accepted = server_env.get(ENV_SPEC_DECODE_ACCEPTED)
    if accepted and aligned["spec_decode_force_accepted"] is None:
        notes.append(f"{ENV_SPEC_DECODE_ACCEPTED}={accepted!r} is not a number")
    return {
        "case": arm.case,
        "arm": arm.name,
        "test_id": test_id,
        "concurrency": _int_or_none(
            option_value(client_cmd, "--max-concurrency")
            or option_value(client_cmd, "--concurrency")
        ),
        "dataset": dataset_kind(client_cmd),
        "run": run,
        "metrics": metrics,
        "frontend_cpu_s": frontend_cpu_s,
        "output_tokens": output_tokens,
        "frontend_cpu_us_per_token": per_token,
        "chain_proof": chain_proof,
        "alignment": aligned,
        "versions": {"trtllm": trtllm_sha(env), "glm": env.get(ENV_GLM_SHA) or None},
        "node": socket.gethostname(),
        "job_id": env.get("SLURM_JOB_ID") or None,
        "notes": notes,
    }


_ALIGNMENT_KEYS = (
    "enable_block_reuse",
    "reasoning_parser",
    "tool_parser",
    "disable_mm_encoder",
    "stream_interval",
    "num_serve_frontends",
    "num_postprocess_workers",
    "spec_decode_force_accepted",
)
_RESULT_KEYS = (
    "case",
    "arm",
    "test_id",
    "concurrency",
    "dataset",
    "run",
    "metrics",
    "frontend_cpu_s",
    "output_tokens",
    "frontend_cpu_us_per_token",
    "chain_proof",
    "alignment",
    "versions",
    "node",
    "job_id",
    "notes",
)
_LATENCY_METRICS = ("ttft_ms", "tpot_ms", "itl_ms", "e2el_ms")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_result(doc: Any) -> list[str]:
    """Problems with a result document; empty when it matches the result schema."""
    if not isinstance(doc, dict):
        return [f"result is a {type(doc).__name__}, not an object"]
    errors = []
    if set(doc) != set(_RESULT_KEYS):
        missing = sorted(set(_RESULT_KEYS) - set(doc))
        extra = sorted(set(doc) - set(_RESULT_KEYS))
        errors.append(f"keys: missing {missing}, unexpected {extra}")
        return errors
    arm = result_arm(doc["arm"]) if isinstance(doc["arm"], str) else None
    if arm is None:
        errors.append(
            f"arm {doc['arm']!r} is not one of {list(ARMS)} or <disagg arm>:<role> (result_arm)"
        )
    elif doc["case"] != arm.case:
        errors.append(f"case {doc['case']!r} is not arm {arm.name}'s {arm.case!r}")
    if not isinstance(doc["test_id"], str) or not doc["test_id"]:
        errors.append("test_id must be a non-empty string")
    if doc["concurrency"] is not None and not (
        _is_count(doc["concurrency"]) and doc["concurrency"] > 0
    ):
        errors.append("concurrency must be a positive integer or null")
    if doc["dataset"] not in ("token_ids", "text"):
        errors.append(f"dataset {doc['dataset']!r} is not token_ids or text")
    if not (_is_count(doc["run"]) and doc["run"] >= 1):
        errors.append("run must be a positive integer")
    metrics = doc["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {*_LATENCY_METRICS, "output_tok_per_s"}:
        errors.append("metrics must hold ttft_ms, tpot_ms, itl_ms, e2el_ms and output_tok_per_s")
    else:
        for name in _LATENCY_METRICS:
            stats = metrics[name]
            if not isinstance(stats, dict) or set(stats) != {"mean", "median", "p99"}:
                errors.append(f"metrics.{name} must hold mean, median and p99")
            elif any(v is not None and not _is_number(v) for v in stats.values()):
                errors.append(f"metrics.{name} values must be numbers or null")
        if metrics["output_tok_per_s"] is not None and not _is_number(metrics["output_tok_per_s"]):
            errors.append("metrics.output_tok_per_s must be a number or null")
    for key in ("frontend_cpu_s", "frontend_cpu_us_per_token"):
        if doc[key] is not None and not (_is_number(doc[key]) and doc[key] >= 0):
            errors.append(f"{key} must be a non-negative number or null")
    if doc["output_tokens"] is not None and not _is_count(doc["output_tokens"]):
        errors.append("output_tokens must be a non-negative integer or null")
    proof = doc["chain_proof"]
    if arm is not None and arm.in_process:
        if proof is not None:
            errors.append(f"chain_proof must be null for arm {arm.name}, which runs no chain")
    elif not isinstance(proof, dict) or set(proof) != {"generate_sent", "generate_received"}:
        errors.append("chain_proof must hold generate_sent and generate_received")
    elif any(v is not None and not _is_count(v) for v in proof.values()):
        errors.append("chain_proof counts must be non-negative integers or null")
    if not isinstance(doc["alignment"], dict) or set(doc["alignment"]) != set(_ALIGNMENT_KEYS):
        errors.append(f"alignment must hold exactly {list(_ALIGNMENT_KEYS)}")
    versions = doc["versions"]
    if not isinstance(versions, dict) or set(versions) != {"trtllm", "glm"}:
        errors.append("versions must hold trtllm and glm")
    elif any(v is not None and not isinstance(v, str) for v in versions.values()):
        errors.append("versions values must be strings or null")
    if not isinstance(doc["node"], str) or not doc["node"]:
        errors.append("node must be a non-empty string")
    if doc["job_id"] is not None and not isinstance(doc["job_id"], str):
        errors.append("job_id must be a string or null")
    if not isinstance(doc["notes"], list) or not all(isinstance(n, str) for n in doc["notes"]):
        errors.append("notes must be a list of strings")
    return errors


def write_result(doc: dict[str, Any], directory: str) -> str:
    """Validate ``doc`` and write it to a new file in ``directory``; return the path.

    Raises:
        ArmError: The document does not match the result schema, or no free
            file name is left.
        OSError: The directory or the file cannot be written.
    """
    errors = validate_result(doc)
    if errors:
        raise ArmError("invalid perf result: " + "; ".join(errors))
    os.makedirs(directory, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc["test_id"])[-120:]
    stem = f"{doc['arm']}-run{doc['run']}-{slug}"
    for n in range(1, 1000):
        path = os.path.join(directory, f"{stem}.json" if n == 1 else f"{stem}-{n}.json")
        try:
            with open(path, "x", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
                f.write("\n")
        except FileExistsError:
            continue
        return path
    raise ArmError(f"no free result file name for {stem} in {directory}")


def current_test_id() -> str:
    """The node id of the running pytest test, or ``""`` outside one."""
    current = os.environ.get("PYTEST_CURRENT_TEST", "")
    return re.sub(r" \((?:setup|call|teardown)\)$", "", current)


# ----------------------------------------------------------------------------- disaggregated arms
#
# A disaggregated perf sanity test runs each role in a pytest of its own (``DISAGG_SERVING_TYPE``):
# ``CTX_<i>`` and ``GEN_<i>`` start one worker each, ``DISAGG_SERVER`` the disaggregated server
# and ``BENCHMARK`` the client. Under a ``disagg`` arm each worker starts as its role's
# :class:`RoleArm` says, and each worker's pytest answers the benchmark's reading requests
# (:class:`RoleReader`): around every measured client the benchmark asks every worker for its
# counters and front-end CPU time (:class:`DisaggRunProbe`), so that the run's result holds
# per-role readings and the chain proof of both legs.

# Set by a driver that runs every role on one node without srun (the single-node driver in
# jenkins/scripts/perf/local/single_node_disagg/): no srun step gathers the generation worker's
# output into gen_server_<i>.log, so the generation worker's pytest writes it.
ENV_SINGLE_NODE = "TRTLLM_PERF_SINGLE_NODE"
# The perf sanity runtime of a disaggregated test and the benchmark modes the disagg arms take.
DISAGG_RUNTIME = "multi_node_disagg_server"
DISAGG_ARM_MODES = ("e2e", "gen_only")
# The modes in which perf sanity runs one server of a disaggregated test's YAML (the aggr-<mode>
# test ids), with the role whose settings that server takes under a disaggregated arm and the
# perf sanity runtime of the mode: ctx_only runs the context workers' configuration as an
# aggregated server, gen_only_no_context one generation worker behind a disaggregated server
# that has no context worker.
SINGLE_SERVER_ARM_MODES = {"ctx_only": ROLE_CTX, "gen_only_no_context": ROLE_GEN}
SINGLE_SERVER_RUNTIMES = {
    "ctx_only": "aggr_server",
    "gen_only_no_context": "gen_only_no_context_server",
}
SERVER_ROLE_OPTION = "--server_role"
_SERVER_ROLE_VALUES = {ROLE_CTX: "context", ROLE_GEN: "generation"}
# Environment variables of a worker that select or tune the KV-cache transfer.
_KV_TRANSFER_ENV_PREFIXES = ("UCX_", "NIXL_")
_KV_TRANSFER_ENV_NAMES = (
    "TRTLLM_USE_UCX_KVCACHE",
    "TRTLLM_USE_NIXL_KVCACHE",
    "TRTLLM_USE_MOONCAKE_KVCACHE",
    "TRTLLM_USE_MPI_KVCACHE",
    "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP",
    "TLLM_BENCHMARK_REQ_QUEUES_SIZE",
)
ENV_DISAGG_NO_RETRY = "TRTLLM_DISAGG_NO_RETRY"


def disagg_arm(arm: Arm | None) -> Arm | None:
    """``arm`` when it is a disaggregated arm, else ``None``."""
    return arm if arm is not None and arm.topology == TOPOLOGY_DISAGG else None


# The benchmark modes that get perf sanity's warmup pass (warmup_pass.py) only under a
# disaggregated arm. perf sanity sends none in them: their measured requests are then the first
# the workers serve after start-up, and pay the one-time costs of the first requests, which differ
# between an arm's workers in the chain and in process. The pass goes through the server the
# client calls, before the measured window, as in the other modes. In gen_only it opens the
# generation workers' benchmark fill gate (TLLM_BENCHMARK_REQ_QUEUES_SIZE), which opens once, so
# the measured requests then start on arrival instead of all together; TRTLLM_PERF_WARMUP_PASS=0
# keeps that mode's gated burst.
ARM_WARMUP_MODES = ("gen_only", "gen_only_no_context")


def arm_warmup_applies(
    benchmark_mode: str | None, environ: Mapping[str, str] | None = None
) -> bool:
    """Whether a run in ``benchmark_mode`` gets a warmup pass because a disaggregated arm is set.

    Raises:
        ArmError: As :func:`selected_arm`, for a mode of :data:`ARM_WARMUP_MODES`.
    """
    return benchmark_mode in ARM_WARMUP_MODES and disagg_arm(selected_arm(environ)) is not None


def role_arm(arm: Arm, role: str) -> RoleArm:
    """The settings of ``arm`` for the workers of ``role`` (``ctx`` or ``gen``).

    Raises:
        ArmError: ``arm`` has no settings for ``role``.
    """
    value = {ROLE_CTX: arm.ctx, ROLE_GEN: arm.gen}.get(role)
    if value is None:
        raise ArmError(f"arm {arm.name} has no {role!r} workers")
    return value


def role_view(arm: Arm, role: str) -> Arm:
    """``arm`` as the one-server arm the workers of ``role`` start with."""
    settings = role_arm(arm, role)
    return Arm(
        f"{arm.name}:{role}",
        arm.case,
        settings.chain,
        False,
        settings.frontends,
        settings.postprocess_workers,
    )


def single_server_view(arm: Arm | None, benchmark_mode: str | None) -> Arm | None:
    """The arm the one server of a perf sanity run in ``benchmark_mode`` starts with.

    Under a disaggregated arm, a run in a one-server mode (:data:`SINGLE_SERVER_ARM_MODES`)
    starts its server as the arm's settings of that mode's role say (:func:`role_view`);
    any other arm, or none, is returned as it is.
    """
    role = SINGLE_SERVER_ARM_MODES.get(benchmark_mode or "")
    if role is None or disagg_arm(arm) is None:
        return arm
    return role_view(arm, role)


def result_arm(name: str) -> Arm | None:
    """The one-server arm a result document names, or ``None`` if it names none.

    One of :data:`ARMS`, or a disaggregated arm's view of one role
    (``<arm>:<role>``, :func:`role_view`) that a one-server mode starts with.
    """
    if name in ARMS:
        return ARMS[name]
    base, _, role = name.rpartition(":")
    arm = DISAGG_ARMS.get(base)
    if arm is None or arm.topology != TOPOLOGY_DISAGG or role not in WORKER_ROLES:
        return None
    return role_view(arm, role)


def worker_role(serving_type: str) -> tuple[str, int]:
    """``(role, index)`` of a worker's ``DISAGG_SERVING_TYPE`` such as ``CTX_0`` or ``GEN_1``.

    Raises:
        ArmError: ``serving_type`` names no worker.
    """
    kind, _, index = serving_type.partition("_")
    role = {"CTX": ROLE_CTX, "GEN": ROLE_GEN}.get(kind)
    if role is None or not index.isdigit():
        raise ArmError(
            f"DISAGG_SERVING_TYPE={serving_type!r} names no context or generation worker"
        )
    return role, int(index)


def check_disagg_case(
    arm: Arm,
    *,
    runtime: str,
    benchmark_mode: str | None,
    time_breakdown: bool,
    numa_bind: bool,
    dwdp_size: int,
    worker_gpus: Mapping[str, int],
    gpus_per_node: int,
) -> None:
    """Refuse a perf sanity test a disaggregated arm cannot run.

    A disaggregated arm takes the ``e2e`` and ``gen_only`` modes (a case-03 arm ``e2e``
    only) of a disaggregated test whose workers each fit on one node (a case-03 worker may
    take whole nodes, :func:`ds_spans_nodes`), without the time
    breakdown, DWDP or ``numactl``: the chain starts its engine and front end
    with a ``trtllm-serve`` command line and cannot split one that a
    ``numactl`` prefix starts, and both arms of a comparison start the same way.
    A case-04 arm that starts both roles alike (``E4-base``, ``E4-chain`` and
    their lean variants) also takes the one-server modes ``ctx_only`` and
    ``gen_only_no_context`` (:data:`SINGLE_SERVER_ARM_MODES`) on their runtimes.

    Raises:
        ArmError: The arm is the direct replay tool's, or the test is not one of these.
    """
    where = f"{ENV_ARM}={arm.name}"
    if arm.topology == TOPOLOGY_DIRECT:
        raise ArmError(
            f"{where} is the direct replay tool's arm (openengine_chain/direct_replay_load.py); "
            "perf sanity does not run it"
        )
    single_server = benchmark_mode in SINGLE_SERVER_ARM_MODES
    if single_server and runtime != SINGLE_SERVER_RUNTIMES[benchmark_mode]:
        raise ArmError(
            f"{where}: the {benchmark_mode} mode runs on runtime "
            f"{SINGLE_SERVER_RUNTIMES[benchmark_mode]!r}, not {runtime!r}"
        )
    if not single_server and runtime != DISAGG_RUNTIME:
        raise ArmError(f"{where} applies to disaggregated cases only, not to runtime {runtime!r}")
    if benchmark_mode not in DISAGG_ARM_MODES and not single_server:
        modes = (*DISAGG_ARM_MODES, *SINGLE_SERVER_ARM_MODES)
        raise ArmError(
            f"{where} takes the benchmark modes {', '.join(modes)}, not {benchmark_mode!r}"
        )
    if single_server and (arm.case != CASE_DISAGG or arm.ctx != arm.gen):
        raise ArmError(
            f"{where}: the {benchmark_mode} mode runs one server as the arm's "
            f"{SINGLE_SERVER_ARM_MODES[benchmark_mode]} workers start; only a case-04 arm that "
            "starts both roles alike takes it"
        )
    if arm.case == CASE_DS and benchmark_mode != "e2e":
        # gen_only rests on trtllm-serve's request queue (TLLM_BENCHMARK_REQ_QUEUES_SIZE),
        # which the ds engine does not have.
        raise ArmError(f"{where} takes the e2e benchmark mode only, not {benchmark_mode!r}")
    if time_breakdown:
        raise ArmError(f"{where} does not take the time_breakdown modifier")
    if dwdp_size:
        raise ArmError(f"{where} does not take a DWDP context side")
    if numa_bind:
        raise ArmError(
            f"{where}: set slurm.numa_bind to false; the chain cannot start a worker whose "
            "command line starts with numactl, and both arms must start the same way"
        )
    for role, gpus in worker_gpus.items():
        if ds_spans_nodes(arm, gpus, gpus_per_node):
            continue
        if gpus_per_node and gpus > gpus_per_node:
            raise ArmError(
                f"{where}: a {role} worker takes {gpus} GPUs, more than one node's "
                f"{gpus_per_node}; the arms run each worker on one node"
            )


def prepare_worker(cmd: Sequence[str], arm: Arm, role: str, out_dir: str) -> ServerPlan:
    """The ``trtllm-serve`` command line of a ``role`` worker under a disaggregated arm.

    The role's settings go into a copy of the worker's YAML as
    :func:`prepare_server` writes it. A worker in the chain also gets
    ``--server_role``, by which the chain harness starts a disaggregated
    worker's engine and front end (the in-process worker of the comparison
    keeps the test's command line, which has none). A native ds worker
    (``ds_http``) runs dsv4-serve with the same options and ``--server_role``.

    Raises:
        ArmError: As :func:`prepare_server` raises, the command line gives
            its own ``--server_role``, or a native ds worker has no
            ``TRTLLM_TEST_DS_SERVE_CMD``.
        ValueError: A native ds worker's command line is not a ``trtllm-serve`` one.
    """
    view = role_view(arm, role)
    plan = prepare_server(cmd, view, out_dir)
    tokens = list(plan.cmd)
    if _option_values(tokens, SERVER_ROLE_OPTION):
        raise ArmError(
            f"arm {arm.name}: the {role} worker's command line gives {SERVER_ROLE_OPTION}"
        )
    if not view.in_process:
        tokens += [SERVER_ROLE_OPTION, _SERVER_ROLE_VALUES[role]]
    if role_arm(arm, role).ds_http:
        # dsv4-serve takes its role from --server_role, as $DS/dev/test_disagg.sh gives it.
        tokens = ds_http_command(tokens + [SERVER_ROLE_OPTION, _SERVER_ROLE_VALUES[role]])
    return ServerPlan(cmd=tuple(tokens), config=plan.config)


def worker_frontends(plan: ServerPlan) -> int:
    """The number of front ends a worker's plan configures (trtllm-serve's default 1 for port 0)."""
    value = _as_int(option_value(plan.cmd, "--num_serve_frontends"))
    if value is None:
        value = _as_int(plan.config.get("num_serve_frontends"))
    return value if isinstance(value, int) and value > 0 else 1


def worker_needs_port(view: Arm, plan: ServerPlan) -> bool:
    """Whether a worker starts on a port chosen in advance rather than on port 0.

    A worker in the chain does, so that its address is known to the chain
    proof after it stops; so does one with several front ends, which
    trtllm-serve refuses on port 0.
    """
    return not view.in_process or worker_frontends(plan) > 1


def parallel_gpus(config: Mapping[str, Any]) -> int:
    """GPUs one worker of ``config`` takes: tensor x pipeline x context parallel."""
    size = 1
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size"):
        value = _as_int(config.get(key))
        size *= value if isinstance(value, int) and value > 0 else 1
    return size


def kv_transfer_env(env: Mapping[str, str]) -> dict[str, str]:
    """The variables of ``env`` that select or tune the KV-cache transfer."""
    return {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(_KV_TRANSFER_ENV_PREFIXES) or key in _KV_TRANSFER_ENV_NAMES
    }


class LogMirror:
    """Copies what a log file gains to another file, as it grows.

    Under :data:`ENV_SINGLE_NODE`, the generation worker's
    ``gen_server_<i>.log`` that perf sanity reads the engine's per-iteration
    lines from is written this way (see :func:`gen_log_source`).
    """

    def __init__(self, source: Callable[[], str | None], dest: str) -> None:
        self._source = source
        self.dest = dest
        self._offset = 0
        self._path: str | None = None
        self.copied = 0

    def pump(self) -> int:
        """Copy what the source gained since the last call; return the bytes copied."""
        path = self._source()
        if not path:
            return 0
        if path != self._path:
            self._path, self._offset = path, 0
        try:
            with open(path, "rb") as src:
                src.seek(self._offset)
                data = src.read()
        except OSError:
            return 0
        if not data:
            return 0
        with open(self.dest, "ab") as dst:
            dst.write(data)
        self._offset += len(data)
        self.copied += len(data)
        return len(data)


def gen_log_source(server: "ArmServer", server_log: str) -> Callable[[], str | None]:
    """The log that holds a generation worker's engine output.

    The engine log the chain writes for a worker in the chain; otherwise the
    worker's own log, which its MPI workers inherit when no srun step gathers them.
    """
    if server.handle is not None:
        return lambda: server.handle.engine_log
    return lambda: server_log


def mirrors_gen_log(view: Arm, environ: Mapping[str, str] | None = None) -> bool:
    """Whether a generation worker writes its engine's output to ``gen_server_<i>.log`` itself.

    Only under :data:`ENV_SINGLE_NODE`. Otherwise the worker's srun step
    writes that file, without appending, so a second writer would overwrite
    lines.
    """
    env = os.environ if environ is None else environ
    return env.get(ENV_SINGLE_NODE) == "1"


class MirrorPump:
    """Pumps a :class:`LogMirror` in a thread every ``interval_s`` until :meth:`stop`.

    perf sanity's ``gen_only_no_context`` run starts its generation worker in the
    test's own pytest, and that worker's ``gen_server_<i>.log`` has no other writer;
    under an arm with a chain the engine's output reaches it through this pump.
    :meth:`pump` copies at once (before a byte offset of the log is read).
    """

    def __init__(self, mirror: LogMirror, interval_s: float = 1.0) -> None:
        self.mirror = mirror
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def pump(self) -> int:
        with self._lock:
            return self.mirror.pump()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="perf-arm-gen-log", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.pump()

    def stop(self) -> None:
        """Stop the thread, then copy what is left."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.pump()


_SPAWNED_MARK = "multiprocessing.spawn"
_SPAWN_HELPERS = ("multiprocessing.resource_tracker", "multiprocessing.forkserver")


def _read_cmdline(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return None


# Set by trtllm-llmapi-launch: the executor runs in the launcher's MPI ranks, not under the process.
_SPAWN_PROXY_ENV = "TLLM_SPAWN_PROXY_PROCESS"


def _process_argv(pid: int) -> list[str] | None:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    return [arg.decode("utf-8", "replace") for arg in raw.split(b"\0") if arg]


def _process_environ(pid: int) -> dict[str, str] | None:
    """Process ``pid``'s environment as it started, or ``None`` when it cannot be read."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    env = {}
    for entry in raw.split(b"\0"):
        key, sep, value = entry.decode("utf-8", "replace").partition("=")
        if sep:
            env[key] = value
    return env


def _process_config(pid: int) -> dict[str, Any] | None:
    """The YAML of the last configuration option on process ``pid``'s command line, if readable."""
    argv = _process_argv(pid)
    path = None
    for i, arg in enumerate(argv or ()):
        name, sep, value = arg.partition("=")
        if name in CONFIG_OPTIONS:
            path = value if sep else (argv[i + 1] if i + 1 < len(argv) else None)
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    return config if isinstance(config, dict) else None


def count_spawned(pids: Iterable[int], cmdline: Callable[[int], str | None] = _read_cmdline) -> int:
    """How many of ``pids`` are processes ``multiprocessing`` spawned, its helpers left out.

    The post-processing workers of a front end are such processes.
    """
    count = 0
    for pid in pids:
        args = cmdline(pid) or ""
        if _SPAWNED_MARK in args and not any(helper in args for helper in _SPAWN_HELPERS):
            count += 1
    return count


def _read_ppid(pid: int) -> int | None:
    """The parent PID of ``pid``, or ``None`` if it is gone."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read()
    except OSError:
        return None
    # Field 4 of proc(5), the second after the parenthesised command.
    fields = stat[stat.rfind(b")") + 2 :].split()
    return int(fields[1]) if len(fields) > 1 else None


def count_engine_postproc(
    root: int,
    reader: "CpuReader",
    *,
    ppid: Callable[[int], int | None] | None = None,
    cmdline: Callable[[int], str | None] | None = None,
) -> int:
    """How many post-processing workers the engine processes in ``root``'s tree forked.

    An in-process server's executor starts its post-processing workers in its
    leader rank, an engine process, from a process pool that forks them. A
    forked worker runs its parent's command line, so :func:`count_spawned` does
    not see it, and it is in the engine's tree, which
    :meth:`CpuReader.frontend_pids` leaves out. It is counted here as a child
    of an engine process that runs the same command line and is not an engine
    process itself.
    """
    ppid = ppid or _read_ppid
    cmdline = cmdline or _read_cmdline
    pids = reader.tree(root)
    on_gpu = reader.gpu()
    if on_gpu is None:
        return 0
    engine = pids & on_gpu
    count = 0
    for pid in pids - engine:
        parent = ppid(pid)
        if parent in engine:
            args = cmdline(pid)
            if args and args == cmdline(parent):
                count += 1
    return count


# ----------------------------------------------------------------------------- per-role readings

PHASE_BEFORE = "before"
PHASE_AFTER = "after"
PHASES = (PHASE_BEFORE, PHASE_AFTER)
# How long the benchmark waits for every worker's reading of one phase.
READINGS_TIMEOUT_S = 120.0
_READINGS_POLL_S = 0.2
_HEALTH_TIMEOUT_S = 2.0
_REQUEST = re.compile(r"^req-([0-9a-f]+)-(\d+)-(before|after)\.json$")


def readings_dir(
    test_output_dir: str, server_idx: int, environ: Mapping[str, str] | None = None
) -> str:
    """The directory the benchmark and the workers of one server index exchange readings through."""
    env = os.environ if environ is None else environ
    return os.path.join(
        test_output_dir, f"perf-arm-{env.get('SLURM_JOB_ID', 'local')}-{server_idx}"
    )


def request_path(directory: str, session: str, seq: int, phase: str) -> str:
    return os.path.join(directory, f"req-{session}-{seq}-{phase}.json")


def reading_path(directory: str, session: str, seq: int, phase: str, worker: str) -> str:
    return os.path.join(directory, f"read-{session}-{seq}-{phase}-{worker}.json")


def write_json_atomic(path: str, doc: Any) -> None:
    """Write ``doc`` to ``path`` so that a reader never sees a partial file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _counters_json(counters: Any) -> dict[str, Any] | None:
    if counters is None:
        return None
    return {
        "frontend": counters.frontend,
        "engine": counters.engine,
        "serve": None if counters.serve is None else dict(counters.serve),
        "engine_v4": None if counters.engine_v4 is None else dict(counters.engine_v4),
    }


def _health_ok(address: str) -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{address}/health", timeout=_HEALTH_TIMEOUT_S) as r:
            return r.status == 200
    except (OSError, urllib.error.URLError, ValueError):
        return False


class RoleReader:
    """Answers the benchmark's reading requests for one worker, in a thread of its pytest.

    For each request it reads the worker's counters (a worker in the chain),
    front-end processes and CPU time, and writes one JSON next to the
    request. The reading after a measured client also holds the worker's own
    part of the chain proof, judged as one node of a ``cross_node`` test
    (``openengine_chain.proof.judge_disagg``): the benchmark merges the parts.
    The thread also times the worker's start until its ``/health`` answers and,
    with a :class:`LogMirror`, writes ``gen_server_<i>.log``.
    """

    def __init__(
        self,
        *,
        server: "ArmServer",
        view: Arm,
        role: str,
        worker: str,
        directory: str,
        address: Callable[[], str | None],
        plan: ServerPlan,
        env: Mapping[str, str],
        launched_at: float,
        mirror: LogMirror | None = None,
        reader: CpuReader | None = None,
    ) -> None:
        self.server, self.view, self.role, self.worker = server, view, role, worker
        self.directory = directory
        self._address = address
        self.plan = plan
        self.env = dict(env)
        self.launched_at = launched_at
        self.mirror = mirror
        self.reader = reader or CpuReader()
        self.startup_s: float | None = None
        self._done: set[str] = set()
        self._before: dict[tuple[str, int], tuple[Any, RunProbe]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._loop, name=f"perf-arm-{self.worker}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop answering; copy what the log mirror has not copied yet.

        A thread still busy after the wait keeps the mirror to itself.
        """
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                self.errors.append("the reader thread did not stop within 30 s")
                return
        if self.mirror is not None:
            self.mirror.pump()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception as error:  # noqa: BLE001 - one bad poll must not stop the answers
                self.errors.append(repr(error))
            self._stop.wait(_READINGS_POLL_S)

    def poll(self) -> None:
        """One round: time the start, copy the log, answer new requests."""
        if self.mirror is not None:
            self.mirror.pump()
        if self.startup_s is None:
            address = self._address()
            if address and _health_ok(address):
                self.startup_s = round(time.monotonic() - self.launched_at, 3)
        try:
            names = os.listdir(self.directory)
        except OSError:
            return
        requests = []
        for name in names:
            match = _REQUEST.match(name)
            if match is not None and name not in self._done:
                requests.append((int(match.group(2)), match.group(3) != PHASE_BEFORE, name, match))
        # In the order the benchmark asked: by client, the reading before it first.
        for _, _, name, match in sorted(requests):
            self._done.add(name)
            session, seq, phase = match.group(1), int(match.group(2)), match.group(3)
            try:
                doc = self.read(session, seq, phase)
            except Exception as error:  # noqa: BLE001 - the benchmark still gets an answer
                doc = {
                    "worker": self.worker,
                    "role": self.role,
                    "phase": phase,
                    "seq": seq,
                    "notes": [f"reading failed: {error!r}"],
                }
            # The engine lines of the client just measured go to the log before the answer.
            if self.mirror is not None:
                self.mirror.pump()
            write_json_atomic(reading_path(self.directory, session, seq, phase, self.worker), doc)

    def read(self, session: str, seq: int, phase: str) -> dict[str, Any]:
        """This worker's reading of one phase of one measured client."""
        notes: list[str] = []
        address = self._address()
        doc: dict[str, Any] = {
            "worker": self.worker,
            "role": self.role,
            "chain": self.view.chain,
            "phase": phase,
            "seq": seq,
            "address": address,
            "pid": os.getpid(),
            "node": socket.gethostname(),
            # This node's clock, which the engine's and front end's timestamps also use.
            "t_ns": time.time_ns(),
            "startup_s": self.startup_s,
            "speculative": _speculative(self.plan.config),
            "gpus": parallel_gpus(self.plan.config),
            "frontends": {"configured": worker_frontends(self.plan), "started": None},
            "postproc_workers": {
                "configured": _as_int(self.plan.config.get("num_postprocess_workers")),
                "started": None,
            },
            "stream_interval": self.plan.config.get("stream_interval"),
            "kv_transfer": self._kv_transfer(notes),
            "alignment": worker_alignment(self.plan, self.env),
            "counters": None,
            "frontend_cpu_s": None,
            "leg": None,
            "notes": notes,
        }
        if self.view.case == CASE_DS:
            doc["ds_instance"] = ds_instance_reading(self.server, self.plan, self.env, notes)
        accepted = self.env.get(ENV_SPEC_DECODE_ACCEPTED)
        if accepted and doc["alignment"]["spec_decode_force_accepted"] is None:
            notes.append(f"{ENV_SPEC_DECODE_ACCEPTED}={accepted!r} is not a number")
        pids, note = self.reader.frontend_pids(
            self.server.frontend_pid, in_process=self.view.in_process
        )
        notes.append(note)
        if pids is not None:
            doc["postproc_workers"]["started"] = count_spawned(pids)
            if self.view.in_process:
                forked = count_engine_postproc(self.server.frontend_pid, self.reader)
                notes.append(f"{forked} post-processing workers forked by the engine's processes")
                doc["postproc_workers"]["started"] += forked
        doc["frontends"]["started"] = self._frontends_started(notes)
        counters = (
            _read_counters(self.server.handle, notes) if self.server.handle is not None else None
        )
        doc["counters"] = _counters_json(counters)
        key = (session, seq)
        if phase == PHASE_BEFORE:
            probe = RunProbe(
                self.server,
                self.reader,
                self.reader.snapshot(pids) if pids is not None else None,
                counters,
                [],
                self.reader.now(),
            )
            self._before[key] = (counters, probe)
            return doc
        before = self._before.pop(key, None)
        if before is None:
            notes.append("no reading of this worker before the client; nothing to compare with")
            return doc
        start, probe = before
        doc["frontend_cpu_s"] = probe.frontend_cpu_s()
        notes.extend(probe.notes)
        if self.server.handle is not None and start is not None:
            doc["leg"] = self._judge_leg(start, address, notes)
        return doc

    def _kv_transfer(self, notes: list[str]) -> dict[str, Any]:
        """The KV-transfer settings of the process that runs this worker's engine, as it runs.

        Read from ``/proc``: the configuration file on its command line and its
        environment. That process is a chained worker's engine, whose
        configuration the harness rewrites (resolving backend DEFAULT, for
        one), else the worker's own server process. What ``/proc`` does not
        give, the worker's planned configuration and environment stand in for,
        with a note.
        """
        handle = self.server.handle
        process = handle.engine if handle is not None else self.server.proc
        pid = process.pid if process is not None else None
        config = _process_config(pid) if pid is not None else None
        if config is None:
            notes.append(f"KV transfer: no configuration file read from process {pid}; planned one")
            config = self.plan.config
        env = _process_environ(pid) if pid is not None else None
        if env is None:
            notes.append(f"KV transfer: environment of process {pid} not readable; planned one")
            env = self.env
        if env.get(_SPAWN_PROXY_ENV) == "1":
            notes.append(
                f"KV transfer: {_SPAWN_PROXY_ENV}=1, so the executor runs in the launcher's MPI "
                "ranks, whose environment is not read"
            )
        transceiver = config.get("cache_transceiver_config") or {}
        if not isinstance(transceiver, Mapping):
            transceiver = {}
        return {
            "backend": transceiver.get("backend"),
            "runtime": transceiver.get("transceiver_runtime"),
            "env": kv_transfer_env(env),
        }

    def _frontends_started(self, notes: list[str]) -> int | None:
        from openengine_chain import proof

        frontend = (
            self.server.handle.frontend if self.server.handle is not None else self.server.proc
        )
        try:
            return proof.frontend_count(types.SimpleNamespace(frontend=frontend))
        except (AssertionError, OSError) as error:
            notes.append(f"front ends not counted: {error}")
            return None

    def _judge_leg(self, start: Any, address: str | None, notes: list[str]) -> dict[str, Any]:
        """This worker's part of the chain proof, as one node of a ``cross_node`` test."""
        from openengine_chain import proof

        handle = self.server.handle
        try:
            info = proof.engine_info(handle)
        except Exception as error:  # noqa: BLE001 - an unreadable engine only makes the proof incomplete
            notes.append(f"reading the engine's server info failed: {error!r}")
            info = None
        topology = proof.check_topology(handle)
        inprocess = proof.inprocess_workers(os.getpid())
        rule = proof.LegRule(leg="cross_node")
        normalized = proof.normalize_address(address) if address else None
        for attempt in range(_SETTLE_TRIES):
            end = _read_counters(handle, notes)
            first = proof.counting_start(start, end)
            leg = proof.Leg(self.role, normalized, first, end, info, topology)
            result = proof.judge_disagg([leg], None, inprocess, rule)
            settled = not any(
                name in ("CH-2", "CH-4") and role is not None and c.status == proof.FAIL
                for name, role, c in result.checks
            )
            if settled or attempt + 1 == _SETTLE_TRIES:
                break
            time.sleep(_SETTLE_INTERVAL_S)
        counts = dict(result.counts)
        counts["node_names"] = proof.node_names()
        return {
            "status": result.status,
            "roles": dict(result.roles),
            "counts": counts,
            "details": result.details(),
        }


def orchestrator_pid_path(directory: str) -> str:
    """Where the disaggregated server's pytest writes the server's pid."""
    return os.path.join(directory, "orchestrator.pid")


def publish_orchestrator_pid(directory: str, pid: int) -> None:
    write_json_atomic(orchestrator_pid_path(directory), {"pid": pid, "node": socket.gethostname()})


def _orchestrator_pid(directory: str) -> int | None:
    """The disaggregated server's pid when it runs on this node (its CPU time is read here)."""
    try:
        with open(orchestrator_pid_path(directory), encoding="utf-8") as f:
            doc = json.load(f)
        pid = int(doc["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return pid if doc.get("node") == socket.gethostname() else None


def _tree_ticks(root: int | None) -> int | None:
    """CPU ticks of the live processes of ``root``'s tree, or ``None`` without one."""
    if root is None:
        return None
    total = None
    for pid in _process_tree(root):
        value = read_cpu_ticks(pid)
        if value is not None:
            total = (total or 0) + value[1]
    return total


def _children_cpu_s() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


@dataclass
class DisaggRunProbe:
    """The benchmark's side of the readings of one measured client.

    :meth:`start` asks every worker for its reading and waits for them, reads
    the disaggregated server's counters and CPU time and the client's CPU so
    far; :meth:`stop` does the same after the client. :meth:`result` builds the
    run's result document.
    """

    arm: Arm
    directory: str
    workers: tuple[str, ...]
    orchestrator_url: str
    session: str
    seq: int
    readings: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    orchestrator: dict[str, Any] = field(default_factory=dict)
    orch_ticks: dict[str, int | None] = field(default_factory=dict)
    client_cpu: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    timeout: float = READINGS_TIMEOUT_S
    # time.time_ns() when the measured client starts (after the readings before it) and ends.
    window: dict[str, int] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        arm: Arm,
        *,
        directory: str,
        workers: Sequence[str],
        orchestrator_url: str,
        seq: int,
        session: str | None = None,
        timeout: float = READINGS_TIMEOUT_S,
    ) -> "DisaggRunProbe":
        import uuid

        probe = cls(
            arm,
            directory,
            tuple(workers),
            orchestrator_url,
            session or uuid.uuid4().hex[:16],
            seq,
            timeout=timeout,
        )
        probe._phase(PHASE_BEFORE)
        return probe

    def stop(self) -> None:
        """Take the readings after the client."""
        self._phase(PHASE_AFTER)

    def _phase(self, phase: str) -> None:
        if phase == PHASE_AFTER:
            self.window[phase] = time.time_ns()
            self.client_cpu[phase] = _children_cpu_s()
        write_json_atomic(
            request_path(self.directory, self.session, self.seq, phase),
            {"session": self.session, "seq": self.seq, "phase": phase, "t": time.time()},
        )
        self.readings[phase] = self._wait(phase)
        self.orchestrator[phase] = self._read_orchestrator()
        self.orch_ticks[phase] = _tree_ticks(_orchestrator_pid(self.directory))
        if phase == PHASE_BEFORE:
            self.client_cpu[phase] = _children_cpu_s()
            self.window[phase] = time.time_ns()

    def _wait(self, phase: str) -> dict[str, dict[str, Any]]:
        found: dict[str, dict[str, Any]] = {}
        deadline = time.monotonic() + self.timeout
        while True:
            for worker in self.workers:
                if worker in found:
                    continue
                path = reading_path(self.directory, self.session, self.seq, phase, worker)
                try:
                    with open(path, encoding="utf-8") as f:
                        found[worker] = json.load(f)
                except (OSError, ValueError):
                    continue
            if len(found) == len(self.workers) or time.monotonic() > deadline:
                break
            time.sleep(_READINGS_POLL_S)
        missing = sorted(set(self.workers) - set(found))
        if missing:
            self.notes.append(
                f"no {phase} reading of {', '.join(missing)} within {self.timeout:.0f} s"
            )
        return found

    def _read_orchestrator(self) -> dict[str, int] | None:
        from openengine_chain import proof

        try:
            reading = proof.read_orchestrator(proof.Orchestrator(0, self.orchestrator_url, ""))
        except (OSError, ValueError) as error:
            self.notes.append(f"reading the disaggregated server's counters failed: {error!r}")
            return None
        return dict(reading.counts)

    def _role_workers(self, role: str) -> list[str]:
        return [w for w in self.workers if worker_role(w)[0] == role]

    def roles_block(self, output_tokens: int | None) -> dict[str, Any]:
        """The result's ``roles``: each role's configuration, front ends and front-end CPU time."""
        roles: dict[str, Any] = {}
        for role in WORKER_ROLES:
            workers = self._role_workers(role)
            after = [self.readings.get(PHASE_AFTER, {}).get(w) for w in workers]
            seen = [r for r in after if r is not None]
            first = seen[0] if seen else {}
            cpu = [r.get("frontend_cpu_s") for r in seen]
            cpu_s = None if not seen or None in cpu or len(seen) != len(workers) else sum(cpu)
            startups = [r.get("startup_s") for r in seen]
            roles[role] = {
                "chain": role_arm(self.arm, role).chain,
                "instances": len(workers),
                "gpus_per_instance": first.get("gpus"),
                "frontends": _summed(seen, "frontends"),
                "postproc_workers": _summed(seen, "postproc_workers"),
                "stream_interval": first.get("stream_interval"),
                "frontend_cpu_s": cpu_s,
                "frontend_cpu_us_per_token": (
                    cpu_s * 1e6 / output_tokens if cpu_s is not None and output_tokens else None
                ),
                # Not measured: the per-thread CPU time of the front ends is not read.
                "busiest_thread_cores": None,
                "startup_s": None if not seen or None in startups else max(startups),
            }
        return roles

    def chain_proof(self, members: Mapping[str, Sequence[str]] | None) -> dict[str, Any] | None:
        """The result's ``chain_proof``: per-role CH-2 to CH-5 counts and the merged judgment.

        Each worker in the chain judged its own leg (:meth:`RoleReader._judge_leg`);
        here the legs are merged as the nodes of a ``cross_node`` test, with the
        disaggregated server's counters and its worker list ``members`` (the
        ``urls`` of its YAML in the static URL form). An arm with one role in
        the chain proves that role's legs only.
        """
        from openengine_chain import proof

        chained = [r for r in WORKER_ROLES if not role_arm(self.arm, r).in_process]
        if not chained:
            return None
        legs = {}
        for role in chained:
            for worker in self._role_workers(role):
                reading = self.readings.get(PHASE_AFTER, {}).get(worker) or {}
                legs[worker] = reading.get("leg")
        statuses = [proof.INCOMPLETE if leg is None else leg["status"] for leg in legs.values()]
        details = [f"{w}: {d}" for w, leg in legs.items() for d in (leg or {}).get("details", [])]
        details += [f"{w}: no reading of its leg" for w, leg in legs.items() if leg is None]
        orch = self._orchestrator_delta()
        nodes = [None if leg is None else dict(leg["counts"]) for leg in legs.values()]
        merged = None
        if len(chained) == len(WORKER_ROLES):
            if nodes and nodes[0] is not None:
                nodes[0]["orchestrator"] = (
                    None
                    if orch is None
                    else {
                        "counts": orch,
                        "members": None
                        if members is None
                        else {r: list(members.get(r, ())) for r in WORKER_ROLES},
                        "stale": False,
                    }
                )
            merged = proof.judge_cross_node(nodes)
            statuses.append(merged.status)
            details += [f"merged: {d}" for d in merged.details()]
        role_status = {
            role: proof.worst(
                [
                    (legs[w] or {}).get("roles", {}).get(role, proof.INCOMPLETE)
                    for w in self._role_workers(role)
                ]
                + ([merged.roles[role]] if merged is not None else [])
            )
            if role in chained
            else None
            for role in WORKER_ROLES
        }
        sums = merged.counts["roles"] if merged is not None and merged.counts else _leg_sums(legs)
        return _proof_block(sums, orch, proof.worst(statuses), role_status, chained, details)

    def _orchestrator_delta(self) -> dict[str, int] | None:
        before = self.orchestrator.get(PHASE_BEFORE)
        after = self.orchestrator.get(PHASE_AFTER)
        if before is None or after is None:
            return None
        return {key: after.get(key, 0) - before.get(key, 0) for key in after}


def _summed(readings: Sequence[Mapping[str, Any]], key: str) -> dict[str, int | None]:
    """``configured`` and ``started`` of ``key`` summed over the instances of a role."""
    out: dict[str, int | None] = {}
    for part in ("configured", "started"):
        values = [(r.get(key) or {}).get(part) for r in readings]
        out[part] = None if not values or None in values else sum(values)
    return out


def _leg_sums(legs: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    """Per-role counts summed over the legs that were read (a missing leg leaves ``None``)."""
    sums: dict[str, Any] = {}
    for role in WORKER_ROLES:
        counts: dict[str, int | None] = {}
        for leg in legs.values():
            part = None if leg is None else leg["counts"]["roles"][role]
            if part is None or not part["legs"]:
                continue
            for key, value in part["counts"].items():
                old = counts.get(key, 0)
                counts[key] = None if old is None or value is None else old + value
        sums[role] = {"counts": counts}
    return sums


def _proof_block(
    sums: Mapping[str, Any],
    orch: Mapping[str, int] | None,
    status: str,
    role_status: Mapping[str, str | None],
    chained: Sequence[str],
    details: Sequence[str],
) -> dict[str, Any]:
    from openengine_chain import proof

    def count(role: str, key: str) -> int | None:
        return (sums.get(role) or {}).get("counts", {}).get(key)

    def http(role: str) -> int | None:
        values = [count(role, key) for key in proof.SERVE_HTTP_RECEIVED.values()]
        return None if None in values else sum(values)

    accepted = proof.ENGINE_ACCEPTED_BUCKETS
    ctx = gen = None
    if proof.CTX in chained:
        ctx = {
            "generate_sent": count(proof.CTX, "sent"),
            "generate_received": count(proof.CTX, "received"),
            "accepted_context_only": count(proof.CTX, accepted[proof.BUCKET_CONTEXT_ONLY]),
            "prefill_ready_sent": count(proof.CTX, proof.ENGINE_PREFILL_READY_SENT),
            "prefill_ready_received": count(proof.CTX, proof.SERVE_PREFILL_READY_RECEIVED),
            "http_received": http(proof.CTX),
            "http_rejected": count(proof.CTX, proof.SERVE_HTTP_REJECTED),
        }
    if proof.GEN in chained:
        gen = {
            "generate_sent": count(proof.GEN, "sent"),
            "generate_received": count(proof.GEN, "received"),
            "accepted_generation_only": count(proof.GEN, accepted[proof.BUCKET_GENERATION_ONLY]),
            "accepted_aggregated": count(proof.GEN, accepted[proof.BUCKET_AGGREGATED]),
            "http_received": http(proof.GEN),
            "http_rejected": count(proof.GEN, proof.SERVE_HTTP_REJECTED),
        }
    handoffs = ctx["prefill_ready_sent"] if ctx else None
    taken = gen["accepted_generation_only"] if gen else None
    orphans = None if handoffs is None or taken is None else handoffs - taken
    return {
        "ctx": ctx,
        "gen": gen,
        "orch_requests": {
            role: None if orch is None else orch.get(f"{role}.total_requests")
            for role in proof.ROLES
        },
        "handoffs": handoffs,
        "orphan_handoffs": orphans,
        # Both legs merged as the nodes of a cross_node test; one leg alone when only it is chained.
        "leg_expectation": "cross_node"
        if len(chained) == len(proof.ROLES)
        else f"one_leg:{chained[0]}",
        "status": {"overall": status, **dict(role_status)},
        "details": list(details),
    }


def static_members(config_path: str) -> dict[str, list[str]] | None:
    """The worker ``urls`` of a disaggregated server's YAML, by role; ``None`` if unreadable."""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return {
            ROLE_CTX: [str(u) for u in config["context_servers"].get("urls") or []],
            ROLE_GEN: [str(u) for u in config["generation_servers"].get("urls") or []],
        }
    except (OSError, yaml.YAMLError, KeyError, TypeError, AttributeError):
        return None


def _orchestrator_block(config_path: str, cpu_s: float | None) -> dict[str, Any]:
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        config = {}

    def router(key: str) -> Any:
        section = config.get(key) or {}
        value = section.get("router") if isinstance(section, Mapping) else None
        return value.get("type") if isinstance(value, Mapping) else None

    return {
        "kind": "trtllm",
        "schedule_style": config.get("schedule_style"),
        "router": {ROLE_CTX: router("context_servers"), ROLE_GEN: router("generation_servers")},
        "num_workers": config.get("num_workers"),
        "cpu_s": cpu_s,
        "config": {
            key: config.get(key)
            for key in ("gen_strip_message_history", "gen_tokids_ctxbytes", "max_retries")
            if key in config
        },
    }


def worker_alignment(plan: ServerPlan, env: Mapping[str, str]) -> dict[str, Any]:
    """The settings of one worker that both arms of a comparison must share."""
    config = plan.config
    kv = config.get("kv_cache_config") or {}

    def setting(key: str) -> Any:
        value = option_value(plan.cmd, f"--{key}")
        return value if value is not None else config.get(key)

    return {
        "enable_block_reuse": kv.get("enable_block_reuse") if isinstance(kv, Mapping) else None,
        "reasoning_parser": setting("reasoning_parser"),
        "tool_parser": setting("tool_parser"),
        "disable_mm_encoder": config.get("disable_mm_encoder"),
        "return_perf_metrics": config.get("return_perf_metrics"),
        "spec_decode_force_accepted": _float_or_none(env.get(ENV_SPEC_DECODE_ACCEPTED)),
    }


# The result's alignment block (TH-DSG-6): a role's value is ``{"ctx": ..., "gen": ...}``
# where each worker has one; orig_elapsed_s_forwarded is not observed per run (null).
DISAGG_ALIGNMENT_KEYS = (
    "enable_block_reuse",
    "reasoning_parser",
    "tool_parser",
    "disable_mm_encoder",
    "spec_decode_force_accepted",
    "disagg_no_retry",
    "return_perf_metrics",
    "gen_strip_message_history",
    "gen_tokids_ctxbytes",
    "orig_elapsed_s_forwarded",
    "input_form",
)
_WORKER_ALIGNMENT_KEYS = (
    "enable_block_reuse",
    "reasoning_parser",
    "tool_parser",
    "disable_mm_encoder",
    "spec_decode_force_accepted",
    "return_perf_metrics",
)
_DEVICE_STEP_LINES = {
    "mean": re.compile(r"^Average Per Iter Device Step Time \(ms\):\s+(-?[\d.]+)", re.MULTILINE),
    "median": re.compile(r"^Median Per Iter Device Step Time \(ms\):\s+(-?[\d.]+)", re.MULTILINE),
}


def _role_values(probe: DisaggRunProbe, key: str) -> dict[str, Any]:
    """Each role's value of a worker alignment key (a list when its workers differ)."""
    return _role_values_of(
        probe.readings.get(PHASE_AFTER, {}), probe, lambda r: (r.get("alignment") or {}).get(key)
    )


def build_disagg_result(
    probe: DisaggRunProbe,
    *,
    output: str,
    client_cmd: Sequence[str],
    test_id: str,
    benchmark_mode: str,
    orchestrator_config: str,
    orchestrator_env: Mapping[str, str],
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One measured client's result document under a disaggregated arm (TH-DSG-6).

    Raises:
        ArmError: ``output`` is refused by :func:`check_benchmark_output`, or
            ``TRTLLM_PERF_RUN`` is not a positive integer.
    """
    check_benchmark_output(output)
    env = os.environ if environ is None else environ
    metrics, output_tokens = parse_metrics(output)
    metrics["gen_device_step_ms"] = {
        stat: (float(m.group(1)) if (m := pattern.search(output)) else None)
        for stat, pattern in _DEVICE_STEP_LINES.items()
    }
    notes = list(probe.notes)
    after = probe.readings.get(PHASE_AFTER, {})
    for worker, reading in sorted(after.items()):
        notes.extend(f"{worker}: {n}" for n in reading.get("notes", []))
    gen_readings = [after.get(w) or {} for w in probe._role_workers(ROLE_GEN)]
    if not checks_decoded_tokens(probe.arm) and any(r.get("speculative") for r in gen_readings):
        notes.append(_DECODED_TOKENS_NOTE)
    ticks = probe.orch_ticks
    orch_cpu = None
    if ticks.get(PHASE_BEFORE) is not None and ticks.get(PHASE_AFTER) is not None:
        orch_cpu = (ticks[PHASE_AFTER] - ticks[PHASE_BEFORE]) / os.sysconf("SC_CLK_TCK")
        if orch_cpu < 0:
            # A process of its tree exited in between and took its ticks along.
            notes.append("orchestrator CPU time not measured: its process tree lost processes")
            orch_cpu = None
    client_cpu = None
    if PHASE_BEFORE in probe.client_cpu and PHASE_AFTER in probe.client_cpu:
        client_cpu = round(probe.client_cpu[PHASE_AFTER] - probe.client_cpu[PHASE_BEFORE], 3)
    orch = _orchestrator_block(orchestrator_config, orch_cpu)
    proof_block = probe.chain_proof(static_members(orchestrator_config))
    if proof_block and (proof_block["orphan_handoffs"] or 0) > 0:
        notes.append(
            f"{proof_block['orphan_handoffs']} orphan handoffs in the measured window: the run "
            "is affected by KV cache held for them and is not a formal comparison"
        )
    alignment = {key: _role_values(probe, key) for key in _WORKER_ALIGNMENT_KEYS}
    alignment["disagg_no_retry"] = orchestrator_env.get(ENV_DISAGG_NO_RETRY) == "1"
    for key in ("gen_strip_message_history", "gen_tokids_ctxbytes"):
        alignment[key] = orch["config"].get(key)
    alignment["orig_elapsed_s_forwarded"] = None
    alignment["input_form"] = dataset_kind(client_cmd)
    kv_transfer = {
        key: _role_values_of(after, probe, lambda r, k=key: (r.get("kv_transfer") or {}).get(k))
        for key in ("backend", "runtime", "env")
    }
    concurrency = _int_or_none(
        option_value(client_cmd, "--max-concurrency") or option_value(client_cmd, "--concurrency")
    )
    doc = {
        "case": probe.arm.case,
        "arm": probe.arm.name,
        "topology": probe.arm.topology,
        "test_id": test_id,
        "benchmark_mode": benchmark_mode,
        "concurrency": concurrency,
        "dataset": dataset_kind(client_cmd),
        "run": run_number(env),
        "metrics": metrics,
        "output_tokens": output_tokens,
        "roles": probe.roles_block(output_tokens),
        "orch": orch,
        "client": {"cpu_s": client_cpu},
        "kv_transfer": kv_transfer,
        "chain_proof": proof_block,
        "first_output": first_output_block(probe, concurrency),
        "alignment": {key: alignment[key] for key in DISAGG_ALIGNMENT_KEYS},
        "versions": {"trtllm": trtllm_sha(env), "ds": None},
        "nodes": sorted(
            {r.get("node") for r in after.values() if r.get("node")} | {socket.gethostname()}
        ),
        "job_id": env.get("SLURM_JOB_ID") or None,
        "notes": notes,
    }
    if probe.arm.case == CASE_DS:
        add_ds_blocks(
            doc,
            probe,
            output=output,
            orchestrator_config=orchestrator_config,
            orchestrator_env=orchestrator_env,
            environ=env,
        )
    return doc


def _role_values_of(
    readings: Mapping[str, Mapping[str, Any]],
    probe: DisaggRunProbe,
    getter: Callable[[Mapping[str, Any]], Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role in WORKER_ROLES:
        values = [getter(readings[w]) if w in readings else None for w in probe._role_workers(role)]
        distinct = [v for i, v in enumerate(values) if v not in values[:i]]
        out[role] = distinct[0] if len(distinct) == 1 else distinct
    return out


DISAGG_RESULT_KEYS = (
    "case",
    "arm",
    "topology",
    "test_id",
    "benchmark_mode",
    "concurrency",
    "dataset",
    "run",
    "metrics",
    "output_tokens",
    "roles",
    "orch",
    "client",
    "kv_transfer",
    "chain_proof",
    "first_output",
    "alignment",
    "versions",
    "nodes",
    "job_id",
    "notes",
)
_ROLE_KEYS = (
    "chain",
    "instances",
    "gpus_per_instance",
    "frontends",
    "postproc_workers",
    "stream_interval",
    "frontend_cpu_s",
    "frontend_cpu_us_per_token",
    "busiest_thread_cores",
    "startup_s",
)
_PROOF_KEYS = (
    "ctx",
    "gen",
    "orch_requests",
    "handoffs",
    "orphan_handoffs",
    "leg_expectation",
    "status",
    "details",
)
_PROOF_ROLE_KEYS = {
    ROLE_CTX: (
        "generate_sent",
        "generate_received",
        "accepted_context_only",
        "prefill_ready_sent",
        "prefill_ready_received",
        "http_received",
        "http_rejected",
    ),
    ROLE_GEN: (
        "generate_sent",
        "generate_received",
        "accepted_generation_only",
        "accepted_aggregated",
        "http_received",
        "http_rejected",
    ),
}
_PROOF_STATUSES = ("ok", "fail", "incomplete", "incomplete(killed)")


def _count_or_null(value: Any) -> bool:
    return value is None or _is_count(value)


def validate_disagg_result(doc: Any) -> list[str]:
    """Problems with a disaggregated arm's result document; empty when it matches the schema."""
    if not isinstance(doc, dict):
        return [f"result is a {type(doc).__name__}, not an object"]
    # A case-03 disaggregated result also holds DS_RESULT_KEYS (validate_ds_blocks); a direct
    # one has no orchestrator or client to describe.
    ds_blocks = doc.get("case") == CASE_DS and doc.get("topology") == TOPOLOGY_DISAGG
    keys = set(DISAGG_RESULT_KEYS) | (set(DS_RESULT_KEYS) if ds_blocks else set())
    if set(doc) != keys:
        missing = sorted(keys - set(doc))
        extra = sorted(set(doc) - keys)
        return [f"keys: missing {missing}, unexpected {extra}"]
    errors: list[str] = []
    arm = DISAGG_ARMS.get(doc["arm"])
    if arm is None:
        errors.append(f"arm {doc['arm']!r} is not a disaggregated or direct arm")
    elif doc["case"] != arm.case or doc["topology"] != arm.topology:
        errors.append(f"case and topology must be arm {arm.name}'s {arm.case!r}, {arm.topology!r}")
    for key in ("test_id", "benchmark_mode"):
        if not isinstance(doc[key], str) or not doc[key]:
            errors.append(f"{key} must be a non-empty string")
    if doc["concurrency"] is not None and not (
        _is_count(doc["concurrency"]) and doc["concurrency"] > 0
    ):
        errors.append("concurrency must be a positive integer or null")
    if doc["dataset"] not in ("token_ids", "text"):
        errors.append(f"dataset {doc['dataset']!r} is not token_ids or text")
    if not (_is_count(doc["run"]) and doc["run"] >= 1):
        errors.append("run must be a positive integer")
    metrics = doc["metrics"]
    expected = {*_LATENCY_METRICS, "output_tok_per_s", "gen_device_step_ms"}
    if not isinstance(metrics, dict) or set(metrics) != expected:
        errors.append(f"metrics must hold exactly {sorted(expected)}")
    else:
        for name in _LATENCY_METRICS:
            stats = metrics[name]
            if not isinstance(stats, dict) or set(stats) != {"mean", "median", "p99"}:
                errors.append(f"metrics.{name} must hold mean, median and p99")
            elif any(v is not None and not _is_number(v) for v in stats.values()):
                errors.append(f"metrics.{name} values must be numbers or null")
        step = metrics["gen_device_step_ms"]
        if not isinstance(step, dict) or set(step) != {"mean", "median"}:
            errors.append("metrics.gen_device_step_ms must hold mean and median")
    if not _count_or_null(doc["output_tokens"]):
        errors.append("output_tokens must be a non-negative integer or null")
    errors += _validate_roles(doc["roles"], arm)
    errors += _validate_proof(doc["chain_proof"], arm)
    errors += _validate_first_output(doc["first_output"], arm)
    orch = doc["orch"]
    if not isinstance(orch, dict) or set(orch) != {
        "kind",
        "schedule_style",
        "router",
        "num_workers",
        "cpu_s",
        "config",
    }:
        errors.append("orch must hold kind, schedule_style, router, num_workers, cpu_s and config")
    if not isinstance(doc["client"], dict) or set(doc["client"]) != {"cpu_s"}:
        errors.append("client must hold cpu_s")
    if not isinstance(doc["kv_transfer"], dict) or set(doc["kv_transfer"]) != {
        "backend",
        "runtime",
        "env",
    }:
        errors.append("kv_transfer must hold backend, runtime and env")
    if not isinstance(doc["alignment"], dict) or set(doc["alignment"]) != set(
        DISAGG_ALIGNMENT_KEYS
    ):
        errors.append(f"alignment must hold exactly {list(DISAGG_ALIGNMENT_KEYS)}")
    versions = doc["versions"]
    if not isinstance(versions, dict) or set(versions) != {"trtllm", "ds"}:
        errors.append("versions must hold trtllm and ds")
    if (
        not isinstance(doc["nodes"], list)
        or not doc["nodes"]
        or not all(isinstance(n, str) and n for n in doc["nodes"])
    ):
        errors.append("nodes must be a non-empty list of names")
    if doc["job_id"] is not None and not isinstance(doc["job_id"], str):
        errors.append("job_id must be a string or null")
    if not isinstance(doc["notes"], list) or not all(isinstance(n, str) for n in doc["notes"]):
        errors.append("notes must be a list of strings")
    if ds_blocks:
        errors += validate_ds_blocks(doc, arm)
    return errors


def _validate_roles(roles: Any, arm: Arm | None) -> list[str]:
    if not isinstance(roles, dict) or set(roles) != set(WORKER_ROLES):
        return ["roles must hold ctx and gen"]
    errors = []
    for role, block in roles.items():
        if not isinstance(block, dict) or set(block) != set(_ROLE_KEYS):
            errors.append(f"roles.{role} must hold exactly {list(_ROLE_KEYS)}")
            continue
        if (
            arm is not None
            and arm.topology == TOPOLOGY_DISAGG
            and block["chain"] != role_arm(arm, role).chain
        ):
            errors.append(f"roles.{role}.chain is not arm {arm.name}'s")
        if not (_is_count(block["instances"]) and block["instances"] >= 1):
            errors.append(f"roles.{role}.instances must be a positive integer")
        for key in ("frontends", "postproc_workers"):
            part = block[key]
            if not isinstance(part, dict) or set(part) != {"configured", "started"}:
                errors.append(f"roles.{role}.{key} must hold configured and started")
            elif not all(_count_or_null(v) for v in part.values()):
                errors.append(f"roles.{role}.{key} values must be non-negative integers or null")
        for key in (
            "frontend_cpu_s",
            "frontend_cpu_us_per_token",
            "busiest_thread_cores",
            "startup_s",
        ):
            value = block[key]
            if value is not None and not (_is_number(value) and value >= 0):
                errors.append(f"roles.{role}.{key} must be a non-negative number or null")
    return errors


def _validate_proof(proof_block: Any, arm: Arm | None) -> list[str]:
    in_process = (
        arm is not None
        and arm.topology == TOPOLOGY_DISAGG
        and all(role_arm(arm, r).in_process for r in WORKER_ROLES)
    )
    if in_process:
        return [] if proof_block is None else [f"chain_proof must be null for arm {arm.name}"]
    if not isinstance(proof_block, dict) or set(proof_block) != set(_PROOF_KEYS):
        return [f"chain_proof must hold exactly {list(_PROOF_KEYS)}"]
    errors = []
    for role in WORKER_ROLES:
        part = proof_block[role]
        chained = (
            arm is None or arm.topology != TOPOLOGY_DISAGG or not role_arm(arm, role).in_process
        )
        if not chained:
            if part is not None:
                errors.append(
                    f"chain_proof.{role} must be null: the arm's {role} workers are in-process"
                )
            continue
        if not isinstance(part, dict) or set(part) != set(_PROOF_ROLE_KEYS[role]):
            errors.append(f"chain_proof.{role} must hold exactly {list(_PROOF_ROLE_KEYS[role])}")
        elif not all(_count_or_null(v) for v in part.values()):
            errors.append(f"chain_proof.{role} counts must be non-negative integers or null")
    if not isinstance(proof_block["orch_requests"], dict) or set(
        proof_block["orch_requests"]
    ) != set(WORKER_ROLES):
        errors.append("chain_proof.orch_requests must hold ctx and gen")
    if not _count_or_null(proof_block["handoffs"]):
        errors.append("chain_proof.handoffs must be a non-negative integer or null")
    orphans = proof_block["orphan_handoffs"]
    if orphans is not None and not (isinstance(orphans, int) and not isinstance(orphans, bool)):
        errors.append("chain_proof.orphan_handoffs must be an integer or null")
    status = proof_block["status"]
    if not isinstance(status, dict) or set(status) != {"overall", *WORKER_ROLES}:
        errors.append("chain_proof.status must hold overall, ctx and gen")
    elif status["overall"] not in _PROOF_STATUSES or any(
        v is not None and v not in _PROOF_STATUSES for v in status.values()
    ):
        errors.append(f"chain_proof.status values must be one of {_PROOF_STATUSES} or null")
    if not isinstance(proof_block["details"], list):
        errors.append("chain_proof.details must be a list")
    return errors


def write_disagg_result(doc: dict[str, Any], directory: str) -> str:
    """Validate a disaggregated arm's result and write it to a new file in ``directory``.

    Raises:
        ArmError: The document does not match the result schema, or no free
            file name is left.
        OSError: The directory or the file cannot be written.
    """
    errors = validate_disagg_result(doc)
    if errors:
        raise ArmError("invalid perf result: " + "; ".join(errors))
    os.makedirs(directory, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", doc["test_id"])[-120:]
    stem = f"{doc['arm']}-run{doc['run']}-{slug}"
    for n in range(1, 1000):
        path = os.path.join(directory, f"{stem}.json" if n == 1 else f"{stem}-{n}.json")
        try:
            with open(path, "x", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
                f.write("\n")
        except FileExistsError:
            continue
        return path
    raise ArmError(f"no free result file name for {stem} in {directory}")


def results_directory(default_dir: str, environ: Mapping[str, str] | None = None) -> str:
    """Where result files go: ``TRTLLM_OPENENGINE_RESULTS_DIR/perf``, else ``default_dir``."""
    env = os.environ if environ is None else environ
    results = env.get(ENV_RESULTS_DIR)
    return os.path.join(results, "perf") if results else default_dir


# Per-request timestamps. A worker in the chain gets FIRST_OUTPUT_DIR_ENV; its engine
# then records when each Generate call's first response leaves it, and its front end when that
# response arrives (tensorrt_llm/grpc/openengine/servicer.py FirstOutputTimestamps), one JSONL
# file per process under the worker's directory.
FIRST_OUTPUT_DIR_ENV = "TRTLLM_OPENENGINE_FIRST_OUTPUT_DIR"
ENGINE_FIRST_OUTPUT_PROBE = "engine_first_output_sent"
FRONTEND_FIRST_OUTPUT_PROBE = "frontend_first_output_received"


def first_output_dir(directory: str, worker: str) -> str:
    """Where ``worker``'s engine and front end write their first-output timestamps."""
    return os.path.join(directory, "first-output", worker)


def first_output_env(view: Arm, directory: str, worker: str) -> dict[str, str]:
    """The variable that turns the timestamps on for a worker in the chain; nothing in-process."""
    if view.in_process:
        return {}
    return {FIRST_OUTPUT_DIR_ENV: first_output_dir(directory, worker)}


def read_first_output(directory: str) -> dict[str, dict[str, list[int]]]:
    """``{probe: {request_id: [t_ns, ...]}}`` of the JSONL files in ``directory`` (none: ``{}``)."""
    found: dict[str, dict[str, list[int]]] = {}
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return found
    for name in names:
        if not name.endswith(".jsonl"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                lines = f.readlines()
        except (OSError, ValueError):
            continue
        for line in lines:
            try:
                record = json.loads(line)
                probe, request_id = record["probe"], record["request_id"]
                t_ns = int(record["t_ns"])
            except (ValueError, KeyError, TypeError):
                continue
            found.setdefault(probe, {}).setdefault(str(request_id), []).append(t_ns)
    return found


def _gap_stats(gaps_ms: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(gaps_ms)
    if not ordered:
        return {"count": 0, "median_ms": None, "p99_ms": None, "max_ms": None}
    p99 = ordered[min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))]
    return {
        "count": len(ordered),
        "median_ms": round(ordered[len(ordered) // 2], 3),
        "p99_ms": round(p99, 3),
        "max_ms": round(ordered[-1], 3),
    }


def first_output_gaps(
    records: Mapping[str, Mapping[str, Sequence[int]]],
    window: tuple[int, int] | Callable[[str], tuple[int, int] | None] | None,
    first_wave: int,
) -> dict[str, Any]:
    """The first-output gap (front end received - engine sent) of the calls sent in ``window``.

    A call counts when both probes recorded it exactly once and the engine sent
    its first output inside ``window``; a call the engine sent inside ``window``
    that either probe did not record exactly once is ``unmatched``. The
    ``first_wave`` calls the engine sent first are summarized apart from the
    rest. ``window`` may also map a call's id to the window of the clock that
    call's timestamps use.
    """
    sent = records.get(ENGINE_FIRST_OUTPUT_PROBE, {})
    received = records.get(FRONTEND_FIRST_OUTPUT_PROBE, {})
    joined = []
    unmatched = 0
    for request_id, times in sent.items():
        bounds = window(request_id) if callable(window) else window
        if not times or bounds is None or not bounds[0] <= times[0] <= bounds[1]:
            continue
        other = received.get(request_id, ())
        if len(times) != 1 or len(other) != 1:
            unmatched += 1
            continue
        joined.append((times[0], (other[0] - times[0]) / 1e6))
    joined.sort()
    gaps = [gap for _, gap in joined]
    return {
        "calls": len(joined),
        "unmatched": unmatched,
        "first_wave": _gap_stats(gaps[:first_wave]),
        "rest": _gap_stats(gaps[first_wave:]),
    }


def first_output_block(probe: "DisaggRunProbe", first_wave: int | None) -> dict[str, Any] | None:
    """The result's ``first_output``: per chained role, the first-output gaps; None in-process.

    ``first_wave`` is the client's concurrency: the calls of its first wave are summarized apart.
    """
    chained = [r for r in WORKER_ROLES if not role_arm(probe.arm, r).in_process]
    if not chained:
        return None
    window = None
    if PHASE_BEFORE in probe.window and PHASE_AFTER in probe.window:
        window = (probe.window[PHASE_BEFORE], probe.window[PHASE_AFTER])

    def worker_window(call: str) -> tuple[int, int] | None:
        # The engine and front end of a worker share its node's clock, which the worker's
        # readings record; the benchmark's own clock stands in when a reading lacks it.
        worker = call.split("/", 1)[0]
        times = [(probe.readings.get(p, {}).get(worker) or {}).get("t_ns") for p in PHASES]
        if all(isinstance(t, int) for t in times):
            return (times[0], times[1])
        return window

    block: dict[str, Any] = {"directory": os.path.join(probe.directory, "first-output")}
    for role in WORKER_ROLES:
        if role not in chained:
            block[role] = None
            continue
        merged: dict[str, dict[str, list[int]]] = {}
        for worker in probe._role_workers(role):
            for name, calls in read_first_output(first_output_dir(probe.directory, worker)).items():
                for request_id, times in calls.items():
                    merged.setdefault(name, {}).setdefault(f"{worker}/{request_id}", []).extend(
                        times
                    )
        block[role] = first_output_gaps(merged, worker_window, first_wave or 0)
    return block


_GAP_KEYS = ("count", "median_ms", "p99_ms", "max_ms")


def _validate_first_output(block: Any, arm: Arm | None) -> list[str]:
    chained = (
        []
        if arm is None or arm.topology != TOPOLOGY_DISAGG
        else [r for r in WORKER_ROLES if not role_arm(arm, r).in_process]
    )
    if not chained:
        return [] if block is None else ["first_output must be null without a worker in the chain"]
    if not isinstance(block, dict) or set(block) != {"directory", *WORKER_ROLES}:
        return ["first_output must hold directory, ctx and gen"]
    errors = []
    for role in WORKER_ROLES:
        part = block[role]
        if role not in chained:
            if part is not None:
                errors.append(
                    f"first_output.{role} must be null: the arm's {role} workers are in-process"
                )
            continue
        if not isinstance(part, dict) or set(part) != {"calls", "unmatched", "first_wave", "rest"}:
            errors.append(f"first_output.{role} must hold calls, unmatched, first_wave and rest")
            continue
        if not (_is_count(part["calls"]) and _is_count(part["unmatched"])):
            errors.append(f"first_output.{role} calls and unmatched must be non-negative integers")
        for group in ("first_wave", "rest"):
            stats = part[group]
            if not isinstance(stats, dict) or set(stats) != set(_GAP_KEYS):
                errors.append(f"first_output.{role}.{group} must hold exactly {list(_GAP_KEYS)}")
            elif not _is_count(stats["count"]) or any(
                v is not None and not _is_number(v) for k, v in stats.items() if k != "count"
            ):
                errors.append(f"first_output.{role}.{group} values must be numbers or null")
    return errors


# ----------------------------------------------------------------------------- case-03 (ds-engine)

# dsv4-serve's command prefix (for example ``python3 -m dsv4_serve``), the same variable the
# chain harness's ``ds`` chain starts its engine with.
ENV_DS_SERVE_CMD = "TRTLLM_TEST_DS_SERVE_CMD"
# The ds-engine checkout: its commit goes into the result, and the AgentX client runs its
# ``dev/run_agentx.sh``.
ENV_DS_ROOT = "TRTLLM_TEST_DS_ROOT"
# The ds-engine commit written to the result; when unset, the HEAD of TRTLLM_TEST_DS_ROOT.
ENV_DS_SHA = "TRTLLM_PERF_DS_SHA"
# Orchestrator values that differ between dsv4-serve's and trtllm-serve's disaggregated servers
# by default: the aligned arms write dsv4-serve's defaults for both, so
# the values are the same whichever server reads them.
DS_ALIGNED_ORCHESTRATOR = {"max_retries": 3, "server_keep_alive_timeout": 600}
# dsv4-serve's disaggregated-server variables: the aligned native arm does not route new
# conversations by load (trtllm-serve has no such router) and runs one server process, the
# number trtllm-serve's disaggregated server runs under perf sanity.
ENV_DS_PROXY_LOAD_ROUTE = "DSV4_PROXY_LOAD_ROUTE"
ENV_DS_PROXY_WORKERS = "DSV4_PROXY_WORKERS"
ENV_DS_PROXY_FE_AFFINITY = "DSV4_PROXY_FE_AFFINITY"
# Instance parameters of a ds worker that the aligned arms must share.
DS_INSTANCE_PARAMS = (
    "tensor_parallel_size",
    "moe_expert_parallel_size",
    "enable_attention_dp",
    "attention_dp_config",
    "max_batch_size",
    "max_num_tokens",
    "max_seq_len",
    "kv_cache_config",
    "speculative_config",
    "custom_tokenizer",
)
# Settings of the serving front end in front of a ds engine (the chain's worker front end, or
# dsv4-serve's own HTTP front end) that the result records.
DS_FRONTEND_SETTINGS = (
    "num_serve_frontends",
    "enable_tokenization_cache",
    "serve_frontend_affinity_port_stride",
)
_DS_ENV_PREFIX = "DSV4_"
# ds variables that only choose what the engine logs; they need not match across arms.
# DSV4_REQ_LOG is not among them: with it the engine runs its step-boundary work serially
# instead of on the DSV4_HOST_PAR pool.
_DS_LOG_ONLY_ENV = ("DSV4_STEP_LOG", "DSV4_XFER_PROF", "DSV4_TOKCACHE_VERIFY")
# The ds engine ignores DSV4_GEN_OVERLAP=1 on a context instance, with block reuse or under
# DSV4_SCHED_PIPE, and logs a warning then (DK-13).
ENV_DS_GEN_OVERLAP = "DSV4_GEN_OVERLAP"
# Node rank of a multi-node ds instance, as dsv4-serve reads it ($DS/python/dsv4_serve/cli.py _multinode).
_DS_NODE_RANK_ENV = ("DSV4_NODE_RANK", "SLURM_NODEID", "SLURM_PROCID")


def ds_command_prefix(environ: Mapping[str, str] | None = None) -> list[str]:
    """dsv4-serve's command prefix from ``TRTLLM_TEST_DS_SERVE_CMD``.

    Raises:
        ArmError: The variable is unset or empty.
    """
    env = os.environ if environ is None else environ
    prefix = shlex.split(env.get(ENV_DS_SERVE_CMD, ""))
    if not prefix:
        raise ArmError(f"the case-03 arms need {ENV_DS_SERVE_CMD}, dsv4-serve's command prefix")
    return prefix


def ds_http_command(tokens: Sequence[str], environ: Mapping[str, str] | None = None) -> list[str]:
    """A ``trtllm-serve`` worker command line as dsv4-serve's: the same model and options.

    dsv4-serve takes trtllm-serve's options and warns about the configuration keys it ignores.

    Raises:
        ArmError: ``TRTLLM_TEST_DS_SERVE_CMD`` is unset.
        ValueError: The command line is not a ``trtllm-serve`` one.
    """
    from openengine_chain.split import parse_serve_command

    command = parse_serve_command(list(tokens))
    return ds_command_prefix(environ) + [command.model] + list(command.args)


def ds_orchestrator_command(
    tokens: Sequence[str], environ: Mapping[str, str] | None = None
) -> list[str]:
    """A ``trtllm-serve disaggregated ...`` command line as dsv4-serve's, without ``--report_addr``.

    dsv4-serve's disaggregated server takes the same subcommand and options but does not
    publish its address; :func:`prepare_orchestrator` gives it a port chosen in advance.

    Raises:
        ArmError: ``TRTLLM_TEST_DS_SERVE_CMD`` is unset, or the command line is not
            ``trtllm-serve disaggregated``.
    """
    tokens = list(tokens)
    if tokens[:2] != ["trtllm-serve", "disaggregated"]:
        raise ArmError(f"not a 'trtllm-serve disaggregated' command line: {tokens[:2]}")
    args: list[str] = []
    skip = False
    for token in tokens[2:]:
        if skip:
            skip = False
        elif token == "--report_addr":
            skip = True
        elif not token.startswith("--report_addr="):
            args.append(token)
    return ds_command_prefix(environ) + ["disaggregated"] + args


def is_ds_engine_role(arm: Arm, role: str) -> bool:
    """Whether ``role``'s workers of ``arm`` run the ds engine (natively or in the chain)."""
    settings = role_arm(arm, role)
    return settings.ds_http or settings.chain == CHAIN_DS


def ds_spans_nodes(arm: Arm, gpus: int, gpus_per_node: int) -> bool:
    """Whether a case-03 worker of ``gpus`` GPUs runs as one multi-node ds instance.

    dsv4-serve runs an instance larger than a node as one process per node (CP-11): the
    workers of a case-03 arm may take whole nodes.
    """
    return (
        arm.case == CASE_DS
        and arm.topology == TOPOLOGY_DISAGG
        and gpus_per_node > 0
        and gpus > gpus_per_node
        and gpus % gpus_per_node == 0
    )


def _step_tasks_per_node(environ: Mapping[str, str]) -> list[int]:
    """Tasks per node of the Slurm step this process runs in (``SLURM_STEP_TASKS_PER_NODE``)."""
    counts = []
    for part in environ.get("SLURM_STEP_TASKS_PER_NODE", "").split(","):
        count = part.strip().split("(", 1)[0]
        if count.isdigit():
            counts.append(int(count))
    return counts


def ds_node_rank(
    plan: ServerPlan, local_gpus: int, environ: Mapping[str, str] | None = None
) -> int | None:
    """This node's rank in a multi-node ds instance of ``plan``; ``None`` for one node.

    The rules are dsv4-serve's: a ``tensor_parallel_size`` larger than this node's GPUs
    (``CUDA_VISIBLE_DEVICES``, else ``local_gpus``) makes the instance multi-node, and the
    rank is ``DSV4_NODE_RANK``, else ``SLURM_NODEID``, else ``SLURM_PROCID``. Each node of
    the instance runs one task of the worker's step: rank 0 leads and serves, the others
    follow.

    Raises:
        ArmError: The plan sets pipeline or context parallelism, which dsv4-serve ignores
            (the worker would get GPUs its instance does not use), the instance's GPUs are
            not a multiple of this node's, the step runs more than one task on a node, or
            the rank is not a non-negative integer.
    """
    env = os.environ if environ is None else environ
    for key in ("pipeline_parallel_size", "context_parallel_size"):
        size = _as_int(_ds_param(plan.cmd, plan.config, key))
        if isinstance(size, int) and size > 1:
            raise ArmError(f"dsv4-serve ignores {key}={size}; a ds worker must set it to 1")
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        local_gpus = len([gpu for gpu in visible.split(",") if gpu.strip()])
    tp = _as_int(_ds_param(plan.cmd, plan.config, "tensor_parallel_size"))
    tp = tp if isinstance(tp, int) and tp > 0 else 1
    if local_gpus <= 0 or tp <= local_gpus:
        return None
    if tp % local_gpus:
        raise ArmError(f"a ds instance of {tp} GPUs does not fill nodes of {local_gpus} GPUs")
    tasks = _step_tasks_per_node(env)
    if any(count > 1 for count in tasks):
        raise ArmError(
            "a multi-node ds instance runs one process per node: its worker step must run one "
            f"task per node, not SLURM_STEP_TASKS_PER_NODE={env['SLURM_STEP_TASKS_PER_NODE']}"
        )
    for name in _DS_NODE_RANK_ENV:
        value = env.get(name)
        if value is not None:
            if not value.isdigit():
                raise ArmError(f"{name}={value!r} is not a node rank")
            return int(value)
    return 0


def ds_follower_command(plan: ServerPlan, environ: Mapping[str, str] | None = None) -> list[str]:
    """The command of a follower node of a multi-node ds instance.

    A follower runs dsv4-serve with the worker's command line and YAML; dsv4-serve sees its
    node rank and only loads its part of the model and follows the leader's steps. It serves
    neither HTTP nor gRPC, so a chain worker's follower is the same command as a native one's
    (the chain's leader is the OpenEngine engine the harness starts from a rewritten YAML).

    Raises:
        ArmError: ``TRTLLM_TEST_DS_SERVE_CMD`` is unset.
        ValueError: The plan's command line is neither trtllm-serve's nor dsv4-serve's.
    """
    tokens = list(plan.cmd)
    prefix = ds_command_prefix(environ)
    if tokens[: len(prefix)] == prefix:
        return tokens
    return ds_http_command(tokens, environ)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_process_group(proc: subprocess.Popen, grace_s: float) -> None:
    """Stop ``proc``, started in a session of its own, and every process of its group.

    SIGTERM to the group; SIGKILL to it when any of its processes, ``proc`` or another,
    still runs ``grace_s`` seconds later. Reaps ``proc`` when it has exited; raises nothing
    when a process outlives the SIGKILL too.
    """
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace_s
        while True:
            proc.poll()
            if not _group_alive(proc.pid):
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
    proc.poll()


# Grace before SIGKILL when this process is being stopped: shorter than the 30 s the
# single-node driver gives a role before it kills the role's process group.
_STOPPING_GRACE_S = 10


def _exit_on_sigterm(signum: int, _frame: Any) -> None:
    # A further SIGTERM must not interrupt the cleanup this one starts.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


@contextlib.contextmanager
def _sigterm_raises() -> Iterator[None]:
    """Within the block, SIGTERM raises ``SystemExit`` once; later ones are ignored.

    Only in the main thread. A child started in a session of its own is outside the reach of a
    signal sent to this process's group; the caller's cleanup of that child then still runs
    when this process is stopped with SIGTERM.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_ds_follower(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    log_path: str,
    wait: Callable[[Any], None],
) -> int | None:
    """Run a follower node of a multi-node ds instance until ``wait(proc)`` returns.

    ``wait`` is the worker's wait for the end of the benchmark. The follower's process group
    is then stopped (:func:`_stop_process_group`, 60 s grace); so it is, with a shorter grace,
    when ``wait`` raises or this process gets SIGTERM. Returns its exit code, or ``None`` when
    it outlived the SIGKILL.

    Raises:
        OSError: The follower cannot be started.
    """
    with open(log_path, "w", encoding="utf-8") as sink, _sigterm_raises():
        proc = subprocess.Popen(
            list(cmd),
            env=dict(env),
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait(proc)
            _stop_process_group(proc, 60)
        except BaseException:
            _stop_process_group(proc, _STOPPING_GRACE_S)
            raise
    return proc.returncode


def ds_orchestrator_env(arm: Arm | None, env: Mapping[str, str]) -> dict[str, str]:
    """The disaggregated server's environment under ``arm``, from the one the test gives it.

    The aligned native arm runs dsv4-serve's disaggregated server as one process without its
    load router; other arms keep ``env``. A pure function of ``env``, so that the benchmark's
    process, which writes the result, derives what the server's process ran with.
    """
    out = dict(env)
    if (
        arm is not None
        and arm.case == CASE_DS
        and arm.orchestrator == ORCHESTRATOR_DS
        and arm.aligned
    ):
        out.pop(ENV_DS_PROXY_LOAD_ROUTE, None)
        out[ENV_DS_PROXY_WORKERS] = "1"
    return out


@dataclass(frozen=True)
class OrchestratorPlan:
    """How a disaggregated arm starts its disaggregated server.

    Attributes:
        cmd: The command line.
        port: The port chosen in advance (for dsv4-serve, which does not publish its
            address); ``None`` when the server publishes it through ``--report_addr``.
        env: The server's environment.
    """

    cmd: tuple[str, ...]
    port: int | None
    env: dict[str, str]


def b64_prompt_ids_off(arm: Arm | None, completions_client: bool) -> bool:
    """Whether ``arm``'s trtllm-serve disaggregated server runs without ``gen_tokids_ctxbytes``.

    It does in front of dsv4-serve's context workers when a client calls ``/v1/completions``:
    there dsv4-serve returns the prompt as ``prompt_token_ids_b64``, which trtllm-serve's
    ``CompletionResponse`` does not take, while the chain's trtllm-serve front ends return
    the token id list, which the server then reads in both arms.
    """
    return (
        arm is not None
        and arm.case == CASE_DS
        and arm.topology == TOPOLOGY_DISAGG
        and arm.orchestrator == ORCHESTRATOR_TRTLLM
        and role_arm(arm, ROLE_CTX).ds_http
        and completions_client
    )


def prepare_orchestrator(
    arm: Arm,
    cmd: Sequence[str],
    config_path: str,
    hostname: str,
    env: Mapping[str, str],
    environ: Mapping[str, str] | None = None,
    *,
    completions_client: bool = False,
) -> OrchestratorPlan:
    """The disaggregated server of a case-03 arm; other arms keep ``cmd`` and ``env``.

    An aligned arm writes :data:`DS_ALIGNED_ORCHESTRATOR` into the server's YAML
    (``config_path``, rewritten in place, so that the result reads what the server read), and
    ``gen_tokids_ctxbytes: false`` when :func:`b64_prompt_ids_off` (``completions_client``:
    a client of the server calls ``/v1/completions``). The native arms' server is
    dsv4-serve's: it gets a port chosen in advance, written into the YAML. Its environment is
    :func:`ds_orchestrator_env`'s.

    Raises:
        ArmError: As :func:`ds_orchestrator_command` raises, or the YAML is not a mapping.
        OSError: The YAML cannot be read or written.
        yaml.YAMLError: The YAML cannot be parsed.
    """
    out_env = ds_orchestrator_env(arm, env)
    if arm.case != CASE_DS or arm.topology != TOPOLOGY_DISAGG:
        return OrchestratorPlan(tuple(cmd), None, out_env)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ArmError(f"{config_path} must hold a mapping, got {type(config).__name__}")
    if arm.aligned:
        config.update(DS_ALIGNED_ORCHESTRATOR)
    if b64_prompt_ids_off(arm, completions_client):
        config[_B64_PROMPT_IDS_KEY] = False
    port = None
    tokens = list(cmd)
    if arm.orchestrator == ORCHESTRATOR_DS:
        tokens = ds_orchestrator_command(tokens, environ)
        port = free_port(hostname)
        config["port"] = port
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return OrchestratorPlan(tuple(tokens), port, out_env)


def _ds_env_split(env: Mapping[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    """The ``DSV4_*`` variables of ``env``: (those the arms share, those that only choose logging)."""
    shared, log_only = {}, {}
    for key, value in sorted(env.items()):
        if key.startswith(_DS_ENV_PREFIX):
            (log_only if key in _DS_LOG_ONLY_ENV else shared)[key] = value
    return shared, log_only


# dsv4-serve's other names for the instance parameters' options, and its flag options.
_DS_PARAM_OPTIONS = {
    "tensor_parallel_size": ("--tensor_parallel_size", "--tp_size"),
    "pipeline_parallel_size": ("--pipeline_parallel_size", "--pp_size"),
    "context_parallel_size": ("--context_parallel_size", "--cp_size"),
    "moe_expert_parallel_size": ("--moe_expert_parallel_size", "--ep_size"),
}
_DS_FLAG_PARAMS = ("enable_attention_dp",)
_DS_KV_FRACTION_OPTIONS = ("--free_gpu_memory_fraction", "--kv_cache_free_gpu_memory_fraction")


def _ds_param(argv: Sequence[str], config: Mapping[str, Any], key: str) -> Any:
    """Instance parameter ``key``: its option in ``argv`` (dsv4-serve's names), else ``config``.

    Does not apply ``--kv_cache_dtype`` or ``--set``, which perf sanity does not pass.
    """
    if key in _DS_FLAG_PARAMS:
        return True if f"--{key}" in argv else config.get(key)
    if key == "kv_cache_config":
        value = copy.deepcopy(config.get(key))
        fraction = None
        for name in _DS_KV_FRACTION_OPTIONS:
            found = [item for item in _option_values(argv, name) if item[1] is not None]
            if found and (fraction is None or found[-1][0] > fraction[0]):
                fraction = found[-1]
        if fraction is not None:
            value = dict(value) if isinstance(value, Mapping) else {}
            value["free_gpu_memory_fraction"] = yaml.safe_load(fraction[1])
        return value
    found = []
    for name in _DS_PARAM_OPTIONS.get(key, (f"--{key}",)):
        found += [item for item in _option_values(argv, name) if item[1] is not None]
    if found:
        return yaml.safe_load(max(found)[1])
    return config.get(key)


def ds_instance_reading(
    server: "ArmServer", plan: ServerPlan, env: Mapping[str, str], notes: list[str]
) -> dict[str, Any]:
    """A ds worker's instance parameters, ds variables and front-end settings, as it runs.

    Read from ``/proc`` of the process running the ds engine (a chain worker's engine, else
    the worker's server): the configuration file and options on its command line and its
    environment; the planned ones stand in, with a note, for what ``/proc`` does not give.
    The front-end settings come from the worker's planned configuration.
    """
    handle = server.handle
    process = handle.engine if handle is not None else server.proc
    pid = process.pid if process is not None else None
    config = _process_config(pid) if pid is not None else None
    if config is None:
        notes.append(f"ds instance: no configuration file read from process {pid}; planned one")
        config = plan.config
    argv = _process_argv(pid) if pid is not None else None
    if argv is None:
        notes.append(f"ds instance: command line of process {pid} not readable; planned one")
        argv = list(plan.cmd)
    run_env = _process_environ(pid) if pid is not None else None
    if run_env is None:
        notes.append(f"ds instance: environment of process {pid} not readable; planned one")
        run_env = env
    params = {key: _ds_param(argv, config, key) for key in DS_INSTANCE_PARAMS}
    shared, log_only = _ds_env_split(run_env)
    # DK-13: the warning exists only with DSV4_GEN_OVERLAP=1, and the engine log is not read.
    overlap_warning = False if run_env.get(ENV_DS_GEN_OVERLAP) != "1" else None
    if overlap_warning is None:
        notes.append(
            f"ds instance: {ENV_DS_GEN_OVERLAP}=1; whether the engine ignored it is not read"
        )
    return {
        "params": params,
        "env": shared,
        "log_only_env": log_only,
        "gen_overlap_ignored_warning": overlap_warning,
        "frontend": {key: plan.config.get(key) for key in DS_FRONTEND_SETTINGS},
    }


# The orchestrator setting that has the context workers return the prompt in base64.
_B64_PROMPT_IDS_KEY = "gen_tokids_ctxbytes"
_B64_PROMPT_IDS_NOTE = (
    "case-03: trtllm-serve's disaggregated server ran with gen_tokids_ctxbytes false: on "
    "/v1/completions dsv4-serve's context workers would return prompt_token_ids_b64, which "
    "trtllm-serve's CompletionResponse does not take; the chain's front ends return the token "
    "id list there, as these workers then do"
)
# The blocks only a case-03 disaggregated result holds, on top of DISAGG_RESULT_KEYS.
DS_RESULT_KEYS = ("ds_instances", "ds_alignment", "ds_client")
_DS_ORCH_KEYS = (
    "max_retries",
    "server_keep_alive_timeout",
    "gen_strip_message_history",
    "gen_tokids_ctxbytes",
    "conversation_affinity_header_for_subagents",
    "subagent_affinity_scope",
    "frontend_affinity",
    "num_proxy_workers",
)
_DS_ALIGNMENT_KEYS = ("orchestrator", "orchestrator_env", "frontend")
_DS_KV_NOTE = (
    "case-03: dsv4-serve accepts cache_transceiver_config.backend but ignores it; the ds engine "
    "moves the KV cache over its own NVLink fabric transfer, so kv_transfer.backend is the "
    "configured value only"
)
# The line the AgentX wrapper (run_ds_agentx) prints with the client's settings.
DS_CLIENT_LINE = "[ds-agentx] settings "
# The line it prints with where the run's output token count comes from in aiperf's export.
DS_TOKENS_LINE = "[ds-agentx] total generated tokens from "


def ds_sha(environ: Mapping[str, str] | None = None) -> str | None:
    """The ds-engine commit: ``TRTLLM_PERF_DS_SHA``, else the HEAD of ``TRTLLM_TEST_DS_ROOT``."""
    env = os.environ if environ is None else environ
    if env.get(ENV_DS_SHA):
        return env[ENV_DS_SHA]
    root = env.get(ENV_DS_ROOT)
    return _git_head(root) if root else None


def _client_settings(output: str) -> dict[str, Any] | None:
    for line in output.splitlines():
        if line.startswith(DS_CLIENT_LINE):
            try:
                value = json.loads(line[len(DS_CLIENT_LINE) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def add_ds_blocks(
    doc: dict[str, Any],
    probe: "DisaggRunProbe",
    *,
    output: str,
    orchestrator_config: str,
    orchestrator_env: Mapping[str, str],
    environ: Mapping[str, str] | None = None,
) -> None:
    """Add a case-03 result's blocks: each role's ds instance, the aligned settings, the client.

    ``orchestrator_env`` is the environment the test gives the disaggregated server; the
    server's own is derived from it (:func:`ds_orchestrator_env`). Also fills in the
    orchestrator's kind and the ds-engine commit, and notes what ``kv_transfer.backend``
    means for the ds engine.
    """
    server_env = ds_orchestrator_env(probe.arm, orchestrator_env)
    after = probe.readings.get(PHASE_AFTER, {})
    doc["ds_instances"] = _role_values_of(after, probe, lambda r: r.get("ds_instance"))
    try:
        with open(orchestrator_config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        config = {}
    if not isinstance(config, Mapping):
        config = {}
    doc["ds_alignment"] = {
        "orchestrator": {key: config.get(key) for key in _DS_ORCH_KEYS},
        "orchestrator_env": {
            key: server_env.get(key)
            for key in (ENV_DS_PROXY_LOAD_ROUTE, ENV_DS_PROXY_WORKERS, ENV_DS_PROXY_FE_AFFINITY)
        },
        "frontend": _role_values_of(
            after, probe, lambda r: (r.get("ds_instance") or {}).get("frontend")
        ),
    }
    doc["ds_client"] = _client_settings(output)
    if (
        probe.arm.orchestrator == ORCHESTRATOR_TRTLLM
        and role_arm(probe.arm, ROLE_CTX).ds_http
        and config.get(_B64_PROMPT_IDS_KEY) is False
    ):
        doc["notes"].append(_B64_PROMPT_IDS_NOTE)
    for line in output.splitlines():
        if line.startswith(DS_TOKENS_LINE):
            doc["notes"].append(
                f"case-03 AgentX: output_tokens is aiperf's {line[len(DS_TOKENS_LINE) :]}"
            )
    doc["orch"]["kind"] = probe.arm.orchestrator
    doc["versions"]["ds"] = ds_sha(environ)
    doc["notes"].append(_DS_KV_NOTE)


def validate_ds_blocks(doc: Mapping[str, Any], arm: Arm | None) -> list[str]:
    """Problems with the case-03 blocks of a disaggregated result (:data:`DS_RESULT_KEYS`)."""
    errors = []
    instances = doc["ds_instances"]
    if not isinstance(instances, dict) or set(instances) != set(WORKER_ROLES):
        errors.append("ds_instances must hold ctx and gen")
    else:
        for role, value in instances.items():
            parts = value if isinstance(value, list) else [value]
            for part in parts:
                if part is not None and (
                    not isinstance(part, dict)
                    or set(part)
                    != {"params", "env", "log_only_env", "gen_overlap_ignored_warning", "frontend"}
                ):
                    errors.append(f"ds_instances.{role} must be null or a ds instance reading")
    alignment = doc["ds_alignment"]
    if not isinstance(alignment, dict) or set(alignment) != set(_DS_ALIGNMENT_KEYS):
        errors.append(f"ds_alignment must hold exactly {list(_DS_ALIGNMENT_KEYS)}")
    elif not all(isinstance(alignment[key], dict) for key in ("orchestrator", "orchestrator_env")):
        errors.append("ds_alignment.orchestrator and orchestrator_env must be objects")
    elif arm is not None and arm.aligned:
        orch = alignment["orchestrator"]
        for key, value in DS_ALIGNED_ORCHESTRATOR.items():
            if orch.get(key) != value:
                errors.append(f"ds_alignment.orchestrator.{key} must be {value} for arm {arm.name}")
        # trtllm-serve's disaggregated server then retries 0 times; dsv4-serve's still 3.
        if isinstance(doc.get("alignment"), dict) and doc["alignment"].get("disagg_no_retry"):
            errors.append(f"arm {arm.name} must not set {ENV_DISAGG_NO_RETRY}=1")
        if arm.orchestrator == ORCHESTRATOR_DS:
            env = alignment["orchestrator_env"] or {}
            if env.get(ENV_DS_PROXY_LOAD_ROUTE) not in (None, "0"):
                errors.append(f"arm {arm.name} must not route by load ({ENV_DS_PROXY_LOAD_ROUTE})")
            if env.get(ENV_DS_PROXY_WORKERS) != "1":
                errors.append(f"arm {arm.name} must run one ds proxy process")
    if doc["ds_client"] is not None and not isinstance(doc["ds_client"], dict):
        errors.append("ds_client must be an object or null")
    if arm is not None and isinstance(doc.get("orch"), dict):
        if doc["orch"].get("kind") != arm.orchestrator:
            errors.append(f"orch.kind must be arm {arm.name}'s {arm.orchestrator!r}")
    return errors


def compare_ds_instances(first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    """Differences between two case-03 results' ds instances (parameters and shared variables).

    E3-native-al and E3-chain must have none; the variables that
    only choose logging are not compared. A role without a reading is a difference.
    """
    differences = []
    for role in WORKER_ROLES:
        a = (first.get("ds_instances") or {}).get(role)
        b = (second.get("ds_instances") or {}).get(role)
        if not isinstance(a, dict) or not isinstance(b, dict):
            differences.append(f"{role}: no single ds instance reading on both sides")
            continue
        for part in ("params", "env"):
            left, right = a.get(part) or {}, b.get(part) or {}
            for key in sorted(set(left) | set(right)):
                if left.get(key) != right.get(key):
                    differences.append(
                        f"{role}.{part}.{key}: {left.get(key)!r} != {right.get(key)!r}"
                    )
    return differences


# ----------------------------------------------------------------------------- case-03 layouts (CP-11)


@dataclass(frozen=True)
class DsInstance:
    """One ds instance of a layout.

    Attributes:
        worker: Its perf sanity role, ``CTX_<i>`` or ``GEN_<i>``.
        nodes: Its nodes; the first leads (serves), the others follow.
        gpus: ``CUDA_VISIBLE_DEVICES`` on each of its nodes.
        tp: Its ``tensor_parallel_size`` (all its GPUs).
    """

    worker: str
    nodes: tuple[str, ...]
    gpus: str
    tp: int


@dataclass(frozen=True)
class DsLayout:
    """The ds instances of a layout and the node of the disaggregated server and the client."""

    name: str
    instances: tuple[DsInstance, ...]
    orchestrator_node: str


_NODE_GPUS = "0,1,2,3"
# Context instances of the mid layout: PU-08's middle row has 2 (agentx_deploy.sh's CTX_N=2;
# the script's own default is 3).
DS_MID_CTX_DEFAULT = 2


def ds_layout(
    name: str,
    nodes: Sequence[str],
    *,
    ctx_instances: int | None = None,
    client_node: str | int | None = None,
) -> DsLayout:
    """Where each ds instance of a case-03 layout runs, as ds's deployment scripts place it.

    For a job body that starts the instances itself; the single-node driver assigns GPUs
    its own way. Layouts: ``n0`` (PU-06, ``$DS/dev/test_disagg.sh``: context DEP2 on GPUs 0-1
    and generation TP2 on GPUs 2-3 of one node), and as ``$DS/dev/agentx_deploy.sh`` places
    them, ``mini`` (PU-07: context DEP2 on node 0, generation TP4 on node 1) and ``mid``
    (PU-08: ``ctx_instances`` context DEP2, two per node, then one generation DEP16 on four
    nodes). The disaggregated server and the client run on node 0; ``client_node="last"``
    gives them the last node to themselves, an index puts them on that node.

    Raises:
        ArmError: The layout is unknown, or ``nodes`` has too few nodes for it.
    """
    pool = list(nodes)
    orchestrator = pool[0] if pool else ""
    if client_node == "last":
        if len(pool) < 2:
            raise ArmError("client_node='last' needs a node besides the instances'")
        orchestrator = pool.pop()
    elif isinstance(client_node, int):
        if not 0 <= client_node < len(pool):
            raise ArmError(f"client_node {client_node} is not one of the {len(pool)} nodes")
        orchestrator = pool[client_node]

    def need(count: int) -> None:
        if len(pool) < count:
            raise ArmError(f"layout {name} needs {count} instance nodes, got {len(pool)}")

    instances: list[DsInstance] = []
    if name == "n0":
        need(1)
        instances = [
            DsInstance("CTX_0", (pool[0],), "0,1", 2),
            DsInstance("GEN_0", (pool[0],), "2,3", 2),
        ]
    elif name == "mini":
        need(2)
        instances = [
            DsInstance("CTX_0", (pool[0],), "0,1", 2),
            DsInstance("GEN_0", (pool[1],), _NODE_GPUS, 4),
        ]
    elif name == "mid":
        count = DS_MID_CTX_DEFAULT if ctx_instances is None else ctx_instances
        if count < 1:
            raise ArmError(f"layout mid needs at least one context instance, got {count}")
        ctx_nodes = (count + 1) // 2
        need(ctx_nodes + 4)
        for i in range(count):
            gpus = "0,1" if i % 2 == 0 else "2,3"
            instances.append(DsInstance(f"CTX_{i}", (pool[i // 2],), gpus, 2))
        instances.append(
            DsInstance("GEN_0", tuple(pool[ctx_nodes : ctx_nodes + 4]), _NODE_GPUS, 16)
        )
    else:
        raise ArmError(f"unknown case-03 layout {name!r}; expected n0, mini or mid")
    return DsLayout(name, tuple(instances), orchestrator)


# ----------------------------------------------------------------------------- AgentX client (CP-07)

AGENTX_CLIENT = "agentx"
# The trace corpus $DS/dev/run_agentx.sh replays (it takes no dataset argument).
DS_AGENTX_DATASET = "semianalysis_cc_traces_weka_062126"
# The client settings all case-03 arms share (PU-08's); they replace the
# lane's own values.
DS_AGENTX_SETTINGS = {
    "AGENTX_DURATION": "3600",
    "AGENTX_WARMUP_PER_LANE": "3",
    "AGENTX_EXTRA_ARGS": "",
}
_AGENTX_LEARN_ENV = (
    "AIPERF_AGENTX_WARMUP_LEARN_REQUESTS",
    "AIPERF_AGENTX_WARMUP_LEARN_MAX_TOKENS",
    "AIPERF_AGENTX_WARMUP_LEARN_FREEZE_FILE",
)
ENV_AGENTX_ARTIFACT_DIR = "TRTLLM_AGENTX_ARTIFACT_DIR"
DS_AGENTX_COMMAND = "ds-agentx"


def takes_client(arm: Arm | None, benchmark_client: str) -> bool:
    """Whether ``arm`` measures a client of ``benchmark_client`` (``""`` is benchmark_serving).

    The case-03 arms also take the AgentX client, which they run as :func:`run_ds_agentx`.
    """
    if not benchmark_client:
        return True
    return (
        arm is not None
        and arm.case == CASE_DS
        and arm.topology == TOPOLOGY_DISAGG
        and benchmark_client == AGENTX_CLIENT
    )


def arm_client_cmd(arm: Arm | None, benchmark_client: str, cmd: Sequence[str]) -> list[str]:
    """The client command line under ``arm``: an AgentX client of a case-03 arm runs ds's client.

    perf sanity's AgentX command (``agentx_client.py --model M --concurrency C --dataset D``)
    becomes this module's ``ds-agentx`` command with the same options; the others are kept.

    Raises:
        ArmError: The AgentX command line gives an option the ds client does not take, or a
            dataset other than the one it replays.
    """
    tokens = list(cmd)
    if not (takes_client(arm, benchmark_client) and benchmark_client == AGENTX_CLIENT):
        return tokens
    options = {}
    rest = tokens[2:]
    if len(rest) % 2:
        raise ArmError(f"cannot read the AgentX command line {tokens}")
    for name, value in zip(rest[::2], rest[1::2]):
        if name not in ("--model", "--concurrency", "--dataset"):
            raise ArmError(
                f"arm {arm.name}: the ds AgentX client takes --model, --concurrency and "
                f"--dataset, not {name}"
            )
        options[name] = value
    if options.get("--dataset", DS_AGENTX_DATASET) != DS_AGENTX_DATASET:
        raise ArmError(
            f"arm {arm.name}: the ds AgentX client replays {DS_AGENTX_DATASET}, not "
            f"{options['--dataset']}"
        )
    out = [sys.executable, os.path.abspath(__file__), DS_AGENTX_COMMAND]
    for name in ("--model", "--concurrency", "--dataset"):
        if name in options:
            out += [name, options[name]]
    return out


def _agentx_client_module() -> types.ModuleType:
    """perf/agentx_client.py, whose checks and report format perf sanity reads."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentx_client.py")
    spec = importlib.util.spec_from_file_location("perf_agentx_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _total_output_tokens(export: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """The run's output token count from an aiperf export, and how it was obtained."""
    # output_token_count is per request: only its sum is the run's total.
    for field_name in ("total_output_tokens", "output_token_count"):
        entry = export.get(field_name)
        if isinstance(entry, Mapping):
            for stat in ("sum", "total"):
                if _is_number(entry.get(stat)):
                    return int(entry[stat]), f"{field_name}.{stat}"
    rate = export.get("output_token_throughput")
    duration = export.get("benchmark_duration")
    if isinstance(rate, Mapping) and isinstance(duration, Mapping):
        if _is_number(rate.get("avg")) and _is_number(duration.get("avg")):
            return (
                int(round(rate["avg"] * duration["avg"])),
                "output_token_throughput.avg x benchmark_duration.avg",
            )
    return None, None


def run_ds_agentx(argv: Sequence[str], environ: Mapping[str, str] | None = None) -> int:
    """The AgentX client of the case-03 arms: ``$TRTLLM_TEST_DS_ROOT/dev/run_agentx.sh``.

    Runs ds's client (aiperf with ds's warmup-learning patch when
    ``AIPERF_AGENTX_WARMUP_LEARN_REQUESTS`` is above 0) against ``--host``/``--port`` with
    :data:`DS_AGENTX_SETTINGS`, writing its records to ``concurrency_<C>`` under
    ``TRTLLM_AGENTX_ARTIFACT_DIR`` (an export an earlier run left there is removed first,
    as perf/agentx_client.py does); then checks aiperf's export as agentx_client.py does and
    prints its metrics in benchmark_serving's format, the total output token count, and the
    lines of the client's settings (:data:`DS_CLIENT_LINE`) and of where that count comes
    from (:data:`DS_TOKENS_LINE`) for the result. Returns 0, or raises.

    Raises:
        ArmError: The ds checkout, its script or the artifact directory is missing, the
            dataset is not the one run_agentx.sh replays, the run outlived its duration plus
            ``AGENTX_TIMEOUT_GRACE`` seconds (its processes are stopped), or it wrote no
            export.
        RuntimeError: The export shows an incomplete or failed run (agentx_client.py's checks)
            or holds no output token count.
        ValueError: ``AGENTX_TIMEOUT_GRACE`` is not an integer, or the export is not JSON
            (``json.JSONDecodeError``).
    """
    import argparse

    parser = argparse.ArgumentParser(prog=f"openengine_arms.py {DS_AGENTX_COMMAND}")
    for name in ("--model", "--concurrency", "--host", "--port"):
        parser.add_argument(name, required=True)
    parser.add_argument("--dataset", default=DS_AGENTX_DATASET)
    args = parser.parse_args(list(argv))
    if args.dataset != DS_AGENTX_DATASET:
        raise ArmError(
            f"run_agentx.sh replays {DS_AGENTX_DATASET}, not the lane's dataset {args.dataset}"
        )
    env = dict(os.environ if environ is None else environ)
    replaced = {
        key: env[key]
        for key, value in DS_AGENTX_SETTINGS.items()
        if key in env and env[key] != value
    }
    if replaced:
        print(f"[ds-agentx] the case-03 arms replace the lane's {replaced}", flush=True)
    env.update(DS_AGENTX_SETTINGS)
    root = env.get(ENV_DS_ROOT, "")
    script = os.path.join(root, "dev", "run_agentx.sh")
    if not root or not os.path.isfile(script):
        raise ArmError(f"the ds AgentX client needs {ENV_DS_ROOT} with dev/run_agentx.sh")
    parent = env.get(ENV_AGENTX_ARTIFACT_DIR, "")
    if not parent:
        raise ArmError(f"the ds AgentX client needs {ENV_AGENTX_ARTIFACT_DIR}")
    out_dir = os.path.abspath(os.path.join(parent, f"concurrency_{args.concurrency}"))
    os.makedirs(out_dir, exist_ok=True)
    export_path = os.path.join(out_dir, "profile_export_aiperf.json")
    if os.path.exists(export_path):
        print(f"[ds-agentx] removing the export of an earlier run: {export_path}", flush=True)
        os.remove(export_path)
    env["MODEL"] = args.model
    duration = int(env["AGENTX_DURATION"])
    settings = {
        "client": "ds dev/run_agentx.sh",
        "ds_sha": ds_sha(env),
        "window_s": duration,
        "warmup_per_lane": int(env["AGENTX_WARMUP_PER_LANE"]),
        "extra_args": env["AGENTX_EXTRA_ARGS"],
        "dataset": DS_AGENTX_DATASET,
        "warmup_learning": {key: env.get(key) for key in _AGENTX_LEARN_ENV},
    }
    print(DS_CLIENT_LINE + json.dumps(settings, sort_keys=True), flush=True)
    url = f"http://{args.host}:{args.port}"
    grace = int(env.get("AGENTX_TIMEOUT_GRACE", "900"))
    # A session of its own: on a timeout, aiperf and its workers stop with the script. So
    # they do when this process is stopped (SIGTERM, Ctrl-C), which they would otherwise
    # outlive, being outside its process group.
    with _sigterm_raises():
        proc = subprocess.Popen(
            ["bash", script, url, str(args.concurrency), str(duration), out_dir],
            env=env,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=duration + grace)
        except subprocess.TimeoutExpired:
            _stop_process_group(proc, 60)
            raise ArmError(
                f"run_agentx.sh still ran {duration + grace} s after it started; stopped it"
            )
        except BaseException:
            _stop_process_group(proc, _STOPPING_GRACE_S)
            raise
    if returncode != 0:
        print(f"[ds-agentx] run_agentx.sh exited {returncode}; checking its export", flush=True)
    if not os.path.exists(export_path):
        raise ArmError(f"run_agentx.sh exited {returncode} and wrote no {export_path}")
    with open(export_path, encoding="utf-8") as f:
        export = json.load(f)
    client = _agentx_client_module()
    client.check_run_health(export)
    client.report_metrics(export)
    tokens, source = _total_output_tokens(export)
    if tokens is None:
        raise RuntimeError(f"{export_path} holds no output token count")
    print(f"Total generated tokens: {tokens}", flush=True)
    print(DS_TOKENS_LINE + source, flush=True)
    return 0


def main(argv: Sequence[str]) -> int:
    """``openengine_arms.py ds-agentx ...``: see :func:`run_ds_agentx`."""
    if not argv or argv[0] != DS_AGENTX_COMMAND:
        print(
            f"usage: openengine_arms.py {DS_AGENTX_COMMAND} --model M --concurrency C "
            "--host H --port P [--dataset D]",
            file=sys.stderr,
        )
        return 2
    return run_ds_agentx(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
