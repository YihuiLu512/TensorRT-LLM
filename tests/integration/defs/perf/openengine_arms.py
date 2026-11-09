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
with the TensorRT-LLM chain, the harness also starts the engine with
``--openengine-enable-load-metrics`` when the test's configuration turns on
``enable_iter_perf_stats``. Every other setting of the test is kept; a server that needs a fixed port gets
one chosen in advance. Only the plain aggregated perf sanity cases (no
benchmark mode such as ``ctx_only``, no ``time_breakdown``) and the legacy
serve perf tests take an arm; the legacy tests still write their usual perf
records under the test's own name. The other perf sanity cases (multi-node
ones included), perf sanity test ids that upload results, clients other than
benchmark_serving (accuracy runs included) and the legacy trtllm-bench tests
refuse one, and the arms with a chain refuse the options that the OpenEngine
front end's option check refuses; perf tests that do not read these
variables ignore them.

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

- ``TRTLLM_PERF_ARM``: the arm, one of :data:`ARMS`.
- ``TRTLLM_PERF_RUN``: the run number written to the result (default 1).
- ``TRTLLM_PERF_GLM_SHA``: the glm engine's commit written to the result.
- ``TRTLLM_OPENENGINE_RESULTS_DIR``: result files go to its ``perf``
  subdirectory; unset, they go next to the test's own logs.
- ``TRTLLM_TEST_GLM_SERVE_CMD``: the glm server's command prefix, for the
  case-01 arms.
"""

from __future__ import annotations

import copy
import functools
import importlib.util
import json
import os
import random
import re
import shlex
import socket
import subprocess
import tempfile
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
class Arm:
    """One comparison arm.

    Attributes:
        name: The value of ``TRTLLM_PERF_ARM``.
        case: ``case-02`` (TRT-LLM engine) or ``case-01`` (glm engine).
        chain: ``none``, ``trtllm`` or ``glm``.
        glm_http: Serve with the glm server's own HTTP front end instead of trtllm-serve.
        frontends: ``num_serve_frontends`` the arm sets, or ``None`` to keep the test's.
        postprocess_workers: ``num_postprocess_workers`` the arm sets, or ``None`` to keep the test's.
    """

    name: str
    case: str
    chain: str
    glm_http: bool
    frontends: int | None
    postprocess_workers: int | None

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
    if not name:
        if chain:
            raise ArmError(
                f"{ENV_CHAIN}={chain!r} is set for a perf test; set {ENV_ARM} to one of "
                f"{', '.join(ARMS)} to choose the comparison arm"
            )
        return None
    arm = ARMS.get(name)
    if arm is None:
        raise ArmError(f"{ENV_ARM}={name!r}; expected one of {', '.join(ARMS)}")
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
    such cases say so instead.
    """
    return arm is None or arm.in_process


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
            ``TRTLLM_TEST_GLM_SERVE_CMD``.
        ValueError: The glm HTTP arm cannot split the command line.
        OSError: The YAML cannot be read or its copy written.
        yaml.YAMLError: The YAML or an option value cannot be parsed.
        ImportError: An arm with a chain cannot import TensorRT-LLM's
            OpenEngine front end, or the glm HTTP arm the harness.
    """
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
        for pid, (start, ticks) in after.items():
            before = self.cpu_before.get(pid)
            if before is not None and before[0] == start:
                total += ticks - before[1]
            elif pid not in pids:
                # The PID of a front-end process that exited, now another process's.
                continue
            elif self.started is None or start >= self.started:
                total += ticks
            else:
                older += 1
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
    arm = ARMS.get(doc["arm"])
    if arm is None:
        errors.append(f"arm {doc['arm']!r} is not one of {list(ARMS)}")
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
