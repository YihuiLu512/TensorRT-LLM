#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one disaggregated perf sanity test with every role on one node.

The multi-node job body (``jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh``)
gives each context and generation server nodes of its own and starts each role
as an ``srun`` step. This driver runs the same roles of one test id
(``perf/test_perf_sanity.py::test_e2e[disagg-<mode>-<config>]``) as processes on
the node it runs on, inside one container:

- one pytest per role, the role chosen by ``DISAGG_SERVING_TYPE`` (``GEN_<i>``,
  ``CTX_<i>``, ``DISAGG_SERVER``, ``BENCHMARK``), started in that order;
- each worker gets GPUs of its own (``CUDA_VISIBLE_DEVICES``), the generation
  workers' first, as many as its parallel sizes take; a worker with several GPUs
  starts its MPI workers itself, not through ``srun --mpi``;
- the workers get the UCX variables the submit scripts set for the cluster
  (``tests/test_common/cluster_env.py``); a ``gen_only`` test's generation
  workers also get the fill gate the local submit script sets;
- ``TRTLLM_PERF_SINGLE_NODE=1`` makes each generation worker's pytest write its
  server's output to ``gen_server_<i>.log`` (no srun step gathers it), and the
  driver touches ``gen_server_<i>.done`` once that pytest exits.

Only the disaggregated comparison arms write that log, so the driver needs
``TRTLLM_PERF_ARM`` naming one (``E4-base`` is the arm without the chain), with
``TRTLLM_TEST_OPENENGINE_CHAIN`` as that arm requires; every role then loads
the harness's pytest plugin, as ``openengine_chain/runner.py`` sets it up.

The driver refuses an output directory with the state of an earlier run in it,
waits for every role once the benchmark is done, then stops whatever process
of the run is left (each carries a tag in its environment) and says so.
It writes ``single-node-driver.json`` with each role's exit code and log, the
GPUs and variables each role got, and the processes it had to stop.

It does not call sbatch or srun: run it inside an allocation of one node, for
example from a job script that starts the container. Exit status: 0 if every
role's pytest passed and no process of the run was left, 1 otherwise, 2 for
bad arguments or a test the driver cannot run.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import yaml

_PERF_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_PERF_SCRIPTS)))


def _load(name: str, path: str) -> types.ModuleType:
    """The module at ``path``, loaded under ``name`` without changing ``sys.path``.

    It is registered in ``sys.modules`` first, as dataclasses look their module up.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The helpers the submit scripts parse a test's YAML with (jenkins/scripts/perf/benchmark_utils.py).
benchmark_utils = _load(
    "single_node_disagg_benchmark_utils", os.path.join(_PERF_SCRIPTS, "benchmark_utils.py")
)
# The UCX rule table the submit scripts use (tests/test_common/cluster_env.py).
cluster_env = _load(
    "single_node_disagg_cluster_env", os.path.join(_REPO, "tests", "test_common", "cluster_env.py")
)

ENV_TAG = "TRTLLM_PERF_DRIVER_TAG"
ENV_SINGLE_NODE = "TRTLLM_PERF_SINGLE_NODE"
ENV_CHAIN = "TRTLLM_TEST_OPENENGINE_CHAIN"
ENV_ARM = "TRTLLM_PERF_ARM"
ENV_RESULTS_DIR = "TRTLLM_OPENENGINE_RESULTS_DIR"
HARNESS_PACKAGE = "openengine_chain"
DISAGG_TOPOLOGY = "disagg"
PLUGIN_OPTION = f"-p {HARNESS_PACKAGE}.pytest_plugin"
DISAGG_MODES = ("e2e", "gen_only")
TIME_BREAKDOWN_MODIFIER = "time_breakdown"
# No role runs under an MPI launcher here (a worker with several GPUs spawns its
# MPI workers itself), so no role gets the launcher's MPI or Slurm variables, as
# slurm_run.sh clears them for a one-node job; every role then names its exchange directories "local".
_MPI_ENV_PREFIXES = ("PMI", "OMPI_", "MPI_", "SLURM_")
# State an earlier run leaves in a test's output directory.
STALE_PATTERNS = (
    "hostnames-*",
    "benchmark_status.*.txt",
    "gen_server_*.done",
    "gen_server_*.log",
    "perf-arm-*",
)
SETTLE_S = 5.0
STOP_GRACE_S = 30.0


class DriverError(RuntimeError):
    """The driver cannot run this test as asked."""


@dataclass(frozen=True)
class TestPlan:
    """What the driver reads of a test id and its YAML."""

    test_id: str
    mode: str
    config_path: str
    num_ctx: int
    num_gen: int
    gpus: Mapping[str, int]
    concurrency: int
    supported_gpus: tuple[str, ...]
    gen_config: Mapping
    roles: tuple[str, ...] = field(default=())


def parse_test_id(test_id: str) -> tuple[str, str]:
    """``(mode, config stem)`` of ``disagg-<mode>-<config>`` (``disagg_upload`` is refused).

    Raises:
        DriverError: The id is not a disaggregated test id the driver runs.
    """
    labels = test_id.split("-")
    if len(labels) < 3 or labels[0] != "disagg":
        raise DriverError(f"{test_id!r} is not a 'disagg-<mode>-<config>' test id")
    mode = labels[1]
    if mode not in DISAGG_MODES:
        raise DriverError(f"{test_id!r}: the driver runs the modes {DISAGG_MODES}, not {mode!r}")
    if labels[2] == TIME_BREAKDOWN_MODIFIER:
        raise DriverError(f"{test_id!r}: the driver does not run the time_breakdown modifier")
    return mode, "-".join(labels[2:])


def _parallel(config: Mapping) -> int:
    size = 1
    for key in ("tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size"):
        size *= int(config.get(key) or 1)
    return size


def load_plan(test_id: str, config_dir: str) -> TestPlan:
    """The plan of ``test_id`` from its YAML in ``config_dir``.

    Raises:
        DriverError: The id or the YAML is not one the driver runs.
        OSError: The YAML cannot be read.
    """
    mode, stem = parse_test_id(test_id)
    path = os.path.join(config_dir, f"{stem}.yaml")
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    hardware = config.get("hardware") or {}
    workers = config.get("worker_config") or {}
    num_ctx = int(hardware.get("num_ctx_servers") or 0)
    num_gen = int(hardware.get("num_gen_servers") or 0)
    if num_ctx < 1 or num_gen < 1:
        raise DriverError(f"{path}: needs at least one context and one generation server")
    if benchmark_utils.is_gen_only_no_context(mode, config):
        raise DriverError(f"{path}: the driver does not run the gen_only_no_context mode")
    concurrency = benchmark_utils.parse_positive_concurrency(
        (config.get("benchmark") or {}).get("concurrency_list", "1")
    )
    roles = tuple(f"GEN_{i}" for i in range(num_gen)) + tuple(f"CTX_{i}" for i in range(num_ctx))
    return TestPlan(
        test_id=test_id,
        mode=mode,
        config_path=path,
        num_ctx=num_ctx,
        num_gen=num_gen,
        gpus={
            "ctx": _parallel(workers.get("ctx") or {}),
            "gen": _parallel(workers.get("gen") or {}),
        },
        concurrency=concurrency,
        supported_gpus=tuple((config.get("metadata") or {}).get("supported_gpus") or ()),
        gen_config=workers.get("gen") or {},
        roles=roles,
    )


def available_gpus(environ: Mapping[str, str], query: Callable[[], int]) -> list[str]:
    """The GPUs the driver may hand out: ``CUDA_VISIBLE_DEVICES`` if set, else ``0..query()-1``."""
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    return [str(i) for i in range(query())]


def count_gpus() -> int:
    """How many GPUs ``nvidia-smi -L`` lists."""
    listing = subprocess.run(
        ["nvidia-smi", "-L"], check=True, capture_output=True, text=True, timeout=60
    ).stdout
    return sum(1 for line in listing.splitlines() if line.startswith("GPU "))


def assign_gpus(plan: TestPlan, gpus: Sequence[str]) -> dict[str, list[str]]:
    """GPUs of each worker role, the generation workers' first, none shared.

    Raises:
        DriverError: There are fewer GPUs than the workers take.
    """
    needed = plan.num_gen * plan.gpus["gen"] + plan.num_ctx * plan.gpus["ctx"]
    if needed > len(gpus):
        raise DriverError(
            f"{plan.test_id}: the workers take {needed} GPUs "
            f"({plan.num_gen} x {plan.gpus['gen']} generation, "
            f"{plan.num_ctx} x {plan.gpus['ctx']} context), only {len(gpus)} are available"
        )
    assignment = {}
    cursor = 0
    for role in plan.roles:
        count = plan.gpus["gen" if role.startswith("GEN_") else "ctx"]
        assignment[role] = list(gpus[cursor : cursor + count])
        cursor += count
    return assignment


def gen_fill_queue_size(plan: TestPlan) -> int:
    """The gen_only fill target the local submit script sets on the generation workers."""
    concurrency = plan.concurrency
    capacity = int(plan.gen_config.get("max_batch_size", concurrency))
    if plan.gen_config.get("enable_attention_dp", False):
        capacity *= int(plan.gen_config.get("tensor_parallel_size", 1))
    return min(capacity, concurrency)


def ucx_env(cluster: str, gpu_type: str) -> dict[str, str]:
    """The UCX variables the submit scripts set on the workers of ``cluster``."""
    return cluster_env.get_ucx_env(cluster, gpu_type)


def role_env(
    base: Mapping[str, str],
    plan: TestPlan,
    role: str,
    *,
    gpus: Sequence[str] | None,
    tag: str,
    ucx: Mapping[str, str],
) -> dict[str, str]:
    """The environment of ``role``'s pytest.

    Every role: ``DISAGG_SERVING_TYPE``, the config folder of the test, the
    single-node switch and the run's tag, without the launcher's MPI and Slurm
    variables.
    A worker also gets its GPUs and the cluster's UCX variables (after clearing
    the ones the submit scripts clear); a ``gen_only`` generation worker gets
    the fill gate.
    """
    env = {k: v for k, v in base.items() if not k.startswith(_MPI_ENV_PREFIXES)}
    # A role's pytest is not one started by a test.
    env.pop("PYTEST_CURRENT_TEST", None)
    env["DISAGG_SERVING_TYPE"] = role
    env["DISAGG_CONFIG_FOLDER"] = os.path.dirname(os.path.abspath(plan.config_path))
    env[ENV_SINGLE_NODE] = "1"
    env[ENV_TAG] = tag
    if gpus is None:
        return env
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    for key in cluster_env.BASE_UCX_UNSET_VARS:
        env.pop(key, None)
    env.update(ucx)
    env["FLASHINFER_JIT_DIR"] = f"/tmp/flashinfer_jit_cache_{role.lower()}"
    env.setdefault("HF_HOME", "/tmp/hf_home")
    if plan.mode == "gen_only" and role.startswith("GEN_"):
        # Generation workers only, as the local submit script sets it.
        env["TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP"] = "1"
        env["TLLM_BENCHMARK_REQ_QUEUES_SIZE"] = str(gen_fill_queue_size(plan))
    return env


def stale_state(test_output_dir: str) -> list[str]:
    """Entries of an earlier run in ``test_output_dir`` (sorted)."""
    if not os.path.isdir(test_output_dir):
        return []
    return sorted(
        name
        for name in os.listdir(test_output_dir)
        if any(fnmatch.fnmatch(name, pattern) for pattern in STALE_PATTERNS)
    )


def load_arms(defs_dir: str) -> types.ModuleType:
    """``perf/openengine_arms.py`` of the checkout."""
    return _load(
        "single_node_disagg_openengine_arms", os.path.join(defs_dir, "perf", "openengine_arms.py")
    )


def check_arm(env: Mapping[str, str], arms: Mapping) -> str:
    """The disaggregated arm ``TRTLLM_PERF_ARM`` names.

    Raises:
        DriverError: It is unset, unknown or not a disaggregated arm, or
            ``TRTLLM_TEST_OPENENGINE_CHAIN`` is not what the arm needs.
    """
    name = env.get(ENV_ARM, "")
    disagg = sorted(n for n, arm in arms.items() if arm.topology == DISAGG_TOPOLOGY)
    if arms.get(name) is None or arms[name].topology != DISAGG_TOPOLOGY:
        raise DriverError(
            f"{ENV_ARM}={name!r}: the driver runs the disaggregated arms {', '.join(disagg)}"
        )
    # The roles' pytest checks this too, but only after every role has started.
    expected = "" if arms[name].in_process else arms[name].chain
    chain = env.get(ENV_CHAIN, "")
    if chain != expected:
        want = f"set to {expected!r}" if expected else "unset"
        raise DriverError(f"{ENV_ARM}={name} needs {ENV_CHAIN} {want}; it is {chain!r}")
    return name


def harness_env(env: dict[str, str], *, defs_dir: str, run_dir: str) -> None:
    """Make the harness's pytest plugin load in every role, as ``openengine_chain/runner.py`` does.

    ``openengine_chain`` goes on ``PYTHONPATH`` through a directory holding only
    it, the plugin into ``PYTEST_ADDOPTS``, and the harness results directory
    defaults to ``<run_dir>/openengine``.
    """
    path = os.path.join(run_dir, "pythonpath")
    os.makedirs(path, exist_ok=True)
    link = os.path.join(path, HARNESS_PACKAGE)
    target = os.path.join(defs_dir, HARNESS_PACKAGE)
    if not os.path.islink(link):
        os.symlink(target, link)
    elif os.path.realpath(link) != os.path.realpath(target):
        raise DriverError(f"{link} points to {os.path.realpath(link)}, not {target}")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (path, env.get("PYTHONPATH", "")) if p)
    addopts = env.get("PYTEST_ADDOPTS", "")
    if PLUGIN_OPTION not in addopts:
        env["PYTEST_ADDOPTS"] = f"{addopts} {PLUGIN_OPTION}".strip()
    env.setdefault(ENV_RESULTS_DIR, os.path.join(run_dir, "openengine"))


def pytest_command(
    python: str, plan: TestPlan, role: str, output_dir: str, run_dir: str
) -> list[str]:
    """The pytest command of ``role`` (run from ``tests/integration/defs``)."""
    return [
        python,
        "-m",
        "pytest",
        "-v",
        f"perf/test_perf_sanity.py::test_e2e[{plan.test_id}]",
        f"--output-dir={output_dir}",
        "-o",
        "junit_logging=out-err",
        f"--junitxml={os.path.join(run_dir, f'results-{role.lower()}.xml')}",
    ]


@dataclass
class RoleRun:
    """One role's pytest."""

    role: str
    cmd: list[str]
    log: str
    gpus: list[str] | None
    proc: subprocess.Popen | None = None
    started: float = 0.0
    ended: float | None = None
    returncode: int | None = None
    stopped: bool = False

    def summary(self) -> dict:
        return {
            "role": self.role,
            "cmd": self.cmd,
            "log": self.log,
            "gpus": self.gpus,
            "pid": self.proc.pid if self.proc else None,
            "returncode": self.returncode,
            "elapsed_s": None if self.ended is None else round(self.ended - self.started, 3),
            "stopped_by_driver": self.stopped,
        }


def start_role(run: RoleRun, env: Mapping[str, str], cwd: str) -> None:
    """Start ``run``'s pytest in a session of its own, its output in ``run.log``."""
    with open(run.log, "ab") as log:
        run.proc = subprocess.Popen(
            run.cmd,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    run.started = time.monotonic()


def tagged_pids(tag: str, *, exclude: Sequence[int] = ()) -> list[int]:
    """Processes whose environment carries ``TRTLLM_PERF_DRIVER_TAG=<tag>`` (readable ones only)."""
    needle = f"{ENV_TAG}={tag}".encode()
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) in exclude:
            continue
        try:
            with open(f"/proc/{name}/environ", "rb") as f:
                entries = f.read().split(b"\0")
        except OSError:
            continue
        if needle in entries:
            found.append(int(name))
    return sorted(found)


def stop_pids(
    pids: Sequence[int], grace_s: float, signal_fn: Callable[[int, int], None] = os.kill
) -> None:
    """SIGTERM ``pids``, then SIGKILL those still alive after ``grace_s``."""
    for pid in pids:
        try:
            signal_fn(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_s
    alive = list(pids)
    while alive and time.monotonic() < deadline:
        time.sleep(0.5)
        alive = [pid for pid in alive if _alive(pid)]
    for pid in alive:
        try:
            signal_fn(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            state = f.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def touch(path: str) -> None:
    with open(path, "a", encoding="utf-8"):
        pass


def supervise(
    runs: Sequence[RoleRun],
    *,
    test_output_dir: str,
    timeout_s: float,
    after_benchmark_s: float,
    poll_s: float = 1.0,
) -> list[str]:
    """Wait for every role; returns notes on what had to be stopped.

    A generation worker's pytest exiting touches its ``gen_server_<i>.done``.
    Once the benchmark's pytest exits, the other roles get ``after_benchmark_s``
    to finish; the roles still running then, or when ``timeout_s`` runs out,
    are stopped (their process groups). A role that fails before the benchmark
    has written its status file stops the others at once. Roles still running
    a while after being stopped are noted and left to the caller.
    """
    notes = []
    start = time.monotonic()
    benchmark_done = None
    failed = None
    stopped_at = None
    while True:
        now = time.monotonic()
        for run in runs:
            if run.returncode is None and run.proc.poll() is not None:
                run.returncode = run.proc.returncode
                run.ended = now
                if run.role.startswith("GEN_"):
                    touch(os.path.join(test_output_dir, f"gen_server_{run.role[4:]}.done"))
                if run.role == "BENCHMARK":
                    benchmark_done = now
                elif (
                    run.returncode != 0
                    and failed is None
                    and not _benchmark_status(test_output_dir)
                ):
                    failed = run
        running = [run for run in runs if run.returncode is None]
        if not running:
            return notes
        if stopped_at is not None:
            if now - stopped_at > STOP_GRACE_S + _STOP_WAIT_S:
                notes.append(f"still running after being stopped: {[r.role for r in running]}")
                return notes
            time.sleep(poll_s)
            continue
        if failed is not None:
            reason = f"{failed.role} exited with {failed.returncode} before the benchmark"
            failed = None
        elif now - start > timeout_s:
            reason = f"the run took over {timeout_s:.0f}s"
        elif benchmark_done is not None and now - benchmark_done > after_benchmark_s:
            reason = f"still running {after_benchmark_s:.0f}s after the benchmark exited"
        else:
            time.sleep(poll_s)
            continue
        for run in running:
            notes.append(f"stopped {run.role}: {reason}")
            run.stopped = True
            stop_group(run.proc.pid, STOP_GRACE_S)
        stopped_at = time.monotonic()


# How long a stopped role may take to exit after its SIGKILL.
_STOP_WAIT_S = 60.0


def _benchmark_status(test_output_dir: str) -> bool:
    """Whether the benchmark has written a status file (its measured clients are over)."""
    try:
        names = os.listdir(test_output_dir)
    except OSError:
        return False
    return any(fnmatch.fnmatch(name, "benchmark_status.*.txt") for name in names)


def group_members(pgid: int) -> list[int]:
    """The live (not zombie) processes of process group ``pgid``."""
    members = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", encoding="utf-8") as f:
                fields = f.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) > 2 and fields[2] == str(pgid) and fields[0] != "Z":
            members.append(int(name))
    return members


def stop_group(pgid: int, grace_s: float) -> None:
    """SIGTERM process group ``pgid``; SIGKILL it if a member is left after ``grace_s``.

    The group's leader is a role's pytest, which the driver has not reaped, so
    its id is not reused while this runs; members outlive an exited leader.
    """
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while group_members(pgid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if group_members(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def plan_runs(
    plan: TestPlan,
    assignment: Mapping[str, list[str]],
    *,
    python: str,
    output_dir: str,
    run_dir: str,
) -> list[RoleRun]:
    """The four kinds of roles in the order they start: generation, context, server, benchmark."""
    runs = []
    for role in (*plan.roles, "DISAGG_SERVER", "BENCHMARK"):
        runs.append(
            RoleRun(
                role=role,
                cmd=pytest_command(python, plan, role, output_dir, run_dir),
                log=os.path.join(run_dir, f"{role.lower()}.log"),
                gpus=assignment.get(role),
            )
        )
    return runs


def _env_summary(env: Mapping[str, str], base: Mapping[str, str]) -> dict:
    """What the driver set or removed in a role's environment."""
    return {
        "set": {k: v for k, v in sorted(env.items()) if base.get(k) != v},
        "removed": sorted(k for k in base if k not in env),
    }


def write_report(path: str, report: Mapping) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("test_id", help="disagg-<e2e|gen_only>-<config stem>")
    parser.add_argument("--output-dir", required=True, help="pytest --output-dir of every role")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Directory holding <config stem>.yaml (default: DISAGG_CONFIG_FOLDER, "
        "else tests/scripts/perf-sanity/disaggregated)",
    )
    parser.add_argument(
        "--llm-root", default=None, help="TensorRT-LLM checkout (default: this one)"
    )
    parser.add_argument(
        "--cluster",
        default=None,
        help="Cluster name for the UCX rule (default: SLURM_CLUSTER_NAME)",
    )
    parser.add_argument(
        "--gpu-type",
        default=None,
        help="GPU type for the UCX rule (default: the YAML's supported_gpus)",
    )
    parser.add_argument(
        "--timeout", type=float, default=7200.0, help="Seconds before every role is stopped"
    )
    parser.add_argument(
        "--after-benchmark-timeout",
        type=float,
        default=600.0,
        help="Seconds the other roles get to finish after the benchmark's pytest exits",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the plan; start nothing")
    return parser.parse_args(argv)


def _exit_on_sigterm(signum, frame):
    sys.exit(128 + signum)


def main(argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    # A job's SIGTERM still stops every role and writes the report.
    previous = signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        return _main(parse_args(argv), environ)
    finally:
        signal.signal(signal.SIGTERM, previous)


def _main(args: argparse.Namespace, environ: Mapping[str, str] | None) -> int:
    outer = dict(os.environ if environ is None else environ)
    base = dict(outer)
    llm_root = os.path.abspath(
        args.llm_root or os.path.join(_PERF_SCRIPTS, os.pardir, os.pardir, os.pardir)
    )
    defs_dir = os.path.join(llm_root, "tests", "integration", "defs")
    config_dir = args.config_dir or base.get(
        "DISAGG_CONFIG_FOLDER", os.path.join("tests", "scripts", "perf-sanity", "disaggregated")
    )
    if not os.path.isabs(config_dir):
        config_dir = os.path.join(llm_root, config_dir)
    output_dir = os.path.abspath(args.output_dir)
    run_dir = os.path.join(output_dir, "single-node", args.test_id)
    test_output_dir = os.path.join(output_dir, args.test_id)
    try:
        arm = check_arm(base, load_arms(defs_dir).DISAGG_ARMS)
        plan = load_plan(args.test_id, config_dir)
        stale = stale_state(test_output_dir)
        if os.path.exists(run_dir):
            stale.append(run_dir)
        if stale:
            raise DriverError(
                f"{test_output_dir} holds the state of an earlier run ({', '.join(stale)}); "
                "use a new --output-dir"
            )
        assignment = assign_gpus(plan, available_gpus(base, count_gpus))
        cluster = args.cluster if args.cluster is not None else base.get("SLURM_CLUSTER_NAME", "")
        gpu_type = args.gpu_type or cluster_env.gpu_type_from_supported_gpus(
            list(plan.supported_gpus)
        )
        ucx = ucx_env(cluster, gpu_type)
    except (DriverError, OSError, ValueError, yaml.YAMLError, subprocess.SubprocessError) as error:
        print(f"[single-node disagg] {error}", file=sys.stderr)
        return 2
    os.makedirs(run_dir)
    os.makedirs(test_output_dir, exist_ok=True)
    base.setdefault("LLM_ROOT", llm_root)
    tag = f"{args.test_id}-{uuid.uuid4().hex[:12]}"
    harness_env(base, defs_dir=defs_dir, run_dir=run_dir)
    runs = plan_runs(
        plan, assignment, python=sys.executable, output_dir=output_dir, run_dir=run_dir
    )
    envs = {
        run.role: role_env(base, plan, run.role, gpus=run.gpus, tag=tag, ucx=ucx) for run in runs
    }
    report = {
        "test_id": plan.test_id,
        "mode": plan.mode,
        "config": plan.config_path,
        "output_dir": output_dir,
        "test_output_dir": test_output_dir,
        "tag": tag,
        "arm": arm,
        "chain": base.get(ENV_CHAIN, ""),
        "ucx": {"cluster": cluster, "gpu_type": gpu_type, "set": ucx},
        "roles": [],
        "environment": {role: _env_summary(env, outer) for role, env in envs.items()},
        "notes": [],
        "leftover_pids": [],
        "status": "planned",
    }
    report_path = os.path.join(run_dir, "single-node-driver.json")
    if args.dry_run:
        report["roles"] = [run.summary() for run in runs]
        write_report(report_path, report)
        print(report_path)
        return 0
    try:
        for run in runs:
            start_role(run, envs[run.role], defs_dir)
            print(f"[single-node disagg] started {run.role} (pid {run.proc.pid}) -> {run.log}")
        report["notes"] = supervise(
            runs,
            test_output_dir=test_output_dir,
            timeout_s=args.timeout,
            after_benchmark_s=args.after_benchmark_timeout,
        )
    finally:
        for run in runs:
            if run.proc is not None and run.proc.poll() is None:
                run.stopped = True
                stop_group(run.proc.pid, STOP_GRACE_S)
                try:
                    run.returncode = run.proc.wait(timeout=_STOP_WAIT_S)
                except subprocess.TimeoutExpired:
                    report["notes"].append(f"{run.role} did not exit after SIGKILL")
                run.ended = time.monotonic()
        time.sleep(SETTLE_S)
        leftover = tagged_pids(tag, exclude=(os.getpid(),))
        if leftover:
            report["notes"].append(f"stopped processes left by the run: {leftover}")
            stop_pids(leftover, STOP_GRACE_S)
        report["leftover_pids"] = leftover
        report["roles"] = [run.summary() for run in runs]
        passed = all(run.returncode == 0 for run in runs) and not leftover
        report["status"] = "passed" if passed else "failed"
        write_report(report_path, report)
    for run in runs:
        print(f"[single-node disagg] {run.role}: rc={run.returncode} log={run.log}")
    print(f"[single-node disagg] {report['status']}; report {report_path}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
