# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split one ``trtllm-serve`` command line between the engine and the front end."""

from __future__ import annotations

import enum
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import yaml


class Side(enum.Enum):
    FRONTEND = "frontend"
    ENGINE = "engine"
    BOTH = "both"
    # Set by the harness itself for each process.
    HARNESS = "harness"


@dataclass(frozen=True)
class Option:
    side: Side
    takes_value: bool = True


OPTIONS: Mapping[str, Option] = {
    # Front end only.
    "--tool_parser": Option(Side.FRONTEND),
    "--chat_template": Option(Side.FRONTEND),
    "--allow_request_chat_template": Option(Side.FRONTEND, takes_value=False),
    "--post_processor_hook": Option(Side.FRONTEND),
    "--num_postprocess_workers": Option(Side.FRONTEND),
    "--num_serve_frontends": Option(Side.FRONTEND),
    "--otlp_traces_endpoint": Option(Side.FRONTEND),
    "--middleware": Option(Side.FRONTEND),
    "--served_model_name": Option(Side.FRONTEND),
    "--media_io_kwargs": Option(Side.FRONTEND),
    "--num_media_load_workers": Option(Side.FRONTEND),
    "--num_input_processor_workers": Option(Side.FRONTEND),
    # Both.
    # The engine prunes video tokens (EVS) when it preprocesses the media.
    "--video_pruning_rate": Option(Side.BOTH),
    "--tokenizer": Option(Side.BOTH),
    "--trust_remote_code": Option(Side.BOTH, takes_value=False),
    "--reasoning_parser": Option(Side.BOTH),
    "--config": Option(Side.BOTH),
    "--extra_llm_api_options": Option(Side.BOTH),
    "--set": Option(Side.BOTH),
    # The front end refuses a beam width above 1 at startup; the engine is sized with it.
    "--max_beam_width": Option(Side.BOTH),
    # A disaggregated worker's role and service discovery: the front end registers the worker.
    "--server_role": Option(Side.FRONTEND),
    "--disagg_cluster_uri": Option(Side.FRONTEND),
    # Engine only.
    "--backend": Option(Side.ENGINE),
    "--tp_size": Option(Side.ENGINE),
    "--pp_size": Option(Side.ENGINE),
    "--ep_size": Option(Side.ENGINE),
    "--cp_size": Option(Side.ENGINE),
    "--gpus_per_node": Option(Side.ENGINE),
    "--max_batch_size": Option(Side.ENGINE),
    "--max_num_tokens": Option(Side.ENGINE),
    "--max_seq_len": Option(Side.ENGINE),
    "--kv_cache_free_gpu_memory_fraction": Option(Side.ENGINE),
    "--enable_chunked_prefill": Option(Side.ENGINE, takes_value=False),
    "--enable_attention_dp": Option(Side.ENGINE, takes_value=False),
    # The engine's Prometheus port (launch.ENGINE_METRICS_PORT_OPTION).
    "--openengine-metrics-port": Option(Side.ENGINE),
    # Each process gets its own address.
    "--host": Option(Side.HARNESS),
    "--port": Option(Side.HARNESS),
}

CONFIG_OPTIONS = ("--config", "--extra_llm_api_options")

# Options of a command line the chain cannot start at all, with the reason.
REFUSED_OPTIONS: Mapping[str, str] = {
    "--metadata_server_config_file": (
        "a worker front end does not register through a metadata server; register the "
        "test as skip in the expectation table"
    ),
}

# YAML keys that configure the front end's post-processing and request handling; the engine
# does not get them (with --grpc it rejects allow_request_chat_template).
FRONTEND_ONLY_YAML_KEYS = (
    "num_postprocess_workers",
    "num_serve_frontends",
    "postprocess_tokenizer_dir",
    "post_processor_hook",
    "allow_request_chat_template",
)

# The configuration key whose iteration stats need the engine's KV-cache load.
_ITERATION_STATS_KEY = "enable_iter_perf_stats"

# Environment variables that only configure the engine.
ENGINE_ONLY_ENV = ("TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS",)

_SERVE_MODULE = "tensorrt_llm.commands.serve"


class UnknownServeOptionError(ValueError):
    """The command line has an option the split table does not list."""


class ChainUnrunnableError(ValueError):
    """The test's configuration cannot run in the chain; the message says why."""


@dataclass(frozen=True)
class ServeCommand:
    """A ``trtllm-serve`` command line, taken apart."""

    # The executable part, including the "serve" subcommand when given.
    prefix: tuple[str, ...]
    model: str
    # Every token other than the prefix and the model, in order.
    args: tuple[str, ...]


@dataclass(frozen=True)
class SplitArgs:
    engine: tuple[str, ...]
    frontend: tuple[str, ...]
    # The values of --host and --port, if the command line sets them.
    host: str | None
    port: str | None


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def _iter_options(args: Sequence[str]):
    """Yield ``(name, tokens, value)`` per option, or ``(None, [token], None)`` per positional."""
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            yield None, [token], None
            i += 1
            continue
        name = _option_name(token)
        if name in REFUSED_OPTIONS:
            raise ChainUnrunnableError(
                f"not runnable in the OpenEngine chain: {name}: {REFUSED_OPTIONS[name]}"
            )
        option = OPTIONS.get(name)
        if option is None:
            raise UnknownServeOptionError(
                f"trtllm-serve option {name!r} is not in the OpenEngine chain split table "
                "(tests/integration/defs/openengine_chain/split.py OPTIONS); add it to the "
                "table with the side it belongs to"
            )
        if "=" in token:
            if not option.takes_value:
                raise UnknownServeOptionError(f"flag {name!r} does not take a value: {token!r}")
            yield name, [token], token.split("=", 1)[1]
            i += 1
        elif option.takes_value:
            if i + 1 >= len(args):
                raise UnknownServeOptionError(f"option {name!r} is missing its value")
            yield name, [token, args[i + 1]], args[i + 1]
            i += 2
        else:
            yield name, [token], None
            i += 1


def parse_serve_command(cmd: Sequence[str]) -> ServeCommand:
    """Take apart ``trtllm-serve [serve] MODEL ...`` or ``python -m tensorrt_llm.commands.serve ...``."""
    cmd = list(cmd)
    if cmd and os.path.basename(cmd[0]) == "trtllm-serve":
        n = 1
    elif len(cmd) >= 3 and cmd[1:3] == ["-m", _SERVE_MODULE]:
        n = 3
    else:
        raise ValueError(f"not a trtllm-serve command: {cmd!r}")
    if len(cmd) > n and cmd[n] == "serve":
        n += 1
    rest = cmd[n:]
    positionals = []
    args = []
    for name, tokens, _ in _iter_options(rest):
        if name is None:
            positionals.append(tokens[0])
        else:
            args.extend(tokens)
    if len(positionals) != 1:
        raise ValueError(f"expected exactly one model argument, got {positionals!r} in {cmd!r}")
    return ServeCommand(prefix=tuple(cmd[:n]), model=positionals[0], args=tuple(args))


def engine_yaml(path: str, out_dir: str) -> str:
    """Return a YAML path for the engine: ``path`` itself, or a copy without the front-end keys."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must hold a mapping, got {type(config).__name__}")
    dropped = [key for key in FRONTEND_ONLY_YAML_KEYS if key in config]
    if not dropped:
        return path
    for key in dropped:
        del config[key]
    os.makedirs(out_dir, exist_ok=True)
    # Unique per call: two configs of one command line may share a file name.
    fd, engine_path = tempfile.mkstemp(
        prefix="engine-", suffix=f"-{os.path.basename(path)}", dir=out_dir
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return engine_path


def split_args(args: Sequence[str], *, rewrite_yaml: Callable[[str], str]) -> SplitArgs:
    """Split the options of one command line between the two processes.

    ``rewrite_yaml`` maps a ``--config`` path to the path the engine gets.
    ``--set`` overrides of front-end-only keys stay with the front end, as
    those keys do in a ``--config`` file.
    """
    engine: list[str] = []
    frontend: list[str] = []
    host = port = None
    for name, tokens, value in _iter_options(args):
        if name is None:
            raise ValueError(
                f"unexpected positional argument {tokens[0]!r}; pass the model separately"
            )
        side = OPTIONS[name].side
        if side is Side.HARNESS:
            if name == "--host":
                host = value
            else:
                port = value
            continue
        if side in (Side.FRONTEND, Side.BOTH):
            frontend.extend(tokens)
        if side is Side.ENGINE:
            engine.extend(tokens)
        elif side is Side.BOTH:
            if name in CONFIG_OPTIONS:
                engine.extend([name, rewrite_yaml(value)])
            elif name == "--set" and value.split("=", 1)[0].strip() in FRONTEND_ONLY_YAML_KEYS:
                continue
            else:
                engine.extend(tokens)
    return SplitArgs(engine=tuple(engine), frontend=tuple(frontend), host=host, port=port)


def option_value(args: Sequence[str], name: str) -> str | None:
    """The value ``args`` gives option ``name`` last, or ``None`` if it does not give one."""
    value = None
    for found, _, found_value in _iter_options(args):
        if found == name:
            value = found_value
    return value


def iteration_stats_enabled(args: Sequence[str]) -> bool:
    """Whether the configuration of one command line turns on ``enable_iter_perf_stats``.

    As in ``trtllm-serve``: ``--config`` and ``--extra_llm_api_options`` name
    one option, so only the last file given counts, and ``--set`` overrides it.
    """
    config_file = None
    sets = []
    for name, _, value in _iter_options(args):
        if name in CONFIG_OPTIONS:
            config_file = value
        elif name == "--set":
            sets.append(value)
    enabled = False
    if config_file is not None:
        with open(config_file, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        if isinstance(config, dict) and _ITERATION_STATS_KEY in config:
            enabled = _as_bool(config[_ITERATION_STATS_KEY])
    for assignment in sets:
        key, _, raw = assignment.partition("=")
        if key.strip() == _ITERATION_STATS_KEY:
            enabled = _as_bool(yaml.safe_load(raw))
    return enabled


def _as_bool(value: Any) -> bool:
    """A configuration boolean as pydantic reads it, strings such as ``"false"`` included."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "t", "yes", "y", "on")
    return bool(value)


def frontend_env(env: Mapping[str, str]) -> dict[str, str]:
    """The front end's environment: ``env`` without the engine-only variables."""
    return {key: value for key, value in env.items() if key not in ENGINE_ONLY_ENV}


# --- Disaggregated workers --------------------------------------------------------------
#
# A worker is disaggregated when its command line gives a role or a service-discovery URI,
# or its configuration has a service-discovery or KV-transfer section. Only such workers
# get the rules below; an aggregated server is split as before.

# YAML keys only the front end gets in a disaggregated worker: the engine's gRPC mode
# refuses service discovery, and only the front end registers the worker.
DISAGG_FRONTEND_ONLY_YAML_KEYS = ("disagg_cluster",)
# Environment variables only the engine reads, in a disaggregated worker.
DISAGG_ENGINE_ONLY_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY",
    "TRTLLM_DISAGG_NO_RETRY",
    "TRTLLM_WORKER_DISABLE_GC",
)
DISAGG_ENGINE_ONLY_ENV_PREFIXES = ("TRTLLM_KVCACHE_", "TLLM_")
# KV-transfer backends of the C++ transceiver, which needs KV cache manager V1.
CPP_TRANSCEIVER_BACKENDS = ("UCX", "MPI", "MOONCAKE")
# The transceiver runtime a configuration gets when it names none.
DEFAULT_TRANSCEIVER_RUNTIME = "PYTHON"
# Router types of the disaggregated server that read the workers' KV cache events.
KV_AWARE_ROUTERS = ("kv_cache_aware",)
# The configuration keys of the three parallel sizes, and their command-line options.
_PARALLEL_KEYS = (
    ("tensor_parallel_size", "--tp_size"),
    ("pipeline_parallel_size", "--pp_size"),
    ("context_parallel_size", "--cp_size"),
)


def last_config_file(args: Sequence[str]) -> str | None:
    """The configuration file ``trtllm-serve`` reads: the last ``--config`` or ``--extra_llm_api_options``."""
    path = None
    for name, _, value in _iter_options(args):
        if name in CONFIG_OPTIONS:
            path = value
    return path


def load_config(path: str | None) -> dict[str, Any]:
    """The mapping in YAML file ``path``, or an empty one for ``None``."""
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{path} must hold a mapping, got {type(config).__name__}")
    return config


def is_disagg_worker(args: Sequence[str]) -> bool:
    """Whether a ``trtllm-serve`` command line (without its model) starts a disaggregated worker."""
    if option_value(args, "--server_role") is not None:
        return True
    if option_value(args, "--disagg_cluster_uri") is not None:
        return True
    config = load_config(last_config_file(args))
    return "disagg_cluster" in config or "cache_transceiver_config" in config


def disagg_frontend_env(env: Mapping[str, str]) -> dict[str, str]:
    """A disaggregated worker front end's environment: ``env`` without what only the engine reads."""
    return {
        key: value
        for key, value in frontend_env(env).items()
        if key not in DISAGG_ENGINE_ONLY_ENV and not key.startswith(DISAGG_ENGINE_ONLY_ENV_PREFIXES)
    }


def parallel_size(args: Sequence[str]) -> int:
    """GPUs one instance of the command line's model takes: tensor x pipeline x context parallel.

    The configuration file wins over the command-line options, as in ``trtllm-serve``.
    """
    config = load_config(last_config_file(args))
    size = 1
    for key, option in _PARALLEL_KEYS:
        value = config.get(key, option_value(args, option))
        size *= int(value) if value is not None else 1
    return size


def kv_events_wanted(config: Mapping[str, Any], kv_aware_router: bool | None) -> bool:
    """Whether the engine needs the streaming KV event publisher.

    A worker whose configuration sets ``kv_cache_config.event_buffer_max_size``
    serves KV cache events. The engine of the chain publishes them only through
    ``kv_events_config``. ``kv_aware_router`` says whether the disaggregated
    server routes on KV cache events; ``None`` (the harness does not know the
    server's configuration) counts as yes.
    """
    kv_cache = config.get("kv_cache_config") or {}
    if not kv_cache.get("event_buffer_max_size"):
        return False
    return kv_aware_router is not False


def _transceiver_template(config: dict[str, Any]) -> None:
    """Spell out the KV-transfer settings the engine needs, in ``config``.

    The backend is always written, so that the engine builds a transceiver:
    NIXL for the Python runtime, the only backend it takes; DEFAULT for the C++
    runtime when the configuration names none. A C++-only backend gets the C++
    runtime and KV cache manager V1 unless the configuration chooses otherwise.
    Nothing else is set: ``kv_transfer_timeout_ms`` and ``max_tokens_in_buffer``
    keep the configuration's values or their defaults.
    """
    transceiver = dict(config.get("cache_transceiver_config") or {})
    backend = transceiver.get("backend")
    if backend in CPP_TRANSCEIVER_BACKENDS:
        transceiver.setdefault("transceiver_runtime", "CPP")
        kv_cache = dict(config.get("kv_cache_config") or {})
        kv_cache.setdefault("use_kv_cache_manager_v2", False)
        config["kv_cache_config"] = kv_cache
    elif transceiver.get("transceiver_runtime", DEFAULT_TRANSCEIVER_RUNTIME) == "PYTHON":
        if backend in (None, "DEFAULT"):
            transceiver["backend"] = "NIXL"
    elif backend is None:
        transceiver["backend"] = "DEFAULT"
    config["cache_transceiver_config"] = transceiver


def transceiver_runtime(config: Mapping[str, Any]) -> str:
    """The transceiver runtime an engine configuration selects: ``PYTHON`` or ``CPP``."""
    transceiver = config.get("cache_transceiver_config") or {}
    runtime = transceiver.get("transceiver_runtime", DEFAULT_TRANSCEIVER_RUNTIME)
    return "PYTHON" if runtime == "PYTHON" else "CPP"


def attention_dp_ranks(config: Mapping[str, Any], args: Sequence[str]) -> int:
    """KV event publishers of one engine: one per attention-DP rank."""
    enabled = config.get("enable_attention_dp")
    if enabled is None:
        enabled = any(name == "--enable_attention_dp" for name, _, _ in _iter_options(args))
    if not enabled:
        return 1
    value = config.get("tensor_parallel_size", option_value(args, "--tp_size"))
    return max(1, int(value) if value is not None else 1)


def _kv_events_template(
    config: dict[str, Any], args: Sequence[str], ports: Callable[[int], int]
) -> None:
    """Replace ``event_buffer_max_size`` with the streaming publisher, in ``config``.

    ``ports(n)`` returns the first of ``n`` consecutive free ports. Each
    attention-DP rank binds the base port plus its rank, for the live and the
    replay endpoint alike.
    """
    for key, option in _PARALLEL_KEYS[1:]:
        value = config.get(key, option_value(args, option))
        if value is not None and int(value) > 1:
            raise ChainUnrunnableError(
                f"not runnable in the OpenEngine chain (DV-23): KV cache events need {key} 1, "
                f"got {value}"
            )
    if transceiver_runtime(config) == "CPP":
        raise ChainUnrunnableError(
            "not runnable in the OpenEngine chain (DV-23): KV cache events need KV cache "
            "manager V2, which the C++ transceiver runtime cannot use"
        )
    kv_cache = dict(config.get("kv_cache_config") or {})
    if kv_cache.get("use_kv_cache_manager_v2") is False:
        raise ChainUnrunnableError(
            "not runnable in the OpenEngine chain (DV-23): KV cache events need KV cache "
            "manager V2, and the configuration turns it off"
        )
    kv_cache.pop("event_buffer_max_size", None)
    kv_cache["use_kv_cache_manager_v2"] = True
    kv_cache.setdefault("enable_block_reuse", True)
    ranks = attention_dp_ranks(config, args)
    base = ports(2 * ranks)
    kv_cache["kv_events_config"] = {
        "enable_kv_cache_events": True,
        "publisher": "zmq",
        "endpoint": f"tcp://127.0.0.1:{base}",
        "replay_endpoint": f"tcp://127.0.0.1:{base + ranks}",
    }
    config["kv_cache_config"] = kv_cache


def disagg_engine_yaml(
    path: str | None,
    out_dir: str,
    *,
    args: Sequence[str],
    kv_aware_router: bool | None,
    ports: Callable[[int], int],
) -> str:
    """Write the engine's configuration for a disaggregated worker and return its path.

    ``path`` is the worker's configuration file (``None``: it has none) and
    ``args`` its command line. The copy drops the front-end-only keys, spells
    out the KV-transfer settings and, when the worker serves KV cache events,
    sets up the streaming publisher (:func:`kv_events_wanted`).
    """
    config = load_config(path)
    for key in FRONTEND_ONLY_YAML_KEYS + DISAGG_FRONTEND_ONLY_YAML_KEYS:
        config.pop(key, None)
    _transceiver_template(config)
    if kv_events_wanted(config, kv_aware_router):
        _kv_events_template(config, args, ports)
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(path) if path is not None else "config.yaml"
    fd, engine_path = tempfile.mkstemp(prefix="engine-", suffix=f"-{name}", dir=out_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return engine_path


# --- Moving an aggregated test to the disaggregated form --------------------------------

# The free GPU memory share trtllm-serve gives the KV cache by default.
DEFAULT_KV_CACHE_FREE_GPU_MEMORY_FRACTION = 0.9
WORKER_ROLES = ("context", "generation")


@dataclass(frozen=True)
class DisaggWorkerCommands:
    """One aggregated command line taken apart for a context and a generation worker."""

    # ``role -> command line`` for the two workers, in WORKER_ROLES order.
    commands: Mapping[str, tuple[str, ...]]
    # The configuration file each worker got (written under the output directory).
    configs: Mapping[str, str]
    # The two workers use the same GPUs (a model that fits one GPU).
    share_gpu: bool
    # GPUs one worker takes.
    gpus_per_worker: int


def disagg_worker_commands(
    cmd: Sequence[str],
    *,
    ports: Mapping[str, int],
    host: str,
    auth_key: str,
    out_dir: str,
) -> DisaggWorkerCommands:
    """Take one aggregated ``trtllm-serve`` command line apart for two disaggregated workers.

    Each worker gets every option of ``cmd`` except ``--host`` and ``--port``:
    the model and engine options start an engine per worker, the front-end
    options a front end per worker. Each also gets ``--host host``, its port
    from ``ports``, ``--server_role``, and a configuration file: the one
    ``cmd`` reads, plus a KV transceiver (unless it has one), the
    disaggregated server's signing key and, when the two workers share a GPU,
    half of the KV cache memory share each.
    """
    command = parse_serve_command(cmd)
    if is_disagg_worker(command.args):
        raise ValueError(f"already a disaggregated worker, not moved: {list(cmd)!r}")
    config_path = last_config_file(command.args)
    config = load_config(config_path)
    config.setdefault("cache_transceiver_config", {"backend": "DEFAULT"})
    config["internal_request_auth_key"] = auth_key
    gpus = parallel_size(command.args)
    share_gpu = gpus == 1
    if share_gpu:
        kv_cache = dict(config.get("kv_cache_config") or {})
        fraction = kv_cache.get("free_gpu_memory_fraction")
        if fraction is None:
            given = option_value(command.args, "--kv_cache_free_gpu_memory_fraction")
            fraction = (
                float(given) if given is not None else (DEFAULT_KV_CACHE_FREE_GPU_MEMORY_FRACTION)
            )
        kv_cache["free_gpu_memory_fraction"] = float(fraction) / 2
        config["kv_cache_config"] = kv_cache
    kept: list[str] = []
    for name, tokens, _ in _iter_options(command.args):
        if name not in ("--host", "--port") and name not in CONFIG_OPTIONS:
            kept.extend(tokens)
    os.makedirs(out_dir, exist_ok=True)
    commands: dict[str, tuple[str, ...]] = {}
    configs: dict[str, str] = {}
    for role in WORKER_ROLES:
        fd, path = tempfile.mkstemp(prefix=f"{role}-", suffix=".yaml", dir=out_dir)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        configs[role] = path
        commands[role] = (
            *command.prefix,
            command.model,
            "--host",
            host,
            "--port",
            str(ports[role]),
            "--server_role",
            role,
            *kept,
            "--extra_llm_api_options",
            path,
        )
    return DisaggWorkerCommands(
        commands=commands, configs=configs, share_gpu=share_gpu, gpus_per_worker=gpus
    )


def disagg_server_config(
    *, host: str, port: int, workers: Mapping[str, str], auth_key: str
) -> dict[str, Any]:
    """The disaggregated server's configuration for one worker per role at ``workers[role]`` (host:port)."""
    return {
        "hostname": host,
        "port": port,
        "context_servers": {"num_instances": 1, "urls": [workers["context"]]},
        "generation_servers": {"num_instances": 1, "urls": [workers["generation"]]},
        "internal_request_auth_key": auth_key,
    }


def worker_gpus(visible: Sequence[str], gpus_per_worker: int, share_gpu: bool) -> dict[str, str]:
    """``role -> CUDA_VISIBLE_DEVICES`` for two workers on the GPUs ``visible``.

    Workers that share a GPU both get the first one; otherwise the context
    worker gets the first ``gpus_per_worker`` and the generation worker the
    next ones. Raises :class:`ChainUnrunnableError` if there are too few.
    """
    needed = gpus_per_worker if share_gpu else 2 * gpus_per_worker
    if len(visible) < needed:
        raise ChainUnrunnableError(
            f"not runnable here: the disaggregated form needs {needed} GPUs "
            f"({gpus_per_worker} per worker), {len(visible)} visible"
        )
    if share_gpu:
        shared = ",".join(visible[:gpus_per_worker])
        return {role: shared for role in WORKER_ROLES}
    return {
        "context": ",".join(visible[:gpus_per_worker]),
        "generation": ",".join(visible[gpus_per_worker:needed]),
    }
