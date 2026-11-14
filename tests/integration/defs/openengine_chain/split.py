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
    "--custom_tokenizer": Option(Side.BOTH),
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

# Other names trtllm-serve accepts for options of the table; they split, and are read, as
# the table's name.
OPTION_ALIASES: Mapping[str, str] = {
    "--tensor_parallel_size": "--tp_size",
    "--pipeline_parallel_size": "--pp_size",
    "--context_parallel_size": "--cp_size",
    "--moe_expert_parallel_size": "--ep_size",
    "--free_gpu_memory_fraction": "--kv_cache_free_gpu_memory_fraction",
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
    name = token.split("=", 1)[0]
    return OPTION_ALIASES.get(name, name)


def _iter_options(args: Sequence[str], strict: bool = True):
    """Yield ``(name, tokens, value)`` per option, or ``(None, [token], None)`` per positional.

    ``strict`` refuses the options the split table does not list or refuses.
    Without it, as for a command line that runs unchanged, such an option takes
    the next token as its value unless that token starts with ``--``.
    """
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            yield None, [token], None
            i += 1
            continue
        name = _option_name(token)
        if not strict and (name in REFUSED_OPTIONS or name not in OPTIONS):
            if "=" in token:
                yield name, [token], token.split("=", 1)[1]
                i += 1
            elif i + 1 < len(args) and not args[i + 1].startswith("--"):
                yield name, [token, args[i + 1]], args[i + 1]
                i += 2
            else:
                yield name, [token], None
                i += 1
            continue
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


def parse_serve_command(cmd: Sequence[str], strict: bool = True) -> ServeCommand:
    """Take apart ``trtllm-serve [serve] MODEL ...`` or ``python -m tensorrt_llm.commands.serve ...``.

    ``strict`` as in :func:`_iter_options`.
    """
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
    for name, tokens, _ in _iter_options(rest, strict):
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
            elif name == "--set" and (
                value.split("=", 1)[0].strip() in FRONTEND_ONLY_YAML_KEYS
                or _set_path(value)[0] in DISAGG_FRONTEND_ONLY_YAML_KEYS
            ):
                continue
            else:
                engine.extend(tokens)
    return SplitArgs(engine=tuple(engine), frontend=tuple(frontend), host=host, port=port)


def option_value(args: Sequence[str], name: str, strict: bool = True) -> str | None:
    """The value ``args`` gives option ``name`` last, or ``None`` if it does not give one."""
    value = None
    for found, _, found_value in _iter_options(args, strict):
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
# or its configuration has a service-discovery section. Only such workers get the rules
# below; an aggregated server is split as before, even with a KV-transfer section.

# YAML keys only the front end gets in a disaggregated worker: the engine's gRPC mode
# refuses service discovery, and only the front end registers the worker.
DISAGG_FRONTEND_ONLY_YAML_KEYS = ("disagg_cluster",)
# Environment variables a disaggregated worker's front end does not get, besides the
# ENGINE_ONLY_ENV of any front end: settings only a worker's engine reads (two
# disaggregated-serving switches and the executor worker's garbage-collection switch).
DISAGG_ENGINE_ONLY_ENV = (
    "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY",
    "TRTLLM_DISAGG_NO_RETRY",
    "TRTLLM_WORKER_DISABLE_GC",
)
# The variable trtllm-serve sets in its own process from --server_role (lower case) before
# it builds the model; some models and cache managers read it.
DISAGG_ROLE_ENV = "TRTLLM_DISAGG_ROLE"
# KV-transfer backends of the C++ transceiver, which needs KV cache manager V1.
CPP_TRANSCEIVER_BACKENDS = ("UCX", "MPI", "MOONCAKE")
# The transceiver runtime a configuration gets when it names none.
DEFAULT_TRANSCEIVER_RUNTIME = "PYTHON"
# The legacy environment variables that resolve backend DEFAULT, by priority, as in
# tensorrt_llm.llmapi.llm_args._CACHE_TRANSCEIVER_BACKEND_ENV_VARS.
BACKEND_ENV_VARS = (
    ("TRTLLM_USE_NIXL_KVCACHE", "NIXL"),
    ("TRTLLM_USE_UCX_KVCACHE", "UCX"),
    ("TRTLLM_USE_MOONCAKE_KVCACHE", "MOONCAKE"),
    ("TRTLLM_USE_MPI_KVCACHE", "MPI"),
)
# Router types of the disaggregated server that read the workers' KV cache events
# (the server compares router types in lower case).
KV_AWARE_ROUTERS = ("kv_cache_aware",)
# The configuration keys of the three parallel sizes, and their command-line options.
_PARALLEL_KEYS = (
    ("tensor_parallel_size", "--tp_size"),
    ("pipeline_parallel_size", "--pp_size"),
    ("context_parallel_size", "--cp_size"),
)
# ``--set`` paths a disaggregated worker cannot take: the harness writes these settings into
# the engine's configuration, and a ``--set`` would override them there.
_DISAGG_RULE_SET_PATHS = (
    ("cache_transceiver_config",),
    ("kv_cache_config", "event_buffer_max_size"),
    ("kv_cache_config", "use_kv_cache_manager_v2"),
    ("kv_cache_config", "kv_events_config"),
)


def is_kv_aware_router(router_type: str | None) -> bool:
    """Whether a disaggregated server's router type routes on KV cache events."""
    return router_type is not None and router_type.lower() in KV_AWARE_ROUTERS


def last_config_file(args: Sequence[str], strict: bool = True) -> str | None:
    """The configuration file ``trtllm-serve`` reads: the last ``--config`` or ``--extra_llm_api_options``."""
    path = None
    for name, _, value in _iter_options(args, strict):
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


def _set_path(assignment: str) -> tuple[str, ...]:
    """The configuration path of one ``--set PATH=VALUE``."""
    return tuple(part.strip() for part in assignment.split("=", 1)[0].split("."))


def set_overrides(args: Sequence[str], strict: bool = True) -> list[tuple[tuple[str, ...], Any]]:
    """The ``--set`` overrides of a command line, in order: ``(path, value)``.

    A malformed one (no ``=``, or a value that is not YAML) is left out:
    ``trtllm-serve`` refuses it when it starts.
    """
    found = []
    for name, _, value in _iter_options(args, strict):
        if name != "--set" or value is None or "=" not in value:
            continue
        try:
            parsed = yaml.safe_load(value.split("=", 1)[1])
        except (yaml.YAMLError, ValueError, OverflowError, RecursionError):
            continue
        found.append((_set_path(value), parsed))
    return found


def setting(
    config: Mapping[str, Any],
    args: Sequence[str],
    path: tuple[str, ...],
    option: str | None = None,
    strict: bool = True,
) -> Any:
    """The value ``trtllm-serve`` gives configuration ``path``, or ``None`` for its default.

    The last ``--set`` of the path or of a path above it wins, as
    ``trtllm-serve`` applies ``--set`` last and replaces the whole value at
    its path (``null`` included); then the command-line ``option``, when the
    command line gives it; then the configuration file.
    """
    overrides = [
        (found, found_value)
        for found, found_value in set_overrides(args, strict)
        if path[: len(found)] == found
    ]
    if overrides:
        found, node = overrides[-1]
        return _lookup(node, path[len(found) :])
    if option is not None:
        given = option_value(args, option, strict)
        if given is not None:
            return given
        if not OPTIONS.get(option, Option(Side.ENGINE)).takes_value:
            if any(name == option for name, _, _ in _iter_options(args, strict)):
                return True
    return _lookup(config, path)


def _lookup(node: Any, path: tuple[str, ...]) -> Any:
    """The value at ``path`` below ``node``, or ``None`` if there is none."""
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def is_disagg_worker(args: Sequence[str], strict: bool = True) -> bool:
    """Whether a ``trtllm-serve`` command line (without its model) starts a disaggregated worker."""
    if option_value(args, "--server_role", strict) is not None:
        return True
    if option_value(args, "--disagg_cluster_uri", strict) is not None:
        return True
    if any(path[0] == "disagg_cluster" for path, _ in set_overrides(args, strict)):
        return True
    return "disagg_cluster" in load_config(last_config_file(args, strict))


def disagg_frontend_env(env: Mapping[str, str]) -> dict[str, str]:
    """A disaggregated worker front end's environment: ``env`` without what it does not get."""
    return {
        key: value for key, value in frontend_env(env).items() if key not in DISAGG_ENGINE_ONLY_ENV
    }


def disagg_engine_env(
    env: Mapping[str, str], args: Sequence[str], strict: bool = True
) -> dict[str, str]:
    """A disaggregated worker engine's environment: ``env``, with :data:`DISAGG_ROLE_ENV` set.

    The engine does not get ``--server_role``, so it gets the variable that
    option sets, with the same value (and in place of any value ``env`` has).
    """
    engine_env = dict(env)
    role = option_value(args, "--server_role", strict)
    if role is not None:
        engine_env[DISAGG_ROLE_ENV] = role.lower()
    return engine_env


def parallel_size(args: Sequence[str], strict: bool = True) -> int:
    """GPUs one instance of the command line's model takes: tensor x pipeline x context parallel.

    Each size is resolved as :func:`setting` resolves it.
    """
    config = load_config(last_config_file(args, strict))
    size = 1
    for key, option in _PARALLEL_KEYS:
        value = setting(config, args, (key,), option, strict)
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


def env_backend(env: Mapping[str, str]) -> str | None:
    """The backend the legacy environment variables select for backend DEFAULT, if any."""
    for name, backend in BACKEND_ENV_VARS:
        if env.get(name) == "1":
            return backend
    return None


def _transceiver_template(config: dict[str, Any], env: Mapping[str, str]) -> None:
    """Spell out the KV-transfer settings the engine needs, in ``config``.

    The backend is always written, so that the engine builds a transceiver.
    Without one, or with DEFAULT, the Python runtime, ``auto`` and a null
    runtime with ``enable_pipelined_transfer`` get the backend the legacy
    environment variables select, as the engine would resolve DEFAULT, or
    else NIXL; the C++ runtime gets DEFAULT. (The engine
    resolves ``auto`` to the model's preferred runtime, or else to the Python
    one, falling back to the C++ one for a backend other than NIXL or a null
    ``kv_transfer_timeout_ms``.) A C++-only backend gets the C++ runtime and KV cache manager V1
    unless the configuration chooses otherwise. Nothing else is set:
    ``kv_transfer_timeout_ms`` and ``max_tokens_in_buffer`` keep the
    configuration's values or their defaults.
    """
    transceiver = dict(config.get("cache_transceiver_config") or {})
    backend = transceiver.get("backend")
    runtime = transceiver.get("transceiver_runtime", DEFAULT_TRANSCEIVER_RUNTIME)
    if backend in CPP_TRANSCEIVER_BACKENDS:
        transceiver.setdefault("transceiver_runtime", "CPP")
        kv_cache = dict(config.get("kv_cache_config") or {})
        kv_cache.setdefault("use_kv_cache_manager_v2", False)
        config["kv_cache_config"] = kv_cache
    elif runtime in ("PYTHON", "auto") or (
        runtime is None and transceiver.get("enable_pipelined_transfer")
    ):
        if backend in (None, "DEFAULT"):
            transceiver["backend"] = env_backend(env) or "NIXL"
    elif backend is None:
        transceiver["backend"] = "DEFAULT"
    config["cache_transceiver_config"] = transceiver


def transceiver_runtime(config: Mapping[str, Any]) -> str:
    """The transceiver runtime an engine configuration selects: ``PYTHON``, ``CPP`` or ``AUTO``.

    ``AUTO`` is ``auto``, which the configuration alone does not resolve:
    the engine takes the model's preferred runtime when the model declares
    one, and otherwise the Python runtime unless the configuration rules it
    out. A null runtime is the C++ one, except with
    ``enable_pipelined_transfer`` and backend NIXL, where the engine takes the
    Python one. DEFAULT counts as not NIXL here: :func:`disagg_engine_yaml`
    resolves it for this case first.
    """
    transceiver = config.get("cache_transceiver_config") or {}
    runtime = transceiver.get("transceiver_runtime", DEFAULT_TRANSCEIVER_RUNTIME)
    if runtime == "auto":
        return "AUTO"
    if runtime == "PYTHON":
        return "PYTHON"
    if (
        runtime is None
        and transceiver.get("enable_pipelined_transfer")
        and transceiver.get("backend") == "NIXL"
    ):
        return "PYTHON"
    return "CPP"


def attention_dp_ranks(config: Mapping[str, Any], args: Sequence[str]) -> int:
    """KV event publishers of one engine: one per attention-DP rank."""
    enabled = setting(config, args, ("enable_attention_dp",), "--enable_attention_dp")
    if enabled is None or not _as_bool(enabled):
        return 1
    value = setting(config, args, ("tensor_parallel_size",), "--tp_size")
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
        value = setting(config, args, (key,), option)
        if value is not None and int(value) > 1:
            raise ChainUnrunnableError(
                f"not runnable in the OpenEngine chain (DV-23): KV cache events need {key} 1, "
                f"got {value}"
            )
    runtime = transceiver_runtime(config)
    if runtime == "CPP":
        raise ChainUnrunnableError(
            "not runnable in the OpenEngine chain (DV-23): KV cache events need KV cache "
            "manager V2, which the C++ transceiver runtime cannot use"
        )
    transceiver = config.get("cache_transceiver_config") or {}
    backend = transceiver.get("backend")
    if runtime == "PYTHON" and backend != "NIXL":
        raise ChainUnrunnableError(
            "not runnable in the OpenEngine chain (DV-23): KV cache events need KV cache "
            f"manager V2 and so the Python transceiver, which takes only NIXL, not {backend}"
        )
    if runtime == "AUTO" and (
        backend != "NIXL"
        or (
            "kv_transfer_timeout_ms" in transceiver
            and transceiver["kv_transfer_timeout_ms"] is None
        )
    ):
        raise ChainUnrunnableError(
            "not runnable in the OpenEngine chain (DV-23): KV cache events need KV cache "
            "manager V2 and so the Python transceiver, which takes only NIXL and a finite "
            "kv_transfer_timeout_ms; with transceiver_runtime auto the engine falls back to "
            "the C++ transceiver here, or fails to build the one the model prefers"
        )
    kv_cache = dict(config.get("kv_cache_config") or {})
    if kv_cache.get("use_kv_cache_manager_v2") is False:
        raise ChainUnrunnableError(
            "not runnable in the OpenEngine chain (DV-23): KV cache events need KV cache "
            "manager V2, and kv_cache_config.use_kv_cache_manager_v2 is false"
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


def check_disagg_sets(args: Sequence[str]) -> None:
    """Refuse the ``--set`` overrides a disaggregated worker cannot take (the harness writes them).

    An override of a path above one of them is refused too: it replaces it.
    """
    for path, _ in set_overrides(args):
        for refused in _DISAGG_RULE_SET_PATHS:
            if path[: len(refused)] == refused or refused[: len(path)] == path:
                raise UnknownServeOptionError(
                    f"--set {'.'.join(path)}: the harness writes this setting into a "
                    "disaggregated worker's engine configuration; give it in the configuration "
                    "file instead"
                )


def disagg_engine_yaml(
    path: str | None,
    out_dir: str,
    *,
    args: Sequence[str],
    env: Mapping[str, str],
    kv_aware_router: bool | None,
    ports: Callable[[int], int],
) -> str:
    """Write the engine's configuration for a disaggregated worker and return its path.

    ``path`` is the worker's configuration file (``None``: it has none),
    ``args`` its command line and ``env`` the engine's environment. The copy
    drops the front-end-only keys, spells out the KV-transfer settings and,
    when the worker serves KV cache events, sets up the streaming publisher
    (:func:`kv_events_wanted`).
    """
    check_disagg_sets(args)
    config = load_config(path)
    for key in FRONTEND_ONLY_YAML_KEYS + DISAGG_FRONTEND_ONLY_YAML_KEYS:
        config.pop(key, None)
    _transceiver_template(config, env)
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
    strict: bool = True,
) -> DisaggWorkerCommands:
    """Take one aggregated ``trtllm-serve`` command line apart for two disaggregated workers.

    Each worker gets the options of ``cmd`` but ``--host``, ``--port`` and the
    configuration file options: the model and engine options start an engine
    per worker, the front-end options a front end per worker. Each also gets
    ``--host host``, its port from ``ports``, ``--server_role``, and a
    configuration file: the one ``cmd`` reads, plus a KV transceiver (unless it
    has one), the disaggregated server's signing key and, when the two workers
    share a GPU, half of the KV cache memory share each. That share is resolved
    as ``trtllm-serve`` resolves it (:func:`setting`), so the command line's own
    ``--kv_cache_free_gpu_memory_fraction`` and ``--set`` of it are dropped too;
    a ``--set`` of ``kv_cache_config`` as a whole is refused then
    (:class:`UnknownServeOptionError`). ``strict`` as in :func:`_iter_options`
    (``False`` when no chain splits the workers).
    """
    command = parse_serve_command(cmd, strict)
    if is_disagg_worker(command.args, strict):
        raise ValueError(f"already a disaggregated worker, not moved: {list(cmd)!r}")
    config = load_config(last_config_file(command.args, strict))
    config.setdefault("cache_transceiver_config", {"backend": "DEFAULT"})
    config["internal_request_auth_key"] = auth_key
    gpus = parallel_size(command.args, strict)
    share_gpu = gpus == 1
    fraction_path = ("kv_cache_config", "free_gpu_memory_fraction")
    fraction_option = "--kv_cache_free_gpu_memory_fraction"
    if share_gpu:
        if any(path == fraction_path[:1] for path, _ in set_overrides(command.args, strict)):
            raise UnknownServeOptionError(
                "--set kv_cache_config: the harness halves the KV cache memory share of two "
                "workers on one GPU; set kv_cache_config's keys one by one instead"
            )
        fraction = setting(config, command.args, fraction_path, fraction_option, strict)
        if fraction is None:
            fraction = DEFAULT_KV_CACHE_FREE_GPU_MEMORY_FRACTION
        kv_cache = dict(config.get("kv_cache_config") or {})
        kv_cache["free_gpu_memory_fraction"] = float(fraction) / 2
        config["kv_cache_config"] = kv_cache
    kept: list[str] = []
    for name, tokens, value in _iter_options(command.args, strict):
        if name in ("--host", "--port") or name in CONFIG_OPTIONS:
            continue
        if share_gpu and (
            name == fraction_option or (name == "--set" and _set_path(value) == fraction_path)
        ):
            continue
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
