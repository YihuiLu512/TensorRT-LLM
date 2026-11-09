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
    # Each process gets its own address.
    "--host": Option(Side.HARNESS),
    "--port": Option(Side.HARNESS),
}

CONFIG_OPTIONS = ("--config", "--extra_llm_api_options")

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
