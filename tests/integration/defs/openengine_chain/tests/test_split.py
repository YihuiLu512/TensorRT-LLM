# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which process gets which argument."""

import pytest
import yaml
from openengine_chain import split as sp

pytestmark = pytest.mark.cpu_only

_VALUE = {"--max_beam_width": "4", "--num_serve_frontends": "1"}


@pytest.mark.parametrize("name", sorted(sp.OPTIONS))
def test_every_table_option_goes_to_its_side(name):
    option = sp.OPTIONS[name]
    tokens = [name, _VALUE.get(name, "x")] if option.takes_value else [name]
    if name in sp.CONFIG_OPTIONS:
        result = sp.split_args(tokens, rewrite_yaml=lambda path: f"engine:{path}")
        assert result.frontend == (name, "x")
        assert result.engine == (name, "engine:x")
        return
    result = sp.split_args(tokens, rewrite_yaml=lambda path: path)
    expected_frontend = tuple(tokens) if option.side in (sp.Side.FRONTEND, sp.Side.BOTH) else ()
    expected_engine = tuple(tokens) if option.side in (sp.Side.ENGINE, sp.Side.BOTH) else ()
    assert result.frontend == expected_frontend
    assert result.engine == expected_engine


def test_host_and_port_go_to_neither_process():
    result = sp.split_args(["--host", "0.0.0.0", "--port=8123"], rewrite_yaml=lambda p: p)
    assert (result.engine, result.frontend) == ((), ())
    assert (result.host, result.port) == ("0.0.0.0", "8123")


@pytest.mark.parametrize("args", [["--log_level", "info"], ["--revision=abc"], ["--grpc"]])
def test_unknown_option_is_an_error(args):
    with pytest.raises(sp.UnknownServeOptionError, match="split table"):
        sp.split_args(args, rewrite_yaml=lambda p: p)


def test_equals_form_and_flags():
    result = sp.split_args(
        ["--tp_size=2", "--enable_chunked_prefill", "--tool_parser=qwen3"], rewrite_yaml=lambda p: p
    )
    assert result.engine == ("--tp_size=2", "--enable_chunked_prefill")
    assert result.frontend == ("--tool_parser=qwen3",)


def test_flag_with_value_and_missing_value_are_errors():
    with pytest.raises(sp.UnknownServeOptionError):
        sp.split_args(["--trust_remote_code=1"], rewrite_yaml=lambda p: p)
    with pytest.raises(sp.UnknownServeOptionError, match="missing its value"):
        sp.split_args(["--tp_size"], rewrite_yaml=lambda p: p)


def test_beam_width_goes_to_both_so_the_front_end_refuses_beam_servers():
    result = sp.split_args(["--max_beam_width", "4"], rewrite_yaml=lambda p: p)
    assert result.engine == ("--max_beam_width", "4")
    assert result.frontend == ("--max_beam_width", "4")


def _config(tmp_path, name, config):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_iteration_stats_come_from_the_last_config_file_and_the_set_overrides(tmp_path):
    on = _config(tmp_path, "on.yaml", {"enable_iter_perf_stats": True})
    off = _config(tmp_path, "off.yaml", {"enable_iter_perf_stats": False})
    other = _config(tmp_path, "other.yaml", {"return_perf_metrics": True})
    assert not sp.iteration_stats_enabled([])
    assert not sp.iteration_stats_enabled(["--extra_llm_api_options", other])
    assert sp.iteration_stats_enabled(["--extra_llm_api_options", on])
    # The two names are one option of trtllm-serve: only the last file given counts.
    assert not sp.iteration_stats_enabled(["--config", on, "--extra_llm_api_options", other])
    assert not sp.iteration_stats_enabled(["--config", on, "--extra_llm_api_options", off])
    assert sp.iteration_stats_enabled(["--config", other, "--extra_llm_api_options", on])
    # --set wins over the file.
    assert not sp.iteration_stats_enabled(["--set", "enable_iter_perf_stats=false", "--config", on])
    assert sp.iteration_stats_enabled(["--config=" + off, "--set=enable_iter_perf_stats=true"])
    assert not sp.iteration_stats_enabled(["--set", "enable_iter_req_stats=true"])
    # A string is read as pydantic reads it.
    quoted_off = _config(tmp_path, "quoted_off.yaml", {"enable_iter_perf_stats": "false"})
    assert not sp.iteration_stats_enabled(["--config", quoted_off])
    assert not sp.iteration_stats_enabled(["--config", on, "--set", "enable_iter_perf_stats='off'"])


def test_set_of_a_frontend_only_key_stays_with_the_frontend():
    result = sp.split_args(
        ["--set", "num_postprocess_workers=2", "--set", "max_batch_size=8"],
        rewrite_yaml=lambda p: p,
    )
    assert result.frontend == ("--set", "num_postprocess_workers=2", "--set", "max_batch_size=8")
    assert result.engine == ("--set", "max_batch_size=8")


def test_engine_yaml_drops_frontend_only_keys(tmp_path):
    config = {
        "num_postprocess_workers": 2,
        "num_serve_frontends": 4,
        "postprocess_tokenizer_dir": "/t",
        "max_batch_size": 8,
        "kv_cache_config": {"enable_block_reuse": True},
    }
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(config))
    engine_path = sp.engine_yaml(str(path), str(tmp_path / "out"))
    assert engine_path != str(path)
    assert yaml.safe_load(open(engine_path)) == {
        "max_batch_size": 8,
        "kv_cache_config": {"enable_block_reuse": True},
    }
    assert yaml.safe_load(path.read_text()) == config


def test_engine_yaml_reuses_a_file_without_frontend_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"max_batch_size": 8}))
    assert sp.engine_yaml(str(path), str(tmp_path / "out")) == str(path)


@pytest.mark.parametrize(
    "cmd, prefix",
    [
        (["trtllm-serve", "/m", "--tp_size", "1"], ("trtllm-serve",)),
        (["/usr/bin/trtllm-serve", "serve", "/m"], ("/usr/bin/trtllm-serve", "serve")),
        (
            ["python3", "-m", "tensorrt_llm.commands.serve", "--host", "h", "/m"],
            ("python3", "-m", "tensorrt_llm.commands.serve"),
        ),
    ],
)
def test_parse_serve_command(cmd, prefix):
    command = sp.parse_serve_command(cmd)
    assert command.prefix == prefix
    assert command.model == "/m"
    assert [a for a in cmd[len(prefix) :] if a != "/m"] == list(command.args)


@pytest.mark.parametrize(
    "cmd", [["python3", "x.py", "/m"], ["trtllm-serve"], ["trtllm-serve", "/a", "/b"]]
)
def test_parse_rejects_other_commands(cmd):
    with pytest.raises(ValueError):
        sp.parse_serve_command(cmd)


def test_frontend_env_drops_engine_only_variables():
    env = {"TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "2", "PATH": "/bin"}
    assert sp.frontend_env(env) == {"PATH": "/bin"}


def test_table_matches_the_trtllm_serve_cli():
    """Every table option exists on ``trtllm-serve`` and takes a value exactly when the CLI says so."""
    serve = pytest.importorskip("tensorrt_llm.commands.serve")
    params = {}
    for param in serve.serve.params:
        for opt in param.opts:
            params[opt] = param
    for name, option in sp.OPTIONS.items():
        assert name in params, f"{name} is not a trtllm-serve option"
        assert params[name].is_flag == (not option.takes_value), name
