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


def test_custom_tokenizer_goes_to_both_processes():
    result = sp.split_args(
        ["--custom_tokenizer", "deepseek_v41", "--tp_size", "8"], rewrite_yaml=lambda p: p
    )
    assert result.frontend == ("--custom_tokenizer", "deepseek_v41")
    assert result.engine == ("--custom_tokenizer", "deepseek_v41", "--tp_size", "8")


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


def test_option_value_is_the_last_value_given():
    args = ["--openengine-metrics-port", "1", "--tp_size=2", "--openengine-metrics-port=2"]
    assert sp.option_value(args, "--openengine-metrics-port") == "2"
    assert sp.option_value(args, "--tp_size") == "2"
    assert sp.option_value(args, "--pp_size") is None


# --- Disaggregated workers ---------------------------------------------------------------


def test_role_and_service_discovery_options_go_to_the_front_end_only():
    result = sp.split_args(
        ["--server_role", "context", "--disagg_cluster_uri=http://h:1", "--tp_size", "2"],
        rewrite_yaml=lambda p: p,
    )
    assert result.frontend == ("--server_role", "context", "--disagg_cluster_uri=http://h:1")
    assert result.engine == ("--tp_size", "2")


def test_metadata_server_option_is_not_runnable_in_the_chain():
    with pytest.raises(sp.ChainUnrunnableError, match="metadata_server_config_file"):
        sp.split_args(["--metadata_server_config_file", "m.yaml"], rewrite_yaml=lambda p: p)


def test_unknown_options_of_a_disaggregated_worker_are_still_errors():
    with pytest.raises(sp.UnknownServeOptionError, match="split table"):
        sp.split_args(["--server_role", "context", "--log_level", "info"], rewrite_yaml=lambda p: p)


@pytest.mark.parametrize(
    "args, config, disagg",
    [
        ([], {}, False),
        ([], {"max_batch_size": 8}, False),
        (["--server_role", "generation"], {}, True),
        (["--disagg_cluster_uri", "http://h:1"], {}, True),
        ([], {"disagg_cluster": {"cluster_uri": "http://h:1"}}, True),
        ([], {"cache_transceiver_config": {"backend": "DEFAULT"}}, True),
    ],
)
def test_is_disagg_worker(tmp_path, args, config, disagg):
    path = _config(tmp_path, "c.yaml", config)
    assert sp.is_disagg_worker(args + ["--config", path]) is disagg


def test_disagg_frontend_env_keeps_only_what_the_front_end_reads():
    env = {
        "PATH": "/bin",
        "CUDA_VISIBLE_DEVICES": "3",
        "TLLM_LOG_LEVEL": "info",
        "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "2",
        "TRTLLM_KVCACHE_TIME_OUTPUT_PATH": "/k",
        "TRTLLM_DISAGG_BENCHMARK_GEN_ONLY": "1",
        "TRTLLM_DISAGG_NO_RETRY": "1",
        "TRTLLM_WORKER_DISABLE_GC": "1",
        "TRTLLM_SERVER_DISABLE_GC": "1",
        "UCX_TLS": "^ib",
    }
    assert sp.disagg_frontend_env(env) == {
        "PATH": "/bin",
        "TRTLLM_SERVER_DISABLE_GC": "1",
        "UCX_TLS": "^ib",
    }
    # An aggregated server's front end keeps them, as before.
    assert sp.frontend_env(env)["CUDA_VISIBLE_DEVICES"] == "3"


def _no_ports(count):
    raise AssertionError("no KV event ports expected")


def _engine_config(tmp_path, config, *, args=(), kv_aware_router=None, ports=_no_ports):
    path = _config(tmp_path, "worker.yaml", config) if config is not None else None
    engine_path = sp.disagg_engine_yaml(
        path,
        str(tmp_path / "out"),
        args=list(args),
        kv_aware_router=kv_aware_router,
        ports=ports,
    )
    if path is not None:
        assert yaml.safe_load(open(path)) == config, "the worker's own file must not change"
    return yaml.safe_load(open(engine_path))


def test_disagg_engine_yaml_splits_the_worker_configuration(tmp_path):
    shared = {
        "internal_request_auth_key": "k",
        "disable_overlap_scheduler": True,
        "max_batch_size": 8,
        "kv_cache_config": {
            "tokens_per_block": 32,
            "kv_cache_event_hash_algo": "sha256",
            "free_gpu_memory_fraction": 0.2,
        },
        "cache_transceiver_config": {"backend": "NIXL"},
    }
    config = {
        **shared,
        "disagg_cluster": {"cluster_uri": "http://h:1"},
        "num_postprocess_workers": 2,
    }
    # Service discovery and front-end keys stay with the front end; the signing key,
    # the KV-transfer settings and the keys both processes check go to the engine
    # unchanged, from the same file the front end reads.
    assert _engine_config(tmp_path, config) == shared


def test_disagg_engine_yaml_keeps_transceiver_prerequisites(tmp_path):
    """NT-173: the engine YAML spells out what the KV cache transceiver needs."""

    def engine(config):
        return _engine_config(tmp_path, config, kv_aware_router=False)

    # EP-01, EP-02: the backend is always written; the default runtime gets NIXL.
    for given in (None, {}, {"cache_transceiver_config": {}}):
        assert engine(given)["cache_transceiver_config"] == {"backend": "NIXL"}
    default = engine({"cache_transceiver_config": {"backend": "DEFAULT"}})
    assert default["cache_transceiver_config"] == {"backend": "NIXL"}
    # Two workers of one test get the same backend.
    ctx = engine({"cache_transceiver_config": {"backend": "DEFAULT"}, "max_batch_size": 4})
    gen = engine({"cache_transceiver_config": {"backend": "DEFAULT"}, "max_batch_size": 8})
    assert ctx["cache_transceiver_config"] == gen["cache_transceiver_config"]
    # EP-03: the default runtime does not get KV cache manager V1.
    assert "kv_cache_config" not in default
    # EP-04: a C++-only backend gets the C++ runtime and KV cache manager V1.
    for backend in ("UCX", "MPI", "MOONCAKE"):
        cpp = engine({"cache_transceiver_config": {"backend": backend}})
        assert cpp["cache_transceiver_config"] == {
            "backend": backend,
            "transceiver_runtime": "CPP",
        }
        assert cpp["kv_cache_config"] == {"use_kv_cache_manager_v2": False}
    cpp_default = engine({"cache_transceiver_config": {"transceiver_runtime": "CPP"}})
    assert cpp_default["cache_transceiver_config"]["backend"] == "DEFAULT"
    # EP-05, EP-11: the timeout and the buffer size keep the test's values or their defaults.
    plain = engine({"cache_transceiver_config": {"backend": "NIXL"}})
    assert "kv_transfer_timeout_ms" not in plain["cache_transceiver_config"]
    assert "max_tokens_in_buffer" not in plain["cache_transceiver_config"]
    given = {"backend": "NIXL", "max_tokens_in_buffer": 8192, "kv_transfer_timeout_ms": 5000}
    assert engine({"cache_transceiver_config": given})["cache_transceiver_config"] == given
    # EP-09: a model family that needs V2 keeps the default runtime and NIXL.
    v2 = engine(
        {
            "cache_transceiver_config": {"backend": "DEFAULT"},
            "kv_cache_config": {"use_kv_cache_manager_v2": True},
        }
    )
    assert v2["cache_transceiver_config"] == {"backend": "NIXL"}
    assert v2["kv_cache_config"] == {"use_kv_cache_manager_v2": True}


def _events_config(**kv_cache):
    return {
        "cache_transceiver_config": {"backend": "DEFAULT"},
        "kv_cache_config": {"enable_block_reuse": True, "event_buffer_max_size": 1024, **kv_cache},
    }


def test_kv_events_replace_the_event_buffer_when_the_server_routes_on_them(tmp_path):
    asked = []

    def ports(count):
        asked.append(count)
        return 41000

    for router in (True, None):
        config = _engine_config(tmp_path, _events_config(), kv_aware_router=router, ports=ports)
        assert config["kv_cache_config"] == {
            "enable_block_reuse": True,
            "use_kv_cache_manager_v2": True,
            "kv_events_config": {
                "enable_kv_cache_events": True,
                "publisher": "zmq",
                "endpoint": "tcp://127.0.0.1:41000",
                "replay_endpoint": "tcp://127.0.0.1:41001",
            },
        }
    assert asked == [2, 2]


def test_kv_events_stay_off_without_the_event_buffer_or_a_kv_aware_server(tmp_path):
    config = _events_config()
    assert (
        _engine_config(tmp_path, config, kv_aware_router=False)["kv_cache_config"]
        == (config["kv_cache_config"])
    )
    plain = {"kv_cache_config": {"enable_block_reuse": True}}
    assert (
        _engine_config(tmp_path, plain, kv_aware_router=True)["kv_cache_config"]
        == (plain["kv_cache_config"])
    )


def test_kv_events_publish_one_port_per_attention_dp_rank(tmp_path):
    config = {**_events_config(), "enable_attention_dp": True, "tensor_parallel_size": 4}
    engine = _engine_config(tmp_path, config, ports=lambda count: 42000)
    events = engine["kv_cache_config"]["kv_events_config"]
    assert (events["endpoint"], events["replay_endpoint"]) == (
        "tcp://127.0.0.1:42000",
        "tcp://127.0.0.1:42004",
    )
    engine = _engine_config(
        tmp_path,
        _events_config(),
        args=["--enable_attention_dp", "--tp_size", "2"],
        ports=lambda count: 43000,
    )
    assert engine["kv_cache_config"]["kv_events_config"]["replay_endpoint"] == (
        "tcp://127.0.0.1:43002"
    )


@pytest.mark.parametrize(
    "config, args, match",
    [
        ({"pipeline_parallel_size": 2}, [], "pipeline_parallel_size 1"),
        ({"context_parallel_size": 2}, [], "context_parallel_size 1"),
        ({}, ["--pp_size", "2"], "pipeline_parallel_size 1"),
        ({"cache_transceiver_config": {"backend": "UCX"}}, [], "C\\+\\+ transceiver"),
        (
            {"kv_cache_config": {"event_buffer_max_size": 8, "use_kv_cache_manager_v2": False}},
            [],
            "turns it off",
        ),
    ],
)
def test_kv_events_are_not_runnable_where_the_publisher_cannot_run(tmp_path, config, args, match):
    merged = _events_config()
    merged.update({k: v for k, v in config.items() if k != "kv_cache_config"})
    merged["kv_cache_config"].update(config.get("kv_cache_config", {}))
    with pytest.raises(sp.ChainUnrunnableError, match=f"DV-23.*{match}"):
        _engine_config(tmp_path, merged, args=args, ports=lambda count: 44000)


# --- Moving an aggregated test ----------------------------------------------------------


def test_parallel_size_takes_the_configuration_over_the_command_line(tmp_path):
    path = _config(tmp_path, "c.yaml", {"tensor_parallel_size": 4})
    assert sp.parallel_size(["--tp_size", "2", "--pp_size", "2"]) == 4
    assert sp.parallel_size(["--tp_size", "2", "--config", path]) == 4
    assert sp.parallel_size([]) == 1


def _moved(tmp_path, cmd):
    workers = sp.disagg_worker_commands(
        cmd,
        ports={"context": 7001, "generation": 7002},
        host="localhost",
        auth_key="key",
        out_dir=str(tmp_path / "out"),
    )
    configs = {role: yaml.safe_load(open(path)) for role, path in workers.configs.items()}
    return workers, configs


def test_moved_aggregated_command_line_becomes_two_workers(tmp_path):
    own = _config(tmp_path, "own.yaml", {"max_batch_size": 8, "kv_cache_config": {"x": 1}})
    cmd = [
        "trtllm-serve",
        "/m",
        "--host",
        "localhost",
        "--port",
        "8000",
        "--tool_parser",
        "qwen3",
        "--extra_llm_api_options",
        own,
        "--num_serve_frontends",
        "1",
    ]
    workers, configs = _moved(tmp_path, cmd)
    for role, port in (("context", "7001"), ("generation", "7002")):
        assert workers.commands[role] == (
            "trtllm-serve",
            "/m",
            "--host",
            "localhost",
            "--port",
            port,
            "--server_role",
            role,
            "--tool_parser",
            "qwen3",
            "--num_serve_frontends",
            "1",
            "--extra_llm_api_options",
            workers.configs[role],
        )
        assert configs[role] == {
            "max_batch_size": 8,
            "kv_cache_config": {"x": 1, "free_gpu_memory_fraction": 0.45},
            "cache_transceiver_config": {"backend": "DEFAULT"},
            "internal_request_auth_key": "key",
        }
        assert sp.is_disagg_worker(workers.commands[role][2:])
    assert (workers.share_gpu, workers.gpus_per_worker) == (True, 1)
    assert yaml.safe_load(open(own)) == {"max_batch_size": 8, "kv_cache_config": {"x": 1}}


def test_moved_workers_halve_the_kv_cache_share_only_when_they_share_a_gpu(tmp_path):
    _, configs = _moved(
        tmp_path, ["trtllm-serve", "/m", "--kv_cache_free_gpu_memory_fraction", "0.6"]
    )
    assert configs["context"]["kv_cache_config"] == {"free_gpu_memory_fraction": 0.3}
    own = _config(
        tmp_path,
        "own.yaml",
        {"kv_cache_config": {"free_gpu_memory_fraction": 0.4}},
    )
    _, configs = _moved(tmp_path, ["trtllm-serve", "/m", "--config", own])
    assert configs["generation"]["kv_cache_config"] == {"free_gpu_memory_fraction": 0.2}
    workers, configs = _moved(tmp_path, ["trtllm-serve", "/m", "--tp_size", "2"])
    assert (workers.share_gpu, workers.gpus_per_worker) == (False, 2)
    assert "kv_cache_config" not in configs["context"]
    assert "--tp_size" in workers.commands["generation"]


def test_a_disaggregated_worker_is_not_moved(tmp_path):
    with pytest.raises(ValueError, match="already a disaggregated worker"):
        _moved(tmp_path, ["trtllm-serve", "/m", "--server_role", "context"])


def test_disagg_server_config_points_at_the_two_workers():
    assert sp.disagg_server_config(
        host="localhost",
        port=8000,
        workers={"context": "localhost:1", "generation": "localhost:2"},
        auth_key="k",
    ) == {
        "hostname": "localhost",
        "port": 8000,
        "context_servers": {"num_instances": 1, "urls": ["localhost:1"]},
        "generation_servers": {"num_instances": 1, "urls": ["localhost:2"]},
        "internal_request_auth_key": "k",
    }


def test_worker_gpus():
    assert sp.worker_gpus(["3", "5"], 1, True) == {"context": "3", "generation": "3"}
    assert sp.worker_gpus(["0", "1", "2", "3"], 2, False) == {
        "context": "0,1",
        "generation": "2,3",
    }
    with pytest.raises(sp.ChainUnrunnableError, match="needs 8 GPUs"):
        sp.worker_gpus([str(i) for i in range(7)], 4, False)
