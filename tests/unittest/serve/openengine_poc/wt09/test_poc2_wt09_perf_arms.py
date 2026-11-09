# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The perf comparison arms (tests/integration/defs/perf/openengine_arms.py) on CPU."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

_TESTS = Path(__file__).resolve().parents[4]
_DEFS = _TESTS / "integration" / "defs"


@pytest.fixture(scope="module")
def harness_path(tmp_path_factory):
    """Put the harness package on ``sys.path`` under its top-level name, as runner.py does."""
    directory = tmp_path_factory.mktemp("pythonpath")
    (directory / "openengine_chain").symlink_to(_DEFS / "openengine_chain")
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path.remove(str(directory))


@pytest.fixture(scope="module")
def arms(harness_path):
    spec = importlib.util.spec_from_file_location(
        "openengine_arms_under_test", _DEFS / "perf" / "openengine_arms.py"
    )
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up while the class bodies run.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.fixture(autouse=True)
def _no_container_ports(monkeypatch):
    """free_port avoids CI's container port range; the tests that want one set it."""
    monkeypatch.delenv("CONTAINER_PORT_START", raising=False)
    monkeypatch.delenv("CONTAINER_PORT_NUM", raising=False)


@pytest.fixture(autouse=True)
def _accept_frontend_options(request, arms, monkeypatch):
    """The OpenEngine front end's option check needs TensorRT-LLM; tests that want it ask for ``facade``."""
    if "facade" not in request.fixturenames:
        monkeypatch.setattr(arms, "_frontend_check", lambda: lambda args: None)


@pytest.fixture
def facade():
    return pytest.importorskip("tensorrt_llm.serve.openengine_poc.facade")


# name -> (case, chain, glm HTTP, front ends, postproc workers).
# None keeps the test's own value.
EXPECTED_ARMS = {
    "A-def": ("case-02", "none", False, 8, None),
    "A-1f": ("case-02", "none", False, 1, None),
    "A-1f0": ("case-02", "none", False, 1, 0),
    "B-s1": ("case-02", "trtllm", False, 1, 0),
    "B-full": ("case-02", "trtllm", False, 8, None),
    "C-http": ("case-01", "none", True, None, None),
    "C-s1": ("case-01", "glm", False, 1, 0),
    "C-full": ("case-01", "glm", False, 8, None),
}

BENCHMARK_OUTPUT = """\
============ Serving Benchmark Result ============
Total requests:                          16
Successful requests:                     16
Failed requests:                         0
Benchmark duration (s):                  3.21
Total input tokens:                      2048
Total generated tokens:                  512
Request throughput (req/s):              4.98
Output token throughput (tok/s):         159.50
Total Token throughput (tok/s):          797.51
---------------Time to First Token----------------
Mean TTFT (ms):                          21.50
Median TTFT (ms):                        20.10
P99 TTFT (ms):                           40.25
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          5.10
Median TPOT (ms):                        5.00
P99 TPOT (ms):                           6.75
---------------Inter-token Latency----------------
Mean ITL (ms):                           5.20
Median ITL (ms):                         5.05
P99 ITL (ms):                            7.00
----------------End-to-end Latency----------------
Mean E2EL (ms):                          180.00
Median E2EL (ms):                        175.50
P99 E2EL (ms):                           220.10
==================================================
"""

CLIENT_TOKEN_IDS = [
    "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving", "--model", "m",
    "--max-concurrency", "4", "--dataset-name", "random", "--random-ids",
    "--tokenize-on-client", "--random-input-len", "128",
]  # fmt: skip
CLIENT_TEXT = [
    "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving", "--model", "m",
    "--max-concurrency", "64", "--tokenize-on-client", "--dataset-name", "trtllm_custom",
    "--dataset-path", "/data/x.json",
]  # fmt: skip

CONFIG = {
    "max_batch_size": 64,
    "kv_cache_config": {"enable_block_reuse": False, "free_gpu_memory_fraction": 0.8},
    "stream_interval": 20,
    "num_postprocess_workers": 4,
    "disable_mm_encoder": True,
    "speculative_config": {"decoding_type": "MTP", "max_draft_len": 3},
}


def _write_config(tmp_path, config=CONFIG, name="extra.yml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config))
    return str(path)


# ----------------------------------------------------------------------------- arms table


def test_arm_table_matches_spec(arms):
    assert list(arms.ARMS) == list(EXPECTED_ARMS)
    for name, (case, chain, glm_http, frontends, workers) in EXPECTED_ARMS.items():
        arm = arms.ARMS[name]
        assert (arm.case, arm.chain, arm.glm_http, arm.frontends, arm.postprocess_workers) == (
            case,
            chain,
            glm_http,
            frontends,
            workers,
        ), name
        assert set(arm.settings()) <= set(arms.ARM_SETTINGS)


def test_full_arms_match_default_arm(arms):
    """B-full and C-full use A-def's front-end and post-processing settings."""
    a_def = arms.ARMS["A-def"].settings()
    assert arms.ARMS["B-full"].settings() == a_def
    assert arms.ARMS["C-full"].settings() == a_def


def test_names_match_harness_conventions(arms):
    from openengine_chain import conventions as cv

    assert arms.ENV_CHAIN == cv.ENV_CHAIN
    assert arms.ENV_GLM_SERVE_CMD == cv.ENV_GLM_SERVE_CMD
    assert arms.ENV_RESULTS_DIR == cv.ENV_RESULTS_DIR
    assert (arms.CHAIN_NONE, arms.CHAIN_TRTLLM, arms.CHAIN_GLM) == (
        cv.CHAIN_NONE,
        cv.CHAIN_TRTLLM,
        cv.CHAIN_GLM,
    )
    from openengine_chain.split import CONFIG_OPTIONS, ENGINE_ONLY_ENV

    assert arms.CONFIG_OPTIONS == CONFIG_OPTIONS
    assert arms.ENV_SPEC_DECODE_ACCEPTED in ENGINE_ONLY_ENV


@pytest.mark.parametrize("name", list(EXPECTED_ARMS))
def test_apply_arm_changes_only_arm_settings(arms, name):
    arm = arms.ARMS[name]
    out = arms.apply_arm(CONFIG, arm)
    changed = {key for key in set(out) | set(CONFIG) if out.get(key) != CONFIG.get(key)}
    assert changed <= set(arms.ARM_SETTINGS)
    for key, value in arm.settings().items():
        assert out[key] == value
    assert CONFIG["num_postprocess_workers"] == 4, "the input must not change"


def test_explicit_port(arms):
    explicit = {name for name, arm in arms.ARMS.items() if arm.explicit_port}
    assert explicit == {"A-def", "B-s1", "B-full", "C-http", "C-s1", "C-full"}


# ----------------------------------------------------------------------------- selection


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, None),
        ({"TRTLLM_PERF_ARM": "A-def"}, "A-def"),
        ({"TRTLLM_PERF_ARM": "C-http"}, "C-http"),
        ({"TRTLLM_PERF_ARM": "B-s1", "TRTLLM_TEST_OPENENGINE_CHAIN": "trtllm"}, "B-s1"),
        ({"TRTLLM_PERF_ARM": "C-full", "TRTLLM_TEST_OPENENGINE_CHAIN": "glm"}, "C-full"),
    ],
)
def test_selected_arm(arms, env, expected):
    arm = arms.selected_arm(env)
    assert (arm.name if arm else None) == expected


@pytest.mark.parametrize(
    "env, message",
    [
        ({"TRTLLM_PERF_ARM": "X"}, "expected one of"),
        ({"TRTLLM_TEST_OPENENGINE_CHAIN": "trtllm"}, "choose the comparison arm"),
        ({"TRTLLM_PERF_ARM": "B-s1"}, "set to 'trtllm'"),
        ({"TRTLLM_PERF_ARM": "B-s1", "TRTLLM_TEST_OPENENGINE_CHAIN": "glm"}, "set to 'trtllm'"),
        ({"TRTLLM_PERF_ARM": "A-1f", "TRTLLM_TEST_OPENENGINE_CHAIN": "trtllm"}, "unset"),
        ({"TRTLLM_PERF_ARM": "C-http", "TRTLLM_TEST_OPENENGINE_CHAIN": "glm"}, "unset"),
    ],
)
def test_selected_arm_refuses(arms, env, message):
    with pytest.raises(arms.ArmError, match=message):
        arms.selected_arm(env)


@pytest.mark.parametrize("run", ["x", "0", "-1"])
def test_selected_arm_checks_the_run_number(arms, run):
    with pytest.raises(arms.ArmError, match="TRTLLM_PERF_RUN"):
        arms.selected_arm({"TRTLLM_PERF_ARM": "A-def", "TRTLLM_PERF_RUN": run})


def test_selected_arm_needs_the_harness(arms, monkeypatch):
    monkeypatch.setattr(arms, "HARNESS_PACKAGE", "openengine_chain_missing_for_test")
    with pytest.raises(arms.ArmError, match="runner.py"):
        arms.selected_arm({"TRTLLM_PERF_ARM": "A-def"})
    assert arms.selected_arm({}) is None


def test_free_port_is_not_held(arms):
    """The port is free again for a server that binds without SO_REUSEPORT."""
    port = arms.free_port("127.0.0.1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


# (ephemeral floor, the ports expected; None: a port the kernel picks)
@pytest.mark.parametrize(
    "floor, ports",
    [
        (32768, range(10000, 32768 - 4096)),
        # The static window reaches down to 10000, leaving no room below it.
        (14096, range(10000, 14096)),
        (5000, None),
        (0, None),
    ],
)
def test_free_port_avoids_the_ephemeral_range_and_the_static_window(
    arms, monkeypatch, floor, ports
):
    """Below the static window under the ephemeral range, else below that range, else any port."""
    monkeypatch.setattr(arms, "_ephemeral_floor", lambda: floor)
    port = arms.free_port("127.0.0.1")
    if ports is not None:
        assert port in ports
    else:
        assert not arms._PORT_FLOOR <= port < floor


def test_free_port_takes_the_container_port_range_rather_than_the_ephemeral_range(
    arms, monkeypatch
):
    """With the container range covering every port below the ephemeral range."""
    bounds = set()
    pick = arms.random.SystemRandom()

    def randrange(start, stop):
        bounds.add((start, stop))
        return pick.randrange(start, stop)

    monkeypatch.setattr(
        arms.random, "SystemRandom", lambda: types.SimpleNamespace(randrange=randrange)
    )
    monkeypatch.setattr(arms, "_ephemeral_floor", lambda: 32768)
    monkeypatch.setenv("CONTAINER_PORT_START", str(arms._PORT_FLOOR))
    monkeypatch.setenv("CONTAINER_PORT_NUM", str(32768 - arms._PORT_FLOOR))
    assert arms._PORT_FLOOR <= arms.free_port("127.0.0.1") < 32768
    assert bounds == {(arms._PORT_FLOOR, 32768)}


def test_free_port_draws_below_the_static_window(arms, monkeypatch):
    """The bounds of the draw, whatever port it lands on."""
    bounds = []

    def randrange(start, stop):
        bounds.append((start, stop))
        return start

    monkeypatch.setattr(
        arms.random, "SystemRandom", lambda: types.SimpleNamespace(randrange=randrange)
    )
    monkeypatch.setattr(arms, "_ephemeral_floor", lambda: 32768)
    arms.free_port("127.0.0.1")
    assert bounds[0] == (arms._PORT_FLOOR, 32768 - arms._STATIC_PORT_RANGE_SIZE)


def test_free_port_skips_the_container_port_range(arms, monkeypatch):
    """Ports get_free_port_in_ci hands out from CONTAINER_PORT_START are not drawn.

    The range covers the whole draw but one free port, which a draw that only
    skipped it would hardly ever reach in its tries.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    monkeypatch.setattr(arms, "_ephemeral_floor", lambda: free + 1 + arms._STATIC_PORT_RANGE_SIZE)
    monkeypatch.setenv("CONTAINER_PORT_START", str(arms._PORT_FLOOR))
    monkeypatch.setenv("CONTAINER_PORT_NUM", str(free - arms._PORT_FLOOR))
    assert arms.free_port("127.0.0.1") == free


# (container range, values the draw returns, ports yielded, upper bound of the draw)
@pytest.mark.parametrize(
    "handed_out, values, ports, stop",
    [
        (range(0), [10000, 19999], [10000, 19999], 20000),
        (range(12000, 13000), [10000, 11999, 12000, 18999], [10000, 11999, 13000, 19999], 19000),
        (range(5000, 11000), [10000, 18999], [11000, 19999], 19000),
        (range(19000, 30000), [10000, 18999], [10000, 18999], 19000),
        (range(10000, 20000), [], [], None),
    ],
)
def test_candidate_ports_are_drawn_around_the_container_range(
    arms, monkeypatch, handed_out, values, ports, stop
):
    """Below 20000, the draw covers the window with the range cut out, mapped back around it."""
    bounds = set()

    def randrange(start, end):
        bounds.add((start, end))
        return values.pop(0)

    monkeypatch.setattr(
        arms.random, "SystemRandom", lambda: types.SimpleNamespace(randrange=randrange)
    )
    monkeypatch.setattr(arms, "_PORT_TRIES", len(ports))
    assert list(arms._candidate_ports(20000, handed_out)) == ports
    assert bounds == ({(arms._PORT_FLOOR, stop)} if stop else set())


@pytest.mark.parametrize(
    "start, count, ports",
    [(None, None, range(0)), ("20000", "8", range(20000, 20008)), ("x", "8", range(0))],
)
def test_container_ports(arms, monkeypatch, start, count, ports):
    for name, value in (("CONTAINER_PORT_START", start), ("CONTAINER_PORT_NUM", count)):
        if value is not None:
            monkeypatch.setenv(name, value)
    assert arms._container_ports() == ports


def test_static_window_matches_common(arms):
    """The window size is the one tests/integration/defs/common.py hands ports from."""
    import re

    text = (_DEFS / "common.py").read_text()
    sizes = re.findall(r"^STATIC_PORT_RANGE_SIZE = (\d+)$", text, re.MULTILINE)
    assert sizes == [str(arms._STATIC_PORT_RANGE_SIZE)]
    assert "high - STATIC_PORT_RANGE_SIZE + 1" in text, "common.py's window changed shape"


@pytest.mark.parametrize(
    "content, floor",
    [
        ("32768\t60999\n", 32768),
        ("1024 65535", 1024),
        ("60999 32768", 0),  # reversed
        ("0 60999", 0),
        ("32768 70000", 0),
        ("32768", 0),
        ("x y", 0),
    ],
)
def test_ephemeral_floor_is_checked(arms, tmp_path, monkeypatch, content, floor):
    path = tmp_path / "ip_local_port_range"
    path.write_text(content)
    monkeypatch.setattr(arms, "_EPHEMERAL_RANGE_FILE", str(path))
    assert arms._ephemeral_floor() == floor


def test_ephemeral_floor_without_the_file(arms, tmp_path, monkeypatch):
    monkeypatch.setattr(arms, "_EPHEMERAL_RANGE_FILE", str(tmp_path / "missing"))
    assert arms._ephemeral_floor() == 0


def test_free_port_uses_the_family_of_the_host(arms, monkeypatch):
    real = socket.getaddrinfo

    def only(family):
        def fake(host, port, *args, **kwargs):
            return [info for info in real(host, port, *args, **kwargs) if info[0] == family]

        return fake

    monkeypatch.setattr(arms.socket, "getaddrinfo", only(socket.AF_INET))
    assert arms._family("localhost") == socket.AF_INET
    if not socket.has_ipv6:
        pytest.skip("no IPv6")
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
    except OSError:
        pytest.skip("no IPv6 loopback")
    monkeypatch.setattr(arms.socket, "getaddrinfo", only(socket.AF_INET6))
    assert arms._family("::1") == socket.AF_INET6
    port = arms.free_port("::1")
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        s.bind(("::1", port))


def test_free_port_skips_ports_in_use(arms, monkeypatch):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        taken = held.getsockname()[1]
        held.listen()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            free = probe.getsockname()[1]
        ports = iter([taken, taken, free])
        pick = types.SimpleNamespace(randrange=lambda lo, hi: next(ports))
        monkeypatch.setattr(arms.random, "SystemRandom", lambda: pick)
        monkeypatch.setattr(arms, "_PORT_FLOOR", 1)
        monkeypatch.setattr(arms, "_PORT_TRIES", 3)
        monkeypatch.setattr(arms, "_ephemeral_floor", lambda: 65535)
        assert arms.free_port("127.0.0.1") == free


def _perf_sanity_function(name, namespace, owner=None):
    """A function of test_perf_sanity.py (a method of class ``owner``), compiled from its source alone.

    Importing test_perf_sanity needs TensorRT-LLM and the integration conftest;
    ``namespace`` stands in for the module globals the function uses.
    """
    import ast

    path = _DEFS / "perf" / "test_perf_sanity.py"
    body = ast.parse(path.read_text()).body
    if owner is not None:
        (cls,) = [n for n in body if isinstance(n, ast.ClassDef) and n.name == owner]
        body = cls.body
    (node,) = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _reserve_multi_frontend_port():
    """test_perf_sanity.reserve_multi_frontend_port and the list of sockets it holds."""
    held = []
    namespace = {"socket": socket, "print_info": lambda *a, **k: None}
    namespace["_RESERVED_PORT_SOCKETS"] = held
    return _perf_sanity_function("reserve_multi_frontend_port", namespace), held


def _bind(port, *options, listen=False):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        for option in options:
            s.setsockopt(socket.SOL_SOCKET, option, 1)
        s.bind(("127.0.0.1", port))
        if listen:
            s.listen()
    except OSError:
        s.close()
        raise
    return s


def test_multi_frontend_reservation_lets_the_launcher_probe():
    """A reserved port lets trtllm-serve start its front ends, and refuses a plain bind.

    trtllm-serve's multi-front-end launcher first binds the port with
    SO_REUSEADDR only, then each front end binds it with SO_REUSEADDR and
    SO_REUSEPORT and listens.
    """
    reserve, held = _reserve_multi_frontend_port()
    port = reserve("127.0.0.1")
    opened = []
    try:
        opened.append(_bind(port, socket.SO_REUSEADDR))  # the launcher's probe
        opened.pop().close()
        for _ in range(2):
            opened.append(_bind(port, socket.SO_REUSEADDR, socket.SO_REUSEPORT, listen=True))
        with pytest.raises(OSError):
            opened.append(_bind(port))  # anything else on the node
    finally:
        for s in opened + held:
            s.close()


def test_a_reuseport_only_reservation_fails_the_launcher_probe():
    """Why the reservation also sets SO_REUSEADDR."""
    with _bind(0, socket.SO_REUSEPORT) as reservation:
        port = reservation.getsockname()[1]
        with pytest.raises(OSError):
            _bind(port, socket.SO_REUSEADDR).close()


# ----------------------------------------------------------------------------- server command


def _serve_cmd(config_path, *extra):
    return ["trtllm-serve", "/models/m", "--backend", "pytorch", "--config", config_path, *extra]


@pytest.mark.parametrize("name", [n for n in EXPECTED_ARMS if n != "C-http"])
def test_prepare_server_rewrites_only_the_config(arms, tmp_path, name):
    arm = arms.ARMS[name]
    source = _write_config(tmp_path)
    cmd = _serve_cmd(source, "--reasoning_parser", "deepseek-r1", "--host", "h", "--port", "1")
    plan = arms.prepare_server(cmd, arm, str(tmp_path / "out"))
    index = cmd.index("--config") + 1
    assert list(plan.cmd[:index]) + list(plan.cmd[index + 1 :]) == cmd[:index] + cmd[index + 1 :]
    written = yaml.safe_load(Path(plan.cmd[index]).read_text())
    assert written == arms.apply_arm(CONFIG, arm) == plan.config
    assert Path(plan.cmd[index]).parent == tmp_path / "out"
    assert yaml.safe_load(Path(source).read_text()) == CONFIG, "the test's YAML must not change"


def test_prepare_server_equals_form_and_no_config(arms, tmp_path):
    source = _write_config(tmp_path)
    plan = arms.prepare_server(
        ["trtllm-serve", "m", f"--extra_llm_api_options={source}"],
        arms.ARMS["A-1f0"],
        str(tmp_path),
    )
    assert plan.cmd[2].startswith("--extra_llm_api_options=")
    assert (
        yaml.safe_load(Path(plan.cmd[2].split("=", 1)[1]).read_text())["num_postprocess_workers"]
        == 0
    )

    plan = arms.prepare_server(["trtllm-serve", "m"], arms.ARMS["A-def"], str(tmp_path))
    assert plan.cmd[:2] == ("trtllm-serve", "m") and plan.cmd[2] == "--config"
    assert yaml.safe_load(Path(plan.cmd[3]).read_text()) == {"num_serve_frontends": 8}


def test_prepare_server_glm_http(arms, tmp_path):
    source = _write_config(tmp_path)
    cmd = _serve_cmd(source, "--tool_parser", "glm47", "--host", "localhost", "--port", "9")
    env = {"TRTLLM_TEST_GLM_SERVE_CMD": "/venv/bin/glm5-serve --flag"}
    plan = arms.prepare_server(cmd, arms.ARMS["C-http"], str(tmp_path / "out"), env)
    assert list(plan.cmd) == ["/venv/bin/glm5-serve", "--flag"] + cmd[1:]
    assert plan.config == CONFIG
    assert not (tmp_path / "out").exists(), "C-http changes no setting, so writes no YAML"
    with pytest.raises(arms.ArmError, match="TRTLLM_TEST_GLM_SERVE_CMD"):
        arms.prepare_server(cmd, arms.ARMS["C-http"], str(tmp_path), {})


@pytest.mark.parametrize(
    "extra",
    [
        ["--num_serve_frontends", "2"],
        ["--num_postprocess_workers=2"],
        ["--set", "num_serve_frontends=2"],
        ["--set", "kv_cache_config.enable_block_reuse=true"],
        ["--set=stream_interval=4"],
    ],
)
def test_prepare_server_refuses_cli_values_the_arm_sets(arms, tmp_path, extra):
    with pytest.raises(arms.ArmError, match="sets "):
        arms.prepare_server(
            _serve_cmd(_write_config(tmp_path), *extra), arms.ARMS["B-s1"], str(tmp_path)
        )
    # An arm that keeps the test's value leaves the option alone.
    if extra[0].startswith("--num_postprocess_workers"):
        arms.prepare_server(
            _serve_cmd(_write_config(tmp_path), *extra), arms.ARMS["B-full"], str(tmp_path)
        )


def test_prepare_server_refuses_what_the_frontend_refuses(arms, tmp_path, monkeypatch):
    """A chain arm's front end starts after the engine loads; its refusal must come first."""
    seen = []

    def check(args):
        seen.append(vars(args))
        if getattr(args, "enable_iter_perf_stats", False):
            raise ValueError("enable_iter_perf_stats needs the OpenEngine metrics recorder")

    monkeypatch.setattr(arms, "_frontend_check", lambda: check)
    source = _write_config(tmp_path, {**CONFIG, "enable_iter_perf_stats": True}, "iter.yml")
    with pytest.raises(arms.ArmError, match="B-s1: enable_iter_perf_stats needs"):
        arms.prepare_server(_serve_cmd(source), arms.ARMS["B-s1"], str(tmp_path))
    arms.prepare_server(_serve_cmd(source), arms.ARMS["A-1f0"], str(tmp_path))
    assert len(seen) == 1, "arms without a chain run no front-end check"

    seen.clear()
    plain = _write_config(tmp_path)
    cmd = _serve_cmd(
        plain,
        "--max_beam_width", "4",
        "--post_processor_hook", "pkg.mod:hook",
        "--set", "otlp_traces_endpoint=http://x",
    )  # fmt: skip
    arms.prepare_server(cmd, arms.ARMS["C-s1"], str(tmp_path))
    assert "max_beam_width" not in seen[0], "the harness gives --max_beam_width to the engine only"
    assert seen[0]["post_processor_hook"] == "pkg.mod:hook"
    assert seen[0]["otlp_traces_endpoint"] == "http://x"
    assert seen[0]["num_postprocess_workers"] == 0, "the check sees the arm's settings"


def test_prepare_server_refuses_values_of_the_wrong_type(arms, tmp_path, monkeypatch):
    """The YAML is not validated yet: a value of the wrong type is refused as ArmError too."""

    def check(args):
        if args.num_postprocess_workers > 0:  # TypeError for a string
            raise ValueError("unreachable")

    monkeypatch.setattr(arms, "_frontend_check", lambda: check)
    source = _write_config(tmp_path, {**CONFIG, "num_postprocess_workers": "4"}, "str.yml")
    with pytest.raises(arms.ArmError, match="C-full: TypeError.*'>'"):
        arms.prepare_server(_serve_cmd(source), arms.ARMS["C-full"], str(tmp_path))


def test_prepare_server_uses_the_frontend_check(arms, tmp_path, facade, monkeypatch):
    # As when no metrics recorder is installed, so enable_iter_perf_stats is refused.
    monkeypatch.setattr(
        facade.contracts,
        "resolve_slot",
        lambda name: (facade.contracts.slots()[name].default, "default"),
    )
    source = _write_config(tmp_path, {**CONFIG, "enable_iter_perf_stats": True}, "iter.yml")
    with pytest.raises(arms.ArmError, match="enable_iter_perf_stats"):
        arms.prepare_server(_serve_cmd(source), arms.ARMS["B-s1"], str(tmp_path))
    arms.prepare_server(_serve_cmd(_write_config(tmp_path)), arms.ARMS["B-s1"], str(tmp_path))


def test_prepare_server_refuses_two_configs(arms, tmp_path):
    source = _write_config(tmp_path)
    with pytest.raises(arms.ArmError, match="2 YAML configurations"):
        arms.prepare_server(
            _serve_cmd(source, "--extra_llm_api_options", source), arms.ARMS["A-1f"], str(tmp_path)
        )


# ----------------------------------------------------------------------------- parsing


def test_parse_metrics(arms):
    metrics, tokens = arms.parse_metrics(BENCHMARK_OUTPUT)
    assert tokens == 512
    assert metrics == {
        "ttft_ms": {"mean": 21.5, "median": 20.1, "p99": 40.25},
        "tpot_ms": {"mean": 5.1, "median": 5.0, "p99": 6.75},
        "itl_ms": {"mean": 5.2, "median": 5.05, "p99": 7.0},
        "e2el_ms": {"mean": 180.0, "median": 175.5, "p99": 220.1},
        "output_tok_per_s": 159.5,
    }
    empty, none = arms.parse_metrics("")
    assert none is None and empty["ttft_ms"] == {"mean": None, "median": None, "p99": None}


def test_parse_metrics_matches_benchmark_serving_labels(arms):
    """benchmark_serving prints every label the parser reads, in the form it reads."""
    source = (_TESTS.parent / "tensorrt_llm/serve/scripts/benchmark_serving.py").read_text()
    for label in (
        "Total generated tokens:",
        "Output token throughput (tok/s):",
        "Failed requests:",
        'f"Mean {metric_name}{display_unit}:"',
        'f"Median {metric_name}{display_unit}:"',
        'f"P{p_word} {metric_name}{display_unit}:"',
        'display_unit = f" ({unit_suffix})"',
    ):
        assert label in source, label
    for name in ("TTFT", "TPOT", "ITL", "E2EL"):
        assert f'"{name}"' in source or f"'{name}'" in source


@pytest.mark.parametrize(
    "cmd, kind",
    [
        (CLIENT_TOKEN_IDS, "token_ids"),
        (CLIENT_TEXT, "text"),
        (["--dataset-name", "random"], "text"),
        (["--tokenize-on-client"], "text"),
        (["--dataset-name=random_image", "--tokenize-on-client"], "token_ids"),
    ],
)
def test_dataset_kind(arms, cmd, kind):
    assert arms.dataset_kind(cmd) == kind


def test_alignment_prefers_command_line(arms):
    plan = arms.ServerPlan(
        cmd=("trtllm-serve", "m", "--reasoning_parser", "deepseek-r1", "--num_serve_frontends=3"),
        config={
            **CONFIG,
            "reasoning_parser": "qwen3",
            "tool_parser": "glm47",
            "num_serve_frontends": 8,
        },
    )
    got = arms.alignment(plan, {"TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "2.384"})
    assert got == {
        "enable_block_reuse": False,
        "reasoning_parser": "deepseek-r1",
        "tool_parser": "glm47",
        "disable_mm_encoder": True,
        "stream_interval": 20,
        "num_serve_frontends": 3,
        "num_postprocess_workers": 4,
        "spec_decode_force_accepted": 2.384,
    }
    assert arms.alignment(arms.ServerPlan(("x",), {}), {})["enable_block_reuse"] is None


# ----------------------------------------------------------------------------- result file


def _result(arms, name, **overrides):
    arm = arms.ARMS[name]
    kwargs = dict(
        arm=arm,
        plan=arms.ServerPlan(("trtllm-serve", "m"), arms.apply_arm(CONFIG, arm)),
        server_env={},
        output=BENCHMARK_OUTPUT,
        client_cmd=CLIENT_TOKEN_IDS,
        test_id="perf/test_perf_sanity.py::test_e2e[aggr-x]|server=0|client=0",
        frontend_cpu_s=2.56,
        chain_proof=None if arm.in_process else {"generate_sent": 17, "generate_received": 17},
        notes=["n"],
        environ={"TRTLLM_PERF_RUN": "2", "SLURM_JOB_ID": "123", "TRTLLM_PERF_GLM_SHA": "abc"},
    )
    kwargs.update(overrides)
    return arms.build_result(**kwargs)


@pytest.mark.parametrize("name", list(EXPECTED_ARMS))
def test_result_matches_schema(arms, name, tmp_path):
    doc = _result(arms, name)
    assert arms.validate_result(doc) == []
    assert doc["case"] == EXPECTED_ARMS[name][0] and doc["arm"] == name
    assert doc["run"] == 2 and doc["job_id"] == "123" and doc["concurrency"] == 4
    assert doc["dataset"] == "token_ids" and doc["output_tokens"] == 512
    assert doc["frontend_cpu_us_per_token"] == pytest.approx(2.56e6 / 512)
    assert doc["versions"]["glm"] == "abc"
    assert doc["alignment"]["num_serve_frontends"] == EXPECTED_ARMS[name][3]
    path = arms.write_result(doc, str(tmp_path))
    assert json.loads(Path(path).read_text()) == doc
    again = arms.write_result(doc, str(tmp_path))
    assert again != path and Path(path).exists(), "a second run must not overwrite the first"


def test_unparsable_accepted_tokens_are_noted(arms):
    doc = _result(arms, "A-def", server_env={"TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "auto"})
    assert doc["alignment"]["spec_decode_force_accepted"] is None
    assert any("'auto' is not a number" in n for n in doc["notes"])
    assert arms.validate_result(doc) == []


def test_trtllm_sha_is_the_checkout(arms):
    expected = subprocess.run(
        ["git", "-C", str(_TESTS), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if not expected:
        pytest.skip("not a git checkout")
    assert arms.trtllm_sha({}) == expected


def test_result_file_goes_to_results_dir(arms, tmp_path, monkeypatch):
    """RunProbe.finish writes under $TRTLLM_OPENENGINE_RESULTS_DIR/perf when it is set."""
    server = _FakeServer(arms, "A-1f0")
    reader = _FakeReader(arms, ticks={10: [(1, 100), (1, 300)]}, gpu={11})
    probe = arms.RunProbe.start(server, reader)
    env = {"TRTLLM_OPENENGINE_RESULTS_DIR": str(tmp_path / "res"), "TRTLLM_PERF_RUN": "1"}
    path = probe.finish(
        output=BENCHMARK_OUTPUT,
        client_cmd=CLIENT_TEXT,
        test_id="t",
        default_dir=str(tmp_path / "default"),
        environ=env,
    )
    assert Path(path).parent == tmp_path / "res" / "perf"
    doc = json.loads(Path(path).read_text())
    assert doc["frontend_cpu_s"] == pytest.approx(2.0) and doc["chain_proof"] is None
    assert doc["dataset"] == "text"
    path = probe.finish(
        output=BENCHMARK_OUTPUT,
        client_cmd=CLIENT_TEXT,
        test_id="t",
        default_dir=str(tmp_path / "default"),
        environ={"TRTLLM_PERF_RUN": "1"},
    )
    assert Path(path).parent == tmp_path / "default"


@pytest.mark.parametrize(
    "name, change, message",
    [
        ("A-def", lambda d: d.update(extra=1), "unexpected"),
        ("A-def", lambda d: d.pop("alignment"), "missing"),
        ("A-def", lambda d: d.update(case="case-01"), "case"),
        (
            "A-def",
            lambda d: d.update(chain_proof={"generate_sent": 1, "generate_received": 1}),
            "null",
        ),
        ("B-s1", lambda d: d.update(chain_proof=None), "chain_proof must hold"),
        ("B-s1", lambda d: d["chain_proof"].update(generate_sent=-1), "non-negative"),
        ("A-def", lambda d: d.update(dataset="ids"), "dataset"),
        ("A-def", lambda d: d.update(run=0), "run"),
        ("A-def", lambda d: d["metrics"]["ttft_ms"].pop("p99"), "ttft_ms"),
        ("A-def", lambda d: d["metrics"].update(output_tok_per_s="fast"), "output_tok_per_s"),
        ("A-def", lambda d: d["alignment"].pop("tool_parser"), "alignment"),
        ("A-def", lambda d: d.update(frontend_cpu_s=-1.0), "frontend_cpu_s"),
        ("A-def", lambda d: d.update(arm="Z"), "arm"),
    ],
)
def test_invalid_results_are_rejected(arms, tmp_path, name, change, message):
    doc = _result(arms, name)
    change(doc)
    errors = arms.validate_result(doc)
    assert errors and any(message in e for e in errors), errors
    with pytest.raises(arms.ArmError, match="invalid perf result"):
        arms.write_result(doc, str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_run_number_must_be_positive(arms):
    with pytest.raises(arms.ArmError, match="TRTLLM_PERF_RUN"):
        _result(arms, "A-def", environ={"TRTLLM_PERF_RUN": "x"})


# ----------------------------------------------------------------------------- front-end CPU


class _FakeServer:
    def __init__(self, arms, name, handle=None, root=10, config=None):
        self.arm = arms.ARMS[name]
        self.plan = arms.ServerPlan(("trtllm-serve", "m"), config or {})
        self.env = {}
        self.handle = handle
        self.frontend_pid = root


class _FakeReader:
    """Process 10 is the root; 11 holds a GPU and has child 12; 13 is a front end."""

    def __init__(self, arms, *, ticks, gpu, tree=None, now=0):
        self._ticks = {pid: list(values) for pid, values in ticks.items()}
        tree = tree or {10: {10, 11, 12, 13}, 11: {11, 12}}
        self.reader = arms.CpuReader(
            tree=lambda pid: set(tree.get(pid, {pid})),
            gpu=lambda: gpu,
            ticks=self._read,
            ticks_per_s=100.0,
            now=lambda: now,
        )

    def _read(self, pid):
        values = self._ticks.get(pid)
        if not values:
            return None
        return values.pop(0) if len(values) > 1 else values[0]

    def __getattr__(self, name):
        return getattr(self.reader, name)


@pytest.mark.parametrize("in_process", [True, False])
def test_engine_in_the_root_has_no_cpu_time(arms, in_process):
    """The engine runs inside the root (glm5-serve's own HTTP server, say)."""
    fake = _FakeReader(arms, ticks={10: [(1, 5)]}, gpu={10, 999})
    pids, note = fake.frontend_pids(10, in_process=in_process)
    assert pids is None and "root process 10 holds the engine's GPU memory" in note


def test_engine_processes_are_the_ones_with_engine_sized_memory(arms, monkeypatch):
    """Front ends with a CUDA context of their own (~0.7 GiB) are not engine processes."""
    out = "10, 682\n13, 600\n11, 143420\n12, 2048\nbad line\n14, [N/A]\n"

    def run(cmd, **kwargs):
        assert cmd[0] == "nvidia-smi"
        return subprocess.CompletedProcess(cmd, 0, stdout=out)

    monkeypatch.setattr(arms.subprocess, "run", run)
    assert arms._engine_gpu_pids() == {11, 12}
    fake = _FakeReader(arms, ticks={}, gpu=arms._engine_gpu_pids())
    pids, _ = fake.frontend_pids(10, in_process=True)
    assert pids == {10, 13}


def test_engine_pids_without_nvidia_smi(arms, monkeypatch):
    def run(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(arms.subprocess, "run", run)
    assert arms._engine_gpu_pids() is None


def test_in_process_excludes_gpu_subtree(arms):
    fake = _FakeReader(arms, ticks={}, gpu={11, 999})
    pids, _ = fake.frontend_pids(10, in_process=True)
    assert pids == {10, 13}


@pytest.mark.parametrize("gpu", [set(), None])
def test_in_process_without_gpu_process_has_no_cpu_time(arms, gpu):
    fake = _FakeReader(arms, ticks={10: [(1, 5)]}, gpu=gpu)
    pids, note = fake.frontend_pids(10, in_process=True)
    assert pids is None and "cannot tell" in note
    probe = arms.RunProbe.start(_FakeServer(arms, "A-def"), fake)
    assert probe.frontend_cpu_s() is None


def test_new_processes_started_before_the_run_are_not_counted(arms):
    """A process missed by the first read that is older than the run would add its whole life."""
    fake = _FakeReader(
        arms,
        ticks={10: [(1, 100), (1, 300)], 14: [(5, 9000)], 15: [(60, 30)]},
        gpu={11},
        tree={10: {10, 11, 12}, 11: {11, 12}},
        now=50,
    )
    probe = arms.RunProbe.start(_FakeServer(arms, "A-def"), fake)
    fake.reader.tree = lambda pid: {10: {10, 11, 12, 14, 15}, 11: {11, 12}}[pid]
    assert probe.frontend_cpu_s() == pytest.approx((200 + 30) / 100.0)
    assert any("started before it" in note for note in probe.notes)


def test_reused_pid_of_an_exited_front_end_process_is_not_counted(arms):
    """Front-end process 15 exits; a process outside the server takes its PID during the run."""
    fake = _FakeReader(
        arms,
        ticks={10: [(1, 100), (1, 300)], 15: [(3, 40), (60, 5000)]},
        gpu={11},
        tree={10: {10, 11, 12, 15}, 11: {11, 12}},
        now=50,
    )
    probe = arms.RunProbe.start(_FakeServer(arms, "A-def"), fake)
    fake.reader.tree = lambda pid: {10: {10, 11, 12}, 11: {11, 12}}[pid]
    assert probe.frontend_cpu_s() == pytest.approx(200 / 100.0)
    assert any("exited" in note for note in probe.notes)


def test_chain_without_nvidia_smi_counts_the_whole_tree(arms):
    fake = _FakeReader(arms, ticks={}, gpu=None, tree={10: {10, 13}})
    pids, note = fake.frontend_pids(10, in_process=False)
    assert pids == {10, 13} and "whole front-end tree" in note


def test_cpu_delta(arms):
    # 10: +200 ticks; 13 restarted under the same PID: its new 50 ticks count;
    # 14 appears during the run: all 30 count; 15 exits: lost; 11/12 are the engine.
    fake = _FakeReader(
        arms,
        ticks={
            10: [(1, 100), (1, 300)],
            11: [(1, 1000), (1, 9000)],
            12: [(1, 1000), (1, 9000)],
            13: [(5, 70), (6, 50)],
            14: [(9, 30)],
            15: [(3, 40)],
        },
        gpu={11},
        tree={10: {10, 11, 12, 13, 15}, 11: {11, 12}},
    )
    probe = arms.RunProbe.start(_FakeServer(arms, "A-def"), fake)
    fake.reader.tree = lambda pid: {10: {10, 11, 12, 13, 14}, 11: {11, 12}}[pid]
    fake._ticks[15] = []
    assert probe.frontend_cpu_s() == pytest.approx((200 + 50 + 30) / 100.0)
    assert any("exited" in note for note in probe.notes)


def test_cpu_of_real_processes(arms):
    """The default readers find a busy child of this process and its CPU time."""
    busy = subprocess.Popen(
        [sys.executable, "-c", "import sys, time\nsys.stdin.readline()\nt=time.process_time()\n"
         "while time.process_time() - t < 1.0: pass\nsys.stdin.readline()"],
        stdin=subprocess.PIPE,
        text=True,
    )  # fmt: skip
    try:
        reader = arms.CpuReader(gpu=lambda: set())
        server = _FakeServer(arms, "B-s1", root=os.getpid())
        server.handle = None  # no counters to read
        probe = arms.RunProbe.start(server, reader)
        assert busy.pid in probe.cpu_before
        busy.stdin.write("go\n")
        busy.stdin.flush()
        deadline = time.monotonic() + 30
        while reader.ticks(busy.pid)[1] - probe.cpu_before[busy.pid][1] < 0.5 * reader.ticks_per_s:
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert probe.frontend_cpu_s() >= 0.5
    finally:
        busy.stdin.write("done\n")
        busy.stdin.close()
        busy.wait(timeout=30)


# ----------------------------------------------------------------------------- chain counts


def test_chain_counts_settle(arms, monkeypatch):
    from openengine_chain.proof import Counters

    reads = iter(
        [
            Counters(None, 5, frontend_unwritten=True),  # before: no counter file yet
            Counters(3, 9),  # the front end's last count not written yet
            Counters(4, 9),
        ]
    )
    monkeypatch.setattr(arms, "_read_counters", lambda handle, notes: next(reads))
    monkeypatch.setattr(arms, "_SETTLE_INTERVAL_S", 0)
    reader = _FakeReader(arms, ticks={}, gpu=set(), tree={10: {10}})
    probe = arms.RunProbe.start(_FakeServer(arms, "B-s1", handle=object()), reader)
    assert probe.chain_counts() == {"generate_sent": 4, "generate_received": 4}


def test_chain_counts_without_frontend_counter(arms, monkeypatch):
    from openengine_chain.proof import Counters

    monkeypatch.setattr(arms, "_read_counters", lambda handle, notes: Counters(None, 2, True))
    reader = _FakeReader(arms, ticks={}, gpu=set(), tree={10: {10}})
    probe = arms.RunProbe.start(_FakeServer(arms, "C-s1", handle=object()), reader)
    assert probe.chain_counts() == {"generate_sent": None, "generate_received": 0}
    assert any("no Generate count of the front end" in note for note in probe.notes)


def test_chain_counts_that_disagree_are_noted(arms, monkeypatch):
    from openengine_chain.proof import Counters

    reads = iter([Counters(0, 0)] + [Counters(5, 4)] * 4)
    monkeypatch.setattr(arms, "_read_counters", lambda handle, notes: next(reads))
    monkeypatch.setattr(arms, "_SETTLE_INTERVAL_S", 0)
    reader = _FakeReader(arms, ticks={}, gpu=set(), tree={10: {10}})
    probe = arms.RunProbe.start(_FakeServer(arms, "B-s1", handle=object()), reader)
    assert probe.chain_counts() == {"generate_sent": 5, "generate_received": 4}
    assert any("sent 5 Generate calls, the engine received 4" in n for n in probe.notes)


@pytest.mark.parametrize("bug", [AttributeError, KeyError])
def test_counter_bugs_are_not_hidden(arms, monkeypatch, bug):
    from openengine_chain import proof

    def broken(handle):
        raise bug("harness API changed")

    monkeypatch.setattr(proof, "read_counters", broken)
    reader = _FakeReader(arms, ticks={}, gpu=set(), tree={10: {10}})
    with pytest.raises(bug):
        arms.RunProbe.start(_FakeServer(arms, "B-s1", handle=object()), reader)


def test_unreadable_counters_count_as_unknown(arms, monkeypatch):
    from openengine_chain import proof

    def broken(handle):
        raise OSError("counter file vanished")

    monkeypatch.setattr(proof, "read_counters", broken)
    monkeypatch.setattr(arms, "_SETTLE_INTERVAL_S", 0)
    reader = _FakeReader(arms, ticks={}, gpu=set(), tree={10: {10}})
    probe = arms.RunProbe.start(_FakeServer(arms, "B-s1", handle=object()), reader)
    assert probe.chain_counts() == {"generate_sent": None, "generate_received": None}
    assert any("counter file vanished" in note for note in probe.notes)


@pytest.mark.parametrize("content", ["null", '{"pid": 1, "generate_sent": null}'])
def test_counter_file_holding_null_counts_as_unknown(arms, tmp_path, content):
    """The real reader on a counter file with no count: ValueError, no count."""
    (tmp_path / "1.json").write_text(content)
    handle = types.SimpleNamespace(alive=True, counter_dir=str(tmp_path))
    notes = []
    counters = arms._read_counters(handle, notes)
    assert (counters.frontend, counters.engine) == (None, None)
    assert any("ValueError" in note for note in notes), notes


# ----------------------------------------------------------------------------- decoded tokens


def test_checks_decoded_tokens(arms):
    """Perf sanity's check of 'Mean Avg Decoded Tokens per Iter' skips the arms with a chain."""
    assert arms.checks_decoded_tokens(None)
    assert {name: arms.checks_decoded_tokens(arm) for name, arm in arms.ARMS.items()} == {
        name: chain == "none" for name, (_, chain, *_) in EXPECTED_ARMS.items()
    }
    assert arms.DECODED_TOKENS_LINE == "Mean Avg Decoded Tokens per Iter"


def _check_pattern():
    """The pattern test_perf_sanity.py parses the line with (read from its source)."""
    import re

    text = (_DEFS / "perf" / "test_perf_sanity.py").read_text()
    found = re.findall(r'"al": re\.compile\(r"([^"]+)"\)', text)
    assert len(found) == 1, found
    return re.compile(found[0])


AL_LINE = "Mean Avg Decoded Tokens per Iter:        3.38\n"


@pytest.mark.parametrize(
    "name, config, output, noted",
    [
        ("B-s1", CONFIG, BENCHMARK_OUTPUT, True),
        ("C-full", CONFIG, BENCHMARK_OUTPUT, True),
        ("B-s1", CONFIG, AL_LINE + BENCHMARK_OUTPUT, True),  # not checked even when printed
        ("B-s1", {}, BENCHMARK_OUTPUT, False),  # no speculative decoding
        # No decoding_type: perf sanity does not count it as speculative decoding.
        ("B-s1", {"speculative_config": {"max_draft_len": 3}}, BENCHMARK_OUTPUT, False),
        ("A-1f0", CONFIG, BENCHMARK_OUTPUT, False),  # the in-process arms keep the check
    ],
)
def test_unchecked_decoded_tokens_are_noted(
    arms, tmp_path, monkeypatch, name, config, output, noted
):
    from openengine_chain.proof import Counters

    assert _check_pattern().search(AL_LINE), "the line the note is about is the one checked"
    monkeypatch.setattr(arms, "_read_counters", lambda handle, notes: Counters(0, 0))
    arm = arms.ARMS[name]
    server = _FakeServer(arms, name, handle=None if arm.in_process else object(), config=config)
    reader = _FakeReader(arms, ticks={10: [(1, 100), (1, 300)]}, gpu={11})
    path = arms.RunProbe.start(server, reader).finish(
        output=output,
        client_cmd=CLIENT_TEXT,
        test_id="t",
        default_dir=str(tmp_path),
        environ={},
    )
    notes = json.loads(Path(path).read_text())["notes"]
    assert any(arms.DECODED_TOKENS_LINE in note for note in notes) == noted, notes


@pytest.mark.parametrize("name", [None, *sorted(EXPECTED_ARMS)])
def test_perf_sanity_checks_decoded_tokens_of_in_process_servers_only(arms, name):
    """PerfSanityTestConfig.check_test_failure, run from its source on a case without the line."""
    arm = None if name is None else arms.ARMS[name]
    warnings = []
    namespace = {
        "openengine_arms": types.SimpleNamespace(
            selected_arm=lambda: arm, checks_decoded_tokens=arms.checks_decoded_tokens
        ),
        "PERF_METRIC_LOG_QUERIES": {},
        "AGENTX_BENCHMARK_CLIENT": "agentx",
        "DISAGG_CONFIG_RUNTIMES": (),
        "GEN_ONLY_MODES": (),
        "TIME_BREAKDOWN_MODIFIER": "time_breakdown",
        "print_warning": warnings.append,
    }
    check = _perf_sanity_function("check_test_failure", namespace, owner="PerfSanityTestConfig")
    client = types.SimpleNamespace(
        only_run_accuracy=False, benchmark_client="benchmark_serving", spec_decoding=True
    )
    config = types.SimpleNamespace(
        server_client_configs={0: [client]},
        _perf_results={0: [{}]},
        server_configs={0: types.SimpleNamespace()},
        runtime="aggr_server",
        time_breakdown=False,
    )
    if name is None or EXPECTED_ARMS[name][1] == "none":
        with pytest.raises(RuntimeError, match="missing 'Mean Avg Decoded Tokens per Iter'"):
            check(config)
        assert warnings == []
    else:
        check(config)
        assert len(warnings) == 1 and name in warnings[0], warnings


@pytest.mark.parametrize(
    "output, message",
    [
        ("Successful requests: 4\n", "no 'Total generated tokens'"),
        (None, "2 failed requests"),
        ("Total generated tokens:                  0\n", "generated no tokens"),
        ("x", "!CHECK LOG FOR ERRORS!"),
    ],
)
def test_finish_refuses_incomplete_output(arms, tmp_path, output, message):
    if output == "x":
        output = BENCHMARK_OUTPUT + "=====!CHECK LOG FOR ERRORS!=====\n"
    if output is None:
        output = BENCHMARK_OUTPUT.replace(
            "Failed requests:                         0",
            "Failed requests:                         2",
        )
        assert output != BENCHMARK_OUTPUT
    reader = _FakeReader(arms, ticks={10: [(1, 100)]}, gpu={11})
    probe = arms.RunProbe.start(_FakeServer(arms, "A-1f0"), reader)
    with pytest.raises(arms.ArmError, match=message):
        probe.finish(
            output=output,
            client_cmd=CLIENT_TEXT,
            test_id="t",
            default_dir=str(tmp_path),
            environ={},
        )
    assert not list(tmp_path.iterdir())


def test_in_process_arm_reads_no_counters(arms):
    reader = _FakeReader(arms, ticks={}, gpu={11})
    probe = arms.RunProbe.start(_FakeServer(arms, "A-def"), reader)
    assert probe.counters_before is None and probe.chain_counts() is None


def test_current_test_id(arms, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "a/b.py::test_x[p q] (call)")
    assert arms.current_test_id() == "a/b.py::test_x[p q]"
