# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The single-node disaggregated perf driver (jenkins/scripts/perf/local/single_node_disagg) on CPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

# The local submit script, whose fill gate the driver repeats.
__extra_import_path__ = ["~/jenkins/scripts/perf/local"]

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[5]
DRIVER = REPO / "jenkins" / "scripts" / "perf" / "local" / "single_node_disagg"

PU02 = "b200_qwen3-0.6b-bf16_1k256_con16_ctx1_tp1_gen1_tp1_eplb0_mtp0_ccb-DEFAULT"
CONFIGS = DRIVER / "configs"


def _yaml(
    tmp_path, *, ctx_tp=1, gen_tp=2, num_ctx=1, num_gen=2, concurrency="16", mode="e2e", gen=None
):
    config = yaml.safe_load((CONFIGS / f"{PU02}.yaml").read_text())
    config["hardware"].update(num_ctx_servers=num_ctx, num_gen_servers=num_gen)
    config["worker_config"]["ctx"]["tensor_parallel_size"] = ctx_tp
    config["worker_config"]["gen"]["tensor_parallel_size"] = gen_tp
    config["worker_config"]["gen"].update(gen or {})
    config["benchmark"].update(concurrency_list=concurrency, mode=mode)
    path = tmp_path / "b200_test-config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_pu02_config_is_a_one_plus_one_gpu_case(driver):
    plan = driver.load_plan(f"disagg-e2e-{PU02}", str(CONFIGS))
    assert (plan.num_ctx, plan.num_gen, dict(plan.gpus), plan.concurrency) == (
        1,
        1,
        {"ctx": 1, "gen": 1},
        16,
    )
    assert plan.roles == ("GEN_0", "CTX_0") and plan.supported_gpus == ("B200",)
    config = yaml.safe_load((CONFIGS / f"{PU02}.yaml").read_text())
    assert config["metadata"]["model_name"] == "qwen3_0.6b"
    assert config["slurm"]["numa_bind"] is False, "the chain arms refuse numa_bind"
    assert "TRTLLM_DISAGG_NO_RETRY=1" in config["environment"]["server_env_var"]


@pytest.mark.parametrize(
    "test_id, message",
    [
        ("aggr-x", "not a 'disagg-<mode>-<config>'"),
        ("disagg_upload-e2e-x", "not a 'disagg-<mode>-<config>'"),
        ("disagg-ctx_only-x", "modes"),
        ("disagg-e2e-time_breakdown-x", "time_breakdown"),
    ],
)
def test_parse_test_id_refuses(driver, test_id, message):
    with pytest.raises(driver.DriverError, match=message):
        driver.parse_test_id(test_id)


def test_parse_test_id(driver):
    assert driver.parse_test_id("disagg-gen_only-b200_a-b") == ("gen_only", "b200_a-b")


def test_gen_only_no_context_is_refused(driver, tmp_path):
    path = _yaml(tmp_path, mode="gen_only_no_context")
    with pytest.raises(driver.DriverError, match="gen_only_no_context"):
        driver.load_plan(f"disagg-gen_only-{path.stem}", str(tmp_path))


def test_gpus_go_to_generation_workers_first(driver, tmp_path):
    path = _yaml(tmp_path, ctx_tp=1, gen_tp=2, num_ctx=1, num_gen=2)
    plan = driver.load_plan(f"disagg-e2e-{path.stem}", str(tmp_path))
    gpus = ["0", "1", "2", "3", "4", "5"]
    assert driver.assign_gpus(plan, gpus) == {
        "GEN_0": ["0", "1"],
        "GEN_1": ["2", "3"],
        "CTX_0": ["4"],
    }
    with pytest.raises(driver.DriverError, match="take 5 GPUs"):
        driver.assign_gpus(plan, gpus[:4])


def test_available_gpus(driver):
    assert driver.available_gpus({"CUDA_VISIBLE_DEVICES": "2, 5"}, lambda: 8) == ["2", "5"]
    assert driver.available_gpus({}, lambda: 3) == ["0", "1", "2"]


BASE = {
    "PATH": "/usr/bin",
    "PMIX_RANK": "0",
    "PMI_SIZE": "1",
    "OMPI_COMM_WORLD_SIZE": "1",
    "SLURM_JOB_ID": "77",
    "UCX_TLS": "rc",
    "UCX_NET_DEVICES": "mlx5_0:1",
    "PYTEST_CURRENT_TEST": "outer.py::t (call)",
    "CUDA_VISIBLE_DEVICES": "0,1,2",
}


@pytest.mark.parametrize("ucx", [{}, {"UCX_TLS": "cuda_ipc,tcp", "UCX_TCP_AF_PRIO": "inet"}])
def test_role_env(driver, tmp_path, ucx):
    path = _yaml(tmp_path, mode="gen_only", gen={"max_batch_size": 8})
    plan = driver.load_plan(f"disagg-gen_only-{path.stem}", str(tmp_path))
    envs = {
        role: driver.role_env(BASE, plan, role, gpus=gpus, tag="t1", ucx=ucx)
        for role, gpus in (("GEN_0", ["0", "1"]), ("CTX_0", ["2"]), ("BENCHMARK", None))
    }
    for role, env in envs.items():
        assert env["DISAGG_SERVING_TYPE"] == role and env["TRTLLM_PERF_DRIVER_TAG"] == "t1"
        assert env["TRTLLM_PERF_SINGLE_NODE"] == "1" and env["DISAGG_CONFIG_FOLDER"] == str(
            tmp_path
        )
        assert env["SLURM_JOB_ID"] == "77", "every role names the exchange directory after the job"
        assert not {"PMIX_RANK", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "PYTEST_CURRENT_TEST"} & set(
            env
        )
    assert (
        envs["GEN_0"]["CUDA_VISIBLE_DEVICES"] == "0,1"
        and envs["CTX_0"]["CUDA_VISIBLE_DEVICES"] == "2"
    )
    assert envs["BENCHMARK"]["CUDA_VISIBLE_DEVICES"] == "0,1,2"
    for role in ("GEN_0", "CTX_0"):
        ucx_vars = {k: v for k, v in envs[role].items() if k.startswith("UCX_")}
        assert ucx_vars == ucx, (
            "both worker roles: the submit scripts' cleared variables plus the rule's"
        )
    assert envs["BENCHMARK"]["UCX_TLS"] == "rc"
    # The fill gate of a gen_only test: generation workers only, capped by max_batch_size.
    assert envs["GEN_0"]["TLLM_BENCHMARK_REQ_QUEUES_SIZE"] == "8"
    assert envs["GEN_0"]["TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP"] == "1"
    assert "TLLM_BENCHMARK_REQ_QUEUES_SIZE" not in envs["CTX_0"]
    assert "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP" not in envs["CTX_0"]


def test_e2e_has_no_fill_gate(driver):
    plan = driver.load_plan(f"disagg-e2e-{PU02}", str(CONFIGS))
    env = driver.role_env(BASE, plan, "GEN_0", gpus=["0"], tag="t", ucx={})
    assert "TLLM_BENCHMARK_REQ_QUEUES_SIZE" not in env


def test_fill_gate_matches_the_submit_script(driver, tmp_path):
    import submit

    for gen in (
        {"max_batch_size": 8},
        {"max_batch_size": 4, "enable_attention_dp": True},
        {"max_batch_size": 64},
    ):
        path = _yaml(tmp_path, gen=gen)
        plan = driver.load_plan(f"disagg-gen_only-{path.stem}", str(tmp_path))
        config = yaml.safe_load(path.read_text())
        assert driver.gen_fill_queue_size(plan) == submit.get_benchmark_request_queue_size(
            config, 16
        )


def test_ucx_rule_is_the_submit_scripts(driver):
    from test_common import cluster_env

    for cluster in ("", "computelab", "oci-jhb-x", "nsc-svg-slurm-1"):
        assert driver.ucx_env(cluster, "B200") == cluster_env.get_ucx_env(cluster, "B200")
    # UV-53: a cluster with no rule of its own clears the submit scripts' variables and sets none.
    assert driver.ucx_env("computelab", "B200") == {}


def test_stale_state(driver, tmp_path):
    assert driver.stale_state(str(tmp_path / "missing")) == []
    for name in (
        "hostnames-1-0",
        "benchmark_status.0.txt",
        "gen_server_0.done",
        "perf-arm-1-0",
        "keep.log",
    ):
        (tmp_path / name).mkdir()
    assert driver.stale_state(str(tmp_path)) == [
        "benchmark_status.0.txt",
        "gen_server_0.done",
        "hostnames-1-0",
        "perf-arm-1-0",
    ]


def test_check_arm(driver):
    arms = driver.load_arms(str(REPO / "tests" / "integration" / "defs")).DISAGG_ARMS
    assert driver.check_arm({"TRTLLM_PERF_ARM": "E4-chain"}, arms) == "E4-chain"
    for env in ({}, {"TRTLLM_PERF_ARM": "E4-direct"}, {"TRTLLM_PERF_ARM": "A-def"}):
        with pytest.raises(driver.DriverError, match="E4-base, E4-base-lean, E4-chain"):
            driver.check_arm(env, arms)


def test_harness_env(driver, tmp_path):
    defs = str(REPO / "tests" / "integration" / "defs")
    env = {"PYTHONPATH": "/x", "PYTEST_ADDOPTS": "-q"}
    driver.harness_env(env, defs_dir=defs, run_dir=str(tmp_path))
    link = tmp_path / "pythonpath" / "openengine_chain"
    assert link.is_symlink() and link.resolve() == Path(defs, "openengine_chain").resolve()
    assert env["PYTHONPATH"] == f"{tmp_path / 'pythonpath'}{os.pathsep}/x"
    assert env["PYTEST_ADDOPTS"] == "-q -p openengine_chain.pytest_plugin"
    assert env["TRTLLM_OPENENGINE_RESULTS_DIR"] == str(tmp_path / "openengine")
    again = dict(env)
    driver.harness_env(again, defs_dir=defs, run_dir=str(tmp_path))
    assert again["PYTEST_ADDOPTS"] == env["PYTEST_ADDOPTS"], "the plugin is added once"


def test_pytest_command(driver):
    plan = driver.load_plan(f"disagg-e2e-{PU02}", str(CONFIGS))
    runs = driver.plan_runs(
        plan, {"GEN_0": ["0"], "CTX_0": ["1"]}, python="py", output_dir="/o", run_dir="/r"
    )
    assert [r.role for r in runs] == ["GEN_0", "CTX_0", "DISAGG_SERVER", "BENCHMARK"]
    assert [r.gpus for r in runs] == [["0"], ["1"], None, None]
    assert runs[0].cmd == [
        "py", "-m", "pytest", "-v", f"perf/test_perf_sanity.py::test_e2e[disagg-e2e-{PU02}]",
        "--output-dir=/o", "-o", "junit_logging=out-err", "--junitxml=/r/results-gen_0.xml",
    ]  # fmt: skip
    assert runs[3].log == "/r/benchmark.log"


def _script(code: str) -> list[str]:
    return [sys.executable, "-c", code]


@pytest.fixture
def quick(driver, monkeypatch):
    monkeypatch.setattr(driver, "SETTLE_S", 0.2)
    monkeypatch.setattr(driver, "STOP_GRACE_S", 2.0)
    return driver


def test_supervise_touches_done_and_stops_what_outlives_the_benchmark(quick, tmp_path):
    driver = quick
    runs = [
        driver.RoleRun(
            "GEN_0", _script("import time; time.sleep(0.3)"), str(tmp_path / "g.log"), ["0"]
        ),
        driver.RoleRun(
            "CTX_0", _script("import time; time.sleep(600)"), str(tmp_path / "c.log"), ["1"]
        ),
        driver.RoleRun("BENCHMARK", _script("pass"), str(tmp_path / "b.log"), None),
    ]
    for run in runs:
        driver.start_role(run, dict(os.environ), str(tmp_path))
    notes = driver.supervise(
        runs, test_output_dir=str(tmp_path), timeout_s=60, after_benchmark_s=1, poll_s=0.1
    )
    assert (tmp_path / "gen_server_0.done").exists()
    assert [r.returncode for r in runs][0::2] == [0, 0]
    assert runs[1].stopped and runs[1].returncode != 0
    assert notes == ["stopped CTX_0: still running 1s after the benchmark exited"]


def test_supervise_stops_everything_at_the_timeout(quick, tmp_path):
    driver = quick
    runs = [
        driver.RoleRun(
            "BENCHMARK", _script("import time; time.sleep(600)"), str(tmp_path / "b.log"), None
        )
    ]
    driver.start_role(runs[0], dict(os.environ), str(tmp_path))
    start = time.monotonic()
    notes = driver.supervise(
        runs, test_output_dir=str(tmp_path), timeout_s=0.5, after_benchmark_s=60, poll_s=0.1
    )
    assert len(notes) == 1 and notes[0].startswith("stopped BENCHMARK: the run took over")
    assert runs[0].stopped and time.monotonic() - start < 30


def test_tagged_processes_are_found_and_stopped(quick, tmp_path):
    driver = quick
    env = dict(os.environ, TRTLLM_PERF_DRIVER_TAG="wt70-test-tag")
    proc = subprocess.Popen(
        _script("import time; time.sleep(600)"), env=env, start_new_session=True
    )
    try:
        deadline = time.monotonic() + 10
        while proc.pid not in driver.tagged_pids("wt70-test-tag") and time.monotonic() < deadline:
            time.sleep(0.1)
        assert driver.tagged_pids("wt70-test-tag") == [proc.pid]
        assert driver.tagged_pids("wt70-other-tag") == []
        driver.stop_pids([proc.pid], 2.0)
        assert proc.wait(10) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def _main(driver, tmp_path, env, *extra):
    argv = [
        "--output-dir",
        str(tmp_path / "out"),
        "--config-dir",
        str(CONFIGS),
        *extra,
        f"disagg-e2e-{PU02}",
    ]
    return driver.main(argv, environ=env)


def _report(tmp_path):
    return json.loads(
        (
            tmp_path / "out" / "single-node" / f"disagg-e2e-{PU02}" / "single-node-driver.json"
        ).read_text()
    )


ARM_ENV = {
    "TRTLLM_PERF_ARM": "E4-chain",
    "TRTLLM_TEST_OPENENGINE_CHAIN": "trtllm",
    "CUDA_VISIBLE_DEVICES": "4,5,6",
}


def test_main_dry_run_writes_the_plan(driver, tmp_path):
    env = dict(os.environ, **ARM_ENV, SLURM_CLUSTER_NAME="computelab")
    assert _main(driver, tmp_path, env, "--dry-run") == 0
    report = _report(tmp_path)
    assert (report["arm"], report["chain"], report["status"]) == ("E4-chain", "trtllm", "planned")
    assert [(r["role"], r["gpus"]) for r in report["roles"]] == [
        ("GEN_0", ["4"]), ("CTX_0", ["5"]), ("DISAGG_SERVER", None), ("BENCHMARK", None)
    ]  # fmt: skip
    assert report["ucx"] == {"cluster": "computelab", "gpu_type": "B200", "set": {}}
    assert report["environment"]["GEN_0"]["set"]["CUDA_VISIBLE_DEVICES"] == "4"
    # The run directory now exists: the same output directory is refused.
    assert _main(driver, tmp_path, env, "--dry-run") == 2


@pytest.mark.parametrize(
    "env, stale",
    [
        ({"CUDA_VISIBLE_DEVICES": "0,1"}, None),
        (dict(ARM_ENV, CUDA_VISIBLE_DEVICES="0"), None),
        (ARM_ENV, "hostnames-local-0"),
    ],
)
def test_main_refuses(driver, tmp_path, capsys, env, stale):
    if stale:
        (tmp_path / "out" / f"disagg-e2e-{PU02}" / stale).mkdir(parents=True)
    assert _main(driver, tmp_path, dict(os.environ, **env), "--dry-run") == 2
    assert "[single-node disagg]" in capsys.readouterr().err
    assert not (tmp_path / "out" / "single-node").exists(), "nothing is started or written"


def _fake_pytest(code_by_role):
    def command(python, plan, role, output_dir, run_dir):
        return [sys.executable, "-c", code_by_role.get(role, "pass")]

    return command


def test_main_runs_every_role_and_reports(quick, tmp_path, monkeypatch):
    driver = quick
    monkeypatch.setattr(
        driver, "pytest_command", _fake_pytest({"GEN_0": "import time; time.sleep(0.3)"})
    )
    assert _main(driver, tmp_path, dict(os.environ, **ARM_ENV)) == 0
    report = _report(tmp_path)
    assert report["status"] == "passed" and report["leftover_pids"] == [] and report["notes"] == []
    assert [r["returncode"] for r in report["roles"]] == [0, 0, 0, 0]
    assert (tmp_path / "out" / f"disagg-e2e-{PU02}" / "gen_server_0.done").exists()


def test_main_stops_what_a_role_leaves_behind(quick, tmp_path, monkeypatch):
    driver = quick
    orphan = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'], start_new_session=True)"
    )
    monkeypatch.setattr(driver, "pytest_command", _fake_pytest({"CTX_0": orphan}))
    assert _main(driver, tmp_path, dict(os.environ, **ARM_ENV)) == 1
    report = _report(tmp_path)
    assert report["status"] == "failed" and len(report["leftover_pids"]) == 1
    deadline = time.monotonic() + 10
    while any(driver._alive(p) for p in report["leftover_pids"]) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not any(driver._alive(p) for p in report["leftover_pids"])
