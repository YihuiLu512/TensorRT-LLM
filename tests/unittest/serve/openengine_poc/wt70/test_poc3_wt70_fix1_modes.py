# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one-server modes ctx_only and gen_only_no_context under the case-04 arms, on CPU.

perf/openengine_arms.py lets E4-base and E4-chain (and their lean variants)
take these two modes of a disaggregated YAML, and the single-node driver
(jenkins/scripts/perf/local/single_node_disagg) runs their aggr-<mode> test ids.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[5]
DEFS = REPO / "tests" / "integration" / "defs"
CONFIGS = REPO / "jenkins" / "scripts" / "perf" / "local" / "single_node_disagg" / "configs"
PU02 = "b200_qwen3-0.6b-bf16_1k256_con16_ctx1_tp1_gen1_tp1_eplb0_mtp0_ccb-DEFAULT"
ONE_SERVER = {"ctx_only": "aggr_server", "gen_only_no_context": "gen_only_no_context_server"}
BOTH_ALIKE = ["E4-base", "E4-chain", "E4-base-lean", "E4-chain-lean"]


def _case(**changes):
    values = dict(
        runtime="aggr_server",
        benchmark_mode="ctx_only",
        time_breakdown=False,
        numa_bind=False,
        dwdp_size=0,
        worker_gpus={"ctx": 4},
        gpus_per_node=8,
    )
    values.update(changes)
    return values


# ----------------------------------------------------------------------------- the arms


@pytest.mark.parametrize("name", BOTH_ALIKE)
@pytest.mark.parametrize("mode", sorted(ONE_SERVER))
def test_arms_that_start_both_roles_alike_take_the_one_server_modes(arms, name, mode):
    arms.check_disagg_case(
        arms.DISAGG_ARMS[name], **_case(benchmark_mode=mode, runtime=ONE_SERVER[mode])
    )


@pytest.mark.parametrize(
    "name, changes, message",
    [
        # A mode no arm takes is refused, whatever the runtime.
        ("E4-chain", {"benchmark_mode": "ctx_first"}, "disaggregated cases only"),
        ("E4-chain", {"benchmark_mode": "ctx_first", "runtime": "multi_node_disagg_server"},
         "takes the benchmark modes"),
        # Each one-server mode on its own runtime only.
        ("E4-chain", {"benchmark_mode": "ctx_only", "runtime": "multi_node_disagg_server"},
         "runs on runtime 'aggr_server'"),
        ("E4-chain", {"benchmark_mode": "gen_only_no_context"},
         "runs on runtime 'gen_only_no_context_server'"),
        # The arms whose two roles differ, and the case-03 arms, do not take them.
        ("E4-ctx", {}, "starts both roles alike"),
        ("E4-gen", {"benchmark_mode": "gen_only_no_context",
                    "runtime": "gen_only_no_context_server"}, "starts both roles alike"),
        ("E3-chain", {}, "case-04 arm"),
        ("E4-direct", {}, "direct replay"),
        # The disaggregated cases' other limits hold too.
        ("E4-chain", {"time_breakdown": True}, "time_breakdown"),
        ("E4-chain", {"benchmark_mode": "gen_only_no_context",
                      "runtime": "gen_only_no_context_server", "numa_bind": True}, "numa_bind"),
        ("E4-chain", {"worker_gpus": {"ctx": 16}}, "more than one node"),
    ],
)  # fmt: skip
def test_one_server_modes_refused(arms, name, changes, message):
    with pytest.raises(arms.ArmError, match=message):
        arms.check_disagg_case(arms.DISAGG_ARMS[name], **_case(**changes))


def test_the_modes_and_runtimes_are_perf_sanitys(arms):
    # openengine_arms does not import test_perf_sanity (it imports TensorRT-LLM's test deps).
    source = (DEFS / "perf" / "test_perf_sanity.py").read_text()
    assert re.search(r'^GEN_ONLY_NO_CONTEXT_RUNTIME = "gen_only_no_context_server"$', source, re.M)
    assert re.search(r'^AGGREGATED_DISAGG_YAML_MODES = \("ctx_only", GEN_ONLY_NO_CONTEXT_MODE\)$',
                     source, re.M)  # fmt: skip
    assert re.search(r'^GEN_ONLY_NO_CONTEXT_MODE = "gen_only_no_context"$', source, re.M)
    assert arms.SINGLE_SERVER_RUNTIMES == ONE_SERVER
    assert arms.SINGLE_SERVER_ARM_MODES == {"ctx_only": "ctx", "gen_only_no_context": "gen"}
    # perf sanity starts the ctx_only server on the aggregated path ("aggr_server").
    assert 'self.runtime = "aggr_server"' in source


@pytest.mark.parametrize(
    "name, mode, expected",
    [
        ("E4-chain", "ctx_only", ("E4-chain:ctx", "trtllm", 1, None)),
        ("E4-chain", "gen_only_no_context", ("E4-chain:gen", "trtllm", 1, None)),
        ("E4-base", "ctx_only", ("E4-base:ctx", "none", 1, None)),
        ("E4-chain-lean", "gen_only_no_context", ("E4-chain-lean:gen", "trtllm", 1, 0)),
    ],
)
def test_single_server_view_takes_the_mode_roles_settings(arms, name, mode, expected):
    view = arms.single_server_view(arms.DISAGG_ARMS[name], mode)
    assert (view.name, view.chain, view.frontends, view.postprocess_workers) == expected
    assert view.topology == arms.TOPOLOGY_AGG and view.case == arms.CASE_DISAGG


def test_single_server_view_leaves_other_runs_alone(arms):
    chain = arms.DISAGG_ARMS["E4-chain"]
    assert arms.single_server_view(chain, "e2e") is chain
    assert arms.single_server_view(chain, None) is chain
    assert arms.single_server_view(None, "ctx_only") is None
    poc2 = arms.ARMS["B-full"]
    assert arms.single_server_view(poc2, "ctx_only") is poc2


def test_result_arm(arms):
    assert arms.result_arm("B-full") is arms.ARMS["B-full"]
    assert arms.result_arm("E4-chain:ctx") == arms.role_view(arms.DISAGG_ARMS["E4-chain"], "ctx")
    for name in ("E4-chain", "E4-direct:ctx", "E4-chain:mid", "nope:gen", "E4-chain:"):
        assert arms.result_arm(name) is None, name


def test_one_server_view_prepares_and_proves_as_one_server(arms, tmp_path):
    view = arms.single_server_view(arms.DISAGG_ARMS["E4-chain"], "ctx_only")
    config = tmp_path / "ctx.yml"
    config.write_text(yaml.safe_dump({"num_serve_frontends": 8, "max_batch_size": 16}))
    plan = arms.prepare_server(["trtllm-serve", "m", "--config", str(config)], view, str(tmp_path))
    assert plan.config["num_serve_frontends"] == 1 and "--server_role" not in plan.cmd
    gen = arms.role_view(arms.DISAGG_ARMS["E4-chain"], "gen")
    worker = arms.prepare_worker(
        ["trtllm-serve", "m", "--config", str(config)], arms.DISAGG_ARMS["E4-chain"], "gen",
        str(tmp_path),
    )  # fmt: skip
    assert worker.cmd[-2:] == ("--server_role", "generation") and not gen.in_process


BENCHMARK_OUTPUT = """\
Successful requests:                     16
Total generated tokens:                  16
Output token throughput (tok/s):         9.50
Mean TTFT (ms):                          121.50
Median TTFT (ms):                        120.10
P99 TTFT (ms):                           140.25
Mean TPOT (ms):                          0.00
Median TPOT (ms):                        0.00
P99 TPOT (ms):                           0.00
Mean ITL (ms):                           0.00
Median ITL (ms):                         0.00
P99 ITL (ms):                            0.00
Mean E2EL (ms):                          121.60
Median E2EL (ms):                        120.20
P99 E2EL (ms):                           140.30
"""
CLIENT = [
    "python", "-m", "tensorrt_llm.serve.scripts.benchmark_serving", "--model", "m",
    "--max-concurrency", "16", "--dataset-name", "random", "--random-ids",
]  # fmt: skip


def _result(arms, name, mode, proof):
    view = arms.single_server_view(arms.DISAGG_ARMS[name], mode)
    return arms.build_result(
        arm=view,
        plan=arms.ServerPlan(("trtllm-serve", "m"), {"num_serve_frontends": 1}),
        server_env={},
        output=BENCHMARK_OUTPUT,
        client_cmd=CLIENT,
        test_id=f"perf/test_perf_sanity.py::test_e2e[aggr-{mode}-x]|server=0|client=0",
        frontend_cpu_s=1.0,
        chain_proof=proof,
        environ={"TRTLLM_PERF_RUN": "2"},
    )


@pytest.mark.parametrize("mode", sorted(ONE_SERVER))
def test_one_server_results_name_the_role_view_and_validate(arms, tmp_path, mode):
    chain = _result(arms, "E4-chain", mode, {"generate_sent": 16, "generate_received": 16})
    base = _result(arms, "E4-base", mode, None)
    role = arms.SINGLE_SERVER_ARM_MODES[mode]
    assert (chain["arm"], base["arm"]) == (f"E4-chain:{role}", f"E4-base:{role}")
    assert chain["case"] == base["case"] == "case-04" and chain["run"] == 2
    assert arms.validate_result(chain) == [] and arms.validate_result(base) == []
    path = arms.write_result(chain, str(tmp_path))
    assert json.loads(Path(path).read_text())["arm"] == f"E4-chain:{role}"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"arm": "E4-chain"}, "is not one of"),
        ({"arm": "E4-direct:ctx"}, "is not one of"),
        ({"arm": 7}, "is not one of"),
        ({"case": "case-02"}, "is not arm E4-chain:ctx's 'case-04'"),
        ({"chain_proof": None}, "generate_sent and generate_received"),
    ],
)
def test_one_server_results_are_checked(arms, change, message):
    doc = _result(arms, "E4-chain", "ctx_only", {"generate_sent": 1, "generate_received": 1})
    doc.update(change)
    assert any(message in error for error in arms.validate_result(doc)), arms.validate_result(doc)


def test_in_process_role_view_must_have_null_proof(arms):
    doc = _result(
        arms, "E4-base", "gen_only_no_context", {"generate_sent": 1, "generate_received": 1}
    )
    assert any("must be null" in error for error in arms.validate_result(doc))


def test_mirror_pump_copies_while_running_and_at_stop(arms, tmp_path):
    source, dest = tmp_path / "engine.log", tmp_path / "gen_server_0.log"
    source.write_text("a\n")
    pump = arms.MirrorPump(arms.LogMirror(lambda: str(source), str(dest)), interval_s=0.05)
    pump.start()
    try:
        deadline = time.monotonic() + 10
        while (dest.read_text() if dest.exists() else "") != "a\n":
            assert time.monotonic() < deadline, "the thread did not copy"
            time.sleep(0.02)
        with source.open("a") as f:
            f.write("b\n")
        assert pump.pump() in (0, 2)  # the thread may have copied it first
        assert dest.read_text() == "a\nb\n"
        with source.open("a") as f:
            f.write("c\n")
    finally:
        pump.stop()
    assert dest.read_text() == "a\nb\nc\n", "stop copies what is left"
    assert not pump._thread.is_alive()


def test_gen_only_no_context_is_single_side_on_the_trtllm_chain(harness_path):
    from openengine_chain import expectations as ex

    tables = ex.load()
    nodeid = "perf/test_perf_sanity.py::test_e2e[aggr-gen_only_no_context-b200_x-NIXL]"
    leg = ex.leg_expectation(ex.applicable(tables, nodeid, "trtllm"))
    assert (leg.wt, leg.leg) == ("WT-70-fix1", ex.SINGLE_SIDE)
    for other, chain in (
        ("perf/test_perf_sanity.py::test_e2e[aggr-ctx_only-b200_x-NIXL]", "trtllm"),
        ("perf/test_perf_sanity.py::test_e2e[aggr-gen_only_no_context-b200_x-NIXL]", "ds"),
    ):
        assert ex.leg_expectation(ex.applicable(tables, other, chain)) is None, (other, chain)
    leg = ex.leg_expectation(
        ex.applicable(tables, "perf/test_perf_sanity.py::test_e2e[disagg-e2e-b200_x]", "trtllm")
    )
    assert (leg.wt, leg.leg) == ("WT-70", ex.CROSS_NODE)


# ----------------------------------------------------------------------------- the driver


def _yaml(tmp_path, *, ctx_tp=4, gen_tp=4):
    config = yaml.safe_load((CONFIGS / f"{PU02}.yaml").read_text())
    config["worker_config"]["ctx"]["tensor_parallel_size"] = ctx_tp
    config["worker_config"]["gen"]["tensor_parallel_size"] = gen_tp
    path = tmp_path / "b200_fix1-config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_driver_repeats_the_arms_modes(driver, arms):
    assert driver.SINGLE_SERVER_MODES == arms.SINGLE_SERVER_ARM_MODES
    assert driver.DISAGG_MODES == arms.DISAGG_ARM_MODES


@pytest.mark.parametrize("mode", sorted(ONE_SERVER))
def test_parse_test_id_takes_the_one_server_ids(driver, mode):
    assert driver.parse_test_id(f"aggr-{mode}-b200_a-b") == (mode, "b200_a-b")


@pytest.mark.parametrize(
    "test_id, message",
    [
        ("aggr-e2e-x", "aggr modes"),
        ("aggr-ctx_first-x", "aggr modes"),
        ("aggr-x-y", "aggr modes"),
        ("disagg-gen_only_no_context-x", "disagg modes"),
        ("disagg-ctx_only-x", "disagg modes"),
        ("aggr_upload-ctx_only-x", "is not a"),
        ("aggr-ctx_only", "is not a"),
        ("aggr-ctx_only-time_breakdown-x", "time_breakdown"),
    ],
)
def test_parse_test_id_refuses_other_ids(driver, test_id, message):
    with pytest.raises(driver.DriverError, match=message):
        driver.parse_test_id(test_id)


@pytest.mark.parametrize("mode, gpus", [("ctx_only", "4567"), ("gen_only_no_context", "0123")])
def test_one_server_takes_its_roles_gpus(driver, tmp_path, mode, gpus):
    path = _yaml(tmp_path)
    plan = driver.load_plan(f"aggr-{mode}-{path.stem}", str(tmp_path))
    assert plan.mode == mode and plan.roles == ("SERVER",)
    assignment = driver.assign_gpus(plan, [str(i) for i in range(8)])
    assert assignment == {"SERVER": list(gpus)}, "the GPUs of its role in the e2e run"
    with pytest.raises(driver.DriverError, match="take 8 GPUs"):
        driver.assign_gpus(plan, [str(i) for i in range(7)])


def test_the_disagg_gen_only_id_with_a_no_context_config_is_still_refused(driver, tmp_path):
    config = yaml.safe_load(_yaml(tmp_path).read_text())
    config["benchmark"]["mode"] = "gen_only_no_context"
    path = tmp_path / "b200_fix1-nocontext.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(driver.DriverError, match="aggr-gen_only_no_context-<config> only"):
        driver.load_plan(f"disagg-gen_only-{path.stem}", str(tmp_path))
    assert driver.load_plan(f"aggr-gen_only_no_context-{path.stem}", str(tmp_path)).roles == (
        "SERVER",
    )


BASE = {
    "PATH": "/usr/bin",
    "SLURM_JOB_ID": "77",
    "UCX_TLS": "rc",
    "PYTEST_CURRENT_TEST": "outer.py::t (call)",
    "DISAGG_SERVING_TYPE": "BENCHMARK",
}


@pytest.mark.parametrize("mode", sorted(ONE_SERVER))
def test_one_server_env(driver, tmp_path, mode):
    path = _yaml(tmp_path)
    plan = driver.load_plan(f"aggr-{mode}-{path.stem}", str(tmp_path))
    env = driver.role_env(BASE, plan, "SERVER", gpus=["4", "5"], tag="t", ucx={"UCX_TLS": "tcp"})
    assert "DISAGG_SERVING_TYPE" not in env, "one pytest runs every process of the test"
    assert env["CUDA_VISIBLE_DEVICES"] == "4,5" and env["UCX_TLS"] == "tcp"
    assert env["TRTLLM_PERF_SINGLE_NODE"] == "1" and env["TRTLLM_PERF_DRIVER_TAG"] == "t"
    assert not {"SLURM_JOB_ID", "PYTEST_CURRENT_TEST"} & set(env)
    assert env.get("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == (
        "1" if mode == "gen_only_no_context" else None
    ), "as the local submit script exports it"
    assert not {"TLLM_BENCHMARK_REQ_QUEUES_SIZE", "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP"} & set(
        env
    ), "no gen_only fill gate"
    assert (
        driver.role_env(BASE, plan, "SERVER", gpus=None, tag="t", ucx={})["TRTLLM_PERF_DRIVER_TAG"]
        == "t"
    )


ARM_ENV = {"TRTLLM_PERF_ARM": "E4-chain", "TRTLLM_TEST_OPENENGINE_CHAIN": "trtllm"}


def _main(driver, tmp_path, test_id, env, *extra):
    argv = ["--output-dir", str(tmp_path / "out"), "--config-dir", str(tmp_path), *extra, test_id]
    return driver.main(argv, environ=env)


@pytest.mark.parametrize("mode, gpus", [("ctx_only", ["4"]), ("gen_only_no_context", ["3"])])
def test_main_dry_run_plans_one_pytest(driver, tmp_path, mode, gpus):
    path = _yaml(tmp_path, ctx_tp=1, gen_tp=1)
    test_id = f"aggr-{mode}-{path.stem}"
    env = dict(os.environ, **ARM_ENV, CUDA_VISIBLE_DEVICES="3,4,5", SLURM_CLUSTER_NAME="x")
    assert _main(driver, tmp_path, test_id, env, "--dry-run") == 0
    report = json.loads(
        (tmp_path / "out" / "single-node" / test_id / "single-node-driver.json").read_text()
    )
    assert report["mode"] == mode and report["status"] == "planned"
    assert [(r["role"], r["gpus"]) for r in report["roles"]] == [("SERVER", gpus)]
    assert report["roles"][0]["cmd"][-5].endswith(f"test_e2e[{test_id}]")
    assert "DISAGG_SERVING_TYPE" in report["environment"]["SERVER"]["removed"] or (
        "DISAGG_SERVING_TYPE" not in env
    )


def test_main_refuses_an_unknown_mode(driver, tmp_path, capfd):
    path = _yaml(tmp_path)
    env = dict(os.environ, **ARM_ENV)
    assert _main(driver, tmp_path, f"aggr-ctx_first-{path.stem}", env, "--dry-run") == 2
    assert "aggr modes" in capfd.readouterr().err
    assert not (tmp_path / "out" / "single-node").exists()


def test_main_runs_the_one_pytest(driver, tmp_path, monkeypatch):
    path = _yaml(tmp_path, ctx_tp=1, gen_tp=1)
    test_id = f"aggr-ctx_only-{path.stem}"
    seen = []

    def command(python, plan, role, output_dir, run_dir):
        seen.append(role)
        return [
            python, "-c",
            "import os, sys; sys.exit(0 if 'DISAGG_SERVING_TYPE' not in os.environ else 3)",
        ]  # fmt: skip

    monkeypatch.setattr(driver, "pytest_command", command)
    monkeypatch.setattr(driver, "SETTLE_S", 0.1)
    env = dict(os.environ, **ARM_ENV, CUDA_VISIBLE_DEVICES="0,1", DISAGG_SERVING_TYPE="CTX_0")
    assert _main(driver, tmp_path, test_id, env) == 0
    report = json.loads(
        (tmp_path / "out" / "single-node" / test_id / "single-node-driver.json").read_text()
    )
    assert seen == ["SERVER"] and report["status"] == "passed"
    assert [(r["role"], r["returncode"]) for r in report["roles"]] == [("SERVER", 0)]
