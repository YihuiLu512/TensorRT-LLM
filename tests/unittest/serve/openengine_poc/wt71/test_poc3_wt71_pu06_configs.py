# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The PU-06 configurations (case-03, layout N-0) under the single-node driver and the E3 arms."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[5]
DRIVER = REPO / "jenkins" / "scripts" / "perf" / "local" / "single_node_disagg"
CONFIGS = REPO / "tests" / "integration" / "defs" / "openengine_chain" / "records" / "WT-71"
STEM = "gb300_deepseek-v41-flash-nvfp4-dspark5_8k1k_con{}_ctx1_dep2_gen1_tp2_ccb-NIXL"
# concurrency -> requests (concurrency x multi_round)
REQUESTS = {1: 50, 8: 80, 32: 160}
# The keys a ds context and generation instance of N-0 set differently.
ROLE_KEYS = {"enable_attention_dp", "attention_dp_config", "kv_cache_config"}


@pytest.fixture(scope="module")
def driver():
    name = "wt71_single_node_driver_under_test"
    spec = importlib.util.spec_from_file_location(name, DRIVER / "run_single_node_disagg.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_the_directory_holds_the_three_configs():
    assert sorted(p.name for p in CONFIGS.iterdir()) == sorted(
        f"{STEM.format(c)}.yaml" for c in REQUESTS
    )


@pytest.mark.parametrize("concurrency", sorted(REQUESTS))
def test_pu06_config(driver, arms, concurrency):
    stem = STEM.format(concurrency)
    plan = driver.load_plan(f"disagg-e2e-{stem}", str(CONFIGS))
    assert (plan.num_ctx, plan.num_gen, dict(plan.gpus), plan.concurrency) == (
        1,
        1,
        {"ctx": 2, "gen": 2},
        concurrency,
    )
    assert plan.supported_gpus == ("GB300",)
    assert driver.assign_gpus(plan, ["0", "1", "2", "3"]) == {
        "GEN_0": ["0", "1"],
        "CTX_0": ["2", "3"],
    }
    config = yaml.safe_load((CONFIGS / f"{stem}.yaml").read_text())
    bench = config["benchmark"]
    assert concurrency * bench["multi_round"] == REQUESTS[concurrency]
    assert (bench["input_length"], bench["output_length"], bench["dataset_file"]) == (
        8192,
        1024,
        "",
    )
    assert config["hardware"]["gpus_per_node"] == 4
    ctx, gen = config["worker_config"]["ctx"], config["worker_config"]["gen"]
    for key in set(ctx) | set(gen):
        if key not in ROLE_KEYS:
            assert ctx.get(key) == gen.get(key), key
    assert (ctx["enable_attention_dp"], gen["enable_attention_dp"]) == (True, False)
    assert (
        ctx["kv_cache_config"]["enable_block_reuse"],
        gen["kv_cache_config"]["enable_block_reuse"],
    ) == (
        True,
        False,
    )
    for name in ("E3-native", "E3-native-al", "E3-mid", "E3-chain", "E3-chain-lean"):
        arms.check_disagg_case(
            arms.DISAGG_ARMS[name],
            runtime="multi_node_disagg_server",
            benchmark_mode=bench["mode"],
            time_breakdown=False,
            numa_bind=config["slurm"]["numa_bind"],
            dwdp_size=0,
            worker_gpus=dict(plan.gpus),
            gpus_per_node=config["hardware"]["gpus_per_node"],
        )
