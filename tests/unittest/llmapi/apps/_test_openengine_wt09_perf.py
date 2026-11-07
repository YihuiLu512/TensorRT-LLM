# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One measured perf run of arm B-s1 on the OpenEngine chain writes a complete result.

Runs under ``openengine_chain/runner.py --chain trtllm`` (the regression smoke
set), which puts the harness on ``PYTHONPATH`` and loads its plugin.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_common.http_utils import wait_for_endpoint_ready
from utils.llm_data import llm_models_root

_ARMS_PATH = Path(__file__).resolve().parents[3] / "integration/defs/perf/openengine_arms.py"
NUM_PROMPTS = 16
OUTPUT_LEN = 32
READY_TIMEOUT_S = 1800


@pytest.fixture(scope="module")
def arms():
    spec = importlib.util.spec_from_file_location("openengine_arms_smoke", _ARMS_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look the module up while the class bodies run.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        del sys.modules[spec.name]


def test_perf_arm_b_s1(arms, tmp_path):
    if os.environ.get(arms.ENV_CHAIN) != arms.CHAIN_TRTLLM:
        pytest.skip(f"needs {arms.ENV_CHAIN}=trtllm (openengine_chain/runner.py --chain trtllm)")
    arm = arms.ARMS["B-s1"]
    model = str(llm_models_root() / "Qwen3" / "Qwen3-0.6B")
    config = tmp_path / "extra.yml"
    config.write_text(
        "max_batch_size: 8\nkv_cache_config:\n  free_gpu_memory_fraction: 0.3\n"
        "  enable_block_reuse: false\n"
    )
    port = arms.free_port("localhost")
    cmd = ["trtllm-serve", model, "--backend", "pytorch", "--config", str(config)]
    cmd += ["--host", "localhost", "--port", str(port)]
    plan = arms.prepare_server(cmd, arm, str(tmp_path))
    log_path = tmp_path / "server.log"
    with open(log_path, "w") as log:
        server = arms.launch(plan, arm, env=os.environ, sink=log, ready_timeout=READY_TIMEOUT_S)
        try:
            wait_for_endpoint_ready(
                f"http://localhost:{port}/health",
                timeout=READY_TIMEOUT_S,
                check_files=[str(log_path)],
                server_proc=server,
            )
            client_cmd = [
                sys.executable, "-m", "tensorrt_llm.serve.scripts.benchmark_serving",
                "--model", model, "--tokenizer", model,
                "--num-prompts", str(NUM_PROMPTS), "--max-concurrency", "4",
                "--dataset-name", "random", "--random-ids", "--tokenize-on-client",
                "--random-input-len", "128", "--random-output-len", str(OUTPUT_LEN),
                "--random-range-ratio", "0.0", "--ignore-eos", "--no-test-input",
                "--percentile-metrics", "ttft,tpot,itl,e2el",
                "--host", "localhost", "--port", str(port),
            ]  # fmt: skip
            run = server.start_run()
            output = subprocess.run(
                client_cmd, capture_output=True, text=True, check=True, timeout=600
            ).stdout
            path = run.finish(
                output=output,
                client_cmd=client_cmd,
                test_id=arms.current_test_id(),
                default_dir=str(tmp_path),
            )
        finally:
            server.terminate()
            server.wait()
    doc = json.loads(Path(path).read_text())
    print(json.dumps(doc, indent=1))
    assert arms.validate_result(doc) == []
    assert (doc["case"], doc["arm"], doc["dataset"]) == ("case-02", "B-s1", "token_ids")
    assert doc["output_tokens"] == NUM_PROMPTS * OUTPUT_LEN
    for name in ("ttft_ms", "tpot_ms", "itl_ms", "e2el_ms"):
        assert None not in doc["metrics"][name].values(), name
    assert doc["metrics"]["output_tok_per_s"] > 0
    assert doc["frontend_cpu_s"] > 0 and doc["frontend_cpu_us_per_token"] > 0
    assert doc["chain_proof"] == {
        "generate_sent": NUM_PROMPTS,
        "generate_received": NUM_PROMPTS,
    }, doc["notes"]
    assert doc["alignment"]["num_serve_frontends"] == 1
    assert doc["alignment"]["num_postprocess_workers"] == 0
    assert doc["alignment"]["enable_block_reuse"] is False
