# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Names the harness shares with the serving front end, the engines and job scripts."""

# Selects the chain: unset or empty runs the single in-process server.
ENV_CHAIN = "TRTLLM_TEST_OPENENGINE_CHAIN"
CHAIN_NONE = "none"
CHAIN_TRTLLM = "trtllm"
CHAIN_GLM = "glm"
CHAINS = (CHAIN_TRTLLM, CHAIN_GLM)

# Command prefix of the glm engine; required when the chain is "glm".
ENV_GLM_SERVE_CMD = "TRTLLM_TEST_GLM_SERVE_CMD"

# Where the harness writes the inner pytest junit files and per-process logs.
ENV_RESULTS_DIR = "TRTLLM_OPENENGINE_RESULTS_DIR"

# The three names below repeat tensorrt_llm/grpc/openengine_conventions.py, so the
# harness imports without TensorRT-LLM; a WT-01 unit test checks that they match.
# Directory where each front-end process writes <pid>.json with its counter.
ENV_COUNTER_DIR = "TRTLLM_OPENENGINE_COUNTER_DIR"
# Field of <pid>.json and of /server_info's "engine" block.
FRONTEND_COUNTER_FIELD = "generate_sent"
# Key of the engine's GetLoad().attributes.
ENGINE_COUNTER_KEY = "trtllm.counters.generate_received"

# Module name of the pytest plugin; the harness refuses to start a chain without it.
PLUGIN_MODULE = "openengine_chain.pytest_plugin"
