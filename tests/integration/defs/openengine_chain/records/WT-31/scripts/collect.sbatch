#!/bin/bash
#SBATCH --job-name=j17874-collect
#SBATCH --output=/home/scratch.yihlu_coreai/tmp/20261113-wt31-r10/results/collect-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# WT-31 (R10): collect the node ids of the R10 test files (no GPU).
# Partition, account and QoS of a CPU queue go on the sbatch command line; LLM_MODELS_ROOT,
# MODELS_MOUNT and LOCAL_CANDIDATES come from the environment (see collect.sh).
exec bash /home/scratch.yihlu_coreai/dev/feat-dev-J17874/tests/integration/defs/openengine_chain/records/WT-31/scripts/collect.sh /home/scratch.yihlu_coreai/tmp/20261113-wt31-r10/results
