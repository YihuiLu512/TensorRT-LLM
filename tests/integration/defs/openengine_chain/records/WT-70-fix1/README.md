<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WT-70-fix1: PU-04 in the ctx_only and gen_only_no_context modes

This fix lets the case-04 arms `E4-base` and `E4-chain` (and the single-node driver) run the two one-server
benchmark modes of PU-04, then measures them as a supplement to PB-1b (WT-72), which could only run `gen_only`.
This file is a summary; the raw results and job scripts stay in the local evidence directory named in the work
item's STATUS file and in appendix BE of the PoC-S3 STATUS file.

## What the harness change does

- `ctx_only` (runtime `aggr_server`): the context configuration runs as one aggregated server. Under a case-04 arm
  the server is the arm's context role (`E4-chain:ctx`: one engine and one worker front end, no `--server_role`), so
  the chain is proven the aggregated way (counted calls, `generate_sent == generate_received`).
- `gen_only_no_context` (runtime `gen_only_no_context_server`): one generation worker behind a disaggregated server
  with no context worker, `TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1`. Under `E4-chain` the worker starts as the arm's
  generation role (`--server_role generation`), and `expectations/WT-70-fix1.py` registers `single_side` for the
  trtllm chain, so only the generation side must take calls.
- Only case-04 arms that start both roles alike take these modes; `E4-ctx`, `E4-gen` and case-03 arms are refused,
  as are unknown modes and a mode on the wrong runtime.
- The single-node driver accepts `aggr-<ctx_only|gen_only_no_context>-<config>`, runs one pytest on the GPUs of the
  matching role in the e2e layout (context GPUs 4-7, generation GPUs 0-3) and sets the generation-only variable.
- Result documents name the arm `<disagg arm>:<role>`; a background pump copies the engine's step-time lines into
  the generation log the step-time parser reads.

## Measurements

One node, 8 B200 GPUs held alone, GLM-5.3-Flash FP8 (PL-10 single-node configuration), concurrency 16. Three rounds
per mode, arms paired within a round, arm order rotated. Noise is the coefficient of variation (CV) of `E4-base` over
the rounds; a paired median difference is beyond noise when its absolute value is more than 2 x CV. All 12 shots
exited 0; the chain proof of all 6 `E4-chain` shots is `ok` (and stays `ok` when the recorded counts are judged
again with the `single_side` rule of WT-38-fix2).

| Mode | Metric | E4-base (3 rounds) | CV | E4-chain - E4-base (median) |
|---|---|---|---|---|
| ctx_only | TTFT median | 2495.6 / 2498.3 / 2495.2 ms | 0.07% | +0.09% (within noise) |
| ctx_only | output tok/s | 6.37 / 6.37 / 6.37 | 0.00% | -0.31% |
| gen_only_no_context | E2EL median | 4613.6 / 4680.2 / 4657.6 ms | 0.73% | -1.95% |
| gen_only_no_context | output tok/s | 3520.5 / 3469.2 / 3489.8 | 0.74% | +1.80% |
| gen_only_no_context | device step median | 14.57 / 14.80 / 14.69 ms | 0.81% | -2.64% |
| gen_only_no_context | TTFT median | 196.3 / 202.4 / 194.8 ms | 2.03% | +814% |

- `ctx_only`: no visible difference; the small differences of TTFT mean and throughput (about 6 ms in 2.29 s) are
  beyond a noise of 0.05% but negligible.
- `gen_only_no_context`: throughput, end-to-end latency and step time of the chain are not worse than the base.
  Its first token arrives 0.86 to 1.72 s later while the end-to-end time is unchanged, so TPOT (derived from
  E2EL - TTFT) drops; and its generation worker front end uses 72 to 78 CPU seconds per client run (about 4.6 ms per
  output token), while the same front end used about 1.7 CPU seconds for a whole e2e run of 81 requests (about
  21 us per token). Both are unexplained and left to a separate investigation.
