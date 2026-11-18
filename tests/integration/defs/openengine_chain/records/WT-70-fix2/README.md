<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WT-70-fix2: PU-04 gen-only modes with a warmup pass

This fix sends a warmup pass before the measured run of the two gen-only modes of PU-04 under the case-04 arms,
measures both modes again, and locates the first-token delay of the chain arm that PB-1b (WT-72) and WT-70-fix1
reported. This file is a summary; the raw results and job scripts stay in the local evidence directory named in the
work item's STATUS file and in appendix BG of the PoC-S3 STATUS file.

## What the harness change does

- Under a disaggregated arm (`E4-base`, `E4-chain`, ...), `gen_only` and `gen_only_no_context` get the warmup pass the
  other modes already had: one wave of requests through the server the client calls, before the measured window.
  The pass is in neither the client's output, the arm's readings nor the device step window; with more than one
  client the previous client's window is closed before the next client's pass. Without an arm nothing changes, and
  `TRTLLM_PERF_WARMUP_PASS=0` still turns it off. In `gen_only` the pass opens the generation worker's benchmark fill
  gate, which opens once, so the measured requests then start on arrival instead of all together.
- The readings of a case-04 in-process server also count the post-processing workers its engine forks; they run the
  engine leader's command line in the engine's process tree and read as 0 before (`postproc_workers.started` of
  `E4-base` is now 4).
- Each run's result notes how the front end's CPU time splits over its processes.
- The single-node driver gives each worker role a torch profiler trace path of its own when
  `TLLM_TORCH_PROFILE_TRACE` is set.

## Measurements

One node, 8 B200 GPUs held alone, GLM-5.3-Flash FP8 (PL-10 single-node configuration), concurrency 16, ISL 8k /
OSL 1k. Three rounds per point, arms paired within a round, arm order rotated. Noise is the coefficient of variation
(CV) of `E4-base` over the rounds; a paired median difference is beyond noise when its absolute value is more than
2 x CV. All 18 shots exited 0 and the chain proof of every `E4-chain` shot holds. `gen_only` ran both ways in the
same job: without a warmup pass (as in WT-72, the fill gate waits for all 16 requests) and with one (the measured
requests are not held by the gate, so this variant is not comparable with WT-72's `gen_only`).

| Point | Metric | E4-base (3 rounds) | CV | E4-chain - E4-base (median) |
|---|---|---|---|---|
| gen_only_no_context, warm | TTFT median | 65.0 / 70.0 / 66.7 ms | 3.7% | +42.5% (about +28 ms) |
| gen_only_no_context, warm | E2EL median | 4469 / 4375 / 4430 ms | 1.1% | -0.8% (within noise) |
| gen_only, warm | TTFT median | 1521 / 1712 / 1526 ms | 6.9% | +1.5% (within noise) |
| gen_only, warm | device step median | 14.3 / 14.2 / 14.2 ms | 0.6% | +0.3% (within noise) |
| gen_only, cold | TTFT median | 3486 / 2748 / 3252 ms | 11.9% | +26.3% (+620 to +920 ms) |
| gen_only, cold | E2EL median | 8014 / 7220 / 7671 ms | 5.2% | +11.0% |

- The warmup pass removes most of the chain's first-token difference: in `gen_only_no_context` it drops from +814%
  (WT-70-fix1, no warmup) to about 28 ms, and warm `gen_only` shows no difference beyond noise. TPOT, inter-token
  latency, throughput and step time are within noise at every point.
- The 72 to 79 CPU seconds of the chain's generation front end seen without a warmup are its four post-processing
  workers importing after a `spawn` start (about 22 to 27 CPU seconds each, within 20 s of starting); they only show
  up when a measurement starts that early. After the warmup the chain front end uses 0.25 to 0.41 CPU seconds per run.
- The cold `gen_only` difference sits in the chain context engine's first context iteration (1511 ms against
  725 ms, later iterations equal). That engine runs with a context-only KV cache (`TRTLLM_DISAGG_ROLE=context`), so
  its start-up attention warmup forward is skipped for this hybrid model and the first real request absorbs the
  first-touch costs (FlashInfer RMSNorm first calls, memory pool growth, a torch.compile re-trace, the first GEMM).
  The base context server does not get that variable, keeps the full cache and runs the warmup. This was read from
  logs and profiler traces; an ablation has not been run.
