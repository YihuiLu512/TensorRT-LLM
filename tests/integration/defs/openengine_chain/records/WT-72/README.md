<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WT-72 (PB-1b): case-04 chain against today's TRT-LLM disaggregated deployment

This work item changes no product code. It measures the case-04 chain (`E4-chain`: TRT-LLM orchestrator, worker
front ends, gRPC, TRT-LLM OpenEngine servers) against today's TRT-LLM disaggregated deployment (`E4-base`: the same
orchestrator with in-process `trtllm-serve` workers) on one node, and attributes the differences to the context or
generation leg. This file is a summary; the raw results, job scripts and cluster details stay in the local evidence
directory named in the work item's STATUS file and in appendix BC of the PoC-S3 STATUS file.
Seven-digit numbers in parentheses are batch job ids.

## Setup

- TRT-LLM: the stage03 integration branch at `ee3c5d17ba7644c2cde10c674661bcec04e8705c`. PoC image
  `e17503-9fd3822a00-x86-sm100-cv2.ready.sqsh` (its C++ is unchanged against that commit); the 41 `tensorrt_llm`
  Python files the branch changes relative to the image's commit are copied over the installed package.
- Single node: 8 B200 GPUs, the job holds the node alone. GLM-5.3-Flash FP8; context TEP4 on GPUs 4-7, generation
  TEP4 on GPUs 0-3; KV transfer NIXL; MTP 3 with the accepted draft length pinned to 2.384 on generation; block
  reuse off; random token ids through `/v1/completions`. Configuration: PL-10's B200 files changed for one node
  (`numa_bind: false`, `TRTLLM_DISAGG_NO_RETRY=1` on both arms); points with a concurrency above 32 raise the
  generation `max_batch_size` and CUDA graph sizes to the concurrency.
- Points (ISL/OSL, concurrency, requests): PU-03 8192/1024 con16 (80); PU-05 con1 (50), con8 (40), con64 (320),
  con256 (1280); the TTFT point 13312/33 con8 (160, the shape of PoC-S2's U4); PU-04 `gen_only` 8192/1024 con16
  (16). All run through WT-70's single-node driver.
- One round is one `E4-base` and one `E4-chain` shot, each starting its own servers; base first in odd rounds,
  chain first in even rounds; differences are paired within a round (chain minus base) and the table gives the
  median over rounds. Noise is the coefficient of variation (CV) of `E4-base` over the rounds (across jobs); a
  difference is "beyond noise" when its absolute value is more than 2 x CV, else "within noise, resolution X%"
  with X = 2 x CV. Points with a beyond-noise verdict after three rounds got more rounds, up to nine.
- Alignment read from the result files: both arms have one worker front end per role configured and started,
  `stream_interval` 20 and 4 post-processing workers on generation, the same NIXL version; `E4-base` has no
  chain proof. The `E4-base` result reports 0 started post-processing workers although its generation log shows 4
  feed-in queues: a reading problem of the harness, not a difference between the arms.
- All 90 shots of the formal points exited 0; the chain proof of the 45 `E4-chain` shots is `ok` with no orphan
  (one handoff per request plus the connection warm-up request; `gen_only` has no warm-up request). One job
  (6069031) failed on a node with a broken GPU before any measurement and was rerun elsewhere (6071402).

## Paired differences

Jobs: con16 (6067726, 6070052, 6071659), con8 (6067727, 6070053, 6071660), con1 (6069028), con64 (6069029),
con256 (6069030), TTFT point (6071402), `gen_only` (6069032, 6071422). Median chain minus base in percent (in ms
for latencies); bold = beyond noise; the base column is the median of `E4-base` over the rounds.

| Point | Rounds | Metric | Base | Base CV | Chain - base |
|---|---|---|---|---|---|
| con16 | 9 | TTFT median | 397.9 ms | 18.1% | -1.09% (-3.3 ms) |
| con16 | 9 | TTFT mean | 592.8 ms | 4.7% | +1.12% (+6.7 ms) |
| con16 | 9 | TTFT p99 | 2503 ms | 0.38% | **+1.24% (+30.9 ms)** |
| con16 | 9 | TPOT median | 4.29 ms | 0.69% | +0.47% |
| con16 | 9 | E2EL median | 4762 ms | 0.81% | +0.24% |
| con16 | 9 | output tok/s | 3173 | 0.62% | -0.33% |
| con16 | 9 | gen device step median | 14.5 ms | 0.58% | +0.57% |
| con8 | 9 | TTFT median | 257.2 ms | 8.3% | +5.83% (+14.8 ms) |
| con8 | 9 | TTFT mean | 401.1 ms | 2.6% | **+7.00% (+28.0 ms)** |
| con8 | 9 | TTFT p99 | 1291 ms | 1.9% | +1.84% (+23.6 ms) |
| con8 | 9 | TPOT median | 3.57 ms | 1.4% | +0.86% |
| con8 | 9 | E2EL median | 3974 ms | 1.6% | +0.44% |
| con8 | 9 | output tok/s | 1970 | 1.3% | -0.58% |
| con8 | 9 | gen device step median | 12.05 ms | 1.2% | +1.20% |
| con1 | 3 | TTFT median | 203.5 ms | 0.30% | **+6.17% (+12.6 ms)** |
| con1 | 3 | TTFT mean | 232.7 ms | 1.3% | **+6.76% (+16.0 ms)** |
| con1 | 3 | E2EL median | 2311 ms | 0.19% | **+0.58% (+13.5 ms)** |
| con1 | 3 | output tok/s | 439.3 | 0.29% | -0.51% |
| con64 | 3 | TTFT p99 | 9688 ms | 0.13% | **+0.60% (+57.8 ms)** |
| con64 | 3 | output tok/s | 5949 | 0.26% | +0.10% |
| con256 | 3 | TTFT p99 | 38370 ms | 0.15% | **+0.63% (+241.9 ms)** |
| con256 | 3 | TPOT median | 6.38 ms | 1.4% | +1.57% |
| con256 | 3 | output tok/s | 6398 | 0.06% | **-0.22%** |
| 13312/33 con8 | 3 | TTFT median | 2066 ms | 0.09% | **+0.32% (+6.6 ms)** |
| 13312/33 con8 | 3 | TTFT mean | 2024 ms | 0.17% | **+0.38% (+7.7 ms)** |
| 13312/33 con8 | 3 | TTFT p99 | 2112 ms | 0.96% | **+10.91% (+230 ms)** |
| 13312/33 con8 | 3 | E2EL median | 2136 ms | 0.07% | **+0.35% (+7.5 ms)** |
| 13312/33 con8 | 3 | output tok/s | 123.3 | 0.17% | **-0.37%** |
| `gen_only` con16 | 6 | TTFT median | 3205 ms | 8.5% | +14.12% (+433 ms) |
| `gen_only` con16 | 6 | gen device step median | 14.31 ms | 1.7% | -0.39% |

- The other metrics of con64 and con256 (TTFT median and mean, TPOT, E2EL, generation step) are within noise; at
  con256 the E2EL mean (+0.23%) is beyond its 0.06-0.07% noise as well. At the TTFT point the TPOT mean is +0.92%
  and the ITL median +2.7%, both beyond noise.
- con16 and con8 are the AC3-P2 points: at con16 only the TTFT p99 is beyond noise (positive in 8 of 9 rounds); at
  con8 only the TTFT mean, whose per-round differences are bimodal (+7 to +13% in five rounds, -1.5 to +2.1% in
  four). The con8 TTFT p99 was beyond noise after six rounds (+2.25%) and is within noise after nine. The TTFT
  median is not resolved at either point after nine rounds: the first wave is quantized by context iterations, so
  the base CV is large by construction.
- `gen_only`: the difference is positive in all six rounds (+0.4% to +64%) but within the 8.5% noise; the point
  stopped at six rounds because its purpose (the generation step is the same on both arms) is met.
- Engine batches: the number and composition of context iterations and the generation step are the same on both
  arms at the median of the rounds (for example con64: 162 context iterations, 159 of them with two requests; TTFT
  point: 161, all with one request). The chain does not change the batch shape systematically.
- TPOT (PT-2, at most 2%): met at every point.

## Attribution to the legs

Three attribution jobs (6070151: con16 and con8, two rounds each; 6071423: con1 and `gen_only`, two rounds each;
6071424: con256, one round) ran both arms with per-request performance records on: the orchestrator's
`perf_metrics_output_dir`, the workers' `return_perf_metrics`, and on the chain the OpenEngine server's leg
durations. Their absolute values are not compared with the formal points, only the segments between arms. For each
request, `ctx_wrap` is the orchestrator's context leg (worker received to first token) minus the context engine's
own arrival-to-first-token time, i.e. the layer around the engine (on the base the in-process HTTP server, on the
chain the worker front end, gRPC and the `PrefillReady` path); `gen_wrap` is the same for the generation leg;
`handoff` is the orchestrator's hop from the end of the context leg to the generation worker.

- **A fixed cost per request of about 9-13 ms, half on each leg's worker front end and gRPC.** Medians: `ctx_wrap`
  5.8-6.6 ms on the chain against 1.6-2.0 ms on the base (+4 to +4.6 ms); `gen_wrap` +4.3 to +5 ms on the chain;
  the orchestrator's entry, `handoff` (6-9 ms) and return are the same on both arms. The first-output probe (CP-09)
  on the formal points gives about 1.1 ms on both legs from the servicer sending the first output to the front end
  receiving it. At con1 the orchestrator's TTFT is +12.9 ms on the chain, matching the formal +12.6 ms. This cost
  explains the beyond-noise differences at con1 and at the TTFT point.
- **Serialization of the first wave on the context front end.** When `concurrency` requests arrive at once, the
  chain's `ctx_wrap` grows with the position in the wave: its mean over the first wave is 23-25 ms at con16 and
  15-16 ms at con8 against about 2 ms on the base; at con256 it is 273 ms (the slowest requests 285-287 ms) against
  2-3 ms, about 1.1 ms per request in series. Context queueing, `handoff` and the generation leg are the same on
  both arms. This puts the TTFT p99 differences of every concurrency, and the small throughput loss at con256, on
  the context leg. That a late request can miss a context iteration boundary and wait a whole iteration (the TTFT
  point: +230 ms against 267 ms per iteration, in two of three rounds) is an inference, not measured per request.
- **The con8 TTFT mean follows the context batching of each round.** Per round, the chain-minus-base difference of
  the TTFT mean correlates with the difference in the number of context iterations that hold two or more requests
  (Pearson 0.94 at con8, 0.92 at con16). At con8 the chain had more such iterations in five of nine rounds (TTFT
  mean +7 to +13%), the same or fewer in four (-1.5 to +2.1%); at con16 the direction is mixed and the median is
  near zero. A reading (not tested): the closed-loop client's arrival phase decides whether two requests share a
  context iteration, and the chain's per-request cost shifts that phase. Whether the chain causes more batching
  systematically is not shown.
- **`gen_only`: the first context iteration on the chain is slow.** The difference lies in the context engine on
  the context leg: the second request's queueing time, which equals the first context iteration, is 896 / 1079 ms
  on the chain against 277 / 277 ms on the base, and later iterations take about 310 ms on both; `ctx_wrap` adds
  about 26 ms, and the generation leg is shorter on the chain. The `gen_only` mode sends no warm-up request (16
  records for 16 requests); that this is a cold start of the chain's context engine is an inference.
- Not located: which step inside the worker front end and gRPC costs the milliseconds above, and what the slow
  first context iteration of `gen_only` does. Next measurements: per-stage timing in the worker front end or
  sampling its main thread; for `gen_only`, a warm-up request before measuring, or a timeline of the first
  iteration.

## The PB-0 late first token under disaggregation (AC3-P5)

In PoC-S2's U4 (aggregated serving, 13312/33 con8) the chain's first token was 181-357 ms later, traced to a
changed context schedule inside the engine. At the same shape under disaggregation the chain's TTFT median is only
+6.6 ms (+0.32%), and the context iterations (161, each with one request) and the generation step are the same on
both arms: the late first token of aggregated serving does not appear. A reading (not tested): with one 13k-token
request per context iteration, the context engine's schedule no longer depends on the arrival phase. The remaining
+230 ms TTFT p99 is the first-wave effect above. The con8 batching effect at 8192/1024 is of the same kind as PB-0 but
an order of magnitude smaller (+28 ms).

For PT-3 (TTFT at most one engine iteration, UV-45) the reference matters: against a context iteration (230-310 ms)
every point meets it; against a generation step (6.9 ms at con1, 14.4 ms at con16) the con1 difference (+12.6 ms)
and the con8 TTFT mean (+28 ms) do not. Under disaggregation the context iteration is the suggested reference.

## Not run

- PU-04 `ctx_only` and `gen_only_no_context`: not supported by the performance harness at this commit; WT-70-fix1
  added them and records their results separately.
- The single-side arms (`E4-ctx`, `E4-gen`), the lean arm and `E4-direct`: the attribution above uses the
  lower-cost means of the plan instead.
- Cross-node KV transfer (one node only; PB-3 is not part of this work item).
- Per-request client timestamps: the benchmark client does not keep them, so the first-wave grouping comes from the
  orchestrator's records in the attribution jobs only.

## Summary

- At the AC3-P2 points (con16, con8; nine rounds each) the chain's TPOT, end-to-end latency, throughput and
  generation step are within the noise of today's disaggregated deployment (differences of at most 1.2%, resolution
  1.2-3.2%). The chain's first token is later in the tail at con16 (p99 +31 ms) and on average at con8 (mean
  +28 ms, in five of nine rounds).
- The chain costs about 9-13 ms per request, half on the context leg's and half on the generation leg's worker front
  end and gRPC; the orchestrator adds nothing measurable, and the engines' batch shape and step times are
  unchanged. Requests arriving together are serialized on the context front end (about 1.1 ms each), which
  lengthens the TTFT tail at every concurrency.
- PoC-S2's late first token at 13312/33 con8 does not appear under disaggregation (+6.6 ms).
