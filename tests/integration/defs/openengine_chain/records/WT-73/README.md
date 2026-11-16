<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WT-73 (PB-2a): case-03 chain against the ds-engine native deployment

This work item changes no product code. It measures the case-03 chain (`E3-chain`: TRT-LLM orchestrator, worker
front ends, gRPC, ds-engine OpenEngine server) against the ds-engine native deployment (`E3-native-al`) and the
middle arm (`E3-mid`: TRT-LLM orchestrator, native ds-engine workers) on one node, gives the capacity of the
ds-engine OpenEngine server with the direct replay tool (`E3-direct`), and runs the two multi-node smoke tests of
NT-147. This file is a summary; the raw results, job scripts and cluster details stay in the local evidence directory
named in the work item's STATUS file and in appendix AX of the PoC-S3 STATUS file.
Seven-digit numbers in parentheses are Slurm job ids.

## Setup

- TRT-LLM: the stage03 integration branch at `6c3380c8734b8ee66c0ec124e7f8ffe5318b500d`; ds-engine:
  `94005ff481ac708506751304c8fe8ca44ef8be54` with the compiled extension WT-62 and WT-71 used (its C++ is
  unchanged since that build). PoC image `e17503-9b4022dd0d-aarch64-sm103-ds0163a5d9c5.ready.sqsh`; the 22
  `tensorrt_llm` Python files the branch changes relative to the image's commit are copied over the installed
  package.
- Single node: 4 GB300 GPUs, the job holds the node alone. Layout N-0: context DEP2 on GPUs 0-1, generation TP2 on
  GPUs 2-3. Load PU-06: the three configuration files of `records/WT-71/` (DeepSeek-V4.1-Flash NVFP4, DSpark 5
  with the accepted draft length pinned to 2.51, ISL/OSL 8192/1024, random token ids, `/v1/completions`),
  concurrency 1 / 8 / 32 with 50 / 80 / 160 requests, through the single-node driver.
- `E3-native` and `E3-native-al` are one group on N-0 (UV-36: their switch sets are the same; `E3-native-al` only
  writes four values that equal ds-engine's defaults). `E3-native-al` (N) is run for both.
- One job per concurrency runs three rounds of the three arms; every shot starts its own servers; the arm order
  rotates per round (N/M/C, M/C/N, C/N/M); differences are paired within a round. Noise is the coefficient of
  variation (CV) of N over the rounds; a paired median difference is "beyond noise" when its absolute value is more
  than 2 x CV, else "within noise, resolution X%" with X = 2 x CV (`03_perf-tests.md` §4.2 of the plan).
- All 27 PU-06 shots exited 0; the chain proof of the 9 `E3-chain` shots is `ok` (one handoff per request, no
  orphan).

## E3-direct: capacity of the ds-engine OpenEngine server (1911821)

The direct replay tool sends the PU-06 load (8192/1024, random ids below 10000, a different seed per level)
straight to the two OpenEngine servers, without front ends or orchestrator. One shot per level.

| Concurrency | Output tok/s | TTFT median (ms) | TPOT median (ms) | Server main thread, max cores (gen / ctx) |
|---|---|---|---|---|
| 1 | 401 | 193.6 | 2.18 | 0.035 / 0.005 |
| 8 | 1687 | 176.6 | 3.82 | 0.04 / 0.005 |
| 32 | 4704 | 229.0 | 6.18 | 0.095 / 0.015 |
| 64 | 5173 | 6004 (queued) | 5.87 | 0.20 / 0.02 |

The server's asyncio main thread stays at or below 0.2 cores; the busy threads of the process are three engine
threads at about one core each. At the PU-06 levels the OpenEngine server is not the limit; at concurrency 64 the
generation instance's `max_batch_size` of 32 is. The prompts differ from PU-06's, so compare throughput and TPOT
only, not TTFT.

## PU-06 (1909140, 1908941, 1910386)

Paired median differences in percent (three rounds); bold = beyond noise.

| Concurrency | Metric | N CV | C - N | M - N | C - M |
|---|---|---|---|---|---|
| 1 | TTFT median | 4.9% | +8.3% (+11.7 ms) | +0.9% | +6.1% |
| 1 | TPOT mean | 3.6% | -2.8% | +4.2% | **-8.4%** |
| 1 | E2EL mean | 1.6% | **-5.3%** | +3.0% | **-8.3%** |
| 1 | output tok/s | 1.8% | **+5.7%** | -2.9% | **+9.1%** |
| 8 | TTFT median | 49.1% | +22.8% | +72.8% | +8.8% |
| 8 | TTFT mean | 23.6% | +38.4% | +2.5% | **+64.0%** |
| 8 | TPOT mean | 2.2% | **-5.1%** | 0.0% | **-6.5%** |
| 8 | E2EL mean | 3.3% | -2.6% | -1.2% | -1.5% |
| 8 | output tok/s | 3.2% | +0.8% | +1.8% | -1.0% |
| 32 | TTFT median | 11.7% | **+57.4% (+168 ms)** | **+30.5%** | **+46.4%** |
| 32 | TTFT mean | 8.9% | +11.6% | **-24.3%** | **+59.0%** |
| 32 | TPOT median | 2.4% | **+23.2%** | +1.2% | **+22.7%** |
| 32 | ITL median | 0.8% | +0.9% | -0.3% | **+1.7%** |
| 32 | E2EL median | 2.3% | **+26.6%** | +2.8% | **+23.2%** |
| 32 | output tok/s | 5.6% | **-11.6%** | +10.9% | **-20.4%** |

- These are the first three rounds of each concurrency. More rounds (below) do not confirm the beyond-noise
  verdicts at concurrency 8 and 32: at both points the chain's differences are within noise over six to nine
  rounds, except the ones listed there.
- Concurrency 1: the chain is faster end to end (E2EL mean -5.3%, tok/s +5.7%); see the step logs below.
- Startup times are not paired: 124-210 s on the chain, 128-261 s on the middle arm, 149-224 s on the native arm.

## DS-O03: prompt ids on the generation hop

Offline micro-benchmark in the PoC image, 8192 prompt ids, median of 300 repetitions: the gRPC `GenerateRequest`
is 16.3 KB (ids below 10000) / 23.6 KB (ids below 129280), the HTTP JSON body 48-59 KB; the front end parses the
`CompletionRequest` in 0.31 ms and builds and serializes the gRPC request in 0.17-0.19 ms; the ds-engine server
decodes it in 0.02-0.03 ms and validates it in 0.32 ms in all. About 0.8 ms per request, a tenth or less of the
extra time the chain spends between the context instance's `PrefillReady` and the generation instance's first
token (front-end probes: 12-15 / 35-52 / 58-62 ms on the chain against 2.6 / 19.6 / 34 ms on E3-direct at
concurrency 1 / 8 / 32). On `/v1/completions` the chain's `gen_tokids_ctxbytes: true` has no effect: TRT-LLM's
`CompletionResponse` carries no `prompt_token_ids_b64` (as for `E3-mid`, N113), so the generation hop gets the
ids as a JSON array.

## Aligned and not aligned

Aligned (the same value on every arm, written in the result JSON): ds-engine build and extension, worker YAML,
pinned draft acceptance, context router `conversation`, `gen_strip_message_history`, `max_retries` 3 /
`server_keep_alive_timeout` 600, one front end per worker, one orchestrator process, client command and seed,
KV transfer over ds-engine's own fabric. Not aligned: `E3-mid` has `gen_tokids_ctxbytes: false` (N113); the chain's
`gen_tokids_ctxbytes` has no effect on completions (above); the TRT-LLM front end ignores
`enable_tokenization_cache` for `deepseek_v41` while the native worker uses it (UV-17; no effect on token-id
prompts); only the chain has the first-output probes; ds-engine reports no `gen_device_step_ms` on any arm.

## NT-147: multi-node smoke tests (1917058, 1917312)

One `E3-chain` shot each, the PU-06 concurrency-8 file with only the generation instance's
`tensor_parallel_size` changed, one task per node in every worker step (CP-11), each ds-engine worker with
`DSV4_XFER_HOST` set to its node and `DSV4_NODE_RANK` to its node rank in the step.

| Layout | Nodes | Generation | Result |
|---|---|---|---|
| `mini` (DS-S19a, `two_node`) | 2: context DEP2 on node 0, generation on node 1 | TP4 on one node | every role exited 0; chain proof `ok` with `leg_expectation: cross_node`, 81 handoffs, no orphan; 2424 tok/s, TTFT median 160.7 ms, TPOT median 2.96 ms |
| `mini8` (DS-S19b, `gen_multinode`) | 3: context DEP2 on node 0, generation on nodes 1-2 | TP8 over two nodes: node 1 leads, node 2 follows | every role exited 0; the follower logged "multi-node instance: node 1 of 2 (4 GPUs each)" and "Follower node: loading weights; steps come from the leader", loaded ranks 4-7 and was ready; the pytest of both generation nodes passed; chain proof `ok`, `cross_node`, 81 handoffs, no orphan; 2132 tok/s, TTFT median 164.9 ms, TPOT median 3.17 ms |

These are smoke tests: one shot, not compared with a native arm (PU-07 was not run). The KV transfer between the
nodes uses ds-engine's own transfer path (the NIXL settings are accepted and ignored); which interconnect it took
was not checked.

## Concurrency 8: extra rounds (1917062, 1920247)

N's TTFT varied too much over the first three rounds (CV 49% of the median) to judge the chain's later first
token, so six more rounds ran in two more jobs (one `E3-mid` shot of round 6 failed: its orchestrator exited with
code 1 about 20 s after start and left an empty log; not explained, not seen again). Over the nine rounds (the CV
includes the differences between the three jobs):

| Metric | N CV | C - N | M - N | C - M |
|---|---|---|---|---|
| TTFT median | 30.4% | +21.4% | +15.2% | +4.2% |
| TTFT mean | 17.9% | +16.8% (7 of 9 rounds positive) | +8.3% | +6.3% |
| TPOT mean | 6.8% | -4.5% | -0.3% | -3.2% |
| E2EL mean | 7.0% | +0.9% | -0.8% | -1.8% |
| output tok/s | 6.9% | -1.8% | +0.1% | +0.6% |

All within noise. The beyond-noise verdicts of the first three rounds (C - M TTFT mean +64%, C - N TPOT mean
-5.1%) do not hold over nine rounds. At this point the TTFT of the chain cannot be told from the native arm's to
better than about 36%.

## Concurrency 32: repeat (1921964)

The first three rounds showed the chain 11.6% below the native throughput in every round. Three more rounds
without any diagnostics (one job, rotated order) do not repeat it: C - N output tok/s is -2 / +15 / -7%. Over the
six rounds (the CV includes the difference between the two jobs):

| Metric | N CV | C - N | M - N | C - M |
|---|---|---|---|---|
| TTFT median | 27.3% | +37.0% (all six rounds positive, +9 to +98%) | +8.4% | +40.7% |
| TTFT mean | 20.9% | +9.9% | -20.8% | +38.8% |
| TPOT median | 6.4% | +11.3% | +0.6% | **+13.1%** |
| ITL median | 2.9% | +1.1% | -0.7% | +2.1% |
| E2EL median | 5.9% | +10.5% | +2.5% | **+15.1%** |
| output tok/s | 7.7% | -7.5% | +6.0% | -9.7% |

So the throughput loss of the first three rounds does not hold. What remains: the chain's median TTFT is higher
than the native arm's in every one of the six rounds, by more than the round-to-round noise allows to call; and
against the middle arm the chain's median TPOT and E2EL are beyond noise.

## Where the time goes (diagnostics jobs; not measurements)

Three more jobs ran the PU-06 load with ds-engine's own diagnostics, switched on per engine process through a
wrapper named by `TRTLLM_TEST_DS_SERVE_CMD` (no harness or driver change): `DSV4_STEP_LOG` (one line per engine
step) in all three, `DSV4_REQ_LOG` (one line per request) in the second, plus a per-thread CPU sampler of every
process of the run. Their throughput is of the same order as PU-06's (no effect of the logging seen).

- **Step logs, concurrency 32 (1917667, 1920047).** In every shot the generation engine runs a full batch (about 24
  decoding sequences per step on average, about 31 when the long steps happen) on every arm; steps take about
  22.4 ms at the median. What makes a shot slow is a number of long steps (over 30 ms, up to about 220 ms): slow
  shots have 46-73 of them and lose 7-9 s, fast shots 0-16. They occur on every arm, spread over the whole run;
  part of them wait for the step's inputs to be launched or uploaded (`lau` / `upl` of 100-220 ms), the rest run
  longer on the GPUs (`loc`). Their cause was not found. The context engine never queues more than the step
  boundary (its `wait` count stays 0) and is busy 40-54% of the time.
- **Request logs, concurrency 32 (1920047).** Time from the context engine finishing a request to the generation
  engine receiving it, medians per shot: native 3 ms, middle arm 16-17 ms, chain 19-21 ms, in every round (means
  3-15 / 17-28 / 39-67 ms). The TRT-LLM orchestrator adds about 13 ms on this hop over ds-engine's proxy, the
  worker front ends and gRPC about 3-4 ms more. The time a request spends in the context engine (arrival to end)
  grows in the slow shots on every arm. A reading consistent with both logs (not tested): long generation steps
  end several requests at once, the closed-loop client sends their successors at once, and they queue in the
  context engine, raising TTFT and lowering throughput together.
- **CPU, concurrency 32.** On the chain, the generation worker front end uses 0.18 cores on average (main thread at
  most 0.17), the orchestrator 0.09, the OpenEngine server's main thread at most 0.1: nothing on the chain's own
  path is CPU-bound.

- **Step logs, concurrency 1 (1921963).** C - N over these three rounds: TTFT median +8.6% (+9 / +14 / +7%,
  beyond noise; N's CV is 0.9% here), output tok/s +3.9% (within noise). Together with PU-06's first three rounds,
  the chain's median TTFT is 7-14% (about 11 ms) above the native arm's in all six rounds, beyond noise; M - N is
  +1.8% and C - M +6.2% (both within noise, C - M positive in all six rounds), so most of these milliseconds lie on
  the worker front end and gRPC side. The front-end probes of the chain agree: from the context front end receiving
  `PrefillReady` to the generation front end receiving the first token takes 12-15 ms on the chain against 2.6 ms
  between the two OpenEngine servers on E3-direct. The end-to-end advantage of the chain in PU-06's first three
  rounds (E2EL mean -5.3%) is not seen again (C - N E2EL mean -3.6%, within noise).

## Summary

- PU-06 on N-0, six to nine rounds per point (at concurrency 1 three of the six are the step-log rounds): the chain's throughput, TPOT and end-to-end latency are within the
  noise of the native deployment at concurrency 1, 8 and 32 (resolution 7-15%). The chain's first token is later:
  by about 11 ms at concurrency 1 (beyond noise), and in most rounds at 8 and 32 (+17% / +37% at the median of the
  rounds, within noise).
- The fixed cost of the chain's path: the TRT-LLM orchestrator's handoff hop (about 13 ms over ds-engine's proxy at
  concurrency 32) plus the worker front ends and gRPC (about 3-4 ms at 32, most of the 11 ms at 1). The variance
  between shots comes from long steps of the generation engine on every arm (cause not found).
- The ds-engine OpenEngine server is not a limit at these loads; NT-147's two multi-node layouts run the chain.
- Not run: PU-07 (two-node comparison), PB-2b (AgentX load, PU-08).
