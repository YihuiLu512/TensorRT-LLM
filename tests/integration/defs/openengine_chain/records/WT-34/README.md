<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WT-34 (PB-0): where the chain's later first token is spent

This probe changes no product code in this branch. It answers one question left open by PoC-S2: on load U4, the
chain arms (front end -> gRPC -> engine process) reached the first token later than the in-process arms, while the
end-to-end latency did not change (PoC-S2's readings). Is the extra time spent in the engine or on the gRPC and
front-end path?

Answer: **in the engine.** At concurrency 8, almost all of the extra time lies between the rank-0 worker accepting
the request and the executor emitting its first response. At concurrency 1 the whole difference is +3.5 to +5.4 ms
and lies outside that span. Why the engine takes longer on the chain arms was not measured.

Readings are in `readings.json`. Seven-digit numbers in parentheses are Slurm job ids. The local evidence
directory `/home/scratch.yihlu_coreai/tmp/20261114-wt34-u4probe/` keeps the probe tree (`src/`), the probe
commits as patches (`analysis/probe-patches/`), the job inputs and logs (`runs/`), the per-request tables
(`analysis/a1/`) and the script that derives `readings.json` from them (`analysis/make_records_readings.py`);
cluster details are not repeated here.

## Setup

- Model and load: GLM-5.3-Flash FP8, TP4 on 4 GPUs of an 8-GPU B200 node held by the job alone, the PoC-S2 U4
  configuration unchanged (ISL 13312 / OSL 33, `max_num_tokens` 16384, chunked prefill, MTP 3,
  `stream_interval` 20). Four clients per step, as in PoC-S2: concurrency 8 and 1, each with text prompts and with
  token-id prompts (`con8_text`, `con1_text`, `con8_tok`, `con1_tok`), 160 / 30 / 160 / 30 measured requests;
  each client's warm-up run is left out.
- Arms, as in PoC-S2: in-process `A-def` (8 front ends) and `A-1f0` (1 front end), chain `B-s1` and `B-full`.
  The engine configuration files of the four arms differ only in the front-end settings (`num_serve_frontends`,
  `num_postprocess_workers`).
- One job per round runs the four arms one after the other in a rotated order: round 1 (5915905), round 2
  (5915906), round 3 (5915909). All 12 steps returned 0 and passed every validity check of the analysis.
- Source: the integration branch at `f1d6fcda8ddcf49fcdfbd4f9d89c671e4cf9b833` plus a temporary timestamp
  patch (three local commits ending at `d89d41117f124e5a0ba0782f6d9af9848edc727c`, never pushed). Every
  `tensorrt_llm` Python file that this tree changes relative to the image's commit (40 files: the branch's own
  changes and the patch) was copied over the container's installed package; the image
  `e17503-9fd3822a00-x86-sm100-cv2.ready.sqsh` itself was not changed. All four arms carry the same patch.

### Probe points

Each process appends JSON lines with wall-clock nanoseconds. Points used here (the complete list with their exact
meaning is `points` in `readings.json`), in path order:

| Point | Where | Arms |
|---|---|---|
| `f_call`, `s_rpc` | front end about to issue the gRPC call; servicer entered | chain |
| `w_submit` | rank-0 worker `submit()` entered | all |
| `e_activate`, `e_first_ctx`, `e_last_ctx` | executor activates the request; first / last schedule with a context chunk of it | all |
| `e_first_rsp` | executor about to enqueue the request's first response | all |
| `s_first_yield`, `f_first_token` | servicer about to yield the first token; front end receives it | chain |
| `ttft` | client first token (the PoC-S2 per-request client log) | all |

A client request is joined to the front end's record when exactly one record with the same prompt hash entered
the front end between 0.5 s before and 2 s after the client sent it. Every step joined all 380 measured requests,
with no ambiguous match. A step counts only if every probe point of its arm is present on at least 95% of its
successful requests; all 12 steps had 100%.

## Result

Paired differences, chain arm minus in-process arm in the same job: medians per arm and round, then the median
over the three rounds; requests after the first wave (ms). Each column is a median of its own, so the columns do
not add up exactly.

| Client | Pair | TTFT | to engine | engine | from engine |
|---|---|---|---|---|---|
| con8_text | B-s1 - A-1f0 | +303.4 | +3.3 | +295.3 | +1.1 |
| con8_text | B-full - A-def | +380.1 | +3.7 | +378.8 | +1.8 |
| con8_tok | B-s1 - A-1f0 | +207.1 | +3.4 | +203.2 | +1.1 |
| con8_tok | B-full - A-def | +236.5 | +4.5 | +232.1 | +1.8 |
| con1_text | B-s1 - A-1f0 | +3.5 | +2.4 | +0.0 | +1.1 |
| con1_text | B-full - A-def | +4.9 | +3.0 | -0.2 | +1.6 |
| con1_tok | B-s1 - A-1f0 | +4.2 | +3.2 | -0.3 | +1.2 |
| con1_tok | B-full - A-def | +5.4 | +3.5 | -0.1 | +1.6 |

"To engine" is client send -> `w_submit`, "engine" is `w_submit` -> `e_first_rsp`, "from engine" is
`e_first_rsp` -> client first token. "To engine" plus "from engine" is 3.5-6.3 ms per row.

- PoC-S2's own analysis of the same load (without this probe) gave con8_text +269 (B-s1) / +357 (B-full),
  con8_tok +181 / +182 and con1 +3 to +5 ms. The two analyses do not select requests the same way, so compare the
  sign and the size only. End-to-end latency differs by -16 to +5 ms here (paired, as in the table).
- Per round, the con8 TTFT differences range from +296 to +389 ms (text) and +92 to +309 ms (tok), and in every
  round the engine segment carries 95-100% of the con8 TTFT difference (per-round values in `readings.json`).
- Segments that only the chain arms have, per-round medians of the chain arms over all clients: `f_call` ->
  `s_rpc` 1.0-1.5 ms, `e_first_rsp` -> `s_first_yield` 1.1-1.3 ms, `s_first_yield` -> `f_first_token` 0.8-1.1 ms.
  `f_first_token` -> client first token is 0.4 ms on `B-s1` and 1.6-1.7 ms on `B-full`.

### Inside the engine

Paired differences of the engine points, as in the Result table (ms):

| Client | Pair | `w_submit` | `e_activate` | `e_first_ctx` | `e_last_ctx` | `e_first_rsp` |
|---|---|---|---|---|---|---|
| con8_text | B-s1 - A-1f0 | +3.3 | -1.5 | +39.4 | +349.2 | +302.0 |
| con8_text | B-full - A-def | +3.7 | +8.9 | +60.8 | +358.4 | +384.3 |
| con8_tok | B-s1 - A-1f0 | +3.4 | +40.1 | +43.6 | +62.7 | +205.3 |
| con8_tok | B-full - A-def | +4.5 | +147.0 | +150.4 | +79.8 | +233.5 |

With text prompts the chain requests are activated at about the same time and first scheduled 39-61 ms later,
and the median time from their first to their last context schedule is 303-308 ms against 0 on the in-process
arms (`median_ms_first_to_last_context` in `readings.json`). With token-id prompts the difference is spread over
the wait before activation and the time after the last context schedule. The schedules also look different. All
three rounds; per arm, the first column counts the 456 requests after the first wave and the other three count
the context schedules in the con8 windows (424-477; the last schedule of each window is left out):

| Client | Arm | Requests whose context spans 2+ schedules | Context schedules with 2+ requests | Context tokens per context schedule (median) | Interval after a context schedule (median ms) |
|---|---|---|---|---|---|
| con8_text | A-def / A-1f0 | 42% / 38% | 47% / 44% | 13312 / 13312 | 268.3 / 268.2 |
| con8_text | B-s1 / B-full | 59% / 65% | 68% / 74% | 16249 / 16256 | 307.9 / 309.9 |
| con8_tok | A-def / A-1f0 | 14% / 17% | 16% / 20% | 13312 / 13312 | 243.9 / 245.6 |
| con8_tok | B-s1 / B-full | 33% / 39% | 36% / 45% | 13312 / 13312 | 265.5 / 265.6 |

The interval after a schedule includes any wait for work, so it is not the execution time of the schedule. Not
measured: why requests meet the scheduler in a different pattern on the chain arms. The paired median time to
`w_submit` is only 2.4-4.5 ms later than on the in-process arms, and the scheduler entries of the configuration
are the same.
Open leads for the follow-up: the time the executor loop spends between schedules on the two architectures, and
when final responses reach the clients (they set the next request's send time). With token-id prompts the
in-process arms also show a long interval after decode-only schedules (76-92 ms, against 12.9 ms on the chain
arms); this was not explained either.

### Schedule intervals (reference for PT-3, UV-45)

From the con8 windows, the interval from one schedule to the next. It includes any wait for work, so it is an
upper bound of one engine iteration, not its execution time:

- a schedule with context work: median 244-268 ms on the in-process arms and 266-310 ms on the chain arms;
- a schedule with decode work only: median 11.8-12.9 ms on the chain arms; on the in-process arms 11.9 ms with
  text prompts and 76-92 ms with token-id prompts;
- at concurrency 1, `e_first_ctx` -> `e_first_rsp` is 259-263 ms on all arms (per-round medians).

## Known limits

- The probe hashes the prompt for its join key (median 0.6 ms for 13312 token ids and 0.2 ms for a text prompt
  of similar length, timed on the agent host; `analysis/hash_time.py`). On the chain arms this runs before the
  gRPC call is issued, on the in-process arms after the request is handed to the worker, so it is part of the
  chain's "to engine" difference above.
- The cost of the other probe points was not measured; all four arms carry the same patch, but the chain arms
  have more points on their path.
- Wall-clock timestamps from different processes on one host are compared directly.
- Client labels are assigned by the clients' start order; for all 12 steps they were checked against each
  client's `--max-concurrency` and dataset arguments.
