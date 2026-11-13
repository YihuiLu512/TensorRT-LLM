<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WT-32 direct handoff probe: readings

`readings.json` holds the readings of the WT-32 probe (TRTLLM-17875), extracted by a
script from the raw results of three jobs on one B200 (context and
generation engines on the same GPU, Qwen3-0.6B, the PoC-S3 image with the branch's
Python overlaid). The raw results and the probe driver are not in the repository.

| Job | Source | What ran |
|---|---|---|
| 5909835 | `a5ba8df9a4` | NT-041 once; probe measurements 1 and 3 to 8 |
| 5910462 | `a5ba8df9a4` | Reference matrix, prompt logprobs with and without a cached prefix, X != Y with a 300 s deadline |
| 5911451 | `e1940769c9` | NT-041 three times; reference matrix with block reuse off on both engines |

Keys of `readings.json`, by measurement:

- `m1_handoff`: context_only, PrefillReady, kv.session, decode. First token, usage,
  bytes sent and received, the resolved transceiver, and the reference matrix.
- `m3_orphans`: KV held by context legs whose generation leg never comes, and when
  the transfer timeout frees it.
- `m4_x_ne_y`: `disagg_request_id` replaced while `session_id` is kept, and the other
  way round.
- `m5_gen_only`: `TRTLLM_DISAGG_BENCHMARK_GEN_ONLY` with and without the session check
  disabled (in the probe process only), and with too little KV cache.
- `m6_engine_killers`: an unreachable `ctx_info_endpoint`; an engine without a cache
  transceiver.
- `m7_fill_gate`: `TLLM_BENCHMARK_REQ_QUEUES_SIZE` and
  `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP`.
- `m8`: ServerInfo's KV connector, context stream endings, `orchestrator_type: ray`,
  prompt logprobs on a context stream.

The defect list and the conclusions for the PoC-S3 plan are in the stage STATUS
(appendix for WT-32), not here.
