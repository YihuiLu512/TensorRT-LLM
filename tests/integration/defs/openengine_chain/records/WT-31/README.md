<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# R10: baseline of the existing disagg tests on the control form

R10 ran the disaggregated-serving tests of the PoC plan (XD-001..XD-319, 149 planned rows, 169
pytest entries) on the control form T: no `TRTLLM_TEST_OPENENGINE_CHAIN`, no
`TRTLLM_TEST_OPENENGINE_TOPOLOGY`, no waives file (waived tests ran too), no `-x`. The 2-node group
except XD-311 was run again on another cluster (2-node group rerun below).

- Source: `9b4022dd0dfe7581dd0580750d79f3261d7fc5be`, its Python files overlaid on the images
  `e17503-9fd3822a00-x86-sm100-cv2.ready.sqsh` (B200) and `e17503-9fd3822a00-x86-sm90-cv2.ready.sqsh`
  (Hopper), both built from `9fd3822a00`.
- Drivers: `scripts/`. The 11-entry reruns of the first run (below) used them as committed in
  `115ad0e039` (frozen copy
  `/home/scratch.yihlu_coreai/tmp/20261113-wt31-r10/drivers/284da90b1b64d69390c75beec69add1b5dfa574a/`,
  the same files); the first run, including the shards run again, used an earlier, unpushed version (frozen copy
  `/home/scratch.yihlu_coreai/tmp/20261113-wt31-r10/drivers/ed53945ee86f37c5f0d0e8de28723b3aa065b23b/`)
  that lacked the per-entry `timeout(1)`, `--src-rw`, `--clean-slurm-env` and some bookkeeping
  fixes.
- Results: `r10_results.tsv`, one line per pytest entry after the header (169 lines; `xd` repeats
  where a plan row has several entries). `group` is b200 (one node), hopper or 2node; `gpus` is GPUs
  per node; `gpu_model` is the GPU model the entry ran on (B200, H100-80GB or H200, from the job's
  node precheck). `outcome` / `time_s` / `job` are the first run, for the three shards run again as
  a whole (below) their second attempt; `rerun_*` the single rerun of the 11 single-node entries
  listed below (empty otherwise). `verdict` is the classification after those reruns and `reason`
  its cause: for skipped entries the test's skip message, for failures a summary from reading the
  entry's log, empty for passed entries. Both were assigned from the summary of
  `scripts/summarize_r10.py`. `waived_in` lists the platforms `waives.txt` waives the entry on
  (`any` = all, `-` = none). `time_s` is empty where no junit was written. The 2-node rerun has its
  own table, `r10_2node_rerun.tsv` (2-node group rerun below).
- Raw logs and per-entry junit: `/home/scratch.yihlu_coreai/tmp/20261113-wt31-r10/results/`.

## Totals (verdict)

| Group | passed | skipped | failed | not-runnable | killed | entries |
|---|---|---|---|---|---|---|
| B200, single node (1/2/4/8 GPUs) | 113 | 1 | 12 | 9 | 0 | 135 |
| Hopper (H100-80GB / H200, `skip_no_hopper` rows) | 21 | 0 | 1 | 0 | 0 | 22 |
| B200, 2 nodes x 8 GPUs (see the 2-node group rerun below) | 0 | 0 | 11 | 0 | 1 | 12 |

Reruns (11 entries, all single node): 10 passed. In the first run they had failed because the
source tree was mounted read-only (tests that write `output.json` / logs into their working
directory: XD-055, 073, 074, 312, 313), because `mpirun` inherited the job step's Slurm variables
and got one slot (XD-315..317), or because the server did not start within the test's 300 s
limit (XD-035, 054, 072, each the first entry of its job; the cause was not measured). The
reruns of the last three were again first entries: XD-035 and 054 passed, XD-072 timed out
again and stays `failed`.

Three first-run shards were run again as a whole, and their results come from the second
attempt: b200-4g-s04 (its first job stopped in the node precheck because one of its GPUs already
had a compute process; the owner was not recorded), b200-8g-s07 (on its first node three of its four entries failed in NCCL
initialization with "unhandled cuda error", which no other node showed) and hopper-8g-s00
(requeued by Slurm after a node failure; the first attempt left no entry results).

## Not runnable on the PoC machines or in this setup

| Entry | Reason |
|---|---|
| XD-019, XD-020 | the test requires GB200 / GB300 |
| XD-202 | the test requires Rubin |
| XD-230 | `skip_less_device_memory(200000)`: a B200 reports 183359 MiB |
| XD-003 | the test puts the source tree on `PYTHONPATH`; a tree without built bindings fails to import |
| XD-101 | first run: read-only source mount (`output.json`); not rerun, because the test also writes `examples/disaggregated/*` relative to the defs directory, where it does not exist (read from the test, not run) |
| XD-125 | the cancel-stress harness imports `test_disaggregated` by bare name; each R10 entry runs in its own pytest session, where that name does not resolve (`ModuleNotFoundError`) |
| XD-318, XD-319 | the image has no Ray |
| XD-314 | 2 nodes, from the 2-node group rerun (first-run verdict `killed`, as in the totals): with the test's TP4 / EP4 per server the model weights do not fit in B200 GPU memory (2-node group rerun below) |

## Existing failures on the control form

| Entries | Cause |
|---|---|
| XD-201 (Hopper, on H100-80GB), 208 (x2), 209, 210 (x2), 221 (1 of 2), 224 (1 of 3) | executor creation fails: insufficient GPU memory for the KV cache (`CuOOMError`; XD-221 `torch.OutOfMemoryError`) |
| XD-060 | DeepSeek-V4.1-Flash ctx TP4 / gen TP4: the ctx worker exits (rc 1) during warmup; the part of its log the test prints has no error line, cause not identified (XD-061, attention DP4, passes) |
| XD-056 | Llama-4 long context: requests rejected, 400 `extra_forbidden` |
| XD-071 | Qwen3-32B FP8 stress: 98% request errors (500, and 400 "Error in kv cache transfer") |
| XD-072 | Nemotron-3-Super FP8: server start exceeded the test's 300 s limit in both runs (cause not measured) |
| XD-123 | `test_workers_kv_cache_aware_router_eviction`: `assert 2016 < 1984` |
| XD-301..309 | 2 nodes (first run; all pass in the rerun below): the cross-node KV transfer fails (the NIXL wait returns transfer state FAILURE after about 33 s; gen answers 400 "Error in kv cache transfer") |
| XD-310 | 2 nodes, etcd service discovery (first run; passes in the rerun below): requests answered 400; no response body and no KV transfer failure in the logs, cause not identified |
| XD-311 | 2 nodes, HTTP service discovery (not rerun): the disagg server exits at start ("HTTP cluster storage is only supported for loopback-only disaggregated serving") because the test's cluster URI names the first node; `/health` never answers |

XD-314 (2 nodes) did not finish in the first run: it was the last of the 12 entries of the 4 h
2-node job, started with about 94 min left and was ended by the driver's timeout on the container
step (exit 124), before its own 7200 s server-start limit; no junit, and the test's output was not
kept, so where it stood is not known. In the 2-node group rerun below its servers exit within a
minute with a GPU memory error, so it is listed as not runnable above. XD-071, XD-123 and XD-209 are
waived on the platform they ran on (`waived_in`). XD-118 skips itself (nvbugs 5372970).

## KV transfer path

- One B200 node (DQ-24): with `cache_transceiver_config.backend` DEFAULT or NIXL, ctx and gen on
  the same GPU or on two GPUs, three requests each returned 200; the workers log
  `Using KvCacheTransceiverV2` and a native NIXL agent, so DEFAULT is NIXL through the Python
  transceiver. Backend UCX fails at worker start: "Python transceiver currently only supports
  NIXL backend, got UCX" (it asks for `transceiver_runtime: CPP` with KV cache manager V1).
- One Hopper node (UV-75): 21 of the 22 entries pass; the failing one is the KV-cache OOM above
  (XD-201, on H100-80GB; the Hopper shards ran on H100-80GB or H200, see `gpu_model`). The
  transport was not probed separately on Hopper.
- Two B200 nodes (UV-08, UV-32, G1b): the tests use `KvCacheTransceiverV2` with a NIXL agent on the
  NIXL UCX backend (worker logs). XD-301..309 reached the transfer and failed in it, XD-310 got 400s
  without a logged cause, XD-311 failed before serving (HTTP cluster storage refuses a non-loopback
  URI) and XD-314 did not finish, so in the first run no 2-node test passed and the cross-node KV
  path was not shown to work. Inside the container `ucx_info` lists RDMA (mlx5) devices, `cuda_ipc`
  and `gdr_copy`. In this run no UCX tuning was retried (the 2-node group was allocated once). G1b
  did not hold after this run; the 2-node group rerun below settles it.

## 2-node group rerun (WT-31-fix1)

In the first run XD-301..309 failed in the cross-node KV transfer, so the 2-node group except XD-311
was run again on another cluster, with 8 x B200 per node and an InfiniBand fabric between the nodes,
on the same control form, source and image (a copy with the same sha256). The two runs differ in
cluster, interconnect, GPU driver and container environment; these differences were not listed one
by one. The drivers were `scripts/` as changed by the WT-31-fix1 drivers commit (git tree
`27f2d505c8`; the jobs ran a frozen copy of that tree). A preflight job on two nodes checked the
GPUs, the full image checksum and the transport facts; then XD-301..305 and XD-306..310 ran as two
jobs with a 2 h limit and XD-314 as one job with a 5 h limit, each job with its own node precheck.
XD-311 was not rerun. Results: `r10_2node_rerun.tsv`, one line per rerun entry, with the columns of
`r10_results.tsv` except `rerun_*`: `outcome` and `time_s` are node 0's, as
`scripts/summarize_r10.py` reports them for these jobs, and `verdict` / `reason` come from reading
the logs. Raw logs and per-entry junit:
`/home/scratch.yihlu_coreai/tmp/20261114-wt31fix1-eg39/results/`.

| Group | passed | not-runnable | entries |
|---|---|---|---|
| B200, 2 nodes x 8 GPUs (rerun) | 10 | 1 | 11 |

- XD-301..310 pass: on node 0, which runs the client and checks the completions, and on node 1,
  which hosts the gen server, each entry's junit records one passed test with no failure, error or
  skip. Their junit time on node 0 is 194 to 633 s.
- XD-314 is not runnable on B200 with the test's configuration. Measured: on both nodes executor
  creation fails for the Model component with "insufficient GPU memory" (178.35 GiB per GPU, 177.06
  or 177.74 GiB free), and the test errors at setup (junit time 57 s on node 0, 73 s on node 1).
  Estimated, not measured: each server runs TP4 / EP4 with attention DP and the checkpoint is 879
  GiB, so each GPU needs over 200 GiB for the weights. Unlike the KV-cache OOMs of the first run,
  which fail later, when the KV cache is allocated, the model weights themselves do not fit, so the
  entry is `not-runnable` rather than `failed`. It could run on GPUs with more memory per GPU; that
  is left to R12 / R14. This job was submitted with `UCX_NET_DEVICES` set in its environment (not by
  the drivers), which the test's worker environment uses; the test never reached a KV transfer. The
  test class is marked `skip_less_device_memory(140000)`, below that estimate, so on B200 it starts
  instead of skipping; the threshold is left to R12 / R14 as well.
- XD-311 keeps its first-run result (`failed`) and is not in the rerun table: a code limit
  (HTTP cluster storage only for loopback-only serving, see above), not the transport.

KV transfer path (G1b): the two test files that XD-301..310 run
(`tests/unittest/llmapi/apps/_test_disagg_serving_multi_nodes.py` and
`_test_disagg_serving_multi_nodes_service_discovery.py` beside it) start the ctx server on node 0
and the gen server on node 1, so a passed completion means the KV cache crossed the nodes. Every job
was submitted with `UCX_PROTO_INFO=y` in its environment and the XD-306..310 job also with
`UCX_LOG_FILE` set to a shared directory (neither is set by the drivers); that job left 65 UCX logs
(43 non-empty). Each of XD-306..310 has inter-node protocol tables there, and the 1438 inter-node
tables name no transport other than `rc_mlx5` (RDMA) in the rows they print; six of the tables are
cut off at the end of their log, two before any row and four after their first two rows. Four of the
1438 describe puts from GPU memory to a remote GPU (`ucp_put*` from `cuda/GPU<n>` to `cuda/dev`), in
processes of XD-306, 307 and 308; all four use zero-copy over two paths for every non-empty message.
For XD-301..305 the protocol was not recorded: the output UCX prints goes to the test's captured
output, which pytest does not keep for a passed test. Every 2-node entry has a result or a reason
and ten of them show the cross-node KV path working, so G1b holds on nodes joined by InfiniBand; why
the transfer failed on the first cluster was not identified.

## Behaviors of the control form (probe, Qwen3-0.6B, 2 x B200)

- UV-25: completions `n=2`, completions `best_of=2, n=1` and chat `n=2` all return 200 with the
  expected number of choices.
- UV-33: the first generated token is " Paris". With `stop=[" Paris"]` the non-streaming
  response is empty text, `finish_reason: stop`, 1 completion token, so generation does not
  continue; the streaming response carries only `data: [DONE]` (no choice chunk).
- UV-29 (`subagent_affinity_scope: both`, parent-session header present): both the ctx
  (context_only) and the gen (generation_only) worker derive `subagent_affinity_id = parent-A`;
  without the header both derive none. The disagg server warns that the generation fleet does not
  use the `conversation` router, so instance affinity is inactive there.

## Other readings

- UV-22: the two DeepSeek-V4.1 disagg tests (XD-060, XD-061) configure NIXL with the Python
  transceiver and do not set `bounded_replay_on_generation` (default false), so they do not use
  remote-tail; XD-061 passes on the control form.
- UV-02: `DeepSeek-V4.1-Flash-NVFP4` is in the model store (XD-061 loaded it). UV-30: it runs
  on 8 x B200 with attention DP4 (XD-061, 627 s); TP4 fails at ctx worker start (XD-060).
- UV-37: a B200 reports 183359 MiB (nvidia-smi) / 182672 MiB (torch), below 200000, so XD-230
  skips.
- UV-48: `gpt_oss/gpt-oss-20b` (`config.json`: sliding window 128 on 12 of 24 layers) and
  `gpt-oss-120b` (18 of 36) are in the model store; the gpt-oss-20b disagg tests pass on 4 x B200 (XD-058, 88 s)
  and 2 x B200 (XD-219, 182 s).
- UV-70 (first half): `Qwen3/Qwen3-30B-A3B` and `Qwen3/Qwen3-30B-eagle3` are in the model store.
- UV-74: `DeepSeek-V4-Pro-nvfp4-DSpark` (XD-314's model) is in the model store (read-only listing,
  879 GiB); with the test's TP4 / EP4 per server it does not fit on B200 (2-node group rerun below).
- UV-23: the first run was 12 submissions with 33 array tasks (31 shards and two attempts of the
  probe, the first stopped in the node precheck), plus 2 resubmitted shards; the 11-entry reruns
  were 5 jobs. Queue wait from submission to start was 0 min for most single-node tasks, up to 154
  min for 8-GPU tasks (mostly the array's 4-at-a-time limit), and 0 min for the first run's 2-node
  job. Wall time of the first run: 00:16 to 04:08 (UTC+8). The 2-node rerun was 4 jobs (the
  preflight and three test jobs); each waited about 2 min or less except the XD-314 job (70 min);
  the first was submitted at 14:26 and the last ended at 16:01 (UTC+8).

## Durations and the smoke budget

`time_s` is the test's junit time. Entries that passed in the first run, including the three
shards run again (seconds):

| Group | n | median | max |
|---|---|---|---|
| B200 1 GPU | 17 | 71 | 100 |
| B200 2 GPUs | 13 | 338 | 795 |
| B200 4 GPUs | 32 | 182 | 3816 |
| B200 8 GPUs | 42 | 285 | 8938 |
| Hopper 1 GPU (H200) | 2 | 259 | 261 |
| Hopper 2 GPUs (H100-80GB) | 1 | 613 | 613 |
| Hopper 4 GPUs (H100-80GB) | 6 | 153 | 235 |
| Hopper 4 GPUs (H200) | 8 | 145 | 4246 |
| Hopper 8 GPUs (H200) | 3 | 232 | 6327 |

PS-12..PS-17 of the perf plan are XD-053, 021, 005, 009, 072 and 056. All ran; PS-16 (XD-072)
and PS-17 (XD-056) fail on the control form, so the table has no duration for them.

Smoke budget proposal (1 x B200, ctx / gen on one GPU, Qwen3-0.6B): each smoke case at most
180 s of junit time (the control form takes 49 to 100 s; each entry's wall time, including
pytest start-up, is 64 to 122 s; the margin is for the chain's extra start-up, not yet measured), and the whole PoC-S3 smoke set at most 30 min.
