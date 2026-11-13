<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# WT-33: case-03 environment and asset probe (GB300)

This probe changes no product code. It records whether case-03 (ds-engine) can run on the PoC-S3 GB300 cluster,
before the case-03 GPU steps of later work items start (milestone M30, gate G3).

ds-engine is `0163a5d9c58b31681ee03054d01c17d8fc7e96b8`; TRT-LLM is the stage03 integration branch at
`9b4022dd0dfe7581dd0580750d79f3261d7fc5be`. File:line references to ds-engine are at that commit. Seven-digit
numbers in parentheses are Slurm job ids. Cluster names, node names, paths on the cluster, disk room, logs and job
scripts are in the local evidence directory `/home/scratch.yihlu_coreai/workfile/feat_dev-J17876/data/stage-00/test-00/`
and in appendix C of `/home/scratch.yihlu_coreai/dev/feat-dev-J17503/tmp/data/stage-03/test-00/STATUS.md`; they are
not repeated here.

## Gate G3

G3 holds (2026-11-13, cluster time):

- The case-03 assets are copied into this account's project directory on the cluster and are readable (sha256 equal
  on both sides; 1814387).
- ds-engine's native N-0 layout runs as is, and its GSM8K-200 score is above the ds-engine bar of 95 in all six
  rounds: three in the author's container (1814988) and three in the PoC image (1815299), see the N-0 table below.
- The aarch64 PoC image holds TRT-LLM and ds-engine in one container and is listed in the image manifests on the
  cluster and on the agent host. ds-engine builds in it (1814470), runs N-0 in it (1815299), and imports in one
  process with TRT-LLM (1817130). This answers UV-26.

## Assets (card item 1)

| Asset | Check |
|---|---|
| Checkpoint `DeepSeek-V4.1-Flash-NVFP4`: 85 files, 492G, without the author's `.git` | sha256 of every file equal on both sides (1814387) |
| The ds-engine author's development container (TRT-LLM `072c1abd45`, CUDA archs 100-real and 103-real) | sha256 `641d4d3b6b681752e223ad78db79579c98a08821f88d0175b101626e9719cb5f` on both sides |
| CI sbsa base image (`LLM_SBSA_DOCKER_IMAGE` of the integration branch), copied from the CI image cache | sha256 `0539328228bbd0d2da059db4450da5466dddfe0ef74a000bb88108dc5d8e4379` on both sides |
| PoC image `e17503-9b4022dd0d-aarch64-sm103-ds0163a5d9c5.ready.sqsh` (built here, see below) | sha256 `0ee4260116a3c31d90582b198173a8b9910381a4b9f430112c1b11c6fcb9409d` |

- Disk room and quota were checked before copying; the readings are in the local evidence directory.
- Some file creation on the cluster's shared file system fails with `EROFS`: creating directories failed in 1814487,
  and a pip install failed in the first build attempt of 1814470. Both succeeded on a later attempt (the pip install
  on the same node). Jobs that create files there should be able to retry.
- The cluster's CI model tree has the same checkpoint: the same file names and byte counts, except one extra
  `encoding/__pycache__/*.pyc` in the author's copy, and equal md5 of `config.json` and `hf_quant_config.json`. The
  weights were not hashed there.
- From the cluster's data-transfer node, `git ls-remote` reaches the aiperf fork `SemiAnalysisAI/agentx-harness`
  (the pinned commit `754356e9a39acc6cc6afb242d123bb57c3fb6f75` was not fetched), and the Hugging Face API answers
  for the dataset `semianalysisai/cc-traces-weka-062126`; compute nodes were not checked. The author's client venv
  hard-codes the author's paths and was not copied. `dev/setup_agentx.sh` cannot be run as is, because it creates
  the venv under the author's directory (`dev/setup_agentx.sh:5-6`); run its steps with a venv path in this account.

## IMEX and the NVLink fabric (card item 2)

In both N-0 jobs (1814988, 1815299) the job had one IMEX channel (`/dev/nvidia-caps-imex-channels/channel<N>`),
visible on the host and in the pyxis container, and `nvidia-smi -q` reported the fabric as Completed / Success. In
all six N-0 rounds the generation instance logged "KV pulled from context instances over the NVLink fabric".

## Native N-0 baseline (card item 3)

`dev/test_disagg.sh 200` was run unchanged three times per image: context DEP2 on GPUs 0-1, generation TP2 on
GPUs 2-3, the ds proxy, then GSM8K-200 through the proxy at concurrency 32, greedy. The job set two of the script's
inputs, `MODEL` (this account's checkpoint copy) and `GSM_EXTRA=--out <json>`, and also `HOME` (a writable
directory) and `PYTHONUNBUFFERED=1`; the container otherwise inherits the job's environment. Per-round numbers and
the between-round comparisons are in `n0_native_gsm8k.yaml`.

| Image (ds-engine build job / N-0 job) | GSM8K-200 per round (strict = flexible) | Mean | To proxy ready | GSM8K-200 time |
|---|---|---|---|---|
| author's container (1814987 / 1814988) | 99.0 / 99.5 / 99.5 | 99.33 | 218.5 / 179.0 / 193.2 s | 9.7 / 10.7 / 10.5 s |
| PoC image, `/opt/ds-engine` (1814470 / 1815299) | 99.5 / 98.5 / 99.5 | 99.17 | 269.6 / 183.4 / 210.4 s | 11.2 / 10.9 / 11.3 s |

"To proxy ready" is measured from the `srun` that runs the round in the job's container, which the job's first step
had already created.

- Between any two of the six rounds, at most 2 of 200 questions change between right and wrong. Doc 119 is wrong in
  all six rounds.
- The answers are not reproducible word for word: two rounds give the same answer text on 59-87 of 200 questions
  within an image and on 55-71 across the images. Compare accuracy, not text.
- The two ds-engine builds come from different CUDA versions (13.2 in the author's container, 13.4 in the PoC
  image), and their accuracy cannot be told apart in these rounds. Proposed by this probe: later native arms use
  the PoC image, the build the chain arms also use, and do not mix in numbers from the author's container.
- Proposed by this probe for the paired comparison in AC3-F7 (DQ-41, not decided): three rounds per arm, native and
  chain alternating in one job; every round above 95; the chain mean at least the native mean minus 1.0;
  per-question disagreements reported but not gated.

### Switches the N-0 script turns on (UV-36)

Read from `dev/test_disagg.sh:10-17` and `tests/configs/disagg_{ctx_dep2,gen_tp2,local}.yaml`, and checked against
the running processes of the same three commands in the author's container (1814989: command lines, `DSV4_*` and
`TLLM_*` environment, proxy and context logs). The PoC-image run (1815299) uses the same script; its process
environment was not captured.

| DS-O item | N-0 script |
|---|---|
| context router of the proxy | `conversation` |
| ADP conversation affinity in the context instance | on (`attention_dp_config.kv_cache_routing_conversation_affinity: true`) |
| DS-O01 generation hop gets only the last message | on (`gen_strip_message_history: true`) |
| DS-O02 / DS-O03 prompt ids from the context server, as int32 base64 | on (`gen_tokids_ctxbytes: true`) |
| DS-O04 context requests pinned to a front-end process | affinity on by default (the proxy calls `GET /frontends`), but each instance has one front end; tokenization cache off (`DSV4_TOKCACHE_ENTRIES` unset) |
| DS-O05 route new sessions by `/load` | off (`DSV4_PROXY_LOAD_ROUTE` unset) |
| DS-O06 sub-agent routing by parent session | off (no `conversation_affinity_header_for_subagents`) |
| DS-O07 / DS-O11 parent session as `conv` | off (`DSV4_ADP_CONV_KEY` unset) |
| DS-O08 proxy processes | 1 |
| DS-O09 front-end processes per instance | 1 (no `--num_serve_frontends`) |
| DS-O10 `conv` pinning | off (`DSV4_ADP_CONV_PIN` unset) |
| DS-O12 request log tag | off (`DSV4_REQ_LOG` unset) |

Both instances also draft 5 tokens per step (DSpark, `max_draft_len: 5`), and block reuse is on for the context
instance only (both as logged in each instance's engine config). Both configs also set
`cache_transceiver_config: {backend: NIXL}`, but each instance logs "cache_transceiver_config is accepted but
ignored by dsv4-serve"; the KV cache moves by ds-engine's own pull over the NVLink fabric.

The running processes match this reading, which is the condition the perf plan sets for deciding the native arms on
N-0. Proposed by this probe (not decided): on N-0, run E3-native and E3-native-al as one arm, namely the script
plus the alignment items that the perf plan fixes to E3-chain's values (retries, keep-alive and the others it lists).

## Same-seed repeatability (UV-62)

1814989, author's container, through the ds proxy, `max_tokens` 96. The same request was sent three times:

| Case | The three outputs |
|---|---|
| completion, greedy, sent alone | identical |
| completion, temperature 1.0, seed 1234, sent alone | identical |
| chat completion, temperature 1.0, seed 1234, sent alone | identical |
| completion, temperature 1.0, seed 1234, each time sent concurrently with 7 other requests | all different |
| completion, temperature 1.0, no seed (control) | all different |

In this test, same-seed requests gave identical outputs only when sent alone, so a same-seed assertion should send
its requests one at a time. Inference: the sampled output depends on the other requests running with it.

## PoC image and ds-engine build (card items 4 and 5)

- Built on one 76-core aarch64 CPU node (1814470, 1 h 50 min 45 s): `build_wheel -a 103-real` of the integration
  branch on the CI sbsa base, then ds-engine in `/opt/ds-engine` with the steps of `dev/fetch_third_party.sh` and
  `dev/build.sh` (return code 0, 237 s inside the container, `sm_103a` only). The first `build_wheel` attempt
  failed with `EROFS` in pip and cost 23 minutes.
- Neither `tensorrt_llm` nor `dsv4_serve._dsv4` imports on the CPU build node (`libcuda.so.1` missing). On a GB300
  node (1817130) one process imports `tensorrt_llm`, `tensorrt_llm.bindings`, `tensorrt_llm.grpc.openengine` and
  `dsv4_serve._dsv4` in either order and sees 4 GPUs; `trtllm-serve --help` returns 0.
- UV-04, gRPC dependencies:

| Package | PoC image (1814470) | Author's container (1814987) |
|---|---|---|
| grpcio | 1.84.0 | 1.83.1 |
| grpcio-health-checking | 1.81.2 | missing |
| protobuf | 6.33.6 | 6.33.6 |
| tensorrt_llm | 1.4.0rc3 | 1.3.0rc25 |

Inference: OpenEngine server steps on GPU should run in the PoC image. The author's container lacks
`grpcio-health-checking`, which the OpenEngine server code imports as `grpc_health`
(`tensorrt_llm/grpc/openengine/_health.py:10`); whether that container's TRT-LLM has `tensorrt_llm.grpc.openengine`
at all was not checked.

## Iteration cost on the isolated cluster (card item 6, UV-23)

| Step | Time |
|---|---|
| Sync a source tree from the agent host to the cluster | ds-engine 7 s; TRT-LLM (840 MB) 98 s |
| Queue for one exclusive CPU node | about 6 min (1814470) and 1 min (1814987) |
| Queue for one GB300 node | 3 s from eligible to start (1814988, 1814989) |
| Build ds-engine in a source tree the container mounts, as one job step (container start, third-party fetch, build) | 215 s (1814987) |
| Start the container of a GPU job | 26-33 s |
| One N-0 round in a running container (start, load, GSM8K-200, stop) | 214-305 s |
| Build the PoC image from scratch (cold ccache) | 1 h 51 min, including the failed 23 min attempt |

The queue readings come from a few jobs on 2026-11-13 between 06:30 and 07:10 PST; busy hours were not measured.
Adding up the rows: a ds-engine change costs about 9-16 minutes from sync to an N-0 result (sync, CPU queue, build
step, GPU queue, container start, one round); a Python-only change of ds-engine about 4-6 minutes (sync, GPU queue,
container start, one round); plus any longer queue. Putting a new ds-engine build into the PoC image adds an image
export (172 s in 1814470).

## Multi-node ds-engine instances (card item 7, UV-60, read only)

- One multi-node instance is one `srun -N <nodes> --ntasks-per-node=1` step that runs the same
  `python3 -m dsv4_serve ... --tp_size <tp> --server_role <role>` command on every node
  (`dev/agentx_deploy.sh:127-132`).
- `dsv4_serve` decides the role itself: when `tp_size` exceeds the node's GPUs, the node rank comes from
  `DSV4_NODE_RANK`, else `SLURM_NODEID`, else `SLURM_PROCID`; the leader is `DSV4_LEADER` or the first Slurm host;
  the distributed port is `DSV4_DIST_PORT` (default 29600) (`python/dsv4_serve/cli.py:776-799`).
- Node 0 (the leader) serves HTTP and schedules. Ranks above 0 only build the engine, load weights and follow the
  leader's steps; they serve nothing (`cli.py:697-701`, `cli.py:831-840`). Extra front-end processes
  (`--num_serve_frontends`) skip the multi-node check (`cli.py:696`).
- `DSV4_XFER_HOST` is the instance's first node (`dev/agentx_deploy.sh:127`); context instances advertise their KV
  transfer endpoint there, on an OS-chosen port unless `XFER_PORT_BASE` is set (`agentx_deploy.sh:105-109`). The
  proxy lists only each instance's first node (`agentx_deploy.sh:101-104`).
- Inference, not run: in OpenEngine server mode the follower nodes need only ds-engine's own launch, and the
  OpenEngine server and the worker front end belong on the leader node only.
