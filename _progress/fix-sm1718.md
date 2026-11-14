# fix-sm1718 进度（2026-11-14）

- 子批次：fix-sm1718 —— Fix SM-17 slurm-view ssh failures and SM-18 gRPC unrequested-call cancellation
- 范围：
  - SM-17：前端 slurm_view 观察一项的 ssh 暂态失败（rc 255、未重试）直接使启动 ENV_INVALID（E1-0 h3 首投 5940970）。
  - SM-18：gRPC 过载档（tier 6 up-over）里排队超过 30 s 的未请求流被取消，导致 e0w60 的 e0b 树 VOID（C3 RED）。
  - 契约：WT/INTERFACES.md 末尾新增勘误节「Fix SM-17 slurm-view ssh failures and SM-18 gRPC unrequested-call cancellation」，被覆盖的正文句加指针。
  - 计划依据：W/RUNBOOK-stage2-draft.md §2（子批次表与 2.0 流程）及其引用的设计文档；W/00_plan/{PLAN,CONSTRAINTS,DATA-AND-ANALYSIS}.md；累计规则集 RS1–RS22（capkit-J17871/R4/brief.md、triage.md）。
- 基线：main @ 1176861dd1d8233b8de1063c2e620fe9aee21e3f
- 分支 / 工作区：capkit-fix-sm1718 @ `01_code/_local/wt-fix-sm1718`
- 验证约束：无集群访问（fakebin 10 个替身 exit 97）；每次测试 `ulimit -v 8388608` + `timeout 900` + `python -B`；测试目录逐个串行跑，不加 -x。

## 日志
- 2026-11-14 WT 已建：capkit-fix-sm1718 @ 1176861d；建本进度文件
- 2026-11-14 21:38:55 建 _local/fix-sm1718（fakebin 十个替身 ssh scp sbatch srun salloc scancel scontrol squeue sacct cdb、pyc、logs、rt.sh 照 wt-sm16 改路径）；读完 SEAMS SM-17/18、slurmview/submit/gate、capgrpc common/nullgrpc/relay；venv grpcio 1.85.1 的 cygrpc 里有字符串 grpc.server_max_unrequested_time_in_server 与 max_time_in_pending_queue；下一步本地复现 SM-18
- 2026-11-14 21:41:54 复现 unreq.py（grpc.aio 服务端首个调用 time.sleep 阻塞事件循环，其后的调用在内核未领取队列里等）：不设 ⇒ 等 30.2 s 的调用 CANCELLED；设 29 ⇒ CANCELLED；设 31/60 ⇒ OK；设 2、阻塞 6 s ⇒ CANCELLED（取消在循环恢复后才送达客户端）。结果在 repro/unreq_30s.txt、unreq_edge.txt
- 2026-11-14 21:45:53 SM-18 代码已写（common.add_server_args/resolve_max_unrequested/make_server 必填参数；nullgrpc、relay 接线并自报）；跑新测试 tests/grpc/test_unrequested.py（约 1–2 分钟）
- 2026-11-14 21:49:17 新测试 17 passed（63 s）；跑 SM-18 变异 M1（make_server 不设选项=修前）M2（nullgrpc 写死 30）M3（缺省只取超时）M4（去掉低于超时的拒绝），约 4 分钟
- 2026-11-14 21:54:27 SM-18 变异：M1（修前）2 failed（短值不再取消；relay 32 s 阻塞 ⇒ load grpc_CANCELLED 2，复现 SM-18）、M2 1 failed、M3 1 failed、M4 4 failed；日志 logs/mut_M*.log。下一步写 SM-17
- 2026-11-14 21:55:42 SM-17 代码已写（slurmview 有界重试 + failed_items；slurm-view 有失败项不写文件、退出 4、新 --attempts/--retry-wait-s）；跑 tests/jobs/test_slurmview_retry.py 与 test_units/test_submit 相关
- 2026-11-14 21:56:05 SM-17 变异：M5（不看失败项=修前）1 failed、M6（不重试）9 failed、M7（未跑项也算失败）2 failed、M8（不退避）3 failed；另 1 个 error 是变异副本没有 configs/（env 夹具），与变异无关。下一步全量跑 tests/jobs、tests/grpc（约 3–5 分钟）
- 2026-11-14 22:02:14 全量 tests/jobs 218 passed、tests/grpc 52 passed、tests/envgate 71 passed（未改，读方兼容）；INTERFACES 勘误节与正文指针、jobs/nullgrpc/capselfcheck README 已改；提交第一版
- 2026-11-14 22:02:37 作业路径 e2e：从 wt-b2-spb 的 e2e 脚本拷到 _local/fix-sm1718/e2e 改路径；建 P0 覆盖清单（HEAD fa2c56b2），再跑 e0b-s1-single（约 2–3 分钟）
- 2026-11-14 22:01 作业路径 e2e（脚本 _local/fix-sm1718/e2e，从 wt-b2-spb 拷来改路径；P0 覆盖清单 HEAD fa2c56b2；导出 PYTHONDONTWRITEBYTECODE=1）：e0b-s1-single（J9300001）submit rc=0、sbatch rc=0、L0 VALID（拷回 222 文件）、judge rc=11（NOT_RUN，C1/C7/C13 NOT_RUN，与往轮同形）、0 RED、无 INPUT_INVALID；beat、relay1 的就绪文件与 12 个档 summary 都记 `grpc_server_options` = 18 s（登记超时 5 s、档位表长约 17.x s ⇒ 取较长者向上取整），source 为 default。simgate rc 6 与往轮相同（dmi + 1 s slurm_view 等待、本机没有 Slurm）。
- 2026-11-14 22:03 slurm-view 命令行对 ssh 替身（exit 97）：`--attempts 2 --retry-wait-s 0.5` ⇒ 退出 4、作业目录为空、标准输出 failed 列出第 1 项（rc 97、retried_rcs [97]）、rcs [97,null,null]；fakebin.log 两次 ssh。

## 结果汇总

### 核实（MEASURED）
- grpcio 1.85.1（venv）：选项名 `grpc.server_max_unrequested_time_in_server`，整秒；不设时排队 30.2 s 的调用 CANCELLED，设 29 CANCELLED、31/60 OK；设 0、−1 时排队的调用（3 s）CANCELLED；设 2³¹−1 OK（repro/unreq_30s.txt、unreq_edge.txt）。取消要等服务端事件循环恢复后才送达客户端。
- SM-18 复现（tests/grpc/test_unrequested.py::test_relay_default_outlasts_the_grpc_30s_default）：relay 事件循环阻塞 32 s，修前（变异 M1，make_server 不设选项）负载端 grpc_CANCELLED 2；修后缺省 45 s ⇒ 4 个流全部完成。
- 测试：tests/jobs 218 passed、tests/grpc 52 passed、tests/envgate 71 passed（未改，读方兼容）；变异 M1–M8 全部被测试抓到（logs/mut_M*.log）。

### 修复
| 处 | 改动 |
|---|---|
| `tools/capjobs/slurmview.py` | 每项查询有界重试（attempts、线性退避）；条目加 `attempts`、`retried_rcs`；`failed_items` |
| `tools/capjobs/submit.py` | `slurm-view` 有失败项不写文件、退出 4、输出 `failed`；新 `--attempts`、`--retry-wait-s`；作业目录不存在先报错不发 ssh |
| `tools/capgrpc/common.py` | `UNREQUESTED_OPTION`、`add_server_args`、`resolve_max_unrequested`；`make_server` 第二参数必填 |
| `tools/capgrpc/nullgrpc/server.py`、`tools/capselfcheck/relay.py` | 接 `--max-unrequested-s`，就绪 info 与每档 summary 记 `grpc_server_options` |
| 文档 | INTERFACES 勘误节 + §5.5.2、§11 三行指针；jobs、nullgrpc、capselfcheck README |
- 2026-11-14 22:05:11 全量验证 v1（HEAD 2c8bbf4，脚本 _local/fix-sm1718/fullv.sh 逐步调用：10 个测试目录逐个、selftest 25、patchcheck、py_compile、bash -n，约 10 分钟）
- 2026-11-14 22:07:29 capprobe 8 failed：测试读 WT/probe-tree（照 fix-SM14/SM16 建测试用软链 WT/probe-tree -> ../../probe-tree，收尾移到 _local/fix-sm1718/probe-tree.link），重跑 capprobe
- 2026-11-14 22:18:07 tests/launch 7 failed 全是 test_aux_helper multiprocessing forkserver「AF_UNIX path too long」（TMPDIR=$L/tmp/v1 过长，环境问题）；改用短 TMPDIR /home/scratch.yihlu_coreai/tmp/20261114-capkit-sm1718（含 PROVENANCE.md）重跑 tests/launch
- 2026-11-14 22:21:43 作业路径 e2e 终版：重建 P0 覆盖清单（HEAD 2c8bbf4a），跑 e0b-s1-single（J9300011）与 e0a-nullsse（J9300012），各约 1–3 分钟
- 2026-11-14 23:12:17 恢复（暂停后重启）：此前全量验证 v1 与 e2e 作废；恢复测试用软链 WT/probe-tree（从 _local/fix-sm1718/probe-tree.link.230403 改名回来）；开始全量验证 v2（HEAD 2c8bbf4，TMPDIR /home/scratch.yihlu_coreai/tmp/20261114-capkit-sm1718/v2，10 个测试目录逐个，约 15 分钟）
- 2026-11-14 23:24:39 全量验证 v2 完成：10 个目录全过（1018 passed）、selftest 25 过、patchcheck PASS、py_compile 190/0、bash -n 3/0；下一步作业路径 e2e（e0b-s1-single、e0a-nullsse，约 5 分钟）
- 2026-11-14 23:27:40 作业路径 e2e v2（overlay 重建于 e2e/ovl/v2，HEAD 2c8bbf4a）：e0b-s1-single J9300021、e0a-nullsse J9300022 均 L0 VALID、judge rc=11 NOT_RUN、0 RED、无 INPUT_INVALID；e0b 的 beat、relay1 就绪与 24 个服务端 summary 记 grpc_server_options=18 s；下一步 SM-18 过载档本地复现（修前 1176861 导出树 vs HEAD）
- 2026-11-14 23:30:39 SM-18 过载档复现 v2：repro/sm18-v2/{pre,post} 由 git archive 导出（pre=1176861d main 基线、post=HEAD 2c8bbf4a），同一复现测试用 e0w60 e0b-s1 参数（beat 4 ms、50 msg/流、relay 每 msg busy 1 ms、up-over 30 rps × 60 s、超时 60 s），每侧约 2–3 分钟
- 2026-11-14 23:34:30 SM-18 修前（pre 1176861d）：drain grpc_CANCELLED 158/1800（tier 2 up-over），relay1 streams_accepted 1642 = ok；修后（post HEAD）开跑约 2–3 分钟
- 2026-11-14 23:38:12 SM-17 假 ssh 复现 v2：repro/sm17-v2（bin/ssh 按调用序号返回 255，其余回放 Slurm 文本；pre-tools=1176861d 导出），5 个用例约 1 分钟
- 2026-11-14 23:4x SM-17 假 ssh 复现 v2 完成；结果见下节「全量验证 v2（恢复后重做）」

## 全量验证 v2（恢复后重做，HEAD 2c8bbf4a；v1 与此前 e2e 作废）

### MEASURED
| 项 | 结果 |
|---|---|
| tests/capcommon | 190 passed |
| tests/capload | 33 passed |
| tests/capprobe（测试用软链 WT/probe-tree） | 159 passed |
| tests/envgate | 71 passed |
| tests/grpc | 52 passed |
| tests/jobs | 218 passed |
| tests/judge | 97 passed |
| tests/nullsse | 4 passed |
| tests/sidecar | 5 passed |
| tests/launch（短 TMPDIR） | 189 passed |
| `samples/run_selftest.py --package capjudge --expect-samples 25` | rc 0，25 个样本，errors [] |
| patchcheck（探针树只读，base 2f22a98d，--rev HEAD） | PASS，violations [] |
| py_compile | 190 ok / 0 bad |
| bash -n | 3 ok / 0 bad（capkit_inner.sh、capkit_job.sbatch、prep_tokenizer.sbatch） |
| 作业路径 e2e e0b-s1-single（J9300021，overlay 重建 e2e/ovl/v2） | L0 VALID（拷回 222 文件），judge rc 11 NOT_RUN，0 RED，无 INPUT_INVALID；beat、relay1 就绪与 24 个服务端 summary 记 grpc_server_options = 18 s（default） |
| 作业路径 e2e e0a-nullsse（J9300022） | L0 VALID（拷回 318 文件），judge rc 11 NOT_RUN，0 RED，无 INPUT_INVALID |
| SM-18 过载档复现（repro/sm18-v2，e0w60 e0b-s1 参数：beat 4 ms、50 msg/流、relay 每 msg busy 1 ms、up-over 30 rps × 60 s、超时 60 s） | 修前 1176861d：grpc_CANCELLED 158/1800，失败流 frames 0、done−sent 30.00–30.07 s，relay1 streams_accepted 1642 = ok；修后 HEAD：0 失败、1800 ok = streams_accepted，relay1 自报 127 s（default），首帧晚于 30 s 的流 367 个全部完成（最长 37.8 s） |
| SM-17 假 ssh 复现（repro/sm17-v2，bin/ssh 按调用序号返回 255） | 修前：255 一次 ⇒ rc 0 且写出含 item 1 rc 255 的 slurm_view.json（只发 1 次 ssh）；修后：第 1 次或第 2 次调用 255 ⇒ 重试（attempts 2、retried_rcs [255]）后 rc 0、文件三项 rc 0；始终 255 ⇒ rc 4、不写文件（attempts 3）；`--attempts 2` 两次 255 ⇒ rc 4、不写；item 3 连续 3 次 255 ⇒ rc 4、不写。修后写出的文件里没有失败项 |

### 说明
- 本机同时有其他会话的 pytest 在跑（b2-judge 树、J17912），不是本子批次的进程；本子批次的 pytest 始终一次一个。
- fakebin.log：v2 的 10 个测试目录与 selftest 期间无新增行；e2e 每个作业 3 行（simgate 的 scontrol show job/node、squeue，均为 exit 97 替身，与往轮同形）；SM-17 复现的 ssh 调用走 repro/sm17-v2/bin/ssh（不连网）。
- 2026-11-15 00:30:38 开始 R1 修复（slug capkit-fixsm1718-J17871，base head 4cb310a1cb）：F1–F9，先读码
- 2026-11-15 00:34:42 R1 修复：跑新用例 test_unrequested 的 resolve/queue-length 部分（约 1 分钟）
- 2026-11-15 00:35:48 R1 修复变异 M9（make_server 不设排队条数上限）/M10（第 2 项失败仍查第 3 项）/M11（不留之前各次 stderr），约 2 分钟
- 2026-11-15 00:37:33 R1 修复：代码与测试已改（F1–F4、F7），变异 M9/M10/M11 均被抓到；排队条数复现 repro/r1-pending 重跑；下一步改 INTERFACES 勘误与 README
- 2026-11-15 00:40:22 R1 修复：勘误第 1–3 条重写、§5.5.2 指引注、jobs/README 已同步；累计规则 RS1–RS27 全量核对后再收紧 2 处措辞（测试 docstring 不写内核缺省值、勘误变异数写明一次运行）；提交
- 2026-11-15 00:40:41 全量验证 v3（HEAD a3b6cecd95，fullv.sh 轮名 v3，TMPROOT 短路径；先 tests/grpc、tests/jobs，约 4 分钟）
- 2026-11-15 00:44:31 v3：tests/capcommon capload capprobe envgate nullsse sidecar（约 3 分钟）
- 2026-11-15 00:46:55 v3：tests/judge、tests/launch（约 5 分钟）
- 2026-11-15 00:52:49 作业路径 e2e v3：重建 P0 覆盖清单（HEAD a3b6cecd，e2e/ovl/v3），再跑 e0b-s1-single（J9300031）与 e0a-nullsse（J9300032），各约 1–3 分钟
- 2026-11-15 00:54:45 SM-18 过载档复现 v3（repro/sm18-v3，pre=1176861d、post=HEAD a3b6cecd95 由 git archive 导出，同 v2 测试文件），每侧约 2–3 分钟
- 2026-11-15 00:59:35 SM-17 假 ssh 复现 v3（repro/sm17-v3；pre=1176861d、post=HEAD 导出树），6 个用例约 1 分钟

## R1 修复（slug capkit-fixsm1718-J17871，修复提交 a3b6cecd95；base 4cb310a1cb）

| 项 | 改动 |
|---|---|
| F1 | `capgrpc.common.server_queue_options`：时限 + `grpc.server.max_pending_requests` / `_hard_limit` 都取 2³¹−1；`make_server` 与自报 `grpc_server_options` 取自它；新测试 `test_queue_length_limit_drops_no_call_behind_a_stalled_loop`（两进程，2500 个调用排在阻塞的事件循环后） |
| F2 | `slurmview.collect`：第 1 项仍失败或没给出单个节点 ⇒ 第 2、3 项不查；第 2 项仍失败 ⇒ 第 3 项不查（`attempts` 0，`stderr` 写 `not run: …`） |
| F3 | 条目加 `retried_stderr`；`slurm-view` 标准输出 `failed[]` 也带上 |
| F4 | `slurmview.check_job_dir`，`cmd_slurm_view` 与 `write` 都调它 |
| F5–F9 | INTERFACES 勘误第 1–3 条重写（版本范围、最坏耗时、证据位置、被测系统范围、S-PB 时点）；§5.5.2 指引注；`jobs/README.md`；两处测试 docstring 不强于断言 |

### MEASURED（HEAD a3b6cecd95，全量验证 v3，logs/v3/summary.txt）
| 项 | 结果 |
|---|---|
| 变异 M9（`make_server` 不设两项条数上限） | `test_queue_length_limit…` 失败：`{"OK": 2042, "CANCELLED": 458}`（logs/r1_mut_M9.log） |
| 变异 M10（第 2 项仍失败后仍查第 3 项） | 2 failed（logs/r1_mut_M10.log） |
| 变异 M11（不留之前各次 stderr） | 6 failed（logs/r1_mut_M11.log） |
| 条数上限复现重跑（repro/r1-pending，grpcio 1.85.1，阻塞 8 s） | 不设：N 1500 ⇒ 61 CANCELLED、N 4000 ⇒ 1454；显式 1000/3000：N 1500 ⇒ 47；2³¹−1：N 4000 ⇒ 全部 OK |
| tests/grpc | 53 passed |
| tests/jobs | 221 passed |
| tests/capcommon / capload / capprobe（测试用软链 WT/probe-tree）/ envgate / nullsse / sidecar | 190 / 33 / 159 / 71 / 4 / 5 passed |
| tests/judge / tests/launch（短 TMPDIR） | 97 / 189 passed |
| selftest（25 样本） | rc 0，errors [] |
| patchcheck（--rev HEAD） | PASS，violations [] |
| py_compile / bash -n | 190 ok 0 bad / 3 ok 0 bad |
| 作业路径 e2e e0b-s1-single（J9300031，覆盖清单 e2e/ovl/v3） | L0 VALID（222 文件），judge rc 11 NOT_RUN（C1/C7/C13，与往轮同形），0 RED，无 INPUT_INVALID；beat、relay1 的就绪文件与 24 个 summary 的 `grpc_server_options` 都是三个选项键（时限 18 s，两项条数上限 2147483647）+ source |
| 作业路径 e2e e0a-nullsse（J9300032） | L0 VALID（318 文件），0 RED，无 INPUT_INVALID |
| SM-18 过载档复现 v3（repro/sm18-v3，同 v2 参数） | 修前 1176861d：drain `grpc_CANCELLED` 187/1800，relay1 streams_accepted 1613；修后 HEAD：0 失败，1800 ok = streams_accepted，relay1 自报三个选项键（时限 127 s） |
| SM-17 假 ssh 复现 v3（repro/sm17-v3） | 修前：第 3 项 255 一次 ⇒ rc 0、写出含 item 3 rc 255 的文件；修后：第 3 项或第 1 项 255 一次 ⇒ 重试后 rc 0、文件三项 rc 0、`retried_stderr` 记下 255 的 stderr；始终 255 ⇒ rc 4、3 次 ssh、不写；第 2 项连续 3 次 255 ⇒ rc 4、4 次 ssh（第 3 项未查）、不写；第 3 项连续 3 次 255 ⇒ rc 4、5 次 ssh、不写。修后写出的文件里没有失败项 |

### NOT RUN
- grpcio 1.84.0（集群镜像版本）上的选项语义：本机没有 1.84.0，按勘误第 2 条由首个集群作业复核。
- follow-up 登记（真实链路排队提前取消的归类；ControlMaster 复用 ssh）不在本仓库，交主会话写 `W/02_e0/SEAMS.md` 或 `W/STATUS.md`。
