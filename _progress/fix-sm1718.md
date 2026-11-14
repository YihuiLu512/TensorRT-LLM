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
