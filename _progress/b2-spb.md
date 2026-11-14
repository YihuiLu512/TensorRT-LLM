# b2-spb（第二批 2B：S-PB 预制桩、指纹接缝与请求字节输入）进度

- 子批：b2-spb（RUNBOOK-stage2-draft.md §2 子批表第 3 行「2B S-PB + 指纹接缝 + 请求字节输入」）。
- WT = `_local/wt-b2-spb`（分支 capkit-b2-spb，起点 0a1410a4206501650f3fa71b43706b2a4bb1847f）；C（01_code main）只读；探针树 C/probe-tree 只读。
- 范围：+ `tools/capstubs/prebaked.py`（容器内生成预制 `GenerateResponse` 字节）；+ `tools/capstubs/prebaked_server.py` 与 `__main__`（复用 nullgrpc 的 `BeatSchedule`、待发队列；解析 `stopping.max_tokens`；注入 `send.msg`/`beat.step`；short 开关；`streams` 带请求号；backlog 槽；可选的 Control/Health）；~ `caplaunch/scenario.py`（`TOOL_TABLE`，`INPUT_KEYS` 加 `request_bytes`）；~ `capjobs/launches.py` 与 `jobs/capkit_inner.sh`（门禁后生成 fingerprint 与 reqbytes，SM-6）；~ 判定器输入核对；`fake_openengine.py` 的发送方式见 D16；+ `configs/gen_e0s_configs.py`（补充冒烟场景）；+ `tests/capstubs/`。
- 契约：WT/INTERFACES.md 末尾新增勘误节「Batch 2B: S-PB prebaked stub, fingerprint seam and request-bytes input」，被覆盖的正文句子加指引注。
- 验证：单测加本地 e2e（nullgrpc 式读者对 S-PB，pb2 用测试替身）；容器内真 pb2 与 `--request-format openengine` 的冒烟属 1.4 补充冒烟，本子批不跑。

## 进度日志
- 2026-11-14 20:12:56 开工：建 worktree（capkit-b2-spb @ 0a1410a4206501650f3fa71b43706b2a4bb1847f），写本进度日志头。
- 2026-11-14 20:19:53 2B-1-prebaked 开工：读完设计 §2.3 S-PB、E2、§7、§12.7、RUNBOOK §2、nullgrpc/reqbytes/capgrpc.common、产品 generation.proto 与 client stream 契约；开始写 capstubs/prebaked.py、prebaked_server.py 与 tests/capstubs/
- 2026-11-14 20:30:44 prebaked/prebaked_server/tests 写完，tests/capstubs 两文件全绿（u2 63、s2 17）；开始变异 M1–M8（_local/b2-spb/mut，逐个串行，约 5 分钟）
- 2026-11-14 20:35:48 自审修 4 处（inspect 的 request_id 线型、短缩不叠加、死属性、S-FE 选项报错），变异 M1–M11 全被抓；跑 tests/capstubs 全量、test_fingerprint、py_compile（约 1 分钟）
- 2026-11-14 20:36:59 2B-1-prebaked 完成待提交：tests/capstubs 82 passed（logs/v1-capstubs.log）、tests/launch/test_fingerprint.py 2 passed、py_compile 通过；变异 M1–M11 全被抓（_local/b2-spb/mut/summary.txt）；fakebin.log 无记录。WT/probe-tree 软链（测试 real_pb2 用）已移到 _local/b2-spb/probe-tree.link，重跑 tests/capstubs 前要移回。INTERFACES 新增勘误节「Batch 2B …」第 1–8 条与 6 处正文指针。
- 2B-1 未做（本子批后续步骤）：caplaunch/scenario.py（TOOL_TABLE、INPUT_KEYS 加 request_bytes）、capjobs/launches.py 与 jobs/capkit_inner.sh 作业内生成 fingerprint / reqbytes / prebaked（SM-6）、判定器输入核对（C10 核 sfe_module sha）、configs/gen_e0s_configs.py 补充冒烟场景、D16 的 fake_openengine.py 进覆盖清单。
- 2B-1 NOT RUN：真 fake_openengine.py 的 _Control/_Health、真前端连 S-PB、容器内 python -m capstubs.prebaked（真 pb2 模块名导入）——都属 1.4 补充冒烟。
- 2026-11-14 20:44:48 2B-2-wiring 开工：读完 RUNBOOK §1.4/§2、SEAMS SM-6、设计 §2.3、scenario/orchestrator/launches/inner.sh/overlay/judge inputs/C10/smoke/gen_e0_configs；方案：场景 job_inputs（fingerprint/request_bytes/prebaked）由 capjobs.launches 逐启动在门禁后生成并以 --job-input 交给 caplaunch，D16 用覆盖清单 stub_modules + C10 核 sha
- 2026-11-14 20:52:41 代码初稿完成（jobinputs、launches、orchestrator、scenario、judge、C10、overlay stub_modules、smoke 角色、gen_e0s_configs）；开始逐目录跑现有测试（launch/jobs/judge/capprobe/capstubs，各约 1–5 分钟）
- 2026-11-14 21:06:30 新测试：tests/jobs/test_job_inputs.py 13 passed、capprobe overlay stub 32 passed、tests/launch/test_job_inputs_launch.py 29 passed；接着写 C10 桩模块核对测试、S-PB 作业路径 e2e、smoke 角色测试
- 2026-11-14 21:09:08 测试补齐（C10 桩模块 4、S-PB 作业路径 e2e 1、smoke 角色 2、e0s 配置 5）；开始写文档：INTERFACES 勘误 9–13 与正文指针、各 README
- 2026-11-14 21:12:13 文档写完（INTERFACES 勘误 9–14 与 6 处正文指针、7 个 README、configs/e0s/README）；开始 py_compile 与全量测试（launch/jobs/judge/capprobe/capstubs/capcommon/grpc + selftest，约 15 分钟，逐目录串行）
- 2026-11-14 21:24:11 全量全绿（launch 225、jobs 211、capprobe 171、capstubs 83、judge 97、capcommon 190、grpc 36；selftest EXIT=0 25 样本）；开始变异 J1–J14（_local/b2-spb/mut2，逐个串行，约 6 分钟）
- 2026-11-14 21:26:03 变异 J1–J14 全部被杀（每个 rc=1，见 _local/b2-spb/mut2/summary.txt）；probe-tree 软链移回 _local/b2-spb/probe-tree.link；提交 2B-2-wiring
- 2026-11-14 21:32:28 全量验证开工（子批 HEAD f9246b5 起）：probe-tree 软链移回 WT/probe-tree；后台跑 fullv.sh y1（11 个测试目录逐个 + selftest 25 + patchcheck --rev HEAD + py_compile + bash -n，约 15 分钟）
- 2026-11-14 21:45:25 y1：10 目录全绿；tests/launch 7 个 test_aux_helper 失败为环境问题（fullv 设了很长的 TMPDIR ⇒ forkserver AF_UNIX path too long；x1 未设 TMPDIR 时 225 passed）；selftest 漏了 --tools/--tmp。改为不设 TMPDIR 重跑 launch 与 selftest（约 4 分钟）
- 2026-11-14 21:49:06 launch 重跑 225 passed（logs/y1b-launch.log）、selftest 25 EXIT=0（logs/y1b-selftest.log）、patchcheck --rev HEAD PASS 7/8。作业路径 e2e：r5b 脚本随 wt-r4 已删，按 02_e0/sweep-sim + e0r1_{build,plan}.py 重建到 _local/b2-spb/e2e（真 capkit_job.sbatch + capkit_inner.sh，srun 替身，throwaway venv 收覆盖，gate 包装器只忽略退出码）；先跑 e2e-spb-null J9200001（约 2–3 分钟）
- 2026-11-14 21:49:59 e2e-spb-null J9200001：submit rc 0、launches L0 VALID、judge rc 11（NOT_RUN）、0 RED、无 INPUT_INVALID、C10 PASS 5/5；开跑回归 e0a-nullsse J9200002（约 2 分钟）
- 2026-11-14 21:50:39 e0a-nullsse J9200002：launches VALID、judge rc 11、0 RED、无 INPUT_INVALID；开跑 e0b-s1-single J9200003（约 2 分钟）
- 2026-11-14 21:52 e0b-s1-single J9200003：launches VALID、judge rc 11、0 RED、无 INPUT_INVALID。probe-tree 软链移回 _local/b2-spb/probe-tree.link。

## 子批 HEAD 全量验证（f9246b5，代码未改）

| 项 | 结果 | 证据 |
|---|---|---|
| tests/capcommon | 190 passed | _local/b2-spb/logs/y1/t_capcommon.log |
| tests/capload | 33 passed | y1/t_capload.log |
| tests/capprobe | 171 passed | y1/t_capprobe.log |
| tests/capstubs | 83 passed | y1/t_capstubs.log |
| tests/envgate | 71 passed | y1/t_envgate.log |
| tests/grpc | 36 passed | y1/t_grpc.log |
| tests/jobs | 211 passed | y1/t_jobs.log |
| tests/judge | 97 passed | y1/t_judge.log |
| tests/launch | 225 passed（y1 首跑 7 failed：fullv 设了过长 TMPDIR，forkserver AF_UNIX path too long；不设 TMPDIR 重跑全过） | logs/y1b-launch.log |
| tests/nullsse | 4 passed | y1/t_nullsse.log |
| tests/sidecar | 5 passed | y1/t_sidecar.log |
| selftest --package capjudge --expect-samples 25 | EXIT=0，25 个样本，errors [] | logs/y1b-selftest.log |
| patchcheck --rev HEAD（探针树只读） | PASS，7 位点 / 8 处 | y1/patchcheck.json |
| py_compile | 202 / 0 | y1/summary.txt |
| bash -n | 3 / 0（capkit_inner.sh、capkit_job.sbatch、prep_tokenizer.sbatch） | y1/summary.txt |

### 作业路径 e2e（_local/b2-spb/e2e，脚本 build.py / run.sh / mk_overlay.sh）
流程：P0 覆盖清单按 WT HEAD 生成（导出目录在 WT 内）→ build.py 建假 W、登记、写计划、真 `capjobs.submit submit`（桩 runner、固定 plan id）→ 真快照里的 `jobs/capkit_job.sbatch` 与 `jobs/capkit_inner.sh`（srun 替身只去掉选项；python3 = 一次性 venv，覆盖装进它的 site-packages，PYTHONPATH 未设；submit 用 `--gate-module capkit_e2e_simgate` 包真 capenv.gate 并只忽略其退出码）→ capjobs.launches → caplaunch → capjudge（WT tools）。

| 场景 | 作业 | submit | 作业级 STATE | launches | judge | RED | INPUT_INVALID |
|---|---|---|---|---|---|---|---|
| e2e-spb-null（e0b-s1 的读者 capgrpc.load --request-format openengine 打 S-PB；作业输入 prebaked + request_bytes 由 launches 生成，pb2 = oe_double） | J9200001 | rc 0 | 无（未 ENV_INVALID），finalize ok | L0 VALID | rc 11（NOT_RUN：C1/C6/C7/C13，无 D14 等） | 0 | 无 |
| e0a-nullsse | J9200002 | rc 0 | 同上 | L0 VALID | rc 11 | 0 | 无 |
| e0b-s1-single | J9200003 | rc 0 | 同上 | L0 VALID | rc 11 | 0 | 无 |

- S-PB 场景逐档：drain ok = planned（50/150/300/600/300/150）、failed 0；每请求 21 条消息（20 token + finished），spb msgs_sent = drain msgs；C10 PASS 5/5、C3/C5/C9/C11/P-SENT/RECORDS 全 PASS。
- 真 capenv.gate 退出 6（UNREGISTERED、无 Slurm，预期）；fakebin.log 只有门禁的 scontrol/squeue（替身 exit 97），没碰集群。
- 未覆盖：真容器、真 pb2 模块、`--request-format openengine` 打真前端（1.4 补充冒烟）。
- 2026-11-14 23:10:44 暂停后重启：在 HEAD 5a0046e（与 f9246b5 只差进度日志）重做全量验证 z1（fullv2.sh：短 TMPDIR=~/tmp/20261114-capkit-sm1718/b2-spb/z1、selftest 带 --tools/--tmp；11 目录逐个 + selftest 25 + patchcheck + py_compile + bash -n，约 15 分钟）；probe-tree 软链移回 WT/probe-tree
- 2026-11-14 23:24:33 z1 结果：10 目录全绿，tests/launch 1 failed（test_full_launch_passes：末档 drain-5 的 pids_stable=False，capload 在 wend 扫描前已退出，wend 扫描比窗口末晚 38 ms；同机有其他会话的 pytest 在跑，判为时序抖动）；selftest 25 EXIT=0、patchcheck PASS、py_compile 202/0、bash -n 3/0。重跑 tests/launch（z1b，约 3 分钟）
- 2026-11-14 23:29:20 z1b tests/launch 重跑 225 passed（logs/z1b-launch.log）；单测 test_full_launch_passes 再串行跑 5 次全过（logs/z1c-rep1..5.log）。覆盖清单按 HEAD 5a0046ec 重生成（e2e/ovl/P0-5a0046ec，37 文件 sha 与 f9246b59 版逐一相同，只差 src 路径）；开跑作业路径 e2e：e2e-spb-null J9200011（约 2–3 分钟）
- 2026-11-14 23:30:40 e2e-spb-null J9200011：submit rc 0、L0 VALID、judge rc 11（NOT_RUN：C1/C6/C7/C13）、0 RED、无 INPUT_INVALID、C10 PASS；开跑回归 e0a-nullsse J9200012 与 e0b-s1-single J9200013（串行，各约 2 分钟）
- 2026-11-14 23:32 e0a-nullsse J9200012、e0b-s1-single J9200013：launches VALID、judge rc 11、0 RED、无 INPUT_INVALID。probe-tree 软链移回 _local/b2-spb/probe-tree.link。

## 暂停后重启的全量复验（HEAD 5a0046e，代码与 f9246b5 相同，只差本进度日志）

全量脚本 `_local/b2-spb/fullv2.sh z1 25`（短 TMPDIR = ~/tmp/20261114-capkit-sm1718/b2-spb/z1；每个目录 `ulimit -v 8388608` + `timeout 900` + fakebin 替身，逐个串行）。

| 项 | 结果 | 证据 |
|---|---|---|
| tests/capcommon | 190 passed | _local/b2-spb/logs/z1/t_capcommon.log |
| tests/capload | 33 passed | z1/t_capload.log |
| tests/capprobe | 171 passed | z1/t_capprobe.log |
| tests/capstubs | 83 passed | z1/t_capstubs.log |
| tests/envgate | 71 passed | z1/t_envgate.log |
| tests/grpc | 36 passed | z1/t_grpc.log |
| tests/jobs | 211 passed | z1/t_jobs.log |
| tests/judge | 97 passed | z1/t_judge.log |
| tests/launch | z1 首跑 1 failed / 224 passed（test_full_launch_passes：末档 drain-5 的 pids_stable=False，capload 在 wend 扫描前退出，扫描比窗口末晚 38 ms；同机有别的会话在跑 pytest）；重跑整目录 225 passed，单测再串行 5 次全过 | z1/t_launch.log、logs/z1b-launch.log、logs/z1c-rep1..5.log |
| tests/nullsse | 4 passed | z1/t_nullsse.log |
| tests/sidecar | 5 passed | z1/t_sidecar.log |
| selftest --package capjudge --expect-samples 25 | EXIT=0，ran 25，errors [] | z1/selftest.log |
| patchcheck --rev HEAD（探针树只读） | PASS，7 位点 / 8 处 | z1/patchcheck.json |
| py_compile | 202 / 0 | z1/summary.txt |
| bash -n | 3 / 0 | z1/summary.txt |

作业路径 e2e（覆盖清单按 HEAD 重生成：e2e/ovl/P0-5a0046ec，37 个产品文件 sha 与 f9246b59 版相同；launch.json 的 capkit_sha = 5a0046ec…）：

| 场景 | 作业 | submit | launches | judge | RED | INPUT_INVALID |
|---|---|---|---|---|---|---|
| e2e-spb-null（capgrpc.load --request-format openengine 读者打 S-PB，pb2 = oe_double） | J9200011 | rc 0 | L0 VALID | rc 11（NOT_RUN：C1/C6/C7/C13） | 0 | 0 |
| e0a-nullsse | J9200012 | rc 0 | L0 VALID | rc 11 | 0 | 0 |
| e0b-s1-single | J9200013 | rc 0 | L0 VALID | rc 11 | 0 | 0 |

- S-PB 逐档：drain ok = planned（50/150/300/600/300/150）、failed 0；spb msgs_sent = 21 × 请求数；C10 PASS（5/5）、C3/C5/C9/C11/P-SENT/P-PIDS/RECORDS 全 PASS。输出存 e2e/runs/*-J920001[123].out。
- fakebin.log 新增 9 行，都是真 capenv.gate 的 scontrol/squeue（替身 exit 97）；没碰集群。
- 2026-11-15 00:28:33 开始 R1 修复（slug capkit-b2spb-J17871，head b43d6e6）：先读码核实 C1/C2/C4/C3/C5–C17，再逐条写失败测试、修复
- 2026-11-15 00:46:07 R1 代码修复已落（C1/C2/C5+C19/C6/C7/C8/C10 测试/C14/C15/C16+C17），各自先红后绿、变异核过；接着改文档（INTERFACES 勘误、README）
- 2026-11-15 00:50:34 文档修完（INTERFACES B2B-2/3/4/7/8/9/10/11/13/14 改写、新增 B2B-15；R4-11、R5-3、R5-5 加指引；jobs/configs/tools README），提交后在新 HEAD 上跑全量验证 r1v（约 15–20 分钟）与三个作业路径 e2e
- 2026-11-15 01:07:22 r1v 全量验证完（tests/grpc 首跑 1 个时序用例 test_mode2_two_chained_relays 超界，整目录重跑 39 passed、该用例再串行 3 次全过；其余全绿）；开跑作业路径 e2e：e2e-spb-null J9200021（约 2–3 分钟，覆盖清单 P0-9dc8c002）

## R1 自审修复（slug capkit-b2spb-J17871，审阅 head b43d6e6 → 修复提交 9dc8c00）

修复清单 18 项全部落实（`code-review/self-review-compare/capkit-b2spb-J17871/R1/triage.md` 第 4 节）；不修 C18、C20、C21。每项先写测试并在旧代码上跑红（日志 `_local/b2-spb/logs/r1/*-red*.log`），修后转绿；C1、C2、C8 另做变异核对（`*-mut.log`）。

| 簇 | 改动 | 失败在先的测试 |
|---|---|---|
| C1 | `capcommon.lifecycle.inprocess_ready_info` / `inprocess_tool_info`（写读同处），`LaunchData.ready_tool_info`；C10 经它读，缺失/形状不对/无 control 键两种模式都记问题 | `tests/judge/test_c10_stub_module.py`（就绪文件用真写方生成）、`test_prebaked_server.py` 两种模式真实就绪文件喂 C10、`test_spb_job_path.py` 前端模式 |
| C2 | `caplaunch.httpprobe --started-file`（`capkit.httpprobe_started/1`）；e0s-spb-frontend 的 httpprobe 就绪改为它 | `tests/launch/test_smoke.py`、`test_e0s_configs.py` |
| C5+C19 | `load_sfe_module` 读一次、`compile` 同一份字节执行，任何异常（含 SystemExit）→ ConfigError、清 `sys.modules`；`capgrpc.reqbytes.import_pb2` 同改 | `test_prebaked_server.py`（5 种异常、不二次读）、`tests/grpc/test_reqbytes.py` |
| C6 | e0s S-PB 加 `--max-stream-records {param:spb_max_stream_records}` | `test_e0s_configs.py` |
| C7 | 指纹作业输入写死 `--trust-remote-code false --tokenizer-mode auto`，去掉两个参数；e0/e0s 重新生成；`fingerprint.py` help 改正 | `test_e0s_configs.py` |
| C8 | `capjobs.launches.run_helper`（独立会话、轮询；超时或 SIGTERM 杀进程组）；绑核表与生成器都走它；`generate(run=)` | `tests/jobs/test_job_inputs.py`（真进程发 SIGTERM、经 helper、超时杀组） |
| C14 | 场景读不到 ⇒ `skipped: scenario: …`（计数仍 `pin_table_failed`） | `tests/jobs/test_container.py` |
| C15 | 生成前失败（缺 param、输出已存在、登记输入读不到）也是 `JobInputError(records)`，保留已成功种类 | `test_job_inputs.py` 参数化 3 例 |
| C16+C17 | `capcommon.templates`（占位符、`parse_params`）、`capcommon.jobinputs.load_verified`；编排器、判定器、scenario、capjobs 改用 | `tests/capcommon/test_templates.py`；原有测试原样通过 |
| C3/C4/C9/C10/C11/C12/C13/17/18 | 文档：INTERFACES B2B-2/3/4/7/8/9/10/11/13/14 改写、新增 B2B-15；R4-11、R5-3、R5-5 指引注；jobs/README、configs/e0(s)/README、capstubs/capjudge/capcommon/caplaunch README | — |

### 全量验证 r1v（HEAD 9dc8c00；`fullv2.sh r1v 25`，短 TMPDIR）

| 项 | 结果 | 证据 |
|---|---|---|
| tests/capcommon | 193 passed | `_local/b2-spb/logs/r1v/t_capcommon.log` |
| tests/capload | 33 passed | r1v/t_capload.log |
| tests/capprobe | 171 passed | r1v/t_capprobe.log |
| tests/capstubs | 90 passed | r1v/t_capstubs.log |
| tests/envgate | 71 passed | r1v/t_envgate.log |
| tests/grpc | 首跑 1 failed / 38 passed（`test_relay_modes.py::test_mode2_two_chained_relays[2000000-500000]` 的 per_msg_ns 4.28 ms 超出上界 3.08 ms，时序用例，本轮未改 relay）；整目录重跑 39 passed；该用例再串行 3 次全过 | r1v/t_grpc.log、logs/r1/r1v-grpc-rerun.log、logs/r1/r1v-relay-rep1..3.log |
| tests/jobs | 217 passed | r1v/t_jobs.log |
| tests/judge | 98 passed | r1v/t_judge.log |
| tests/launch | 227 passed | r1v/t_launch.log |
| tests/nullsse | 4 passed | r1v/t_nullsse.log |
| tests/sidecar | 5 passed | r1v/t_sidecar.log |
| selftest --package capjudge --expect-samples 25 | EXIT=0，ran 25，errors [] | r1v/selftest.log |
| patchcheck --rev HEAD（探针树只读） | PASS，7 位点 / 8 处 | r1v/patchcheck.json |
| py_compile | 204 / 0 | r1v/summary.txt |
| bash -n | 3 / 0 | r1v/summary.txt |

### 作业路径 e2e（覆盖清单按 HEAD 9dc8c00 重生成：e2e/ovl/P0-9dc8c002，37 个产品文件 sha 与 5a0046ec 版相同）

| 场景 | 作业 | submit | launches | judge | RED | INPUT_INVALID | C10 |
|---|---|---|---|---|---|---|---|
| e2e-spb-null（capgrpc.load --request-format openengine 读者打 S-PB，pb2 = oe_double） | J9200021 | rc 0 | L0 VALID | rc 11（NOT_RUN：C1/C6/C7/C13） | 0 | 0 | PASS 5/5（含 S-PB 一项，经新读法） |
| e0a-nullsse | J9200022 | rc 0 | L0 VALID | rc 11 | 0 | 0 | PASS 8/8 |
| e0b-s1-single | J9200023 | rc 0 | L0 VALID | rc 11 | 0 | 0 | PASS 5/5 |

- fakebin.log 新增 9 行，都是真 capenv.gate 的 scontrol/squeue（替身 exit 97）；没碰集群。输出存 e2e/runs/*-J920002[123].out。
- 2026-11-15 01:09:52 R1 修复验证全部完成，记录进度并提交
- 2026-11-15 02:03:29 开始 R2 修复（slug capkit-b2spb-J17871，head cb583882cc）：C4/C5/C6（作业输入场景校验）、C1（S-PB S-FE 设置整段边界）、C2（schedule_sha256）、C10/C9（措辞）；先写失败测试再改
- 2026-11-15 02:09:28 R2 修复已写完（新测试修前 17 个失败、修后全过）；开始变异检查（约 8 个，每个 10–30 s）
- 2026-11-15 02:13:38 变异 m1–m10 全部被杀（m5、m9 首轮存活，已补断言/兜底测试后被杀）；文档改完（INTERFACES B2B-2/8/9/14/15、§5.4.1 指引、capstubs/e0s/jobs README）；开始全量验证 r2v（约 15–20 分钟）
