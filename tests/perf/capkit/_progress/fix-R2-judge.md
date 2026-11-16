# FX2-judge 进度（R2 修复）

- 2026-11-13 开工：建 fakebin（_local/fx2-judge/fakebin）、读 triage/brief。
- 22:5x 读完 judge/d14/pilot/checks；Q12 核实：判定器不读 tier.json snapshot_copies 的 snapshot_missing（RS1 读方缺失），准备修。
- 基线：跑 tests/judge（修前，约 2-5 分钟）
- 基线 tests/judge：61 passed（exit 0）。
- Q1 修前复现（MEASURED，_local/fx2-judge/q1/pre/）：真实 e2e 树 W-local-r1b-b 拷贝，撤掉 judge/ 与台账行、STATE 只留 caplaunch VALID 再追加 `VOID capjobs.launches pids survived SIGKILL (simulated)`，跑 capjudge → exit 11，STATE 末行 `NOT_RUN capjudge`；pilot 照收该树 tiers_used [2,4,6,8]。
- 发现：E0 树的判定器结论天然是 NOT_RUN（无噪声带），triage「pilot 只收启动级结论 VALID」字面照做会让 pilot 在 E0 无树可用 ⇒ 改为 pilot 只收「上游（caplaunch/capjobs 的 STATE 与 launch_end）结论 VALID」的树，判定器自己的 NOT_RUN 不排除。
- 设计：inputs.verify 新增 `_load_upstream`（读全部非 capjudge 的 STATE 行 + launch_end.state + leftovers.after，取最差；没有上游行 ⇒ INPUT_INVALID）；judge_tree 新增启动级检查行 LAUNCH-STATE（VALID→PASS、NOT_RUN→NOT_RUN、VOID/ENV_INVALID→RED）；写 STATE 取 max(上游, 判定)。
- 已改：treedata 共用窗口扫描（Q16/Q7）、C13 只扣 softirq（Q4）、D14 按作业运行计数（Q6）、LAUNCH-STATE 与 STATE 取较差（Q1）、judge/ 存在时补写（Q3）、GPU-IDLE 本启动进程（Q5）。下一步：Q12 RECORDS 读 snapshot_missing、pilot 排除上游非 VALID 树、样本。
- 新样本自测：修后 25/25 无错；HEAD 版判定器跑新样本 → upstream_void、probe_entry_unmatched 报错（修前会红，MEASURED）。Q1 修后同一实验 STATE 末行 VOID、pilot 拒收（exit 3 no tier survived）。真实 e2e 两棵树修后判定与修前逐档一致（只多 LAUNCH-STATE PASS），pilot 噪声带与 D14 读数不变。下一步写单测。
- 22:xx 用新生成器重生成 samples/**（覆盖同名文件，新增 STATE 与两组新样本，无删除）
- 单测已加（judge_once 8、checks_rows 11、pilot 5），README 同步；跑全量 tests/judge
- 收尾（MEASURED）：tests/judge 85 passed exit 0；tests/capcommon 188 passed exit 0；仓库内 samples 自测 run_selftest --expect-samples 25 → 25 个样本、0 错误、exit 0；新测试对 HEAD 版判定器：judge_once 7 红、checks_rows 7 红、pilot 5 红、selftest 2 个新样本红。Q1 最终复验：真实树追加 VOID 后判定 exit 10，STATE 末行 VOID，再判 12，pilot exit 3（树被剔除）。fakebin.log 为空；无残留进程。
- 交主会话：INTERFACES 勘误原文、STATE 排序是否下沉 capcommon.runtree、FX2-launch 场景校验（产品角色开探针组时 snapshots 不得为 none）、Q4 内核 IRQ 时间记账未核。
