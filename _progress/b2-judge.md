# b2-judge（第二批 2A+2E：标定文件、判定器阶段 2 检查、定位 I、干预 V、等级与 E1 分析）进度

- 子批：b2-judge（RUNBOOK-stage2-draft.md §2.1 子批表第 1 行「2A 标定格式与构建器 + 判定器阶段 2 检查」与第 2 行「2E 定位 I + 干预 V + 等级（核心）+ E1 分析」，按 §2.2 第 3 条两者都在 capjudge 里，合在一个分支叠着写）。
- WT = `_local/wt-b2-judge`（分支 capkit-b2-judge，起点 1176861dd1d8233b8de1063c2e620fe9aee21e3f）；C（01_code main）只读；探针树 C/probe-tree 只读。
- 范围（2A）：+ `tools/capjudge/calib.py`（loadgen_band、stub_calib 域外段、near_boundary_noise 的构建与 schema 校验）；+ `analysis/calib_build.py`（薄 CLI，同 `pilot_build.py`）；~ `capjudge/checks.py`（C2 域外桩部分、C4、C12）；~ `capjudge/inputs.py`（C2/C4/C12 移出 `BATCH2_CHECKS`；C1 改读 loadgen_band）；~ `capjudge/thresholds.py`（按档位 regime 选带：近边界档优先取 near_boundary_noise，同一个键允许两份来源各管一个 regime）；~ `samples/make_samples.py`（每项一个造坏样本）；+ `tests/judge/test_calib.py`。
- 范围（2E）：+ `capjudge/localize.py`（I1 按 GIL 域求和，以每核 `/proc/stat` 为准；I3 按积压所在一侧判）；+ `capjudge/intervene.py`（跨启动剂量–反应：先核心跳计数，再看单调下移是否超出噪声带，做反向注入）；+ `capjudge/grade.py`（P+X+I+V ⇒ 等级）；~ `symptoms.py`（接 I/V 钩子；`reclassify_failure` 认负载器的连接类失败）；~ `judge.py`；+ `analysis/boundary.py`（每启动 T4、包络、T5）；+ `analysis/scaling.py`（T6、对侧翻倍归因）；+ `analysis/e1_report.py`（斜率对账、字节 / accepts 相等、空汇健康）；每个产物写 `.cmd.txt`。
- 契约：WT/INTERFACES.md 末尾新增勘误节「Batch 2A+2E: calibration files, stage-2 judge checks, localization, intervention, grading and E1 analysis」，被覆盖的正文句子加指引注；涉及 §9.6–9.8、§10.2–10.4、新增 §14「分析脚本输入输出」。
- 验证：判定器全量测试、样本自测（含「读者慢 / 服务端慢」一对、`sat_spin_thread_same_process` 判出「域到顶」）；e2e：E0 第一轮真树上重判（C12 为 PASS 或 NOT_RUN，不报错），E0(b) s1～s4 真树跑 localize（s3 在辅助线程侧判出）；均为本地读树，不占集群。

## 进度日志
- 2026-11-14 20:15:11 开工：建 worktree（capkit-b2-judge @ 1176861dd1d8233b8de1063c2e620fe9aee21e3f），写本进度日志头。
- 2026-11-14 20:24:46 2A-calib-checks 开工：已读设计 §2.6/§2.7/E1/E2、INTERFACES §9/§10/勘误、capjudge 全部相关代码；方案：按档 regime 选带（登记表 judge.near_boundary_tiers）、loadgen_band/stub_calib 用 shapes + 登记表 bind、C1 改读 loadgen_band、C2 域外、C4、C12。先写 thresholds/calib/inputs/checks，再样本与测试（预计 2–3 h）。
- 2026-11-14 20:36:44 代码写完（thresholds regime、calib.py 校验/统计/构建、inputs bind/regime、checks C1 改读 loadgen_band + C2/C4/C12、样本 +8 个共 33）；新建样本自测 33/33 过（_local/b2-judge/st1）。下一步写 tests/judge/test_calib.py、改旧测试、重生成提交的样本（约 1 h）。
- 2026-11-14 20:39:39 e2e：拷 E0 第一轮两棵树（e0a J5935100-L1、e0b J5935100-L0）到 _local/b2-judge/e2e 并建最小 wd，用修改后的判定器重判（约 10 分钟）
- 2026-11-14 20:42:30 文档（INTERFACES §9.6–9.8 + 勘误 B2-2A、两份 README）写完；跑 tests/judge 全量 + 提交样本自测（约 6 分钟）
- 2026-11-14 20:46:58 变异检查 M1–M10（_local/b2-judge/mut，每个跑 test_calib.py，约 5 分钟）
- 2026-11-14 20:50:09 变异 M11/M12（样本自测层，约 6 分钟）
- 2026-11-14 20:59:14 2A-calib-checks 验证完成（MEASURED）：tests/judge 157 passed（_local/b2-judge/logs/j2.log）；提交样本自测 33/33 EXIT=0（logs/st3.log）；变异 M1–M12 全部 KILLED（mut/summary.txt；M10 首轮存活后补了按时间表份额的 C1 测试再杀死）；e2e：E0 第一轮 e0a J5935100-L1、e0b J5935100-L0 拷贝重判，CLI 退出 11，checks.tsv 前六列除 e0a 档 4 C7（SM-14 修复）外不变；进程内加 C2/C4/C12：C2 NOT_RUN（无 stub_calib）、C4 PASS、C12 PASS/NOT_RUN（e0a 档 4 nullsse worker 1 本档 0 请求 ⇒ NOT_RUN），无报错；fakebin.log 0 行。提交。
- 2026-11-14 21:09:06 2E-localize-intervene-grade 开工：已读设计 §2.6/§2.7、DATA §4/§6、INTERFACES §5/§7/§10/勘误、capjudge 现有代码。方案：judge.mode 新增 P+X+I+V；localize.py（I1 单核=核空闲、多核=GIL 域求和+运行队列、非 GIL 线程单列，阈值 pilot.i1_calibration + T2 带；I3 按登记的 judge.localize 读数的接收/发送方判积压所在一侧）；intervene.py（逐档心跳 + 跨启动剂量–反应 CLI）；grade.py（逐启动等级 + 合并 V 的最终等级）；reclassify_failure 认负载器连接类失败；样本 +3（i_server_slow、i_reader_slow、i_conflict），sat_spin 改 P+X+I+V 判出域到顶。先写代码（约 1.5 h），再测试与样本。
- 2026-11-14 21:19:47 2E 代码与新测试写完（localize/intervene/grade、symptoms 钩子、inputs 模式与 localize 校验、pilot i1_calibration；样本 +3 共 36，sat_spin 改 P+X+I+V）；新测试 test_localize 16、test_intervene 47 过。跑 tests/judge 全量（约 5 分钟）
- 2026-11-14 21:25:27 tests/judge 220 passed（j3 一处测试替身缺 roles 已补，j4 复跑过）；样本自测 36/36（st5）；tests/capcommon 190 passed。开始变异 N1–N13（mut/run2.py，约 10 分钟）
- 2026-11-14 21:32:56 e2e：E0(b) J5933747 L1–L4（s1–s4）raw 拷到 _local/b2-judge/e2e/trees2，进程内以 P+X+I+V 跑 localize（诊断用阈值 0.90，非登记值；约 5 分钟）
- 2026-11-14 21:38:31 改 I1：统计量不论 pilot 有无门槛 / D14 都记录（单核分不含 / 含软中断两键，按 D14 取其一），样本重生成；复跑 tests/judge 全量（约 5 分钟）
- 2026-11-14 21:47:27 2E-localize-intervene-grade 验证完成（MEASURED）：tests/judge 220 passed（logs/j5.log）；样本自测 36/36（logs/st6.log）；tests/capcommon 190 passed（logs/cc1.log）；变异 N1–N14 全部 KILLED（mut/summary2.txt）；e2e：E0(b) J5933747 L1–L4 进程内 P+X+I+V（诊断门槛 0.90，非登记值）：s1/s2/s3 档 6 relay1 I1 成立（核忙碌 0.996–0.998），s3 线程份额 capkit-inject-aux-relay.msg 0.915 核、MainThread 0.046 核；s2 relay2 不成立；s4（sleep）各档 I1 不成立；X 因 E0 第一轮无带为 NOT_RUN；fakebin.log 不存在（0 次调用）。提交。
- 2026-11-14 21:57:14 2E-analysis 开工：已读设计 §2.6 T4–T6、E1 卡、E0(b)、DATA §4/§6、INTERFACES §10/勘误 B2-2E、stats/intervene/grade。注意 STATUS 21:42 用户要求 1 小时内暂停（b2-judge 到点冻结 diff），本步按优先级：boundary.py → scaling.py → §14/勘误 → sm9.md → e1_report.py，到点前提交完整一致的部分。
- 2026-11-14 21:59:53 跑 tests/judge/test_analysis.py（约 2 分钟）
- 2026-11-14 22:00:12 e2e：E0(b) J5933747 L1–L4 进程内 localize 复跑（loc4，诊断门槛 0.90，后台串行约 5 分钟），供 sm9.md
- 2026-11-14 22:01:13 e2e：capjudge.boundary 对 E0 第一轮 15 棵已判真树（W 只读，产物写 _local/b2-judge/e2e/an）
- 2026-11-14 22:03:30 boundary/scaling/cmdrec + §14.1/14.2 + 勘误 B2-2E-A 1–6 写完；test_analysis 18 passed；变异 M1–M7 全杀（mut/summary3.txt，M6 首轮存活后补测试）；E0(b) 12 棵真树 boundary e2e EXIT=0；sm9.md 写完。先提交，再写 e1_report。
- 2026-11-14 22:05:18 e2e：e0a 3 棵已判真树跑 boundary + e1_report（字节 / accepts 对账，W 只读）
- 2026-11-14 22:07:53 e2e：E1-0 15 棵已判真树（W 只读）按轴分两组跑 boundary + e1_report
- 2026-11-14 22:08:07 跑 tests/judge 全量（约 5 分钟）
- 2026-11-14 22:11:53 跑 tests/capload（sink.py 抽出 response_head，约 3 分钟）
- 2026-11-14 22:13:18 样本自测 36 个（st7，约 3 分钟）
- 2026-11-14 22:13:58 2E-analysis 验证完成（MEASURED）：tests/judge 247 passed（logs/j6.log）；tests/capload 33 passed（logs/cl1.log）；样本自测 36/36（logs/st7.log）；变异 M1–M7（summary3）、E1–E6（summary4）全部 KILLED；e2e（W 只读，产物 e2e/an/）：E0(b) 12 树 boundary EXIT=0；e0a 3 树、E1-0 15 树 boundary+e1_report EXIT=0，字节对账全部 PASS（HTTP 重建头后相等、gRPC 直接相等），HTTP accepts 全部 PASS，gRPC accepts NOT_RUN，T4 全为 none、斜率 NOT_RUN（判定时无带、无剂量）；localize 复跑 loc4 与 loc3 去 pid 后逐行相同；SM-9 判读写在 _local/b2-judge/sm9.md；fakebin.log 不存在（0 次调用）。提交。
- 2026-11-14 22:15:22 补：counts 要求两侧每个档目录每个 worker 都有摘要（缺 ⇒ NOT_RUN），测试 28 passed（an4），变异 E7 KILLED（summary5）；E1-0 与 e0a e2e 复跑（*.r3.json）结论不变。提交。
- 2026-11-14 22:18:02 全量验证 v1 开工（HEAD d3ff38e002c40540af823b058b083d20e16e3a08）：脚本 _local/b2-judge/fullv.sh（改自 fix-sm1718/fullv.sh），10+ 个测试目录逐个串行跑 + selftest 36 + patchcheck + py_compile + bash -n，约 15–20 分钟；建测试用软链 WT/probe-tree -> ../../probe-tree（被 git 忽略，结束后移到 _local/b2-judge/probe-tree.link）
- 2026-11-14 23:09:48 重启（暂停后）：v1 全量验证中断作废（停在 tests/judge），从头重做为 v2；fullv.sh 的 TMPDIR/basetemp 改到短路径 /home/scratch.yihlu_coreai/tmp/20261114-capkit-sm1718/b2-judge/<轮次>（原件留 fullv.sh.v1）；软链 probe-tree 恢复到 WT 根（未入库）。顺序：v2 全量验证（约 15–20 分钟）→ E0 第一轮 J5933747-r0-L0..L4 拷贝重判 + boundary + localize → e0a-nullsse 作业路径 e2e。
