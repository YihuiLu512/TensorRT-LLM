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
