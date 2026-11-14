# b2-judge（第二批 2A+2E：标定文件、判定器阶段 2 检查、定位 I、干预 V、等级与 E1 分析）进度

- 子批：b2-judge（RUNBOOK-stage2-draft.md §2.1 子批表第 1 行「2A 标定格式与构建器 + 判定器阶段 2 检查」与第 2 行「2E 定位 I + 干预 V + 等级（核心）+ E1 分析」，按 §2.2 第 3 条两者都在 capjudge 里，合在一个分支叠着写）。
- WT = `_local/wt-b2-judge`（分支 capkit-b2-judge，起点 1176861dd1d8233b8de1063c2e620fe9aee21e3f）；C（01_code main）只读；探针树 C/probe-tree 只读。
- 范围（2A）：+ `tools/capjudge/calib.py`（loadgen_band、stub_calib 域外段、near_boundary_noise 的构建与 schema 校验）；+ `analysis/calib_build.py`（薄 CLI，同 `pilot_build.py`）；~ `capjudge/checks.py`（C2 域外桩部分、C4、C12）；~ `capjudge/inputs.py`（C2/C4/C12 移出 `BATCH2_CHECKS`；C1 改读 loadgen_band）；~ `capjudge/thresholds.py`（按档位 regime 选带：近边界档优先取 near_boundary_noise，同一个键允许两份来源各管一个 regime）；~ `samples/make_samples.py`（每项一个造坏样本）；+ `tests/judge/test_calib.py`。
- 范围（2E）：+ `capjudge/localize.py`（I1 按 GIL 域求和，以每核 `/proc/stat` 为准；I3 按积压所在一侧判）；+ `capjudge/intervene.py`（跨启动剂量–反应：先核心跳计数，再看单调下移是否超出噪声带，做反向注入）；+ `capjudge/grade.py`（P+X+I+V ⇒ 等级）；~ `symptoms.py`（接 I/V 钩子；`reclassify_failure` 认负载器的连接类失败）；~ `judge.py`；+ `analysis/boundary.py`（每启动 T4、包络、T5）；+ `analysis/scaling.py`（T6、对侧翻倍归因）；+ `analysis/e1_report.py`（斜率对账、字节 / accepts 相等、空汇健康）；每个产物写 `.cmd.txt`。
- 契约：WT/INTERFACES.md 末尾新增勘误节「Batch 2A+2E: calibration files, stage-2 judge checks, localization, intervention, grading and E1 analysis」，被覆盖的正文句子加指引注；涉及 §9.6–9.8、§10.2–10.4、新增 §14「分析脚本输入输出」。
- 验证：判定器全量测试、样本自测（含「读者慢 / 服务端慢」一对、`sat_spin_thread_same_process` 判出「域到顶」）；e2e：E0 第一轮真树上重判（C12 为 PASS 或 NOT_RUN，不报错），E0(b) s1～s4 真树跑 localize（s3 在辅助线程侧判出）；均为本地读树，不占集群。

## 进度日志
- 2026-11-14 20:15:11 开工：建 worktree（capkit-b2-judge @ 1176861dd1d8233b8de1063c2e620fe9aee21e3f），写本进度日志头。
