# spinners 进度（2026-11-15）

- 子批次：SB-B1 spinners —— Calibration v2 SB-B1: capenv.spinners aux tool for the C13 positive control
- 范围（DESIGN-r7a V1CD-01）：
  - 新模块 `tools/capenv/spinners.py`：按核清单起 N 个纯 Python 自旋进程，各绑一核，记录每核 busy；`--target-role <角色>` 读本启动的绑核表，把 1 个自旋进程绑到该角色的第一个核（自己调用 `sched_setaffinity`，不改绑核表生成），供 V0 的 c13-pc（C13 阳性对照）使用。
  - `caplaunch/scenario.py` 三处小改：`TOOL_TABLE` 加 `capenv.spinners`（origin/ready/pin_table）；`MANAGED_FLAGS` 加 `--pin-table`（场景手写 ⇒ 校验失败）；`build_argv` 对 `pin_table: True` 的工具追加 `--pin-table <启动树>/env/pin_table.json`。
  - 测试第 26 条（本机，不做性能测量）；不碰负载器、中继、判定器、探针树。
  - 契约：WT/INTERFACES.md 末尾新增勘误节「Calibration v2 SB-B1: capenv.spinners aux tool for the C13 positive control」，被覆盖的正文句加指针。
  - 计划依据：W/RUNBOOK-stage2-draft.md §2（子批次表与 2.0 流程）及其引用的设计文档（09_experiment_design.md，§12/§13 修订前文）；W/02_e0/E0/analysis/calib-v2/DESIGN-r7a.md（SB-B1 / V1CD-01、测试第 26 条）；W/00_plan/{PLAN,CONSTRAINTS,DATA-AND-ANALYSIS}.md；累计规则集 RS1–RS22（capkit-J17871/R4/brief.md、triage.md）。
- 基线：main @ fae19c43e7cc24f7943611064a0738f2f2bf578c（= M_bj）
- 分支 / 工作区：capkit-spinners @ `01_code/_local/wt-spinners`
- 审阅 slug：capkit-spinners-J17871
- 验证约束：无集群访问（fakebin 10 个替身 exit 97）；每次测试 `ulimit -v 8388608` + `timeout 900` + `python -B`；测试目录逐个串行跑，不加 -x；跑前 `free -g` 可用 ≥ 8 GB。

## 日志
- 2026-11-15 05:28:05 WT 已建：capkit-spinners @ fae19c43；建本进度文件
