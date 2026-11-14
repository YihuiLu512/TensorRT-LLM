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
