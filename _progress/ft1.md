# ft1（快速通道 FT1：S-NL、oe_sink、冒烟与 E2a/E3/E4 场景模板、产品侧注入）进度

- 子批：ft1（W/00_plan/FASTTRACK-1117-PLAN.md 快速通道；RUNBOOK-stage2-draft.md §2 子批表与 §2.0 流程）。
- WT = `_local/wt-ft1`（分支 capkit-ft1，起点 756b9e7fe6ef5fb7726b0dd3f6a81869545f184d）；C（01_code main）只读；探针树 C/probe-tree 只读（除非范围明确要求，且只在其自有分支上改）。
- 范围：S-NL 桩、oe_sink、冒烟场景模板与 E2a/E3/E4 场景模板、产品侧注入（细节按 FASTTRACK-1117-PLAN.md 与设计文档 09_experiment_design.md §12/§13）。
- 契约：WT/INTERFACES.md 末尾新增勘误节「Fast track FT1: S-NL, oe_sink, smoke and E2a/E3/E4 scenario templates, product-side injection」，被覆盖的正文句子加指引注。
- 写代码时按累计规则集 RS1–RS22（code-review/self-review-compare/capkit-J17871/R4/{brief,triage}.md）自查。
- 验证：venv 单测，每次在子 shell 内 `ulimit -v 8388608` + `timeout 900`、`python -B`、PYTHONPYCACHEPREFIX=WT/_local/ft1/pyc、fakebin 集群命令替身；测试目录逐个串行跑，不加 -x。

## 进度日志
- 2026-11-15 23:49:02 开工：建 worktree（capkit-ft1 @ 756b9e7fe6ef5fb7726b0dd3f6a81869545f184d），写本进度日志头。
