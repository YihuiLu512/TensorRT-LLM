# ft1（快速通道 FT1：S-NL、oe_sink、冒烟与 E2a/E3/E4 场景模板、产品侧注入）进度

- 子批：ft1（W/00_plan/FASTTRACK-1117-PLAN.md 快速通道；RUNBOOK-stage2-draft.md §2 子批表与 §2.0 流程）。
- WT = `_local/wt-ft1`（分支 capkit-ft1，起点 756b9e7fe6ef5fb7726b0dd3f6a81869545f184d）；C（01_code main）只读；探针树 C/probe-tree 只读（除非范围明确要求，且只在其自有分支上改）。
- 范围：S-NL 桩、oe_sink、冒烟场景模板与 E2a/E3/E4 场景模板、产品侧注入（细节按 FASTTRACK-1117-PLAN.md 与设计文档 09_experiment_design.md §12/§13）。
- 契约：WT/INTERFACES.md 末尾新增勘误节「Fast track FT1: S-NL, oe_sink, smoke and E2a/E3/E4 scenario templates, product-side injection」，被覆盖的正文句子加指引注。
- 写代码时按累计规则集 RS1–RS22（code-review/self-review-compare/capkit-J17871/R4/{brief,triage}.md）自查。
- 验证：venv 单测，每次在子 shell 内 `ulimit -v 8388608` + `timeout 900`、`python -B`、PYTHONPYCACHEPREFIX=WT/_local/ft1/pyc、fakebin 集群命令替身；测试目录逐个串行跑，不加 -x。

## 进度日志
- 2026-11-15 23:49:02 开工：建 worktree（capkit-ft1 @ 756b9e7fe6ef5fb7726b0dd3f6a81869545f184d），写本进度日志头。
- 2026-11-16 00:01:10 FT1-snl 开始：读完设计稿 §2.3 S-NL、E3 卡、产品 servicer/server/streaming（探针树），开始写 capstubs/nullllm.py 与 null_llm_server.py（预计 1～2 h）。
- 2026-11-16 00:16:44 FT1-snl 变异 M0–M6 串行（约 5 分钟）
- 2026-11-16 00:22:52 提交 907a2e30（S-NL 代码与测试；变异 M1–M9 全被抓，M7 为等价变异已换 M7b）。开跑作业路径 e2e：覆盖清单按 HEAD 生成，e2e-snl-full / e2e-snl-mono / e0a-nullsse 回归（各约 2–3 分钟）
- 2026-11-16 00:32:56 跑 tests/capstubs 全目录（约 2–3 分钟），之后 tests/launch（约 5 分钟）；建测试用软链 WT/probe-tree（结束后改名移到 _local/ft1/probe-tree.link）
