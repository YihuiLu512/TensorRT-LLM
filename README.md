# 01_code：承载上限测试的新代码（capkit）

本目录是只在本地的 git 仓库（提交身份 `Yihui Lu <yihlu@nvidia.com>`，不设 remote，不推送）。
第一个提交只含本文件与 `.gitignore`，作为每一轮三路自审的 base。

- 计划与约束：`../00_plan/`（先读 `../00_plan/README.md`）
- 实验设计定稿：`/home/scratch.yihlu_coreai/dev/feat-dev-J17503/tmp/meta/explain/06_arch-perf-capacity/_src/research/09_experiment_design.md`（§2、§7、E0/E1 卡）
- 接口约定：`INTERFACES.md`（各组件之间的文件格式、目录树、参数约定）
- 自审记录指针：`REVIEW.md`
- 进度：本文件末尾「进度表」；更细的事件流水在 `../STATUS.md`

## 布局

见 `../00_plan/DATA-AND-ANALYSIS.md` 1.1 节：`tools/`、`configs/`、`analysis/`、`jobs/`、`samples/`、`probe-tree/`（TRT-LLM 探针树，git worktree，不纳入本仓库）、`probe-tree-glm/`（同上）。

## 进度表

| 件 | 批 | 状态 | 说明 |
|---|---|---|---|
| 仓库骨架 | — | 完成 | 首个提交 = 自审 base |
