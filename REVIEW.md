# 自审记录（指针）

本仓库每一轮三路自审（`review-pr` 本地模式 + `/code-review` + `/security-review`，见 `../00_plan/CONSTRAINTS.md` §3 与用户全局约定）的结论**不写在这里**，只在这里留指针。原始报告与逐条比对落在：

- 比对台账根目录：`/home/scratch.yihlu_coreai/code-review/self-review-compare/<slug>/R<k>/`（`meta.json`、`review-pr.md`、`code-review.md`、`security-review.md`、`compare.md`）
- 跨轮汇总：`/home/scratch.yihlu_coreai/code-review/self-review-compare/LEDGER.tsv`

每开一轮，在下表追加一行（只追加，不改旧行）。`base` / `head` 写完整 40 位 sha；三路必须审同一个 `base...head`。

- `base`：首个提交 `Initialize capkit repository (review base)`（只含 README 与 `.gitignore`）。
- 探针树（`probe-tree/`，TRT-LLM worktree，不纳入本仓库）的补丁另按基线 `2f22a98d6e8b32ed185bcdfe051ef0e4405ae4e8...<探针树 HEAD>` 一并送审；探针树 HEAD 写在「备注」列。

| 轮 | slug | 日期 | base | head | 触发原因 | 连续轮次 | 结论指针 | 备注 |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | 尚未开审（第一批集成提交之后开 R1） |

触发原因取值：`pre-test`（投作业前）、`post-test-fix`（测试暴露问题修复后）、`post-human-request`（人工要求修改后）、`pre-push-change`。连续轮次 = 自上次重置以来第几轮（上限 3）。
