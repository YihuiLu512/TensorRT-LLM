# calibv2-a（Calibration v2 SB-A：pilot/2、统计量、阈值、判定等级、laddergen 与预算闸）进度

- 子批：calibv2-a（标定 v2 子批次 SB-A；设计 `W/02_e0/E0/analysis/calib-v2/DESIGN-r7a.md` §3 的 P2-00～P2-10，含 P2-08a/P2-08b、P2-09b、P2-09c；步骤清单 `calib-v2/impl-steps-SB-A.json`）。
- WT = `_local/wt-calibv2-a`（分支 capkit-calibv2-a，起点 fae19c43e7cc24f7943611064a0738f2f2bf578c）；C（01_code main）只读；探针树 C/probe-tree 只读（本子批范围不含探针树）。
- 范围：pilot/2 与统计量（bootstrap 阈值等）、阈值源与档位 regime、判定等级、laddergen、capjobs 预算闸 / 入选与缺额补足 / expect-node-exclude（以 DESIGN-r7a §3 各步「范围」句为验收边界，跨步改动退回）。
- 契约：WT/INTERFACES.md 末尾新增勘误节「Calibration v2 SB-A: pilot/2, statistics, thresholds, judge levels, laddergen and budget gates」，被覆盖的正文句子加指引注。
- 流程：RUNBOOK-stage2-draft.md §2.0（自测在前 → 三路自审 → 收敛后合入）；写作时套用累计规则集 RS1～RS22（SR/capkit-J17871/R4/{brief,triage}.md）。
- 验证环境：venv /home/scratch.yihlu_coreai/tmp/20261106-wt03-venv/venv；每次测试子 shell 内 `ulimit -v 8388608` + `timeout 900`、`python -B`、PYTHONPYCACHEPREFIX=WT/_local/calibv2-a/pyc、PATH 前置 10 个集群命令替身的 fakebin（exit 97）；不同时跑两个 pytest；不加 -x。

## 进度日志
- 2026-11-15 05:25:57 开工：建 worktree（capkit-calibv2-a @ fae19c43e7cc24f7943611064a0738f2f2bf578c），写本进度日志头。
