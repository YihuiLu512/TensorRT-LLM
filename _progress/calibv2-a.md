# calibv2-a（Calibration v2 SB-A：pilot/2、统计量、阈值、判定等级、laddergen 与预算闸）进度

- 子批：calibv2-a（标定 v2 子批次 SB-A；设计 `W/02_e0/E0/analysis/calib-v2/DESIGN-r7a.md` §3 的 P2-00～P2-10，含 P2-08a/P2-08b、P2-09b、P2-09c；步骤清单 `calib-v2/impl-steps-SB-A.json`）。
- WT = `_local/wt-calibv2-a`（分支 capkit-calibv2-a，起点 fae19c43e7cc24f7943611064a0738f2f2bf578c）；C（01_code main）只读；探针树 C/probe-tree 只读（本子批范围不含探针树）。
- 范围：pilot/2 与统计量（bootstrap 阈值等）、阈值源与档位 regime、判定等级、laddergen、capjobs 预算闸 / 入选与缺额补足 / expect-node-exclude（以 DESIGN-r7a §3 各步「范围」句为验收边界，跨步改动退回）。
- 契约：WT/INTERFACES.md 末尾新增勘误节「Calibration v2 SB-A: pilot/2, statistics, thresholds, judge levels, laddergen and budget gates」，被覆盖的正文句子加指引注。
- 流程：RUNBOOK-stage2-draft.md §2.0（自测在前 → 三路自审 → 收敛后合入）；写作时套用累计规则集 RS1～RS22（SR/capkit-J17871/R4/{brief,triage}.md）。
- 验证环境：venv /home/scratch.yihlu_coreai/tmp/20261106-wt03-venv/venv；每次测试子 shell 内 `ulimit -v 8388608` + `timeout 900`、`python -B`、PYTHONPYCACHEPREFIX=WT/_local/calibv2-a/pyc、PATH 前置 10 个集群命令替身的 fakebin（exit 97）；不同时跑两个 pytest；不加 -x。

## 进度日志
- 2026-11-15 05:25:57 开工：建 worktree（capkit-calibv2-a @ fae19c43e7cc24f7943611064a0738f2f2bf578c），写本进度日志头。
- 2026-11-15 05:32:24 SB-A-1 开工：读完 impl-steps 第 1 条、DESIGN-r7a §2.2/2.3/2.6/2.7/2.10/3.1/3.5/4.5、triage §4.1、repro_b2.py；建 _local/calibv2-a（fakebin 十个替身、pyc、logs、tmp、runt.sh）。跑改前基线 tests/judge（约 3～6 分钟）
- 2026-11-15 05:43:14 stats.py 新增 P2-01 全部原语与 FU-4（t4_boundary 计 UNDETERMINED）；grade 抽出 first_non_stable，calib.build_shapes 改取等级区间下端（FU-3）。本机直跑核对数值（kmax、rank、t_isf、tol_cell 表、shortfall）全部与设计一致。下一步写测试 tests/judge/test_stats_calibv2.py 与 test_calib / test_stats 增补。
- 2026-11-15 05:46:58 测试写完（test_stats_calibv2.py 新文件；test_stats.py、test_calib.py 增补，test_calib 既有「无下端」用例改为整树无稳定档）。先单跑新/改的三个文件（约 2～4 分钟）
- 2026-11-15 05:50:34 新/改测试单跑通过（test_stats_calibv2 41、test_stats+test_calib 共 125→改正 1 处期望后全过）。后台跑变异 r1（24 项 + 未变异对照，逐个串行，约 15～25 分钟）
- 2026-11-15 06:02:59 变异 r1 完成：未变异对照 128 passed；24 项变异全部变红（_local/calibv2-a/mut/r1/summary.txt）。INTERFACES 勘误节「Calibration v2 SB-A …」与 calib-v2-A1 小节已写，§9.6、B2-2E-A-2、B2-R2 第 4 条加指引。跑全量 tests/judge（约 5～6 分钟，后台）
- 2026-11-15 06:08:57 全量 tests/judge 343 passed（改前 295）。跑 capjudge 样本自测（n=38）与 py_compile（约 3～6 分钟）
- 2026-11-15 06:10 SB-A-1 收尾（P2-00 + P2-01），实测（MEASURED）：
  - 改前基线 tests/judge 295 passed（logs/pre；注：运行期间 stats.py 已开始编辑，子进程型用例可能读到中间版本）；改后 tests/judge 343 passed（logs/post，345.8 s）；capjudge 样本自测 n=38 EXIT=0、errors []；py_compile 207/0；fakebin.log 0 行（没碰集群命令）。
  - 变异 r1（mut/r1/summary.txt）：未变异对照 128 passed；M01～M24 全部变红（FU-4 不计 UNDETERMINED、上限档改回 T4 下端、censored 改回 T4 kind、去掉等级区间核对、去掉非单调提示、tol_cell 常数 0.6 / r4 公式 / 去 Bonferroni、上界换点估计、任一候选通过、t 换正态、1.2533→1、n_eff→n_windows、rank→k、fold_rank 不缩放、补足乘数 1、q̂ 不截断、符号翻转 ≥→>、去计数下限、T5 门槛 7、ICC 上界不截 0、thr_hi 取小端、MAD=0 不回退 SD、kmax 放宽 γ）。
  - 与设计稿测试表的出入登记在 INTERFACES calib-v2-A1 第 4 条（L=20 的 z_d SD 1.08；TOL 0.1 时约 2/3 unconfirmed；(100 s, 20σ) 1/50 停在 300；r3 逐格比较代替汇总比较）。
  - 未做（NOT RUN）：第 8 条真实片段子窗部分（属 P2-12）；其余测试目录（本步只改 capjudge 与 tests/judge）。
