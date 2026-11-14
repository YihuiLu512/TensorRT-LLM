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
- 2026-11-15 05:37:08 已读设计 V1CD-01、scenario.py、orchestrator（绑核表拷贝、进程登记、版本自报、监测类子进程）、judge C9/C10/C13。设计决定：自旋子进程不进就绪文件 children（登记了就要版本记录，否则 C10 NOT_RUN）；角色须 measure_class observer（子进程进 monitor_children、不进 unattributed）；输出 raw/spinners/<角色>.json。开始写 tools/capenv/spinners.py
- 2026-11-15 05:41:28 spinners.py、scenario.py 三处、tests/envgate/test_spinners.py 已写；跑 tests/envgate/test_spinners.py（约 1 分钟）
- 2026-11-15 05:43:37 envgate/test_spinners 25 passed、launch/test_scenario 57 passed；加了启动 e2e 两条，跑这两条（约 1 分钟）
- 2026-11-15 05:44:28 跑变异 M0–M11（每项复制 tools/tests/configs 后跑相关测试，约 3–5 分钟）
- 2026-11-15 05:49:13 变异 M0–M11 全部按预期（_local/spinners/mut/summary.txt）；INTERFACES 勘误 SB-B1 与 4 处正文指针、capenv README 已写。开始逐个跑测试目录 tests/envgate、tests/launch（各约 1–4 分钟）
- 2026-11-15 05:53:23 tests/launch v1：11 failed，原因是环境（TMPDIR 太深 ⇒ forkserver AF_UNIX path too long；缺 probe-tree）。TMPDIR 改 /tmp/spnb1，建测试用软链 WT/probe-tree -> ../../probe-tree（提交前移走）。重跑 tests/launch（约 3 分钟）
- 2026-11-15 06:02:28 tests/launch v2：197 passed / 2 failed（snapshot_child 与 gate_stalls，均为时序类；单跑：stalls 过，snapshot_child WT 2/14 败、基线副本 0/14 败，见 logs/v1/snapchild_cmp.txt；败因是快照 seq 未推进，与本改动无代码路径）。跑 tests/jobs、tests/sidecar（约 3–5 分钟）
- 2026-11-15 06:05 tests/sidecar 5 passed、tests/jobs 221 passed；tools+tests 200 个 .py 编译 0 错；fakebin.log 0 行。测试用软链 probe-tree 已移到 _local/spinners/probe-tree.link（未删）。准备提交 V1CD-01。

## V1CD-01 结果

### 改动
| 处 | 内容 |
|---|---|
| + `tools/capenv/spinners.py` | 监督进程（不自旋，绑 `--cpus`）＋每个 `--cpus` 核一个 fork 的纯 Python 自旋进程；`--target-role` 读 `--pin-table` 另起一个绑到该角色第一个核；全部报回亲和性读回后才写就绪文件（`children: []`）；SIGTERM → 宽限后 SIGKILL，第二次信号立即 SIGKILL；自旋进程每 64 单位查父进程；输出 `raw/spinners/<角色>.json`；退出码 0/3/4 |
| ~ `tools/caplaunch/scenario.py` | `TOOL_TABLE` 加 `capenv.spinners`（origin/ready/pin_table）；`MANAGED_FLAGS` 加 `--pin-table`；`build_argv` 对 `pin_table` 组追加 `RunTree(<T>).env("pin_table.json")` |
| ~ `tests/launch/fakepkgs/capenv/__init__.py` | 假 capenv 包后面接上真实 `tools/capenv`（只有 canary、sidecar 是假的） |
| + `tests/envgate/test_spinners.py`；~ `tests/launch/test_scenario.py`、`test_launch_e2e.py` | 测试第 26 条及失败路径、启动 e2e |
| ~ `INTERFACES.md`、`tools/capenv/README.md` | 勘误节「Calibration v2 SB-B1」（6 条）＋ 正文 4 处指针 |

### MEASURED
- `tests/envgate` 96 passed（rc 0）；`tests/launch` 197 passed / 2 failed（见下）；`tests/sidecar` 5 passed；`tests/jobs` 221 passed；日志 `_local/spinners/logs/v1/`。
- 变异 M0–M11（`_local/spinners/mut/summary.txt`）：未变异 82 passed；11 个变异各自让对应测试变红。
- 启动 e2e（真实 capenv.spinners）：rc 0、STATE VALID、`--pin-table` = `<T>/raw/env/pin_table.json`、目标核 = product 的第一个核、自旋进程全部进 monitor_children、unattributed 为空、各档 pids_stable、无残留；非监测类反例：自旋进程进 unattributed。
- `tests/launch` 两条失败：`test_gate_survives_orchestrator_stalls` 单跑通过；`test_tool_role_snapshot_child_is_unattributed` 单跑 WT 14 次败 2 次、基线副本（fae19c43 的 tools/tests/configs）14 次败 0 次，败因是清空门「快照 seq 未推进」（`_local/spinners/logs/v1/snapchild_cmp.txt`）。

### INFERRED
- 上述两条失败与本改动无关：这两个测试不起 capenv 的任何模块，改动在它们路径上只有 scenario.py 多 import 了 RunTree；同期另一会话在跑 pytest（负载约 7），测试按亲和性取前几个核，会与别的会话的测试抢核。没有找到因果路径，但 WT/基线的失败次数差异未被完全解释。
- c13-pc 上 C9 不会因自旋进程判 RED（自旋进程未登记，C9 只看登记进程）；设计稿说 C9 RED 预期但不计入判据，判据不受影响。

### NOT RUN
- 作业路径 e2e（capjobs submit → capkit_inner.sh → caplaunch → capjudge）与 capjudge 对含自旋器启动树的判定（C13 是否真的 RED）：本机不做性能测量，留给 V0 的 c13-pc。
- 三路自审（本步只实现，按 2.0 流程由子批次后续步骤做）。
- 2026-11-15 06:05:54 全量验证开工（HEAD f1dadc262f）：建 _local/spinners/fullv.sh（从 b2-judge 拷来改路径；TMPDIR=/home/scratch.yihlu_coreai/tmp/20261114-capkit-sm1718/spinners/<轮>）与 e2e/jobpath；逐目录跑 tests（共约 15–25 分钟）
- 2026-11-15 06:06:14 软链 probe-tree.link 移回 WT/probe-tree（测试用，提交前再移走）；后台跑全量验证 v2（fullv_all.sh v2，约 15–25 分钟）
- 2026-11-15 06:18:54 v2 进行中：capcommon 190、capload 33、capprobe 159、envgate 96、grpc 53、jobs 221、judge 295 全过；正在跑 tests/launch（约 5–8 分钟）。e2e 脚本已备好（_local/spinners/e2e/jobpath：build.py 加「<base>+spin:<目标角色>」场景变体、run.sh 加自旋器绑定核对），测试跑完再跑
- 2026-11-15 06:22:26 全量验证 v2 全绿（HEAD f1dadc26，logs/v2/summary.txt）。开跑作业路径 e2e：e0a-nullsse（J8530001）、e0a-nullsse+spin:nullsse（J8530002），各约 1–3 分钟
- 2026-11-15 06:5x 作业路径 e2e 完成（见下）；pgrep 无残留；测试用软链已移回 _local/spinners/probe-tree.link（未删）。

## V1CD-01 全量验证（HEAD f1dadc262f47da6cb09d16a97eba2336957440c5）

### MEASURED
| 项 | 结果 |
|---|---|
| tests/capcommon | 190 passed |
| tests/capload | 33 passed |
| tests/capprobe | 159 passed |
| tests/envgate | 96 passed |
| tests/grpc | 53 passed |
| tests/jobs | 221 passed |
| tests/judge | 295 passed |
| tests/launch | 199 passed（这一轮 snapshot_child / gate_stalls 两条也过了，SM-21 未复现） |
| tests/nullsse | 4 passed |
| tests/sidecar | 5 passed |
| samples/run_selftest.py --package capjudge --expect-samples 38 | EXIT 0，ran 38，errors 0 |
| patchcheck --rev HEAD（探针树只读） | PASS，7/7 位点、8/8 位置、0 违例 |
| py_compile | 208 ok / 0 bad |
| bash -n | 3 ok / 0 bad（capkit_inner.sh、capkit_job.sbatch、prep_tokenizer.sbatch） |
| 作业路径 e2e e0a-nullsse（J8530001） | build submit_rc 0、inner/launches rc 0、STATE VALID、judge rc 11（NOT_RUN）、RED 0、INPUT_INVALID 0、pyc 0；非 PASS 集合与 b2-judge 的 J8520003 逐行相同 |
| 作业路径 e2e e0a-nullsse+spin:nullsse（J8530002，加一个 capenv.spinners aux 角色 `--target-role nullsse`） | STATE VALID、launches rc 0、各角色退出码 0；launch.json 里 spin 的 `--pin-table` = 本次启动树的 raw/env/pin_table.json；pin_table：spin [20]、nullsse [14, 15]；spin.json ready、problems 空、target_cpu 14；自旋进程 own cpu 20 亲和性 [20]、target cpu 14 亲和性 [14]，都已报告，last_cpu 分别 20、14；读入的 pin table sha 与拷回的一致；unattributed 空、自旋进程都在 monitor_children；judge rc 11、RED 0，非 PASS 集合与 J8530001 相同（C9 三档 PASS） |
| fakebin.log | 2 行，都是门禁的 `scontrol show job 853000x`（替身退出 97，模拟里按设计忽略），无真实集群调用 |

日志：`_local/spinners/logs/v2/summary.txt`；e2e：`_local/spinners/e2e/jobpath/*.run.log`。

### INFERRED
- 本地模拟里含自旋器的启动 C13 仍是 NOT_RUN（pilot.json 没有 D14 结果），所以这次 e2e 不说明 C13 阳性对照会不会判 RED；那要在集群上的 c13-pc 看。

### NOT RUN
- 集群上的 c13-pc 阳性对照、任何性能测量。
