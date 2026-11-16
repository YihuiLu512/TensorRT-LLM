# fix-R1 FX-judge 进度

- 2026-11-13 开工。负责簇：K5、K7（checks 侧）、K13、K20；RS2 样本对齐。独占文件：`tools/capjudge/**`、`samples/**`、`tests/judge/**`。
- fakebin 已建：`_local/fx-judge/fakebin/`（10 个脚本，argv 追加到 `_local/fx-judge/fakebin.log` 后 exit 97）。

## 进展日志

- 已读 triage/brief、R5/R7/R13/B09 原文、INTERFACES §10、judge.py、checks.py GPU-IDLE、run_selftest.py、make_samples.py。
- K20 实证线索：`_local/e2e-20261113-195336-nullsse` 的启动树已有 `judge/`（只有 inputs.json=INPUT_INVALID + judge.log），JUDGED.tsv 不存在——正是 B09 场景，该树此后只能 ALREADY_JUDGED。
- 注意：INTERFACES §10.1 原文把「INPUT_INVALID 后 judge/ 已存在、不能再判」写成了约定；本次按 triage + 用户补充改为「失败尝试落 judge.attempts/，judge/ 只在判完后原子改名建立」，需主会话出 INTERFACES 勘误（原文见文末）。
- RS2 比对已跑（脚本 scratchpad/rs2_compare.py，只读；对象：samples/good 的树 vs `_local/e2e-20261113-195834-nullsse`、`_local/e2e-20261113-200042-relay` 的启动树）。初步结果见下文「RS2」节（稍后整理）。
- 发现跨组件问题：按 K6 口径请求 0 张 GPU 时，判定器 GPU-IDLE 恒 NOT_RUN（必检项）⇒ 启动恒 NOT_RUN；交主会话。
- 20:43 跑基线 tests/judge（改动前），预计 1–3 分钟
- 20:44 已改 checks.py：K5/K7（GPU-IDLE 读 gpu_every_ns、不可读行不计 checked）、额外 C3 应检/实检对账、额外 C8 直方图槽求和。开始改 judge.py（K20）。
- 20:46 judge.py（K20）与 run_selftest.py（K13 + K20 断言）已改；开始改 make_samples.py（RS2 对齐 + K5 周期 2 s + K7 新样本）
- 20:49 新增 tests/judge/test_checks_rows.py、test_judge_once.py；test_selftest.py 改 N_SAMPLES=23 并加 K13 用例。跑 tests/judge（约 1 分钟）
- 20:50 样本已重新生成并覆盖到 samples/（1430 个文件改动 + 新样本 gpu_util_unreadable，无残留旧文件）。跑 tests/judge 与 run_selftest
- 20:51 修前对照：hybrid 树 /home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02/01_code/_local/fx-judge/hybrid（capjudge 与 run_selftest.py 取 HEAD，其余为当前），跑新测试
- 20:52 真实 e2e 树副本重判：e2e4/e2e7 仍 11（与原判一致），e2e2 INPUT_INVALID 只留 attempt、可再判（rc 13），已判树第二次 rc 12。开始改 README
- 20:54 终跑 tests/judge 与 run_selftest --expect-samples 23

---

# 汇总（收尾时写）

## 逐簇结论

| 簇 | 核实 | 证据 | 改动 | 新增测试 | 修前会红 |
|---|---|---|---|---|---|
| K5 | 属实 | HEAD `checks.py` 应检 = `window_ns // NS × gpu_count`，不读 sidecar `meta.json` 的 `gpu_every_ns`；新 good 样本（周期 2 s）用 HEAD 判定器判出 11（档 2/4 NOT_RUN） | `tools/capjudge/checks.py:525-575`（`check_gpu_idle`：应检 = Σ(window_ns // gpu_every_ns) × gpu_count，`gpu_every_ns` 缺失/≤0 ⇒ NOT_RUN） | `test_checks_rows.py::test_expected_rows_follow_the_sidecar_gpu_period`、`::test_missing_gpu_period_is_not_run`；good 样本改为 2 s 周期 | 是（hybrid 修前树 FAILED；样本自测 good 11≠0） |
| K7（checks 侧） | 属实 | HEAD `(row.get("util_pct") or 0) > 0`：None 计入 checked 且不判红；1 s 周期单测里 util=None 行修前 PASS | `checks.py` 同函数：读数不可读（rc≠0、util 非数、compute_pids 非列表）不计入 checked、记 NOT_RUN；忙碌证据（util>0 或 pids 非空）即使另一读数缺失也判 RED | `test_unreadable_util_is_not_counted_and_not_run`、`test_missing_compute_pids_is_not_run`、`test_failed_query_is_not_counted`、`test_busy_evidence_is_red_even_without_util`；新样本 `gpu_util_unreadable`（期望 11） | 是（前三条修前 FAILED；第四条修前也 RED，是回归护栏） |
| K13 | 属实 | HEAD `run_selftest.py` `--expect-samples` default None，省略时只按目录数 | `samples/run_selftest.py:189-193`（必填、≥1）；文档串 | `test_selftest.py::test_runner_requires_expect_samples`（省略或 0 ⇒ 退出 2）、`::test_runner_red_on_wrong_sample_count` | 前者是（修前 FAILED）；后者修前也过（原有行为的护栏） |
| K20 | 属实 | HEAD 先 `os.mkdir(judge)` 再核输入；真实树 `_local/e2e-20261113-195336-nullsse/W/runs/Jlocal-e2e2-…/judge/` 只有 INPUT_INVALID 的 inputs.json + judge.log、无 JUDGED.tsv，此后只能 ALREADY_JUDGED | `tools/capjudge/judge.py:89-193`、`:203-240`：过两道已判检查后建 `<T>/judge.attempts/<UTC>-p<pid>/`，全部输出先写这里并写 `attempt.json`；只有产出判定的运行才再查一次台账后 `os.rename` 成 `judge/`（原子，即「已判」标记），再追加 JUDGED.tsv 与 STATE；INPUT_INVALID / 异常（BaseException，记 traceback 后原样抛出，退出码 1）留在 attempts、不建 judge/；期间别人先建 judge/ 或台账行 ⇒ 12，本次记 ALREADY_JUDGED | `test_judge_once.py` 5 条（输入无效→留记录→修好再判 0→第三次 12；二次判定 12 且不写任何东西；异常留 ERROR 记录、无 judge/、可再判；judge/ 抢先建立；台账行抢先出现）；runner 对 13 样本断言无 judge/、恰一条 INPUT_INVALID 记录、再判仍 13，对 good 断言 attempt.json=JUDGED、二判 12 且不新增记录 | 是（5 条修前全 FAILED；runner 修前报 `INPUT_INVALID left judge/ behind`） |

## RS 规则额外扫出（均已修）

- **RS3/C3**：`check_c3` 不比对已检与应检，部分 worker 缺结果时 C3 可 PASS ⇒ 已检 < 应检记 NOT_RUN（`checks.py:160-162`）；测试 `test_c3_missing_outcomes_is_not_run`（修前 FAILED）。真实树重判 C3 仍 PASS，无回归。
- **RS1/C8**：覆盖清单的 lag 组位点（Lfe/Lmfl/Lmfa）槽是 loghist 直方图，快照 `values[slot]` 是 list，HEAD C8 做 `list - list` ⇒ TypeError，判定器崩溃（退出 1）。改为直方图按桶计数之和算增量（`checks.py:385-394`、`:396` 起）；测试 `test_c8_counts_histogram_slots_by_bucket_total`、`test_c8_histogram_without_new_samples_is_red`（修前均 FAILED）。
- **RS3/GPU-IDLE**：`gpu.jsonl` 头的 `positive_control_ok`（sidecar 自己的 GPU 列举）此前不读 ⇒ 现要求为 true，否则 NOT_RUN；测试 `test_sidecar_positive_control_failure_is_not_run`（修前 FAILED）。
- **RS12**：`inputs.py` 文档串（judge/inputs.json → 尝试目录）、`checks.py` 文档串、`tools/capjudge/README.md`（判定流程、GPU-IDLE/C3/C8 口径、`--expect-samples`）、`samples/README.md`（23、新样本、断言）同步。

## RS2：samples/good 与真实启动树比对

方法（MEASURED）：`scratchpad/rs2_compare.py`（只读）把文件名归一（档目录、pid、wstart/wend）后比文件清单，再对共有文件逐个比递归键路径与值类型。真实树：`_local/e2e-20261113-195834-nullsse/W/runs/Jlocal-e2e4-r0-L0_e0a-nullsse`、`_local/e2e-20261113-200042-relay/W/runs/Jlocal-e2e7-r0-L0_e0b-s1-single`。

修前「样本有、真实没有或形状不同」及处理：

| 文件 / 字段 | 差异 | 处理 |
|---|---|---|
| `raw/env/versions/*.json` | 样本是扁平无 identity 的记录；真实是 `capkit.version_record/1`（stamped，记录嵌在 `versions`，带 `source`） | 样本侧对齐（`make_samples.py:373-377`） |
| `raw/tiers/*/tier.json` | 样本有 `pids_window_start/end`、`nic_counters.unreadable`、`snapshot_copies[].edge/offset_ns`；真实是 `scans.{wstart,wend}`、`nic_counters.{wstart,wend}`、`snapshot_copies[].which/offset_from_boundary_ns/role/seq/file`、`thread_rules_sha`、`incomplete`，线程项带 `class`、`cpus_allowed` | 样本侧对齐（`scan()`、`threads_at()`、tier.json 写法） |
| `raw/sidecar/meta.json` | 样本只有 4 个键、无 identity、无 `gpu_every_ns` | 样本侧对齐为 stamped 完整键集，`gpu_every_ns = 2 s` |
| `raw/sidecar/gpu.jsonl` 头 | 样本缺 `positive_control_ok`，`written_wall` 是假值 | 对齐 |
| `nullsse.w0.summary.json` | 样本有 `failures`，真实没有 | 删掉样本里的键 |
| `register/scenario.json` 的 `roles[].cpus` | 真实场景角色不带 cpus（绑核来自绑核表） | 删掉样本场景里的 cpus（launch.json 的 roles 仍带） |
| `raw/env/overlay_manifest.json` | 真实树没有（K2）；样本键集比 `capprobe.overlay` 少，位点用 `line` 而当前写方是 `lines` | 样本按当前 `capprobe.overlay` 键集对齐；**真实写方要靠 K2（FX-launch/FX-jobs）补** |
| `raw/env/gpu_idle.json` | 真实树没有 | 样本按 `capjobs.envfiles gpu-idle`（HEAD 形状）写；**真实流程里没有人把作业级 env/ 交给编排器，见「需要他方配合」第 1 条** |
| `raw/sidecar/gpu.jsonl` | 真实树没有：本地 e2e 的 `sidecar_gpu_every_s` 为 "0" | 正常（本地无 GPU）；集群作业须把该参数设为 > 0，否则 GPU-IDLE 恒 NOT_RUN |

修后复跑：除上表最后三项（真实树缺文件）与合理差异（`synthetic` 标记、`probe_tier.groups` 非空、门读数名 `sink_streams` 来自样本自己的档位表、`gpu_positive_control`/`rows.gpu` 因真实树未开 GPU 采样）外，无「样本有、真实没有」的键。真实独有的键（如 `launch.json.orchestrator/params`、`launch_end.events`、canary 文件）判定器不读，未补。

## 需要他方或主会话配合

1. **（RS1，FX-jobs/FX-launch/主会话）`--job-env-dir` 无人传**：`gpu_idle.json` 由内层脚本写到 `$JOB_DIR/env/`，编排器只在 `--job-env-dir` 给出时才把作业级 `env/*` 拷进每棵树的 `raw/env/`；`capjobs.launches.launch_argv` 不追加它，`launchplan` 的计划 argv 是否带它取决于写计划的人。不带 ⇒ 每棵树缺 `gpu_idle.json` ⇒ GPU-IDLE（强制项）NOT_RUN ⇒ 每次启动 NOT_RUN。建议由 `capjobs.launches` 固定追加 `--job-env-dir {JOB_DIR}/env` 并列入 `RESERVED_FLAGS`（MEASURED：全仓 `command grep -RIn job-env-dir tools jobs configs` 只命中 caplaunch 自己）。
2. **（挑战设计，主会话）K6 请求 0 张 GPU 与 GPU-IDLE**：按 K6 口径请求 0 张时不跑阳性对照、记 NOT_RUN；判定器 GPU-IDLE 是强制项，阳性对照不过 ⇒ NOT_RUN ⇒ 这类作业的每次启动都 NOT_RUN。要么第一批作业一律申请 1 张卡，要么判定器需要一个可信的「请求 0 张」来源（如 submissions.tsv 的 `gpus` 列）并为它定义单独口径。本轮判定器未改。
3. **（FX-jobs）`gpu_idle.json` 形状在变**：FX-jobs 正把 `envfiles gpu-idle` 改为调用 `capenv.gpu.check_visible`，输出键集变化；判定器只读 `positive_control_ok`（新旧都有）。他们定稿后，样本 `gpu_idle.json` 可再对齐一次（只影响样本外观，不影响判定）。
4. **（FX-launch）tier.json**：样本按 HEAD 的 `_maybe_write_tier` 形状对齐；K3 若改 `scans`/`descendants` 口径或加字段，样本需要再对齐（判定器只读 `pids_stable` 与 `threads`）。
5. **（挑战设计，主会话）C13 启动级的红证据被降成 NOT_RUN**：`Result.finish` 在 expected==0 时一律 NOT_RUN；e0a-nullsse 场景没有 under_test 角色，真实树 e2e4 的 C13 写着 `2 unattributed process(es) in roles.json`（red=1）却记 NOT_RUN。这些未登记进程是 `multiprocessing.resource_tracker`（K27/RS13 归 FX-launch）。是否让「有红证据」优先于「应检为 0」，涉及 INTERFACES §10.2 的口径，本轮未改。
6. **（INFERRED，主会话/FX-probe）C8 与覆盖清单的位点选择**：`capprobe.overlay` 不带 `--sites` 时取「所开组的全部位点」，C8 要求清单里每个位点在窗口内都有增量；单前端场景里 Lmfl/Lmfa（多前端位点）不会触发 ⇒ C8 RED。需要按场景传 `--sites`，或让 C8 按 `proc_kind` 只查场景里存在的进程类。未实测。
7. **已有的旧 judge/**：`_local/e2e-20261113-195336-nullsse/W/runs/Jlocal-e2e2-r0-L0_e0a-nullsse/judge/` 是旧判定器 INPUT_INVALID 留下的；新判定器仍按「judge/ 已存在」拒判。要重判需主会话把它移开（我没动，只读）。

## INTERFACES 勘误原文（请主会话写入文末勘误）

> **§10.1 / §10.2 一棵树只判一次（R1 K20 修订）**：判定器先查 `<W>/JUDGED.tsv`，同一 `(job_id, restart_count, launch_seq, arm)` 已在表里 → 12；再查 `<tree>/judge`，已存在 → 12；这两步不写任何文件。之后新建本次尝试目录 `<tree>/judge.attempts/<UTC 时刻>-p<pid>/`，输入核对与判定的全部输出（`inputs.json`、`verdicts.json`、`checks.tsv`、`judge.log`，以及新增的 `attempt.json`：`status` ∈ `JUDGED` / `INPUT_INVALID` / `ERROR` / `ALREADY_JUDGED`、`exit_code`、起止时刻、异常时的 traceback）先写在这里。只有产出了判定结果的运行，才在再查一次 `JUDGED.tsv` 后把尝试目录原子改名为 `<tree>/judge`（这一步即「已判」标记），然后追加 `JUDGED.tsv` 一行、`STATE` 一行；改名时 `judge/` 已被别的运行建立，或台账已有该行 → 12，本次尝试记 `ALREADY_JUDGED` 留在 `judge.attempts/`。输入核对不过（13）或异常中断（1）时不建 `judge/`、不写台账、不写 `STATE`，尝试目录留作复查；输入修好后可以再判。原文「`os.mkdir(<tree>/judge)`」与「输入核对在 `judge/` 建好之后做……`judge/` 已存在，所以同一棵树也不能再判」两句作废。
>
> **§10.2 GPU-IDLE 应检条数（R1 K5、K7 修订）**：应检条数 = 各被判档 `window_ns // gpu_every_ns`（`sidecar/meta.json`）之和 × `gpu.jsonl` 头的 `gpu_count`；`gpu_every_ns` 缺失或 ≤ 0、`gpu.jsonl` 头的 `positive_control_ok` 不为 true ⇒ NOT_RUN。窗口内读不出的行（`query_rc ≠ 0`、`util_pct` 不是数值（`[N/A]`）、`compute_pids` 不是列表）不计入已检并记 NOT_RUN；`util_pct > 0` 或 `compute_pids` 非空的行即使另一读数缺失也判 VOID。原文「测量窗口秒数之和 × gpu_count」作废。
>
> **§10.4 自测（R1 K13、K20 修订）**：`run_selftest.py --expect-samples N` 必填（N ≥ 1，由调用方给出，等于样本目录数）；期望 13 的样本另断言没有 `judge/`、`judge.attempts/` 下恰有一条 `INPUT_INVALID` 记录、再判仍为 13；良品另断言 `judge/attempt.json` 为 `JUDGED`、再判 12 且不新增尝试记录。当前样本数 23（新增 `gpu_util_unreadable`）。

## 测试（MEASURED，命令前缀均为 `bash -c 'ulimit -v 8388608; exec systemd-run --user --scope -q -p MemoryMax=6G timeout 600 env PATH=<C>/_local/fx-judge/fakebin:$PATH PYTHONPYCACHEPREFIX=<C>/_local/fx-judge/pyc <venv python> -B …'`，cwd = <C>）

| 命令 | 退出码 | 结果 |
|---|---|---|
| 改动前基线 `-m pytest -q -p no:cacheprovider tests/judge` | 0 | 27 passed |
| 终跑 `-m pytest -q -p no:cacheprovider tests/judge` | 0 | 48 passed |
| `samples/run_selftest.py --tools <C>/tools --package capjudge --tmp <C>/_local/fx-judge/rt4 --expect-samples 23`（提交的样本） | 0 | 23 个样本全过，errors [] |
| 同上不带 `--expect-samples` | 2 | argparse 报缺参 |
| 修前对照：hybrid 树（capjudge 与 run_selftest.py 取 HEAD）`pytest tests/judge/test_checks_rows.py test_judge_once.py test_selftest.py -k 'not tampered'` | 1 | 15 failed / 7 passed / 1 skipped（之后补的 sidecar 阳性对照单测再跑 test_checks_rows.py：9 failed / 5 passed） |
| 修前对照：HEAD 判定器跑新样本 `run_selftest.py --expect-samples 23` | 1 | good 11≠0、INPUT_INVALID 留 judge/ 等 19 条错误 |
| 真实 e2e 树副本重判（`_local/fx-judge/rejudge/`，副本，原树只读） | — | e2e4、e2e7 attempt.json=JUDGED、exit 11（与原判一致）；e2e2 两次 13、只留 attempts、无 judge/；已判树再判 rc 12 |

## MEASURED / INFERRED / NOT RUN

- MEASURED：上表全部；RS2 比对输出（`scratchpad/rs2_out.txt` 修前、`rs2_out2.txt` 修后）；`--job-env-dir` 全仓只在 caplaunch 出现（grep）；真实树 C13 的 unattributed 是 `multiprocessing.resource_tracker`（读 sidecar meta）。
- INFERRED：C8 在单前端场景里遇到多前端位点会 RED（未实跑真实 P2 清单）；K6 0 卡请求使每次启动 NOT_RUN（由代码路径推出）；`os.rename` 遇到一个**空**的 `judge/` 会覆盖它（Linux rename 语义；只在与旧版判定器并发时可能发生，未处理）。
- NOT RUN：带 GPU 的真实树（本地 e2e 未开 GPU 采样、无作业级 env）——GPU-IDLE 在真实写方产物上的端到端行为未验证；集群相关一律未碰。

## 收尾

- fakebin.log：空（0 字节）。
- 残留进程：`pgrep -af 'capjudge|run_selftest|make_samples|pytest.*tests/judge'` 无。
- 我的改动（`git status --short`）：`M tools/capjudge/{README.md,checks.py,inputs.py,judge.py}`、`M samples/{README.md,make_samples.py,run_selftest.py}`、`M samples/<22 个样本目录>/**`（重新生成，1428 个数据文件）、`?? samples/gpu_util_unreadable/`、`M tests/judge/test_selftest.py`、`?? tests/judge/test_checks_rows.py`、`?? tests/judge/test_judge_once.py`。未删除任何文件；未 commit。
- 我自己的临时物都在 `_local/fx-judge/`（fakebin、pyc、gen/、rt*/、headtools/、hybrid/、rejudge/），可删，待主会话决定。
- 20:59 收到主会话裁定：C13 RED 优先、C8 按 proc_kind 适用性、K6 0 卡保持 NOT_RUN。开始改。
- 21:00 已改：Result.finish RED 优先；C8 按 launch.json roles[].proc_kinds 判适用；C13 启动级记录 aux_helpers（不计入被测）。接着改样本与测试
- 21:01 单测已补；重新生成样本

---

# 第二轮（主会话裁定后）

裁定：(5) C13 RED 优先；(6) C8 按 proc_kind 适用性；(2) K6 不加载模型的作业一律申请 1 张卡，0 张路径保持 NOT_RUN（判定器不改）；(1) `--job-env-dir` 转 FX-jobs/FX-launch；(7) e2e2 旧树不处理。

## 改动

| 项 | 改动 | 新增测试 | 修前是否红（hybrid 树，capjudge 取 HEAD） |
|---|---|---|---|
| 状态优先级 RED > NOT_RUN > PASS | `checks.py:41-50` `Result.finish`：有红证据即 RED，再看应检/已检为 0 ⇒ NOT_RUN；模块文档串同步 | `test_red_evidence_wins_over_nothing_expected` | 是 |
| C13 启动级 | 无 under_test 角色时 `unattributed` 也判 RED（经上一条）；`roles.json` 的 `aux_helpers` 只在明细列出、不计入任何角色进程集合（`checks.py:775-783`）。核实：判定器的角色进程集合只读 `roles.json.procs`（`treedata.procs_by_pid/procs_of_role`），线程清单只读 `tier.json.threads`（编排器只为 `procs` 写）；FX-launch 当前写法把辅助进程放在 `roles.json.aux_helpers` 与 `tier.json.scans.*.aux_helpers`，两处判定器都不当被测进程 | `test_c13_unattributed_is_red_without_under_test_roles`、`test_c13_aux_helpers_are_not_processes_under_test` | 是（后者修前状态同为 PASS，红在明细断言） |
| C8 适用性 | `checks.py:397-445`：`launch_proc_kinds` 取 `launch.json` 各角色 `proc_kinds`（字符串列表）的并集；`proc_kind` 不在其中的位点列入明细「not applicable in this topology: …」，不计入应检/已检；清单缺失或无位点、无角色声明 `proc_kinds`（或格式错）、无一个适用位点 ⇒ 显式 NOT_RUN | `test_c8_site_of_an_absent_process_kind_is_not_applicable`、`test_c8_no_applicable_site_is_not_run`、`test_c8_without_declared_proc_kinds_is_not_run`、`test_c8_union_of_role_kinds_and_red_on_applicable_site` | 前三条是；最后一条修前也红（护栏） |
| 样本 | `make_samples.py`：nullsse 角色 `proc_kinds: ["frontend"]`；覆盖清单加不适用位点 `W1`（worker）；capload 的 resource_tracker 记入 `roles.json.aux_helpers` 与 `tier.json.scans.*.aux_helpers`（并补 `monitor_descendants: []`，对齐 FX-launch 当前 `_scan` 形状）。重新生成，文件集合不变（无残留、无新增），23 个样本 | — | good 的 C8 明细：`not applicable in this topology: W1(worker)`，PASS；C13 明细：`1 aux helper(s) attributed to ['capload'] (not under test)` |
| README | `tools/capjudge/README.md`（优先级说明、C8、C13 行）、`samples/README.md`（共有设定） | — | — |

## 测试（MEASURED）

| 命令 | 退出码 | 结果 |
|---|---|---|
| `pytest -q -p no:cacheprovider tests/judge` | 0 | 55 passed |
| `samples/run_selftest.py --tools <C>/tools --package capjudge --tmp <C>/_local/fx-judge/rt5 --expect-samples 23`（23 = 带 expect.json 的样本目录数，现数） | 0 | 23 个，errors [] |
| hybrid（HEAD capjudge）跑新增 7 条 | 1 | 6 failed / 1 passed |
| 真实树副本重判（`_local/fx-judge/rejudge2/`） | — | e2e4（旧编排器产物，unattributed = 2 个 resource_tracker）：C13 RED ⇒ 退出 10；e2e7：C13 PASS，退出 11（GPU-IDLE NOT_RUN） |

## 需要他方配合（新增）

- **FX-launch（RS1，写方）**：C8 读 `launch.json roles[].proc_kinds`（产品角色声明自己运行的 capprobe 进程类，如 `["frontend"]`、多前端为 `["frontend_launcher","frontend_attached"]`）。现在场景与 launch.json 都没有这个键 ⇒ 真实树上登记了 C8 的启动一律 C8 NOT_RUN（fail-closed）。需要 scenario 角色键与 launch.json roles 增加 `proc_kinds`，取值与 `capprobe.sites.SITES[*].proc_kind` 同一词表。
- **INFERRED**：FX-launch 的 `aux_helpers` 改动落地后，e2e4 那类 C13 RED 会消失（未用新编排器实跑验证）。

## INTERFACES 勘误原文（追加）

> **§10.2 检查状态优先级（R1 主会话裁定）**：每项检查的状态按 RED > NOT_RUN > PASS 取：有任何红证据即 RED（即使应检为 0）；否则 `expected_count == 0` 或 `checked_count == 0`、或有 NOT_RUN 理由 ⇒ NOT_RUN；都没有才 PASS。原文「`expected_count == 0` 或 `checked_count == 0` → 该项 `NOT_RUN`」按此读。
>
> **§10.2 C8 适用位点（R1 主会话裁定）**：`launch.json` 的 `roles[]` 新增可选键 `proc_kinds`（字符串列表，产品角色声明自己运行的 capprobe 进程类，词表同 `SITES[*].proc_kind`）。C8 只检查 `proc_kind` 属于各角色 `proc_kinds` 并集的位点，应检条数 = 这些位点的个数；其余位点在 `checks.tsv` 明细里列为「本拓扑不适用」，不算 PASS 也不算 RED。覆盖清单缺失或无位点、没有角色声明 `proc_kinds`、或没有一个适用位点 ⇒ C8 NOT_RUN。直方图槽按各桶计数之和算窗口增量。
>
> **§5.3 / §10.2 C13 与辅助进程（R1 主会话裁定）**：`roles.json` 的 `aux_helpers`（工具角色运行时自起的辅助进程，如 `multiprocessing.resource_tracker`）归属其父角色，但不属于该角色的进程集合：判定器的 C10/C11/C13/快照对账等一律只用 `roles.json.procs`；`aux_helpers` 不算 `unattributed`，只在 C13 明细里列出。
