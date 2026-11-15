# capkit-b2-spb 合入 main 的进度与冲突记录

- 任务：把 `capkit-b2-spb`（head 756b9e7fe6ef5fb7726b0dd3f6a81869545f184d，base 0a1410a4，R1～R4 自审、R4 干净）`--no-ff` 合入 main（b3b3368e73f2f488eb53129bb8ecefbd47420887）。
- 授权：DECISIONS 第十四、十五次答复（合入不再触发自审）；FASTTRACK-1117-PLAN v2 第 5 节「合入 b2-spb」。
- 工作树：`01_code/_local/wt-merge-spb`，分支 `capkit-merge-spb`。

## 日志

- 2026-11-15 建 worktree，`git merge --no-ff --no-commit capkit-b2-spb`：6 个文件内容冲突（INTERFACES.md、tools/capjobs/submit.py、tools/capjudge/inputs.py、tools/caplaunch/README.md、tools/caplaunch/orchestrator.py、tools/caplaunch/scenario.py），其余 5 个双方都改的文件自动合并（jobs/README.md、jobs/capkit_inner.sh、tools/capgrpc/common.py、tools/capjudge/README.md、tools/capjudge/checks.py）。
- 2026-11-15 23:56:41 合入前先跑 tests/capstubs（约 3 分钟），核对 S-PB 的 SM-18 适配。
- 首次 tests/capstubs：94 passed、1 failed（`test_real_generated_module`：WT 下没有 `probe-tree` 软链；与 b2-spb 进度里同一现象，属环境，不是合并问题）。全量验证时按 verify-main-b3b3368 的做法建 `probe-tree -> ../../probe-tree` 软链，跑 e2e 前移到 `_local/merge-spb/probe-tree.link`（`.gitignore` 的 `probe-tree/` 只匹配目录，软链会让树变脏）。

## 冲突与处理

| # | 文件 | 两边各改了什么 | 处理 |
|---|---|---|---|
| 1 | `tools/capjobs/submit.py`（import 行） | main（SB-A）加 `jobplan`；2B 加 `jobinputs` | 两个都留：`from capcommon import jobinputs, jobplan, ledger, policy`（两者在文件里都有调用） |
| 2 | `tools/capjudge/inputs.py` `verify()` 的加载顺序 | main（2A+2E、SB-A）在档位表之后加 `_load_tier_regimes`、`_check_threshold_window`、`_load_localize`；2B 在同一位置加 `_load_job_inputs` | 全留，main 的三项在前、`_load_job_inputs` 在后、`_check_reconcile_and_scenario` 之前。两边互不读对方设的 ctx 字段（`_load_job_inputs` 只读 `register_files`、`launch`、`root`），顺序不影响结果；勘误 B2B-10 写的「档位表与时间表之后、对账段之前」仍成立 |
| 3 | `tools/caplaunch/orchestrator.py` | main（SB-B1）新增方法 `_check_job_env_pin_table`；2B 新增方法 `_load_job_inputs`，同一插入点 | 两个方法都留（main 的在前）；两处调用点各自在自动合并的部分里（`__init__` 第 203 行、第 355 行） |
| 4 | `tools/caplaunch/scenario.py`（import 与 TOOL_TABLE 注释） | main 加 `RunTree` import 与 `pin_table`、`role_fixed` 两段注释；2B 加 `templates` import 与 `single` 一段注释 | import 都留；注释合成一份：`sidecar / canary`、`pin_table`、`single` 三行 + `role_fixed` 说明。TOOL_TABLE 本体（`capenv.spinners`、`capstubs.prebaked_server`、`caplaunch.smoke`）自动合并，`role_fixed` 与 `single` 两个校验都在 `validate_role` 里 |
| 5 | `tools/caplaunch/README.md`（模块表与参数说明） | main 改 tiergen 行、新增 laddergen 行、`--job-env-dir` 加 `pin_table.json` 一句；2B 改 httpprobe、smoke、生成器三行，新增 `--job-input` 一条 | 模块表：main 的 tiergen + laddergen，2B 的 httpprobe + smoke + 生成器；参数：2B 的 `--job-input` 一条在前，main 的 `--job-env-dir`（含 `pin_table.json` 一句）在后 |
| 6 | `INTERFACES.md` §11 命令表 4 行 | main：`capjudge.pilot` 行加 `capjudge.intervene` 指针、`capjobs.submit` 行加 SM-17 与 calib-v2-A4 的新参数；2B：`caplaunch` 行加 `--job-input`、`capstubs.fingerprint` 行加 S-PB 指针 | 逐行取改过的一边：pilot、submit 取 main，caplaunch、fingerprint 取 2B（逐词比对确认每行只有一边改了） |
| 7 | `INTERFACES.md` 勘误末尾 | main 追加 SM-16、SM-17/18、2A+2E、SB-B1、SB-A 五节；2B 追加「Batch 2B …」一节 | 全留，按合入顺序：main 的五节在前、2B 一节在后，其后新增「合入 b2-spb 到 main（勘误 M-SPB-n）」一节。编号核对：勘误号 `B2-2A/2E-n`、`SB-B1-n`、`calib-v2-An`、`B2B-n` 互不重叠；正文新增节号 §13（2B 的 S-PB 契约，B2B-2）与 §14（2A+2E 的分析脚本）不重号；2B 改的正文指针（6 处）与 main 改的正文都在自动合并的部分，无冲突 |

## 文本无冲突、语义上要对齐的地方

| # | 问题 | 处理 |
|---|---|---|
| S1 | main 的 SM-18 修复把 `capgrpc.common.make_server(reuseport, max_unrequested_s)` 的第二个参数改为必填，并在 INTERFACES「Fix SM-17 … SM-18 …」第 2 条最后一项写明「S-PB（2B）尚未合入 main，合入后起进程时 TypeError，必须照此改」。2B 的 `capstubs.prebaked_server` 仍是 `make_server(reuseport=False)`，文本自动合并，不改就起不来 | 照 `capselfcheck.relay` 的做法改 S-PB：`add_server_args`（可选 `--max-unrequested-s`）、`main()` 里在读档位表后 `resolve_max_unrequested`、`make_server(reuseport=False, max_unrequested_s=…)`，就绪文件自报与每档 summary 写 `grpc_server_options`。测试：`tests/capstubs/test_prebaked_server.py` 端到端用例加三项断言（缺省来源、两项条数上限、每档与就绪一致），启动拒绝用例加 `max_unrequested_zero`。文档：`tools/capstubs/README.md` 参数与输出、INTERFACES SM-18 那一项改为指向 M-SPB-1、B2B-3 的 S-PB 行加参数、新增勘误 M-SPB-1 |
- 2026-11-15 23:58:52 合并提交 0b8816b4db1fb45666f78dd661656f579fa5c55f；开跑全量验证 `MAINV_WT=<WT> fullv_all.sh merge-spb 52`（12 个测试目录 + selftest 52 + patchcheck + py_compile + bash -n，约 30 分钟，后台）。
- 2026-11-16 00:08:26 全量验证 tests/jobs 6 failed：SB-A 的 `laddergen.packing` 用只有 `budget_s` 的合成计划调 `submit.time_budget`，2B 的 time_budget 要从每个启动的 argv 读场景算作业输入时限 ⇒ KeyError 'argv'。真语义冲突（S2），修复中；全量验证在后台继续跑完其余目录。

| # | 问题 | 处理 |
|---|---|---|
| S2 | 全量验证 tests/jobs 的 `test_v0_tables_fit` 6 例 KeyError 'argv'：main（SB-A）的 `caplaunch.laddergen.packing` 用只有 `budget_s` 的合成计划调 `capjobs.submit.time_budget`；2B 让 `time_budget` 经 `launchplan.in_job_bound_s` 从每个启动 argv 读场景、加 `job_inputs_s`。两边文本都没冲突 | `time_budget(plan, renv, job_input_kinds=None)`：None 照旧读场景（提交端不变）；整数则每个启动按这么多种类计。`laddergen.packing` 传 0（v2 标定场景都不声明 `job_inputs`），`packing` 段（进 `ladder.json`）的键与数值与合入前相同，没加新键以免改变 `ladder.json` 字节。测试：`tests/jobs/test_job_inputs.py::test_in_job_bound_counts_each_declared_kind` 加显式计数的三条断言；`test_v0_tables_fit` 覆盖 laddergen 路径。INTERFACES 新增 M-SPB-2 |
