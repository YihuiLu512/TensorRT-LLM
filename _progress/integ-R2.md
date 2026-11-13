# integ-R2（R2 自审修复集成）进度

- 2026-11-13 23:14:38 开工：<C> HEAD 1be4670c22（工作树含 FX2-judge/probe/jobs/launch 四方改动）；探针树 HEAD 733edd4edf（facade.py 一处未提交）。已读 R2 triage/brief、R1 triage、4 份 fix-R2 进度、integ-R1。建 _local/integ-R2/{fakebin(10 个替身),pyc,logs}。
- 注：4 份 fix-R2 进度文件里没有「INTERFACES 勘误原文」一节（grep「勘误」0 命中），勘误按进度文件 + 代码 diff 由本集成整理。
- 2026-11-13 23:16:07 1(a) STATE 严重度下沉 capcommon/runtree.py（STATE_SEVERITY/state_rank/worst_state）；inputs.py 删 STATE_RANK/worst_state，judge.py、capjobs.launches.classify 与 run_all（survived/copy-back 失败）改用 worst_state（原来 survived/copy-back 失败会把 ENV_INVALID 改成 VOID、预算超时也会覆盖 ENV_INVALID，按 RS14 修）；测试 capcommon +2、jobs +1。
- 2026-11-13 23:17:55 1(b) caplaunch/scenario.py 产品角色 probe_tier P1/P2 时 snapshots 不得为 none；tests/launch/test_scenario.py +2 参数化用例（变异：去掉检查 ⇒ 2 红）。configs/e0 产品角色都是 P0、生成器无需改（核对：e0a-frontend-sfe / e0c-frontend-http 的 sfe、frontend 均 P0 none）；samples 合成场景的 P2 角色 nullsse 是快照写者，不受影响。1(a) 变异：classify 不取较差 ⇒ jobs 2 红。
- 2026-11-13 23:19:28 1(c) sites.py 文档读方清单加 caplaunch.scenario（按路径单独加载 PROC_KINDS，要求 sites.py 无 import）+ tests/capprobe 新增 test_sites_module_loads_alone_by_path。1(d) 扫接缝：jobs/README 与 launches 文档串的 STATE 规则改为 worst_state 口径；launches 文档串「write the first STATE line」字面不成立（编排器先写）→ 改为在编排器之后追加。
- 2026-11-13 23:20:11 第 3 步：后台跑 verify_all.sh v1（10 个测试目录 + selftest + patchcheck + py_compile + bash -n，预计 6–10 分钟）；同时写 INTERFACES 勘误。
- 2026-11-13 23:23:11 第 2 步：INTERFACES 新增「### R2 自审修复（2026-11-14）」18 条（按章节，§5.7→§12），正文/旧勘误 16 处句子加「见勘误 R2-n」指向；Q25 与 Q4 遗留登记为 R2-18 已知缺口。进度文件里无勘误原文，由集成者按进度文件与代码 diff 整理。
- 2026-11-13 23:27:33 verify v1 全过：capcommon 190、capload 29、capprobe 159、envgate 60、grpc 32、jobs 165、judge 85、launch 68、nullsse 4、sidecar 5（均 EXIT=0）；selftest --expect-samples 25 EXIT=0；patchcheck 工作树 PASS；py_compile 185+4 无失败；bash -n 3 个无失败。下一步 e2e。
- 2026-11-13 23:27:38 e2e e0a-nullsse（JOB local-ir2a，约 1–2 分钟）
- 2026-11-13 23:28:14 e2e nullsse：caplaunch rc=0、STATE VALID、judge rc=11、0 RED（C1 无噪声带、C6 应检 0、C13 launch 无 under_test/逐档无 D14）。开始 e0b-s1-single（JOB local-ir2b）
- 2026-11-13 23:29:00 e2e relay：caplaunch rc=0、STATE VALID、judge rc=11、0 RED（各档 C1/C7/C13 NOT_RUN）；launch LAUNCH-STATE/C10/GPU-IDLE/C13 PASS。pilot --d14-rule rc=0：1 噪声带/4 档，d14=True（relay1 核 15：空载 softirq 0/195，负载 1–8/111–119），n_job_runs_complete=1。下一步提交探针树。
- 2026-11-13 23:29:51 探针树提交 add0f86748515be4bd370bf539f48d3d3b993e6e（首次提交 40e5e6d636 正文措辞过宽，未推送，amend 改为字面成立的一句）；patchcheck --rev HEAD PASS 8/8，行号无漂移，sites.py 不用改。

## 第 1 步 接缝处理
- (a) STATE 严重度：`capcommon/runtree.py` 新增 `STATE_SEVERITY`、`state_rank`、`worst_state`（唯一实现）；`capjudge/inputs.py` 删 `STATE_RANK`/`worst_state`，`judge.py` 与 `capjobs/launches.py` 改用它。顺带按 RS14 修 launches：超预算被杀、扛过 SIGKILL、拷回未核对原来直接写 VOID，会把编排器已记的 ENV_INVALID 改「好」成 VOID → 现取较差者。测试：capcommon `test_state_severity_is_one_order`、`test_no_second_state_order_in_the_tools`；jobs `test_budget_kill_never_improves_the_launcher_record`。变异（classify 不取较差）⇒ jobs 2 红。
- (b) `caplaunch/scenario.py`：产品角色 `probe_tier` P1/P2 时 `snapshots` 不得为 `none`；`tests/launch/test_scenario.py::test_probed_product_role_must_expect_snapshots[P1|P2]`，变异 ⇒ 2 红。configs/e0 与生成器无需改（产品角色均 P0）。
- (c) `capprobe/sites.py` 文档读方清单加 caplaunch.scenario（按路径单独加载，须无 import）；`tests/capprobe/test_probe_tree_sites.py::test_sites_module_loads_alone_by_path`。
- (d) jobs/README 与 launches 文档串的 STATE 规则改成 worst_state 口径；launches 文档串「write the first STATE line」改为在编排器之后追加。其余接缝核对无问题：编排器经 `probe_entry.build_rule` 带 `cwd`；`tier.json` 的 `snapshot_missing` 写方（orchestrator）与读方（RECORDS）键一致；caplaunch `_private_dir` 与 launches `private_mkdir` 口径一致。

## 留给主会话（未改）
- Q4 遗留：目标节点内核的 irq/softirq 记账配置（`CONFIG_IRQ_TIME_ACCOUNTING`）未核，登记为勘误 R2-18②，待用户裁决。
- Q25：原架构前端无 lag 位点，登记为勘误 R2-18①（第二批、E7 之前）。
- 勘误 R2 由集成者按进度文件与 diff 整理（四份进度文件没有勘误原文），建议主会话抽读。
- R1 遗留未变（integ-R1「留给主会话」各条）。
- 本地 e2e 只覆盖 capkit 角色（P0、无产品角色）：Q2/Q13 的入口匹配、Q21 的 C8 进程类、Q22 的 F4 守卫都没有真实产品进程验证（INFERRED 依据：单测 + patchcheck）。

## 第 3 步 全量验证（v1，日志 `_local/integ-R2/logs/v1/`）
| 项 | 退出码 | 结果 |
|---|---|---|
| tests/capcommon | 0 | 190 passed |
| tests/capload | 0 | 29 passed |
| tests/capprobe | 0 | 159 passed |
| tests/envgate | 0 | 60 passed |
| tests/grpc | 0 | 32 passed |
| tests/jobs | 0 | 165 passed |
| tests/judge | 0 | 85 passed |
| tests/launch | 0 | 68 passed |
| tests/nullsse | 0 | 4 passed |
| tests/sidecar | 0 | 5 passed |
| run_selftest --package capjudge --expect-samples 25 | 0 | errors [] |
| patchcheck（探针树工作树） | 0 | PASS 8/8 |
| py_compile（185 + 探针树 4） | 0 | 0 失败 |
| bash -n（3 个 .sh/.sbatch） | 0 | 0 失败 |

## 第 4 步 本地 e2e（`_local/integ-R2/e2e/`）
- e0a-nullsse（local-ir2a）：caplaunch rc=0、STATE VALID；judge rc=11（STATE 末行 capjudge NOT_RUN，upstream=VALID）；0 RED；非 PASS：launch C13（无 under_test，应检 0）、各档 C1（无登记噪声带）、C6（应检 0）、C13（pilot 无 D14）；LAUNCH-STATE、GPU-IDLE、C10 PASS。
- e0b-s1-single（local-ir2b）：caplaunch rc=0、STATE VALID；judge rc=11；0 RED；非 PASS：各档 C1、C7、C13（同上）；launch 级 LAUNCH-STATE、C10、GPU-IDLE、C13 PASS。
- pilot --d14-rule（e0b 树，档 2,4,6,8）：rc=0，1 噪声带 / 4 档，d14 True，n_trees_complete=1、n_job_runs_complete=1。
- fakebin.log 2 行：`scontrol show job local-ir2a|b`（capenv.gate 读 Slurm 视图，被替身拦下 exit 97，门禁按预期 ENV_INVALID）。

## 第 5、6 步
- 探针树：add0f86748515be4bd370bf539f48d3d3b993e6e；patchcheck --rev HEAD PASS 8/8。
- 残留进程：无。新建 /dev/shm（未删）：/dev/shm/yihlu-local-ir2a-integR2、/dev/shm/yihlu-local-ir2b-integR2。
