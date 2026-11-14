# fix-R4（R3 自审追加一轮修复，用户 2026-11-14 批准）进度

- WT = `_local/wt-r4`（分支 capkit-r4，起点 9f435929411185c2587a4821c6241941242deb4f）；C（01_code main）只读；探针树只读。
- 工作目录 `WT/_local/r4-fix/`（fakebin 十个替身、pyc、logs）。
- 2026-11-14 11:52:08 开工：已读 R3 triage/compare（capkit、capprobe）、R1/R2 fx_prompt、integ-R2 进度。

## 范围
S1、S2、RS20（PATCH_SHA / CODE_SHA 写方）、文档 S12/S13/S20/T1。其余 Minor 不碰。

## 进度日志
- 2026-11-14 11:57:02 续接（新修复者）：读完 triage/compare/SEAMS（SM-1..SM-4，尚无 SWEEP-DONE），开始读码核实 S1/S2/RS20/SM-1/SM-2。
- 2026-11-14 12:01:24 核实结论（读码）：
  - SM-1 属实：launchplan.worst_stop_s 要求 `--scenario {SNAP}/…`；caplaunch `_load_inputs` 要求场景在 --workdir 内且 W 相对路径 = register files.scenario.relpath；判定器 `_load_register` 也按 W 相对路径读、要求 REGISTRY 登记早于 submitted_at ⇒ {SNAP} 形式在作业里必 exit 3。选 SEAMS 建议 (b)：计划 argv 的 `--scenario` 指 W 里已登记的那份（相对 --workdir 或 W 内绝对路径，不得含占位符），kill_grace 从它算；caplaunch/判定器不改（它们本来就按 W 核对，内容由 register sha 钉住）。submit 额外核对 relpath/sha 与登记表该 arm 的 files.scenario 相符、argv --workdir 与 --workdir 相同（提交时就拒，不等到作业里 exit 3）。
  - SM-2 属实：capkit_inner.sh/launches 都不生成绑核表。改：capjobs.launches 每个启动前在节点上按该启动场景的角色需求跑 `capenv.pins generate`（启动身份、`--source-sha` = 本作业 env/hw_profile.json 的 profile_sha256、只用本进程 affinity 内的 CPU），追加 `--pin-table`；`--pin-table` 进 RESERVED_FLAGS（计划里出现即拒）。
- 2026-11-14 12:05:10 SM-1/SM-2 代码与 tests/jobs 改完，跑 tests/jobs
- 2026-11-14 12:06:20 SM-4 改（inner 传 overlay base_commit 作 --code-baseline）；tests/jobs 相关文件通过。重读 SEAMS。
- 2026-11-14 12:07:45 SEAMS 新增 SM-5..SM-8（无 SWEEP-DONE）。开始 S1/S2/RS20（caplaunch）。
- 2026-11-14 12:11:57 测试用：WT/probe-tree -> ../../probe-tree 软链（只读引用探针树；提交前 mv 到 _local/r4-fix/，不删）。
- 2026-11-14 12:13:23 S1/S2/RS20 代码+测试写完，后台跑 tests/launch（约 2–3 分钟）
- 2026-11-14 12:15:15 作业路径 e2e 脚本写好：_local/r4-fix/jobpath/{build.py,inner.sh,run.sh}（改编自轨道 A sweep），开跑 e0a-nullsse
- 2026-11-14 12:16:08 tests/launch 70 passed?（见 logs/dev/launch2.log）；作业路径 e2e e0a-nullsse：submit rc=0（真 plan_id）→ launches VALID → judge rc=11、0 RED、无 INPUT_INVALID。开跑 e0b-s1-single。
- 2026-11-14 12:17:34 T1 文案改完（sites.py F4、capprobe README）；后台跑全量验证 v1（10 目录 + selftest + patchcheck + py_compile + bash -n，约 8–10 分钟）
- 2026-11-14 12:19:52 SEAMS 已 SWEEP-DONE 12:15，主会话裁定 R4 修 SM-1(推荐a，我选b须写理由)/SM-2/SM-3文档/SM-4/SM-5/SM-10/SM-12；延后 SM-6/7/8/9/11。全量验证 v1 在跑（基于 SM-5/10/12 之前的代码，之后要重跑 v2）；变异 12 项全红、基线全绿（mut/r1）。开始 SM-5。
- 2026-11-14 12:21:59 SM-5（launches：exit 3 且无 raw/launch.json ⇒ NOT_RUN+stderr 末行；空树拷回不记未核对）、SM-10（gate 第 1 项按 ReqTRES，无 ReqTRES 时按分配数）改完并有测试。开始 SM-12。
- 2026-11-14 12:26:46 SM-12（identity nics = PCI 网卡清单，netdev 移到 observed "netdevs"）改完并有测试；文档（INTERFACES 勘误 R4-1..R4-11、jobs/README、README、REVIEW、capenv/caplaunch/capprobe README）改完。重读 SEAMS：12:16 裁定之后无新增条目。后台跑全量验证 v2（约 8–10 分钟）。
- 2026-11-14 12:28:01 作业路径 e2e 重跑（SM-5/10/12 之后）：e0a-nullsse J8400005、e0b-s1-single J8400006 均 launches VALID、judge rc=11、0 RED；画像 nics 为 PCI 清单、code_baseline=base_commit。开跑直接 caplaunch e2e（不导出 CAPKIT_CODE_SHA，验证 RS20）。
- 2026-11-14 12:35 直接 caplaunch e2e（integ-R2 做法，脚本拷到 `_local/r4-fix/direct/`，caplaunch 用 `env -u CAPKIT_CODE_SHA -u CAPPROBE_PATCH_SHA` 起）：e0a-nullsse（local-r4da）caplaunch rc=0、STATE VALID、judge rc=11、0 RED、C10 PASS 8/8；e0b-s1-single（local-r4db）同，C10 PASS 5/5。第一次用了 job id `r4da` 被 local-record 拒（要 `local-<x>`），目录改名为 `W-r4da.badjobid` 保留。
- 2026-11-14 12:35 变异第二批（mut/r2）：SM-5、SM-10、SM-12 去掉修复后对应新测试全红；未变异副本 4 passed。
- 2026-11-14 12:34 全量验证 v2（logs/v2/summary.txt）：capcommon 190、capload 29、capprobe 159、envgate 63、grpc 32、jobs 174、judge 85、launch 98、nullsse 4、sidecar 5，均 EXIT=0；selftest --expect-samples 25 EXIT=0；patchcheck（探针树，只读）PASS；py_compile 186 个 0 失败；bash -n 3 个 0 失败。

## 结果汇总

### 处理
| 项 | 处理 | 测试 / 证据 |
|---|---|---|
| S1 | `scenario.product_writes_snapshots`（产品角色且 P2）；这样的角色 `snapshots` 必须 `all`；P0/P1 任意值接受；编排器 `_snapshot_procs` 不把不写快照的产品角色列进应拷名单（勘误 R4-1 推翻 R2-17） | test_scenario 新用例；test_probe_env 应拷名单逐值；变异 S1-* 红 |
| S2 | lag 组 + 产品角色 ⇒ `params_required` 必含 `lag_period_us`；编排器校验正整数并设 `CAPPROBE_LAG_PERIOD_US`；外层非空 `CAPPROBE_INJECT`（有产品角色时）拒绝 exit 3；外层 LAG 值冲突拒绝（勘误 R4-2） | test_probe_env、test_scenario、test_launch_e2e 参数化；变异 S2-* 红 |
| RS20 | 选「编排器写」：每个角色 `CAPKIT_CODE_SHA`=--capkit-sha；P1/P2 `CAPPROBE_PATCH_SHA`=覆盖清单 target_commit（非 sha ⇒ exit 3）；P0 去掉外层 PATCH_SHA；外层冲突 exit 3；submit 要求计划 --capkit-sha = HEAD（勘误 R4-3） | test_probe_env、test_submit；直接 caplaunch e2e 不导出变量 C10 PASS；变异 RS20-* 红 |
| SM-1 | 选 (b)，理由见勘误 R4-4：计划 argv `--scenario` 指 W 内已登记的场景（字面、无占位符），`--workdir` 字面绝对路径；submit 核对 workdir、relpath+sha 与登记表一致 | test_units、test_submit；作业路径 e2e |
| SM-2 | launches 在节点上按启动身份生成绑核表，`--pin-table` 自动追加且计划中保留；失败 ⇒ NOT_RUN；pins 新 `--only-affinity`；launches 新必填 `--hw-profile`（勘误 R4-6） | test_container 新用例；作业路径 e2e 绑核表身份为本作业 |
| SM-3 | 只改文档（jobs/README、勘误 R4-8） | — |
| SM-4 | inner 传覆盖清单 base_commit 作 `--code-baseline`，读不到 ⇒ env_invalid（勘误 R4-7①） | test_scripts；作业路径 e2e 画像 code_baseline=base_commit |
| SM-5 | exit 3 且无 raw/launch.json 且无编排器 STATE 行 ⇒ NOT_RUN（stderr 末行）；空树拷回不记未核对、不算拷回失败（勘误 R4-9） | test_container；变异 SM5 红 |
| SM-10 | 默认决策：门禁第 1 项按 ReqTRES，缺失时按分配数；记 req_gpus/alloc_gpus/gpus_checked_from（勘误 R4-10，措辞待用户确认） | test_envgate_gpucount；变异 SM10 红 |
| SM-12 | 默认决策：identity nics = PCI 网卡清单（vendor:device×count，排除 VF）；旧列表移到 observed `netdevs`（勘误 R4-7②，待用户确认） | test_envgate_gate；变异 SM12 红；作业路径 e2e 实机 /sys 读到 1 个 0x1af4:0x1041 |
| S12 | README 说明进度表命令是历史记录、`--expect-samples` 取带 expect.json 的样本数（历史行不改） | — |
| S13 | jobs/README 两处过时说法改正 | — |
| S20 | REVIEW.md 补 R1–R3 行 | — |
| T1 | sites.py SITE_NOTES["F4"]、capprobe README、勘误 R4-5（R2-6 加指针） | — |

### 未修（按范围）
- R3 S3–S11、S14–S19；capprobe T2–T6。
- SEAMS SM-6、SM-7、SM-8、SM-9、SM-11（主会话裁定延后）。
- 产品侧探针自检注入在编排器下暂不可用（S2 改为拒绝外层 INJECT 的后果，勘误 R4-11①）。
- 待用户确认：SM-10 的 CONSTRAINTS 措辞、SM-12 的 identity 口径；R4 合入后两个画像都要重采再登记。

### 其他
- 测试用软链 `WT/probe-tree -> ../../probe-tree` 提交前移到 `_local/r4-fix/probe-tree.link`（不删）。
- 为让 jobs 测试与作业路径 e2e 找到 SAVED_CDB，在 C/_local（被 git 忽略）建了软链 `00_plan -> ../../00_plan`；C 的 HEAD、index、工作树未动。
