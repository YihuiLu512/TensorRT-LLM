# fix-R5b（R5 三路审阅唯一 Important：路 A R5-R1 / RS22）进度

- WT = `_local/wt-r4`（分支 capkit-r4，起点 0ca36e32febb9274837e22792b85343c74e91147）；C（01_code main）只读；探针树只读。
- 工作目录 `WT/_local/r5b-fix/`（fakebin 十个替身、pyc、logs、作业路径 e2e，从 r5-fix 拷来改路径）。
- 范围：只修 R5-R1（起进程前的 caplaunch 失败 + submit 预检计划 argv）；其余 Minor/Nit 不碰；R5 之后不再开三路复审（用户已定）。

## 进度日志
- 2026-11-14 14:05:50 开工：已读 R5 review-pr R1 节、launches/submit/caplaunch/launchplan、orchestrator 起进程前路径；建 _local/r5b-fix（含 build_badplan.py：去掉 --housekeeping-ms）。下一步：复现坏计划。
- 2026-11-14 14:05:57 跑修前坏计划复现（作业路径 e2e，BUILD=build_badplan.py，J8510001，约 1–2 分钟）
- 2026-11-14 14:06:10 修前复现成功（e0a-nullsse-bad-pre，J8510001）：build submit_rc=0（submit 接受缺 --housekeeping-ms 的计划）；launches rc=4，L0 VOID「copy-back not verified (NOT_RUN); launcher exit 2」，launches.json config_refused=0。开始写修前会红的测试。
- 2026-11-14 14:07:53 新测试写完（test_container 2 个：退出码 1/2/3 参数化 + 判据单测；test_submit 2 个：预检 6 例参数化 + 构造 argv 单测；submit fixture argv 补全为完整 caplaunch argv）。跑修前 tests/jobs（约 2–4 分钟）
- 2026-11-14 14:08:45 修前 tests/jobs：10 failed / 187 passed（logs/pre/t_jobs.log）；红的正是新测试：launches 退出 2、1 时 main 返回 4，判据对 rc 1 返回 False，submit 6 例都 rc 0 接受、check_plan_argv 不存在。开始改 launches.py 与 submit.py。
- 2026-11-14 14:09:48 修复写完，tests/jobs 197 passed（logs/post/t_jobs.log）。下一步：查旧措辞残留、改文档。
- 2026-11-14 14:11:15 文档改完（INTERFACES 新增「R5 后 Important 修复」R5b-1/R5b-2 与 4 处指针；jobs/README 两处）。跑变异 4 项（_local/r5b-fix/mut/r1，约 2–4 分钟）
- 2026-11-14 14:11:41 变异 4 项全红（mut/r1/summary.txt：未变异 11 passed；M1 判据退回只认 3 ⇒ 3 failed/1 passed（退出 3 那例本该过）；M2 去掉 submit 预检调用 ⇒ 6 failed；M3 去掉 error 覆盖 ⇒ 5 failed/1 passed（-h 例本该过）；M4 不追加 launches 参数 ⇒ 1 failed）。后台跑全量验证 v1（10 目录 + selftest + patchcheck + py_compile + bash -n，约 8–10 分钟）
- 2026-11-14 14:11:52 建测试用软链 WT/probe-tree -> ../../probe-tree（tests/capprobe 用；提交前移到 _local/r5b-fix/probe-tree.link）
- 2026-11-14 14:13:39 协调者补充检查点（RS15：run_all 的 config_refused 与 classify 的结论同一判据）：两处本来就调用同一个 prestart_config_refusal（launches.py :407 与 classify 内）；补测试 test_config_refused_and_status_come_from_one_criterion（rc×超时×SIGTERM×STATE×launch.json 全组合断言两者一致）。
- 2026-11-14 14:20:11 全量验证 v1 全绿（logs/v1/summary.txt：capcommon 190、capload 33、capprobe 159、envgate 71、grpc 36、jobs 198、judge 85、launch 189、nullsse 4、sidecar 5；selftest EXIT=0（25 样本）；patchcheck PASS；py_compile 187/0；bash -n 3/0）。跑变异 r2（全 6 项，含 RS15 的 M5/M6，约 1 分钟）
- 2026-11-14 14:20:42 变异 r2 全 6 项全红（未变异 12 passed；M5 run_all 自算判据 ⇒ 2 failed/1 passed；M6 classify 自算判据 ⇒ 2 failed）。开跑作业路径 e2e 终版：e0a-nullsse（J8510011）、e0b-s1-single（J8510012）、坏计划（J8510013），约 3–5 分钟
- 2026-11-14 14:22:40 作业路径 e2e 终版（导出 PYTHONDONTWRITEBYTECODE=1，环境同 capkit_inner.sh）：e0a-nullsse（J8510011）、e0b-s1-single（J8510012）build submit_rc=0、launches rc=0、STATE VALID、judge rc=11（NOT_RUN，与 R5 终版同形）、0 RED、无 INPUT_INVALID。坏计划（J8510013，缺 --housekeeping-ms）：submit rc=3「launch 0: caplaunch would refuse its argv: the following arguments are required: --housekeeping-ms」，没建快照、没写台账。补充：用修复后的 WT tools 对修前坏计划快照跑作业侧（inner_wt.sh，J8510014）：L0 NOT_RUN「launcher exited before starting any role (rc=2): …--housekeeping-ms; nothing to copy back」，launches rc=3，config_refused=1。
- fakebin.log：只有门禁的 scontrol show job（替身 exit 97），没有碰集群。
- 准备提交：WT/probe-tree 软链移到 _local/r5b-fix/probe-tree.link（不删）。

## 结果汇总

### 处理（R5-R1，RS22 + RS15）
| 处 | 改动 | 测试 / 证据 |
|---|---|---|
| ① `capjobs.launches` | `prestart_config_refusal` 判据改为「launcher 自己非 0 退出（非超预算被杀、非作业 SIGTERM）、无 `raw/launch.json`、无编排器 STATE」，不看退出码；原因 `launcher exited before starting any role (rc=N): <stderr 末行>`；`run_all` 的 config_refused 与 `classify` 用同一函数；main 按 R5-3 返回 CONFIG(3) | test_container：退出码 3/2/1 参数化（main=3、NOT_RUN、config_refused、原因含 rc 与末行、有 launch.json 退出 4 仍 VOID）；判据单测（rc 1/2/3/4/-9 × launch.json/STATE/超时/SIGTERM）；两处一致性全组合单测 |
| ② `capjobs.submit` | 新 `check_plan_argv`：用 `launches.launch_argv`（占位符代入 `EXAMPLE_PLACEHOLDER_VALUES`、追加 launches 会加的参数）构造作业实际 argv，取 `-m caplaunch` 之后用 `caplaunch.__main__.build_parser()` 只解析不执行（error 改抛 SubmitError，-h 等 SystemExit 也拒，stdout 不外泄）；在 prepare_submit 的 capkit-sha 核对之后调用 | test_submit：缺参、缩写（必填被缩写 / --par）、未知参数、int 类型错、-h 六例全部 rc 3、不调 ssh、不建快照、不写台账；构造 argv 单测（无残留占位符、含全部保留参数、无 -m caplaunch 拒绝）；fixture 计划 argv 补全为完整 caplaunch argv |
| 文档 | INTERFACES 新增「R5 后 Important 修复（2026-11-14）」R5b-1/R5b-2，旧条目 4 处加指针；jobs/README 第 10 步与 launches 判据两处 | — |

### 编排器起进程后的失败不会被误归
caplaunch 唯一起进程处 `_start_role`（orchestrator.py :599）只在 `_startup` 里，`run()` 先 `_prepare_tree`、`_write_launch_json`（write_json_stamped 原子写）再 `_startup`；写 launch.json 之前可能的退出只有 ConfigError 3、CapkitError 4、argparse 2、未捕获异常 1（orchestrator 不产生 ENV_INVALID）。起进程之后一定已有 launch.json ⇒ 不进这一类。

### 变异
mut/r1（4 项）、mut/r2（6 项，含协调者补充的 RS15 两项）：去掉修复后对应测试全红，未变异副本全绿。

### 未修
R5 其余 Minor / Nit（按用户决定）。
