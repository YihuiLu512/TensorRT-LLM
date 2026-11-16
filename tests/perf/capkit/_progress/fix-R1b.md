# R1b 修复进度（用户三项裁决：P3 只在指定进程启用、D14 按 E0 实测定、tokenizer 拷贝作业正式入口）

起点：<C> main a39c006f70；探针树 capprobe-J17871 733edd4edf。裁决原文：00_plan/DECISIONS.md「勘误与补充」最后一条（2026-11-13 23:5x）。

## 时间线
- 2026-11-13 读完必读材料（DECISIONS 末条、INTERFACES 勘误、R1 triage RS1–RS13、brief、CONSTRAINTS、fix-R1-probe P3 证据、设计稿 E0/D14）。
- 22:02:45 跑 tests/capprobe（P3 改动后，≤900 s）
- 22:05:39 P3 编排器接线完成；跑 tests/launch（≤900 s）
- 22:11:12 D14：gate item6 记 softirq、capjudge/d14.py、pilot --d14-rule、样本加 exclusivity.json；跑 tests/judge/test_pilot.py
- 22:17:19 tokenizer 正式入口写完；跑 tests/jobs（≤900 s）
- 22:22 INTERFACES R1b 勘误已写；开始重生成样本+selftest
- 22:23 样本重生成（+22 exclusivity.json，1475 个文件只改时间戳/sha/pilot），capjudge selftest --expect-samples 23 EXIT=0
- 22:23 开始逐目录跑 tests（每个 ≤900 s）
- 22:26 全部 tests 目录跑完（见 _local/r1b/logs/final_summary.txt）；py_compile 183 文件 rc=0；bash -n 3 脚本通过；patchcheck --rev HEAD PASS；开始本地 e2e

## 设计决定

- P3：判据放在 capprobe 导入时、按本进程命令行（`sys.orig_argv`，旧版本读 `/proc/self/cmdline`）匹配编排器写的 `CAPPROBE_ENTRY`（`capkit.probe_entry/1`：program realpath + args 原样）。不用 pid/ppid（re-exec 的多前端 launcher 要能匹配；pid 在 exec 前未知），不用 `sys.argv`（spawn 子进程会换成父进程的）。值校验在所有带 `CAPPROBE_*` 的进程里做（矛盾配置在哪都报错），身份文件只在匹配进程里读；缺 TIER 只在匹配进程报错。fork 未 exec 的子进程维持原行为（仍启用、各自写快照）。
- D14：空载读数取门禁第 6 项空转窗口（作业内第一次启动之前），在 gate 里加每核 irq/softirq tick；负载读数取 sidecar cpus.jsonl 档稳态窗口首末行差。阈值不写死，放在必须先于输入提交登记的 `capkit.d14_rule/1` 文件里（kind=d14_rule）；不够树数或登记晚则不写 `irq_softirq_is_experiment`（C13 第二半 NOT_RUN，RS3）。pilot 不再读 loopback_softirq.json（原来读了但不用，死代码，RS12）。
- tokenizer：作业体只用 bash+coreutils（无 python/find/awk/grep），容器内自查只读挂载与库不可见；只拷顶层白名单、权重后缀优先拒绝、软链出目录拒绝、超限即失败、sha256 清单、结束前复核；PLANNED 行用独立 purpose/assertions，其余计划挂模型库仍拒绝；测量快照排除该脚本。

## 改动

见 commit（<C> 一个提交；探针树产品文件未改，不提交）。新增：`tools/capcommon/probe_entry.py`、`tools/capjudge/d14.py`、`tests/capprobe/test_entry_spawn.py`；改名：`jobs/prep_tokenizer.sbatch.draft` → `jobs/prep_tokenizer.sbatch`（重写）；样本重生成（+22 个 `raw/env/exclusivity.json`，其余只变时间戳/sha/pilot 键集）；INTERFACES 勘误 R1b-1～6 + 正文指针。

## 测试

- 逐目录（rt.sh：venv python、ulimit -v 8G、MemoryMax=6G、timeout 900、fakebin PATH）：capcommon 188、capload 29、capprobe 130、envgate 60、grpc 32、jobs 141、judge 61、launch 61、nullsse 4、sidecar 5，全部 EXIT=0（`_local/r1b/logs/final_*.log`）。
- 修前变红：test_entry_spawn 4 failed（行为：3 个快照写者；旧报错文本）；test_config 收集错误；launch 入口测试 rc=5（外层 CAPPROBE_ENTRY 泄漏到 dummy）；D14 旧 pilot 在新样本上 `d14_softirq=None`；jobs 7 failed + test_scripts 收集错误（入口不存在，属缺失型红）。
- `samples/run_selftest.py --package capjudge --expect-samples 23` EXIT=0。
- py_compile 183 个 .py rc=0；`bash -n` jobs/capkit_inner.sh、capkit_job.sbatch、prep_tokenizer.sbatch 通过；`capprobe.patchcheck --rev HEAD`（探针树 733edd4edf，base 2f22a98d6e）PASS。
- 本地 e2e（`_local/r1b/e2e/`）：e0a-nullsse、e0b-s1-single 均 STATE VALID、judge rc=11（无 RED；NOT_RUN 来自未登记噪声带与 C13 第二半，同 integ-R1）；真实 capenv.gate --phase pre 产出 exclusivity.json（rc=6 ENV_INVALID 预期）；e0b 树上 `capjudge.pilot --d14-rule` rc=0，relay1 核 15：空载 softirq 0/200 tick，负载档 3～4/120～126 tick，结论 True（本地 VM、1.5 s 档，不是测量）。

## RS 规则扫描（额外扫出）

- RS12：INTERFACES §5.5 文件表里 `loopback_softirq.json` 一行仍写「D14 小程序的结果」→ 改为「不进 D14 结论」；pilot 读 loopback_softirq.json 的死代码删除；samples/README 加 exclusivity.json 说明。
- RS1/RS2：d14 读的 per_core 键（softirq_ticks/irq_ticks/total_ticks）与 gate 写方一致；sidecar cpus 列与 d14 一致；样本 exclusivity.json 与真实 gate 产出同形（e2e 实测）。
- RS3：缺空载读数、tick 总数为 0、无负载序、树不够、登记晚 ⇒ 不写结论键（测试覆盖）。
- RS6：tokenizer 作业目录 `J<job>-r<restart>`，已存在即拒绝。
- RS10：目的地与 .partial 已存在（含软链，lexists）拒绝；plan dir 按唯一 plan id 新建（`os.makedirs` 不带 exist_ok，已存在时抛 FileExistsError 而非 SubmitError，未改，记为 Nit）。
- RS13：spawn 子进程、MPI worker、其他导入者不再是快照写者（test_entry_spawn 实测）。

## MEASURED / INFERRED / NOT RUN

- MEASURED：上面全部测试/自测/patchcheck/e2e 的退出码与计数（日志在 `_local/r1b/logs/`、`_local/r1b/e2e/*.log`）。
- INFERRED：C13 减的是 irq+softirq，规则只看 softirq（irq 在本地读数为 0，真实节点未知）；产品角色真实 trtllm-serve 命令行（`#!` 脚本）能匹配，只由单测（shebang 用例）推断，未在真实镜像里跑。
- NOT RUN：集群上任何作业（tokenizer 拷贝作业、E0）；真实产品进程 + 探针覆盖层的端到端（本地 e2e 两个场景都没有 product 角色，launch.json `probe_entry.roles` 为 `{}`）；D14 规则文件的正式值（需主会话/用户写好并在首个 E0 作业提交前登记）。
