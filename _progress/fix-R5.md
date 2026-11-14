# fix-R5（R4 自审第二个追加轮修复）进度

- WT = `_local/wt-r4`（分支 capkit-r4，起点 84c06955d387449807a36262bed22cefacdaf199）；C（01_code main）只读；探针树只读。
- 工作目录 `WT/_local/r5-fix/`（fakebin 十个替身、pyc、logs、作业路径 e2e）。
- 范围：R4 triage 标「R5 修」的 U1–U11；其余不碰。

## 进度日志
- 2026-11-14 13:02:15 开工：已读 R4 triage/brief、路 A R1–R6、路 B 全文、fix-R4 进度。
- 2026-11-14 13:05:53 U1：修复与单测写完（tests/launch/test_aux_helper.py 64 passed，e2e 参数化 3 passed）；作业路径 e2e 脚本拷到 _local/r5-fix/jobpath 并按 capkit_inner.sh 设环境；临时还原 orchestrator.py 到 HEAD 跑 RED（约 2 分钟）
- 2026-11-14 13:06:28 U1 RED 复现（修复前 orchestrator）：作业路径 e2e e0a-nullsse（J8500001，导出 PYTHONDONTWRITEBYTECODE=1）launches VALID → judge rc=10、C13 RED「1 unattributed」；已恢复修复版 orchestrator.py，开跑修复后 e0a
- 2026-11-14 13:09:17 U1 修复后作业路径 e2e e0a-nullsse（J8500002）judge rc=11、0 RED、C13 明细 2 aux helper。U2 改完：submit 覆盖检查加 <W>（ro）与登记表各 files.* 的 realpath；job.env 新增 CAPJOBS_WORKDIR；inner --must-read W、缺失 ENV_INVALID；tests/jobs 新用例通过。开始 U3。
- 2026-11-14 13:10:51 U3 改完：launches.prestart_config_refusal（classify 同一判据），记录 config_refused，main 在无拷回失败/ENV_INVALID 时返回 CONFIG(3)；launches.json 顶层 config_refused 计数；tests/jobs 新用例通过。开始 U4–U11。
- 2026-11-14 13:15:18 U4（scenario 拒 role env 的 CAPPROBE_*）、U5（demand 写入进 try）、U6（lag 只认 ASCII 数字；同根因另修 capload/capgrpc 端口、gate TRES、submit 作业号 3 处）、U8（req_gpus 只取 ReqTRES，新增 gpus_checked）、U9（PCI vendor/device 读不到 ⇒ nics unreadable）、U10（PINS_TIMEOUT_S 移到 launchplan，进作业内剩余时间检查与 submit 时间预算）、U11（inputs.py 注释）改完并有测试。开始 U7。
- 2026-11-14 13:17:05 U7（caplaunch allow_abbrev=False，测试遍历全部长选项）完成；INTERFACES 新增「R5 自审修复」10 条与 6 处指针。下一步 jobs/README、caplaunch/capjobs README 文档，然后变异与全量验证。
- 2026-11-14 13:21:24 RS22 同根因补修：绑核表生成失败的启动记 pin_table_failed，main 同样退出 3；submit 的 arm files 查找收成一处（arm_files，RS15）；ASCII 数字补测（capload/capgrpc/submit reconcile 作业号；slurm-view 一处本来就被 JOB_ID_RE 挡住，改动撤回）。后台跑变异 21 项（约 5–8 分钟，_local/r5-fix/mut/r1）。
- 2026-11-14 13:22:29 变异运行中；INTERFACES R5-9 措辞收窄（int() 崩溃 vs 放过两类分开写）；直接 caplaunch e2e 脚本拷到 _local/r5-fix/direct（路径 r4-fix→r5-fix、shm 后缀 r5d）；建测试用软链 WT/probe-tree（提交前移走）。
- 2026-11-14 13:24:26 变异 21 项全红（mut/r1；U3-exit 第一次的变异不是合法 Python，r2 重跑为 3 failed），未变异副本全绿。后台跑全量验证 v1（10 目录 + selftest + patchcheck + py_compile + bash -n，约 8–10 分钟）。
- 2026-11-14 13:32:37 全量验证 v1 全绿（logs/v1/summary.txt：capcommon 190、capload 33、capprobe 159、envgate 71、grpc 36、jobs 186、judge 85、launch 189、nullsse 4、sidecar 5；selftest 25；patchcheck PASS；py_compile 187/0；bash -n 3/0）。开跑作业路径 e2e 终版（e0a、e0b，约 3 分钟）。
- 2026-11-14 13:33:47 作业路径 e2e 终版：e0a-nullsse（J8500011）、e0b-s1-single（J8500012）launches VALID、judge rc=11、0 RED、无 INPUT_INVALID；aux helper 命令行带 -B 被认出、unattributed 0；job.env 有 CAPJOBS_WORKDIR。开跑直接 caplaunch e2e 两条。
- 直接 caplaunch e2e（R4 做法，脚本在 _local/r5-fix/direct，caplaunch 用 `env -u CAPKIT_CODE_SHA -u CAPPROBE_PATCH_SHA` 起）：e0a-nullsse（local-r5da）caplaunch rc=0、STATE VALID、judge rc=11、0 RED、C10 PASS 8/8；e0b-s1-single（local-r5db）同，C10 PASS 5/5。
- fakebin.log：只有门禁调用的 `scontrol show job <id>`（替身 exit 97），没有碰集群。
- 残留：/dev/shm/yihlu-local-r5da-r5d、yihlu-local-r5db-r5d 两个空目录（e2e 的 shm 根，R4 也留下了 r4d 两个）；按约束不删，待用户决定。

## 结果汇总

### 处理
| 簇 | 处理 | 测试 / 证据 |
|---|---|---|
| U1 | `aux_helper_rule` 接受 `python [解释器选项] -c <代码>`：无参单字母 `bBdEIOPqsSuv`（可合写）、`-W`/`-X`（同词或下一词）；`-m`、脚本、`-i`、`-Bc`、代码后有参数仍拒（勘误 R5-1） | `tests/launch/test_aux_helper.py`（19 种等价形式 × 2 规则、19 种反例、7 组真实写方：子进程里拦截 multiprocessing 实际生成的 argv）；e2e 参数化 3 个环境；作业路径 e2e 修复前 RED（J8500001：judge rc=10、C13「1 unattributed」，argv `[python,-B,-c,…]`）→ 修复后 0 RED |
| U2 | submit 覆盖检查加 `--workdir`（ro）与登记表及各 arm `files.*` 的 realpath（`workdir_inputs`）；`job.env` 新增 `CAPJOBS_WORKDIR`；inner `--must-read` W，缺变量 ENV_INVALID（勘误 R5-2） | test_submit 两个新用例（W 未挂载拒绝 / ro 挂载通过；软链指出挂载拒绝）；test_scripts must-read 与缺变量用例 |
| U3 | 仍 NOT_RUN；`config_refused` 逐启动与顶层计数；`launches.main` 返回 CONFIG(3)（RUNTIME、ENV_INVALID 优先）；同根因（RS22）把绑核表生成失败也记 `pin_table_failed` 并同样退出 3；停止规则口径写进 jobs/README 与勘误 R5-3（00_plan 未改） | test_container 改断言（原来断言 0）、参数化退出码用例 |
| U4 | 场景任何角色 `env` 有 `CAPPROBE_*` 键 ⇒ 拒绝 exit 3（勘误 R5-4） | test_scenario 6 键 × 3 角色 × P0/P1，含反例 `XCAPPROBE_INJECT` 通过 |
| U5 | demand 文件写入进 try ⇒ NOT_RUN | test_unwritable_demand_file_is_not_run |
| U6 | lag 只认 ASCII 数字；同根因另修 capload/capgrpc `--target` 端口、gate TRES、submit 作业号（reconcile-submit、sbatch --parsable）（勘误 R5-9） | e2e lag 6 反例 + 正对照；各处单测 |
| U7 | caplaunch `allow_abbrev=False`（勘误 R5-6） | test_units 遍历全部长选项的缩写（空格与 `=` 两种）均 exit 2，正对照解析通过 |
| U8 | `req_gpus` 只取 ReqTRES，新增 `gpus_checked`（勘误 R5-7） | test_envgate_gpucount 无 ReqTRES 用例 |
| U9 | PCI vendor/device 读不到或空 ⇒ `nics` unreadable（勘误 R5-8） | test_envgate_gate 三种缺失 |
| U10 | `PINS_TIMEOUT_S` 移到 launchplan，进作业内剩余时间检查与 submit 时间预算 `pin_tables_s`（勘误 R5-5②） | 边界用例（差 1 秒 NOT_RUN / 恰好够跑）；test_submit 预算断言 |
| U11 | `capjudge/inputs.py` 注释改为绑核表逐启动 | — |

### 变异
`_local/r5-fix/mut/r1`（21 项）+ `r2`（U3-exit 重跑）：去掉每项修复后对应新测试全红；未变异副本全绿。

### 未修
- R4 分诊 U12–U17（按范围）。
- 同类未改：画像 `local_disk_type` 的 fstype/rota/model 读不到仍记 `?`（md/dm 设备本无 device/model，改成 unreadable 会使这类节点画像恒 unreadable，需真实节点核对）；`capcommon.clock.parse_wall_iso` 的 `isdigit`（`int()` 失败本来就是约定的 ValueError，行为一致）；`/proc`、`/sys` 读数的 `isdigit`（内核只写 ASCII）；正则 `\d` 接受非 ASCII 十进制数字（`int()` 能正确转换，不崩）。
- 时间不足而跳过的启动（NOT_RUN）仍不影响退出码：属预算设计，非起进程前的失败。
