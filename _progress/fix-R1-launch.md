# fix-R1-launch（FX-launch）进度

负责簇：K2（拷入）、K3、K12（launch 侧）、K19、K21、K27、K28（orchestrator 侧）
独占文件：tools/caplaunch/**、tests/launch/**、configs/**

## 时间线
- 2026-11-13 开工：建 fakebin（_local/fx-launch/fakebin，10 个脚本，exit 97），读 triage/brief。
- 读完 orchestrator/procfs/scenario/__main__、tests/launch、judge C11/C13/P-PIDS、capcommon/gate、launches.py、INTERFACES §5/§6.6/§8.3。
- 初步核实（读码）：
  - K2：编排器唯一拷入口是 --job-env-dir，作业 env/ 里没有 overlay_manifest.json ⇒ 属实（待测试复核）。
  - K3：_scan 的 descendants 含所有顶层角色的子孙（含 sidecar 的 nvidia-smi）；另 _discover 每个 housekeeping 周期把这些子孙写进 unattributed ⇒ C13 启动级也会红（RS13 额外扫出，同根因）。
  - K19：handler 在 _stop_all/_verify_exit/_finalize 期间仍挂着 ⇒ 属实；另 _startup 把信号 LaunchAbort 重包成 not_ready_before_origin（RS4 额外）。
  - K21：每轮只 poll 一次、落后时背靠背两次 poll ⇒ seq 不前进 ⇒ 不满足、清零 ⇒ 属实（判定器 C11 要求逐个计划时刻都有记录，不能跳过；修法=落后时把这次轮询推迟到上次实际轮询后 ≥1.5 个快照周期，且不越过期限）。
  - K27：快照分支对工具角色后代 `continue` 静默；但下面 proc.children 循环会把它记进 unattributed（seen_via proc.children），所以「C13 看不到」在现码下不成立，丢的是快照证据；K3 把监测/负载类子树排除后会变成完全静默 ⇒ 部分属实，修。
- 开始改 tools/caplaunch/procfs.py、orchestrator.py、__main__.py。
- 代码改动已落（未测）：procfs.descendant_records（替换死代码 descendants / is_descendant_of）；orchestrator：MONITOR_CLASSES、_load_overlay、_private_dir、_on_signal/cleaning/hurry、_gate_next_abs、_proc_scan/_note_monitor_child、procs/monitor_children.json；__main__ --overlay-manifest 必填。
- 下一步：改 tests/launch 的 conftest 与 fakes，新增测试。
- 20:49:08 开跑 tests/launch（预计 2–4 分钟）
- 20:51:10 run1: 50 passed / 1 failed（K21 用例阈值写错：最大滞后≈650ms<800ms，门已过）；改阈值 500ms。接主会话转达：加 --nvidia-smi（必填）、{nvidia_smi} 占位、launch.json 记录、监测子进程记 exe/is_nvidia_smi。
- 20:51:44 run2 tests/launch（含 --nvidia-smi、smoke RS3）
- 20:53:49 run2: 52 passed rc=0。加 tiergen 顺序修复+用例；建 _local/fx-launch/pre（HEAD caplaunch + CLI/占位垫片）跑修前对照
- 20:57:41 K21 用例 settle 改 4：修后 1 passed、修前（pre）门未过红。tiergen 修前红。开始本地真实工具端到端 _local/fx-launch/e2e（e0a-nullsse，真实 sidecar + 假 nvidia-smi，约 1–2 分钟）
- 20:58:46 e2e fxl2（修后）VALID、8 档 pids_stable 全 True、monitor_children 75 条；改用 ensure_own_dir + run-dir 检查；跑修前 e2e（pre/tools）
- 20:59:40 修前 e2e（pre）：8 档 pids_stable 全 False、P-PIDS RED、unattributed 74、judge rc=10。接主会话裁定：--job-env-dir 必填、aux_helper 白名单
- 21:00:20 已改：--job-env-dir 必填（非目录⇒NOT_RUN job_env_dir_missing）、AUX_HELPER_RULES + roles.json aux_helpers + tier.json scans.aux_helpers；跑 run3
- 21:05:40 加 proc_kinds（场景必填、产品角色才可非空、launch.json roles 带出；生成器重生成 7 个场景）；跑 run5
- 21:08:27 e2e fxl3（终版前）：VALID、8 档 pids_stable True、unattributed []、aux 2（nullsse/capload resource_tracker）；C13 档级红来自 sidecar GPU 线程未声明（他方）。_register 时把已登记进程移出 monitor 清单；跑 run6

## 结论（终版，2026-11-13）

| 簇 | 核实 | 证据 | 改动 | 新增/改动测试 | 修前红 |
|---|---|---|---|---|---|
| K2 | 属实 | 唯一拷入口是 --job-env-dir，作业 env/ 无 overlay_manifest.json | `__main__` 新增必填 `--overlay-manifest`；`orchestrator._load_overlay` 原样写 raw/env/overlay_manifest.json，launch.json `overlay_manifest{source,present,sha256,needed_by_roles}`；有 probe_tier 角色而缺失 ⇒ `_startup` 前 NOT_RUN（stop reason overlay_manifest_missing）；job env 里另有一份且内容不同 ⇒ 退出码 3 | test_full_launch_passes（增断言）、test_overlay_missing_with_probe_role_is_not_run、test_overlay_missing_without_probe_role_runs | 是（pre：缺文件/KeyError/rc 0≠5） |
| K3 | 属实，且多一处同根因 | 修前 e2e（真实 sidecar + 假 nvidia-smi）8 档 pids_stable 全 False、P-PIDS RED、unattributed 74；_discover 还把这些子进程写进 unattributed（C13 红） | MONITOR_CLASSES=(load_source, observer)；`_scan` 的 descendants 不含这些角色的子树，`monitor_descendants` 另列；`_discover` 不再把它们写进 unattributed；逐个记 raw/procs/monitor_children.json | test_monitor_children_do_not_break_pid_stability、test_under_test_children_still_break_pid_stability | 是 |
| K12 | 有意为之（launch 侧） | test_fingerprint 依赖 probe-tree | docstring 写明依赖与「不 skip」理由，缺失时 assert 给出清晰信息 | — | — |
| K19 | 属实 | 修前：第二个 SIGTERM 让 LaunchAbort 从 _stop_all 逃出，rc=1、无 launch_end | `_on_signal`：第一次设 cleaning 并中止；收尾期间再收信号只记录并 hurry（跳过剩余宽限直接 SIGKILL）；信号安装挪进 try；`_startup` 不再把信号中止改写成 not_ready_before_origin；launch_end `signals` | test_second_signal_during_cleanup_still_finalizes | 是 |
| K21 | 属实 | 修前：恢复后背靠背轮询 seq 不前进，门未过 | `_gate_next_abs`：落后时与上次轮询至少隔 1.5 快照周期（不越过期限），每个计划时刻仍各有记录 | test_gate_survives_orchestrator_stalls（SIGSTOP 1 s/运行 0.6 s 循环，settle 4） | 是（pre 门 passed=False） |
| K27 | 部分属实 | 现码里该进程会经 proc.children 循环进 unattributed（C13 看得到），丢的是快照证据；K3 改后会变成完全静默 | 快照分支：非产品角色下的未登记写快照进程一律进 unattributed（seen_via probe.snapshot，带 registered_ancestor） | test_tool_role_snapshot_child_is_unattributed | 是（pre seen_via=proc.children） |
| K28 | 属实（加固） | makedirs(exist_ok) 接管已存在目录/跟随软链 | `_private_dir` = capcommon.snapshot.ensure_own_dir + 组/他人不可写；用于 --shm-dir、本次快照子目录、--run-dir；在建树之前检查 | test_shm_dir_not_private_is_refused、test_shm_subdir_symlink_is_refused | 是 |

主会话追加（均已做）：--nvidia-smi 必填 + `{nvidia_smi}` 占位 + launch.json `nvidia_smi{path,realpath,sha256}`；--job-env-dir 必填（非目录 ⇒ NOT_RUN job_env_dir_missing）；AUX_HELPER_RULES（resource_tracker、forkserver）+ roles.json `aux_helpers`；场景必填 `proc_kinds`，launch.json roles 带出。

### RS 额外扫出
- RS4：`_startup` 把信号中止重包成 not_ready_before_origin（已修）。
- RS13：同 K3，_discover 的 unattributed（已修）；监测清单里会出现就绪前尚未登记的工具 worker，_register 时移出（已修）。
- RS3：smoke `tree.empty_delta` 在配置缺 httpprobe_requests 时以被检对象自己的条数为准 ⇒ 改为 NOT_RUN（test_smoke_empty_delta_without_expected_count_is_not_run，修前 PASS）。
- RS5：tiergen 在 --schedule-out 已存在时先写了 --out 再拒绝，留下半份登记 ⇒ 改为先查后写（test_main_bad_schedule_out_writes_nothing，修前红）。
- RS12：procfs.descendants / is_descendant_of 死代码删除（函数级，非文件删除）。
- 未改：smoke env.import_probe 的 PASS 不依赖对照是否「有信息量」（只记录 control_informative）；BaseException 分支不写 launch_end（重抛，STATE 缺失 ⇒ launches 判 VOID）。

### 给 FX-judge 的形状（最终）
- tier.json `scans.<wstart|wend>` = {t_mono_ns, registered_live:[[pid,st]], registered_total, descendants:[[pid,st]]（不含 MONITOR_CLASSES 角色子树、不含 aux helper）, monitor_descendants:[[pid,st,role]]（监测类角色的未登记子孙）, aux_helpers:[[pid,st,role]]}。
- roles.json：新增 `aux_helpers`:[{pid,ppid,starttime_ticks,cmdline,role,measure_class,category:"aux_helper",rule,first_seen_mono_ns}]；unattributed 的 probe.snapshot 项多 `registered_ancestor`（null 或 {pid,role}）。
- raw/procs/monitor_children.json（capkit.monitor_children/1）：{monitor_classes, n_proc_scans, coverage, children:[{pid,ppid,comm,starttime_ticks,role,measure_class,exe,is_nvidia_smi,first_seen_mono_ns,last_seen_mono_ns,n_seen,seen_via}]}。
- launch.json：overlay_manifest、nvidia_smi、roles[].proc_kinds、orchestrator.monitor_classes、orchestrator.aux_helper_rules；launch_end.json：signals。

### 需要他方 / 主会话
- FX-jobs：classify 要按 caplaunch STATE 认 NOT_RUN 的新 stop reason：overlay_manifest_missing、job_env_dir_missing、orchestrator_signalled（launches.py 已追加三个参数）。K14：编排器最坏收尾时长 = Σ角色(stop_grace_s + kill_after_s) + 残留核验最多 2×kill_after_s；KILL_GRACE_S 应不小于它（第二个信号现在会让编排器跳过宽限直接 SIGKILL）。
- FX-env / FX-judge：真实 sidecar 开 GPU 采样后有一个未声明名字的 Python 线程（e2e fxl3：sidecar pid 的 tid 未识别）⇒ C13 档级 RED。要么 sidecar 在就绪文件 info.threads 声明，要么判定器 C13 的线程检查只看 under_test。
- INTERFACES 勘误原文（主会话改）：见最终报告。

### MEASURED / INFERRED / NOT RUN
- MEASURED：tests/launch 终版 60 passed rc=0（t_launch_6.log）；修前对照（_local/fx-launch/pre）新用例全部按预期红（t_pre.log、t_pre2.log、t_k21_pre.log）；e2e fxl3（终版前一版）VALID、pids_stable 8/8 True、unattributed []、aux 2；修前 e2e 8/8 False。
- INFERRED：真实 nvidia-smi 为 ELF 时 is_nvidia_smi 可为 True（本机只有脚本替身，exe 是 shell ⇒ False）；前端 proc_kinds=["frontend"] 未用真实 trtllm-serve 核实。
- NOT RUN：含产品角色的真实场景（e0a-frontend-sfe、e0c）；集群。
