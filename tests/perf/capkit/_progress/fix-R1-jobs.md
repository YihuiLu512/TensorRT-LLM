# FX-jobs R1 修复进度（capkit-J17871）

负责簇：K1（接线）、K2（作业侧）、K6（envfiles）、K8、K9、K11、K12（jobs 侧）、K14–K17、K18（envfiles 侧）、K22、K23、K25、K26
独占文件：`jobs/**`、`tools/capjobs/**`、`tests/jobs/**`、`tools/capcommon/policy.py`
fakebin：`_local/fx-jobs/fakebin/`（10 个脚本，argv 追加到 `_local/fx-jobs/fakebin.log` 后 exit 97）

## 日志
- 2026-11-13 开始：已读 triage.md、brief.md、路 A R1/R2/R6/R8/R9/R11/R12、路 B B01–B16、路 C O2/O3；已建 fakebin。下一步读代码。

## 逐簇核实与改动

| 簇 | 核实 | 证据 | 改动 | 新测试 | 修前红 |
|---|---|---|---|---|---|
| K1（接线） | 属实 | 修前 inner 门禁调用无 `--idle-window-s/--idle-busy-max-frac/--max-alloc-gpus`，jobs/ 与 tools/capjobs/ 无任何 hwprofile 调用（读码；修前副本跑新测试 `KeyError/断言`） | register.json 顶层必填 `environment` 段（submit.register_environment，现算 sha 核对）→ job.env `CAPJOBS_MAX_ALLOC_GPUS/IDLE_WINDOW_S/IDLE_BUSY_MAX_FRAC/SLURM_VIEW_WAIT_S/CHASSIS_SOURCE`；inner：等 slurm_view.json → `capenv.hwprofile`（--nvidia-smi、--role hwprofile、--container-runtime pyxis-enroot）→ 门禁带全部参数 + `--hw-profile`；`--registered-hw-sha` 收 `UNREGISTERED` | test_inner_run_wires_registered_limits_into_profile_gate_and_launches（并用真实 capenv.gate 解析器、真实 hwprofile.main 跑录下的 argv）、test_inner_unregistered_profile_capture_stops_at_gate、test_inner_waits_for_slurm_view_then_env_invalid、test_unregistered_hw_profile_is_submittable、forbidden 新增 register_* 4 例 | 是（破坏性：去掉 hwprofile / 去掉 idle 参数各 1 红） |
| K2（作业侧） | 属实 | launches 只读清单、不传给编排器 | launch_argv 追加 `--overlay-manifest $SNAP/overlay_manifest.json`（另按主会话追加 `--nvidia-smi`、`--job-env-dir <作业目录>/env`），三者入 RESERVED_FLAGS | test_launches_statuses（断言编排器收到三项）、test_plan_rejects 新 3 例 | 是（破坏性） |
| K6（envfiles） | 属实 | 修前 check_gpu_idle 2 张可见/请求 1 → passed（修前副本跑 test_gpu_idle 在 `gpu_count == 2` 处红）；`--gpus 0` 必 NOT_RUN | envfiles 改调 `capenv.gpu.check_visible`；submit 要求 `--gpus == environment.gpus`，gpus=0 打印 GPU-IDLE 会 NOT_RUN 的警告 | test_gpu_idle（请求 0/1 共 9 情形）、test_gpu_idle_cli_requires_explicit_inputs、test_zero_gpus_warns、forbidden gpus_vs_register | 是 |
| K8 | 属实 | compare 只比 `["f", size]` | listing 带 sha256，记录 `content_check: sha256` | test_copy_and_compare（同大小不同字节） | 是 |
| K9 | 属实 | policy.check_mounts 只 normpath | policy：字面路径与 realpath、库路径与其 realpath 两两比较（`_related`→公开 `paths_related`）；envfiles.check_mounts：挂载目的与 must-read 的路径/realpath 碰库即不过 | test_check_mounts_resolves_symlinks、test_check_mounts（K9 段）、forbidden mount_symlink_to_lib | 是 |
| K11 | 属实（文档） | README 偏差 3 已由 inputs.JOB_KEY_FIELDS 解决 | 改为一句指向勘误；同时删过期的偏差 1（包位置）、偏差 4（ts） | — | — |
| K12（jobs 侧） | 有意为之（维持报错） | 两份仓库外输入 | conftest.external_input：缺文件时 `pytest.fail` 写明缺哪个；README「测试」列出两份依赖 | （辅助函数） | — |
| K14 | 属实 | killpg 只到编排器进程组；角色 start_new_session | launches 成为 child subreaper；运行中每 0.5 s 记录后代；编排器退出/超时被杀后对存活后代 SIGTERM→kill_grace_s→SIGKILL，记 `leftovers`，扛过 SIGKILL 记 VOID 并跳过后续；新增计划必填 `kill_grace_s`，预算与剩余时间判据计入 | test_leftover_processes_are_killed（子进程跑 main）| 是（破坏性） |
| K15 | 属实 | classify 只看退出码，exit 5 → VOID | 以编排器自己写的最后一行 STATE 为准；超时→VOID；编排器写 VALID 而退出码非 0/5 → VOID | test_launcher_state_is_kept | 是 |
| K16 | 属实 | rc=None 写 ENDED/SUBMIT_FAILED | rc ∈ {None, 0(无作业号), 255} ⇒ NOTE 行 `UNKNOWN_SUBMIT: …`（台账 status 集合不含新值，见他方配合），不重投；record-end 拒绝；新子命令 `reconcile-submit --job-id J`（scontrol Command= 本计划脚本，或 batch.tsv 写着本计划号）/`--not-submitted`（squeue rc 0 且无本计划、无 batch.tsv） | test_submit_outcome_unknown_is_not_ended[3]、test_reconcile_submit_by_scontrol_then_record_end、test_reconcile_submit_by_batch_tsv_and_not_submitted、test_reconcile_not_submitted | 是 |
| K17 | 属实 | squeue rc≠0 一律 UNKNOWN | 仅 stderr 含 `Invalid job id specified` 时退到 sacct；其他错误仍 UNKNOWN | test_jobstate_purged_job_falls_back_to_sacct[5]、test_jobstate_query | 是（修前副本 `UNKNOWN != ENDED`） |
| K18（envfiles） | 属实 | envfiles 自拼两条 nvidia-smi 查询 | 改为 capenv.gpu.check_visible；`--nvidia-smi` 必须绝对路径（RS9） | test_gpu_idle（断言执行的是给定路径）、forbidden nvidia_smi_from_path | 是 |
| K22 | 属实 | 本地根名不含重启次数；mkdir 失败 trap 仍 sweep | 目录 `<base>/<user>-<job>-r<restart>`（scratchpath 的 job-id 传 `<job>-r<restart>`）；`run` 模式只 sweep/删 mkdir 成功的根，`finalize-only` 只认存在、非软链、属主本用户的根；copyback sweep 的两个根改为可省略 | test_inner_requeue_and_foreign_scratch_dir、test_sweep_skips_roots_not_owned | 是（破坏性 2 例） |
| K23 | 属实 | sbatch 在分配里 sha256 整个镜像 | 取「提交端算一次」：submit 现算 sha 等于 `--image-sha256`（前后 stat 一致），job.env 记 size+mtime_ns；sbatch 只比 `stat` 大小与 `date -r +%s%N`；理由：不占分配、不用猜哈希耗时计入预算 | test_submit_end_to_end（job.env）、forbidden image_sha_wrong、test_batch_* | 是 |
| K25 | 属实（死代码） | ts 组保留，copy_probe_ts 恒 0 | 删函数及 glob/shutil/launch_shm_subdir 导入、`probe_ts_files` 字段与 README 偏差 4（函数级删除，非整文件，未用 git rm） | test_launches_statuses 去掉 ts 期望 | — |
| K26 | 属实 | 原文末无换行时 `# exit:` 粘行；rc None 写 `exit: None` | 补换行；rc None 写 `none (ssh not run or timed out)` | test_exclude_live_raw_text | 是（修前副本粘行） |

## RS 规则额外扫出

- **X1（RS1/RS2）拷回核对必败**：caplaunch 在启动树根写 `STATE`，copyback.compare 只从目的端剔除顶层 STATE ⇒ 真实启动 diff=["STATE"]、FAIL、记 VOID（MEASURED：`_local/fx-jobs/repro` 用 copy_tree 复现 `FAIL ['STATE']`）。修：顶层 STATE 只许追加（副本以原件字节开头，sha 比对）。测试 test_copy_tree_with_launcher_state、test_launcher_state_is_kept。
- **X2（RS4）作业收到 SIGTERM**：launches 无处理器，Slurm 的 SIGTERM 直接杀死它，当前启动的进程树不收、launches.json 不写。修：main 装 SIGTERM 处理器（只置标志），当前启动按超时路径收树（宽限取 min(kill_grace_s, 10 s)）、记 `terminated`/VOID，其余启动 NOT_RUN，照常写记录。测试 test_sigterm_stops_launch_and_skips_the_rest（破坏性：去掉处理器即红）。
- **X3（RS7）预算**：submit 的 `--time` 核对加上门禁空转窗口与 slurm_view 最长等待、每启动 kill_grace_s；剩余时间判据同样加 kill_grace_s（破坏性 RS7_budget_no_waits 红）。
- **X4（RS8）`slurm_view_wait_s`**：INTERFACES §5.5.2 写「S 取自登记表」，修前是 submit 命令行参数 ⇒ 改为 register.json `environment.slurm_view_wait_s`，删除 `--slurm-view-wait-s`。
- **X5（RS9）`--nvidia-smi` 允许裸名**（走 PATH）⇒ 必须绝对路径（破坏性 RS9_nvsmi_from_path 红）。
- **X6（RS12）删复述 / 无读者的写方**：job.env 的 `CAPJOBS_KILL_GRACE_S`（无读者，计划里已有）删除；gpu_idle.json 的 `expected_gpus_requested`（与 check_visible 的 `expected_gpus` 重复）删除；README 过期的偏差 1、3、4 删除或改为指针；`CAPJOBS_EXPECT_GPUS` 改名 `CAPJOBS_MAX_ALLOC_GPUS`（与门禁参数同名）。
- **X7（RS5/RS3）**：sbatch 退出 0 却无作业号、ssh 255 也是「结果不明」，与 rc=None 同按 UNKNOWN_SUBMIT 处理（修前记 SUBMIT_FAILED）。
- 扫过未改：RS6 copy_tree 目的已存在即 FileExistsError（记 FAIL），树名含重启次数；RS10 job dir/快照 `mkdir` 不复用、`open(...,"x")`；RS13 leftovers 已记录。残余（INFERRED，未改）：hwprofile 每条外部命令 30 s 超时、overlay 拷贝不在预算里（量级小，由每启动 margin_s 吸收）。

## 需要他方配合（含 INTERFACES 勘误原文）

1. **主会话 / capcommon.ledger 属主**：`SUBMISSION_STATUSES` 没有 `UNKNOWN_SUBMIT`，本轮用 `event=NOTE`、`reason` 以 `UNKNOWN_SUBMIT` 开头表达；用途标记写在 PLANNED 行 `reason`（`purpose=measurement|hw_profile_capture`），NOTE/ENDED 行前缀重复。若要成为正式状态/列，需改 ledger（不归我）。代码里没有「连续两次 ENV_INVALID」的计数实现，只在台账字段上区分。
2. **FX-judge**：envfiles 写的 `gpu_idle.json`（schema `capkit.gpu_idle/1`）最终键集（MEASURED，`_local/fx-jobs/repro/gpu_idle.json`）：`schema identity written_mono_ns written_wall` + `status reasons expected_gpus nvidia_smi visible_gpus gpus compute_apps queries count_check positive_control apps_check positive_control_ok` + `gpu_count passed outcome`（`outcome` ∈ PASS/FAIL/NOT_RUN）。判定器只读 `positive_control_ok`；样本请按此对齐。另：`capenv.gpu check` CLI 也以同一 schema 名写另一种形状（idle_check 载荷），作业流程不调用它——两个形状同名，建议 FX-env/主会话改名或在勘误里写明。
3. **FX-judge / 主会话**：请求 0 张 GPU 时 `positive_control_ok=false` ⇒ GPU-IDLE NOT_RUN（fail-closed，已按裁定在 submit 打警告）。
4. **FX-launch**：编排器收到 `--overlay-manifest`、`--nvidia-smi`、`--job-env-dir <作业目录>/env`（均由 capjobs.launches 追加）；编排器写在启动树根的 `STATE` 现在被 copyback 接受（只许追加）。launches 已是 child subreaper 并在编排器退出后清理残留后代（记 `leftovers`），与编排器自身的 `_verify_exit` 不冲突。
5. **主会话**：`capcommon.scratchpath` 未改；作业把 `<job>-r<restart>` 当作 job-id 传入（INTERFACES §5.9 的路径形状因此变化，见勘误）。

### INTERFACES 勘误原文（请主会话追加到「勘误与补充」）

> - **2026-11-13，R1 自审修复（capjobs）**：
>   1. **§9.2 `register.json` 新增顶层必填段 `environment`**：恰好五个键——`gpus`（整数，0 或 1，向 Slurm 申请的 GPU 数；不加载模型的作业取 1）、`idle_window_s`（数值，(0, 3600]，门禁第 6 项空转窗口）、`idle_busy_max_frac`（数值，(0, 1]，第 6 项每核忙碌比例上限）、`slurm_view_wait_s`（数值，(0, 3600]，等 `slurm_view.json` 的最长秒数；§5.5.2 的 S 即此值）、`chassis_source`（`slurm_view` 或 `dmi`，传给 `capenv.hwprofile`）。`capjobs.submit submit` 按 `REGISTRY.tsv` 中该 sha 的 `relpath` 读 `register.json`、现算 sha 核对，缺段、多键、取值越界一律拒绝提交；`--gpus` 必须等于 `environment.gpus`；`--slurm-view-wait-s` 命令行参数删除。这些值经 `job.env`（`CAPJOBS_MAX_ALLOC_GPUS`、`CAPJOBS_IDLE_WINDOW_S`、`CAPJOBS_IDLE_BUSY_MAX_FRAC`、`CAPJOBS_SLURM_VIEW_WAIT_S`、`CAPJOBS_CHASSIS_SOURCE`）由内层脚本显式传给 `capenv.hwprofile`、`capenv.gate` 与 `capjobs.envfiles gpu-idle`，没有缺省值。判定器与编排器不读该段。
>   2. **作业内顺序（§5.5）**：内层脚本在门禁之前写 `env/hw_profile.json` 与 `env/hw_raw.json`（`capenv.hwprofile`，`--role hwprofile`）；`chassis_source = slurm_view` 时先等 `env/slurm_view.json`。`--registered-hw-sha` 可为 `UNREGISTERED`（只采画像的作业，门禁恒 ENV_INVALID），其台账 PLANNED 行 `reason` 为 `purpose=hw_profile_capture`，其余作业为 `purpose=measurement`；ENDED / NOTE 行重复该标记。
>   3. **§5.5 `gpu_idle.json`** 由 `capenv.gpu.check_visible` 产生（可见数 == `environment.gpus`；≥1 时先要列出 ≥1 张；0 时要求无卡，阳性对照记 NOT_RUN），载荷键为 `status reasons expected_gpus nvidia_smi visible_gpus gpus compute_apps queries count_check positive_control apps_check positive_control_ok gpu_count passed outcome`；`image.json` 的 sha256 由提交端计算（`sha256_source`），作业内只核大小与 `mtime_ns`。
>   4. **§5.9 节点本地目录**改为 `<base>/<用户名>-<作业号>-r<重启次数>`（`capcommon.scratchpath` 的 `--job-id` 传 `<作业号>-r<重启次数>`）；收尾只处理本轮创建（`run`）或属主为本用户的真实目录（`finalize-only`）。拷回核对比较每个文件的 sha256；启动树顶层 `STATE` 只许追加。
>   5. **§9.5 `submissions.tsv`**：sbatch 结果不明（ssh 超时或失败、退出 0 无作业号）写 `event=NOTE`、`reason` 以 `UNKNOWN_SUBMIT` 开头、`submitted_at` 为 sbatch 时刻，不重投；`record-end` 拒绝此状态，由新子命令 `reconcile-submit`（`--job-id J` 或 `--not-submitted`，凭 scontrol / batch.tsv / squeue 证据）落 `SUBMITTED` 或 `ENDED/SUBMIT_FAILED`。作业结束判据：`squeue` 报 `Invalid job id specified` 视为已不在 slurmctld、只看 sacct。
>   6. **启动计划 `capkit.launch_plan/1`** 新增必填 `kill_grace_s`（≥1）；`--overlay-manifest`、`--nvidia-smi`、`--job-env-dir` 与身份参数一样由 `capjobs.launches` 追加、计划里出现即拒绝。启动的第一行（capjobs.launches 写的）`STATE` 以编排器自己写的最后一行为准（超时被杀与退出码矛盾时为 VOID）。

## 测试

命令：`_local/fx-jobs/runtests.sh <dir>` = `cd <C>; PATH=<C>/_local/fx-jobs/fakebin:$PATH PYTHONPYCACHEPREFIX=<C>/_local/fx-jobs/pyc bash -c 'ulimit -v 8388608; exec systemd-run --user --scope -q -p MemoryMax=6G timeout 600 <venv>/python -B -m pytest -q -p no:cacheprovider <dir>'`

| 目录 | 结果 | 退出码 |
|---|---|---|
| tests/jobs（修改前基线） | 86 passed | 0 |
| tests/jobs（最终） | 115 passed | 0 |
| tests/capcommon（policy.py 回归） | 187 passed | 0 |
| 修前对照：`_local/fx-jobs/prefix/`（HEAD 版本的我的 9 个源文件 + 新测试，其余为当前工作树） | 47 failed / 65 passed（行为性红：K6 gpu_count==2、K8 同大小、K9 realpath、K17 UNKNOWN≠ENDED、K26 粘行、STATE 拷回；其余多为签名/字段变化） | 1 |
| 破坏性自测 `_local/fx-jobs/sab_run.py`（每例一份当前树副本，只回退一处） | 16 例全红：K1×2、K2、K6、K14、K15、K16×2、K22×2、K23、RS7、K8、STATE、RS4、RS9；对照组 12 passed | 0（脚本本身） |
| `bash -n jobs/capkit_inner.sh`、`bash -n jobs/capkit_job.sbatch` | 通过 | 0 |

## MEASURED / INFERRED / NOT RUN

- MEASURED：上表全部命令与结果；STATE 拷回 FAIL 的复现；gpu_idle.json 键集（真实 envfiles + 真实 capenv.gpu + 假 nvidia-smi）；内层脚本 `run` 模式录下的 hwprofile/gate argv 经真实 `capenv.gate.build_parser()` 解析、真实 `capenv.hwprofile.main()` 执行退出 0（本机硬件，假 nvidia-smi）；launches 作为独立进程成为 child subreaper（`child_subreaper: true`）并收掉孤儿角色。
- INFERRED：Slurm 真实环境下 `squeue -j` 对已清出作业的 stderr 是 `Invalid job id specified`（依 Slurm 文档/经验，未在本集群核对）；`scontrol show job` 的 `Command=` 是批处理脚本路径（未核对）；`date -r FILE +%s%N` 与 VM 上 `st_mtime_ns` 在同一 NFS 上一致（同一服务器时间戳，未在集群核对）；enroot `--container-remap-root` 下 `[[ -O dir ]]` 对本作业建的目录为真（未在容器里核对）。
- NOT RUN：任何集群命令（fakebin.log 为空）；容器内真实 overlay / 真实门禁全流程；`reconcile-submit` 对真实 squeue `--me` 输出格式。
- 20:39:19 基线：跑 tests/jobs（修改前），预计 <2 分钟
- 基线（修改前）：`tests/jobs` 86 passed，EXIT=0（runtests.sh，venv py3.11）。
- 主会话裁定（收到）：capenv 接口按我提议的名字（gpu 函数 `nvsmi` 关键字；gate/hwprofile `--nvidia-smi` 必填；gate `--max-alloc-gpus` 必填；hwprofile 收 identity + `--role hwprofile`；`--container-runtime pyxis-enroot`）；register.json 顶层必填段 `environment`。
- 核实中额外发现（RS1/RS2）：caplaunch 在启动树根写 `STATE`（orchestrator.py:1058），而 copyback.compare 只从目的端剔除顶层 STATE ⇒ 真实启动的拷回核对必然 FAIL（diff=["STATE"]），每个启动都会被记 VOID；单测的假编排器不写 STATE 所以没暴露。一起修。
- 修改顺序计划：policy.py → submit.py → jobstate.py → launchplan.py → launches.py → copyback.py → envfiles.py → capkit_inner.sh / capkit_job.sbatch → README → tests。
- 20:45:26 已改 policy.py(K9)、submit.py(K1/K6/K16/K23/K26/RS7)、jobstate.py(K17)、launchplan.py(kill_grace_s、--overlay-manifest 保留)、launches.py(K2/K14/K15/K25)；下一步 copyback/envfiles/脚本
- 20:48:16 已改 copyback.py(K8 内容 sha256、STATE 前缀规则、根目录可省略)、envfiles.py(K6/K9/K18、image 改核 size+mtime)、capkit_inner.sh(K1/K22)、capkit_job.sbatch(K23)；bash -n 通过；下一步 tests 与 README
- 20:50:19 tests: 已改 test_units(K17/K9/plan)、test_submit(K1/K6/K16/K23/K26/K9)、conftest(K12)；下一步 test_container、test_scripts
- 20:53:21 test_container 全过（含 K14 子进程版、K15、STATE 拷回）；下一步 test_scripts（内层脚本 run 模式接线测试，用假 python）
- 20:54:21 test_scripts 15 passed；开始全量 tests/jobs 与修前对照（_local/fx-jobs/prefix/）
- 20:55:22 破坏性自测（_local/fx-jobs/sab_run.py，每例一份副本）开始
- 20:56:18 破坏性自测 14 组全红、对照组 12 passed；下一步 README（K11/K12/K25 等）
- 21:00:45 已加 --job-env-dir、SIGTERM 处理(RS4)、--nvidia-smi 必须绝对路径(RS9)、去掉 job.env 里无读者的 KILL_GRACE、gpu_idle 去重复键；tests/jobs 全过；下一步跑其他相关目录回归与写进度
- 21:02:40 收尾：pgrep 无本组残留（看到的 pytest tests/launch 与 caplaunch/capload/sidecar 子进程属 FX-launch，未动）；fakebin.log 0 字节；未 commit。完成。
