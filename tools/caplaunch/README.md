# caplaunch(launch 编排器)

运行:`PYTHONPATH=<C>/tools`。包在 `tools/caplaunch/`(INTERFACES §1.1;第一批写在 `tools/launch/caplaunch/`,集成时移过来，import 名不变)。
E0 配置生成器在 `configs/gen_e0_configs.py`。

## 模块

| 模块 | 作用 |
|---|---|
| `caplaunch.__main__` | `python -m caplaunch`:一次 launch(一个 launch 树 `<T>`)的全部编排 |
| `caplaunch.scenario` | 场景配置 `capkit.scenario/1` 的校验、工具参数表(INTERFACES §11)、argv 与占位符、与档位表/绑核表的交叉检查 |
| `caplaunch.orchestrator` | 主循环：起进程(taskset 绑核、独立进程组、launch 标记 env)、就绪、origin、档位边界事件、清空门、窗口扫描、停止与残留核验、收尾文件 |
| `caplaunch.procfs` | /proc 读取：starttime、子孙进程、标记扫描(`/proc/*/environ`)、线程清单与识别规则、网卡计数 |
| `caplaunch.ready` | 就绪判据:`capkit_ready` / `plain_port_file` / `report_addr` / `http_get` / `file_exists`(可组合，全部满足才算就绪) |
| `caplaunch.tiergen` | `python -m caplaunch.tiergen`:把档位表模板里的 `TBD:<名字>` 填成登记的数，生成并校验 `capkit.tiers/1`(可选同时写到达时刻表) |
| `caplaunch.httpprobe` | E0(c) HTTP 冒烟客户端(作为场景里的 aux 角色运行):逐帧记录文本增量是否为空;可选 `--started-file`(发第一个请求前写，回答要等到原点之后的场景用它作就绪文件;INTERFACES 勘误 B2B-15) |
| `caplaunch.smoke` | `python -m caplaunch.smoke`:E0(c) 可行性冒烟各项，输出 `capkit.smoke/2`(PASS / FAIL / RECORDED / NOT_RUN);也可作为场景里的 aux 角色(TOOL_TABLE `caplaunch.smoke`,参数同 httpprobe;`--items` 选项组、`--hold` 写完后等 SIGTERM、`--result-dir` 是 `--out` 的别名;INTERFACES 勘误 B2B-12) |
| `configs/gen_e0_configs.py`、`configs/gen_e0s_configs.py`、`configs/gen_e2a_e3_e4_configs.py`(不在本包内) | `configs/e0/*.json`、`configs/e0s/*.json`、`configs/{e2a,e3,e4}/*.json`(快轨场景模板，INTERFACES 勘误 FT1-15)的生成器(交付物是生成出的 JSON;改配置请改生成器，生成到新目录后 diff) |

## `python -m caplaunch` 参数

身份参数(`capcommon.cli.add_identity_args`,`--role` 填 `launch`)外，以下全部必填，出处类的值可写 `none`(落 null):
`--run-dir --workdir --register --scenario --pin-table --shm-dir --snapshot-interval-ms --origin-lead-ms
--housekeeping-ms --kill-after-s --capkit-sha --probe-tree-sha --baseline-sha --snapshot-dir --image-path
--image-sha256 --hw-profile-sha --overlay-manifest --nvidia-smi --job-env-dir`;`--param KEY=VALUE` 可重复(场景 `params_required` 全部要给);`--job-input KIND=PATH` 可重复(场景 `job_inputs` 声明的每个种类恰好一个，见下)。长选项只认完整拼写(`allow_abbrev=False`：`capjobs` 按完整拼写核对计划 argv)。

- `--job-input KIND=PATH`(INTERFACES 勘误 B2B-9):场景可选段 `job_inputs` 声明的作业输入(`fingerprint` / `request_bytes` / `prebaked`),由 `capjobs.launches` 在作业里生成(作业外用 `python -m capjobs.jobinputs`)。`PATH` 绝对路径;文件 schema 是该种类的;`PATH.record.json` 的种类、模块、`rc`、`out_sha256`、所用登记输入的 sha 都要对得上，否则退出码 3、不起进程。文件与记录拷进 `raw/env/job_inputs/`,角色参数里的 `{input:KIND}` 解析为这份拷贝;`launch.json` 的 `job_inputs` 记 relpath、sha256、record_sha256、模块与来源路径。
- `--job-env-dir`:作业级 `env/` 目录(门禁结果、`gpu_idle.json` 等),其中的文件拷进 `raw/env/`;不是目录 ⇒ 起任何进程之前停止,NOT_RUN(`stop.json` reason `job_env_dir_missing`)。

- `--overlay-manifest`:作业级覆盖清单，原样拷进 `raw/env/overlay_manifest.json`,sha 记进 `launch.json` 的 `overlay_manifest`。文件不存在时只记录;但场景里有带 `probe_tier` 的角色时，启动在起任何进程之前停止,STATE 为 NOT_RUN(`stop.json` reason `overlay_manifest_missing`)。
- `--nvidia-smi`:nvidia-smi 可执行文件的绝对路径，展开场景参数里的 `{nvidia_smi}` 占位(sidecar 的 GPU 查询),路径与 sha 记进 `launch.json`。
- `--shm-dir` 与本次启动的快照子目录：经 `capcommon.snapshot.ensure_own_dir` 创建或接受(不存在就以 0700 新建;已存在则必须是本用户的真实目录、不是软链),另外要求组和其他人不可写，否则退出码 3。

输入核对(任何一项不符 → 退出码 3,不起任何进程):register 里的文件 sha 与当前文件一致、到达时刻表头里的档位表 sha 一致、
场景/档位表/绑核表交叉检查(每角色核数、D1 角色间核不相交、门的 `poll_ns`/`max_snapshot_age_ns` ≥ 2×快照间隔、门读的槽位已在角色上声明)。
清空门只支持 `abs_max`;`baseline_band` 报 CONFIG(未实现)。

## 结果与退出码

| 情况 | STATE 行(by=caplaunch) | 退出码 |
|---|---|---|
| 全部档位跑完、退出核验干净 | VALID | 0 |
| enforce 门超时：按 stop.json 提前停 | VALID(reason 写明 drain_gate_timeout) | 5 |
| origin 前未就绪 / 起不来 / 收到信号 | NOT_RUN | 5(起进程失败为 4) |
| 进程在允许之前退出 | VOID | 5 |
| 停止后仍有残留进程(标记扫描) | VOID | 4 |
| 有 `probe_tier` 角色而覆盖清单缺失 | NOT_RUN(reason 写明 overlay manifest) | 5 |
| 参数或输入不合法(含场景角色 `env` 里有 `CAPPROBE_*` 键、`--param lag_period_us` 不是 ASCII 正整数)、外层环境的 `CAPKIT_CODE_SHA` 与 `--capkit-sha` 不符、`CAPPROBE_PATCH_SHA` 与覆盖清单不符、`CAPPROBE_TIER` / `CAPPROBE_GROUPS` / `CAPPROBE_LAG_PERIOD_US` 与场景不符或有非空的 `CAPPROBE_INJECT`(后四项只在场景有产品角色时)、启动树里已有 `raw/launch.json` 或 `STATE` | 不写(不往启动树写任何文件、不起进程) | 3 |

收到第一个 SIGTERM/SIGINT:停止启动(NOT_RUN),进入收尾。收尾(停进程、残留核验、写收尾文件)期间再收到信号只记录，并跳过剩余的宽限等待、直接 SIGKILL;`launch_end.json` 的 `signals` 列出每个信号。

## 进程集合与监测类角色

`measure_class` 为 `load_source`、`observer` 的角色(`MONITOR_CLASSES`)的未登记子孙(例如 sidecar 每次 GPU 查询起的 nvidia-smi)不进入窗口始末 `descendants` 的比较(`pids_stable`),也不进 `roles.json` 的 `unattributed`;它们逐个记在 `raw/procs/monitor_children.json`(pid、ppid、comm、exe、是否为 `--nvidia-smi`、首次/末次看到的时刻、看到次数),每档的扫描另记在 `tier.json` 的 `monitor_descendants`。清单只含某次 /proc 扫描(housekeeping 或窗口边界)碰到的进程。其他类别(`under_test`、`stub_*`)的子孙照旧比较、照旧进 `unattributed`。任何非产品角色下面写快照的未登记进程都进 `unattributed`(`seen_via: probe.snapshot`,带 `registered_ancestor`)。

capkit 工具角色(任何 `measure_class`)的运行时辅助进程按命令行白名单 `AUX_HELPER_RULES`(`multiprocessing` 的 resource tracker、forkserver)归属到该角色(命令行 `python [解释器选项] -c <代码>`，`<代码>` 整串匹配规则；解释器选项只认 `multiprocessing` 从父进程标志带过来的那些与 `-u`，如作业里 `PYTHONDONTWRITEBYTECODE=1` 带来的 `-B`，见 `aux_helper_rule`)：记在 `roles.json` 的 `aux_helpers`(pid、ppid、starttime、cmdline、role、measure_class、`category: aux_helper`、命中的规则),每档扫描记在 `tier.json` 的 `aux_helpers`;不进 `unattributed`,也不进比较用的 `descendants`。白名单在 `launch.json` 的 `orchestrator.aux_helper_rules`。产品角色的子进程不套用白名单。

产品侧注入(INTERFACES 勘误 FT1-18):档位表 `launch_inject` 里指名产品角色的规格就是该角色的 `CAPPROBE_INJECT`(只此一个来源;外层值与角色 `env` 里的 `CAPPROBE_*` 都拒绝),`cross_check` 在起进程之前核对档位、位点、模式、进程类与 `snapshots`;这样的 P1 / P2 角色写快照，编排器把它列进每档的应拷快照，登记的规格记在 `launch.json` 的 `probe_inject`。

清空门轮询落后(多个计划时刻同时到期)时，每次轮询至少与上一次相隔 1.5 个快照周期(不越过门的期限),每个计划时刻仍各有一条记录。

与 INTERFACES §5.7 的差异:§5.7 规定 STATE 第一行由 B6 写;本实现由编排器追加一行，供 B6 合并时裁决。

## 已知限制(详见 `_progress/B7-launch.md` 与最终报告)

- 线程识别：工具要在 ready 文件 `info.threads` 里发布 `{native_id: name}`,否则 Python 线程无法识别(C13 会红)。
- P0 产品进程的子进程只能出现在 `unattributed`(INTERFACES §12 U-5)。
- 各角色 `expected_procs` 取值(nullsse/capload 的父+worker 数、前端的真实进程数)未经真实工具核实。
- `launch.json` 里角色参数的 `{addr:R}` 保持未展开(运行时才知道)。
- 本机 RHEL8 cgroup v1 上 `systemd-run --user --scope -p MemoryMax=` 不生效(B4 发现);`caplaunch.smoke` 的 `read.memory_guard` 只读不判。
