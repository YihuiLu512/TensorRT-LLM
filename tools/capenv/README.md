# capenv（B4）：环境门禁、硬件画像、绑核表、D13/D14、时钟自检

## 包的布局

`tools/capenv/` 是一个普通包（INTERFACES.md §1.1）。第一批由 B4 分三处写成命名空间包（写权限所限），集成时合并到本目录并加了 `__init__.py`；代码里的 import 都是 `capenv.<模块>` 或包内相对 import，合并时没有改代码。

- 仓库里跑：`PYTHONPATH=<C>/tools python -m capenv.<模块>`。
- 容器里跑：与其他包一样整目录拷进 `dist-packages/capenv/`（INTERFACES §1.2）。

依赖：只用标准库与 `capcommon`（不用 numpy、psutil、grpc），python3.11 与容器里的 3.12 都能跑。

## 入口一览

身份参数（`--experiment --arm --launch-seq --job-id --restart-count --register-sha --role`，见 INTERFACES §2.1）每个入口都必需，写进输出文件的 `identity`。退出码沿用 INTERFACES §2.4：0 通过、2 用法、3 配置、6 `ENV_INVALID`（检查结果是 RED 或 NOT_RUN）。

### `python -m capenv.gate`（INTERFACES §11 的参数全部支持，另加下列必填参数）

```
python -m capenv.gate <身份> --run-dir <作业目录> --phase {pre,post}
    --registered-hw-sha {SHA|UNREGISTERED} (--comparable-file PATH | --no-comparable)
    --slurm-view-file PATH --slurm-view-wait-s S
    --max-alloc-gpus N --nvidia-smi PATH
    --idle-window-s S --idle-busy-max-frac F        # --phase pre 必填，post 不用
    [--hw-profile PATH] [--roles-file PATH] [--timeout-s S]
```

- 必填参数缺了：argparse 退出码 2；`--phase pre` 缺第 6 项的两个参数、`--max-alloc-gpus < 0`、`--idle-window-s <= 0`、`--idle-busy-max-frac` 不在 [0, 1]、`--registered-hw-sha` 既不是 64 位小写十六进制也不是 `UNREGISTERED`：退出码 3。这些值由调用方从登记处显式传入，门禁不设缺省。
- `--registered-hw-sha UNREGISTERED`：该型号还没有登记画像时的第一个作业。门禁照常跑第 1～6 项、读画像，`hw_gate.json` 记 `status: NOT_RUN`、`node_hw_sha`（供登记），判定恒为 `ENV_INVALID`、退出码恒为 6（给了 `--comparable-file` 也一样），作业在门禁后停止。

- 标准输出打印**一行** JSON 判定：`{"verdict":"PASS"|"ENV_INVALID","reasons":[…],"items":{"1":"PASS",…},"files":{…sha256…},"phase":…}`；同一份写进 `env/gate.<phase>.json`（`capkit.env_gate/1`）。
- `pre`：RESOURCES §3 第 1～6 项 → `env/exclusivity.json`；`hw_gate.json`；`memguard.json`（本进程 + `--roles-file` 里的 pid）。`post`：第 1～5 项 → `env/exclusivity.post.json`。
- **任何一项不是 PASS（RED 或 NOT_RUN）门禁就是 `ENV_INVALID`**：查不了的项不算通过。
- 第 1～3 项先在容器里跑 `scontrol` / `squeue`；命令不存在或返回非 0 时等 `--slurm-view-file` 出现（最多 `--slurm-view-wait-s` 秒），用它的原文判同样的条件，`source` 写 `slurm_view`。
- 各项判据：1 `JobId` 等于本作业、`OverSubscribe=NO`、申请的 GPU 数（`ReqTRES`；输出里没有它时按分配数 `TRES`/`AllocTRES`）≤ `--max-alloc-gpus`，分配数照原样记在 `alloc_gpus`（整机独占时是整台的卡数；可用几张由第 5 项核），申请数只取自 `ReqTRES` 记在 `req_gpus`（没有 `ReqTRES` 时为 null），实际比较的数记在 `gpus_checked`、取自哪个字段记在 `gpus_checked_from`（三者都没有记红：读不到不等于 0 张）；2 `NodeName` 对、`CPUAlloc == CPUTot`；3 退出码 0、列出本作业（阳性对照，列不出 ⇒ NOT_RUN）、没有别的作业；4 `/proc/self/status` 的 `Cpus_allowed_list` 覆盖全部在线核（掩码里多出的离线 possible 核不算问题，本 VM 上就是 `0-95` 对在线 `0-23`）；5 用 `--nvidia-smi` 指定的可执行文件、经 `capenv.gpu.check_visible` 判「可见数 == `--max-alloc-gpus`」且无计算进程：请求 ≥1 张时先要列出 ≥1 张（阳性对照，否则 NOT_RUN），张数不等记红；请求 0 张时要求 nvidia-smi 报告没有卡（查询成功且为空，或输出 `No devices were found`），看见卡记红、查不了记 NOT_RUN，阳性对照与计算进程检查需要一张卡，记 NOT_RUN 并写明原因；6 空转 `--idle-window-s` 秒、每个在线核的忙碌比例（user+nice+system+irq+softirq+steal ÷ 全部 tick）≤ `--idle-busy-max-frac`；每核另记 `irq_ticks`、`softirq_ticks`（D14 的空载读数）。**瞬时噪声重采样**：第一个窗口红时，先停一个窗口长度，再读一个同长窗口，最多再读 `IDLE_RESAMPLE_MAX` = 3 个（最坏多花 3 × 2 × 窗口；第一个窗口通过时不重采、不多花时间）；任一窗口里全部在线核都 ≤ 阈值即判通过，核集合与阈值不变，全部窗口都红仍记红。`detail.windows` 逐窗口记 `busy_cores`、`missing`、红核的忙碌比例与全部核的最大值、中位数、前 8 名；`detail.passed_on_window` 是通过的窗口序号（1 起，全红为 null），重采后通过时 `detail.note` 写「passed on window k of n」；顶层 `window_ns busy_cores missing per_core` 取通过的那个窗口，全红时取第一个窗口。`capjobs.submit.time_budget` 为此另记一项 `gate_idle_resample_s`。**不给第 6 项的两个参数时第 6 项 NOT_RUN ⇒ 门禁 ENV_INVALID**（窗口与阈值「待 E0 试点」，由登记表给出）。
- `hw_gate.json`：本节点画像 sha 等于 `--registered-hw-sha` 即过；不等时只有 `--comparable-file` 里登记了全部差异字段才过。可比登记文件格式（本组件定义，INTERFACES 未定义）：`{"schema":"capkit.hw_comparable/1","registered_profile":{…主型号画像，恰好 §5.5.3 的键…},"comparable_fields":["kernel",…],"evidence":{…}}`；门禁先核对 `profile_sha256(registered_profile) == --registered-hw-sha`。

### `python -m capenv.hwprofile`

```
python -m capenv.hwprofile <身份> --out env/hw_profile.json --raw-out env/hw_raw.json
    --local-dir <节点本地运行目录> --chassis-source {slurm_view,dmi} --nvidia-smi PATH
    [--slurm-view-file PATH] [--image-sha256 HEX64] [--code-baseline SHA40]
    [--container-runtime TEXT] [--timeout-s S]
```

- `hw_profile.json`（`capkit.hw_profile/1`）的载荷：`profile`（**恰好** `capcommon.hwprofile` 的 IDENTITY_FIELDS + OBSERVED_FIELDS）、`profile_sha256`、`sha_fields`、`raw_file_sha256`、`unreadable_identity_fields`。读：`capenv.hwprofile.load_profile(path)`（会重算 sha 核对）。
- **进 sha 的字段**（identity）：`chassis_model cpu_arch cpu_model sockets cores_per_socket threads_per_core microcode numa_layout cpu_governor mem_total_gib mem_per_numa_gib gpu_model gpu_count gpu_mem_mib gpu_driver nics local_disk_type kernel container_runtime image_sha256 code_baseline`。**不进 sha**（observed）：`node_name partition collected_at collected_mono_ns cpu_freq_khz mem_available_kib local_disk_free_bytes netdevs nic_link sources`。读不到写 `{"unreadable": 原因}`，identity 键的这个标记也进 sha（原因文字也在内）。
- 取法：lscpu（`LC_ALL=C`）、`/proc/cpuinfo` 的 microcode、`/sys/devices/system/node`、cpufreq、`/proc/meminfo`、`--nvidia-smi` 指定的 `nvidia-smi --query-gpu`（`memory.total` 读不到时 `gpu_mem_mib` 记 unreadable）、`nics` 取 `/sys/bus/pci/devices` 里 class 为 `0x02xxxx` 的设备（SR-IOV 虚拟功能除外）按 PCI `vendor:device` 计数；`/sys/class/net/*`（只有带 `device` 链接的才算）的型号与速率记进 observed 的 `netdevs`（速率读不到写 unreadable）、mountinfo + `/sys/dev/block` 定本地盘类型。`chassis_model` 的来源由 `--chassis-source` 显式指定：`slurm_view`（`scontrol show node` 原文里的 `SystemProductName`，与 capcommon 文档一致）或 `dmi`（`/sys/class/dmi/id/product_name`）。同一研究里必须始终用同一个来源，否则 sha 会因来源不同而不同。
- `hw_raw.json`（`capkit.hw_raw/1`）放没有画像键的原始证据：`lscpu`、`lscpu -e`、`lscpu -p`、`numactl -H`、`free -b`、`lsblk`、`df` 原文与退出码，`iommu_groups_count`、DMI 产品名、超线程兄弟表（topology.summary）、uname。

### `python -m capenv.pins`

```
python -m capenv.pins generate <身份> --demand FILE --out env/pin_table.json --source-sha <hw sha>
    [--lscpu-file PATH] [--avoid-cpus LIST] [--only-affinity] [--primary-numa N]
python -m capenv.pins verify <身份> --pin-table PATH --roles-file PATH --out env/pin_readback.json
```

- 拓扑来自 `lscpu -p=CPU,CORE,SOCKET,NODE`（或 `--lscpu-file` 给的同格式文本）。物理核 = (socket, core)；每个物理核只用编号最小的逻辑核，兄弟核留空（`siblings_left_idle`）；`--avoid-cpus` 里任一核所在的整个物理核都不分。
- `--demand`：`[{"role":…,"cpus_needed":N,"numa":节点号或null}]`。显式给了节点的角色放在该节点；`numa:null` 的角色全部放进同一个「主节点」：`--primary-numa`，否则（有显式节点时）出现最多的显式节点，否则编号最小、能装下全部需求的节点。按需求顺序、核号升序连续分配。装不下以退出码 3 退出，stderr 打印 `{"verdict":"DOES_NOT_FIT","gap":[…]}`，**不跨节点、不用兄弟核凑**。
- 输出：INTERFACES §5.5.1 的键（`source_sha rules siblings_left_idle roles unassigned`）外加 `primary_numa avoid_cpus demand avoided_cores topology topology_source table_sha256`（`table_sha256` 只算 `roles`）。
- `verify`：对 `roles.json` 里每个角色在表里的进程，逐线程读 `Cpus_allowed_list` 与最近运行核；越出本角色核集、与他角色核集相交、最近运行核不在本角色核集，都算红；登记了但表里没有的角色、表里有但没登记进程的角色也算红；应检进程数为 0 ⇒ NOT_RUN。退出码 0 / 6。

### `python -m capenv.memguard`

`<身份> --out PATH (--pids LIST | --roles-file PATH)`。对每个 pid：`/proc/<pid>/cgroup` → memory 控制器路径（v1 `memory.limit_in_bytes`，v2 `memory.max`），从叶子一路读到挂载根，取最小的有限值；`max` 与 v1 的「不限」哨兵值（≥ 2^62）不算护栏。没有任何有限值的 pid 记红；pid 列表为空 ⇒ NOT_RUN。

> 本 VM 实测（2026-11-13）：RHEL8 cgroup v1、systemd 239 下，`systemd-run --user --scope -p MemoryMax=6G` 里的进程 memory 控制器路径仍是登录会话的 scope，`memory.limit_in_bytes` = 9223372036854771712（不限），memguard 判红。即这台 VM 上那条护栏对内存并不生效；计算节点上以作业 cgroup 的读数为准。

### `python -m capenv.gpu check`

`<身份> --out PATH --samples N --every-s S --timeout-s S --expect-gpus N --nvidia-smi PATH` → `capkit.gpu_idle_check/1`（作业流程不调用它；作业的 `env/gpu_idle.json` 由 `capjobs.envfiles gpu-idle` 写，schema `capkit.gpu_idle/1`，形状不同）：`visible`（`check_visible` 的结果）、`positive_control_ok`、`gpu_count`、逐次采样（每卡 `util_pct mem_used_mib compute_pids`）、`verdict`：可见数与请求数不等、任一读数利用率 > 0 或有计算进程 ⇒ RED；阳性对照不过、任一次采样失败、任一利用率读不到（`[N/A]` → null）⇒ NOT_RUN；请求 0 张且看不见卡 ⇒ PASS、不采样。pid 读不到的计算进程照样算计算进程（`compute_pids` 里记 null）。

**`capenv.gpu` 是 capkit 里 nvidia-smi 查询的唯一实现**：门禁、画像、sidecar 与 `capjobs.envfiles` 都调它。函数签名（`nvsmi` 缺省 `"nvidia-smi"` 只给单元测试用，生产调用方一律显式传；传 `None`/空串 ⇒ 不执行，记录 `reason: "nvidia-smi path not given"`）：

- `query_gpus(timeout_s, runner=runcmd.run, nvsmi=…) -> (record, [gpu])`、`query_apps(timeout_s, runner, nvsmi=…) -> (record, [app])`、`positive_control(timeout_s, runner, nvsmi=…) -> (ok, count, record)`、`sample(timeout_s, runner, nvsmi=…) -> (rc, rows)`；
- `check_visible(expect_gpus, timeout_s, runner=runcmd.run, nvsmi=…) -> dict`：`status`（PASS/RED/NOT_RUN）、`reasons`、`expected_gpus`、`visible_gpus`（读不到为 null）、`gpus`、`compute_apps`（没查为 null）、`positive_control_ok`、`count_check`/`positive_control`/`apps_check`（各 `{status, why}`）、`queries`（每次查询的 `cmd rc stdout stdout_sha256 stderr reason`）；`expect_gpus` 不是 ≥0 的 int 抛 `ConfigError`；
- `idle_check(samples, every_s, timeout_s, expect_gpus, runner, sleep, nvsmi=…) -> dict`（上面 CLI 的载荷）。

### `python -m capenv.loopback`（D14）

`<身份> --sender-cpu A --receiver-cpu B --duration-s S --msg-bytes N --out env/loopback_softirq.json`。协调进程（应在第三个核上）起一对绑核的回环 TCP 收发进程，发送端在窗口内全力 `sendall`；窗口前后读两条线程的 `utime/stime` 与两个核的 `/proc/stat` 全部列。输出每个核：各列 tick 增量、忙碌 tick、绑在该核上的线程的 `utime+stime`、两者之差（不属于这对线程的忙碌时间，主要在 softirq/irq 列）。只给数，不下结论；`capjudge.pilot` 的 D14 结论不读它（用户 2026-11-13 裁决改为比较门禁第 6 项的空载窗口与负载档，见 `capjudge` README「D14」）。

### `python -m capenv.clockcheck`

INTERFACES §11 的参数。server 写端口文件，client 做 `--round-trips` 次往返：单程差（对端戳 − 本端发送时刻）应落在 `[0, rtt]`；外加 `--back-to-back` 次连读 `monotonic_ns` 的差值分布、两边的 `/proc/self/ns/{pid,mnt}` 与 cgroup。client 写 `clock_check.json`，`consistent` = 往返数 > 0 且没有落在 `[0, rtt]` 之外的单程差。

### `python -m capenv.switchinterval`（D13）

`<身份> --versions-dir raw/env/versions --expected-count N --out env/switchinterval.json`。逐个版本自报文件取 `switchinterval_s`；`N` 由调用方给（`launch.json` 的 `expected_procs` 之和）；`N=0` ⇒ NOT_RUN，少于 N 个或缺字段 ⇒ RED。值只记录不判定。

### `python -m capenv.spinners`（SB-B1）

caplaunch 场景的 `aux` 角色（`measure_class: observer`）：`--cpus` 每核一个纯 Python 自旋进程，`--target-role ROLE` 另起一个绑到绑核表里 ROLE 的第一个核（C13 阳性对照）。`--pin-table` 由编排器给（启动树的 `raw/env/pin_table.json`）。参数、进程模型、角色写法、退出码与输出 `raw/spinners/<角色>.json` 见 INTERFACES 勘误「Calibration v2 SB-B1」。

## 已知限制

- 第 6 项只按核的忙碌比例判，不扣除「本作业自己的进程」的 CPU（按核归属进程时间在 /proc 里拿不到）；pre 阶段本作业只有门禁自己与作业脚本在睡眠，阈值留出这部分。
- `pins verify` 把「最近运行核不在本角色核集」也判红：线程若在改亲和性之前睡下、之后一直没运行，`processor` 字段是旧值，会误红。E0 冒烟时看是否出现；出现就在判定口径里单列。

## capenv.sidecar（B4）：sidecar v4


### 命令行（INTERFACES §11 的参数全部必需）

```
python -m capenv.sidecar <身份> --run-dir <T> --cpus <sidecar 的核> --stop-grace-s S
    --shm-dir <本次启动的快照目录> --origin-ns N --roles-file <T>/raw/procs/roles.json
    --interval-ms 1000 --out-dir <T>/raw/sidecar --cpus-watch {all,LIST}
    --gpu-every-s S --numa-every-s S --logs-dir <T>/raw/logs
    [--ready-file PATH] [--pin-table PATH] [--gpu-timeout-s S] [--nvidia-smi PATH]
```

- `--nvidia-smi`：GPU 查询用的可执行文件（INTERFACES §11 之外的可选参数）。`--gpu-every-s > 0` 而不给它时不执行任何 nvidia-smi：`gpu.jsonl` 头 `positive_control_ok: false`、每行 `query_rc ≠ 0`，GPU-IDLE 因而 NOT_RUN（不会退回 PATH 里的同名命令）。利用率 `[N/A]` 的行记 null，计数写进 `meta.stats.gpu_util_unreadable_rows` 与 `meta.unreadable.gpu_util`。

- `--gpu-every-s 0` / `--numa-every-s 0` 关掉对应文件（关掉 GPU 后 GPU-IDLE 只能 NOT_RUN）。
- `--pin-table`：给了就核对 `--cpus-watch` ⊇ 表里全部角色的核，不覆盖以退出码 3 退出（INTERFACES §5.6 写的是编排器核对，这里多一道）。
- `--ready-file`：第一拍采完后写就绪文件（INTERFACES §11 的 sidecar 行没有列它，但 §2.2 说长驻进程都要就绪文件，编排器按原点前是否就绪判 NOT_RUN，所以这里做成可选）。
- 只在收到 SIGTERM / SIGINT 时退出，退出码 0。

### 做什么

- **进程只按 `roles.json` 认**（pid + `starttime_ticks`）；`generation` / mtime 变了就重读。starttime 对不上（pid 复用）或进程没了，记进 `meta.json` 的 `pids_gone`，不再采；同一 pid 以新的 starttime 重新登记会重新采。没有任何按命令行或进程名的匹配。
- 每拍（绝对时刻 `origin + k × interval`，落后就跳过并计 `missed_ticks`，不补拍）：
  - `threads.jsonl`：每个登记进程的每条线程 `utime/stime`（分列）、`schedstat` 的运行队列等待 ns、自愿 / 非自愿切换数、最近运行核、`Cpus_allowed_list` 原文（每行都写）。
  - `procs.jsonl`：`role`、`worker` 取自登记表；RSS、线程数、进程级 `Cpus_allowed_list`。
  - `cpus.jsonl`：`--cpus-watch` 的每个核的 `/proc/stat` 全部列（tick 原值）与当前频率（先 `scaling_cur_freq`，读不到退回 `/proc/cpuinfo` 的 `cpu MHz`，来源写进 `meta.freq_source`；都没有则写 null）。
  - `snapshots.jsonl`：`--shm-dir` 里登记进程的快照（含金丝雀发布的 `canary.loop.*` 计数）解码后与 `/proc` 读数用**同一个** `t_rel_ns`；快照 meta 的 pid / starttime 与登记表不符的跳过并计数；未登记 pid 的快照文件列进 `meta.unregistered_snapshot_pids`。
  - `logs.jsonl`：`--logs-dir` 下每个文件的大小。
  - `tids.jsonl`（**INTERFACES §5.6 之外新增**，`capkit.sidecar.tids/1`，列 `t_rel_ns, pid, role, tid, event, comm, in_inventory, identified_as`）：线程出现 / 改名 / 消失事件；编排器的线程清单 `procs/threads/<角色>.<pid>.json` 存在时，`in_inventory` 与 `identified_as` 取自它（不在清单里 = 未认出的线程）；清单还没写出时两列为 null。
- 慢线程（与每拍线程分开，`nvidia-smi` 慢也不拖每拍）：`gpu.jsonl`（开头先做阳性对照，头里写 `gpu_count` 与 `positive_control_ok`；查询失败的那次每张已知卡一行、`query_rc ≠ 0`，已知卡数为 0 时写一行 `gpu_index=null`）与 `numa.jsonl`。
- `meta.json`（`capkit.sidecar.meta/1`，开始写一次、结束覆盖一次，`final` 标明）：`clk_tck`、周期、`freq_source`、`unreadable`（schedstat、切换数、频率、GPU 读不到的原因与行数）、`stats`（拍数、漏拍、最大迟到、每拍耗时、读失败计数）、`pids_gone`、`roles_generations`、各文件行数、sidecar 自身 CPU、结束时的 `control/stop.json`。

全部文件只在 `--out-dir`（节点本地启动树）里写，逐拍 flush 一次。

## capenv.canary（B4）：并发金丝雀（设计稿 §2.5 D5）


```
python -m capenv.canary <身份> --run-dir <T> --cpus <恰好 1 个核> --stop-grace-s S
    --origin-ns N --interval-ms 1000 --out <T>/raw/canary/canary.jsonl
    [--ready-file PATH] [--shm-dir <快照目录>] [--unit-ops N]
```

- 单线程、绑在一个核上，反复执行固定的纯 Python 工作单元（`UNIT_OPS = 2000` 步 31 位 LCG；`--unit-ops` 可改，实际值写进 `canary.meta.json`）。不建任何别的线程（快照在边界处由主线程内联写，不起 `SnapshotWriter` 线程）。
- 每到 `origin + k × interval` 之后的第一个单元边界写一行 `canary.jsonl`（`capkit.canary/1`）：`t_rel_ns`（实际时刻）、`iters`（上一行以来完成的单元数）、`thread_cpu_ns`（上一行以来本线程 CPU 时间）、`cpu`（`/proc/thread-self/stat` 的最近运行核）。整段漏掉时重新对齐，不补行。
- `--shm-dir`：发布快照，计数槽 `canary.loop.iters`（累计单元数）、`canary.loop.thread_cpu_ns`；金丝雀登记进 `roles.json` 后，sidecar 会把它与 `/proc` 读数写进同一拍。
- `--cpus` 不是恰好 1 个核以退出码 3 退出；SIGTERM / SIGINT 后写 `canary.meta.json`（与输出同目录）并以 0 退出。
