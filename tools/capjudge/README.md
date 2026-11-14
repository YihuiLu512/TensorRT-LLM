# capjudge（B5：判定器、统计量 T1～T6、试点构建）

> 包在 `tools/capjudge/`（INTERFACES §1.1）。第一批写在 `tools/judge/`，集成时改名；包内全部是相对 import，改名没有动代码。

只依赖标准库与 `capcommon`，在 VM 上用 `python3.11` 跑。

## 入口

| 命令 | 作用 |
|---|---|
| `PYTHONPATH=<C>/tools python3.11 -m capjudge --tree <T> --workdir <W>` | 判一棵启动树（INTERFACES §10.1，只接受这两个参数） |
| `PYTHONPATH=<C>/tools python3.11 -m capjudge.calib {near-boundary --regime R,loadgen-band,stub-calib} --inputs <列表> --workdir <W> --out <文件>`，或 `python3.11 <C>/analysis/calib_build.py …` | 构建 `near_boundary_noise`、`loadgen_band`、`stub_calib`（INTERFACES §9.6～§9.8、勘误 B2-2A-8；列表文件：near-boundary 同 pilot，另两种每行 `<树>\t<档,…>\t<角色>\t<形态名>`） |
| `PYTHONPATH=<C>/tools python3.11 -m capjudge.pilot --inputs <列表> --workdir <W> --d14-rule <规则> --out <pilot.json>`，或 `python3.11 <C>/analysis/pilot_build.py …`（同一实现） | 从已判过的亚饱和档构建 E0 的 `pilot.json`（INTERFACES §9.4） |
| `PYTHONPATH=<C>/tools python3.11 -m capjudge.boundary --inputs <列表> --workdir <W> --out <文件> [--compare A,B …]`，或 `python3.11 <C>/analysis/boundary.py …` | 每启动 T4、每臂包络、T5（INTERFACES §14.1）；旁边写 `.cmd.txt` |
| `PYTHONPATH=<C>/tools python3.11 -m capjudge.scaling --boundary <boundary.json> --pair 基准,加倍 … --grows-min R --flat-max R --out <文件>`，或 `python3.11 <C>/analysis/scaling.py …` | T6 与对侧翻倍归因（INTERFACES §14.2）；旁边写 `.cmd.txt` |
| `PYTHONPATH=<C>/tools python3.11 -m capjudge.intervene --inputs <列表> --workdir <W> --out <文件>` | 跨启动的剂量–反应（V）与最终结论等级：列表每行一棵已按 `P+X+I+V` 判过的树（INTERFACES 勘误 B2-2E-7、B2-2E-8）；不覆盖已有输出，出错 → 3 |
| `python3.11 <C>/samples/make_samples.py --out <目录>` | 生成自测样本（合成，`launch.json` 带 `"synthetic": true`） |
| `python3.11 <C>/samples/run_selftest.py --tools <C>/tools --package capjudge --tmp <已存在的临时目录> --expect-samples <样本数>` | 跑全部样本并逐条断言；`--expect-samples` 必填，由调用方给出（当前样本数见 `samples/README.md`） |

退出码按 INTERFACES §10.3：0 / 1 / 2 / 10 / 11 / 12 / 13（`UNDETERMINED` 与 `UNSTABLE` 一样不算失败）。空目录（没有 `raw/launch.json`）→ 11，不写任何文件。`pilot`、`intervene`、`boundary`、`scaling` 出错 → 3。

## 判定流程（一棵树）

1. 参数核对（`* ? [` → 2）→ `<T>/judge` 已存在 → 见下文「补写」，否则 12 → `JUDGED.tsv` 已有同一 `(job_id, restart_count, launch_seq, arm)` → 12。这两步除补写外不写任何文件。之后新建一个本次尝试的目录 `<T>/judge.attempts/<UTC 时刻>-p<pid>/`，以下输出都先写在这里。
2. 输入核对（`inputs.py`，任一不过 → 尝试目录里的 `inputs.json` 写明哪一项，13）：`launch.json`；`submissions.tsv` 里该作业的 `submitted_at`；`register.json` 与它列出的每个文件、每个标定文件现算 sha、已登记、登记时刻严格早于提交；`launch.json` 的 tier_table/schedule/prompts/scenario 与 `register.json` 一致；`inline_thresholds` 只许 E0；pilot 的 `environment` 与本树一致；时间表头的档位表 sha；`reconcile` 过 `expect.validate_reconcile`；`raw/` 下全部带 `identity` 的文件（json、jsonl 头、capbin 头、快照元数据）前六个字段一致；最后读判定之前已记下的启动级结论（上游）：`STATE` 里全部非 `capjudge` 写的行（caplaunch、capjobs.launches）、`raw/launch_end.json` 的 `state`（缺失或不是四个状态之一按 NOT_RUN）与 `leftovers.after`（非空按 VOID），取最差（VALID < NOT_RUN < VOID < ENV_INVALID）。`STATE` 读不了、或没有任何非 `capjudge` 的行（作业还没记下结果）⇒ 13。
3. 判定器代码 sha：`CAPKIT_CODE_SHA`（40 位）优先；否则 `git rev-parse HEAD` 且 `git status --porcelain -- tools` 为空。取不到或工作树脏 ⇒ 整次判定 `NOT_RUN`。
4. 启动级前提：LAUNCH-STATE（上游 VALID ⇒ PASS，NOT_RUN ⇒ NOT_RUN，VOID / ENV_INVALID ⇒ RED）、C10、GPU-IDLE、C13（第一半，登记了 C13 时）、PROBE-TIER（有产品角色时）。任一 RED ⇒ 全部被判档 VOID；任一 NOT_RUN ⇒ 全部 NOT_RUN。另：缺 `launch_end.json` 的 `tiers_completed`、档位表没有基线档 ⇒ NOT_RUN。
5. 逐档（只判 `up` / `down`；`warmup`、`drain` 档 `judged: false`、`verdict: null`）：前提 P = RECORDS、P-PIDS、P-SENT + 适用检查；P 有 RED ⇒ VOID；有 NOT_RUN ⇒ NOT_RUN；否则看 X：矛盾 ⇒ VOID；X 算不出 ⇒ NOT_RUN；X 成立 ⇒ UNSTABLE，否则 STABLE。模式 `P+X+I+V`（登记表 `judge.mode`）下另算 I 与逐档 V 心跳（`localize.py`、`intervene.py`，经 `symptoms.evaluate_x` 的 `hooks`），UNSTABLE 档的 I 信号指向不同角色 ⇒ UNDETERMINED；启动级再给结论等级（`grade.py`）。
6. 在尝试目录写 `inputs.json`、`verdicts.json`、`checks.tsv`、`judge.log`、`attempt.json`（`status = JUDGED`）；再查一次 `JUDGED.tsv`，然后把尝试目录原子改名为 `<T>/judge`（这一步就是「已判」标记；期间别的判定先建了 `judge/` 或台账行 ⇒ 12，本次尝试记 `ALREADY_JUDGED` 留在原处）；最后追加 `JUDGED.tsv` 一行、`STATE` 一行（`by = capjudge`；状态 = 上游与本次 `launch_verdict` 中较差者，所以判定器不会把 caplaunch / capjobs 记的 VOID、ENV_INVALID、NOT_RUN 改好；原因写 `launch_verdict=… upstream=… verdicts_sha=…`；这一行的状态与原因也记在 `verdicts.json` 的 `state_line`，上游明细记在 `upstream`）。

**一棵树只判一次，失败可复查**：只有产出了判定结果的那次运行会建立 `judge/`。输入无效（13）或异常中断（1）时，尝试目录留在 `judge.attempts/` 下（`attempt.json` 记 `status`：`INPUT_INVALID` / `ERROR`（含 traceback），连同 `inputs.json`、`judge.log`），不建 `judge/`、不写台账；输入修好后可以再判，已判过的树再判 ⇒ 12。

**补写**：改名为 `judge/` 之后、追加台账行或 `STATE` 行之前出错或被杀时，树已是「已判」，但缺台账行或 `STATE` 行。再跑同一命令时：`judge/` 存在而台账行（按启动键）或带本 `verdicts_sha` 的 `capjudge` 行缺失 ⇒ 按 `judge/verdicts.json`（`identity`、`state_line`、文件 sha）补上缺的那一项，不重判，`judge/attempt.json` 改回 `JUDGED` 并记 `commit_completed`，退出码取 `verdicts.json` 的 `exit_code`。两项都在 ⇒ 12。台账行的 `verdicts_sha256` 与 `judge/verdicts.json` 不符、`judge/` 读不了、或 `attempt.json` 记的写者在本机仍在运行（可能还在提交）⇒ 12，不补写。

**适用检查** = 强制集 `{C3, C10, C11, GPU-IDLE}` ∪ `register.json` 的 `judge.checks`。登记表可以加检查，不能去掉强制集。C2、C4、C12 按下表判（INTERFACES 勘误 B2-2A）；C2 只在有 `stub_out_of_domain` 角色、C12 只在有加倍角色的场景登记，否则应检为 0 ⇒ NOT_RUN。另有三项始终跑的前提：`RECORDS`（档目录可读、文件 identity 的档号与所在档一致、capbin `meta.overflow == 0`、每进程恰好两份窗口快照；`tier.json` 的 `snapshot_copies` 必须是列表，其中记 `snapshot_missing` 的项（编排器按角色 `snapshots` 范围应拷而在两个快照周期内没等到的进程，例如命令行与 `CAPPROBE_ENTRY` 不匹配、探针没启用的产品进程）或列了却读不到副本的项 ⇒ RED）、`P-PIDS`（`tier.json` 的 `pids_stable`）、`P-SENT`（负载器每 worker 的 `planned` = 时间表推出的条数，否则 RED；`sent ≠ planned` ⇒ NOT_RUN）。

## 各检查的应检条数与口径

每项状态的优先级 RED > NOT_RUN > PASS：有任何红证据就是 RED（即使应检为 0）；否则应检或已检为 0、或有 NOT_RUN 理由 ⇒ NOT_RUN；都没有才 PASS。

| 检查 | 应检条数来源 | 红 / NOT_RUN |
|---|---|---|
| C1 | `measure_class=load_source` 角色的 `workers` 之和（逐 worker 自报） | 按登记项 `bind` 取该角色的 `loadgen_band` 形态（INTERFACES §9.6）：档在形态 `match` 里时，worker 的计划负载（档总负载 × 时间表份额）> `per_worker_max` ⇒ RED；四项健康量 `T1:send_lag_p50_ns`、`T3:send_lag_p99_ns`、`T3:fionread_nonzero_frac`、`T3:loop_lag_p99_ns` 超形态带上界 ⇒ RED；没有 `loadgen_band`、角色未绑定、档不在 `match`、上界为 null、值或带缺 ⇒ NOT_RUN；worker 缺摘要或请求记录 ⇒ RED |
| C2 | `measure_class=stub_out_of_domain` 角色的 `workers` 之和 | 同 C1，形态取自 `stub_calib.out_of_domain`（§9.7）：计划负载按 worker 均分对 `per_process_max`；`T3:loop_lag_p99_ns`（摘要 `loop_lag_ns`）、`T3:thread_cpu_cores`（摘要 pid 的全部线程在窗口内的 utime+stime，单位核）对带；`stub_in_domain` 只列出，不进 P |
| C4 | 时间表 `meta.per_tier` × `kind=client` 角色数 | 计划在本档的请求：负载器 `tokens` 列，为 -1 时取本档 `server.streams` 同一 `req_id` 的 `tokens`；有计数且 ≠ `max_tokens` ⇒ RED（不论状态）；无计数的失败请求算已检；无计数的成功请求、缺记录 ⇒ NOT_RUN |
| C12 | `reconcile` 的 `requests` 链上 tier 跳：摘要来源且 `workers ≥ 2` 的角色按 worker、槽来源且 `expected_procs ≥ 2` 的角色按登记进程 | 任一单位本档计数 ≤ 0 ⇒ NOT_RUN（加倍没有生效，不算通过，也不算红）；取不到计数 ⇒ NOT_RUN |
| C3 | 时间表 `meta.per_tier`；已检 = `kind=client` 角色各 worker 的 `ok + failed` | 非过载型失败 > 0 ⇒ RED；已检 < 应检（有计划请求没有结果）⇒ NOT_RUN。只认负载器 `class_hint == overload`（登记超时）为过载型，其余（含 `unclassified`）按非过载型；负载器的连接类失败（`conn_refused` / `conn_fail` / `conn_reset`）不论提示一律非过载型，明细 `by_category` 记作 `loadgen_connection`（`symptoms.reclassify_failure`，勘误 B2-2E-6） |
| C5 / C6 / C7 | `reconcile` 里该类链的 `scope: tier` 跳数；C7 另加「抽样请求数」= 时间表里本档、`req_id % chunk_sample_mod == 0` 的请求 | 每跳值 ∉ [`expected_hop_count(n − 过载失败)`, `expected_hop_count(n)`] ⇒ RED；C7 抽样请求的块行数（本档与其后清空档的 `chunks` 文件合计，块行按接收时刻归档）≠ 每请求帧数 ⇒ RED；本档文件里不属于本档抽样请求的行（stray）⇒ RED |
| C8 | 覆盖清单里 `proc_kind` 属于本次启动的位点个数（进程类 = `launch.json` 各角色 `proc_kinds` 的并集；产品角色声明自己运行的 capprobe 进程类） | 适用位点的槽在全部窗口快照里都不存在，或窗口增量之和 ≤ 0 ⇒ RED（直方图槽按各桶计数之和算增量）；其余位点在明细里列为「本拓扑不适用」，不算 PASS 也不算 RED；清单缺失或无位点、没有角色声明 `proc_kinds`（或格式不对）、没有一个适用位点 ⇒ NOT_RUN |
| C9 | 窗口长 ÷ `sidecar/meta.json` 的 `interval_ns` × 线程清单里的线程数 | 任一行 `cpus_allowed` 越出本角色核或与他角色核相交、`last_cpu` 不在本角色核 ⇒ RED；已检行数 < 应检 ⇒ RED |
| C10 | 各角色 `expected_procs` 之和 | `versions.check_versions` 不过 ⇒ 整次启动 NOT_RUN（D11） |
| C11 | `planned_polls(清空档开始, t_pass 或期限, poll_ns)` | 逐次复核快照字段（`gate.check_snapshot`，门涉及角色的**全部**登记进程都必须有可用快照）、读数 = 各项之和、每项 `n_procs` = 该角色登记进程数、规则、`settle_polls`；复核的 passed / t_pass 与记录不符 ⇒ RED；`record_only` 未过 ⇒ RED；`enforce` 未过 ⇒ NOT_RUN；`baseline_band` 规则的复核 ⇒ NOT_RUN（第二批） |
| C13 | 启动级：`under_test` 角色 `expected_procs` 之和；逐档：`under_test` 角色数 | 线程清单缺、任一线程 `identified_as` 为空、`roles.json` 有 `unattributed`（`aux_helpers` 里的辅助进程归属其父角色，只在明细里列出，不算被测进程、不进任何角色的进程集合）、`tier.json` 窗口始末线程未认出 ⇒ RED；第二半「外来忙碌占比」（被测核窗口内忙碌 tick（user nice system irq softirq steal）− 本角色登记线程的 utime+stime；D14 结论为真时再减 softirq，只减 softirq，硬中断 irq 照算外来，因为登记的 D14 规则只裁决 softirq）超带 ⇒ RED；`pilot.json` 无 `d14_softirq` ⇒ 第二半 NOT_RUN；被测核窗口内有整行 null（核下线）或不足两行 ⇒ 该角色第二半 NOT_RUN |
| GPU-IDLE | 被判档 Σ(窗口长 ÷ `sidecar/meta.json` 的 `gpu_every_ns`) × `gpu.jsonl` 头的 `gpu_count` | `util_pct > 0`，或 `compute_pids` 里有不属于本次启动的进程（本次启动的进程 = `roles.json` 的 `procs` 与 `aux_helpers`，加上各档 `tier.json` 窗口边界扫描记下的 `descendants`、`monitor_descendants`、`aux_helpers`；pid 为 null 的不算本次启动的）⇒ 整次启动 VOID；本次启动的进程在利用率 0 时持有计算上下文（如空 executor 启动时的 CUDA 调用）不判红，按 pid 在明细里逐条列出（行数、首末时刻）；`gpu_idle.json` 或 `gpu.jsonl` 头的阳性对照不过、`gpu_count ≤ 0`、`gpu_every_ns` 缺失或 ≤ 0 ⇒ NOT_RUN；读不出的行（`query_rc ≠ 0`、`util_pct` 不是数（`[N/A]`）、`compute_pids` 不是列表）不计入已检并记 NOT_RUN；已检行数不足 ⇒ NOT_RUN |
| PROBE-TIER | `probe_tier` 非空的角色数 | 覆盖清单 `tier`、`launch.json` 的 `probe_tier`、角色声明三者不一致 ⇒ RED |

## 外部症状 X（设计稿 §2.6）

- **X2（主量）**：逐秒中位数序列的 T1（窗口对半切，后半中位数 − 前半中位数），任一序列超出带上界 ⇒ X2 成立。序列：每个有 `loadgen.requests` 的角色的请求层「首块 − 计划发出」；`under_test` 的负载器另加「实发 − 计划」；有 `chunks` 且本档有节拍时的消息层「收到 − 该块对应步的计划时刻」（按 `req_id` 连接服务端 `streams` 的 `first_step`，步偏移来自 `expect.msg_step_offsets(N,k,s,"product")`；`first_step = −1`（asap）⇒ 该序列 NOT_RUN）；排空端 `drain.msgs` 的「收到 − planned_rel_ns」。没有序列超带、但有序列算不出 ⇒ X2 NOT_RUN。
- **X1（对账）**：`窗口内完成数 ÷ 窗口内计划数`，除以同一次启动基线档的同一比值。基线档自身的归一值恒为 1，不记作样本。
- **X3（对账）**：由逐请求记录推出的在途数（每秒中点 `sent ≤ t < done`）的 T1；以及 `sidecar/snapshots.jsonl` 里各角色 `backlog.*` **gauge**（种类取自窗口快照的槽表；`backlog.unread_bytes` 除外）逐秒和的 T1。
- 过载型失败 > 0 ⇒ X 成立。X1、X3 至少要算出一个，否则 X NOT_RUN。
- **矛盾 ⇒ VOID（对称）**：X2 成立而 X1 显著高于带（交付超过提供）或 X3 显著低于带（积压在降）；X2 不成立而 X1 显著低于带或 X3 显著高于带。
- **I、V 与等级**：只在模式 `P+X+I+V` 下算，定义、门槛来源与输出字段见 INTERFACES 勘误 B2-2E（`localize.py`、`intervene.py`、`grade.py` 的模块说明是同一份规则的代码版）。模式 `P+X` 下每档的 `X.I`、`X.V` 与顶层 `localisation_I`、`intervention_V`、`grade` 写 `NOT_REQUESTED`，UNSTABLE 只表示「X 成立」。

## 统计量键（`<T>:<量>:<对象>`，对象 = 角色名）

判定器把每档算出的值写进 `verdicts.json` 的 `tiers[].stats`（键 → 值列表），`pilot` 只从这里取值。

| 键 | 含义 | 用在 |
|---|---|---|
| `T1:req_first_lag_p50_ns:<角色>` | 请求层首块滞后逐秒中位数的 T1 | X2 |
| `T1:msg_lag_p50_ns:<角色>` | 消息层滞后逐秒中位数的 T1 | X2 |
| `T1:send_lag_p50_ns:<角色>` | 发送滞后逐秒中位数的 T1 | C1；`under_test` 时也进 X2 |
| `T2:delivery_ratio_norm:<角色>` | X1 归一后的交付率 | X1 |
| `T1:inflight_derived:<角色>` | 推出的在途数逐秒序列的 T1 | X3 |
| `T1:backlog.<槽>:<角色>` | sidecar 里 `backlog.*` gauge 逐秒和的 T1 | X3 |
| `T3:send_lag_p99_ns:<角色>` | 窗口内发送滞后 p99（逐 worker） | C1 |
| `T3:fionread_nonzero_frac:<角色>` | FIONREAD 非零抽样占比（逐 worker） | C1 |
| `T3:loop_lag_p99_ns:<角色>` | 事件循环延迟直方图 p99 桶上界（逐 worker） | C1 |
| `T3:foreign_busy_frac:<角色>` | 被测核集上不属于已登记线程的忙碌 tick 占比 | C13 第二半 |
| `T3:thread_cpu_cores:<角色>` | 桩 worker 进程全部线程在窗口内的 CPU（核） | C2 |
| `T2:core_busy_frac:<角色>`、`T2:core_busy_frac_softirq:<角色>` | 单核进程的核忙碌占比（不含 / 含软中断）相对基线档的偏移；按 D14 结论取其一 | I1（单核） |
| `T2:gil_domain_cores:<角色>`、`T2:gil_run_delay_frac:<角色>` | GIL 域（`class = python` 的线程）核数、运行队列等待占比相对基线档的偏移 | I1（多核） |
| `T2:nongil_thread_cores:<角色>` | 非 GIL 线程核数相对基线档偏移的最大值 | I1（多核） |
| `T1:i3_backlog:<读数名>`、`T2:i3_backlog:<读数名>` | 登记的 I3 积压读数（`judge.localize`）逐秒序列的 T1、相对基线档的 T2 | I3 |
| `T2:loop_lag_p50_ns:<角色>` | 事件循环延迟 p50 相对基线档的偏移 | I2（只作旁证） |

阈值只来自登记表 `calibration` 里的标定文件与 E0 的 `inline_thresholds`（形状 `{"noise_bands": {键: {"min","max","n"}}}`）。噪声带（`pilot`、`near_boundary_noise`、`inline_thresholds` 的 `noise_bands`）按档的 regime 选：登记表 `judge.near_boundary_tiers` 列出的档只读 `regime: near_boundary` 的带，其余负载档只读 `subsat` 的带（带上没写按 `subsat`），近边界档不退回亚饱和带；同一 (键, regime) 出现在两个来源 ⇒ 13。查不到 ⇒ 用它的检查 NOT_RUN。C1、C2 的带不在 `noise_bands` 里，而在 `loadgen_band` / `stub_calib` 的形态里（按登记项 `bind` 取）。每种标定文件都核 `environment` 与本树一致。

## 本包对其他组件文件的读取约定（INTERFACES 没写死、本包按此读；对应组件请对齐或提出）

- `launch.json` `roles[]` 必须带 `kind`（场景配置的 `server`/`client`/`aux`）：C3、P-SENT 用 `kind == client` 认负载器；缺 ⇒ 这两项 NOT_RUN。`launch.json` 的 `probe_tier` 可为字符串或 `{"tier": …}`。
- `tier.json` 的 `threads` 形如 `{"wstart": {pid: [线程…]}, "wend": {…}}`。
- `sidecar/meta.json` 带 `interval_ns`（采样周期，整数 ns，`capenv.sidecar` 的写法）与 `clk_tck`。
- `env/versions/*.json` 是版本记录本身，或 `{"versions": 记录}`；pid 取自记录里的 `pid`，角色经 `roles.json` 按 pid 查。
- `env/gpu_idle.json` 带 `positive_control_ok`、`gpu_count`；`sidecar/gpu.jsonl` 头带 `gpu_count`。
- `pilot.json` 的 `d14_softirq`：判定器（C13 第二半）只读其中的布尔 `irq_softirq_is_experiment`，没有这个键 ⇒ NOT_RUN。它由 `pilot` 按登记的规则算出（见下节）。
- 窗口快照副本：每进程按 `t_mono_ns` 排序取首末两份，不按文件名里的 `wstart`/`wend`。

## pilot 列表文件

每行 `<启动树>\t<档序号,档序号…>`（必须写明用哪几档；不接受通配符、同一树两行）。每棵树必须已判过；任一检查 RED 或 VOID 的档被剔除并列在 `inputs[].tiers_excluded`。判定之前已记下的启动级结论不是 VALID 的整棵树被剔除（`inputs[].tree_excluded` 写原因，全部档列进 `tiers_excluded`，也不进 D14 读数）：判定时记下的 `verdicts.json` `launch_checks.LAUNCH-STATE` 不是 PASS，或现在重读的 `STATE`（非 `capjudge` 行）与 `launch_end.json` 不是 VALID；`verdicts.json` 没有 `LAUNCH-STATE`（旧版判定器判的）⇒ 3。判定器自己的 `launch_verdict` 不要求 VALID（E0 判定时还没有噪声带，那里是 NOT_RUN）。全部树的环境（画像 sha、镜像 sha、capkit sha、基线 sha）必须相同。产出：`noise_bands`（T3：各键在入选档上的 [min, max]、n、来源档）、`window_ns_by_arm`（每个臂（启动身份的 `arm`）各自搜索：本臂序列等长时，前 L 秒的 T1 都落进全窗口噪声带的最短偶数秒 L；只在 L = 全窗时归零记 null 并在 `window_search_by_arm` 里标 `saturated_at_full_window`，臂内不等长记 null 与 `unequal_series_lengths`）、`warmup_ns_by_arm`（用本臂窗口，从档开始偏移 w 秒后在途数无趋势的最小 w）、顶层 `window_ns` / `warmup_ns`（各臂都非 null 时取最大，否则 null；INTERFACES 勘误 SM-16）、`drain_max_wait_ns`、`d13_switchinterval_s`、`d14_softirq`（下节）；`tier_resolution`、`launches_per_arm`、`min_detectable_diff`、`i2_calibration`、`core_budget`、`canary_band` 写 `null` 并列在 `not_computed`（第二批）。不覆盖已有文件。

## D14：被测核上的 softirq 算不算实验自己的 CPU 时间（`capjudge.d14`，勘误 R1b）

用户 2026-11-13 的裁决：按 E0 实测、按登记的规则定。`--d14-rule` 必填，是一个 `capkit.d14_rule/1` 文件，必须已在 `<W>/REGISTRY.tsv` 以 `kind=d14_rule` 登记（未登记、键不全、取值越界 ⇒ 退出码 3）：

```json
{"schema":"capkit.d14_rule/1","idle_softirq_frac_max":0.002,"load_min_excess_frac":0.005,
 "monotonic_slack_frac":0.002,"min_trees":1}
```

- 读数（每棵树、每个 `under_test` 角色、该角色在 `pin_table.json` 里的核）：空载 = `raw/env/exclusivity.json` 第 6 项（门禁空转窗口，作业里第一个启动之前）各核的 `softirq_ticks`、`irq_ticks`、`total_ticks`；负载 = 本树入选的每个负载档，`sidecar/cpus.jsonl` 在档稳态窗口内各核首末两行之差（与 C13 第二半同一实现 `treedata.cpu_tick_deltas`；窗口内有整行 null（核下线）或不足两行的核使这一档读数不完整，记进 `why`，不计入）。`softirq_frac` = softirq tick ÷ 同区间全部 tick（user nice system idle iowait irq softirq steal）。
- 规则：空载 `softirq_frac` ≤ `idle_softirq_frac_max`（接近 0）；每个负载档 ≥ 空载 + `load_min_excess_frac`，且按负载（档位表轴的方向）排序后相邻两档下降不超过 `monotonic_slack_frac`（随负载增长）。
- 结论：空载读数是作业级的（门禁每次作业运行做一次，拷进该次运行的每棵树），所以完整的树另按作业运行 `(job_id, restart_count)` 计数。完整的树（每个 `under_test` 角色都有空载读数和至少一个负载档读数）≥ `min_trees`、且这些树来自 ≥ `min_trees` 个作业运行时，`irq_softirq_is_experiment` = 全部完整树的全部角色都满足规则；否则、或规则文件的登记时刻不严格早于某棵输入树的 `submitted_at`，不写这个键（C13 第二半保持 NOT_RUN），`conclusion_reason` 写明原因。
- `d14_softirq` 另带 `rule`、`rule_file`（`relpath sha256 registered_at`）、`readings`（逐树逐角色的原始读数与 `idle_close_to_zero`、`grows_with_load`，逐树的 `job_run`）、`n_trees_complete`、`n_job_runs_complete`。
- 没有 `under_test` 角色的树（如 `e0a-nullsse`）不完整，只列出不计数。
