# 判定器自测样本（合成）

INTERFACES §10.4、设计稿 §2.7 末段的三类样本 + 两个可满足性样本。**全部是合成样本**（每个 `launch.json`、`register.json`、`pilot.json` 都带 `"synthetic": true`）；按计划 P8，第一个 E0 作业采到真实样本后，以真实良品替换 `good` 作参照并重跑自测。

- 生成：`python3.11 make_samples.py --out <目录>`（目录下不得已有同名样本；`--only <名字>` 只建一个）。数据来自一次确定性模拟（固定随机种子）；`written_mono_ns`、`written_wall` 等时间戳每次不同，其余内容不变。
- `make_samples.build()` 另有只供测试用的选项（`calib_v2`、`v2`、`calibration_entries` 等，INTERFACES 勘误 calib-v2-A2 第 8 条），下表的样本不用它们。（生成器的缺省标定已改为 v2 源，另有选项 `calib_v1`、`c13_foreign_rate`；本目录的样本已按 v2 缺省重新生成，见勘误 calib-v2-A5 第 6 条。）`calibv2_real/` 不是样本（没有 `expect.json`），是测试用的真实片段，来源与 sha 见其 `SOURCES.md`。
- 运行：`python3.11 run_selftest.py --tools <C>/tools --package <capjudge|judge> --tmp <已存在的临时目录> --expect-samples 52`（`--expect-samples` 必填，等于下表行数）。每个样本先整份拷到临时目录再判（判定器只判一次，不放宽任何输入核对）；期望 13 的样本另断言没有 `judge/`、`judge.attempts/` 下恰有一条 `INPUT_INVALID` 记录，且再判仍是 13；良品另做：`judge/attempt.json` 为 `JUDGED`、再判一次 → 12 且不新增尝试记录、树拷到别处 → 12（JUDGED.tsv）、`--tree` 带通配符 → 2、不存在 → 2、按 `sabotage.json` 改一处（先断言目标文件确实变了）→ 期望 C5 红。
- 文件形状按真实写方（caplaunch 的 `tier.json`、`raw/env/versions/*.json`，sidecar 的 `meta.json`、`gpu.jsonl` 头，capjobs.envfiles 的 `gpu_idle.json`，capprobe.overlay 的覆盖清单），并与集成时本地端到端产出的启动树逐文件比对过；sidecar 的 GPU 采样周期取 2 s（`meta.json` 的 `gpu_every_ns`），GPU-IDLE 的应检行数要按它算。
- 所有样本共有：nullsse（被测、P2）按新架构单前端声明 `proc_kinds: ["frontend", "frontend_openengine", "frontend_single"]`（进程类词表见 capprobe 位点表）；覆盖清单除 F2（`frontend_openengine`）外还列一个 `W1`（`proc_kind: worker`），本拓扑没有 worker 进程，C8 把它列为「不适用」；capload 有一个 `multiprocessing.resource_tracker` 辅助进程，按 caplaunch 的写法记在 `roles.json` 的 `aux_helpers` 与 `tier.json` 的 `scans.*.aux_helpers`，判定器不把它当被测进程。
- 每棵启动树另有 `STATE`：先是 caplaunch 的 `VALID` 一行，再是 capjobs.launches 的一行（`upstream_void` 为 `VOID`，其余为 `VALID`）；`launch_end.json` 带 `state: VALID` 与空的 `leftovers.after`。判定器把它们当作判定之前已记下的启动级结论，自己的结论只能更差（LAUNCH-STATE）。期望里写了 `state_last` 的样本另断言判后 `STATE` 末行的状态。
- 每棵启动树另有 `raw/env/exclusivity.json`：只合成门禁第 6 项（被测核在空转窗口内 softirq 为 0），是 D14 的空载读数；合成 `pilot.json` 的 `d14_softirq` 只有 `capjudge.d14` 写出的键集，没有读数，结论 `irq_softirq_is_experiment` 是合成的 `true`（INTERFACES 勘误 R1b；C13 第二半与 I1 只读这个布尔值）。合成 `pilot.json` 另带 `i1_calibration`（`near_one_core_min` 0.95）与 I 用的带；模式 `P+X+I+V` 的样本在 `expect.json` 里另有 `tier_localisation`（比对 `tiers[].X.I`）与 `grade`（比对顶层 `grade`），由 `run_selftest.py` 断言（INTERFACES 勘误 B2-2E-9）。
- 每个样本 = 一个最小工作目录 W：`REGISTRY.tsv`、`submissions.tsv`（`local-sample-*` 作业，PLANNED + SUBMITTED 两行）、空的 `JUDGED.tsv`、`register/`（档位表、时间表、prompt 池、场景、pilot、`loadgen_band`（形态 `capload-r` 绑给 capload）、个别样本另有 `stub_calib` / `near_boundary_noise`、登记表，全部已登记且登记早于提交）、`runs/<树>/`、`expect.json`。

- **已知不符（SB-C 之前）**：模式 `P+X+I+V` 的 5 个样本（`sat_spin_thread_same_process`、`i_server_slow`、`i_reader_slow`、`i_reader_slow_real_pins`、`i_conflict`）的期望是 I 定位成立；I1 改读 pilot/2 的 `i1_calibration` 属 P2-13（SB-C），在那之前 bootstrap pilot/2 没有它 ⇒ I `NOT_RUN`，自测对这 5 个样本报不符、退出码 1。期望不改，等 P2-13 落地后复核。
- DESIGN-r7a §3.5 第 22 条的两个可满足性样本是两遍版 `v2_sat_overloaded_level_2p`、`v2_near_stable_level_2p`（生成器选项 `two_pass`：档位表在档 4 之后再接 down r40 档 6 与清空档 7，`ladder.json` 的级 1 为 `up_seq` 4、`down_seq` 6）；`expect.json` 的 `levels` 按级号断言 `verdicts.json` 的 `levels[]`。单遍版 `v2_sat_overloaded_level`、`v2_near_stable` 保留。

| 样本 | 改了什么 | 期望 |
|---|---|---|
| `good` | — | 0；档 2、4 STABLE；`STATE` 末行 VALID |
| `upstream_void` | capjobs.launches 在判定之前记了 `VOID`（残留进程） | 10；档 2、4 VOID（LAUNCH-STATE）；`STATE` 末行仍是 VOID |
| `probe_entry_unmatched` | 档 4 nullsse 没有任何窗口快照（探针没在被测进程里启用），`tier.json` 的 `snapshot_copies` 记 `snapshot_missing` | 10；档 4 VOID（RECORDS、C8） |
| `c1_send_lag` | worker 0 在档 4 有 30% 请求发送滞后 +500 µs | 10；档 4 VOID（C1） |
| `c3_non_overload` | 档 4 两个请求 `conn_refused` | 10；档 4 VOID（C3，连带 C5/C6/C7） |
| `c5_requests` | nullsse 档 4 自报的请求数少 1 | 10（C5） |
| `c6_probe_miscount` | 探针计数 `count.F2.msgs` 在 15 s 后少计 3 | 10（C6） |
| `c7_dropped_chunks` | 负载器丢掉 1/10 请求的第 2 块 | 10（C7） |
| `c8_probe_unwired` | 档 4 窗口末快照的位点计数没有增长 | 10（C8） |
| `c9_pinning` | 一行线程亲和集合越出本角色核 | 10（C9） |
| `c10_versions` | nullsse 版本自报缺 `capkit_code_sha` | 11；全部 NOT_RUN（D11） |
| `c11_stale_gate` | 清空门通过的那次轮询用了陈旧快照 | 10；档 4 VOID（C11） |
| `c13_unknown_thread` | 线程清单里有未认出的线程 | 10；全部 VOID（C13，连带 C9） |
| `c13_foreign_busy` | 被测核上有 0.3 核不属于任何登记线程 | 10（C13 第二半） |
| `gpu_busy` | 一次 GPU 采样利用率 5% | 10；整次启动 VOID（GPU-IDLE） |
| `gpu_util_unreadable` | 档 4 窗口内一次 GPU 采样利用率读不出（`[N/A]` → null） | 11；GPU-IDLE NOT_RUN，档 2、4 NOT_RUN |
| `p_sent_short` | worker 1 档 4 实发少 1 | 11；档 4 NOT_RUN |
| `records_overflow` | 档 4 chunks 的 `meta.overflow = 3` | 10（RECORDS） |
| `probe_tier_mismatch` | 覆盖清单写 P1、登记 P2 | 10（PROBE-TIER） |
| `x_contradiction` | 滞后无趋势，但 sidecar 积压读数逐秒 +10 | 10；档 4 VOID（X3 与 X2 矛盾）（生成器的期望已改为 0、档 4 `CONTRA`，见勘误 calib-v2-A3 第 6、9 条） |
| `sat_overloaded_tier` | 档 4 滞后每秒 +50 ms，另有 5 个登记超时（过载型） | 0；档 4 UNSTABLE（不是 VOID） |
| `sat_spin_thread_same_process` | 模式 `P+X+I+V`；nullsse 进程内多一条已认出的自旋线程（0.7 核），事件循环在档 4 从 0.10 核升到 0.27 核（核 99 % 忙，基线档 82 %），档 4 滞后增长 | 0；档 4 UNSTABLE，I 定位到 nullsse、`finding: domain_saturated`（「域到顶」，不是 VOID）；等级 `LOCALIZED` |
| `i_server_slow` | 模式 `P+X+I+V`，登记 I3 读数（nullsse 的待发队列、nullsse→capload 的在途数）；档 4 nullsse 事件循环 0.96 核、待发队列每秒 +5、在途数持平 | 0；档 4 UNSTABLE，I 定位到 nullsse（I1 与 I3）；等级 `LOCALIZED` |
| `i_reader_slow` | 同上，但档 4 capload 两个 worker 各 0.97 核、在途数顶在 64（平台） | 0；档 4 UNSTABLE，I 定位到 capload（负载源）；等级 `LOWER_BOUND` |
| `i_reader_slow_real_pins` | 同 `i_reader_slow`，但用集群上 caplaunch 写出的拓扑：capload 的父进程也登记（`expected_procs` 3），父进程与两个 worker 都绑在角色的核 2–3 上 | 同 `i_reader_slow`；I1 把核 2–3 当一个核组（多核：按进程看 GIL 域），`pids` 列出父进程与两个 worker |
| `good_real_pins` | 只换成上面的拓扑，其余同 `good` | 同 `good` |
| `i_conflict` | 同上，nullsse 0.96 核（I1）而在途数顶在平台（I3 指向 capload） | 0；档 4 UNDETERMINED；等级 `UNDETERMINED` |
| `c1_loadgen_overdriven` | `loadgen_band` 的每 worker 上界 15 req/s（档 4 每 worker 计划 20） | 10；档 4 VOID（C1） |
| `c2_stub_good` | nullsse 是域外桩（`stub_out_of_domain`，检查清单 C2 代替 C13），登记 `stub_calib` | 0；档 2、4 STABLE；C2 PASS |
| `c2_stub_loop_lag` | 同上，档 4 nullsse 事件循环延迟 p99 ≈ 50 ms | 10；档 4 VOID（C2） |
| `c4_short_output` | 档 4 一个请求只出 2 个 token（capload 记 `short_output`） | 10；档 4 VOID（C4，连带 C3） |
| `c12_idle_worker` | 档 4 capload worker 1 的请求一个都没发出（摘要 `counts.sent` 为 0，全部未发出即超时） | 11；档 4 NOT_RUN（C12） |
| `regime_nb_missing` | 档 4 登记为近边界档，近边界带只有 C13 的键 | 13（`near_boundary_noise` 已不是阈值源，勘误 calib-v2-A2 第 1 条；原期望 11 作废） |
| `regime_nb_good` | 档 4 登记为近边界档，另登记一份与 pilot 同键的 `near_boundary_noise` | 13（同上；原期望 0 作废） |
| `v2_good` | v2 登记表（`arm_config`、`ladder.json`：档 2 基线、档 4 单遍级）+ bootstrap pilot/2 + `e0_d14.json` | 0；档 2、4 STABLE；`levels[]` 一级 STABLE |
| `v2_sat_overloaded_level` | 同 `v2_good`，档 4 过载（同 `sat_overloaded_tier`） | 0；档 4 UNSTABLE（明显过载级判 UNSTABLE） |
| `v2_near_stable` | 同 `v2_good`，档 4 在 `ladder.json` 里是 `near` 角色 | 0；档 2、4 STABLE（近边界稳定档在内） |
| `v2_sat_overloaded_level_2p` | 两遍档位表（档 4 上行、档 6 下行为同一级），两遍都过载（同 `sat_overloaded_tier`） | 0；档 4、6 UNSTABLE；`levels[]` 级 1 UNSTABLE（明显过载级两遍都红） |
| `v2_near_stable_level_2p` | 两遍档位表，不过载 | 0；档 2、4、6 STABLE；`levels[]` 级 1 STABLE（近边界稳定级两遍都在内） |
| `v2_ladder_missing` | 登记过的 `ladder.json` 被改名移走（`*.moved-away`） | 13 |
| `v2_ladder_sha` | `ladder.json` 登记后多写一个字节 | 13 |
| `v2_pilot1` | v2 登记表引用 v1 `pilot.json` | 13（v1 calibration is not a threshold source） |
| `v2_window` | thresholds 阶段 pilot/2 的 `window_ns` 60 s，档窗口 4 s | 13（勘误 calib-v2-A5 第 6 条） |
| `v2_d14_missing` | 登记表不列 `e0_d14.json` | 11；档 2、4 NOT_RUN（C13 第二半无 D14） |
| `v2_d14_env` | `e0_d14.json` 的 `environment` 画像与启动不符 | 13（calibration environment） |
| `v2_arm_config_missing` | `arm_config` 缺本臂 | 13 |
| `v2_near_boundary_tiers` | v2 登记表写了 `judge.near_boundary_tiers` | 13 |
| `v2_inline_thresholds` | v2 登记表 `inline_thresholds` 非空 | 13 |
| `input_calib_invalid` | `loadgen_band` 一个带 min > max | 13 |
| `input_registered_late` | `register.json` 登记时刻晚于提交 | 13 |
| `input_identity_mismatch` | 一个自报文件的 `identity.arm` 不同 | 13 |
| `empty_tree` | 空目录 | 11 |
