# 判定器自测样本（合成）

INTERFACES §10.4、设计稿 §2.7 末段的三类样本 + 两个可满足性样本。**全部是合成样本**（每个 `launch.json`、`register.json`、`pilot.json` 都带 `"synthetic": true`）；按计划 P8，第一个 E0 作业采到真实样本后，以真实良品替换 `good` 作参照并重跑自测。

- 生成：`python3.11 make_samples.py --out <目录>`（目录下不得已有同名样本；`--only <名字>` 只建一个）。数据来自一次确定性模拟（固定随机种子）；`written_mono_ns`、`written_wall` 等时间戳每次不同，其余内容不变。
- 运行：`python3.11 run_selftest.py --tools <C>/tools --package <capjudge|judge> --tmp <已存在的临时目录> --expect-samples 22`。每个样本先整份拷到临时目录再判（判定器只判一次，不放宽任何输入核对）；良品另做：再判一次 → 12、树拷到别处 → 12（JUDGED.tsv）、`--tree` 带通配符 → 2、不存在 → 2、按 `sabotage.json` 改一处（先断言目标文件确实变了）→ 期望 C5 红。
- 每个样本 = 一个最小工作目录 W：`REGISTRY.tsv`、`submissions.tsv`（`local-sample-*` 作业，PLANNED + SUBMITTED 两行）、空的 `JUDGED.tsv`、`register/`（档位表、时间表、prompt 池、场景、pilot、登记表，全部已登记且登记早于提交）、`runs/<树>/`、`expect.json`。

| 样本 | 改了什么 | 期望 |
|---|---|---|
| `good` | — | 0；档 2、4 STABLE |
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
| `gpu_busy` | 一秒 GPU 利用率 5% | 10；整次启动 VOID（GPU-IDLE） |
| `p_sent_short` | worker 1 档 4 实发少 1 | 11；档 4 NOT_RUN |
| `records_overflow` | 档 4 chunks 的 `meta.overflow = 3` | 10（RECORDS） |
| `probe_tier_mismatch` | 覆盖清单写 P1、登记 P2 | 10（PROBE-TIER） |
| `x_contradiction` | 滞后无趋势，但 sidecar 积压读数逐秒 +10 | 10；档 4 VOID（X3 与 X2 矛盾） |
| `sat_overloaded_tier` | 档 4 滞后每秒 +50 ms，另有 5 个登记超时（过载型） | 0；档 4 UNSTABLE（不是 VOID） |
| `sat_spin_thread_same_process` | nullsse 进程内多一条已认出的自旋线程（0.7 核），档 4 滞后增长 | 0；档 4 UNSTABLE（「域到顶」，不是 VOID；第一批无定位） |
| `input_registered_late` | `register.json` 登记时刻晚于提交 | 13 |
| `input_identity_mismatch` | 一个自报文件的 `identity.arm` 不同 | 13 |
| `empty_tree` | 空目录 | 11 |
