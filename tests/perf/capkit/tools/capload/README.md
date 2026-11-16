# capload（B1）：HTTP SSE 开环负载器与空汇

本包有两个入口（`INTERFACES.md` §11）：

- `python -m capload`：多进程 HTTP/1.1 SSE 开环负载器（设计稿 §2.2 第 1～10 条）。
- `python -m capload.nullsse`：多进程 `SO_REUSEPORT` + `asyncio.Protocol` 的 SSE 空汇。

只依赖标准库与 `capcommon`；`uvloop` 只在 `--loop uvloop` 时导入（导入失败退出码 3）。

## 进程模型（两个入口相同）

- 编排器起的那一个进程是**父进程**：用 `spawn` 起 W（或 Z）个 worker，只有它处理 `SIGTERM`/`SIGINT`（`StopController`）：第一次信号转成 `stop`、第二次转成 `hard`，经 Pipe 发给 worker。worker 忽略这两个信号（进程组信号不会被算两次）；Pipe 断开（父进程没了）按 `stop` 处理。
- 父进程也跑一个同实现的事件循环（版本自报里有 `loop_impl`），并发布快照：本工具的 `SLOTS` 全部登记、值恒为 0。原因：清空门读角色的**每个**登记 pid，缺快照算「不满足」（§8.3）。
- 所以场景配置里 `expected_procs = workers + 1`。
- worker k 绑到 `--cpus` 的第 k 项；父进程绑到整个列表。就绪文件的 `children` 列出全部 worker pid，`info.workers` 是各 worker 的就绪信息（版本自报、亲和性读回、`aux_threads` 等）。
- nullsse 的父进程先绑定（不监听）占住端口，全部 worker 都 `listen` 之后才写端口文件，再写就绪文件。

## capload 参数

必需参数见 `INTERFACES.md` §11。补充说明：

| 参数 | 说明 |
|---|---|
| `--fionread-every-ms` | 内务 tick 的周期：每个 tick 对每条打开的连接取一次 `FIONREAD`（写 `fionread` 记录、更新 `backlog.unread_bytes`），并扫一遍已登记超时。所以超时的判定精度就是这个周期 |
| `--max-records-per-worker` | `chunks`、`fionread`、`conns` 三个记录数组各自的容量。`requests` 的容量 = 本 worker 在时间表里的行数；`lag` 的容量按「原点前剩余时间 + 档位表总时长」÷ 周期算（与空汇相同，集成时改）；这两个都不会溢出 |
| `--pool-pick roundrobin` | 空闲连接 FIFO（最久没用的先用）；`random` = 在空闲连接里按 `--pool-seed`（每 worker 派生不同种子）均匀抽 |
| `--pool-reset per-tier` | 每个负载档开始时关掉全部空闲连接（计 `pool_reset_closed`）；`never` 不关 |

可选参数（不改变任何规定的输出格式）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host-header` | `--target` 的值，否则 `localhost` | HTTP `Host` 头 |
| `--mode {open,wave}` | `open` | `wave` = 闭环波模式（见下） |
| `--wave-size C`、`--wave-delays-ns d0,…`、`--wave-gap-ns G` | —、全 0、0 | 只在 `--mode wave` 时有效 |
| `--slow-read-pause-us N` | 0 | 阳性对照：每次读到数据后 `pause_reading` N µs（「故意慢读」，验证 `FIONREAD` 读数会变） |

## capload 的行为要点

- **绝对时间表**：每个 worker 一个分发定时器（`capcommon.aio.call_at_ns`）对准下一条的计划时刻；到点把计划时刻已过的全部发出，**不跳过、不提前**。时延从计划时刻起算，另记实际写出时刻（`requests` 的 `planned_rel_ns`、`sent_rel_ns`）。
- **请求体**启动时按（档、prompt 项）预先渲染；发送路径只拼「固定前缀 + 十进制 req_id + 预制后缀」。请求体带 `max_tokens`、`stream`、`stream_options.include_usage=true`、`ignore_eos=true`；头带 `x-capload-id/k/s/tier`（`capcommon.wire.http_headers`）。`/v1/chat/completions` 的 token id 输入走 TRT-LLM 的 `prompt_token_ids` 字段（`messages` 必填，给一个空 user 消息）。
- **接收**：`asyncio.Protocol` 内解析 HTTP 响应头与 chunked 编码，负载喂给 `wire.SseCounter`（计块口径 §4.5）。`--recv-mode full` 另用 `FullSseParser` 逐帧 `json.loads`，帧数、usage 与计块口径不一致即 `protocol_error`。响应头里若有 `x-capload-id` 且与请求不符 → `protocol_error`。
- **结果判定**：非 2xx → `http_<码>`；没见到 `[DONE]` 或末尾半帧 → `protocol_error`；没有 usage → `protocol_error`；`completion_tokens ≠ max_tokens` → `short_output`。
- **连接池**按需增长、不设上限。对端在空闲时关掉的连接计 `peer_idle_closed`；复用一条空闲连接后、一个字节都没收到它就断了 → 换新连接重发一次，计 `reconnects`，不算请求失败（设计稿 §2.2 第 4 条）；收到过任何字节（含不完整的响应头）就不重发。重发不再计 `self.written` 与发送滞后，`sent_rel_ns` 仍是第一次写出的时刻，所以这类请求的 `first_from_sent`/`done_from_sent_ok` 含重连耗时。`new-per-request` 时请求带 `Connection: close`，建连失败计 `conn_refused`/`conn_fail`，`failures_split` 把「连接失败」与「请求失败」分开。
- **两端都显式设 `TCP_NODELAY`**。实测：空汇的监听 socket 若用 `proto=0` 创建，asyncio 不会给接受的连接设 `TCP_NODELAY`，复用连接上每个响应多出约 40 ms（Nagle + 延迟确认）。
- **超时**：只读档位表的 `request_timeout_ns`；截止 = 计划时刻 + 超时（`config.timeout_from = "planned"`），由内务 tick 扫描，过期即 abort 连接，记 `timeout_registered`（`class_hint = overload`）。
- **注入位点**：`send.req`（每请求一次；`busy`/`sleep`）、`recv.chunk`（每个计到的 SSE 帧一次；`busy`/`sleep`/`drop`/`miscount`）。不支持的位点或 mode 在启动时退出码 3。逐档注入在档边界生效，逐档自报的 `inject` 由 `capcommon.inject.tier_report` 生成，另附 `inject_marks`（取 mark 的相对时刻）。
- **停止**：跑完最后一档后，等在途请求最多 `--stop-grace-s`，剩下的记 `aborted_at_stop`，退出码 0。在最后一档结束前收到停止 → 不再发新请求、在途的等 `--stop-grace-s`，退出码 5；`control/stop.json` 原样抄进每份 summary 的 `stop`。
- **最后一档之后**事件循环延迟只进直方图，不再写逐样本记录（防止停止信号来得晚时 `lag` 溢出）。

### 闭环波模式（`--mode wave`，E12 L2 用）

每个负载档按时间表里本 worker 的 req_id 顺序成波：一波 C 条，第 j 条的计划时刻 = 波起点 + `wave_delays_ns[j]`；一波全部完成（成功或失败）后，下一波的起点 = max(档开始, 完成时刻 + `wave_gap_ns`)。本档的请求全部发完才轮到下一档（可能越过档末，落进清空档）。`requests` 里的 `planned_rel_ns` 是上述波计划时刻；应发条数仍取时间表（`counts.planned`），所以时间表要按想发的条数生成。

## capload 的输出（每 worker 每档，§5.4）

`summary.json`（`capkit.tier_summary/1`）在 INTERFACES §5.4.1 的键之外另有：

| 键 | 含义 |
|---|---|
| `counts.sent` | 本档（按计划档）**已分发**的请求数，含建连失败的；`counts.not_sent = planned − sent` |
| `counts.recorded`、`counts.unfinished` | `requests` 里本档的行数；`sent − recorded`（应为 0） |
| `failures_split` | `{"connect": 建连失败数, "request": 其余失败数}` |
| `latency_ns` | `first_from_planned`、`first_from_sent`（首帧）、`done_from_planned_ok`、`done_from_sent_ok`（成功请求的完成），均为 `loghist/1` |
| `conn_pool` | `size_max`、`new_conns`、`reconnects`、`peer_idle_closed`、`conn_fail`、`pool_reset_closed`、`unsolicited`（空闲时收到字节）——按事件时刻归档；`requests_on_new_conn` 按计划档 |
| `stop_reason`、`stopped_early`、`config`、`status_codes` | 停止来源（`signal`/`parent_pipe_closed`）、是否在最后一档前被停、本次参数、状态码表 |

逐行记录：`requests`、`chunks`（`nbytes` = 完成该帧的那次读的字节数；按事件时刻归档）、`fionread`、`lag`、以及新增的 **`conns`**（`kind: loadgen.conns`：`conn_id`、`local_port`、`peer_port`、`opened_rel_ns`、`closed_rel_ns`（仍打开为 −1）、`n_requests`、`close_reason`（`meta.close_reasons`）、`tier_seq`（按建连时刻））——E6 的「连接 → 前端」映射用它与 `requests.conn_id` 连接。表里每一档都写一份文件（没有行也写），缺文件不会被读成 0 行。

失败原因在 §5.4.1 的表之外新增 **`aborted_at_stop`**（`class_hint = unclassified`）：停止或最后一档结束后宽限期满时仍在途的请求。

## nullsse

- 每请求参数只取自请求：`max_tokens`（或 `max_completion_tokens`）与 `stream_options.include_usage`（有 `stream_options` 没写 `include_usage` 按 OpenAI 缺省 true）从请求体按 JSON 键搜索；k、s、计划档、req_id 从 `x-capload-*` 头读。没有这些头的请求（如 `benchmark_serving.py`）按 k = s = 1、到达时刻所在档计，记 `foreign_requests`。
- 响应：`transfer-encoding: chunked`，每个 SSE 事件一个 HTTP chunk；M = `expect.msgs_per_request(N, k, s, "product")` 个内容帧（每个恰好 `--chunk-bytes` 字节），请求要 usage 时加一个 usage 帧，然后 `[DONE]`。回显 `x-capload-id`。
- `--pacing beat`：全进程一个 `BeatSchedule`；一条流的第 j 帧在步 `first_step + expect.msg_step_offsets(…)[j]` 的计划时刻写出，`first_step = step_at_or_after(接纳时刻)`；写不等待对端读（transport 缓冲无界）。超出档位表最后一步的帧立即写，记 `beyond_beat_schedule`。`asap`：接纳后一次写完。
- `--idle-close-s`：连接空闲这么久就关（模仿 uvicorn 的 5 s keep-alive）；≤ 0 不关。
- 注入位点：`send.chunk`（每内容帧；`busy`/`sleep`/`drop`/`miscount`）、`short.req`（每请求；`short`）。
- `--max-records-per-worker`（可选）：`streams` 记录容量，缺省 = 档位表推出的时间表总行数（`count_schedule_rows`）；`lag` 的容量按「原点前剩余时间 + 档位表总时长」÷ 周期算。
- `summary.json` 的 `counts`：`requests`、`frames_sent`、`bytes_sent`（按计划档）、`accepts`、`idle_closed`（按事件时刻），另有 `streams_completed`、`streams_aborted`（对端断开）、`streams_unfinished_at_stop`、`bad_requests`、`foreign_requests`、`beyond_beat_schedule`、`short_requests`；口径逐条写在 summary 的 `count_rules` 里。

## 快照槽（`SLOTS`，§6.2）

`capload.LOADGEN_SLOTS`（= `capload.SLOTS`）与 `capload.SINK_SLOTS`，启动时一次性全部登记。清空门用：负载器 `backlog.inflight`（gauge）、空汇 `backlog.active_streams`（gauge）；`backlog.unread_bytes` 只作记录。其余是 `self.*` 计数、`lag.loop` 直方图、`inject.<位点>.{hits,applied,applied_ns}`。

## 本机测试（只验功能，不取性能数）

```bash
cd <C>
systemd-run --user --scope -q -p MemoryMax=6G timeout 600 \
  /home/scratch.yihlu_coreai/tmp/20261106-wt03-venv/venv/bin/python -m pytest -q tests/capload tests/nullsse -p no:cacheprovider
```

`tests/capcommon` 的测试用 `from conftest import …`，与其他测试目录同一次收集时会互相遮蔽，请分开跑。
