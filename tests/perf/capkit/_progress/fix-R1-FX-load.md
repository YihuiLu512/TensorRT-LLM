# fix-R1-FX-load 进度（K24 + RS6/RS3/RS4 扫 capload；capgrpc 只读扫）

- 2026-11-13 开工：建 fakebin（_local/fx-load/fakebin，10 个脚本 exit 97），pyc 前缀 _local/fx-load/pyc。

## K24 核实（读码）
- 属实。`loadgen.py` 的 `on_lost` 重连分支（:688-700）对同一个 Req 调 `_connect`→`_write`，`_write`（:491-501）无条件 `self.vals[i_written] += 1`、`self.written += 1`、`h_lag.add(r.sent - r.planned)`，并把 `r.sent` 覆盖成重发时刻 ⇒ 快照 `self.written` > `self.dispatched`、`self.send_lag` 多一个含重连耗时的样本；同时 requests.capbin 的 `sent_rel_ns` 也被改成重发时刻，C1 的 T3 send_lag_p99 / T1 趋势（checks.py:113、request_lag_series）同样混进重连耗时（B13 未点名的第二处）。
- 口径决定：发送滞后是「负载源自己到顶」的信号（设计稿 §2.2 第 2 条），重连是对端关连接的事件（另计 `reconnects`）⇒ 每个请求只在第一次写出时计 `written` 与 `send_lag`，`sent_rel_ns` 保持第一次写出的时刻，与直方图一致。代价：被重发的请求 `first_from_sent`/`done_from_sent` 含重连耗时（README 写明）。
- 20:42 已改 loadgen.py(_write 只在首次写出计数；on_lost 收到半个响应头不重发)、_proc.py(on_ready OSError)、_util.py(删 pass_site)、README；新增 tests/capload/test_retry_counts.py。下一步：先跑新测试（预计 <1 分钟），再用修前代码证明会红。
- 新测试修后：`pytest tests/capload/test_retry_counts.py` → 3 passed（13.9 s，rc=0）。
- 修前（`git archive HEAD` 导出到 _local/fx-load/prefix/ 再放入新测试）：3 failed —— silent：`self.written` 21 ≠ planned 12；partial：reconnects 6 ≠ 0；on_ready OSError：nullsse 30 s 不退出（回调异常只被 loop 记日志）。残留进程 pgrep 为空。
- 下一步：逐个跑 tests/capload、tests/nullsse 全目录（预计各 1–3 分钟）。
- 全目录：tests/capload 29 passed（73.9 s，rc=0）；tests/nullsse 4 passed（10.9 s，rc=0）。

## 改动清单
- `tools/capload/loadgen.py` `_write`：只在 `r.sent < 0`（第一次写出）时设 `r.sent`、计 `self.written`、加 `self.send_lag` 样本。（K24）
- `tools/capload/loadgen.py` `on_lost` 重连条件加 `not r.rbuf`：收到过不完整响应头的请求不再重发。（RS6 额外扫出 X1）
- `tools/capload/_proc.py` `ParentRunner`：`on_ready` 抛 `OSError` 也走「记错误 + hard stop 全部 worker」，并把失败码并进父进程退出码（原来 `on_ready` 失败时父进程照样退 0/5）。（RS4/RS3 额外扫出 X2）
- `tools/capload/_util.py`：删无人调用的 `pass_site`。（RS12 顺手）
- `tools/capload/README.md` 连接池一条：补「收到字节不重发；重发不计 written/发送滞后，sent_rel_ns 为第一次写出」。
- 新增 `tests/capload/test_retry_counts.py`：3 个用例，见上。

## RS6/RS3/RS4 扫 tools/capload/**（逐计数核对）
- 恰好一次（读码核对）：`self.dispatched`、`backlog.inflight`（+1 `_send_row`，−1 只在 `_finish` 的 `inflight.pop` 成功后）、`self.ok/failed`、`rec_req` 行、`self.conn_new`/`new_conns`（每条连接 `on_made` 一次）、`self.conn_reconnect`/`reconnects`（`r.retried` 保证每请求最多一次）、`self.conn_peer_closed_idle`、`self.conn_fail`、`pool_reset_closed`、`unsolicited`、`rec_conn`（`on_lost` 或 teardown 二选一，`dumped` 互斥）；超时、hard stop（`_abort_all`）、宽限到期、对端关闭各路径都只经 `_finish` 一次。
- sink.py：`requests` 只在 `not conn.closing` 时计，空闲关闭与请求竞态时请求不计、客户端重发后只计一次；`streams_completed/aborted/unfinished_at_stop` 由 `_end_stream` 的 `st in self.streams` 守卫保证一次；`pending_msgs` 出队与收尾各减一次。
- 观察未改（不是计数重复，无消费者）：① 硬停后才建好的连接被 `_release` 关闭时 close_reason 记成 `closed_after_response`（n_requests=0）；② 极少见的「新建连接在 `_connect` 恢复前就被对端关掉」会被记成 `peer_closed_idle`（INFERRED，asyncio 回调顺序下难以出现）；③ `reconnects` 是 `new_conns` 的子集（重连那条新连接两边都计），README 未写明子集关系。

## capgrpc（tools/capgrpc/**）只读扫——交主会话，FX-load 未改
- G1（RS4/RS3，INFERRED，NOT RUN）`loader.py:376-470` `LoadWorker.one()`：只捕获 `grpc.aio.AioRpcError` 与 `asyncio.CancelledError`；其他异常（如 `grpc.aio.UsageError`/`InternalError`、记录数组之外的意外异常）逃出时 `finally` 只回收 `backlog.active_streams`，`backlog.inflight` 不减、`ok/failed` 不计、`requests` 不落行 ⇒ inflight 永久 +1，清空门可能一直不满足，`sent ≠ ok+failed`。建议：加 `except Exception` 归为 `internal_error` 失败，统一在 finally 后记账。
- G2（RS4，INFERRED，NOT RUN）`common.py:441-` `run_supervisor`：`on_all_ready`（nullgrpc 写端口文件）或 `write_ready_file` 抛 `OSError`/`CapkitError` 时异常直接冒出 `supervise`，不 `relay_stop`；子进程是非 daemon 的 spawn 进程，父进程退出时 multiprocessing atexit 会 join 它们 ⇒ nullgrpc（不会自停）永远挂住、capgrpc.load 跑完整张表才退。与本轮在 capload `_proc.py` 修的 X2 同类。建议：包住这两步，失败时 `relay_stop(True)` 并把失败码并入退出码。
- G3（口径，非计数重复）`loader.py:412-419` `recv_types`：`bytes` 只在每条流第一条消息计一次（按流），非 bytes 类型按消息计且立即 break，与模块 docstring「every received message's type … counted per type」不符。建议改 docstring 或改成逐消息计数。
- 核对为恰好一次：capgrpc 无应用层重试（gRPC 断连直接 UNAVAILABLE 失败，不重发）；`self.sent`/`lag.send`/`sent_by_tier` 在 `one()` 开头各一次；硬停时未开始执行的任务不计 sent、不落行（与 not_sent 一致）。nullgrpc `behavior` 的 accepted/active/completed/truncated/cancelled 与 `pending_msgs` 由 try/finally 保证各一次。

## 需要他方或主会话配合
- INTERFACES 勘误（主会话改）：§5.4 `requests` 表 `sent_rel_ns` 一格，原文「`sent_rel_ns q`（实际写出请求的时刻）」改为「`sent_rel_ns q`（实际写出请求的时刻；keep-alive 重连重发时仍为第一次写出的时刻）」。
- capgrpc G1/G2/G3 由主会话决定是否派人修（不在 FX-load 独占范围）。

## MEASURED / INFERRED / NOT RUN
- MEASURED：新测试修后 3 passed / 修前 3 failed（数值见上）；tests/capload 29 passed rc=0；tests/nullsse 4 passed rc=0。
- MEASURED（修前，_local/fx-load/prefix 里去掉前置断言后单跑）：`self.written`=21、`self.send_lag.n`=21、`self.send_lag.max`=502 ms（含 500 ms 重连）、planned=12；requests.capbin 的 sent−planned 断言同样失败 ⇒ C1 T3 send_lag_p99 的输入确被重连耗时污染。
- INFERRED：capgrpc G1/G2 读码推断。
- NOT RUN：capgrpc 任何测试；集群相关一律未碰。

## capgrpc/capselfcheck（主会话裁定扩大范围：G1、G2 修，G3 统一口径；扫 RS3/RS4/RS6）
- 20:47 开工：先读 tests/grpc 与 tools/capselfcheck。
- 20:49 写了 tests/grpc/test_failure_paths.py（G1 故障注入、G2 nullgrpc 监督/进程内、relay 端口文件失败），先对未改代码跑看哪些红。
- 20:54 G1/G2/G3 已修；额外：relay _pump/建上游调用异常记账（R1），nullgrpc request_types/rejects 与 relay upstream_failures 改为逐档（R2：原为整进程累计、每档 summary 重复写，跨档求和多计）。下一步跑新测试与修前对照。
- 20:55 新测试修后 7 passed；修前 5 failed / 2 passed（进程内 nullgrpc、relay 端口文件失败修前就会非零退出，非缺陷，保留作回归）。下一步跑 tests/grpc 全目录（预计数分钟）。

### 核实与改动（capgrpc/capselfcheck）
- G1 属实（修前复现）：故障注入让第 2 次 `read()` 抛 `RuntimeError`，修前 load summary `failed`=0、sent=10，没有 requests 行，inflight 卡住。修：`capgrpc/loader.py` `one()` 加 `except Exception` → `client_error`（只在第一次出现时打印），之后统一记账。测试 `test_loader_client_fault_counted_once`。
- G2 属实（修前复现）：nullgrpc 监督模式端口文件写不了 → `NotADirectoryError` 冒出 `supervise`，30 s 不退出。修：`capgrpc/common.py` `run_supervisor` 把 `on_all_ready`+`write_ready_file` 包进 try，`CapkitError`/`OSError` 时记 fails(-1, 码) 并 `relay_stop(True)`。测试 `test_nullgrpc_port_file_failure_exits_nonzero[supervisor]`。进程内 nullgrpc、relay 同一路径修前已非零退出（不挂），用例保留作回归，修前不红。
- G3 属实：按 README 与 test_chain 已用的口径（每条流只计首条消息的类型）统一——改 `loader.py` docstring；非 bytes 分支只在 `nmsg == 0` 时计类型（修前「先 bytes 后非 bytes」的流会被计两次：修前实测 {'bytes': 10, 'str': 10}）。理由：README、现有测试 test_chain 都按「每流首条」写，改口径要动两处，改实现只动一处且与已有约定一致。测试 `test_loader_recv_types_counts_first_message_per_stream`。
- 额外 R1（RS4/RS3，同 G1 类）`capselfcheck/relay.py`：`_pump` 遇到非 gRPC 异常时 code 为 None → `upstream_failures` 记成 `grpc_None`、`context.abort(None)`；建上游调用（`call_fn`）在 try 之外，抛异常时 accepted/active 已加、没有任何结局计数、active 永久 +1。修：`_pump` 加 `except Exception` → UNKNOWN + details；`call_fn` 挪进 try，失败计 `streams_failed`。测试 `test_relay_fault_counts_each_stream_once[call|read]`，修前两个都红（call：streams_failed 0≠5；read：{'grpc_None': 10}≠{'grpc_UNKNOWN': 5}）。
- 额外 R2（恰好一次，逐档自报）：nullgrpc 的 `request_types`、`rejects` 与 relay 的 `upstream_failures` 是整进程累计的 dict，却写进每一档 summary ⇒ 跨档求和会把同一事件计 N 次（修前实测 sink request_types 跨 2 档 {'bytes': 20}，实际 10 个请求）。修：改成逐档（nullgrpc 按事件档，与 `streams_rejected` 同；relay 按计划档，与 `streams_*` 同），README 写明。断言加在上面两组用例里。
- README：capgrpc（`client_error`）、nullgrpc（两项按事件档）、capselfcheck（中继自身出错计 streams_failed；upstream_failures 按计划档）。

### 扫描观察（未改）
- nullgrpc/relay 在 `server.stop(grace)` 期间收到第二个信号不会缩短宽限期（受 `--stop-grace-s` 限制，有上界）。
- `run_supervisor` 轮询循环里其他意外异常仍不会 relay_stop（读码推断，概率低）。
- loader `plan.build` 在 try 之外：理论上抛异常时请求不计 sent、不落行（与 not_sent 一致）；模板在原点前已全量校验。
- relay 读上游时自身出错，记在 `upstream_failures.grpc_UNKNOWN`（键名叫 upstream，实为中继侧）；README 已注明下游收到 UNKNOWN。

### 测试
- tests/grpc/test_failure_paths.py：修后 7 passed（rc=0）；修前（_local/fx-load/prefix）5 failed、2 passed。
- tests/grpc 全目录：32 passed（109.7 s，rc=0）。
- 残留进程：无。fakebin.log 0 字节。

### 需要主会话
- INTERFACES 勘误：§5.4.1 失败原因表新增一行「| `client_error` | 负载器自身在请求处理中抛出的非协议异常（如 gRPC 库用法错误） | `unclassified` |」。
