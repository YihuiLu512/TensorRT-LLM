# capgrpc.nullgrpc（B2：gRPC 空汇 / 绝对节拍源）

入口是 `python -m capgrpc.nullgrpc`（INTERFACES.md §11）。服务端用通用处理器回应**任意方法路径**（unary_stream，`request_deserializer=None`、`response_serializer=None`），所以没有 `--method` 参数；每个请求对象的类型按类型名计数（`summary.request_types`），首个请求的类型打印到标准错误（U-1 的冒烟读数）。

## 请求与应有消息数

请求必须是 capkit gRPC 请求（`capcommon.wire.GRPC_REQ` 头 + prompt）。每请求的 N、k、s、计划档只取自请求头（§4.4），消息数、每条消息的步偏移与 token 数只由 `capcommon.expect`（`product` 闸门）算。头不合法 → `INVALID_ARGUMENT`，计入 `streams_rejected`（按事件档归档，因为没有可信的计划档）。

## 两种节奏

- `--pacing asap`：逐条 `yield` 预制字节（同一个 bytes 对象，无分配）；可多进程（`--workers Z`，`SO_REUSEPORT`：父进程先用一个只绑定不监听的套接字占住端口，子进程绑同一端口，全部就绪后写端口文件再释放占位套接字）。
- `--pacing beat`（节拍源，要求 `--workers 1`）：全进程一个 `BeatSchedule`；节拍协程在每一步的计划时刻醒来，把到期的消息放进各流的**无界**待发队列，不等任何写完成；流 j 条消息在步 `first_step + msg_step_offsets[j]` 入队（入队时登记下一条）。`first_step = step_at_or_after(接纳时刻)`；若节拍协程已经越过了那一步（只在接纳恰好落在步的计划时刻且节拍已处理时发生），改用下一步并计 `first_step_bumped`。流的步超出档位表末尾 → `OUT_OF_RANGE`，计 `streams_truncated`。`--stamp` 在每条消息头部写 16 字节步号戳 `(step, planned_rel_ns)`（要求 `--msg-bytes ≥ 16`、`--pacing beat`）。

## 参数

§11 的全部参数；新增可选：`--in-process`（本进程直接服务，要求 `--workers 1`）、`--max-stream-records N`（`streams` 容量；缺省 = `count_schedule_rows(档位表)`，即整次启动可能的全部请求数）、`--lag-records-capacity N`、`--max-unrequested-s N`（gRPC 内核的 `grpc.server_max_unrequested_time_in_server`，整秒：调用在「应用还没领取」的队列里等过这么久就被取消；缺省 = 档位表 `request_timeout_ns` 与档位表长度中较长者向上取整，显式值不得短于 `request_timeout_ns`，见 `capgrpc.common.resolve_max_unrequested`）。`steps` 容量 = `BeatSchedule.n_steps`（精确）。**`expected_procs` = Z + 1**（`--in-process` 时为 1）。

## 注入位点

`send.msg`（每条消息交给 gRPC 之前；`busy`/`sleep`）、`beat.step`（每个节拍步；`busy`/`sleep`，用来把节拍源本身拖慢）。

## 快照槽

`backlog.active_streams`、`backlog.pending_msgs`（gauge，待发队列里的消息数）、`backlog.yielded`（counter，已交给 gRPC 的消息累计；与读者的 `backlog.read` 相减得在途数）、`self.accepted/completed/cancelled`、`self.children_alive`、`bytes.sent`、`count.beat_steps`、`count.beat_late_steps`（醒来时已过下一步计划时刻的步数）、`count.beat_enqueued`、`lag.beat_wake`（醒来滞后）、`lag.loop`、`inject.<site>.*`。

## 输出（每档）

- `streams`（`server.streams`）：每接纳一个请求一行，按请求自带的计划档归档；`first_step`（asap 为 −1）、`n_msgs`（实际交出的消息数）、`tokens`（这些消息的 token 数之和）。
- `steps`（`beat.steps`，只在 beat）：每步一行，`msgs` = 本步入队的消息数，按步的计划时刻归档。
- `summary.json`：`counts`（`streams_accepted`、`streams_completed`、`streams_cancelled`、`streams_truncated`、`streams_rejected`、`first_step_bumped`、`msgs_sent`、`bytes_sent`，按计划档）、`beat`（`steps`、`late_steps`、`enqueued`、`max_pending`、`wake_lag_ns`，按事件档）、`yielded_event_time`、`grpc_server_options`（实际设给 gRPC 的服务端选项与来源，就绪文件每个 worker 的 `info` 里也有）、`request_types`、`rejects`（这两项按事件档，各档只含本档的数）、`inject`、`loop_lag_ns`、`cpu`、`overflow`。对账段可用的源：`summary.counts.streams_accepted`、`summary.counts.msgs_sent`。

服务端只在收到信号时退出（`--stop-grace-s` 内收尾，之后取消仍在发的流，计 `streams_cancelled`），退出码 0；配置错误 3；绑定失败 4。
