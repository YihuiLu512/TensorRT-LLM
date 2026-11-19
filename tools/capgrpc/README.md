# capgrpc.load（B2：gRPC 负载器 / 排空端）

入口是 `python -m capgrpc.load`（INTERFACES.md §11；实现在 `capgrpc/loader.py`，`load.py` 只是入口，这样多进程子进程按稳定的模块名导入）。只依赖标准库、`grpcio`、`capcommon`；`--loop uvloop` 时才导入 uvloop，导入失败退出码 3。

## 入口

| 入口 | 作用 |
|---|---|
| `python -m capgrpc.load` | 多进程 `grpc.aio` 开环负载器；`--stamped` 时作排空端 |
| `python -m capgrpc.reqbytes raw …` / `openengine …` | 生成请求字节文件（`capkit.reqbytes/1`）；`openengine` 子命令只在装了产品包的容器里能用，导入失败退出码 3 |
| `python -m capgrpc.oe_sink` | 经产品 OpenEngine 客户端类读流的开环驱动（设计稿 E3 卡 3b 的 `oe_sink.py`；INTERFACES 勘误 FT1-8～FT1-12），见文末一节 |
| `capgrpc.common` | 三个 gRPC 工具共用的运行时（原始字节的通用处理器、通道、端口预留、`ToolRuntime`、多进程监督者）；`capgrpc.nullgrpc`、`capselfcheck.relay` 都 import 它 |

## 参数（§11 之外新增的都是可选参数）

§11 的必需参数全部支持：通用（身份、`--run-dir`、`--stop-grace-s`、`--cpus`）、快照、档位、`--loop`、`--ready-file`、`--inject`、`--target`｜`--target-port-file`、`--method`、`--workers`、`--channels-per-worker`、`--schedule`、`--request-format`、`--msg-sample-mod`、`--lag-period-us`、`--max-records-per-worker`、`--stamped`。

| 新增参数 | 含义 |
|---|---|
| `--in-process` | worker 就在本进程里跑（要求 `--workers 1`）。**弱读者模式** = `--in-process --workers 1 --channels-per-worker 1` |
| `--shared-subchannel-pool` | **只作对照**：不设 `grpc.use_local_subchannel_pool`，同一 worker 的多条 channel 共用一条 TCP 连接 |
| `--prompts PATH` | `capkit` 格式必需（prompt 池，§4.3）；token id 编码为 int32 小端，文本为 UTF-8 |
| `--request-bytes PATH` | `openengine` / `raw` 格式必需（`reqbytes` 生成的文件） |
| `--max-msg-records-per-worker N` | `msgs` 记录容量；`capkit` 格式下不给就按时间表与 `expect.msgs_per_request` 精确算出 |
| `--lag-records-capacity N` | 事件循环延迟的逐样本记录（缺省 0 = 只有直方图） |
| `--planned-msgs` | 不盖戳的流（S-PB、S-NL）也算消息层滞后：计划时刻由时间表推出（`capgrpc.msgplan`，勘误 FT1-9）；与 `--stamped` 互斥，`raw` 格式不可用，要给 `--max-msg-records-per-worker` |
| `--consume raw\|client` | 读端的消费方式：`raw`（缺省，原样）为原始字节读流；`client` 时每个 worker 用产品 `OpenEngineClient.generate` → `GenerateStream` 读流（protobuf 请求与响应、每条消息一个读任务、逐条流契约校验、终止事件后的收尾读），实现为 `capgrpc.oe_sink.LoadClientWorker`；只能配 `--request-format openengine`、`--channels-per-worker 1`，与 `--stamped` 互斥；不计字节；summary 多 `consume`、`imported` 两键（PLAN-R2R5 第 2 节；INTERFACES「S-NX 线 §2」） |

`--request-format`：`capkit`（`wire.GRPC_REQ` 头 + prompt 字节，每请求现拼头部）、`openengine`（请求字节文件里的 `GenerateRequest` 模板前面拼上 `request_id` 字段：protobuf 解析把串接的序列化合并，所以发送路径上不建 protobuf 对象）、`raw`（新增取值：每请求同一段字节，没有请求号）。请求超时只读档位表的 `request_timeout_ns`，**从计划时刻起算**（发送时剩余时间 < 1 ms 时按 1 ms 发）；超时记 `timeout_registered`。请求处理中抛出的非 gRPC 异常（客户端自身故障）记 `client_error`（`class_hint = unclassified`），首次出现打印到标准错误。

## 进程结构

- 缺省：一个监督者进程（`spawn` 出 W 个 worker 子进程）。监督者用同一种事件循环跑（所以它的版本自报里也有 `loop_impl`），也写快照（全部槽为 0，外加 `self.children_alive`），等全部 worker 在原点之前就绪后写就绪文件（`children` 列出每个 worker 的 pid）。停止请求经管道下发（不转发信号，避免被编排器直接发了信号的子进程把转发当成「第二次信号」）；父进程消失时子进程按管道 EOF 优雅停止。
- **场景配置的 `expected_procs` = W + 1**（`--in-process` 时为 1）。
- `--cpus`：第 k 个 worker 绑第 k 项；监督者绑前 W 项的并集。

## 注入位点（`SITES`）

| 位点 | 每次经过 | 可用模式 |
|---|---|---|
| `send.req` | 每请求，发出调用之前 | `busy`、`sleep` |
| `recv.msg` | 每条收到的消息 | `busy`、`sleep`（`sleep` 可作「排空端暂停读取」的破坏样本） |
| `drop.msg` | 每条收到的消息 | `drop`：被选中的消息不计入任何计数与记录（C7 失败样本） |

## 快照槽（`SLOTS`，启动时一次性登记）

`backlog.inflight`（gauge，已发出未结束的请求）、`backlog.active_streams`（gauge，还没读到流末尾的请求）、`backlog.read`（counter，已读消息累计；与上游的 `backlog.yielded` 相减得 gRPC 在途数）、`self.sent/ok/failed`、`self.children_alive`、`bytes.recv`、`lag.send`（发送滞后）、`lag.msg`（`--stamped` 时的消息层滞后）、`lag.loop`，以及每个位点的 `inject.<site>.hits/applied/applied_ns`。

## 输出（每 worker 每档，`raw/tiers/<档>/drain.w<k>.*` 之类）

- `requests`（`loadgen.requests`）：按计划档归档。`frames` = 收到的消息数，`tokens` 恒为 −1（gRPC 负载器不数 token），`conn_id` = channel 序号，`new_conn` 恒为 0；`meta.status_codes`。
- `msgs`（`drain.msgs`，只在 `--stamped`）：`stream_id % msg_sample_mod == 0` 的流逐消息一行，按收到时刻归档；`meta.msg_sample_mod`（同时写 `chunk_sample_mod` 方便判定器共用代码）。
- `lag`（只在 `--lag-records-capacity > 0`）。
- `summary.json`（`capkit.tier_summary/1`）：`counts`（`planned` 来自时间表表头，`sent`、`ok`、`failed`、`not_sent`）、`failures`、`msgs`（`msgs`、`bytes` 按计划档；`expected_per_request` 与 `ok_requests_with_unexpected_msg_count` 只在 `capkit` 格式）、`send_lag_ns`、`msg_lag_ns`、`read_msgs_event_time`、`recv_types`（**每条消息都检查 `type(msg) is bytes`**；按类型计首条消息，非 bytes 记 `protocol_error`）、`inject`（`inject.tier_report`）、`loop_lag_ns`、`cpu`（进程与事件循环线程的 CPU 增量）、`overflow`、`completed`、`stop`，以及 `n_workers`、`schedule_sha256`、`tier_table_sha256`、`request_timeout_ns`、`channels_per_worker`、`local_subchannel_pool`、`in_process`。对账段可用的源：`summary.counts.ok`、`summary.msgs.msgs`。

## 退出码

0 正常跑完；3 配置错误（含 `--loop uvloop` 而无 uvloop、未知注入位点）；4 原点之前连不上目标；5 档位表走完之前被停止。多进程时监督者按 3 > 4 > 其他非零 > 5 > 0 合并各 worker 的码。

## 与 INTERFACES.md 的偏差

1. 包名：第一批写在 `capload_grpc`、`nullgrpc`、`selfcheck` 三个包里（任务分配的目录）；集成时已按 §1.1 改成 `capgrpc`（含子包 `capgrpc.nullgrpc`）与 `capselfcheck`。
2. `--request-format` 多一个取值 `raw`；`openengine` 第一批就实现了（依赖请求字节文件）。
3. 进程数：监督者 + W 个 worker（`expected_procs = W + 1`）。

## `python -m capgrpc.oe_sink`（E3-3b 的客户端侧驱动）

`LoadWorker` / `RequestPlan` 的子类：时间表、档、发送滞后、`requests` / `msgs` 记录与每档 summary 与 `capgrpc.load` 同结构，判定器当负载角色读。契约见 INTERFACES 勘误 FT1-8～FT1-12，这里只列用法。

- 单进程、一条 channel。读者参数与 `capgrpc.load` 完全相同（`loader.add_reader_args`），另加 `--tier {generate-stream,engine-event-stream}`、`--connect {product,channel}`（必填）、`--control-timeout-s`、`--slow-read-us N`。弱读者对照 = 同一组参数的 `capgrpc.load --in-process --workers 1 --channels-per-worker 1`：场景里换读者只换模块名并去掉这几个参数。
- `--tier generate-stream`：`OpenEngineClient.generate` → `GenerateStream`；`--tier engine-event-stream`：`OpenEngineEngineClient.generate(EngineRequest)`（另加两次 dataclass ↔ protobuf 转换）。
- `--connect product` 走客户端自己的 `connect()`，对端要答 Control（S-PB 前端模式、S-NL）；`--connect channel` 不发 Control 调用，可读只服务 Generate 的 S-PB。
- 产品模块只在容器里有：导入失败退出码 3（「container only」），导入了什么写进就绪文件与每档 summary 的 `imported`。本机测试用 `tests/capstubs/oe_product.py` 把探针树里产品客户端文件原样拼成一个最小的 `tensorrt_llm` 包。
- `recv.msg` 的 `busy` / `sleep` 阻塞整个事件循环；`--slow-read-us` 只让每条流读完一项后 `asyncio.sleep`（E3「读者慢」已知样本）。
