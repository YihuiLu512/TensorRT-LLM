# capselfcheck.relay（B2：已知瓶颈中继）

入口是 `python -m capselfcheck.relay`（INTERFACES.md §11）。单进程（`expected_procs = 1`，`--cpus` 恰好一个核），`--loop {asyncio,uvloop}`。

## 行为

服务任意方法路径（原始字节），对每个下游调用用同一方法、同一请求字节调上游（`--upstream HOST:PORT` 或 `--upstream-port-file`，一条 channel，原点之前连通，否则退出码 4）。每条上游消息在**读上游的任务里**（事件循环上）依次：计 `backlog.read` → 经过位点 `relay.msg`（已知的额外工作）→ 放进本流的队列；处理器从队列取出交给下游（`backlog.yielded`）。所以中继是瓶颈时积压出现在中继的上游（上游 `yielded` − 中继 `read`），中继内部队列 `backlog.queue_msgs` 只在下游读得慢时增长。上游非 OK 状态原样转给下游（计 `streams_failed`）。

## 四种已知答案模式（都用注入表达，§11）

| E0 场景 | 注入 |
|---|---|
| 1 单级忙等 | `site=relay.msg,mode=busy,delta_ns=δ` |
| 2 两级串联 | 两个中继进程首尾相连，角色名 `relay1`、`relay2`；档位表里按角色各给一条规格，互换 δ = 互换两条规格的 `role` |
| 3 忙等在另一条线程上 | `site=relay.msg,mode=busy,delta_ns=δ,where=aux_thread`（辅助线程启动时预建，名字写进就绪文件 `info.threads`） |
| 4 不耗 CPU 的阻塞 | `site=relay.msg,mode=sleep,delta_ns=δ` |

## 快照槽

`backlog.queue_msgs`、`backlog.active_streams`（gauge）；`backlog.read`（已从上游读的消息）、`backlog.yielded`（已交给下游的消息）（counter）；`self.accepted/completed/failed/cancelled`、`bytes.forwarded`、`lag.relay_service`（每条消息在第 2 步花的时间，事件循环上量）、`lag.relay_hold`（读到 → 交给下游）、`lag.loop`、`inject.relay.msg.*`。

## 输出（每档）

`summary.json`：`counts`（`streams_*`、`msgs_received`、`msgs_forwarded`、`bytes_forwarded`、`requests_without_capkit_header`，按请求头里的计划档；没有 capkit 头的请求按事件档）、`event_time`（`read`、`yielded`、`service_ns`、`hold_ns`）、`upstream_failures`、`inject`、`loop_lag_ns`、`cpu`（`process_ns` 与 `loop_thread_ns`：场景 3 时两者之差就是辅助线程的忙等）、`overflow`；`lag`（可选）。只在收到信号时退出，退出码 0。
