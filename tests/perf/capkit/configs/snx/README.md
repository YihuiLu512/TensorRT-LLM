# configs/snx：S-NX 实验（E2b、E7、E8、E9）的场景模板

由 `configs/gen_snx_configs.py` 生成（`python configs/gen_snx_configs.py <新目录>`，再与本目录 diff）。每个臂三个文件：`.scenario.json`、`.tiers.template.json`、`.values.todo.json`；TBD 值在登记时按 `caplaunch.tiergen` 填。

| 实验 | 臂 | 拓扑 |
|---|---|---|
| E2b | e2b-sa-{mc,md,r}、e2b-px-mc | `capstubs.nx_role` standalone（无 IPC、自排空）；proxy（产品 proxy ↔ ZMQ ↔ NullWorker，原始字节排空） |
| E7 | e7-raw-k2-{mc,md,r}、e7-raw-k1-mc、e7-deser-k2-mc | `capstubs.nx_role` proxy，K 条活跃 lane（K=1 为 tcp PAIR 单列）；deser = 隔离检验 (b) |
| E8 | e8-k4-{mc,md,r}、e8-k1-mc | `capgrpc.load`（K 条连接）→ `capstubs.serve_nx` 宿主（`--grpc --grpc-protocol openengine`）↔ ZMQ ↔ NullWorker |
| E9 | e9-{o,n}{1,8}f0-{mc,r}、e9-{o,n}8fP-mc | O-NX：`capload` → `serve_nx` 原架构（n 前端、池 w）；N-NX：`capload` → `serve_nx` 包装的新架构前端 → gRPC → `serve_nx` 宿主 |

要点：
- S-NX 角色自己读档位表与起点（`--tier-table {input:tier_table} --origin-ns {origin_ns}`），NullExecutor 的步网格 = 档位表的节拍表，每步 token 数与 s 取自档位（`tier_shape`）。`nx_stream_interval` 必须等于档位的 s（NullWorker 交叉核对，不等即拒绝起动）。
- 门：E2b/E7 用 `nx` 的 `backlog.inflight / engine_open / pending_keys`；E8 用读者在途；E9 用 capload 在途。
- 对账链只有 requests 一条（负载方 `summary.counts.ok`）；S-NX 自报（`{raw}/snx/snx.<pid>.*`）由 `python -m capstubs.null_executor summarize` 离线汇总，judge 不读。
- 产品角色 P0：不写快照，C10 对产品进程 NOT_RUN（与 E0 补充冒烟相同）；MPI worker、附属前端是未登记的子进程（C13 会列为 unattributed）。
