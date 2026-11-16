# configs/e6 — E6 多前端扩展，臂 (iii)：n 个互相独立的单前端链

全部 JSON 由 `configs/gen_e6_configs.py` 生成（重生成到新目录再比对；测试 `tests/launch/test_e6_configs.py`）。依据：设计稿 E6 卡（判断「是否成比例」只用臂 (iii)）；`00_plan/REPLAN-20261116.md` 覆盖轮（n ∈ {2, 4}，R 轴与 M 轴 N = 33，每点 1 次启动）。登记目录按 `04_frontend/E6/register` 写。

## 拓扑

n 份互不相连的 E4-4b 主档链：`capload<i>` → 真 `trtllm-serve` 新架构单前端 `frontend<i>`（`--num_serve_frontends 1 --num_postprocess_workers 0 --num_input_processor_workers 8`，11 核，被测）→ `spb<i>`（S-PB 前端模式，1 核）。

- 每条链的三个角色只引用自己的地址：`frontend<i>` 的 `--openengine-endpoint {addr:spb<i>}`，`capload<i>` 的 `--target {addr:frontend<i>}`。没有共享的桩、没有多前端机制，也就没有连接落点问题；每前端的请求数就是 `capload<i>` 的 `counts.ok`，对账链一对一条（`requests<i>`：`capload<i> counts.ok` ↔ `spb<i> streams_accepted`）。
- 每条链的前端、桩、负载器参数与 `configs/e4` 的 `e4b-mc` / `e4b-r` 逐项相同（只换角色名），档位表模板的档、轴、公共字段也相同（测试逐项比对）。
- **档位是每前端的负载**：n 个 `capload<i>` 读同一份登记的时间表与 prompt，组负载 = n × 档位值；n 条到达序列在时间上完全一致（同一个到达种子）。
- **核数**：`capload<i>` 的 worker 数 n = 2 时 2、n = 4 时 1（目标机一个 NUMA 节点 56 个物理核，n = 4 共 4 × (11 + 1 + 1) + 2 = 54）。worker 数按 E1 的 capload 单 worker 健康带与 E4 实测（6400 msgs/s 时 4 个 worker 每进程约 0.04 核）定，n 不同 worker 数不同，负载源翻倍的对照留给细化轮。
- 清空门：`loadgen_inflight`（全部 `capload<i>` 之和）、`spb_pending`、`spb_active`（全部 `spb<i>` 之和）。

## 臂

| 臂 | 用途 |
|---|---|
| `e6-iii-n2-mc`、`e6-iii-n4-mc` | M 轴（与 `e4b-mc` 同形状，登记 N = 33） |
| `e6-iii-n2-r`、`e6-iii-n4-r` | R 轴（与 `e4b-r` 同形状，登记 beat 1 ms） |

## 未核对

- 两个以上真 `trtllm-serve` 进程同时在一个 launch 里起：编排器按角色逐个处理（本地 e2e 用假产品两条链通过），真前端未在作业里跑过。
- 同步到达对并行前端的影响（每条链各占各的核，推断影响小，未测）。
