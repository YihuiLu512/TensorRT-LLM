# configs/glm：E10-S（E10 替身臂 G0、G1）的场景模板

由 `configs/gen_glm_configs.py` 生成（`python configs/gen_glm_configs.py <新目录>`，再与本目录 diff）。每个臂三个文件：`.scenario.json`、`.tiers.template.json`、`.values.todo.json`；TBD 值在登记时按 `caplaunch.tiergen` 填。

| 臂 | 拓扑 |
|---|---|
| e10-g0-{mc,r} | `capgrpc.load`（4 条连接，`--planned-msgs`）→ `capstubs.serve_glm --engine mock`：glm 自带 `MockEngine`（G0） |
| e10-g1-{mc,r} | 同上，`--engine g1`：`capstubs.glm_g1.G1Engine`（G1：绝对节拍、每步一次入队） |
| e10-{g0,g1}-{mc,r}-vload | 与对应臂同一场景；档位表 `launch_inject` 在读者（`capgrpc.load`）的 `recv.msg` 位点注入忙等 `TBD:delta_ns`（每条消息、每个 worker 一次）：设计卡 E10 的负载源一侧 V 对照，预期边界不动 |
| e10-{g0,g1}-{mc,r}-k8 | 与对应臂相同，只把负载源翻倍：`capgrpc.load` 8 条连接（8 个 worker、8 个核）；设计卡 E10 的「负载源翻倍」对照，预期边界不动 |

要点：
- 实验编号 `E10-S`（`capcommon.policy.NO_MODEL_EXPERIMENTS`）；裸 `E10` 与真引擎臂不在白名单里（Q6 不做）。
- 被测进程由 glm 的 venv Python 起（`{param:glm_python}`，在 `/home/scratch.yihlu_coreai_1/repro-env/j17740-oe-eda0ca6/venv/`，`include-system-site-packages`，容器内 Python 3.12）；capkit 包经覆盖清单装进容器的 site-packages，venv 看得到。glm 检出 `eda0ca6` 一字不改，替身与探针都是 `serve_glm` 在运行时套上的。
- 替身按固定步长出响应：`glm_step_ms` 必须等于档位表各负载档的 `beat_ns`，否则 `serve_glm` 拒绝起动。mc 档 5 ms、r 档 1 ms（登记时定）。
- 探针层：`--inject off|M1:count|M1:busy:NS`（M1 = `servicer._token_output`，每条 token 消息一次）；`--diag` 每秒一行 `{raw}/snx/serve_glm.<pid>.jsonl`（派发批大小、事件循环滞后、替身步统计、在途请求、线程表、gc）；`capstubs.procmon` 写 `{raw}/snx/procmon.host.jsonl`（逐线程 CPU 与运行队列等待）。
- 门：读者在途与活跃流归零；对账链只有 requests 一条（读者 `summary.counts.ok`）。
- 产品角色 P0：不写快照，C10 对产品进程 NOT_RUN。
