# configs/e2a — E2a 桩标定（S-PB、S-NL）的场景模板

全部 JSON 由 `configs/gen_e2a_e3_e4_configs.py` 生成（同时生成 `configs/e3`、`configs/e4`；改配置请改生成器，生成到新目录后与本目录 diff；`tests/launch/test_e2a_e3_e4_configs.py` 核对两者逐字节相同）。依据：快轨计划 v2 第 1 节次序 3、第 3 节 E2a 一行；设计稿 E2 卡（E2a 部分）；RUNBOOK-stage2 §1.5。登记目录按 `03_calib/E2a/register` 写。

**这些都是模板**：档位表里的数（速率、在途流数 C、节拍 Δ、输出长度 N、注入 δ、窗口、门参数）都是 `TBD:<名字>`，登记时按 E1 / E2a 的结果填（填 `*.values.todo.json` → `python -m caplaunch.tiergen`，流程同 `configs/e0/README.md`）。进程数、核数是生成器里的常量，改它们要重新生成。

## 臂的命名

`e2a-<桩>-<连接形态>-<轴>[-<对照>]`，每个臂一组 `scenario` + `tiers.template` + `values.todo`。

| 部分 | 取值 | 含义 |
|---|---|---|
| 桩 | `spb`、`snl` | S-PB 直连模式（`capstubs.prebaked_server --schedule`）；S-NL full（`capstubs.null_llm_server --mode full --schedule`，产品 server 模块用缺省的 `tensorrt_llm.grpc.openengine.server`）。桩都是 1 个进程绑 1 核，`measure_class` 为 `under_test` |
| 连接形态 | `1c`、`kc`、`k2c` | 读者 `capgrpc.load`：`1c` = 单进程、单 channel（`--in-process`，E3-3b / E4 前端看到的形态，也是弱读者检验的参照）；`kc` = K 个 worker 进程各一条 channel（K 条连接，E3-3a 的形态）；`k2c` = 2K（负载器翻倍，C12） |
| 轴 | `mc`、`md`、`r` | `mc`：M 轴固定 Δ 加 C；`md`：M 轴固定 C 缩 Δ；`r`：R 轴最短输出（`max_tokens` 1）加 λ |
| 对照 | `vrdr`、`vstub` | 启动级注入一个 busy δ：`vrdr` 在读者的 `recv.msg`（单连接形态判读者是否先到顶）；`vstub` 在桩的 `send.msg`（V：边界应随 δ 下移） |

## 共同约定

- **读者**：所有臂的读者角色都叫 `reader`，`--request-format openengine` + 作业输入 `request_bytes`（`capgrpc.reqbytes openengine`，真 `generation_pb2`），`--planned-msgs`（S-PB / S-NL 的消息不带步号戳，勘误 FT1-9），`--method` 写死为产品的 Generate 路径。
- **作业输入**：S-PB 臂另有 `prebaked`（`capstubs.prebaked`，真 `generation_pb2`、`--tier-table {input:tier_table}`）。两类请求字节与 S-NL 的 `--model` 用同一个参数 `model_name`（真 servicer 接受任何非空名字）。
- **对账**：`requests` 链（读者 `counts.ok` ↔ 桩 `streams_accepted`）与 `messages` 链（桩 `msgs_sent` / `msgs_yielded` ↔ 读者 `msgs.msgs`，两跳各加 1 条 finished），与 FT1-sink 的作业路径 e2e 相同。
- **清空门**：gRPC 在途数（桩 `backlog.yielded` − 读者 `backlog.read`）、桩待发队列与活跃流、读者活跃流与在途请求，全部归零。
- **档位表**：一个暖机档 + 8 个成倍加压的 `up` 档（之间都有带门的 drain）；第一个 `up` 档是基线。二分是另一次启动。`r` 按 Δ 至少登记两次（设计稿 §2.1「R 轴的 Δ」）；`mc` / `md` 按输出长度 N 登记两次（§2.1，两个 N 的上限不同就分开报）。`md` 每一档（含其后的 drain）的时长都必须是该档 Δ 的整数倍。
- 观察者 `sidecar`、`canary` 与 E0 / E1 相同。

## 未核对

本机只做了结构校验、档位表展开与 `cross_check`（无进程）。容器里的真实运行（真 pb2、真 S-NL 产品模块）属集群冒烟与 E2a 本身。K（`kc` 的进程数）是生成器常量，应按 E1-gRPC 的每 worker 健康带核对后再登记。

## S-FE 臂（前端线第 8 项，2026-11-20）

`e2a-sfe-1c-mc`、`e2a-sfe-1c-r`：`capgrpc.load` 单连接 → 一个零改动的 S-FE（`fake_openengine.py`，直连形态、无指纹，读者按 `{addr:sfe}` 连），与 `e2a-spb-1c-*` 同一模板形状、同作业对照。S-FE 每条流各自睡 `step_delay`，没有 md 臂；S-FE 不自报，门与链只读读者。S-FE 的多连接行是 `configs/e3` 的 `e3c-sfe-kc-*`。详见 `INTERFACES.md`「前端线第 8 项」。
