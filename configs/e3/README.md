# configs/e3 — E3 gRPC 一跳（3a 服务端侧、3b 客户端侧）的场景模板

全部 JSON 由 `configs/gen_e2a_e3_e4_configs.py` 生成（约定、模板的含义、档位表形状与 `configs/e2a/README.md` 相同）。依据：快轨计划 v2 第 1 节次序 4、5，第 3 节 E3-3a / E3-3b 两行；设计稿 E3 卡。登记目录按 `04_frontend/E3/register` 写。

## 臂

| 前缀 | 拓扑 | 被测 | 对照臂 |
|---|---|---|---|
| `e3a-snl-kc-<轴>` | `capgrpc.load`（K 个进程、K 条连接）→ 一个 S-NL full | S-NL 进程（真 servicer + `_format_result`，节拍器同循环，是域内桩） | `e3a-snl-k2c-mc`：负载器翻倍；`e3a-snl-kc-mc-vsrv`：在 S-NL 的 `send.msg`（servicer 的循环上）注入 δ，边界应下移；`e3a-snl-kc-mc-vrdr`：在读者 `recv.msg` 注入 δ，边界应不动 |
| `e3a-mono-<轴>` | 只有 S-NL metronome-only（节拍器、NullLlm、队列，无 servicer、无 gRPC 流量） | 节拍器自身的开销（「只有节拍器」空跑档） | 与同轴的 `e3a-snl-kc-<轴>` **用同一份档位值登记**，给「含 / 不含节拍器」两个数；没有负载角色，判定器的 C3、P-SENT 为 NOT_RUN（勘误 FT1-6②） |
| `e3b-gs-<轴>`、`e3b-ees-<轴>` | `capgrpc.oe_sink --tier generate-stream / engine-event-stream --connect channel` ← 一个 S-PB（直连模式，1 核） | 产品客户端库的读路径（`oe_sink`，`under_test`） | `e3b-weak-<轴>`：弱读者检验（同一个 S-PB、同一组读者参数，只把读者换成 `capgrpc.load` 单连接）；`e3b-ees-mc-vrdr`：在 `oe_sink` 的 `recv.msg` 注入 δ（边界应下移）；`e3b-ees-mc-vstub`：在 S-PB 的 `send.msg` 注入 δ（边界应不动） |

轴：`mc`、`md`、`r`，含义见 `configs/e2a/README.md`。E3 问题 (3)（逐档加 C 直到出现非 OK 状态码或新流被推迟，含 SM-19 的挂起流上限实测）用 `e3a-snl-kc-mc` 登记足够大的 C 档即可，不另设模板。阴性（C 小、μ 低时积压读数在零附近）读阶梯的低档。

- `e3b-*` 三类读者的档位表模板逐字节相同（除 `name`），一份档位值登记三臂；场景只差读者的模块名、`--tier` / `--connect` 与 `measure_class`（测试核对）。
- `oe_sink` 用 `--connect channel`：直连模式的 S-PB 不答 Control。第二档要求请求字节是 token id prompt（勘误 FT1-8），所以 prompt 集 `short` 必须是 token id 集。
- 快轨不做的 E3 项（K 轴、长 prompt 混合、3c、纯 Python protobuf 阳性、读者慢 / 服务端慢已知样本）见快轨计划第 4 节，这里没有模板。

## 未核对

同 `configs/e2a/README.md`；另外 `oe_sink` 读真产品包、`metronome-only` 在集群上的节拍器份额都只在集群上能看到。

## S-FE 臂（前端线第 8 项，2026-11-20）

E3 卡「3c 对照」：`e3c-sfe-kc-{mc,r}` = 3a 的 S-NL 换成零改动的 S-FE（对 `e3a-snl-kc-*`，同一套值）；`e3c-ees-snl-{mc,r}` = 3b 的 `oe_sink --tier engine-event-stream` 对端由 S-PB 换成 S-NL full（对 `e3b-ees-*`，同一套值）。S-FE 臂的门与链只读读者。详见 `INTERFACES.md`「前端线第 8 项」。
