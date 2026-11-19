# configs/e4 — E4 单前端（4b 主臂、4b′ 换桩对照）的场景模板

全部 JSON 由 `configs/gen_e2a_e3_e4_configs.py` 生成（模板的含义、档位表形状与 `configs/e2a/README.md` 相同）。依据：快轨计划 v2 第 1 节次序 6、第 3 节 E4 一行；设计稿 E4 卡。登记目录按 `04_frontend/E4/register` 写。

## 拓扑

`capload`（W 个 worker）→ 真 `trtllm-serve` 新架构前端（`--num_serve_frontends 1 --num_postprocess_workers 0`，被测）→ **一个**桩进程（1 核）。前端角色照 E0 的写法（`configs/e0` 的 `e0a-frontend-sfe`：`{param:trtllm_serve} {param:tokenizer_dir} --report_addr … --openengine-endpoint {addr:<桩>}`，探针档 P0），另加 `--num_input_processor_workers 8`（产品缺省值，显式写出，使线程池大小成为写明的条件）。

- **绑核（E4 卡主档）**：前端的核数 = 1（事件循环）+ 8（输入处理线程池）+ 2（grpcio 自己的线程，数目未核对）。敏感性档（整进程 1 核）不在快轨内。
- **tokenizer**：`tokenizer_dir` 填已拷好的 GLM-5.3-Flash 副本 `01_code/configs/tokenizers/GLM-5.3-Flash`（不含权重；该目录被 git 忽略，按绝对路径给）。前端的 served model name 是这个目录，`capload --model`、S-PB 的 `--model`、S-NL 的 `--model` 都用同一个参数。
- **4b（`e4b-*`）**：S-PB 前端模式（Control + Health 来自 S-FE 模块，`--sfe-module`、`--max-context-length` 与 `configs/e0s` 的 `e0s-spb-frontend` 相同）；作业输入 `fingerprint`（tokenizer 设置固定为前端 argv 隐含的 `trust_remote_code false`、`tokenizer_mode auto`）与 `prebaked`。
- **4b′（`e4bp-*`）**：S-NL full，产品 server 模块用缺省，`--tokenizer {param:tokenizer_dir} --trust-remote-code false --tokenizer-mode auto`（与前端相同，S-NL 自己用产品的 loader 加载，Control 据此算指纹）；没有作业输入。
- **桩不带 `--schedule`**：前端发给桩的 `request_id` 不是时间表的 `req_id`，所以流按到达时刻归档（勘误 FT1-1、B2B-8 ⑤）。
- **prompt 集 `tid128`**：token id、长度 128（E4 卡 R 轴：token-id、L = 128）；M 轴也用它（短 L）。登记时由 GLM 副本的词表生成，集名必须是 `tid128`。
- **对账**：只有 `requests` 链（`capload counts.ok` ↔ 桩 `streams_accepted`）；P0 的前端不发布计数，SSE 块与桩消息的对应关系未定（INTERFACES §12 U-6），不设 `messages` 链。清空门读 `capload backlog.inflight` 与桩的待发队列、活跃流。

## 臂

| 臂 | 用途 |
|---|---|
| `e4b-<轴>`、`e4bp-<轴>` | 4b 主臂与 4b′ 换桩对照，轴 `mc`、`md`、`r`（含义见 `configs/e2a/README.md`） |
| `e4b-nb-mc`、`e4b-nb-r` | E4 卡的第一步：粗扫找到平台后，在平台下方最近的稳定档上重复启动，量真前端的近边界噪声（模板只有 1 个 `up` 档） |
| `e4b-weak-<轴>`、`e4bp-weak-<轴>` | 弱读者检验：同一个桩（与对应前端臂逐字段相同的角色），读者换成 `capgrpc.load` 单连接；按前端的边界档登记。这些臂另有作业输入 `request_bytes`（`--model {param:tokenizer_dir}`）与 `messages` 链 |
| `e4b-w2-mc` | 负载源翻倍（`capload` 2W） |
| `e4b-mc-vstub`、`e4b-mc-vload` | V 的「另一端注入不动」：δ 注在 S-PB 的 `send.msg`、`capload` 的 `recv.chunk` |
| `e4b-mc-vf2`、`e4b-mc-vf4` | V 的被测端（产品侧注入，INTERFACES 勘误 FT1-18）：δ 由 capprobe 注在前端的 F2（`GenerateStream._read`，每读一条 gRPC 消息）或 F4（每请求队列出队，每个事件项）。相对 `e4b-mc`，除臂名、说明与按臂名取的登记路径外只改三处：启动与前端角色的 `probe_tier` 为 `P1`、前端 `snapshots` 为 `all`、档位表多一条 `launch_inject`。作业的覆盖清单必须是 P1（探针补丁、不开组），所以这两臂不能与 P0 的臂同一作业；探针补丁本身的开销（P1 − P0）按设计稿 §2.4 另行评估 |

阴性对照（桩节拍很低时积压读数在零附近）读阶梯的低档。快轨不做的 E4 项（4a、4c、K 轴、L 轴文本长 prompt、互扰档、陪跑阳性、chat 对照、1 核敏感性、`--random-ids`）见快轨计划第 4 节。

## 未核对

本机只做了结构校验、档位表展开与 `cross_check`（`vf2` / `vf4` 另有本机的编排器端到端：替身产品进程里真 capprobe 的心跳逐窗可读，勘误 FT1-20）。真前端里 F2 / F4 的注入与心跳要靠集群冒烟确认。真前端连 S-PB / S-NL、`--num_input_processor_workers` 在镜像加覆盖后的实际行为、前端在负载下实际可运行的线程数、前端是否把 `max_tokens` 截短（R 轴 `max_tokens` 1 没有对应的预制流时 S-PB 会拒绝）都要靠 E0 补充冒烟与 E4 自身的冒烟确认。

## S-FE 臂（前端线第 8 项，2026-11-20）

E4 卡 4c：`e4c-mc` = `e4b-mc` 的 S-PB 换成零改动的 S-FE（前端形态：`--model {param:tokenizer_dir}` + 指纹作业输入），每条流各自睡 `step_delay`（不齐发）；前端、capload、观察者与 `e4b-mc` 相同，门只读 `capload backlog.inflight`，链只有 `capload counts.ok`。4a（S-FC）仍没有模板。详见 `INTERFACES.md`「前端线第 8 项」。
