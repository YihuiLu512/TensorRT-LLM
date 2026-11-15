# configs/e0s — E0 补充冒烟场景(RUNBOOK-stage2 1.4 第 1、2、3 项)

全部 JSON 由 `configs/gen_e0s_configs.py` 生成(改配置请改生成器，生成到新目录后与本目录 diff;`tests/launch/test_e0s_configs.py` 核对两者逐字节相同)。
登记流程与 `configs/e0/README.md` 相同(填 `values.todo.json` → `caplaunch.tiergen` → 写 register.json);登记目录按 `02_e0/E0s/register` 写。

## 文件

| 文件 | 内容 |
|---|---|
| `e0s-import-probe.scenario.json` | 第 1 项：`importprobe` 角色(`caplaunch.smoke --items env --hold`,配置 `e0s-smoke.json`)在 origin 前跑 `env.import_probe`(带与不带 compat 路径各一次);`capload → capload.nullsse` 只打一档低速 warmup 与一个带门的 drain(档位表要求最后是带门的 drain) |
| `e0s-spb-frontend.scenario.json` | 第 2 项：`importprobe` → S-PB(`capstubs.prebaked_server` 前端模式)→ 真前端(`trtllm-serve --openengine-endpoint {addr:spb}`)→ `httpprobe`(completions 与 chat 各一条)→ `capload` 一档 warmup 与带门的 drain |
| `e0s-snl-frontend.scenario.json` | 第 3 项(勘误 FT1-14):`importprobe` → S-NL 最小版(`capstubs.null_llm_server --mode minimal`,产品 server 模块用缺省)→ 真前端 → `httpprobe`(只有一个 `/v1/completions`,`e0s-snl-httpprobe.requests.json`)→ `capload` 一档 warmup 与带门的 drain |
| `e0s-smoke.json` | `caplaunch.smoke` 的配置(`capkit.smoke_config/1`):`import tensorrt_llm, uvloop, grpc_health.v1.health_pb2` |
| `*.tiers.template.json`、`*.values.todo.json` | 档位表模板与要填的名字 |

## e0s-spb-frontend 的要点

- **作业输入**(INTERFACES 勘误 B2B-9):`fingerprint`(与 `configs/e0` 的 S-FE 相同：只有 `tokenizer_dir` 一个参数，`--trust-remote-code false --tokenizer-mode auto` 写死，取前端角色 argv 隐含的设置)与 `prebaked`(`--pb2-module tensorrt_llm.grpc.openengine._generated.generation_pb2 --tier-table {input:tier_table} --token-ids {param:spb_token_ids} --prompt-tokens {param:spb_prompt_tokens}`),由 `capjobs.launches` 在门禁之后生成。
- **S-FE 模块**(D16,勘误 B2B-11):覆盖清单要带 `--stub-module sfe=tests/unittest/grpc/openengine_client/fake_openengine.py`;计划写 `--param sfe_module={SNAP}/<清单里 stub_modules 的 src>`。S-PB 只自报它读到的字节的 sha,是否等于清单由判定器 C10 的 S-PB 一项判；本场景的 C10 因 `importprobe`、`httpprobe` 没有版本记录本来就是 NOT_RUN(见「判读」),所以要看 C10 的 NOT_RUN 原因里有没有 `role spb:` 开头的一项，没有才算核过。`--sfe-module` 指向另一个带同样入口的 S-FE 文件时 S-PB 照样能起，只有这一项会报出来(缺入口的文件以退出码 3 拒绝)。
- **档位表**:`max_tokens` 固定为 16,等于 `configs/e0/e0c-httpprobe.requests.json` 两个请求的 `max_tokens`(S-PB 拒绝没有预制流的 `max_tokens`);`beat_ns` 必须 > 0(S-PB 没有 asap)。S-PB 不给 `--schedule`,流按接纳时刻归档。
- **`spb_max_stream_records`**(S-PB 的 `--max-stream-records`):不小于时间表行数 + httpprobe 的请求数(2)。不给时 S-PB 的缺省容量只按档位表算时间表行数，httpprobe 的请求不在时间表里，会溢出(判定器报 rows dropped)。
- `httpprobe` 的请求文件用 `--param httpprobe_requests_json=…/e0c-httpprobe.requests.json`(`model` 换成登记的 served model name 后的副本)。
- **httpprobe 何时就绪**:S-PB 按节拍发，第一步在原点，所以 httpprobe 的请求在原点前发出、在原点之后才拿到回答。它的就绪条件是 `--started-file`(`raw/httpprobe.started.json`,发第一个请求之前写)，不是 `--result`(`raw/httpprobe.json`,两个回答都收完才写)。`httpprobe_timeout_s` 是每次阻塞读的超时(`http.client`),必须不小于「第一个请求发出到原点」的等待(上界是 `origin_lead_ms`)加上若干个 `beat_ns` 与余量；取 `origin_lead_ms / 1000 + 16 × beat_ns / 1e9 + 30` 秒即可。
- **`origin_lead_ms`**:原点前要串行做完 importprobe 的两次冷 `import tensorrt_llm`(带与不带 compat 路径，配置里每次最长 600 s)、S-PB 导入 S-FE 模块、真前端启动(含加载 tokenizer)。E0 计划用的 6000 ms 不能照抄，否则会 `not_ready_before_origin`;1.4 冒烟先按 importprobe 的上界取足，再按实测的就绪时刻收紧。

## e0s-snl-frontend 的要点

- **S-NL 最小版**:真 `OpenEngineServer` 与真 servicer 跑在 NullLlm 上，请求一到就把全部项放进队列(不需要节拍，档位表 `beat_ns` 为 0)。这是设计稿 §5.2 阶段 0 的 S-NL 最小冒烟；不过 ⇒ E3-3a 与 E8 的阶梯基线改用 S-FE、E2a 取消 S-NL 两臂(RUNBOOK-stage2 1.4)。
- **tokenizer**:S-NL 用产品的 loader 自己加载 `--tokenizer {param:tokenizer_dir}`,`--trust-remote-code false --tokenizer-mode auto` 写死，取前端 argv 隐含的设置(与 S-PB 一项的 `fingerprint` 作业输入相同);`--model` 也是 `{param:tokenizer_dir}`(前端的 served model name)。本项没有作业输入。
- **不给 `--schedule`**:前端发给 S-NL 的 `request_id` 不是时间表的 `req_id`,流按到达时刻归档(勘误 FT1-1)。`snl_max_stream_records`(S-NL 的 `--max-stream-records`)不小于时间表行数 + 1(httpprobe 的请求不在时间表里)。
- **httpprobe 何时就绪**:S-NL 最小版到达即答，与 E0(c) 的 S-FE 相同，就绪条件是 `--result`(`raw/httpprobe.json`),不需要 `--started-file`;`httpprobe_timeout_s` 只要覆盖前端加 S-NL 的一次往返。请求文件用 `--param httpprobe_requests_json=…/e0s-snl-httpprobe.requests.json`(`model` 换成登记的 served model name 后的副本)。
- **`origin_lead_ms`**:原点前要串行做完 importprobe 的两次冷 import、S-NL 导入产品并加载 tokenizer、真前端启动(含它对 S-NL 的三项启动检查)。

## 判读

三个场景里 `importprobe`、`httpprobe` 没有就绪文件与快照，也就没有版本记录，判定器的 C10 对这三个场景是 NOT_RUN(与 E0(c) 相同)。冒烟结论读 `raw/smoke/smoke.json`、`raw/httpprobe.json` 与 `python -m caplaunch.smoke --tree <树>`。

## 未核对

全部只在本机做过结构校验与档位表展开;容器里的真实运行(三个生成器、真 `fake_openengine.py`、真前端连 S-PB)属 1.4 补充冒烟。其中要核的前提还有：真前端是否把 `max_tokens` 截到上下文剩余长度(截后的 N 没有预制流 ⇒ S-PB 以 `no_prebaked_stream` 拒绝);前端发给 S-PB 的 `request_id` 是什么形态(勘误 B2B-8 ⑤)。第 3 项另有：真 server 模块在 NullLlm 上能否构造、`NullArgs` 是否缺 Control 直接读的字段、真前端对 S-NL 的三项启动检查(勘误 FT1-6①)。
