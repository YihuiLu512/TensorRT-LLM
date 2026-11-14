# configs/e0s — E0 补充冒烟场景(RUNBOOK-stage2 1.4 第 1、2 项)

全部 JSON 由 `configs/gen_e0s_configs.py` 生成(改配置请改生成器，生成到新目录后与本目录 diff;`tests/launch/test_e0s_configs.py` 核对两者逐字节相同)。
登记流程与 `configs/e0/README.md` 相同(填 `values.todo.json` → `caplaunch.tiergen` → 写 register.json);登记目录按 `02_e0/E0s/register` 写。

## 文件

| 文件 | 内容 |
|---|---|
| `e0s-import-probe.scenario.json` | 第 1 项：`importprobe` 角色(`caplaunch.smoke --items env --hold`,配置 `e0s-smoke.json`)在 origin 前跑 `env.import_probe`(带与不带 compat 路径各一次);`capload → capload.nullsse` 只打一档低速 warmup 与一个带门的 drain(档位表要求最后是带门的 drain) |
| `e0s-spb-frontend.scenario.json` | 第 2 项：`importprobe` → S-PB(`capstubs.prebaked_server` 前端模式)→ 真前端(`trtllm-serve --openengine-endpoint {addr:spb}`)→ `httpprobe`(completions 与 chat 各一条)→ `capload` 一档 warmup 与带门的 drain |
| `e0s-smoke.json` | `caplaunch.smoke` 的配置(`capkit.smoke_config/1`):`import tensorrt_llm, uvloop, grpc_health.v1.health_pb2` |
| `*.tiers.template.json`、`*.values.todo.json` | 档位表模板与要填的名字 |

## e0s-spb-frontend 的要点

- **作业输入**(INTERFACES 勘误 B2B-9):`fingerprint`(参数同 S-FE:`tokenizer_dir`、`trust_remote_code`、`tokenizer_mode`)与 `prebaked`(`--pb2-module tensorrt_llm.grpc.openengine._generated.generation_pb2 --tier-table {input:tier_table} --token-ids {param:spb_token_ids} --prompt-tokens {param:spb_prompt_tokens}`),由 `capjobs.launches` 在门禁之后生成。
- **S-FE 模块**(D16,勘误 B2B-11):覆盖清单要带 `--stub-module sfe=tests/unittest/grpc/openengine_client/fake_openengine.py`;计划写 `--param sfe_module={SNAP}/<清单里 stub_modules 的 src>`,判定器 C10 核就绪文件报的 sha。
- **档位表**:`max_tokens` 固定为 16,等于 `configs/e0/e0c-httpprobe.requests.json` 两个请求的 `max_tokens`(S-PB 拒绝没有预制流的 `max_tokens`);`beat_ns` 必须 > 0(S-PB 没有 asap)。S-PB 不给 `--schedule`,流按接纳时刻归档。
- `httpprobe` 的请求文件用 `--param httpprobe_requests_json=…/e0c-httpprobe.requests.json`(`model` 换成登记的 served model name 后的副本)。

## 判读

两个场景里 `importprobe`、`httpprobe` 没有就绪文件与快照，也就没有版本记录，判定器的 C10 对这两个场景是 NOT_RUN(与 E0(c) 相同)。冒烟结论读 `raw/smoke/smoke.json`、`raw/httpprobe.json` 与 `python -m caplaunch.smoke --tree <树>`。

## 未核对

全部只在本机做过结构校验与档位表展开;容器里的真实运行(三个生成器、真 `fake_openengine.py`、真前端连 S-PB)属 1.4 补充冒烟。第 3 项(S-NL 最小版)属 2C-min。
