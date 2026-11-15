# ft1（快速通道 FT1：S-NL、oe_sink、冒烟与 E2a/E3/E4 场景模板、产品侧注入）进度

- 子批：ft1（W/00_plan/FASTTRACK-1117-PLAN.md 快速通道；RUNBOOK-stage2-draft.md §2 子批表与 §2.0 流程）。
- WT = `_local/wt-ft1`（分支 capkit-ft1，起点 756b9e7fe6ef5fb7726b0dd3f6a81869545f184d）；C（01_code main）只读；探针树 C/probe-tree 只读（除非范围明确要求，且只在其自有分支上改）。
- 范围：S-NL 桩、oe_sink、冒烟场景模板与 E2a/E3/E4 场景模板、产品侧注入（细节按 FASTTRACK-1117-PLAN.md 与设计文档 09_experiment_design.md §12/§13）。
- 契约：WT/INTERFACES.md 末尾新增勘误节「Fast track FT1: S-NL, oe_sink, smoke and E2a/E3/E4 scenario templates, product-side injection」，被覆盖的正文句子加指引注。
- 写代码时按累计规则集 RS1–RS22（code-review/self-review-compare/capkit-J17871/R4/{brief,triage}.md）自查。
- 验证：venv 单测，每次在子 shell 内 `ulimit -v 8388608` + `timeout 900`、`python -B`、PYTHONPYCACHEPREFIX=WT/_local/ft1/pyc、fakebin 集群命令替身；测试目录逐个串行跑，不加 -x。

## 进度日志
- 2026-11-15 23:49:02 开工：建 worktree（capkit-ft1 @ 756b9e7fe6ef5fb7726b0dd3f6a81869545f184d），写本进度日志头。
- 2026-11-16 00:01:10 FT1-snl 开始：读完设计稿 §2.3 S-NL、E3 卡、产品 servicer/server/streaming（探针树），开始写 capstubs/nullllm.py 与 null_llm_server.py（预计 1～2 h）。
- 2026-11-16 00:16:44 FT1-snl 变异 M0–M6 串行（约 5 分钟）
- 2026-11-16 00:22:52 提交 907a2e30（S-NL 代码与测试；变异 M1–M9 全被抓，M7 为等价变异已换 M7b）。开跑作业路径 e2e：覆盖清单按 HEAD 生成，e2e-snl-full / e2e-snl-mono / e0a-nullsse 回归（各约 2–3 分钟）
- 2026-11-16 00:32:56 跑 tests/capstubs 全目录（约 2–3 分钟），之后 tests/launch（约 5 分钟）；建测试用软链 WT/probe-tree（结束后改名移到 _local/ft1/probe-tree.link）
- 2026-11-16 00:41:10 提交 e9584185（Control 推理探针、勘误 FT1-1～7）；tests/capstubs 128 passed、tests/launch 230 passed、tests/jobs 250 passed；M10（去掉 TOOL_TABLE 行）被抓。按新 HEAD 重跑作业路径 e2e（e2e-snl-full、e2e-snl-mono、e0a-nullsse，约 6 分钟）
- 2026-11-16 作业路径 e2e（HEAD e9584185，覆盖清单 _local/ft1/e2e/ovl/P0-e9584185）：e2e-snl-full J9300011、e2e-snl-mono J9300012、e0a-nullsse J9300013 全部 submit rc 0、L0 VALID、judge rc 11（NOT_RUN）、0 RED、无 INPUT_INVALID；fakebin.log 只有真 capenv.gate 的 scontrol/squeue（替身 exit 97）。

## FT1-snl 结果

### MEASURED（本机，命令与日志在 _local/ft1/logs、_local/ft1/mut、_local/ft1/e2e/runs）
- tests/capstubs 128 passed（logs/v1-capstubs.log）；tests/launch 230 passed（v1-launch.log）；tests/jobs 250 passed（v1-jobs.log）；之后改动只涉及 S-NL 探针路径，test_null_llm_server.py 与 test_nullllm.py 已重跑通过（s4/s5/u3）。
- 变异（mut/summary.txt）：M1 节拍模式退化为一次入队、M2 不计 yielded、M3 不打 send.msg、M4 一次取多项、M5 不打 beat.step、M6 去掉包装核对、M7b 节拍器提前醒、M8 short 失效、M9 不截断、M10 去掉 TOOL_TABLE 行——全部被测试抓到；M7（首项偏移）是等价变异（product 闸门 offs[0] 恒为 0）。
- 作业路径 e2e：e2e-snl-full（capgrpc.load 2 进程、openengine 请求字节 → S-NL full + snl_double）逐档 streams_accepted = drain ok（合计 1550）、msgs_yielded = drain msgs（合计 32550），C3/C5/C9/C10/C11/P-PIDS/P-SENT/RECORDS PASS；e2e-snl-mono（metronome-only）VALID，C3/P-SENT NOT_RUN（没有负载角色，预期）；e0a-nullsse 回归 VALID。
### INFERRED
- 产品 OpenEngineServer 的构造路径会经过 server 模块里的 OpenEngineInferenceServicer 名字（读探针树 server.py:146 附近），所以子类替换在真产品上应当生效；启动时的「恰好实例化一次」核对会在不生效时退出 3。
### NOT RUN
- 真 tensorrt_llm（真 servicer / _format_result / Control 启动检查 / ModelLoader 加载 tokenizer）跑在 NullLlm 上：本机无法 import，属 E0 补充冒烟（勘误 FT1-6①）。
- raw 请求格式读 S-NL：真 servicer 要求非空且唯一的 request_id，原始格式不适用（勘误 FT1-1）；只测了「无 request_id 被拒」。
- 2026-11-16 00:50:23 FT1-sink 开始：读完快轨计划、E3 卡 3b、产品客户端 client/stream/engine_client_impl；试做「探针树真客户端文件 + 合成包」的本地替身（约 10 分钟）
- 2026-11-16 01:00:31 FT1-sink：loader 重构（共用读者参数、--planned-msgs、连接钩子、begin/end_request）完成，开始写 capgrpc/oe_sink.py（约 30 分钟）
- 2026-11-16 01:04:55 FT1-sink 首跑 tests/capstubs/test_oe_sink.py（约 2–3 分钟）
- 2026-11-16 01:07:59 跑 tests/grpc 全目录（loader 重构回归，约 3–5 分钟）
- 2026-11-16 01:09:59 跑 tests/capstubs 全目录（约 4 分钟）
- 2026-11-16 01:12:59 跑 tests/launch 全目录（约 5 分钟）
- 2026-11-16 01:17:22 FT1-sink 变异 S0–S17 串行（后台，约 10 分钟）
- 2026-11-16 01:23:18 提交 4d616f7a；开始作业路径 e2e：覆盖清单按 HEAD 生成，e2e-sink-spb（oe_sink 替身包 → S-PB）、e2e-weak-spb（同参数换 capgrpc.load）（约 10 分钟）
- 2026-11-16 01:27:54 e2e 完成（e2e-sink-spb J9300023、e2e-weak-spb J9300024、e0a-nullsse J9300025 全部 VALID、0 RED）；按 HEAD 重跑 tests/capstubs 全目录（约 3 分钟）

## FT1-sink 结果

- 代码：`tools/capgrpc/oe_sink.py`（新入口 `python -m capgrpc.oe_sink`，LoadWorker / RequestPlan 子类）、`tools/capgrpc/msgplan.py`（`--planned-msgs`）、`capgrpc.loader` 重构（`add_reader_args`、连接钩子、`begin_request` / `end_request`，行为不变）、`capgrpc.common.product_versions`（S-NL 改用同一函数）、`TOOL_TABLE` 一行；勘误 FT1-8～FT1-13；`tools/capgrpc/README.md`。
- 位置偏离：RUNBOOK §2.1 第 7 行写 `capstubs/oe_sink.py`，实际放 `capgrpc`（理由在勘误 FT1-8）。

### MEASURED（本机；日志在 _local/ft1/logs、_local/ft1/mut-sink、_local/ft1/e2e/runs）
- tests/capstubs 159 passed（logs/sink-capstubs2.log，HEAD 4d616f7a 之后只改了文档）；tests/grpc 39 passed（sink-grpc）；tests/launch 233 passed（sink-launch）；test_oe_sink.py 31 passed（sink-s4）。
- 变异 S1～S17 全部被抓到（mut-sink/summary.txt；S0 锚点不唯一未施加，基线即全目录运行）。
- 作业路径 e2e（覆盖清单 P0-4d616f7a；替身包 e2e/simpkgs-sink 装进一次性 venv）：e2e-sink-spb J9300023（oe_sink engine-event-stream、--connect channel、--planned-msgs）、e2e-weak-spb J9300024（同场景换 capgrpc.load）均 L0 VALID、judge rc 11、0 RED、无 INPUT_INVALID；C3/C5/C6/P-SENT/RECORDS/C9/C10/C11 PASS；两场景只差 drain 的 module、`--tier`/`--connect` 与名字；e0a-nullsse J9300025 回归 VALID、0 RED。
### INFERRED
- 真产品包里 oe_sink 走的是同一份客户端代码（替身包里的文件就是探针树里的产品文件原样），所以读路径与计数口径在集群上应一致；导入整库的开销与线程未知。
### NOT RUN
- 真 `tensorrt_llm` 安装里导入客户端、真 S-PB 前端模式 + 真 S-FE 下的 `--connect product`（本机只用探针树文件跑过）、oe_sink 读 S-NL、集群上的 E3-3b。
- 2026-11-16 01:41:46 FT1-scen 开始：读完快轨计划、E2/E3/E4 卡、gen_e0/gen_e0s、E0 补充冒烟登记与计划、S-NL/oe_sink 参数；开始写 gen_e0s 第 3 项与 E2a/E3/E4 生成器（约 60 分钟）
- 2026-11-16 01:55:01 FT1-scen 首跑 test_e2a_e3_e4_configs.py + test_e0s_configs.py（约 1–2 分钟）
- 2026-11-16 01:58:19 FT1-scen 跑 tests/launch 全目录（约 5 分钟）
