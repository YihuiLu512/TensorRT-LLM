# capstubs

包在 `tools/capstubs/`(INTERFACES §1.1)。仓库里跑：`PYTHONPATH=<C>/tools`。

## `python -m capstubs.fingerprint`

从产品自己的 tokenizer 指纹生成 S-FE(`tests/unittest/grpc/openengine_client/fake_openengine.py`,零改动)的命令行参数。
绝不手抄词表大小 / eos / bos。

- 必须在 TRT-LLM 容器里跑(要 import `tensorrt_llm`;无驱动节点要先把 CUDA compat 库目录放进 `LD_LIBRARY_PATH`)。
- 参数：`--tokenizer <只含 tokenizer 文件的模型目录> --out <json> --trust-remote-code {true,false} --tokenizer-mode {auto,slow}`;
  后两项必须与前端的设置一致(前端用 `ModelLoader.load_hf_tokenizer(..., use_fast=mode != "slow")` 加载)。
- 计算：`tensorrt_llm.grpc.openengine_conventions.compute_tokenizer_fingerprint(tokenizer, model_dir)`;
  输出前用产品自己的 `TokenizerFingerprint.from_extra` + `compare_tokenizer_fingerprints` 做往返自检，不一致退出码 4。
- 输出 `capkit.fingerprint/1`,其中 `sfe_args` 是可直接拼进 S-FE 命令行的参数列表：
  `--vocab-size N [--eos-token-ids ...] --profile-json {"bos_token_id":..,"fingerprint":<to_extra()>}`
  (S-FE 在引擎上报 `ModelInfo.extra` 时以 `fingerprint` 为准)。作业里不手跑：场景在 `job_inputs.fingerprint` 里声明它的参数，`capjobs.launches` 在门禁之后生成(INTERFACES 勘误 B2B-9),角色参数用 `{"splice_json": "{input:fingerprint}", "key": "sfe_args"}` 引用。

测试 `tests/launch/test_fingerprint.py` 用 probe-tree 里真实的 `openengine_conventions.py` + 假的 tokenizer 加载器；
真实 tokenizer + 真实容器的运行属于 E0(c),未在本机运行。

## `python -m capstubs.prebaked`（S-PB 的预制响应字节，容器内运行）

类比 `capgrpc.reqbytes`：只在装了产品包的容器里能用（`--pb2-module tensorrt_llm.grpc.openengine._generated.generation_pb2`），导入期的任何异常（不只是缺产品包）都退出码 3。

- 参数：`--pb2-module M --tier-table T --token-ids 1,2,… --prompt-tokens P --out PATH`，可选 `--short-tokens D`（可重复）、`--method`（缺省 `/openengine.v1.Inference/Generate`）。
- 输出 `capkit.prebaked/1`：对档位表各负载档的每个（`max_tokens` N、k、s）——以及档位表里与 `--short-tokens` 给出的每个 `short.req` 量对应的缩短后的 N——存一整条流的字节：按 `capcommon.expect` 的 `product` 闸门每条消息一个 `TokenOutput`（`output_index` 0，只有 `token_id`，不填 `text`；token 号沿整条流循环取 `--token-ids`），最后一条 `GenerationFinished`（`output_index` 0，`LENGTH`）带 `usage`（`prompt_tokens` = P、`completion_tokens` = N）。所有响应都不带 `request_id`。
- 生成前用 `check_field_numbers` 核对产品模块的字段号与桩的线格式读取器（`prebaked.FIELDS`）一致，生成后用桩启动时的同一套检查再读一遍。
- 档位表各负载档的（k、s）必须相同（请求里只有 `max_tokens`，没有 k、s）；K 轴扫描按 k 分启动。

## `python -m capstubs.prebaked_server`（S-PB）

单进程 `grpc.aio` 桩。Generate（预制文件的 `method`）用恒等（反）序列化注册；请求字节只读 `request_id` 与 `stopping.max_tokens`（手写的 protobuf 线格式读取器，不建 protobuf 对象），响应字节全部来自预制文件。

- 参数：通用、快照、档位、`--loop`、`--ready-file`、`--port-file`、`--inject`（可选，可重复）；`--host`、`--port`、`--workers 1`（只能为 1）、`--prebaked PATH`、`--lag-period-us`；可选 `--schedule PATH`、`--max-stream-records N`、`--lag-records-capacity N`、`--max-unrequested-s N`（同 `capgrpc.nullgrpc`：gRPC 内核的 `grpc.server_max_unrequested_time_in_server`，缺省不短于档位表的 `request_timeout_ns`；实际值写进就绪文件的自报与每档 summary 的 `grpc_server_options`，INTERFACES 勘误「Fix SM-17 … SM-18 …」第 2 条）。
- 节拍：与 `capgrpc.nullgrpc --pacing beat` 相同（一个 `BeatSchedule`、每流一个无界待发队列、不等写完成）；第 j 条消息在步 `first_step + expect.msg_step_offsets(N, k, s, "product")[j]` 入队，finished 消息与最后一条 token 消息同一步入队。N 取自请求，k、s 取档位表负载档共同的值。
- 档的归属：给了 `--schedule` 时，`request_id` 必须是时间表里的 `req_id`（十进制），每条流按它的计划档归档；不给时按接纳时刻所在的档归档（前端模式下请求号由前端生成，无法对回计划档）。
- 拒绝（`INVALID_ARGUMENT`，按事件档计入 `rejects`）：`not_bytes`、`malformed`、`no_max_tokens`（未设或为 0）、`no_prebaked_stream`、`request_id_not_in_schedule`。其他方法路径 → `UNIMPLEMENTED`（前端模式下 Control 与 Health 除外）。
- 启动即拒绝（退出码 3）：负载档的（k、s）不唯一；预制文件缺档位表或 `short.req` 规格需要的流；注入位点不认识或模式不支持；`--workers` ≠ 1；给了 S-FE 选项却没给 `--sfe-module`；`--sfe-module` 读不到、导入期抛出任何异常（缺产品包、语法错、运行期错误）、它的解析器拒绝这些选项、缺 S-FE 入口、建 `FakeEngine` 或服务失败、把服务注册到 server 时抛异常；时间表不是由这份档位表生成的。
- 前端模式（`--sfe-module PATH`，即探针树基线里未改动的 `fake_openengine.py`）：按路径导入它，用它自己的 `_parse_args` + `_profile_from_args` 解析本桩收到的 S-FE 选项（`--sfe-profile` 对应 S-FE 的 `--profile`，`--model`、`--max-context-length`、`--vocab-size`、`--eos-token-ids`、`--profile-json` 同名），建 `FakeEngine(profile, tokens_per_step=1, step_delay=0.0, record=False)`，注册它的 `_Control`、`_Health`。指纹 JSON 的 `sfe_args` 可原样拼进命令行。文件只读一次，执行的就是算 sha256 的那份字节；sha256 写进就绪文件（`info.workers[0].info.control`，判定器用 `LaunchData.ready_tool_info` 读）与每档自报的 `control`。只在容器里能用。
- 注入位点：`send.msg`（每条消息交给 gRPC 之前；`busy`/`sleep`）、`beat.step`（每个节拍步；`busy`/`sleep`）、`short.req`（每个被接纳的请求一次；`short`：该请求按 `max(N − short_tokens, 1)` 出流，按 nullsse 的规则）。
- 快照槽：`backlog.active_streams`、`backlog.pending_msgs`、`backlog.yielded`（counter）、`self.accepted/completed/cancelled/rejected`、`bytes.sent`、`count.beat_steps`、`count.beat_late_steps`、`count.beat_enqueued`、`lag.beat_wake`、`lag.loop`、`inject.<site>.*`。
- 输出（每档）：`streams`（`server.streams`；`req_id` = 十进制 `request_id`，不是十进制时 −1；`n_msgs` = 实际发出的 gRPC 消息数，含 finished；`tokens` = 这些消息的 token 数；`tier_seq` 见上「档的归属」，`meta.tier_seq` 写 `schedule` 或 `event`）、`steps`（`beat.steps`）、`summary.json`：`counts`（`streams_accepted/completed/cancelled/truncated/rejected`、`first_step_bumped`、`msgs_sent`、`bytes_sent`、`finished_sent`、`tokens_sent`、`short_requests`、`request_id_not_decimal`）、`beat`、`yielded_event_time`、`rejects`、`request_id_samples`（勘误 M-FT1-2）、`method`、`prebaked`（路径与 sha256）、`schedule_sha256`（给了 `--schedule` 时是该文件的 sha256，否则 `null`；就绪文件的自报里也有）、`tokens_per_msg`、`stream_interval`、`tier_attribution`、`control`、`grpc_server_options`、`inject`、`loop_lag_ns`、`cpu`、`overflow`。
- **场景配置的 `expected_procs` = 1**(TOOL_TABLE 的 `single`:`workers`、`expected_procs` 不是 1 即拒绝)。
- 作业里的用法(INTERFACES 勘误 B2B-9、B2B-11):`--prebaked {input:prebaked}`(场景 `job_inputs.prebaked`,由 `capjobs.launches` 用产品的 `generation_pb2` 在作业里生成);前端模式的 `--sfe-module` 指向覆盖清单 `stub_modules` 里 `sfe` 在快照中的文件(计划写 `--param sfe_module={SNAP}/<src>`),判定器的 C10 核对就绪文件里报的 sha 等于清单的；指纹参数拼自 `{input:fingerprint}`。示例场景见 `configs/e0s/e0s-spb-frontend.scenario.json`。

测试 `tests/capstubs/`：线格式读取器以真 protobuf 解析为准绳；预制文件用 pb2 测试替身（`oe_double.py`）与探针树里产品的 `_generated/generation_pb2.py`（按路径作私有包导入，只需 protobuf）各生成一次；本地 e2e 用 `capgrpc.load --request-format openengine`（请求字节同样由替身生成）读 S-PB；前端模式用一个替身 S-FE 模块。真 `fake_openengine.py` 与真前端连 S-PB 属于 1.4 补充冒烟，本机没有运行。

## `python -m capstubs.null_llm_server`（S-NL）

产品的 `OpenEngineServer`（真 servicer、真 `_format_result`）跑在 `capstubs.nullllm.NullLlm` 上；单进程。契约见 INTERFACES 勘误 FT1-1～FT1-6（模式、命令行、位点与积压槽、输出、已知缺口），这里只列用法。

- `--mode minimal`：每个请求的全部项一到就入队，不需要节拍；E0 补充冒烟的最小版（一个请求能回）。
- `--mode full`：一个节拍器按档位表推进，每条流一个无界待发队列，`NullHandle.__anext__` 每次只取一项；已 yield 计数与 `send.msg` 注入在包住产品 servicer `Generate` 的子类里（D17，不改产品文件）。
- `--mode metronome-only`：只有节拍器、队列与进程内消费协程，按 `--schedule` 开流，没有 servicer 与 gRPC 流量；E3-3a 的同进程空跑档。只为占端口而起的 gRPC server 用 capkit 的队列选项（可选 `--max-unrequested-s`，缺省规则与 S-PB 相同；勘误 M-FT1-1）；`minimal` / `full` 下产品 server 用它自己的选项，给 `--max-unrequested-s` 即退出码 3。就绪文件与每档自报的 `grpc_server_options` 在 `metronome-only` 下是实际值，其余模式为 `null`。
- 每档自报的 `request_id_samples`：本档（事件档，含被拒的请求）最先收到的至多 8 个原始 `request_id`（勘误 M-FT1-2），S-PB 同。
- 产品模块只在 `minimal` / `full` 下导入（`--server-module`，缺省 `tensorrt_llm.grpc.openengine.server`），任何失败退出码 3；导入了什么写进就绪文件与每档自报的 `imported`。只在容器里能用；本机测试用 `tests/capstubs/snl_double.py`。
- 读 S-NL 用 `capgrpc.load --request-format openengine`（原始请求格式没有 `request_id`，真 servicer 拒绝）。

## `capstubs/fe_inject`（真前端与 postproc 池进程里的运行期注入位点，R3 的 E5 / E6 V 干预）

- 用法：把 `capstubs/fe_inject` 目录放在前端角色 `PYTHONPATH` 的最前面（池进程是前端的 spawn 子进程，继承它），环境变量 `CAPKIT_FE_INJECT` 写位点规格（与 `snx_inject` 同语法，如 `Pp:busy:200000`、`Fr:count`）。不改产品文件。
- 位点：`Fr` = `GenerateStream._validate`（前端进程，每读到一条 Generate 响应一次，F2 的路径）；`Pp` = `postproc_pool._handle`（每个池进程，每从 inbox 取一条消息一次：开流、每个事件、收尾）。
- 机制：`sitecustomize` 先执行 sys.path 上下一个 `sitecustomize.py`（镜像自己的），再装一个导入钩子；位点所在模块在本进程加载完时，用 `snx_inject.install` 包住该函数（先计数，再按规格忙等，持 GIL）。规格写错直接退出码 2（site.py 会吞掉 sitecustomize 的异常，所以这里自己退出）。
- 输出：每个进程每 2 s 与退出时在 stderr 写一行 `capkit-fe-inject-report {"pid", "sites": {site: hits/applied/applied_ns}}`；设了 `CAPKIT_FE_INJECT_DIR` 时改由 `snx_inject` 写 `<dir>/inject.<pid>.jsonl`。加载时与每装上一个位点时各写一行标记。
- 测试 `tests/capstubs/test_fe_inject.py`：替身模块树上验证父进程与 spawn 子进程里的包裹、忙等时长与计数、无规格时不包、未知位点拒绝启动。

## `capstubs/q1_seg` + `capstubs.q1seg`（真前端请求路径的逐段 CPU 计时，前端线第 10 项）

- 用法：把 `capstubs/q1_seg` 目录放在前端角色 `PYTHONPATH` 的最前面；`CAPKIT_Q1SEG` = `all` 或段号列表（`S1j,S1k,S1v,S2,S3,S4,S5`），`CAPKIT_Q1SEG_INJECT` = `H1:count` 或 `H1:busy:<ns>`，`CAPKIT_Q1SEG_REPORT_MS`（缺省 250）。两个开关都没设时只链式执行镜像自带的 sitecustomize，不导入 capstubs。不改产品文件。
- 段：S1j = `tensorrt_llm.serve.openai_server` 模块命名空间里的 `json.loads`（请求体 JSON 解析；产品的路由类 `_MsgspecRoute` 让 FastAPI 经 `_MsgspecRequest.json` 读请求体，它调用的是 openai_server 自己的 `json.loads`，不经 Starlette 的 `Request.json`；只把该模块的全局 `json` 换成代理，别处的 `json.loads` 不变）、S1k = `KvHintsRequest.filter_kv_hints`（`tensorrt_llm.serve.kv_hints`，对解码后的请求体执行 hint 信任策略）、S1v = `fastapi._compat.v2.ModelField.validate`（pydantic 校验请求体）、S2 = `OpenEngineLLM.generate_async`、S3 = `completion_stream_post_processor`（`openai_server` 与 `postprocess_handlers` 里换成同一个包装对象）、S4 = `OpenAIServer.await_disconnected`（协程函数，只计次数）、S5 = `GenerateStream._validate`。H1 = `ServerArrivalTimeMiddleware.__call__`（http、路径以 `/v1/` 开头时计一次；`busy` 时在主线程忙等 `<ns>` 再交回原协程），在所有段之外。
- 计量：只在主线程（事件循环）上计 `n / cpu_ns（thread_time_ns）/ wall_ns（perf_counter_ns）`；别的线程只计 `n_other`；计时调用里再进入的计时调用只计 `n`、`n_nested`，不重复计时。实测每次调用的包装开销约 0.7 µs（本机，单测上限 1 µs）。
- 拒绝：段名或注入规格写错退出码 2；包装点不存在、类型不对（计时段遇到协程函数、S4 / H1 不是协程函数、静态方法等）在模块加载时退出码 3；第一次计时调用时还有段没装上，也退出码 3。
- 输出：主线程在计时调用结束后（计时区间之外）至多每 `CAPKIT_Q1SEG_REPORT_MS` 写一行、退出时再写一行 `capkit-q1seg-report {json}` 到 stderr（累计值，`t_ns` = `time.monotonic_ns()`，与编排器的档窗口同一时钟）；没有额外线程。`q1seg.parse_reports`、`q1seg.window_delta` 读这些行并在档窗口两端线性插值；`q1seg.zero_hit_sites` 列出窗口内调用次数不 > 0 的已装位点（命中数检查），`q1seg.arrivals` 给出每请求的分母（H1 hits，没有 H1 时用 S2 调用次数）。
- 测试 `tests/capstubs/test_q1_seg.py`：已知 CPU / 墙钟的阳性对照 ±5 %、包装开销 ≤ 1 µs、协程拒绝、嵌套与其他线程、链式 sitecustomize、开关关闭、错开关退出 2、包装点缺失或类型错退出 3、替身树上的命中数逐个精确（S1j 只包 openai_server 的名字，Starlette 与全局 `json.loads` 不变）、H1 忙等阳性对照、报告行插值、探针树里各位点的类型、S1j / S1k 在请求体热路径上（AST）、命中数检查的阴性对照（J6190736 的 S1j = 0 形态判不过）与分母。
