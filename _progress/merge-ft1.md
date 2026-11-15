# capkit-ft1 合入 main 的进度与冲突记录

- 任务：把 `capkit-ft1`（head 4c59bbcb8d452039828a0b03eeda1209e74bdfa3，起点 b2-spb 756b9e7f；快轨 FT1：S-NL、oe_sink、E0 冒烟第 3 项与 E2a/E3/E4 场景模板、产品侧注入）`--no-ff` 合入 main（fb61526dd3627d46f583919242cb2d4b8f90f9f9 = b3b3368e SB-A + b2-spb 合并 91c6e683）。
- 授权：FASTTRACK-1117-PLAN v2（11-15 23:4x 批准）；DECISIONS 第十五次答复（合入、本地测试、单个短冒烟不触发自审）。
- 工作树：`01_code/_local/wt-merge-ft1`，分支 `capkit-merge-ft1`。

## 日志

- 2026-11-16 03:30:59 建 worktree，`git merge --no-ff --no-commit capkit-ft1`：只有 INTERFACES.md 内容冲突（3 处）；其余双方都改的文件自动合并（tests/launch/test_launch_e2e.py、tools/capgrpc/common.py、tools/caplaunch/README.md、tools/caplaunch/orchestrator.py、tools/caplaunch/scenario.py、tools/capstubs/README.md、tools/capstubs/prebaked_server.py）。
- 2026-11-16 03:33:23 解冲突与语义修复写完（见下表），合并提交前先跑 tests/capstubs（约 4 分钟；临时软链 WT/probe-tree）
- tests/capstubs（解冲突与 S1、S2 修复后、合并提交前）：161 passed（logs/merge-ft1/pre/t_capstubs.log）。

## 冲突与处理

| # | 文件 | 两边各改了什么 | 处理 |
|---|---|---|---|
| 1 | `INTERFACES.md` §11 命令表（`capgrpc.load`、`capgrpc.nullgrpc`、`capselfcheck.relay` 三行） | main（SM-18）给 nullgrpc、relay 两行加 `--max-unrequested-s`；FT1 给 load 一行加 `--planned-msgs` 指向（FT1-9） | 逐行取改过的一边（与 merge-base 756b9e7f 逐行比对确认每行只有一边改了）：load 取 FT1，nullgrpc、relay 取 main |
| 2 | `INTERFACES.md` §10.2 `measure_class` 一行 | main（SB-B1-3）给 `observer` 加「自旋器」；FT1 给域外桩加 S-NL 三种模式指向（FT1-1）、给 E2a/E3/E4 加指向（FT1-15） | 合成一行：FT1 的文字 + main 的「自旋器（勘误 SB-B1-3）」 |
| 3 | `INTERFACES.md` 勘误末尾 | main 追加 SM-16 … 「合入 b2-spb 到 main（M-SPB-n）」；FT1 追加「Fast track FT1 …」一节 | 全留，main 的在前、FT1 一节在后，其后新增「合入 capkit-ft1 到 main（勘误 M-FT1-n）」一节 |

其余双方都改的 7 个文件文本自动合并；`tools/capstubs/prebaked_server.py` 的自动合并结果核对过：main 的 M-SPB-1（`--max-unrequested-s`、`grpc_server_options`）与 FT1 的 `check_inject_specs(specs, site_modes, who)` 互不相干。

## 文本无冲突、语义上要对齐的地方

| # | 问题 | 处理 |
|---|---|---|
| S1 | (a) SM-18：main 的 `make_server(reuseport, max_unrequested_s)` 第二个参数必填。FT1 里建 gRPC server 的地方：S-NL `metronome-only` 的占端口空 server 仍是 `make_server(reuseport=False)`（合入后该模式 TypeError，`test_metronome_only_dry_run` 会挂）；S-NL `minimal`/`full` 的 server 由产品 `OpenEngineServer` 自建；`oe_sink` 只建客户端 channel | 照 S-PB（M-SPB-1）改 S-NL：可选 `--max-unrequested-s`（只用于 `metronome-only`，产品模式给了即退出码 3——产品 server 的排队选项正是 E3/SM-19 要测的，不改），`resolve_max_unrequested` 定值传入 `make_server`，就绪文件与每档自报 `grpc_server_options`（产品模式 `null`）。测试与文档见勘误 M-FT1-1 |
| S2 | (b) `submit.time_budget(plan, renv, job_input_kinds=None)`（M-SPB-2）：FT1 没有调用 `time_budget` / `in_job_bound_s` / `laddergen` 的代码 | 不改；FT1 的 E2a/E3/E4 模板都声明 `job_inputs`，以后若经 laddergen 装箱要按种类数传入，记为 follow-up（M-FT1-4） |
| S3 | (c) 真前端发非十进制 request_id（j2 实测 22/22），S-PB/S-NL 带 `--schedule` 会全拒。核对 E3-3b 直连：`oe_sink --request-format openengine` 两档都把十进制 `str(req_id)` 交给产品客户端类；产品 `OpenEngineClient.generate` 只在 request_id 为空时补 `new_request_id()`，`to_generate_request` 原样抄 `EngineRequest.request_id`（读探针树 `openengine_client/client.py:370-371`、`openengine_poc/engine_client_impl.py:84-91`）⇒ 十进制 id 到得了 S-PB | 成立的是「十进制可达」：E3-3b 模板保持 S-PB 带 `--schedule`，不改生成器；E4 / e0s 前端臂本来就不带。加测试：`test_oe_sink.py` 两档断言 S-PB 收到的样本都是所发 `req_id`、原始格式断言样本非十进制（M-FT1-3） |
| S4 | 桩没有记录原始 request_id 原文（只有 `request_id_not_decimal` 计数） | S-PB、S-NL 每档自报加 `request_id_samples`（事件档最先 8 个原文，含被拒）；测试见 M-FT1-5，文档 `tools/capstubs/README.md`、勘误 M-FT1-2 |
