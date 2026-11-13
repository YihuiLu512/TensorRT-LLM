# fix-R1-probe 进度（FX-probe）

负责簇：P1、P2、P4–P8、K28（snapshot 侧）；P3 只核实、整理证据。
独占文件：probe-tree/**、tools/capprobe/**、tests/capprobe/**、tools/capcommon/snapshot.py

## 日志
- 2026-11-13 建 fakebin（_local/fx-probe/fakebin/，10 个脚本）；开始读 triage/brief/原始报告。
- 读完 triage、brief、capprobe R1 路 A/B 条目、probe-tree diff、tools/capprobe/*、snapshot.py、tests/capprobe/*。
- 核实结论（读码）：
  - P1 属实：launch.py:135、multi_frontend.py:485/649 的 install_loop_lag 在 try 之前。
  - P2 属实：facade.py:1599-1601 池开时 __anext__ 直接走 _next_postprocessed；_feed(:1406 queue.get) 无位点。
  - P5 属实：三处都在 facade.start() 之前。
  - P6：单前端第一批窗口内无非负载请求（orchestrator 只在 origin 前轮询 readiness：orchestrator.py:797-818；httpprobe 在 origin 前发完）。但多前端时附属前端的 AttachedFrontendWatchdog 每 5 s 经 launcher UDS GET /health（serve/multi_frontend.py:227,252），launcher 的同一 app 带 ServerArrivalTimeMiddleware ⇒ 窗口内 H1 会多计（INFERRED，未实跑）。修：只计 /v1/ 路径。
  - P7 属实（stream.py:267 两次 read.result()）。
  - P4：第一批 capload 端点全是 /v1/completions（configs/e0/*.scenario.json），且全部场景 probe_tier=P0；H3 在 chat 流为 0 时 C8 会以「window delta <= 0」显性判红（checks.py:382-410）⇒ 有意为之成立，不改。
  - P8 属实：SITE_NOTES base_line 复述，删。
  - K28：snapshot.py:264 makedirs(exist_ok) 不查属主/软链；:307 open("wb") 跟随软链。
- 现有能力确认：patchcheck 不给 --rev 时就是检查工作树（_diff_args(base, None) + 读文件 + untracked 检查），无需新增。
- P2 方案选择：SITES 改为 `lines` 列表（一个位点、一个槽、一个函数 `_capprobe.F4()`，多处代码位置）。不拆成两个位点的理由：拆开后两个位点共享槽，C8 对其中一个位点的「窗口增量>0」会被另一个位点的计数满足，等于假装两个位点都接上了；注入也会分成两个 site 名。代价：INTERFACES §6.5/§6.6 的 `"line"` 要改成 `"lines"`（勘误交主会话），make_samples.py 的合成清单要同步（交 FX-judge）。
- 基线：改动前 tests/capprobe 87 passed（rc=0）。
- 已改 probe-tree（未提交）：launch.py Lfe 挪进 try、facade.start() 之后；multi_frontend.py Lmfl 挪到「全部附属前端已连上」日志之后、_serve 之前（try 内）；Lmfa 挪到 _wait_for_launcher 返回之后、_serve 之前（try 内）；facade.py _feed 出队处加 F4；responses_utils.py H1 守卫加 `and scope["path"].startswith("/v1/")`。
- P7 试改后撤回：`result = read.result()` 的写法让 patchcheck FAIL（删了 1 行、新增行不属于位点、行号漂移），与「补丁只增不删、每个位点一个守卫调用」的约束冲突；代价只在 ON_F2 打开时多一次 Future.result()。MEASURED：临时改后 patchcheck rc=3，3 条 violation；已从 _local/fx-probe/stream.py.bak 还原，git diff 无 stream.py。
- 已改 tools/capprobe/sites.py（`line`→`lines` 列表、删 base_line、SITE_NOTES 只留 what、sites_for_groups 深拷）、patchcheck.py（一个位点多处位置；lines 必须非空严格升序；报告加 expected/found_location_count）、overlay.py（摘要带位置数）。
- patchcheck 对工作树实跑（MEASURED）：rc=0，PASS，7 位点 / 8 处位置，0 violation；输出在 _local/fx-probe/patchcheck-worktree.json。
- 已改 tools/capcommon/snapshot.py（K28）：ensure_own_dir（新建 0700；拒绝软链、非目录、非本人目录）；临时文件 O_NOFOLLOW|O_CREAT|O_TRUNC|O_CLOEXEC，0644。
- 下一步：改测试（gitfix/test_patchcheck/test_overlay/test_config）+ 新增 test_probe_tree_sites.py、test_snapshot_dir.py。
- 20:46:39 开始修前红灯验证：git archive HEAD 的 tools 到 _local/fx-probe/redcheck，配新测试跑（约 1 分钟）
- 修前红灯（MEASURED，_local/fx-probe/redcheck = git archive HEAD 的 tools + 新测试）：test_snapshot_dir 6 个用例全红（只有「已存在的本人目录可用」这个阳性对照是绿的）、test_patchcheck 的多位置用例 5 个全红（旧代码不认 `lines`，KeyError）；位置辅助函数对 HEAD d9380322a3 的旧文件：F4 只在 `__anext__`、三个 lag 都报「不在 try 体内」、H1 对 7 个路径计 7 次（新代码 3 次）。破坏性自测：把 Lfe 放进 try 但在 facade.start() 之前，报「is not after facade.start()」。
- P3 的 spawn 演示第一版写错（子进程抛错后父进程无超时地等队列，挂住），已用 TaskStop 停掉该后台任务，确认 spawn_demo 进程与其 resource_tracker 都已退出；第二版加了超时与异常回传。

## P3 证据（只核实、不改；供主会话写方案）

问题：补丁在 `tensorrt_llm/serve/responses_utils.py` 顶层 `import capprobe as _capprobe`（为了 H1），而 `capprobe` 导入时就严格解析 `CAPPROBE_*`（tools/capprobe/config.py:104-110：TIER 缺失或为 P0 都抛 `CapprobeConfigError`，它是 RuntimeError）。其余 5 个补丁模块（openai_server、stream、facade、launch、multi_frontend）也是顶层导入。

### 哪些进程会导入补丁模块（读码，probe-tree 工作树）

| 进程 | 导入路径 | 它的环境从哪来 | P1/P2 时看到什么 |
|---|---|---|---|
| 新架构前端（单前端、launcher、附属前端） | 本来就导入 openai_server / facade / launch / multi_frontend | 编排器 `_role_env`（tools/caplaunch/orchestrator.py:330-348）只给 `launcher == "product"` 的角色注入 `CAPPROBE_TIER/GROUPS/SHM_DIR/INTERVAL_MS/IDENTITY_FILE/ROLE_HINT`；附属前端由 launcher `subprocess.Popen` 重新执行，继承 launcher 的整份环境 | 设计内的写者 |
| 新架构 postproc 池进程（`num_postprocess_workers>0`） | `postproc_pool.py:744` 用 `multiprocessing.get_context("spawn")`；池进程在处理请求时懒导入 `postprocess_handlers`（`postproc_pool.py:160/181/208`），后者顶层 `from tensorrt_llm.serve.responses_utils import …`（`postprocess_handlers.py:21-22`）⇒ 导入 capprobe | spawn 子进程继承父前端的 `os.environ`，即带着前端的 `CAPPROBE_*` 与 IDENTITY_FILE、ROLE_HINT | 见下方实测 B：P2 下每个池进程起一条 `capkit-snapshot` 线程、写一个全零的 `<pid>.bin`，identity 与 role_hint 都是前端的 |
| 原架构 MPI rank 0 worker | 前端把 `PostprocParams(post_processor=completion_stream_post_processor, postproc_args=CompletionPostprocArgs)` 放进请求；worker 在 `executor/worker.py:362-374`（`pickle.loads`）反序列化时导入 `tensorrt_llm.serve.postprocess_handlers` → responses_utils → capprobe；`llmapi/serialization.py:113-118` 白名单里正有这些名字 | MpiPoolSession 只显式转发 `TRTLLM*`/`TLLM*` 与几个具名变量（`llmapi/mpi_session.py:36-50`）；其余取决于 MPI 运行时从哪个进程 fork（注释 `mpi_session.py:52-58`：singleton 的 DVM 在第一次 spawn 时 fork 一次并保留那时的环境）——能否带上 `CAPPROBE_*` 未实测（INFERRED：多半会带上） | 带不上：首个 completions 请求反序列化时抛 `CapprobeConfigError`，worker 收请求循环中断；带上且 P2：worker 里多一个零计数快照写者 |
| 原架构 PostprocWorker 进程（原架构 `num_postprocess_workers>0`） | `executor/postproc_worker.py:207` 取 `postproc_params.post_processor/postproc_args`，经 IPC 反序列化同一批 handler | 继承启动它的进程 | 同上 |
| 新架构 engine（S-FE/S-TM 等） | 模块级闭包不含补丁模块（路 A 的 AST 扫描）；gRPC/protobuf 不带 pickled handler | 若被登记成 product 角色，也会拿到 `CAPPROBE_*` | 不导入（INFERRED，未在运行期核对懒导入） |
| 辅助/工具进程（httpprobe、capload…） | 不导入 tensorrt_llm.serve | 编排器不给非 product 角色注入 `CAPPROBE_*`，但 `dict(os.environ)` 会把外层环境里已有的 `CAPPROBE_*` 带过去 | 外层环境若有 `CAPPROBE_*`，只有 product 角色会被校验冲突（orchestrator.py:340-342） |

### 实测（MEASURED，`_local/fx-probe/p3/spawn_demo.py`：spawn 子进程懒导入一个顶层 `import capprobe` 的模块，模拟池进程）

- A：父进程不带 `CAPPROBE_TIER` ⇒ 子进程 `CapprobeConfigError: CAPPROBE_TIER must be one of ['P0', 'P1', 'P2'] when capprobe is imported, got None`（rc=0 是 demo 自己吞了异常回传）。
- B：父进程（自己不导入 capprobe）带 P2 count 环境 ⇒ 子进程 `ACTIVE=True`，线程 `['MainThread', 'capkit-snapshot']`，shm 里出现子进程的 `<pid>.bin`，`identity.arm=p3`、`role_hint=frontend`，四个 count 槽全 0。
- 路 A 另有 pickle 反序列化的同类实测（capprobe R1 review-pr R1「证据」E2/E3）。

### 后果与第一批的关系

- 第一批 configs/e0 全部 `probe_tier: P0`（`grep '"probe_tier"' configs/e0/*.json` 全是 P0），P0 覆盖层不含补丁 ⇒ 第一批不导入 capprobe，不受影响。
- 第二批：原架构（E7、E9）或开池（E5）的 P1/P2 启动会碰到：(a) 环境没到 ⇒ 启动后首个请求才崩（门禁看不到）；(b) 环境到了且 P2 ⇒ 被测进程里多一条写快照线程、多一个零计数写者（与 RS13「测量者不污染被测」冲突），编排器按子孙关系会登记成前端角色的 `probe.snapshot` 进程。
- 与设计稿冲突：09_experiment_design.md:280 写「capprobe … 拷进 dist-packages，这样 MPI worker、spawn 起的池子进程、被重新执行的附属前端都能导入它，不依赖环境变量的继承」，而 capprobe 的「TIER 必需、缺失即抛」规则恰恰依赖继承。
- 可供选择的方向（只列，不评；由主会话写方案）：① 把 H1 的导入从 responses_utils 挪走（例如在 openai_server 里给中间件打点），让池/worker 不经 postprocess_handlers 导入 capprobe；② capprobe 在「没有 TIER」时降级为全关并在快照/日志里留痕，而不是抛错；③ 只有 pid 是登记过的产品进程时才激活写者（例如 IDENTITY_FILE 里带 pid 白名单，或要求 `CAPPROBE_ROLE_HINT` 所指角色的 pid == os.getpid() 的父链）；④ 维持现状，在 overlay/场景校验里禁止 P1/P2 与原架构、池开同时出现。
- 20:50:53 跑测试：tests/capprobe、tests/capcommon（snapshot.py 的既有测试，非我独占，只读运行），每个 ≤600 s
- 20:51:13 影响面测试（SnapshotWriter 使用方，别人的目录，只读运行）：capload、grpc、nullsse、sidecar
- 20:54:37 capload 29 passed、grpc 30 passed、nullsse 4 passed、sidecar 5 passed（均 rc=0）；开始跑 tests/launch（FX-launch 的目录，只读运行，看 snapshot 改动的影响，≤600 s）

## 各簇结论

| 簇 | 核实 | 改动 | 测试 |
|---|---|---|---|
| P1 | 属实 | probe-tree launch.py、multi_frontend.py：三处 install_loop_lag 挪进 try | test_probe_tree_sites.py::test_lag_installed_after_start_inside_try[Lfe/Lmfa/Lmfl]（修前 3 处都报「不在 try 体内」） |
| P2 | 属实 | facade.py `_feed` 出队处加同一守卫调用（进同一槽 count.F4.dequeued）；sites.py `line`→`lines`；patchcheck 支持一个位点多处位置；overlay 摘要带位置数 | test_f4_counts_on_both_queue_paths（修前只有 `__anext__`）；test_patchcheck::test_site_at_two_locations、test_site_locations_must_match_exactly×4（修前全红） |
| P4 | 有意为之成立 | 不改 | — |
| P5 | 属实 | 同 P1：Lfe 在 `facade.start()` 之后；Lmfl 在 `children = await asyncio.shield(spawn)`（全部附属前端 READY）之后、`_serve` 之前；Lmfa 在 `_wait_for_launcher`（launcher 已开始监听）之后、`_serve` 之前 | 同 P1 的用例另查「在启动完成调用之后、服务调用之前」；破坏性自测见上 |
| P6 | 部分属实：单前端第一批窗口内没有非负载请求；多前端时有（附属前端看门狗每 5 s 经 launcher UDS GET /health，INFERRED） | H1 守卫改成 `ON_H1 and scope["path"].startswith("/v1/")` | test_h1_counts_v1_requests_only（修前 7 个路径计 7 次，修后 3 次） |
| P7 | 属实但不改 | 试改后撤回：违反 patchcheck「只增不删、新增行只属于位点」 | patchcheck 实跑 FAIL（见上） |
| P8 | 属实 | 删 base_line；SITE_NOTES 只留 what；README 位点表不再复述行号 | test_config::test_slot_and_site_tables_consistent 断言 SITE_NOTES 只有 what |
| K28 | 属实 | snapshot.py：ensure_own_dir + O_NOFOLLOW 临时文件 | test_snapshot_dir.py 7 个（修前 6 红 1 绿，绿的是阳性对照） |

## RS 规则扫描（范围：probe-tree 补丁、tools/capprobe/**、tests/capprobe/**、tools/capcommon/snapshot.py）

- RS1/RS2：manifest `sites[*].lines` 的读方只有判定器 C8（只读 `slots`，checks.py:382-410），不受影响；合成样本 samples/make_samples.py:360 已是 `"lines": [1]`（不是我改的，FX-judge 或主会话已同步）。INTERFACES §6.5、§6.6 仍写 `"line"` ⇒ 勘误交主会话（原文见下）。
- RS3（额外扫出）：test_patchcheck::test_real_probe_tree_cli 与 test_overlay::test_real_image_to_baseline_range 在探针树缺失时 `skipif` 静默跳过 ⇒ 改为 `require_probe_tree()` 断言失败（与 K12 的处理口径一致），并在 capprobe README 写明依赖；test_real_probe_tree_cli 原来写死 `"found_site_count": 7` ⇒ 改为与 `len(SITES)` 及位置总数比较。patchcheck 新增：某位点 `lines` 为空或非严格升序 ⇒ FAIL。
- RS4：install_loop_lag 已进 try；LoopLagProbe 用 loop timer（capcommon/aio.py:87-110），不留 task，循环关闭即停，无需另行回收。
- RS10：snapshot 目录与临时文件已加固；overlay 的 P0 `--export-dir` 用 makedirs + jsonio.atomic_write_bytes（不是我的文件里的写法，路径来自操作者 CLI），未改，记一笔。
- RS11：F4 两条出队路径都覆盖；H1 只计 /v1/；H3 chat 流未打点（P4，SITE_NOTES/README 写明，C8 会显性判红）。
- RS12（额外扫出）：capprobe README 位点表复述了全部行号 ⇒ 删掉行号列，只指向 SITES；`__init__.py` 文档串「A patched site is one guarded call」⇒ 改为每处代码位置一个守卫调用；sites.py 文档串 `file:line` ⇒ `file:lines`。test_overlay 注释里的「37 个文件」是固定提交区间的事实，保留。
- RS13：P3（待裁决，见上）。

## 需要他方 / 主会话配合

1. INTERFACES.md 勘误（主会话改），建议原文：
   > §6.5 末条与 §6.6 示例：`SITES` 每项的 `"line": 行号` 改为 `"lines": [行号, …]`（升序、非空）：同一位点的事件若有多条代码路径，每条路径放同一个守卫调用，共用一个槽与一个探针函数；覆盖清单的 `sites` 原样带 `lines`。capprobe.patchcheck 要求每个位点恰好出现在 `lines` 所列的每一行；覆盖清单的 `patchcheck` 摘要另带 `expected_location_count`、`found_location_count`。F4 现覆盖 `__anext__`（无池）与 `_feed`（有池）两条路径。H1 只计路径以 `/v1/` 开头的 http 请求。
2. FX-launch：snapshot.py 现在拒绝软链、非本人目录作为快照目录；orchestrator.py:229 的 `os.makedirs(self.shm_dir, exist_ok=True)` 若要与之对齐（K28 orchestrator 侧），可直接调用 `capcommon.snapshot.ensure_own_dir`。
3. 主会话：P3 方案（证据见上）；P7 是否接受「保留两次 read.result()」（只在 ON_F2 打开时多一次调用）。
4. probe-tree 的改动未提交；`capprobe.overlay` 的 P1/P2 要求探针树 `tensorrt_llm/` 干净且 patchcheck 对 HEAD 通过 ⇒ 生成覆盖清单前需要主会话提交这些改动（我按约定不 commit）。

## 收尾（20:56:44）
- tests/launch 53 passed（rc=0，影响面检查）。最终复跑 tests/capprobe 与 patchcheck 结果见最终报告。
- fakebin.log 0 字节；probe-tree HEAD 仍是 d9380322a3、分支 capprobe-J17871，改动未提交。
