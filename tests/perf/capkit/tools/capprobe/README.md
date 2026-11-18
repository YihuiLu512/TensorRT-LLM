# capprobe（B3）

产品进程里的探针模块（不依赖 `tensorrt_llm`），外加三个工具：快照解码、补丁静态检查、覆盖清单生成。约定以 `INTERFACES.md` §6、§7、§11 为准；本文件只列本包的入口、位点与新增的可选参数。

## 入口

| 入口 | 用途 |
|---|---|
| `import capprobe`（产品代码，经探针补丁） | 导入时读一次 `CAPPROBE_*` 环境变量（§6.5）；只有命令行与 `CAPPROBE_ENTRY` 匹配的进程才可能启用（勘误 R1b），打开的组与注入决定哪些位点生效 |
| `python -m capprobe.decode FILE [--no-hist]` | 把一个快照解码成 JSON；坏文件退出码 3，读不到 4 |
| `python -m capprobe.patchcheck --probe-tree P --base-commit SHA [--rev REV] [--out F]` | 补丁静态检查（见下）；PASS 退出 0，FAIL / NOT_RUN 退出 3 |
| `python -m capprobe.overlay …`（§11 的参数） | 生成 `env/overlay_manifest.json`（§6.6） |

工具进程（`python -m capprobe.decode|overlay|patchcheck`）按原始命令行里 `-m` 后的模块名进入「工具模式」：不读环境变量、不起探针，所以在 P2 的环境里运行也不会在快照目录里多出一个 `<pid>.bin`。

## 只在被测入口进程里启用（勘误 R1b，实现在 `capprobe.config` 与 `capcommon.probe_entry`）

| 进程 | 结果 |
|---|---|
| 没有任何 `CAPPROBE_*` 变量 | 全关（`ENABLED`、`ACTIVE`、各 `ON_*` 都是 False，不起线程、不写快照、不报错；`CONFIG.reason = no_capprobe_env`） |
| 有 `CAPPROBE_*`，没有 `CAPPROBE_ENTRY` | 先校验在场的值（见下），之后全关（`no_entry_rule`） |
| 有 `CAPPROBE_*`，命令行与 `CAPPROBE_ENTRY` 不匹配（spawn 池子进程、MPI worker、其他导入者） | 先校验在场的值，之后全关（`not_the_registered_entry`），不读身份文件 |
| 命令行匹配（编排器起的产品进程，及多前端 launcher 用 `[sys.executable] + sys.argv` 重新执行出来的附属前端；匹配形式见下） | 全套规则：`CAPPROBE_TIER` 缺失报错，`P0` 报错（P0 的覆盖层不应含探针补丁），其余同下 |

`CAPPROBE_ENTRY` 由编排器给每个产品角色写（`capkit.probe_entry/1`：`cwd` 是该角色工作目录的 realpath；`program` 是该角色 argv[0] 解析后的 realpath，含 `/` 的按 `cwd` 解析，裸名按角色的 PATH 查、PATH 里的相对项也按 `cwd`；`args` 是其余 argv 原样）。规则全文与两种匹配形式以 `capcommon.probe_entry` 模块说明为准，要点：

- 解析一律按规则里的 `cwd`，不按进程当前目录，所以产品先 `chdir` 再导入 `capprobe` 也照样匹配；只解析含 `/` 的元素和第 0 个位置的裸名。
- 直接形式：命令行以 `args` 结尾，紧挨着的前一个元素是 `program`。覆盖原生程序、`python 脚本`、`python -m 模块`、`#!` 脚本（内核在前面加解释器）、相对路径启动，以及脚本启动后的重新执行（命令行与原来相同）。
- 模块重新执行形式：只在 `args` 以 `-m <模块>` 开头时适用。`python -m 模块` 启动后 Python 把 `sys.argv[0]` 换成模块文件，重新执行变成 `python <模块文件> <其余参数>`；命令行以其余参数结尾、前一个元素是以 `<模块路径>.py` 或 `<模块路径>/__main__.py` 结尾的路径（只看名字，不 import）、再前一个元素是 `program`，就算匹配。`-m` 之前有解释器选项（`python -u -m 模块`）时这些选项也不在 `sys.argv` 里，重新执行出来的进程**不匹配**（探针关）。

fork 出来、没有 exec 的子进程命令行不变，仍启用（各自归零计数、按自己的 pid 写快照，同 R1 之前）。

## 开关的补充规则（在 §6.5 之上，全部是导入时 `CapprobeConfigError`，它是 `RuntimeError`）

- 只要有 `CAPPROBE_*`，不论哪个进程都校验在场的值：未知的 `CAPPROBE_` 变量名、非法 `CAPPROBE_TIER`、`CAPPROBE_GROUPS` 与 TIER 不符（P0、P1 ⇒ 组为空；P2 ⇒ 组非空；没有 TIER 却有组或注入）、未知组名、坏注入规格、坏 `CAPPROBE_ENTRY`、非正整数周期、相对路径 `CAPPROBE_SHM_DIR` 都报错。
- 本版本实现 `count`、`lag`、`stack` 三组；`bytes`、`queue`、`ts`、`pin` 是已知组名但未实现，打开即报错。`stack` 没有代码位点（`sites_for_groups` 为空，覆盖清单的 `sites` 为空，C8 记 NOT_RUN）。
- 任何组或注入打开时，`CAPPROBE_SHM_DIR`（绝对路径）、`CAPPROBE_INTERVAL_MS`、`CAPPROBE_IDENTITY_FILE` 必需；`lag` 打开时 `CAPPROBE_LAG_PERIOD_US` 必需。
- 注入只支持位点 `H1`、`H3`、`F2`、`F4`，模式只支持 `busy`、`sleep`、`miscount`；`miscount` 要求 `count` 组打开（否则误计不可见）。注入打开时，四个位点都会经过注入器，没有规格的位点也计 `inject.<site>.hits`。在编排器下，注入只能经档位表 `launch_inject` 指名产品角色登记（INTERFACES 勘误 FT1-18）。

## 位点（`capprobe.SITES`）

每个位点标一个进程类（`proc_kind`，词表 `capprobe.sites.PROC_KINDS`）：一个类 = 该类进程在服务本研究的负载（`/v1/completions` 流）时都会经过的那些位点。产品角色在场景里用 `proc_kinds` 声明自己的进程树里有哪些类，判定器 C8 只在声明了的类的位点上要求窗口增量。

| 进程类 | 位点 | 哪些进程 |
|---|---|---|
| `frontend` | H1、H3 | 经 `OpenAIServer` 服务 OpenAI HTTP 接口的前端：新架构单前端、多前端 launcher 与附属前端、原架构 `trtllm-serve` 前端 |
| `frontend_openengine` | F2、F4 | 经 OpenEngine 客户端与 facade 连 engine 的新架构前端（单前端、launcher、附属前端） |
| `frontend_single` | Lfe | 新架构单前端（`--num_serve_frontends 1`，走 `launch.py`） |
| `frontend_launcher` | Lmfl | 多前端的 launcher |
| `frontend_attached` | Lmfa | 多前端的附属前端 |

角色的 `proc_kinds`：新架构单前端 `["frontend", "frontend_openengine", "frontend_single"]`；新架构多前端（launcher 与附属前端同一个角色）`["frontend", "frontend_openengine", "frontend_launcher", "frontend_attached"]`；原架构前端 `["frontend"]`；不跑探针代码的角色 `[]`。

文件与行号只以 `capprobe.SITES` 为准（`lines` 是打过补丁的探针树里每处守卫 `if` 所在行；同一事件有几条代码路径，就在每条路径上放同一个守卫调用、列几个行号，共用一个槽）。本表不复述行号。

| 位点 | 文件 | 槽 | 一次经过的含义 |
|---|---|---|---|
| H1 | `tensorrt_llm/serve/responses_utils.py` | `count.H1.arrivals` | 路径以 `/v1/` 开头的每个 `http` 类型 ASGI 请求，不论路由与状态码：生成请求之外，`GET /v1/models`、其他 `/v1/` 路由和 `/v1/` 下的 404 也计（capkit 的负载器只发 `/v1/completions`、`/v1/chat/completions`；`/health`、`/metrics` 不在 `/v1/` 下）。请求经过几个打了探针的进程就计几次：多前端时附属前端把 `/v1/messages/batches` 系列路由转发给 launcher，launcher 再计一次；disagg 路由也装同一个中间件，每一跳各计一次 |
| H3 | `tensorrt_llm/serve/openai_server.py` | `count.H3.chunks` | `completion_generator` 每个 `pp_res`（只有 `/v1/completions` 流；chat 流未打点） |
| F2 | `tensorrt_llm/grpc/openengine_client/stream.py` | `count.F2.msgs` | `_read` 读到的每条消息（不含 EOF 与本地取消；含终止事件之后排空读到的） |
| F4 | `tensorrt_llm/serve/openengine_poc/facade.py` | `count.F4.dequeued` | 请求仍在进行时从每请求队列取出的每个事件项（结束标记与错误项不计，engine 取消的那一项计）。无 postproc 池：`__anext__` 只在响应处理方等下一个输出时取，请求结束后不再取。有池：`_feed` 一到就取，只在后处理会话打开时计，积压在 F4 之后（池里与结果队列里）。所以 F2 − F4 只在无池、且请求全部正常结束时才是队列里的积压：F2 还计 error、未知与溢出事件；请求中止、失败、命中停止串（有池时：会话先关闭）之后读到的事件 F2 计、F4 不取，留下残留。读 F2 − F4 之前先按失败计数筛档（见勘误 R4-5）。两条路径在请求收尾时口径也不同：`_feed` 把终止事件交给池之后、会话关闭之前取到的项会计入，无池路径的 `__anext__` 不取这些 |
| Lfe | `tensorrt_llm/serve/openengine_poc/launch.py` | `lag.Lfe`（hist）及 `.n/.sum/.max/.neg`、`lag.Lfe.installs` | 单前端事件循环；`facade.start()` 返回之后、HTTP 服务启动之前安装 |
| Lmfl | `tensorrt_llm/serve/openengine_poc/multi_frontend.py` | `lag.Lmfl` … | 多前端的 launcher 循环；全部附属前端连上之后、开始服务之前安装 |
| Lmfa | `tensorrt_llm/serve/openengine_poc/multi_frontend.py` | `lag.Lmfa` … | 多前端的附属前端循环；launcher 开始监听之后、开始服务之前安装 |

三个安装点都在各自 `try`（其 `finally` 负责关闭 facade 等清理）之内，所以安装出错时清理照常执行。`lag.<位点>` 直方图的口径（`capprobe.sites.LAG_NOTE`）：

- 探针从安装起一直跑到进程退出，不会停：HTTP 服务自己的启动（uvicorn 启动、应用 lifespan 的启动部分）和关停（排空、关闭 facade、停掉附属前端）都在直方图里，也在其后的每份快照和退出时的终值快照里。判定器只读每档窗口首末两份快照之差。
- uvloop 的 `call_at` 是毫秒精度，每个样本可能在真实停顿之外再晚最多约 1 ms：1 ms 以下的滞后分辨不出。

注入心跳槽：`inject.<H1|H3|F2|F4>.{hits,applied,applied_ns}`。机器可读清单：`capprobe.SLOTS`（组打开时在导入时一次性登记，没经过的位点表现为「槽在、值为 0」）。`SITE_NOTES` 只写每个位点「一次经过」的含义。

## 运行时

- 位点写法：`if _capprobe.ON_F2: _capprobe.F2()`（守卫可以是 `ON_<site> and <条件>`，条件只在位点打开时求值）；关闭时只剩一次模块属性读取与一次分支。
- 快照：后台线程每 `CAPPROBE_INTERVAL_MS` 写 `<CAPPROBE_SHM_DIR>/<pid>.bin`（`capcommon.snapshot`），进程正常退出时再写一份终值。元数据里 `identity` 取自 `CAPPROBE_IDENTITY_FILE`（`pid` 换成本进程的），另有 `probe`（档位、组、注入规格、`sites_on`）。
- 版本自报（D11）：导入时写一份（`loop_impl` 为 null）；`install_loop_lag` 之后带上循环类与 `probe_loops`；产品在 `capprobe` 之后才 import `grpc` 的，写快照线程发现后自动刷新 `grpc`、`protobuf`、`protobuf_impl`。
- `fork`：子进程里计数清零、换成子进程自己的 pid 与写快照线程、辅助线程重建；事件循环探针不继承。
- `install_loop_lag(site)` 必须在运行中的循环里调用；同一进程同一安装点装两次报错。

## 栈采样组 `stack`（`capprobe.stack`）

- 打开后 `_setup` 启动 `capkit-stack` 线程：每 4 ms 读一次 `sys._current_frames()`，主线程的栈从最内层向外找第一个能定类的帧，归入 `sites.STACK_CATEGORIES` 之一（规则表在 `capprobe.stack._FILE_RULES`；asyncio、uvloop 等循环框架帧透明；pydantic、json 等序列化帧取外侧第一个定类帧的类别，外侧是流式生成器时记 `encode`；最内层就是循环 runner 时记 `idle`，即等 I/O 或在循环的 C 代码里）。
- 每个样本按距上一个样本的墙钟时间加权（`stack.main.wall_ns.<类>`），同时记主线程这段时间的 CPU（`stack.main.cpu_ns.<类>`）和样本数（`stack.main.n.<类>`）。采样要拿 GIL，主线程长时间占着 GIL 的 C 调用会算到调用它的 Python 帧上。
- 其他 Python 线程（分词线程池等）只在不处于等待点时计入 `stack.thr.wall_ns.<类>`（线程·秒）；capkit 自己的线程跳过。
- 每 0.5 s 读 `/proc/<pid>/task/*/children` 与子进程 `stat`：`stack.children_cpu_ns` 为见过的每个子进程的 utime+stime+cutime+cstime 之和，`stack.children` 为最近一次的存活子进程数。
- 合计：`stack.samples`、`stack.wall_ns`、`stack.main_cpu_ns`、`stack.proc_cpu_ns`、`stack.sampler_cpu_ns`（采样线程自身 CPU，即直接开销）、`stack.errors`。按函数的明细（定类帧的 `类别|文件:函数` → 墙钟 ns，累计，最多 200 项）每 2 s 写进快照 meta 的 `stack_funcs`。

## 覆盖清单（`capprobe.overlay`）的补充参数

- `--export-dir DIR`：P0 必需。探针树的工作树是打过补丁的，P0 的文件从基线提交的 blob 导出到这里，`src` 指向导出的文件。
- `--stub-module KEY=PATH`（可重复，每个 KEY 一次；KEY 只有 `sfe`；INTERFACES 勘误 B2B-11）：把 `tensorrt_llm/` 之外的一个桩脚本（如 `sfe=tests/unittest/grpc/openengine_client/fake_openengine.py`）从**基线提交**的 blob 零改动导出到 `<--export-dir>/<PATH>`，记进清单的 `stub_modules`（`key`、`repo_path`、`commit`、`src`、`sha256`、`size`）。要 `--export-dir`（任何档位）。它随快照进作业，但不装进 site-packages；S-PB 前端模式从快照里加载它，判定器 C10 核 sha。
- `--allow-unshipped PATH`（可重复）：`tensorrt_llm/` 下改了、但不是安装包内容的非 Python 文件，逐个显式放行并记进 `unshipped_allowed`。9fd3822a00 → 2f22a98d6e 之间有一个：`tensorrt_llm/grpc/openengine/README.md`（`setup.py` 的 `package_data` 不含它）。放行一个没有变的路径会报错。
- `--sites LIST`：仅 P2；从打开组的位点里选出本场景会经过的那些（如单前端不经过 `Lmfl`、`Lmfa`），记 `sites_selection: "cli"`。不给时列出打开组的全部位点。
- 清单另加字段：`image_is_ancestor_of_target`（9fd3822a00 不是 2f22a98d6e 的祖先，镜像提交是侧枝）、`unshipped_allowed`、`sites_selection`、`patchcheck`（P1/P2 的检查摘要）、`created_wall`。
- P1/P2 要求：探针树 HEAD 包含基线提交、`tensorrt_llm/` 工作树干净、`patchcheck` 对 HEAD 通过；`--packages` 必须含 `capprobe` 与 `capcommon`；`--capkit-sha` 必须等于 `<C>` 的 HEAD，且所列包没有未提交改动。每个文件都逐字节核对 `src` 与提交里的 blob 一致。

## patchcheck 检查什么

相对基线（缺省比工作树，即覆盖时实际会拷贝的内容）：只改 `tensorrt_llm/**.py`、只增不删；每个改动文件能编译；`_capprobe` 只出现在模块级 `import capprobe as _capprobe`、`if` 守卫、守卫体内唯一的那条位点调用里；找到的位点与 `SITES` 逐个对上：每个位点只在它的 `file` 里、恰好出现在 `lines` 所列的每一行（应检的位点数与位置数取自 `SITES`，位点数为 0 记 NOT_RUN）；每一行新增都属于导入或某个位点。

`tests/capprobe` 里读真实补丁的用例（`test_patchcheck.py::test_real_probe_tree_cli`、`test_overlay.py::test_real_image_to_baseline_range`、`test_probe_tree_sites.py`）依赖仓库旁的探针树 `probe-tree/`（TRT-LLM worktree，分支 capprobe-J17871）；探针树缺失时这些用例报错，不跳过。
