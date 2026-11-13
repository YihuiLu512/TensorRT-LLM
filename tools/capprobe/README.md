# capprobe（B3）

产品进程里的探针模块（不依赖 `tensorrt_llm`），外加三个工具：快照解码、补丁静态检查、覆盖清单生成。约定以 `INTERFACES.md` §6、§7、§11 为准；本文件只列本包的入口、位点与新增的可选参数。

## 入口

| 入口 | 用途 |
|---|---|
| `import capprobe`（产品代码，经探针补丁） | 导入时读一次 `CAPPROBE_*` 环境变量（§6.5）；打开的组与注入决定哪些位点生效 |
| `python -m capprobe.decode FILE [--no-hist]` | 把一个快照解码成 JSON；坏文件退出码 3，读不到 4 |
| `python -m capprobe.patchcheck --probe-tree P --base-commit SHA [--rev REV] [--out F]` | 补丁静态检查（见下）；PASS 退出 0，FAIL / NOT_RUN 退出 3 |
| `python -m capprobe.overlay …`（§11 的参数） | 生成 `env/overlay_manifest.json`（§6.6） |

工具进程（`python -m capprobe.decode|overlay|patchcheck`）按原始命令行里 `-m` 后的模块名进入「工具模式」：不读环境变量、不起探针，所以在 P2 的环境里运行也不会在快照目录里多出一个 `<pid>.bin`。其他任何进程导入 `capprobe` 都会严格校验开关。

## 开关的补充规则（在 §6.5 之上，全部是导入时 `CapprobeConfigError`，它是 `RuntimeError`）

- `CAPPROBE_TIER` 未设置也报错（不悄悄关掉）。`P0` 下导入 `capprobe` 报错：P0 的覆盖层不应含探针补丁。辅助进程若会导入打过补丁的模块（如 import 了 `tensorrt_llm.serve`），要显式设 `CAPPROBE_TIER=P1`、其余 `CAPPROBE_*` 不设；**`CAPPROBE_*` 只能给被测产品进程**，否则辅助进程也会写快照（判定器会把它当作未登记进程）。
- 本版本只实现 `count`、`lag` 两组；`bytes`、`queue`、`ts`、`pin` 是已知组名但未实现，打开即报错。
- 任何组或注入打开时，`CAPPROBE_SHM_DIR`（绝对路径）、`CAPPROBE_INTERVAL_MS`、`CAPPROBE_IDENTITY_FILE` 必需；`lag` 打开时 `CAPPROBE_LAG_PERIOD_US` 必需。
- 注入只支持位点 `H1`、`H3`、`F2`、`F4`，模式只支持 `busy`、`sleep`、`miscount`；`miscount` 要求 `count` 组打开（否则误计不可见）。注入打开时，四个位点都会经过注入器，没有规格的位点也计 `inject.<site>.hits`。

## 位点（`capprobe.SITES`；行号是打过补丁的探针树里守卫 `if` 所在行）

| 位点 | 文件:行 | 槽 | 一次经过的含义 |
|---|---|---|---|
| H1 | `tensorrt_llm/serve/responses_utils.py:3539` | `count.H1.arrivals` | 每个 `http` 类型的 ASGI 请求（含健康检查） |
| H3 | `tensorrt_llm/serve/openai_server.py:3064` | `count.H3.chunks` | `completion_generator` 每个 `pp_res`（只有 `/v1/completions` 流；chat 流未打点） |
| F2 | `tensorrt_llm/grpc/openengine_client/stream.py:267` | `count.F2.msgs` | `_read` 读到的每条消息（不含 EOF 与本地取消；含终止事件之后排空读到的） |
| F4 | `tensorrt_llm/serve/openengine_poc/facade.py:1628` | `count.F4.dequeued` | 非 postproc 池路径每个出队的事件项（结束标记与错误项不计） |
| Lfe | `tensorrt_llm/serve/openengine_poc/launch.py:135` | `lag.Lfe`（hist）及 `.n/.sum/.max/.neg`、`lag.Lfe.installs` | 单前端事件循环 |
| Lmfl | `tensorrt_llm/serve/openengine_poc/multi_frontend.py:485` | `lag.Lmfl` … | 多前端的 launcher 循环 |
| Lmfa | `tensorrt_llm/serve/openengine_poc/multi_frontend.py:649` | `lag.Lmfa` … | 多前端的附属前端循环 |

注入心跳槽：`inject.<H1|H3|F2|F4>.{hits,applied,applied_ns}`。机器可读清单：`capprobe.SLOTS`（组打开时在导入时一次性登记，没经过的位点表现为「槽在、值为 0」）。`SITE_NOTES` 记着设计稿在基线提交上的行号，仅供对照。

## 运行时

- 位点写法：`if _capprobe.ON_F2: _capprobe.F2()`；关闭时只剩一次模块属性读取与一次分支。
- 快照：后台线程每 `CAPPROBE_INTERVAL_MS` 写 `<CAPPROBE_SHM_DIR>/<pid>.bin`（`capcommon.snapshot`），进程正常退出时再写一份终值。元数据里 `identity` 取自 `CAPPROBE_IDENTITY_FILE`（`pid` 换成本进程的），另有 `probe`（档位、组、注入规格、`sites_on`）。
- 版本自报（D11）：导入时写一份（`loop_impl` 为 null）；`install_loop_lag` 之后带上循环类与 `probe_loops`；产品在 `capprobe` 之后才 import `grpc` 的，写快照线程发现后自动刷新 `grpc`、`protobuf`、`protobuf_impl`。
- `fork`：子进程里计数清零、换成子进程自己的 pid 与写快照线程、辅助线程重建；事件循环探针不继承。
- `install_loop_lag(site)` 必须在运行中的循环里调用；同一进程同一安装点装两次报错。

## 覆盖清单（`capprobe.overlay`）的补充参数

- `--export-dir DIR`：P0 必需。探针树的工作树是打过补丁的，P0 的文件从基线提交的 blob 导出到这里，`src` 指向导出的文件。
- `--allow-unshipped PATH`（可重复）：`tensorrt_llm/` 下改了、但不是安装包内容的非 Python 文件，逐个显式放行并记进 `unshipped_allowed`。9fd3822a00 → 2f22a98d6e 之间有一个：`tensorrt_llm/grpc/openengine/README.md`（`setup.py` 的 `package_data` 不含它）。放行一个没有变的路径会报错。
- `--sites LIST`：仅 P2；从打开组的位点里选出本场景会经过的那些（如单前端不经过 `Lmfl`、`Lmfa`），记 `sites_selection: "cli"`。不给时列出打开组的全部位点。
- 清单另加字段：`image_is_ancestor_of_target`（9fd3822a00 不是 2f22a98d6e 的祖先，镜像提交是侧枝）、`unshipped_allowed`、`sites_selection`、`patchcheck`（P1/P2 的检查摘要）、`created_wall`。
- P1/P2 要求：探针树 HEAD 包含基线提交、`tensorrt_llm/` 工作树干净、`patchcheck` 对 HEAD 通过；`--packages` 必须含 `capprobe` 与 `capcommon`；`--capkit-sha` 必须等于 `<C>` 的 HEAD，且所列包没有未提交改动。每个文件都逐字节核对 `src` 与提交里的 blob 一致。

## patchcheck 检查什么

相对基线（缺省比工作树，即覆盖时实际会拷贝的内容）：只改 `tensorrt_llm/**.py`、只增不删；每个改动文件能编译；`_capprobe` 只出现在模块级 `import capprobe as _capprobe`、`if` 守卫、守卫体内唯一的那条位点调用里；找到的位点与 `SITES` 逐个对上（应检数取自 `SITES`，为 0 记 NOT_RUN）；每一行新增都属于导入或某个位点。
