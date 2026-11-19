# capkit 接口约定（INTERFACES.md）

版本：`capkit.interfaces/1` 修订 r2（2026-11-13；r1 为首版，r2 按审阅意见原地修订，见文末「勘误与补充」与「审阅记录」）。本文件是各组件之间唯一的、有约束力的约定：文件格式、目录树、命令行、退出码、开关的名字与语义。组件的作者照本文件写代码，**不需要彼此沟通**；本文件与设计稿（`09_experiment_design.md`）在「方法定义」上冲突时以设计稿为准，在「格式与接口」上以本文件为准。改本文件只在末尾「勘误与补充」追加，带日期与原因；改了格式就升 `schema` 的版本号。

共享库 `tools/capcommon/` 是本文件的代码化部分：凡是本文件说「用 `capcommon.X`」的地方，组件必须调用它，不得自己另写一份同功能的实现（这样格式只有一个实现）。

<!-- SECTIONS BELOW -->

## 0. 名词与通用规则

| 名词 | 含义 |
|---|---|
| `<C>` | 代码仓库 `01_code/`（实际路径 `/home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02/01_code`） |
| `<W>` | 工作目录 `test-02/`（`<C>` 的父目录） |
| 一次启动（launch） | 起一组进程 → 预热 → 档位表逐档 → 停止 → 落盘（设计稿 §2.8）。一次启动一棵数据树 |
| 启动树 `<T>` | `runs/J<作业号>-r<重排次数>-L<启动序号>_<臂名>/`（DATA-AND-ANALYSIS §3）。**目录名只给人看，任何代码都不从目录名或文件名里解析身份**，身份一律从文件内容的 `identity` 读 |
| 角色（role） | 一次启动里的一类进程：`capload`、`nullsse`、`frontend`、`stub`、`relay1`、`relay2`、`sidecar`、`canary`…；正则 `^[a-z][a-z0-9_-]{0,31}$`。**拓扑位置不同的进程用不同的角色名**（例：两级串联的中继是 `relay1` 与 `relay2`，注入、清空门、判定器的定位都按角色区分它们）；**可以互换的同类进程**（同一负载器的多个 worker、多前端里的各个前端）用 `worker` 序号区分（0 起），不另起角色名。编排器对每个角色只起一个顶层进程；多进程由该进程自己派生（§5.2 的 `workers`、`expected_procs`） |
| 原点 origin | 本次启动的时间零点：一个 `time.monotonic_ns()` 值。**编排器在起第一个进程之前选定** `origin = now + lead`（`lead` 由 `--origin-lead-ms` 显式给出），先写进 `raw/launch.json`，再用 `--origin-ns` 传给每个工具；工具不等任何「开始」信号，按档位表在原点处自己开始。原点到达时还有角色没写就绪文件，编排器就停止本次启动（`stop.json` 的 `reason: "not_ready_before_origin"`），整次启动记 `NOT_RUN`。档位表与时间表里的时刻都是**相对原点的 ns**（`*_rel_ns`） |
| 档（tier） | 档位表里的一行：预热 / 上行 / 下行 / 清空四种 |

通用规则（每个组件都适用；评审时按这张表逐条核对）：

1. **时间**：一律 `time.monotonic_ns()`（`capcommon.clock.now_ns`），整数 ns。墙钟只用于人读的时间戳与台账（`capcommon.clock.wall_iso`），不参与任何差值计算。同一节点上不同进程、不同容器共用同一个 `CLOCK_MONOTONIC`【推断：同一内核；E0 的时钟自检实测】。
2. **时延从计划时刻起算**（协同遗漏）：每个请求、每条消息都同时记「计划时刻」与「实际时刻」，两者都落盘；不得只记实际时刻。
3. **测量热路径上不写盘**：逐请求 / 逐消息的记录进 `capcommon.capbin.RecordArray`（预分配），在启动结束时落盘。允许的运行中 I/O 只有：后台线程每个快照周期写一次 `/dev/shm` 快照（§6）；独立进程 sidecar、canary 的逐秒序列（它们自己不是被测对象）；启动时写端口文件与就绪文件。
4. **不写死**路径、资源、硬件规格：全部是命令行参数（或本文件列出的环境变量），没有隐含默认值的路径。数值型的「方法参数」（快照周期、轮询周期）也一律显式传入；工具对缺失的必需参数以退出码 2 退出。
5. **身份显式传入并写进每个输出文件**（§2.1）：实验、臂、启动序号、作业号、重排次数、登记表 sha、角色由调用方传入；档序号与档标签由档位表给出。不按名字子串决定任何控制流。
6. **「零异常即通过」的检查都带应检条数**：应检条数来自配置（档位表、时间表、进程数），不由检查者自己数；条数为 0 判 `NOT_RUN`，不判通过。
7. **不依赖**标准库、`grpcio`、`protobuf`、`psutil` 之外的包。`uvloop` 只能可选（`--loop uvloop` 时才 import；导入失败以退出码 3 退出，不悄悄退回 asyncio）。热路径上不用 `numpy`。`capcommon`、`capjudge`、`capjobs`、分析脚本只用标准库（要在 agent VM 的 `python3.11` 上跑）。
8. **原子写**：所有 JSON / 二进制结果文件先写同目录的临时名再 `os.replace`（`capcommon.jsonio.atomic_write_bytes`）。读者永远看不到半个文件。
9. 新源文件开头两行：`# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` 与 `# SPDX-License-Identifier: Apache-2.0`。

## 1. 仓库布局、包名与调用方式

### 1.1 目录

```
<C>/
├── INTERFACES.md          本文件
├── README.md / REVIEW.md
├── tools/                 全部 Python 包（每个子目录是一个顶层包）
│   ├── capcommon/         共享库（Spec 负责）
│   ├── capload/           B1：HTTP SSE 负载器（capload）与空汇（capload.nullsse）
│   ├── capgrpc/           B2：grpc 负载器 / 排空端（capgrpc.load）、空汇 / 节拍源（capgrpc.nullgrpc）；经产品客户端读流的 capgrpc.oe_sink 见勘误 FT1-8
│   ├── capselfcheck/      B2：已知瓶颈中继（capselfcheck.relay）
│   ├── capprobe/          B3：探针模块（被产品进程 import）、快照解码命令、覆盖清单生成
│   ├── capenv/            B4：sidecar_v4、金丝雀、环境门禁、硬件画像、绑核表、D14 小程序
│   ├── capjudge/          B5：判定器、统计量 T1～T6、试点构建（pilot.json）
│   ├── caplaunch/         B7：编排器 capkit-launch
│   ├── capstubs/          B7（本批只有 fingerprint）；第二批的桩也放这里（S-PB 见勘误 B2B-1）
│   └── capjobs/           B6：submit.py 的 Python 部分（在 VM 上跑）
├── tests/<包名>/          各包的 pytest；每个目录自带 conftest.py 把 <C>/tools 放进 sys.path
├── jobs/                  B6：sbatch、容器内脚本（*.sh）、tokenizer 拷贝作业；占 GPU 的模板只以 .draft 存在
├── configs/               B7：场景配置、档位表与时间表的生成参数；tokenizers/<名字>/（只含非权重文件）
├── samples/               B5：判定器自测样本（良品、每项一个造坏的、空目录、两个可满足性样本）
├── analysis/              第二批：分析脚本
├── _progress/<组件>.md     各组件的进度记录（中文）
├── probe-tree/            TRT-LLM 探针树（git worktree，不纳入本仓库）
└── probe-tree-glm/        glm 探针树（同上）
```

### 1.2 包名与调用方式（选定：`python -m <包>[.<模块>]`）

- 每个工具都是一个**带 `__main__` 的模块**，入口一律 `python -m <包>.<模块>`（包本身就是工具时 `python -m <包>`）。全部包名以 `cap` 开头，避免拷进 `dist-packages` 后与镜像里已有的包重名（`stubs`、`selfcheck` 这类通用名都可能撞名）。
- **在仓库里跑**：`PYTHONPATH=<C>/tools python -m capload …`（测试由 `tests/<包>/conftest.py` 插 `sys.path`）。
- **在容器里跑**：作业的内层脚本把 `tools/` 下本次要用的包整目录拷进容器的 `dist-packages`（覆盖清单里逐文件列出 sha256，§6.6），并 `unset PYTHONPATH`；之后 `python -m capload …` 原样可用。
- 理由：(1) 同一条命令在仓库与容器里都成立，不靠路径拼接；(2) `capprobe` 必须能被产品代码按名字 import——包括 MPI 拉起的 worker、`spawn` 的池子进程、被重新执行的附属前端，它们不继承 `PYTHONPATH`（设计稿 §2.4），拷进 `dist-packages` 是唯一对它们都成立的做法，其余包随它用同一方式部署，免得两套规则；(3) 包内用相对 import，模块之间不靠 `sys.path` 技巧；(4) `multiprocessing` 的 `spawn` 子进程能按模块名重新导入入口。
- 包之间的依赖只允许指向 `capcommon`（以及 `capjudge` 可以 import `capenv` 的纯解析函数、`caplaunch` 可以 import 其他包的「常量与解析函数」；`capstubs`、`capselfcheck` → `capgrpc` 的例外见勘误 B2B-1）；**`capprobe` 只允许 import `capcommon.snapshot`、`capcommon.hist`、`capcommon.inject`、`capcommon.clock`、`capcommon.versions`（以及它们依赖的 `capcommon.exitcodes`）**（被产品进程 import，导入面要小；这几个模块只依赖标准库）。
- 每个包有 `README.md`（中文）列出自己的入口与本文件之外新增的可选参数。可选参数不得改变本文件规定的任何输出格式与语义。

## 2. 通用命令行约定

所有在启动树里产出数据的工具都用 `capcommon.cli.add_common_args(parser, …)` 注册下面这些参数，用 `capcommon.cli.common_from_args(args)` 取回并校验。参数名、含义、校验规则只在这里定义。

### 2.1 身份参数（全部必需）

| 参数 | 类型与校验 | 写进 `identity` 的字段 |
|---|---|---|
| `--experiment` | `^E\d{1,2}[A-Za-z0-9-]*$`（如 `E0`、`E2a`、`E15-A`） | `experiment` |
| `--arm` | `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$` | `arm` |
| `--launch-seq` | 整数 ≥ 0 | `launch_seq` |
| `--job-id` | `^[A-Za-z0-9._-]{1,64}$`（Slurm 作业号；本机测试用 `local-<任意>`） | `job_id` |
| `--restart-count` | 整数 ≥ 0（`${SLURM_RESTART_COUNT:-0}`） | `restart_count` |
| `--register-sha` | 64 位小写十六进制（登记表 `register.json` 的 sha256，§9.2） | `register_sha` |
| `--role` | 角色名正则见 §0 | `role` |

`identity` 对象（`capcommon.identity.Identity`）在每个输出文件里出现时还会补上：`worker`（同角色多进程时的序号，单进程为 0）、`pid`、`tier_seq` 与 `tier_label`（只在逐档文件里有，其余为 `null`）。字段集合固定：

```json
{"experiment":"E0","arm":"nullsse-a","launch_seq":2,"job_id":"5601234","restart_count":0,
 "register_sha":"<64hex>","role":"capload","worker":1,"pid":12345,"tier_seq":3,"tier_label":"up-r2000"}
```

多进程工具的父进程用 `common_from_args` 得到自己的 `Identity`，每个子进程用 `identity.with_worker(k).with_pid()` 换成自己的序号与 pid；写逐档文件时再 `with_tier(seq, label)`。

**读者一律用 `identity` 字段**，不从路径反推。判定器核对同一棵树里全部文件的前六个字段（experiment、arm、launch_seq、job_id、restart_count、register_sha）彼此相同，且与 `raw/launch.json` 相同；不同即该树 `INPUT_INVALID`（§10）。

### 2.2 其余通用参数

| 参数 | 谁要 | 含义 |
|---|---|---|
| `--run-dir PATH` | 全部 | 启动树根 `<T>`（作业运行中是节点本地盘上的那一份）。工具只往 §5 指定给自己角色的位置写 |
| `--worker N` | 由编排器或父进程起的单个工作进程 | 写进 `identity.worker` |
| `--loop {asyncio,uvloop}` | 全部 asyncio 工具，**必需，无默认值** | 事件循环实现。实际用到的实现（`type(loop)` 的模块与类名）写进版本自报（§2.5） |
| `--tier-table PATH` | 跟档位表走的工具（负载器、空汇、节拍源、排空端、中继） | 档位表（§3） |
| `--origin-ns INT` | 同上，以及 sidecar、金丝雀 | 原点的 `monotonic_ns` 值（编排器在起进程之前选定，§0） |
| `--shm-dir PATH` | 全部发快照的进程 | 快照目录（§6）。**每次启动一个子目录**：`/dev/shm/<用户名>-<作业号>/snap/r<重排次数>-L<启动序号>`（`capcommon.runtree.launch_shm_subdir`），写进 `launch.json` 的 `shm_dir`；同一作业里前一次启动留下的快照文件因此不会被这一次的编排器、清空门读到 |
| `--snapshot-interval-ms INT` | 同上 | 快照周期，> 0 |
| `--port-file PATH` | 服务端 | 绑定成功后写端口文件（§2.3） |
| `--ready-file PATH` | 全部长驻进程 | 就绪后写就绪文件（§2.3） |
| `--cpus LIST` | 全部 | 本进程（多 worker 工具：第 k 个 worker 用列表第 k 项）绑到的逻辑核；格式 `3,5,7-9`（`capcommon.cli.parse_cpu_list`）。工具自己调用 `os.sched_setaffinity` 并把读回值写进就绪文件；多 worker 工具要求列表长度 ≥ worker 数 |
| `--inject SPEC` | 全部（可重复） | 启动级注入（§7）；逐档注入在档位表里 |
| `--stop-grace-s FLOAT` | 全部长驻进程 | 收到停止信号后最多等多久再落盘退出 |

场景参数里指向作业输入（指纹、请求字节、预制字节）的占位符 `{input:fingerprint}`、`{input:request_bytes}`、`{input:prebaked}` 见勘误 B2B-9。

### 2.3 端口文件、就绪文件、停止

- **端口文件**：服务端绑定 `127.0.0.1:0`（或 `--host`/`--port` 显式给定）后，用 `capcommon.lifecycle.write_port_file(path, identity, host, port)` 写 JSON：`{"schema":"capkit.portfile/1","identity":…,"host":…,"port":…,"pid":…,"t_mono_ns":…}`。多进程共享端口（`SO_REUSEPORT`）的服务：父进程先绑定拿到端口，再起子进程绑同一端口，全部子进程就绪后才写端口文件。读：`capcommon.lifecycle.wait_port_file(path, timeout_s)`。
- **就绪文件**：进程准备好接收 / 发送后写 `capcommon.lifecycle.write_ready_file(path, identity, versions, affinity=None, children=None, info=None)`：`{"schema":"capkit.ready/1","identity":…,"t_mono_ns":…,"versions":<§2.5>,"affinity":[…读回的核…],"children":[{"pid":…,"worker":…,"role":…}],"info":{…工具自定…}}`。**多进程工具必须在 `children` 里列出每个子进程的 pid**——编排器据此登记「pid → 角色」（§5.3），sidecar 不按命令行认进程。（例外：`capenv.spinners` 的自旋进程不列，见勘误 SB-B1-4。）
- **停止**：`SIGTERM` 或 `SIGINT` = 优雅停止：不再发新请求 / 不再接新流，在 `--stop-grace-s` 内收尾，落盘全部记录与自报，退出。第二次信号 = 立即落盘已有记录并退出。用 `capcommon.lifecycle.StopController` 安装处理器。编排器在发信号之前先写 `<T>/control/stop.json`（`{"schema":"capkit.stop/1","reason":…,"t_mono_ns":…,"after_tier_seq":…}`）；工具落盘时若该文件存在，把它的内容原样抄进自己的末尾自报（`stop` 字段）。`<T>/control/` 在 `raw/` 之外，只是编排器到工具的单向信道。
- 负载器、排空端按档位表跑完最后一档（加 `--stop-grace-s` 的收尾）后自己退出；服务端、桩、中继、sidecar、金丝雀只在收到信号时退出。

### 2.4 退出码（`capcommon.exitcodes`）

| 码 | 名字 | 含义 |
|---|---|---|
| 0 | `OK` | 正常结束（服务端被信号停止也是 0） |
| 1 | — | 未捕获的异常（Python 默认） |
| 2 | `USAGE` | 命令行错误（argparse 默认码） |
| 3 | `CONFIG` | 配置或输入文件无效：档位表 / 时间表校验失败、sha 不符、身份字段非法、`--loop uvloop` 而 uvloop 不可导入 |
| 4 | `RUNTIME` | 运行期失败：绑定端口失败、对端连不上、子进程异常退出 |
| 5 | `STOPPED_EARLY` | 负载器 / 排空端在时间表跑完之前被停止（自报里的 `stop` 字段给原因） |
| 6 | `ENV_INVALID` | 环境门禁不过（只由 `capenv.gate` 使用） |

判定器另有自己的退出码（§10.3）：0、1、2 的含义与上表相同，失败码用 10～13，与上表不重叠。

### 2.5 版本自报（D11）

每个被测进程与每个 capkit 进程在就绪时调用 `capcommon.versions.version_record(loop=…, extra=…)`，得到：

```json
{"python":"3.12.3 …","implementation":"cpython","executable":"/usr/bin/python3","pid":123,
 "grpc":"1.66.0"|null,"protobuf":"5.27.0"|null,"protobuf_impl":"upb"|"python"|"cpp"|null,
 "loop_impl":"uvloop.Loop"|"asyncio.unix_events._UnixSelectorEventLoop"|null,
 "gc_enabled":true,"switchinterval_s":0.005,
 "capkit_code_sha":"<40hex>"|null,"probe_tier":"P0"|"P1"|"P2"|null,"probe_patch_sha":"<40hex>"|null,
 "probe_groups":["count","lag"],"env":{"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION":null,…}}
```

`capkit_code_sha`、`probe_tier`、`probe_patch_sha`、`probe_groups` 由环境变量 `CAPKIT_CODE_SHA`、`CAPPROBE_TIER`、`CAPPROBE_PATCH_SHA`、`CAPPROBE_GROUPS` 读（作业内层脚本显式设置；编排器给每个角色设置版本自报变量，见勘误 R4-3）；缺失就写 `null`。`grpc` / `protobuf` 没装或本进程不用时写 `null`（`version_record(include_grpc=False)`），不 import 失败。

**C10 怎样判（D11）**：必需字段**按角色**定，不是一张表套全部进程。场景配置给每个角色声明 `asyncio`、`uses_grpc`、`loop`（`asyncio` / `uvloop` / `null`）、`probe_tier`（产品进程写 `P0`/`P1`/`P2`，capkit 工具写 `null`）（§11 场景配置），`capcommon.versions.required_fields(声明)` 给出该角色的必需字段：

| 条件 | 必需字段 |
|---|---|
| 全部进程 | `python`、`implementation`、`gc_enabled`、`switchinterval_s`、`capkit_code_sha` |
| `asyncio: true` | 再加 `loop_impl`，且它必须与声明的 `loop` 一致（`asyncio.` / `uvloop.` 开头） |
| `uses_grpc: true` | 再加 `grpc`、`protobuf`、`protobuf_impl` |
| `probe_tier` 为 `P1` / `P2` | 再加 `probe_tier`、`probe_patch_sha`（`P0` 只加 `probe_tier`） |

**必须跨进程一致**的字段（出现即比，`capcommon.versions.CONSISTENT_FIELDS`）：`python`、`implementation`、`grpc`、`protobuf`、`protobuf_impl`、`capkit_code_sha`、`probe_tier`、`probe_patch_sha`。**允许不同**：`loop_impl`（按角色；只核对与该角色的声明一致）、`gc_enabled`、`switchinterval_s`（记进维度表，D9、D13）、`pid`、`executable`、`env`。应检进程 = `launch.json` 的 `expected_procs` 之和（不是「报上来的进程」）；缺版本记录也算一条问题。判定由 `capcommon.versions.check_versions(records, decls)` 给出；**有任何问题 ⇒ 整次启动 `NOT_RUN`**（D11 原文「缺项或不一致 ⇒ NOT RUN」），不是红。S-PB 前端模式加载的 S-FE 模块 sha 也在 C10 里核（见勘误 B2B-11）。

## 3. 档位表（tier table）

一次启动的全部负载与节拍由**一份**档位表决定（设计稿 §2.2 第 10 条）。它是登记文件：写好后算 sha256 进 `REGISTRY.tsv`，之后不改。档位表里**没有绝对时刻**——原点在启动时才定，用 `--origin-ns` 传入，所以同一份档位表可以被多次启动复用。读写与校验只用 `capcommon.tiers`：`TierTable.load(path)`（校验失败抛 `TierTableError`，工具以退出码 3 退出）。

### 3.1 格式（`schema: capkit.tiers/1`，JSON）

```json
{
  "schema": "capkit.tiers/1",
  "experiment": "E0",
  "name": "e0a-nullsse-subsat",
  "arrival": {"dist": "fixed", "seed": 20261113},
  "axis": {"name": "R", "load_key": "rate_rps", "unit": "req/s", "order": "asc"},
  "request_timeout_ns": 30000000000,
  "beat_origin_rel_ns": 0,
  "default_gate": { …§8.1… },
  "launch_inject": [ …§7 的 InjectSpec… ],
  "tiers": [
    {"seq": 0, "label": "warmup", "kind": "warmup",
     "start_ns": 0, "ramp_ns": 5000000000, "window_ns": 20000000000, "tail_ns": 5000000000,
     "rate_rps": 50.0, "streams_c": null,
     "max_tokens": 33, "tokens_per_msg": 1, "stream_interval": 1, "beat_ns": 20000000,
     "prompt_set": "tok128", "inject": [], "gate": null, "baseline": false, "note": "",
     "load_value": 50.0},
    {"seq": 1, "label": "drain-01", "kind": "drain",
     "start_ns": 30000000000, "ramp_ns": 0, "window_ns": 30000000000, "tail_ns": 0,
     "rate_rps": 0.0, "streams_c": null,
     "max_tokens": 0, "tokens_per_msg": 1, "stream_interval": 1, "beat_ns": 20000000,
     "prompt_set": null, "inject": [], "gate": null, "baseline": false, "note": "",
     "load_value": null}
  ]
}
```

| 字段 | 含义 |
|---|---|
| `arrival.dist` | `fixed`（定间隔，定上限用）或 `poisson`（只作敏感性）。一张档位表只有一种 |
| `arrival.seed` | 整数；泊松的随机数种子（定间隔也必须写，便于核对） |
| `axis` | 本表扫的轴（设计稿 §2.1）：`name` ∈ `R`/`M`/`K`/`L`；`load_key` ∈ `rate_rps`、`nominal_msgs_per_s`（= `streams_c × 10⁹ ÷ (beat_ns × stream_interval)`，即设计稿的名义 μ = C ÷ (Δ·s)）、`streams_c`、`beat_ns`、`tokens_per_msg`、`prompt_len`；`unit` 文字单位；`order` ∈ `asc`（`load_value` 越大负载越重）/ `desc`（越小越重，如扫 `beat_ns`）。判定器的 T4 区间、D10 迟滞配对（按 `(load_value, kind)`）都只用它排序，不各自猜 |
| `request_timeout_ns` | 负载器「事先登记的超时」（失败原因 `timeout_registered`，过载型，设计稿 §2.6）；`null` = 不设超时。它在档位表里所以随档位表登记，负载器只从这里读，不接受命令行传入 |
| `beat_origin_rel_ns` | 节拍原点，必须等于 `tiers[0].start_ns`（= 0）。显式写出是为了让读档位表的人知道节拍从哪起算 |
| `default_gate` | 清空档没写 `gate` 时用它（§8） |
| `launch_inject` | 整次启动不变的注入（§7）；探针（`capprobe`）一侧只认这一种 |
| `seq` | 0 起连续整数 |
| `label` | `^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$`，表内唯一 |
| `kind` | `warmup` / `up` / `down` / `drain` |
| `start_ns` | 相对原点。`tiers[0].start_ns == 0`；**相邻档首尾相接**：`start[i+1] == start[i] + ramp + window + tail` |
| `ramp_ns`、`window_ns`、`tail_ns` | 爬升段、稳态窗口、收尾段的长度。稳态窗口 = `[start+ramp, start+ramp+window)`。清空档只用 `window_ns`，它就是清空门的最长等待（`ramp_ns == tail_ns == 0`） |
| `rate_rps` | 本档的到达率 λ（请求/s）。负载档 > 0；清空档 == 0。到达覆盖本档全程 `[start, end)`（爬升、窗口、收尾都按同一 λ 发） |
| `streams_c` | M 轴的名义在途流数 C（R 轴写 `null`）。给了就必须满足 `rate_rps == C ÷ 流时长`（相对误差 ≤ 1e-9），流时长 = `ceil(max_tokens ÷ tokens_per_step) × beat_ns`；由生成档位表的脚本算好，校验器只核对 |
| `max_tokens` | 每请求输出 token 数 N（负载档 ≥ 1；清空档 0） |
| `tokens_per_msg` | k：每条响应消息的 token 数 |
| `stream_interval` | s：每隔几步出一条消息。必须 `tokens_per_msg % stream_interval == 0`；每步 token 数 `tokens_per_step = k ÷ s`（`Tier.tokens_per_step`） |
| `beat_ns` | 本档桩 / 节拍源的步周期 Δ（0 = 本档不用节拍）。> 0 时本档总时长必须是 `beat_ns` 的整数倍；**清空档的 `beat_ns` 必须等于前一档的**（上一档没走完的流在清空档里按原节拍继续出 token） |
| `prompt_set` | prompt 池里的集合名（§4.3）；清空档为 `null` |
| `inject` | 本档生效的注入列表（只对读档位表的 capkit 工具有效，§7） |
| `gate` | 清空档的清空门配置（§8）；负载档必须是 `null` |
| `baseline` | `true` = 本档是「同一次启动的亚饱和基线档」（T2、X1 归一用）。每张表至多一个；判定器要求被判的表恰好有一个 |
| `note` | 自由文本 |
| `load_value` | 本档在 `axis` 上的负载值（生成脚本算好写入）。`up`/`down` 档必需（> 0）；`warmup` 档可为 `null`；`drain` 档必须为 `null`。校验器按 `load_key` 用本档原始参数重算并核对（相对误差 ≤ 1e-9）；`prompt_len` 的长度不在档位表里，只核对它是正整数 |

### 3.2 校验规则（`TierTable.load` 全部执行；任一不过即 `TierTableError`）

1. `schema` 正确；必需字段齐全，类型正确（整数字段不接受浮点、布尔）。
2. `seq` 从 0 连续；`label` 唯一且合法；`tiers[0].start_ns == 0 == beat_origin_rel_ns`；相邻档首尾相接。
3. 第一档必须是 `warmup`。每个 `up` / `down` 档的前一档必须是 `drain`（档与档之间都有清空门）。最后一档必须是 `drain`（最后一档走完再停）。
4. 负载档：`rate_rps > 0`、`max_tokens ≥ 1`、`window_ns > 0`、`prompt_set` 非空、`gate` 为 `null`。清空档：`rate_rps == 0`、`max_tokens == 0`、`ramp_ns == tail_ns == 0`、`window_ns > 0`、有 `gate` 或表里有 `default_gate`。
5. `tokens_per_msg ≥ 1`、`stream_interval ≥ 1`、`tokens_per_msg % stream_interval == 0`。
6. `beat_ns ≥ 0`；`beat_ns > 0` 时档总时长整除 `beat_ns`；清空档的 `beat_ns` 等于前一档的。
7. `streams_c` 给了就核对与 `rate_rps` 一致（见上表）；这时 `beat_ns` 必须 > 0。
8. 注入条目逐条按 §7 校验。
9. `baseline` 为真的档至多一个，且必须是 `up` 或 `down` 档（预热档不作基线）。
10. `axis` 四个键齐全且取值合法；`request_timeout_ns` 为 `null` 或正整数；每档 `load_value` 满足上表，并与按 `load_key` 重算的值一致（`nominal_msgs_per_s`、`streams_c` 要求本档有 `streams_c` 与 `beat_ns > 0`）。
11. 清空门配置（§8.1）的 `poll_ns`、`settle_polls`、`max_snapshot_age_ns` 为正整数。

### 3.3 各工具怎样用档位表（不重启换档）

- 每个工具在启动时读一次档位表，用 `--origin-ns` 把相对时刻换成绝对时刻：`abs = origin_ns + rel_ns`（`TierTable.abs_ns`）。
- **换档没有任何进程间通信**：到了下一档的开始时刻，各工具各自切换（负载器按时间表发、节拍源按本档 Δ 出步、注入按本档 `inject` 换）。热路径上判断是否换档只做一次整数比较（与「下一档的开始时刻」比），换档动作由定时回调或节拍线程在边界处做。
- 查询：`tier_at(rel_ns)` 给出该时刻所在的档（原点之前 → `None`，最后一档之后 → `None`）；`window_of(seq)` 给出稳态窗口的相对起止。
- 清空门的判定与「门不过就停止」由编排器做（§8）；工具只需要按档位表照常运行、按 §6 暴露积压读数、收到停止信号时收尾。

### 3.4 节拍（桩、节拍源）的计划时刻

步的计划时刻由档位表决定，算法在 `capcommon.schedule.BeatSchedule`（全部节拍源、桩与按序号算滞后的读者都用它）：

- 只有 `beat_ns > 0` 的档有步。档 i 的第 j 步（j = 0, 1, …）的计划时刻 = `start_i + j × beat_ns`，j 取到档末之前为止；全局步号按档顺序累计（`first_step(seq)`）。
- 因为清空档沿用前一档的 Δ、且每档时长整除 Δ，所以同一 Δ 的相邻档之间节拍连续，没有相位跳变。
- `planned_rel_ns(step)` 与 `step_at_or_after(rel_ns)` 是一对互逆的查询；流的「首步」`first_step = step_at_or_after(接纳时刻的 rel_ns)`，由接纳该请求的一方（capkit 的空汇、节拍源；第二批的桩）记进逐请求记录 `streams`（§5.4.2）。
- **一条流在哪些步出消息、每条几个 token**只由 `capcommon.expect` 算（§5.8）：第 j 条消息在步 `first_step + msg_step_offsets(N, k, s, gate)[j]` 发出，计划时刻 = `planned_rel_ns(该步)`。X2 的消息层滞后 = 收到时刻 − 这个计划时刻；判定器按 `req_id` 把客户端的 `chunks` / 排空端的 `msgs` 与服务端的 `streams` 连接。
- **`--pacing beat` 的含义（nullsse、nullgrpc 相同）**：全进程一个 `BeatSchedule` 绝对节拍器；每条流一个待发队列；第 j 条消息在上一条所说的那一步入队。不是「每条流从接纳起各自计时」——那样负载器一侧算不出计划时刻。`asap` 模式不按节拍发，X2 不适用（判定器对 `asap` 档的 X2 记 `NOT_RUN`）。

## 4. 到达时间表与 prompt 池

### 4.1 二进制容器 capbin（全部逐行记录与时间表共用）

`capcommon.capbin` 是唯一的读写实现。布局（小端）：

```
offset 0   8 字节   magic  b"CAPBIN01"
offset 8   8 字节   u64    header_len（JSON 头的字节数，含尾部空格填充，是 8 的倍数）
offset 16  header_len 字节  UTF-8 JSON 头，尾部用空格填到 8 的倍数
随后       各列依次排放，每列是 array 模块的原始字节（小端），每列尾部用 0 填到 8 的倍数
```

JSON 头：

```json
{"schema":"capkit.capbin/1","kind":"<内容类型，如 schedule / loadgen.requests>",
 "identity":{…§2.1…}|null,"n_rows":12345,
 "columns":[{"name":"planned_rel_ns","typecode":"q","itemsize":8,"offset":0,"nbytes":98760}, …],
 "meta":{…按 kind 约定…},"byteorder":"little"}
```

- `offset` 相对数据区起点（= `16 + header_len`）。允许的 typecode：`b B h H i I q Q f d`；读者核对 `itemsize` 与本机 `array` 的一致，不一致即报错（不猜）。
- 所有列行数相同（`n_rows`）。读：`read_capbin(path)` 返回 `(header, {列名: array})`；大文件按行步长读：`read_capbin(path, columns=[…], stride=(k, W))` 只取第 k、k+W、… 行（用 `mmap`，不整读）。
- 逐行记录在运行中进 `RecordArray(fields, capacity)`（列式、预分配）；满了不扩容，只把 `overflow` 计数加一并丢掉这一行；落盘时 `overflow` 写进 `meta.overflow`，判定器把 `overflow > 0` 判为对账失败（不是「少了几行没关系」）。`dump_split(path_for, kind, identity_for, key="tier_seq", meta=None)` 按某一列的值分文件落盘（逐档目录各一份；`path_for(值)`、`identity_for(值)` 由调用方给出）。

### 4.2 到达时间表（`kind: schedule`）

一次启动的全部请求的计划发出时刻，由 `capcommon.schedule.generate_schedule(tier_table, max_rows)` 从档位表确定地生成，`write_schedule(path, …)` 写成 capbin。它是登记文件（sha256 进 `REGISTRY.tsv`）。

| 列 | typecode | 含义 |
|---|---|---|
| `planned_rel_ns` | `q` | 计划发出时刻（相对原点）。全表单调不减 |
| `tier_seq` | `h` | 所属档 |

- 行号就是请求号 `req_id`（0 起，全表唯一）。**分给负载器 worker 的规则固定为 `req_id % n_workers == worker`**（设计稿 §2.2 第 1 条）；`n_workers` 是负载器的参数，不在时间表里。读：`load_schedule(path, worker, n_workers, tier_table_sha=…)` 返回该 worker 的 `(req_ids, planned_rel_ns, tier_seq, meta)`（前三个是 array），并核对头里的档位表 sha。每个 worker 每档的应发条数只由表头算：`per_worker_tier_counts(meta, n_workers)`。
- 生成算法（写死，改了要升版本）：每个负载档独立生成，`[start, end)` 内：
  - `fixed`：第 i 个到达 `start + (i × 10^9) // rate_rps`（整数运算，`rate_rps` 先转成 `Fraction` 以免浮点漂移），i 从 0 起直到 ≥ end。
  - `poisson`：`random.Random(seed)` 一条随机数流按档顺序贯穿整表；间隔 `-ln(1 − u) / rate_rps` 秒（`u = rng.random()`，不用 `expovariate`，避免版本差异），取整到 ns；第一个到达是 `start + 第一个间隔`。
- `meta`：`{"generator":"capkit.schedule/1","tier_table_sha256":…,"dist":…,"seed":…,"per_tier":{"<seq>":<该档请求数>},"prompt_assign":"mod","n_requests":…}`。`per_tier` 就是「应发请求数」，判定器从这里取应检条数，不从负载器的自报里取。
- `max_rows`（调用方必须给）：生成的总行数超过它就抛错，不截断。
- 体积：每请求 10 字节；生成前用 `count_schedule_rows(tier_table)` 先算行数。

### 4.3 prompt 池（`schema: capkit.prompts/1`，JSON）

```json
{"schema":"capkit.prompts/1","source":{"tokenizer":"configs/tokenizers/<名字>","note":"…"},
 "sets":{"tok128":{"kind":"token_ids","items":[[1,2,3,…],…]},
         "txt1024":{"kind":"text","items":["…",…]}}}
```

- 请求 `req_id` 用档 `prompt_set` 集合里的第 `req_id % len(items)` 项（`prompt_assign: "mod"`，`capcommon.schedule.prompt_index`）。要「全新 prompt」就让集合的项数 ≥ 该档请求数。
- 池文件是登记文件。负载器启动时把每个集合预先渲染成请求体字节（发送路径上不做 JSON 序列化）。

### 4.4 每请求的固定请求内容（负载器负责，判定器核对）

**每请求参数只随请求本身传递**（`capcommon.wire`）：一个请求用的是它**计划时刻所在的档**（时间表的 `tier_seq` 列）的 `max_tokens`、`tokens_per_msg`（k）、`stream_interval`（s）。服务端（空汇、节拍源、中继、第二批的桩）**不得**按到达时刻查档位表（S-PB 不给 `--schedule` 时按接纳时刻归档，见勘误 B2B-4）——在负载档末尾计划、因发送滞后晚到的请求会落进清空档（那里 `max_tokens = 0`），按到达时刻查就会算错应有条数。

- HTTP：请求体显式带 `max_tokens`（= 计划档的 `max_tokens`）、`ignore_eos: true`、`stream: true`、`stream_options: {"include_usage": true}`；不带 stop 字符串。请求头（`capcommon.wire.http_headers`）：`x-capload-id: <req_id 十进制>`、`x-capload-k: <k>`、`x-capload-s: <s>`、`x-capload-tier: <计划档 seq>`。`nullsse` 从请求体读 `max_tokens` 与 `stream_options.include_usage`，从请求头读另外三项；产品前端忽略这些头。
- gRPC（`--request-format capkit`）：请求字节 = `capcommon.wire.GRPC_REQ`（`struct "<qiiii"`：`req_id`、`max_tokens`、`tokens_per_msg`、`stream_interval`、`tier_seq`）+ prompt 字节，启动时按（档、prompt）预先构造；`pack_grpc_request` / `unpack_grpc_request`。`openengine` 格式（第二批）的 `request_id` 字段写 `<req_id 十进制>`。
- 同一启动内 `req_id` 唯一，所以 `x-capload-id` 在启动内唯一；跨启动靠 `identity` 区分。

### 4.5 SSE 的字节格式与计块口径（`capcommon.wire`）

`nullsse` 对一个请求写出（全部帧以 `\n\n` 结尾，不用 `\r\n`）：

1. `M = expect.msgs_per_request(N, k, s, "product")` 个内容帧，每个恰好 `--chunk-bytes` 字节：`wire.padded_content_frame(req_id, chunk_bytes)`，形如 `data: {"id":"capkit-<req_id>","object":"text_completion","choices":[{"index":0,"text":"xxx…","finish_reason":null}]}`；
2. 请求带 `include_usage: true` 时，一个 usage 帧 `wire.usage_frame(req_id, prompt_tokens, completion_tokens)`（`choices` 为空，`usage.completion_tokens` = 实际出的 token 数）；
3. 结束标记 `data: [DONE]`（`wire.DONE_FRAME`）。

客户端（`capload` 的 count 模式）计块只用 `wire.SseCounter`：`frames` = 除 `[DONE]` 之外的全部 SSE 事件数（**usage 帧算一帧**）；`done_seen` 单独记；`completion_tokens` 取最后一个非 `[DONE]` 帧里的 `usage`。所以 nullsse 一个请求的应有帧数 = `expect.sse_frames_per_request(M, include_usage)` = M + 1。流结束时没见到 `[DONE]`、或末尾有半个帧（`leftover > 0`）→ `protocol_error`。`chunks` 记录的 `idx` 是按这个口径数的帧序号（0 起），所以 usage 帧的 `idx = M`。产品前端的块数口径相同，已知不等的来源（`PromptInfo` 不产生块等）写在场景配置的 `reconcile.corrections` 里（§5.8）。

## 5. 一次启动的数据树与各角色的输出

### 5.1 树的形状（DATA-AND-ANALYSIS §3 的展开；路径助手 `capcommon.runtree.RunTree`）

```
<T>/                                   runs/J<作业号>-r<重排次数>-L<启动序号>_<臂名>/
├── raw/
│   ├── launch.json                    编排器：启动前写一次，之后不改（§5.2）
│   ├── launch_end.json                编排器：启动结束时写（§5.2；§3 骨架之外新增的一个文件，见 §12）
│   ├── dims.json                      编排器：维度表（§5.2）
│   ├── env/                           门禁与环境（§5.5）
│   ├── procs/
│   │   ├── roles.json                 pid → 角色登记表（§5.3），运行中可更新（原子替换）
│   │   ├── ports/<角色>.json           端口文件
│   │   ├── ready/<角色>[.w<k>].json    就绪文件
│   │   └── threads/<角色>.<pid>.json   线程清单（§5.3）
│   ├── tiers/<序号3位>_<标签>/          每档一个目录（§5.4），如 003_up-r2000
│   ├── tiers/pre/ 、tiers/post/          原点之前 / 最后一档之后的记录（预热前的启动期、停止收尾）
│   ├── sidecar/                       sidecar 逐秒序列（§5.6）
│   ├── canary/                        金丝雀逐秒序列（§5.6）
│   ├── spinners/<角色>.json           自旋器的记录（勘误 SB-B1-5）
│   └── logs/<角色>[.w<k>].<pid>.{out,err}   标准输出与错误（编排器重定向）
├── control/                           编排器 → 工具的信道（stop.json）；不是原始数据
├── judge/                             判定器输出（§10），只生成一次
└── STATE                              一行一条，只追加（§5.7）
```

档目录名用 `capcommon.runtree.tier_dir_name(seq, label)` = `f"{seq:03d}_{label}"`。

**作业目录**（一个作业的多次启动共用）：`runs/J<作业号>-r<重排次数>/`（`capcommon.runtree.job_dir_name`），与启动树并列，放 `job.json`（钉节点作业另记 `expect_node`，见勘误 calib-v2-A4 第 6 条）、sbatch 日志、脚本快照指针，以及 `env/slurm_view.json`（§5.5.2）。作业级门禁（`capenv.gate`）的结果写在作业目录的 `env/`，编排器在每次启动开始时把它原样拷进该启动树的 `raw/env/`（判定器只读启动树）。

### 5.2 编排器写的启动级文件

全部 JSON 都用 `capcommon.jsonio.write_json_stamped(path, schema, identity, payload)` 写，结构是 `{"schema":…,"identity":…,"written_mono_ns":…,"written_wall":"<ISO8601 UTC>", …payload 的键…}`；读用 `read_json_stamped(path, schema)`（schema 不符即报错）。

`launch.json`（`capkit.launch/1`，`identity.role = "launch"`）。**写的时刻**：编排器选定原点之后、起第一个进程之前（§0），之后不改：

| 键 | 内容 |
|---|---|
| `origin_ns` | 原点（`选定时刻 + origin_lead_ns`） |
| `origin_lead_ns` | `--origin-lead-ms` 换算的 ns；全部角色必须在原点之前就绪 |
| `shm_dir` | 本次启动的快照子目录（§2.2） |
| `register` | `{"relpath":…,"sha256":…}`（相对 `<W>`；sha 同 `identity.register_sha`） |
| `tier_table`、`schedule`、`prompts` | 各自 `{"relpath":…,"sha256":…}` |
| `scenario` | 场景配置 `{"relpath":…,"sha256":…}` 与它的全文副本 |
| `code` | `{"capkit_sha":"<40hex>","probe_tree_sha":…,"baseline_sha":"2f22a98d6e…","snapshot_dir":…}` |
| `image` | `{"path":…,"sha256":…}` |
| `probe_tier` | `P0` / `P1` / `P2`，以及打开的探针组列表 |
| `roles` | 计划起的角色，场景配置（§11）逐字抄来再补全：`[{"role":…,"module":…,"args":[…完整命令行…],"cpus":[…],"workers":W,"expected_procs":P,"measure_class":…,"asyncio":…,"uses_grpc":…,"loop":…,"probe_tier":…}]`。**判定器的一切「应检进程数」都取自这里的 `workers` 与 `expected_procs`**，再与 `roles.json` 的观测对账：观测少于应有 ⇒ 红，不判通过 |
| `reconcile` | 场景配置的对账段（§5.8）原样副本 |
| `hw_profile_sha` | 本节点画像校验值（§5.5） |
| `submission` | `{"job_id":…,"restart_count":…,"submitted_at":…}`（submitted_at 从 `submissions.tsv` 抄） |

`launch_end.json`（`capkit.launch_end/1`）：`stop`（stop.json 的内容或 `null`）、`ready_at_rel_ns`（每个角色就绪文件的时刻，相对原点；应全为负）、`tiers_completed`（走完的最后一档 seq）、`gate_outcomes`（每个清空档 `{"seq":…,"passed":…,"t_pass_rel_ns":…}`）、`exit_codes`（`{"<角色>[.w<k>]": 码}`）、`ended_mono_ns`。

`dims.json`（`capkit.dims/1`）：DATA-AND-ANALYSIS §5 的每个格子一个键，**值未知写 `null`（= 未核对），不省略键**：`code_version`、`image_sha`、`node`、`machine_model`、`hw_profile_sha`、`cores_numa`、`canary_rate_baseline`（亚饱和档的金丝雀速率，启动结束时补）、`tokenizer`（名字与选择理由）、`pin_table_sha`、`env_vars`（全部相关环境变量的键值）、`stub`（类型与参数）、`load_shape`、`probe_tier`、`log_sink`、`gc`、`response_timing_switches`（如 `enable_early_first_token_response`）、`loop_impls`（各角色的版本自报摘要）、`protobuf_impls`、`interpretation`（判读口径）。`dims.json` 在启动结束时写（这时才有全部读数）。

### 5.3 进程登记

`procs/roles.json`（`capkit.roles/1`，`identity.role = "launch"`），编排器每次有变化就原子替换，`generation` 加一；sidecar 发现 `generation` 变了就重读：

```json
{"generation":3,"procs":[
  {"pid":4321,"starttime_ticks":987654,"role":"capload","worker":0,"parent_pid":4300,
   "cmdline":["python","-m","capload",…],"cpus":[3],"discovered_via":"ready.children",
   "registered_mono_ns":…}],
 "unattributed":[{"pid":4400,"starttime_ticks":…,"seen_via":"probe.snapshot","argv":[…]}]}
```

（监测类角色的子进程、工具自起的辅助进程与 `aux_helpers` 键见勘误 R1-4、R1-5。）

- `starttime_ticks` = `/proc/<pid>/stat` 第 22 列，与 pid 一起才唯一（防 pid 复用）。`discovered_via` ∈ `launched`（编排器直接起的）、`ready.children`（就绪文件报的子进程）、`probe.snapshot`（产品进程的探针快照里自报的 pid，§6）、`proc.children`（编排器扫 `/proc` 找到的子孙，**只用于报告「未登记的进程」，不自动给角色**）。
- 测量期间 pid 集合不变是前提 P 的一项：编排器在每档窗口开始与结束各扫一次，结果写进该档的 `tier.json`（§5.4）。
- `roles.json` 是**观测**；应有多少进程来自 `launch.json` 的 `roles[].expected_procs`（含产品进程按其参数派生的子进程，如 `--num_postprocess_workers w` 时多出的 w 个池进程）。某个角色没起来或没写就绪文件，它在 `roles.json` 里就少一项，判定器按应有数判红，而不是「少查一个也通过」。

`procs/threads/<角色>.<pid>.json`（`capkit.threads/1`）：预热之后、第一个负载档之前，编排器为每个被测进程列出全部线程：

```json
{"pid":4321,"taken_mono_ns":…,"threads":[
  {"tid":4321,"comm":"python","py_name":"MainThread","class":"python",
   "identified_as":"event_loop","source":"probe.pin"},
  {"tid":4330,"comm":"grpc_global_tim","py_name":null,"class":"grpc_core",
   "identified_as":"grpc timer","source":"comm"},
  {"tid":4331,"comm":"ZMQbg/IO/0","py_name":null,"class":"zmq_io","identified_as":null,"source":"comm"}]}
```

`class` ∈ `python` / `grpc_core` / `zmq_io` / `other`；`identified_as` 为 `null` 的线程 = 未认出，C13 判红。认线程的规则由编排器（B7）维护在 `caplaunch` 里，规则表本身写进该文件的 `rules_sha`。

这份清单只是起点：grpc 的线程懒创建、分词线程池按需起线程，都可能在它之后出现。所以**每档窗口的始末各再取一次**全部登记进程的 tid 集合与识别结果，写进 `tier.json` 的 `threads`（§5.4）；窗口内出现的新 tid 认不出来 ⇒ 该档 C13 红。capkit 工具自己的线程不许懒建：注入的辅助线程（`capkit-inject-aux-<位点>`）在启动时由 `InjectorSet.prestart_aux(档位表.all_inject_for(角色) + 命令行规格)` 一次建好（懒建会抛错），线程名写进就绪文件的 `info.threads`。

### 5.4 逐档目录 `raw/tiers/<序号>_<标签>/`

| 文件 | 谁写 | 内容 |
|---|---|---|
| `tier.json`（`capkit.tier/1`） | 编排器 | 本档规格的原样副本、绝对起止 `abs_start_ns`/`abs_window_start_ns`/`abs_window_end_ns`/`abs_end_ns`、窗口始末两次扫描的 pid 集合、`pids_stable`；窗口始末两次的 `threads`（`{pid: [{"tid","comm","py_name","identified_as"}]}`）；窗口始末两次的 `/sys/class/net/*/statistics/{rx,tx}_bytes`（`nic_counters`，读不到写 `{"unreadable": 原因}`；同机回环时证明「网卡不在路径上」，DATA §8.1）；`snapshot_copies`（见下） |
| `drain_gate.json`（`capkit.drain_gate/1`） | 编排器 | 只在清空档：§8.3 |
| `snapshots/<角色>.<pid>.{wstart,wend}.bin` | 编排器 | 窗口开始 / 结束各拷一份全部登记进程的快照文件（§6），原样字节。**取边界之后的第一份**：拷贝前解码，要求快照自身的 `t_mono_ns ≥` 边界的绝对时刻；不满足就等（最多 2 个快照周期），超时在 `tier.json` 的 `snapshot_copies` 里记 `snapshot_missing`。`snapshot_copies` 每份记 `seq`、`t_mono_ns`、相对边界的偏差 ns。判定器用快照自身的 `t_mono_ns` 作区间端点，不用边界时刻 |
| `<角色>.w<k>.summary.json` | 负载器、排空端、空汇、节拍源、中继 | 本档自报（§5.4.1） |
| `<角色>.w<k>.<名字>.capbin` | 同上 | 本档的逐行记录（§5.4.2），由 `RecordArray.dump_split` 在进程退出前写 |

**每个文件的 `identity` 带 `tier_seq`、`tier_label`、`worker`、`pid`。** 记录归档的规则：逐请求记录按**计划时刻所在的档**归档（`tier_seq` 取时间表那一列）；逐事件记录（滞后采样、FIONREAD 采样、消息到达）按**事件发生时刻所在的档**归档（`TierTable.tier_seq_at(rel_ns)`，原点之前记 `-1` → `tiers/pre/`，最后一档之后记 `-2` → `tiers/post/`）。判定器按请求核对块行时，把本档与其后清空档的文件合计（见勘误 SM-14-1）。

#### 5.4.1 逐档自报 `<角色>.w<k>.summary.json`（`capkit.tier_summary/1`）

```json
{"schema":"capkit.tier_summary/1","identity":{…},"tool":"capload",
 "n_workers":4,"schedule_sha256":"<64hex>","tier_table_sha256":"<64hex>",
 "request_timeout_ns":30000000000,
 "counts":{"planned":1200,"sent":1200,"ok":1198,"failed":2,"not_sent":0},
 "failures":[{"reason":"http_503","count":2,"class_hint":"unclassified"}],
 "frames":{"frames":40000,"bytes":5230000,"completion_tokens":39600},
 "send_lag_ns":{"hist":{"bins":"loghist/1","counts":[…]},"max":…,"n":…},
 "fionread":{"samples":3000,"nonzero":0,"max_bytes":0},
 "conn_pool":{"size_max":64,"new_conns":64,"reconnects":0,"peer_idle_closed":0,"conn_fail":0},
 "inject":[{"site":"recv.chunk","spec":{…},"activation_unchanged":true,"hits_delta":40000,
            "applied_delta":40000,"applied_ns_delta":800000000,
            "hits_since_activation_at_start":0,"hits_since_activation_at_end":40000,
            "expected_applied_delta":40000,"aux_pending_at_end":0}],
 "loop_lag_ns":{"hist":{…},"n":…},
 "overflow":{"requests":0,"chunks":0,"fionread":0},
 "stop":null}
```

- `n_workers`、`schedule_sha256`（服务端写 `null`；S-PB 见勘误 B2B-15）、`tier_table_sha256` 每个工具都写，判定器与 `launch.json` 交叉核对；负载器另写 `request_timeout_ns`（取自档位表）。
- `inject` 每项由 `capcommon.inject.tier_report(本档开始的 mark, 本档结束的 mark)` 生成（工具在档边界上：先取上一档结束的 `mark()`，再 `apply_tier_specs`，再取本档开始的 `mark()`）。各量都是**本档增量**；选中规则按「生效以来」的经过次数计（§7.3）。
- `counts` 的五个键对负载器必需；`planned` 是本 worker 在本档按时间表应发的条数（由时间表算出，不由发送路径数），`planned ≠ sent` 时判定器判本档 `NOT_RUN`。服务端类工具（空汇、节拍源）用 `counts` 里与自己相关的键（如 `streams_accepted`、`msgs_sent`、`bytes_sent`、`accepts`），在包的 README 里列出。
- **失败原因**：`reason` 用下表的固定码，`class_hint` ∈ `overload` / `non_overload` / `unclassified`。负载器只能确定少数类别（事先登记的超时 → `overload`；连接被拒、协议错误 → `non_overload`），其余写 `unclassified`，由判定器结合服务端日志与探针计数改判（设计稿 §2.6）；判定器分不出的按非过载型处理（连接类失败一律非过载型，见勘误 B2-2E-6）。新增的 `client_error` 见勘误 R1-7。

| `reason` | 含义 | 负载器的 `class_hint` |
|---|---|---|
| `http_<码>` | 非 2xx 状态码 | `unclassified` |
| `conn_refused` | 建连被拒 | `non_overload` |
| `conn_fail` | 其他建连失败（含临时端口耗尽） | `unclassified` |
| `conn_reset` | 传输中被对端断开 | `unclassified` |
| `timeout_registered` | 超过档位表 / 登记的超时 | `overload` |
| `protocol_error` | SSE / gRPC 帧解析失败、`x-capload-id` 不符 | `non_overload` |
| `grpc_<状态名>` | gRPC 非 OK 状态 | `unclassified` |
| `short_output` | 实际 token 数 ≠ `max_tokens` | `non_overload` |

- 直方图一律用 `capcommon.hist` 的 `loghist/1` 分桶（§6.3）。

#### 5.4.2 逐行记录（capbin `kind` 与列，都带 `identity`）

| 文件名 `<名字>` | `kind` | 列 |
|---|---|---|
| `requests` | `loadgen.requests` | `req_id q`、`planned_rel_ns q`、`sent_rel_ns q`（实际写出请求的时刻；重连重发见勘误 R1-7）、`first_rel_ns q`（首个数据帧，无则 −1）、`done_rel_ns q`（完成或失败时刻）、`frames i`、`bytes q`、`tokens i`、`status h`（0 = 成功，其余为 `meta.status_codes` 里的失败原因编号）、`conn_id i`、`new_conn b`、`tier_seq h` |
| `chunks` | `loadgen.chunks` | `req_id q`、`idx i`（该流的第几个数据帧，0 起）、`recv_rel_ns q`、`nbytes i`、`tier_seq h`。只记 `req_id % chunk_sample_mod == 0` 的请求；`meta.chunk_sample_mod` 必写，判定器据此算应有行数（按接收时刻归档，一个请求的行可能跨到下一档，见勘误 SM-14-1） |
| `fionread` | `loadgen.fionread` | `t_rel_ns q`、`conn_id i`、`bytes i`、`tier_seq h` |
| `lag` | `loop.lag` | `planned_rel_ns q`、`late_ns q`、`tier_seq h`（只在开了逐样本记录时有，§6.4） |
| `steps` | `beat.steps` | `step q`、`planned_rel_ns q`、`actual_rel_ns q`、`msgs i`、`tier_seq h`（节拍源每步一行） |
| `msgs` | `drain.msgs` | `stream_id q`、`idx i`、`step q`（消息里盖的步号，§7.4；`--planned-msgs` 时为从时间表推出的步号，见勘误 FT1-9）、`planned_rel_ns q`、`recv_rel_ns q`、`tier_seq h`（排空端；采样规则同 `chunks`） |
| `streams` | `server.streams` | `req_id q`（HTTP 取自 `x-capload-id`，gRPC 取自请求头）、`accept_rel_ns q`、`first_step q`（`BeatSchedule.step_at_or_after(accept_rel_ns)`；`asap` 模式记 −1）、`n_msgs i`（实际发出的内容消息数）、`tokens i`、`tier_seq h`（**请求自带的计划档**，§4.4）。空汇、节拍源（第二批：桩）每接纳一个请求一行，按 `tier_seq` 归档（S-PB 的 `req_id`、`n_msgs`、`tier_seq` 口径见勘误 B2B-4）；判定器用它与 `chunks` / `msgs` 按 `req_id` 连接，用 `first_step + expect.msg_step_offsets(…)[idx]` 得到每块对应的步（X2）|

`meta.status_codes` 是 `{"1":"http_503",…}` 的映射。各工具可以在 `meta` 里加键，不能改列。

### 5.5 `raw/env/`（环境与门禁，B4 与 B6 写）

| 文件 | 写者 | 内容（schema 名） |
|---|---|---|
| `hw_profile.json` | B4 `capenv.hwprofile` | `capkit.hw_profile/1`：恰好 §5.5.3 的两张键表；`profile_sha256` = `capcommon.hwprofile.profile_sha256`（只算 identity 键） |
| `clock_check.json` | B4 `capenv.clockcheck` | `capkit.clock_check/1`：两个进程互发带 `monotonic_ns` 的消息做往返的次数、单程差（对端时间戳 − 本端发送时刻）分布、同一位点连打两次 `monotonic_ns` 的差值分布（时钟噪声），两个进程各自的 `/proc/self/ns/{pid,mnt}` 与 cgroup 路径（证明是否跨容器 / 命名空间）。E0 的「时钟自检」与 §12 U-4 用它；它是 `pilot.json` 的输入之一 |
| `hw_gate.json` | B4 | 本节点 sha、登记的主型号 sha、`comparable_fields` 登记、`passed` |
| `exclusivity.json` | B4 `capenv.gate` | RESOURCES §3 第 1～6 项：每项 `{"id":1..6,"cmd":…,"rc":…,"stdout_sha":…,"passed":…,"detail":…,"source":"container"|"slurm_view"}`（第 1～3 项在容器里跑不了时取自 §5.5.2 的回退文件，`source` 写明）；第 3 项的阳性对照（输出里有本作业）单列 `positive_control_ok` |
| `memguard.json` | B4 | 每个被测 pid 的 `/proc/<pid>/cgroup`、对应 cgroup 的 `memory.max`（或 v1 的 `memory.limit_in_bytes`）读数；没有任何读数即门禁不过 |
| `pin_table.json` | B4 `capenv.pins` | 绑核表（§5.5.1） |
| `pin_readback.json` | B4 | 启动时一次：每个登记进程每条线程的 `Cpus_allowed_list` 与最近运行核；与表比对的结果。**运行中的逐秒读数在 `sidecar/threads.jsonl` 的 `cpus_allowed` 列**（D2：worker 会在运行中自己解除绑核，只读一次抓不到），C9 用逐秒读数 |
| `switchinterval.json` | B4 | 每个被测进程的 `sys.getswitchinterval()`（D13，取自版本自报）汇总 |
| `loopback_softirq.json` | B4 `capenv.loopback` | 回环 softirq 小程序的结果（只在 E0 的专门一步里有；不进 D14 结论，见勘误 R1b-4） |
| `overlay_manifest.json` | B3 `capprobe.overlay` | 覆盖清单（§6.6） |
| `overlay_verify.json` | B6 内层脚本 | 拷进容器后逐文件 sha256 的读回比对 |
| `image.json`、`mounts.json`、`local_disk.json`、`gpu_idle.json` | B6 | 镜像 sha、挂载检查（含模型库不在场）、本地盘余量、`nvidia-smi` 计算进程与利用率（`gpu_idle.json` 含阳性对照 `positive_control_ok`：查询本身列出了本节点的 GPU，张数写进 `gpu_count`）。判定见 §10.2 的 `GPU-IDLE`；`gpu_idle.json` 的形状与 GPU 数口径见勘误 R1-8 |
| `versions/<角色>[.w<k>].<pid>.json` | 编排器 | 每个进程就绪文件里的版本自报的副本（产品进程取自探针快照的 `meta.versions`） |

#### 5.5.1 绑核表 `pin_table.json`（`capkit.pin_table/1`）

```json
{"source_sha":"<hw_profile 的 sha>","rules":"one-logical-per-physical; roles disjoint",
 "siblings_left_idle":[[0,112],[1,113]],
 "roles":{"capload":{"cpus":[2,3,4,5],"numa_node":0},"nullsse":{"cpus":[6,7],"numa_node":0},
          "sidecar":{"cpus":[60],"numa_node":0},"canary":{"cpus":[61],"numa_node":0}},
 "unassigned":[…]}
```

由 `capenv.pins` 从硬件探测结果与场景配置里的角色需求（`[{"role":…,"cpus_needed":…,"numa":…}]`）生成；放不下时以退出码 3 退出并列出缺口（不临时放宽 D1 / D3）。编排器只按这张表绑核，不自己算。

#### 5.5.2 Slurm 读数的回退文件 `<作业目录>/env/slurm_view.json`（`capkit.slurm_view/1`）

RESOURCES §3「作业内验证独占」第 1～3 项（`scontrol show job`、`scontrol show node`、`squeue -w`）在计算节点或容器里不可用时，由提交端在作业开跑后从前端查、写进这个文件：`{"schema":…,"job_id":…,"items":[{"id":1,"cmd":[…],"rc":…,"stdout":"<原文>","stdout_sha256":…,"taken_wall":"<ISO8601 ms>"},…]}`。写者：`capjobs.submit slurm-view --job-id J --job-dir D`（B6，在 VM 上跑，只调用 CONSTRAINTS §7 允许的前端只读命令；每项有界重试、有查询仍失败时不写文件并以 4 退出，条目另有 `attempts`、`retried_rcs`、`retried_stderr`，见勘误「Fix SM-17 slurm-view ssh failures and SM-18 gRPC unrequested-call cancellation」第 1 条）。读者：`capenv.gate --slurm-view-file PATH --slurm-view-wait-s S`（两个参数都必需；S 取自登记表）——容器内命令可用时不用它；不可用时等这个文件出现，最多 S 秒，超时即 `ENV_INVALID` 退出。硬件画像的 `chassis_model` 与 `partition` 也取自它（容器里通常没有 `scontrol`）。

#### 5.5.3 硬件画像的键（`capcommon.hwprofile`）

画像**恰好**有下面两组键（多一个少一个都报错，`validate_profile`）。每个值要么是读到的值，要么是 `{"unreadable": "<原因>"}`；identity 键的「读不到」标记**也进 sha**。

- `IDENTITY_FIELDS`（进 `profile_sha256`；「一致」= 这些全部相同）：`chassis_model`、`cpu_arch`、`cpu_model`、`sockets`、`cores_per_socket`、`threads_per_core`、`microcode`、`numa_layout`、`cpu_governor`、`mem_total_gib`（按 GiB 取整）、`mem_per_numa_gib`、`gpu_model`、`gpu_count`、`gpu_mem_mib`、`gpu_driver`、`nics`（`{model, speed_mbps}` 的有序列表，不含接口名；已改为 PCI 设备清单，见勘误 R4-7）、`local_disk_type`、`kernel`、`container_runtime`、`image_sha256`、`code_baseline`。
- `OBSERVED_FIELDS`（记录，不进 sha）：`node_name`、`partition`、`collected_at`、`collected_mono_ns`、`cpu_freq_khz`、`mem_available_kib`、`local_disk_free_bytes`、`nic_link`（状态与收发计数）、`netdevs`（勘误 R4-7）、`sources`（每个键取自容器 / 回退文件 / 读不到）。

每个键的取法写在 `capcommon/hwprofile.py` 的两张表里；登记主型号时算 sha 与作业内 `capenv.hwprofile` 算 sha 用同一个函数。

### 5.6 sidecar 与金丝雀（B4 写，流式 JSONL）

逐秒序列由独立进程写，允许运行中写本地盘。格式统一为 **capkit JSONL**（`capcommon.jsonio.JsonlWriter`）：第一行是头 `{"_header":true,"schema":…,"identity":…,"columns":[…]}`，其后每行一个 JSON 数组，顺序与 `columns` 相同。读：`read_jsonl(path)` 返回 `(header, rows)`。

| 文件 | `schema` | 列 |
|---|---|---|
| `sidecar/threads.jsonl` | `capkit.sidecar.threads/1` | `t_rel_ns, pid, tid, comm, utime_ticks, stime_ticks, run_delay_ns, voluntary_ctxt, nonvoluntary_ctxt, last_cpu, cpus_allowed`（`cpus_allowed` = `/proc/<pid>/task/<tid>/status` 的 `Cpus_allowed_list` 原文，**每行都写**；C9 的应检条数 = 窗口秒数 × 登记线程数） |
| `sidecar/cpus.jsonl` | `capkit.sidecar.cpus/1` | `t_rel_ns, cpu, user, nice, system, idle, iowait, irq, softirq, steal, freq_khz`（`/proc/stat` 的 tick 计数，原样） |
| `sidecar/procs.jsonl` | `capkit.sidecar.procs/1` | `t_rel_ns, pid, role, worker, rss_kb, nthreads, cpus_allowed` |
| `sidecar/snapshots.jsonl` | `capkit.sidecar.snapshots/1` | `t_rel_ns, pid, seq, snap_t_rel_ns, counters, hists`（`counters`：counter 与 gauge 槽的 `{名字: 值}`；`hists`：hist 槽的 `{名字: [194 个累计计数]}`；`snap_t_rel_ns` 是快照自己的 `t_mono_ns` 换算到原点，用来判断读到的是不是旧快照） |
| `sidecar/gpu.jsonl` | `capkit.sidecar.gpu/1` | `t_rel_ns, gpu_index, util_pct, mem_used_mib, compute_pids, query_rc`；头里写 `gpu_count`（本节点 `nvidia-smi` 列出的 GPU 数）。查询失败的那一秒每张卡一行，`query_rc ≠ 0`、其余为 `null`（成功查询里的 `null` 读数见勘误 R1-10） |
| `sidecar/numa.jsonl` | `capkit.sidecar.numa/1` | `t_rel_ns, pid, pages_by_node`（`/proc/<pid>/numa_maps` 按 `N<k>=` 汇总的 `{节点: 页数}`；低频，周期由 `--numa-every-s` 给出；读不到写 `{"unreadable": 原因}`）。D4 的数据验证 |
| `sidecar/logs.jsonl` | `capkit.sidecar.logs/1` | `t_rel_ns, path, bytes`（`raw/logs/` 下每个日志文件的大小，每秒一行）。D8 的「日志字节增长率」 |
| `sidecar/meta.json` | `capkit.sidecar.meta/1` | `clk_tck`、采样周期、读不到的项（`schedstat`、频率等）及原因 |
| `canary/canary.jsonl` | `capkit.canary/1` | `t_rel_ns, iters, thread_cpu_ns, cpu` |

`t_rel_ns` 一律相对原点（sidecar 也拿 `--origin-ns`；原点之前的行是负数，照样记）。`cpus.jsonl` 只覆盖 `--cpus-watch` 给的核：编排器传入的列表必须 ⊇ `pin_table.json` 里全部角色的核（或 `all`），不覆盖时编排器以退出码 3 退出——C13 第二半要逐核读被测核集。

### 5.7 `STATE`

`<T>/STATE`，只追加，用 `capcommon.runtree.append_state(tree, status, reason, by)`：每行 `<ISO8601 UTC>\t<VALID|VOID|NOT_RUN|ENV_INVALID>\t<写者>\t<原因>`。作业内层脚本（B6）在拷回并校验之后写第一行；判定器在 `judge/` 写完之后追加一行（以判定结果为原因的改判，§10.2）（写者顺序与「只许变差」见勘误 R2-2、R2-12）。读当前状态 = 最后一行。

### 5.8 对账段 `reconcile`（场景配置里，`capcommon.expect`）

C5、C6、C7 要两样东西：每条链上哪几跳的哪个计数应当相等，以及每请求应有几条消息、几个 SSE 帧。**两样都只有一份实现**：公式在 `capcommon.expect`（桩、空汇、节拍源、判定器都调用它，不各写一份），哪几跳相等写在场景配置的必填段 `reconcile`（`capkit.reconcile/1`；编排器原样抄进 `launch.json`；判定器只读这一段，**不按角色名硬编码**）：

```json
{"schema":"capkit.reconcile/1","chains":[
  {"name":"requests","kind":"requests","gate":null,"include_usage":null,
   "hops":[{"role":"capload","source":"summary.counts.ok","scope":"tier"},
           {"role":"nullsse","source":"summary.counts.requests","scope":"tier"}],
   "corrections":[]},
  {"name":"sse","kind":"chunks","gate":"product","include_usage":true,
   "hops":[{"role":"capload","source":"summary.frames.frames","scope":"tier"},
           {"role":"nullsse","source":"summary.counts.frames_sent","scope":"tier"}],
   "corrections":[]}]}
```

- `kind`：`requests`（每请求 1）、`messages`（每请求 `expect.msgs_per_request(N, k, s, gate)`）、`chunks`（再按 `expect.sse_frames_per_request(…, include_usage)` 加 usage 帧）。`gate` ∈ `product`（真实闸门：首步、每 s 步、末步，`py_executor.py:10468-10470`）/ `every_step_plus_final`（S-FE：每步一条加结束一条）；capkit 的空汇与节拍源一律 `product`。
- `hops[].source`：`summary.<点分路径>`（该角色逐档自报里的一个数，如 `summary.counts.ok`、`summary.frames.frames`；多 worker 时各 worker 相加）或 `slot:<槽名>`（快照的窗口增量）；`scope` ∈ `tier`（逐档比）/ `launch`（整次启动比）。
- `corrections`：已知不等的修正项 `{"name","per_request":整数,"hop":角色}`，加在该跳的每请求应有数上（例：产品前端 `PromptInfo` 不产生块 → 前端写出的块数那一跳 `-1`；数值由 E0 的冒烟确认后登记，本约定不预设）。
- 一跳的应有数 = `expect.expected_hop_count(chain, 本档应发请求数, N, k, s, 该跳角色)`；应发请求数取时间表 `meta.per_tier`，扣除按原因单列的过载型失败（设计稿 §2.6）。`validate_reconcile(doc, 角色集合)` 在编排器启动时与判定器读入时各跑一次。

### 5.9 本地盘与快照目录的清理

只由 B6 的作业内层脚本做，只在**拷回并校验通过之后**做。被删的根路径一律是 `<本地盘>/<用户名>-<作业号>`（已改为带重启次数，见勘误 R1-11；`/dev/shm/<用户名>-<作业号>` 同理，它下面的 `snap/…` 随之删掉）。删除前调用 `python3 -m capcommon.scratchpath --path "${P:?}" --base "${BASE:?}" --user "${USER:?}" --job-id "${SLURM_JOB_ID:?}" --verified-file "${V:?}"`，退出码 0 才删（它断言路径**逐字等于** `<base>/<user>-<job>`、不是软链、`--verified-file` 的 `passed` 为真）；删除命令写成 `rm -rf -- "${P:?}"`，命令与结果追加进作业目录的日志。别的路径一律不删（CONSTRAINTS §6）。

## 6. 实时快照（capprobe 与全部 capkit 工具）

被测产品进程里的探针（`capprobe`）与全部 capkit 工具用**同一种**实时计数机制：进程内一个 `int64` 数组（每个槽只由一条线程写，不加锁），一条后台线程每 `--snapshot-interval-ms`（探针：`CAPPROBE_INTERVAL_MS`）把整个数组拷成一个快照文件。sidecar 逐秒读它们；编排器的清空门读它们；判定器从档目录里的窗口始末快照与 sidecar 序列读它们。实现只有 `capcommon.snapshot`。

### 6.1 文件与布局

文件：`<shm_dir>/<pid>.bin`，每次整份重写（同目录临时名 `.<pid>.bin.tmp` + `os.replace`）。布局（小端）：

```
offset 0   8 字节  magic b"CAPSNAP1"
offset 8   u32     version = 1
offset 12  u32     meta_len（JSON 元数据字节数，含空格填充，8 的倍数）
offset 16  u64     seq（本进程第几次写快照，1 起）
offset 24  i64     t_mono_ns（拷贝数组的时刻）
offset 32  u32     n_values（int64 个数）
offset 36  u32     meta_gen（槽表的代数；新登记槽时加一）
offset 40  meta_len 字节 JSON 元数据
随后       n_values × int64
```

元数据（`schema: capkit.snapshot/1`）：`identity`（产品进程里 `role` 可能是 `null`，见 §6.5）、`pid`、`ppid`、`starttime_ticks`、`argv`、`role_hint`、`versions`（§2.5）、`interval_ns`、`slots`：

```json
[{"name":"count.F2.msgs","kind":"counter","offset":0,"length":1},
 {"name":"backlog.inflight","kind":"gauge","offset":1,"length":1},
 {"name":"lag.loop","kind":"hist","offset":2,"length":194,"bins":"loghist/1"},
 {"name":"lag.loop.n","kind":"counter","offset":196,"length":1}]
```

解码：`decode_snapshot(bytes) -> Snapshot`（`.seq`、`.t_mono_ns`、`.meta`、`.values`（名字 → int 或 list））；`read_snapshot_file(path)`；`list_snapshot_files(shm_dir)`。解码器核对 magic、version、长度，不符抛 `SnapshotError`。

### 6.2 槽的命名与种类

槽名 `<组>.<位点或对象>.<量>`，正则 `^[a-z][a-z0-9_]*(\.[A-Za-z0-9_]+)+$`。

| 种类 | 语义 | 读者怎样用 |
|---|---|---|
| `counter` | 单调不减的累计数 | 两次快照相减得区间增量 |
| `gauge` | 当前值（可增可减） | 直接读 |
| `hist` | `loghist/1` 分桶的累计计数，长度 194 | 两次快照逐桶相减得区间分布；同名另有 `.n`、`.sum`、`.max`、`.neg` 四个 counter（`.max` 是累计最大值，`.neg` 是负值个数）。用 `SlotRegistry.hist(name).add(v)` 写 |

**清空门（§8）只读名字以 `backlog.` 开头的槽**：gauge 直接读（如 `backlog.inflight`）；跨进程的在途数用两个 `backlog.` counter 相减（如节拍源的 `backlog.yielded` 减读者的 `backlog.read`）。

固定的组：`count`（条数）、`bytes`（字节）、`lag`（事件循环延迟）、`queue`（队列深度）、`ts`（时间戳环形数组的写入计数）、`pin`（线程登记数）、`inject`（注入心跳，§7.3）、`backlog`（积压读数，§8）、`self`（工具自身的计数，如 `self.sent`）。

**机器可读的槽清单**：每个发布槽的包导出常量 `SLOTS = {槽名: {"kind": counter|gauge|hist, "site": 位点名或 null, "group": 组名（= 槽名第一段）, "per": request|message|step|event|sample|none}}`（`capcommon.snapshot.validate_slot_decls` 校验）。**组打开时，该组的全部槽在 import / 启动时一次性登记**（`SlotRegistry.register_decls(SLOTS, groups=打开的组)`），不在第一次经过位点时懒登记——这样「位点没接上」表现为「槽在、增量为 0」，与「槽不存在」可以区分。README 里的文字清单只给人看，判定器与编排器只认 `SLOTS`。

### 6.3 直方图分桶 `loghist/1`（`capcommon.hist`）

对非负整数 v（ns）：v < 1024 → 桶 0；否则 e = v.bit_length() − 1，o = e − 10，sub = (v >> (e − 3)) & 7，桶 = 1 + 8·o + sub；o ≥ 24（v ≥ 2^34 ns ≈ 17 s）→ 桶 193（溢出桶）。共 194 个桶，相对分辨率 1/8。负数（不应出现）记进桶 0 并另计 `<名字>.neg`。`bucket_index(v)`、`bucket_bounds(i) -> (lo, hi)`、`quantile(counts, q)`（返回该分位所在桶的上界，保守）、`diff(counts_a, counts_b)`。

### 6.4 事件循环延迟探针（`capcommon.aio.LoopLagProbe`）

- 一个周期性回调：每次被调用时记 `late = now_ns − planned_ns`，再以 `planned = now_ns + period_ns` 重新挂（**相对重挂**：长阻塞之后不补发一串回调，量的是「每个定时回调被推迟了多久」）；样本数下降本身也是信号（`.n` 计数）。
- 写进快照的槽：`lag.<名字>`（hist）及 `.n`、`.sum`、`.max`、`.neg`；可选逐样本记录（`RecordArray`，`lag` 文件，§5.4.2）。
- 周期由调用方给（`--lag-period-us`，探针为 `CAPPROBE_LAG_PERIOD_US`），没有默认值。uvloop 的定时器只有毫秒精度【读码：libuv `uv_now`】，所以同一读数在两种循环实现下不可直接比；I2 的标定按实现分开做（设计稿 §2.6）。

### 6.5 capprobe 的开关（环境变量，模块导入时读一次；只在被测入口进程里启用，见勘误 R1b-1）

| 变量 | 取值 | 缺省 / 校验 |
|---|---|---|
| `CAPPROBE_GROUPS` | 逗号分隔的子集：`count,bytes,lag,queue,ts,pin`；空或未设 = 全关（P1） | 出现未知组名 → 导入时 `RuntimeError`（进程起不来，比悄悄关掉好） |
| `CAPPROBE_TIER` | `P0` / `P1` / `P2` | 必须与 `CAPPROBE_GROUPS` 一致：P1 ⇒ 组为空；P2 ⇒ 组非空。不一致 → `RuntimeError` |
| `CAPPROBE_SHM_DIR` | 快照目录 | 任何组打开或有注入时必需，缺 → `RuntimeError` |
| `CAPPROBE_INTERVAL_MS` | 快照周期 | 同上 |
| `CAPPROBE_LAG_PERIOD_US` | 延迟探针周期 | `lag` 组打开时必需 |
| `CAPPROBE_INJECT` | 分号分隔的注入规格（§7.1） | 可空 |
| `CAPPROBE_PATCH_SHA` | 探针补丁提交的 40 位 sha | 写进版本自报；缺 → 自报里为 `null` |
| `CAPPROBE_IDENTITY_FILE` | 一个 JSON 文件 `{"schema":"capkit.identity/1","identity":{…§2.1…}}`（编排器写，`role` 可为 `null`，`pid` 为 `null`） | 快照元数据的 `identity` 取自它（探针把 `pid` 换成自己的） |
| `CAPPROBE_ROLE_HINT` | 可选；起该产品进程的命令里显式给出的角色名 | 只作 `role_hint`，**编排器只凭 pid 登记角色** |

- 探针进程的 pid 自报：快照文件名就是 pid，元数据里有 `pid`、`ppid`、`starttime_ticks`、`argv`。编排器发现 `<shm_dir>` 里出现新的 `<pid>.bin`，按「该 pid 是哪个已登记进程的子孙、`role_hint`」登记为 `discovered_via: probe.snapshot`；对不上任何已登记进程的写进 `roles.json` 的 `unattributed` 列表（判定器 C13 判红）。
- `ts` 组的环形数组在进程退出时（`atexit`）落到 `<shm_dir>/<pid>.ts.capbin`（`kind: probe.ts`）；作业收尾时由内层脚本拷进对应启动树。
- 探针自检注入（§7）在探针里由 `CAPPROBE_INJECT` 给出，整个进程生命周期不变（档位表的逐档注入只对 capkit 工具有效）。编排器目前拒绝外层 `CAPPROBE_INJECT`，见勘误 R4-2；产品侧注入改由档位表 `launch_inject` 登记，见勘误 FT1-18。
- `capprobe` 另导出 `SITES = {位点编号: {"group": 组, "proc_kind": 所在进程类型（如 frontend、postproc、host、worker；词表见勘误 R2-5）, "file": 相对产品树的路径, "line": 行号, "slots": [该位点的槽名…]}}`，`capprobe.overlay` 把它写进覆盖清单（§6.6）。`line` 已改为 `lines` 列表，见勘误 R1-12。

### 6.6 覆盖清单（P0 / P1 / P2，B3 的 `capprobe.overlay` 生成）

`env/overlay_manifest.json`（`capkit.overlay/1`）：

```json
{"tier":"P2","image_commit":"9fd3822a00…","base_commit":"2f22a98d6e…","target_commit":"<探针树 HEAD>",
 "groups":["count","lag"],
 "files":[{"src":"probe-tree/tensorrt_llm/serve/openai_server.py",
           "dst":"tensorrt_llm/serve/openai_server.py","sha256":…,"size":…}],
 "capkit_packages":[{"src":"tools/capprobe/__init__.py","dst":"capprobe/__init__.py","sha256":…,"size":…}],
 "capkit_code_sha":"<40hex>","non_python_changed":[],
 "sites":{"F2":{"group":"count","proc_kind":"frontend","file":"tensorrt_llm/…","line":123,
               "slots":["count.F2.msgs"]}}}
```

`sites` = `capprobe.SITES` 里属于本次打开的组的那些位点（P0、P1 为空表）。C8（探针已接上）的应检位点数取自这里（只算本拓扑适用的位点，见勘误 R1-19），不取自「快照里出现了的槽」：每个应检位点在每个被判档的窗口增量 > 0，且与 §5.8 对账链给出的应有条数一致（该位点在某条链上时）。

`dst` 相对容器的 `dist-packages`（`stub_modules` 没有 `dst`、不装进 `dist-packages`，见勘误 B2B-11）。`non_python_changed` 必须为空（镜像提交到目标提交之间 C++ 与非 Python 的包内容没有变），否则生成器以退出码 3 退出（要重烤镜像，先问用户）。P0 的 `target_commit` = `base_commit`（无探针补丁）；P1 与 P2 的 `files` 相同，只差 `groups`。

## 7. 注入与自检开关

一套开关给所有工具与探针共用（设计稿 §2.4 末两段）：干预检验的剂量—反应、E0 的已知答案与分辨力、各项阳性对照都用它。实现只有 `capcommon.inject`。

### 7.1 注入规格 InjectSpec

| 键 | 取值 | 说明 |
|---|---|---|
| `site` | 位点名，`^[A-Za-z0-9_.]{1,64}$` | 由各组件在 README 里列出自己支持的位点（探针用设计稿 §2.4 的编号，如 `F2`、`G3`；工具用 `recv.chunk`、`send.req`、`relay.msg` 这类名字）。**不认识的位点 → 启动时退出码 3**（防「开关没接上」） |
| `mode` | `busy` / `sleep` / `miscount` / `drop` / `short` | 见 7.2 |
| `delta_ns` | 整数 > 0 | `busy`、`sleep` 必需 |
| `every_n` | 整数 ≥ 1，缺省 1 | 每 n 次经过作用一次 |
| `phase` | 0 ≤ phase < every_n，缺省 0 | 第 h 次经过（h 从 1 起）被选中 ⇔ `(h − 1) % every_n == phase` |
| `miscount_delta` | 非零整数 | `miscount` 必需：被选中的那次，计数加 `1 + miscount_delta`（−1 = 这一次不计） |
| `short_tokens` | 整数 ≥ 1 | `short` 必需：被选中的请求少出这么多 token |
| `where` | `inline`（缺省）/ `aux_thread` | `busy` 专用：`aux_thread` = 把忙等交给同进程的一条辅助 Python 线程去做（与调用方抢 GIL，E0 场景 3） |
| `role` | 角色名 | **只在档位表里出现**，指明这条规格给哪个角色；命令行与环境变量里的规格作用于本进程，不带 `role` |

文本形式（命令行 `--inject` 与 `CAPPROBE_INJECT`）：`site=F2,mode=busy,delta_ns=20000,every_n=1`；环境变量里多条用 `;` 分隔。解析：`capcommon.inject.parse_inject_specs(text)`；JSON 形式：`InjectSpec.from_dict` / `to_dict`。同一进程里同一 `site` 只能有一条生效的规格：启动级（命令行 / `launch_inject`）与本档的 `inject` 撞了同一 `site` → 档位表校验失败。

### 7.2 语义

| `mode` | 位点怎样调用 | 效果 |
|---|---|---|
| `busy` | `inj.hit()` | 被选中时自旋到 `now_ns() ≥ t0 + delta_ns`（纯 Python，持 GIL）；`where=aux_thread` 时把一次自旋交给辅助线程，调用方立即返回 |
| `sleep` | `inj.hit()` | 被选中时 `time.sleep(delta_ns / 1e9)`：阻塞调用它的线程（在事件循环上就阻塞循环），不占 CPU（E0 场景 4） |
| `miscount` | `n = inj.count_delta()`，用 `n` 代替 1 去加计数 | 被选中时返回 `1 + miscount_delta`，否则 1（C6 的失败样本） |
| `drop` | `if inj.should_drop(): 丢弃这一项` | 被选中时返回真（C7 的失败样本：排空端 / 负载器丢块） |
| `short` | `k = inj.short_by()`（每请求调一次） | 被选中时返回 `short_tokens`，否则 0（C4 的失败样本） |

没有生效规格的位点调用 `hit()` 等只做一次属性判断就返回（`Injector.active` 为假）。

### 7.3 心跳（干预有效性的正面证据）

每个位点三个 counter 槽：`inject.<site>.hits`（经过次数，规格未生效时也计）、`inject.<site>.applied`（被选中并作用的次数；`aux_thread` 在辅助线程做完时才加）、`inject.<site>.applied_ns`（实际忙等 / 睡眠的总 ns）。它们同时出现在快照（§6）与逐档自报的 `inject` 列表（§5.4.1）。

**选中按「生效以来」计**：`set_spec` 时记下当时的累计经过次数；之后第 h 次经过（h 从 1 起，**从生效起数**）被选中 ⇔ `(h − 1) % every_n == phase`。逐档注入在某档开始时生效，所以选中哪几次与之前各档经过了多少次无关。逐档自报的每项（`inject.tier_report`）给出本档增量 `hits_delta`、`applied_delta`、`applied_ns_delta`，本档始末的「生效以来经过次数」，以及 `expected_applied_delta = expected_applied(末) − expected_applied(始)`（`capcommon.inject.expected_applied_delta`；本档内生效发生过变化、或没有生效的规格时为 `null` ⇒ 该位点的心跳核对 `NOT_RUN`）。

分析时先核对 `applied_delta == expected_applied_delta`，且 `hits_delta` 等于该位点在本档的应有经过次数（每消息位点 = §5.8 对账链给出的应有消息数；每请求位点 = 应发请求数），再读边界；不符则该次干预 `NOT_RUN`（陷阱清单 §7）。`aux_thread` 另报 `aux_pending_at_end`（已交给辅助线程、还没做完的忙等次数）；停止时它不为 0 ⇒ 计入不符。

### 7.4 节拍消息的步号戳（节拍源 → 中继 → 排空端）

E0 已知答案链上的消息由 capkit 自己产生，不是产品 protobuf，所以消息前 16 字节固定为步号戳：`struct.pack("<qq", step, planned_rel_ns)`（`capcommon.schedule.STEP_STAMP`）。中继原样转发字节；排空端每收到一条就用戳算「收到时刻 − 计划时刻」（X2 的消息层滞后），不需要跨进程连接记录。节拍源 `--stamp` 打开时盖戳；给产品前端用的预制字节不盖戳。

## 8. 清空门协议（C11）

档位表里每两个负载档之间是一个清空档（§3.2 第 3 条）。清空档里没有新到达，上一档没走完的流照常走完；编排器在清空档内轮询积压读数，判断「门过了没有」。下一档的开始时刻是档位表里写死的（不随门早过而提前），所以工具之间不需要任何换档通信；门在清空档结束前没过，编排器就停止整次启动。

### 8.1 门的配置（档位表里清空档的 `gate`，或表级 `default_gate`）

```json
{"mode":"enforce",
 "rule":"abs_max",
 "readings":[
   {"name":"loadgen_inflight","terms":[{"role":"capload","slot":"backlog.inflight","sign":1,"agg":"sum"}],"abs_max":0},
   {"name":"grpc_in_flight","terms":[{"role":"stub","slot":"backlog.yielded","sign":1,"agg":"sum"},
                                     {"role":"frontend","slot":"backlog.read","sign":-1,"agg":"sum"}],"abs_max":0}],
 "poll_ns":100000000,
 "settle_polls":3,
 "deadline_guard_ns":2000000000}
```

| 键 | 含义 |
|---|---|
| `mode` | `enforce`：门在期限内不过就停止启动。`record_only`：只记录不停止（C11 的阳性对照：故意不等清空就进下一档，判定器必须报红） |
| `rule` | `abs_max`：每个读数 ≤ 它的 `abs_max`。`baseline_band`：每个读数落在 `band_ref` 指向的已登记标定文件（§9）给出的噪声带内。E0 第一轮还没有 `pilot.json`，只能用 `abs_max` |
| `readings[].terms` | 一个读数 = 若干项之和：每项取 `role` 角色全部进程的快照里名为 `slot` 的槽，按 `agg`（`sum` / `max`）合并，再乘 `sign`（±1）。**槽名必须以 `backlog.` 开头**（各组件 README 的槽清单里标出） |
| `poll_ns` | 轮询周期 |
| `settle_polls` | 连续这么多次轮询都满足才算过（单次读数可能恰好在两次快照之间） |
| `deadline_guard_ns` | 期限 = 清空档结束 − 这个值；留给停止信号送达的时间 |
| `max_snapshot_age_ns` | 一份快照多旧就不能用（§8.3）。随档位表登记 |

编排器启动时核对（否则退出码 3）：`poll_ns ≥ 2 × 快照周期`（每次轮询都应看到每个进程至少一份新快照）、`max_snapshot_age_ns ≥ 2 × 快照周期`。

### 8.2 各角色必须发布的积压读数（`backlog.` 槽）

| 角色 | 槽 | 含义 |
|---|---|---|
| 负载器（`capload`、`capgrpc.load`） | `backlog.inflight` | 已发出、还没完成或失败的请求数 |
| 负载器 | `backlog.unread_bytes` | 最近一次 FIONREAD 抽样的合计（只作记录，不进门：它可以在门过之后合法地非零一瞬） |
| 空汇（`nullsse`、`nullgrpc`） | `backlog.active_streams` | 正在发的流数 |
| 节拍源（`nullgrpc --pacing beat`） | `backlog.pending_msgs`、`backlog.active_streams`（gauge）；`backlog.yielded`（counter，已交给 grpc 的消息累计） | 待发队列里的消息数；活跃流数；与读者的 `backlog.read` 相减得 gRPC 在途数（S-PB 相同，见勘误 B2B-6） |
| 中继 | `backlog.queue_msgs`、`backlog.active_streams` | 中继内部队列的消息数；活跃流数 |
| 排空端（`capgrpc.load --stamped`；`capgrpc.oe_sink` 同，见勘误 FT1-11） | `backlog.active_streams`（gauge）；`backlog.read`（counter，已读消息累计） | 还没收到末条消息的流数；见上一行 |
| 探针（产品进程，第二批起） | 设计稿 §2.8 列的各项：前端每请求队列、gRPC 已 `yield` / 已读计数、IPC 已发 / 已收计数、worker 待取字典 | 由 B3 在 README 的槽清单里给出名字；两端计数之差用两个 `terms` 表达 |

### 8.3 编排器的动作与记录

1. 轮询按**计划时刻**做：清空档开始起每 `poll_ns` 一次（`capcommon.gate.planned_polls(start, end, poll_ns)`），直到门过或期限。每次轮询用 `capcommon.gate.snapshots_by_role(launch.json 的 shm_dir, roles["procs"], launch_key, gate.max_snapshot_age_ns, prev_seqs=上一次轮询的 {pid: seq})` 读全部登记进程的快照：pid 或 `starttime_ticks` 与登记不符、`identity` 的 launch key 不是本次启动的、快照比 `max_snapshot_age_ns` 旧（进程崩溃后留下的「`inflight = 0`」就是这样被拒的）、`seq` 没有比上一次轮询前进——这样的快照不用，记进该次轮询的错误表。再用 `evaluate_gate(gate, snaps, band)` 算每个读数，用 `GateTracker` 计连续满足次数。缺快照、缺槽、快照不可用的读数算「不满足」，不当 0。
2. 连续 `settle_polls` 次全部满足 → 门过，记 `t_pass_rel_ns`，停止轮询，等档位表上的下一档照常开始。
3. 到期限仍未过：`enforce` → 写 `control/stop.json`（`reason: "drain_gate_timeout"`、`after_tier_seq` = 本清空档的前一档），给全部工具发 `SIGTERM`，之后的档全部记 `NOT_RUN`（原因「清空门超时」）；`record_only` → 只记录，照常继续。
4. 写 `tiers/<本清空档>/drain_gate.json`（`capkit.drain_gate/1`）：`gate`（配置原样）、`source`（配置来自本档还是 `default_gate`；`baseline_band` 时还有所用标定文件的 relpath 与 sha）、`polls`（每次轮询一项 `{"planned_rel_ns","actual_rel_ns","now_ns","values":{读数名: 值},"details":{…},"snapshots":{pid: gate.snapshot_fields(…)},"errors":{pid: 原因}}`）、`n_polls`、`passed`、`t_pass_rel_ns`、`deadline_rel_ns`、`action`（`none` / `stopped` / `recorded_only`）。
5. 判定器的 C11：用 `gate.check_snapshot` 对每次轮询记录的 `snapshots` 字段按同一规则复核（不能复读 `/dev/shm`，所以规则作用在记录下来的字段上），再按记录的读数重算满足与否；应有轮询次数 = `len(planned_polls(清空档开始, t_pass 或期限, poll_ns))`，与 `n_polls` 和逐次的 `planned_rel_ns` 对账，对不上 ⇒ C11 红。下一档的前一个清空档复核后 `passed` 为真才算该档过了清空门；`record_only` 而未过 → 下一档 C11 红；应有轮询次数为 0 → `NOT_RUN`。

## 9. 标定文件、登记表与台账

### 9.1 sha256 与规范 JSON（`capcommon.hashing`）

- 文件 sha：`sha256_file(path)`，对文件全部字节，64 位小写十六进制。登记、核对一律用它。
- 对象 sha（硬件画像等「去掉某些键后」的校验值）：`canonical_json_sha256(obj)` = 对 `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` 的 UTF-8 字节求 sha256。浮点按 Python `repr` 输出；需要跨机器稳定的量先转成整数或字符串再放进去。

### 9.2 登记表 `register.json`（每个实验一份或多份，放在实验目录的 `register/`）

`schema: capkit.register/1`。它是一次启动「事先登记」的全部机器可读内容；`--register-sha` 就是它的 sha256。新增的必填段 `environment` 见勘误 R1-13。

```json
{"schema":"capkit.register/1","experiment":"E0","register_md":{"relpath":"02_e0/E0/register/REGISTER.md","sha256":…},
 "arms":["nullsse-a","sfe-a"],
 "files":{"tier_table":{"relpath":…,"sha256":…},"schedule":{"relpath":…,"sha256":…},
          "prompts":{"relpath":…,"sha256":…},"scenario":{"relpath":…,"sha256":…}},
 "calibration":[{"kind":"pilot","relpath":"02_e0/E0/analysis/pilot.json","sha256":…}],
 "inline_thresholds":{…只允许 E0 用：E0 还没有 pilot.json 时登记的绝对门槛…},
 "expected":{"launches_per_arm":3,"job_budget":4},
 "judge":{"mode":"P+X","checks":["C1","C3","C5","C6","C7","C8","C9","C10","C11","C12","C13"]}}
```

- `relpath` 一律相对 `<W>`。各臂的档位表 / 时间表不同时，不写 `files`，改写 `files_by_arm: {"<臂名>": {…与 files 同形…}}`；两个键恰好出现一个。
- `calibration[].kind` ∈ `pilot`（E0 的 `pilot.json`）、`loadgen_band`（E1 的负载源健康带）、`stub_calib`（E2 的桩标定表）、`near_boundary_noise`（各实验第一步量的近边界噪声）。判定器**只**从这些文件与 `inline_thresholds`（仅 E0）读阈值（哪些文件仍是阈值源、`inline_thresholds` 的限制与新增的 `d14_result`、`arm_config`、`arm_threshold_config`、按臂 `ladder` 已改，见勘误 calib-v2-A2），不接受命令行阈值（DATA-AND-ANALYSIS §4 第 6 条）。后三种的格式见 §9.6～§9.8；`loadgen_band`、`stub_calib` 的登记项另带必填的 `bind`，`judge` 段可另带 `near_boundary_tiers`（见勘误 B2-2A-1、B2-2A-2）。`judge.mode` 另有 `P+X+I+V`，该模式下可带 `localize`（见勘误 B2-2E-1、B2-2E-3）。

### 9.3 `<W>/REGISTRY.tsv`（只追加）

列（第一行是表头，制表符分隔）：

| 列 | 内容 |
|---|---|
| `registered_at` | ISO8601 UTC，**毫秒精度**，带 `Z`（`capcommon.clock.wall_iso()`，形如 `2026-11-13T08:00:00.123Z`）。全部台账（`REGISTRY.tsv`、`submissions.tsv`、`JUDGED.tsv`）与 `STATE` 用同一格式；`parse_wall_iso` 只认这一种，秒精度的旧格式报错。「登记早于提交」按毫秒严格小于判，同一毫秒算不早于（判定器判 `INPUT_INVALID`，宁严勿宽） |
| `experiment` | 如 `E0` |
| `relpath` | 相对 `<W>` |
| `sha256` | 文件 sha |
| `note` | 说明；以 `kind=<种类>;` 开头（`register`、`register_md`、`tier_table`、`schedule`、`prompts`、`scenario`、`pilot`、`loadgen_band`、`stub_calib`、`near_boundary_noise`、`prediction`、`hw_profile`；`d14_rule` 见勘误 R1b-4；calib-v2 新增的种类见勘误 calib-v2-A2 第 6 条） |

用 `capcommon.ledger.registry_append(path, experiment, relpath, sha256, note)` 写（文件不存在则先写表头；同一 sha 已登记过则拒绝，提示用已有的那行；只用追加模式打开，从不重写）。查：`registry_lookup(path, sha256) -> 行 | None`。

### 9.4 `pilot.json`（E0 的试点产出，`schema: capkit.pilot/1`）

由 B5 的 `capjudge.pilot` 从 E0 的亚饱和档构建；只从 `judge/` 输出与原始数据算，不手填。（判定器已不把本格式当阈值源，登记了即 `INPUT_INVALID`，见勘误 calib-v2-A2 第 1 条。）`d14_softirq` 的算法与形状见勘误 R1b-4。

```json
{"schema":"capkit.pilot/1","experiment":"E0",
 "environment":{"hw_profile_sha":…,"image_sha":…,"capkit_sha":…,"baseline_sha":…},
 "inputs":[{"tree":"02_e0/E0/runs/J…/","judge_verdicts_sha256":…}],
 "noise_bands":{"<统计量键>":{"regime":"subsat","min":…,"max":…,"n":…,"unit":"ns","source_tiers":[…]}},
 "window_ns":…,"warmup_ns":…,"tier_resolution":{…},"launches_per_arm":…,"min_detectable_diff":{…},
 "i2_calibration":{"enters_criteria":false,"per_loop_impl":{"uvloop":{…曲线…},"asyncio":{…}}},
 "d13_switchinterval_s":{…},"d14_softirq":{…},"core_budget":{…},"drain_max_wait_ns":…,
 "canary_band":{…}}
```

- 统计量键的写法：`<T 代号>:<量>:<对象>`，如 `T1:send_lag_p50_ns:capload`、`T2:loop_lag_p50_ns:relay1`。量与对象的完整清单由 B5 在 `capjudge` 的 README 里给出；判定器查不到某个键 → 对应检查 `NOT_RUN`（不是通过）。带的 `regime` 决定它用在哪些档上（见勘误 B2-2A-2）。
- `n` 是样本数；噪声带的名义误报率 1 ÷ (n + 1) 由读者算，不存。可选键 `i1_calibration` 给 I1 的「接近 1 核」门槛（见勘误 B2-2E-2）。
- `window_ns`、`warmup_ns` 按臂分别搜索，另有 `window_ns_by_arm`、`warmup_ns_by_arm`、`window_search_by_arm`、`warmup_search_by_arm`；顶层两项是各臂都有值时的最大值，否则 null（见勘误 SM-16）。
- 「同一环境」：判定器核对 `environment.hw_profile_sha`、`image_sha` 与被判启动树的一致（不一致 → 该树 `INPUT_INVALID`，除非画像已按 DATA-AND-ANALYSIS §8.1 登记为可比，判定器在 `hw_gate.json` 里读这个登记）。

### 9.5 `<W>/submissions.tsv`（作业台账，B6 的 `capjobs.submit` 写）

只追加的**事件行**：一个作业的「计划」「已提交」「已结束」各一行，用 `plan_id` 串起来，当前状态 = 同一 `plan_id` 的最后一行（`capcommon.ledger.submissions_current(path)`）。这样满足 CONSTRAINTS §8「每个作业一行、先计划后补号再补状态」的信息要求，又不原地改写台账（见 §12）。列：

`event_at, plan_id, event, job_id, status, reason, experiment, arms, launches, basis, account, partition, qos, exclude, time_limit, mem, cpus_per_task, gpus, exclusive, snapshot_dir, snapshot_sha256, sbatch_argv, submitted_at, mounts_json, tokenizer_dir_sha256, assertions_json`

- `event` ∈ `PLANNED` / `SUBMITTED` / `ENDED` / `NOTE`；`status` ∈ 空 / `VALID` / `VOID` / `NOT_RUN` / `ENV_INVALID` / `CANCELLED` / `SUBMIT_FAILED`（另有 `UNKNOWN_SUBMIT`，见勘误 R1-14）。
- `basis`：「依据哪条批准」的原文（CONSTRAINTS §2 第 6 条）；`PLANNED` 行缺它 → 拒绝提交。
- `gpus` 是申请的 GPU 张数（> 1 拒绝提交）；`sbatch_argv` 是 JSON 数组形式的完整命令。
- 写：`capcommon.ledger.submissions_append(path, row_dict)`（缺列或多列即报错）。
- **提交前的五项断言**（CONSTRAINTS §2「靠机制落实」第 1、2、5、6 条；`capcommon.policy`）：`capjobs.submit` 在写 `PLANNED` 行之前调用 `policy.check_no_model_submission(experiment, gpus, mounts, tokenizer_dir, basis)`，按固定顺序断言 ① 实验编号在 `policy.NO_MODEL_EXPERIMENTS` 里（E0～E9 及 E2a、E2b、E15-A；E10 替身臂与 E13 主体与要加载模型的部分共用编号，**不在表里**，要先由用户定出独立编号、在本文件追加勘误）；② 申请的 GPU ≤ 1；③ 挂载清单里没有任何一项等于、位于或包含模型库路径（`policy.FORBIDDEN_MOUNT_PATHS`；挂 `/home` 也算包含）；④ tokenizer 副本里没有权重后缀（`policy.WEIGHT_SUFFIXES`）、没有软链，并算出 `tokenizer_dir_sha256`（不用 tokenizer 的作业写 `none`）；⑤ 依据原文非空。任一不过即退出、不写台账、不提交（受门控的登记表另要过六道提交门并写 `BUDGET.tsv`，见勘误 calib-v2-A4 第 5 条；哪些登记表受门控以勘误 calib-v2-A6 第 1 条为准）。`PLANNED` 行必须带 `mounts_json`（挂载清单 JSON 数组）、`tokenizer_dir_sha256`、`assertions_json`（五个键全为 `true`），`submissions_append` 在缺列、断言不全为真、挂载复核不过、实验编号不在表里时拒绝写入。`probe` 子命令的命令行必须过 `policy.check_test_only_argv`（`sbatch` 之后有 `--test-only`）。tokenizer 拷贝作业（唯一允许挂模型库的作业）走另一套断言，见勘误 R1b-5。
- **本机运行**（冒烟、判定器自测）的作业号是 `local-<任意>`，由 `capjobs.submit --local-record` 写 `PLANNED` 与 `SUBMITTED` 两行（同样过五项断言，`gpus = 0`、`sbatch_argv` 为空），判定器对它们**不放宽**任何规则。

### 9.6 `loadgen_band`（E1 的负载源健康带，`schema: capkit.loadgen_band/1`）

由 `capjudge.calib loadgen-band`（`analysis/calib_build.py` 同一实现）从已判过的 E1 启动树构建，不手填。按「形态」（shape）给出每 worker 的健康上界与健康量的带；形态名由构建者在列表文件里起，登记表用 `bind` 把本实验的角色绑到形态上（勘误 B2-2A-1）。

```json
{"schema":"capkit.loadgen_band/1","experiment":"E1",
 "environment":{"hw_profile_sha":…,"image_sha":…,"capkit_sha":…,"baseline_sha":…},
 "inputs":[{"tree":…,"judge_verdicts_sha256":…,"role":"ld","shape":"http-r-ka","tiers_used":[…],"tiers_excluded":{…},"boundary_T4":{…},
           "attribution":{"grade":…,"role":…,"tier_seq":…,"why":…}}],
 "shapes":{"<形态>":{"match":{"tokens_per_msg":[1],"streams_c":[null]},
                     "per_worker_max":{"load_key":"rate_rps","value":…,"censored":false,"n_launches":3},
                     "bands":{"T1:send_lag_p50_ns":{"min":…,"max":…,"n":…,"unit":"ns","source_tiers":[…]},
                              "T3:send_lag_p99_ns":{…},"T3:fionread_nonzero_frac":{…},"T3:loop_lag_p99_ns":{…}}}}}
```

- `bands` 的键只能是这四个统计量（`<T>:<量>`，对象就是被绑定的角色），每个带 `min ≤ max`、整数 `n ≥ 1`；值是列表里每棵树入选档（判定为 STABLE、没有 RED 检查）上每个 worker 的读数的 [最小, 最大]（T3）。
- `per_worker_max`：每次启动取 `verdicts.json` 的 `boundary_T4` 下端（最高 STABLE 档）上最忙 worker 的计划负载（档的总负载 × 该 worker 在时间表里的份额），多次启动取最小（T4 包络下端）。（取哪一档、没有该档时的报错与 `censored` 条件已改，见勘误 calib-v2-A1 第 1 条。）`load_key` ∈ `rate_rps`、`nominal_msgs_per_s`（由档参数算，§3.1）。列表里参与构建的启动，T4 没有下端或该档的计划负载算不出 ⇒ 构建报错并写明是哪棵树（退出码 3），不从最小值里静默去掉它；要不要这棵树由构建者从列表里显式删。`censored: true`（上界只是「不低于」）的条件：贡献了上界的启动里，有一次从没 UNSTABLE，或它的结论等级没有把第一个非稳定档定位到被绑定的角色（`grade.role` ≠ 该角色，包括按 `P+X` 判、没有等级的启动）：到顶的是别的组件或谁都没到顶时，这个数只说明该角色至少能到这里。每个输入的 `attribution` 写明这次启动的等级、定位到的角色与理由。判定器对 `censored` 的上界只在计划负载不超过它时比较，超过 ⇒ 该项 NOT_RUN（不判 RED，勘误 B2-2A-3）。
- `match`：贡献了读数或上界的档的 `tokens_per_msg` 与 `streams_c`（R 轴为 null）的集合。判定器只在被判档落在 `match` 里时比较健康上界。

### 9.7 `stub_calib`（E2 的桩标定表，`schema: capkit.stub_calib/1`）

由 `capjudge.calib stub-calib` 构建。本格式只定义**域外**部分 `out_of_domain`（形态的写法与 §9.6 相同，只是上界键叫 `per_process_max`、份额按 worker 均分（同一个监听端口，没有计划的分配；档的总负载当作桩的负载，只在启动恰有一个 `kind: client` 角色时成立：每个客户端角色各跑整张时间表，几个客户端角色怎样分到同一个桩没有计划，这时桩的计划负载算不出，构建报错、判定器 C2 的负载项 NOT_RUN），`bands` 的键是 `T3:loop_lag_p99_ns`（桩 worker 摘要里的事件循环延迟直方图 p99）与 `T3:thread_cpu_cores`（该 worker 进程全部线程在窗口内的 utime + stime，按 sidecar `threads.jsonl` 首末两行，单位：核；有线程在窗口内只有一行 ⇒ 它的 CPU 不知道，这一项没有值，不按其余线程之和算））。`in_domain` 必须为 null 或不出现：域内桩的自身需求表由后续子批次定义（设计稿 §2.6「域内桩」）。

```json
{"schema":"capkit.stub_calib/1","experiment":"E2a","environment":{…},"inputs":[…],
 "out_of_domain":{"<形态>":{"match":{…},"per_process_max":{"load_key":"nominal_msgs_per_s","value":…,"censored":…,"n_launches":…},
                            "bands":{"T3:loop_lag_p99_ns":{…},"T3:thread_cpu_cores":{…}}}},
 "in_domain":null}
```

### 9.8 `near_boundary_noise`（各实验第一步量的噪声带，`schema: capkit.near_boundary_noise/1`）

由 `capjudge.calib near-boundary --regime {subsat,near_boundary}` 构建：与 `pilot.json` 的 `noise_bands` 同一算法（判定器 `verdicts.json` 里各档 `stats` 的 [最小, 最大]），只是入选档另外剔除 UNSTABLE 与 UNDETERMINED（X 成立、原因未定的档）（pilot 只剔除 VOID 与 RED）。

```json
{"schema":"capkit.near_boundary_noise/1","experiment":"E1","regime":"near_boundary",
 "environment":{…},"inputs":[{"tree":…,"judge_verdicts_sha256":…,"tiers_used":[…],"tiers_excluded":{…}}],
 "noise_bands":{"<统计量键>":{"regime":"near_boundary","min":…,"max":…,"n":…,"unit":…,"source_tiers":[…]}}}
```

- 文件级 `regime` ∈ `subsat`、`near_boundary`，每个带的 `regime` 必须写出且与文件级相同。
- 以 `capkit.pilot/1` 格式写的文件也接受登记为本类（RUNBOOK 阶段 2 的 D5：E1-0 的带由 `capjudge.pilot` 构建）：只读其 `noise_bands`，各带按自己的 `regime`（缺省 `subsat`）。（本类与本格式都已不是阈值源，登记了即 `INPUT_INVALID`，见勘误 calib-v2-A2 第 1 条；构建器照旧可用。）
- 本类可以登记多份（如一份 pilot 格式的 `subsat`、一份 `near_boundary`），每个 (键, regime) 仍只能有一个来源（勘误 B2-2A-1、B2-2A-2）。（单份、多份都已不接受，见勘误 calib-v2-A2 第 1 条。）

## 10. 判定器的输入、输出与退出码

### 10.1 命令（逐字固定）

```
python3 -m capjudge --tree <启动树的路径> --workdir <W>
```

- 只接受这两个参数（外加 `--help`）。`--tree` 必须是一个已存在的目录、含 `raw/launch.json`、路径里没有 `* ? [` 字符；不接受多个树、不接受通配符。阈值、标定文件全部从 `launch.json → register.json` 取（§9.2）。
- 在 VM 上的完整命令是 `PYTHONPATH=<C>/tools python3.11 -m capjudge --tree <T> --workdir <W>`。
- **一棵树只判一次**（流程已改，见勘误 R1-16）：判定器先查 `<W>/JUDGED.tsv`（列 `judged_at, job_id, restart_count, launch_seq, arm, experiment, tree_relpath, verdicts_sha256`），同一 `(job_id, restart_count, launch_seq, arm)` 已在表里 → 退出码 12（防止把树拷到别处再判一次）；再 `os.mkdir(<tree>/judge)`，已存在 → 退出码 12。两步都不写任何东西就退出。判完在 `JUDGED.tsv` 追加一行。
- 输入核对在 `judge/` 建好之后做（已改为先写尝试目录，见勘误 R1-16；任一不过 → 在 `judge/inputs.json` 写明哪一项，退出码 13，不追加 `JUDGED.tsv`；`judge/` 已存在，所以同一棵树也不能再判）：`launch.json` 的 `register.sha256` 在 `REGISTRY.tsv` 里，且 `registered_at` 早于 `submissions.tsv` 里该作业的 `submitted_at`（毫秒，严格小于；本机运行的 `local-*` 作业同样要有 `--local-record` 写的台账行）；`launch.json` 的 `reconcile` 过 `expect.validate_reconcile`；`register.json` 列出的每个文件现算 sha 与登记一致；每个标定文件同样已登记且早于提交；树内全部文件的 `identity` 前六个字段一致（§2.1）。
- 另核 `launch.json` 的 `job_inputs` 与登记场景声明的作业输入一致、树里的拷贝与记录未变（见勘误 B2B-10）。

### 10.2 输出（全部在 `<tree>/judge/`，不写 `raw/`）

| 文件 | 内容 |
|---|---|
| `inputs.json`（`capkit.judge.inputs/1`） | 判定器代码 sha（`CAPKIT_CODE_SHA` 或 `git -C <C> rev-parse HEAD`，取不到写 `null` 并把整次判定降为 `NOT_RUN`）、读过的每个文件的 relpath 与 sha256、所用标定文件的 kind / relpath / sha256 |
| `verdicts.json`（`capkit.judge.verdicts/1`） | 每档一条：`{"tier_seq":…,"tier_label":…,"kind":…,"verdict":"STABLE"|"UNSTABLE"|"VOID"|"NOT_RUN"|"UNDETERMINED","reasons":[…],"checks":{…}}`；启动级：`launch_verdict`、`exit_code`（另有 `mode`、每档 `X.I` / `X.V`、顶层 `localisation_I`、`intervention_V`、`grade`，见勘误 B2-2E-1、B2-2E-4；`CONTRA`、`zstats`、`levels`、`boundary`、`t7`、`c13_mean` 见勘误 calib-v2-A3） |
| `checks.tsv` | 一项检查一行：`tier_seq, check_id, expected_count, checked_count, red_count, status(PASS/RED/NOT_RUN), source_of_expected, detail` |
| `judge.log` | 过程日志 |

- 每项检查的 `expected_count` 来自配置（时间表 `meta.per_tier`、`launch.json` 的 `roles[].workers` 与 `expected_procs`、`reconcile`、`chunk_sample_mod`、覆盖清单的 `sites`、窗口秒数），写明 `source_of_expected`；`expected_count == 0` 或 `checked_count == 0` → 该项 `NOT_RUN`（红证据优先，见勘误 R1-17）。
- **各 C 项作用在哪些角色上，只看 `launch.json` 里每个角色的 `measure_class`**（§11 场景配置），不按角色名或模块名判断：C1（负载源健康）作用于 `load_source`；C2（桩健康）作用于 `stub_out_of_domain`（域外桩：空汇、排空端、节拍源、S-PB / S-FE / S-NL）与 `stub_in_domain`（只查自身需求，不进 P）；被测部件（`under_test`）不做健康检查；`observer`（sidecar、金丝雀）只查它们自己的数据是否齐全。C1、C2 的应检进程数 = 对应类别角色的 `expected_procs` 之和（实际按 `workers` 计，见勘误 B2-2A-4）。
- 几项检查的具体口径：
  - **C9**（绑核读回一致）：用 `sidecar/threads.jsonl` 窗口内的逐秒 `cpus_allowed` 与 `last_cpu`，与 `pin_table.json` 核对；应检条数 = 窗口秒数 × 登记线程数；任一行亲和集合与他角色的核相交、或越出本角色的核 ⇒ 红。
  - **C10**（版本自报）：`versions.check_versions`（§2.5）；不过 ⇒ 整次启动 `NOT_RUN`（不是红）。
  - **C13**（线程清单与独占核集；辅助进程见勘误 R1-5）：前一半 = 启动时的线程清单与每档窗口始末的 `threads` 全部认出（§5.3）；后一半 = 对每个 `under_test` 角色的核集逐核算「`cpus.jsonl` 里窗口内的忙碌 tick（user+nice+system+irq+softirq+steal）− 绑在这些核上的已登记线程的 `utime + stime` 增量」，减去按 D14 结果（`pilot.json` 的 `d14_softirq`）认定属于本实验的软中断部分（只减 softirq，见勘误 R2-14），剩余超出噪声带 ⇒ 红；`pilot.json` 还没有 D14 结果时这一半记 `NOT_RUN`。（D14 的来源已改为 `d14_result`，见勘误 calib-v2-A2 第 7 条。）（上限改读 pilot/2 的 `c13_cap`，另有启动级 `C13-MEAN`，见勘误 calib-v2-A3 第 5 条。）
  - **GPU-IDLE**（启动级，CONSTRAINTS §2 第 4 条；检查号不用 `G1`，避免与设计稿的位点 G1 撞名）：应检条数 = 测量窗口秒数之和 × `gpu.jsonl` 头里的 `gpu_count`（已改，见勘误 R1-18）；`gpu_idle.json` 的阳性对照不过、`gpu_count` 为 0、或 `gpu.jsonl` 没有窗口内的行 ⇒ `NOT_RUN`；窗口内有任何 `query_rc ≠ 0` 的行也记 `NOT_RUN`；有任何 `util_pct > 0` 或非空的 `compute_pids` ⇒ 整次启动 `VOID`（`STATE` 写 VOID，并上报）（只算不属于本次启动的进程，见勘误 R2-15）。
- 档的判定：前提 P 任一项 `RED` → `VOID`；任一项 `NOT_RUN` → `NOT_RUN`；P 全过时看 X：成立 → `UNSTABLE`，不成立 → `STABLE`；X 的主量与对账量方向矛盾 → `VOID`（设计稿 §2.6）。（矛盾改判 `CONTRA`，不作废，见勘误 calib-v2-A3 第 6 条。）第一批只实现 P 与 X；I、V 是第二批，届时 `UNDETERMINED` 才会出现（已实现：模式 `P+X+I+V`，`UNDETERMINED` 的条件与新增输出见勘误 B2-2E-4）。清空档不判稳定性，只给 C11 的结果。
- 判定器写完 `judge/` 之后在 `<tree>/STATE` 追加一行（`by = capjudge`，原因 = `launch_verdict` 与 `verdicts.json` 的 sha 前 12 位）。

### 10.3 退出码

| 码 | 名字 | 条件 |
|---|---|---|
| 0 | `PASS` | 全部被判的档都不是 `VOID` / `NOT_RUN`（`UNSTABLE` 是正常的测量结果，不算失败；`UNDETERMINED` 同样，见勘误 B2-2E-5） |
| 1 | — | 未捕获的异常 |
| 2 | `USAGE` | 参数错误 |
| 10 | `RED` | 至少一档 `VOID`，或启动级检查（C9、C13、GPU-IDLE 等）红 / VOID。C10 不过不在这里，它按 D11 记 `NOT_RUN`（码 11） |
| 11 | `NOT_RUN` | 没有 `VOID`，但至少一档 `NOT_RUN`；**空目录、没有任何负载档也是这个码**（有 `ladder.json` 的启动与 `C13-MEAN` 的判法见勘误 calib-v2-A3 第 7 条） |
| 12 | `ALREADY_JUDGED` | 见 10.1 |
| 13 | `INPUT_INVALID` | 见 10.1 |

优先级：2 > 12 > 13 > 10 > 11 > 0。

### 10.4 判定器自测样本（`<C>/samples/`）

每个样本是一个**最小工作目录 W**：一棵完整的启动树，加上它自己的 `REGISTRY.tsv`、`submissions.tsv`（`local-*` 作业，用 `--local-record` 的格式）、空的 `JUDGED.tsv`、登记表与被登记的文件，再加一个 `expect.json`（`{"exit_code":…,"tier_verdicts":{…},"red_checks":[…]}`）。自测脚本把整个样本 W 拷到临时目录，用 `--workdir <拷贝的 W>` 判（判定器只判一次；不放宽任何输入核对），比对 `expect.json`（`--expect-samples` 必填与新增断言见勘误 R1-20）；同时对「良品」样本做一次破坏（按样本的 `sabotage.json` 改一处）并断言变红、且断言破坏确实改动了目标文件（陷阱清单 §1）。

## 11. 各组件的最小命令行（编排器据此起进程）

「通用」= §2.1 的身份参数 + `--run-dir` + `--stop-grace-s` + `--cpus`；「快照」= `--shm-dir` + `--snapshot-interval-ms`；「档位」= `--tier-table` + `--origin-ns`。下表的参数**全部必需**（除非标「可选」），组件可以加可选参数，但不能少。编排器（B7）只用本表与场景配置里逐字给出的额外参数起进程。

| 入口 | 参数 | 发布的位点（注入）与 backlog 槽 | 输出（§5.4） |
|---|---|---|---|
| `python -m capload`（B1） | 通用、快照、档位、`--loop`、`--ready-file`、`--inject`（可选，可重复）；`--target HOST:PORT` 与 `--target-port-file PATH` 二选一；`--endpoint {/v1/completions,/v1/chat/completions}`；`--model NAME`；`--workers W`；`--schedule PATH`；`--prompts PATH`；`--conn-policy {keepalive,new-per-request}`；`--pool-pick {roundrobin,random}`；`--pool-seed INT`；`--pool-reset {per-tier,never}`；`--recv-mode {count,full}`；`--chunk-sample-mod M`；`--fionread-every-ms INT`；`--lag-period-us INT`；`--max-records-per-worker N`。请求超时**不是**命令行参数：只读档位表的 `request_timeout_ns`（§3.1），值与档位表 sha 写进 summary。请求头与计块口径按 §4.4、§4.5 | 位点 `send.req`、`recv.chunk`；`backlog.inflight`、`backlog.unread_bytes` | 每 worker 每档：`summary.json`、`requests`、`chunks`、`fionread`、`lag`；退出码 0 / 3 / 4 / 5 |
| `python -m capload.nullsse`（B1） | 通用、快照、档位、`--loop`、`--ready-file`、`--port-file`、`--inject`（可选）；`--host`；`--port`（0 = 任取）；`--workers Z`（`SO_REUSEPORT`）；`--chunk-bytes N`；`--pacing {beat,asap}`（`beat`：全进程一个 `BeatSchedule`，第 j 条内容帧在步 `first_step + expect.msg_step_offsets(N,k,s,"product")[j]` 发，§3.4）；`--idle-close-s FLOAT`；`--lag-period-us INT`。每请求的 N 取自请求体、k / s / 计划档取自请求头（§4.4），不按到达时刻查档位表；帧格式按 §4.5 | 位点 `send.chunk`（每内容帧）、`short.req`（每请求，`mode=short` 让该请求少出 token，C4 的失败样本）；`backlog.active_streams` | 每 worker 每档 `summary.json`（`counts`: `requests`、`frames_sent`（按 §4.5 口径，含 usage 帧、不含 `[DONE]`）、`bytes_sent`、`accepts`、`idle_closed`）、`streams`（§5.4.2）、`lag` |
| `python -m capgrpc.load`（B2） | 同 `capload` 的通用 / 快照 / 档位 / 时间表 / 记录参数；`--target` 与 `--target-port-file` 二选一；`--method /pkg.Service/Method`；`--channels-per-worker N`（每条 channel 设 `grpc.use_local_subchannel_pool=1`）；`--request-format {capkit,openengine}`（第一批只实现 `capkit`：`capcommon.wire.GRPC_REQ` = `struct "<qiiii"`（req_id、max_tokens、tokens_per_msg、stream_interval、计划档 tier_seq）+ prompt 字节，§4.4）；`--stamped`（按 §7.4 的步号戳算消息层滞后，作排空端用；不盖戳的流用 `--planned-msgs`，见勘误 FT1-9）；`--msg-sample-mod M` | 位点 `send.req`、`recv.msg`；`backlog.inflight`、`backlog.active_streams`、`backlog.read`（counter） | 每 worker 每档：`summary.json`、`requests`、`msgs`（`--stamped` 时）、`lag` |
| `python -m capgrpc.nullgrpc`（B2） | 通用、快照、档位、`--loop`、`--ready-file`、`--port-file`、`--inject`（可选）；`--host`、`--port`、`--workers Z`；`--msg-bytes N`；`--pacing {beat,asap}`（`beat` = 节拍源：全进程一个 `BeatSchedule` 绝对节拍器、每流一个无界待发队列、不等写完成；`beat` 时 `--workers` 必须为 1）；`--stamp`（盖步号戳）；`--lag-period-us INT`；`--max-unrequested-s INT`（可选，见勘误「Fix SM-17 slurm-view ssh failures and SM-18 gRPC unrequested-call cancellation」第 2 条）。每请求的 N、k、s、计划档取自请求头（§4.4）；每请求的消息数与出消息的步按 `expect`（`product` 闸门） | 位点 `send.msg`、`beat.step`；`backlog.pending_msgs`、`backlog.active_streams`、`backlog.yielded`（counter） | 每档 `summary.json`、`steps`、`streams`（§5.4.2）、`lag` |
| `python -m capselfcheck.relay`（B2） | 通用、快照、档位、`--loop`、`--ready-file`、`--port-file`、`--inject`（可选；四种已知答案模式都用注入表达：单级 = `site=relay.msg,mode=busy`；另一线程 = 加 `where=aux_thread`；不耗 CPU = `mode=sleep`；两级串联 = 两个中继进程首尾相连，**角色名分别为 `relay1`、`relay2`**（§0：拓扑位置不同的进程用不同角色名），档位表里的注入按角色分别给 δ，互换 δ 就是互换两条规格的 `role`）；`--host`、`--port`；`--upstream HOST:PORT` 与 `--upstream-port-file` 二选一；`--lag-period-us INT`；`--max-unrequested-s INT`（可选，见勘误「Fix SM-17 slurm-view ssh failures and SM-18 gRPC unrequested-call cancellation」第 2 条） | 位点 `relay.msg`；`backlog.queue_msgs`、`backlog.active_streams` | 每档 `summary.json`、`lag` |
| `import capprobe`（B3） | 环境变量 §6.5 | 位点见设计稿 §2.4 位点表（第一批：`count`、`lag` 两组） | 快照 §6、`ts` 落盘 |
| `python -m capprobe.decode FILE`（B3） | — | — | 把快照解码成 JSON 打到标准输出 |
| `python -m capprobe.overlay`（B3） | `--tier {P0,P1,P2}`、`--probe-tree PATH`、`--image-commit SHA`、`--base-commit SHA`、`--capkit-root <C>`、`--capkit-sha SHA`、`--packages LIST`、`--groups LIST`、`--out PATH`；可选 `--stub-module KEY=PATH`（见勘误 B2B-11） | — | §6.6 |
| `python -m capenv.sidecar`（B4；新参数见勘误 R1-10） | 通用、快照目录 `--shm-dir`、`--origin-ns`、`--roles-file PATH`、`--interval-ms INT`、`--out-dir PATH`、`--cpus-watch {all,LIST}`（LIST 必须 ⊇ 绑核表全部角色的核，§5.6）、`--gpu-every-s FLOAT`、`--numa-every-s FLOAT`、`--logs-dir PATH` | — | §5.6 |
| `python -m capenv.canary`（B4） | 通用、`--origin-ns`、`--interval-ms INT`、`--out PATH`（`--cpus` 恰好 1 个核） | — | §5.6 |
| `python -m capenv.spinners`（SB-B1，见勘误 SB-B1-1） | 通用、`--origin-ns`、`--ready-file`、`--pin-table PATH`（编排器给）、`--target-role ROLE`（可选） | — | `raw/spinners/<角色>.json`（勘误 SB-B1-5） |
| `python -m capenv.gate`（B4；新参数见勘误 R1-21） | `--run-dir`（作业目录）、`--phase {pre,post}`、`--registered-hw-sha SHA`、`--comparable-file PATH` 或 `--no-comparable`、`--slurm-view-file PATH`、`--slurm-view-wait-s FLOAT`（§5.5.2）、身份参数（`--role gate`） | — | 作业目录 `env/` 各文件；退出码 0 / 6 |
| `python -m capenv.clockcheck`（B4） | 通用、`--peer-mode {server,client}`、`--port-file PATH`、`--round-trips N`、`--back-to-back N`、`--out PATH` | — | `env/clock_check.json`（§5.5） |
| `python -m capenv.hwprofile`、`capenv.pins`、`capenv.loopback`（B4；hwprofile 新参数见勘误 R1-21） | 各自 README | — | `env/hw_profile.json`、`env/pin_table.json`、`env/loopback_softirq.json` |
| `python -m capjudge`（B5） | §10.1 | — | §10.2 |
| `python -m capjudge.pilot`（B5） | `--inputs FILE`（逐行列出启动树，不接受通配符）、`--workdir <W>`、`--out PATH` | — | §9.4（跨启动的剂量–反应另有 `python -m capjudge.intervene`，见勘误 B2-2E-7） |
| `python -m caplaunch`（B7；新必填参数见勘误 R1-1；`--job-input` 见勘误 B2B-9） | 身份参数（`--role launch`）、`--run-dir`、`--scenario PATH`、`--pin-table PATH`、快照（`--shm-dir` 给作业级根 `/dev/shm/<用户名>-<作业号>/snap`，编排器自己建本次启动的子目录）、`--origin-lead-ms INT`（**起第一个进程之前**选定 `origin = now + lead`；原点到达时还有角色没就绪 ⇒ 停止、`NOT_RUN`，§0；没有单独的就绪超时参数，就绪期限就是原点） | — | §5.2、§5.3、§5.4 的编排器部分、§8.3 |
| `python -m capstubs.fingerprint`（B7） | `--tokenizer DIR`、`--out PATH` | — | 指纹 JSON 与生成的 S-FE 命令行（S-PB 的两个入口见勘误 B2B-3） |
| `python3.11 -m capjobs.submit`（B6） | 子命令 `probe`（只做 `--test-only`，过 `policy.check_test_only_argv`）、`submit`（写 `PLANNED` 前过 §9.5 的五项断言；挂载另有覆盖要求，见勘误 R2-9）、`slurm-view`（§5.5.2；新可选参数 `--attempts`、`--retry-wait-s`，失败时退出 4，见勘误「Fix SM-17 slurm-view ssh failures and SM-18 gRPC unrequested-call cancellation」第 1 条）、`--local-record`（本机运行的台账行）、`--dry-run`；资源参数全部显式（新增 `submit --expect-node / --plan-item / --cluster-utc-offset`、`exclude --all-but / --also-exclude` 与子命令 `pin-next`，见勘误 calib-v2-A4） | — | `submissions.tsv`（§9.5）、快照目录、`env/slurm_view.json` |

场景配置（`configs/<实验>/<场景>.json`，B7 定义，`schema: capkit.scenario/1`）至少有：`roles`（每个角色一个顶层进程：`module`、`args`（逐字的额外参数）、`kind`（`server` / `client` / `aux`，只管起停顺序）、`start_after`（依赖哪些角色的端口文件）、`cpus_needed`、`numa`，以及下面这些**必填**的声明）、`tier_table`、`schedule`、`prompts`（均为相对 `<W>` 的路径；可选的 `job_inputs` 段见勘误 B2B-9）、`probe_tier`、`stop_order`、`reconcile`（§5.8，必填）。

| 角色的必填声明 | 含义 |
|---|---|
| `workers` | 工具内部的 worker 数；编排器据此生成 `--workers`（不接受场景在 `args` 里另写 `--workers`）。不分 worker 的工具与产品进程写 1 |
| `expected_procs` | 该角色应有的进程总数，含它派生的子进程（产品进程按其参数推出，如前端 + `num_postprocess_workers` 个池进程）。判定器的应检进程数一律取它 |
| `measure_class` | `load_source` / `stub_out_of_domain`（域外桩：空汇、排空端、节拍源、S-PB、S-FE、S-NL；S-NL 的三种模式与 `metronome-only` 空跑见勘误 FT1-1）/ `stub_in_domain` / `under_test` / `observer`（sidecar、金丝雀、自旋器（勘误 SB-B1-3））。同一个模块在不同实验里类别不同（E1 里 `capload` 是 `under_test`，E0 里是 `load_source`；E2a / E3 / E4 模板里各角色的类别见勘误 FT1-15）；判定器只按它决定 C 项的作用对象（§10.2） |
| `asyncio`、`uses_grpc`、`loop`、`probe_tier` | C10 的必需字段与 `loop_impl` 核对（§2.5） |编排器把 `--target-port-file`、`--upstream-port-file` 这类「指向另一个角色」的参数用 `{port_file:<角色>}` 占位符写在 `args` 里，由编排器替换成实际路径。

### 11.1 `capcommon` 速查（各组件必须用这些实现，不另写）

| 模块 | 主要接口 | 用途 |
|---|---|---|
| `clock` | `now_ns`、`wall_iso`（毫秒）、`parse_wall_iso`（返回毫秒）、`busy_wait_ns`、`sleep_until_ns(target, spin_ns=0)`、`RelClock(origin_ns)` | 时间 |
| `hashing` | `sha256_file`、`sha256_bytes`、`canonical_json_sha256`、`verify_sha256_file`、`is_sha256_hex`、`is_git_sha` | §9.1 |
| `identity` | `Identity`（`with_tier`、`with_worker`、`with_pid`、`with_role`、`launch_key`、`from_dict`）、`check_same_launch` | §2.1 |
| `jsonio` | `atomic_write_bytes`、`write_json_stamped`、`read_json_stamped`、`JsonlWriter`、`read_jsonl` | §5.2、§5.6 |
| `cli` | `add_common_args(parser, loop=, tiers=, snapshot=, ready=, port=, cpus=, inject=, stop=)`、`common_from_args`、`parse_cpu_list` | §2 |
| `tiers` | `TierTable.load`（`.tiers`、`.sha256`、`.axis`、`.request_timeout_ns`、`tier_at`、`tier_seq_at`、`label_of`、`window_of`、`gate_of`、`baseline_seq`、`inject_for(role, seq)`、`all_inject_for(role)`、`nominal_load(tier)`、`load_order_key(seq)`）、`Tier`（`end_ns`、`window_start_ns`、`tokens_per_step`、`steps_per_stream`、`stream_duration_ns`）、`TierTableError` | §3 |
| `schedule` | `generate_schedule`、`write_schedule`、`load_schedule`、`count_schedule_rows`、`per_worker_tier_counts`、`prompt_index`、`BeatSchedule`、`STEP_STAMP`、`stamp`、`unstamp` | §3.4、§4、§7.4 |
| `capbin` | `write_capbin`、`read_capbin(path, columns=None, stride=None)`、`RecordArray(fields, capacity)`（`append`、`dump`、`dump_split`、`column`、`overflow`） | §4.1、§5.4.2 |
| `hist` | `bucket_index`、`bucket_bounds`、`quantile`、`diff`、`NBINS` | §6.3 |
| `snapshot` | `SlotRegistry`（`counter`、`gauge`、`hist`、`get`、`register_decls(SLOTS, groups)`）、`validate_slot_decls`、`SnapshotWriter(registry, shm_dir, interval_ns, meta_extra)`（`start`、`write_now`、`stop`）、`decode_snapshot`、`read_snapshot_file`、`list_snapshot_files`、`proc_starttime_ticks` | §6 |
| `inject` | `InjectSpec`、`parse_inject_spec(s)`、`Injector(site, registry)`（`hit`、`count_delta`、`should_drop`、`short_by`、`set_spec`、`prepare_aux`、`counters`、`stats`）、`InjectorSet(sites, registry)`（`set_launch_specs`、`apply_tier_specs`、`prestart_aux(specs)`、`mark`）、`tier_report(mark_a, mark_b)`、`expected_applied`、`expected_applied_delta` | §7 |
| `aio` | `run(main, loop_impl)`、`new_event_loop`、`call_at_ns`、`sleep_until_ns`、`LoopLagProbe(loop, registry, name, period_ns, records=, origin_ns=, tier_seq_at=)` | §2.2、§6.4 |
| `lifecycle` | `write_port_file`、`wait_port_file`、`read_port_file`、`write_ready_file`、`wait_ready_file`、`read_stop_file`、`pin_self`、`StopController` | §2.3 |
| `versions` | `version_record(loop=, extra=, include_grpc=)`、`loop_impl_name`、`required_fields(decl)`、`missing_fields(rec, required)`、`check_versions(records, decls)`、`CONSISTENT_FIELDS` | §2.5 |
| `ledger` | `registry_append`、`registry_lookup`、`registry_kind`、`registered_before`、`submissions_append`、`submissions_current`、`submission_for_job`、`judged_lookup`、`judged_append`、列名常量 | §9.3、§9.5、§10.1 |
| `gate` | `snapshots_by_role(shm_dir, procs, launch_key, max_age_ns, prev_seqs=, now_ns=)`、`snapshot_fields`、`check_snapshot`、`planned_polls`、`evaluate_reading`、`evaluate_gate(gate, snaps, band=None)`、`GateTracker(gate)`（`update(satisfied, t_rel_ns)`） | §8 |
| `runtree` | `RunTree(root)`（`make_skeleton`、`tier_dir`、`tier_file`、`ready_file`、`port_file`、`log_file`…）、`tier_dir_name`、`tree_dir_name`、`job_dir_name`、`launch_shm_subdir`、`append_state`、`read_state` | §5 |
| `expect` | `iterations`、`msg_iterations`、`msgs_per_request(N,k,s,gate)`、`msg_tokens`、`msg_step_offsets`、`sse_frames_per_request(n, include_usage)`、`validate_reconcile`、`expected_hop_count`、`GATES` | §3.4、§4.5、§5.8 |
| `wire` | `GRPC_REQ`、`pack_grpc_request`、`unpack_grpc_request`、`http_headers`、`HDR_*`、`padded_content_frame`、`usage_frame`、`DONE_FRAME`、`SseCounter` | §4.4、§4.5 |
| `hwprofile` | `IDENTITY_FIELDS`、`OBSERVED_FIELDS`、`validate_profile`、`profile_sha256`、`diff_identity`、`unreadable` | §5.5.3 |
| `policy` | `NO_MODEL_EXPERIMENTS`、`FORBIDDEN_MOUNT_PATHS`、`WEIGHT_SUFFIXES`、`ASSERTION_KEYS`、`check_no_model_submission`、`check_mounts`、`tokenizer_dir_sha256`、`check_test_only_argv` | §9.5 |
| `scratchpath` | `assert_job_scratch_path`、`check_verified`；`python3 -m capcommon.scratchpath`（退出码 0 / 3） | §5.9 |
| `exitcodes` | 工具与判定器的退出码常量、`ConfigError`（→ 3）、`CapkitError` | §2.4、§10.3 |

工具的 `main()` 约定：顶层捕获 `capcommon.exitcodes.CapkitError`，打印原因到标准错误，以它的 `exit_code` 退出；其余异常让它冒泡（退出码 1）。

## 12. 已知与设计稿 / 计划的偏差、未决问题

**偏差（每条写明出处与理由；评审时请判「接受 / 改」）**

1. **清空门不提前开下一档**（设计稿 §2.8 写「回到基线才进入下一档；最长等待时间待试点」）：本约定把清空档的长度固定为「最长等待」，下一档的开始时刻写死在档位表里，门早过也不提前。理由：这样各工具换档不需要任何进程间通信，时间表与节拍可以事先算好并登记；代价是每个清空档最多浪费「最长等待 − 实际清空用时」。门不过时的处理与设计稿一致（结束本次启动）。
2. **清空门的默认判据是「积压归零」（`abs_max: 0`）**，不是设计稿的「回到亚饱和档的基线（T3 噪声带内）」。理由：清空档里没有新到达，在途数的基线本来就是 0；E0 第一轮还没有噪声带。`baseline_band` 规则保留，供有了 `pilot.json` 之后用。
3. **`raw/launch_end.json`** 是 DATA-AND-ANALYSIS §3 骨架之外新增的文件：`launch.json` 在启动前写一次、不改，结束信息另放一个文件，防止崩溃时连 `launch.json` 都没有。
4. **逐行记录按档拆分落盘**：DATA-AND-ANALYSIS §3 的 `tiers/` 写的是「负载器输出、桩自报、探针快照、清空门读数」；本约定把跨档的逐行记录在落盘时按档拆开放进各档目录，另加 `tiers/pre/`、`tiers/post/` 两个非档目录，装原点之前与最后一档之后的事件。
5. **`submissions.tsv` 用事件行**（§9.5），不是「一个作业一行、原地补字段」：只追加、不改写，与 `REGISTRY.tsv`、`STATE` 的规则一致；「每个作业一行」的视图由 `submissions_current` 给出。
6. **探针的逐档注入不支持**：设计稿 §2.6 的剂量—反应在同一次启动里换 δ 时，探针一侧只能靠「每个 δ 一次启动」（`CAPPROBE_INJECT` 在进程生命周期内不变）。capkit 工具一侧支持逐档换 δ。探针一侧的登记方式见勘误 FT1-18。
7. **事件循环延迟探针是相对重挂**（§6.4）：长阻塞之后不补发，样本数下降与 `.max` 一起反映阻塞。设计稿只说「实际触发时刻 − 计划时刻」，没有规定重挂方式。
8. **快照目录按启动分子目录**（§2.2）：设计稿 §2.4 写的是 `/dev/shm/capprobe/<pid>.bin`；本约定改为 `/dev/shm/<用户名>-<作业号>/snap/r<重排次数>-L<启动序号>/<pid>.bin`。理由：同一作业的多次启动不能互相读到对方的快照；子目录名带用户名与作业号，满足 CONSTRAINTS §1 的本地盘命名规则。
9. **清空门要求 `poll_ns ≥ 2 × 快照周期`**（§8.1），并且每次轮询要求每个进程的快照 `seq` 前进、不比 `max_snapshot_age_ns` 旧。设计稿没有规定快照新鲜度；代价是写快照的线程被饿住时门会晚过（偏保守，不会把崩溃的进程当作已清空）。
10. **GPU 为 0 的检查号是 `GPU-IDLE`**（§10.2），不是审阅意见建议的 `G1`：设计稿里 `G1` 已是宿主进程的位点编号与 §12.6 的操纵核对，撞名会让判定器输出被误读。
11. **不加载模型的实验编号白名单比 CONSTRAINTS §2 窄**（§9.5）：E10 的替身臂、E13 的主体与要加载模型的部分共用实验编号，按编号无法机械区分，所以暂不收进白名单；要提交它们，先由用户定出独立编号并在本文件追加勘误。
12. **`asap` 模式的档不算 X2**（§3.4）：没有计划步就没有消息层的计划时刻；这类档的 X2 记 `NOT_RUN`，只能用请求层滞后与 X1、X3。

**未决问题（不阻塞第一批编码；各组件按「缺省做法」写，执行时实测）**（R2 新增的已知缺口见勘误 R2-18）

- U-1：`grpc.aio` 通用处理器配恒等序列化能否直接收发原始字节（设计稿 §2.2 末段【未核对】）。缺省做法：`capgrpc` 用 `grpc.aio.server` 的 `method_handlers_generic_handler` 加 `request_deserializer=None, response_serializer=None`，冒烟时打印收到对象的类型并写进就绪文件的 `info`。
- U-2：asyncio transport 上怎样取 `FIONREAD`（设计稿 §2.2 第 9 条【未核对】）。缺省做法：`fcntl.ioctl(transport.get_extra_info("socket").fileno(), termios.FIONREAD, …)`；E1 用「故意慢读」对照确认读数会变。
- U-3：uvloop 的 `loop.time()` 精度（毫秒）对按计划时刻发送的影响。缺省做法：发送调度用 `capcommon.aio.call_at_ns`，滞后照实记录；E1 标定 uvloop 与 asyncio 两种实现。
- U-4：节点上 `CLOCK_MONOTONIC` 跨容器一致（推断），E0 时钟自检（`capenv.clockcheck`，§5.5）实测。
- U-5：P0 档（无探针补丁）的产品进程不 import `capprobe`，没有快照也就没有版本自报；P0 启动的 C10 怎样取产品进程的版本（例如由内层脚本用同一解释器另起一次 `version_record` 并登记为「同镜像同解释器」的旁证）待 B3 / B5 定，在此之前 P0 启动的 C10 会是 `NOT_RUN`。
- U-6：产品前端一侧的块数修正项（`reconcile.corrections`，如 `PromptInfo` 不产生块、到 `max_tokens` 的一步与结束合并）的具体数值，由 E0 冒烟确认后登记；本约定不预设。

## 14. 分析脚本的输入与输出（DATA-AND-ANALYSIS §6 第②～④步，见勘误 B2-2E-A）

跨启动的合并只在分析脚本里做（DATA-AND-ANALYSIS §4 第 1 条）。每个脚本都是 `capjudge` 里的一个模块，`analysis/<名>.py` 是薄封装（与 `calib_build.py` 相同，用 `python3.11 analysis/<名>.py …` 或 `PYTHONPATH=<C>/tools python3 -m capjudge.<名> …`）。共同规则：

- **输入**：列表文件每行一棵已判的启动树（字面路径，不许通配符、制表符，不许重复）；每棵树的 `judge/verdicts.json` 的 sha 必须与 `<W>/JUDGED.tsv` 里该启动的行一致，否则退出码 3。
- **筛选**（DATA-AND-ANALYSIS §4 第 4 条）：判定前的启动级结论不是 VALID 的树整棵剔除，并在输出的 `excluded` 里列出原因。这里的「判定前的结论」取三者中最差的：`verdicts.json` 的 `upstream`、`launch_checks.LAUNCH-STATE`、现在读到的 `STATE` 里不是 `capjudge` 写的全部行（判定之后追加的行也算）。判定器自己写的 NOT_RUN 不剔除整树：档级的 VOID / NOT_RUN 只让区间变宽。
- **产物**：`--out` 指定的新文件，已存在 ⇒ 退出码 3，不覆盖。旁边写同名的 `.cmd.txt`（`x.json` → `x.cmd.txt`，`capjudge.cmdrec`），制表符分隔的键值行：`product`、`product_sha256`、`module`、`code_sha`（取法同判定器，§10.2）、`cwd`、`pythonpath`、`invoked`（解释器与 `sys.argv` 原样）、`command`（等价的 `python -m` 命令）、`exit_code`、`written_at`，以及每个输入文件一行 `input\t<绝对路径>\t<sha256>`。`.cmd.txt` 在产物之后写，有产物没有 `.cmd.txt` ⇒ 那次运行被打断，产物不可用。
- **退出码**：0 成功；2 参数错误；3 输入或配置错误（不写任何文件）。

### 14.1 `capjudge.boundary`（每启动 T4、包络、T5；`schema: capkit.boundary/1`）

```
python3 -m capjudge.boundary --inputs <列表> --workdir <W> --out <新文件> [--compare <臂A>,<臂B> …]
```

- 全部入选的树必须扫同一条轴（`verdicts.json` 的 `axis` 完全相同），否则 3。
- `launches[]`：`tree`、`identity`（实验、臂、作业号、重启次数、启动序号）、`block`（区组 = 作业号：一个作业里每臂一次启动）、`mode`、`launch_verdict`、`roles`（`raw/launch.json` 每个角色的 `kind`、`measure_class`、`workers`）、`grade`（`P+X+I+V` 模式的逐启动等级，否则 null）、`doses` 与 `heartbeat`（`intervention_V` 的同名字段，否则 null），以及两个区间：
  - `t4`：判定器写的 `boundary_T4`。脚本按 `tiers` 重算一遍，`lower`、`upper`、`lower_seq`、`upper_seq`、`kind` 有一个不同 ⇒ 3。
  - `interval`（跨启动用的区间，T5 与 V 用它）：在第一个「工具到顶档」以下的负载档上算 T4。工具到顶档指 RED 检查只有 C1 / C2 的 VOID 档，或 I 定位到 `load_source` / `stub_out_of_domain` 的 UNSTABLE 档（设计稿 §2.6「负载源或域外桩先离开健康带 ⇒ 下界」，与 `capjudge.grade` 同一判据）。截断发生在第一个不稳定档之前 ⇒ 上端为 null，`kind: lower_bound_only`，`censored_by` 写明是哪一档、为什么。`skipped` 列出既不是 STABLE 也不是 UNSTABLE 的档。
- `arms.<臂>`：`envelope`（该臂全部启动的 `interval` 包络，按 `axis.order` 取 [各次下端的最小值, 各次上端的最大值]，任一次缺端 ⇒ 该端 null）、`by_block`（每个区组一个包络）、`n_launches`；另有 `envelope_t4`、`by_block_t4`：同样的包络，但取判定器的 `t4`（不截断），给 14.2 用。
- `t5[]`（每个 `--compare` 一项）：只用两臂都有的区组，`stats.t5_compare` 逐区组比 `by_block`。结果是 `different`（附方向）或 `indistinguishable`（附本次能分辨的最小差）；有开口端的区组不能算作分开；没有共同区组，或没有一个区组两臂的包络都闭合 ⇒ `NOT_RUN`（不是 `indistinguishable`）。`blocks_missing` 列出只有一臂的区组。臂名不在输入里 ⇒ 3。

### 14.2 `capjudge.scaling`（T6 与对侧翻倍归因；`schema: capkit.scaling/1`）

```
python3 -m capjudge.scaling --boundary <boundary.json> --pair <基准臂>,<加倍臂> [--pair …]
    --grows-min <比值> --flat-max <比值> --out <新文件>
```

- 输入是 14.1 的产物（T4 只在那里算一次）。用每臂的 `by_block_t4`（判定器的 T4，不在工具到顶档截断）：对侧翻倍要判的正是哪一侧先到顶，空汇或负载器到顶的档在这里是证据，不是截断点；没有 `by_block_t4` 的文件 ⇒ 3。轴必须是 `asc`。两个界取实验登记表里事先登记的值，显式传入并写进输出；要求 0 < `flat-max` < `grows-min`。
- 每一对：两臂的角色集合、`kind`、`measure_class` 相同；只有**一侧**（`kind` 为 `client`，即负载器；或为 `server`，即空汇 / 桩）的角色改了 `workers`，这一侧的全部角色按同一个倍数 f > 1 改，另一侧不变。不满足 ⇒ 3。一臂的各次启动角色不同 ⇒ 3。
- 逐区组（两臂都有的区组）：基准臂包络 E_b，加倍臂包络 E_s，边界比值落在 [E_s.lower ÷ E_b.upper, E_s.upper ÷ E_b.lower] 内；缺端 ⇒ 对应一端为 null。同时给出效率（比值 ÷ f）。`t6_lower_ends`、`t6_upper_ends` 分别是只用下端、只用上端的 T6 区间（`stats.t6_ratio_range`），作对照。
- 判断用不利端，对全部共同区组（设计稿 §2.6 T6）：最小比值 ≥ `grows-min` ⇒ `grows`；最大比值 ≤ `flat-max` ⇒ `flat`；否则 `unresolved`。某个区组的下端比值算不出 ⇒ 最小比值未知（`ratio.min` 为 null，`ratio.min_unknown_blocks` 列出这些区组），不给 `grows`；上端比值算不出 ⇒ 最大比值未知（`ratio.max`、`ratio.max_unknown_blocks` 同理），不给 `flat`；不用其余区组下结论。没有一个共同区组的下端比值算得出（基准臂包络不闭合，或加倍臂没有下端）⇒ `NOT_RUN`。`grows` ⇒ `limit_side` 为加倍的那一侧；`flat` ⇒ 另一侧。
- `attribution`（E1 卡「隔离」：两种情形各出现一次才知道谁是谁）：同时有 server 侧加倍为 `flat`、client 侧加倍为 `grows` ⇒ `client`（限制在负载器）；镜像情形 ⇒ `server`（限制在空汇）。两侧都有已判的对，但组合不是这两种，或同一侧的几对结论不一致 ⇒ `UNDETERMINED`；缺一侧的已判对 ⇒ `NOT_RUN`。

### 14.3 `capjudge.e1report`（E1 的斜率对账、字节与 accepts 对账、空汇健康；`schema: capkit.e1_report/1`）

```
python3 -m capjudge.e1report --boundary <boundary.json> --workdir <W> --out <新文件> [--sink-bands <标定文件>]
```

封装是 `analysis/e1_report.py`。三部分：

- **`slope`（E1 卡「已知答案」：1/边界对 δ 的斜率为 1）**：
  - 只用 `P+X+I+V` 模式判过的启动。不带剂量的是 δ = 0。带剂量的启动必须只给一个角色、一个位点加剂量，各启动相同，心跳为 PASS；不满足的列进 `excluded`。
  - 只用一种拓扑（角色名、`kind`、`measure_class`、`workers` 都相同，`capjudge.boundary.topology`）：带剂量的启动必须同一拓扑，否则 NOT_RUN；拓扑与它们不同的不带剂量启动（别的臂，如 worker 数不同，边界本来就不同）不当 δ = 0，列进 `excluded`。
  - δ 按每次经过的注入时长算（`delta_ns ÷ every_n`，秒）。每个 δ 一个包络。
  - 单位换算：W = 加剂量角色的 `workers`；p = 每单位 `load_value` 的经过次数，由构造给出：用该启动的对账链（位点的单位，`intervene.SITE_UNIT`）和档位参数，经 `capcommon.expect` 算出。`rate_rps` 轴 p = 每请求的经过次数；`nominal_msgs_per_s` 轴再乘 `rate_rps ÷ load_value`。各负载档的 p 必须相同，档位表的 sha 必须与 `launch.json` 登记的一致，否则 NOT_RUN。
  - 对每一对 δ_a < δ_b，归一斜率落在 [(1/b.upper − 1/a.lower), (1/b.lower − 1/a.upper)] ÷ (δ_b − δ_a) × W ÷ p 内。
  - 判定（逐对）：已知的一端把 1 排除在外 ⇒ FAIL；区间闭合、包含 1、且下端 > 0 ⇒ PASS；其余（有开口端，或区间同时包含 0：剂量没让边界可分辨地移动时也会得到这样的区间）⇒ INCONCLUSIVE。总体：有 FAIL ⇒ FAIL，否则有 INCONCLUSIVE ⇒ INCONCLUSIVE，否则 PASS；少于 3 个 δ，或轴不是 `asc` ⇒ NOT_RUN。
- **`counts`（E1 卡「计数真实性」，逐启动，启动范围求和）**：两个工具按计划档或事件档归档，口径不同，所以按全部档目录求和后再比。
  - 拓扑必须是 E1 形：恰好一个 `kind: client` 角色，恰好一个 `kind: server` 角色，且它是 `stub_out_of_domain`；否则 NOT_RUN。
  - 两个角色在每个档目录（`raw/tiers/NNN_*`）里每个 worker 都要有逐档摘要，缺一份 ⇒ NOT_RUN。缺的摘要会同时从两侧的和里消失，不能让它变成 PASS。
  - 字节：HTTP 时，空汇的 `counts.bytes_sent` 包括每个响应的头，capload 的 `frames.bytes` 不包括。所以按空汇逐流记录（`streams.capbin` 的 `req_id`）用 `capload.sink.response_head` 重建头的字节数，比较空汇字节 − 头字节与 capload 字节。有外来请求、流记录溢出，或流记录条数 ≠ `counts.requests` ⇒ NOT_RUN。gRPC 时两侧都是消息负载字节，直接比。相等 ⇒ PASS；负载器收到的多 ⇒ FAIL；空汇发出的多（空汇计的是交给传输层的字节，被丢下的流里有在途字节）：负载器有失败或未完成的请求（capload 的 `counts.failed`、`counts.unfinished`，capgrpc.load 的 `counts.failed`）⇒ NOT_RUN 并写出差值，没有 ⇒ FAIL。
  - accepts：capload 的 `conn_pool.new_conns` 之和 = 空汇的 `counts.accepts` 之和。nullgrpc 不计 accepts ⇒ NOT_RUN。
  - 用到的计数（`frames.bytes` / `msgs.bytes`、`counts.bytes_sent`、`counts.foreign_requests`、`counts.requests`、`overflow.streams`、`conn_pool.new_conns`、`counts.accepts`、`counts.failed`）在任一份摘要里缺键 ⇒ 该项 NOT_RUN，不按 0 相加。
- **`sink_health`（E1 卡「隔离」：空汇的线程 CPU 与事件循环延迟留在空载带内）**：
  - 只在给了 `--sink-bands` 时算。该文件的 sha 必须在 `<W>/REGISTRY.tsv` 里，否则退出码 3。只用其 `noise_bands` 里 `regime` 为 `subsat` 的有效带。文件的 `environment`（硬件画像、镜像）必须与该启动一致（判定器对每种标定文件的同一判据，`capjudge.inputs._check_env`），否则该启动 NOT_RUN 并写明原因。
  - 每个负载档、每个 `stub_out_of_domain` worker，用判定器 C2 的同一实现（`capjudge.calib.stub_worker_stats`）算 `T3:loop_lag_p99_ns` 与 `T3:thread_cpu_cores`，与 `<统计量>:<空汇角色>` 的带比。高于带 ⇒ OUTSIDE；没有值或没有带 ⇒ NOT_RUN。只作 E1 报告的旁证，不进判定（勘误 B2-2E-A 第 7 条）。
  - 读原始数据前，先用 `capjudge.inputs.verify` 把树重新核对一遍。

## 勘误与补充

- **2026-11-13，r2（原地修订，不是追加勘误）**：按审阅意见（24 条，逐条见下方「审阅记录」）修订 §0、§2.2、§2.5、§3、§4.4、新增 §4.5、§5.1～§5.6、新增 §5.5.2、§5.5.3、§5.8、§5.9、§6.2、§6.5、§6.6、§7.3、§8、§9.3、§9.5、§10、§11、§11.1、§12。原地修订而不是逐条追加，是因为写本约定时还没有任何组件按 r1 写出代码或数据（`_progress/` 下只有本约定与评审两份进度），追加 24 条勘误只会让组件作者读两遍。schema 名的版本号没有升：没有任何文件以 r1 的格式写出过。capcommon 同步新增 `expect`、`wire`、`hwprofile`、`policy`、`scratchpath` 五个模块，并改了 `clock`（毫秒）、`ledger`、`tiers`、`inject`、`gate`、`versions`、`snapshot`、`runtree`。r1 没有提交进 git，原文不另保留；r1 → r2 改了什么以本条与下方「审阅记录」为准。

- **2026-11-13，集成（第一批）**：集成者按本约定对齐各组件，并据本地功能性端到端（`_local/`，不是测量）补下列条目。格式版本号不变（没有已发布的数据按旧口径写出过）。
  1. **包的位置与入口与 §1.1、§11 一致**：第一批各组件因写权限写在别处，集成时搬到 `tools/<包>/`：`capgrpc`（`python -m capgrpc.load` 是入口，实现在 `capgrpc.loader`，使多进程子进程按稳定模块名导入；`capgrpc.nullgrpc` 是子包）、`capselfcheck`、`capenv`（三处命名空间包合并为普通包）、`capjudge`、`caplaunch`、`capstubs`、`capjobs`。仓库内一律 `PYTHONPATH=<C>/tools`。`configs/gen_e0_configs.py` 是 `configs/e0/` 的生成器。
  2. **§5.3 线程自报**：`capcommon.snapshot.SnapshotWriter` 在快照 meta 里写 `threads: {native_id: Python 线程名}`（该进程全部活着的 Python 线程，集合变化时刷新）。编排器的线程清单把它当作「进程自己声明的线程名」（`source: declared`）。理由：多进程工具的子进程没有自己的就绪文件，原先它们的写快照线程一律认不出，C13 逐档红。产品进程经 capprobe 也带上这张表。
  3. **§2.1 / §10.1 身份核对的作业级例外**：`raw/env/` **直属**的文件是作业级文件的原样副本（门禁结果、绑核表等，§5.1），带作业的 `arm` 与 `launch_seq 0`；判定器对它们只核对 `experiment`、`job_id`、`restart_count`、`register_sha` 四项（一个作业只用一份登记表）。`raw/env/` 的子目录（如 `versions/`）与其余文件仍核对六项。**待用户 / spec 负责人确认**：这是放宽，理由是不放宽则每棵真实启动树都判 `INPUT_INVALID`。
  4. **§5.6 sidecar meta 的采样周期键名是 `interval_ns`**（整数 ns），判定器按它读；判定器自测样本同步改。
  5. **caplaunch 线程规则表**新增 grpcio 1.85 的 `event_engine`、`lifeguard` 两类 grpc 核心线程（`rules_sha` 随之变化）。
  6. **E0(b) 场景**：排空端的消息数在 `summary.msgs.msgs`（不在 `summary.counts`）；对账链把各级中继的 `msgs_received`、`msgs_forwarded` 与 `streams_accepted` 都列为跳；节拍源与排空端加 `--in-process`，使 `expected_procs = 1` 与实际进程数相符（多进程模式是监督者 + W 个 worker，`expected_procs = W + 1`）。
  7. **已解决（见下方勘误 R1-5）**：用 `multiprocessing` spawn 起子进程的工具（`capload`、`capload.nullsse`、多进程模式的 `capgrpc.*`）会多出一个 `multiprocessing.resource_tracker` 进程，它不在任何角色的登记里，原先编排器记为 `unattributed`。原文「E0(a) 没有 `under_test` 角色，不受影响」有误：只有 E0(a) 的 nullsse 线（`e0a-nullsse`）没有 `under_test` 角色，`e0a-frontend-sfe` 的前端就是 `under_test`；且按勘误 R1-17 的状态优先级，没有 `under_test` 角色时 `unattributed` 也判红。现由编排器把它记入 `roles.json` 的 `aux_helpers`，不算 `unattributed`。

### R1 自审修复（2026-11-13）

三路自审 R1（簇 K1–K28、P1–P8，见 `code-review/self-review-compare/capkit-J17871/R1/triage.md`）修复后，各组件修复者给出的勘误原文与集成时补的条目合并在此，按章节排序、去重；正文里与之冲突的句子各加了一句指向本节（写作「见勘误 R1-n」，n 为下列条号）。格式版本号不变（没有已发布的数据按旧口径写出过）。

1. **§5.2 / §11 `caplaunch` 新必填参数**：`--overlay-manifest PATH`、`--nvidia-smi PATH`、`--job-env-dir DIR` 三项必填，与身份参数一样只由 `capjobs.launches` 追加（启动计划里出现即拒绝，见第 15 条）。场景参数可用占位 `{nvidia_smi}`（sidecar 的 `--nvidia-smi` 取它）。覆盖清单原样拷进每棵启动树的 `raw/env/overlay_manifest.json`，`launch.json` 记 `overlay_manifest{source,present,sha256,needed_by_roles}`、`nvidia_smi{path,realpath,sha256}`；场景里有带 `probe_tier` 的角色而清单缺失 ⇒ 不起任何进程，`STATE` NOT_RUN（stop reason `overlay_manifest_missing`，退出码 5）；作业 `env/` 里另有一份内容不同的清单 ⇒ 退出码 3。`--job-env-dir` 不是目录 ⇒ NOT_RUN（`job_env_dir_missing`）。`--shm-dir`、本次启动的快照子目录与 `--run-dir` 在建树之前必须是本人私有目录（`capcommon.snapshot.ensure_own_dir`：新建 0700，拒绝软链、非目录、他人目录；另拒绝组或他人可写；只查最后一级，见勘误 R2-4）。
2. **§5.2 场景角色键 `proc_kinds`（必填）**：本角色运行的 capprobe 进程类（词表同 `capprobe.SITES[*].proc_kind`，如单前端 `["frontend"]`；词表与各拓扑的值已改，见勘误 R2-5）；只有产品角色可以非空，其余必须为 `[]`。`launch.json` 的 `roles[]` 原样带出；判定器 C8 据此判位点适用性（第 15 条）。
3. **§5.2 编排器收到信号**：第一个停止信号 ⇒ 本次启动 NOT_RUN（stop reason `orchestrator_signalled`，退出码 5），进入收尾；收尾（停各角色、残留核验、写结束文件）期间再收到信号只记录，并跳过剩余宽限直接 SIGKILL，结束文件照写；`launch_end.json` 新增 `signals`。信号中止不再被改写成 `not_ready_before_origin`。编排器被信号停下时的最坏收尾时长 = 场景 `stop_order` 角色数 × (`stop_grace_s` + `--kill-after-s`) + 2 × `--kill-after-s`（残留核验的 SIGTERM、SIGKILL 各最多等一次）。
4. **§5.3 测量者自己的子进程**：`measure_class` 为 `load_source`、`observer` 的角色（监测类）的未登记子孙（如 sidecar 每次 GPU 采样起的 nvidia-smi）不进 `tier.json` 的 `descendants`、不影响 `pids_stable`，也不写进 `unattributed`；它们逐个记在 `raw/procs/monitor_children.json`（`capkit.monitor_children/1`：`monitor_classes`、`n_proc_scans`、`coverage`、`children[{pid,ppid,comm,starttime_ticks,role,measure_class,exe,is_nvidia_smi,first_seen_mono_ns,last_seen_mono_ns,n_seen,seen_via}]`），并在 `tier.json` 的 `scans.<wstart|wend>.monitor_descendants` 里列出。`tier.json` 的 `scans.<wstart|wend>` = `{t_mono_ns, registered_live, registered_total, descendants, monitor_descendants, aux_helpers}`。非产品角色下未登记、却写快照的进程一律进 `unattributed`（`seen_via: probe.snapshot`，带 `registered_ancestor`）。
5. **§5.3 / §10.2 工具运行时自起的辅助进程**：`multiprocessing.resource_tracker`、`forkserver`（`AUX_HELPER_RULES`，按 `python -c` 的固定代码匹配；`-c` 前的解释器选项见勘误 R5-1）记入 `roles.json` 的 `aux_helpers`（`{pid,ppid,starttime_ticks,cmdline,role,measure_class,category:"aux_helper",rule,first_seen_mono_ns}`）与 `tier.json` 的 `scans.*.aux_helpers`：归属其父角色，但不属于该角色的进程集合；判定器的 C10/C11/C13/快照对账一律只用 `roles.json.procs`，`aux_helpers` 不算 `unattributed`，只在 C13 明细里列出。这解决了上一节第 7 条的未决。
6. **§5.3 sidecar 的线程自报**：`capenv.sidecar` 开 GPU 或 numa_maps 采样时多一条 Python 线程 `capenv-sidecar-slow`；sidecar 在就绪文件 `info.threads` 里以 `{native_id: 线程名}` 声明主线程以外的全部 Python 线程，编排器据此识别（`source: declared`）。
7. **§5.4 `requests` 的 `sent_rel_ns`**：实际写出请求的时刻；keep-alive 重连重发时仍为第一次写出的时刻（发送滞后直方图与 `written` 同口径，每个请求只计一次）。收到过不完整响应头的请求不再重发。**§5.4.1 失败原因表**新增 `client_error`：负载器自身在请求处理中抛出的非协议异常（如 gRPC 库用法错误），`class_hint` 为 `unclassified`。nullgrpc 的 `request_types`、`rejects` 与中继的 `upstream_failures` 改为逐档计数（原为整进程累计、每档重复写）。
8. **§5.5 作业内顺序与环境文件**：内层脚本在门禁之前写 `env/gpu_idle.json`、（`chassis_source = slurm_view` 时先等 `env/slurm_view.json`）`env/hw_profile.json` 与 `env/hw_raw.json`（`capenv.hwprofile`，`--role hwprofile`、`--nvidia-smi` 必填）。`gpu_idle.json`（`capkit.gpu_idle/1`，`capjobs.envfiles gpu-idle` 写）由 `capenv.gpu.check_visible` 产生：可见 GPU 数 == 登记的 `environment.gpus`；≥1 张时先要列出 ≥1 张（阳性对照）；0 张时要求 nvidia-smi 报告没有卡，阳性对照记 NOT_RUN；载荷键为 `status reasons expected_gpus nvidia_smi visible_gpus gpus compute_apps queries count_check positive_control apps_check positive_control_ok gpu_count passed outcome`。`capenv.gpu check` 命令行另写的采样检查改用 schema `capkit.gpu_idle_check/1`（作业流程不调用它，与 `gpu_idle.json` 不同形状）。`image.json` 的 sha256 由提交端计算（`sha256_source`），作业内只核大小与 `mtime_ns`。
9. **§5.5 门禁 `exclusivity.json` 与 `hw_gate.json`**：第 5 项改为「可见 GPU 数 == `--max-alloc-gpus` 且无计算进程」：请求 ≥1 张时先要列出 ≥1 张（否则 NOT_RUN），请求 0 张时要求 nvidia-smi 报告没有卡，阳性对照与计算进程检查记 NOT_RUN 并写明原因。第 1 项的 `scontrol` 输出没有 TRES 时记红（不再按 0 张计；第 1 项改按申请数核对，见勘误 R4-10）。`hw_gate.json` 增加 `status`（PASS / RED / NOT_RUN）。`--registered-hw-sha UNREGISTERED`：门禁照常做第 1～6 项并读画像，`hw_gate.json` 记 `status: NOT_RUN` 与 `node_hw_sha`，判定恒为 `ENV_INVALID`（退出码恒为 6），用于某型号第一个作业只采画像、供登记。画像 sha 含可见卡的 `gpu_count` 等键，申请 0 张与 1 张的作业画像 sha 必然不同。
10. **§5.6 sidecar 的 GPU 读数**：`--nvidia-smi PATH` 可选，不给则不执行任何 nvidia-smi，`gpu.jsonl` 头 `positive_control_ok: false`。`gpu.jsonl` 的 `compute_pids` 可含 `null`（pid 读不到的计算进程，仍算计算进程）；`util_pct` 为 `null` 表示读不到，不是 0，`meta.json` 的 `unreadable.gpu_util` 与 `stats.gpu_util_unreadable_rows` 记录条数。集群作业的 `sidecar_gpu_every_s` 必须 > 0，否则判定器 GPU-IDLE 恒 NOT_RUN。
11. **§5.9 节点本地目录与拷回**：本地盘与 shm 的作业根改为 `<base>/<用户名>-<作业号>-r<重启次数>`（`capcommon.scratchpath` 的 `--job-id` 传 `<作业号>-r<重启次数>`；重排队回到同一节点不会撞上上一轮的目录）；收尾只处理本轮创建的（`run`）或属主为本用户的真实目录（`finalize-only`），mkdir 失败时不清理该目录。拷回核对比较每个文件的 sha256（不只路径、类型、大小）；启动树顶层 `STATE` 只许追加（副本以原件字节开头）。
12. **§6.1 / §6.5 / §6.6 capprobe 位点与快照目录**：`SITES` 每项的 `"line": 行号` 改为 `"lines": [行号, …]`（升序、非空）：同一位点的事件若有多条代码路径，每条路径放同一个守卫调用，共用一个槽与一个探针函数；覆盖清单的 `sites` 原样带 `lines`，`patchcheck` 要求每个位点恰好出现在 `lines` 所列的每一行，摘要另带 `expected_location_count`、`found_location_count`。F4 覆盖 `__anext__`（无池）与 `_feed`（有池）两条出队路径；H1 只计路径以 `/v1/` 开头的 http 请求；三个 lag 位点装在启动完成之后、服务调用之前的 `try` 内（lag 的口径见勘误 R2-6）。`SITE_NOTES` 不再带行号。`capcommon.snapshot` 的快照目录用 `ensure_own_dir` 建（新建 0700，拒绝软链、非目录、他人目录），临时文件以 `O_NOFOLLOW` 打开。
13. **§9.2 `register.json` 新增顶层必填段 `environment`**：恰好五个键——`gpus`（整数，0 或 1，向 Slurm 申请的 GPU 数；不加载模型的作业取 1）、`idle_window_s`（(0, 3600]，门禁第 6 项空转窗口）、`idle_busy_max_frac`（(0, 1]，第 6 项每核忙碌比例上限）、`slurm_view_wait_s`（(0, 3600]，等 `slurm_view.json` 的最长秒数；§5.5.2 的 S 即此值）、`chassis_source`（`slurm_view` 或 `dmi`，传给 `capenv.hwprofile`）。`capjobs.submit submit` 按 `REGISTRY.tsv` 中该 sha 的 `relpath` 读 `register.json`、现算 sha 核对，缺段、多键、取值越界一律拒绝提交；`--gpus` 必须等于 `environment.gpus`；`--slurm-view-wait-s` 命令行参数删除。这些值经 `job.env`（`CAPJOBS_MAX_ALLOC_GPUS`、`CAPJOBS_IDLE_WINDOW_S`、`CAPJOBS_IDLE_BUSY_MAX_FRAC`、`CAPJOBS_SLURM_VIEW_WAIT_S`、`CAPJOBS_CHASSIS_SOURCE`）由内层脚本显式传给 `capenv.hwprofile`、`capenv.gate` 与 `capjobs.envfiles gpu-idle`，没有缺省值。判定器与编排器不读该段。
14. **§9.5 `submissions.tsv`**：`status` 新增正式取值 `UNKNOWN_SUBMIT`，只能出现在带 `submitted_at` 的 `NOTE` 行上：sbatch 结果不明（ssh 超时或失败、退出 0 却没有作业号）时写它，`submitted_at` 为 sbatch 时刻，不重投；`record-end` 拒绝此状态，由子命令 `reconcile-submit`（`--job-id J` 或 `--not-submitted`，凭 scontrol / batch.tsv / squeue 证据）落 `SUBMITTED` 或 `ENDED`/`SUBMIT_FAILED`。`submissions_current` 合并时，`SUBMITTED`、`ENDED` 行的 `status` 即使为空也覆盖之前的值（对账后的 `SUBMITTED` 行就此关闭 `UNKNOWN_SUBMIT`）。每个计划的 `PLANNED` 行 `reason` 是用途标记 `purpose=measurement` 或 `purpose=hw_profile_capture`（`--registered-hw-sha UNREGISTERED` 的只采画像作业），`NOTE`、`ENDED` 行的 `reason` 以同一标记开头。作业结束判据：`squeue` 报 `Invalid job id specified` 视为已不在 slurmctld、只看 sacct。
15. **启动计划 `capkit.launch_plan/1`（`capjobs.launchplan`）**：新增必填 `kill_grace_s`（整数 ≥ 1，启动超预算被 SIGTERM 后到整树 SIGKILL 的宽限），提交端（`capjobs.submit`）与作业内（`capjobs.launches`）都按第 3 条的最坏收尾时长核对：每个启动的 argv 必须带 `--scenario {SNAP}/…`（且该文件在作业快照里；已作废，场景改为 W 里已登记的那份，见勘误 R4-4）与 `--kill-after-s`，`kill_grace_s` 小于任一启动的最坏收尾时长 ⇒ 拒绝计划。`--overlay-manifest`、`--nvidia-smi`、`--job-env-dir` 与身份参数一样由 `capjobs.launches` 追加，计划里出现即拒绝（`--pin-table` 也是，见勘误 R4-6）。每个启动的结论以编排器自己写的最后一行 `STATE` 为准（`capjobs.launches` 再追加一行）；超时被杀、或编排器写 VALID 而退出码不是 0/5 ⇒ VOID（只许变差、作业收 SIGTERM 记 NOT_RUN，见勘误 R2-2）；没有编排器 `STATE` 时，`control/stop.json` 的 stop reason 为 `not_ready_before_origin`、`overlay_manifest_missing`、`job_env_dir_missing`、`orchestrator_signalled` 之一且退出码为 5 ⇒ NOT_RUN，其余 VOID（起进程之前的退出码 3 另记 NOT_RUN，见勘误 R4-9；不限于 3，见勘误 R5b-1）。`capjobs.launches` 是 child subreaper：编排器退出或超时被杀后，对仍存活的后代 SIGTERM → `kill_grace_s` → SIGKILL，记 `leftovers`；作业收到 SIGTERM 时停下当前启动、其余启动记 NOT_RUN，照常写记录。
16. **§10.1 一棵树只判一次**：判定器先查 `<W>/JUDGED.tsv`，同一 `(job_id, restart_count, launch_seq, arm)` 已在表里 → 12；再查 `<tree>/judge`，已存在 → 12（补写例外见勘误 R2-11）；这两步不写任何文件。之后新建本次尝试目录 `<tree>/judge.attempts/<UTC 时刻>-p<pid>/`，输入核对与判定的全部输出（`inputs.json`、`verdicts.json`、`checks.tsv`、`judge.log`，以及新增的 `attempt.json`：`status` ∈ `JUDGED` / `INPUT_INVALID` / `ERROR` / `ALREADY_JUDGED`、`exit_code`、起止时刻、异常时的 traceback）先写在这里。只有产出了判定结果的运行，才在再查一次 `JUDGED.tsv` 后把尝试目录原子改名为 `<tree>/judge`（这一步即「已判」标记），然后追加 `JUDGED.tsv` 一行、`STATE` 一行；改名时 `judge/` 已被别的运行建立，或台账已有该行 → 12，本次尝试记 `ALREADY_JUDGED` 留在 `judge.attempts/`。输入核对不过（13）或异常中断（1）时不建 `judge/`、不写台账、不写 `STATE`，尝试目录留作复查；输入修好后可以再判。原文「`os.mkdir(<tree>/judge)`」与「输入核对在 `judge/` 建好之后做……`judge/` 已存在，所以同一棵树也不能再判」两句作废。
17. **§10.2 检查状态优先级**：每项检查的状态按 RED > NOT_RUN > PASS 取：有任何红证据即 RED（即使应检为 0）；否则 `expected_count == 0` 或 `checked_count == 0`、或有 NOT_RUN 理由 ⇒ NOT_RUN；都没有才 PASS。原文「`expected_count == 0` 或 `checked_count == 0` → 该项 `NOT_RUN`」按此读。C3 已检条数少于应检（部分 worker 缺结果）⇒ NOT_RUN。
18. **§10.2 GPU-IDLE 应检条数**：应检条数 = 各被判档 `window_ns // gpu_every_ns`（`sidecar/meta.json`）之和 × `gpu.jsonl` 头的 `gpu_count`；`gpu_every_ns` 缺失或 ≤ 0、`gpu_idle.json` 或 `gpu.jsonl` 头的 `positive_control_ok` 不为 true ⇒ NOT_RUN。窗口内读不出的行（`query_rc ≠ 0`、`util_pct` 不是数值（`[N/A]`）、`compute_pids` 不是列表）不计入已检并记 NOT_RUN；`util_pct > 0` 或 `compute_pids` 非空的行即使另一读数缺失也判 VOID（`compute_pids` 只算不属于本次启动的进程，见勘误 R2-15）。原文「测量窗口秒数之和 × gpu_count」作废。请求 0 张 GPU 的作业 GPU-IDLE 恒 NOT_RUN（不加载模型的作业一律申请 1 张）。
19. **§10.2 C8 适用位点**：C8 只检查 `proc_kind` 属于 `launch.json` 各角色 `proc_kinds` 并集的位点，应检条数 = 这些位点的个数；其余位点在 `checks.tsv` 明细里列为「本拓扑不适用」，不算 PASS 也不算 RED。覆盖清单缺失或无位点、没有角色声明 `proc_kinds`、或没有一个适用位点 ⇒ C8 NOT_RUN。直方图槽（lag 组）按各桶计数之和算窗口增量。
20. **§10.4 自测**：`run_selftest.py --expect-samples N` 必填（N ≥ 1，由调用方给出，等于样本目录数）；期望 13 的样本另断言没有 `judge/`、`judge.attempts/` 下恰有一条 `INPUT_INVALID` 记录、再判仍为 13；良品另断言 `judge/attempt.json` 为 `JUDGED`、再判 12 且不新增尝试记录。新增样本 `gpu_util_unreadable`。
21. **§11 命令行**：`capenv.gate` 新增必填 `--max-alloc-gpus N`（向 Slurm 申请的 GPU 数）与 `--nvidia-smi PATH`；`--phase pre` 另需 `--idle-window-s S`、`--idle-busy-max-frac F`（取自登记表 `environment` 段，门禁不设缺省，缺了以退出码 3 退出）；`--registered-hw-sha` 可取 `UNREGISTERED`（第 9 条）。`capenv.hwprofile` 新增必填 `--nvidia-smi PATH`。`capenv.sidecar` 新增可选 `--nvidia-smi PATH`（第 10 条）。`capjobs.submit` 新增子命令 `reconcile-submit`（第 14 条），删除 `--slurm-view-wait-s`（第 13 条）。**§11.1 速查**补一行：nvidia-smi 查询只由 `capenv.gpu` 实现（`query_gpus`、`query_apps`、`positive_control`、`sample`、`check_visible`、`idle_check`，关键字参数 `nvsmi` 显式传入可执行文件，生产调用方不靠 PATH）。

### R1b（2026-11-14）

用户 2026-11-13 23:5x 的三项裁决（`00_plan/DECISIONS.md`「勘误与补充」末条第 2～4 项：P3 只在指定进程启用、D14 按 E0 实测定、批准 tokenizer 拷贝作业）落实后的约定。格式版本号不变（没有已发布的数据按旧口径写出过）；新增的 schema 是 `capkit.probe_entry/1`、`capkit.d14_rule/1`。

1. **§6.5 capprobe 只在被测入口进程里启用**（`capprobe.config`、`capcommon.probe_entry`）：新增环境变量 `CAPPROBE_ENTRY`，值为一个 JSON 对象 `{"schema":"capkit.probe_entry/1","program":"<绝对 realpath>","args":[…]}`。导入 `capprobe` 的进程按下面的顺序定状态（`capprobe.ENABLED`、`CONFIG.reason`）：① 没有任何 `CAPPROBE_*` ⇒ 全关（全部开关 False、不起线程、不写快照、不报错；`no_capprobe_env`）；② 有 `CAPPROBE_*` ⇒ 先校验在场的值，不论是哪个进程：未知的 `CAPPROBE_` 变量名、非法 `CAPPROBE_TIER`、`CAPPROBE_GROUPS` 与 TIER 不符（P0、P1 ⇒ 组为空，P2 ⇒ 组非空，没有 TIER 却有组或注入）、未知组名、坏注入规格、坏 `CAPPROBE_ENTRY`、非正整数周期、相对路径 `CAPPROBE_SHM_DIR` ⇒ `RuntimeError`；③ 没有 `CAPPROBE_ENTRY`（`no_entry_rule`）或本进程命令行与它不匹配（`not_the_registered_entry`：spawn 池子进程、MPI worker、其他导入者）⇒ 全关，不读身份文件；④ 匹配的进程 ⇒ 原 §6.5 的全套规则，另加「`CAPPROBE_TIER` 缺失 ⇒ `RuntimeError`」。匹配 = 进程的 `sys.orig_argv`（3.10 之前读 `/proc/self/cmdline`；不用 `sys.argv`，spawn 子进程会把它换成父进程的）以 `args` 原样结尾，且紧挨着的前一个元素解析后等于 `program`（含 `/` 的取 realpath；只有第 0 个元素是裸名时按 PATH 查）。所以 `#!` 脚本（内核在前面加解释器）和多前端 launcher 重新执行的同一命令行都匹配（规则新增 `cwd` 与模块重新执行形式，`python -m` 启动的重新执行原先不匹配，见勘误 R2-7）。fork 出来、没有 exec 的子进程命令行不变，仍启用（与之前一样各自归零、按自己的 pid 写快照）。工具模式（`python -m capprobe.decode|overlay|patchcheck`）不变。原 §6.5「`CAPPROBE_TIER` … 不一致 → `RuntimeError`」与 capprobe README「`CAPPROBE_TIER` 未设置也报错」按本条读：缺 TIER 只在匹配的进程里报错。
2. **§5.2 / §5.3 编排器写入入口规则**：编排器给每个 `launcher: product` 的角色在起进程时写 `CAPPROBE_ENTRY`：`program` = 该角色实际 argv（taskset 之后、`{addr:…}` 已解析）的第 0 个元素按该角色环境的 `PATH`（含 `/` 时按角色工作目录 `control/cwd/<角色>`）解析后的 realpath，`args` = 其余 argv 原样；解析不出 ⇒ 本次启动 NOT_RUN。非产品角色的环境里一律去掉 `CAPPROBE_ENTRY`（外层环境里的值不会传下去）。记录：`launch.json` 新增 `probe_entry`（`env`、`schema`、`match`（匹配规则原文）、`exact_rule_in`、`roles{<产品角色>: {program, args_template（argv 的非运行期展开，`{addr:…}` 仍是占位）, error}}`，已改为角色名列表，见勘误 R2-1）；精确规则在该角色起进程时定，记在 `roles.json` 该进程 `procs[]` 项的 `probe_entry` 与 `orchestrator` 事件日志的 `start` 事件里。`dims.json` 的 `injected_for_product_roles` 加上 `CAPPROBE_ENTRY`。
3. **§5.5 门禁第 6 项**：`exclusivity.json` 第 6 项 `detail.per_core[<核>]` 新增 `irq_ticks`、`softirq_ticks`（空转窗口内的 tick 增量），作为 D14 的空载读数。它经 `--job-env-dir` 拷进每棵启动树的 `raw/env/exclusivity.json`。
4. **§9.3 / §9.4 D14 由登记的规则算**（`capjudge.d14`）：`capjudge.pilot` 新增必填 `--d14-rule <文件>`（`capkit.d14_rule/1`：恰好 `schema`、`idle_softirq_frac_max`、`load_min_excess_frac`、`monotonic_slack_frac`（三者为 [0, 1] 的数）、`min_trees`（≥ 1 的整数）），文件必须已在 `<W>/REGISTRY.tsv` 以新增的 kind `d14_rule` 登记，否则 pilot 退出码 3。读数：每棵输入树、每个 `under_test` 角色、它在 `pin_table.json` 里的核——空载取 `raw/env/exclusivity.json` 第 6 项，负载取本树入选的每个负载档稳态窗口内 `sidecar/cpus.jsonl` 各核首末两行之差；`softirq_frac` = softirq tick ÷ 同区间八列 tick 之和。规则：空载 `softirq_frac ≤ idle_softirq_frac_max`（接近 0），每个负载档 ≥ 空载 + `load_min_excess_frac`，按负载（档位表轴的方向）排序后相邻两档下降不超过 `monotonic_slack_frac`（随负载增长）。完整树（每个 `under_test` 角色都有空载读数和至少一个负载档读数）≥ `min_trees`（另要求来自 ≥ `min_trees` 个作业运行，见勘误 R2-8）时写 `irq_softirq_is_experiment` = 全部完整树的全部角色都满足；完整树不够、或规则的登记时刻不严格早于某棵输入树 `launch.json` 的 `submission.submitted_at`，**不写**这个键（C13 第二半保持 NOT_RUN）。`d14_softirq` 的形状：`{rule, rule_file{relpath,sha256,registered_at}, readings{<树>: {roles{<角色>: {cores, idle, load[], why, idle_close_to_zero, grows_with_load}}, complete, why}}, n_trees_complete, conclusion_reason[, irq_softirq_is_experiment]}`。pilot 不再读 `env/loopback_softirq.json`（`capenv.loopback` 仍是 E0 冒烟的一项读数，不进结论）。判定器只读布尔键，不变。**执行要求**：规则文件要在第一个 E0 (a)(b) 作业提交之前写好并登记，否则 E0 的数据不能用于 D14 结论。
5. **§9.5 / §11 tokenizer 拷贝作业**（`capjobs.submit prep-tokenizer`，`jobs/prep_tokenizer.sbatch`，原 `.draft` 改名）：唯一允许挂模型库的作业。登记值在 `capcommon.policy`：`PREP_TOKENIZER_MODEL_DIRS`（`/home/scratch.trt_llm_data_ci/llm-models/GLM-5.3-Flash`）、目的地 `<代码根>/configs/tokenizers/<名字>`、白名单 `PREP_TOKENIZER_ALLOW_GLOBS`、词表 `PREP_TOKENIZER_VOCAB_GLOBS`、单文件上限 `PREP_TOKENIZER_MAX_FILE_BYTES`（64 MiB）、权重后缀仍是 `WEIGHT_SUFFIXES`。`PLANNED` 行：`reason = purpose=prep_tokenizer`、`experiment = PREP-TOKENIZER`、`gpus = 0`、`exclusive = false`、`tokenizer_dir_sha256 = none`、`assertions_json` 的键为 `PREP_ASSERTION_KEYS`（`purpose_prep_tokenizer gpus_zero model_mount_registered_ro other_mounts_no_model basis_present`），`mounts_json` 过 `policy.check_prep_tokenizer_mounts`：恰好一个挂载碰到模型库，且是登记的目录、同路径、`ro`，其余挂载按 `check_mounts`。其他计划（任何别的 `reason`）挂模型库仍被 `submissions_append` 拒绝。资源：不申请 GPU、不带 `--exclusive`、`--nodes=1 --ntasks=1`，`--partition` 可不给（不给就不加）。作业体只用 bash 与 coreutils；容器内核对模型目录是只读挂载、模型库其余部分不可见，只拷顶层白名单文件、拒绝权重后缀与指向目录外的软链（拒绝规则已扩充，见勘误 R2-10）、白名单文件超限即失败，逐文件核对源与副本 sha256，写 `COPY_MANIFEST.tsv`（文件名、字节数、sha256）与 `PROVENANCE.txt`，结束前再核一遍没有任何权重文件与清单外文件。细节见 `jobs/README.md`「tokenizer 拷贝作业」。测量作业的快照不含这个脚本。
6. **§10.4 自测样本**：样本树新增 `raw/env/exclusivity.json`（只合成第 6 项，D14 的空载读数），合成 `pilot.json` 的 `d14_softirq` 改为第 4 条的键集。

### R2 自审修复（2026-11-14）

三路自审 R2（簇 Q1–Q25，见 `code-review/self-review-compare/capkit-J17871/R2/triage.md`）修复后的约定。四个修复者（FX2-judge、FX2-probe、FX2-jobs、FX2-launch）的进度文件没有单列勘误原文，本节由集成者按各进度文件的簇结论与代码改动整理，并加上集成时补的条目（第 2 条的严重度统一实现、第 17 条的场景校验）；按章节排序、去重。正文里与之冲突的句子各加了一句「见勘误 R2-n」。格式版本号不变（没有已发布的数据按旧口径写出过）；`capkit.probe_entry/1` 多了必填键 `cwd`（第 7 条），schema 名不变。
1. **§5.2 编排器的外层环境核对与 `launch.json` 的 `probe_entry`**（核对的变量已扩大，见勘误 R4-2、R4-3）：场景有产品角色时，外层环境里与场景不符的 `CAPPROBE_TIER` / `CAPPROBE_GROUPS` 在编排器构造时就拒绝（退出码 3，不往启动树写任何文件、不起进程；原先要到起第一个产品角色时才发现）。`launch.json` 的 `probe_entry.roles` 改为产品角色名的列表（不再有 `program`、`args_template`、`error`）：入口规则只在起进程时按运行期 argv 算一次，注入环境的、`roles.json` 该进程 `procs[].probe_entry` 记的、`orchestrator` 事件日志 `start` 事件记的是同一个对象；`roles.json` 登记该进程时一次写全，不再先写一版缺 `probe_entry` 的再补写。勘误 R1b-2 的 `roles{<产品角色>: {program, args_template, error}}` 按此读。
2. **§5.7 `STATE` 的写者顺序与「只许变差」**：一棵启动树的 `STATE` 依次由三方追加：`caplaunch` 在启动结束时写第一行（建树时树里已有 `STATE` 或 `raw/launch.json` ⇒ 退出码 3，不写任何文件）；`capjobs.launches` 拷回核对之后追加一行；`capjudge` 判定后追加一行。原文「作业内层脚本（B6）在拷回并校验之后写第一行」作废。启动级结论的严重度 VALID < NOT_RUN < VOID < ENV_INVALID 只在 `capcommon.runtree` 实现一次（`STATE_SEVERITY`、`state_rank`、`worst_state`），后写方先读已有结论，只能取两者中较差者，不能改好：`capjobs.launches` 的超预算被杀、有进程扛过 SIGKILL、拷回未核对 ⇒ 与编排器自己的结论取较差者再与 VOID 取较差者（编排器已记 ENV_INVALID 就保持）；作业自己收到 SIGTERM（`launches.json` 的 `terminated`）⇒ 与 NOT_RUN 取较差者，原因以 `terminated` 开头，不再记 VOID（不计入「连续两次 VOID」）。判定器的规则见第 12 条。读当前状态仍是最后一行。
3. **§5.9 本地启动树与作业目录的权限**：`capjobs.launches` 用 `private_mkdir` 建 `<L>/runs`、`<L>/logs` 与每棵本地启动树：模式恰为 0700（不受 umask 影响；建后 `chmod` 并读回，已存在的必须是本用户的真实目录）；`jobs/capkit_job.sbatch` 的作业目录与 `env/`、`prep_tokenizer.sbatch` 的作业目录一律 `mkdir -m 700`。理由：`caplaunch` 拒绝组或他人可写的 `--run-dir`（第 4 条），umask 002 的节点上原先会整次启动起不来。
4. **§6.1 快照目录（`capcommon.snapshot.ensure_own_dir`）**：除软链、非目录、他人属主之外，组或他人可写（`mode & 0o022`）的目录也拒绝；只检查最后一级，上级目录由调用方负责（启动里是作业私有的 0700 快照根）。`caplaunch` 的 `--run-dir`、`--shm-dir` 直接用这一实现（原先在编排器里另写了一遍 0o022 检查）。
5. **§6.2 / §6.5 位点的进程类（`capprobe.sites.PROC_KINDS`）**：一个进程类 = 该类进程在服务本研究的负载（`/v1/completions` 流）时都会经过的位点集合，每个位点只标一个类。词表：`frontend`（H1、H3；经 `OpenAIServer` 服务 OpenAI HTTP 接口的前端，含原架构 `trtllm-serve`）、`frontend_openengine`（F2、F4；经 OpenEngine 客户端与 facade 连 engine 的新架构前端）、`frontend_single`（Lfe；新架构单前端 `--num_serve_frontends 1`）、`frontend_launcher`（Lmfl；多前端 launcher）、`frontend_attached`（Lmfa；多前端附属前端）。角色的 `proc_kinds`：新架构单前端 `["frontend", "frontend_openengine", "frontend_single"]`；新架构多前端（launcher 与附属前端同一个角色）`["frontend", "frontend_openengine", "frontend_launcher", "frontend_attached"]`；原架构前端 `["frontend"]`；不跑探针代码的角色 `[]`。场景校验（`caplaunch.scenario`）按路径单独加载 `sites.py` 读 `PROC_KINDS`，未知名字拒绝。§6.5 示例里的 `proc_kind`（`frontend`、`postproc`、`host`、`worker`）与勘误 R1-2「如单前端 `["frontend"]`」按此读；disagg 路由经过 H1、不经过 H3，第一批没有 disagg，它的角色写 `[]`。
6. **§6.5 位点的计数与 lag 口径**（`capprobe.sites.SITE_NOTES`、`LAG_NOTE`，capprobe README）：H1 计路径以 `/v1/` 开头的每个 http 请求，不论路由与状态码（`GET /v1/models`、`/v1/` 下的 404 也计）；请求经过几个打了探针的进程就计几次（多前端附属前端把 `/v1/messages/batches` 系列转发给 launcher；disagg 各跳各计）。F4：无 postproc 池时 `__anext__` 只在响应处理方等下一个输出时取，F2 − F4 是队列积压（只在请求全部正常结束时成立，见勘误 R4-5）；有池时 `_feed` 一到就取，只在后处理会话打开时计（探针树补丁 `facade.py` 该行守卫改为 `_capprobe.ON_F4 and not self._session_closed`，同一行、行号不变），F2 − F4 接近 0，积压在 F4 之后；`_feed` 把终止事件交给池之后、会话关闭之前取到的项会计入，无池路径不计这些。lag：安装点不变（各自 `try` 内、HTTP 服务调用之前），探针从安装起一直跑到进程退出、不会停，HTTP 服务自己的启动（uvicorn 启动、应用 lifespan 的启动部分）与关停都在直方图里，也在其后的每份快照与退出快照里（判定器只用每档窗口首末两份快照之差）；uvloop 的 `call_at` 是毫秒精度，每个样本可能在真实停顿之外再晚最多约 1 ms。勘误 R1-12「三个 lag 位点装在启动完成之后」按此读：HTTP 服务自己的启动不在「启动完成」之内。
7. **§6.5 `CAPPROBE_ENTRY` 规则（`capcommon.probe_entry`）**：规则为恰好四键 `{"schema":"capkit.probe_entry/1","program":…,"args":[…],"cwd":"<角色工作目录的 realpath>"}`。解析一律按规则里的 `cwd`，不按进程当前目录（产品先 `chdir` 再导入 `capprobe` 也照样匹配）：含 `/` 的元素按 `cwd` 取 realpath；第 0 个位置的裸名按角色 PATH 查，PATH 里的相对项也按 `cwd`。两种匹配形式：① 直接：命令行以 `args` 结尾、紧挨着的前一个元素是 `program`（原生程序、`python 脚本`、`python -m 模块`、`#!` 脚本、相对路径启动，以及脚本启动后的重新执行）；② 模块重新执行：只在 `args` 以 `-m <模块>` 开头时适用——`python -m 模块` 启动后多前端 launcher 用 `[sys.executable] + sys.argv` 重新执行，命令行变成 `python <模块文件> <其余参数>`；命令行以其余参数结尾、前一个元素是以 `<模块路径>.py` 或 `<模块路径>/__main__.py` 结尾的路径（只看名字，不 import）、再前一个是 `program` 即匹配。`-m` 之前有解释器选项（`python -u -m 模块`）时，重新执行出来的进程不匹配（探针关，按第 13、17 条在判定层显性暴露）。勘误 R1b-1 的「匹配 = …」与「多前端 launcher 重新执行的同一命令行都匹配」按此读。
8. **§9.4 D14（`capjudge.d14`、`capjudge.pilot`）**：空载读数是作业级的（门禁每次作业运行做一次，拷进该次运行的每棵树），所以 `min_trees` 同时要求完整树数 ≥ `min_trees` 与这些树来自 ≥ `min_trees` 个作业运行 `(job_id, restart_count)`；`d14_softirq` 新增 `n_job_runs_complete` 与逐树的 `job_run`。负载读数与 C13 第二半共用 `treedata.cpu_tick_deltas`：窗口内有整行 null（核下线）或不足两行的核使该档读数不完整，记进 `why`、不计入（原先做减法抛 TypeError，pilot 崩溃）。pilot 只收「判定之前已记下的启动级结论」为 VALID 的树：判定时记下的 `verdicts.json` `launch_checks.LAUNCH-STATE` 不是 PASS，或现在重读的 `STATE`（非 `capjudge` 行）与 `launch_end.json` 不是 VALID ⇒ 整树剔除（`inputs[].tree_excluded`，全部档列进 `tiers_excluded`，也不进 D14 读数）；`verdicts.json` 没有 `LAUNCH-STATE`（旧版判定器判的）⇒ 退出码 3。判定器自己的 `launch_verdict` 不要求 VALID（E0 判定时还没有噪声带，那里是 NOT_RUN）。勘误 R1b-4 的「完整树 ≥ `min_trees` 时写」按此读。
9. **§9.5 / §11 提交挂载（`capjobs.submit` 的 `submit` 与 `prep-tokenizer` 共用 `job_mounts`）**：挂载清单必须有次级盘 `/home/scratch.yihlu_coreai_1` 的「源 = 目的」挂载（`REQUIRED_IDENTITY_MOUNTS`；主盘上的目录可能是指向它的软链）；运行根、本地盘根、shm 根（要 rw）、快照根与 tokenizer 目录（ro 即可）及其 realpath 都要在某个「源 = 目的」挂载之下（`check_coverage`；`<W>` 与其中登记的文件也要，见勘误 R5-2）。`capcommon.policy` 纵深防御：挂载串任意位置含 `,`、路径含 `:` 一律拒绝；`prep-tokenizer` 的挂载模式只认 `ro` / `rw`；`policy.library_roots()` 给出每个模型库的字面路径与 realpath。`tokenizer_dir_sha256`（断言 `tokenizer_no_weights`）除权重后缀外再按文件头拒绝权重内容（`policy.weight_content`，与作业脚本同四条规则，见第 10 条）。
10. **§9.5 / §11 tokenizer 拷贝作业**（`capjobs.submit prep-tokenizer`、`jobs/prep_tokenizer.sbatch`）：新增必填 `--mount SRC:DST[:ro|rw]`（可多次；容器挂载 = 登记的模型目录同路径 `ro` 加在最前 + 各 `--mount`，过第 9 条与 `policy.check_prep_tokenizer_mounts`）。提交前还要求 `jobs/prep_tokenizer.sbatch` 在代码仓 HEAD 已提交且工作区与之逐字节相同（`--dry-run` 只报告 `would_refuse_uncommitted`）；计划目录里放 HEAD 版脚本、`job.env`、`mounts.json` 与 `MANIFEST.json`（代码 HEAD 与前三个文件的 sha256、字节数），`PLANNED` 行的 `snapshot_sha256` = `MANIFEST.json` 的 sha256；`job.env` 新增 `CAPJOBS_CODE_SHA`、`CAPJOBS_PREP_SCRIPT_SHA256`，作业先核计划目录里脚本的 sha256（不符退出 3），`PROVENANCE.txt` 记两者。只看模型目录顶层，白名单名字之外另拒绝并记下：与作业输出同名（`COPY_MANIFEST.tsv`、`PROVENANCE.txt`）、名字带反斜杠、软链的解析目标不是模型目录内的普通文件或目标名带权重后缀、内容开头是权重格式（pickle `\x80`、zip `PK\x03\x04`、GGUF `GGUF`、safetensors：前 8 字节小端长度 N ≤ 文件长 − 8 且后跟 `{"`）。最终核对用 `sha256sum -c --strict`，并要求核过的行数等于清单行数；库根的可见性按字面路径与 realpath 各核一次。勘误 R1b-5 的「拒绝权重后缀与指向目录外的软链」按此读（64 MiB 上限是兜底，不是唯一防线）。
11. **§10.1 判定的补写**：改名为 `judge/` 之后、追加 `JUDGED.tsv` 行或 `STATE` 行之前出错或被杀时，再跑同一命令：`judge/` 存在而台账行（按启动键）或带本 `verdicts_sha` 的 `capjudge` 行缺失 ⇒ 按 `judge/verdicts.json`（`identity`、`state_line`、文件 sha）补上缺的那一项，不重判，`judge/attempt.json` 改回 `JUDGED` 并记 `commit_completed`，退出码取 `verdicts.json` 的 `exit_code`；两项都在 ⇒ 12；台账行的 `verdicts_sha256` 与 `judge/verdicts.json` 不符、`judge/` 读不了、或 `attempt.json` 记的写者在本机仍在运行 ⇒ 12，不补写。勘误 R1-16「`judge/` 已存在 → 12」按此读。
12. **§10.1 / §10.2 判定器读上游结论（LAUNCH-STATE）**：输入核对的最后一步读判定之前已记下的启动级结论（上游）：`STATE` 里全部非 `capjudge` 写的行、`raw/launch_end.json` 的 `state`（缺失或不是四个状态之一按 NOT_RUN）与 `leftovers.after`（非空按 VOID），取最差（第 2 条的 `worst_state`）；`STATE` 读不了、或没有任何非 `capjudge` 的行（作业还没记下结果）⇒ 13。新增启动级检查 LAUNCH-STATE：上游 VALID ⇒ PASS，NOT_RUN ⇒ NOT_RUN，VOID / ENV_INVALID ⇒ RED。判定器写的 `STATE` 行 = 上游与本次 `launch_verdict` 中较差者（原因 `launch_verdict=… upstream=… verdicts_sha=…`；也记在 `verdicts.json` 的 `state_line`，上游明细在 `upstream`），所以判定器不会把 caplaunch / capjobs 记的 VOID、ENV_INVALID、NOT_RUN 改好。原文「以判定结果为原因的改判」按此读。
13. **§10.2 RECORDS 读 `snapshot_copies`**：`tier.json` 的 `snapshot_copies` 必须是列表；其中记 `snapshot_missing` 的项（编排器按角色 `snapshots` 范围应拷而在两个快照周期内没等到的进程，例如探针没在被测进程里启用）或列了却读不到副本的项 ⇒ RED。这是「设了 `CAPPROBE_*` 但命令行不匹配 ⇒ 探针静默关闭」在判定层的显性暴露（配合第 17 条）。
14. **§10.2 C13 第二半**：D14 结论为真时只减 softirq tick；硬中断 irq 照算外来（登记的 D14 规则只裁决 softirq）。被测核窗口内有整行 null（核下线）或不足两行 ⇒ 该角色第二半 NOT_RUN。窗口首末行扫描与绑核表取核（`treedata.cpu_tick_deltas`、`treedata.role_cpus`）与 pilot 的 D14 共用一处实现。正文「减去按 D14 结果认定属于本实验的软中断部分」按此读。
15. **§10.2 GPU-IDLE 的外来进程**：`compute_pids` 里只有**不属于本次启动**的进程判 VOID；本次启动的进程 = `roles.json` 的 `procs` 与 `aux_helpers`，加上各档 `tier.json` 窗口边界扫描记下的 `descendants`、`monitor_descendants`、`aux_helpers`；pid 为 null 的不算本次启动的。本次启动的进程在利用率 0 时持有计算上下文（如空 executor 启动时的 CUDA 调用）不判红，按 pid 在明细里逐条列出（行数、首末时刻）。`util_pct > 0` 仍判 VOID。正文与勘误 R1-18「`compute_pids` 非空 ⇒ VOID」按此读。
16. **§10.4 自测样本**：样本数 25（`run_selftest.py --expect-samples 25`）；每棵样本树有 `STATE`（caplaunch 的 `VALID` 一行，再是 capjobs.launches 的一行）与带 `state: VALID`、空 `leftovers.after` 的 `launch_end.json`；新增 `upstream_void`（capjobs.launches 判定前记了 VOID ⇒ 10，`STATE` 末行仍是 VOID）与 `probe_entry_unmatched`（档 4 nullsse 没有窗口快照、`snapshot_copies` 记 `snapshot_missing` ⇒ 10，RECORDS、C8 红）；期望里写了 `state_last` 的样本另断言判后 `STATE` 末行。样本的 nullsse（被测、P2）按第 5 条声明 `proc_kinds: ["frontend", "frontend_openengine", "frontend_single"]`。
17. **§11 场景：开了探针组的产品角色必须是快照写者**（规则已改，见勘误 R4-1）：`launcher: product` 且 `probe_tier` 为 `P1` / `P2` 的角色，`snapshots` 不得为 `none`（`caplaunch.scenario` 拒绝，退出码 3）。理由：`snapshots: none` 的角色不在编排器的应拷快照名单里，探针被静默关闭（例如命令行与 `CAPPROBE_ENTRY` 不匹配）时 `tier.json` 不会出现 `snapshot_missing`，判定器的 RECORDS（第 13 条）看不到缺失。`configs/e0` 的产品角色都是 `P0`，不受影响。
18. **§12 已知缺口（新增）**：① **原架构前端没有 lag 位点**（R2 簇 Q25）：基线（原架构）无法在事件循环延迟上与新架构对比；第二批、E7 之前在原架构前端补 lag 位点（并按第 5 条给它定进程类）。② **C13 / D14 的 irq / softirq 读数口径未按目标节点内核核对**（R2 簇 Q4 遗留，待用户裁决）：`/proc/stat` 的 irq、softirq 列怎样记账与内核的中断时间记账配置（`CONFIG_IRQ_TIME_ACCOUNTING`）有关，本批没有在目标节点上核对；核对之前，D14 结论与 C13 第二半（第 14 条）只按现口径解读。

### R4 自审修复（2026-11-14）

R3 到上限后用户批准追加的一轮修复（R3 分诊簇 S1、S2，规则 RS19、RS20 的同类位置，文档簇 S12、S13、S20、T1），加上 E0 作业路径核查（`02_e0/SEAMS.md`）主会话裁定在本轮修的接缝 SM-1～SM-5、SM-10、SM-12。按章节排序；正文与旧勘误里被推翻的句子各加了「见勘误 R4-n」。格式版本号不变；硬件画像的字段集变了（第 7 条），所以登记过的画像要重采。
1. **§11 场景：只约束会写快照的产品角色**（推翻勘误 R2-17）：产品角色只有在 `probe_tier` 为 `P2`（打开了探针组）时，进程才经 capprobe 写快照（P0、P1 不开组；产品侧注入被编排器拒绝，见第 2 条（已改为经档位表登记，登记了产品侧注入的 P1 / P2 角色也写快照，见勘误 FT1-18）；角色 `env` 里的 `CAPPROBE_*` 也拒绝，见勘误 R5-4），这样的角色 `snapshots` 必须是 `all`（`none` 或 `workers` 都拒绝，退出码 3：探针在被启动的那个进程里没打开时，名单里没有它，RECORDS 看不到缺失）。P0、P1 产品角色的 `snapshots` 取任何值都接受，编排器不把它们列进应拷快照名单（`caplaunch.scenario.product_writes_snapshots`、`orchestrator._snapshot_procs`），所以 P1 写 `all` 不会每档出现 `snapshot_missing`。
2. **§5.2 / §6.5 编排器给产品角色的 capprobe 开关**：场景打开 `lag` 组且有产品角色时，`params_required` 必须含 `lag_period_us`（场景校验拒绝，退出码 3），编排器核对它是大于 0 的整数，并以它为产品角色设置 `CAPPROBE_LAG_PERIOD_US`（与 capkit 工具的 `--lag-period-us` 取同一个场景参数）。场景有产品角色时，外层环境里非空的 `CAPPROBE_INJECT` 一律拒绝（退出码 3，起进程之前）：场景没有承载产品侧注入的字段，外层值不能未经登记就透传进产品进程。§6.5「探针自检注入由 `CAPPROBE_INJECT` 给出」在编排器下暂时用不了（第 9 条；已由勘误 FT1-18 改为经档位表 `launch_inject` 登记）。外层的 `CAPPROBE_TIER`、`CAPPROBE_GROUPS`、`CAPPROBE_LAG_PERIOD_US` 与编排器要设置的值不同，同样拒绝（勘误 R2-1 的核对扩到这三个）。
3. **§2.5 版本自报变量的写方**：编排器给每个角色设置 `CAPKIT_CODE_SHA` = `--capkit-sha`；启动的 `probe_tier` 为 P1、P2 时再设置 `CAPPROBE_PATCH_SHA` = 覆盖清单的 `target_commit`（清单在而 `target_commit` 不是 40 位 sha ⇒ 退出码 3；清单缺失时照旧以 `overlay_manifest_missing` 停止）；P0 与无档位的启动去掉外层的 `CAPPROBE_PATCH_SHA`。外层已有、但与编排器要设置的值不同 ⇒ 退出码 3。这样直接跑 `caplaunch`（不经 `capjobs.launches`）时 C10 必需的 `capkit_code_sha`、`probe_patch_sha` 也有写方。作业里外层的 `CAPKIT_CODE_SHA` 来自覆盖清单的 `capkit_code_sha`，所以 `capjobs.submit` 另要求计划里每个启动的 `--capkit-sha` 等于代码仓 HEAD。选「由编排器写」而不是「缺失就拒绝」：直接运行的启动不再依赖操作者导出变量，且值与 `launch.json` 的 `code.capkit_sha` 同源；两边不一致时仍然拒绝，不放宽。
4. **§9.5 / §11 启动计划的场景路径**（推翻勘误 R1-15 的「`--scenario {SNAP}/…`（且该文件在作业快照里）」）：计划里每个启动的 argv 必须带 `--workdir`（绝对路径）与 `--scenario`，两者都不许含占位符；`--scenario` 是 `<W>` 里已登记的那份场景，写相对 `--workdir` 的路径或 `<W>` 内的绝对路径（`capjobs.launchplan.scenario_path`，按 realpath 求相对路径，与 caplaunch 的算法相同）。`kill_grace_s` 的核对（提交端与作业内）读这份文件。`capjobs.submit` 另核对：每个启动的 `--workdir` 就是 `--workdir`，场景的相对路径与 sha 等于登记表中该启动所属 arm 的 `files.scenario`。理由：caplaunch 与判定器都按 `<W>` 相对路径与登记表核对场景，登记又必须早于提交（判定器要求 `registered_at` 早于 `submitted_at`），而 `{SNAP}` 含提交时才生成的计划号，所以 `{SNAP}/…` 的场景在作业里必然以退出码 3 失败。没有采用 SEAMS 建议 (a)（登记表改记代码仓相对路径、caplaunch 从 `--snapshot-dir` 取）：(a) 要改 caplaunch、判定器读场景的地方与 REGISTRY 行的路径口径（REGISTRY 的 relpath 一律相对 `<W>`），改动面更大；(b) 只改提交端与计划校验，caplaunch 与判定器不变。场景不在只读快照里，但它的内容由登记表 sha 钉住：提交之后被改，caplaunch 起进程之前就拒绝（退出码 3），不会按改过的内容测量。
5. **§6.5 F4 与 F2 − F4 的口径**（勘误 R2-6 的「F2 − F4 是队列积压」按此读）：F2 − F4 只在无 postproc 池、且请求全部正常结束时才是队列里的积压。F2 还计 error、未知与溢出事件；请求中止、失败、命中停止串之后（有池时：会话先关闭之后）读到的事件 F2 计、F4 不取，留下残留。分析读 F2 − F4 之前先按失败计数筛档。另有一处与中止无关的口径差：有池时 `_feed` 把终止事件交给池之后、会话关闭之前取到的项计入 F4，无池路径的 `__anext__` 不取这些（`capprobe.sites.SITE_NOTES["F4"]`、capprobe README）。
6. **§5.5.1 / §11 绑核表在作业里、按本作业身份生成**：`capjobs.launches` 在每个启动之前（门禁已通过）在本节点上跑 `capenv.pins generate`：需求 = 该启动场景各角色的 `{pin_role, cpus_needed, numa}`，身份 = 该启动（arm、launch_seq 为该启动的，角色 `pins`），`--source-sha` = 本作业 `env/hw_profile.json` 的 `profile_sha256`（`capjobs.launches` 新增必填 `--hw-profile`，内层脚本传 `<作业目录>/env/hw_profile.json`），新开关 `--only-affinity` 把本进程亲和集之外的核（作业步骤的 cpuset）都当作不可用。表写在 `<作业目录>/pins/<启动树名>.pin_table.json`，以 `--pin-table` 追加给 caplaunch；`--pin-table` 加进计划的保留参数（出现即拒绝）。生成失败（画像读不到、需求放不下等）⇒ 该启动记 NOT_RUN（`skipped: pin table: …`），不起编排器（作业退出码见勘误 R5-3）。判定器对 `raw/env/` 直属文件只核作业四项（勘误「集成」第 3 条），这张表带的是本作业的作业号与重启次数，所以不再判 INPUT_INVALID。
7. **§5.5.3 硬件画像**：① `code_baseline` 取覆盖清单的 `base_commit`（TRT-LLM 基线），不再取 capkit 代码 HEAD（`CAPJOBS_CODE_SHA`），画像 sha 不随 capkit 提交变化；内层脚本读不到 `base_commit` ⇒ ENV_INVALID。② identity 键 `nics` 改为 PCI 设备清单：`/sys/bus/pci/devices` 里 class 为 `0x02xxxx` 的设备（SR-IOV 虚拟功能除外）按 `vendor:device` 计数，`[{model, count}]`（`vendor` / `device` 读不到见勘误 R5-8），与驱动建了哪些网络接口、链路状态、速率无关；原先按 `/sys/class/net` 采的 `{model, speed_mbps}` 列表移到新的 observed 键 `netdevs`（不进 sha）。两处都改变画像 sha，已登记的画像要重采。
8. **§6.6 P0 覆盖清单的导出位置**：`capprobe.overlay --tier P0` 的 `--export-dir` 必须在代码根之内（清单的 `src` 是代码根相对路径，提交端按代码根解析），用代码根下被 git 忽略的固定目录（如 `<C>/_local/e0-P0-export/`）；镜像提交到基线之间改了不进 wheel 的 `tensorrt_llm/grpc/openengine/README.md`，P0 清单要加 `--allow-unshipped tensorrt_llm/grpc/openengine/README.md`（jobs/README 同）。
9. **§5.7 / §9.5 `capjobs.launches` 对起进程之前的配置错误**：caplaunch 退出码 3（不限于 3，见勘误 R5b-1）、启动树里没有 `raw/launch.json`（caplaunch 写完它才起角色）且没有编排器自己的 `STATE` 行 ⇒ NOT_RUN，原因取 launcher stderr 的最后一行；这样的树没有文件可拷，拷回结论 `NOT_RUN` 不再另记「copy-back not verified」，也不算 `capjobs.launches` 的拷回失败（但作业退出码与台账要能看见，见勘误 R5-3）。退出码 3 而 `raw/launch.json` 已存在 ⇒ 照旧 VOID。勘误 R1-15「其余 VOID」按此读。
10. **§5.5.2 门禁第 1 项**（主会话默认决策，CONSTRAINTS「分到的 GPU ≤ 1」的措辞待用户确认）：按申请数核对：`scontrol show job` 的 `ReqTRES` 里的 GPU 数 ≤ `--max-alloc-gpus`；输出里没有 `ReqTRES` 时按分配数（`TRES` / `AllocTRES`）核对，不会更松。整机独占时 Slurm 把整台的卡都记进 `AllocTRES`（实测 8 张），这个数照原样记在 `exclusivity.json` 第 1 项的 `alloc_gpus`，申请数记 `req_gpus`（只取自 `ReqTRES`，见勘误 R5-7），所用字段记 `gpus_checked_from`；作业实际能用几张卡由第 5 项（可见卡数等于申请数、无计算进程）与 GPU-IDLE 负责。
11. **§12 已知缺口（新增）**：① 产品侧探针自检注入在编排器下用不了（第 2 条），要用时先给场景加登记字段（已由勘误 FT1-18 补上，登记在档位表）。② SEAMS 里本轮不修、由主会话裁定延后的接缝：SM-6（`fingerprint_json` 参数没有写方，挡 e0a-frontend-sfe；已由勘误 B2B-9 处理，见该条）、SM-7（prompts 池没有生成器）、SM-8（计划里的来源 sha 手抄、未与作业实测值核对）、SM-9（E0(b) 已知答案判读未实现，挑战设计-待用户裁决）、SM-11（本集群 `scontrol show node` 没有 `SystemProductName`）。③ R3 其余 Minor / Nit 簇（S3～S11、S14～S19，capprobe 的 T2～T6）本轮按范围未修，见 `_progress/fix-R4.md`。

### R5 自审修复（2026-11-14）

R4 追加轮分诊中标「R5 修」的簇 U1～U11（`code-review/self-review-compare/capkit-J17871/R4/triage.md`；新规则 RS21 本地验证环境与作业一致、RS22 起进程前的失败也要让上层看得见）。按章节排序；正文与旧勘误里被推翻或补充的句子各加了「见勘误 R5-n」。格式版本号不变；新增的字段：`job.env` 的 `CAPJOBS_WORKDIR`、`launches.json` 的 `config_refused` 与 `pin_table_failed`（顶层计数与逐启动布尔）、`exclusivity.json` 第 1 项的 `gpus_checked`、`capjobs.submit` 时间预算的 `pin_tables_s`。

1. **§5.3 / §10.2 辅助进程白名单的命令行**（细化勘误 R1-5「按 `python -c` 的固定代码匹配」）：命令行是 `python [解释器选项] -c <代码>`，`<代码>` 是最后一个参数，按规则整串匹配。解释器选项只认 CPython 的无参单字母选项 `b B d E I O P q s S u v`（可合写，如 `-BE`、`-OO`）与带参的 `-W`、`-X`（参数在同一个词里或下一个词），也就是 `multiprocessing` 按父进程的标志与环境变量（`subprocess._args_from_interpreter_flags`）加在 `-c` 前的那些，外加 `-u`（`caplaunch.orchestrator.aux_helper_rule`、`AUX_HELPER_NOARG_OPTS`、`AUX_HELPER_ARG_OPTS`）。原因：作业导出 `PYTHONDONTWRITEBYTECODE=1`，工具的 resource tracker 实际以 `python -B -c …` 运行，原先只认三元素命令行，把它记成 `unattributed`，C13 判红、启动作废。`-m`、脚本、`-i`、`-c` 与其他选项合写、代码之后还有参数，仍不认。
2. **§9.5 提交端挂载覆盖与作业内可读检查包含 `<W>`**（补勘误 R2-9）：作业从起第一个启动前就读 `<W>` 里的登记表与场景（勘误 R4-4），所以 `capjobs.submit` 的覆盖检查加上 `--workdir`（只读即可），并对登记表文件和计划里每个启动所属 arm 的每个 `files.*` 逐个按路径与 realpath 核覆盖（`submit.workdir_inputs`；软链指到没挂载的目录同样拒绝，退出码 3）。`job.env` 新增 `CAPJOBS_WORKDIR`（= `--workdir`），内层脚本的 `capjobs.envfiles mounts` 增加 `--must-read "$CAPJOBS_WORKDIR"`，缺这个变量 ⇒ ENV_INVALID。
3. **§5.7 / §9.5 起进程前配置错的可见性**（补勘误 R4-9）：这类启动仍记 NOT_RUN（什么都没测），但在 `launches.json` 里带 `config_refused: true`，顶层另有计数 `config_refused`；绑核表生成失败而跳过的启动（勘误 R4-6）同样记 NOT_RUN 并带 `pin_table_failed: true`（RS22 同一根因）。只要有一个这样的启动，`capjobs.launches` 以 `CONFIG`（3）退出（拷回失败的 `RUNTIME`（4）与 `ENV_INVALID`（6）优先），内层脚本与批处理脚本照传，作业在 Slurm 里是 FAILED 而不是 COMPLETED。CONSTRAINTS「同一个原因连续两次使作业 `ENV_INVALID` 或使运行 VOID，就停下上报」在本工具包里的口径：起进程前的配置错（作业退出 3、`launches.json` 的 `config_refused`、`pin_table_failed` 或 `job_input_failed` > 0，后者见勘误 B2B-9）按同一规则计数，原因取该启动的 `reason`（launcher stderr 的最后一行，另带退出码，见勘误 R5b-1）；计划里每个启动都因同一配置错被拒时，连续两个这样的作业就停下上报。
4. **§11 场景角色的 `env` 不得含 `CAPPROBE_*` 键**（补勘误 R4-1、R4-2）：任何角色的 `env` 里出现以 `CAPPROBE_` 开头的键 ⇒ 场景校验拒绝（退出码 3）。capprobe 开关只由编排器按登记的场景字段（与档位表的 `launch_inject`，勘误 FT1-18）设置；否则 `CAPPROBE_INJECT` 可经角色 `env` 进入 P0 / P1 产品进程，进程写快照却不在应拷名单里，勘误 R4-1 的前提不成立。
5. **§5.5.1 / §9.5 绑核表生成的两处补漏**（补勘误 R4-6）：① 需求文件 `<启动树名>.pin_table.json.demand.json` 写不进 ⇒ 该启动 NOT_RUN（`skipped: pin table: cannot write …`，`pin_table_failed`，见第 3 条），不再让整个 `capjobs.launches` 以 4 退出、不写 `launches.json`。② 生成最长等 `capjobs.launchplan.PINS_TIMEOUT_S`（120 s），它计入每个启动前的剩余时间检查（`budget_s + kill_grace_s + margin_s + PINS_TIMEOUT_S`；作业输入的时间另加，见勘误 B2B-9）与 `capjobs.submit` 的时间预算（新增 `pin_tables_s` = 120 × 启动数）。
6. **§11 caplaunch 不接受缩写的长选项**：`python -m caplaunch` 的解析器 `allow_abbrev=False`。`capjobs` 按完整拼写核对计划 argv（保留参数 `launchplan.RESERVED_FLAGS`，`flag_value` 取 `--workdir`、`--scenario`、`--kill-after-s`、`--capkit-sha`），`--scen x` 这类缩写原先能绕过这些核对（提交端另用这个解析器预检整条 argv，见勘误 R5b-2）。
7. **§5.5.2 门禁第 1 项的字段**（补勘误 R4-10）：`req_gpus` 只取自 `ReqTRES`，输出里没有 `ReqTRES` 时为 null；实际比较的数另记 `gpus_checked`（取自哪个字段仍记 `gpus_checked_from`）。判据不变。
8. **§5.5.3 硬件画像 `nics`**（补勘误 R4-7②）：某个网卡的 PCI `vendor` 或 `device` 读不到或为空 ⇒ `nics` 记 unreadable（进 `unreadable_identity_fields`），不再记成 `?` 型号。
9. **只认 ASCII 数字**：`--param lag_period_us`、`capload` 与 `capgrpc` 的 `--target` 端口、门禁从 TRES 读的 GPU 数、`capjobs.submit` 的作业号（`reconcile-submit --job-id`、`sbatch --parsable` 的输出）。`str.isdigit()` 还接受 `'²'`、`'١'` 等字符：后面接 `int()` 的地方（lag、`capload` 端口、TRES）遇到 `'²'` 会抛出未捕获的 `ValueError`，工具以退出码 1 结束而不是配置错 3；其余地方把它们当成合法值放过。
10. **§12 已知缺口（新增）**：① R4 分诊 U12（门禁第 1 项与 CONSTRAINTS 措辞，待用户确认）、U13～U17 本轮按范围未修。② 与 U9 同类而未改：画像 `local_disk_type` 的 fstype、`rota`、`model` 读不到时仍记 `?`（md / dm 设备本来就没有 `device/model`，改成 unreadable 会让这类节点的画像恒为 unreadable，要先在真实节点核对）。见 `_progress/fix-R5.md`。

### R5 后 Important 修复（2026-11-14）

R5 三路自审里唯一的 Important（路 A R5-R1，`code-review/self-review-compare/capkit-J17871/R5/review-pr.md`；核对规则 RS22 起进程前的失败也要让上层看得见、RS15 同一事实只算一次）。只改这两处，R5 其余 Minor / Nit 按用户决定不在本轮处理。格式版本号与字段不变。

1. **§5.7 / §9.5 起进程前的失败不论退出码**（推翻勘误 R4-9、R5-3 里「退出码 3」这一条件）：caplaunch 写完 `raw/launch.json` 才起角色，所以判据是「launcher 自己以非 0 退出（不是超预算被杀，也不是作业收到 SIGTERM；后者的判法见勘误 B2B-15「停止标志先于退出码」）、启动树里没有 `raw/launch.json`、也没有编排器自己的 `STATE` 行」，不看退出码是几：配置错 3、argparse 用法错 2（缺必填参数、缩写、未知参数）、未捕获的异常 1（如登记表读不到的 `FileNotFoundError`）都记 NOT_RUN 并带 `config_refused`，原因是 `launcher exited before starting any role (rc=<退出码>): <launcher stderr 的最后一行>`，`capjobs.launches` 按勘误 R5-3 的口径以 `CONFIG`（3）退出（`launches.prestart_config_refusal`，`classify` 用同一判据）。原先只认 3：退出 2、1 被记成 VOID，空树的拷回结论 NOT_RUN 又把原因改写成「copy-back not verified」，作业以 4 退出，看起来像拷回失败，`config_refused` 为 0。编排器起了进程之后的失败不会落进这一类：唯一起进程的地方（`_start_role`）在写完 `launch.json` 之后，之后的退出照旧按 `STATE` 或退出码判（例如有 `launch.json` 而退出 3 或 4 ⇒ VOID）。
2. **§9.5 提交端用 caplaunch 的解析器预检计划 argv**（补勘误 R5-6）：`capjobs.submit submit` 对计划里每个启动，按作业里实际运行的样子构造 argv（`capjobs.launches.launch_argv`：占位符代入示例值 `submit.EXAMPLE_PLACEHOLDER_VALUES`，追加身份参数、`--run-dir`、`--pin-table` 等 `capjobs.launches` 会加的参数），取 `-m caplaunch` 之后的部分，用 caplaunch 自己的 `build_parser()` 只解析、不执行（`submit.check_plan_argv`）。缺必填参数、缩写、未知参数、值类型不对、`-h` 都在提交时拒绝（退出码 3，不写台账、不建快照、不调 ssh）；argv 里没有 `-m caplaunch` 同样拒绝。提交端原先只取 `--workdir`、`--scenario`、`--kill-after-s`、`--capkit-sha` 四个值核对，少一个 `--housekeeping-ms` 的计划能提交，到作业里 caplaunch 才以 2 退出。

### SM-14 判定器修复（2026-11-14）

E0 作业 5933747 的 L0（`e0a-nullsse`）启动 VALID、数据完整，判定器却在第 4 档（`up-sub-hi`）判 C7 RED「1 sampled requests with != 17 rows」，档与启动随之 VOID（`02_e0/SEAMS.md` SM-14）。原因：C7 的抽样请求按**计划档**取（时间表 `tier_seq`），块行却只数本档目录里的 `chunks` 文件，而 capload 按**接收时刻所在的档**归档块行（§5.4）；req 5599 计划 141.990 s、完成 142.007 s，17 行里 8 行在档 4、9 行在档 5（`drain-03`，档界 142.000 s）。只改判定器，capload 的写文件规则不变，集群上已采的树用修好的判定器重判即可。格式版本号与字段不变。

1. **§5.4.2 / §10.2 C7 的逐请求块行跨档合计**（补 §5.4 的归档规则在判定器一侧的用法）：对计划在本档、`req_id % chunk_sample_mod == 0`、成功完成的请求，块行数 = 本档的 `chunks` 文件里它的行数 + **紧接其后的清空档**的 `chunks` 文件里它的行数（`checks._chunk_rows` 的 `next_drain`），仍要求恰好等于每请求帧数。按规则 3（§3.2）负载档后面一定是清空档，清空门要求在途数归零才算过，所以合法的块行只可能落在这两档；再往后的档里的行不计入（缺了照旧 RED），清空档文件里不属于本档抽样请求的行不计入、也不算 stray。stray id 的口径不变：本档自己的文件里、不是「计划在本档的抽样请求」的行，一行一个 RED。C7 的 `detail` 另记「N rows of M sampled requests counted from tier <清空档>」。
2. **同一根因的扫描**（「按请求的计划档取请求 × 按档文件数逐事件行」）：C7 是判定器里唯一按请求数逐事件行的检查。C5、C6 的跳值来自逐档自报（capload 的 `frames`、nullsse 的 `requests` / `frames_sent`、节拍源的 `msgs_sent`、中继的 `msgs_received` / `msgs_forwarded`、排空端的 `summary.msgs.msgs` 都按计划档计）或快照槽从本档开始到下一个清空档结束的增量，不受影响。未改、只登记：① `symptoms.msg_lag_from_chunks`（X2 消息层，`pilot._lag_series` 同）按 `req_id` 把本档的块行与本档的 `server.streams` 连接，块行若来自更早计划档的请求会记「chunk rows without a server stream」⇒ X2 该项 NOT_RUN；按规则 3 本档前一档是没有请求的清空档，只有请求跨过整个清空档才会出现，不改。② `msg_lag_from_chunks` 与 `msg_lag_from_msgs` 按计划时刻取窗口内的消息、只读本档文件：计划在窗口内、接收晚于本档结束（滞后 > `tail_ns`）的消息不在本档文件里，不进 X2 的逐秒序列（滞后极大时序列被截尾）。这是 X2 的灵敏度问题，不是 P 的计数口径，另案处理。③ 原有缺口：清空档不判（§10.2），清空档文件里的块行从不做 stray 核对。
3. **验证**：`tests/judge/test_c7_chunk_rows.py`；把拷出的 J5933747 L0 树重判，第 4 档 C7 PASS（见 `_progress/fix-SM14.md`）。

### SM-16 pilot 按臂搜索（2026-11-14）

E0 第一轮的 pilot（`02_e0/E0/analysis/pilot.json`）定不出 `window_ns` / `warmup_ns`（`02_e0/SEAMS.md` SM-16）。原因：稳定窗口的搜索把全部臂的序列截到「全体最短」（e0b 各臂 20 s），e0a 的 30 s 序列在 L = 20 s 时仍是截断窗，有 3 条落在带外，没有任何 L 归零；预热依赖窗口，也随之为 null。按用户 2026-11-14 第三次值守前的答复（`00_plan/DECISIONS.md`「按臂搜索+补长窗口」），pilot 改为按臂分别搜索。噪声带的构成、T1 的算法、「落在带内」的判据都不变；格式版本号不变（`capkit.pilot/1`），字段有增删，见第 1、3 条。

1. **§9.4 按臂搜索与新字段**：「臂」是启动身份里的 `arm`。每个臂只用本臂入选档的序列分别做稳定窗口搜索（`capjudge.pilot.search_by_arm`、`_window_length`）与预热搜索（`_warmup_length`，用本臂自己的窗口长度）。输出四个新字段，每臂一张表，表的格式与原 `window_search` / `warmup_search` 相同（`{"<L 或 w>": {"n_series", "n_outside"}}`）：`window_ns_by_arm`、`warmup_ns_by_arm`（`{arm: ns 或 null}`）、`window_search_by_arm`、`warmup_search_by_arm`（`{arm: 表}`）。原来全体一张的 `window_search` / `warmup_search` 不再输出。
2. **同一臂内只在等长时比较**：同一臂的全部序列长度相同（本臂的全窗长度）才在同一 L 上比较；长度不一 ⇒ 该臂 `window_ns_by_arm[arm] = null`，表为 `{"unequal_series_lengths": [各长度]}`，不再截到臂内最短。不同长度的臂互不截断。
3. **全窗归零不算找到**：某臂只在 L = 该臂全窗长度时才归零 ⇒ `window_ns_by_arm[arm] = null`，表里另加 `"saturated_at_full_window": true`。理由：噪声带就是各档全窗 T1 的 [min, max]，每条序列的全窗 T1 本身是带的一个样本，全窗归零是同义反复，不能当作找到了更短的稳态窗。窗口为 null 的臂预热也为 null（表为 `{}`）。搜索仍只取偶数秒 L；全窗长度为奇数秒时最大只搜到全窗减 1 秒，那一档归零照常算找到。
4. **顶层 `window_ns` / `warmup_ns` 的口径**：所有臂的 `window_ns_by_arm`（或 `warmup_ns_by_arm`）都非 null 时取各臂的最大值，有任一臂为 null（或没有臂）就是 null。取最大值是让同一个窗口 / 预热长度对每个臂都够用；某臂没有结论时不拿其他臂的值代替。
5. **行为变化**：判定器自测样本 `good`（单臂 `synthetic-a`，序列 4 s）原先在 L = 4 s 归零、得出 `window_ns = 4 s`、`warmup_ns = 0`；这正是第 3 条的全窗归零，现在两者都是 null（`saturated_at_full_window`）。单臂、等长、在短于全窗的 L 归零的情形，结果与原先逐字相同（`tests/judge/test_pilot.py::test_one_arm_of_equal_lengths_keeps_the_old_search`）。
6. **验证**：`tests/judge/test_pilot.py` 新增 4 个测试（两臂不同长度：短臂全窗饱和、长臂在中间 L 归零；单臂等长回归；臂内不等长；预热用各臂自己的窗口、顶层取最大），修改 `test_pilot_from_good_sample`；变异与对 E0 正式输入的重跑见 `_progress/fix-SM16.md`。

### Fix SM-17 slurm-view ssh failures and SM-18 gRPC unrequested-call cancellation（2026-11-14）

两个接缝（`02_e0/SEAMS.md` SM-17、SM-18），都只改写方，读方（`capenv.gate`、`capenv.hwprofile`、判定器）不变；判定器对失败原因的分类不变。`capkit.slurm_view/1`、`capkit.tier_summary/1` 的版本号不变，只加键。

1. **§5.5.2 `slurm-view` 有界重试，有查询失败时不写文件**（SM-17：E1-0 作业 5940970 的第 3 项 `squeue` 遇到一次 ssh rc 255「kex_exchange_identification: Connection closed by remote host」，原先照样写文件、退出 0，调用方不再重试，门禁第 3 项读到 rc 255 判 RED、作业 ENV_INVALID）。
   - 每项查询失败（rc ≠ 0，或 rc 为 null：ssh 起不来或超时）就重试，最多 `--attempts` 次（缺省 3，取值 1～10），第 k 次失败后等 `--retry-wait-s` × k 秒（缺省 2，取值 0～60）。条目记最后一次的结果，另加三个键：`attempts`（试了几次）、`retried_rcs` 与 `retried_stderr`（之前各次失败的 rc 与 stderr，等长、同序；超时与 ssh 255 由 stderr 区分，超时的 stderr 以 `TimeoutExpired:` 开头）。
   - 某一项最后仍失败后，后面的项不再查询（文件反正不写）。没查询的项记 rc null、`attempts` 0、`retried_rcs` 与 `retried_stderr` 为空，`stderr` 写原因（`not run: …`）。有两种情形：第 1 项没给出单个节点（仍失败，或退出 0 但没有单个 `NodeList`）时的第 2、3 项；第 2 项仍失败时的第 3 项。
   - 跑过的查询里只要有一项最后仍失败（`slurmview.failed_items`，没查询的项不算），就**不写** `slurm_view.json`、以退出码 4（RUNTIME）退出；标准输出的 JSON 里 `path` 为 null，`failed` 列出这些项的 `id`、`cmd`、`rc`、`stderr`、`retried_rcs`、`retried_stderr`，`rcs`、`attempts` 照常列出全部三项。作业在 `slurm_view_wait_s` 内等这个文件，调用方（观察脚本）下一轮再调即可补写。第 1 项退出 0 但给不出单个节点时，第 2、3 项不算失败的查询，文件照旧写出，门禁照旧判 RED。
   - 一次调用的最坏耗时 = 已成功各项的耗时 + 仍失败那一项的 `attempts × --ssh-timeout-s + --retry-wait-s × attempts(attempts−1)/2`（缺省 3 次、2 s，`--ssh-timeout-s 60` 时约 186 s）。前端反复先挂住、重试后才通的情形，单次调用仍可能超过作业的 `slurm_view_wait_s`：此时作业照旧 ENV_INVALID，文件可能在作业结束后才写进它的 `env/`（门禁已退出，不影响判定）。
   - 持续失败时，作业最多空等 `slurm_view_wait_s` 后 ENV_INVALID（门禁报「slurm_view file did not appear within S s」）；失败项的命令、rc、stderr 在调用方（观察脚本）的日志里（`slurm-view` 标准输出的 `failed`），不在作业目录里。
   - 作业目录不存在时先报错（退出码 3），不发 ssh；这项检查只有一处实现（`slurmview.check_job_dir`），写文件前再查一次。
   - 与数据侧临时办法 `03_calib/E1/plans/slurmview_watch2.sh` 的关系：那个脚本把文件写到暂存目录、只在 `rcs` 全为 0 时改名搬进作业目录，自己 5 s 后重试；它读的 `rcs` 键保留，所以它在新代码下照样能用，只是多了进程内的重试。`02_e0/E0/plans/slurmview_watch.sh` 只看退出码（0 才记完成），在新代码下失败时不再记完成、下一轮重试。
2. **§11 capkit 的每个 gRPC 服务端显式设未领取队列的时限与条数上限**（SM-18：e0w60 的 e0b 过载档里，relay1 的事件循环跟不上，排队超过 30 s 还没被应用领取的流被 gRPC 内核取消，负载端记 `grpc_CANCELLED`，C3/C5/C6 RED）。gRPC 内核会在两种情形下结束一个应用还没领取的调用：等得比时限 `grpc.server_max_unrequested_time_in_server` 久；队列比条数上限 `grpc.server.max_pending_requests`（软）/ `grpc.server.max_pending_requests_hard_limit`（硬）长。两者客户端看到的都是 `CANCELLED`，与 SM-18 同一签名。
   - 核实时限（本地 venv grpcio 1.85.1，`_local/fix-sm1718/repro/unreq.py`：服务端第一个调用用 `time.sleep` 阻塞事件循环，其后的调用在未领取队列里等）：选项名就是 `grpc.server_max_unrequested_time_in_server`，单位整秒；不设时等 30.2 s 的调用被取消、设 31 或 60 不取消、设 29 取消，即缺省 30 s；设 0 或负数时排队的调用全部被取消（不是「关闭」）。取消要等事件循环恢复、应用领取下一个调用时才送达客户端。
   - 核实条数上限（同一环境，`_local/fix-sm1718/repro/r1-pending/pending.py`，服务端与客户端分两个进程，服务端事件循环阻塞 8 s，期间客户端在一条连接上发 N 个调用；`pending.triage.out` 是 R1 分诊的运行、`pending.out` 是重跑）：不设两项时 N = 1500 有 61～69 个、N = 4000 有约 1450 个被丢弃；显式设 1000 / 3000 时 N = 1500 丢 47～59 个，与不设同一量级；两项都设 2³¹−1 时 N = 4000 全部完成。
   - `capgrpc.common.make_server(reuseport, max_unrequested_s)` 的第二个参数必填、没有缺省；它设的未领取队列选项由 `capgrpc.common.server_queue_options` 给出：时限取 `max_unrequested_s`，两项条数上限都是 2³¹−1（不接参数）。`capgrpc.nullgrpc`（每个 worker）与 `capselfcheck.relay` 新增可选参数 `--max-unrequested-s`（整秒），由 `capgrpc.common.resolve_max_unrequested` 定值：不给时取「档位表 `request_timeout_ns` 与档位表长度 `end_ns` 中较长者，按整秒向上取整」，所以不会短于登记的超时；`request_timeout_ns` 为 null 时就是档位表长度。显式给的值要在 [1, 2³¹−1] 内且不短于 `request_timeout_ns`，否则启动时退出码 3。
   - 这样两条内核路径都不会早于登记的超时结束排队的调用：时限不短于 `request_timeout_ns`（更短的显式值被拒），条数上限是 2³¹−1；排队的调用由负载器按登记的超时结束（`timeout_registered`，过载型）。依据是上面两项核实与第 3 条的测试。没有登记超时（`request_timeout_ns` 为 null）时，时限取档位表长度，排队积压的条数只受时间表的请求总数限制。
   - 实际设的值写进自报：就绪文件的 `info`（nullgrpc 在每个 worker 的 `info`）与快照 meta 的 `info`、每档 `summary.json`，键都是 `grpc_server_options`：`{"grpc.server_max_unrequested_time_in_server": <秒>, "grpc.server.max_pending_requests": 2147483647, "grpc.server.max_pending_requests_hard_limit": 2147483647, "source": "<时限的来源：default: … | --max-unrequested-s>"}`，三个选项键与 `make_server` 设的取自同一个 `server_queue_options`。判定器不读这个键。
   - 版本：以上选项语义只在本地 venv grpcio 1.85.1 上核实；集群镜像自报 grpcio 1.84.0（E0 运行树 `raw/env/versions/*.json` 的 `grpc`），没有在 1.84.0 上核实。首个用新代码的集群作业复核：就绪文件与每档 summary 的 `grpc_server_options` 三个选项键都在，过载档负载端 summary 没有 `grpc_CANCELLED`。
   - 已登记的场景配置（`configs/e0/`）不用改：不给参数即取缺省。
   - **S-PB（`capstubs.prebaked_server`，子批次 2B；合入时已照此改，见勘误 M-SPB-1）必须照此**：它在 2B 分支上调用的是旧的 `make_server(reuseport=False)`，合入后 S-PB 起进程时 `make_server` 因缺少必填参数抛 TypeError，角色起不来；要改成接 `--max-unrequested-s` 与 `resolve_max_unrequested`，并在自报里写 `grpc_server_options`。两项条数上限在 `make_server` 内设，S-PB 走 `make_server` 即自动获得。以后任何 capkit gRPC 服务端同理。
   - 没改：负载端把 `CANCELLED` 记 `grpc_CANCELLED`（`unclassified`），判定器 batch 1 把它算非过载型，两处分类都不变。取消发生在采数时，已采的 e0w60 树用新代码重判结果不变。被测系统的 gRPC 服务端（探针树 `tensorrt_llm/grpc/openengine/server.py`）不设这些选项（测量工具不改被测系统），保持内核缺省；真实链路上排队时限或条数上限造成的提前取消仍记 `grpc_CANCELLED`、判非过载，怎么归类未定（follow-up）。
3. **验证**：`tests/jobs/test_slurmview_retry.py`（暂态失败重试后写出、持续失败有界并点名、之前各次的 stderr 分得开超时与 rc 255、`--attempts 1` 不重试、第 1 项失败时只算第 1 项、第 2 项仍失败时不查第 3 项、第 1 项没给出单个节点不算失败、重试参数越界拒绝；命令行：仍失败 ⇒ 退出 4 且作业目录里什么都不写、`failed` 只列跑过且失败的项，暂态失败 ⇒ 写出并记 `attempts`/`retried_rcs`，作业目录不存在 ⇒ 退出 3 且不发 ssh；门禁照旧读重试后写出的文件）；`tests/grpc/test_unrequested.py`（缺省与显式值的取法和拒绝；自报记三个选项键；两个工具对低于超时的值退出 3；nullgrpc 事件循环阻塞 3 s 时设 1 s ⇒ 排队的流被取消、设 10 s ⇒ 全部完成；SM-18 复现：relay 阻塞 32 s、登记超时 45 s，不设参数 ⇒ 4 个流全部完成，把时限选项从 `make_server` 去掉（即修前）⇒ 负载端 `grpc_CANCELLED` 2 个；条数上限：经 `make_server` 起的服务端事件循环阻塞期间，另一进程发 2500 个调用 ⇒ 全部完成，`make_server` 不设两项条数上限（变异）⇒ 一次运行有 458 个 `CANCELLED`）。变异与运行记录见 `_progress/fix-sm1718.md`。

### Batch 2A+2E: calibration files, stage-2 judge checks, localization, intervention, grading and E1 analysis

第二批子批次 2A 与 2E（RUNBOOK-stage2-draft.md §2.1 第 1、2 行），分支 capkit-b2-judge。按子批次分节；正文被推翻或补充的句子各加了「见勘误 B2-2A-n」「见勘误 B2-2E-n」。

#### B2-2A：标定文件、按 regime 选带、C1 改读 loadgen_band、C2 / C4 / C12（2026-11-14）

新增三种标定文件格式（正文 §9.6～§9.8）与构建器；判定器的 C2、C4、C12 从「`NOT_IMPLEMENTED_BATCH2` ⇒ 该档 NOT_RUN」变为真正的检查。已有格式（档位表、时间表、登记表的其余段、`pilot.json`、判定器输出文件）版本号不变；登记表新增可选键，判定器输出新增字段，见各条。

1. **§9.2 标定登记项与环境**：`calibration[]` 每项只许 `kind`、`relpath`、`sha256` 三个键，`loadgen_band`、`stub_calib` 另有**必填**的 `bind: {"<本实验的角色>": "<文件里的形态名>"}`（非空；形态名必须在文件里，否则 13；可以列别的臂的角色；本次启动没有被绑定的 `load_source` / `stub_out_of_domain` 角色，其 C1 / C2 项 NOT_RUN）。多余的键 ⇒ 13。每个标定文件按种类做格式校验（`capjudge.calib.validate`，§9.4、§9.6～§9.8）⇒ 不过即 13；**每一种**标定文件都核 `environment` 的 `hw_profile_sha`、`image_sha` 与本树一致（原先只核 `pilot`），不一致 ⇒ 13（CONSTRAINTS §4 第 2 条「只用于同一环境」）。`near_boundary_noise` 可以登记多份（同一 relpath 两次 ⇒ 13），其余种类仍只许一个文件。（`near_boundary_noise` 已不接受，见勘误 calib-v2-A2 第 1 条。）
2. **§9.2 / §9.4 噪声带按档的 regime 选**（RUNBOOK §2.2 第 2 条；解决「同一个键不能有两份来源」）：每个噪声带属于一个 regime：`subsat`（亚饱和档上量的）或 `near_boundary`（近边界档上量的）；带上没有 `regime` 字段按 `subsat`。登记表的 `judge` 段新增可选的 `near_boundary_tiers: {"<臂>": ["<档标签>", …]}`（臂必须在 `arms` 里；标签必须是该臂档位表里的 `up` / `down` 档，否则 13）；列出的档是近边界档，其余负载档是亚饱和档（带 `arm_config` 的登记表不许写本键，regime 改由该臂 `ladder.json` 的角色定，见勘误 calib-v2-A2 第 4 条）。被判档只读**本 regime** 的带（X 的全部量、C13 第二半）：同一 (键, regime) 出现在两个文件里仍是 13；同一个键在两个文件里各属一个 regime 是允许的，所以 pilot、一份 `subsat` 的 `near_boundary_noise`（如 pilot 格式的 E1-0 `e1_noise`）与一份 `near_boundary` 的 `near_boundary_noise` 可以同时登记（RUNBOOK E1 正式臂、E2a）。**近边界档查不到近边界带时不退回亚饱和带，该项 NOT_RUN**：RUNBOOK 写的是「近边界档优先取 near_boundary_noise」，这里按 DATA-AND-ANALYSIS §4 第 6 条「近边界的档不沿用亚饱和档的噪声带」与设计稿 §2.6 T3 取严格读法（代价：近边界档要有覆盖其全部统计量的近边界带，否则判不了；样本 `regime_nb_missing`）。输出：`verdicts.json` 每个被判档加 `regime`；`inputs.json` 的 `band_lookups` 的键改为 `<regime>|<统计量键>`。（X 已不读噪声带，见勘误 calib-v2-A3 第 1 条。）
3. **§10.2 C1 改读 loadgen_band**：对每个 `load_source` 角色的每个 worker，按 `bind` 找到它的形态：① 被判档在形态的 `match` 里时，该 worker 的计划负载（档的总负载 × 时间表份额，单位取形态的 `load_key`）> `per_worker_max.value` ⇒ RED，但上界 `censored`（§9.6：只知道「不低于」）时超出它 ⇒ NOT_RUN（超出标定范围），不判 RED；不在 `match` 里、`value` 为 null、负载按该键算不出 ⇒ NOT_RUN；② 四个健康量（`T1:send_lag_p50_ns`、`T3:send_lag_p99_ns`、`T3:fionread_nonzero_frac`、`T3:loop_lag_p99_ns`，与原 C1 同一算法，`capjudge.calib.loadgen_worker_stats`）超出形态带的上界 ⇒ RED，值或带缺 ⇒ NOT_RUN。C1 **不再**读 `pilot.json` / `inline_thresholds` 里以 `T*:…:<负载器角色>` 为键的带（pilot 里这几个键仍会被写出，判定器不用）；没有登记 `loadgen_band` ⇒ C1 NOT_RUN（E0 第一轮重判时状态不变，只是原因措辞变了）。各档 `stats` 里记录的键不变。
4. **§10.2 C2（域外桩部分）**：对每个 `stub_out_of_domain` 角色的每个 worker，按 `stub_calib` 的 `bind` 找形态：计划负载（档的总负载 ÷ worker 数，按 worker 均分；同一监听端口的实际分配不受计划控制；档的总负载当作桩的负载只在启动恰有一个 `kind: client` 角色时成立，否则或有角色缺 `kind` ⇒ 负载项 NOT_RUN，§9.7）对 `per_process_max`，`T3:loop_lag_p99_ns`（该 worker 摘要的 `loop_lag_ns` 直方图）与 `T3:thread_cpu_cores`（摘要 `identity.pid` 的全部线程，`capjudge.calib.stub_worker_stats`；有线程在窗口内只有一行 ⇒ 没有值）对形态的带，规则同第 3 条；worker 没有摘要 ⇒ RED。**C1、C2 的应检数按 `workers` 计**（每个 worker 一份摘要），不按 `expected_procs`（推翻 §10.2「应检进程数 = `expected_procs` 之和」：`expected_procs` 含不服务请求的父进程，如 e0a 的 nullsse 是 2 worker + 1 父进程，父进程没有摘要，按它计会让 C2 永远 NOT_RUN）。`stub_in_domain` 角色不进 P，只在明细里列出（自身需求表见 §9.7 `in_domain`，后续子批次）。设计稿 §2.6 / §2.7 C2 里的「积压所在的一侧」与「弱读者检验」**本条没有实现**：前者要读者侧在途数，归 2E 的 I3（`capjudge/localize.py`）；后者是 E2a 的实验步骤（RUNBOOK §1.5），见第 10 条。
5. **§10.2 C4（每请求输出 token 数 = `max_tokens`）**：应检 = 时间表 `meta.per_tier` × `kind=client` 角色数（每个客户端角色跑整张时间表）。每个计划在本档的请求（`loadgen.requests` 的 `tier_seq` = 本档）的计数取负载器的 `tokens` 列（≥ 0 时不论请求状态都比：capload 把少出的请求记为 `short_output` 失败，C3 同时红）；负载器不计数（`-1`：`capgrpc.load` 一律写 -1，capload 没收到 usage 帧的失败请求写 -1）的**成功**请求（`status = 0`）取本档 `server.streams` 里同一 `req_id` 的 `tokens`。服务端记录只在 `req_id` 能唯一对应一条流时读：有请求需要回退，而启动的客户端角色多于一个（每个客户端角色跑整张时间表，`req_id` 会重复，别的客户端角色的同号流可能落在同一个服务端上，只有一个角色需要回退时也一样），或写流记录的服务端角色多于一个 ⇒ 这些请求不比、计 NOT_RUN。有计数且 ≠ `max_tokens` ⇒ RED；负载器不计数的失败请求只计为已检（由 C3 / X 处理），不读服务端记录：被取消或超时的流在服务端记的是中止前已发的 token 数，读它会把每个登记超时判成 RED；没有计数的成功请求、或有计划而没有记录的请求 ⇒ NOT_RUN。
6. **§10.2 C12（加倍的进程确实承担了流量）**：作用对象 = `reconcile` 里 `kind: requests` 链上 `scope: tier` 的跳所在的角色中，摘要来源（`summary.*`）且 `workers ≥ 2` 的角色（每个 worker 一个单位），或快照槽来源（`slot:*`）且 `expected_procs ≥ 2` 的角色（每个登记进程一个单位，槽增量取本档开始到其后清空档结束、清空门已过，与 C5 同一实现 `checks._hop_parts`）；同一角色只查一次。应检 = 单位数之和。单位的计数：负载器（`kind: client`、摘要来源）取每个 worker 摘要的 `counts.sent`（实发数，设计稿 §2.6 的前提是「实发数、收到数增量 > 0」：请求全部超时的 worker 也承担了流量），不取对账跳的 `counts.ok`；其余单位取该跳自己的计数。任一单位本档计数 ≤ 0 ⇒ NOT_RUN（设计稿 §2.7 C12 的阳性对照：没有流量必须记 NOT RUN 而不是通过），不判 RED；取不到计数 ⇒ NOT_RUN。没有这样的角色 ⇒ 应检 0 ⇒ NOT_RUN（所以只在确有加倍的场景登记 C12）。
7. **§9.2 / §10.2 检查清单**：`judge.checks` 可以列 C2、C4、C12，它们按第 4～6 条判，不再出 `NOT_IMPLEMENTED_BATCH2`。强制集不变。
8. **构建器**（§9.6～§9.8）：`python3 -m capjudge.calib {near-boundary,loadgen-band,stub-calib}` 或 `python3.11 analysis/calib_build.py …`（同一实现，与 `pilot_build.py` 一样的薄 CLI）。列表文件：`near-boundary` 与 pilot 相同（`<树>\t<档序号,…>`）；另两种每行 `<树>\t<档序号,…>\t<角色>\t<形态名>`，同一棵树每个角色可以出现一次（一次启动的两个角色可以各自贡献），同一 (树, 角色) 出现两次 ⇒ 3；参与构建的启动 T4 没有下端（已改，见勘误 calib-v2-A1 第 1 条）或上限负载算不出 ⇒ 3，写明是哪棵树（§9.6）。共同规则（与 pilot 共用 `capjudge.pilot.select_tiers`）：每棵树必须已判过、输入核对通过、判定前的启动级结论为 VALID（否则整树剔除并写原因）；列出的档有 RED 检查或判定为 VOID ⇒ 剔除并列出；`near-boundary` 另剔除 UNSTABLE 与 UNDETERMINED，`loadgen-band` / `stub-calib` 只用 STABLE 档；全部树环境相同；不覆盖已有输出（输出或其 `.cmd.txt` 已存在 ⇒ 退出码 3），产物后写 `.cmd.txt`（`capjudge.cmdrec`，§14）。
9. **§10.4 样本**：新增 8 个（全表见 `samples/README.md`）：`c1_loadgen_overdriven`、`c2_stub_good`、`c2_stub_loop_lag`、`c4_short_output`、`c12_idle_worker`、`regime_nb_missing`、`regime_nb_good`、`input_calib_invalid`。（`regime_nb_missing`、`regime_nb_good` 的期望已改为 13，见勘误 calib-v2-A5 第 6 条。）全部样本都登记一份合成的 `loadgen_band`（形态 `capload-r` 绑给 capload）；`good` 等样本的检查清单加了 C4、C12；合成的 capload 记录对没有 usage 帧的失败请求写 `tokens = -1`（与 capload 一致）；合成的 nullsse 摘要加了 `loop_lag_ns` 直方图（真实 nullsse 有）。
10. **已知缺口（§12 追加）**：① C2 没有「积压所在的一侧」与弱读者检验（第 4 条）。② 桩的计划负载按 worker 均分；E0 第一轮重判（本地拷贝，C12 加入后在进程内判）实测 e0a（J5935100 L1）档 4 的 nullsse worker 1 本档请求数为 0（2 个长连接都落在 worker 0），所以「均分」在 `SO_REUSEPORT` 下不一定成立；C12 会把这样的档记 NOT_RUN，C2 的负载项在这种档上偏松。C12 是否检查由启动所引登记表的 `judge.checks` 决定，只在加倍本身是处理时（scaling 的加倍臂）列入；登记表不列它的启动不会因此 NOT_RUN。轻载档（本档计划请求少于单元数，或长连接都落在一个 worker）单元计数为 0 时同样记 NOT_RUN（按推荐项维持，R2 自审 C9：NOT_RUN 不会给出错的结论，只是少一个档）。③ `match` 只做精确匹配，不外推。④ 两侧都不计 token 的拓扑（读者与桩都不写 `tokens`）C4 恒为 NOT_RUN。⑤ `stub_calib.in_domain`（域内桩自身需求）未定义。⑥ E1-0 的 `e1_noise` 若用 `capjudge.pilot` 构建，其带是 `subsat`，只用于亚饱和档（第 2 条）；要给近边界档用，须另用 `calib_build.py near-boundary --regime near_boundary` 从近边界档构建，作为第二份 `near_boundary_noise` 登记（第 1 条）。⑦ C12 的槽来源单位按 `expected_procs` 计，包含父进程（父进程不处理请求，其槽增量为 0 ⇒ 该档 NOT_RUN）；现有配置里没有槽来源的 `requests` 跳，未触发；出现时先按真实写方（`roles.json`、sidecar 快照）确认单位该是哪些进程，再定口径。
11. **验证**：`tests/judge/test_calib.py`（格式校验正反例、regime 选带、`bind` 与 regime 的输入核对、C1 / C2 形态项、C4、C12、三个构建器）；样本自测全过；E0 第一轮两棵树（e0a J5935100 L1、e0b J5935100 L0）拷到 `_local/b2-judge/e2e` 重判：CLI 重判 `checks.tsv` 前六列除 e0a 档 4 C7（SM-14 修复，原判早于它）外逐行相同，C1 只是 NOT_RUN 原因的措辞变了；在进程内加上 C2、C4、C12 再判：C2 NOT_RUN（没有 stub_calib）、C4 PASS、C12 为 PASS 或 NOT_RUN（见第 10 条②），没有报错。记录在 `_progress/b2-judge.md`。
12. **NOT_RUN 的原因**：C1、C2、C4、C12 应检为 0 时（本次启动没有 `load_source` 角色、没有 `stub_out_of_domain` 角色、本档没有计划请求、没有加倍的角色），`checks.tsv` 的明细写明是哪一种，不留空。

#### B2-2E：定位 I、干预 V、结论等级（2026-11-14）

判定器新增判定模式 `P+X+I+V`：逐档算内部定位 I 与逐档的干预心跳，逐启动给结论等级；跨启动的剂量–反应与最终等级由新的 `python3 -m capjudge.intervene` 给出。`P+X` 模式的判定结果不变。已有格式的版本号不变：登记表、`pilot.json` 只加可选键，判定器输出只加字段。

1. **§9.2 `judge.mode`**：取值 `P+X` 或 `P+X+I+V`，其他值 ⇒ 13。`judge.localize`（第 3 条）只许出现在 `P+X+I+V` 模式下，否则 ⇒ 13。`P+X` 模式下，`verdicts.json` 的 `tiers[].X.I`、`tiers[].X.V` 与顶层 `localisation_I`、`intervention_V`、`grade` 都写 `NOT_REQUESTED`（原来写 `NOT_IMPLEMENTED_BATCH2`）。两种模式的 `verdicts.json` 都新增 `mode`，`batch` 改为 2。（现为 `"calib-v2"`，见勘误 calib-v2-A3 第 7 条。）
2. **I1（CPU）**，按核组算。作用对象：`measure_class` 不是 `observer` 的全部角色的登记进程；只用本档窗口内的数据。
   - **核组**：同一角色里 `roles.json` 的 `procs[].cpus` 相交（直接或经同角色的另一进程）的登记进程合成一组。caplaunch 把一个角色的父进程与全部 worker 都绑在该角色的整组核上（E1-0：空汇父进程与 4 个 worker 同在核 0–3，负载器父子同在核 4），所以一个角色通常就是一个核组。
   - **前提**：核组的核不与**其他角色**的任何登记进程相交（同角色的进程不算），否则 NOT_RUN；没有 `cpus` 的进程 NOT_RUN。单核规则另需 `pilot.json` 的 D14 结论（软中断算不算本进程的工作），没有 ⇒ NOT_RUN。
   - **读数**：判据读什么由核组的核数决定（单核 / 多核分支）。单核核组读该核的 `/proc/stat`（`cpus.jsonl` 的 `core_busy_frac*`；D14 只用在这里）。多核核组读每个进程的 GIL 域：线程 utime+stime（`threads.jsonl`），`class = python` 的线程合计为该进程的 GIL 域（每个进程一个 GIL）；组内每核读数与 `unattributed_cores`（核上的忙碌减去组内全部线程）照样算出、并列输出，但不进判据：一个核组里有几个进程（几个 GIL）时，每核忙碌分不出是哪个 GIL 域在忙（与设计稿 §2.6 的偏差，见第 10 条⑩）。线程的 `class`、`identified_as` 先取线程清单，再用 `tier.json` 的窗口始末线程表覆盖。GIL 域读数不完整时，该进程的 GIL 域一路 NOT_RUN，GIL 域的两个偏移（`T2:gil_domain_cores`、`T2:gil_run_delay_frac`）也不记进 `stats`：本档或基线档窗口里有 CPU 的线程查不到类别（`unknown`；这类线程也不当非 GIL 候选），或 `python` / `unknown` 线程在窗口里只有一行（它的 CPU 不知道）。组内任一进程在窗口里没有两行线程读数 ⇒ 该组 NOT_RUN。
   - **单元**：单核的核组是一个单元（读的就是那一个核）；多核的核组每个进程一个单元。每个单元都列出组内全部 `pids`。
   - **单核核组**：核的忙碌占比 ≥ `near_one_core_min`，并且它相对本次启动基线档的偏移在带上界之外，I1 才成立。这是设计稿「整进程只给 1 核时，看这个核的空闲时间是否接近零」的写法。D14 为真时用含软中断的 `T2:core_busy_frac_softirq:<角色>`，为假时用不含软中断的 `T2:core_busy_frac:<角色>`。
   - **多核核组的每个进程**，两条路，满足一条即成立：
     - GIL 域的核数 ≥ `near_one_core_min`，偏移超过 `T2:gil_domain_cores:<角色>`，并且 GIL 线程的运行队列等待占比 `T2:gil_run_delay_frac:<角色>` 没有超出带上界；
     - 某一条非 GIL 线程（grpc_core / zmq_io / other）单独 ≥ `near_one_core_min`，且偏移超过 `T2:nongil_thread_cores:<角色>`。
   - **为什么同时要绝对门槛和相对基线的偏移**：设计稿要求「一律与亚饱和档比」。如果自旋线程在空载时就接近 1 核，只看绝对门槛会误判。
   - **门槛来源**：`near_one_core_min` 来自 `pilot.json` 新增的可选键 `i1_calibration: {"near_one_core_min": (0, 1], "source_tiers": [...], "n": int ≥ 1}`，格式不对 ⇒ 13；缺这个键，或缺带、缺基线读数 ⇒ I1 NOT_RUN。（`pilot.json` 已不是阈值源，来源改为 pilot/2，见勘误 calib-v2-A2 第 7 条。）
   - **记录**：两种忙碌占比（每个单核核组一次）、GIL 域、运行队列、非 GIL 线程的偏移（每个多核核组里的每个进程一次）都逐档写进 `stats`，不论 pilot 有没有门槛或 D14 结论；这样 pilot 可以从它出现之前判过的树构建这些带。基线档本身的偏移按定义为 0，不记。
3. **I3（积压在哪一侧）与 I2**：
   - **登记 I3 读数**：登记表 `judge.localize: {"readings": [{"name", "terms", "receiver", "sender"}]}`。`terms` 的写法与清空门读数相同（§8.1：只许 `backlog.*` 槽、本次启动的角色、`sign` 取 ±1、`agg` 取 sum / max）。`receiver`、`sender` 是本次启动的角色或 null，两者不能都为 null，也不能相同；`name` 必须唯一。两端计数之差（在途数）记作 receiver = 读者、sender = 发送方；一个角色自己的队列记作 receiver = 它、sender = null（设计稿 §2.4 的归因规则）。
   - **读数状态**：逐秒序列取 `snapshots.jsonl` 里每个登记进程在该秒的最后一个值，缺任一进程 ⇒ 该秒为空。状态分四种：
     - `growing`：T1 超出 `T1:i3_backlog:<名>` 的带上界；
     - `elevated`：T1 未超，但相对基线档的 T2 超出 `T2:i3_backlog:<名>`，即顶在平台上；
     - `flat`：两者都未超；
     - NOT_RUN：缺带或缺数据。
   - **判定**：对角色 R，入口读数（receiver = R）有一项 growing 或 elevated，且全部出口读数（sender = R）都是 flat ⇒ I3 成立；出口读数有 growing 或 elevated ⇒ 不成立；出口读数里有 NOT_RUN ⇒ NOT_RUN。
   - **I2**：只作旁证。记录 `T2:loop_lag_p50_ns:<角色>`，不进判据，因为设计稿说要先过 E0 的标定曲线。
4. **合并与输出（§10.2）**：
   - **合并**：成立的信号（I1、I3）指向的角色：
     - 只有一个 ⇒ `LOCALIZED`，附 `finding`；但其他候选角色有信号是 NOT_RUN（没有被排除）⇒ NOT_RUN，`holding` 照列，原因写明哪些竞争角色没算出（同一角色自己的另一个信号 NOT_RUN 不拦）。`finding` 取值：`domain_saturated`（I1 由核空闲或 GIL 域判出，即「域到顶」）、`thread_saturated`（非 GIL 线程）、`backlog_at_entry`（只有 I3）。
     - 多于一个 ⇒ `UNDETERMINED`；
     - 没有，且有信号没算出 ⇒ NOT_RUN；
     - 没有，且全部信号都算出了 ⇒ `NOT_LOCALIZED`。
     - 只有 X 成立的档才下结论；其余档写 `NOT_APPLICABLE`，统计量照记。
   - **`tiers[].X.I` 的字段**：`I1`（逐单元的读数与线程份额，每个单元带 `unit`（`core_group` / `process`）、`pid`（多核单元的进程，单核单元为 null）、`pids`（组内全部进程））、`I2`、`I3`（读数与逐角色结果）、`holding`、`not_run`、`status`、`role`、`via`、`finding`、`measure_class`、`granularity`（`process`：成立的单元里有多核单元的进程；`core_group`：只有单核核组；`role`：只有 I3）、`pids`（成立单元所在核组的全部进程）、`thread_shares`（这些进程的线程份额，按 pid 分）。
   - **档的判定**：P 全过、X 成立、而 I 是 `UNDETERMINED` 的档，判 `UNDETERMINED`（原因未定，不出上限）。被推翻的是正文「第一批只实现 P 与 X」那句。
   - **`tiers[].X.V`**：本档的心跳（第 7 条）。
   - **顶层新增字段**：
     - `localisation_I`：等级所在档 I 的摘要；
     - `intervention_V`：`doses`，即逐角色的 busy / sleep 规格，各负载档都一样才写出，否则写 `mixed`；`heartbeat`；`measure_class`；
     - `grade`：第 8 条；
     - `boundary_T4.undetermined_seqs`。
5. **§10.3 退出码不变**：`UNDETERMINED` 与 `UNSTABLE` 一样是测量结果，不让退出码变成非 0。没有新增退出码。
6. **失败分类（§5.4.1 与设计稿 §2.6）**：`symptoms.reclassify_failure(role, failure, measure_class)` 返回（类别，归类）：
   - 负载器的连接类失败（`conn_refused`、`conn_fail`、`conn_reset`，由 `load_source`、`under_test` 或类别未知的角色报告）一律记非过载型，归类为 `loadgen_connection`，即使 `class_hint` 写的是 `overload`；
   - 其余失败：`class_hint == overload` ⇒ 过载型；其他一律非过载型。
   - C3、C5～C7、X、V 心跳用同一个分类，都传入 `ctx.roles`。C3 的明细加了 `by_category`。
   - 服务端日志里的原因（前端每请求积压上限的 503、看门狗中止）还没有写方，本条不读。
7. **V**：
   - **逐档心跳**：每个负载档对档位表声明的每个 busy / sleep 注入（`launch_inject` 加上本档的 `inject`）核对。「没有数据」与「注入失败」分开：
     - 计数的窗口是本档加其后的清空档，与对账链 C5～C7 同一窗口（`checks._hop_parts`）：工具按事件时刻归档，档末在途的消息在清空档里经过注入点，辅助线程（`where: aux_thread`）也会在档后才施加。E0(b) 第一轮实测：s1 档 6 经过 24741 + 14259 = 39000；s3 档 6 施加 25616 + 13384 = 39000（负载档末 `aux_pending_at_end` 13244）。所以下面的计数比较都按 worker 把两档的摘要相加后再比；
     - 某个 worker 本档没有摘要（档没跑完、没写摘要）⇒ NOT_RUN，此时不判 `applied > 0`、不比计数；
     - 本档摘要在、但没报该位点 ⇒ FAIL；两档里报了的规格与声明的不同 ⇒ FAIL；
     - 两档摘要的 `hits_delta`、`applied_delta`、`applied_ns_delta`、`expected_applied_delta`（清空档另有 `aux_pending_at_end`）任一缺键或为 null ⇒ NOT_RUN（不按 0 算）；`activation_unchanged` 不为真 ⇒ NOT_RUN；
     - 没有其后的清空档（或它没有清空门记录）、清空门没过、清空档里有 worker 没有报该位点的摘要 ⇒ 计数比较 NOT_RUN；
     - 窗口内 `applied == expected_applied`，`applied_ns ≥ applied × delta_ns`，清空档末 `aux_pending_at_end == 0`（负载档末的在途不算）；本档摘要齐全时另要 `applied > 0`（清空档可用时按窗口计，不可用时按本档计）；
     - 窗口内经过次数之和落在对账链给出的该角色那一跳的 [应有数 − 过载失败, 应有数] 内（应有数与 C5～C7 用同一个函数 `checks.hop_range`）。每请求位点查 `requests` 链，每消息位点查 `messages` / `chunks` 链（`intervene.SITE_UNIT`）。没有对应的跳（如 `beat.step`、产品探针位点）⇒ NOT_RUN。
     - 摘要里出现了没有声明的注入 ⇒ FAIL。
   - **跨启动**：`python3 -m capjudge.intervene --inputs <列表> --workdir <W> --out <新文件>`。列表与已判核对同 `capjudge.boundary`（§14）：每行一个已判（`JUDGED.tsv` 的 sha 与 `verdicts.json` 一致）、模式为 `P+X+I+V` 的树。入口过滤与 `capjudge.boundary` 相同（`boundary.upstream_now`，§14 共同规则「筛选」）：判定前的启动级结论现在不是 VALID 的树（判定后追加的 STATE 行也算）列进 `V.excluded`，不当对照、不当剂量级、不进最终等级；全部被剔除 ⇒ 3（「no launch left after the exclusions」）。其余树的实验与负载轴必须相同。不覆盖已有输出（输出或 `.cmd.txt` 已存在 ⇒ 3），产物后写 `.cmd.txt`；出错 ⇒ 退出码 3。输出 `capkit.intervention/1`。每次启动的边界用 §14.1 的 `interval`（在第一个工具到顶档截断），判定器的 T4 并列为 `boundary_t4`。
   - **剂量–反应的规则**：
     - 先看心跳：带剂量的启动，心跳必须为 PASS 才用，否则排除并写明原因；
     - 拓扑：只用一种拓扑（角色名、`kind`、`measure_class`、`workers` 都相同，`capjudge.boundary.topology`）。对照多于一种拓扑 ⇒ V NOT_RUN（列出各拓扑），最终等级也是 NOT_RUN（不同拓扑的等级不合并）；拓扑与对照不同的带剂量启动列进 `excluded`；
     - 对照：不带剂量的启动，它们都必须 `LOCALIZED` 到同一个角色；
     - 正向：只给该角色加剂量，至少要有两个 δ > 0。按 δ 从小到大，相邻两档的包络用 T5 比较，必须是「different，δ 小的一方边界更高」；
     - 反向：剂量加在 `load_source` / `stub_out_of_domain` 上，T5 必须是 `indistinguishable`。反向启动先要可用：没有 VOID 档、区间没有被工具到顶截断（`censored_by` 为空：截断说明加剂量的工具或别的工具先到顶，区间不是被定位角色的边界）、对照与反向两边的包络都闭合（开口端分不出动没动）；否则列进 `excluded` 写明原因，不进 `reverse`，也不判 FAIL。可用的反向启动 T5 为 `different` ⇒ 边界动了（FAIL）。为 `indistinguishable` 时，还要这次比较分辨得出同样大小的移动才算「边界没动」的证据：它的 `min_resolvable_diff`（两包络并集的宽度）必须小于同一标量 δ 的正向位移（该正向剂量级包络与对照包络的中点距离，该级对对照的 T5 必须是 `different`）；没有同 δ 的正向级、该级对对照不是 `different`、或分辨不出 ⇒ 列进 `excluded`，两个数都写进记录；
     - 结论：任何一步边界上移，或反向剂量让边界移动 ⇒ FAIL；正向不足两个 δ，或没有可用的反向剂量 ⇒ NOT_RUN；全部满足 ⇒ PASS；其余（某一步没有超出噪声带）⇒ `INCONCLUSIVE`。
     - 结果只用 δ = 0 时的边界，不用剂量–反应的截距去推上限。
8. **结论等级（`capjudge.grade`）**：
   - **逐启动**：按负载序，由第一个不是 STABLE 的档决定：
     - `UNSTABLE` 档：I 定位到 `under_test` / `stub_in_domain` ⇒ `LOCALIZED`；定位到 `load_source` / `stub_out_of_domain` ⇒ `LOWER_BOUND`（工具先到顶）；其余 ⇒ `NOT_LOCALIZED`（I 为 UNDETERMINED 的档判定器已改判 `UNDETERMINED`，第 4 条）；
     - `UNDETERMINED` 档 ⇒ `UNDETERMINED`；
     - 只因 C1 / C2 红而 VOID 的档 ⇒ `LOWER_BOUND`（下端为它下面最高的稳定档）；
     - 其余 VOID / NOT_RUN 档跳过（区间变宽，跳过的档逐个列出）；
     - 没有不稳定档：稳定档以上还有跳过的档 ⇒ NOT_RUN（X 在那里没有算），否则 ⇒ `LOWER_BOUND`（X 在可达范围内没有成立）；
     - 没有任何档过了 P ⇒ NOT_RUN。
     - 判定器给出的等级最高到 `LOCALIZED`，V 写「跨启动」。
     - `I_status`：决定等级的那一档是 UNSTABLE 或 UNDETERMINED 时，写该档 I 的状态（`LOCALIZED` / `NOT_LOCALIZED` / `NOT_RUN` / `UNDETERMINED`），其余写 null。`NOT_LOCALIZED` 等级有两种来源，靠它区分：I 的信号都算出了、都不成立（`I_status: NOT_LOCALIZED`），或有信号没算出（`I_status: NOT_RUN`，例如缺 `i1_calibration`，第 10 条①）；后者是「没算」，不能读成「谁都没到顶」。
   - **最终等级**（`intervene` 输出的 `grade`）：只看对照启动。`I_status` 汇总各对照启动的 `I_status`（{状态: 启动数}，null 记作 `"null"`）；对照多于一种拓扑 ⇒ NOT_RUN（第 7 条）。其余：
     - 有 UNDETERMINED，或定位到不同的角色 ⇒ `UNDETERMINED`；
     - 否则有 `LOWER_BOUND` ⇒ `LOWER_BOUND`；
     - 否则有 `NOT_LOCALIZED` ⇒ `NOT_LOCALIZED`；
     - 否则：V 为 PASS 且作用在同一角色上 ⇒ `CONFIRMED`；V 为 FAIL ⇒ `UNDETERMINED`；其余 ⇒ `LOCALIZED`。
   - **粒度**：进程（GIL 域）；定位到的进程同在一个核上时是核组（等级的 `granularity` 取 I 的）。线程份额只并列报告。
   - **两处不是设计稿原文的规定，按推荐项定（R2 自审 C25）：两条都维持**：
     - `NOT_LOCALIZED`：设计稿没有给「P + X 成立而无 I」定等级。维持，因为它从不升级；并入 UNDETERMINED 会把「没到顶原因」与「信号矛盾」混成一类。
     - 「V FAIL ⇒ UNDETERMINED」：干预结果与定位矛盾时怎么处理，设计稿没有写。维持，与设计稿「有信号指向不同进程时该点标原因未定」同一精神；保留 LOCALIZED 加注会让被干预推翻的定位留在结论里。
9. **§10.4 样本**：
   - **新增三个**：`i_server_slow`（服务端慢：待发队列增长、在途数持平 ⇒ nullsse，`LOCALIZED`）、`i_reader_slow`（读者慢：在途数顶在平台 ⇒ capload，即负载源，`LOWER_BOUND`）、`i_conflict`（I1 指向 nullsse、I3 指向 capload ⇒ 档 4 `UNDETERMINED`）。
   - **`sat_spin_thread_same_process` 改用 `P+X+I+V`**：事件循环在档 4 从 0.10 核升到 0.27 核，自旋线程为 0.70 核，档 4 判出 `LOCALIZED`、`finding: domain_saturated`。
   - **全部样本的合成 `pilot.json`**：加了 `i1_calibration`（0.95）和 I 用的带；样本线程的 CPU 改成可以按档变化（`rate_at`），其余样本的读数不变。
   - **`expect.json`**：新增 `tier_localisation`、`grade`，`run_selftest.py` 比对 `tiers[].X.I` 与顶层 `grade`。
10. **已知缺口（§12 追加）**：
    - ① `capjudge.pilot` 还不产出 `i1_calibration`（RUNBOOK §2.1 第 8 行 3B1-rest）。在它补上之前，真实启动的后果一路传到上层：每个候选角色的 I1 都 NOT_RUN ⇒ 有多于一个候选角色时，每个 X 成立的档上 I 都是 NOT_RUN（第 4 条：竞争角色的信号没算出就不给 LOCALIZED，只靠 I3 的定位也被拦下）⇒ 逐启动等级 `NOT_LOCALIZED`（`I_status: NOT_RUN`，第 8 条）⇒ V 没有已定位的对照，剂量–反应 NOT_RUN，最终等级到不了 `CONFIRMED` ⇒ 标定构建器的上限归因（§9.6，`calib.max_attribution`）没有定位到被绑定角色的启动，`loadgen_band` / `stub_calib` 的上限一律 `censored` ⇒ E1「已确认」要等 pilot 产出 `i1_calibration`。R2 自审（C3）按推荐项维持组合规则（用户定「准确度优先」：这样不会给出错的结论，代价只是能力推迟，而 E1 正式臂目前本来就暂停）；备选 (c)：缺 `i1_calibration` 时把 I1 对所有角色对称地移出组合，只用 I3 定位，其余 NOT_RUN 照样拦（改动在 `localize._conclude` 的一个条件，用户要在 3B1-rest 之前拿 E1「已确认」时再开）。RUNBOOK §2.1 第 2 行的交付口径与第 8 行的先后交用户定（本子批不改 RUNBOOK）。I 的各条带要从**以 `P+X+I+V` 判过**的亚饱和档构建，因为 pilot 读的是 `verdicts.json` 的 `stats`；用 `P+X` 判的树没有这些键。
    - ② I2 不进判据。
    - ③ 产品进程的探针侧注入（`CAPPROBE_INJECT`）没有逐档 `inject` 自报，心跳核对不了 ⇒ 这类干预的 V 记 NOT_RUN（3B2-rest）。
    - ④ `beat.step` 等没有对账跳的位点，心跳 NOT_RUN。
    - ⑤ 设计稿「E0 自检在哪类已知答案上失败，对应形态的等级封顶在已定位」没有实现。
    - ⑥ 弱读者检验仍未实现（B2-2A-10①），C2 的「积压所在一侧」由 I3 承担，但 I3 不进 P。
    - ⑦ I3 按角色判，I1 按核组 / 进程判（第 2 条）。多进程角色的「指向同一进程」按角色比较，`pids` 列出 I1 成立的单元所在核组的全部进程。
    - ⑧ 合成的读者慢样本为了让 I 被判到，保持 C1 绿；真实的读者慢多半会让 C1 的 FIONREAD 变红，档 VOID，等级同样是 `LOWER_BOUND`（经 C1 那条路）。
    - ⑨ 分析脚本 `boundary.py`、`scaling.py`、`e1_report.py` 与 §14 不在本步（已补，见勘误 B2-2E-A）。
    - ⑩ 与设计稿 §2.6「读数以独占核集上的每核 /proc/stat 为准」的偏差：只有单核核组按每核 `/proc/stat` 判；多核核组按每进程的 GIL 域（线程 CPU）判，每核读数与 `unattributed_cores` 只并列输出（第 2 条）。理由：核组里有几个进程（几个 GIL）时，每核忙碌分不出是哪个 GIL 域在忙，按它判会把多进程角色的总忙碌当成一个 GIL 到顶，正是设计稿 §2.6 I-4 要避免的。按推荐项定（R2 自审 C7），只让文字与代码一致，不改判定。
11. **验证**：
    - **测试**：`tests/judge/test_localize.py`、`tests/judge/test_intervene.py`（I1 单核 / 多核 / 非 GIL / 独占核，I3 四种状态与两侧规则，合并，失败归类，心跳逐项，剂量–反应 PASS / FAIL / INCONCLUSIVE / NOT_RUN，降序轴，等级，`mode` 与 `localize` 的输入核对，`i1_calibration` 格式，CLI）；`tests/judge` 全量、样本自测全过；变异 N1～N14 全部杀死（`_local/b2-judge/mut/summary2.txt`；实测数字记在 `_progress/b2-judge.md`）。
    - **e2e**：E0(b) J5933747 L1～L4 拷到 `_local/b2-judge/e2e/trees2`，在进程内按 `P+X+I+V` 判。门槛用诊断值 0.90，不是登记值；E0 第一轮没有 X 的带，所以 X 为 NOT_RUN，只看 I1。结果：
      - s1、s2、s3 在过载档（档 6）都是 relay1 的 I1 成立（核忙碌 0.996～0.998），亚饱和档与近边界档都不成立；
      - s2 的 relay2（δ 较小）不成立；
      - s3 的线程份额：`capkit-inject-aux-relay.msg` 0.915 核、`MainThread` 0.046 核，辅助线程一侧在 GIL 域内判出；
      - s4（sleep，不耗 CPU）各档 I1 都不成立，与已知答案「I1 认不出不耗 CPU 的瓶颈」一致。
    - 记录在 `_progress/b2-judge.md`。

#### B2-2E-A：分析脚本（2026-11-14）

新增正文 §14（分析脚本的输入与输出）。判定器的输入、输出与退出码都不变。

1. **`capjudge.boundary`、`capjudge.scaling`（§14.1、§14.2）**：薄封装 `analysis/boundary.py`、`analysis/scaling.py`。共同的 `.cmd.txt` 由 `capjudge.cmdrec` 写，满足 DATA-AND-ANALYSIS §4 第 8 条；`capjudge.calib`（构建器）与 `capjudge.intervene` 同样写 `.cmd.txt`、同样拒绝已存在的产物或记录。
2. **跨启动用的区间不完全等于判定器的 T4**：在第一个工具到顶档以下截断（§14.1），工具先到顶时上端写 null。判定器的 `boundary_T4` 原样并列输出。`capjudge.grade` 的逐启动区间是另一种口径：它取第一个非稳定档，非单调时不同。每个读方各选口径：T5（`boundary --compare`）与 V（`intervene` 的正向、反向）用截断后的 `interval`，问的是被定位角色的边界；T6 与对侧翻倍归因（`scaling`）用判定器的 T4（`by_block_t4`），问的正是哪一侧先到顶，工具到顶档是证据；标定上限（§9.6）用 T4 下端（已改为等级区间下端，见勘误 calib-v2-A1 第 1 条），并按等级把未定位到被绑定角色的启动记为 `censored`。
3. **区组 = 作业号**（设计稿 §2.6 T5「同区组」、RUNBOOK §1.3「3 个作业 = 3 个区组」）。同一作业里同一臂有几次启动时，它们进同一个区组的包络。
4. **T6 用区间算比值**：设计稿 T6 的分子分母是数，这里两臂的边界都是区间，所以逐区组给比值的上下界，判断取不利端（§14.2）。只用下端、只用上端的两个 T6 区间并列给出，作对照。`grows-min`、`flat-max` 的取值设计稿没有给，要在实验登记表里事先登记，运行时显式传入。
5. **对侧翻倍按 `kind` 分侧**：client 侧是负载器，server 侧是空汇 / 桩，aux 角色不参与。E1 卡「只加 Z 不加 W 时不再增长 ⇒ 限制在负载器；只加 W 不加 Z 时不再增长 ⇒ 限制在空汇」写成：加倍的一侧 `grows` ⇒ 限制在这一侧，`flat` ⇒ 限制在另一侧；两侧的结论吻合才给出 `client` / `server`。
6. **`capjudge.e1report`（§14.3）**：为了让字节对账不复写空汇的头格式，`capload.sink` 抽出了 `response_head(rid, foreign)`，空汇与分析共用。空汇的写法与字节数不变。
7. **空汇健康的带从哪来**：E1-0 的 `e1_noise` 里没有空汇的 `T3:loop_lag_p99_ns`、`T3:thread_cpu_cores`。这两个统计量只有判定器跑 C2 时才写进 `stats`（没有 `stub_calib` 时 C2 是 NOT_RUN，但仍会记录），而 E1-0 的检查清单没有 C2。所以对现有 E1 树，`sink_health` 是 NOT_RUN。要用它，标定树的 `judge.checks` 要加 C2，再用 `calib_build.py near-boundary --regime subsat` 建带并登记。注意：没有登记 `stub_calib` 时，列进 `judge.checks` 的 C2 是 NOT_RUN，会让这些标定树的每个负载档判定都是 NOT_RUN，只有 `stats` 可用；`calib_build.py near-boundary` 不剔除 NOT_RUN 档，所以带照样能建，但这批树的判定本身不能再当稳定性结论用。另外，「线程 CPU 留在空载带内」按字面实现：空汇的 CPU 随负载线性上升，高负载档可能因此 OUTSIDE。已按推荐项定（R1 自审 C30）：维持字面实现，OUTSIDE 只作 E1 报告的旁证，不进判定；是否改成「离 1 核足够远」（需要一个另行登记的阈值）列为 follow-up。
8. **斜率对账只用 `P+X+I+V` 判过的启动**，因为剂量与心跳只在这种模式下写进 `verdicts.json`。p 由构造给出，不从心跳计数反推。
9. **已知缺口**：
   - `scaling` 的两个界没有登记表的机器可读入口，只靠命令行加 `.cmd.txt` 记录；`slope` 没有另设容差：「斜率为 1」要求区间包含 1、且下端 > 0（同时包含 0 的区间分不出「斜率 1」与「剂量没让边界动」，记 INCONCLUSIVE），区间宽度由档距决定；
   - `boundary` 的筛选不读 `launch_end.json`（判定器的 `upstream` 已包括它）；
   - gRPC 的 accepts 没有计数方；
   - `slope` 不检验截距。
10. **验证**：
    - **测试**：`tests/judge/test_analysis.py`，覆盖区间截断、T4 重算核对、T5 区组、对侧翻倍的分侧与归因、斜率 PASS / FAIL / INCONCLUSIVE / NOT_RUN、字节与 accepts、逐档摘要齐全、拓扑、空汇健康的登记核对与带比较、`.cmd.txt`、不覆盖。变异 M1～M7、E1～E7 全部杀死（`_local/b2-judge/mut/summary3.txt`～`summary5.txt`）。`tests/judge`、`tests/capload` 全量与样本自测全过（实测数字记在 `_progress/b2-judge.md`）。
    - **e2e**：全部在 W 只读下进行，产物在 `_local/b2-judge/e2e/an/`：
      - E0(b) 12 棵已判真树跑 `boundary`；
      - e0a 3 棵树、E1-0 15 棵树（按 R、M 两条轴分组）跑 `boundary` 与 `e1report`：字节对账全部 PASS（HTTP 重建头后相等，gRPC 直接相等）；HTTP 的 accepts 全部 PASS（new-per-request 臂 30200 = 30200）；gRPC 的 accepts 是 NOT_RUN；这些树判定时还没有噪声带，T4 都是 `none`，斜率都是 NOT_RUN。
    - SM-9 的离线判读写在 `_local/b2-judge/sm9.md`。

#### B2-R1：本子批 R1 自审修复（2026-11-15）

R1 自审（slug `capkit-b2judge-J17871`）发现的问题已直接改进本子批的正文（§9.6、§9.8、§14）与上面 B2-2A、B2-2E、B2-2E-A 各条，这里只列改了哪里：

1. I1 按核组算，同角色共享核（父进程、兄弟 worker）不再算「不独占」：B2-2E-2、B2-2E-4 的字段、B2-2E-10⑦。
2. 合并：其他候选角色有信号 NOT_RUN 时不给 LOCALIZED：B2-2E-4。
3. T5：没有一个区组两边都闭合 ⇒ NOT_RUN：§14.1。V 的正向、反向改用截断后的区间；反向启动被截断、或对照 / 反向包络开口 ⇒ 排除，不判 FAIL：B2-2E-7。心跳分开「没有数据」（NOT_RUN）与「注入失败」（FAIL）：B2-2E-7。
4. 斜率对账的 PASS 另要区间下端 > 0：§14.3、B2-2E-A-9。字节对账分方向，空汇多发且负载器有失败 / 未完成请求 ⇒ NOT_RUN；计数缺键 ⇒ NOT_RUN：§14.3。
5. `scaling` 用判定器的 T4（`boundary` 新增 `envelope_t4`、`by_block_t4`）：§14.1、§14.2、B2-2E-A-2。
6. C4 失败请求不读服务端的中止前计数、服务端记录不跨角色合并：B2-2A-5。C12 的负载器单位读 `counts.sent`：B2-2A-6。C1 / C2 对 `censored` 上界超出 ⇒ NOT_RUN：B2-2A-3、§9.6。
7. 标定：`near_boundary_noise` 可登记多份：B2-2A-1、B2-2A-2、§9.8。构建器剔除 UNDETERMINED：§9.8、B2-2A-8。`loadgen_band` / `stub_calib` 上界按等级归因记 `censored`、输入写 `attribution`：§9.6。`calib`、`intervene` 写 `.cmd.txt`：B2-2A-8、B2-2E-7。
8. 等级：去掉判定器不会走到的「UNSTABLE 档 I 为 UNDETERMINED」分支：B2-2E-8。空汇健康的判据按推荐项定：B2-2E-A-7。已知缺口追加 C12 槽来源单位含父进程：B2-2A-10⑦。
9. 样本：新增 `i_reader_slow_real_pins`、`good_real_pins`（集群上的拓扑：父进程登记、角色核组共享）；`c12_idle_worker` 改为 worker 1 一个请求都没发出（`samples/README.md`）。
10. 验证：`tests/judge` 全量、样本自测、变异（`_local/b2-judge/mut/summary6.txt`）与 e2e 的实测记录在 `_progress/b2-judge.md`。

#### B2-R2：本子批 R2 自审修复（2026-11-15）

R2 自审（slug `capkit-b2judge-J17871`）发现的问题已直接改进本子批的正文（§9.6、§9.7、§14.2、§14.3）与上面 B2-2A、B2-2E、B2-2E-A 各条，这里只列改了哪里：

1. V 心跳的计数窗口改为本档加其后清空档（经过次数、施加数、注入时长都按 worker 两档相加后比，清空档末才看 `aux_pending_at_end`），应有数与 C5～C7 共用 `checks.hop_range`：B2-2E-7。
2. `intervene` 走与 `boundary` 相同的入口过滤：B2-2E-7。跨启动只合并同一拓扑：`intervene` 的对照与剂量级（B2-2E-7、B2-2E-8 最终等级），`e1report` 的斜率（§14.3）。反向剂量要分辨得出同 δ 正向位移才算证据：B2-2E-7。
3. `scaling` 不再跳过比值算不出的区组：§14.2。
4. C4 有请求要回退时，客户端角色或写流记录的服务端角色多于一个 ⇒ 回退部分 NOT_RUN：B2-2A-5。桩的均分负载只在恰有一个客户端角色时成立：§9.7、B2-2A-4。构建器参与的启动没有 T4 下端（已改，见勘误 calib-v2-A1 第 1 条）或上限负载算不出 ⇒ 报错；列表按 (树, 角色) 去重：§9.6、B2-2A-8。
5. I1 的 GIL 域读数不完整（类别不明而有 CPU 的线程，或窗口内只有一行的 python / 类别不明线程）⇒ 该进程 GIL 域 NOT_RUN；C2 的线程 CPU 有单行线程 ⇒ 没有值：B2-2E-2、§9.7、B2-2A-4。I1 实际读什么、与设计稿的偏差：B2-2E-2、B2-2E-10⑩。
6. 等级输出加 `I_status`：B2-2E-8。缺 `i1_calibration` 的后果链与组合规则的裁决：B2-2E-10①。B2-2E-8 两条推断改为裁决结论。
7. `sink_health` 核对带文件的环境：§14.3。没有 `stub_calib` 时列 C2 的后果：B2-2E-A-7。C12 只在加倍是处理时列入：B2-2A-10②。
8. 验证：`tests/judge`、样本自测、变异（`_local/b2-judge/mut/summary7.txt`）与 e2e 的实测记录在 `_progress/b2-judge.md`。

### Calibration v2 SB-B1: capenv.spinners aux tool for the C13 positive control

依据：E0 v2 设计 `02_e0/E0/analysis/calib-v2/DESIGN-r7a.md` 的 V1CD-01 与测试第 26 条（V0 的 `c13-pc` 启动，以及 SB-B2 之后 V1-c/d 的陪跑负载）。本节的条目覆盖正文中对应的句子，正文相关处已加指针。

1. **新入口 `python -m capenv.spinners`**（`tools/capenv/spinners.py`）。参数：通用（§2.1 身份、`--run-dir`、`--stop-grace-s`、`--cpus`）、`--origin-ns`、`--ready-file`、`--pin-table PATH`，全部必需；`--target-role ROLE` 可选；不收 `--inject`；参数不接受前缀缩写。行为：
   - `--cpus` 的每个核起一个自旋进程（fork），各自 `sched_setaffinity` 绑到该核，跑纯 Python 的 31 位 LCG（每单位 2000 步，单线程，持 GIL）。
   - 给了 `--target-role` 时，另起一个自旋进程，绑到绑核表 `roles[ROLE].cpus` 列表的第一项。ROLE 是绑核表 `roles` 的键（即被测角色的 `pin_role`）。这个自旋进程自己调用 `sched_setaffinity`，绑核表只读不改，绑核表的生成也不变。
   - 起进程的那个进程只做监督，不自旋：它绑到 `--cpus`，等每个自旋进程报回亲和性读回值（恰好是自己那个核；全部合计最多等 10 s），然后才写就绪文件。
   - 第一次 SIGTERM/SIGINT：向自旋进程发 SIGTERM，`--stop-grace-s` 之后还在的发 SIGKILL。第二次信号：剩下的立即 SIGKILL。
   - 自旋进程每 64 个单位查一次父进程是否还在，监督者被 SIGKILL 时自己退出。
   - 退出码：
     - 0：被信号停止，每个自旋进程都报了结果（被第二次信号杀掉的不算缺）；或者在就绪之前就收到停止信号，而已经报回的亲和性都正确（记录里 `ready: false`，不要求结果，没报回亲和性的也不算缺）。
     - 3：绑核表读不到、不是 `capkit.pin_table/1` 或没有 `roles`、ROLE 不在表里或没有核、目标核落在 `--cpus` 里。这些都在起任何自旋进程之前判，不写就绪文件，也不写记录。ROLE 的条件在场景校验阶段已经核过（第 3 条），工具里的检查是兜底。
     - 4：监督者把自己绑到 `--cpus` 失败（照常写记录，`supervisor_affinity: null`、`spinners: []`，不起自旋进程）；某个自旋进程绑核失败或读回不符，没报回亲和性就退出，或者在等待时限内没有报回亲和性（这几种都不写就绪文件）；在停止之前就退出了；宽限期过后仍没有报结果。凡是退出 4，都写记录，原因在 `problems`。
   - 本工具不做性能测量。记录只说明跑了什么，不作为判定依据。
2. **接线**（`caplaunch/scenario.py`；作业级副本的核对在 `caplaunch/orchestrator.py`）：
   - `TOOL_TABLE` 加 `"capenv.spinners": {"origin": True, "ready": True, "pin_table": True, "role_fixed": {…}}`（`role_fixed` 见第 3 条）。
   - 新参数组 `pin_table`：`build_argv` 追加 `--pin-table <T>/raw/env/pin_table.json`，路径用 `RunTree(<T>).env("pin_table.json")` 求出。这正是编排器在启动开始时放绑核表副本的位置（§5.1 的 `raw/env/`；设计稿写作「<启动树>/env/pin_table.json」，指的就是这个文件）。作业级 env 目录（`--job-env-dir`）里若有同名文件，编排器要求它与 `--pin-table` 字节相同，否则在起任何进程之前拒绝（退出码 3）；所以启动树里的副本就是绑核用的那份。场景里不写这个路径。
   - `--pin-table` 加入 `MANAGED_FLAGS`，对所有 capkit 工具生效：场景 `args` 里出现 `--pin-table` 或 `--pin-table=…` ⇒ 场景校验失败（退出码 3）。
3. **场景里的角色写法**：`kind: aux`，`measure_class: observer`，`workers: 1`，`expected_procs: 1`，`asyncio: false`，`loop: null`，`uses_grpc: false`，`snapshots: none`，`backlog_slots: []`，`readiness: [{"kind": "capkit_ready"}]`，`proc_kinds: []`。`cpus_needed` 等于本角色自己的自旋核数（至少 1）；给 `--target-role` 时，总自旋进程数是 `cpus_needed + 1`。
   - `kind`、`measure_class`、`expected_procs`、`asyncio`、`uses_grpc`、`snapshots` 这六项由场景校验强制（`TOOL_TABLE` 的 `role_fixed`，不符 ⇒ 退出码 3）。`measure_class` 必须属于编排器的 `MONITOR_CLASSES`，这一对齐由编排器的导入期断言保证。理由见第 4 条：自旋进程是未登记的子进程，在其他类别的角色下面会被记为 `unattributed`。
   - `--target-role X` 由场景校验核对（不符 ⇒ 退出码 3）：X 是字面值（不含 `{` 占位符），只出现一次；X 是场景里另一个角色的名字；该角色的 `pin_role` 等于 X（工具按 `pin_role` 在绑核表里取核，判定器的 C9/C13 按角色名取核，两者要指向同一条）；X 在本角色的 `start_after` 里（目标角色就绪之后才起自旋进程）。
   - 本工具只记录，不发快照，不判档。
4. **§2.3「多进程工具必须列出子进程」的例外**：`capenv.spinners` 的就绪文件 `children` 是 `[]`，自旋进程不登记。
   - 原因：登记了的进程，C10 要求它有版本自报（§2.5）。编排器只从快照 meta，或从起进程那个进程自己的就绪文件里取版本自报（`orchestrator._versions`），自旋进程两者都没有，一旦登记，整次启动就 C10 NOT_RUN。
   - 后果：自旋进程是 observer 角色的未登记子孙，记在 `procs/monitor_children.json` 与各档 `tier.json` 的 `monitor_descendants`，不进窗口始末的 pid 比较，也不进 `unattributed`。
   - 对 `c13-pc` 的影响（按代码推断，没有在真实 sidecar 与判定器上跑过，由 V0 的 c13-pc 读回确认）：`check_c9` 跳过不属于登记 pid 的线程行，`_foreign_frac` 用核忙碌减去 relay1 的登记线程。所以目标自旋进程的忙碌应当算进 C13 第二半的外来忙碌，C9 不应因它判 RED。设计稿里的「C9 RED 预期」按代码推断不会出现，见上。c13-pc 的判据不变（C13 每窗与 `C13-MEAN` 都 RED），C9 的结果在该启动上不作要求。
5. **输出 `raw/spinners/<角色>.json`**（`capkit.spinners/1`，带本次启动的身份，判定器的身份扫描照常核对；判定器不读内容）。键：
   - `origin_ns`、`unit_ops`、`workload`、`cpus`、`supervisor_affinity`；
   - `pin_table`：`path` 与读到的字节的 `sha256`；
   - `target_role`、`target_cpu`（不给 `--target-role` 时为 null）；
   - `ready`、`ready_rel_ns`、`stop_rel_ns`（相对原点）；
   - `spinners`：每个自旋进程一行，含 `index cpu purpose(own|target) target_role pid affinity pinned_ok reported units cpu_ns wall_ns cpu_frac orphaned last_cpu exit killed(false|second_signal|grace_expired)`；
   - `cores`：每个自旋核从就绪到停止的 `/proc/stat` 各列 tick 差、`busy_ticks`（同 `procfs.BUSY_COLS`）、`total_ticks`、`busy_frac`。读不到时为 null，原因写在 `cores_unreadable`；
   - `clk_tck`、`stop`（`control/stop.json` 的原文，§2.3）、`signals`、`problems`。
6. **测试**：
   - `tests/envgate/test_spinners.py`：纯函数、监督者的等待与停止逻辑，以及本机真实 fork 的 CLI 的正常与各失败路径。
   - `tests/launch/test_scenario.py`：本节对场景的接受与拒绝规则，以及 `build_argv` 给出的参数。
   - `tests/launch/test_launch_e2e.py`：用真实 `capenv.spinners` 的启动 e2e，以及起进程之前的拒绝。
   - `tests/launch/fakepkgs/capenv/__init__.py` 现在把真实的 `tools/capenv` 接在假包之后：只有 canary、sidecar 是假的，其余 capenv 模块用真的。

### Calibration v2 SB-A: pilot/2, statistics, thresholds, judge levels, laddergen and budget gates

依据：calib-v2 `W/02_e0/E0/analysis/calib-v2/DESIGN-r7a.md` §3（P2-00～P2-10）与步骤清单 `calib-v2/impl-steps-SB-A.json`。本节按步骤分小节，排在后面的小节优先；正文与前面各勘误中被取代的句子只加指引，不改写。

#### calib-v2-A1：b2-judge FU-3 / FU-4 与 stats 原语（2026-11-15）

1. **形态构建器的上限档（FU-3；取代 §9.6 `per_worker_max` 的「`boundary_T4` 下端（最高 STABLE 档）」与 B2-2E-A-2 的「标定上限（§9.6）用 T4 下端」）**：每次启动的上限档 = 第一个非稳定的已判负载档之下最高的 STABLE 档。「第一个非稳定档」按 `capjudge.grade.first_non_stable` 走（`launch_grade` 用的同一个函数）：按负载序走已判负载档，RED 检查不只 C1 / C2 的 VOID / NOT_RUN 档跳过，第一个 UNSTABLE、UNDETERMINED 或只有 C1 / C2 RED 的 VOID 档就是它。所以上限档就是等级的 `interval.lower_seq`，归因（`max_attribution`）说的正是这个第一个非稳定档。
   - 启动带等级（按 `P+X+I+V` 判）时，构建器核对等级的 `interval.lower_seq` 与上面算出的档相同，不同 ⇒ 报错（退出码 3）。
   - 没有这一档（第一个非稳定档之下没有 STABLE 档）⇒ 报错并写明是哪棵树；它取代原「T4 没有下端 ⇒ 报错」。
   - `censored` 的条件「有一次从没 UNSTABLE」改为「有一次没有第一个非稳定档」；归因条件不变。
   - 每个输入新增 `max_tier: {"tier_seq", "first_non_stable_seq", "t4_lower_seq"}`（取的档、第一个非稳定档、T4 下端），`boundary_T4` 照旧原样记下。上限档低于 T4 下端的情形：第一个非稳定档之上还有 STABLE 档（T4 非单调，或只有 C1 / C2 RED 的 VOID 档之上有 STABLE 档）。
   - B2-2E-A-2 其余读方的口径不变：T5 与 V 用截断后的 `interval`，T6 与对侧翻倍用判定器的 T4。
2. **`non_monotone` 计入 UNDETERMINED 档（FU-4）**：`stats.t4_boundary` 的 `non_monotone` = 最高 STABLE 档高于最低的「X 成立」档（UNSTABLE 或 UNDETERMINED）。区间仍只由 STABLE / UNSTABLE 档构成，`lower`、`upper`、`lower_seq`、`upper_seq`、`kind`、`n_stable`、`n_unstable` 都不变，所以 §14.1 的 T4 重算核对（只比其中五项）不受影响。判定器写的 `boundary_T4.non_monotone` 随之；`grade.launch_grade` 的提示随之出现，措辞为「T4: a stable tier lies above the lowest unstable or undetermined tier」。E0 v2 阶梯启动的 b 与级上的 `non_monotone`（DESIGN-r7a §2.4）不读这个字段，由 P2-06 在 `ladder.json` 的级上另算。
3. **stats 原语（P2-01）**：`capjudge.stats` 新增下列纯函数，只用标准库，本步没有调用方（`t3_band`、`t5_compare` 照旧）。算不出时返回 `stats.NotRun(reason)`（同 reason 的两个相等），违反入参约定时抛 `ValueError`；T1～T6 旧函数仍返回 None。
   - `zd(series, quantum=None)`：DESIGN-r7a §2.2 的 z_d，返回 `{z, D, sigma, n1, n2}`；差分不足 ⇒ `NotRun("too_few_diffs")`，尺度为 0 而 D ≠ 0 ⇒ `NotRun("degenerate_scale")`。SD 回退用总体 SD。
   - `t_sf(x, df)`、`t_isf(p, df)`：t 分布尾概率与分位（不完全 beta，移植 `calib-v2/r4/tdist.py`）。
   - `kmax(n, p, gamma=0.95)`、`n_eff(n_windows, m_mean, icc_ub)`、`rank_for_k(k, n_windows, n_eff)`、`fold_rank(r, n, n_fold)`、`order_stat_thresholds(values, n_eff, p)`：§2.3 的 k、有效窗口数、全样本秩 r、留一折秩 r′ 与双侧阈值（k = 0 ⇒ `uncalibrated`）。`n_eff` 的商加 1e-9 再取整，防止整数商被浮点舍入成少 1。
   - `icc_oneway(groups)`、`icc_cluster_boot(groups, B, seed)`：单因素 ICC(1)（不等组长，移植 `calib-v2/r2/icc_zd.py`）与按启动整群自助的 95 % 上界（第 ceil(0.95·有定义的重抽次数) 小，下限截为 0）。
   - `signflip_exact(diffs)`：精确双侧符号翻转 p（n ≤ 20 全枚举，更大折半枚举；先换成同一刻度的整数，0 与并列精确计入）。`signflip_shift_ci(diffs, alpha)`：反演检验得平移区间（n ≤ 20 用成对子集均值的次序统计量精确给出，更大时二分）。`t5_block(diffs, alpha=0.05, min_blocks=8)`：§2.10 的 T5 v2，不足 8 个区组 ⇒ `NotRun("blocks_lt_8")`，`direction` 取 Σd 的符号（d = A − B，正为 `a_higher`）。
   - `t7_fit(points)`：§2.7 的 g = λ/μ + β 最小二乘（points 为 (λ, g)，g = 2·T1/L 无量纲）。`t7_single(lam, g)`：β 固定 −1，μ = λ/(g + 1)。
   - `tol_cell(n, F, n_cand=10, floor=0.6)`：§2.6 第 3 步的 TOL_cell（二维 Simpson 积分 121 × 121 点、二分 32 次，移植 `calib-v2/r5/settle_r5.py`），按全部入参缓存。`settle_rule(cells, cands, F)`：§2.6 第 3～5 步，每格给 n、`tol`、`s_ref`、各候选的 m / s / ub / pass（未检验的候选 pass 为 null）、`s_cell`（或 `unconfirmed`）、`insufficient`（n < 60，交第 4 步的后备补投判定；此时 `s_cell` 照算但不是定值）、`kurt_ref` 与最大候选的超额峰度；第 2 步的排除与第 6 步的 S_reg 由调用方做。
   - `excess_kurtosis(values)`：矩估计 m4/m2² − 3，只作描述。`shortfall_topup(d, lost, ended, mult=1.25)`：§4.5 的 ceil(mult·d/(1 − q̂)) + 1，有理数精确计算。
4. **与 DESIGN-r7a §3.5 测试表的出入**（实现逐字按正文公式；测试按实测性质断言，出入在此登记）：
   - 第 1 条「sd 1.00 ± 0.05」只在 L = 60 成立（实测 1.03，含随机空秒 1.04）。L = 20 实测 1.08（含空秒 1.15），与 §2.2「L ≥ 14 时 sd 1.085～1.175」一致，测试按这个性质断言 1.05～1.20。登记窗口是 L = 60。
   - 第 8 条「TOL 换成 0.1 时阴性样本几乎全部 unconfirmed」：n = 140 时实测约三分之二 unconfirmed，其余停在 240～400；测试断言没有一格得 S = 0、且至少一半 unconfirmed。
   - 第 8 条（100 s, 20σ）的「S ≥ 解析真值」：真值 400，候选 300 处的期望偏移是 TOL 的 1.01 倍，50 格里有 1 格停在 300；测试断言中位数 ≥ 真值且 ≥ 95 % 的格 ≥ 真值。另两种趋近逐格 ≥ 真值。
   - 第 8 条「与 `settle_r3.txt` 同种子的数一致」：存档只有 200 格的汇总，没有逐格结果；测试在同种子、同序列上逐格与 r3 的规则（Cornish-Fisher t95、TOL 0.6）比较 S_cell，6 种情形各 50 格全部相同。
   - 第 8 条的真实片段子窗部分属于 P2-12，本步没有做。
5. **验证**：`tests/judge/test_stats_calibv2.py`（第 1、2、3、8（stats 部分）、11 条），`tests/judge/test_stats.py`、`tests/judge/test_calib.py`（第 41 条：`repro_b2.py` 的三例，外加跳过 VOID 档、等级区间不符、样本树上的构建器实跑）。变异与全量结果记在 `_progress/calibv2-a.md`。

#### calib-v2-A2：阈值源（pilot/2 两种 stage、d14_result）、v1 标定与 inline_thresholds 拒收、arm_config / arm_threshold_config / ladder、按 ladder 角色定 regime、新 REGISTRY kinds（2026-11-15）

1. **v1 标定文件不再是阈值源**（取代 §9.2「`calibration[].kind` ∈ … `near_boundary_noise`」中 v1 文件可作阈值源的部分、§9.4 作为阈值源的用法、§9.8 末两条、B2-2A-1「`near_boundary_noise` 可以登记多份」）：任何登记表（有没有 `arm_config` 都一样）的 `calibration[]` 里，kind `near_boundary_noise`（一份或多份；不读文件，先拒）或 kind `pilot` 而文件 schema 为 `capkit.pilot/1` ⇒ 13，`inputs.json` 的 `detail` 恰为 `v1 calibration is not a threshold source (calib-v2 erratum)`。
   - 后果：引用 E0 第一轮 `pilot.json`（sha `8e823f12…`）的登记表（E1-0 两份）上的树，判定器、`pilot.select_tiers`、`e1report` 都得 13。E0 第一轮登记表 `calibration` 为空，照旧通过（`pilot2 d14` 取树要用它）。
   - v1 构建器（`capjudge.pilot`、`capjudge.calib` 的三个子命令）照旧可用，`calib.validate` 照旧核对它们的输出；只是判定器不读。
2. **阈值源 `capjudge.nullz`**（DESIGN-r7a P2-02、§3.2）：`calibration[].kind` 新增 `d14_result`；kind `pilot` 只接受 `capkit.pilot/2`。每种至多一份；格式不对 ⇒ 13；两种都核 `environment` 的 `hw_profile_sha`、`image_sha` 与本树一致（同 B2-2A-1，原因含 `calibration environment`），pilot/2 的 `environment` 另须有 `judge_sha`。
   - `capkit.pilot/2` 共有：`stage` ∈ `thresholds`、`bootstrap`；`d14: {relpath, sha256}`；`c13_cap: {"T3:foreign_busy_frac": {window, launch_mean}}`（都 > 0）。同时登记了 `d14_result` 时，`d14.sha256` 必须等于它的 sha，否则 13。
   - `stage: bootstrap`：`x_const: {lag: {z > 0, min_D_ns ≥ 0}, count: {z > 0}}`、`ratio_tol: {"T2:delivery_ratio_norm": {min_per_request: 整数 ≥ 1, tol ≥ 0}}`。其余字段（`rule`、`inputs`、`max_abs` 等）由 P2-09c 写，判定器不读。
   - `stage: thresholds`：`window_ns`（整秒）；`null_z` 的键 `<T>:<量>:<角色>|<配置>|subsat 或 near|L=<秒>`（L 必须等于 `window_ns`），值 `family` ∈ `lag`、`count`，`uncalibrated` 为布尔，非 uncalibrated 时 `thr_lo ≤ thr_hi` 为数；`ratio` 的键 `T2:delivery_ratio_norm:<角色>|*|*`，值 `abs_tol ≥ 0`；`i1_calibration` 可选，格式同 B2-2E-2。
   - `capkit.d14_result/1`：`rule: {relpath, sha256}`、`environment`（四个环境 sha）、`inputs` 列表、`d14_softirq` 带布尔 `irq_softirq_is_experiment`。
   - 读法（供 P2-04、P2-05、P2-13 调用；本步没有调用方）：`get(统计量键, 配置, regime)`，regime 用判定器的 `subsat` / `near_boundary`（键里写作 `subsat` / `near`）。bootstrap 对任何配置与 regime 返回同一族值：滞后族 = T1 的 `req_first_lag_p50_ns`、`send_lag_p50_ns`、`msg_lag_p50_ns`；计数族 = T1 的 `inflight_derived` 与 `backlog.*`；比值 = `T2:delivery_ratio_norm`；其余键没有阈值。`compare(z, 项, D=…, planned_requests=…)`：`null_z` 与计数族、比值按严格 `>` / `<`；bootstrap 滞后族 `z ≥ z_lag 且 D ≥ min_D_ns` ⇒ above，不会 below；bootstrap 比值的容差 = max(`min_per_request` / 计划请求数, `tol`)；uncalibrated 键、没有项、没有值 ⇒ None（该项 NOT_RUN）。`cap(键)`、`d14()`、`i1_near()`（thresholds 的 `i1_calibration.near_one_core_min`）。
   - 输出：`inputs.json` 新增 `thresholds`（每个源的 kind、relpath、sha256、schema、stage）与 `threshold_lookups`（`<键>|<配置>|<regime>` 或 `cap|<键>` → `found` / `missing` / `uncalibrated`）。
   - 本步定下、设计稿没写死的两点：`null_z` 的 `family` 只认 `lag`、`count`，P2-14 给 I 键另定族名时要同时扩这里；thresholds 的比值容差不加 2/计划请求数的结构下限（设计稿只对 bootstrap 写了它）。（**后半句已由勘误 calib-v2-A8 第 4 条取代**：DESIGN-r7a §2.9 给 V2 的比值容差写了下限 1/(该档计划请求数)，`ratio_abs` 按它实现。）
3. **`arm_config` / `arm_threshold_config`**（DESIGN-r7a P2-03）：没有 `arm_config` 的登记表照 v1 读（不取配置、不读 ladder）；只有 `arm_threshold_config` ⇒ 13。有 `arm_config` 时它的键必须恰好是 `arms`（缺本臂、多出别的臂都 13），值是非空、不含 `|` 与 `*` 的配置名；`arm_threshold_config` 可选，缺省等于 `arm_config`，有则键集同样必须恰好是 `arms`。判定器取阈值只用 `arm_threshold_config[臂]`；`ladder.json` 的配置核对与 `verdicts.json` 用 `arm_config[臂]`。`verdicts.json` 顶层新增 `arm_config`、`arm_threshold_config`（本臂的两个配置名；v1 登记表为 null）。臂名不解析。
   - **臂名的字符集**：`capcommon.identity.ARM_RE` 只许 `[A-Za-z0-9._-]`（首字符字母数字，≤ 64），DESIGN-r7a P2-03 的臂名约定 `<配置>@<block_id>.<slot>`、`<配置>@n<序号>` 含 `@`，照写的臂起不了树。生成臂名的一方（P2-07 / 各期登记表）要换成 `ARM_RE` 接受的写法；配置名不受此限（它不进树名与 identity）。
4. **按臂的 `ladder.json` 与 regime**（取代 B2-2A-2 对 v2 登记表的用法）：有 `arm_config` 且用 `files_by_arm` 的登记表，`files_by_arm[臂]` 可多一个 `ladder`（kind `ladder`，与其余文件一样现算 sha、已登记且早于提交）；v1 登记表与 `files` 的键集不变。v2 登记表写 `judge.near_boundary_tiers`（即使为空）⇒ 13。
   - `launch.json` 核对：caplaunch 既不读也不写 ladder（`launch.json` 没有这个键），所以它经登记表（identity 的 `register_sha`）绑定到启动，判定器另核它与本树档位表一致：schema `capkit.ladder/1`；`config` = `arm_config[臂]`；`levels[]` 的 `up_seq` 是 `up` 档、`down_seq` 是 `down` 档或 null；`near`、`cal`、`holds`、`slope` 是带整数 `seq` 的对象列表（键必须在，可为空）；`baseline_seq` 为整数或 null；每个被点名的档都是负载档、不被点名两次；每个负载档恰被一个角色点名（**另核 `tier_table_sha256` 等于登记的档位表、`arm` 等于本启动的臂：勘误 calib-v2-A9 第 7 条**）。任一不符 ⇒ 13（`item` 为 `ladder.json`）。以后若要 caplaunch 把 ladder 记进 `launch.json`，届时再加逐字比较。
   - regime：`levels`、`near` ⇒ `near_boundary`；`cal`、`baseline_seq` ⇒ `subsat`；`holds`、`slope` ⇒ `near_boundary`，且它们的 X 结论不进级合成与 b（由 P2-06 按 `ctx.tier_role` 处理，X 统计量照算，DESIGN-r7a §2.11.1 r7a 注）。`holds` / `slope` 取 `near_boundary` 是本步定的：§2.3 的 subsat 阈值只用于基线档与亚饱和档。v2 登记表的臂没有 ladder ⇒ 全部负载档 `subsat`。
5. **`inline_thresholds`**：有 `arm_config` 的登记表 `inline_thresholds` 非空 ⇒ 13，`detail` 为 `inline thresholds are not a calibration file (CONSTRAINTS 4.2; calib-v2 r7)`；缺省或为空照常。v1 登记表照旧（仅 E0）。
6. **REGISTRY 种类**（§9.3）新增 `errata`、`ladder`、`e0_params`、`e0_summary`、`d14_result`、`e0_list`、`job_plan`、`node_exclusions`、`node_list`。
7. **过渡状态（SB-A 内）**：X（`symptoms`）、C13 第二半（`checks`）、I1（`localize`）仍读 `Bands` 与 pilot/1 的 `d14_softirq` / `i1_calibration`。pilot/1 被拒之后，只登记 v2 源的启动在 P2-04 / P2-05 之前 X 与 C13 第二半都 NOT_RUN；I1 的 `near_one_core_min` 与 D14 要到 P2-13（SB-C）改读 `nullz.i1_near()` / `d14()` 才回来。v1 登记表的 `inline_thresholds`（E0）照旧给 X 带。（X 与 C13 第二半已改读 nullz，末句已不成立，见勘误 calib-v2-A3 第 1、5 条。）
8. **样本与测试**：`samples/make_samples.py` 新增选项 `calib_v2`（d14_result + bootstrap 或 thresholds 的 pilot/2，代替 pilot/1）、`v2`（`arm_config`、`arm_threshold_config`、按臂 ladder、`near_boundary_tiers`）、`calibration_entries` / `calibration_files`（逐字照抄别的登记表的标定项与文件）、`arm`、`extra_arms`、`inline`、`near_boundary_tiers`；默认样本没有改，仍登记 pilot/1（缺省已改为 v2 源，见勘误 calib-v2-A3 第 9 条），所以自测样本与依赖它们的测试在本步得 13（第 7 条：要等 X 与 C13 改读 nullz 后才能把默认样本换成 v2 源；`input_calib_invalid`、`input_identity_mismatch` 的 13 现在先来自 pilot/1 被拒，换源后要复核它们原意的检查）。`samples/calibv2_real/` 放三份真实片段（E1-0 http 登记表、E0 第一轮 `pilot.json`、E0 第一轮登记表，来源与 sha 见其 `SOURCES.md`）。测试：`tests/judge/test_calibv2_sources.py`（第 17 条、第 25 条、第 34 条判定器一侧、第 22 条的 `inline_thresholds` 项、bootstrap 样本），`tests/judge/test_calib.py`（v1 接受改为拒收），`tests/capcommon/test_ledger_cli_runtree.py`（新种类）。实测与变异记录在 `_progress/calibv2-a.md`。

#### calib-v2-A3：X 的 z_d 判据与无阈值照记、C13 上限与 C13-MEAN、档 CONTRA、levels[] / boundary / t7、保持与斜率档照算 X 统计量（2026-11-15）

1. **X2、X3 用 z_d 对阈值源比较**（DESIGN-r7a §2.2、§2.9、P2-04；取代 §10.2 X 的「T1 对噪声带」、勘误 B2-2A-2 中 X 读本 regime 噪声带的部分，以及勘误 calib-v2-A2 第 7 条中 X 的过渡状态）：每条逐秒序列算 `stats.zd`（计数族 `inflight_derived`、`backlog.*` 带量子 1），阈值取 `nullz.get(键, arm_threshold_config[臂], 档的 regime)`，比较用 `nullz.compare`：thresholds 的键 `z > thr_hi` 为 above、`z < thr_lo` 为 below；bootstrap 滞后族 `z ≥ z` 且 `D ≥ min_D_ns` 为 above（不会 below）；bootstrap 计数族 `z > z` 为 above、`z < −z` 为 below。三种情形该键 NOT_RUN（`reason` 写明）：z_d 算不出（差分不足、尺度退化、半窗为空、没有序列）、没有阈值源、键 `uncalibrated`。`_combine` 不变。
   - 没有 `arm_config` 的登记表（v1）没有配置，只读与配置无关的值：bootstrap 的按族值、thresholds 的 `ratio`；`null_z` 键查不到（该键 NOT_RUN）。`inputs.json` 的 `threshold_lookups` 把配置记成 `-`。
   - 噪声带（`Bands`）不再进入 X：v1 登记表的 `inline_thresholds` 照旧可以登记（仅 E0），但对 X 不再起作用（取代勘误 calib-v2-A2 第 7 条末句）。`Bands` 只剩 C1、C2 与 I（I 改读 nullz 在 P2-13）使用。
2. **X1**：每个负载角色记 `|norm − 1|`，比较用 `nullz.compare(norm − 1, 项, planned_requests=本档窗口内计划的请求数)`；bootstrap 的容差 = max(`min_per_request` / 计划请求数, `tol`)，thresholds 的 `ratio` 用 `abs_tol`。条目新增 `abs_dev`、`planned_requests`、`threshold`。
3. **无阈值照记**：`verdicts.json` 每个被判档的 `stats` 记原单位读数，新增 `zstats` 记 z_d（同样是「键 → 列表」），二者都不看阈值在不在：X2、X3 的 D（`stats.t1_trend` 算得出就记）与 z_d（`stats.zd` 算得出就记），X1 的 `|norm − 1|`（基线档自己的比值按构造为 1，不记），C13 的每窗读数（第 5 条）。键 `T2:delivery_ratio_norm:<角色>` 在 `stats` 里的值由 `norm` 改为 `|norm − 1|`：v1 构建器（`capjudge.pilot`、`capjudge.calib`）照旧读 `stats`，它们写出的这条带随之变成 `|norm − 1|` 的带（v1 文件已不是阈值源）。每档另加 `ladder_role`（该档在 `ladder.json` 里的角色；没有 ladder 为 null）。实现：`capjudge.symptoms.Record`（字典本身是 `stats`，`.z` 是 `zstats`）。
4. **诊断量与计数下限函数**：X2 每条序列另记 `level_switch_ratio`（逐秒中位数的 SD ÷ 各秒中位数抽样 SE 的均方根，`stats.level_switch_ratio`，定义同 `calib-v2/drift/excess.py`），只报告、不进判定（DESIGN-r7a §2.13）；算不出时为 null。`stats.count_floor_z(L)`（§2.9 的 z_1q，L = 60 时 10.705）与 `stats.floor_count_thresholds`（经验阈值低于下限取下限、标 `floor_bound`）供 P2-14 的 `pilot2 thresholds` 用，本步只有测试调用。
5. **C13 第二半与 C13-MEAN**（P2-05；取代 §10.2 C13 的「剩余超出噪声带 ⇒ 红」与「`pilot.json` 还没有 D14 结果时这一半记 `NOT_RUN`」，以及勘误 calib-v2-A2 第 7 条中 C13 的过渡状态）：D14 读 `nullz.d14()`（登记的 `d14_result`），上限读 `nullz.cap("T3:foreign_busy_frac")`（pilot/2 的 `c13_cap`）。没有 D14 ⇒ 不算读数，第二半 NOT_RUN；有 D14 ⇒ 每个 `under_test` 角色算得出的读数都写进 `stats`（核下线、缺 sidecar 等算不出的照旧 NOT_RUN），有上限时读数 `>` 每窗上限 ⇒ RED，没有上限 ⇒ 第二半 NOT_RUN（V0-cal）。
   - 新的启动级检查 `C13-MEAN`（只在 `judge.checks` 含 C13 时运行）：逐档循环之后，对每个 `under_test` 角色取全部被判负载档（含保持档、斜率档）的读数均值，`>` 启动均值上限 ⇒ RED，启动 VOID。缺上限或缺 D14 ⇒ NOT_RUN（均值照报）；某角色没有任何读数 ⇒ NOT_RUN。结果进 `launch_checks`、`checks.tsv`（`tier_seq` 为 `launch`），`verdicts.json` 顶层新增 `c13_mean: {<角色>: {mean, n, cap}}`。§10.3 码 10 的「启动级检查红」包括它。
6. **档 CONTRA**（P2-06 ①；取代 §10.2「X 的主量与对账量方向矛盾 → `VOID`」，§10.2 输出表 `verdicts.json` 的 `verdict` 取值随之多一个 `CONTRA`）：矛盾的档判 `CONTRA`，在判定顺序里占原来 VOID 的位置（P 的 RED / NOT_RUN 仍在它之前），不使档或启动 VOID。随之调整、使旧读方语义不变的三处（**读方清单已由勘误 calib-v2-A8 第 2 条补全：`intervene` 的反向剂量启动也把 CONTRA 档当不可用**）：`grade.first_non_stable` 像原来跳过这种 VOID 档一样跳过 CONTRA 档（列进 `skipped`）；v1 构建器 `pilot.build` 与 `calib` 的噪声带构建（`NOISE_DROP`）照旧不收 CONTRA 档；`pilot.select_tiers` 的缺省（`d14` 与将来的 `bootstrap` 取档用）只丢 RED 检查与 VOID 档，所以收 CONTRA 档（取档只看确定性检查，DESIGN-r7a §2.11.3）。
7. **`levels[]`、`boundary`、启动结论、`t7`、`batch`**（P2-06 ②～⑥；有 `ladder.json` 时写 `levels`、`boundary`、`t7`、`criterion_bias` 四个顶层键，没有时都不写）：
   - `levels[]`：`ladder.json` 每个级一条 `{level, rho, load_value, up_seq, down_seq, verdict, keys_up, keys_down, flags}`。`down_seq` 为 null（`passes: 1`）⇒ 级的 verdict = 该档 verdict。两遍时按 DESIGN-r7a §2.4 自上而下取第一条：任一遍 VOID ⇒ VOID；两遍都是 CONTRA 且有同一种矛盾（`X.contradiction` 里的同一条文字）⇒ CONTRA（`flags` 记 `contradiction:<矛盾>`）；否则 CONTRA 的那遍按「X 不成立」参与合成，`flags` 记 `contradiction_single:up|down`；任一遍 NOT_RUN ⇒ NOT_RUN；两遍 X 都成立（档 UNSTABLE 或 UNDETERMINED）且在同一个 X2 键上越 `thr_hi`、或两遍都有过载型失败 ⇒ UNSTABLE，其中一遍是 UNDETERMINED ⇒ UNDETERMINED；两遍都不成立 ⇒ STABLE；其余 AMBIGUOUS（两遍都成立而键不同时 `flags` 记 `keys_differ`）。`keys_up` / `keys_down` 是该遍越 `thr_hi` 的 X2 键。
   - `boundary`：`s_star`、`u_star`（`{level, rho, load_value}` 或 null）、`b = (s* + u*)/2`（负载单位）、`b_robust`（u* 换成「它与下一级都 UNSTABLE」的最低级；没有这种连续两级 ⇒ null）、`non_monotone`（s* 高于 u*）、`ambiguous_inside`、`contra_inside`、`notrun_inside`、`undetermined_inside`、`n_ambiguous_below_b`（负载严格低于 b 的 AMBIGUOUS 级数）、`n_contra_levels`、`censored`。inside 按级号严格落在 min(s*, u*) 与 max(s*, u*) 之间。AMBIGUOUS、CONTRA 级不进 s*、u*，落在 inside 只标记、b 不变；NOT_RUN、UNDETERMINED 级落在 inside ⇒ `censored`。本步另加 `void_inside`（VOID 级落在 inside 也按 `censored` 处置；VOID 级本已使启动 VOID）与 `censored_reasons`（`no STABLE level`、`no UNSTABLE level` 或上面三个 inside 标记）。没有 STABLE 或没有 UNSTABLE ⇒ b 为 null 且 `censored`；inside 标记使 `censored` 时 b 照算照报。
   - 启动结论（取代 §10.3 码 0、10、11 三行的判法）：VOID ⇐ 启动级前提 RED（含 GPU-IDLE、`C13-MEAN`）或任一被判档 VOID；否则 NOT_RUN ⇐ 启动级前提 NOT_RUN（含 `C13-MEAN`）、没有被判档，或任一被判档 NOT_RUN 而它不是 `ladder.json` 级的上行 / 下行档；否则 VALID。档或级的 CONTRA、级档的 NOT_RUN、`boundary` 的各标记都不改变它。没有 `ladder.json` 时每个档都算非阶梯档，与原判法相同（只多了 CONTRA 不作废）。
   - `t7`（**只用跑完且 P 为 PASS 的斜率档：勘误 calib-v2-A9 第 6 条**）：对 `ladder.json` 的 `slope` 档，取各档 `stats` 里 `T1:req_first_lag_p50_ns:<角色>` 的 D，`g = 2·D/L`（L 为该档窗口，ns），λ = 该档 `load_value`，`stats.t7_fit` 拟合得 `{role, mu, beta, n, points}`；没有斜率档读数、点不够或斜率不为正 ⇒ `{status: NOT_RUN, reason}`；有不止一个请求角色带这个读数 ⇒ NOT_RUN 并在 `by_role` 列出各角色的拟合。`criterion_bias = (b − mu)/mu`，任一缺失为 null。
   - `batch` 改为 `"calib-v2"`（取代勘误 B2-2E-1 第 1 条的「`batch` 改为 2」）。
   - 保持档、斜率档（P2-06 ⑦）与其余档一样调用 `evaluate_x`、写 `stats` / `zstats`、得到档 verdict；只是它们不是级，不进 `levels[]` 与 `boundary`；它们的 VOID / NOT_RUN 照常影响启动结论。
8. **`ladder.json` 的级字段核对**（补勘误 calib-v2-A2 第 4 条）：`levels[]` 每项另须有整数 `level`（互不相同）与数值 `rho`、`load_value`；`load_value` 必须等于其 `up_seq`、`down_seq` 档在档位表里的 `load_value`（相对误差 1e-9）。不符 ⇒ 13（`item` 为 `ladder.json`）。（**另核 `tier_table_sha256` 与 `arm`：勘误 calib-v2-A9 第 7 条**。）
9. **样本**：`samples/make_samples.py` 的缺省标定改为 v2 源（`d14_result` + bootstrap pilot/2，即选项 `calib_v2` 的缺省）；新选项 `calib_v1` 恢复登记 v1 `pilot.json`（判定器拒收），`calib_v2` 新增 `pilot: false`（只登记 `d14_result`，V0-cal 的形态），`c13_foreign_rate: {"<档号>" 或 "all": 比例}`（在 nullsse 的核上按比例加外来忙碌 tick）。`samples/calibv2_real/j5940628_l1_t4_requests.json` 是真实片段（来源与 sha 见同目录 `SOURCES.md`）。生成器里只改了 `x_contradiction` 的期望（10、档 4 VOID → 0、档 4 `CONTRA`）；样本 `d14_result` 的规则文件写在 `register/d14_rule.sample.json`（不占测试用的 `register/d14_rule.json`）。按新缺省生成的样本自测，只有 I 的 5 个样本（P2-13 之前 I1 读不到 `i1_calibration`）与 `regime_nb_good`、`regime_nb_missing`（登记 `near_boundary_noise`，勘误 calib-v2-A2 起 13）不符期望。已提交的样本目录（§10.4）本步没有重新生成，仍登记 pilot/1、得 13；重新生成与这 7 个样本的改写留给 DESIGN-r7a P2-10 与 SB-C 的 P2-13。（已重新生成、`regime_nb` 两个改期望 13，见勘误 calib-v2-A5 第 6 条；I 的 5 个仍待 P2-13。）
10. **测试**：`tests/judge/test_judge_levels.py`（DESIGN-r7a §3.5 第 6、9、10、14、15、16、18、19 条）。与测试表的出入：第 18 条的 e0w60 真实树在原处只读使用（`J5940628-r0-L2_e0b-s1-single-w60`，58 MB，没有复制；本机没有该树时跳过），v2 源在内存里加给判定器，而没有写一份 v2 登记表——`register_sha` 写在树里每个文件的 identity 中，换登记表就要改写整棵树；第 9 条的整树部分用合成样本的两个负载档（r20、r40）当斜率档，μ 的精度另在纯函数上测；第 15 条的「全 0 标定 ⇒ 10.705 且 `floor_bound`」用第 4 条的下限函数测（构建器属 P2-14）。实测与变异记录在 `_progress/calibv2-a.md`。

#### calib-v2-A4：laddergen 与 ladder.json、按 e_c 核对装箱、job_plan.tsv、BUDGET.tsv 与预算闸（按不同 item_id 计数）、并发 / 盘 / 维护三道提交门、--expect-node、exclude --all-but / --also-exclude、node_exclusions、node_list 与 pin-next、remote.py 白名单加 sinfo（2026-11-15）

1. **`python -m caplaunch.laddergen --kind {ladder,refine,bracket,hold,null,pc,e2e,cal}`**（DESIGN-r7a P2-07、§3.4、§4.1、§4.2；`tiergen.py`、`capcommon/tiers.py` 未改；**kind `c13pc`、`--request-timeout-ns` 与 `tiergen` 的登记 null 见勘误 calib-v2-A7 第 3、4 条**）：一次启动生成三个新文件（`open(..., "x")`，任一已存在就一个都不写）：`<名>.tiers.template.json`、`<名>.values.json`（交给 `caplaunch.tiergen` 展开；本工具先用 `tiergen.expand` 展开、校验一遍）与 `<名>.ladder.json`（`capkit.ladder/1`）。
   - **轴与档**：R 轴（`load_key = rate_rps`），`streams_c` 一律 null；顶层的到达、超时、门禁、注入与 `common` 取自 `--base-template`（场景的 `capkit.tiers_template/1`），它们的 TBD 值取自 `--base-values`。预热 10 s（0.3·中心），每个负载档之后一个清空档。负载档 `ramp_ns = 4 s + settle`、窗口 60 s、tail 2 s；斜率档（ρ 1.05、1.10、1.20）settle 4 s；保持档 ramp 0、窗口 600 s。各 kind 的档序：`ladder` 基线 + K 级（`ρ_k = 1 − span_half + h·(k + u)`）上行、下行 + 3 斜率档；`refine` 基线 + 21 级 0.95～1.05（h = 0.5%）单遍 + 3 斜率档；`bracket` 基线 + `--rho-first`·`--ratio`^k 共 `--n-levels` 级单遍；`hold` 基线 + 0.95、0.98、0.99 三个保持档（顺序 = 按种子 u 取的排列）+ 3 斜率档；`null` 基线 + near 上行 0.90、0.95、ρ_top×3、下行反序 + 亚饱和 0.3（下行）+ 3 斜率档；`pc` 基线 + 过载 1.5 + 两个亚饱和 0.3（整次启动 settle 取 `--settle-s`，V0 用 4）；`e2e` 基线 + 级 {0.5, 1.0, 1.5} + h·u（h = 0.25%）两遍 + 3 斜率档；`cal` 基线 + 0.3、0.4、0.5、0.6 上行、再反序下行（8 档）。
   - **`ladder.json` 的角色**：`levels`（`{level, rho, load_value, up_seq, down_seq}`，单遍时 `down_seq = null`）、`near`、`cal`（`{rho, seq, pass}`）、`holds`（`{rho, seq, hold_ns}`）、`slope`（`{rho, seq}`）、`baseline_seq`；`pc` 的过载档列在 `slope`、两个亚饱和档列在 `cal`，`null` 的亚饱和档也列在 `cal`（判定器按 `cal` 取亚饱和 regime；`pilot2 bootstrap` 只读它的输入清单里的 V0-cal 树，这两种启动不进去）。另记 `kind`、`phase`、`config`、`arm`、`c_lad`（中心值）、`center_source`、`axis`、`h`、`K`、`span_half`、`passes`、`u`、`settle_ns`、`ramp_base_ns`、`seed`（`{base, block_id, slot, sha}` 或 null）、`drain_rule`（公式、`e_c`、逐个清空档的秒数）、`packing`（下面第 2 点的读数）、`template_sha256`、`values_sha256`，以及 `tier_table_sha256`（= `tiergen` 用同一模板与取值写出的档位表的 sha256，测试核对两者相等）。判定器对它的核对见勘误 calib-v2-A2 第 4 条、calib-v2-A3 第 8 条。
   - **中心**：`--center <文件> --center-key {c1, c_hat, c_b, c_n, c_lad}`，文件必须在 REGISTRY 里登记为 kind `e0_params`、路径与 sha256 都一致，`schema = capkit.e0_params/1`，`stage` 与键对应（`c1`→`bracket`、`c_hat`→`capacity`、`c_b`→`recenter`、`c_n`→`settle`、`c_lad`→`ladder`），值取 `by_config[<--config>][<键>]`（> 0）；`center_source = {relpath, sha256, key}`。**SB-A-5 / SB-C 写 `e0_params` 时须按这个形状给出各配置的中心。** `--center-literal` 只给 kind `ladder`、`pc`、`e2e`、`cal`（V0、V0-cal 的 ĉ0）与 `bracket`（V1-a1 的先验，`e0_params` 里没有对应的键），记作 `center_source = {"literal": 值}`；其余 kind 用它 ⇒ 失败。
   - **种子**：kind `ladder`、`e2e`、`hold` 必须给 `--seed-base`、`--block-id`、`--slot`，`u = int(sha256(base|config|block_id|slot)[:13], 16) / 16**13`（§3.4）；其余 kind `u = null`。
   - **清空槽**（§4.1）：清空档 = `max(60, ceil10(1.5·B + 30))` s，`B = (ρ·(1 + e_c) − 1)+ × 前一档长度`；`bracket` 中 `ρ·(1 + e_c) > 1` 的档之后取 180 s。`e_c` 由 `--phase` 定：V0cal、V0 0.30，V1b、V2n 0.02，V2b 0.01；V1a1、V1a2、V2c 没有登记值，必须给 `--e-c`；有登记值的期再给不同的 `--e-c` ⇒ 失败。注：§4.2 的 `s1-pc` 行写「槽 90 s」，那是按名义 ρ 算的数；按 §4.1 的 V0 `e_c = 0.30`，过载 1.5 档之后的槽是 130 s（更长，排空更充分），本工具按 §4.1 生成。
2. **装箱核对**（P2-07 r7，CR6-M1）：表长 T（档位表末档结束时刻）按 `--launches-per-job n` 次同长启动核两条，任一不满足 ⇒ 退出码 3、不写文件：§4.1 作业条件 `n·(T + 6.5) + job_overhead_s_max ≤ 14340`（`--job-overhead-s-max` 缺省 763.5 = 1.5 × 509，小于它 ⇒ 失败）；`capjobs.submit.time_budget`（同一函数）在 `budget_s = ceil10(T + 6.5 + 60)`、`--kill-grace-s`（缺省 90）、`--margin-s`（60）、`--idle-window-s`（30）、`--slurm-view-wait-s`（300）下 ≤ `--time-limit-s`（缺省 14400，不得更大）。实测（MEASURED，本机生成）：V0 满长度计时行（K = 24、S = 90 s、e_c 0.30、`--center-literal 19.09`）表长 13 264 s 通过，K = 26 表长 14 308 s 失败；V0-cal 一次启动 1474 s、4 次一作业 `time_budget` 7610 s（与 `r7/out/plan_r7.json` 相同）。
3. **`--plan`**：§2.7 / §4.1 的 r4 保守模型（每级都按最高 ρ、V2-b 的 e_c 0.01、清空槽积压按 60 s 超时封顶），输出 `S_fit_by_K`（K = 12～32）、给了 `--settle-s` 时的 `K_fit`，再给 `--K`（可加 `--span-half`）时的启动长度、`per_job` 与是否装得下。`S_fit_by_K` 与 `r4/out/plan_r4.json` 逐项相等（测试用其摘录 `samples/calibv2_real/plan_r4_S_fit_by_K.json`）。
4. **`job_plan.tsv`**（`capcommon.jobplan`，§3.2）：列 `order item_id item_class arms launches block_id slot config_counts pin`；`item_class` 只有 `planned`、`surplus`、`post_gate:{shift, backstop, coverage, blind_topup, shortfall}`，**没有任何替换 / 重投类**；`order` 从 1 起按文件顺序连续，同一 `order` 只许一个条目或同一区组的 slot 1、2 两个条目；`surplus` 排在全部 `planned` 之后；门后计划文件只含一种门后规则；`config_counts` 是 `{配置: 启动数}`，和 = `launches`；`arms` 是该作业各启动的臂（首次出现顺序，= `capjobs.launchplan.arms`）。生成：`laddergen --job-plan <spec> --out <tsv>`，spec 为 `capkit.job_plan_spec/1`（`phase_id`、`seed_base`、`launches_per_job`、`pin`、`configs[{config, arm, planned, surplus}]`、`blocks[{comparison, configs[[配置, 臂] ×2], planned, surplus}]`）：每类（计划、超额）的启动按 `sha256(seed_base|类|配置|臂|序号)` 混装后按 `launches_per_job` 切成作业（**V0-cal 改为每作业每配置各一次：勘误 calib-v2-A7 第 2 条**），区组每个一对 `pin = true` 的单启动条目（`block_id = <comparison>-bNN`，两个 slot 的先后按 `u(seed_base, comparison, block_id, 0) < 0.5`，臂名 `<臂>.<block_id>.<slot>`——DESIGN 的 `<配置>@<block_id>.<slot>` 含 `@`，不合 `capcommon.identity.ARM_RE`），再把作业与区组按种子排序、先计划后超额编 `order`。`--plan-from <门产物>`：产物必须在 REGISTRY 登记为 kind `e0_params`、路径与 sha256 一致、`schema = capkit.e0_params/1`，`items[{item_class, config, arm, launches, pin}]` 只能是同一种 `post_gate:*` 类；`replace`、`post_gate:replace`、`planned`、`surplus` 都 ⇒ 失败、不写文件。（**`--plan-from` 的门产物一律要整数 `round` ≥ 1，单启动条目号带轮次：勘误 calib-v2-A8 第 1 条**。）（**门产物带 `cap.launches_per_job` 时 `--launches-per-job` 必须与它相等：勘误 calib-v2-A9 第 2 条**。）
5. **`BUDGET.tsv` 与提交门**（P2-08b；`<W>/BUDGET.tsv`，`capcommon.ledger.BUDGET_COLUMNS = event_at event budget_id item_id item_class plan_sha256 register_sha256 plan_id job_id`，只追加；`budget_append` 拒收 `event ∉ {RESERVED, JOB_ID}`、`RESERVED` 带作业号、`JOB_ID` 不带、非法类或 sha）。**预算计数 = 该 `budget_id` 下有 `RESERVED` 行的不同 `item_id` 个数**（`ledger.budget_count`）。`capjobs.submit submit` 对登记表有 `expected` 或 `arm_config` 段的提交（`capjobs.budget.gates`；**适用范围已由勘误 calib-v2-A6 第 1 条收窄：只有 v1 两键的 `expected` 不受门控**），在全部既有检查之后、写任何东西之前依次核（**真提交时脏树检查改在门之前、门到 `RESERVED` 行持锁：勘误 calib-v2-A8 第 7、11 条**）：① `expected` 有 `budget_id`、`job_budget`、`plan {relpath, sha256}`、`disk {fs_path（绝对路径）, floor_bytes, per_job_bytes}`，计划文件 sha 与之相等且在 REGISTRY 里登记为 kind `job_plan`、relpath 相同；必须用 `--exclude-file`，且它含最新一版已登记 `node_exclusions` 的全部节点（没有登记过任何一版时此项不查）；② `--plan-item` 必填、在计划里、在 BUDGET.tsv 里还没有 `RESERVED` 行（任何 budget_id），同一计划里 `order` 更小的条目都已有 `RESERVED` 行，`--arms` / `--launches` 等于条目的 `arms` / `launches`，`pin = true` 的条目必须给 `--expect-node` 且 `--exclude-file` 是 `exclude --all-but` 写出的、`all_but` 等于它，`pin = false` 的条目不许给 `--expect-node`；③ 计数 + 1 ≤ `job_budget`；④ `submissions.tsv` 里未 `ENDED` 的计划数（排除 `local-*` 作业；结果不明的也算）+ 同一 `order` 中尚无 `RESERVED` 行的条目数 ≤ 8；⑤ `os.statvfs(fs_path)` 的 `f_bavail × f_frsize` ≥ `floor_bytes + 1.5 × per_job_bytes × (④ 的两数之和)`；⑥ 经 ssh 读 `scontrol show reservation`，有 reservation 时再按 `--partition` 逐分区读 `sinfo -N -h -o %N -p <分区>` 得节点集，对节点集与之有交集、`Nodes = ALL` / `Flags` 含 `ALL_NODES`、或 `PartitionName` 是 `--partition` 之一的 reservation，若 `now + --time + 600 s > StartTime` 且 `now < EndTime` ⇒ 推迟；ssh 失败、输出为空或解析不了、sinfo 失败同样推迟（**解析改为按记录、按键取值：勘误 calib-v2-A8 第 6 条**）。`scontrol` 打印的是控制器本地时间，没有时区，所以门要求 `--cluster-utc-offset ±HH:MM`（必填，格式不对 ⇒ 3），`now` 按它换成控制器本地时间（`EndTime` 为 `Unknown` / `None` / `INFINITE` 视为无穷）。每次读到的原文、换算后的时刻、判定与冲突写一行 JSON 追加到 `<W>/02_e0/E0/cluster/reservations.log`（**不跟随软链：勘误 calib-v2-A9 第 10 条**）。退出码：①～④ 3，⑤ 5，⑥ 4（**按退出码与报错文字区分结局：勘误 calib-v2-A9 第 11 条的对照表**）；任一不过都不写 `submissions.tsv` 与 `BUDGET.tsv`（⑥ 只写 reservations.log）。全部通过后：快照 → `BUDGET.tsv` 一行 `RESERVED`（带本次 `plan_id`）→ 原有的 `PLANNED` / sbatch / `SUBMITTED` 流程（`submissions.tsv` 列不变）→ 拿到作业号后一行 `JOB_ID`。（**写入顺序已改为先 `PLANNED` 后 `RESERVED`：勘误 calib-v2-A9 第 5 条**。）sbatch 结果不明（`UNKNOWN_SUBMIT`）时只有 `RESERVED`，该条目已被占用；`reconcile-submit --job-id` 落 `SUBMITTED` 时，若该计划有 `RESERVED` 行，另追加一行 `JOB_ID`（不改变计数）。`--dry-run` 跑完 ①～⑥ 但什么都不写（包括 reservations.log），输出里有 `budget_gate`。`submit` 没有任何「替换」「重投」参数。没有 `expected` / `arm_config` 的登记表照旧走原流程，给了 `--plan-item` ⇒ 3（**见勘误 calib-v2-A6 第 1 条**：没有 `arm_config`、`expected` 只含 v1 两键的登记表同样照旧）。
6. **`--expect-node`**（P2-08a ①）：`submit --expect-node <短主机名>`（**名字空间是 Slurm 节点名，不是主机名：勘误 calib-v2-A8 第 5 条**） 写进 `job.env` 的 `CAPJOBS_EXPECT_NODE`（未给为空串）；作业里 `capkit_inner.sh` 把它交给 `capjobs.envfiles job --expect-node`，该检查在快照核对之外断言 `socket.gethostname().split(".")[0]` 等于它（**已由勘误 calib-v2-A8 第 5 条改为比对 Slurm 节点名 `--node`；主机名只作诊断**），不等 ⇒ 检查 FAIL、退出 6 ⇒ 作业级 `ENV_INVALID`（原因「job check failed (snapshot or expected node <名>)」），在画像、门禁与任何启动之前。`job.json` 新增 `expect_node`、`hostname_short`、`expect_node_ok`；不符时 `snapshot_problems` 末尾写明两台主机名（**改为节点名、主机名与期望节点：勘误 calib-v2-A8 第 5 条**）。
7. **`exclude --all-but` / `--also-exclude`**（P2-08a ②③）：`--all-but <目标>` 与 `--cdb-file`、`--live` 三选一，另需 `--node-list <已登记的 node_list>`、`--workdir`、`--login-host`、`--ssh-timeout-s`；目标必须在 `node_list` 里，且出现在 `--partition` 各分区现查的 `sinfo -N -h -o %N -p <分区>` 结果里；任一 sinfo 失败 ⇒ 失败、不写文件。输出仍是 `capkit.cdb_exclude/1`（`submit --exclude-file` 照读），`exclude` = 现查全集 − 目标，另记 `all_but`、`partition_nodes`、`node_list {relpath, sha256}`、各次 sinfo 的命令、退出码与输出 sha256。`--also-exclude <node_exclusions>`（可与三种来源合用，需 `--workdir`）：文件必须已登记（kind `node_exclusions`、路径与 sha 一致），其节点并入 `exclude`，输出记 `also_exclude {relpath, sha256, version, nodes}`；`--all-but` 的目标在清单里 ⇒ 失败。`--main-profile-spec` 改为只在 `--cdb-file` / `--live` 时必填。
8. **`node_exclusions` 与 `node_list` 文件**（§3.2）：两者都要有 `schema` 键（`capkit.node_exclusions/1`：`{schema, version, nodes:[{node, reason, evidence}], previous:{relpath, sha256}|null}`，节点不重复、`reason` 非空；`capkit.node_list/1`：`{schema, taken_at, source, nodes}`，`nodes` 按名字升序、不重复、非空）。「最新一版」= REGISTRY 里最后登记的那一行 kind `node_exclusions`。
9. **`capjobs.submit pin-next`**（P2-08a ⑤，§4.5）：参数 `--workdir --node-list --exclusions --key <比较名> --seed-base --plan <该比较的全部 job_plan，可多次> --item <即将提交的钉节点条目> --partition --log <W>/02_e0/E0/cluster/pin_next.log --login-host --ssh-timeout-s`（`--item`、`--partition`、`--seed-base` 是 DESIGN 的命令行之外加的：`--item` 用来认出区组的第二个 slot 与重复提交，`--partition` 是 sinfo 现查的分区，`--seed-base` 是轮转的种子）。`--exclusions` 必须是最新一版；各 `--plan` 必须已登记为 kind `job_plan`，`item_id` 跨文件不重复；`--item` 必须是其中 `pin = true` 的条目且在 BUDGET.tsv 里还没有 `RESERVED` 行。轮转序列 = `node_list` 按 `sha256(seed_base|key|节点)` 升序。区组的另一 slot 已有 `RESERVED` 行 ⇒ 直接取它的节点；否则「上一个」= BUDGET.tsv 中这些计划的钉节点条目里最后一行 `RESERVED` 的 `plan_id`（**只取属于 `--key` 这个比较的条目，`--item` 属于别的比较、或同一比较换了种子 / `node_list` ⇒ 3：勘误 calib-v2-A9 第 1 条**） → `submissions.tsv` 的快照 → `job.env` 的 `CAPJOBS_EXPECT_NODE`（作业已写出的 `job.json` 若有 `expect_node`，必须与之相同），从它的下一位开始、绕回一次，取第一台不在排除清单里、出现在 `sinfo -N -h -o "%N %t" -p <分区>` 现查结果里、状态（去掉 `*~#!%$@^-` 后缀）不以 `down`、`drain`、`drng`、`fail` 开头的节点（**已由勘误 calib-v2-A8 第 3 条改为白名单**）；第一次从序列第 1 位开始。每次调用一行 JSON 追加进 `--log`（比较名、条目、种子、两个清单的 relpath / sha、计划文件、上一个、选中、跳过的节点与原因、规则 `rotation` / `block partner`）；整圈都不可用 ⇒ 照样写日志，退出码 3。只读台账、快照与 sinfo，不读任何判定结果。
10. **`capjobs/remote.py`**（P2-08a ④，CR6-M2）：`ALLOWED_REMOTE` 增加 `sinfo`（只读，不限参数），仍经 `remote.ssh_argv` / `run_remote`；`scancel` 仍不在白名单。`tests/jobs/conftest.py` 的禁止执行清单同步加 `sinfo`。
11. **测试**：`tests/launch/test_laddergen.py`（DESIGN-r7a §3.5 第 12 条、第 33 条的生成器部分）、`tests/jobs/test_budget_gates.py`（第 24 条的预算部分、第 33 条、第 36 条）、`tests/jobs/test_pinning.py`（第 24 条的 `--all-but` / `--expect-node` 部分、第 37 条、第 39 条）、`tests/jobs/test_scripts.py` 两条（作业脚本把 `--expect-node` 交给 job 检查；主机名不符 ⇒ ENV_INVALID（**节点名不符：勘误 calib-v2-A8 第 5 条**） 且不起画像、门禁、启动）、`tests/capcommon/test_budget_ledger.py`。实测与变异记录在 `_progress/calibv2-a.md`。

#### calib-v2-A5：pilot2（d14、shortfall、bootstrap 与入选 _select）、bootstrap 规则原文、自测样本（2026-11-15）

1. **`python -m capjudge.pilot2 <子命令>` 的共同框架**（DESIGN-r7a P2-09、§3.3；`capjudge.pilot` v1 不动）。本步实现 `d14`、`bootstrap`、`shortfall`；`bracket`、`capacity`、`recenter`、`settle`、`thresholds`、`drift`、`summary` 属 SB-C / SB-D，用同一框架。
   - **列表文件**（`--list`）：每行 `<树>\t<档序号（逗号分隔）或 ALL>\t<regime>`，regime ∈ `subsat`、`near`；`#` 开头与空行忽略；树或档列含 `* ? [`、列数不是 3、档号不是整数或重复、同一棵树出现两次、文件里没有任何树（`shortfall` 除外：全部启动都丢失时入选清单可以为空）⇒ 退出码 3。
   - **每棵树**：`<W>/JUDGED.tsv` 里有 `tree_relpath` 等于它的行，且最后一行的 `verdicts_sha256` 等于 `judge/verdicts.json` 现算的 sha；`judge/inputs.json` 的 `code_sha` 非空；全部输入树的判定器 sha 只有一个。否则退出码 3。
   - **产物**：`open(..., "x")` 写新文件，之后写旁边的 `<名>.cmd.txt`（`capjudge.cmdrec`：模块、代码 sha、逐字命令、输入文件及其 sha、退出码）；产物或记录已存在 ⇒ 退出码 3、什么都不写。任何拒收都是退出码 3、不写文件。
   - **供 SB-C 子命令用的核对函数**（本步有单元测试，第 13 条）：`check_windows`（列出的负载档窗口 = 60 s）、`check_ramp`（`ramp_ns = 4 s + S`）、`unique`（环境、判定器 sha 唯一）、`check_v2`（`verdicts.json` 的 `batch` 是 `calib-v2`，否则是 v1 树）、`load_e0_params`（REGISTRY 里 kind `e0_params`、路径与 sha 一致、`stage` 对）、`check_list_matches`。
   - **入选核对**：除 `d14`、`bootstrap` 外，列表文件的树（按行序）必须**恰好等于** `_select` 的 `selected`（多一棵、少一棵、换序 ⇒ 退出码 3，报错列出多出与缺少的前 5 棵）。`d14`、`bootstrap` 只按 `pilot.select_tiers` 取档，不读判定器的 `launch_verdict`。
2. **`_select(plans, W)`**（DESIGN-r7a §4.5、P2-09 r5/r7）：输入一份或多份已登记（kind `job_plan`）的 `job_plan.tsv`；含 `planned` / `surplus` 条目的是原段，其余是门后计划，原段排在前面，各组内按给出的顺序。
   - 每个条目：BUDGET.tsv 的 `RESERVED` 行（`plan_sha256` 必须等于该计划文件的 sha，否则退出码 3；同一 `item_id` 两行 ⇒ 3）→ `plan_id` → `submissions.tsv` 合并后的行 → 作业号 → JUDGED.tsv 中该作业的各启动（按 `launch_seq`；跨多个 `restart_count` ⇒ 3）→ 各树 `verdicts.json`。每个启动的配置取 `verdicts.json` 的 `arm_config`，必须在条目的 `config_counts` 里且不超过其个数（否则 3）；`config_counts` 中没有判定结果的名额记为不可用。
   - **可用**：作业 `ENDED` 且状态不是 `ENV_INVALID` / `CANCELLED` / `SUBMIT_FAILED`、该启动有判定结果、`launch_verdict ≠ VOID`。（**已结束作业里还没判定的启动树不再算「没有判定结果」，而是退出码 3：勘误 calib-v2-A9 第 4 条**。）不可用的原因：没有 `RESERVED` 行或不在 `submissions.tsv`（未提交）、作业未结束、作业级状态、没有判定结果、`launch_verdict VOID`。只读计划、三本台账与 `verdicts.json` 的 `launch_verdict`、`arm_config`、`batch`（必须是 `calib-v2`），不读 z、b、X。
   - **单位与 N**：区组条目（`block_id = <比较>-bNN`）的单位是比较 `cmp:<比较>`，一个区组算一项：两个 slot 都可用、且两棵树 `raw/dims.json` 的 `node` 相同且已知，才可用；其余条目的单位是配置。N = 原段 `planned` 条目在该单位上的启动数（区组数）。
   - **选择**：每个单位依登记顺序（原段的计划与超额，然后 `post_gate:shortfall` 条目）取前 N 个可用的为 `selected`，其后可用的为 `surplus_unused`；`post_gate:{coverage, blind_topup, backstop, shift}` 的条目可用即全部入选（`n_all_selected` 加 1，即 N 相应增大），不落入「超额、未用」；全部不可用项列进 `unusable`（单位、条目、树、原因、是否作业级）。`by_unit[单位] = {N, n_all_selected, usable（原段与补足条目中的可用数）, shortfall = max(0, N − usable)}`。
   - 本步的取舍：`shift` 按「全部入选」处理，被平移条目的一一对应取代由 P2-11（SB-C）实现；门后「全部入选」类的不可用项不计入缺额（缺额只按原段的 N 算）。
3. **`pilot2 d14`**（P2-09，§2.12）：`--rule` 必须是 REGISTRY 里 kind `d14_rule` 的文件（`capjudge.d14.load_rule`）；每行要给明确的档号（`ALL` ⇒ 3）；先核全部树的判定器 sha 唯一，再逐树 `pilot.select_tiers(drop_verdicts=("VOID",))`、`d14.tree_readings`、`d14.conclude`。`environment = {hw_profile_sha, image_sha, capkit_sha, baseline_sha}` 取自全部输入树（含整树被排除的），不唯一 ⇒ 3。输出 `capkit.d14_result/1`：`rule {relpath, sha256}`、`select: "capjudge.pilot.select_tiers"`、`environment`、`inputs[]`（`select_tiers` 的记录：`tree`、`judge_verdicts_sha256`、`tiers_used`、`tiers_excluded`、`tree_excluded`）、`d14_softirq`（`conclude` 原样输出）。实测（MEASURED，本机读第一轮 15 棵树，约 10 s）：读数与已登记 `pilot.json`（sha `8e823f12…`）的 `d14_softirq.readings` 逐字相等，`irq_softirq_is_experiment = false`，完整树 12 棵；15 棵树里 `launch_verdict = VALID` 的为 0（按它取树就是空输入）。
4. **`pilot2 bootstrap`**（P2-09c，§2.11.3，门 G0c）：`--register` 是 V0-cal 的登记表（REGISTRY kind `register`），其 `expected.bootstrap_rule` 必须逐字等于代码常量 `pilot2.BOOTSTRAP_RULE`（= DESIGN-r7a §2.11.3 的一行规则原文，测试时已与设计稿逐字比对），否则 3；`calibration` 必须恰有一项 `d14_result`（sha 现算一致），产物的 `d14` 即它。列表每行必须是 `<树>\tALL\tsubsat`。（**列表必须恰好是该登记表 `budget_id` 下全部条目的被判树：勘误 calib-v2-A9 第 3 条**。）
   - 每棵树：JUDGED 与 sha 核对、`batch = calib-v2`、启动所用登记表 sha 等于 `--register`（否则 3）；取档 = `verdicts.json` 中 `ladder_role = cal` 的档（没有 ⇒ 3）交给 `pilot.select_tiers`（启动级前提 `LAUNCH-STATE ≠ PASS` 或上游不是 VALID ⇒ 整树不用；有 RED 检查或 VOID 的档不用；NOT_RUN 的档照用）。
   - 读数（只读数值，不读任何 X 结论）：`zstats` 中滞后族键的 |z_d|（`lag_z`）、计数族键的 |z_d|（`count_z`）；`stats` 中滞后族键的 |D|（`lag_D`，ns）、`T2:delivery_ratio_norm:*` 的 |norm − 1|（`ratio`）、`T3:foreign_busy_frac:*` 的每档读数（`c13_window`）；每棵树每个 `T3:foreign_busy_frac:<角色>` 键在所用 cal 档上的读数均值的绝对值（`c13_launch_mean`；**口径见勘误 calib-v2-A8 第 9 条：与判定器 `C13-MEAN` 同一总体，即该启动全部被判负载档**）。「max」取全部树、档、键、配置、角色。
   - 阈值 = max(r6 常量, 3 × 最大值)：`lag_z` 10、`lag_D` 10 000 000 ns、`count_z` 10.705、`ratio` 0.02、`c13_window` 0.10、`c13_launch_mean` 0.05。任一族读数个数为 0 ⇒ 3。环境（四个 sha + `judge_sha`）与 `experiment` 在全部输入树上必须唯一，否则 3。
   - 输出 `capkit.pilot/2`（`stage: bootstrap`）：`experiment`、`environment`、`d14`、`select`、`rule`（规则原文）、`register {relpath, sha256}`、`inputs[]`（`tree`、`judge_verdicts_sha256`、`config`（`arm_config`）、`node`（`raw/dims.json`，没有为 null）、`tiers_used`、`tiers_excluded`、`tree_excluded`）、`x_const`、`ratio_tol`（`min_per_request: 2`）、`c13_cap`，以及 `max_abs[族] = {value, tree, tier_seq（launch_mean 为 null）, key, n_readings, three_x_max, design_floor, bound_by: design_floor | data}`（下限 ≥ 3 × 最大值时为 `design_floor`）。产物被 `nullz` 接受，对任意配置与 regime 返回同一族值（测试第 40 条）。只改各档 X 结论时，产物除各输入的 `judge_verdicts_sha256`（它记的就是被改的文件）外逐字节不变。
5. **`pilot2 shortfall`**（P2-09b，§4.5）：参数 `--workdir --list --register --plan（可多次：原段计划与其门后计划，含已登记的补足计划） --launches-per-job --phase --segment [--job-level-only] --out`。
   - 拒收（3）：段内任一条目未结束（未提交、未 `ENDED`）（**已结束作业里有还没判定的启动树同样拒收：勘误 calib-v2-A9 第 4 条**）；区组段（V2-b 的区组缺额补足本步未实现，见第 8 条；**已由勘误 calib-v2-A5b 第 1 条实现，区组段不再拒收**）；列表文件不等于 `_select`；轮次（= 已有 `post_gate:shortfall` 计划数 + 1；**给出的计划必须齐全：勘误 calib-v2-A8 第 8 条**）> 2（N15）；登记表没有整数 `expected.shortfall_cap_jobs` 或没有 `arm_config`；某个有缺额的配置在原段没有臂。
   - `q̂ = min(0.5, max(0.05, lost/ended))`，`ended` = 段内条目（作业）数，`lost` = 有不可用启动的条目数；补足数 = `stats.shortfall_topup(d, lost, ended)`。`--job-level-only`（V0-cal、V0）：`_select` 把 VOID 启动当可用，`lost` 只数作业级不可用（ENV_INVALID、CANCELLED、SUBMIT_FAILED、未跑完 / 没有判定结果），VOID 不算。
   - **上限**：本轮可用作业数 = `shortfall_cap_jobs` − 已有补足计划的条目数；作业数按 `laddergen --plan-from` 的装法算（钉节点与不钉节点分开、每作业 `--launches-per-job` 次）；按配置名顺序每个配置取能装下的最多启动数，有削减 ⇒ `capped: true`。
   - 臂 = 登记表 `arm_config` 在原段各条目的臂上取逆（一个配置只能有一个臂与一个 `pin` 值，否则 3）；`pin` 沿用原段。
   - 输出 `capkit.e0_params/1`（`stage: shortfall`）：`experiment`、`phase`、`segment`、`round`、`job_level_only`、`rule`、`register`、`plans[{relpath, sha256, post_gate_rule}]`、`q_hat {lost, ended, value}`、`by_config[单位] = {N, n_all_selected, usable, shortfall, topup, listed}`、`cap {shortfall_cap_jobs, prior_jobs, launches_per_job, jobs}`、`capped`、`items[{item_class: post_gate:shortfall, config, arm, launches, pin}]`（即 `laddergen --plan-from` 读的形状）、`select {selected, surplus_unused, unusable}`。产物不含任何时间戳与 `verdicts.json` 的 sha：只改 z、b、X 时逐字节不变。
6. **判定器与自测样本**（P2-10，§3.5 第 22 条）：
   - **thresholds 阶段 pilot/2 的窗口核对**（新增，`inputs._check_threshold_window`，在按 ladder 定 regime 之后）：登记了 `stage: thresholds` 的 pilot/2 时，除 `ladder.json` 保持档外的每个负载档窗口必须等于它的 `window_ns`（其 `null_z` 键的 `L=`），否则 13（`register.calibration`）。bootstrap pilot/2 按族取值、没有窗口，不核。`make_samples` 的 `calib_v2` 新增 `window_s`（thresholds 的 `window_ns` 与键的 `L=`，缺省 60）。
   - **已提交样本按 v2 缺省重新生成**（`samples/`，50 个；取代勘误 calib-v2-A3 第 9 条末句的「本步没有重新生成」）。新增 12 个 v2 样本：良品 `v2_good`；两个可满足性样本 `v2_sat_overloaded_level`（明显过载级 UNSTABLE）、`v2_near_stable`（近边界稳定档在内）——样本档位表每个负载只有一档，所以是单遍版，两遍合成由 `test_judge_levels.py` 覆盖（**两遍版样本已由勘误 calib-v2-A5b 第 2 条补上，样本数 52**）；造坏各一：`v2_ladder_missing`、`v2_ladder_sha`、`v2_pilot1`、`v2_window`、`v2_d14_missing`（11，C13 第二半 NOT_RUN）、`v2_d14_env`、`v2_arm_config_missing`、`v2_near_boundary_tiers`、`v2_inline_thresholds`（其余都是 13）。`regime_nb_missing`、`regime_nb_good` 的期望改为 13（登记 `near_boundary_noise` 即拒收，勘误 calib-v2-A2 第 1 条）。全表见 `samples/README.md`。
   - **已知不符**：模式 `P+X+I+V` 的 5 个 I 样本期望 I 定位成立；I1 改读 pilot/2 `i1_calibration` 属 P2-13（SB-C），之前 I 为 `NOT_RUN`，自测对它们报不符（**处理方式已由勘误 calib-v2-A7 第 1 条改为延后清单与 xfail**）。场景臂良品树与「同一棵 V2-n 树按两种 mode 判」两项（第 22 条）同属 SB-C。
   - **V0 取消计数**（第 23 条）：`pilot2.v0_cancelled(rows, status_codes, start, end, stop)` 数 reqlog 中 `status_codes` 映射为 `grpc_CANCELLED`、完成时刻在 `[start, end)`（过载档与其后清空槽）且早于负载器停止时刻的请求；停止时刻及之后的取消不计。
7. **测试**：`tests/judge/test_pilot2.py`（第 13、23、27、31、34、35、40 条；第 27、34 条读本机第一轮 15 棵树，W 不在时跳过；`samples/calibv2_real/e0_round1.pilot_inputs.tsv` 是 `pilot_inputs.tsv` 的副本，来源与 sha 见 `SOURCES.md`）。变异与实测记录在 `_progress/calibv2-a.md`。
8. **本步没有做**：第 21 条作业路径 e2e（本机 fakebin，`capjobs.submit --dry-run` → `caplaunch` → 判定器）——需要探针树覆盖清单，本 worktree 没有 `probe-tree`；V2-b 区组段的缺额补足（`shortfall` 拒收区组段；`laddergen --plan-from` 也只写单启动条目）。（**两项都已由勘误 calib-v2-A5b 完成：区组缺额补足见第 1 条，第 21 条 e2e 见第 3 条**。）

#### calib-v2-A5b：V2-b 区组的缺额补足、两遍可满足性样本、第 21 条作业路径 e2e（2026-11-15）

1. **区组缺额补足**（DESIGN-r7a P2-09b、§4.5；取代 A5 第 5 条「区组段拒收」与第 8 条的同一项）：
   - `pilot2 shortfall` 接受含区组条目的段。单位沿用 `_select`：区组条目为比较 `cmp:<比较>`，其余为配置；缺额 d、`by_config[单位]` 同 A5 第 2 条。
   - **q̂ 的计数单位**：区组算一项（两个作业合起来），非区组条目（作业）算一项；一项「丢失」= 其中任一启动不可用（`--job-level-only` 时只算作业级不可用，VOID 当可用）；区组两作业不在同一台已知节点上也是不可用（不是作业级）。同一段里既有区组又有单臂配置时，两类项合在一个 q̂ 里。
   - **补足项**：比较单位的补足数 = `stats.shortfall_topup(d, lost, ended)` 个区组；条目形状 `{item_class: post_gate:shortfall, comparison, configs: [[配置, 臂], [配置, 臂]]（按配置名排序）, blocks, pin: true}`。臂 = 原段区组条目的臂去掉 `.<block_id>.<slot>` 后缀（`laddergen` 的区组臂命名）；后缀不符、一个区组不是两个不同配置、同一比较的区组配对不一致 ⇒ 3。
   - **上限**：一个区组算 2 个作业（各装 1 次启动），与配置单位的作业数一起计入 `shortfall_cap_jobs`；按单位名顺序取能装下的最多数，有削减 ⇒ `capped`。
   - **`laddergen --plan-from`**：先按原规则装配置条目（不钉、钉节点分开），再把区组条目写成区组：每个区组两个钉节点、各 1 次启动的作业，共用一个 `order`；区组号 `<比较>-b<规则><轮次><NN>`（如 `s1-vs-s2-bshortfall101`；轮次取门产物的 `round`，所以不会与原段的 `<比较>-bNN` 或前一轮重复；`_select` 按最后一个 `-b` 前的部分认比较）；两个 slot 的先后按登记种子 `seed_u(seed_base, 比较, 区组号, 0)`，与 `--job-plan` 相同；臂 `<臂>.<区组号>.<slot>`；`item_id` `<phase_id>-<规则>-<区组号>-s<slot>`。门产物里有区组条目却没有整数 `round` ≥ 1（**已由勘误 calib-v2-A8 第 1 条扩到全部门产物**），或区组条目缺字段 / `pin` 不是 true ⇒ 3。
   - 测试：`tests/judge/test_pilot2.py`（区组段的补足与 q̂、`--job-level-only`、每区组 2 个作业的上限、臂后缀不符 ⇒ 3、门产物 → `--plan-from` → 登记后补足区组排在原段之后入选）；`tests/launch/test_laddergen.py`（区组条目的 `order`、区组号、臂、种子决定的 slot 先后；五种畸形区组条目 ⇒ 3）。
2. **两遍可满足性样本**（§3.5 第 22 条；补 A5 第 6 条的单遍版）：`make_samples` 新增选项 `two_pass`（档位表在清空档 5 之后接 down r40 档 6 与清空档 7，r40 级的两遍是档 4 与档 6；模拟里过载与滞后增长作用于两档；时间轴与 sidecar 行数随表长延长，不用该选项的样本内容不变——重新生成后除时间戳外逐字节相同）。新增样本 `v2_sat_overloaded_level_2p`（两遍都过载 ⇒ 档 4、6 UNSTABLE，级 1 UNSTABLE）与 `v2_near_stable_level_2p`（两遍都在内 ⇒ 级 1 STABLE），`expect.json` 新增 `levels`（按级号断言 `verdicts.json` 的 `levels[]`，`run_selftest` 读它）。样本数 52（`--expect-samples 52`）。`tests/judge/test_selftest.py` 新增：把两遍合成改成「两遍都在内 ⇒ UNSTABLE」时，档 verdict 不变、只有 `v2_near_stable_level_2p` 的级期望变红（证明 `levels` 期望被读）。
3. **第 21 条作业路径 e2e**（本机 fakebin；脚本与记录在 worktree 的 `_local/calibv2-a/e2e/`，不入库）：`build.py` → 作业 → 判定 → `check.py`，一个作业两次启动（同一臂 `e0b-s1-single`、同一档位表；L0 的 `--hw-profile-sha` 等于标定文件的 `environment`，L1 换成另一个 sha）。
   - **档位表**：`laddergen --kind e2e --phase V0`（e_c 0.30）、`--center-literal 19.09`（ĉ0）、S = 4 s、种子 `E0v2-e2e-local-20261115|s1|e2e|1`（u = 0.862）：基线 0.3 + 级 ρ ≈ {0.502, 1.002, 1.502}·ĉ0 上下两遍 + 3 斜率档，22 档、表长 1610 s；`tiergen` 展开后 sha 等于 `ladder.json` 的 `tier_table_sha256`；全部档 `streams_c = null`、`load_value = rate_rps`（R 轴）。非档位取值用 e0w60 的 e0b-s1 值（beat 4 ms、50 msg/流、relay 每 msg 忙等 1 ms、请求超时 60 s）。
   - **登记表**（v2 形状）：`arm_config`、`files_by_arm`（含 `ladder`）、`calibration` = `samples/v2_good` 的 bootstrap pilot/2 与 `e0_d14.json`（只把 `environment` 换成 L0 的画像 sha 与假镜像 sha，pilot/2 的 `d14` 指向它）、`inline_thresholds` 为空、`judge.checks = [C3, C5, C6, C9, C10, C11, C13]`、`expected = {budget_id, job_budget 2, plan（已登记的 job_plan.tsv）, disk}`；另登记 `node_exclusions` 与 exclude 文件。
   - **提交**：`capjobs.submit --dry-run` 走完 P2-08b 的六道门（桩 ssh 回答 `scontrol show reservation`（一条不重叠的维护预约，因而也查了 `sinfo`）与 `sinfo`，桩 `statvfs`），退出码 0、维护门 `pass`、不写 BUDGET.tsv；再用同样的桩真提交，BUDGET.tsv 得 `RESERVED` 与 `JOB_ID` 两行。
   - **结果**（MEASURED，作业 J9400011，提交时代码 `bc60637c`，判定时代码 `3a4ce547`——两者在 `tools/` 下只差 `pilot2.py` 与 `laddergen.py`，判定器不用它们）：作业 sbatch 退出码 0（环境门禁替身记下真门禁退出码 6：本机没有 Slurm、画像未登记，与以往 e2e 同形），两次启动都 VALID 并拷回。L0 判定退出码 0，`launch_verdict = VALID`，启动级 C10、C13、C13-MEAN、GPU-IDLE 全 PASS；级 0（ρ 0.5）STABLE、级 2（ρ 1.5）UNSTABLE（两遍都过载）、级 1（ρ 1.0）CONTRA（X3 计数下降与滞后上升并存，靠近容量）；`boundary.b` = 19.13 rps（不截尾，`contra_inside`），T7 μ̂ = 17.88 rps。L1 判定退出码 13，`INPUT_INVALID`：`register.calibration[0]: calibration environment (hw …48bdc271…) != launch (hw …275bd857…)`。`fakebin.log` 只有作业里 simgate 的 3 行（exit 97 替身）。
   - 限制：画像 sha 是脚本里定的合成值（本机作业的真实画像不登记）；镜像是 4 KB 的假文件；判定器的阈值来自合成 bootstrap 样本（r6 常量），不是 V0-cal 实测值。


#### calib-v2-A6：v1 登记表不受提交门控（SB-A 全量验证，2026-11-15）

1. **受门控的登记表**（`capjobs.budget.gated`；收窄 A4 第 5 条「有 `expected` 或 `arm_config` 段」）：有 `arm_config` 段，或有 `expected` 段且它不是只含 `job_budget` / `launches_per_arm` 的对象（`budget.V1_EXPECTED_KEYS`；含其他任何键，或不是对象）。没有 `arm_config`、`expected` 缺省或只含这两键的登记表是 v1 登记表（DESIGN-r7a §3.2「没有 `arm_config` ⇒ v1 语义」），照旧走原提交流程：不过六道门、不写 `BUDGET.tsv`、不读维护预约，给了 `--plan-item` ⇒ 3。已登记的 E0 第一轮、e0w60 与 E1 的登记表都是 `expected = {job_budget, launches_per_arm}`，A4 的写法会让它们的提交在门 ① 退出 3。
2. 测试：`tests/jobs/test_budget_gates.py` 新增 v1 形状（两键、一键、空）照旧提交且不写 `BUDGET.tsv`、给 `--plan-item` ⇒ 3；`expected` 多出 v2 键、`expected` 不是对象、有 `arm_config` 时受门控（不完整的 `expected` 在门 ① 退出 3，什么都不写）。

#### calib-v2-A7：I 样本延后到 P2-13、V0-cal 作业计划每作业每配置各一次、请求超时 null 与 c13-pc 档位表（SB-A 验证缺口，2026-11-15）

1. **依赖 I 定位的测试与自测样本**（取代 A5 第 6 条「已知不符」中「自测对它们报不符」）：v2 判定器按设计拒收 pilot/1，I1 的阈值源（pilot/2 的 `i1_calibration`）属 P2-13（SB-C），在那之前 I 为 `NOT_RUN`。I 定位逻辑与样本期望都不改。
   - `tests/judge` 里断言 I 定位结果的 6 条用例标 `pytest.mark.xfail(strict=True)`（标记是 `tests/judge/conftest.py` 的 `XFAIL_I_P2_13`，原因写明 P2-13 与本条）：`test_analysis.py` 的 `test_boundary_cli_writes_product_and_record`、`test_boundary_keeps_the_judges_t4_next_to_the_cut_interval`；`test_intervene.py` 的 `test_intervene_cli_on_a_judged_control_launch`、`test_load_launch_uses_the_interval_cut_at_the_tool_limit`；`test_calib.py` 的 `test_loadgen_band_maximum_is_measured_only_when_the_role_was_localised` 的两个参数（`i_reader_slow`、`i_server_slow`）。strict：P2-13 落地后它们一旦通过就算失败，标记必须随之去掉。
   - `samples/run_selftest.py` 的 `DEFERRED_P2_13` 列出 5 个样本及各自依赖 I 的期望部分：`i_reader_slow`、`i_reader_slow_real_pins`、`i_server_slow`、`sat_spin_thread_same_process` 是 `tier_localisation` 与 `grade`；`i_conflict` 另加 `tier_verdicts.4`（档 4 的 UNDETERMINED 来自 I 冲突）。它们照常判，期望的其余部分（退出码、其他档的判定、检查等）照常核、不符记入 `errors`；列出部分的不符按样本写进报告的 `deferred` 段，不记入 `errors`；报告另记 `deferred_rule`。列出部分全部成立 ⇒ 记入 `errors`（`listed in DEFERRED_P2_13 but matches its expectation`），所以 P2-13 落地后清单必须同步缩短。只延后列出部分是为了让 `test_selftest.py` 的 X2 篡改用例仍然变红（它只在这几个 I 样本的档 4 上体现）。`test_selftest.py` 的两条全量用例（生成样本、已提交样本）因此按原断言（退出码 0）通过，不标 xfail。
2. **V0-cal 的作业计划**（取代 A4 第 4 条对 V0-cal 的「按种子混装后切成作业」；DESIGN-r7a §4.2「每作业 4 次启动（s1@1 ms、s1-uvloop、s2、s4 各 1 次，按登记种子排序）」）：`capkit.job_plan_spec/1` 新增两个可选键 `layout`（`mixed` 为缺省，即 A4 第 4 条的做法；`one_each`）与 `phase`（取值同 `--phase`）。
   - `one_each`：每个作业恰好跑每个配置一次；作业数 = 每配置的计数；作业内的启动顺序（即条目 `arms` 的顺序）按 `sha256(seed_base|类|one_each|作业序号|配置|臂)` 升序，作业之间的顺序按 `sha256(seed_base|类|one_each|作业序号)` 升序。要求配置、臂各不相同，同一类（计划、超额）里各配置计数相等，`launches_per_job` = 配置数；任一不满足 ⇒ 退出码 3、不写文件。
   - `phase` 为 `V0cal` 而 `layout` 不是 `one_each`（含不写）⇒ 3。其余期不限制 `layout`。区组条目不受 `layout` 影响。
3. **请求超时**（DESIGN-r7a §4.2：s1-pc 为 `request_timeout_ns = null`，s1-fix、s2-fix、s4-fix 为 60 s）：
   - `laddergen --request-timeout-ns {null, <ns>}`：给了就经取值文件把档位表的 `request_timeout_ns` 设为该值（`null`，或十进制整数 > 0；`0`、负数、`None`、`6e10` ⇒ 3）；没给就取 `--base-values` 里的值。kind `pc` 必须给（两种 pc 行只差这一项）。
   - `--base-template` 的 `request_timeout_ns` 必须是 `"TBD:request_timeout_ns"`（`configs/e0/*.tiers.template.json` 都是）；写成字面值（含 `null`）⇒ 3。`--base-values` 里要用到的任一值为 null（未填的 `*.values.todo.json`）⇒ 3；此前这种 null 会原样抄进取值文件，到 `tiergen` 才被拒。
   - 取值文件里超时为 null 时另写 `"registered_nulls": ["request_timeout_ns"]`；不为 null 时不写这个键，所以不用这个选项的输出逐字节不变。
   - `caplaunch.tiergen`（取代 A4 第 1 条「`tiergen.py` 未改」中关于 `tiergen` 的部分）：取值为 null 仍按「未填」拒收，例外只有**登记的 null**：名字在 `NULLABLE_VALUES`（只有 `request_timeout_ns`）里、列在取值文件的 `registered_nulls` 里、且取值确为 null。`registered_nulls` 不是名字不重复的列表、列了不可为 null 的名字、或列了取值不为 null 的名字 ⇒ 3。
4. **kind `c13pc`**（DESIGN-r7a §4.2 c13-pc 行；`W/02_e0/E0/analysis/calib-v2/ERRATA-after-r7a.md` E1、E2）：档序是 `pc` 去掉过载档：预热、基线 0.3、两个亚饱和 0.3 档（下行，列在 `ladder.json` 的 `cal`，pass 1、2），每个负载档之后一个清空档（V0 的 e_c 0.30 下都是 60 s）。`--center-literal` 可用（ĉ0）；`--request-timeout-ns` 可选（不给就取 `--base-values`）。基线档照其他 kind 保留（INFERRED：§4.2 这一行只写「s1；两个亚饱和档 0.3」，没写基线；保留它只多一个 0.3 档，判据「C13 每窗 RED 且 C13-MEAN RED」不受影响）。场景里的 `aux` 自旋器角色与 E2 的人工保证不由本工具生成。
5. **测试**：`tests/launch/test_laddergen.py`（V0-cal 计划每作业每配置各一次、顺序随种子、`--out` 写出后读回相等；`V0cal` 拒收 `mixed` 与缺省，无 `phase` 时混装确会让同一配置在一个作业里出现两次；`one_each` 的五种拒收；`pc` 的 null 超时写成登记的 null、`tiergen` 展开后 sha 与 `ladder.json` 一致、60 s 写成数值；`pc` 不给或给非法超时 ⇒ 失败；其他 kind 取基础取值、基础取值为 null ⇒ 失败；基础模板写字面超时 ⇒ 失败；`tiergen` 对五种非法 `registered_nulls` / 未登记 null 退出 3 且不写文件，`tiergen.unset_values` 单独核同一规则（不可为 null 的名字不靠档位表校验兜底）；`c13pc` 档序、角色、清空槽、比 `pc` 短，判定器的 `_load_ladder` 接受）；`tests/jobs/test_v0_tables_fit.py`（`pc` null、`pc` 60 s、`c13pc` 各 1、2 次一作业：按生成的 `budget_s` 写的启动计划在 `--time` = 生成的 `time_budget_s` 时 `capjobs.submit --dry-run` 退出 0、报告的时间预算相等，少 1 s ⇒ 3「exceeds --time」）；`tests/judge/test_selftest.py`（延后清单 = 5 个样本且都在样本表里；延后样本列出部分的不符不记入 `errors`；退出码不符、列出部分之外的档判定不符仍是 error；改名成延后样本名的良品样本 ⇒ 1）。变异与实测记录在 `_progress/calibv2-a.md`。
6. **本步没有做**：`laddergen --plan-from`（缺额补足）没有 `one_each`：V0-cal 的作业级补足若丢了 ≥ 2 个作业，按 A5b 的装法补足作业里可能同一配置出现两次（V0-cal 不设超额，补足按配置缺额给数，各配置缺额不一定相等）；要不要、怎样按 §4.2 拆，留给 V0-cal 登记之前决定。

#### calib-v2-A8：R1 自审修复（`capkit-calibv2a-J17871` R1，2026-11-15）

本小节取代前面各小节中加了指引的句子；未提到的照旧。

1. **`laddergen --plan-from` 的轮次**（取代 A4 第 4 条与 A5b 第 1 条的相应部分）：门产物（任何门后规则）必须有整数 `round` ≥ 1，否则退出码 3、不写文件（此前只对含区组条目的产物要求）。单启动条目的 `item_id` 由 `<phase_id>-<规则>-NNNN` 改为 `<phase_id>-<规则>-r<round>-NNNN`；区组号 `<比较>-b<规则><round><NN>` 与区组条目号 `<phase_id>-<规则>-<区组号>-s<slot>` 不变。同一 `--phase-id` 的第 1、2 轮补足因此不会同号（此前第 2 轮与第 1 轮同号，会被提交门 ②、`_select`、`pin-next` 拒收）。SB-C 的门产物（覆盖补投、盲态补投、后备补投、平移）要经 `--plan-from` 生成计划时同样要写 `round`。
2. **CONTRA 档的读方**（补全 A3 第 6 条的读方清单）：`intervene.load_launch` 另记 `contra_tiers`（被判、verdict 为 CONTRA 的档），`_reverse_unusable` 把有 CONTRA 档的反向剂量启动当作不可用证据，原因写「CONTRA tiers [...]（the deterministic checks and X disagree）」（**措辞已改：勘误 calib-v2-A9 第 9 条**），与 VOID 档的原因分开。T4 区间不变（它按设计跳过 CONTRA 档）。按档 verdict 过滤或比较的读方（`grade.first_non_stable`、v1 构建器 `pilot.build` 与 `calib` 的噪声带、`pilot.select_tiers`、`intervene`）各自的语义以 A3 第 6 条与本条为准。
3. **`pin-next` 的节点状态**（取代 A4 第 9 条「去掉后缀后不以 `down`、`drain`、`drng`、`fail` 开头」）：`nodes.usable_state` 改为白名单。可用的基础状态只有 `idle`、`mix` / `mixed`、`alloc` / `allocated`、`comp` / `completing`（不分大小写）；后缀只许没有或 `-`（backfill 计划）。`*`（不响应）、`~`、`#`、`!`、`%`、`$`、`@`、`^` 任一后缀，以及其他任何基础状态（`maint`、`resv`、`unk`、`inval`、`boot`、`npc`、`plnd`、`down`、`drain` 等）都不可用。
4. **thresholds 阶段的比值容差**（取代 A2 第 2 条末句的「thresholds 的比值容差不加……结构下限」）：`nullz.compare` 的 `ratio_abs` 与 `ratio_tol` 一样必须给整数 `planned_requests` ≥ 1（否则 `ValueError`），容差 = max(1/`planned_requests`, `abs_tol`)。依据 DESIGN-r7a §2.9（第 584 行）：「V2 用 |norm − 1| 的经验上分位……下限 1/(该档计划请求数)」。判定器的 X1 一直传该档的计划请求数，计划请求数为 0 时不比较（照旧 NOT_RUN）。
5. **钉节点断言用 Slurm 节点名**（取代 A4 第 6 条的 `socket.gethostname()` 断言与第 11 条的「主机名不符」）：`envfiles job` 比对 `--node`（作业脚本传 `SLURMD_NODENAME`，未设时为 `unknown`）与 `--expect-node`；`--node` 为空、`unknown` 或不等 ⇒ 检查 FAIL、退出 6 ⇒ 作业级 `ENV_INVALID`。`submit --expect-node`、`pin-next`、`sinfo %N` 用的都是这个名字空间。`job.json` 照记 `hostname_short`，只作诊断；不符时 `snapshot_problems` 末尾写节点名、主机名与期望节点。
6. **`scontrol show reservation` 的解析**（A4 第 5 条 ⑥）：按 `ReservationName=` 切成记录，记录内按键取 `ReservationName`、`StartTime`、`EndTime`、`Nodes`、`PartitionName`、`Flags`（键前是行首或空白，值到下一个空白为止），所以其他字段的值可以含空格（如 `Comment=weekly maintenance window`）。输出不以 `ReservationName=` 开头、缺 `StartTime` / `EndTime` / `Nodes`、同一记录里某键出现两次 ⇒ 解析失败 ⇒ 门推迟（退出码 4）。
7. **脏树检查在提交门之前**（A4 第 5 条「在全部既有检查之后」）：受门控的真提交先查代码树是否干净（不干净 ⇒ 退出码 3），再跑六道门；所以脏树不会触发 ssh 查询，也不会写 `reservations.log`。`--dry-run` 照旧跑完六道门，只在输出里记 `would_refuse_dirty`。
8. **`_select` 要求给全本段的计划**（A5 第 2 条；RS28）：`pilot2.launch_facts`（`_select` 与 `shortfall` 共用）读 BUDGET.tsv 的 `RESERVED` 行。给出的计划在台账里只能对应一个 `budget_id`（否则退出码 3）。本段的范围：从给出的原段计划的第一行 `RESERVED` 起（原段还没有时从任一给出计划的第一行起），到下一份未给出的原段计划（`item_class` 为 `planned` / `surplus`）的第一行为止；同一 `budget_id` 的各段（V1-b A、B）按这个先后分开。范围内每一行的 `plan_sha256` 都必须是给出的某份计划，否则退出码 3，报错列出漏给的计划 sha（**另要求已结束作业的启动树都已判定：勘误 calib-v2-A9 第 4 条**）。所以漏给一轮补足计划时，轮次、`prior_jobs` 与入选不会从头再算，N15 的 2 轮上限照样生效。限制：A 段的入选若要在 B 段开始之后再算，B 段之后才提交的 A 段门后计划会被当成 B 段的；按 DESIGN-r7a §4.8 的投递顺序（A 段 →（补足）→ G1a′ → B 段）不会出现这种情形。
9. **bootstrap 的 `c13_launch_mean`**（取代 A5 第 4 条的「按本树入选的 cal 档读」）：每棵入选树、每个 `T3:foreign_busy_frac:<角色>` 键，取该启动全部被判负载档（`verdicts.json` 的 `tiers[]` 中 `judged` 为真，不论 verdict 与 ladder 角色）上读数的均值，与判定器 `C13-MEAN` 是同一总体（共用函数 `checks.c13_launch_readings`）。`c13_window` 照旧只读入选的 cal 档。整树被排除的树不出读数。
10. **`stats.t_isf`**：上界从 1 开始倍增，直到 `t_sf(hi) < p`，再二分；上界超过 `T_ISF_MAX`（1e12）⇒ `ValueError`，不再在 1000 处静默截断（此前 `t_isf(1e-4, 1)` 返回 1000.0，真值约 3183.1）。
11. **预算闸的锁**（A4 第 5 条 ②～写 `RESERVED` 行；RS29）：受门控的真提交从跑六道门起，到本次提交返回（含写 `RESERVED` 行）止，持有 `<W>/BUDGET.tsv.lock` 的 `fcntl.flock(LOCK_EX)`（文件不存在时建一个空文件；**是软链 ⇒ 失败：勘误 calib-v2-A9 第 10 条**）。它与 `ledger.append_tsv` 对 `BUDGET.tsv` 本身的锁是两个文件。并发的第二个真提交在锁上等待，拿到锁后在门 ② 读到第一个写的 `RESERVED` 行，退出码 3。`--dry-run` 与不受门控的提交不取锁。
12. **正则锚定**：本子批新增的 `jobplan.ITEM_ID_RE`（`BLOCK_ID_RE` 同一对象）、`budget.OFFSET_RE`、`nodes.NODE_RE`、`nullz.NULL_Z_KEY`、`RATIO_KEY`、`STAT_KEY` 的结尾由 `$` 改为 `\Z`，末尾带换行的字符串不再匹配。`STAT_KEY` 的角色段 `[^:|*]+` 本身接受换行，改锚后行为不变。base 既有的 `ARM_RE`、`HOST_RE` 与 `capjobs.resources.NODE_RE` 不改。
13. **`jobs/README.md`**：远端命令白名单与 `tests/jobs/conftest.py` 的禁止执行清单都补上 `sinfo`（与 A4 第 10 条一致）。
14. **测试**：`tests/launch/test_laddergen.py`（条目号带轮次；`round` 缺失、0、布尔、字符串、浮点 ⇒ 3）；`tests/judge/test_pilot2.py`（同一 phase 两轮补足经 `--plan-from` 条目号不重、`_select` 同时接受；漏给第 2 轮或两轮都漏 ⇒ 3 并列出 sha、都给 ⇒ 第 3 轮 N15；两段同一 `budget_id` 时各段只需给本段计划；`c13_launch_mean` 与 `C13-MEAN` 同一总体）；`tests/judge/test_intervene.py`（CONTRA 档使反向剂量启动不可用，不依赖 I）；`tests/judge/test_calibv2_sources.py`（`ratio_abs` 下限与 `planned_requests` 必填；带换行的 `null_z`、`ratio` 键 ⇒ 13）；`tests/judge/test_stats_calibv2.py`（`t_isf(1e-4, 1)` ≈ 3183.1，越界报错）；`tests/jobs/test_pinning.py`（状态白名单；节点名比对、主机名不读、`unknown` 不满足；新锚定的三个正则）；`tests/jobs/test_budget_gates.py`（带空格的 `Comment=` 能解析；脏树在门之前拒收、不 ssh、不写 `reservations.log`；并发提交同一条目只占一次；`--dry-run` 不建锁文件）。变异与实测记录在 `_progress/calibv2-a.md`。

#### calib-v2-A9：R2 自审修复（`capkit-calibv2a-J17871` R2，2026-11-15）

本小节取代前面各小节中加了指引的句子；未提到的照旧。

1. **`pin-next` 只看本比较的条目**（取代 A4 第 9 条「上一个 = BUDGET.tsv 中这些计划的钉节点条目里最后一行 `RESERVED`」；calibv2a-RS32）：条目的归属：区组条目属于 `jobplan.block_comparison(block_id)`（`block_id` 最后一个 `-b` 之前的部分；`pilot2._unit_of` 用同一个函数）；不在区组里的钉节点条目属于 `--log`（`pin_next.log`）里早先为它记下的 `key`，没有记录的不属于任何比较。「上一个」只从属于 `--key` 的钉节点条目的 `RESERVED` 行里取最后一行；`--item` 属于别的比较 ⇒ 退出码 3、不写日志行。「已有 `RESERVED` 行」的检查照旧对全部条目。同一 `--key` 在日志里已有记录（且记录的计划与本次 `--plan` 至少有一份相同，即同一个比较的同一条序列；别的期复用同一比较名、计划全不同的不算）、而其 `seed_base` 或 `node_list.sha256` 与本次不同 ⇒ 退出码 3、不写日志行（一条轮转序列只有一个种子与一份清单；calibv2a-RS33）。日志读写都不跟随软链；日志里有一行不是 JSON 对象 ⇒ 3。
2. **`laddergen --plan-from` 的每作业启动数**（补 A4 第 4 条、A8 第 1 条；calibv2a-RS33）：门产物有 `cap` 时它必须是对象；有 `cap.launches_per_job` 时 `--launches-per-job` 必须与它相等，否则失败、不写文件（`shortfall` 按这个数算出作业数上限，装得更碎会超出 `shortfall_cap_jobs`）。只含区组条目的产物同样要相等（区组每作业一个启动，这个数不影响装法）。
3. **`pilot2 bootstrap` 的列表是 V0-cal 全集**（补 A5 第 4 条；calibv2a-RS28）：全集由台账机械取得：`--register` 的 `expected.budget_id`（必须是合法的 id，否则 3）在 BUDGET.tsv 下全部 `RESERVED` 条目 → `submissions.tsv` 合并后的行 → 作业号 → JUDGED.tsv 中这些作业的被判树。要求：这些条目都已 `ENDED`（否则 3，列出条目）；不是 `ENV_INVALID` / `CANCELLED` / `SUBMIT_FAILED` 的已结束作业没有还没判定的启动树（判据同第 4 条）；列表的树集合等于全集（多一棵或少一棵 ⇒ 3，报错列出多出与缺少的树）。产物新增 `universe {budget_id, items, jobs}`（条目数与作业数）。
4. **还没判定的启动树**（补 A5 第 2 条「可用」、A5 第 5 条拒收条件与 A8 第 8 条；calibv2a-RS35；挑战设计的裁决：这是对 DESIGN-r7a P2-09b「条目都已结束」前置条件的补充）：对已结束、且不是作业级不可用的作业，在其计划快照 `job.env` 的 `CAPJOBS_RUN_ROOT` 下找 `runtree.tree_dir_glob(作业号)`（`J<作业号>-r*-L*`）里带 `raw/launch.json` 的目录，去掉 JUDGED.tsv 里已有的树：
   - 没有 `judge/`、`judge.attempts/*/attempt.json` 至少有一份（判定尝试已结束、没有判定结果，例如 INPUT_INVALID）⇒ 该启动记为不可用，原因「judge attempted, no verdicts (<树>)」（仍按作业级计入 `lost`，与此前「没有判定结果」相同）；
   - 其余（从没判过、判定还在跑、或已有 `judge/` 而台账行没写完）⇒ `Pilot2Error`，退出码 3，报错列出这些树（「not judged yet」）。把它们判完后重跑同一命令即可；
   - 计划没有 `snapshot_dir`、`job.env` 读不到或没有 `CAPJOBS_RUN_ROOT` ⇒ 3（不知道的不当作丢失）。
   - 没有树目录、或目录里没有 `raw/launch.json`（启动没开始，RS24）⇒ 照旧记作业级丢失，原因改为「no judged launch (no launch tree with raw/launch.json)」。
   `launch_facts`（`_select` 与 `shortfall` 共用）与 `bootstrap` 用同一个判据（`pilot2.unjudged_trees`）。读 `job.env` 的函数移到 `capcommon.ledger.plan_job_env_value`，`capjobs.submit` 的同名读法改调它。
5. **提交的写入顺序**（取代 A4 第 5 条「快照 → `BUDGET.tsv` 一行 `RESERVED` → 原有的 `PLANNED` / sbatch / `SUBMITTED` 流程」；calibv2a-RS34）：快照 → `submissions.tsv` 的 `PLANNED` → `BUDGET.tsv` 的 `RESERVED` → sbatch → 其余照旧。`PLANNED` 写失败 ⇒ 两本台账都没有新行；`RESERVED` 写失败 ⇒ 先给该计划补一行 `ENDED SUBMIT_FAILED`（原因「BUDGET.tsv RESERVED row not written」，不跑 sbatch，不占并发名额），再按原错误退出；两种情形下该条目都没有 `RESERVED` 行，下一次提交门 ② 放行。仍都在锁内、都在 sbatch 之前。
6. **`t7` 只用跑完且 P 为 PASS 的斜率档**（补 A3 第 7 条；RS3）：只取 `tier_seq ≤ launch_end.tiers_completed`（没有该值时按 −1，即一档不取）且档的 `P = PASS` 的斜率档；其余列进 `t7.skipped[{tier_seq, reason}]`（`not completed` 或 `P <状态>`），拟合结果与 NOT_RUN 结果都带 `skipped`。档的 X 结论照旧不读。
7. **`ladder.json` 的档位表 sha 与臂**（补 A2 第 4 条、A3 第 8 条；RS1）：`tier_table_sha256` 必须等于本启动登记的档位表 sha（`register.files[_by_arm].tier_table.sha256`），`arm` 必须等于本启动的臂（即 `files_by_arm` 的键；区组条目的臂带 `.<block_id>.<slot>` 后缀，`laddergen --arm` 要给这个完整臂名，本仓库的生成脚本都这样给）；不符 ⇒ 13（`item` 为 `ladder.json`）。`laddergen` 写的 `tier_table_sha256` 就是 `tiergen` 写出的档位表文件的 sha。`samples/make_samples.py` 的 v2 ladder 补写 `tier_table_sha256`，13 个带 ladder 的 v2 样本据此重新生成（`v2_d14_env`、`v2_d14_missing`、`v2_good`、`v2_inline_thresholds`、`v2_ladder_missing`、`v2_ladder_sha`、`v2_near_boundary_tiers`、`v2_near_stable`、`v2_near_stable_level_2p`、`v2_pilot1`、`v2_sat_overloaded_level`、`v2_sat_overloaded_level_2p`、`v2_window`）：文件集合不变，`expect.json` 不变，`ladder.json` 只多这一个键，其余差异是登记 sha 的连带变化与时间戳。
8. **格式不对的输入走退出码 3**（calibv2a-RS36）：`pilot2` 的登记表 `expected` 有则必须是对象（`null`、列表、字符串 ⇒ 3）、`calibration` 必须是列表、其 `d14_result` 项必须有字符串 `relpath`、`judge/verdicts.json` 必须是对象；`laddergen` 读 `--center` 的 `by_config` 及其条目前核对象类型（否则「has no by_config」，3），`--job-plan` spec 的 `configs`、`blocks` 必须是列表、区组的 `planned` / `surplus` 必须是整数 ≥ 0。没有扩大 `except` 的范围。
9. **CONTRA 的措辞**（取代 A8 第 2 条的原因文字）：`intervene._reverse_unusable` 的原因改为「CONTRA tiers [...] (X2 contradicts X1 / X3; the deterministic checks passed)」，docstring 同义；`grade` 对稳定档之上没有结论的档的说明改为「have no stability conclusion (VOID / NOT_RUN / CONTRA)」，不再说「X 没有算」（CONTRA 档的 X 是算了的）。
10. **锁与日志不跟随软链**（补 A4 第 5 条、A8 第 11 条与本节第 1 条；RS10）：`<W>/BUDGET.tsv.lock`、`reservations.log`、`pin_next.log` 都用 `os.open(..., O_NOFOLLOW)`（`capjobs.budget.open_nofollow`）打开；路径是软链 ⇒ `OSError` ⇒ 退出码 3，软链的目标不变。
11. **`submit` 退出码 → 结局**（DESIGN-r7a P2-08b 的退出码不改；挑战设计 C2 的裁决：只补对照，不改代码）：

   | 退出码 | 区分方法 | 结局 / 下一步 |
   |---|---|---|
   | 0 | — | 已提交 |
   | 3 | stderr 含「budget … no room」 | 预算用尽（N15）：停并上报 |
   | 3 | stderr 含「concurrency: … try again later」 | 并发已满：下一轮再试 |
   | 3 | 其他 | 拒收（条目已有 `RESERVED`、顺序、脏树、参数、登记、软链、`RESERVED` 写失败等），看报错 |
   | 4 | 该条目在 BUDGET.tsv 没有 `RESERVED` 行 | 维护推迟或读不到 reservation：下一轮再试 |
   | 4 | 该条目已有 `RESERVED` 行 | sbatch 失败（`submissions.tsv` 有 `ENDED SUBMIT_FAILED`）或结果不明（`UNKNOWN_SUBMIT`，用 `reconcile-submit` 收尾） |
   | 5 | — | 盘空间不足（N17）：停并上报 |

12. **一个事实一处定义**（RS15）：`checks` 的 C13 统计量名改用 `nullz.C13_STAT`（删去本地副本）；`stats.shortfall_q(lost, ended)`（Fraction）是 q̂ 的唯一定义，`shortfall_topup` 与 `pilot2 shortfall` 的 `q_hat.value`（`float(...)`）都用它；作业的启动树名模式由 `runtree.tree_dir_glob` 给出，与 `tree_dir_name` 放在一起。
13. **`pilot2` 的说明**（RS12）：`check_v2` 写明 `d14` 不调用它（输入是 v1 判的第一轮树）；`check_windows`、`check_ramp`、`load_e0_params`、`v0_cancelled` 各写明供谁调用（后续子批的门子命令、V0 分析），本子批没有调用方。`capjudge/README.md` 的 X1 一行改为「`|norm − 1|`（键名沿用）」。
14. **测试**：`tests/jobs/test_pinning.py`（两个比较交替提交：XB 的第一个区组 `previous` 为 None、AA 的第二个区组接 AA 的节点，`--item` 属于别的比较 ⇒ 3；非区组条目按日志归属；同一比较换种子或 `node_list` ⇒ 3、不写日志，别的期的计划用同一比较名不受影响；日志为软链或含非对象行 ⇒ 3）；`tests/jobs/test_budget_gates.py`（`PLANNED` 写失败 ⇒ 无 `RESERVED`；`RESERVED` 写失败 ⇒ 计划 `ENDED SUBMIT_FAILED`、不占并发、可再提交；锁与 `reservations.log` 为软链 ⇒ 3、目标不变）；`tests/launch/test_laddergen.py`（`cap.launches_per_job` 不等 ⇒ 3、相等或没有 ⇒ 写出；`by_config` 形状不对、spec 形状不对 ⇒ `LadderError`；`judge_accepts` 改为按 tiergen 的字节写档位表并登记其 sha）；`tests/judge/test_pilot2.py`（bootstrap 少一棵、多一棵别的 `budget_id` 的树、条目未结束、已结束作业有未判定的树 ⇒ 3，判定器拒收过的树不算输入；shortfall 遇到未判定的树 ⇒ 3、判完后计入，只有尝试记录 ⇒ 不可用；`expected` 不是对象 ⇒ 3；`verdicts.json` 不是对象 ⇒ 拒收；`shortfall` 产物经 `--plan-from` 时每作业启动数不等 ⇒ 3）；`tests/judge/test_judge_levels.py`（没跑完、P 为 NOT_RUN 或 RED 的斜率档进 `skipped`、不进拟合）；`tests/judge/test_calibv2_sources.py`（`tier_table_sha256` 不等或缺、`arm` 不等 ⇒ 13）；`tests/judge/test_stats_calibv2.py`（`shortfall_q`）。变异与实测记录在 `_progress/calibv2-a.md`。

#### calib-v2 SB-A 收尾：本节各小节的总清单（DESIGN-r7a P2-10）

| 内容 | 小节 |
|---|---|
| `capkit.pilot/2` 两种 stage（thresholds、bootstrap）与 `capkit.d14_result/1`；判定器只从这两种标定文件取阈值 | A2 第 1、2 条；A5 第 3、4 条 |
| 阈值：X2 / X3 的 z_d 对阈值、计数下限、X1 的 \|norm − 1\| 容差（bootstrap 下取 max(2/计划请求数, tol)；thresholds 下取 max(1/计划请求数, abs_tol)：A8 第 4 条） | A1 第 3 条；A3 第 1～4 条 |
| 级合成（两遍、AMBIGUOUS、CONTRA 不作废档与启动、UNDETERMINED）、`levels[]`、`boundary`、`t7` | A3 第 6、7 条（`ladder.json` 级字段：第 8 条） |
| C13 上限（每窗、启动均值 `C13-MEAN`），无上限时第二半 NOT_RUN、读数照记 | A3 第 5 条 |
| 各期检查清单：判定器跑 `MANDATORY_CHECKS ∪ register.judge.checks`（不变）；E0 v2 登记表列 `[C3, C5, C6, C9, C10, C11, C13]`（DESIGN-r7a §2.11.2） | 本表；§10.2 |
| 登记计划 `job_plan.tsv`（没有替换 / 重投类） | A4 第 4 条 |
| `BUDGET.tsv` 与预算闸（按不同 `item_id` 计数） | A4 第 5 条；受门控的登记表：A6 第 1 条 |
| 三道提交门：并发 ≤ 8、盘空间、维护窗口 | A4 第 5 条 ④⑤⑥ |
| 节点排除清单（`node_exclusions`、`--also-exclude`、最新一版核对） | A4 第 7、8 条 |
| 缺额补足（`pilot2 shortfall`） | A5 第 5 条；区组：A5b 第 1 条 |
| 入选规则（`_select`：可用、登记顺序前 N、门后全部入选、补足排在原段之后） | A5 第 2 条 |
| bootstrap pilot/2（`pilot2 bootstrap`、规则原文）；`inline_thresholds` 不再是阈值源（v2 登记表非空即 13） | A5 第 4 条；A2 第 5 条 |
| `arm_threshold_config`（判定器取阈值只经它；`arm_config` 给配置） | A2 第 3 条（按臂 `ladder.json` 定 regime：第 4 条） |
| `node_list` 与钉节点轮转 `pin-next`、`--expect-node`、`exclude --all-but` | A4 第 6～9 条 |
| `remote.py` 白名单加 `sinfo`（`scancel` 仍不在内） | A4 第 10 条 |
| 自测样本（含两遍可满足性样本）；第 21 条作业路径 e2e | A5 第 6 条；A5b 第 2、3 条 |
| I 样本延后到 P2-13（xfail 与 `DEFERRED_P2_13`）；V0-cal 作业计划 `one_each`；`--request-timeout-ns` 与登记的 null；kind `c13pc` | A7 第 1～4 条 |
| R1 自审修复：门产物轮次与条目号、CONTRA 读方、节点状态白名单、thresholds 比值下限、钉节点用 Slurm 节点名、维护预约按键解析、脏树先查、`_select` 要求计划齐全、`c13_launch_mean` 口径、`t_isf`、预算闸锁、正则锚定 | A8 第 1～13 条 |
| R2 自审修复：`pin-next` 按比较取上一个、`--plan-from` 每作业启动数、bootstrap 全集、未判定的启动树、提交写入顺序、`t7` 只用可用斜率档、`ladder.json` 档位表 sha 与臂、格式不对的输入、CONTRA 措辞、锁与日志不跟随软链、`submit` 退出码对照、`C13_STAT` / `shortfall_q` 单一定义 | A9 第 1～13 条 |

### Batch 2B: S-PB prebaked stub, fingerprint seam and request-bytes input（2026-11-14）

第二批子批次 2B（RUNBOOK-stage2-draft.md §2.1 第 3 行；设计稿 §2.3「S-PB 的规格」、E2 卡、§7 S-PB 一行、§12.7）。本节按实现步骤追加：第 1–8 条（2B-1）是 S-PB 的两个入口与测试；第 9–14 条（2B-2）把它接进场景、作业与判定器：作业输入（指纹、请求字节、预制字节，SM-6）、覆盖清单的 `stub_modules`（D16）、C10 的 S-FE 模块核对、`caplaunch.smoke` 角色与补充冒烟场景；第 15 条是 R1–R3 自审修复新增的实现约定（其余修复直接改在第 1–14 条里）。正文里被本节覆盖或补充的句子各加了「见勘误 B2B-n」。新增文件格式 `capkit.prebaked/1`、`capkit.job_input_record/1`、`capkit.httpprobe_started/1`（第 15 条）；`capkit.smoke` 升为 `/2`（第 12 条）；新增字段：场景的 `job_inputs`、`launch.json` 的 `job_inputs`、覆盖清单的 `stub_modules`、`launches.json` 的 `job_input_failed` 与 `job_inputs`、`capjobs.submit` 时间预算的 `job_inputs_s`；其余格式不变。

1. **§1.1 / §1.2 包与依赖**：`capstubs` 新增 `capstubs.prebaked`（预制响应字节的生成与读取）与 `capstubs.prebaked_server`（S-PB）。`capstubs.prebaked_server` import `capgrpc` 的运行时（`capgrpc.common` 的 `ToolRuntime`、`bind_server`、`make_server`、`combine_launch_specs`、`write_inprocess_ready`，以及 `capgrpc.inject_slot_decls`），`capstubs.prebaked` 的命令行 import `capgrpc.reqbytes.import_pb2`。这是 §1.2「包之间的依赖只允许指向 `capcommon`」之外的又一个例外，与已有的 `capselfcheck.relay` → `capgrpc.common` 相同（那一处正文没有登记，这里一并登记）：S-PB 的节拍、待发队列、档界标记与 nullgrpc 是同一套运行时，另写一份会有两个实现。部署上的后果：起 S-PB 的启动，覆盖清单的 `--packages` 必须同时含 `capstubs`、`capgrpc`、`capcommon`。S-PB 的两个模块都不在模块导入时 import `tensorrt_llm`：生成器只在运行时按 `--pb2-module` 导入产品模块，桩只在给了 `--sfe-module` 时按路径导入 S-FE 模块。
2. **§13.1 S-PB 契约（新增）**：
   - **预制文件 `capkit.prebaked/1`**（`python -m capstubs.prebaked`，容器内运行）：`{"schema","method","gate":"product","token_ids","prompt_tokens","finish_reason":"FINISH_REASON_LENGTH","finish_reason_value","short_tokens","source","streams":[{"max_tokens","tokens_per_msg","stream_interval","messages":[{"tokens","sha256","b64"}…],"finished":{"sha256","b64"}}]}`。每个（N、k、s）一条整流：`messages` 是 `expect.msg_tokens(N, k, s, "product")` 各条消息的 `TokenOutput` 响应（`output_index` 0，`TokenInfo` 只有 `token_id`，不填 `text`；token 号沿整条流循环取 `token_ids`），`finished` 是 `GenerationFinished`（`output_index` 0，`LENGTH`）加 `usage`（`prompt_tokens` = 文件的 `prompt_tokens`，`completion_tokens` = N，`total_tokens` 为和）。所有响应都不带 `request_id`（产品客户端只在它非空时才核对，`openengine_client/stream.py` 的 `_validate`）。需要哪些（N、k、s）：档位表各负载档的 N，加上档位表里与 `--short-tokens` 给出的每个 `short.req` 量对应的缩短后的 N（`max(N − d, 1)`，与 nullsse 相同）。
   - **桩读文件时的检查**（生成器写完也用同一套再读一遍）：sha256、消息条数与每条 token 数等于 `expect` 的 `product` 闸门、每条恰好一个事件且 `output_index` 有值、token 消息不带 `usage`、finished 的原因与 `usage` 三项、没有 `request_id`。生成器另用 `check_field_numbers` 核对产品模块的字段号与桩的线格式读取器（`capstubs.prebaked.FIELDS`）一致。
   - **请求**：Generate 的方法路径取预制文件的 `method`，用恒等（反）序列化注册（勘误前的 U-1 是同一个前提）；请求字节只读 `request_id` 与 `stopping.max_tokens`（protobuf 合并语义：都取最后一次出现的值），不建 protobuf 对象。每请求的 N 取 `max_tokens`；k、s 不在请求里，取档位表各负载档共同的（k、s）——负载档的（k、s）不唯一 ⇒ 启动拒绝（退出码 3）；K 轴按 k 分启动。设计稿 §12.7 的「k 取 MTP 实测的 3 与 4 混合」本步不支持。
   - **每请求的消息**：`expect.msgs_per_request(N, k, s, "product")` 条 token 消息 + 1 条 finished 消息。节拍同 nullgrpc `--pacing beat`：第 j 条 token 消息在步 `first_step + msg_step_offsets(N, k, s, "product")[j]` 入队，finished 与最后一条 token 消息同一步入队；`first_step = step_at_or_after(接纳时刻)`。没有 `asap` 模式，档位表必须有 `beat_ns > 0` 的档。
   - **拒绝**（`INVALID_ARGUMENT`，`rejects` 与 `streams_rejected` 按事件档）：`not_bytes`、`malformed`、`no_max_tokens`（未设或为 0）、`no_prebaked_stream`、`request_id_not_in_schedule`。其他方法路径 ⇒ `UNIMPLEMENTED`。
   - **前端模式**（`--sfe-module PATH`，D16：探针树基线里零改动的 `fake_openengine.py`）：按路径以私有名导入；本桩收到的 S-FE 选项（`--sfe-profile` 对应 S-FE 的 `--profile`；`--model`、`--max-context-length`、`--vocab-size`、`--eos-token-ids`、`--profile-json` 同名，指纹 JSON 的 `sfe_args` 可原样拼入）交给它自己的 `_parse_args` 与 `_profile_from_args`，建 `FakeEngine(profile, tokens_per_step=1, step_delay=0.0, record=False)`（不启动它），把它的 `_Control`、`_Health` 注册在同一个 server 上。文件只读一次，执行的就是算 sha256 的那份字节。给了 S-FE 选项而没给 `--sfe-module`、模块读不到、执行它时抛出任何异常（缺产品包、语法错、运行期错误）、它的解析器拒绝这些选项、缺 S-FE 入口、建 `FakeEngine` 或服务失败、把 `_Control`、`_Health` 注册到 server 时抛异常（从取入口到注册完，其间任何异常）⇒ 退出码 3。模块 sha256 与传给它的 argv 写进就绪文件里进程内工具的自报 `control`（`info.workers[0].info.control`，与 nullgrpc 等进程内工具同一位置，第 15 条）与每档自报的 `control`（C10 的核对见第 11 条）。
   - 与设计稿 §2.3 保真度表的对应：F3（字节数）由预制文件决定；F5 所有流回同一串 token；F7、F8 不实现；另有三处设计稿没列的差别：`usage.prompt_tokens` 是文件里的固定值，不随请求的 prompt 变；`TokenOutput.text` 不填（与 S-FE 相同）；没有 prompt 事件（S-FE 只在请求 prompt logprobs 时发）。
3. **§11 新增两行**：
   - `python -m capstubs.prebaked_server`（S-PB）：通用、快照、档位、`--loop`、`--ready-file`、`--port-file`、`--inject`（可选）；`--host`、`--port`、`--workers 1`（只能为 1；`expected_procs` = 1）、`--prebaked PATH`、`--lag-period-us INT`；可选 `--schedule PATH`、`--max-stream-records N`（缺省 `count_schedule_rows`，不含时间表之外的请求，前端模式下由场景给足，见第 13 条）、`--lag-records-capacity N`、`--max-unrequested-s N`（合入时补，见勘误 M-SPB-1）、前端模式的 `--sfe-module` 与 S-FE 选项。位点 `send.msg`、`beat.step`、`short.req`；`backlog.pending_msgs`、`backlog.active_streams`、`backlog.yielded`（counter）。输出：每档 `summary.json`、`steps`、`streams`（§5.4.2）、`lag`。退出码 0 / 3 / 4（绑定失败）。解析器不接受缩写的长选项。
   - `python -m capstubs.prebaked`：`--pb2-module M --tier-table T --token-ids LIST --prompt-tokens P --out PATH`，可选 `--short-tokens D`（可重复）、`--method`（缺省 `/openengine.v1.Inference/Generate`）。导入产品模块时抛出任何异常（`capgrpc.reqbytes.import_pb2`，不只是 ImportError）⇒ 退出码 3。
4. **§4.4 / §5.4.2 S-PB 的 `streams`**：`req_id` = 十进制的 `request_id`（`capgrpc.load --request-format openengine` 写 `str(req_id)`），不是规范十进制（空、带符号、前导零、非 ASCII 数字、超过 18 位）时记 −1 并计 `request_id_not_decimal`。`n_msgs` = 实际发出的 gRPC 消息数，**含 finished 消息**；`tokens` = 这些消息的 token 数（finished 计 0）。`tier_seq`：给了 `--schedule`（必须由同一份档位表生成）时取该 `req_id` 在时间表里的计划档，`request_id` 不在时间表里 ⇒ 拒绝；不给时取**接纳时刻所在的档**（`meta.tier_seq` 与每档自报的 `tier_attribution` 写 `schedule` / `event`）。后者偏离 §4.4「请求自带的计划档」：OpenEngine 请求里没有档号，前端模式下请求号由前端生成，对不回时间表。应有消息数不受影响（N 来自请求，k、s 在一次启动内不变）。同一原因，前端模式下 `streams` 的 `req_id` 对不回负载器的 `req_id`：判定器 X2 的消息层（`symptoms.msg_lag_from_chunks`，`pilot.py` 用同一函数）按 `req_id` 连接负载器的块行与桩的 `streams`，对这类档恒为 NOT_RUN（缺口见第 8 条⑤）。
5. **§5.4.1 S-PB 的 `counts`**：`streams_accepted`、`streams_completed`、`streams_cancelled`、`streams_truncated`、`streams_rejected`、`first_step_bumped`、`msgs_sent`、`bytes_sent`、`finished_sent`、`tokens_sent`、`short_requests`、`request_id_not_decimal`；另有 `beat`、`yielded_event_time`、`rejects`、`method`、`prebaked`（路径与 sha256）、`tokens_per_msg`、`stream_interval`、`tier_attribution`、`control`。对账：S-PB 一跳的消息数 = `summary.counts.msgs_sent`，与 `capgrpc.load` 的 `summary.msgs.msgs` 同口径（都含 finished）；按 `expect` 写 `messages` 链（`gate: product`）时，finished 消息不在 `msgs_per_request` 里：S-PB 与读者两跳各加一条 `per_request: 1` 的修正（§5.8 的修正按跳加）。（合入 main 时每档自报加 `request_id_samples`，见勘误 M-FT1-2。）
6. **§7 / §8.2 S-PB 的位点与积压槽**：位点 `send.msg`（每条消息交给 gRPC 之前，`busy` / `sleep`）、`beat.step`（每步，`busy` / `sleep`）、`short.req`（每个通过全部检查的请求一次，`short`）。与 nullgrpc 不同，S-PB 在启动时按位点核对模式（不支持的模式 ⇒ 退出码 3，而不是在第一次经过时抛错）。积压槽与节拍源相同（`backlog.pending_msgs`、`backlog.active_streams`、`backlog.yielded`），另有 `self.rejected`。
7. **验证**：`tests/capstubs/`（`test_prebaked.py`、`test_prebaked_server.py`）：线格式读取器以真 protobuf 解析为准绳；预制文件用 pb2 测试替身与探针树里产品的 `_generated/generation_pb2.py`（按路径作私有包导入）各生成一次，后者按产品客户端的 `_validate` 规则逐条核对；本地 e2e 是 `capgrpc.load --request-format openengine`（弱读者形态）读 S-PB；前端模式用替身 S-FE 模块。变异 M1～M11 全部被测试抓到（`_local/b2-spb/mut/summary.txt`，不入库）。
8. **§12 已知缺口与未核对（新增）**：① 真 `fake_openengine.py` 的 `_Control` / `_Health` 挂在未启动的 `FakeEngine` 上能否答真前端的启动检查、`Abort` 找不到 S-PB 的流（它们不在 `FakeEngine._active` 里）时前端怎样处理、`GetLoad` 恒报 0 个在途请求（同一原因）时前端是否依赖它、真前端是否接受没有 prompt 事件与 `text` 的流——都只能在容器里验证，属 1.4 补充冒烟【未核对】。② 给了 `--schedule` 时，流若归到一个还没开始就被停掉的档，该档没有自报（`streams` 行仍在该档目录里）；按时间表发送的负载器不会出现这种情况。③ 不给 `--schedule`（前端模式）时按接纳时刻归档，档界附近的请求会落到下一档：经过 S-PB 的对账链在前端模式下用 `scope: launch`。④ `short.req` 的经过次数，以及被选中时的 `applied`，都含「通过全部检查、但缩短后的 N 没有预制流」而被拒绝的请求（这时 `applied` 与 `short_requests` 不等）；启动时已核对档位表的每个 N 都有缩短后的流，只有档位表之外的 N 才会这样。⑤ 前端模式下负载器与 S-PB 之间没有连接键（第 4 条），设计稿 E4/E5 的主桩正是「S-PB + 真前端」：E4/E5 之前须定一种连接方式（按接纳顺序或时刻对齐，或让前端透传负载器的请求号），并重做 X2 的自测样本；1.4 补充冒烟（`e0s-spb-frontend`）从 S-PB 的 `streams` 读出真前端发来的 `request_id` 的形态，作为定连接方式的输入【未核对】。
9. **§2.2 / §11 作业输入 `job_inputs`（SM-6；D15）**：只有容器里才造得出、而启动又要用的文件（S-FE / S-PB 的指纹、S-PB 的预制响应字节、`capgrpc.load --request-format openengine` 的请求字节），由场景的可选顶层段 `job_inputs` 声明，作业在门禁之后按启动生成（`capcommon.jobinputs`、`capjobs.jobinputs`）：
   - **场景**：`"job_inputs": {<种类>: {"args": [...]}}`。种类与生成模块一一对应、写死：`fingerprint` → `capstubs.fingerprint`（`capkit.fingerprint/1`），`request_bytes` → `capgrpc.reqbytes`（`capkit.reqbytes/1`），`prebaked` → `capstubs.prebaked`（`capkit.prebaked/1`）。`args` 只许用 `{param:<params_required 里的名字>}` 与 `{input:tier_table|schedule|prompts}` 两种占位符，不许含 `--out`（由生成方追加）。角色参数里的 `{input:<种类>}` 只在场景声明了该种类时才合法（`caplaunch.scenario.INPUT_KEYS` = 三种登记输入 + 三种作业输入），解析为启动树里那份拷贝的绝对路径；`splice_json` 同样可以引用它（例：`{"splice_json": "{input:fingerprint}", "key": "sfe_args"}`，指纹参数与 S-FE 相同，D15）。
   - **生成**（`capjobs.launches`，门禁通过之后、每个启动的绑核表之后、起编排器之前）：对启动场景声明的每个种类，在本作业环境里跑 `<{PY}> -m <模块> <args，占位符代入该启动 argv 的 --param 值与 <W> 里的登记输入> --out <作业目录>/job_inputs/<启动树名>/<种类>.json`，旁边写 `<种类>.json.record.json`（`capkit.job_input_record/1`：`kind`、`module`、`argv`、`inputs`（args 引用到的登记输入 → 生成前现算的 sha256）、`rc`、`out_sha256`（只在退出 0、文件存在且 schema 正确时才填）、`log`、起止时刻）与 `.log`。场景有 `job_inputs` 键时，生成之前先核场景里要读的部分（`capcommon.jobinputs.check_scenario`：该段过 `validate_section`，种类为空或有未知种类都拒；`params_required` 是名字列表；`tier_table`、`schedule`、`prompts` 是 `<W>` 内的相对路径；与 `caplaunch.scenario.validate_scenario` 同一口径）；没有这个键的场景不生成，也不在这里解析 `--param`（`--param` 写错由编排器拒绝，记 `config_refused`）。场景核对不过，或任一种类失败（生成器退出码非 0、超时、没有输出、schema 不对；或在生成器运行之前就失败：输出已存在、缺 `--param`、登记输入读不到，这种没有记录文件、`rc` 为 null）⇒ 该启动不起、记 NOT_RUN（原因以 `skipped: job inputs:` 开头），`launches.json` 的启动项带 `job_input_failed: true` 与 `job_inputs`（此前已成功的种类与失败种类各一项：`kind`、`rc`、`out_sha256`、`log`；场景核对不过时还没有种类，不带这一项），顶层另有计数 `job_input_failed`；`capjobs.launches` 因此以 3 退出（与 `config_refused`、`pin_table_failed` 同一条，勘误 R5-3）。生成器与绑核表都经 `capjobs.launches.run_helper` 运行（第 15 条）：作业在这期间收到 SIGTERM（信号先到生成器、生成器先退出也算，第 15 条「停止标志先于退出码」）⇒ 杀掉生成器的进程组，该启动记 NOT_RUN（`skipped: the job received SIGTERM during the job inputs`），不记 `job_input_failed`，其余启动跳过。成功则给编排器追加 `--job-input <种类>=<路径>`（每个种类一个；计划 argv 里出现 `--job-input` 即拒绝，`launchplan.RESERVED_FLAGS`）。每个生成最多 `launchplan.JOB_INPUT_TIMEOUT_S`（600 s），计入启动前的剩余时间检查（`budget_s + kill_grace_s + margin_s + launchplan.in_job_bound_s`，后者 = 绑核表 120 s + 600 s × 该启动的种类数）与 `capjobs.submit` 的时间预算（新增 `job_inputs_s`，`pin_tables_s` 不变）；提交端预检计划 argv 时按场景声明的种类补示例 `--job-input`（`submit.check_plan_argv`），有 `job_inputs` 键的场景先过同一个 `check_scenario`，不过即拒绝提交（退出码 3）。按启动、不按作业生成：请求字节与预制字节随启动的档位表、prompts 变；指纹也按启动生成，代价是每个启动多一次 `import tensorrt_llm`。`jobs/capkit_inner.sh` 不变（生成在它调起的 `capjobs.launches` 里），只改了头注释。作业之外可以用 `python -m capjobs.jobinputs --workdir W --scenario <relpath> --python PY --out-dir DIR [--param K=V]...` 造同样的文件与记录，它打印对应的 `--job-input` 参数。
   - **编排器**（`caplaunch --job-input KIND=PATH`，可重复）：种类集合必须等于场景声明的集合；`PATH` 是绝对路径；文件的 `schema` 是该种类的；记录（`PATH.record.json`）过 `capcommon.jobinputs.check_record`：种类与模块对得上、`rc` 0、`out_sha256` 等于文件现算 sha、`inputs` 里每个登记输入的 sha 等于本启动登记的 sha。任一不过 ⇒ 退出码 3、起进程之前、不写树（勘误 R5-3 的 `config_refused`）。文件与记录按读到的字节交给 `capcommon.jobinputs.load_verified`（解析、schema、sha、`check_record`；判定器用同一函数，第 10 条）。`check_record` 不核对记录里的 `argv` 是否用了本启动的 `--param` 值：作业路径里记录由 `capjobs.launches` 按本启动的 argv 生成，天然一致；只有手动用 `python -m capjobs.jobinputs` 造文件时可能不符。通过后文件与记录原样写进 `raw/env/job_inputs/<种类>.json` 与 `<种类>.record.json`，`launch.json` 新增 `job_inputs`：`{<种类>: {relpath, sha256, size, record_sha256, module, source}}`（没有作业输入时为 `{}`）。
   - **SM-6 的处理**：`configs/e0` 的 `e0a-frontend-sfe`、`e0c-frontend-http` 改为声明 `job_inputs.fingerprint`（`--tokenizer {param:tokenizer_dir} --trust-remote-code false --tokenizer-mode auto`：后两项写死为前端角色 argv 隐含的设置——不带 `--trust_remote_code`（`trtllm-serve` 缺省 false）、不给 `--config`（`tokenizer_mode` 为 auto）；给前端加这两项设置时，生成器 `configs/gen_e0_configs.py` 的 `FP_INPUT` 与 `gen_e0s_configs.py` 的同一处必须同改），S-FE 角色用 `{input:fingerprint}` 拼参数；参数 `fingerprint_json` 去掉，不新增参数。已登记在 `<W>` 里的旧场景副本不受影响。
10. **§10.1 判定器的作业输入核对**（`capjudge.inputs._load_job_inputs`，在档位表与时间表之后、对账段之前）：`launch.json` 的 `job_inputs` 种类集合必须等于登记场景文件（`register.json files.scenario`）声明的集合；每项的 `relpath` 必须是 `raw/env/job_inputs/<种类>.json`，树里的文件与记录现算 sha 等于 `sha256`、`record_sha256`，再过 `capcommon.jobinputs.load_verified`（文件 `schema` 是该种类的、记录过 `check_record`；与编排器同一函数；登记输入的 sha 取 `launch.json` 的 `tier_table`/`schedule`/`prompts`）。任一不过 ⇒ `INPUT_INVALID`（13）。没有 `job_inputs` 键的 `launch.json`（本批之前写的）按「没有作业输入」处理，场景也没有声明时照常通过。
11. **§6.6 / §2.5 覆盖清单的 `stub_modules` 与 C10 的 S-FE 模块核对（D16）**：
   - `capprobe.overlay` 新增可选 `--stub-module KEY=PATH`（可重复，每个 KEY 一次；KEY 只有 `sfe`）：PATH 是 `tensorrt_llm/` 之外的 `.py` 仓库相对路径（如 `tests/unittest/grpc/openengine_client/fake_openengine.py`），取**基线提交**的 blob 零改动写到 `<--export-dir>/<PATH>`（任何档位都要 `--export-dir`；不读探针树的工作文件），清单新增 `stub_modules: [{key, repo_path, commit, src, sha256, size}]`（没给时为 `[]`）。
   - 快照：`capjobs.snapshot.overlay_entries` 把 `stub_modules` 一并列入（拷进快照、提交端与建快照时核 sha；条目必须恰好有上述六个键、`key` 不重复、`src` 是代码根相对路径）。作业里的 `capjobs.overlay` **不**把它们装进 site-packages，只核对 `<快照>/<src>` 的字节等于清单（`overlay_verify.json` 里 `list: stub_modules` 的项，带 `path`）。所以 S-PB 的 `--sfe-module` 在计划里写 `--param sfe_module={SNAP}/<src>`。
   - C10 新增一项（每个 `module` 为 `capstubs.prebaked_server` 的角色一项，计入 C10 的应检与已检数）：角色参数里有 `--sfe-module`（前端模式）⇒ 它在就绪文件里自报的 `control.sha256`（`info.workers[0].info.control`，判定器经 `LaunchData.ready_tool_info` 读，第 15 条）必须等于覆盖清单 `stub_modules` 里 `key = sfe` 的 `sha256`；没有 `--sfe-module` ⇒ `control` 必须为 `null`。不符、清单里没有 `sfe`、就绪文件缺失、`info.workers` 不是恰好一项、或自报里没有 `control` 键 ⇒ C10 `NOT_RUN`（与版本记录的问题同一处理，D11；两种模式都一样，读不到不算通过）。第 2 条「C10 的核对属本子批次后续步骤」由本条取代。
12. **§11 新增与修改的入口**：
   - 场景 `TOOL_TABLE` 新增 `capstubs.prebaked_server`（循环、档位、快照、就绪文件、端口文件、`--workers`；另有 `single`：场景里 `workers` 与 `expected_procs` 都必须为 1）与 `caplaunch.smoke`（与 `caplaunch.httpprobe` 相同：只有身份、`--run-dir`、`--stop-grace-s`、`--cpus`，没有就绪文件，就绪用 `file_exists`）。
   - `caplaunch.smoke` 作为启动角色：新增可选 `--run-dir`、`--stop-grace-s`、`--cpus`（只记录）、`--items LIST`（项组，取自 `env,d13,d14,read,cores,tree`，按这个固定顺序跑；缺省全部）、`--hold`（写完 `smoke.json` 后等 SIGTERM；SIGTERM 处理器在跑项之前装好），`--result-dir` 是 `--out` 的别名（`--out` 是编排器管的参数）。输出格式升为 `capkit.smoke/2`：`identity` 改为身份对象（含 pid；原来的身份参数列表移到 `identity_argv`），新增 `item_groups`、`role`。这样它写在 `raw/` 下的 `smoke.json` 能过判定器的身份核对。
13. **补充冒烟场景 `configs/e0s/`（RUNBOOK-stage2 1.4 第 1、2 项；第 3 项见勘误 FT1-14；生成器 `configs/gen_e0s_configs.py`）**：
   - `e0s-import-probe`：第 1 项。`importprobe` 角色（`caplaunch.smoke --items env`，配置 `e0s-smoke.json`：`import tensorrt_llm, uvloop, grpc_health.v1.health_pb2`，带与不带 compat 路径各一次）+ `capload → capload.nullsse`（一档低速 warmup 与一个带门的 drain：档位表要求最后是带门的 drain）。
   - `e0s-spb-frontend`：第 2 项。`importprobe` → S-PB（前端模式：`--prebaked {input:prebaked}`、`--sfe-module {param:sfe_module}`、`--sfe-profile trtllm-like`、指纹拼自 `{input:fingerprint}`；不给 `--schedule`，流按接纳时刻归档）→ 真前端（`--openengine-endpoint {addr:spb}`）→ `httpprobe`（`/v1/completions`、`/v1/chat/completions` 各一条，请求取 `configs/e0/e0c-httpprobe.requests.json`）→ `capload` 的 warmup 与带门的 drain（门读 capload 在途数与 S-PB 的活跃流、待发消息）。S-PB 按节拍发、第一步在原点，所以 httpprobe 的请求在原点之前发出、在原点之后才得到回答：它的就绪条件是 `--started-file`（`raw/httpprobe.started.json`，发第一个请求之前写，第 15 条），`--result`（`raw/httpprobe.json`）照旧在回答收完后写；`httpprobe_timeout_s`（`http.client` 每次阻塞读的超时）须不小于第一个请求到原点的等待（上界 `origin_lead_ms`）加若干 `beat_ns` 与余量。S-PB 的 `--max-stream-records {param:spb_max_stream_records}` 须不小于时间表行数 + httpprobe 的请求数（缺省容量只算时间表行，httpprobe 的请求会溢出）。`origin_lead_ms` 要放得下原点前串行的启动：importprobe 的两次冷 `import tensorrt_llm`（各最长 600 s）、S-PB 导入 S-FE 模块、真前端启动；E0 计划用的 6000 ms 不能照抄（`configs/e0s/README.md`），1.4 冒烟按实测再定。作业输入 `fingerprint` 与 `prebaked`（`--pb2-module tensorrt_llm.grpc.openengine._generated.generation_pb2`）。档位表 `max_tokens` 固定为 16 = httpprobe 两个请求的 `max_tokens`（S-PB 拒绝没有预制流的 `max_tokens`）；`beat_ns` 必须 > 0。
   - 这两个场景里 `httpprobe` 与 `importprobe` 不写就绪文件、也没有快照，没有版本记录，所以判定器的 C10 对它们是 `NOT_RUN`（与 E0(c) 相同）；冒烟结论读 `raw/smoke/smoke.json`、`raw/httpprobe.json` 与 `caplaunch.smoke --tree`，不读判定。第 3 项（S-NL 最小版）属 2C-min（S-NL 见勘误 FT1-1）。
14. **§12 已知缺口与未核对（新增）**：① 三个生成器在容器里的真实运行（`capstubs.fingerprint` 要 `import tensorrt_llm` 与真 tokenizer；`capstubs.prebaked`、`capgrpc.reqbytes` 用产品的 `generation_pb2`）、真 `fake_openengine.py` 被 S-PB 加载、真前端连 S-PB、两个 `e0s` 场景——本机都没有运行，属 1.4 补充冒烟【未核对】。本地的作业路径 e2e 用 pb2 测试替身跑了真 `capstubs.prebaked` 与 `capgrpc.reqbytes`、真 `capjobs.jobinputs` 与 `caplaunch`，Generate-only 与前端模式（替身 S-FE 模块）各一次（`tests/capstubs/test_spb_job_path.py`）；它给子进程设了 `PYTHONPATH`，作业里则 `unset PYTHONPATH`、capkit 包从 site-packages 导入，导入路径与作业不同（作业路径用 `jobs/capkit_inner.sh` 的同环境另跑，见 `_progress/b2-spb.md`）。② 每个启动都重新生成指纹（多一次 `import tensorrt_llm`），同一作业里各启动的指纹文件应当相同，但没有跨启动核对。③ S-PB 只核它读到的字节的 sha 并自报，不知道覆盖清单；是否等于清单的 `sfe` 由判定器 C10 判（第 11 条）。`--sfe-module` 指向另一个带同样入口的 S-FE 文件时 S-PB 照样能起，只有 C10 那一项会报出来（缺入口的文件以退出码 3 拒绝，第 2 条）；而 `e0s` 场景的 C10 因 `httpprobe`、`importprobe` 没有版本记录本来就是 NOT_RUN（第 13 条），所以要看 C10 的 NOT_RUN 原因列表里有没有 `role spb:` 开头的一项。路径指向不存在的文件时 S-PB 起不来（退出码 3）。④ 真前端可能把 `max_tokens` 截到上下文剩余长度，截后的 N 没有预制流 ⇒ S-PB 以 `no_prebaked_stream` 拒绝；属 1.4 补充冒烟【未核对】。
15. **R1–R3 自审修复新增的实现约定**（`code-review/self-review-compare/capkit-b2spb-J17871/R1/triage.md`、`R2/triage.md`、`R3/triage.md`；第 1–14 条里被改的句子直接改在原处）：
   - **进程内工具的就绪文件**：`info` 的形状由 `capcommon.lifecycle.inprocess_ready_info` 定义（`{tool, workers: [<该进程一项>], threads}`，工具自报在 `workers[0].info`），`capgrpc.common.write_inprocess_ready` 用它写；读方用同处定义的 `inprocess_tool_info`（判定器经 `LaunchData.ready_tool_info`），不手写路径。`workers` 不是恰好一项、或该项没有 `info` 对象 ⇒ 读不到（C10 记为问题）。
   - **`caplaunch.httpprobe --started-file PATH`**（可选）：发第一个请求之前原子写 `capkit.httpprobe_started/1`（带本角色身份与 pid，另有 `addr`、`n_requests`），供「回答在原点之后才来」的场景当就绪文件（`e0s-spb-frontend`，第 13 条）；不给时行为不变（`e0c` 不用）。
   - **`capjobs.launches` 的辅助命令**：绑核表与作业输入的生成器都经 `run_helper` 运行（独立会话，每 `POLL_S` 轮询）：超时 ⇒ 杀整个进程组、按超时失败（时长上界与 `subprocess.run` 相同）；作业收到 SIGTERM ⇒ 杀整个进程组，该启动记 NOT_RUN（`skipped: the job received SIGTERM during the pin table|job inputs`），不记 `pin_table_failed` / `job_input_failed`，其余启动跳过，`launches.json` 照写（原先阻塞在 `subprocess.run` 里，最长 120 s + 600 s × 种类数不响应 SIGTERM）。`capjobs.jobinputs.generate` 因此有 `run=` 钩子（缺省 `subprocess.run`，命令行不变）。**停止标志先于退出码**（R3）：作业的 SIGTERM 可能先到子进程（Slurm 的 proctrack/cgroup 给步骤内所有进程发信号），所以子进程退出之后先看作业是否已收到 SIGTERM（`launches.stopped_after_exit`；子进程以 −SIGTERM 退出而本进程还没看到信号时，等一个 `POLL_S` 再看一次）。辅助命令退出时作业已收到 SIGTERM ⇒ 不论退出码，按上面的信号停止记（NOT_RUN，不记 `pin_table_failed` / `job_input_failed`，不因此以 3 退出）。编排器（`run_with_budget`）退出时作业已收到 SIGTERM、且退出码不是 0 或 5 ⇒ 记 `terminated`（NOT_RUN，原因以 `terminated` 开头，不记 `config_refused`）；退出 0 或 5 的仍以它自己的 `STATE` 与退出码为准。作业没收到 SIGTERM 时，子进程被别的原因以 −SIGTERM 杀死，仍按它自己的失败记。场景读不到时原因写 `skipped: scenario: …`（原先写成 pin table），计数仍归 `pin_table_failed`。
   - **共用的场景模板解析**：`capcommon.templates`（`PLACEHOLDER_RE`、`placeholders`、`parse_params`）由 `caplaunch.scenario`、编排器的 `--param`、`capcommon.jobinputs`、`capjobs.jobinputs.params_from_argv` 共用；`--param` 的错误消息统一为编排器原来的两种（`must be KEY=VALUE`、`given twice`）。
   - **S-PB 的 `schedule_sha256`**（R2）：给了 `--schedule` 时，就绪文件里的工具自报与每档 summary 的 `schedule_sha256` 是它读的时间表文件的 sha256（`capcommon.hashing.sha256_file`，与 `capgrpc.load` 同法）；不给时为 `null`。判定器不核 S-PB 的这个字段，它只供事后核对 S-PB 读的是哪份时间表。
   - **验证**：`tests/judge/test_c10_stub_module.py` 的就绪文件用真写方生成；`tests/capstubs/test_prebaked_server.py` 把 S-PB 两种模式真实写出的就绪文件交给 C10 的 S-PB 一项；作业路径 e2e 增加前端模式；`tests/jobs/test_job_inputs.py` 覆盖生成期间的 SIGTERM（只发给 `launches`、先发给生成器及其子进程再发给 `launches` 两种投递）、`run_helper` 与 `run_all` 里的「停止标志先于退出码」、`run_helper` 超时、生成前失败保留记录；`tests/jobs/test_container.py` 覆盖编排器在写 `raw/launch.json` 之前先被作业的 SIGTERM 杀死；`tests/capcommon/test_templates.py`；`tests/launch/test_smoke.py` 的 `--started-file` 先于第一个请求。

### 合入 b2-spb 到 main（勘误 M-SPB-n，2026-11-15）

`capkit-b2-spb`（子批次 2B，上一节）以 `--no-ff` 合入 main（合入前 main 已含 SM-16、SM-17/18、2A+2E、SB-B1、SB-A 各节）。本节只记合入时为两边对齐而改的地方；冲突逐条的处理见 `_progress/merge-spb.md`。勘误各节按合入顺序排列：main 原有各节在前，2B 一节在后；两边的勘误编号（`B2-2A-n`、`B2-2E-n`、`SB-B1-n`、`calib-v2-An`、`B2B-n`）互不重叠，正文新增的 §13（2B 的 S-PB 契约，勘误 B2B-2）与 §14（2A+2E 的分析脚本）也不重号。

1. **S-PB 接 SM-18 的服务端队列选项**（勘误「Fix SM-17 … SM-18 …」第 2 条最后一项点名的事）：`capgrpc.common.make_server` 在 main 上第二个参数 `max_unrequested_s` 必填，2B 分支上的 S-PB 仍按旧签名调用，合入后起不来。合入时改为与 `capselfcheck.relay` 相同：`capstubs.prebaked_server` 接可选 `--max-unrequested-s`（`capgrpc.common.add_server_args`），启动时用 `resolve_max_unrequested(值, 档位表)` 定值（缺省取 `request_timeout_ns` 与档位表长度中较长者、按整秒向上取整；显式值短于 `request_timeout_ns` 或越界 ⇒ 退出码 3），`make_server(reuseport=False, max_unrequested_s=…)`；实际值写进就绪文件自报（`info.workers[0].info.grpc_server_options`）与每档 `summary.json` 的 `grpc_server_options`，判定器不读。`configs/e0s/` 的场景不给这个参数，取缺省。
2. **`capjobs.submit.time_budget` 与 `caplaunch.laddergen` 的装箱核对**（勘误 calib-v2-A 装箱核对一条、B2B-9「生成」一项）：2B 的 `time_budget` 从每个启动 argv 的 `--workdir` / `--scenario` 读场景、按声明的作业输入种类数加 `job_inputs_s`；SB-A 的 `laddergen.packing` 用只有 `budget_s` 的合成计划调它，合入后抛 KeyError。合入时给 `time_budget` 加可选参数 `job_input_kinds`：缺省（None）照旧从场景读（提交端的路径不变）；给整数时每个启动按这么多种类计。`laddergen.packing` 传 0（calibration v2 的场景都不声明 `job_inputs`），所以它算出的 `time_budget_s` 与写进 `ladder.json` 的 `packing` 段与合入前相同。以后 calibration v2 若用声明了作业输入的场景，要改成按该场景的种类数传入。

### Fast track FT1: S-NL, oe_sink, smoke and E2a/E3/E4 scenario templates, product-side injection（2026-11-16）

快轨子批次 FT1（`W/00_plan/FASTTRACK-1117-PLAN.md`；RUNBOOK-stage2-draft.md §2.1 第 4、6 行的 S-NL 部分、第 5 行的补充冒烟场景与第 7 行的 `oe_sink`；设计稿 §2.3「S-NL 的规格」、E2 卡、E3 卡 3a / 3b、E4 卡、§5.2 阶段 0）。本节按实现步骤追加，每步一组编号（FT1-1 起）。正文里被本节覆盖或补充的句子各加了「见勘误 FT1-n」。

**步骤 FT1-snl（S-NL）**：

1. **§1.1 / §11 新入口 `python -m capstubs.null_llm_server`（S-NL）与 `capstubs.nullllm`**：
   - `capstubs.nullllm.NullLlm`：真 servicer 在请求路径上读的 `LLM` 属性（设计稿 D:202）逐个给出：`input_processor`（`supports_token_ids_input` 真、无媒体）、`tokenizer`（不给 `--tokenizer` 时为 `None`）、`preprocess`（媒体请求一律 `ValueError`）、`generate_async`、`_check_health`、`get_stats`（空表）、`shutdown`、`_executor`（只有 `is_shutdown`、`_fatal_error`、`_engine_dead`），`args` 是 `NullArgs`（`max_seq_len`、`max_num_tokens`、`max_batch_size`、`kv_cache_config`（`tokens_per_block` 32、`kv_events_config` 为 `None`）、`guided_decoding_backend`、`reasoning_parser`、`return_perf_metrics`、`enable_iter_perf_stats`，以及 Control 直接读的并行度三项等；别处用 `getattr(…, 缺省)` 读的字段不给，取产品缺省；直接读而缺的字段抛 `AttributeError`，不猜）。模块导入时不 import `tensorrt_llm`。
   - `generate_async` 返回 `NullHandle`：它就是结果对象（与产品 `GenerationResult` 一样，`__anext__` 返回自身），每次 `__anext__` **只取一项**（设计稿 D:198），把该项的 token 追加到累计的 `outputs[0].token_ids`；最后一项之后 `finish_reason` 为 `length`、`finished` 为真；`abort()` 之后下一次 `__anext__` 以 `cancelled` 结束（已排队的项不再交出）；节拍表在流结束前用完 ⇒ `error` 置为说明文字（真 servicer 把它变成 error 事件）。N 取 `sampling_params.max_tokens`（缺或 < 1、`n` ≠ 1、带 `disaggregated_params` ⇒ `ValueError`，servicer 回 `INVALID_ARGUMENT`）。产品 Control 的推理探针（`generate_async([1], max_tokens=1, streaming=False)` 后 `await handle.aresult()`）同样被接受、走同一节拍；这类流不在任何 Generate RPC 里，按事件档计入 `counts.streams_outside_rpc`，不进 `streams`。token 来源是显式参数：`fixed`（`--token-ids` 沿整条流循环）或 `prompt`（请求的 prompt token 号循环；文本 prompt 只在有 tokenizer 时有 token 号）。不保留请求列表。
   - **三种模式**（`--mode`，必填）：`minimal`——产品 server 跑在 NullLlm 上，每个请求的全部项在到达时一次放进队列（不需要节拍，档位表可以 `beat_ns` 为 0）；这是设计稿 §5.2 阶段 0 的最小冒烟（D:203）。`full`——产品 server 跑在 NullLlm 上，全进程一个节拍器按档位表的 `BeatSchedule` 推进（换档即换 Δ，不重启）；第 j 项在步 `first_step + expect.msg_step_offsets(N, k, s, "product")[j]` 入队、带 `msg_tokens(…)[j]` 个 token，`first_step = step_at_or_after(到达时刻 + --prefill-delay-us)`；节拍器不等 servicer，队列在 servicer 的 `yield` 上游、无界。`metronome-only`——同一个节拍器、NullLlm 与队列，不接 servicer、没有 gRPC 流量：按 `--schedule` 每一行的计划时刻开流（N 取该行所在档的 `max_tokens`），每条流一个进程内消费协程逐项取走；这是 E3-3a 的同进程空跑档（设计稿 E3 卡 3a）。三种模式都绑一个端口（`metronome-only` 的 gRPC server 不注册任何方法），角色都有端口文件。
   - **k、s**：与 S-PB 相同取档位表负载档共同的（k、s）（`capstubs.prebaked.uniform_msg_shape`）；`--stream-interval` 只作核对，与档位表不等 ⇒ 退出码 3（真值只有档位表一处）。
   - **已 yield 计数（D17）**：S-NL 在构造产品 `OpenEngineServer` 的那一刻，把 `--server-module`（缺省 `tensorrt_llm.grpc.openengine.server`）里的 `OpenEngineInferenceServicer` 名字换成它的子类，构造完立即换回；子类的 `Generate` 逐条迭代原 `Generate`，每条响应先过 `send.msg` 位点、`backlog.yielded` 加一，再原样交给 gRPC。构造后核对子类恰好被实例化 1 次，否则退出码 3（server 没有按这个名字建 servicer 时，计数会是 0）。这一层包装本身是每消息多一次异步生成器恢复与一次计数，记在被测的 servicer 线程上（测量者开销，RS13）。产品文件一个不改。
   - **按时间表归档**：给了 `--schedule`（由同一份档位表生成）时，`request_id` 必须是时间表里的 `req_id`（十进制），否则包装层在调用原 `Generate` 之前回 `INVALID_ARGUMENT`，按事件档计入 `rejected`；每条流按计划档归档。不给时按到达时刻所在的档（`tier_attribution: event`）。原始请求格式（`capgrpc.reqbytes raw`：没有 `request_id`）对 `full` / `minimal` 不可用：给了时间表时被包装层拒绝，不给时被真 servicer 以「request_id must be non-empty」拒绝（同一个 `request_id` 并发时还会 `ALREADY_EXISTS`）；读 S-NL 一律用 `--request-format openengine`。
2. **§11 命令行**：通用、快照、档位、`--loop`、`--ready-file`、`--port-file`、`--inject`（可选）；`--host`、`--port`、`--workers 1`（只能为 1）、`--mode {minimal,full,metronome-only}`、`--lag-period-us`、`--model`、`--token-source {fixed,prompt}`（`fixed` 时必须、且只有这时才给 `--token-ids LIST`）；可选 `--schedule`（`metronome-only` 必需）、`--prefill-delay-us`（缺省 0；`minimal` 下给非 0 值 ⇒ 拒绝）、`--stream-interval`（只作核对）、`--max-stream-records`（缺省 `count_schedule_rows`）、`--lag-records-capacity`；产品侧（`minimal` / `full`）：`--server-module`、`--tokenizer DIR` + `--trust-remote-code {true,false}` + `--tokenizer-mode {auto,slow}`（三者同给；用产品的 `ModelLoader.load_hf_tokenizer` 加载，与前端相同；`hf_model_dir` = 该目录，Control 用它算指纹）、`--max-seq-len`（32768）、`--max-batch-size`（4096）、`--max-num-tokens`（32768）。`metronome-only` 不接受 `--tokenizer` 与非缺省的 `--server-module`。解析器不接受缩写的长选项。退出码 0 / 3 / 4（`metronome-only` 绑定失败）。**启动即拒绝（退出码 3，起 server 之前）**：`--workers` ≠ 1；导入产品模块时抛出任何异常（本机没有 `tensorrt_llm` 即属此类，消息写明「needs the product server module … (container only)」）；模块缺 `OpenEngineServer` / `OpenEngineInferenceServicer` 或其 `Generate` 不是异步生成器函数；子类没有被实例化恰好一次；`OpenEngineServer` 构造或 `start()` 抛出任何异常（产品绑定端口失败也在其中，不另分 4）；加载 tokenizer 失败；负载档（k、s）不唯一；`full` / `metronome-only` 的档位表没有 `beat_ns > 0` 的档；注入位点不认识或模式不支持；时间表不是这份档位表生成的。场景 `TOOL_TABLE` 新增 `capstubs.null_llm_server`（循环、档位、快照、就绪文件、端口文件、`--workers`、`single`：场景里 `workers` 与 `expected_procs` 都必须为 1），见 §11 场景配置一行。（合入 main 时加可选 `--max-unrequested-s`，只用于 `metronome-only`，见勘误 M-FT1-1。）
3. **§7 / §8.2 位点与积压槽**：位点 `send.msg`（包装层每条响应交给 gRPC 之前；`busy` / `sleep`）、`beat.step`（节拍器每步；`busy` / `sleep`，在该步计时之前）、`short.req`（NullLlm 每开一条流一次；`short`：该流按 `max(N − short_tokens, 1)` 出 token，`finish_reason` 仍为 `length`，与 S-PB 相同）。启动时按位点核对模式（与 S-PB 同一函数 `capstubs.prebaked_server.check_inject_specs`）。积压槽：`backlog.active_streams`（已开、未关的流）、`backlog.pending_msgs`（已入队、未被 `__anext__` 取走的项；每项恰好对应一条 token 消息，finished 消息不在内）、`backlog.yielded`（counter，包装层计数；`metronome-only` 恒为 0）；另有 `self.rpcs/accepted/completed/cancelled/rejected`、`count.items_taken`、`count.beat_steps/beat_late_steps/beat_enqueued`、`count.beat_work_ns`（节拍器每步体的线程 CPU 之和）、`lag.beat_wake`、`lag.loop`、`inject.<site>.*`。清空门可读 `backlog.yielded − 读者 backlog.read`（gRPC 在途消息数，设计稿 §2.6）与 `backlog.pending_msgs`（积压在 `yield` 上游）。
4. **§5.4 输出**：`streams`（`server.streams`：`req_id`（十进制 `request_id`，否则 −1）、`accept_rel_ns`、`first_step`（无节拍为 −1）、`n_msgs`（包装层计的已 yield 响应数，含 finished；`metronome-only` 为 0）、`items`、`tokens`、`tier_seq`；只记有流的 RPC）；`steps`（`beat.steps`：`step`、`planned_rel_ns`、`actual_rel_ns`、`items`、`active`（该步时的活跃流数）、`work_ns`（该步体的线程 CPU）、`tier_seq`；`minimal` 没有这个文件）；`lag`；每档 `summary.json`：`counts`（`rpcs`、`rpcs_without_stream`（servicer 在开流之前就结束的 RPC）、`rejected`、`streams_accepted/completed/cancelled/truncated`、`first_step_bumped`、`msgs_yielded`、`items_taken`、`tokens_produced`、`short_requests`、`request_id_not_decimal`、`streams_outside_rpc`；按流的档归档，`rejected` 与 `streams_outside_rpc` 按事件档）、`beat`（`steps`、`late_steps`、`enqueued`、`work_ns`、`work_share_of_loop_thread` = `work_ns` ÷ 该档 `cpu.loop_thread_ns`、`max_pending`、`wake_lag_ns`）、`yielded_event_time`、`items_taken_event_time`、`mode`、`pacing`、`servicer`、`imported`、`servicer_wrapped`、`model`、`tokens_per_msg`、`stream_interval`、`prefill_delay_ns`、`schedule_sha256`、`tier_attribution`、通用的 `inject`、`loop_lag_ns`、`cpu`、`overflow`。就绪文件里进程内工具的自报（`info.workers[0].info`，勘误 B2B-15）含 `mode`、`imported`（`server_module`、`server_module_file`、`server_class`、`servicer_class`、`tensorrt_llm` 的 `version`/`file`、`protobuf_impl`；`metronome-only` 为 `null`）、`servicer_wrapped`、`token_source`、`token_ids`、`tokenizer`、`null_args`、`llm_id` 等——集群冒烟据此确认导入的是哪一份产品代码。对账：S-NL 一跳的消息数 = `counts.msgs_yielded`，与 `capgrpc.load` 的 `msgs.msgs` 同口径（都含 finished）。（合入 main 时每档自报加 `request_id_samples` 与 `grpc_server_options`，见勘误 M-FT1-1、M-FT1-2。）
5. **§6.1 快照撞名**：S-NL 是 capkit 角色，编排器不给 capkit 角色传 `CAPPROBE_*`（场景校验拒绝角色 env 里的 `CAPPROBE_*`），所以即使覆盖装好了，capprobe 在 S-NL 进程里也不启用，同一 pid 下只有 capkit 自报这一路快照，不会撞名。产品侧位点（G1–G3、F2）属 FT1-inj（结论见勘误 FT1-18、FT1-19）。
6. **§12 已知缺口与未核对（新增）**：① 真 `tensorrt_llm.grpc.openengine.server` 跑在 NullLlm 上——`NullArgs` 是否缺 Control / capabilities / 请求映射直接读的字段、真前端的三项启动检查（指纹在不给 `--tokenizer` 时为空，前端记为「未核对」而不是不符）、`ModelLoader.load_hf_tokenizer` 加载 GLM 副本、子类替换在真 server 模块上生效——本机都没有运行，属 E0 补充冒烟（快轨次序 1）【未核对】；本地全部用 `tests/capstubs/snl_double.py`（按探针树基线的 `servicer.py` / `streaming.py` 在 S-NL 依赖的路径上仿写的 server 模块替身，pb2 用 `oe_double`）。② `metronome-only` 没有负载角色：判定器的 C3、P-SENT 在这种启动里是 NOT_RUN（作业路径 e2e 实测），流数与项数读 S-NL 自己的 `counts`。③ `--prefill-delay-us` 只推迟第一步，不模拟 prefill 占用节拍器；`minimal` 不计步。④ `first_step_bumped` 只在开流的 RPC 上下文里计（`metronome-only` 的消费协程也有上下文）。⑤ 包装层的 `context.abort` 发生在原 `Generate` 之前，这类 RPC 不进 `streams`，也不进真 servicer 的 `generate_counters`。
7. **验证**：`tests/capstubs/test_nullllm.py`（一步一项、累计 token、abort / 截断 / finish、token 来源、拒绝、上下文钩子、不保留请求）；`tests/capstubs/test_null_llm_server.py`（`capgrpc.load` K = 2 进程、openengine 请求字节读 S-NL full：请求数、已 yield 数、项数、token 数、两个位点的心跳逐档对账；原始客户端核节拍时刻；minimal；拒绝、截断与无 `request_id` 的请求；metronome-only；short；Control 推理探针；启动拒绝逐条各一例）；`tests/launch/test_job_inputs_launch.py` 的 S-NL `TOOL_TABLE` 一行。变异 M1～M9（`_local/ft1/mut/summary.txt`，不入库；M7 是等价变异，换成 M7b）全部被抓到。作业路径 e2e（`_local/ft1/e2e`，覆盖清单按 HEAD 生成，`capjobs.submit` → 真 `capkit_job.sbatch` / `capkit_inner.sh` → `capjobs.launches` → `caplaunch` → `capjudge`；替身 server 模块与 pb2 替身装进一次性 venv）：`e2e-snl-full`（`capgrpc.load` 2 进程 → S-NL full）、`e2e-snl-mono`（metronome-only）、`e0a-nullsse` 回归。

**步骤 FT1-sink（oe_sink；设计稿 E3 卡 3b、§2.3「弱读者检验」、§7 `oe_sink.py` 一行）**：

8. **§1.1 / §11 新入口 `python -m capgrpc.oe_sink`（设计稿的 `oe_sink.py`）**：RUNBOOK §2.1 第 7 行写的是 `capstubs/oe_sink.py`，这里放在 `capgrpc`：它是负载角色，是 `capgrpc.loader` 的 `LoadWorker` / `RequestPlan` 的子类，时间表、到达、ramp、档的归属、发送滞后、`requests` / `msgs` 记录与每档 `summary.json` 的结构都与 `capgrpc.load` 相同（`tool` = `capgrpc.oe_sink`），判定器按 `kind` / `measure_class` 与 `loadgen.requests` 把它当负载角色读，输入格式不变。单进程、一条 channel（产品「每前端一条 channel」）。场景 `TOOL_TABLE` 新增一行（循环、档位、快照、就绪文件、`--workers`、`single`；没有端口文件）。
   - 读法：`--tier generate-stream` = `tensorrt_llm.grpc.openengine_client.OpenEngineClient.generate(GenerateRequest, timeout=…)` 返回的 `GenerateStream` 逐项迭代（读 + 流契约校验）；`--tier engine-event-stream` = 经 `tensorrt_llm.serve.openengine_poc.engine_client_impl.OpenEngineEngineClient.generate(EngineRequest)`（每请求一次 `to_generate_request`，每事件一次 protobuf → dataclass）。事件只计数、丢弃。迭代到 `StopAsyncIteration`，结束时调一次 `cancel()`（流已结束时不起作用）。
   - 连接（`--connect`，必填）：`product` = 客户端自己的 `connect()`（Health、GetServerInfo、GetModelInfo；对端必须答 Control，即 S-PB 前端模式或 S-NL；`ready_timeout` = 到原点的剩余时间；失败退出码 4）。`channel` = 用 `connect()` 设的同一组 channel 选项（收发上限 `MAX_MESSAGE_BYTES`）建 channel 并直接构造 `OpenEngineClient`，不发任何 Control 调用；第二档的 `OpenEngineEngineClient` 不经构造函数（它的构造函数只缓存 `server_info` / `model_info`），只挂上这个 client，`generate` 仍是产品自己的方法。两种连接都在原点之前自检一次：用第一个计划请求建一条流，在第一次迭代之前取消（产品不会发出它）；建流抛错，或取消后不是「已取消且未发出」⇒ 退出码 3。
   - 请求：请求字节文件里的模板在原点之前用产品的 `generation_pb2.GenerateRequest` 各解析一次（模板带 `request_id` ⇒ 退出码 3）。`openengine` 格式每请求把 `request_id` 设为十进制 `req_id`（`generate()` 发送时取副本）；`raw` 格式用产品的 `new_request_id()`（带 `--schedule` 的 S-PB 会拒绝这种请求号，不带时接受）。第二档把模板换成 `EngineRequest` 的字段（`model`、`prompt_token_ids`、`max_tokens`、`ignore_eos`、`sampling`），原点之前用产品的 `to_generate_request` 重建，与模板逐字节比较：不等（模板里有 `EngineRequest` 路径不会重建的字段）或模板是文本 prompt ⇒ 退出码 3。不接受 `capkit` 格式。
   - 超时：与 `capgrpc.load` 一样从计划时刻起算。第一档用 gRPC 截止时间（`generate(timeout=…)`）；产品的第二档接口没有截止时间，由驱动在同一时刻 `cancel()` 这条流。两者都记 `timeout_registered`（overload）；每档 summary 的 `oe_sink.timeout_mechanism` 写 `grpc_deadline` 或 `client_cancel`。
   - 失败原因：`OpenEngineRpcError` → `grpc_<状态码>`（`DEADLINE_EXCEEDED` 且设了超时 → `timeout_registered`）；`OpenEngineEngineError` → `engine_error`；`OpenEngineProtocolError` → `protocol_error`（两者 `class_hint` 为 `non_overload`）；被停止时取消 → `grpc_CANCELLED`；其他异常 → `client_error`（首次出现打印到标准错误）。
9. **§5.4 / §7.4 `--planned-msgs`（新增；`capgrpc.load` 与 `capgrpc.oe_sink` 共用 `capgrpc.msgplan`）**：S-PB、S-NL 发的产品消息不带步号戳，读者从时间表与档位表推出每条消息的计划时刻：计划时刻为 p、计划档为 i 的请求，第 j 条消息的计划时刻 = `BeatSchedule.planned_rel_ns(step_at_or_after(p) + offsets_i[j])`，其中 `offsets_i = expect.msg_step_offsets(N_i, k_i, s_i, "product")`，末尾为 finished 消息再重复一次最后一个偏移（桩在最后一条 token 消息的同一步发 finished）。这是「桩在计划时刻接纳了请求」的开环计划；桩自己的首步按接纳时刻取（S-NL 另加 `--prefill-delay-us`），所以推出的滞后还包含发送滞后、请求在途时间和至多一个节拍的取整，只用来看趋势（X2），不读绝对值。没有计划步的消息（下标超出应有条数、所在档没有节拍、步超出时间表）计入每档 `msg_plan.unplanned`，不进滞后。输出与 `--stamped` 同口径：`lag.msg` 槽、每档 `msg_lag_ns`、`drain.msgs` 记录（`step`、`planned_rel_ns` 为推出值；表头 `meta.planned_source` = `schedule`，`--stamped` 时为 `stamp`），判定器 X2 的消息层（`symptoms.msg_lag_from_msgs`）不改即可读。新增 summary 键 `msg_plan`（`source`：`stamp` / `schedule` / `null`；`unplanned`）。约束：与 `--stamped` 互斥；`raw` 请求格式不可用（每个请求的 max_tokens 在请求字节里，读者不知道）；除 `capkit` 格式外必须给 `--max-msg-records-per-worker`；档位表没有节拍档 ⇒ 退出码 3。缺省关闭，`capgrpc.load` 不加这个参数时行为不变。
10. **§11 `capgrpc.oe_sink` 命令行**：`capgrpc.load` 的读者参数原样照收（`capgrpc.loader.add_reader_args`：`--target`｜`--target-port-file`、`--method`、`--workers`、`--in-process`（接受；本工具总是单进程）、`--channels-per-worker`、`--schedule`、`--request-format`、`--prompts`、`--request-bytes`、`--planned-msgs`、`--msg-sample-mod`、`--lag-period-us`、`--max-records-per-worker`、`--max-msg-records-per-worker`、`--lag-records-capacity`），另加 `--tier {generate-stream,engine-event-stream}`、`--connect {product,channel}`（两者必填）、`--control-timeout-s`（缺省 5.0，与产品缺省相同）、`--slow-read-us N`；不接受 `--stamped`、`--shared-subchannel-pool` 与缩写的长选项。**换读者**：弱读者 = `capgrpc.load --in-process --workers 1 --channels-per-worker 1` 加同一组读者参数；场景里两者只差模块名与 `--tier`、`--connect`（以及可选的 `--slow-read-us`）。启动即拒绝（退出码 3，原点之前）：`--workers` ≠ 1、`--channels-per-worker` ≠ 1、`capkit` 格式、`--slow-read-us` < 1、`--control-timeout-s` ≤ 0；导入产品模块时抛出任何异常（消息写明「container only」），或模块缺所需的名字；`--method` 不是产品 `InferenceStub` 注册的 Generate 路径（用一个只记录路径的替身 channel 读出来）；第 8 条的模板问题；自检失败。连不上或 `connect()` 失败 ⇒ 退出码 4。
11. **§7 / §8.2 位点、槽与自报**：位点与 `capgrpc.load` 相同：`send.req`（每请求，建流之前）、`recv.msg`（产品流每交出一项；`busy` / `sleep` 都阻塞事件循环线程）、`drop.msg`。`--slow-read-us N`：每项计数之后，这条流的协程先 `asyncio.sleep(N µs)` 再读下一项，只拖慢这一条流、不阻塞事件循环；这是设计稿 E3「读者慢」已知样本的开关，每档自报 `oe_sink.slow_read_pauses`（等于该档 `msgs.msgs`）。快照槽 = `capgrpc.load` 的槽去掉 `bytes.recv`（产品对象不带线上字节数）：`summary.msgs.bytes` 为 `null`，`requests` 的 `bytes` 列为 −1。`recv_types` 记每个请求首项的类型名（第一档 `GenerateResponse`、第二档 `TokenDelta`）。每档 summary 另有 `oe_sink`（`tier`、`connect`、`timeout_mechanism`、`slow_read_ns`、`slow_read_pauses`）与 `imported`（`tier`、`generate_method`、导入的各模块及其文件、`stream_class`、`event_stream_class`、`tensorrt_llm` 的 `version` / `file`、`protobuf_impl`；后两项与 S-NL 用同一函数 `capgrpc.common.product_versions`）；就绪文件的工具自报另有 `connect`、`imported`、`slow_read_ns`。两处的 `local_subchannel_pool` 都是 `null`（channel 选项由产品定）。消息口径：一条 = 产品流交出的一项，含 finished；产品跳过的未知事件不计。
12. **§12 已知缺口与未核对（新增）**：① 真 `tensorrt_llm` 包里的导入（`import tensorrt_llm` 会加载整个库：导入耗时、新建的线程对 C13 线程清单的影响、无驱动节点需要 CUDA compat 库）、S-PB 前端模式配真 S-FE 时的 `--connect product`——本机用的是替身包 `tests/capstubs/oe_product.py`（探针树里产品客户端相关文件原样，加空的包 `__init__` 与一个标准库 logger），真包属集群冒烟【未核对】。② 第二档 `--connect channel` 绕过 `OpenEngineEngineClient` 的构造函数；产品改了 `generate` 用到的内部属性时，自检以退出码 3 报出。③ 推出的计划时刻与桩实际的首步可能差一个节拍（第 9 条）；要精确的逐消息滞后，需要在判定器里按 `req_id` 连接桩的 `streams.first_step`（与 SSE 的 `msg_lag_from_chunks` 同法），未做。④ 两档的超时机制不同（第 8 条），只影响过载档。⑤ 本机只测了 oe_sink 读 S-PB；oe_sink 读 S-NL 未运行。
13. **验证**：`tests/capstubs/test_oe_sink.py`（客户端替身 `tests/capstubs/oe_product.py`：探针树里产品客户端相关文件原样拼成的最小 `tensorrt_llm` 包）：两档 × `--connect channel` 读 S-PB（逐档请求数、消息数 = S-PB 的 `msgs_sent`、首项类型、`imported`、`send.req` / `recv.msg` 心跳、`requests` 的 `bytes` 为 −1）；`--planned-msgs` 推出的步号逐条核对（= `step_at_or_after(计划时刻) + 偏移`，且不晚于 S-PB 实际的步）；第二档 `--connect product` 读 S-PB 前端模式（S-FE 为探针树里真的 `fake_openengine.py`）；`--connect product` 读只服务 Generate 的 S-PB ⇒ 退出码 4；换读者（同一组读者参数，只差模块名与 `--tier` / `--connect`，两者的计数与 `drain.msgs` 一致）；两档超时、`raw` 请求被带时间表的 S-PB 拒绝（`grpc_INVALID_ARGUMENT`）与不带时间表时被接受、`--slow-read-us`（同一条流相邻两项的间隔不小于停顿）、`recv.msg` busy 心跳；启动拒绝逐条各一例；`EngineRequest` 往返、Generate 路径读取、`MsgPlan` 边界与自检的单测。`tests/launch/test_job_inputs_launch.py` 的 oe_sink `TOOL_TABLE` 一行。变异 S1～S17（`_local/ft1/mut-sink/summary.txt`，不入库）全部被测试抓到。作业路径 e2e（覆盖清单按提交时的 HEAD 生成，替身包装进一次性 venv）：`e2e-sink-spb`（oe_sink 第二档、`--connect channel`、`--planned-msgs` ← S-PB）与 `e2e-weak-spb`（同一场景只把读者换成 `capgrpc.load`）都是 VALID，判定器 rc 11（NOT_RUN），0 RED，无 INPUT_INVALID，C3、C5、C6（新加的 messages 链）、P-SENT、RECORDS 为 PASS，X2 消息层读到了 `drain.msgs`（没有噪声带 ⇒ NOT_RUN）；`e0a-nullsse` 回归 VALID。

**步骤 FT1-scen（E0 补充冒烟第 3 项与 E2a / E3 / E4 的场景模板；快轨计划 v2 第 1 节次序 1、3–6，设计稿 E2 / E3 / E4 卡）**：

14. **`configs/e0s/` 第 3 项 `e0s-snl-frontend`（RUNBOOK-stage2 1.4 第 3 项；设计稿 §5.2 阶段 0）**：`importprobe` → S-NL `--mode minimal`（产品 server 模块用缺省；`--tokenizer {param:tokenizer_dir} --trust-remote-code false --tokenizer-mode auto`，与 S-PB 一项的指纹作业输入同一组写死的设置；`--model {param:tokenizer_dir}` = 前端的 served model name；不给 `--schedule`，前端的 `request_id` 不是时间表的 `req_id`，流按到达时刻归档；`--max-stream-records {param:snl_max_stream_records}` ≥ 时间表行数 + 1）→ 真前端（与 `e0s-spb-frontend` 的前端角色只差 `--openengine-endpoint {addr:snl}`）→ `httpprobe`（一个 `/v1/completions`，请求文件 `configs/e0s/e0s-snl-httpprobe.requests.json`；S-NL 最小版到达即答，就绪条件是 `--result`，不用 `--started-file`）→ `capload` 一档 warmup 与带门的 drain。档位表 `beat_ns` 为 0（最小版没有节拍）。没有作业输入。生成器仍是 `configs/gen_e0s_configs.py`，第 1、2 项的文件逐字节不变。
15. **`configs/e2a/`、`configs/e3/`、`configs/e4/`（新增；生成器 `configs/gen_e2a_e3_e4_configs.py <新目录>`，写出三个子目录）**：每个臂一组 `<臂>.scenario.json` + `<臂>.tiers.template.json` + `<臂>.values.todo.json`；**全部是模板**：档位表里的数（速率、C、Δ、N、δ、窗口、门参数）都是 `TBD:<名字>`，登记时按 E1 / E2a 的结果填，快轨结论全部标「探索性」（`dims.interpretation`）。进程数与核数是生成器常量（`K_CONN`：多连接形态的 `capgrpc.load` 进程数；`W_HTTP`：E4 的 `capload` worker 数；`FE_CPUS` = 1 + `FE_INPUT_WORKERS` + `FE_GRPC_THREADS`），改它们要重新生成。登记目录：E2a `03_calib/E2a/register`，E3 `04_frontend/E3/register`，E4 `04_frontend/E4/register`。约定：
   - **档位表形状**（`mc` / `md` / `r`）：一个暖机档 + `LADDER` 个成倍加压的 `up` 档，档间带门的 drain，第一个 `up` 档是基线；二分另起启动。`mc` = M 轴固定 Δ 加 C（`streams_c` 为 `TBD:c_<i>`，`rate_rps` 由 C 推出）；`md` = M 轴固定 C 缩 Δ（每档 `beat_ns` 为 `TBD:beat_ns_<i>`，每档及其后的 drain 时长必须是该档 Δ 的整数倍）；`r` = R 轴最短输出（`max_tokens` 写死为 1），`rate_rps` 为 `TBD:rate_<i>`。所有负载档 k = 1、s = 1。`r` 按 Δ 至少登记两次，`mc` / `md` 按输出长度 N 登记两次（设计稿 §2.1）；每次登记是一个臂副本（与 `w60` 副本同法改名改路径）。
   - **读者**：角色名一律 `reader`；`capgrpc.load` 或 `capgrpc.oe_sink`，读者参数相同（每进程一条 channel、`--request-format openengine` + 作业输入 `request_bytes`、`--planned-msgs`、`--method` 写死为产品的 Generate 路径）。连接形态由进程数决定：`1c` = 1 个进程（`--in-process`，`expected_procs` 1）；`kc` = `K_CONN` 个 worker 进程（`expected_procs` = K + 1）；`k2c` = 2K（负载器翻倍）。E3-3b 与 E4 弱读者臂的读者是单连接、名字里不带形态。
   - **桩**：S-PB、S-NL 都是 1 个进程 1 核。被 gRPC 读者直接读时带 `--schedule`（`request_id` = `req_id`，按计划档归档）；在前端后面（E4）不带。S-PB 直连模式的作业输入 `prebaked` + `request_bytes`；前端模式（E4）照 `e0s-spb-frontend` 的写法，作业输入 `fingerprint` + `prebaked`。S-NL 直连时 `--model` 与请求字节的 `--model` 是同一个参数 `model_name`；在前端后面时 `--model` 与 tokenizer 三项同第 14 条。产品 server 模块一律用缺省。
   - **`measure_class`**：E2a 的桩是 `under_test`、读者 `load_source`；E3-3a 的 S-NL（含 `metronome-only`）是 `under_test`；E3-3b 的 `oe_sink` 是 `under_test`、S-PB `stub_out_of_domain`，弱读者 `load_source`；E4 的前端 `under_test`（P0），`capload` 与弱读者 `load_source`，桩 `stub_out_of_domain`。
   - **对账与清空门**：直连臂有 `requests`（读者 `counts.ok` ↔ 桩 `streams_accepted`）与 `messages`（S-PB `msgs_sent` / S-NL `msgs_yielded` ↔ 读者 `msgs.msgs`，两跳各 +1 条 finished）两条链；门读 gRPC 在途数（桩 `backlog.yielded` − 读者 `backlog.read`）、桩待发与活跃流、读者活跃流与在途请求。E4 前端臂只有 `requests` 链（P0 前端不发计数，块与消息的对应关系未定，§12 U-6），门读 `capload backlog.inflight` 与桩待发、活跃流。`metronome-only` 臂只有桩一跳，门只读桩。
   - **对照臂**：`-vrdr` / `-vstub` / `-vsrv` / `-vload` = 基线臂加一条启动级 `busy` 注入（`launch_inject`，δ 为 `TBD:delta_ns`），分别在读者 `recv.msg`、桩 `send.msg`、S-NL `send.msg`（servicer 的循环上）、`capload` `recv.chunk`；场景与基线臂逐字段相同。产品侧的 `-vf2` / `-vf4`（前端 F2 / F4）场景另改探针档，见勘误 FT1-18。`k2c`、`e4b-w2-mc` 只把负载角色的进程数翻倍。`e3a-mono-<轴>` 与 `e3a-snl-kc-<轴>` 的档位表模板只差门，用同一份档位值登记。`e3b-gs/ees/weak-<轴>` 三臂模板只差 `name`，场景只差读者的模块、`--tier` / `--connect` 与 `measure_class`。`e4b-nb-mc`、`e4b-nb-r` 只有 1 个 `up` 档（E4 卡第一步：近边界噪声的重复启动）。E4 弱读者臂的桩角色与对应前端臂逐字段相同。
   - **前端（E4）**：照 E0 的 `e0a-frontend-sfe` 写法，另加 `--num_input_processor_workers`（写成产品缺省值，使线程池大小是写明的条件）；`cpus_needed` = `FE_CPUS`（E4 卡主档：不少于事件循环 + 输入处理线程池 + grpcio 线程）。prompt 集名 `tid128`（token id、L = 128），登记时生成；E2a / E3 用 `short`（必须是 token id 集：`oe_sink` 第二档拒绝文本 prompt）。
16. **§12 已知缺口与未核对（新增）**：① 生成的场景只在本机做过结构校验、档位表展开、`cross_check` 与（直连臂的）作业路径 e2e；真产品模块（真 pb2、真 server 模块、真前端）上的运行属集群冒烟【未核对】。② `K_CONN`、`W_HTTP` 是没有 E1 读数时取的常量，登记前按 E1-gRPC 的每 worker 健康带核对（E1 的 HTTP 部分不在快轨内，`capload` 的健康带只能靠 `e4b-w2-mc` 的翻倍对照）；`FE_GRPC_THREADS` 与前端在负载下实际可运行的线程数未核对，由 sidecar 的逐线程 CPU 判断是否有线程缺核。③ R 轴 `max_tokens` 1 经过前端时是否被改写（S-PB 只认预制过的 N）【未核对】。④ E4 的桩不带 `--schedule`，按到达时刻归档，读者 / `capload` 按计划档归档：两者只在请求跨档到达时不一致（档间有 drain，正常不会发生）。⑤ 模板的 `up` 档数固定为 `LADDER`；需要更少档时改一份副本登记。⑥ 前端位点（F2 / F4）与 engine 侧位点的注入属步骤 FT1-inj，这里的 V 臂只有 capkit 一侧的位点（前端 F2 / F4 的 V 臂已由勘误 FT1-18 补上，E3-3a 见 FT1-19）。⑦ 快轨不做的项（快轨计划第 4 节）没有模板。
17. **验证**：`tests/launch/test_e2a_e3_e4_configs.py`（生成器逐字节重现三个目录；每个臂 `validate_scenario`、作业输入的 `check_scenario`、按形状造的档位值 `tiergen.expand`、合成绑核表上的 `cross_check`；`params_required` 恰等于各角色与作业输入用到的参数；连接形态与进程数；桩的核数、进程数与 `--schedule` 规则；弱读者臂只换读者；V 臂与翻倍臂只差一处；`metronome-only` 与 full 臂的档位形状相同；三种轴形；E4 前端主档、tokenizer 设置与 served model name；直连臂的 model 参数）；`tests/launch/test_e0s_configs.py` 加第 3 项（S-NL 最小版的参数、前端只差端点、httpprobe 就绪、`beat_ns` 0、参数集合）。变异 S1～S16、S7b（`_local/ft1/mut-scen/summary.txt`，不入库；S7 是等价变异，换成 S7b）全部被测试抓到。作业路径 e2e（覆盖清单按 HEAD 生成；生成的场景只把作业输入的 pb2 模块换成 `oe_double`、S-NL full 加 `--server-module snl_double`）：`e3a-snl-kc-mc`、`e3a-mono-mc`、`e3b-ees-mc`、`e3b-ees-mc-vstub`、`e2a-spb-1c-r`、`e2a-snl-1c-md` 全部 VALID、判定器 0 RED、无 INPUT_INVALID，`e0a-nullsse` 回归 VALID。

**步骤 FT1-inj（产品侧注入的登记；RUNBOOK-stage2 §2.1 第 9 行 3B2-rest 的登记部分；设计稿 §2.4 注入忙等开关、§2.6 V、E4 卡与 E3 卡 3a 的 V）**：

18. **§6.5 / §7 / §11 产品侧注入的登记（推翻勘误 R4-2 的「在编排器下暂时用不了」与 R4-11 ①；改勘误 R4-1 的写快照规则）**：
   - **登记处**：档位表的 `launch_inject`，`role` 写产品角色名，规格格式照 §7.1。档位表随登记表的 sha 登记，所以产品侧注入与 capkit 工具的启动级注入是同一种登记。没有指名产品角色的条目 ⇒ 产品进程不带 `CAPPROBE_INJECT`（缺省关）。
   - **编排器**：为被指名的产品角色设置 `CAPPROBE_INJECT` = 这些规格的文本形式（去掉 `role`，多条用 `;` 分隔，`InjectSpec.to_text`）；其他角色都不设。外层非空的 `CAPPROBE_INJECT` 照旧拒绝（退出码 3），角色 `env` 里的 `CAPPROBE_*` 照旧拒绝（勘误 R5-4）。`launch.json` 新增顶层键 `probe_inject`：`{产品角色: [规格（含 role）…]}`，没有登记时为 `{}`。`dims.json` 的 `injected_for_product_roles` 加一项 `CAPPROBE_INJECT`。
   - **场景核对**（`caplaunch.scenario.cross_check`，起进程之前，不过即退出码 3）：
     1. 只能写在 `launch_inject`：capprobe 在 import 时读一次开关，整个进程生命周期不变；某档的 `inject` 指名产品角色 ⇒ 拒绝。
     2. 该角色 `probe_tier` 必须是 P1 或 P2：P0 没有探针补丁，开关到不了任何位点。
     3. 位点只能是 capprobe 的注入位点（`capprobe.sites.INJECT_SITES`），模式只能是 `capprobe.sites.PROBE_INJECT_MODES` 里的；`miscount` 还要求场景打开 `count` 组。这几条与 capprobe 在 import 时的检查相同，提前到起进程之前。
     4. 位点的进程类（`capprobe.SITES[位点].proc_kind`）必须在该角色的 `proc_kinds` 里：不在 ⇒ 这个进程不经过该位点，心跳只能是 0。
     5. 该角色的 `snapshots` 必须是 `all`（见下一条）。
   - **快照**：capprobe 只要有注入就写快照，所以登记了产品侧注入的 P1 / P2 角色也是写快照的角色。`caplaunch.scenario.product_writes_snapshots(角色, 档位表)` = P2，或 P1 / P2 且有登记的产品侧注入；编排器按它把这些进程列进每档窗口两端的应拷快照（`tier.json` 的 `snapshot_copies`），判定器的 RECORDS 照此核对。场景单独校验（`validate_role`，手里没有档位表）时只能用「P2」这一条，其余由 `cross_check` 补上。
   - **心跳**（§7.3）：在该进程快照的 `inject.<位点>.{hits,applied,applied_ns}` 里；有注入时 capprobe 的每个注入位点都计 `hits`，没有规格的位点 `applied` 为 0。每档的心跳 = 该档窗口两端两份快照之差。注入从进程导入 capprobe 起生效，快照里的 `hits` 就是「生效以来经过次数」，应有的 `applied` 增量 = `capcommon.inject.expected_applied_delta(hits 始, hits 末, every_n, phase)`。产品进程没有逐档自报；本批判定器不核对这项心跳（V 的判读属 2E）。
   - **一个 δ 一次启动**（§12 第 6 条不变）：剂量—反应的每个 δ 登记为一个臂副本。
   - **作业**：覆盖清单的档位要与场景一致（判定器 PROBE-TIER），所以 P1 的臂不能与 P0 的臂同一作业。P1 比 P0 多了探针补丁（位点关闭时每处一次属性读与一次分支），P1 − P0 按设计稿 §2.4 另行评估。
   - **探针树不改**：F2、F4 的守卫调用与注入已经在探针补丁里（`capprobe.sites`、`capprobe.patchcheck`）。
   - **E4 模板**：`configs/e4/e4b-mc-vf2`、`e4b-mc-vf4`（δ 注在前端的 F2、F4，每条消息 / 每个事件项）；相对 `e4b-mc`，除臂名、说明与登记路径外只改三处：启动与前端角色的 `probe_tier` 为 `P1`、前端 `snapshots` 为 `all`、档位表多一条 `launch_inject`（见 `configs/e4/README.md`）。
19. **§12 E3-3a 的 V 与已知缺口（新增）**：
   - **E3-3a**：被测端的注入就是 S-NL 的 `send.msg`。它在 S-NL 进程里包住真 servicer 的 `Generate`，在 servicer 的事件循环上、每条响应交给 gRPC 之前忙等（勘误 FT1-3；模板 `e3a-snl-kc-mc-vsrv`），即设计稿说的「被测进程的每消息路径」，不需要产品文件里的位点。反向注入在读者的 `recv.msg`（`-vrdr`）。engine 侧产品位点 G1～G3 的探针（RUNBOOK §2.1 第 6 行的探针子集）不在本步。
   - **已知缺口与未核对**：
     1. 真前端里的 F2 / F4 注入：本机只在替身产品进程里跑过（导入真 capprobe，按探针补丁的守卫写法经过 F2、F4）。真 `trtllm-serve` 在 P1 覆盖下的导入、`CAPPROBE_ENTRY` 的匹配、F2 / F4 在服务负载下的经过次数【未核对】，属集群冒烟。
     2. 判定器不核对产品侧心跳；剂量—反应、反向注入与等级属 2E。在那之前，E4 的 V 只能按本条的读法人工核对；核对不了的记 NOT RUN，结论等级按设计稿 §2.6 封顶在「已定位，未干预」。
     3. R 轴的每请求位点（H1）同样可以这样登记，但快轨模板只有 M 轴的 F2、F4 两臂。
     4. 快轨没有单列「P1、无注入」的参照臂：V 臂的边界移动与 P0 的 `e4b-mc` 比时混着探针补丁本身的开销，登记时按默认决策补一个 P1 参照副本或写明这一点。
     5. 作业路径（`capjobs.submit` → `jobs/capkit_inner.sh` → `caplaunch` → `capjudge`）上带产品角色的 P1 启动【未运行】：本机作业路径 e2e 没有产品前端的替身。
20. **验证**：`tests/launch/test_product_inject.py`（`cross_check` 接受的五种登记；拒绝的每条规则各一个反例：P0、`snapshots` 非 `all`、非注入位点（含 capprobe 的 lag 安装点 `Lfe`）、`drop` / `short`、没有 `count` 组的 `miscount`、进程类不符、某档 `inject` 指名产品角色；capkit 角色的注入不受影响；`product_writes_snapshots` 的全部取值）；`tests/launch/test_probe_env.py`（产品角色的 `CAPPROBE_INJECT` 用 capprobe 自己的解析器读回与登记相同、capkit 角色和无登记时没有；应拷快照名单按「P2 或有登记」）；`tests/launch/test_launch_e2e.py` 新增一例（真编排器 + 替身产品进程里的真 capprobe，按守卫写法每 5 ms 经过 F2、F4：有登记时 `launch.json` 的 `probe_inject`、快照元数据的注入规格、每档窗口两端都有该进程的快照副本，三个负载档的心跳 F2 `applied = hits`、`applied_ns ≥ applied × δ`，F4 只计 `hits`；无登记的阴性对照：没有任何快照）；`tests/launch/test_e2a_e3_e4_configs.py`（`vf2` / `vf4` 相对基线只差探针档、`snapshots` 与一条注入，位点是前端进程类经过的 capprobe 注入位点）。变异 T1～T15、T16b 全部被测试抓到（`_local/ft1/mut-inj/summary.txt`，不入库；T16 未被抓到，补了 `Lfe` 反例后换成 T16b）。作业路径 e2e：`e0a-nullsse`、`e3b-ees-mc-vstub` 回归 VALID、判定器 0 RED、无 INPUT_INVALID；带产品角色的 P1 作业路径【未运行】（第 19 条第 5 项）。

### 合入 capkit-ft1 到 main（勘误 M-FT1-n，2026-11-16）

`capkit-ft1`（快轨 FT1，上一节）以 `--no-ff` 合入 main（合入前 main 已含 b2-spb 与「合入 b2-spb 到 main（勘误 M-SPB-n）」一节）。本节只记合入时为两边对齐而改的地方；冲突逐条的处理见 `_progress/merge-ft1.md`。勘误各节按合入顺序排列：main 原有各节在前，FT1 一节在后；编号 `FT1-n` 与 main 上已有的编号不重叠。

1. **S-NL 接 SM-18 的服务端队列选项**（与 M-SPB-1 同一件事）：`capgrpc.common.make_server` 在 main 上第二个参数 `max_unrequested_s` 必填；FT1 的 `capstubs.null_llm_server` 在 `metronome-only` 下为占端口起的空 server 仍按旧签名调用，合入后该模式起进程即 TypeError。合入时改为与 S-PB 相同：S-NL 接可选 `--max-unrequested-s`（`capgrpc.common.add_server_args`），只在 `metronome-only` 下用 `resolve_max_unrequested(值, 档位表)` 定值并传给 `make_server`；`minimal` / `full` 下 gRPC server 是产品 `OpenEngineServer` 自己建的（真 servicer 的排队时限与条数上限正是 E3 问题 (3) 与 SM-19 要实测的对象），capkit 不改它，给 `--max-unrequested-s` 即退出码 3。实际值写进就绪文件自报与每档 `summary.json` 的 `grpc_server_options`（产品模式为 `null`），判定器不读。`capgrpc.oe_sink` 只建客户端 channel，不涉及。
2. **桩自报原始 `request_id` 样本**：S-PB 与 S-NL 每档 `summary.json` 新增 `request_id_samples`：`{"first": [...], "limit": 8, "attribution": ...}`——本档（事件档，含被拒的请求）最先收到的至多 8 个原始 `request_id`（文本，每个至多 80 字符；S-PB 的格式错请求读不出 id，不计）。用途：报告真前端发来的请求号是什么形态（E0 补充冒烟 j2 实测前端发的 22/22 都不是十进制，`request_id_not_decimal` 只给计数、不给原文）。判定器不读。
3. **E3-3b 直连形态的请求号（核对，未改代码）**：S-PB / S-NL 给了 `--schedule` 时只接受时间表里的十进制 `req_id`（`_admit` / `rpc_begin`），真前端自生的请求号会被全部拒绝。核对结果：`capgrpc.oe_sink --request-format openengine` 在两档都把十进制 `str(req_id)` 交给产品客户端类——`generate-stream` 档设 `GenerateRequest.request_id`，产品 `OpenEngineClient.generate` 只在它为空时才用 `new_request_id()` 补；`engine-event-stream` 档设 `EngineRequest.request_id`，产品 `to_generate_request` 原样抄进 `GenerateRequest`（`EngineRequest` 文档写的 `"trtllm-"` + uuid 只是前端的生成方式，产品不校验）。所以 E3-3b 模板（`configs/e3/e3b-*`）里 S-PB 带 `--schedule` 是对的，不改生成器；E4 模板（真前端在中间）与 `e0s-*-frontend` 里桩不带 `--schedule`，也不改。测试：`tests/capstubs/test_oe_sink.py` 两档端到端用例断言 S-PB 收到的样本都是读者所发的十进制 `req_id`，原始格式用例断言样本里没有十进制（产品客户端自生的 id）。
4. **follow-up（未改）**：`caplaunch.laddergen.packing` 按 0 个作业输入算时间预算（M-SPB-2）；FT1 的 E2a / E3 / E4 模板都声明了 `job_inputs`（请求字节、预制流），以后若经 laddergen 装箱，要改成按场景的种类数传入。
5. **验证**：`tests/capstubs/test_null_llm_server.py`（`metronome-only` 的 `grpc_server_options` 缺省来源且逐档与就绪一致、`full` 下为 `null`、`full` 下给 `--max-unrequested-s` 退出码 3；`full` 下样本都是读者所发的 `req_id`；被拒请求的原文按到达顺序进样本）；`tests/capstubs/test_prebaked_server.py`（端到端样本 ⊆ 读者所发；无 `--schedule` 时非十进制原文进样本；样本按事件档；被拒请求进样本）；`tests/capstubs/test_oe_sink.py`（见第 3 条）。全量验证与作业路径 e2e 见 `_progress/merge-ft1.md`。

### E10 线：E10-S 编号、serve_glm、G1 替身与 configs/glm（2026-11-18）

- **实验编号 `E10-S`**：E10 的替身臂（G0 glm `MockEngine`、G1 `capstubs.glm_g1`）。用户第三十一次答复（2026-11-18 21:3x）开 E10 线、免逐项审批；真引擎臂按 Q6 不做。`capcommon.policy.NO_MODEL_EXPERIMENTS` 加 `E10-S`；裸 `E10`、`E10-proxy`、`E13-main` 仍被拒（§12 第 11 条的「需要独立编号」由此落实）。
- **`capstubs.serve_glm`**（产品角色入口，glm venv Python 起）：`--engine mock|g1`、`--step-ms`（= 档位表负载档的 `beat_ns`，不等即拒绝起动）、`--tokens-per-step`、`--inject off|M1:count|M1:busy:NS`、`--diag`、`--pick-port` + `--report-addr`（服务端 READY 且端口能连上才写 `host:port`；所建引擎的种类与 `--engine` 不符则永不写，读者等不到就绪而失败）、`--procmon-*`（`capstubs.affinity`）。不改 glm 检出的任何文件：替身经 `glm5_serve.engine_api.MockEngine` 属性替换生效（`cli._create_engine` 在函数内导入）。
- **`{raw}/snx/serve_glm.<pid>.jsonl`**（`capkit.serve_glm.diag/1`）：首行头（engine、step_ms、tokens_per_step、inject、diags、`sys.getswitchinterval()`），之后每秒一行：`M1`（hits/applied/applied_ns，累计）、`dispatch`（calls/responses 累计、max_batch 本秒）、`loop`（n/lag_ns/hist 累计、max_lag_ns 本秒）、`engine`（G1：`StepStats` 累计 + 本秒 max；G0：`_iter`、在途请求数）、`server`（phase、requests、generate_accepted）、`threads`、`gc`；进程退出时再写一行。`{raw}/snx/host.serve_glm.json`（`capkit.serve_glm/1`）记 glm 根目录、五个被包装文件的 sha256、Python 版本与前缀、解析后的 mock 配置与 serve 参数。
- **`configs/glm/`**（`configs/gen_glm_configs.py`）：e10-{g0,g1}-{mc,r}，拓扑同 E8（K = 4 条 `capgrpc.load` 连接）且读者加 `--planned-msgs`；宿主 4 + 1（procmon）核。

### S-NX 线 B1 / R3：准入拒绝改归过载、Wr / Dr 位点、tname（2026-11-19）

依据：`05_engine/PLAN-R2R5.md` 第 6 节（B1）、第 1 节（R3 原架构 V）、第 3.2 节（B6）；`05_engine/E9/analysis/cancelled-trace/REPORT.md`（准入拒绝的机制与 1.84.0 本地复现）。

1. **判定器：准入拒绝归过载型**（新模块 `capjudge.admission`；`symptoms.failures_by_class` 多一个可选参数 `adm`，`judge_tree` 建 `ctx.admission` 并传给 C3、C5～C7 的过载数下限与 X）。一条失败条目里，满足下面全部条件的请求数（上限为条目的 `count`）记为过载型、`by_category` 记 `admission_rejected`，其余部分按原规则（勘误 B2-2E-6、设计 2.6）：
   - 逐请求（读同档、同角色、同 worker 的 `requests.capbin`，状态号按该文件自己的 `meta.status_codes` 映射，gRPC 读者各 worker 的编号不同）：gRPC 读者 `grpc_CANCELLED` 且 `frames == 0`；HTTP 负载器（新架构前端后面的 capload）`protocol_error` 且 `frames <= 1`、`tokens <= 0`；两者都要 `done - sent <= 5 s`（`ADMISSION_MAX_NS`，远低于 gRPC 30 s 未领取时限与登记超时，远高于 E9 N-8f0 实测最坏 1.2 s）。
   - 整个启动（不满足则本启动一条也不改归）：S-NX rank 0 自报 `raw/snx/snx.<pid>.summary.json` 存在且都没有 `fatal_error`，且候选总数 ≤ 负载器全部档的 `counts.sent` 之和 − rank 0 的 `counts.taken` − 负载器的 `conn_refused` / `conn_fail` 失败数（这些请求连客户端都没离开；「未达 rank 0」的守恒判据）；有 HTTP 候选时，前端日志 `raw/logs/frontend*.out` 里 `failed a request: OpenEngine Generate on … failed with CANCELLED: CANCELLED` 的行数 ≥ HTTP 候选数（HTTP 负载器看不到 gRPC 状态，以前端日志为证）。
   - `verdicts.json` 新增可选键 `admission`（只在有候选原因的启动出现）：`candidates`、`sent`、`rank0_taken`、`fe_log_cancelled`、`admitted`、`why`。
   - 只有判定器（`judge_tree`）改用新规则；不经 `judge_tree` 调 `failures_by_class` 的分析脚本不传 `adm`，保持原规则。
   - 不改带：S-NX 读数（`snx_readout.py` / `ft_readout.py`）的送达、xrel、e1 口径都把任何失败当不稳定，与失败的类别无关。

2. **gRPC 读者（`capgrpc.load`、`capgrpc.oe_sink`）**：读者自己取消的请求（任务被取消，如停止时；oe_sink 的流被本端取消且不是到时）原因由 `grpc_CANCELLED` 改为 `client_cancelled`（`class_hint` `non_overload`），`grpc_CANCELLED` 只表示服务端返回的 CANCELLED 状态。每档 summary 的每条失败条目新增 `zero_msgs`（其中一条消息都没收到的请求数）与 `max_send_to_end_ns`（其中最长的发出到结束时间）。旧树没有这两个键、旧树里两种取消同名；判定器只读逐请求记录，不依赖这两个键。

3. **S-NX 位点 Wr（rank 0 worker）**：`capstubs.snx_inject.SITES["Wr"]` = `tensorrt_llm.executor.base_worker._send_rsp(worker, response, postproc_batches=None, rsp_batch=None)`（镜像 9fd3822a00 与 2f22a98d6e 的 base_worker.py 相同、不在 P0 覆盖清单里；唯一的逐响应调用点是 `AwaitResponseHelper.handle_for_ipc_batched` 的循环，base_worker.py:1586，按模块全局名查找）。worker 进程拿不到 `CAPKIT_SNX_INJECT`（MpiPoolSession 只转发 `TRTLLM*`），所以：`serve_nx --inject Wr:…` 写进 S-NX 配置新键 `worker_inject`（`capkit.snx_config/1` 加键，缺省 `""`），`null_worker.build_null_executor` 在 worker 里 `snx_inject.install_worker` 安装；`install_from_env`（宿主 / 前端进程）拒绝 worker 位点，`install_worker` 拒绝非 worker 位点；`--worker` 不是 `null` 时给 Wr 退出码 3。nx_role / nx_driver 用 `--set worker_inject=Wr:…`。计数照旧写 `<out_dir>/inject.<pid>.jsonl`（首行 `process: worker`）。

4. **nx_driver 排空端位点 Dr**：`nx_driver proxy|drain`、`nx_role` 新参数 `--drain-inject Dr:count | Dr:busy:NS`（只用于 raw / deserialize 排空端；K ≥ 2 时传给每个 drain 子进程），在排空线程每收到一帧之后立即执行；每个 lane 的 `nxd.<lane>.summary.json` 每档（接收时刻）新增 `dr_hits`、`dr_applied`、`dr_applied_ns`（不给参数时不出现这些键）。

5. **宿主诊断 tname（B6）**：`snx_inject` 诊断新增 `tname`：安装之后启动、名字是缺省形式（`Thread-N` 或 `Thread-N (target)`）的线程，改名为其 target 的限定名（`<模块末段>.<qualname>`；覆写 run 的子类用类名）；只改 Python 线程名，threads 普查据此给出名字。serve_nx 的缺省不变（不默认开），由 R3 新运行树的场景显式给 `--diag gc,threads,tname`（见 PLAN-R2R5 勘误 E-R3-3）。

6. **验证**：`tests/judge/test_admission.py`（逐请求签名；状态号按文件映射；阳性对照 E8 形态与 E9 HTTP 形态判过载；阴性对照：本端取消、收到过消息的 CANCELLED、30 s 时限路径、登记超时、前端日志为 UNAVAILABLE、旧架构无 OpenEngine 行、守恒不成立、rank 0 自报缺失或 fatal；条目部分改归；无候选原因时不读任何证据；日志跨块计数）；`tests/grpc/test_unrequested.py`（服务端 CANCELLED 的条目 `zero_msgs`、`max_send_to_end_ns`）；`tests/capstubs/test_snx_inject.py`（Wr 计数与忙等、宿主 / worker 位点互拒、serve_nx 把 Wr 写进配置而不写进环境变量、tname 只改缺省名且可还原）；`tests/capstubs/test_serve_nx.py`（Dr 忙等与逐档计数、count 模式、非法规格拒绝、drain 子命令参数）；`tests/capstubs/test_null_executor.py`（`worker_inject` 键）。镜像内 Wr 的真实安装（worker 进程能导入 `base_worker` 且 `_send_rsp` 被替换）、Dr / tname 在作业里的效果【未运行】，属 R3 首个作业的冒烟核对（PLAN-R2R5 第 1 节阳性对照）。

## 审阅记录（r2，2026-11-13）

审阅意见来自 spec-critic（JSON，24 条）。每条都对照设计稿 / 计划原文核对过；「采纳」= 按意见改，「部分采纳」= 问题属实、改法与意见不同（写理由），「不采纳」= 问题不成立。

| # | 意见（摘要） | 核对依据 | 结论 | 改在哪 |
|---|---|---|---|---|
| 1 | 原点的确定时刻自相矛盾（`--origin-ns` 必需 vs「就绪后定原点」） | r1 §2.2、§5.2、§11 caplaunch 行确实互相矛盾 | 采纳，取方案 (a)：起进程前定 `origin = now + lead`，原点前未全就绪 ⇒ 停止、`NOT_RUN`；去掉 `--ready-timeout-s` | §0、§2.2、§5.2、§11 |
| 2 | C5～C7 缺「哪几跳相等」与每请求消息数公式 | 设计稿 §2.7 C6、C7，§2.3 F1；产品闸门读码 `py_executor.py:10468-10470` | 采纳：新 `capcommon.expect` + 场景必填 `reconcile` 段；`source` 放宽为 `summary.<点分路径>`（负载器的帧数在 `summary.frames` 里，不在 `counts` 里） | §5.8、`expect.py` |
| 3 | 服务端每请求参数来源未定；`<qii` 缺 s；SSE 帧口径未定 | 设计稿 §2.2 第 5、8 条 | 采纳：参数只随请求走（HTTP 请求体 + `x-capload-k/s/tier` 头；gRPC `<qiiii` 另带计划档）；SSE 模板与 `SseCounter` 口径（usage 算一帧、`[DONE]` 不算） | §4.4、§4.5、`wire.py` |
| 4 | SSE 路径没有 X2 的输入；`--pacing beat` 语义不清 | 设计稿 §2.6 X2（首步由接纳方记录、按请求号连接） | 采纳，取「服务端落逐请求 `streams`」那一种（与第二批的桩同一条路，判定器只有一种连接方式）；`beat` 定为全进程一个 `BeatSchedule` | §3.4、§5.4.2、§11 |
| 5 | 清空门会读到陈旧快照；shm 目录跨启动共用；`snapshot_seqs` 没人用 | r1 `gate.snapshots_by_role` 确实不查新鲜度 | 采纳；「pid 是否还活着」用快照年龄判（崩溃进程的快照会变旧），不另读 `/proc`，这样判定器能对记录下来的字段复核同一规则；轮询改为按计划时刻，判定器对账轮询次数 | §2.2、§8、`gate.py`、`runtree.py` |
| 6 | 注入心跳的逐档语义没定 | r1 `Injector._selected` 用累计次数 | 采纳：按生效以来计；`tier_report` 给逐档增量与 `expected_applied_delta`；`aux_pending_at_end` | §5.4.1、§7.3、`inject.py` |
| 7 | C10 必需字段不分角色（结构性永假）；红与 D11 的 NOT RUN 矛盾 | 设计稿 §2.5 D11 原文「缺项或不一致 ⇒ NOT RUN」 | 采纳：按角色声明定必需字段，列出必须一致 / 允许不同的字段；C10 不过 ⇒ `NOT_RUN` | §2.5、§10.2、§10.3、`versions.py` |
| 8 | C13 线程清单只取一次；辅助线程懒建；第二半算法没写 | 设计稿 §2.4 sidecar v4、§2.7 C13 | 采纳：每档窗口始末再取 tid 集合；`prestart_aux` 预建、懒建抛错；写出第二半算法；`--cpus-watch` 必须覆盖绑核表 | §5.3、§5.4、§5.6、§10.2、`inject.py` |
| 9 | sidecar 不逐秒记线程亲和性 | 设计稿 §2.5 D1、D2「数据里的验证」 | 采纳：`threads.jsonl` 加 `cpus_allowed` 列（每行都写）；C9 用逐秒读数 | §5.5、§5.6、§10.2 |
| 10 | 两级串联中继无法区分 | 设计稿 E0 (b) 场景 2；r1 §0 与 §9.4 自相矛盾 | 采纳：拓扑位置不同用不同角色名（`relay1`、`relay2`） | §0、§11 |
| 11 | 应检进程数来自观测 | 设计稿 §2.7 引言；陷阱清单 §1 | 采纳：场景必填 `workers`、`expected_procs`，抄进 `launch.json`；summary 回写 `n_workers` 与两个 sha | §5.2、§5.3、§5.4.1、§10.2、§11 |
| 12 | 档位表缺负载轴的值 | 设计稿 §2.1、§2.6 T4、§2.5 D10 | 采纳：`axis` + 每档 `load_value`，校验器按 `load_key` 重算核对；`order` 的含义写死（`asc` = 值越大负载越重） | §3.1、§3.2、`tiers.py` |
| 13 | 判定器不知道角色在本实验里扮演什么 | 设计稿 §2.6「桩分两类」、§2.7 C1、C2 | 部分采纳：加必填 `measure_class`；但不单设 `sink` 类——设计稿 §2.6 把空汇、排空端明确列为域外桩，单设一类会让 C2 的作用对象出现第二种写法 | §10.2、§11 |
| 14 | 硬件画像的键与进 sha 的键未定（会永假） | DATA §8.1；陷阱清单 §2 | 采纳：两张键表写死在 `capcommon.hwprofile`；读不到的标记也进 sha；机箱型号与分区取自回退文件 | §5.5.3、`hwprofile.py` |
| 15 | Slurm 读数回退文件没有路径与 schema | RESOURCES §3 末段 | 采纳：`<作业目录>/env/slurm_view.json`、`capjobs.submit slurm-view`、`capenv.gate --slurm-view-file/--slurm-view-wait-s` | §5.1、§5.5、§5.5.2、§11 |
| 16 | 提交前的机制性断言没落进约定 | CONSTRAINTS §2「靠机制落实」第 1、2、5、6 条 | 采纳：`capcommon.policy` + `submissions.tsv` 三列 + `submissions_append` 拒收；另加「挂载上级目录也算暴露模型库」；白名单暂不含 E10 替身臂、E13 主体（见 §12 第 11 条） | §9.5、`policy.py`、`ledger.py` |
| 17 | GPU 为 0 的检查无人负责 | CONSTRAINTS §2 第 4 条 | 采纳，检查号改叫 `GPU-IDLE`（理由见 §12 第 10 条）；`gpu.jsonl` 加 `query_rc` | §5.5、§5.6、§10.2、§10.3 |
| 18 | C8 与槽名没有机器可读的期望清单 | 设计稿 §2.4（探针接没接上的阳性对照）；陷阱清单 §1 | 采纳：各包导出 `SLOTS`、capprobe 导出 `SITES` 并写进覆盖清单；组打开时槽一次性登记（`register_decls`） | §6.2、§6.5、§6.6、`snapshot.py` |
| 19 | 时钟自检没分给任何组件 | 设计稿 E0「对照与自检」与「停止 / 继续」 | 采纳：B4 `capenv.clockcheck` → `env/clock_check.json` | §5.5、§11、§12 U-4 |
| 20 | 请求超时只是命令行参数，不在登记文件里 | 设计稿 §2.6「失败分两类」；DATA §4 第 6 条 | 采纳：档位表顶层 `request_timeout_ns`；`capload` 去掉 `--request-timeout-s` | §3.1、§5.4.1、§11 |
| 21 | 窗口始末快照可能是边界之前写的 | 设计稿 §2.6 T1、T2 | 采纳：取 `t_mono_ns ≥ 边界` 的第一份，最多等 2 个周期，记偏差；判定器用快照自身时刻 | §5.4 |
| 22 | D4、D8 与「网卡不在路径上」没有数据文件 | 设计稿 §2.5 D4、D8；DATA §8.1；RESOURCES §5 第 2 项 | 采纳：`numa.jsonl`、`logs.jsonl`、`tier.json` 的 `nic_counters` | §5.4、§5.6、§11 |
| 23 | 登记秒精度、本机运行与自测样本会被判 INPUT_INVALID | DATA §4 第 5 条；设计稿 §2.7 三类样本 | 采纳，取「台账毫秒精度」（不取「登记后等 1 s」：等待是约定，毫秒是机制）；`--local-record`；样本自带最小 W | §9.3、§9.5、§10.1、§10.4、`clock.py`、`ledger.py` |
| 24 | 本地盘与 shm 由谁删、按什么断言删没写 | CONSTRAINTS §6 第 2 条例外② | 采纳：只由 B6 内层脚本在拷回校验之后删；`capcommon.scratchpath` 断言路径逐字等于 `<base>/<用户名>-<作业号>` 且校验记录为通过 | §5.9、`scratchpath.py` |

没有整条不采纳的意见。
