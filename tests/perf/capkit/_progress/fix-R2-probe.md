# FX2-probe 进度（R2 修复）

- 2026-11-13 开工：已建 `_local/fx2-probe/fakebin`（10 个桩，exit 97）。负责簇 Q2、Q13、Q20、Q21（sites）、Q22、Q23、Q24。

## 词表（Q21 进程类，FX2-launch / FX2-judge 对齐用）

**定稿（2026-11-13 22:53，FX2-probe）**。原则：一个进程类 = 「该类进程**必定**经过的位点集合」；每个位点只标一个类；产品角色声明它的进程树里会出现的全部类。依据：探针树读码（`openai_server.py:935` 与 `openai_disagg_server.py:303` 都装 `ServerArrivalTimeMiddleware`；`multi_frontend.py:108-115` 只有单前端走 `launch.py`；F2/F4 只在 OpenEngine 客户端与 facade 里）。

| 进程类（`proc_kind`） | 位点 | 含义（哪类进程必定经过） |
|---|---|---|
| `frontend` | H1、H3 | 经 `OpenAIServer` 服务 OpenAI HTTP 接口的前端进程：新架构单前端、多前端 launcher、多前端附属前端、原架构 `trtllm-serve` 前端 |
| `frontend_openengine` | F2、F4 | 新架构前端（经 OpenEngine 客户端 + facade 连 engine）：单前端、launcher、附属前端；原架构前端**没有** |
| `frontend_single` | Lfe | 新架构单前端（`--num_serve_frontends 1`，走 `launch.py`） |
| `frontend_launcher` | Lmfl | 多前端的 launcher 进程 |
| `frontend_attached` | Lmfa | 多前端的附属前端进程 |

角色 `proc_kinds` 推荐值（FX2-launch 写 configs/生成器、FX2-judge 写样本/C8 测试照此）：

| 拓扑 | `proc_kinds` |
|---|---|
| 新架构单前端 | `["frontend", "frontend_openengine", "frontend_single"]` |
| 新架构多前端（一个角色 = launcher + 附属前端） | `["frontend", "frontend_openengine", "frontend_launcher", "frontend_attached"]` |
| 原架构前端（第二批） | `["frontend"]` |
| 不跑探针代码的角色（S-FE、负载器、空汇等） | `[]` |

已知不覆盖：disagg 路由（`openai_disagg_server`）也经过 H1、不经过 H3，本词表不给它单列类（第一批没有 disagg）；它的角色写 `[]`，H1 的跨跳重复计数口径见 SITE_NOTES（Q23）。

## 进度日志

- 22:53 词表定稿，已写入上节。
- 22:58 核实完毕（读码）：Q2/Q13 在 probe_entry；Q20 snapshot.ensure_own_dir；Q22 facade _feed 加 `not self._session_closed` 守卫条件（同一行，行号不变）；Q23/Q24 写 SITE_NOTES/README。开始改 probe_entry.py。
- 23:00 probe_entry.py 已改（cwd + -m 形式），新增 test_entry_launch_forms.py；跑 test_config + 新测试（约 1 分钟）
- 23:00 新代码：test_entry_launch_forms + test_config 79 passed（rc 0）。修前对照（_local/fx2-probe/prefix，原版 probe_entry）：launch_forms 红 3 条 = python_dash_m 的 re-exec、chdir 后 relative-shebang、relative-interpreter；console script/python 脚本/相对脚本/PATH 查找/选项在 -m 前/他模块 6 条修前也绿（对照）。test_config 修前另红 28 条，主要因规则多了 cwd 键（形状变化，不算行为证据）。B-5 原文例子 `python3 ./serve_entry.py` 修前就匹配（脚本路径在 args 里按字面比），Q13 判部分属实：真正受影响的是「程序位」是相对路径（相对的 #! 脚本、相对的解释器）。
- 23:01 Q20 完成：snapshot.ensure_own_dir 加 0o022 拒绝 + docstring 收窄；test_group_or_other_writable_dir_refused×4 修前红修后绿（11 passed）。开始改 sites.py（Q21 词表、Q22/Q23/Q24 SITE_NOTES）
- 23:02 sites.py 改完（PROC_KINDS、F2/F4→frontend_openengine、Lfe→frontend_single、SITE_NOTES H1/F4 重写、LAG_NOTE）。下一步：探针树 facade.py:1424 加守卫条件（Q22）
- 23:03 探针树 facade.py:1424 守卫改为 `_capprobe.ON_F4 and not self._session_closed`（同一行，行号不变，未 commit）；patchcheck 工作树 PASS 8/8。跑 tests/capprobe 全目录（约 1–2 分钟）
- 23:03 capprobe README 已同步（入口规则两种形式与 cwd、进程类表、H1/F4 口径、lag 口径，删掉「启动期的事件循环阻塞不进 lag.*」）
- 23:04 影响面检查：只读跑 tests/launch（FX2-launch 的目录，正被并发修改；失败只记录不改，预计 ≤5 分钟）
- 23:07 tests/launch 65 passed rc 0。再只读跑 tests/judge（FX2-judge 的目录）看词表影响
- 23:3x 全部完成。终版：tests/capprobe 158 passed（rc 0）；patchcheck 工作树 PASS（8/8 处）；只读影响面 tests/launch 65 passed、tests/judge 85 passed（各 rc 0，时点快照，他方仍在改）；fakebin.log 0 字节；无残留进程。

## 簇结论

| 簇 | 核实 | 改动 | 新增测试（修前） |
|---|---|---|---|
| Q2 | 属实（实测：`python -m` 的 re-exec 修前 OFF） | `capcommon/probe_entry.py` 模块重新执行形式 + 文档字面化；README | `test_entry_launch_forms.py::test_python_dash_m_and_its_reexec`（红）、`test_config.py::test_module_form_*`（新键，修前因形状红） |
| Q13 | 部分属实：B-5 原例 `python3 ./x.py` 修前就匹配（脚本在 args 里按字面比）；真正受影响的是程序位为相对路径（相对 `#!` 脚本、相对解释器）+ chdir | 规则加必填 `cwd`，解析一律按规则 cwd（含 PATH 相对项） | `test_chdir_before_import[relative-shebang/relative-interpreter]`（红）、`test_relative_path_entries_resolve_against_the_rule_cwd` |
| Q20 | 属实（加固） | `snapshot.ensure_own_dir` 拒绝 `mode & 0o022`，docstring 写明只查最后一级 | `test_group_or_other_writable_dir_refused`×4（红） |
| Q21 | 属实 | `sites.py` 新增 `PROC_KINDS`；F2/F4→`frontend_openengine`，Lfe→`frontend_single`；README 进程类表 | `test_proc_kinds_name_the_processes_that_pass_each_site`（红） |
| Q22 | 属实；积压位置的差异是结构性的，守卫只能统一「会话关闭后的尾随事件」 | 探针树 facade.py:1424 `if _capprobe.ON_F4 and not self._session_closed:`；SITE_NOTES/README 写明两条路径口径 | `test_f4_on_the_pool_path_stops_counting_once_the_session_closed`（红：6 vs 3） |
| Q23 | 属实，有意为之（口径写明） | SITE_NOTES/README H1 | — |
| Q24 | 属实，有意为之（C8 只用窗口两份快照之差；capprobe lag 槽无其他读方） | `sites.LAG_NOTE`、README，删「启动期的事件循环阻塞不进 lag.*」 | — |
