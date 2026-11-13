# integ-R1（R1 自审修复集成）进度

- 2026-11-13 开工：<C> HEAD 0c3df232c6（工作树含 6 个修复者改动）；探针树 HEAD d9380322a3（含 FX-probe 改动）。已读 triage/brief、6 份 fix-R1 进度。
- 21:14:30 (a) sidecar info.threads 已改+测试（mut-a 去掉声明即红）；开始 (b) ledger UNKNOWN_SUBMIT
- 21:15:29 (b) ledger UNKNOWN_SUBMIT 正式状态 + submissions_current 状态被 SUBMITTED/ENDED 行覆盖；capjobs 改用正式状态；测试已改；开始 (c)
- 21:19:36 (b)(c) 完成：tests/jobs 129 passed；变异 5 例全红（_local/integ-R1/mutate.py）。开始 (d) 扫其他接缝
- 21:20:31 (d) capenv.gpu check CLI schema 改名 capkit.gpu_idle_check/1；make_samples gpu_idle.json 对齐 envfiles 形状，重新生成样本到 _local/integ-R1/gen
- 21:21:52 样本重生成并覆盖（文件集合不变 1675）；开始写 INTERFACES 勘误
- 21:23:13 INTERFACES：新增「### R1 自审修复（2026-11-13）」21 条；集成节第 7 条按 K10 改。下一步：正文加指向
- 21:24:13 INTERFACES 正文加 20 处指向（勘误 R1-n）。开始第 3 步全量验证
- 21:24:42 后台跑 verify_all.sh r1（10 个测试目录 + selftest + patchcheck + py_compile + bash -n，预计 6–10 分钟）
- 21:26:16 e2e 脚本（_local/integ-R1/e2e/）与两份提交说明已备好；等 verify r1

## 第 1 步 接缝处理（截至 21:28）
- (a) `tools/capenv/sidecar.py`：新增 `Sidecar.declared_threads()`，就绪文件 `info.threads` = `{native_id: 名}`（主线程以外全部 Python 线程；慢线程名 `capenv-sidecar-slow` 原已命名）。测试 `tests/sidecar/test_sidecar_v4.py::test_sidecar_gpu_uses_explicit_path_and_keeps_unreadable` 加断言：/proc 里主线程外的每个 tid 都在声明里。变异（去掉声明）⇒ 红。判定器未放宽。
- (b) `capcommon/ledger.py`：`SUBMISSION_STATUSES` 加 `UNKNOWN_SUBMIT`（只许 NOTE 行且带 submitted_at）；`submissions_current` 中 SUBMITTED/ENDED 行的 status 即使为空也覆盖（否则对账后仍是 UNKNOWN_SUBMIT、record-end 永拒——变异实测）；`submission_unknown()`。capjobs 写 `status=UNKNOWN_SUBMIT`，reason 以用途标记开头。测试：capcommon 新增 `test_unknown_submit_status`；jobs 两处断言更新。
- (c) `capjobs.launches.NOT_RUN_STOPS` 四个 stop reason（无编排器 STATE 且 rc=5 ⇒ NOT_RUN）；编排器对这三个新 reason 都写 STATE NOT_RUN、rc 5（tests/launch 已有用例断言，核实）。`launchplan.worst_stop_s/check_kill_grace`：每个启动 argv 须带 `--scenario {SNAP}/…` 与 `--kill-after-s`，最坏收尾 = len(stop_order)×(stop_grace_s+kill_after_s)+2×kill_after_s；submit（对 code_root，且场景须在快照文件集内）与 launches.main（对 snap）都校验。测试：test_units 新增 10 例、test_container 新增 4 例、test_submit 禁止项 +2。变异 4 例全红。
- (d) `capenv.gpu check` CLI schema 改名 `capkit.gpu_idle_check/1`（与 envfiles 的 gpu_idle.json 同名不同形）；`make_samples.py` 的 gpu_idle.json 改为 envfiles 真实键集并重生成全部样本（文件集合不变）；jobs/README、launches 文档串同步。
- 21:31:41 verify r1：10 个目录全过（188/29/105/59/32/129/55/60/4/5）、selftest 23 EXIT=0（首跑因 --tmp 目录未建失败，补建后重跑）、patchcheck PASS、py_compile 180+4、bash -n 2。开始 e2e nullsse（约 1–2 分钟）
- 21:32:21 e2e nullsse：caplaunch rc=0、STATE VALID、judge rc=11（NOT_RUN），无 RED；sidecar 慢线程被识别；开始 e2e relay
- 21:33:01 e2e relay：caplaunch rc=0、STATE VALID、judge rc=11，无 RED；launch 级 C10/GPU-IDLE/C13 PASS。开始提交探针树
- 21:33:32 探针树提交 733edd4edf，patchcheck --rev HEAD PASS（7 位点/8 处，无漂移，sites.py 不用改）。r1 之后又改了 launches 文档串与 jobs/README，后台复跑 verify r2（约 7 分钟）

## 第 2 步 INTERFACES
- 文末「勘误与补充」新增 `### R1 自审修复（2026-11-13）`，21 条，按章节排序（§5.2…§11），合并 FX-jobs 6 条、FX-env 1 段、FX-judge 6 段、FX-probe 1 段、FX-load 2 条，以及 FX-launch（其进度文件只写「见最终报告」，按其进度文件的结论表与「给 FX-judge 的形状」整理）和本集成的条目（UNKNOWN_SUBMIT、kill_grace 下限、NOT_RUN 停止原因、sidecar 线程自报、gpu_idle_check 改名）。
- K10：集成节第 7 条改为「已解决（见勘误 R1-5）」，并更正「E0(a) 没有 under_test」（只有 e0a-nullsse 没有；e0a-frontend-sfe 的前端是 under_test）。
- 正文 19 处句子加 20 个「见勘误 R1-n」指向（§5.3、§5.4、§5.4.1、§5.5、§5.6、§5.9、§6.5、§6.6、§9.2、§9.5、§10.1×2、§10.2×3、§10.4、§11×4），未大段重写。

## 留给主会话（未改）
- P3（responses_utils 顶层 import capprobe 被池进程 / 原架构 worker 导入）：挑战设计，待用户裁决。
- K4 D14 结论由谁登记（C13 第二半恒 NOT_RUN，fail-closed）。
- K6 请求 0 张 GPU 的判定口径（现 GPU-IDLE 恒 NOT_RUN；submit 打警告）。
- C13 线程检查范围（本轮改为 sidecar 自报，未放宽判定器）。
- P7（F2 两次 read.result()，改了会违反 patchcheck「只增不删」）是否接受。
- UNREGISTERED 只采画像作业是否计入 CONSTRAINTS §7「连续两次 ENV_INVALID」（台账已用 purpose 区分，无计数实现）。
- 画像 sha 随可见 GPU 数变化 ⇒ 主型号按申请卡数分别登记（INFERRED）。
- 集群作业的 `sidecar_gpu_every_s` 必须 > 0（values.todo.json 填值时注意）。
- FX-launch 未改：smoke env.import_probe 的 PASS 不看对照是否有信息量；BaseException 分支不写 launch_end。FX-jobs 残余：hwprofile 每条外部命令 30 s 超时与 overlay 拷贝不在预算里。
- 旧树 `_local/e2e-20261113-195336-nullsse/.../judge/`（旧判定器留下）未处理。
- `monitor_children.json` 的 `is_nvidia_smi` 在本机替身（shell 脚本）下恒 false，真实 ELF 下为 true 未核实。

## 第 3 步 全量验证（r2 为终版，提交前；r1 结果相同）
| 项 | 退出码 | 结果 |
|---|---|---|
| tests/capcommon | 0 | 188 passed |
| tests/capload | 0 | 29 passed |
| tests/capprobe | 0 | 105 passed |
| tests/envgate | 0 | 59 passed |
| tests/grpc | 0 | 32 passed |
| tests/jobs | 0 | 129 passed |
| tests/judge | 0 | 55 passed |
| tests/launch | 0 | 60 passed |
| tests/nullsse | 0 | 4 passed |
| tests/sidecar | 0 | 5 passed |
| samples/run_selftest.py --package capjudge --expect-samples 23 | 0 | errors [] |
| capprobe.patchcheck（探针树工作树） | 0 | PASS，7 位点 / 8 处 |
| py_compile（180 个 .py + 探针树 4 个） | 0 | 0 失败 |
| bash -n（jobs/capkit_inner.sh、capkit_job.sbatch） | 0 | 0 失败 |
日志：`_local/integ-R1/logs/r2/`。fakebin.log 0 行。

## 第 4 步 本地端到端（`_local/integ-R1/e2e/`）
- e0a-nullsse（JOB local-ir1a）：caplaunch rc=0、STATE VALID；判定 rc=11（launch NOT_RUN），无 RED。非 PASS：launch C13（无 under_test 角色，应检 0；明细列 2 个 aux helper）、各档 C1（no registered band）、C6（应检 0）、C13（pilot 无 D14）。GPU-IDLE PASS、8 档 pids_stable 全 True、unattributed []、sidecar 慢线程被识别为 declared。
- e0b-s1-single（JOB local-ir1b）：caplaunch rc=0、STATE VALID；判定 rc=11，无 RED。非 PASS：各档 C1、C7（应检 0）、C13（pilot 无 D14）；launch 级 C10/GPU-IDLE/C13 PASS；12 档 pids_stable 全 True。

## 第 5、6 步
- 探针树提交 733edd4edfbc80463936d928e86f49a295a872b4；patchcheck --rev HEAD PASS（行号无漂移，sites.py 未改）。
- 残留进程：pgrep（caplaunch|capload|nullsse|capenv|capgrpc|capselfcheck|capjudge|run_selftest|make_samples|pytest|fake-nvidia-smi|verify_all|mutate.py）无。
- 本次新建的 /dev/shm 目录（未删）：/dev/shm/yihlu-local-ir1a-integR1、/dev/shm/yihlu-local-ir1b-integR1。
- 之后提交 <C>（本文件随之提交）。
