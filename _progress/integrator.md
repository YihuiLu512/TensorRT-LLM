# integrator 进度（第一批集成）

开始：2026-11-13。任务：读 INTERFACES 与各 _progress → 跑全部测试 → 本地功能性端到端（_local/）→ 修跨组件不一致 → py_compile / bash -n → README 进度表 + REVIEW.md 骨架 → 一次提交 → 探针树提交 → 清理进程。

## 状态（2026-11-13 收尾）
完成。01_code 一个提交「Add batch-1 capkit tools」（sha 见提交后 git log；本文件随该提交一起 amend 进去）；探针树提交 d9380322a3a08c7cfa5375d9b3bf66e850a1cc90（分支 capprobe-J17871，-s，pre-commit 通过）。下一步交给编排者：三路自审 R1（base = 首个提交 cdf75fd，head = 本提交），见 REVIEW.md。

## 未决（交给用户 / spec 负责人）
- INTERFACES 勘误「集成（第一批）」第 3 条：判定器对 raw/env/ 直属文件只核四个作业键（放宽，待确认）。
- 第 7 条：multiprocessing resource_tracker 进程未归属 → E1 的启动级 C13 会红，E1 前要定做法。
- 各 builder 报告里的其余未决（B5 第 7 条 gate 不要求全部进程有快照、B6 第 2 条白名单编号、B7 STATE 首行归属等）原样保留，见 README 进度表「已知缺口」。
- 本地 E2E 留下的 /dev/shm/yihlu-local-e2e*（7 个目录，共约 300 KB）与 01_code/_local/（约 30 MB，gitignored）未删，等用户决定。

## 流水
- 19:47 跑基线测试（全部 tests/* 并行，输出 /tmp/claude-154653/-home-scratch-yihlu-coreai-dev-feat-dev-J17871/240caacd-4a5a-4062-992a-5158e12b461d/scratchpad/integ/r0/）
- 基线测试（改动前）：10 个 tests/* 目录分别跑，全部 rc=0：capcommon 187、capload 26、capprobe 87、envgate 33、grpc 25、jobs 86、judge 27、launch 42、nullsse 4、sidecar 4；judge 自测 run_selftest.py（--package judge）rc=0。日志在 scratchpad integ/r0/。
- 决定：按 INTERFACES §1.1/§1.2 把包挪到规定位置与名字（capgrpc、capselfcheck、capenv 合并、capjudge、caplaunch、capstubs、capjobs 移到 tools/），因为 B7 场景按 INTERFACES 名字起进程、§1.2 防撞名规则、容器内按 tools/<包> 拷 dist-packages。
- 19:52 包已按 INTERFACES 改名/搬家（capgrpc[.load 入口 + loader 实现, .nullgrpc 子包]、capselfcheck、capenv 合并加 __init__、capjudge、caplaunch、capstubs、tools/capjobs）；gen_e0_configs.py 移到 configs/；E0(b) 场景修正：drain 消息数键 summary.msgs.msgs（原 summary.counts.msgs_read 不存在）、对账链加 relay 各跳、beat/drain 加 --in-process 使 expected_procs=1 成立。下一步：重跑全部测试 r1
- 19:53 本地 E2E(nullsse)：建 W=/home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02/01_code/_local/e2e-20261113-195328-nullsse/W
- 19:54 r1（改名后）全部测试 rc=0（各目录通过数同基线）
- 19:54 nullsse E2E：第一次 NOT_RUN（我给的 nullsse_chunk_bytes=64 < 工具下限 126，配置值问题）；改 128 后 caplaunch rc=0、STATE VALID。树：/home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02/01_code/_local/e2e-20261113-195336-nullsse/W/runs/Jlocal-e2e2-r0-L0_e0a-nullsse
- 19:55 judge 身份核对改为：raw/env/ 直属文件只比四个作业键（作业级文件原样拷入，arm/launch_seq 必然不同）；nullsse E2E 新树 /home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02/01_code/_local/e2e-20261113-195531-nullsse/W/runs/Jlocal-e2e3-r0-L0_e0a-nullsse 判定 rc=10，正在看 checks.tsv
- 19:59 nullsse E2E 修了三处跨组件不一致后复跑（树 /home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02/01_code/_local/e2e-20261113-195834-nullsse/W/runs/Jlocal-e2e4-r0-L0_e0a-nullsse）：caplaunch rc=0 VALID；judge rc=11，剩下的 NOT_RUN 只有无 pilot（C1 无登记带、C13 第二半无 D14）、无 GPU（GPU-IDLE）、2 个 multiprocessing resource_tracker 未归属（C13 启动级，E0a 无 under_test 所以 NOT_RUN；E1 会红 → 未决）。修的三处：
  1. capcommon.snapshot：快照 meta 加 "threads" {native_id: name}（编排器线程清单靠它认出子进程里的 Python 线程；原先每个工具的 capkit-snapshot 线程都认不出，C13 逐档红）
  2. capjudge：sidecar meta 读 interval_ns（sidecar 实际写 interval_ns；原读 interval_ms → C9 NOT_RUN）；样本 21 个 sidecar/meta.json 与 make_samples.py 同步改
  3. capload：lag 记录容量按档位表 + 原点前剩余时间推出（原用 --max-records-per-worker → 溢出、RECORDS 红）
- 19:59 开跑 relay E2E（e0b-s1-single，约 30 s）
- 20:00 relay E2E 第一轮（local-e2e5）：caplaunch VALID；judge rc=10，C3/C5(3 跳)/C6(4 跳)/C9/C11 PASS；C13 红 = grpcio 的 event_engine 线程池认不出 → caplaunch.procfs.THREAD_RULES 加 ^event_engine$ 规则，复跑
- 20:01 加 ^lifeguard$ 规则后 relay E2E（local-e2e7，W=/home/scratch.yihlu_coreai/workfile/feat_dev-J17503/data/stage-01/test-02/01_code/_local/e2e-20261113-200042-relay/W）
- 20:01 开跑 r2 全部测试
- 20:03 r2 全部 rc=0。探针树已提交 d9380322a3（pre-commit 的 ruff 重排了 import，行号变了）；下一步 patchcheck --rev HEAD、按需改 sites.py
- 20:04 py_compile 172 个 .py rc=0、bash -n 3 个脚本 rc=0；judge 自测（--package capjudge）rc=0；patchcheck --rev HEAD PASS 7/7（sites.py 不用改）；README 进度表与 REVIEW.md 骨架已写；下一步提交 01_code
