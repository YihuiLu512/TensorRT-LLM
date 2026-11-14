# b2-spb（第二批 2B：S-PB 预制桩、指纹接缝与请求字节输入）进度

- 子批：b2-spb（RUNBOOK-stage2-draft.md §2 子批表第 3 行「2B S-PB + 指纹接缝 + 请求字节输入」）。
- WT = `_local/wt-b2-spb`（分支 capkit-b2-spb，起点 0a1410a4206501650f3fa71b43706b2a4bb1847f）；C（01_code main）只读；探针树 C/probe-tree 只读。
- 范围：+ `tools/capstubs/prebaked.py`（容器内生成预制 `GenerateResponse` 字节）；+ `tools/capstubs/prebaked_server.py` 与 `__main__`（复用 nullgrpc 的 `BeatSchedule`、待发队列；解析 `stopping.max_tokens`；注入 `send.msg`/`beat.step`；short 开关；`streams` 带请求号；backlog 槽；可选的 Control/Health）；~ `caplaunch/scenario.py`（`TOOL_TABLE`，`INPUT_KEYS` 加 `request_bytes`）；~ `capjobs/launches.py` 与 `jobs/capkit_inner.sh`（门禁后生成 fingerprint 与 reqbytes，SM-6）；~ 判定器输入核对；`fake_openengine.py` 的发送方式见 D16；+ `configs/gen_e0s_configs.py`（补充冒烟场景）；+ `tests/capstubs/`。
- 契约：WT/INTERFACES.md 末尾新增勘误节「Batch 2B: S-PB prebaked stub, fingerprint seam and request-bytes input」，被覆盖的正文句子加指引注。
- 验证：单测加本地 e2e（nullgrpc 式读者对 S-PB，pb2 用测试替身）；容器内真 pb2 与 `--request-format openengine` 的冒烟属 1.4 补充冒烟，本子批不跑。

## 进度日志
- 2026-11-14 20:12:56 开工：建 worktree（capkit-b2-spb @ 0a1410a4206501650f3fa71b43706b2a4bb1847f），写本进度日志头。
- 2026-11-14 20:19:53 2B-1-prebaked 开工：读完设计 §2.3 S-PB、E2、§7、§12.7、RUNBOOK §2、nullgrpc/reqbytes/capgrpc.common、产品 generation.proto 与 client stream 契约；开始写 capstubs/prebaked.py、prebaked_server.py 与 tests/capstubs/
- 2026-11-14 20:30:44 prebaked/prebaked_server/tests 写完，tests/capstubs 两文件全绿（u2 63、s2 17）；开始变异 M1–M8（_local/b2-spb/mut，逐个串行，约 5 分钟）
- 2026-11-14 20:35:48 自审修 4 处（inspect 的 request_id 线型、短缩不叠加、死属性、S-FE 选项报错），变异 M1–M11 全被抓；跑 tests/capstubs 全量、test_fingerprint、py_compile（约 1 分钟）
- 2026-11-14 20:36:59 2B-1-prebaked 完成待提交：tests/capstubs 82 passed（logs/v1-capstubs.log）、tests/launch/test_fingerprint.py 2 passed、py_compile 通过；变异 M1–M11 全被抓（_local/b2-spb/mut/summary.txt）；fakebin.log 无记录。WT/probe-tree 软链（测试 real_pb2 用）已移到 _local/b2-spb/probe-tree.link，重跑 tests/capstubs 前要移回。INTERFACES 新增勘误节「Batch 2B …」第 1–8 条与 6 处正文指针。
- 2B-1 未做（本子批后续步骤）：caplaunch/scenario.py（TOOL_TABLE、INPUT_KEYS 加 request_bytes）、capjobs/launches.py 与 jobs/capkit_inner.sh 作业内生成 fingerprint / reqbytes / prebaked（SM-6）、判定器输入核对（C10 核 sfe_module sha）、configs/gen_e0s_configs.py 补充冒烟场景、D16 的 fake_openengine.py 进覆盖清单。
- 2B-1 NOT RUN：真 fake_openengine.py 的 _Control/_Health、真前端连 S-PB、容器内 python -m capstubs.prebaked（真 pb2 模块名导入）——都属 1.4 补充冒烟。
- 2026-11-14 20:44:48 2B-2-wiring 开工：读完 RUNBOOK §1.4/§2、SEAMS SM-6、设计 §2.3、scenario/orchestrator/launches/inner.sh/overlay/judge inputs/C10/smoke/gen_e0_configs；方案：场景 job_inputs（fingerprint/request_bytes/prebaked）由 capjobs.launches 逐启动在门禁后生成并以 --job-input 交给 caplaunch，D16 用覆盖清单 stub_modules + C10 核 sha
- 2026-11-14 20:52:41 代码初稿完成（jobinputs、launches、orchestrator、scenario、judge、C10、overlay stub_modules、smoke 角色、gen_e0s_configs）；开始逐目录跑现有测试（launch/jobs/judge/capprobe/capstubs，各约 1–5 分钟）
- 2026-11-14 21:06:30 新测试：tests/jobs/test_job_inputs.py 13 passed、capprobe overlay stub 32 passed、tests/launch/test_job_inputs_launch.py 29 passed；接着写 C10 桩模块核对测试、S-PB 作业路径 e2e、smoke 角色测试
- 2026-11-14 21:09:08 测试补齐（C10 桩模块 4、S-PB 作业路径 e2e 1、smoke 角色 2、e0s 配置 5）；开始写文档：INTERFACES 勘误 9–13 与正文指针、各 README
- 2026-11-14 21:12:13 文档写完（INTERFACES 勘误 9–14 与 6 处正文指针、7 个 README、configs/e0s/README）；开始 py_compile 与全量测试（launch/jobs/judge/capprobe/capstubs/capcommon/grpc + selftest，约 15 分钟，逐目录串行）
- 2026-11-14 21:24:11 全量全绿（launch 225、jobs 211、capprobe 171、capstubs 83、judge 97、capcommon 190、grpc 36；selftest EXIT=0 25 样本）；开始变异 J1–J14（_local/b2-spb/mut2，逐个串行，约 6 分钟）
- 2026-11-14 21:26:03 变异 J1–J14 全部被杀（每个 rc=1，见 _local/b2-spb/mut2/summary.txt）；probe-tree 软链移回 _local/b2-spb/probe-tree.link；提交 2B-2-wiring
