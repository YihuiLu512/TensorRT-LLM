# S-NX 进度（capkit-snx）

格式：`- <CST 时间>｜做了什么｜结果`

- 2026-11-16 13:15｜建 worktree `_local/wt-snx`（分支 capkit-snx，base ecc418b5）；开始本地镜像部分解包 /home/scratch.yihlu_coreai/tmp/20261116-snx-localrt（供 unshare+chroot 无 GPU 跑真产品）｜进行中
- 2026-11-16 13:29｜写完 null_executor / snx_product / null_worker / serve_nx 初稿；开始本机 chroot 冒烟 r01（O-NX 单前端）｜进行中
- 2026-11-16 13:36｜本机 chroot r01（O-NX：serve_nx 原架构单前端，真产品 1.4.0rc3，CPU、无 GPU，torch.cuda.Stream 打桩）：起服 123 s，/v1/completions 流式 8 token + usage + [DONE]，NullExecutor 自报 steps 8 / released 8 / finished 1，SIGTERM 后 summary 写出｜PASS（本机，MEASURED）；修 1 处：cuda Stream 桩须是类型（产品注解 `torch.cuda.Stream | None`）
- 2026-11-16 13:44｜本机 r03（N-NX：serve_nx 宿主 --grpc --grpc-protocol openengine + 真新架构前端 --openengine-endpoint；rootfs 按 P0 覆盖清单换上基线 37 个文件后）：宿主 74 s 就绪、前端 40 s，/v1/completions 8 token + usage，NullExecutor 自报 released 8｜PASS（本机，MEASURED）。单测 tests/capstubs/test_null_executor.py 22 passed、test_serve_nx.py 11 passed；caplaunch 加 {origin_ns} 占位
- 2026-11-16 14:35｜本机 r05（nx_driver 直跑，真产品 chroot、CPU）：standalone 190/190 完成；proxy raw 消息 525 条、snx finished 190；proxy deserialize 解出 responses 3040 = 190×16、finals 190；proxy K=2 两条 drain 各发 95、snx finished 190。修 1 处：K≥2 时父进程不等表尾而永等 SIGTERM（改为等表尾+tail，--no-hold 时退出）｜PASS（本机，MEASURED）
- 2026-11-16 14:35｜加 capstubs.nx_role（nx_driver 的 capkit 工具形态，TOOL_TABLE 新条目，single、tiers、snapshot、ready）：门用 backlog.inflight / engine_open / pending_keys（proxy 模式从 worker 的 snx.<pid>.live mmap 读，NullExecutor 新增 live 记录）；每档 summary 按计划档记 planned/sent/ok/failed（ok = snx reqs 记录按 client id 匹配）。serve_nx 加 --stream-interval（写 YAML 并 --config）。本机 r06（nx_role 直跑，真产品）：standalone、proxy raw 每档 ok = planned（40/150），快照里门三槽表尾归零｜PASS（本机，MEASURED）
- 2026-11-16 14:35｜configs/gen_snx_configs.py → 23 臂（E2b 4、E7 5、E8 4、E9 10）模板，validate_scenario 全过；单测 capstubs 198 passed（含新 test_nx_role 4 条）、launch 491 passed｜PASS（MEASURED）
- 2026-11-16 15:2x｜本机 r06 proxy K=2（nx_role）：每档 ok = planned；修：K≥2 的 drain 子进程收到的消息数并进每档 summary 的 zmq。本机 r07 失败（trtllm-serve 无 GPU 时 get_llm_args 拒绝：`No GPU devices found`）⇒ 生成器给 serve_nx 角色加 `--gpus_per_node 1`；r08：O-NX n=2（serve_nx 原架构，附属前端以 `python serve_nx.py <serve 参数>` 重执行）与 N-NX n=2（serve_nx 包装新架构前端 → serve_nx 宿主）各 4 个流式请求全部 10 行（8 token + usage + [DONE]）｜PASS（本机，MEASURED）
- 2026-11-16 15:2x｜加 scrub：serve_nx / nx_role 在导入产品前删掉 SLURM_* / SLURMD_* / PMI_* / PMIX_*（`--keep-launcher-env` 关闭），防 MpiPoolSession 的 MPI_Comm_spawn 在 `srun --mpi=none` 下被当成 srun 直启而挂住或中止（本地无 Slurm 环境，未实测；集群冒烟验）｜INFERRED；新增 tests/capstubs/test_snx_configs.py（生成器逐字节复现、23 个场景 validate、{origin_ns} 解析）
