# S-NX 进度（capkit-snx）

格式：`- <CST 时间>｜做了什么｜结果`

- 2026-11-16 13:15｜建 worktree `_local/wt-snx`（分支 capkit-snx，base ecc418b5）；开始本地镜像部分解包 /home/scratch.yihlu_coreai/tmp/20261116-snx-localrt（供 unshare+chroot 无 GPU 跑真产品）｜进行中
- 2026-11-16 13:29｜写完 null_executor / snx_product / null_worker / serve_nx 初稿；开始本机 chroot 冒烟 r01（O-NX 单前端）｜进行中
- 2026-11-16 13:36｜本机 chroot r01（O-NX：serve_nx 原架构单前端，真产品 1.4.0rc3，CPU、无 GPU，torch.cuda.Stream 打桩）：起服 123 s，/v1/completions 流式 8 token + usage + [DONE]，NullExecutor 自报 steps 8 / released 8 / finished 1，SIGTERM 后 summary 写出｜PASS（本机，MEASURED）；修 1 处：cuda Stream 桩须是类型（产品注解 `torch.cuda.Stream | None`）
- 2026-11-16 13:44｜本机 r03（N-NX：serve_nx 宿主 --grpc --grpc-protocol openengine + 真新架构前端 --openengine-endpoint；rootfs 按 P0 覆盖清单换上基线 37 个文件后）：宿主 74 s 就绪、前端 40 s，/v1/completions 8 token + usage，NullExecutor 自报 released 8｜PASS（本机，MEASURED）。单测 tests/capstubs/test_null_executor.py 22 passed、test_serve_nx.py 11 passed；caplaunch 加 {origin_ns} 占位
