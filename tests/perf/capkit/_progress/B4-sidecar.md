# B4-sidecar 进度

- 2026-11-13 开工：已建 tools/{sidecar,canary,envgate}、tests/{sidecar,envgate} 目录；正在读 INTERFACES.md。
- 已读 INTERFACES（§0-2、§5、§6.1、§10.2、§11）、capcommon（cli/identity/jsonio/hwprofile/snapshot/lifecycle/runtree/versions）、设计稿 §2.4/2.5/2.7、RESOURCES §3/§5、DATA §8.1、CONSTRAINTS §4-7、陷阱清单、sidecar_v3。
- **布局决定（与 INTERFACES 的冲突）**：INTERFACES §1.1/§11 规定包名 `capenv`（tools/capenv/），但本任务只允许写 tools/{sidecar,canary,envgate}/。
  折中：`capenv` 做成 PEP 420 命名空间包，三处各放一个**没有 __init__.py** 的 `capenv/` 子目录：
  tools/sidecar/capenv/sidecar.py、tools/canary/capenv/canary.py、tools/envgate/capenv/{procfs,topology,pins,hwprofile,gate,memguard,gpu,loopback,clockcheck,switchinterval,runcmd}.py。
  PYTHONPATH=<C>/tools:<C>/tools/sidecar:<C>/tools/canary:<C>/tools/envgate 时 `python -m capenv.sidecar` 等与 INTERFACES 一致；集成者把三个 capenv/ 合并成 tools/capenv/ 即可（import 名不变）。
- 下一步：写 tools/envgate/capenv/procfs.py（/proc 读取，带 root 前缀便于测试）
- 19:19 写完 procfs/runcmd/topology/roles/pins + tests/envgate/test_envgate_pins.py；正在跑 pins 测试
- 19:19 pins 测试 8 passed；已把 sidecar_v3.py 拷到 tools/sidecar/capenv/sidecar.py（v3 sha 见下），开始重写为 v4
- 19:21 sidecar.py: Writer/Inventory/Sidecar.sample_proc 已写；下一步 cpus/snapshots/logs/slow thread/main
- 19:22 sidecar.py 全部写完（约 552 行）；下一步 canary，然后 sidecar 测试
- 19:22 canary.py 已写；开始写 sidecar/canary 测试
- 19:23 写了 tests/sidecar/test_sidecar_v4.py；正在跑
- 19:24 sidecar+canary 测试 4 passed（修了一个真缺陷：同一 pid 先因 starttime 不符被标 gone，重新登记后永远不采；改为按 (pid, starttime) 记 gone）。下一步 memguard、hwprofile
- 19:25 hwprofile.py: cpu/memory/gpus/nics/disk 已写；下一步 software/observed/finalize/main
- 19:26 hwprofile.py 完成并在本机跑通（rc=0，缺 nvidia-smi/cpufreq 记 unreadable）。下一步 gate.py（RESOURCES §3 六项 + hw_gate + memguard）
- 19:27 gate.py 写完；下一步 gate/hwprofile/memguard/gpu 测试
- 19:28 写了 tests/envgate/test_envgate_gate.py；正在跑
- 19:29 envgate 测试 29 passed（pins+gate+hwprofile+memguard+gpu）。
  **实测发现**：本 VM（RHEL8 cgroup v1，systemd 239）下 `systemd-run --user --scope -p MemoryMax=6G` 并没有把进程放进带上限的 memory cgroup——memory 控制器路径仍是 session-7.scope，memory.limit_in_bytes = 9223372036854771712（不限）。memguard 据此判 RED，是对的。测试改为与独立 cat 读数核对，gate 测试用合成 cgroup 树。
- 下一步：loopback（D14）、clockcheck、switchinterval、README；然后破坏性自测
- 19:30 写了 loopback/clockcheck/switchinterval + tests/envgate/test_envgate_misc.py；正在跑
- 19:32 misc 测试 4 passed；三个 README 已写。下一步：破坏性自测（scratchpad 副本）、全量回归、pgrep 清理、最终报告
- 19:33 全部 B4 测试一起跑 37 passed rc=0；加了清单首次可读时的 inventory 事件。开始破坏性自测（scratchpad/b4/sab_TAMPERED_DO_NOT_USE）
- 19:35 破坏性自测 12/12 变红（脚本 scratchpad/b4/sabotage.py，副本目录 sab_TAMPERED_DO_NOT_USE；抽查 5 组确认红在被针对的断言上）。
  /usr/bin/python3.11 -I 导入 14 个 capenv 模块，无第三方包。未提交 git。pgrep 无残留。
- **完成（2026-11-13）**。交付：tools/sidecar/capenv/sidecar.py、tools/canary/capenv/canary.py、tools/envgate/capenv/{gate,hwprofile,pins,memguard,gpu,loopback,clockcheck,switchinterval,procfs,topology,roles,runcmd}.py、三个 README、tests/{sidecar,envgate}（37 passed）。
- 未决 / 偏差（详见最终报告）：①包布局为命名空间包（INTERFACES 要 tools/capenv/）；②gate 加了可选参数 --max-alloc-gpus/--idle-window-s/--idle-busy-max-frac/--hw-profile/--roles-file/--timeout-s，第 6 项不给窗口即 NOT_RUN→ENV_INVALID；③新增文件 tids.jsonl、gate.<phase>.json、exclusivity.post.json、hw_raw.json、canary.meta.json；hw_comparable/1 格式自定；④hw_profile.json 的画像放在 payload 的 profile 键下；⑤chassis_model 来源由 --chassis-source 显式指定；⑥sidecar/canary 的 --ready-file 做成可选；⑦本 VM 上 systemd-run --user --scope 的 MemoryMax 对 memory cgroup 不生效（实测）。
