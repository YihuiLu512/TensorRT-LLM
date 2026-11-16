# fix-R2-launch（FX2-launch）进度

负责簇：Q11、Q17、Q21（生成器与 configs）。独占：`tools/caplaunch/**`、`configs/**`、`tests/launch/**`。

## 日志
- 2026-11-13 建 `_local/fx2-launch/{fakebin,pyc}`，开始读码。
- Q11：写了 tests/launch/fakes/hooked_launch.py + test_signal_between_popen_and_pid_still_kills_and_finalizes，先跑修前红
- Q11 已修（_stop_all 用 rt.popen.pid）；Q17 已重构（_entry_rule 一次计算、_register 一次写全、launch.json 只列角色名），跑 tests/launch
- Q21：生成器与 configs/e0 两个场景改为 [frontend, frontend_openengine, frontend_single]，重新生成 diff 只有这两处；README 同步；test_e0_proc_kinds 加严
- RS5/RS14 扫出：外部 CAPPROBE_TIER 冲突前移到 __init__；建树时拒绝已有 STATE；各补测试
- RS15：_private_dir 改为只包 ensure_own_dir（FX2-probe 的 Q20 已把 0o022 检查并入 ensure_own_dir）。23:02 跑 tests/launch 全目录（预计 3–6 分钟）
- 23:05 稳定性：Q11 测试连跑 5 次
- 23:06 终跑 tests/launch 全目录
- 收尾：tests/launch 66 passed（rc=0）；fakebin.log 为空；无残留进程。修前红证据：`_local/fx2-launch/pre-r1/`（HEAD 版 orchestrator + 新测试）里 4 条新测试全红。
