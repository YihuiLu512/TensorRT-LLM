# B5-judge 进度(judge.py + 统计 + pilot_build + 样本 + 自测)

- 2026-11-13 开始:无先前进度。正在读 INTERFACES.md(绑定)。
- 已读 INTERFACES.md 全文。冲突:INTERFACES 定包名 tools/capjudge/、pilot 为 capjudge.pilot;任务给的写入路径是 tools/judge/、analysis/。处置:代码写在 tools/judge/(包内全相对 import,改名为 capjudge 即可),analysis/pilot_build.py 作薄 CLI;报告里列出。下一步读设计稿 §2.6/2.7/2.8/E0/E1。
- 已读设计稿 §2.1/2.6/2.7/2.8、E0/E1 卡、DATA §2-6、CONSTRAINTS §4、陷阱清单、capcommon(ledger/runtree/jsonio/identity/expect/gate/versions/tiers/schedule/capbin/hist/snapshot)。
- 设计(定稿,写代码按此):
  - 包 tools/judge/:stats.py(T1-T6)、thresholds.py(只从已登记标定文件读阈值)、inputs.py(§10.1 输入核对)、treedata.py(读树、逐秒序列)、checks.py(C1/C3/C5/C6/C7/C8/C9/C10/C11/C13/GPU-IDLE + P-SENT/P-PIDS/OVERFLOW/PROBE-TIER)、symptoms.py(X1/X2/X3,I/V 输出 NOT_IMPLEMENTED_BATCH2)、judge.py(CLI)、pilot.py。
  - 适用检查 = 强制集 {C3,C10,C11,GPU-IDLE} ∪ register.judge.checks;登记了但第二批才有的(C2/C4/C12)→ NOT_RUN(NOT_IMPLEMENTED_BATCH2)。
  - 阈值键 `<T>:<量>:<角色>`;C1 固定四项(T1:send_lag_p50_ns、T3:send_lag_p99_ns、T3:fionread_nonzero_frac、T3:loop_lag_p99_ns)。
  - slot: 源的对账:用 sidecar/snapshots.jsonl,区间 = [本档开始前最后一份, 下一个清空档结束前最后一份且 ≥ 门过时刻]。
  - X 矛盾按对称规则:X2 真而 X1 显著超交付或 X3 显著下降 / X2 假而 X1 显著不足或 X3 显著上升 ⇒ VOID。
  - 样本由 samples/make_samples.py 生成(阈值直接写进 pilot.json 并登记);runner samples/run_selftest.py。
- 下一步:写 stats.py → thresholds.py → inputs.py → treedata.py → checks.py → symptoms.py → judge.py → pilot.py → 样本生成器 → runner → tests。
- 已写 stats.py、thresholds.py、inputs.py、treedata.py、symptoms.py、checks.py(C1/C3/P-SENT/P-PIDS/RECORDS/C5-7/C8);下一步 checks 的 C9/C10/C11/C13/GPU-IDLE/PROBE-TIER
- checks.py 全部检查已写(C9/C10/C11/C13/GPU-IDLE/PROBE-TIER);下一步 judge.py(CLI/编排/输出)
- judge.py 已写并能 import;下一步 pilot.py,然后样本生成器
- make_samples.py:Sim/Builder/tiers/server/gate/sidecar 已写;下一步变体表、expect.json、main,然后跑一次
- 生成器可跑;正在用 good 样本调判定器
- runner 已写;正在 scratchpad 生成全部样本并跑 runner
- 全部 22 个样本 runner 通过(rc=0),逐个核对了红项原因与设计一致;下一步 pilot.py + analysis/pilot_build.py
- pilot.py 与 analysis/pilot_build.py 已写;下一步在 scratchpad 用已判过的样本试跑 pilot
- tests/judge 写好(stats/thresholds/selftest/tamper/pilot),正在跑
- 已写 tools/judge/README.md、samples/README.md,样本已生成进 samples/;下一步跑全套测试(含已提交样本)
- empty_tree 改为由 runner 现建空目录(git 不收空目录);正在跑全套 tests/judge
- 修:有过载型失败时不判「X2 假而 X1/X3 指向过载」的矛盾(截断是预期)。下一步:补一个单元测试覆盖 _combine 的各分支,再跑全套
- 完成(2026-11-13):tools/judge/(9 个 .py + README)、analysis/pilot_build.py、samples/(make_samples.py、run_selftest.py、README、22 个合成样本 W)、tests/judge/(7 个文件)。
  测试:venv python -m pytest -q tests/judge → 27 passed rc=0(含生成样本自测、已提交样本自测、3 种篡改判定器副本均使自测变红、pilot 3 项);tests/capcommon 187 passed(未改 capcommon)。
  未提交 git;无残留进程。
- 未决 / 偏差(写进最终报告):包名 tools/judge vs INTERFACES 的 capjudge;launch.json roles 需要 kind;tier.json threads 形状、sidecar meta interval_ms/clk_tck、pilot d14_softirq 形状是本包的读取约定;gate.evaluate_gate 不要求角色全部进程都有快照(判定器复核时要求);slot: 源 scope=tier 用 sidecar 快照序列;X 矛盾规则对称化;C2/C4/C12、I、V、baseline_band 复核为第二批。
