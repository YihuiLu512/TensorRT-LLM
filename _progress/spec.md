# spec 进度(INTERFACES.md + tools/capcommon)

- 2026-11-13 开始:无先前进度。正在阅读规划与设计文档。
- 已读 00_plan README/DATA-AND-ANALYSIS/CONSTRAINTS/RESOURCES;下一步读设计稿 §2、§3 E0/E1、§7
- 已读设计稿 §2.1-2.8;下一步读 §3 E0/E1、§7、陷阱清单、WT-21 脚手架
- 阅读完成(设计稿 §2、E0/E1、§7;陷阱清单;WT-21 sidecar 头部)。
- 设计决定(草案,写入 INTERFACES.md 后以其为准):
  - 包名一律 cap* 前缀的顶层包:capcommon/capload/capgrpc/capselfcheck/capprobe/capenv/capjudge/caplaunch/capstubs/capjobs;调用方式 `python -m <pkg>[.<mod>]`;仓库内 PYTHONPATH=<C>/tools,容器内拷进 dist-packages。
  - 档位表:相对 origin 的 ns,连续无缝,up/down 前必须是 drain 档;origin 由编排器运行时以 --origin-ns 传入(不在登记文件里)。
  - 档切换=各工具按固定时刻自己切;清空门=drain 档内轮询 backlog.* 槽位,超时则 stop.json + SIGTERM。
  - 二进制容器格式 capbin(CAPBIN01 + JSON 头 + 列);快照 CAPSNAP1 写 <shm>/<pid>.bin。
- 下一步:写 INTERFACES.md 骨架 → 分段填充 → 实现 capcommon → 测试。
- INTERFACES.md §0-§4 已写;正在写 §5 数据树
- INTERFACES.md §5-§8 已写;正在写 §9-§12
- INTERFACES.md §0-§12 初稿完成(约 600 行)。下一步:实现 tools/capcommon/ 各模块(clock, hashing, identity, jsonio, tiers, schedule, capbin, hist, snapshot, inject, aio, lifecycle, versions, ledger, cli, exitcodes, runtree),再写 tests/capcommon/。实现后回头核对 INTERFACES 里引用的函数名一一存在。
- capcommon: __init__, exitcodes, clock, hashing, hist, identity, jsonio, tiers 已写;下一步 inject、capbin、schedule
- capcommon: inject, capbin, schedule 已写;下一步 snapshot、aio、lifecycle、versions、ledger、cli、runtree;然后 tests
- capcommon 全部模块初稿完成;下一步 import 冒烟 + tests
- tests: conftest, hist, identity_jsonio, tiers 已写
- tests: schedule, capbin 通过(47 passed);下一步 snapshot/inject/aio/lifecycle/ledger/cli/runtree 测试
- 全部 capcommon 测试 67 passed;下一步:破坏性自测(拷贝到 scratchpad 改坏,断言变红)
- 破坏性自测 5/5 变红;INTERFACES 引用的 capcommon 名字 30 个 + 速查表 130 个全部存在。下一步:通读 INTERFACES.md 自查一致性
- INTERFACES 通读自查进行到 §11(已修若干处:§8.2 槽名/flag 名、register files_by_arm、judge 输入核对顺序、sidecar snapshots 列)
- 新增 capcommon.gate(清空门读数的算术,编排器与判定器共用)及测试;全套 71 passed。INTERFACES §8.3、§11.1 已同步。
- 剩余:最终一致性复查(函数名交叉核对再跑一次)、pgrep 清理检查、写最终报告。
- 完成(2026-11-13):INTERFACES.md(约 850 行)+ tools/capcommon(18 个模块 + README)+ tests/capcommon(11 个文件,71 passed,rc=0)。
  破坏性自测 6 组(contig/fixed/phase/histsum/overflow/gate-missing-slot)均变红;gate 一组首次没变红,补测试后变红。
  未提交 git(按规则只由 integrator 提交)。无残留进程(pgrep 空)。
- 未决:见 INTERFACES.md §12 的 7 条偏差与 U-1~U-4;submissions.tsv 用事件行需用户/评审确认。

## 修订轮(审阅意见 22 条)— 2026-11-13
- 开始:读 INTERFACES.md 全文 + capcommon,再逐条对设计稿核对审阅意见。
- 已读 INTERFACES.md 全文;下一步读 capcommon gate/inject/versions/ledger/cli/tiers 与设计稿相关段
- 已读设计稿 §2.1-2.8/E0/E1、计划 CONSTRAINTS §2/§6、RESOURCES §3-5、DATA §4/§8.1、陷阱清单;下一步读 tiers/schedule/snapshot/runtree 代码,然后逐条裁决
- 裁决(24 条全部核对过设计/计划原文):接受 #1(选方案 a:编排器先定 origin=now+lead,原点前未全就绪 → NOT_RUN)、#2(新 capcommon.expect + 场景 reconcile 段)、#3(新 capcommon.wire:HTTP 头 x-capload-k/s/tier,gRPC <qiiii,SSE 模板与 frames 口径)、#4(nullsse/nullgrpc 用 BeatSchedule;nullsse 落 streams capbin)、#5(gate 快照新鲜度/seq/starttime/launch_key;shm 每启动子目录;poll 计划时刻对账)、#6(inject 生效以来计数、tier 增量、pending_at_stop、aux 预建)、#7(C10 按角色声明;不过→NOT_RUN)、#8、#9、#10(relay1/relay2 不同角色名)、#11(workers/expected_procs)、#12(axis/load_value;order 改名见记录)、#13(measure_class;sink 并入 stub_out_of_domain=部分采纳)、#14(新 capcommon.hwprofile 键表)、#15(slurm_view)、#16(新 capcommon.policy + submissions 三列)、#17(检查号 GPU-IDLE,不用 G1 避免与设计位点 G1 撞名)、#18(SLOTS/SITES + register_decls)、#19(capenv.clockcheck)、#20(request_timeout_ns 进档位表)、#21、#22、#23(台账毫秒精度 + --local-record)、#24(新 capcommon.scratchpath)。
- 改法:INTERFACES 原地修订(尚无组件代码基于旧版;_progress 里只有 spec/spec-critic),在「勘误与补充」记一条总说明 + 「审阅记录」逐条。
- 下一步:先改 capcommon 代码(clock→ledger/policy→tiers→inject→gate→versions→expect/wire→hwprofile→snapshot→scratchpath→runtree),再改测试,再改 INTERFACES。
- 已改 clock(毫秒)、policy(新)、ledger(三列+断言)、tiers(axis/load_value/request_timeout_ns)及相应测试;71 passed。下一步 inject
- inject 改完(生效以来计数、mark/tier_report、prestart_aux、aux_pending)+ 测试;下一步 gate
- gate 改完(snapshot_fields/check_snapshot/新 snapshots_by_role 签名/planned_polls;gate 配置加 max_snapshot_age_ns)+ 测试 75 passed;下一步 versions(C10)
- 新增 expect.py、wire.py、hwprofile.py、scratchpath.py;snapshot 加 register_decls/validate_slot_decls;versions 加 required_fields/check_versions。下一步 runtree 助手 + 新测试文件 test_revision.py
- 写了 tests/capcommon/test_revision.py;正在跑
- 187 passed;下一步破坏性自测(scratchpad 副本)
- 破坏性自测 10 组全红(脚本 scratchpad/sabotage_TAMPERED_DO_NOT_USE/run.py)。下一步改 INTERFACES.md(分批:§0-2 → §3-4 → §5 → §6-8 → §9-11 → §12/勘误/审阅记录)
- INTERFACES §0/§2 已改
- INTERFACES §3 已改
- INTERFACES §4.4/§4.5 已改
- INTERFACES §5.1-§5.4 表头已改;下一步 §5.4.1/5.4.2/5.5/5.6
- INTERFACES §5.4.1/§5.4.2/§5.5 已改;下一步 §5.6 + 新 §5.8/§5.9
- INTERFACES §5.6/§5.8/§5.9 已改;下一步 §6-§8
- INTERFACES §6/§7.3/§8 已改;下一步 §9-§11
- INTERFACES §9/§10 已改;下一步 §11/§11.1
- INTERFACES §11 已改;下一步 §11.1 速查 + §12 + 勘误 + 审阅记录
- INTERFACES §11.1/§12/勘误/审阅记录已写;README 已更新。下一步:一致性复查(INTERFACES 引用的 capcommon 名字是否都存在、残留旧口径 grep)
- 名字交叉核对 230 个 0 缺失;187 passed;python3.11 纯标准库导入 23 模块无第三方。下一步:通读改动段落
- r2 完成(2026-11-13):INTERFACES.md 原地修订(审阅意见 24 条:23 条采纳、1 条部分采纳,逐条见文末「审阅记录」);§5.5.x 编号重排(5.5.1 绑核表 / 5.5.2 slurm_view / 5.5.3 画像键)。
  capcommon 新增 expect/wire/hwprofile/policy/scratchpath,改 clock/ledger/tiers/inject/gate/versions/snapshot/runtree;tests 187 passed(rc=0);破坏性自测 10 组全红;名字交叉核对 230/0 缺失。未提交 git。无残留进程。
- 未决:§12 U-5(P0 档产品进程无版本自报)、U-6(产品前端块数修正值待 E0)、偏差第 11 条(E10 替身臂/E13 主体编号待用户定)。
