# B7-launch 进度(capkit-launch 编排器 + configs/e0 + fingerprint.py)

- 2026-11-13 开工：首次运行，无前序进度。正在阅读 INTERFACES.md / 规划 / 设计 §2.x §3 E0/E1。
- 已读 INTERFACES.md 全文(r2);下一步读 capcommon(runtree/gate/lifecycle/jsonio/identity/cli/tiers/snapshot/versions/expect)
- 已读 capcommon runtree/gate/lifecycle/jsonio/identity/cli/clock/hashing/snapshot/expect/tiers(部分);下一步读 00_plan DATA §1-6、CONSTRAINTS
- 已读 DATA §1-6、CONSTRAINTS 全文;下一步读设计稿 §2.2-2.8、E0/E1、§7
- 已读设计 §2.1-2.3;下一步 §2.4-2.8、E0/E1、§7
- 已读设计 §2.4-2.8、E0、E1;下一步 §7、陷阱清单、WT-21 脚手架、fake_openengine CLI、compute_tokenizer_fingerprint
- 已核对 baseline: fake_openengine CLI(:1966-1984 一致,--port-file 写纯文本端口)、compute_tokenizer_fingerprint(openengine_conventions.py:530)、facade 用 ModelLoader.load_hf_tokenizer、trtllm-serve --report_addr 写 host:port;下一步设计编排器

## 设计决定(2026-11-13,写代码前)
- 布局(写权限所限，与 INTERFACES §1.1 的偏差，照 B4 的做法):包 `caplaunch` 放 tools/launch/caplaunch/,`capstubs` 放 tools/stubs/capstubs/(无 __init__.py 的命名空间包)。
  运行:PYTHONPATH=<C>/tools:<C>/tools/launch:<C>/tools/stubs。集成者可 git mv 到 tools/caplaunch、tools/capstubs,import 名不变。
- 模块:scenario(场景配置校验、工具参数表、argv 与占位符)、procfs(/proc 读取、线程识别规则、标记扫描)、ready(就绪判据)、orchestrator(主循环：事件调度+清空门+窗口边界)、smoke、tiergen(档位表模板填数)、__main__。
- 新增必需参数(launch.json 需要):--workdir、--register、--capkit-sha、--probe-tree-sha、--baseline-sha、--snapshot-dir、--image-path、--image-sha256、--hw-profile-sha、--housekeeping-ms、--kill-after-s;值可写 none(落 null)。
- 每个子进程 env 带 CAPKIT_LAUNCH_MARK=<随机>,结束时扫 /proc/*/environ 查残留。
- 就绪判据种类:capkit_ready / plain_port_file / report_addr / http_get / file_exists;可组合(全部满足)。
- 占位符:{port_file:R} {addr:R} {param:NAME} {run_dir} {own_port_file} {own_ready_file} {own_report_addr} 等;argv 元素可为 {"splice_json":路径,"key":键}(fingerprint 输出)。
- STATE:编排器追加一行(by=caplaunch);与 INTERFACES §5.7「B6 写第一行」冲突，报告里列出。
- scenario.py: 角色校验已写;下一步 top-level 校验+argv 构建
- scenario.py 完成(校验+argv);下一步 ready.py
- orchestrator.py: 输入校验段已写;下一步 树/launch.json/起进程
- orchestrator.py: 起进程/登记/就绪 已写;下一步 发现、窗口边界、清空门
- orchestrator.py: 窗口边界/快照拷贝/tier.json 已写;下一步 清空门、线程清单、主循环
- orchestrator.py: run/startup/timeline 已写;下一步 停止/核验/收尾(launch_end、dims、STATE)
- orchestrator.py 主体完成;下一步 __main__.py(参数)、语法检查
- 测试计划：tests/launch/fakepkgs/{capload,capenv}(冒名的假工具，只在测试 PYTHONPATH 上)+ fakes/dummy_product.py;正在写
- conftest(构造最小 W、启动编排器)已写;下一步 test_launch_e2e.py
- 首次手工端到端(scratchpad/b7/try1)rc=0、STATE VALID、3 个清空门都过、无残留；修了 cpus_readback 取早了的问题。下一步写 pytest
- 开跑 tests/launch/test_launch_e2e.py(预计 1-2 分钟)
- e2e 8 passed(53.6s,rc=0);下一步 scenario 单元测试、fingerprint、tiergen、smoke、configs/e0
- fingerprint.py + test_fingerprint.py 已写(真 openengine_conventions.py 从 probe-tree 拷进假包)
- test_scenario/test_fingerprint: 16 passed + 1 测试写错(已修)+ configs 未写;下一步 tiergen + configs/e0
- 加 {input:tier_table|schedule|prompts} 占位符;下一步写 configs/e0
- 正在用 scratchpad/b7/mkconfigs.py 生成 configs/e0/*.json(交付物是生成出的 JSON)
- configs 生成器(scratchpad/b7/mkconfigs.py)可用：6 场景校验通过、6 模板样例值展开通过;下一步 httpprobe + smoke
- smoke.py 写完(环境前提/D13/D14/可读性/核预算/树检查);下一步 configs/e0 落盘
- configs/e0 已落盘(23 个 JSON,生成器 tools/launch/gen_e0_configs.py,重生成 diff 一致);新增 e0c-frontend-http 场景 + httpprobe 入工具表 + tiergen 拒绝 null 值;下一步 README 与测试
- 开跑 tests/launch 全量(预计 2 分钟)
- smoke 加身份参数(capenv.loopback 需要);VM 上实跑 smoke exit 0;开跑 tests/launch 全量
- 2026-11-13 收尾:tests/launch 全量 42 passed(54.9s,exit 0,systemd-run+timeout 包裹);破坏性自检 3 项全红(s1 标记扫描失效→orphan 测试红;s2 去掉请求数核对→smoke 测试红;s3 去掉 unused 检查→tiergen 测试红;副本在 scratchpad/b7/sab/TAMPERED_DO_NOT_USE);VM 实跑 caplaunch.smoke exit 0(FAIL 0/PASS 6/RECORDED 7/NOT_RUN 7);pgrep 无本组件残留进程
- README:tools/launch/README.md、tools/stubs/README.md、configs/e0/README.md
- 状态：完成，待集成者审阅；遗留见最终报告
