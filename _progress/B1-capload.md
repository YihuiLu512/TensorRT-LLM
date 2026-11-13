# B1-capload 进度(capload + nullsse)

- 2026-11-13 开始：读 INTERFACES.md、设计 §2.2 等。尚未写代码。
- 已读 INTERFACES §0-§4.5;下一步 §5-§8、§11
- 已读 INTERFACES 全部相关段(§0-§8,§11,§12);下一步读设计稿 §2.2
- 已读设计稿 §2.2、E1、E12(波模式)、§7;下一步读 capcommon API
- 已读 capcommon 各模块 API、陷阱清单;下一步开始写 tools/capload
- 设计决定(写代码前,中文记录便于接手):
  - 包:tools/capload/ = capload(负载器)与 capload.nullsse(空汇,INTERFACES §11 规定的入口;不另建 tools/nullsse 包)。测试放 tests/capload、tests/nullsse。
  - 进程模型:父进程(asyncio,同 --loop 实现)+ spawn 出 W 个 worker;父进程也发快照(槽全登记,backlog.* 恒 0),因为清空门要求每个登记进程都有快照。worker 忽略 SIGTERM/SIGINT,只听父进程经 Pipe 发的 stop/hard;Pipe EOF = 父进程死 → 优雅停。expected_procs = workers + 1。
  - 开环:每 worker 一个分发定时器(call_at_ns 对准下一条的计划时刻,到点把 planned<=now 的全部发出,不跳过)。
  - 连接池:keepalive 池无上限;roundrobin=空闲连接 FIFO(最久未用先用),random=带种子随机;--pool-reset per-tier 在每个负载档开始关掉全部空闲连接;复用连接上 0 字节回包即断 → 换新连接重发一次并计 reconnects(不算失败)。
  - 接收:Protocol 内 HTTP/1.1 响应解析 + chunked 解码 → wire.SseCounter;full 模式另外逐帧 json.loads 交叉核对帧数。
  - 超时:deadline = 计划时刻 + request_timeout_ns,由内务 tick(--fionread-every-ms 周期)扫;FIONREAD 也在该 tick 里对每条打开的连接取。
  - 慢读阳性对照:可选 --slow-read-pause-us(每次 data_received 后 pause_reading 一段)。
  - 波模式:可选 --mode wave --wave-size C [--wave-delays-ns] [--wave-gap-ns];每个负载档按时间表里本 worker 的 req_id 顺序成波发送,波内全部完成才起下一波;计划时刻=波起点+该槽延迟。
  - 新增输出:每 worker 每档 conns.capbin(连接→对端映射,E6 用);失败原因新增 aborted_at_stop(偏差,报告里列)。
- 已写 capload/__init__.py(SLOTS)、_util.py、_proc.py(父子进程);下一步 httpwire.py(请求模板 + 响应解析)
- httpwire.py 已写;正在写 loadgen.py(worker 核心,分段写)
- loadgen.py: Req/Conn/IdlePool/Worker.__init__/_main 已写;下一步 边界/分发/收包/内务/收尾
- loadgen.py: 边界/分发/建连/写请求 已写;下一步 收包/完成/断连/内务/停止/波模式/收尾
- loadgen.py: 收包/完成/断连 已写;下一步 内务 tick/停止/波模式/收尾落盘
- loadgen.py: tick/停止/波模式 已写;下一步 _teardown + report.py(逐档自报与落盘)
- loadgen.py 完成(含 _teardown/worker_entry);下一步 report.py
- report.py 已写;下一步 main.py(capload 命令行)与 __main__.py
- main.py、__main__.py 已写;下一步 sink.py(nullsse worker)与 nullsse.py(入口)
- sink.py: 常量/Stream/SinkConn 已写;正在写 Sink 类
- sink.py: Sink.__init__/_main/边界 已写;下一步 请求解析/发帧/节拍/空闲关闭/停止/收尾
- sink.py: 请求解析/发帧 已写;下一步 节拍堆/空闲关闭/断连/停止/收尾 + nullsse.py
- sink.py 完成;下一步 nullsse.py 入口
- nullsse.py 已写;lag 逐样本记录在最后一档结束后停止(只留直方图)。下一步:冒烟 import + 写测试夹具
- import 冒烟通过(venv py3.11);正在写 tests/capload/conftest.py(启动夹具)
- 第一个端到端测试已写,开始跑
- test_e2e_basic 通过(rc=0);下一步 看一份 summary 与 ready 文件,再写其余测试
- 修复:nullsse 监听 socket proto=0 导致 asyncio 不设 TCP_NODELAY(复用连接上 ~40ms 延迟确认停顿);现两端显式设 TCP_NODELAY,首帧延迟降到 ~0.2ms。test_e2e_basic 通过。下一步写 test_e2e_modes.py
- test_e2e_modes.py 5 项通过(rc=0,26.8s):new-per-request accept 计数、keepalive 复用、beat 节拍(帧不早于计划步)、中途停止 planned≠sent + rc=5 + stop 抄录、慢读 FIONREAD 非零(对照=0)。下一步 test_e2e_more.py(注入心跳/short、波模式、full+chat、超时、空闲关闭、配置错误、协议错误)
- test_e2e_more.py 9 项通过(rc=0,28.6s)。下一步:httpwire 单元测试、tests/nullsse、破坏性自测、README
- tests/nullsse 4 项通过(rc=0);httpwire 单测 11 项通过。下一步:破坏性自测(scratchpad 副本,改坏后断言变红)
- 破坏性自测 8 组全红(脚本 scratchpad/b1sab/run.py;missing_done 一组首次没红,已把假服务器改成 usage 正确但缺 [DONE] 后变红)。下一步:全套测试复跑、README
- 热路径上的函数内 import 已提到模块级;删除未用代码。下一步 README.md
- README.md 已写;正在做最终复跑(全套 + 破坏性自测)
- 最终复跑:tests/capload+tests/nullsse 30 passed rc=0;破坏性自测 8/8 变红 rc=0;本人起的进程无残留(pgrep 里看到的 capload/nullsse 进程属于别的组件的测试:参数含 --slot/--child-of,run-dir 在 /tmp/pytest-of-yihlu,不是我起的,未动)

## 完成(2026-11-13)
- 代码:tools/capload/{__init__,__main__,_proc,_util,httpwire,loadgen,sink,report,main,nullsse}.py + README.md(中文,含偏差与新增输出)。
- 测试:tests/capload/{conftest,capload_support,test_e2e_basic,test_e2e_modes,test_e2e_more,test_httpwire}.py;tests/nullsse/{conftest,test_nullsse_direct}.py。
  `pytest -q tests/capload tests/nullsse` 30 passed rc=0;破坏性自测 8/8 变红(scratchpad/b1sab/run.py,副本在 scratchpad/b1sab/TAMPERED_DO_NOT_USE/)。
- 未提交 git(由 integrator 提交)。
- 未决 / 待接手者注意:
  1. uvloop 路径本机无法跑(venv 无 uvloop):PseudoSocket 的 fileno/setsockopt、毫秒定时器,须在容器 E0 冒烟时实测。
  2. tests/capcommon 用 `from conftest import`,与其他测试目录同次收集会冲突(spec owner 的问题),目前分目录跑。
  3. 偏差清单见 README「capload 的输出」与最终报告:aborted_at_stop、conns.capbin、counts.sent=已分发、父进程也发快照(expected_procs=W+1)、超时从计划时刻起算、波模式语义、nullsse 的 --max-records-per-worker 可选。
