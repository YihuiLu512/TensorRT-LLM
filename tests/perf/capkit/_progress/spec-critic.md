# spec-critic 进度

- 开始：读 INTERFACES.md / capcommon / 设计 / 计划
- 已读 INTERFACES.md 全文；下一步读设计稿 §2.1-2.8
- 已读设计稿 §2.1-2.8；下一步读 E0/E1/§7 与计划文件
- 已读 E0/E1/§7；下一步读计划文件
- 已读计划文件 DATA §1-8.1、CONSTRAINTS §2/4/6/7、RESOURCES §3-5；下一步读 capcommon 代码与陷阱清单
- 已读 identity/cli/tiers/schedule/gate/snapshot；下一步 aio/capbin/inject/lifecycle/versions/ledger/runtree/clock
- 已读 capcommon 全部关键模块；下一步读陷阱清单并整理发现
- 已读陷阱清单。发现已整理约 20 条（主要：原点时序矛盾、对账链与每请求消息数公式缺失、SSE 帧定义与 nullsse 参数来源、X2 在 SSE 路径无输入、清空门不查快照新鲜度且 shm 目录按作业不按启动、注入心跳的逐档语义、C10 必需字段不分角色、C13 线程清单时机、串联中继无法区分、sidecar 缺逐线程亲和性、GPU 为 0 检查无人负责、时钟自检无人负责、硬件画像 sha 字段未定、应检进程数来自观测、档位表缺负载轴值、角色测量类别缺失、提交断言与独占回退文件未定义等）。
- 下一步：调用 StructuredOutput 交付；本任务只读，未运行任何测试、未启动任何进程。
