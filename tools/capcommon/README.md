# capcommon

capkit 的共享库，只依赖标准库（Python ≥ 3.11；容器里是 3.12）。它是 `<C>/INTERFACES.md` 的代码化部分：文件格式、身份、档位表、时间表、快照、注入、台账都只有这里一份实现。模块与接口的速查表见 `INTERFACES.md` §11.1。

- 入口：只有一个命令行入口 `python3 -m capcommon.scratchpath`（本地盘清理前的路径断言，INTERFACES §5.9）；其余模块只被别的包 import。
- 2B 新增模块：`jobinputs`（场景 `job_inputs` 段的种类、校验、生成命令行与生成记录的核对；INTERFACES 勘误 B2B-9）。
- r2 新增模块：`expect`（每请求消息数 / 帧数与对账段）、`wire`（请求头、gRPC 请求头、SSE 帧与计块）、`hwprofile`（画像键表与校验值）、`policy`（不加载模型的提交断言）、`scratchpath`（清理断言）。
- 测试：`PYTHONPATH=<C>/tools python -m pytest tests/capcommon`（`tests/capcommon/conftest.py` 会自己插 `sys.path`，所以在 `<C>` 下直接 `python -m pytest tests/capcommon` 也可以）。
- 发布的快照槽：本包本身不发布槽；`Injector` 为每个位点登记 `inject.<site>.hits/.applied/.applied_ns`，`LoopLagProbe` 登记 `lag.<name>` 及 `.n/.sum/.max/.neg`。
- 被 `capprobe` 使用的模块（`snapshot`、`hist`、`inject`、`clock`、`versions`、`exitcodes`）不得新增对其他模块的依赖。
