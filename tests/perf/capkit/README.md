# 01_code：承载上限测试的新代码（capkit）

本目录是只在本地的 git 仓库（提交身份 `Yihui Lu <yihlu@nvidia.com>`，不设 remote，不推送）。
第一个提交只含本文件与 `.gitignore`，作为每一轮三路自审的 base。

- 计划与约束：`../00_plan/`（先读 `../00_plan/README.md`）
- 实验设计定稿：`/home/scratch.yihlu_coreai/dev/feat-dev-J17503/tmp/meta/explain/06_arch-perf-capacity/_src/research/09_experiment_design.md`（§2、§7、E0/E1 卡）
- 接口约定：`INTERFACES.md`（各组件之间的文件格式、目录树、参数约定）
- 自审记录指针：`REVIEW.md`
- 进度：本文件末尾「进度表」；更细的事件流水在 `../STATUS.md`

## 布局

见 `../00_plan/DATA-AND-ANALYSIS.md` 1.1 节：`tools/`、`configs/`、`analysis/`、`jobs/`、`samples/`、`probe-tree/`（TRT-LLM 探针树，git worktree，不纳入本仓库）、`probe-tree-glm/`（同上）。

## 进度表

测试一律每个目录单独跑（`tests/capcommon` 的测试 `from conftest import …`，与别的目录同一次收集会互相遮蔽）：

```
cd <C> && systemd-run --user --scope -q -p MemoryMax=6G timeout 600 \
  <venv>/bin/python -m pytest -q -p no:cacheprovider tests/<目录>
```

`<venv>` = `/home/scratch.yihlu_coreai/tmp/20261106-wt03-venv/venv`（Python 3.11、grpcio、psutil、pytest；无 uvloop / numpy / tensorrt_llm）。本 VM 上 `systemd-run --user -p MemoryMax` 实际不生效（cgroup v1，B4 实测）。下表「测试结果」是集成提交前最后一轮（2026-11-13 r2）的记录，其中的命令与数字是当时的，不要照抄：判定器自测的 `--expect-samples` 取 `samples/` 下带 `expect.json` 的样本目录数（见 `samples/README.md`）。

| 件 | 包 / 目录 | 批 | 状态 | 测试与结果 | 已知缺口 |
|---|---|---|---|---|---|
| 仓库骨架 | — | — | 完成 | — | 首个提交 = 自审 base |
| 共享库（spec） | `tools/capcommon` | 1 | 完成 | `tests/capcommon` 187 passed | `gate.evaluate_gate` 不要求一个角色的全部登记进程都有快照（B5 未决 7）；`SnapshotWriter` 的 meta 刷新没有公开接口（capprobe 改私有字段，B3 未决 5） |
| B1 HTTP 负载器 / 空汇 | `tools/capload`（`capload`、`capload.nullsse`） | 1 | 完成，待自审 | `tests/capload` 26 passed；`tests/nullsse` 4 passed | `--loop uvloop`、容器 3.12、真实前端 NOT RUN；spawn 带出的 `resource_tracker` 进程未归属（INTERFACES 勘误「集成」第 7 条） |
| B2 gRPC 负载器 / 空汇 / 节拍源 / 中继 | `tools/capgrpc`（`.load`、`.nullgrpc`、`.reqbytes`）、`tools/capselfcheck` | 1 | 完成，待自审 | `tests/grpc` 25 passed | uvloop、真实 pb2 的 `openengine` 请求 NOT RUN；gRPC 负载器不数 token |
| B3 探针 | `tools/capprobe`；探针树 `probe-tree/`（分支 `capprobe-J17871`） | 1 | 完成，待自审 | `tests/capprobe` 87 passed；`capprobe.patchcheck --rev HEAD` PASS 7/7 | 探针树补丁已提交（pre-commit 重排了 import，位点行号不变）；P1/P2 覆盖清单未生成；`ts`/`bytes`/`queue`/`pin` 组第二批；H3 不覆盖 chat 流、F4 不覆盖池路径 |
| B4 环境 | `tools/capenv`（sidecar、canary、gate、hwprofile、pins、memguard、gpu、loopback、clockcheck、switchinterval） | 1 | 完成，待自审 | `tests/envgate` 33 passed；`tests/sidecar` 4 passed | 真实 `scontrol`/`squeue`/`nvidia-smi`、cgroup v2、多 NUMA NOT RUN；gate 第 6 项的窗口与阈值要由作业脚本传入 |
| B5 判定器 | `tools/capjudge`、`analysis/pilot_build.py`、`samples/` | 1 | 完成，待自审 | `tests/judge` 27 passed；`samples/run_selftest.py --package capjudge --expect-samples 22` rc=0 | 样本全为合成，要用 E0 第一个作业的真实良品替换 `good`（P8）；I、V、C2/C4/C12 第二批 |
| B6 作业脚手架 | `tools/capjobs`、`jobs/` | 1 | 完成，待自审 | `tests/jobs` 86 passed；`bash -n` 三个脚本通过 | 真实 Slurm、容器内 `capkit_inner.sh run` NOT RUN；E10-proxy / E13-main 白名单待用户定编号 |
| B7 编排器 / 指纹 / E0 配置 | `tools/caplaunch`、`tools/capstubs`、`configs/e0`（生成器 `configs/gen_e0_configs.py`） | 1 | 完成，待自审 | `tests/launch` 42 passed | 清空门只支持 `abs_max`；STATE 首行归属与 §5.7 不同；真实 `trtllm-serve` 场景 NOT RUN |
| 集成 | 全仓 | 1 | 完成，待三路自审 | 上面 10 个目录全部 rc=0；`py_compile` 172 个 .py、`bash -n` 3 个脚本通过；本地功能性端到端两条（`_local/`，不入库）：capload→nullsse 与 节拍源→relay1→排空端，均 `caplaunch` rc=0、STATE VALID，判定器无格式错误（只剩无 pilot / 无 GPU 类 NOT_RUN） | 见 `_progress/integrator.md` 与 INTERFACES「勘误与补充」的「集成（第一批）」条 |
