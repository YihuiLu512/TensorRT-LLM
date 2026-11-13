# fix-R1-env（FX-env）进度

负责簇：K1（门禁侧）、K6（gate 第 5 项）、K7（gpu.py）、K18（capenv.gpu 唯一实现）
独占文件：tools/capenv/**、tests/envgate/**、tests/sidecar/**
状态：**完成**（未 commit；fakebin.log 为空；无残留进程）

## 时间线
- 开工：建 fakebin（`_local/fx-env/fakebin/`），读 triage/brief、A-R1/R6/R7、B01/B06/B07、INTERFACES §5.5/§11、CONSTRAINTS §2。
- 收到主会话裁定（`nvsmi` 关键字、gate/hwprofile 必填 `--nvidia-smi`、`--max-alloc-gpus` 必填），按此实现。
- 改 gpu.py / gate.py / hwprofile.py / sidecar.py / README；补测试；修前代码副本与变异自测证明新测试会红；两个测试目录全绿。

## 各簇核实与改动

| 簇 | 核实 | 证据 | 改动 |
|---|---|---|---|
| K1（门禁侧） | 属实 | gate.py 修前：item6 缺参数 ⇒ NOT_RUN；`--max-alloc-gpus` 缺省 None 只记 "not requested"；`--registered-hw-sha` 只收 64 hex | `--max-alloc-gpus`、`--nvidia-smi` 改 argparse 必填；`--phase pre` 缺 `--idle-window-s/--idle-busy-max-frac` ⇒ 退出 3（另校验 window>0、frac∈[0,1]、gpus≥0）；新增 `UNREGISTERED`：照跑 1～6 项、读画像，`hw_gate.json` 记 `status: NOT_RUN`、`node_hw_sha`，`run_gate` 必加一条 reason，`main` 对 UNREGISTERED 无条件返回 6（两道保证）；`hw_gate.json` 增 `status`，gate.pre.json 增 `hw_gate:{status,node_hw_sha}`、`registered_hw_sha` |
| K6（gate 第 5 项） | 属实 | B07 场景：请求 1、可见 2 时修前 item5 PASS（测试 `test_item5_more_gpus_visible_than_requested_is_red` 在修前代码上红） | item5 改调 `capenv.gpu.check_visible`：可见数 == `--max-alloc-gpus`；请求 0 时要求查询成功且为空或输出 `No devices were found`，看见卡 RED、查不了 NOT_RUN，阳性对照与计算进程检查记 NOT_RUN（"0 GPUs requested: needs a listed GPU"），item5 状态取计数检查 |
| K7（gpu.py） | 属实 | gpu.py:154 修前 `(util_pct or 0) > 0`，`[N/A]` 行当 0；另 `sample()` 把 pid `[N/A]` 的计算进程丢掉（RS3 额外） | `idle_check`：util 为 None 的行计 `unreadable_util` ⇒ NOT_RUN（有忙碌证据时 RED 优先）；`_pids()` 保留 pid 读不到的进程为 `null`；sidecar 计 `gpu_util_unreadable_rows` 并写进 `meta.unreadable.gpu_util` |
| K18 | 属实 | gpu.py:27 修前 `NVSMI="nvidia-smi"` 写死、门禁/画像/sidecar 都走 PATH | 所有查询函数加关键字 `nvsmi`（缺省仅供单测；None/空 ⇒ 不执行，reason "nvidia-smi path not given"）；gate、hwprofile 必填 `--nvidia-smi`；sidecar 可选 `--nvidia-smi`（不给就不跑 GPU 查询，不退回 PATH）；`capenv.gpu check` CLI 必填 `--nvidia-smi --expect-gpus` |

## 最终接口签名

```
capenv.gpu（nvidia-smi 查询的唯一实现）
  query_gpus(timeout_s, runner=runcmd.run, nvsmi="nvidia-smi") -> (record, [gpu])
  query_apps(timeout_s, runner=runcmd.run, nvsmi="nvidia-smi") -> (record, [app])
  positive_control(timeout_s, runner=runcmd.run, nvsmi="nvidia-smi") -> (ok, count, record)
  sample(timeout_s, runner=runcmd.run, nvsmi="nvidia-smi") -> (rc, rows)
  check_visible(expect_gpus, timeout_s, runner=runcmd.run, nvsmi="nvidia-smi") -> dict
      keys: status(PASS|RED|NOT_RUN) reasons expected_gpus nvidia_smi visible_gpus(None=读不到)
            gpus compute_apps(None=没查) positive_control_ok count_check positive_control
            apps_check({status,why} 各一) queries([{cmd rc stdout stdout_sha256 stderr reason}])
      expect_gpus 不是 >=0 的 int ⇒ ConfigError
  idle_check(samples, every_s, timeout_s, expect_gpus, runner=runcmd.run, sleep=time.sleep,
             nvsmi="nvidia-smi") -> dict
  python -m capenv.gpu check <身份> --out P --samples N --every-s S --timeout-s S
      --expect-gpus N --nvidia-smi PATH

python -m capenv.gate <身份> --run-dir D --phase {pre,post}
    --registered-hw-sha {64hex|UNREGISTERED} (--comparable-file P | --no-comparable)
    --slurm-view-file P --slurm-view-wait-s S --max-alloc-gpus N --nvidia-smi PATH
    [--idle-window-s S --idle-busy-max-frac F]   # pre 必填（缺 ⇒ 3），post 不用
    [--hw-profile P] [--roles-file P] [--timeout-s S]
  退出码：0 PASS；6 ENV_INVALID（UNREGISTERED 恒为 6）；3 配置错；2 argparse 缺必填

python -m capenv.hwprofile <身份，含 --role hwprofile> --out P --raw-out P --local-dir D
    --chassis-source {slurm_view,dmi} --nvidia-smi PATH
    [--slurm-view-file P] [--image-sha256 HEX] [--code-baseline SHA40]
    [--container-runtime TEXT] [--timeout-s S]
  Collector(..., nvidia_smi=None)  # None ⇒ GPU 键 unreadable("nvidia-smi path not given")

python -m capenv.sidecar ...（INTERFACES §11 全部）... [--nvidia-smi PATH]
  Sidecar(..., nvidia_smi=None)
```

主会话的内层脚本调法（`--max-alloc-gpus N --idle-window-s S --idle-busy-max-frac F --nvidia-smi PATH --hw-profile $JOB_DIR/env/hw_profile.json --registered-hw-sha <64hex|UNREGISTERED>`；hwprofile 带身份参数、`--role hwprofile`、`--container-runtime pyxis-enroot --nvidia-smi PATH`）与上面签名一致：hwprofile 用 `add_identity_args`，接受 `--role hwprofile`。

## RS 规则额外扫出

1. **RS3** `gpu.sample()`：pid 为 `[N/A]` 的计算进程被 `if p is not None` 丢掉 ⇒ 忙卡可报空闲。已修（`compute_pids` 记 `null`），测试 `test_compute_app_with_unreadable_pid_is_busy`。
2. **RS3** gate 第 1 项：`scontrol show job` 输出没有 `TRES`/`AllocTRES` 时 GPU 数按 0 计、≤ 上限通过。已修（记红 "GPU allocation unreadable"），测试 `test_item1_without_tres_is_not_zero_gpus`（修前代码上红）。
3. **RS3** hwprofile：`memory.total` 为 `[N/A]` 时 `sorted([None, …])` 会崩或把 None 写进进 sha 的字段。已修（记 unreadable），测试在 `test_hwprofile_uses_the_explicit_nvidia_smi_path`。
4. **RS3** `idle_check`：修前采样失败优先于忙碌；改为有忙碌证据即 RED（RED 会让启动 VOID 并上报，比 NOT_RUN 更强）；可见数不等仍采样留证。
5. 扫过未改：memguard / pins verify / switchinterval 的空集都已是 NOT_RUN；loopback 的两个子进程在 finally 里 quit→wait→kill；sidecar `makedirs(out_dir, exist_ok=True)` 的目录由编排器 `RunTree.make_skeleton` 预建（RS10 的建目录方在编排器，K28 归 FX-launch）；switchinterval 只判「少于 N」不判「多于 N」，与其 README 口径一致，未改。lscpu/numactl/free/lsblk/df 仍走 PATH（K18 只针对 nvidia-smi，未扩大）。

## 需要他方配合

- **FX-jobs**：
  1. 内层脚本：门禁前先跑 `capenv.hwprofile ... --nvidia-smi "$CAPJOBS_NVIDIA_SMI"`；门禁加 `--max-alloc-gpus --nvidia-smi --idle-window-s --idle-busy-max-frac`（缺任一 ⇒ 门禁退出 2/3）。若以后加 `--phase post`，同样要传 `--max-alloc-gpus --nvidia-smi`。
  2. 已核对 `tools/capjobs/envfiles.py` 现有 `check_gpu_idle` 调 `capgpu.check_visible(max_alloc_gpus, timeout_s, nvsmi=nvidia_smi)`，与本签名一致。
  3. UNREGISTERED 作业门禁恒退出 6，内层脚本会写 STATE=ENV_INVALID；CONSTRAINTS §7「同一原因连续两次 ENV_INVALID 才停下上报」是否把这类「只采画像」作业计入，需主会话定口径（建议 env_invalid 的原因文字区分 "UNREGISTERED"）。
- **FX-launch**：场景配置（configs/e0/*.json）里 sidecar 没有 `--nvidia-smi`，现在 sidecar 不给路径就不跑 GPU 查询 ⇒ 每次启动 GPU-IDLE NOT_RUN（fail-closed）。需要编排器从作业层拿到 nvidia-smi 路径（例如 `capjobs.launches` 传 `caplaunch --nvidia-smi`，编排器替换进 sidecar 参数）。K3：sidecar 每次 GPU 采样起 2 个短命 nvidia-smi 子进程（`--query-gpu`、`--query-compute-apps`），可执行文件路径即 `--nvidia-smi` 的值，可用于 descendants 口径的识别。
- **FX-judge**：`gpu.jsonl` 的 `compute_pids` 可能含 `null`（pid 读不到的计算进程），checks.py:524 现按「列表非空」判忙，兼容；`util_pct` 为 null 的行按 K7 判定侧处理。请求 0 张时 `gpu_idle.json` 的 `positive_control_ok` 为 false ⇒ GPU-IDLE NOT_RUN，见下一条。
- **主会话**：
  1. 请求 0 张 GPU 的启动：作业层门禁/`gpu-idle` 按 K6 口径可通过，但判定器的 GPU-IDLE 会 NOT_RUN（没有卡可查利用率）。这对 0 卡启动的最终结论有什么影响（整次 NOT_RUN 还是该项不适用）需定。
  2. 画像 sha 含 `gpu_count / gpu_mem_mib / gpu_model / gpu_driver`，取的是**可见**卡：请求 0 张与 1 张的作业画像 sha 必然不同（0 张时 GPU 键为 unreadable 标记）⇒ 主型号登记要按申请的 GPU 数分别登记（INFERRED，未在集群验证）。
  3. INTERFACES 勘误原文（建议追加到「勘误与补充」）：

> - **2026-11-13，R1 自审修复（capenv）**：§11 `capenv.gate` 行新增必填参数 `--max-alloc-gpus N`（向 Slurm 申请的 GPU 数）与 `--nvidia-smi PATH`；`--phase pre` 另需 `--idle-window-s S`、`--idle-busy-max-frac F`（取自登记表的 `environment` 段，门禁不设缺省，缺了以退出码 3 退出）。`--registered-hw-sha` 新增取值 `UNREGISTERED`：门禁照常做第 1～6 项并读画像，`hw_gate.json` 记 `status: NOT_RUN` 与 `node_hw_sha`，判定恒为 `ENV_INVALID`（退出码恒为 6），用于某型号第一个作业只采画像、供登记。§5.5 `exclusivity.json` 第 5 项改为「可见 GPU 数 == `--max-alloc-gpus` 且无计算进程」：请求 ≥1 张时先要列出 ≥1 张（阳性对照，否则 NOT_RUN），请求 0 张时要求 nvidia-smi 报告没有卡，阳性对照与计算进程检查记 NOT_RUN 并写明原因；第 1 项的 `scontrol` 输出没有 TRES 时记红。`hw_gate.json` 增加 `status`（PASS / RED / NOT_RUN）。`capenv.hwprofile` 新增必填 `--nvidia-smi PATH`。`capenv.sidecar` 新增可选 `--nvidia-smi PATH`：不给则不执行任何 nvidia-smi，`gpu.jsonl` 头 `positive_control_ok: false`。`gpu.jsonl` 的 `compute_pids` 可含 `null`（pid 读不到的计算进程，仍算计算进程）；`util_pct` 为 `null` 表示读不到，不是 0。nvidia-smi 查询只由 `capenv.gpu` 实现（`query_gpus`、`query_apps`、`positive_control`、`sample`、`check_visible`、`idle_check`，关键字参数 `nvsmi` 显式传入可执行文件），§11.1 速查可补一行。

## 测试

| 命令（均在 `<C>`，`bash -c 'ulimit -v 8388608; exec systemd-run --user --scope -q -p MemoryMax=6G timeout 600 env PATH=<C>/_local/fx-env/fakebin:$PATH PYTHONPYCACHEPREFIX=<C>/_local/fx-env/pyc <venv>/python -B -m pytest -q -p no:cacheprovider <dir>'`） | 退出码 | 结果 |
|---|---|---|
| `tests/envgate` | 0 | 59 passed |
| `tests/sidecar` | 0 | 5 passed |

新增测试：`tests/envgate/test_envgate_gpucount.py`（26 项）、`tests/sidecar/test_sidecar_v4.py::test_sidecar_gpu_uses_explicit_path_and_keeps_unreadable`。

修前是否会红：
- 修前代码副本（`git archive HEAD tools tests/...` 解到 `_local/fx-env/prefix/`，测试换成新版）：envgate 26 failed / 32 passed、sidecar 1 failed / 4 passed，退出码均 1。其中按行为（不是签名）失败的有：`test_item5_more_gpus_visible_than_requested_is_red`（修前 PASS）、`test_item5_zero_requested[*]`（修前 NOT_RUN）、`test_unregistered_collects_profile_and_exits_env_invalid`（修前退出 3）、`test_item1_without_tres_is_not_zero_gpus`（修前 PASS）。
- 变异自测（`_local/fx-env/mutate.py`，在 `_local/fx-env/mut/` 副本里逐个注入）：M1 util None 当 0、M2 丢弃读不到的 pid、M3 可见数 `>=`、M4 UNREGISTERED 判过、M5 忽略显式路径、M6 不认 "No devices were found"，对应测试全部变红（各 rc=1）。

## MEASURED / INFERRED / NOT RUN

- MEASURED：上表两个测试目录的退出码与计数；修前副本与 6 个变异的退出码；fakebin.log 为 0 字节；`pgrep -af "capenv|pytest|nvidia-smi|fx-env"` 无残留。
- INFERRED：真实 nvidia-smi 在看不见卡时输出 `No devices were found`（退出码按记忆为 6，代码只匹配文字、不依赖退出码）；集群上 `--gpus 0` 的容器里是否还有 nvidia-smi 可执行文件不确定——没有的话请求 0 张的门禁第 5 项为 NOT_RUN ⇒ ENV_INVALID（fail-closed）。画像 sha 随可见卡数变化（见上）。
- NOT RUN：任何集群/真实 nvidia-smi 验证；tests/jobs、tests/launch、tests/judge（不归我）；内层脚本到门禁的端到端（等 FX-jobs 接线）。
