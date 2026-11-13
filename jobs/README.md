# jobs/：作业脚手架（B6）

本目录是 capkit 的作业层：提交入口、批处理脚本、容器内脚本、tokenizer 拷贝作业草稿。
接口以 `../INTERFACES.md`（§5.5、§5.7、§5.9、§9.5、§11）为准；与它不一致的地方列在文末「偏差与未决」。

## 文件

| 文件 | 在哪里跑 | 作用 |
|---|---|---|
| `capjobs/submit.py` | agent VM（`python3.11`，只用标准库） | 唯一提交入口：`probe`、`submit`、`local-record`、`slurm-view`、`jobstate`、`record-end`、`exclude`、`prep-tokenizer`（只 dry-run） |
| `capjobs/remote.py` | VM | ssh 命令构造与唯一执行点；远端命令白名单 `sbatch squeue sacct scontrol show cdb find`（没有 `scancel`） |
| `capjobs/resources.py` | VM | 资源参数校验与 sbatch 参数拼装（新写，不继承 WT-21 的资源段） |
| `capjobs/snapshot.py` | VM / 容器 | 不可变快照目录与 `MANIFEST.json`；`verify()` 在作业开头重算 |
| `capjobs/cdbexclude.py` | VM | 从 `cdb find … -o csv` 输出算 `--exclude` |
| `capjobs/jobstate.py` | VM | 作业结束判据（RESOURCES §4） |
| `capjobs/slurmview.py` | VM | `<作业目录>/env/slurm_view.json`（INTERFACES §5.5.2） |
| `capjobs/launchplan.py` | VM / 容器 | 启动计划 `capkit.launch_plan/1` 的校验与占位符替换 |
| `capjobs/overlay.py` | 容器 | 覆盖清单拷进 site-packages，逐字节读回核对 → `env/overlay_verify.json` |
| `capjobs/envfiles.py` | 容器 | `env/image.json`、`mounts.json`、`local_disk.json`、`gpu_idle.json`、`import_origin.json`、`job.json` |
| `capjobs/launches.py` | 容器 | 逐个启动编排器、拷回、核对、`raw/` 只读、写第一行 `STATE` |
| `capjobs/copyback.py` | 容器 | 拷回与核对；收尾 `sweep` |
| `capkit_job.sbatch` | 计算节点宿主（批处理层，不用 Python） | 作业目录、镜像大小与 sha256、`srun` 进容器、兜底收尾步骤 |
| `capkit_inner.sh` | 容器内（与被测进程同一 srun 步骤） | 本地盘目录、覆盖、环境文件、门禁、启动、收尾删除 |
| `prep_tokenizer.sbatch.draft` | 计算节点宿主（草稿，**需用户批准**） | 只拷白名单里的非权重文件，写 `SHA256SUMS.txt` 与 `PROVENANCE.txt` |

## 调用方式

在 VM 上：`PYTHONPATH=<C>/tools python3.11 -m capjobs.submit <子命令> …`（`<C>` = `01_code/`）。
容器内：`capjobs` 随覆盖清单的 `capkit_packages` 拷进 site-packages，之后 `python -m capjobs.<模块>`；
只有覆盖那一步用 `PYTHONPATH=<快照>/tools`。

## 子命令（全部资源参数必需、无默认值）

- `probe`：`sbatch --test-only … --wrap=true`，经 `policy.check_test_only_argv`；命令与完整输出逐行追加进 `--probe-log`（JSONL，`start` / `result` 两行）。`--dry-run` 只打印。
- `submit`：顺序固定，任一不过即退出码 3，不写台账、不建快照、不调 ssh：
  1. 启动计划校验；`--experiment`、`--register-sha`、`--arms`、`--launches` 必须与计划逐字一致；
  2. `--basis` 必须是 `<W>/STATUS.md` 里完整的一行；
  3. `policy.check_no_model_submission`（白名单、GPU ≤ 1、挂载不碰模型库、tokenizer 副本无权重无软链、依据非空）；
  4. 资源参数校验；计划里各启动的 `budget_s + margin_s` 之和不超过 `--time`；
  5. 挂载覆盖：运行根、本地盘根、shm 根（要 rw）、快照根与 tokenizer 目录（ro 即可）及其 realpath 都在某个「源 = 目的」的挂载之下；
  6. 登记表 sha 在 `<W>/REGISTRY.tsv` 里且 `kind=register`；
  7. 镜像文件存在且大小等于 `--image-size`（sha256 在作业里算）；
  8. 覆盖清单 `non_python_changed == []`、`capkit_code_sha` 等于代码仓 HEAD、每个源文件 sha 与清单一致；
  9. 快照路径下（`jobs/`、`tools/`、`--include`）没有未提交的改动（`--dry-run` 只报告 `would_refuse_dirty`）。
  之后：建快照（只读）→ 写 `PLANNED` 行 → ssh `sbatch --parsable` → 写 `SUBMITTED` 行（作业号、`submitted_at`）；失败写 `ENDED`/`SUBMIT_FAILED`，退出码 4。
- `local-record`：本机运行（作业号 `local-*`）的 `PLANNED` + `SUBMITTED` 两行，同样过五项断言、依据行与登记表检查。
- `exclude`：`--cdb-file`（离线，如 `00_plan/cluster-probe/20261113/02d_cdb_b200.csv`）或 `--live`（经 ssh `cdb find`，原文先落 `--raw-out`）；`--main-profile-spec`（`capkit.cdb_main_profile/1`，显式列出画像字段与取值）；输出 `capkit.cdb_exclude/1`，`submit --exclude-file` 读它（分区必须与 `--partition` 相同）。给定分区里一台节点都没有 → 报错（NOT_RUN），没有一台符合主画像 → 报错（无阳性对照）。
- `jobstate`：退出码 0 = ENDED（`squeue -h -j J` 退出码 0 且输出为空，**并且** `sacct` 给出终态）、20 = 未结束、21 = 不知道（任一命令出错）。
- `record-end`：写 `ENDED` 行；集群作业必须先过 `jobstate` 的 ENDED 判据，`--status` 只能是 VALID / VOID / NOT_RUN / ENV_INVALID / CANCELLED。
- `slurm-view`：`scontrol show job`、`scontrol show node <NodeList>`、`squeue -h -w <node> -o '%i %u %T'` 原文与 sha 写进 `<作业目录>/env/slurm_view.json`；作业目录必须已存在（由作业建）。
- `prep-tokenizer`：草稿，只接受 `--dry-run` 并打印命令；不带 `--dry-run` 退出码 3。

## 启动计划 `capkit.launch_plan/1`

```json
{"schema":"capkit.launch_plan/1","experiment":"E0","register_sha":"<64hex>","job_arm":"job","margin_s":120,
 "launches":[{"launch_seq":0,"arm":"nullsse-a","budget_s":900,
              "argv":["{PY}","-m","caplaunch","--scenario","{SNAP}/configs/e0/x.json","…"]}]}
```

占位符只有 `{PY} {SNAP} {JOB_DIR} {SHM_ROOT} {LOCAL_ROOT} {JOB_ID} {RESTART_COUNT}`。身份参数（`--experiment --arm --launch-seq --job-id --restart-count --register-sha --role launch --run-dir`）由 `capjobs.launches` 追加，计划里出现即拒绝——身份只有一个来源。

## 作业内的顺序

批处理层（`capkit_job.sbatch`）：建 `runs/J<作业号>-r${SLURM_RESTART_COUNT:-0}/`（已存在即退出）→ `stat` + `sha256sum` 核对镜像（不符：作业级 `STATE` 写 `ENV_INVALID`，退出码 6，不起容器）→ `srun --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none --mpi=<参数> --container-image … --container-mounts=<参数> --no-container-mount-home --container-remap-root --container-writable` 跑 `capkit_inner.sh … run` → 没有 `finalize.done` 时再起一个 `finalize-only` 步骤兜底。

容器内（`capkit_inner.sh run`）：建 `<本地盘>/<用户>-<作业号>` 与 `<shm>/<用户>-<作业号>`（已存在即 ENV_INVALID）→ 覆盖并逐字节核对 → `unset PYTHONPATH`，核对 capkit 包都从 site-packages 导入 → `job.json`（快照重算核对）、`image.json`、`mounts.json`（模型库不存在或为空、各挂载非空可读）、`local_disk.json`、`gpu_idle.json`（阳性对照：查询列出至少一张 GPU）→ `--mpi=none` 时清掉 `PMIX_*`、`PMI_*`、`SLURM_*` → 门禁 `python -m <--gate-module> --run-dir <作业目录> --phase pre …`（身份参数 `--role gate`）→ `capjobs.launches`。任一步不过：作业级 `STATE` 写 `ENV_INVALID`，退出码 6。

收尾（`EXIT` trap，两种模式都走）：`capjobs.copyback sweep` 把本地盘与 shm 下的全部内容拷回并核对（文件数、总字节、逐项大小）→ 只有核对通过（或本地根下一个普通文件都没有，记 `EMPTY`）时，对每个根先断言路径逐字等于 `<base>/<用户>-<作业号>`，再过 `python -m capcommon.scratchpath`，最后 `rm -rf -- "${p:?}"` → 写 `finalize.done`。

每个启动树的第一行 `STATE`（`capjobs.launches`，拷回核对之后）：拷回未核对 → VOID；`control/stop.json` 的 `reason` 为 `not_ready_before_origin` → NOT_RUN；编排器退出码 0 → VALID；6 → ENV_INVALID（其后的启动全部跳过、记 NOT_RUN）；超时或其他退出码 → VOID。剩余时间不足 `budget_s + margin_s` 的启动不跑、记 NOT_RUN。

## 测试

`tests/jobs/`（只做功能正确性，本机不取性能数）：

```bash
cd <C> && systemd-run --user --scope -q -p MemoryMax=6G timeout 600 \
  /home/scratch.yihlu_coreai/tmp/20261106-wt03-venv/venv/bin/python -m pytest -q -p no:cacheprovider tests/jobs
```

`conftest.py` 让 `subprocess` 拒绝执行 `ssh sbatch srun squeue sacct scontrol scancel salloc cdb`；批处理脚本的测试在 `PATH` 前面放一个假的 `srun` 记录参数。

## 偏差与未决（报告里同样列出）

1. 包的位置：第一批写在 `jobs/capjobs/`；集成时按 INTERFACES §1.1 移到 `tools/capjobs/`（代码只依赖 `capcommon` 与包内相对导入，没有改代码）。测试仍在 `tests/jobs/`。`snapshot.plan_files` 把整个 `tools/` 与 `jobs/` 放进快照；`capkit_inner.sh` 的 `BOOT_PP` 是 `$SNAP/tools`。
2. 白名单：任务文本列了 `E10-proxy`、`E13-main`；INTERFACES §9.5 与 `capcommon.policy.NO_MODEL_EXPERIMENTS` 不含它们（§12 第 11 条），本组件按 INTERFACES，测试断言二者被拒。
3. 作业级文件的身份：`env/*.json`、`job.json`、`launches.json` 不属于任何一次启动，`identity` 用启动计划的 `job_arm` 与 `launch_seq 0`。编排器把作业级 `env/` 拷进启动树后，判定器「树内全部文件身份前六个字段一致」会因 `arm` 不同而判 INPUT_INVALID。门禁（B4）有同样的问题。需要 spec 负责人定规则（例如作业级文件豁免，或编排器拷贝时只核内容不核身份）。
4. `ts` 探针文件在启动树里的位置 INTERFACES §6.5 没有给出；本组件放 `raw/probe_ts/<pid>.ts.capbin`。
5. 作业级 `STATE`（`<作业目录>/STATE`）与 `import_origin.json`、`launches.json`、`batch.tsv`、`copyback/`、`local_leftovers/`、`shm_leftovers/` 是 INTERFACES 之外新增的作业目录内容。
6. `srun` 显式带 `--cpus-per-task=$SLURM_CPUS_PER_TASK`：较新的 Slurm 里 srun 不再继承 sbatch 的 `--cpus-per-task`【推断，未在本集群核对】，缺了它步骤可能只拿到 1 个核。
7. `--export=ALL`（与 WT-21 一致）：前端的环境变量会带进作业；容器内 `unset PYTHONPATH`，其余未清理。
8. `prep-tokenizer` 走的资源参数与 `submit` 相同，因此也带 `--exclusive`（整机独占，只为拷几个小文件）；批准时再定。
