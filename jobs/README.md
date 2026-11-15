# jobs/：作业脚手架（B6）

本目录是 capkit 的作业层：提交入口、批处理脚本、容器内脚本、tokenizer 拷贝作业。
接口以 `../INTERFACES.md`（§5.5、§5.7、§5.9、§9.5、§11）为准；与它不一致的地方列在文末「偏差与未决」。

## 文件

| 文件 | 在哪里跑 | 作用 |
|---|---|---|
| `tools/capjobs/submit.py` | agent VM（`python3.11`，只用标准库） | 唯一提交入口：`probe`、`submit`、`reconcile-submit`、`local-record`、`slurm-view`、`jobstate`、`record-end`、`exclude`、`prep-tokenizer` |
| `tools/capjobs/remote.py` | VM | ssh 命令构造与唯一执行点；远端命令白名单 `sbatch squeue sacct scontrol show cdb find`（没有 `scancel`） |
| `tools/capjobs/resources.py` | VM | 资源参数校验与 sbatch 参数拼装（新写，不继承 WT-21 的资源段） |
| `tools/capjobs/snapshot.py` | VM / 容器 | 不可变快照目录与 `MANIFEST.json`；`verify()` 在作业开头重算 |
| `tools/capjobs/cdbexclude.py` | VM | 从 `cdb find … -o csv` 输出算 `--exclude` |
| `tools/capjobs/jobstate.py` | VM | 作业结束判据（RESOURCES §4） |
| `tools/capjobs/slurmview.py` | VM | `<作业目录>/env/slurm_view.json`（INTERFACES §5.5.2） |
| `tools/capjobs/launchplan.py` | VM / 容器 | 启动计划 `capkit.launch_plan/1` 的校验与占位符替换 |
| `tools/capjobs/overlay.py` | 容器 | 覆盖清单拷进 site-packages，逐字节读回核对 → `env/overlay_verify.json` |
| `tools/capjobs/envfiles.py` | 容器 | `env/image.json`、`mounts.json`、`local_disk.json`、`gpu_idle.json`、`import_origin.json`、`job.json` |
| `tools/capjobs/launches.py` | 容器 | 逐个启动：在本节点生成绑核表、起编排器、拷回、核对、`raw/` 只读、在编排器那一行之后追加 `STATE` |
| `tools/capjobs/copyback.py` | 容器 | 拷回与核对；收尾 `sweep` |
| `capkit_job.sbatch` | 计算节点宿主（批处理层，不用 Python） | 作业目录、镜像大小与 mtime（sha256 由提交端算）、`srun` 进容器、兜底收尾步骤 |
| `capkit_inner.sh` | 容器内（与被测进程同一 srun 步骤） | 本地盘目录、覆盖、环境文件、硬件画像、门禁、启动、收尾删除 |
| `prep_tokenizer.sbatch` | 宿主（批处理层）+ 容器（拷贝），只用 bash 与 coreutils | tokenizer 拷贝作业（见下节）；不进测量作业的快照 |

## 调用方式

在 VM 上：`PYTHONPATH=<C>/tools python3.11 -m capjobs.submit <子命令> …`（`<C>` = `01_code/`）。
容器内：`capjobs` 随覆盖清单的 `capkit_packages` 拷进 site-packages，之后 `python -m capjobs.<模块>`；
只有覆盖那一步用 `PYTHONPATH=<快照>/tools`。

## 子命令（全部资源参数必需、无默认值）

- `probe`：`sbatch --test-only … --wrap=true`，经 `policy.check_test_only_argv`；命令与完整输出逐行追加进 `--probe-log`（JSONL，`start` / `result` 两行）。`--dry-run` 只打印。
- `submit`：顺序固定，任一不过即退出码 3，不写台账、不建快照、不调 ssh：
  1. 启动计划校验；`--experiment`、`--register-sha`、`--arms`、`--launches` 必须与计划逐字一致；
  2. `--basis` 必须是 `<W>/STATUS.md` 里完整的一行；
  3. `policy.check_no_model_submission`（白名单、GPU ≤ 1、挂载的字面路径与 realpath 都不碰模型库、tokenizer 副本无权重无软链、依据非空）；
  4. 资源参数校验；
  5. 挂载（`job_mounts`，与 `prep-tokenizer` 共用）：必须有次级盘 `/home/scratch.yihlu_coreai_1` 的「源 = 目的」挂载（`REQUIRED_IDENTITY_MOUNTS`；主盘上的目录可能是指向它的软链）；运行根、本地盘根、shm 根（要 rw）、快照根、tokenizer 目录与 `--workdir`（ro 即可；作业要读 `<W>` 里的登记表与场景）及其 realpath 都在某个「源 = 目的」的挂载之下；
  6. 登记表 sha 在 `<W>/REGISTRY.tsv` 里且 `kind=register`；按该行的 `relpath` 读 `register.json`，现算 sha 必须等于 `--register-sha`，取它的 `environment` 段（见下）；`--gpus` 必须等于 `environment.gpus`（为 0 时打印一行警告：判定器的 GPU-IDLE 会 NOT_RUN）；
  7. 时间预算：各启动 `budget_s + kill_grace_s + margin_s`、绑核表生成上限 `launchplan.PINS_TIMEOUT_S`（120 s）与作业输入生成上限（`launchplan.JOB_INPUT_TIMEOUT_S` 600 s × 该启动场景声明的种类数，预算项 `job_inputs_s`；INTERFACES 勘误 B2B-9）之和，加门禁空转窗口 `idle_window_s`、`slurm_view.json` 最长等待 `slurm_view_wait_s`，不超过 `--time`；
  8. 镜像：大小等于 `--image-size`，在提交端算一次 sha256 并等于 `--image-sha256`；大小、`mtime_ns` 与 sha 写进 `job.env`，作业里只核大小与 `mtime_ns`（不在分配里重算整份镜像）；
  9. `--registered-hw-sha` 是 64 位十六进制或 `UNREGISTERED`（只采画像的作业，见下）；
  10. 覆盖清单 `non_python_changed == []`、`capkit_code_sha` 等于代码仓 HEAD、每个源文件 sha 与清单一致；每个启动的 `--capkit-sha` 等于代码仓 HEAD；每个启动的 `--workdir` 就是 `--workdir`、`--scenario` 是登记表（该启动的 arm）`files.scenario` 记的那个 `<W>` 相对路径且 sha 相同（下节）；登记表文件与各启动 arm 的每个 `files.*` 及其 realpath 也在挂载之下（`submit.workdir_inputs`；软链指到没挂载的目录同样拒绝）；每个启动的 argv 按作业里实际运行的样子（`launches.launch_argv`：占位符代入示例值，追加 `capjobs.launches` 会加的参数）用 caplaunch 自己的解析器（`caplaunch.__main__.build_parser`）只解析、不执行（`submit.check_plan_argv`），缺必填参数、缩写、未知参数、值类型不对即拒绝；
  11. 快照路径下（`jobs/`、`tools/`、`--include`）没有未提交的改动（`--dry-run` 只报告 `would_refuse_dirty`）。
  之后：建快照（只读）→ 写 `PLANNED` 行（`reason` 列是用途标记 `purpose=measurement` 或 `purpose=hw_profile_capture`）→ ssh `sbatch --parsable` → 写 `SUBMITTED` 行（作业号、`submitted_at`）。sbatch 明确报错（退出码不是 0、255，也不是 ssh 起不来或超时）写 `ENDED`/`SUBMIT_FAILED`；**结果不明**（ssh 超时或失败、sbatch 退出 0 却没有作业号）写 `NOTE` 行，`status` 为 `UNKNOWN_SUBMIT`（台账正式状态，只能出现在带 `submitted_at` 的 `NOTE` 行上；之后的 `SUBMITTED` / `ENDED` 行取代它）、`reason` 以用途标记开头、`submitted_at` 是这次 sbatch 的时刻：作业可能已经排队，计划保持打开，`record-end` 拒绝它，脚本不重投。两种情况退出码都是 4。
- `reconcile-submit`：只处理 `UNKNOWN_SUBMIT` 的计划，由人在查过 squeue / sacct 之后调用，凭证据落状态：`--job-id J` 要求 `scontrol show job J` 的 `Command=` 恰为本计划快照里的 `jobs/capkit_job.sbatch`，或运行根下 `J<J>-r*/batch.tsv` 写着本计划号 → `SUBMITTED`（`submitted_at` 沿用 sbatch 的时刻）；`--not-submitted` 要求 `squeue -h --me -o '%i %o'` 退出 0 且没有一行含本计划快照目录、运行根下也没有写着本计划号的 `batch.tsv` → `ENDED`/`SUBMIT_FAILED`。证据不够就退出码 3，不写台账。
- `local-record`：本机运行（作业号 `local-*`）的 `PLANNED` + `SUBMITTED` 两行，同样过五项断言、依据行与登记表检查。
- `exclude`：`--cdb-file`（离线，如 `00_plan/cluster-probe/20261113/02d_cdb_b200.csv`）或 `--live`（经 ssh `cdb find`，原文先落 `--raw-out`）；`--main-profile-spec`（`capkit.cdb_main_profile/1`，显式列出画像字段与取值）；输出 `capkit.cdb_exclude/1`，`submit --exclude-file` 读它（分区必须与 `--partition` 相同）。给定分区里一台节点都没有 → 报错（NOT_RUN），没有一台符合主画像 → 报错（无阳性对照）。
- `jobstate`：退出码 0 = ENDED（squeue 不再列出该作业，**并且** `sacct` 给出终态）、20 = 未结束、21 = 不知道（任一命令出错）。「不再列出」= `squeue -h -j J` 退出 0 且输出为空，或退出非 0 且 stderr 是 `Invalid job id specified`（作业已被 slurmctld 清出内存，此时只看 sacct）；其他 squeue 错误一律 21。
- `record-end`：写 `ENDED` 行（`reason` 前面带上计划的用途标记）；集群作业必须先过 `jobstate` 的 ENDED 判据，`--status` 只能是 VALID / VOID / NOT_RUN / ENV_INVALID / CANCELLED；`UNKNOWN_SUBMIT` 的计划先走 `reconcile-submit`。
- `slurm-view`：`scontrol show job`、`scontrol show node <NodeList>`、`squeue -h -w <node> -o '%i %u %T'` 原文与 sha 写进 `<作业目录>/env/slurm_view.json`；作业目录必须已存在（由作业建）。
- `prep-tokenizer`：tokenizer 拷贝作业的正式入口（用户 2026-11-13 批准，见下节「tokenizer 拷贝作业」）。

## tokenizer 拷贝作业（`prep-tokenizer`，`purpose=prep_tokenizer`）

用户 2026-11-13 批准（`00_plan/DECISIONS.md`「勘误与补充」末条第 4 项）。这是**唯一**允许挂模型库的作业类型，登记在 `capcommon.policy`（`PREP_TOKENIZER_*`）：

- 模型目录只认 `policy.PREP_TOKENIZER_MODEL_DIRS`（`/home/scratch.trt_llm_data_ci/llm-models/GLM-5.3-Flash`），以同一路径**只读**挂进容器；目的地父目录 `<代码根>/configs/tokenizers` 与计划目录不是自动加的挂载，要在 `--mount` 给出的 rw「源 = 目的」挂载之下（见下面「挂载」一条）。`policy.check_prep_tokenizer_mounts`：恰好一个挂载碰到模型库，且它就是登记的目录、同路径、`ro`；其余挂载按普通作业的 `check_mounts` 查。`ledger.submissions_append` 只对 `reason = purpose=prep_tokenizer`、`experiment = PREP-TOKENIZER`、`gpus = 0`、断言键为 `PREP_ASSERTION_KEYS` 的 `PLANNED` 行放行这类挂载；其他任何计划挂模型库仍拒绝。
- 命令：`capjobs.submit prep-tokenizer --account A --qos Q [--partition P] --exclude none|… --time T --mem M --cpus-per-task N --job-name J --login-host H --ssh-timeout-s S --workdir <W> --code-root <C> --snapshot-root R --model-dir D --name NAME --image <sqsh> --mount SRC:DST[:ro|rw] … --basis <STATUS.md 整行> [--dry-run]`。不申请 GPU（不带 `--gres`/`--gpus`）、不带 `--exclusive`；`--partition` 不给就不加（全局规则「分区能不加就不加」；本集群 `sacctmgr` 的关联不限分区，见 `00_plan/RESOURCES.md` 9.1），给了原样传。
- 挂载：容器挂载 = 登记的模型目录（同路径 `ro`，自动加在最前）+ 各 `--mount`。`--mount` 与 `submit` 走同一套检查（`parse_mount`、`job_mounts`：次级盘必须挂，目的地父目录 `<C>/configs/tokenizers` 与计划目录及其 realpath 要在某个 rw 的「源 = 目的」挂载之下），整份清单再过 `policy.check_prep_tokenizer_mounts`（`--mount` 碰到模型库即拒绝）。
- 提交前（任一不过退出码 3，不写任何文件）：模型目录在登记表里、`--name` 合法、`<C>/configs/tokenizers/<NAME>`（及 `.partial`）不存在、依据是 `STATUS.md` 的整行、镜像是文件、资源参数合法、挂载断言与覆盖、`jobs/prep_tokenizer.sbatch` 在代码仓 HEAD 已提交且工作区与之逐字节相同（`--dry-run` 只报告 `would_refuse_uncommitted`）。之后建计划目录 `<R>/<计划号>/`（HEAD 里的 `jobs/prep_tokenizer.sbatch`、`job.env`、`mounts.json`、`MANIFEST.json`，都只读；`MANIFEST.json` 记代码 HEAD 与前三个文件的 sha256、字节数）→ `PLANNED` 行（`snapshot_sha256` = `MANIFEST.json` 的 sha256）→ ssh `sbatch --parsable` → `SUBMITTED` 行；结果不明与明确失败的处理同 `submit`（`reconcile-submit` 也认这类计划）。镜像只核大小与 `mtime_ns`（拷贝结果由逐文件 sha256 核对，不依赖镜像内容）。
- 作业（`prep_tokenizer.sbatch`，宿主层 → 一个 `srun --container-mounts=… --no-container-mount-home` 步骤，全程只有 bash 与 coreutils）：作业目录 `<计划目录>/J<作业号>-r<重启次数>/`（已存在即退出 3）→ 计划目录里脚本的 sha256 等于 `job.env` 的 `CAPJOBS_PREP_SCRIPT_SHA256`（否则退出 3）→ 镜像大小与 `mtime_ns` → 容器内：模型目录在 `/proc/self/mountinfo` 里是只读挂载、模型库（`job.env` 里的库根：提交端 `policy.library_roots()`，每个库的字面路径与 realpath）除了通往它的路径外什么都看不见（否则退出 6）→ 只看模型目录顶层：权重后缀（`policy.WEIGHT_SUFFIXES`，不分大小写）一律拒绝并记下，其余名字匹配白名单（`tokenizer* *.json *.jinja *.txt *.model *.tiktoken`）才考虑；以下拒绝并记下：与作业自己的输出同名（`COPY_MANIFEST.tsv`、`PROVENANCE.txt`）、名字带反斜杠、软链的解析目标不是模型目录内的普通文件或目标名带权重后缀、内容开头是权重格式（pickle `\x80`、zip `PK\x03\x04`、GGUF `GGUF`、safetensors：前 8 字节小端长度 N ≤ 文件长 − 8 且后跟 `{"`）；白名单文件超过 `PREP_TOKENIZER_MAX_FILE_BYTES`（64 MiB）作业失败（退出 4）→ 每个文件源与副本 sha256 相等 → 至少拷到一个词表文件 → `COPY_MANIFEST.tsv`（文件名、字节数、sha256，另存一份在作业目录）与 `PROVENANCE.txt`（代码 HEAD、脚本 sha256，拷了、跳过、拒绝的清单）→ 结束前核对副本：只有清单里的普通文件、名字与内容都不是权重、没有超限文件、`sha256sum -c --strict` 退出 0 且清单里每个文件各有一行 OK → 只读、改名为最终目录 → 再核一遍。内容特征只认这四种格式的文件头，不认得的二进制仍只靠名字与 64 MiB 上限。
- 拷完的目录 `configs/tokenizers/<NAME>/` 不进 git（`.gitignore`），测量作业用 `--tokenizer-dir` 挂它，`policy.tokenizer_dir_sha256` 再核一次没有权重、没有软链。

## 登记表的 `environment` 段（`register.json` 顶层，必填）

```json
"environment":{"gpus":1,"idle_window_s":30,"idle_busy_max_frac":0.02,"slurm_view_wait_s":300,"chassis_source":"slurm_view"}
```

恰好这五个键：`gpus`（向 Slurm 申请的 GPU 数，0 或 1；不加载模型的作业取 1，使 GPU-IDLE 有阳性对照）、`idle_window_s`（门禁第 6 项的空转窗口，(0, 3600]）、`idle_busy_max_frac`（第 6 项每核忙碌比例上限，(0, 1]）、`slurm_view_wait_s`（等 `slurm_view.json` 的最长秒数，(0, 3600]）、`chassis_source`（`slurm_view` 或 `dmi`，传给 `capenv.hwprofile`）。`submit` 把它们写进 `job.env`（`CAPJOBS_MAX_ALLOC_GPUS`、`CAPJOBS_IDLE_WINDOW_S`、`CAPJOBS_IDLE_BUSY_MAX_FRAC`、`CAPJOBS_SLURM_VIEW_WAIT_S`、`CAPJOBS_CHASSIS_SOURCE`），内层脚本显式传给各检查；没有缺省值，缺段或多键即拒绝提交。

## 启动计划 `capkit.launch_plan/1`

```json
{"schema":"capkit.launch_plan/1","experiment":"E0","register_sha":"<64hex>","job_arm":"job","margin_s":120,
 "kill_grace_s":120,
 "launches":[{"launch_seq":0,"arm":"nullsse-a","budget_s":900,
              "argv":["{PY}","-m","caplaunch","--workdir","<W>","--scenario",
                      "02_e0/E0/register/x.scenario.json","--capkit-sha","<代码 HEAD>","…"]}]}
```

`kill_grace_s`：超出 `budget_s` 的启动收到 SIGTERM 后，等多久再对整棵进程树 SIGKILL。`submit` 与作业内的 `capjobs.launches` 都核对它不小于每个启动的编排器最坏收尾时长 = 场景 `stop_order` 角色数 × (`stop_grace_s` + `--kill-after-s`) + 2 × `--kill-after-s`（`launchplan.check_kill_grace`），所以每个启动的 argv 必须带 `--workdir`、`--scenario` 与 `--kill-after-s`，否则拒绝。`--scenario` 是 `<W>` 里已登记的那份场景（caplaunch 与判定器都按它在 `<W>` 下的相对路径与登记表 `files.scenario` 核对，登记又必须早于提交，所以不能是带计划号的 `{SNAP}/…`）：写相对 `--workdir` 的路径或 `<W>` 内的绝对路径，两者都不许含占位符。占位符只有 `{PY} {SNAP} {JOB_DIR} {SHM_ROOT} {LOCAL_ROOT} {JOB_ID} {RESTART_COUNT}`。身份参数（`--experiment --arm --launch-seq --job-id --restart-count --register-sha --role launch --run-dir`）与 `--overlay-manifest`、`--nvidia-smi`、`--job-env-dir <作业目录>/env`、`--pin-table` 由 `capjobs.launches` 追加，计划里出现即拒绝——每个只有一个来源。绑核表由 `capjobs.launches` 在每个启动之前、在本节点上用 `capenv.pins generate --only-affinity` 生成（需求 = 该启动场景各角色的 `pin_role`、`cpus_needed`、`numa`；身份 = 该启动，角色 `pins`；`--source-sha` = 本作业 `env/hw_profile.json` 的 `profile_sha256`），写到 `<作业目录>/pins/<启动树名>.pin_table.json`；生成失败（读不到画像、需求放不下）该启动记 NOT_RUN，不起编排器。

覆盖清单的 `stub_modules`（`capprobe.overlay --stub-module sfe=<路径>`，INTERFACES 勘误 B2B-11）随快照进作业、不装进 site-packages：作业的覆盖步骤只核对 `<快照>/<src>` 的字节，S-PB 前端模式在计划里写 `--param sfe_module={SNAP}/<src>`。

P0 覆盖清单：`capprobe.overlay --tier P0` 要求 `--export-dir` 在代码根之内（清单里的 `src` 是代码根相对路径），用代码根下被 git 忽略的固定目录（如 `<C>/_local/e0-P0-export/`）；镜像提交到基线之间改了不进 wheel 的 `tensorrt_llm/grpc/openengine/README.md`，P0 清单要加 `--allow-unshipped tensorrt_llm/grpc/openengine/README.md`。

## 作业内的顺序

批处理层（`capkit_job.sbatch`）：建 `runs/J<作业号>-r${SLURM_RESTART_COUNT:-0}/`（已存在即退出）→ `stat` 与 `date -r` 核对镜像大小与 `mtime_ns` 仍是提交时记下的值（不符：作业级 `STATE` 写 `ENV_INVALID`，退出码 6，不起容器）→ `srun --cpus-per-task=$SLURM_CPUS_PER_TASK --cpu-bind=none --mpi=<参数> --container-image … --container-mounts=<参数> --no-container-mount-home --container-remap-root --container-writable` 跑 `capkit_inner.sh … run` → 没有 `finalize.done` 时再起一个 `finalize-only` 步骤兜底。

容器内（`capkit_inner.sh run`）：建 `<本地盘>/<用户>-<作业号>-r<重启次数>` 与 `<shm>/<用户>-<作业号>-r<重启次数>`（已存在即 ENV_INVALID；重排队到同一节点不会撞上上一轮留下的目录）→ 覆盖并逐字节核对 → `unset PYTHONPATH`，核对 capkit 包都从 site-packages 导入 → `job.json`（快照重算核对）、`image.json`、`mounts.json`（模型库不存在或为空；各挂载与快照、作业目录、`<W>`（`job.env` 的 `CAPJOBS_WORKDIR`，缺了即 ENV_INVALID）非空可读，且路径与 realpath 都不碰模型库）、`local_disk.json`、`gpu_idle.json`（`capenv.gpu.check_visible`：可见 GPU 数等于登记的 `gpus`，申请 ≥1 张时先要列出 ≥1 张，无计算进程）→ `--mpi=none` 时清掉 `PMIX_*`、`PMI_*`、`SLURM_*` → `chassis_source` 为 `slurm_view` 时等 `env/slurm_view.json` 出现（最多 `slurm_view_wait_s` 秒）→ `capenv.hwprofile` 写 `env/hw_profile.json`、`env/hw_raw.json`（`--code-baseline` 取覆盖清单的 `base_commit`，即 TRT-LLM 基线；读不到即 ENV_INVALID）→ 门禁 `python -m <--gate-module> --run-dir <作业目录> --phase pre --hw-profile … --max-alloc-gpus … --idle-window-s … --idle-busy-max-frac … --nvidia-smi …`（身份参数 `--role gate`）→ `capjobs.launches`（`--hw-profile <作业目录>/env/hw_profile.json`；每个启动先生成绑核表，见上节；再生成场景 `job_inputs` 声明的作业输入，见下）。任一步不过：作业级 `STATE` 写 `ENV_INVALID`，退出码 6。`--registered-hw-sha UNREGISTERED` 的作业照常写画像，门禁按约定以 ENV_INVALID 结束，`STATE` 原因写明「profile-only job」；它的台账行带 `purpose=hw_profile_capture`，不算作环境失败。

收尾（`EXIT` trap，两种模式都走）：只处理本轮拥有的节点本地目录——`run` 模式是本轮 `mkdir` 成功的那几个，`finalize-only` 模式是存在、不是软链、属主是本用户的那几个；不属于本轮的目录不拷、不删。`capjobs.copyback sweep` 把它们下面的全部内容拷回并核对（文件数、总字节、逐个文件的大小与 sha256；启动树顶层的 `STATE` 只许追加，副本必须以原件的字节开头）→ 只有核对通过（或一个普通文件都没有，记 `EMPTY`）时，对每个根先断言路径逐字等于 `<base>/<用户>-<作业号>-r<重启次数>`，再过 `python -m capcommon.scratchpath`（`--job-id` 传 `<作业号>-r<重启次数>`），最后 `rm -rf -- "${p:?}"` → 写 `finalize.done`。

每个启动在 `capjobs.launches` 里：剩余时间不足 `budget_s + kill_grace_s + margin_s + launchplan.in_job_bound_s`（绑核表 120 s + 每个作业输入种类 600 s）就不跑、记 NOT_RUN；绑核表之后生成场景 `job_inputs` 声明的作业输入（`capjobs.jobinputs`：`<{PY}> -m <生成模块> … --out <作业目录>/job_inputs/<启动树名>/<种类>.json`，旁边写生成记录与日志；INTERFACES 勘误 B2B-9），场景的 `job_inputs` 段或它要读的场景键不合法（`capcommon.jobinputs.check_scenario`），或任一种类失败（生成器失败，或在它运行之前就失败：缺 `--param`、输出已存在、登记输入读不到）⇒ 该启动记 NOT_RUN（`job_input_failed`，`launches.json` 里保留此前已成功种类的记录），成功则给编排器追加 `--job-input <种类>=<路径>`；绑核表与作业输入的生成器都经 `launches.run_helper` 运行（独立会话、每 0.5 s 轮询）：作业在这期间收到 SIGTERM ⇒ 杀掉生成器的整个进程组，该启动记 NOT_RUN（原因 `skipped: the job received SIGTERM during …`，不记 `pin_table_failed` / `job_input_failed`），其余启动跳过，`launches.json` 照写；信号先到生成器、生成器先退出也一样：生成器退出之后先看作业是否已收到 SIGTERM（以 −SIGTERM 退出的多等一个轮询周期再看），已收到就不论退出码按信号停止记（INTERFACES 勘误 B2B-15）；运行时每 0.5 s 记下编排器的全部后代（`launches` 自己是 child subreaper，编排器的角色即使在自己的会话里、父进程退出，也会挂到它下面）；编排器退出（或超时被杀）后，仍活着的后代先 SIGTERM，`kill_grace_s` 后 SIGKILL，逐个记进 `launches.json` 的 `leftovers`，然后才拷回。启动树里 `capjobs.launches` 写的那行 `STATE`（拷回核对之后）：拷回未核对 → VOID；有进程扛过 SIGKILL → VOID；超时被杀 → VOID；作业自己收到 SIGTERM 而停下当前启动（`launches.json` 的 `terminated`；编排器以 0、5 以外的退出码退出、`launches` 发现它退出时作业已收到 SIGTERM（例如编排器被同一个信号先杀死），也算）→ NOT_RUN（原因以 `terminated` 开头，与编排器被信号停下时记的 NOT_RUN 一致，不计入 VOID）；这几条都只会让编排器自己记的结论变差、不会变好（`capcommon.runtree.worst_state`，VALID < NOT_RUN < VOID < ENV_INVALID：例如编排器已记 ENV_INVALID 就保持 ENV_INVALID）；否则以编排器自己在树里写的最后一行 `STATE` 为准（例如清空门超时是 VALID、退出码 5），只有它写 VALID 而退出码不是 0 或 5 时改判 VOID；编排器没写 `STATE` 时才按退出码：`control/stop.json` 的 `reason` 为编排器的提前停止原因之一（`not_ready_before_origin`、`overlay_manifest_missing`、`job_env_dir_missing`、`orchestrator_signalled`）且退出码为 5 → NOT_RUN；编排器以非 0 退出且树里没有 `raw/launch.json`（它在起任何角色之前就退出了，不论退出码：配置错 3、argparse 用法错 2、未捕获的异常 1 等）→ NOT_RUN，原因带退出码与编排器 stderr 的最后一行，并在 `launches.json` 里记 `config_refused`；0 → VALID，6 → ENV_INVALID，其他 → VOID。ENV_INVALID 或有进程扛过 SIGKILL（记 VOID）之后，其余启动全部跳过、记 NOT_RUN。

`capjobs.launches` 的退出码（内层脚本与批处理脚本照传，即作业的退出码）：有启动拷回失败 → 4；否则有 ENV_INVALID → 6；否则有 `config_refused`、`pin_table_failed`（绑核表生成失败而跳过；场景读不到也记在这里，原因以 `skipped: scenario:` 开头）或 `job_input_failed`（作业输入生成失败而跳过）的启动 → 3（配置错：启动记 NOT_RUN，但作业不算成功）；否则 0。停止规则（CONSTRAINTS「同一个原因连续两次使作业 `ENV_INVALID` 或使运行 VOID，就停下上报」）把起进程前的配置错（作业退出 3、`launches.json` 的 `config_refused`、`pin_table_failed` 或 `job_input_failed` > 0，原因取该启动的 `reason`）按同一规则计数：看护见到连续两个作业因同一配置错退出 3，就停下上报，不再重投（INTERFACES 勘误 R5-3）。

## 测试

`tests/jobs/`（只做功能正确性，本机不取性能数）：

```bash
cd <C> && bash -c 'ulimit -v 8388608; exec systemd-run --user --scope -q -p MemoryMax=6G timeout 600 \
  /home/scratch.yihlu_coreai/tmp/20261106-wt03-venv/venv/bin/python -B -m pytest -q -p no:cacheprovider tests/jobs'
```

`conftest.py` 让 `subprocess` 拒绝执行 `ssh sbatch srun squeue sacct scontrol scancel salloc cdb`；批处理脚本的测试在 `PATH` 前面放一个假的 `srun` 记录参数；内层脚本 `run` 模式的测试用一个假的容器 Python 记录各步命令行，再用真实的 `capenv.gate` 解析器与 `capenv.hwprofile` 核对这些命令行。

依赖本仓库之外的两份输入，缺了就**失败**（`conftest.external_input`，提示缺哪个文件），不 skip——跳过的检查读起来像通过：`<C>/../00_plan/cluster-probe/20261113/02d_cdb_b200.csv`（`exclude` 的离线样本）与 WT-21 的 `wt21_round_v4.sbatch`（「写死资源」静态检查的阳性对照，路径见 `conftest.WT21_SBATCH`）。

## 偏差与未决（报告里同样列出）

1. 快照：`snapshot.plan_files` 把整个 `tools/` 与 `jobs/` 放进快照；`capkit_inner.sh` 的 `BOOT_PP` 是 `$SNAP/tools`。
2. 白名单：任务文本列了 `E10-proxy`、`E13-main`；INTERFACES §9.5 与 `capcommon.policy.NO_MODEL_EXPERIMENTS` 不含它们（§12 第 11 条），本组件按 INTERFACES，测试断言二者被拒。
3. 作业级文件的身份：`env/*.json`、`job.json`、`launches.json` 用启动计划的 `job_arm` 与 `launch_seq 0`；判定器对 `raw/env/` 直属文件只核作业键，见 INTERFACES 勘误「§2.1 / §10.1 身份核对的作业级例外」。
4. 作业级 `STATE`（`<作业目录>/STATE`）与 `import_origin.json`、`launches.json`、`batch.tsv`、`copyback/`、`local_leftovers/`、`shm_leftovers/` 是 INTERFACES 之外新增的作业目录内容。
5. `srun` 显式带 `--cpus-per-task=$SLURM_CPUS_PER_TASK`：较新的 Slurm 里 srun 不再继承 sbatch 的 `--cpus-per-task`【推断，未在本集群核对】，缺了它步骤可能只拿到 1 个核。
6. `--export=ALL`（与 WT-21 一致）：前端的环境变量会带进作业；容器内 `unset PYTHONPATH`，其余未清理。
7. `prep-tokenizer` 只申请最小资源（不带 `--exclusive`、不申请 GPU、分区可不给），见下节。
