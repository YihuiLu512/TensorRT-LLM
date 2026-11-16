# fix-R2 FX2-jobs 进度

簇：Q8、Q9、Q10、Q14、Q15、Q18、Q19。独占：`jobs/**`、`tools/capjobs/**`、`tools/capcommon/policy.py`、`tests/jobs/**`。

- 2026-11-13 建 `_local/fx2-jobs/{fakebin,pyc}`，fakebin 10 个脚本。开始读源报告与代码。
- 读完 triage/brief/三路原文（A-R8/R9/R10、B-2/B-7/B-8、C-O1/O2/O4/O5）与 prep_tokenizer.sbatch、submit.py、policy.py、launches.py、相关测试。修前脚本存 `_local/fx2-jobs/prefix/prep_tokenizer.sbatch`（= HEAD）。
- 下一步：在 `_local/fx2-jobs/lab/` 用改写的 prep_lab.sh 复现 Q9/Q14/Q18 修前行为（约 1 分钟）。
- MEASURED 修前实验（`bash _local/fx2-jobs/prep_lab.sh _local/fx2-jobs/prefix/prep_tokenizer.sbatch _local/fx2-jobs/lab/pre`，EXIT=0，输出 `lab/pre.out`）：link_to_weight / link_to_blob / st_as_json / pickle_as_txt / zip_as_model / gguf_as_json 六例 rc=0 且权重字节进了副本（Q9 属实）；provenance_name：源 PROVENANCE.txt 被拷进再被作业覆盖，rc=4「copy has 2 listed files, expected 3」（Q14 属实，后果是大声失败而非静默）；backslash：rc=0、stderr 两次 `WARNING: 1 line is improperly formatted`、日志 `OK 3 files`（Q18 属实）。
- 开始改 `jobs/prep_tokenizer.sbatch`（Q9/Q14/Q18/Q19-O5/Q8 的作业侧）。
- 脚本已改（Q9 软链目标名 + 内容特征；Q14 保留名拒绝；Q18 反斜杠名拒绝 + `--strict` + 逐文件 OK 计数；Q19-O5 库根 realpath 再核；Q8 作业侧：job.env 新增 `CAPJOBS_CODE_SHA`、`CAPJOBS_PREP_SCRIPT_SHA256`，批处理步核计划目录里脚本的 sha，PROVENANCE 记两者）。`bash -n` 通过。
- MEASURED 修后实验（同命令，lab/post，EXIT=0，输出 `lab/post.out`）：9 例全部 rc=0，八种绕过分别以 link-to-weight / weight-content×5 / reserved-name / backslash 拒绝并记入 PROVENANCE，副本只有 config.json、tokenizer.json。
- 下一步：submit.py（Q8 提交侧、Q15 挂载）、policy.py（Q19-O4）。
- submit.py 已改：`job_mounts`（submit 与 prep-tokenizer 共用：要求次级盘恒等挂载 + `check_coverage`）、`git_head`/`committed_file`（prep 脚本须在 HEAD 已提交且工作区逐字节一致，dry-run 只报 `would_refuse_uncommitted`）、prep 新增必填 `--mount`、计划目录写 `MANIFEST.json`（code_sha + 各文件 sha256/size），台账 `snapshot_sha256` 改为 MANIFEST 的 sha。policy.py 已改：`library_roots()`、挂载串任意位置含 `,` / 路径含 `:` 拒绝、prep 挂载模式只认 ro/rw。
- 下一步：launches.py（Q10）。
- launches.py 已改（Q10：`run_with_budget` 分开 `timed_out` 与 `terminated`；`classify(..., terminated)` → 编排器自记的非 VALID 状态保留，否则 NOT_RUN「terminated: …」）。jobs/README.md 同步（submit 第 5 条挂载、prep 挂载/提交前/作业流程、launches 状态规则）。
- 去掉了刚加的容器内 `readlink -f` 二次核对：`ls -A` 本身跟随软链，字面根与解析根看到的是同一份内容，那段是冗余代码；Q19-O5 的实际修复是提交端把库根的 realpath 一并写进 `CAPJOBS_PREP_LIBRARY_ROOTS`。
- 下一步：改测试（tests/jobs）。
- 测试已改（test_scripts/test_container/test_submit），开始逐目录跑 tests/jobs（预计 1–3 分钟）。
- 修前红验证：git archive HEAD 的 tools/jobs/tests/jobs 解到 _local/fx2-jobs/prefix-tree，换上新测试跑（预计 1 分钟）。
- RS16 扫描额外发现：测量提交端 `policy.tokenizer_dir_sha256`（断言 tokenizer_no_weights）只按名字拒权重；已加 `policy.weight_content`（与 bash 同四条规则）并在算 sha 时读文件头拒绝。新增测试 test_tokenizer_dir_refuses_weight_content。
- tests/jobs 162 passed（EXIT=0）。再跑 tests/capcommon（依赖 policy）。
- 主会话转达（FX2-launch 发现）：launches.py 本地启动树 os.mkdir 未定 mode，umask 002 下 caplaunch ensure_own_dir 拒绝；开始改（显式 0700 + chmod + 读回），扫同类目录创建。
- umask 修复：launches.py 新增 `private_mkdir`（mkdir 0700 → 校验本用户真实目录 → chmod 0700 → 读回），用于 `<L>/runs`、`<L>/logs` 与每棵本地启动树；capkit_job.sbatch 的作业目录与 `env/`、prep 作业目录改 `mkdir -m 700`。本地根 / shm 根原已 `mkdir -m 700`。测试夹具改为预建 0700 本地根（与 inner 一致）；新增 test_launch_trees_are_private_under_umask_002（修前树：runs 为 0775，红）。
- 最终 tests/jobs 164 passed（EXIT=0），tests/capcommon 188 passed（EXIT=0）。fakebin.log 0 字节。完成。
