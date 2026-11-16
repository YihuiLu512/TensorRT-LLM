# fix-SM16 进度（2026-11-14）
- WT 已建：capkit-sm16 @ 0a1410a4；开始读材料
- 2026-11-14 19:53:34 读完材料；建 _local/sm16（fakebin 10 个替身 exit 97、pyc、rt.sh）；跑修前 tests/judge/test_pilot.py 基线（约 1–2 分钟）
- 2026-11-14 19:53:59 修前探针：good 样本单臂 synthetic-a、序列长 4，旧代码在 L=4（=全窗）归零 ⇒ window_ns=4 s；新规则下它属于全窗饱和 ⇒ null（test_pilot_from_good_sample 的断言要随之改）。开始改 pilot.py
- 2026-11-14 19:55:45 pilot.py 改完（search_by_arm、_window_length 等长/全窗饱和）；test_pilot.py 改 good 断言 + 新 4 个；跑修后 test_pilot（约 1 分钟）
- 2026-11-14 19:56:27 修前/变异：_local/sm16/mut/r1（PRE=基线 pilot.py + 新测试；M0 未变异；M1 去掉按臂分组（各臂用全体档）；M2 去掉全窗饱和判断；M3 顶层取 min；M4 预热用全体档；M5 臂内不等长截到最短），各跑 tests/judge/test_pilot.py，约 1–2 分钟
- 2026-11-14 19:58:20 文档改完（INTERFACES 勘误 SM-16 + §9.4 指针、capjudge/README）；建测试用软链 WT/probe-tree；后台跑全量验证 v1（10 目录 + selftest + py_compile + bash -n，约 8–10 分钟）
- 2026-11-14 20:01:47 全量验证 v1 进行中（capcommon 190、capload 33、capprobe 159、envgate 71 已过）；等待其余目录
- 2026-11-14 20:07:47 v1：grpc 36、jobs 198、judge 97（原 93 + 新 4）已过；等 launch/nullsse/sidecar/selftest
- 2026-11-14 20:09:25 v1 结果：launch 7 failed 均为 OSError: AF_UNIX path too long（我给 pytest 设了 TMPDIR=_local/sm16/tmp，forkserver socket 路径超长，是验证脚本的问题）；selftest 漏了必填 --tools/--tmp。其余全绿（capcommon 190、capload 33、capprobe 159、envgate 71、grpc 36、jobs 198、judge 97、nullsse 4、sidecar 5；py_compile 188/0；bash -n 3/0）。去掉 TMPDIR 重跑 tests/launch 与 selftest（v1b，约 4 分钟）
- 2026-11-14 20:12:34 v1b：selftest EXIT=0（25 样本，errors 空）；tests/launch 189 passed。全量验证全绿。用工作区新 pilot 对正式输入重跑 → 02_e0/E0/analysis/pilot.sm16.json（约 1–3 分钟）
- 收尾：测试用软链 WT/probe-tree 移到 `_local/sm16/probe-tree.link`；py_compile 等留下的 `tools/{capcommon,capjudge,capload}/__pycache__` 移到 `_local/sm16/stray_pycache/`（不删）。提交。

## 结果汇总

### 改动
| 处 | 改动 |
|---|---|
| `tools/capjudge/pilot.py` | 新 `search_by_arm`：按启动身份的 `arm` 分组，每臂分别跑 `_window_length` 与 `_warmup_length`（预热用本臂窗口）；输出 `window_ns_by_arm`、`warmup_ns_by_arm`、`window_search_by_arm`、`warmup_search_by_arm`；顶层 `window_ns`/`warmup_ns` = 各臂都非 null 时取最大，否则 null；原全体一张的 `window_search`/`warmup_search` 不再输出。`_window_length`：臂内序列不等长 ⇒ null + `unequal_series_lengths`（不截到最短）；只在 L = 全窗归零 ⇒ null + `saturated_at_full_window: true`。噪声带、T1、带内判据不变 |
| `tests/judge/test_pilot.py` | 新 4 个：两臂不同长度（短臂全窗饱和、长臂 L=8 归零）；单臂等长回归（`_window_length` 结果与旧代码逐字相同）；臂内不等长；预热用各臂自己的窗口、顶层取最大。`test_pilot_from_good_sample` 改断言：good 样本单臂 4 s 序列只在 L=4（全窗）归零 ⇒ 现在 null + saturated（旧代码给 4 s，正是被定为同义反复的情形） |
| 文档 | INTERFACES 新增「SM-16 pilot 按臂搜索（2026-11-14）」，§9.4 加指针；`tools/capjudge/README.md` 产出说明一句改为按臂口径 |

### 修前 / 变异（`_local/sm16/mut/r1`，只跑 tests/judge/test_pilot.py）
- PRE（基线 pilot.py + 新测试）：5 failed / 13 passed（good 样本 KeyError window_ns_by_arm；4 个新测试 AttributeError search_by_arm）。单臂回归测试里 `_window_length` 那条断言在基线上通过（失败在其后的 search_by_arm），说明单臂等长行为不变。
- 基线 `_window_length` 对两臂合成数据：`(None, {2: 2/4 带外, 4: 2/4, 6: 1/4})`（`mut/old_on_synthetic.py`）。
- M0 未变异 18 passed；M1 去掉按臂分组（各臂用全体档）2 failed；M2 去掉全窗饱和判断 2 failed；M3 顶层取 min 1 failed；M4 预热用全体档 1 failed；M5 臂内不等长截到最短 1 failed。

### 全量验证（logs/v1 + v1b）
capcommon 190、capload 33、capprobe 159、envgate 71、grpc 36、jobs 198、judge 97（原 93 + 新 4）、launch 189（v1b；v1 里 7 个失败是我给 pytest 设 TMPDIR 导致 AF_UNIX path too long，去掉后全过）、nullsse 4、sidecar 5；selftest EXIT=0（`--expect-samples 25`，v1b）；py_compile 188/0；bash -n 3/0。fakebin.log 见交回报告。

### 正式输入重跑（`02_e0/E0/analysis/pilot.sm16.json`，新建，未覆盖任何文件）
| 臂 | 序列 | window_ns | 带外条数（L=2,4,…） | warmup_ns |
|---|---|---|---|---|
| e0a-nullsse | 12 × 30 s | null，saturated_at_full_window | 6,7,5,4,2,3,4,2,1,3,2,2,2,3,0（L=30） | null |
| e0b-s1-single | 6 × 20 s | 10 s | 4,2,1,1,0 | 0 |
| e0b-s2-two-stage | 6 × 20 s | 6 s | 2,3,0 | 0 |
| e0b-s3-aux-thread | 6 × 20 s | 16 s | 3,1,3,1,2,1,1,0 | 0 |
| e0b-s4-sleep | 6 × 20 s | 16 s | 4,2,3,2,3,1,1,0 | 0 |
顶层 window_ns / warmup_ns = null（e0a 臂为 null）。其余字段（noise_bands、inputs、d14 等）与 pilot.json 完全相同；只多了四个 _by_arm 字段、少了 window_search/warmup_search。
