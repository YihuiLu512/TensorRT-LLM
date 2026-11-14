# fix-SM14（C7 跨档界请求的块行计数）进度

- WT = `_local/wt-sm14`（分支 capkit-sm14，起点 d9c500c72d211006aef6f457ac3c71317f194178）；C（01_code main）只读；02_e0/E0/runs 只读（拷到 WT/_local/sm14/trees/ 再判）。

## 进度日志
- 2026-11-14 16:23:46 开工：建 worktree；读 SEAMS SM-14、checks.py C7/C6、report.py、treedata、symptoms；下一步拷贝 J5933747 四棵树（L0–L3，runs 下无 L4）。
- 2026-11-14 16:26:34 修前重判 L0 拷贝（WT 未改，code sha=git HEAD d9c500c7），约 1 分钟
- 2026-11-14 16:28:25 修复写完（checks.py _chunk_rows 计入其后清空档的块行；stray 口径不变），新测试 tests/judge/test_c7_chunk_rows.py 8 个；跑修后新测试
- 2026-11-14 16:28:50 跑变异 M0–M3（_local/sm14/mut，约 1 分钟）
- 2026-11-14 16:30:16 修后预判（工作区未提交，CAPKIT_CODE_SHA=40 个 0 只为不让判定器因工作区脏而降级）：拷贝 L0–L3 四棵树，约 2–4 分钟
- 2026-11-14 16:30:45 首次预判 L0 因共用 wd 的 JUDGED.tsv 已有 pre 的行而 12（判定器的一次一判保护，符合预期）；改为每次重判各用自己的 wd，旧目录改名保留（rj/postwip-try1-shared-wd、wd-shared-pre-postwip1）；重跑预判
- 2026-11-14 16:31:17 修后预判结果：L0 只有档 4 C7 一行变（RED→PASS，detail 记 9 行来自档 5），L1–L3 checks.tsv 逐字节不变、0 RED。建测试用软链 WT/probe-tree -> ../../probe-tree（提交前移到 _local/sm14/probe-tree.link）；后台跑全量验证 v1（10 目录 + selftest + py_compile + bash -n，约 8–10 分钟）
- 2026-11-14 16:31:43 变异 mut/summary.txt：M0 未变异 8 passed；M1 基线 checks.py ⇒ 8 failed（7 个直调因签名 TypeError，check_chains 那个是行为红 ('RED',1,6)）；M1b 去掉跨档计数（if False）⇒ 2 failed（跨档合计 PASS、check_chains 接线）；M2 check_chains 不传 next_drain ⇒ 1 failed（接线）；M3 清空档行不按抽样集过滤 ⇒ 1 failed（他档行不计入）
- 2026-11-14 16:39:59 全量验证 v1 全绿（_local/sm14/logs/v1/summary.txt：capcommon 190、capload 33、capprobe 159、envgate 71、grpc 36、jobs 198、judge 93（原 85 + 新 8）、launch 189、nullsse 4、sidecar 5；selftest EXIT=0（25 样本）；py_compile 188/0；bash -n 3/0）。WT/probe-tree 软链移到 _local/sm14/probe-tree.link（不删）。准备提交；提交后用提交的 HEAD 再判四棵树（rj/final），结果见交回报告。

## 结果汇总

### 核实（MEASURED）
- 读码：`checks._chunk_rows` 的抽样请求按时间表计划档取，块行只合并本档 TierData 的 `loadgen.chunks`；`capload/report.py` 的块行按接收时刻 `tier_seq_at` 归档。
- 数据（拷贝树，`_local/sm14/repro/count_chunks.py`）：L0 全部 7400 个抽样 req 里只有 req 5599 跨两档文件（004_up-sub-hi 8 行、005_drain-03 9 行，合计 17 = frames）；按本档文件数不等的只有它，跨档合计后 0 个不等。
- 修前判定器（WT 未改，git HEAD d9c500c7）重判 L0 拷贝：EXIT=10，checks.tsv 与原判逐字节相同（档 4 C7 RED）。

### 修复
| 处 | 改动 |
|---|---|
| `tools/capjudge/checks.py` | `check_chains` 把 `next_drain` 传给 `_chunk_rows`；抽样请求的块行 = 本档文件 + 其后清空档文件里该请求的行（清空档里只取本档抽样请求的行）；仍要求恰好等于帧数；stray 口径不变（只看本档自己的文件）；有跨档行时 detail 另记一句 |
| `tests/judge/test_c7_chunk_rows.py` | 新 8 个：跨档合计=帧数 PASS；合计≠帧数（少 1、多 1）RED；行在本档与其后清空档之外不计（next_drain 为空或 None）RED；本档文件里他档请求的行仍 stray RED；清空档里他请求的行不计不 stray；未抽样请求不计；失败请求留给 C3；check_chains 接线 |
| 文档 | INTERFACES 新增「SM-14 判定器修复（2026-11-14）」，§5.4 归档规则与 §5.4.2 chunks 行各加指针；capjudge/README C7 口径一句 |

### 同根因扫描
- 修：C7（唯一按请求数逐事件行的检查）。
- 不受影响：C5、C6（跳值来自按计划档计的自报或到下一清空档结束的槽增量）；C1、C3、P-SENT、X1/X3 只用按计划档归档的 requests 或 summary。
- 只列不改：symptoms.msg_lag_from_chunks / pilot._lag_series 的 streams 连接（规则 3 下不可触发）；msg_lag_from_chunks / msg_lag_from_msgs 对接收晚于档结束的消息截尾（X2 灵敏度，不同根因）；清空档文件从不做 stray 核对（原有缺口）。

### 重判（工作区预判 rj/postwip，CAPKIT_CODE_SHA=40 个 0）
- L0：checks.tsv 只变一行：档 4 C7 RED(red 1) → PASS(red 0)，detail「9 rows of 1 sampled requests counted from tier 5」；launch_verdict VOID/10 → NOT_RUN/11（档 4 VOID → NOT_RUN，原因 C1/C6/C13 NOT_RUN，与档 2、6 相同）。
- L1–L3：checks.tsv 与原判逐字节相同，0 RED，NOT_RUN/11 不变。runs/ 下没有 L4 树。
