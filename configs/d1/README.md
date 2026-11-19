# configs/d1：方案 D1（C++ engine 宿主，TRTLLM-18068）的场景模板

由 `configs/gen_d1_configs.py` 生成（`python configs/gen_d1_configs.py <新目录>`，再与本目录 diff）。每个臂从 `configs/snx/` 的同名来源模板派生，只改 D1 相关的几处，其余（角色、核数、档位模板、门、对账链）与来源逐字相同，所以 D1 臂与 S-NX 臂可以并排比。规划见 `/home/scratch.yihlu_coreai/dev/feat-dev-J17503/tmp/meta/explain/08_engine-host-cpp-d1/04_perf-plan.md` §2～§4。

| 实验 | 臂 | 来源 | 差异 |
|---|---|---|---|
| E16 | e16-py-k4-{mc,r} | e8-k4-* | 无（E8-py：Python 宿主单前端模式，编号改为 E16，便于与 D1 臂同作业） |
| E16 | e16-pynd-k4-{mc,r} | e8-k4-* | 作业输入 `request_bytes` 加 `--extra-detokenize false`（E8-py-nd：每个请求带 `extra.detokenize=false`，与 C++ 宿主的缺省一致） |
| E16 | e16-2l-k4-{mc,r} | e8-k4-* | 宿主角色加 `--openengine-host-impl cpp`（launcher 两 lane 多前端模式，C++ 宿主在 lane 1 空转）与 `--report-target private`：负载打到 launcher 回环端口上的 Python Generate（lane 0），即 E8-2l |
| E16 | e16-nx-k4-{mc,r} | e8-k4-* | 宿主角色 cpp 模式；负载打到 C++ 宿主的公开端口（E16-nx）；`--cpp-host-bin {param:d1_bin}`、`--grpc-threads {param:d1_grpc_threads}`、`--diag-out {raw}/d1`（自报 `diag.jsonl`、`meta.json`） |
| E16 | e16-nx-nodiag-k4-{mc,r} | e8-k4-* | 同 e16-nx，但不开 `--diag-out`（P0：自报的开销） |
| E9 | e9-nc{1,8}f0-{mc,r} | e9-n{1,8}f0-* | 宿主角色同 e16-nx（NC 臂：新架构前端连 C++ 宿主） |
| E9 | e9-n2l8f0-{mc,r} | e9-n8f0-* | 宿主角色同 e16-2l（N2l 臂：前端连 launcher 回环端口） |

要点：
- C++ 宿主由 launcher（`trtllm-serve ... --openengine-host-impl cpp`）作为宿主角色进程的子进程拉起，继承宿主角色自己的核（4 核，与 Python 宿主相同；rank0 的 3 核与 procmon 的 1 核仍按 `--worker-cores` / `--procmon-cores` 切走）。procmon 走进程树，C++ 宿主的线程（`d1-*`、gRPC 的 `event_engine`）都在宿主角色的 procmon 文件里。
- `capstubs.serve_nx` 的 C++ 宿主选项：`--cpp-host-bin`、`--cpp-host-arg`（按序拼成 `TLLM_OPENENGINE_CPP_HOST_ARGS`，旗标写成 `--cpp-host-arg=--diag-out`）、`--cpp-host-inject`（`TLLM_D1_INJECT`，如 `hs:busy:2000`，供 V 干预）、`--report-target private`。这些选项只在服务参数带 `--openengine-host-impl cpp` 时允许；launcher 自己设的 `--listen`、`--launcher-endpoint`、`--template` 不许再给。
- 新参数：`d1_bin`（镜像里是 `/usr/local/bin/trtllm-openengine-host`）、`d1_grpc_threads`（规划：4，核数扫描时取核数）。
- **E16-zr 未生成**：响应源是回放桩 S-ZR（W13：`cpp/tensorrt_llm/openengine_host/tools/szr/` 与 capkit 分支 `szr-d1` 的 `tools/capstubs/szr_launch.py`，尚未交回）。宿主侧选项与 e16-nx 相同（`--cpp-host-*`、4 核、`--grpc-threads`、`--diag-out`）；响应源怎样接到 C++ 宿主的 lane 由 S-ZR 的启动入口决定，等它合入 main 后在本生成器补 `e16-zr-k4-*` 与核数臂 c2 / c8。
- 产品角色 P0：与 configs/snx 相同。
