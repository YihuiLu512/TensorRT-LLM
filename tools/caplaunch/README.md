# caplaunch(launch 编排器)

运行:`PYTHONPATH=<C>/tools`。包在 `tools/caplaunch/`(INTERFACES §1.1;第一批写在 `tools/launch/caplaunch/`,集成时移过来，import 名不变)。
E0 配置生成器在 `configs/gen_e0_configs.py`。

## 模块

| 模块 | 作用 |
|---|---|
| `caplaunch.__main__` | `python -m caplaunch`:一次 launch(一个 launch 树 `<T>`)的全部编排 |
| `caplaunch.scenario` | 场景配置 `capkit.scenario/1` 的校验、工具参数表(INTERFACES §11)、argv 与占位符、与档位表/绑核表的交叉检查 |
| `caplaunch.orchestrator` | 主循环：起进程(taskset 绑核、独立进程组、launch 标记 env)、就绪、origin、档位边界事件、清空门、窗口扫描、停止与残留核验、收尾文件 |
| `caplaunch.procfs` | /proc 读取：starttime、子孙进程、标记扫描(`/proc/*/environ`)、线程清单与识别规则、网卡计数 |
| `caplaunch.ready` | 就绪判据:`capkit_ready` / `plain_port_file` / `report_addr` / `http_get` / `file_exists`(可组合，全部满足才算就绪) |
| `caplaunch.tiergen` | `python -m caplaunch.tiergen`:把档位表模板里的 `TBD:<名字>` 填成登记的数，生成并校验 `capkit.tiers/1`(可选同时写到达时刻表) |
| `caplaunch.httpprobe` | E0(c) HTTP 冒烟客户端(作为场景里的 aux 角色运行):逐帧记录文本增量是否为空 |
| `caplaunch.smoke` | `python -m caplaunch.smoke`:E0(c) 可行性冒烟各项，输出 `capkit.smoke/1`(PASS / FAIL / RECORDED / NOT_RUN) |
| `configs/gen_e0_configs.py`(不在本包内) | `configs/e0/*.json` 的生成器(交付物是生成出的 JSON;改配置请改生成器，生成到新目录后 diff) |

## `python -m caplaunch` 参数

身份参数(`capcommon.cli.add_identity_args`,`--role` 填 `launch`)外，以下全部必填，出处类的值可写 `none`(落 null):
`--run-dir --workdir --register --scenario --pin-table --shm-dir --snapshot-interval-ms --origin-lead-ms
--housekeeping-ms --kill-after-s --capkit-sha --probe-tree-sha --baseline-sha --snapshot-dir --image-path
--image-sha256 --hw-profile-sha`;`--param KEY=VALUE` 可重复(场景 `params_required` 全部要给);`--job-env-dir` 可选。

输入核对(任何一项不符 → 退出码 3,不起任何进程):register 里的文件 sha 与当前文件一致、到达时刻表头里的档位表 sha 一致、
场景/档位表/绑核表交叉检查(每角色核数、D1 角色间核不相交、门的 `poll_ns`/`max_snapshot_age_ns` ≥ 2×快照间隔、门读的槽位已在角色上声明)。
清空门只支持 `abs_max`;`baseline_band` 报 CONFIG(未实现)。

## 结果与退出码

| 情况 | STATE 行(by=caplaunch) | 退出码 |
|---|---|---|
| 全部档位跑完、退出核验干净 | VALID | 0 |
| enforce 门超时：按 stop.json 提前停 | VALID(reason 写明 drain_gate_timeout) | 5 |
| origin 前未就绪 / 起不来 / 收到信号 | NOT_RUN | 5(起进程失败为 4) |
| 进程在允许之前退出 | VOID | 5 |
| 停止后仍有残留进程(标记扫描) | VOID | 4 |

与 INTERFACES §5.7 的差异:§5.7 规定 STATE 第一行由 B6 写;本实现由编排器追加一行，供 B6 合并时裁决。

## 已知限制(详见 `_progress/B7-launch.md` 与最终报告)

- 线程识别：工具要在 ready 文件 `info.threads` 里发布 `{native_id: name}`,否则 Python 线程无法识别(C13 会红)。
- P0 产品进程的子进程只能出现在 `unattributed`(INTERFACES §12 U-5)。
- 各角色 `expected_procs` 取值(nullsse/capload 的父+worker 数、前端的真实进程数)未经真实工具核实。
- `launch.json` 里角色参数的 `{addr:R}` 保持未展开(运行时才知道)。
- 本机 RHEL8 cgroup v1 上 `systemd-run --user --scope -p MemoryMax=` 不生效(B4 发现);`caplaunch.smoke` 的 `read.memory_guard` 只读不判。
