# configs/e0 — E0 场景配置与档位表模板

全部 JSON 由 `configs/gen_e0_configs.py` 生成(改配置请改生成器，生成到新目录后与本目录 diff)。
这里没有任何登记过的数字：所有 E0 自己要定的数都是 `TBD:<名字>` 或 `{param:<名字>}`,在登记(register)时给出。

## 文件

| 文件 | 内容 |
|---|---|
| `e0a-nullsse.scenario.json` | E0(a) 第二条线:capload → nullsse(+ sidecar、canary) |
| `e0a-frontend-sfe.scenario.json` | E0(a):capload → `trtllm-serve --openengine-endpoint`(新架构前端)→ S-FE;产品进程 P0,门只读 capload |
| `e0b-s1-single` … `e0b-s4-sleep.scenario.json` | E0(b) 已知答案中继：节拍源 `capgrpc.nullgrpc --pacing beat` → `capselfcheck.relay`(1 或 2 级)→ 排空端 `capgrpc.load`;场景 1 单级忙等、2 两级(交换 = 第二份 values)、3 辅助线程忙等、4 sleep |
| `e0c-frontend-http.scenario.json` | E0(c):S-FE + 真实 tokenizer + 前端;`httpprobe` 在 origin 前发 completions / chat 各一条流式请求并记录空增量帧;之后 capload 打一档低速 warmup + 一个带门的 drain(档位表 schema 要求最后是带门的 drain,而这里只有 capload 发布 backlog 槽位) |
| `e0c-httpprobe.requests.json` | httpprobe 的请求体(`model` 为 `TBD:served-model-name`,登记时替换) |
| `e0c-smoke.json` | `python -m caplaunch.smoke` 的项目配置(`capkit.smoke_config/1`) |
| `*.tiers.template.json` | 档位表模板(`capkit.tiers_template/1`) |
| `*.values.todo.json` | 对应模板要填的全部名字(值为 null);填完后跑 tiergen,仍有 null 会被拒绝 |

## 登记流程(每个场景)

1. 复制 `<name>.values.todo.json`,把每个 null 换成登记的数(单位见名字：`_ns` 纳秒、`rate_` req/s、`c_` 并发流数)。
2. `python -m caplaunch.tiergen --template <name>.tiers.template.json --values <填好的 values> --out <W>/02_e0/E0/register/<name>.tiers.json --schedule-out <W>/02_e0/E0/register/<name>.schedule.capbin --max-rows <N>`
   (输出已存在会被拒绝，登记文件不覆盖)。
3. 场景里的 `prompts` 路径(`<name>.prompts.json`)由提示词池生成步骤给出(本组件未实现)。
4. 把三份文件的 sha 写进 register.json,launch 时编排器逐个核对。

## S-FE 的指纹参数(SEAMS SM-6)

`e0a-frontend-sfe`、`e0c-frontend-http` 声明作业输入 `job_inputs.fingerprint`(INTERFACES 勘误 B2B-9):作业在门禁之后用 `python -m capstubs.fingerprint --tokenizer {param:tokenizer_dir} --trust-remote-code {param:trust_remote_code} --tokenizer-mode {param:tokenizer_mode}` 生成，S-FE 角色用 `{"splice_json": "{input:fingerprint}", "key": "sfe_args"}` 拼参数。`trust_remote_code`、`tokenizer_mode` 必须与前端加载 tokenizer 的设置一致。

## 冒烟(E0(c))

- `python -m caplaunch.smoke --config configs/e0/e0c-smoke.json --out <新目录> --param python=<容器 python> --param cuda_compat_dir=<compat 目录> --param loopback_sender_cpu=<n> --param loopback_receiver_cpu=<m> --param cores_<角色>=<核数> ...`
- `--tree <e0c-frontend-http 的 launch 树>` 才会跑访问日志扫描与空增量帧检查；没给的项一律 NOT_RUN,不猜。
- `env.mpi_launch_probe` 要等 S-NX / MpiPoolSession(第 2 批)才能做，现在固定 NOT_RUN。

## 未核实 / 待对齐

- `expected_procs`:nullsse/capload 写的是「父进程 + 2 个 worker = 3」,前端写的是 1,均未用真实工具核实。
- E0(b) 对账链里 capgrpc 工具的 summary 键名(`msgs_sent` / `msgs_read` / `streams_accepted`)未与 B2 的实现核对;
  B2 实际模块名是 `capload_grpc`,与 INTERFACES 的 `capgrpc.load` 不一致，场景里按 INTERFACES 写。
- E0(a) 前端那条对账链在 P0 下只有 capload 一跳(产品进程不发布计数)。
- 前端角色的 `proc_kinds` 写 `["frontend", "frontend_openengine", "frontend_single"]`(`--num_serve_frontends 1` 时只跑新架构单前端进程，它经过 capprobe SITES 里这三类的位点：H1/H3、F2/F4、Lfe),未用真实 `trtllm-serve` 核实;S-FE 等不跑探针代码的角色写 `[]`。
- sidecar 的 `--nvidia-smi` 用 `{nvidia_smi}` 占位，由编排器 `--nvidia-smi` 展开，配置里不写路径。
