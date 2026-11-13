# capstubs

包在 `tools/capstubs/`(INTERFACES §1.1)。仓库里跑：`PYTHONPATH=<C>/tools`。

## `python -m capstubs.fingerprint`

从产品自己的 tokenizer 指纹生成 S-FE(`tests/unittest/grpc/openengine_client/fake_openengine.py`,零改动)的命令行参数。
绝不手抄词表大小 / eos / bos。

- 必须在 TRT-LLM 容器里跑(要 import `tensorrt_llm`;无驱动节点要先把 CUDA compat 库目录放进 `LD_LIBRARY_PATH`)。
- 参数：`--tokenizer <只含 tokenizer 文件的模型目录> --out <json> --trust-remote-code {true,false} --tokenizer-mode {auto,slow}`;
  后两项必须与前端的设置一致(前端用 `ModelLoader.load_hf_tokenizer(..., use_fast=mode != "slow")` 加载)。
- 计算：`tensorrt_llm.grpc.openengine_conventions.compute_tokenizer_fingerprint(tokenizer, model_dir)`;
  输出前用产品自己的 `TokenizerFingerprint.from_extra` + `compare_tokenizer_fingerprints` 做往返自检，不一致退出码 4。
- 输出 `capkit.fingerprint/1`,其中 `sfe_args` 是可直接拼进 S-FE 命令行的参数列表：
  `--vocab-size N [--eos-token-ids ...] --profile-json {"bos_token_id":..,"fingerprint":<to_extra()>}`
  (S-FE 在引擎上报 `ModelInfo.extra` 时以 `fingerprint` 为准)。场景配置用 `{"splice_json": "{param:fingerprint_json}", "key": "sfe_args"}` 引用它。

测试 `tests/launch/test_fingerprint.py` 用 probe-tree 里真实的 `openengine_conventions.py` + 假的 tokenizer 加载器；
真实 tokenizer + 真实容器的运行属于 E0(c),未在本机运行。
