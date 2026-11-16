# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m capstubs.fingerprint: the S-FE command-line arguments from the product's own
tokenizer fingerprint (design 2.3 "S-FE"; INTERFACES.md 11). Never hand-copied.

Runs inside the TRT-LLM container (imports tensorrt_llm; needs the CUDA compat library path on
nodes without a driver). The tokenizer is loaded the way the new-architecture frontend loads it
(`ModelLoader.load_hf_tokenizer(model_dir, trust_remote_code=..., use_fast=mode != "slow")`,
tensorrt_llm/serve/openengine_poc/facade.py:399-404 at 2f22a98d6e) and fingerprinted with
`tensorrt_llm.grpc.openengine_conventions.compute_tokenizer_fingerprint(tokenizer, model_dir)`
(:530 at 2f22a98d6e). The output is checked by round-tripping it through the product's own
`TokenizerFingerprint.from_extra` and `compare_tokenizer_fingerprints` (must be empty).

S-FE CLI at 2f22a98d6e (tests/unittest/grpc/openengine_client/fake_openengine.py:1966-1984):
--vocab-size INT, --eos-token-ids INT [INT ...], --profile-json JSON (FakeProfile overrides;
`fingerprint` wins over vocab/eos/bos when the engine reports ModelInfo.extra, :370-378).
"""

import argparse
import hashlib
import json
import os
import sys

SCHEMA = "capkit.fingerprint/1"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sfe_args(fp, extra):
    """S-FE arguments for a TokenizerFingerprint (product object) and its to_extra()."""
    args = ["--vocab-size", str(fp.vocab_size)]
    prof = {"bos_token_id": fp.bos_token_id, "fingerprint": extra}
    if fp.eos_token_ids:
        args += ["--eos-token-ids"] + [str(t) for t in fp.eos_token_ids]
    else:
        prof["eos_token_ids"] = []
    args += ["--profile-json", json.dumps(prof, sort_keys=True, separators=(",", ":"))]
    return args


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m capstubs.fingerprint", description=__doc__)
    p.add_argument("--tokenizer", required=True, help="tokenizer copy (model dir, no weights)")
    p.add_argument("--out", required=True)
    p.add_argument("--trust-remote-code", required=True, choices=("true", "false"),
                   help="must equal the frontend's (trtllm-serve --trust_remote_code; "
                        "default false)")
    p.add_argument("--tokenizer-mode", required=True, choices=("auto", "slow"),
                   help="must equal the frontend's tokenizer_mode (trtllm-serve has no flag "
                        "for it: auto unless its --config sets it)")
    a = p.parse_args(argv)
    if not os.path.isdir(a.tokenizer):
        print("fingerprint: --tokenizer %s is not a directory" % a.tokenizer, file=sys.stderr)
        return 3
    import tensorrt_llm  # noqa: PLC0415
    from tensorrt_llm.grpc.openengine_conventions import (  # noqa: PLC0415
        TokenizerFingerprint, compare_tokenizer_fingerprints, compute_tokenizer_fingerprint)
    from tensorrt_llm.llmapi.llm_utils import ModelLoader  # noqa: PLC0415
    tok = ModelLoader.load_hf_tokenizer(a.tokenizer, trust_remote_code=a.trust_remote_code
                                        == "true", use_fast=a.tokenizer_mode != "slow")
    if tok is None:
        print("fingerprint: no tokenizer could be loaded from %s" % a.tokenizer,
              file=sys.stderr)
        return 4
    fp = compute_tokenizer_fingerprint(tok, a.tokenizer)
    extra = fp.to_extra()
    back = TokenizerFingerprint.from_extra(json.loads(json.dumps(extra)))
    mism = ["round trip failed"] if back is None else \
        [str(m) for m in compare_tokenizer_fingerprints(fp, back)]
    if mism:
        print("fingerprint: self-check failed: %s" % mism, file=sys.stderr)
        return 4
    files = {n: _sha256(os.path.join(a.tokenizer, n)) for n in sorted(os.listdir(a.tokenizer))
             if os.path.isfile(os.path.join(a.tokenizer, n))}
    doc = {"schema": SCHEMA, "tokenizer_dir": os.path.abspath(a.tokenizer),
           "tokenizer_dir_files_sha256": files,
           "loader": {"call": "tensorrt_llm.llmapi.llm_utils.ModelLoader.load_hf_tokenizer",
                      "trust_remote_code": a.trust_remote_code == "true",
                      "use_fast": a.tokenizer_mode != "slow"},
           "function": "tensorrt_llm.grpc.openengine_conventions.compute_tokenizer_fingerprint",
           "tensorrt_llm": {"version": getattr(tensorrt_llm, "__version__", None),
                            "file": getattr(tensorrt_llm, "__file__", None)},
           "fingerprint": {"vocab_size": fp.vocab_size, "eos_token_ids": list(fp.eos_token_ids),
                           "bos_token_id": fp.bos_token_id,
                           "files_sha256": dict(fp.files_sha256)},
           "model_info_extra": extra, "self_check_mismatches": [],
           "sfe_args": sfe_args(fp, extra)}
    tmp = a.out + ".tmp.%d" % os.getpid()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, sort_keys=True, indent=1)
        f.write("\n")
    os.replace(tmp, a.out)
    print(" ".join(doc["sfe_args"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
