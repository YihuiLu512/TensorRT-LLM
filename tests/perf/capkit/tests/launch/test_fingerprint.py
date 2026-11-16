# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capstubs.fingerprint against the REAL compute_tokenizer_fingerprint source.

The product module openengine_conventions.py is stdlib-only at import time, so it is copied
from the probe tree (baseline commit) into a throw-away `tensorrt_llm` package; only the
tokenizer loader is faked. The S-FE argument parser below mirrors fake_openengine.py:1966-1984
at 2f22a98d6e (same-source risk: the real parser is exercised only in the container smoke).

Needs <C>/probe-tree (the capprobe TRT-LLM worktree, not part of this repository). Without it the
tests fail rather than skip: a skipped fingerprint test would read as a pass.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import conftest as cf

SRC = os.path.join(cf.C, "probe-tree", "tensorrt_llm", "grpc", "openengine_conventions.py")

FAKE_LOADER = '''
class _Tok:
    def __init__(self, n, eos, bos):
        self._n, self.eos_token_id, self.bos_token_id = n, eos, bos
    def __len__(self):
        return self._n

class ModelLoader:
    @staticmethod
    def load_hf_tokenizer(model_dir, trust_remote_code=True, use_fast=True):
        import json, os
        cfg = json.load(open(os.path.join(model_dir, "fake_tok.json")))
        return _Tok(cfg["n"], cfg["eos"], cfg["bos"])
'''


def _fake_pkg(tmp):
    root = os.path.join(str(tmp), "pkg")
    for d in ("tensorrt_llm", "tensorrt_llm/grpc", "tensorrt_llm/llmapi"):
        os.makedirs(os.path.join(root, d))
        with open(os.path.join(root, d, "__init__.py"), "w") as f:
            f.write("__version__ = 'fake'\n" if d == "tensorrt_llm" else "")
    assert os.path.isfile(SRC), "needs the probe-tree worktree: %s missing" % SRC
    shutil.copyfile(SRC, os.path.join(root, "tensorrt_llm", "grpc", "openengine_conventions.py"))
    with open(os.path.join(root, "tensorrt_llm", "llmapi", "llm_utils.py"), "w") as f:
        f.write(FAKE_LOADER)
    return root


def _tokdir(tmp, eos, gen_eos):
    d = os.path.join(str(tmp), "tok")
    os.makedirs(d)
    json.dump({"n": 151669, "eos": eos, "bos": None}, open(os.path.join(d, "fake_tok.json"), "w"))
    open(os.path.join(d, "tokenizer.json"), "w").write('{"x": 1}')
    open(os.path.join(d, "tokenizer_config.json"), "w").write('{"y": 2}')
    json.dump({"eos_token_id": gen_eos}, open(os.path.join(d, "generation_config.json"), "w"))
    return d


def _sfe_parser():  # mirrors fake_openengine.py:1966-1984 at 2f22a98d6e
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="trtllm-like")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--port-file")
    p.add_argument("--model")
    p.add_argument("--max-context-length", type=int)
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--eos-token-ids", type=int, nargs="+")
    p.add_argument("--profile-json")
    p.add_argument("--tokens-per-step", type=int, default=1)
    p.add_argument("--step-delay", type=float, default=0.0)
    return p


def _run(tmp, tok, out):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_fake_pkg(tmp)] + cf.PATHS)
    return subprocess.run([sys.executable, "-m", "capstubs.fingerprint", "--tokenizer", tok,
                           "--out", out, "--trust-remote-code", "false",
                           "--tokenizer-mode", "auto"], env=env, capture_output=True,
                          text=True, timeout=60)


def test_fingerprint_matches_product_function(tmp_path):
    tok = _tokdir(tmp_path, 151645, [151645, 151643])
    out = os.path.join(str(tmp_path), "fp.json")
    r = _run(tmp_path, tok, out)
    assert r.returncode == 0, r.stderr
    doc = json.load(open(out))
    fp = doc["fingerprint"]
    assert fp["vocab_size"] == 151669 and fp["eos_token_ids"] == [151643, 151645]
    assert fp["bos_token_id"] is None
    assert sorted(fp["files_sha256"]) == ["tokenizer.json", "tokenizer_config.json"]
    a = _sfe_parser().parse_args(doc["sfe_args"])
    assert a.vocab_size == 151669 and a.eos_token_ids == [151643, 151645]
    prof = json.loads(a.profile_json)
    assert prof["fingerprint"] == doc["model_info_extra"] and prof["bos_token_id"] is None
    # the engine side's view (FakeProfile.fingerprint -> ModelInfo.extra) parses back equal
    sys.path.insert(0, _fake_pkg(tmp_path / "again"))
    try:
        from tensorrt_llm.grpc import openengine_conventions as oc
        eng = oc.TokenizerFingerprint.from_extra(prof["fingerprint"])
        loc = oc.TokenizerFingerprint(fp["vocab_size"], tuple(fp["eos_token_ids"]),
                                      fp["bos_token_id"], tuple(sorted(fp["files_sha256"].items())))
        assert oc.compare_tokenizer_fingerprints(loc, eng) == []
        # positive control: a different tokenizer file must make the comparison fail
        bad = dict(prof["fingerprint"])
        bad[oc.FP_FILES_SHA256] = dict(bad[oc.FP_FILES_SHA256], **{"tokenizer.json": "0" * 64})
        assert oc.compare_tokenizer_fingerprints(loc, oc.TokenizerFingerprint.from_extra(bad))
    finally:
        sys.path.pop(0)
        for m in [m for m in sys.modules if m.startswith("tensorrt_llm")]:
            del sys.modules[m]


def test_empty_eos_goes_into_profile_json(tmp_path):
    tok = _tokdir(tmp_path, None, None)
    out = os.path.join(str(tmp_path), "fp.json")
    r = _run(tmp_path, tok, out)
    assert r.returncode == 0, r.stderr
    args = json.load(open(out))["sfe_args"]
    assert "--eos-token-ids" not in args
    assert json.loads(_sfe_parser().parse_args(args).profile_json)["eos_token_ids"] == []
