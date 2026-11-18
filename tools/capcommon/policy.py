# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submission policy for jobs that must not load a real model (INTERFACES.md 9.5; plan
CONSTRAINTS section 2 "mechanisms" 1, 2, 5, 6), for the tokenizer copy (user decision
2026-11-13, INTERFACES erratum R1b), and for the whitelisted experiments that load a real model
(MODEL_EXPERIMENTS; user decision 2026-11-18, DECISIONS 32nd answer).

These are mechanisms, not judgement calls: the submit entry calls check_no_model_submission()
(or check_prep_tokenizer_submission()) before writing a PLANNED ledger row, and
ledger.submissions_append() refuses PLANNED rows whose assertions_json does not show every
assertion passed.
"""

import hashlib
import json
import os

from .exitcodes import ConfigError

# Experiment ids accepted by the no-model submission entry. Only ids the plan lists as "does not
# load a real model" AND that have their own experiment id. E10's substitute arm and E13's main
# body share an id with model-loading parts, so they are NOT listed; they need distinct ids
# confirmed by the user and an INTERFACES.md erratum before they can be submitted.
NO_MODEL_EXPERIMENTS = ("E0", "E1", "E2", "E2a", "E2b", "E3", "E4", "E5", "E6", "E7", "E8",
                        "E9", "E15-A")

# Model libraries that a no-model job must not be able to see (CONSTRAINTS section 2 item 1).
FORBIDDEN_MOUNT_PATHS = ("/home/scratch.trt_llm_data_ci", "/home/scratch.trt_llm_data")

# Weight-file suffixes that must not appear in a tokenizer copy.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".engine")

MAX_GPUS = 1


def weight_content(prefix, size):
    """True if a file of `size` bytes starting with `prefix` (its first >= 16 bytes, or all of it)
    has a weight-format header: pickle (protocol >= 2), zip (torch .pt/.bin), GGUF, or
    safetensors (little-endian u64 header length N <= size - 8, then '{"'). The bash tokenizer
    copy (jobs/prep_tokenizer.sbatch weight_content) applies the same four rules."""
    if size <= 0 or not prefix:
        return False
    if prefix[:1] == b"\x80" or prefix[:4] in (b"PK\x03\x04", b"GGUF"):
        return True
    if len(prefix) >= 10 and prefix[8:10] == b'{"' and prefix[4:8] == b"\0\0\0\0":
        n = int.from_bytes(prefix[:8], "little")
        return 2 <= n <= size - 8
    return False


ASSERTION_KEYS = ("experiment_no_model", "gpus_le_1", "no_model_mounts", "tokenizer_no_weights",
                  "basis_present")


# ---- the tokenizer copy job (purpose=prep_tokenizer) -------------------------------------------
# The only job type allowed to mount part of a model library: exactly one registered model
# directory, read-only, no GPU; it copies whitelisted non-weight files into the registered
# destination (<code root>/PREP_TOKENIZER_DEST_REL/<name>). Every other job is still refused by
# check_mounts. The bash job (jobs/prep_tokenizer.sbatch) receives these lists through job.env.
PREP_TOKENIZER_PURPOSE = "purpose=prep_tokenizer"
PREP_TOKENIZER_EXPERIMENT = "PREP-TOKENIZER"  # ledger `experiment` of these plans (not an E*)
PREP_TOKENIZER_MODEL_DIRS = ("/home/scratch.trt_llm_data_ci/llm-models/GLM-5.3-Flash",)
PREP_TOKENIZER_DEST_REL = "configs/tokenizers"
# Top-level names copied (bash [[ name == glob ]]), after the weight-suffix refusal.
PREP_TOKENIZER_ALLOW_GLOBS = ("tokenizer*", "*.json", "*.jinja", "*.txt", "*.model",
                              "*.tiktoken")
# At least one of these must be copied (a tokenizer without a vocabulary is a failed copy).
PREP_TOKENIZER_VOCAB_GLOBS = ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt",
                              "*.tiktoken")
# A whitelisted file above this size fails the job (nothing that large belongs to a tokenizer).
PREP_TOKENIZER_MAX_FILE_BYTES = 64 * 1024 * 1024
PREP_ASSERTION_KEYS = ("purpose_prep_tokenizer", "gpus_zero", "model_mount_registered_ro",
                       "other_mounts_no_model", "basis_present")

# ---- jobs that load a real model (user decision 2026-11-18 21:4x, DECISIONS 32nd answer) --------
# The only experiment ids whose jobs may mount part of a model library: each id names the model
# directories it may mount (identity mount, read-only; at least one of them) and its GPU ceiling
# (the design's per-job GPU count). No other id may mount a model library: check_no_model_submission
# is unchanged and still refuses every id below (none is in NO_MODEL_EXPERIMENTS).
#   E11      tiny real model (design 2.3 S-TM: Qwen3-0.6B, load_format dummy, 1 layer), 1 GPU
#   E13-S    E13 spot checks on the E11 set-up, 1 GPU
#   E15-B    host <-> rank 0 hop, single-GPU TP1 (WT-21 pa_u8: gpt-oss-120b), 1 GPU
#   E12-L2a, E12-L2b, E14   U4 point (GLM-5.3-Flash FP8, TP4), 4 GPUs of an exclusive 8-GPU node
MODEL_LIBRARY = "/home/scratch.trt_llm_data_ci/llm-models"
_QWEN3_06B = MODEL_LIBRARY + "/Qwen3/Qwen3-0.6B"
_GPT_OSS_120B = MODEL_LIBRARY + "/gpt_oss/gpt-oss-120b"
_GLM_53_FLASH = MODEL_LIBRARY + "/GLM-5.3-Flash"
MODEL_EXPERIMENTS = {
    "E11": {"max_gpus": 1, "model_dirs": (_QWEN3_06B,)},
    "E13-S": {"max_gpus": 1, "model_dirs": (_QWEN3_06B,)},
    "E15-B": {"max_gpus": 1, "model_dirs": (_GPT_OSS_120B,)},
    "E12-L2a": {"max_gpus": 4, "model_dirs": (_GLM_53_FLASH,)},
    "E12-L2b": {"max_gpus": 4, "model_dirs": (_GLM_53_FLASH,)},
    "E14": {"max_gpus": 4, "model_dirs": (_GLM_53_FLASH,)},
}
MODEL_ASSERTION_KEYS = ("experiment_model_whitelisted", "gpus_within_design",
                        "model_mounts_registered_ro", "other_mounts_no_model",
                        "tokenizer_no_weights", "basis_present")


class PolicyError(ConfigError):
    pass


def _norm(p):
    return os.path.normpath(p).rstrip("/") or "/"


def paths_related(a, b):
    """True if a == b, a is under b, or a is an ancestor of b (a mount of '/home' exposes b)."""
    a, b = _norm(a), _norm(b)
    if a == b:
        return True
    if a == "/" or b == "/":
        return True
    return a.startswith(b + "/") or b.startswith(a + "/")


def library_roots():
    """Every model library path to compare against: the literal and the realpath of each."""
    return sorted({x for f in FORBIDDEN_MOUNT_PATHS for x in (f, os.path.realpath(f))})


def _no_separator(mount, paths):
    """pyxis splits --container-mounts at ',' and each entry at ':': a mount string with a ','
    anywhere, or a path holding either, would name other mounts than the ones checked here."""
    if isinstance(mount, str) and "," in mount:
        raise PolicyError("mount contains ',': %r" % mount)
    for p in paths:
        if isinstance(p, str) and ("," in p or ":" in p):
            raise PolicyError("mount path contains ',' or ':': %r" % (mount,))


def mount_paths(mount):
    """Paths named by one mount entry: 'src:dst[:opts]' string or {'src':..,'dst':..} dict."""
    if isinstance(mount, dict):
        out = [mount.get("src"), mount.get("dst")]
    elif isinstance(mount, str):
        parts = mount.split(":")
        if len(parts) not in (2, 3):
            raise PolicyError("mount must be 'src:dst[:opts]': %r" % mount)
        out = parts[:2]
    else:
        raise PolicyError("mount entry must be a string or object: %r" % (mount,))
    _no_separator(mount, out)
    return [p for p in out if p]


def check_mounts(mounts):
    """Raise PolicyError if any mount equals, is under, or is an ancestor of a model library.

    Both the literal path and its realpath are compared, against both the literal and the
    realpath of each library, so a symlink into (or out of) a model library cannot hide it."""
    if not isinstance(mounts, list):
        raise PolicyError("mounts must be a list")
    bad = []
    libs = library_roots()
    for m in mounts:
        for p in mount_paths(m):
            if not p.startswith("/"):
                raise PolicyError("mount path must be absolute: %r" % p)
            for q in sorted({p, os.path.realpath(p)}):
                for f in libs:
                    if paths_related(q, f):
                        bad.append((p, q, f))
    if bad:
        raise PolicyError("mounts expose a model library: %s" % bad)


def tokenizer_dir_sha256(path):
    """sha256 over sorted (relpath, file sha256) of every file; refuses symlinks and weights (by
    name suffix and by weight_content).

    Returns the string "none" when path is None (the job mounts no tokenizer copy).
    """
    if path is None:
        return "none"
    if not os.path.isdir(path) or os.path.islink(path):
        raise PolicyError("tokenizer dir %s is not a real directory" % path)
    entries = []
    for root, dirs, files in os.walk(path, followlinks=False):
        for d in dirs:
            if os.path.islink(os.path.join(root, d)):
                raise PolicyError("symlinked directory in tokenizer copy: %s" % d)
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path)
            if os.path.islink(full):
                raise PolicyError("symlink in tokenizer copy: %s" % rel)
            if name.lower().endswith(WEIGHT_SUFFIXES):
                raise PolicyError("weight file in tokenizer copy: %s" % rel)
            h = hashlib.sha256()
            with open(full, "rb") as f:
                head = f.read(16)
                if weight_content(head, os.fstat(f.fileno()).st_size):
                    raise PolicyError("weight content in tokenizer copy: %s" % rel)
                h.update(head)
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            entries.append((rel, h.hexdigest()))
    if not entries:
        raise PolicyError("tokenizer copy %s is empty" % path)
    entries.sort()
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def check_no_model_submission(experiment, gpus, mounts, tokenizer_dir, basis):
    """Run the five assertions in a fixed order; raise PolicyError on the first failure.

    Returns (assertions dict with every key True, tokenizer_dir_sha256) for the ledger row.
    """
    if experiment not in NO_MODEL_EXPERIMENTS:
        raise PolicyError("experiment %r is not in the no-model list %s"
                          % (experiment, NO_MODEL_EXPERIMENTS))
    if not isinstance(gpus, int) or isinstance(gpus, bool) or not 0 <= gpus <= MAX_GPUS:
        raise PolicyError("gpus must be an int in [0, %d], got %r" % (MAX_GPUS, gpus))
    check_mounts(mounts)
    tok_sha = tokenizer_dir_sha256(tokenizer_dir)
    if not isinstance(basis, str) or not basis.strip():
        raise PolicyError("the approval basis text is required")
    return {k: True for k in ASSERTION_KEYS}, tok_sha


def is_model_experiment(experiment):
    return isinstance(experiment, str) and experiment in MODEL_EXPERIMENTS


def max_gpus_for(experiment):
    """The per-job GPU ceiling: the whitelist entry's for a model experiment, else MAX_GPUS."""
    return MODEL_EXPERIMENTS[experiment]["max_gpus"] if is_model_experiment(experiment) \
        else MAX_GPUS


def check_model_mounts(experiment, mounts):
    """A model experiment's mounts: every mount that touches a model library is one of the
    experiment's registered model dirs, mounted at the same path, read-only (literal path and
    realpath both equal the registered one); at least one such mount; every other mount is
    checked like any job's (check_mounts). Returns the sorted model dirs mounted."""
    if not is_model_experiment(experiment):
        raise PolicyError("experiment %r is not in the model whitelist %s"
                          % (experiment, sorted(MODEL_EXPERIMENTS)))
    if not isinstance(mounts, list) or not mounts:
        raise PolicyError("mounts must be a non-empty list")
    reg = {(_norm(d), os.path.realpath(d)): _norm(d)
           for d in MODEL_EXPERIMENTS[experiment]["model_dirs"]}
    lib, other = [], []
    for m in mounts:
        src, dst, mode = _mount_parts(m)
        for p in (src, dst):
            if not p.startswith("/"):
                raise PolicyError("mount path must be absolute: %r" % p)
        if _touches_library(src) or _touches_library(dst):
            key = (_norm(src), os.path.realpath(src))
            if key not in reg or dst != src or mode != "ro":
                raise PolicyError("%s may only mount its registered model dirs %s at the same "
                                  "path read-only, got %r"
                                  % (experiment, sorted(reg.values()), m))
            lib.append(reg[key])
        else:
            other.append(m)
    if not lib:
        raise PolicyError("%s mounts none of its model dirs %s" % (experiment, sorted(reg.values())))
    if len(set(lib)) != len(lib):
        raise PolicyError("a model dir is mounted twice: %s" % lib)
    check_mounts(other)
    return sorted(lib)


def check_model_submission(experiment, gpus, mounts, tokenizer_dir, basis):
    """The assertions of a model experiment's job, in a fixed order; raise PolicyError on the
    first failure. Returns (assertions dict with every key True, tokenizer_dir_sha256)."""
    if not is_model_experiment(experiment):
        raise PolicyError("experiment %r is not in the model whitelist %s"
                          % (experiment, sorted(MODEL_EXPERIMENTS)))
    cap = MODEL_EXPERIMENTS[experiment]["max_gpus"]
    if not isinstance(gpus, int) or isinstance(gpus, bool) or not 1 <= gpus <= cap:
        raise PolicyError("%s: gpus must be an int in [1, %d], got %r" % (experiment, cap, gpus))
    check_model_mounts(experiment, mounts)
    tok_sha = tokenizer_dir_sha256(tokenizer_dir)
    if not isinstance(basis, str) or not basis.strip():
        raise PolicyError("the approval basis text is required")
    return {k: True for k in MODEL_ASSERTION_KEYS}, tok_sha


def check_any_submission(experiment, gpus, mounts, tokenizer_dir, basis):
    """check_model_submission for a whitelisted model experiment, else check_no_model_submission
    (which refuses any id outside NO_MODEL_EXPERIMENTS and any model-library mount)."""
    if is_model_experiment(experiment):
        return check_model_submission(experiment, gpus, mounts, tokenizer_dir, basis)
    return check_no_model_submission(experiment, gpus, mounts, tokenizer_dir, basis)


def check_test_only_argv(argv):
    """The probe sub-command may only run sbatch with --test-only."""
    if not isinstance(argv, list) or "sbatch" not in argv:
        raise PolicyError("probe argv must invoke sbatch: %r" % (argv,))
    if "--test-only" not in argv[argv.index("sbatch") + 1:]:
        raise PolicyError("probe argv must pass --test-only to sbatch: %r" % (argv,))


def _mount_parts(mount):
    """(src, dst, mode) of a 'src:dst[:ro|rw]' string (mode defaults to rw)."""
    if not isinstance(mount, str):
        raise PolicyError("prep_tokenizer mounts must be 'src:dst[:ro|rw]' strings: %r"
                          % (mount,))
    parts = mount.split(":")
    if len(parts) not in (2, 3) or not all(parts[:2]):
        raise PolicyError("bad mount %r" % mount)
    _no_separator(mount, parts[:2])
    mode = parts[2] if len(parts) == 3 else "rw"
    if mode not in ("ro", "rw"):
        raise PolicyError("prep_tokenizer mount mode must be ro or rw: %r" % mount)
    return parts[0], parts[1], mode


def _touches_library(path):
    return any(paths_related(q, f) for q in {path, os.path.realpath(path)}
               for f in library_roots())


def check_prep_tokenizer_mounts(mounts):
    """The tokenizer-copy job's mounts: exactly one mount touches a model library, and it is a
    registered model directory mounted at the same path, read-only (literal path and realpath
    both equal the registered one); every other mount is checked like any job's (check_mounts).
    """
    if not isinstance(mounts, list) or not mounts:
        raise PolicyError("prep_tokenizer mounts must be a non-empty list")
    lib, other = [], []
    for m in mounts:
        src, dst, mode = _mount_parts(m)
        for p in (src, dst):
            if not p.startswith("/"):
                raise PolicyError("mount path must be absolute: %r" % p)
        (lib if _touches_library(src) or _touches_library(dst) else other).append(
            (m, src, dst, mode))
    if len(lib) != 1:
        raise PolicyError("prep_tokenizer needs exactly one model-library mount, got %s"
                          % [x[0] for x in lib])
    m, src, dst, mode = lib[0]
    reg = {(_norm(d), os.path.realpath(d)) for d in PREP_TOKENIZER_MODEL_DIRS}
    if (_norm(src), os.path.realpath(src)) not in reg or dst != src or mode != "ro":
        raise PolicyError("prep_tokenizer may only mount a registered model dir %s at the same "
                          "path read-only, got %r" % (list(PREP_TOKENIZER_MODEL_DIRS), m))
    check_mounts([x[0] for x in other])


def check_prep_tokenizer_submission(gpus, mounts, basis):
    """Assertions of a prep_tokenizer plan, in a fixed order; raise PolicyError on the first
    failure. Returns the assertions dict (every key True) for the ledger row."""
    if gpus != 0 or isinstance(gpus, bool):
        raise PolicyError("prep_tokenizer requests no GPU (gpus must be 0), got %r" % (gpus,))
    check_prep_tokenizer_mounts(mounts)
    if not isinstance(basis, str) or not basis.strip():
        raise PolicyError("the approval basis text is required")
    return {k: True for k in PREP_ASSERTION_KEYS}
