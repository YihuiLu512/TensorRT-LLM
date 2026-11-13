# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submission policy for jobs that must not load a real model (INTERFACES.md 9.5; plan
CONSTRAINTS section 2 "mechanisms" 1, 2, 5, 6).

These are mechanisms, not judgement calls: the submit entry calls check_no_model_submission()
before writing a PLANNED ledger row, and ledger.submissions_append() refuses PLANNED rows whose
assertions_json does not show every assertion passed.
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

ASSERTION_KEYS = ("experiment_no_model", "gpus_le_1", "no_model_mounts", "tokenizer_no_weights",
                  "basis_present")


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


def mount_paths(mount):
    """Paths named by one mount entry: 'src:dst[:opts]' string or {'src':..,'dst':..} dict."""
    if isinstance(mount, dict):
        out = [mount.get("src"), mount.get("dst")]
    elif isinstance(mount, str):
        parts = mount.split(":")
        out = parts[:2]
    else:
        raise PolicyError("mount entry must be a string or object: %r" % (mount,))
    return [p for p in out if p]


def check_mounts(mounts):
    """Raise PolicyError if any mount equals, is under, or is an ancestor of a model library.

    Both the literal path and its realpath are compared, against both the literal and the
    realpath of each library, so a symlink into (or out of) a model library cannot hide it."""
    if not isinstance(mounts, list):
        raise PolicyError("mounts must be a list")
    bad = []
    libs = sorted({x for f in FORBIDDEN_MOUNT_PATHS for x in (f, os.path.realpath(f))})
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
    """sha256 over sorted (relpath, file sha256) of every file; refuses symlinks and weights.

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


def check_test_only_argv(argv):
    """The probe sub-command may only run sbatch with --test-only."""
    if not isinstance(argv, list) or "sbatch" not in argv:
        raise PolicyError("probe argv must invoke sbatch: %r" % (argv,))
    if "--test-only" not in argv[argv.index("sbatch") + 1:]:
        raise PolicyError("probe argv must pass --test-only to sbatch: %r" % (argv,))
