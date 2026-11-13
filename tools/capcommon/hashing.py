# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sha256 helpers and canonical JSON (INTERFACES.md section 9.1)."""

import hashlib
import json
import re

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GITSHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CHUNK = 1 << 20


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json_bytes(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def canonical_json_sha256(obj):
    return sha256_bytes(canonical_json_bytes(obj))


def is_sha256_hex(s):
    return isinstance(s, str) and _SHA256_RE.match(s) is not None


def is_git_sha(s):
    return isinstance(s, str) and _GITSHA_RE.match(s) is not None


class ShaMismatch(Exception):
    pass


def verify_sha256_file(path, expected):
    """Raise ShaMismatch unless sha256(path) == expected. Returns the digest."""
    if not is_sha256_hex(expected):
        raise ShaMismatch("expected sha is not 64 lowercase hex: %r" % (expected,))
    got = sha256_file(path)
    if got != expected:
        raise ShaMismatch("%s: sha256 %s != registered %s" % (path, got, expected))
    return got
