# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario template syntax shared by caplaunch (role argv, INTERFACES 2.2) and the job inputs
(capcommon.jobinputs): `{name}` / `{name:arg}` placeholders, and the `--param KEY=VALUE` values a
launch gives them."""

import re

from .exitcodes import ConfigError

PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)(?::([A-Za-z0-9_.-]+))?\}")


def placeholders(text):
    """[(name, arg)] of every {name[:arg]} placeholder in a string (arg None without one)."""
    return [(m.group(1), m.group(2)) for m in PLACEHOLDER_RE.finditer(text)]


def parse_params(pairs):
    """{KEY: VALUE} of "KEY=VALUE" strings (the values of --param); a string without "=" or with
    an empty KEY, or a KEY given twice, raises ConfigError."""
    out = {}
    for kv in pairs:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise ConfigError("--param must be KEY=VALUE, got %r" % kv)
        if k in out:
            raise ConfigError("--param %s given twice" % k)
        out[k] = v
    return out
