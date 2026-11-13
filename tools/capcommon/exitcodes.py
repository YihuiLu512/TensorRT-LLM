# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exit codes shared by capkit tools (INTERFACES.md section 2.4) and the judge (section 10.3)."""

# Tools
OK = 0
INTERNAL = 1  # uncaught exception (Python default)
USAGE = 2  # argparse default
CONFIG = 3
RUNTIME = 4
STOPPED_EARLY = 5
ENV_INVALID = 6

# Judge (distinct range so the two tables never overlap)
JUDGE_PASS = 0
JUDGE_USAGE = 2
JUDGE_RED = 10
JUDGE_NOT_RUN = 11
JUDGE_ALREADY_JUDGED = 12
JUDGE_INPUT_INVALID = 13

TOOL_CODES = {
    "OK": OK,
    "USAGE": USAGE,
    "CONFIG": CONFIG,
    "RUNTIME": RUNTIME,
    "STOPPED_EARLY": STOPPED_EARLY,
    "ENV_INVALID": ENV_INVALID,
}

JUDGE_CODES = {
    "PASS": JUDGE_PASS,
    "USAGE": JUDGE_USAGE,
    "RED": JUDGE_RED,
    "NOT_RUN": JUDGE_NOT_RUN,
    "ALREADY_JUDGED": JUDGE_ALREADY_JUDGED,
    "INPUT_INVALID": JUDGE_INPUT_INVALID,
}


class CapkitError(Exception):
    """Base error carrying the exit code a tool should use when it surfaces at top level."""

    exit_code = RUNTIME


class ConfigError(CapkitError):
    """Invalid configuration or input file (exit code CONFIG)."""

    exit_code = CONFIG
