# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake capload client: backlog.inflight from the tier table; exits after the last tier."""
import sys

from capfake_common import run

if __name__ == "__main__":
    sys.exit(run(port=False, workers=True))
