# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake nullsse server: port file, workers, backlog.active_streams."""
import sys

from capfake_common import run

if __name__ == "__main__":
    sys.exit(run(port=True, workers=True))
