# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry point of ``python -m capload``."""

import sys

from .main import main

if __name__ == "__main__":  # spawned workers import this module as __mp_main__
    sys.exit(main())
