# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m capjudge --tree <T> --workdir <W> (INTERFACES 10.1)."""

import sys

from .judge import main

if __name__ == "__main__":
    sys.exit(main())
