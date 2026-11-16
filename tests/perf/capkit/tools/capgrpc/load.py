# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`python -m capgrpc.load` = the gRPC load generator / stamped drain (INTERFACES.md section 11).

The implementation lives in capgrpc.loader so that multiprocessing children import it by a
stable module name instead of re-running this entry module as __mp_main__.
"""

import sys

from capgrpc.loader import main

if __name__ == "__main__":
    sys.exit(main())
