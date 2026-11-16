# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`python -m capgrpc.load` = the gRPC load generator / stamped drain."""

import sys

from capgrpc.loader import main

if __name__ == "__main__":
    sys.exit(main())
