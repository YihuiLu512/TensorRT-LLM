# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`python -m capgrpc.nullgrpc` = the gRPC null sink / beat source."""

import sys

from capgrpc.nullgrpc.server import main

if __name__ == "__main__":
    sys.exit(main())
