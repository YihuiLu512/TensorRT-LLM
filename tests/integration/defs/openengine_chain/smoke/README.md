<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Regression smoke set

Each merged work item adds `WT-xx.txt` here: test-db entries (the syntax of
`../lists/case02_full.txt`) for its representative tests, at most 3 minutes on
1xB200 in total. `runner.py --smoke` runs the union; every merge keeps it
passing.
