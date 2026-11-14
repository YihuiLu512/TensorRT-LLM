<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Migrated tests

`<WT>.txt` lists the aggregated tests the work item `<WT>` moved to the disaggregated entry
point unchanged (`**【搬迁】**` in the functional test plan): one per line,
`<T-n | XD-7nn> <node id>` or the node id alone; `#` starts a comment. A node id may add a
parameter id to the plan's `<file>::<test>`.

`../check_case_lists.py --doc <plan>` checks the lists against the plan; with `--complete`
the lists must also hold every migrated test of the plan.
