<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# E2E verification records

`WT-xx.yaml` records the end-to-end verifications work item WT-xx performed on
the chain, one entry per feature:

```yaml
- feature: F-33
  tests: [kv_cache/test_prefix_aware_scheduling.py::...]
  commit: <full sha the run tested>
  junit: <path to the run's junit files>
  result: passed
```
