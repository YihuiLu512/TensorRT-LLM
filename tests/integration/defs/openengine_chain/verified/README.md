<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# E2E verification records

`WT-xx.yaml` records the end-to-end verifications work item WT-xx performed on
the chain. Each entry names the feature row it verifies by `row`: one feature can
have a case-01 row and a case-02 row, so feature ids alone do not tell them apart.

```yaml
entries:
  - row: F-33+F-34
    features: [F-33, F-34]
    cases: [kv_cache/test_prefix_aware_scheduling.py::...]
    commit: <full sha of the code the runs tested>
    junit: [<junit file, or directory of junit files, of each run>]
    result: passed
    chain_proof: incomplete
```

`result` and `chain_proof` are about the cases listed in `cases`; the junit
files can hold other tests too. `chain_proof` is `ok` when the `chain_proof`
junit property of every run is `ok`, and `incomplete` otherwise. Cases a row
lists that belong to another row instead go under `moved`, one item per row they
belong to; the entry does not verify them:

```yaml
    moved:
      - to_row: F-17+F-18+F-30
        why: <what they check that belongs to that row>
        cases: [unittest/llmapi/apps/_test_openai_chat.py::...]
```
