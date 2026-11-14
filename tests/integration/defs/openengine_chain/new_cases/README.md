<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# New tests

`<WT>.txt` lists the new tests (`NT-nnn` in the functional test plan) the work item `<WT>`
wrote: one per line, `<NT-nnn> <node id>` or the node id alone; `#` starts a comment. Node
ids are relative to `tests/`, except `openengine_chain/...` (relative to
`tests/integration/defs/`) and `$DS/...` (in the ds-engine repository), as in the plan.

`../check_case_lists.py --doc <plan>` checks the lists against the plan; with `--complete`
the lists must also hold every new test of the plan.
