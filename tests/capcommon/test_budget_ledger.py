# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BUDGET.tsv rows and the budget count (calib-v2 DESIGN-r7a 3.2, P2-08b)."""

import os

import pytest

from capcommon import ledger
from capcommon.exitcodes import ConfigError

ROW = {"event": "RESERVED", "budget_id": "E0v2-V0cal", "item_id": "V0cal-0001",
       "item_class": "planned", "plan_sha256": "a" * 64, "register_sha256": "b" * 64,
       "plan_id": "P1", "job_id": ""}


def test_count_is_distinct_reserved_items_not_rows(tmp_path):
    p = str(tmp_path / "BUDGET.tsv")
    ledger.budget_append(p, ROW)
    ledger.budget_append(p, dict(ROW, event="JOB_ID", job_id="5601"))
    ledger.budget_append(p, dict(ROW, item_id="V0cal-0002", plan_id="P2"))
    ledger.budget_append(p, dict(ROW, item_id="V0cal-0003", plan_id="P3", budget_id="E0v2-V0"))
    assert ledger.budget_count(p, "E0v2-V0cal") == 2
    assert ledger.budget_count(p, "E0v2-V0") == 1 and ledger.budget_count(p, "other") == 0
    assert ledger.budget_reserved_items(p) == {"V0cal-0001", "V0cal-0002", "V0cal-0003"}
    with open(p) as f:
        assert f.readline().rstrip("\n").split("\t") == list(ledger.BUDGET_COLUMNS)
    assert len(ledger.budget_rows(p)) == 4


@pytest.mark.parametrize("change", [
    {"event": "REPLACED"}, {"job_id": "5601"}, {"event": "JOB_ID"}, {"item_class": "replace"},
    {"plan_sha256": "x"}, {"plan_id": ""}, {"budget_id": "a b"}, {"item_id": ""}])
def test_bad_rows_are_refused(tmp_path, change):
    p = str(tmp_path / "BUDGET.tsv")
    with pytest.raises(ConfigError):  # LedgerError, or JobPlanError for the item class
        ledger.budget_append(p, dict(ROW, **change))
    assert not os.path.exists(p)
