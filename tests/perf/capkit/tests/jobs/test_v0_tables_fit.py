# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The V0 s1-pc / s*-fix / c13-pc tables of caplaunch.laddergen (calib-v2 DESIGN-r7a 4.2;
INTERFACES erratum calib-v2-A7) fit `capjobs.submit --dry-run`: a launch plan with the
generated budget_s passes submit's time-budget check at exactly the time budget laddergen
reports, and one second less is refused. Nothing runs on a cluster."""

import json
import os
import types

import pytest
import test_submit as ts

from caplaunch import laddergen as L
from caplaunch import tiergen

TEMPLATE = os.path.join(ts.CODE_ROOT, "configs", "e0", "e0b-s1-single.tiers.template.json")
MS = 10 ** 6
BASE_VALUES = {"gate_poll_ns": 50 * MS, "gate_settle_polls": 3,
               "gate_deadline_guard_ns": 100 * MS, "gate_max_snapshot_age_ns": 200 * MS,
               "arrival_seed": 7, "request_timeout_ns": 60_000 * MS, "beat_ns": 10 * MS,
               "n_tokens": 32, "delta_ns": MS}
ARMS = ("nullsse-a", "sfe-a")


def _generate(tmp_path, kind, per_job, *extra):
    bv = tmp_path / "base.values.json"
    bv.write_text(json.dumps({"schema": tiergen.VALUES_SCHEMA, "values": BASE_VALUES}))
    a = L.build_parser().parse_args([
        "--kind", kind, "--phase", "V0", "--config", "s1@1ms", "--arm", "s1",
        "--name", "t-" + kind, "--base-template", TEMPLATE, "--base-values", str(bv),
        "--out-dir", str(tmp_path), "--launches-per-job", str(per_job), "--center-literal",
        "19.09", "--settle-s", "4"] + list(extra))
    return L.generate(a)[1]["packing"]


def _hms(s):
    return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def _dry_run(env, pack, time_s):
    n = pack["launches_per_job"]
    plan = dict(env.plan, kill_grace_s=pack["kill_grace_s"], margin_s=pack["margin_s"],
                launches=[dict(ln, budget_s=pack["budget_s"]) for ln in env.plan["launches"][:n]])
    lp = env.write_json("plan.%d.%d.json" % (n, time_s), plan)
    argv = env.submit_args(**{"--dry-run": None, "--launch-plan": lp, "--time": _hms(time_s),
                              "--arms": ",".join(ARMS[:n]), "--launches": str(n)})
    return ts.run(argv, types.SimpleNamespace(run=_no_ssh))


def _no_ssh(argv):
    raise AssertionError("a dry run of an ungated register calls no ssh: %s" % argv)


@pytest.mark.parametrize("per_job", [1, 2])
@pytest.mark.parametrize("kind,extra", [("pc", ("--request-timeout-ns", "null")),
                                        ("pc", ("--request-timeout-ns", str(60 * 10 ** 9))),
                                        ("c13pc", ())])
def test_v0_table_fits_submit_dry_run_at_its_time_budget(tmp_path, capsys, kind, extra, per_job):
    pack = _generate(tmp_path, kind, per_job, *extra)
    env = ts.Env(tmp_path / "env")
    assert pack["idle_window_s"] == ts.REG_ENV["idle_window_s"]
    assert pack["slurm_view_wait_s"] == ts.REG_ENV["slurm_view_wait_s"]
    tb = pack["time_budget_s"]
    assert _dry_run(env, pack, tb) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True and out["time_budget"]["total_s"] == tb
    assert _dry_run(env, pack, tb - 1) == 3
    assert "exceeds --time" in capsys.readouterr().err
    assert not os.path.exists(os.path.join(env.w, "submissions.tsv"))
