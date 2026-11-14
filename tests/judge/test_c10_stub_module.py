# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""C10's S-PB stub-module item (errata B2B-11, D16): an S-PB role started with --sfe-module must
report the sha256 of the overlay manifest's stub module "sfe"; one without it reports no
control. Fake launch data; the version-record half of C10 has its own samples."""

import types

from conftest import judge_mod

C = judge_mod("checks")
SHA = "a" * 64


class LD:
    def __init__(self, overlay, ready, records=None):
        self.overlay = overlay
        self._ready = ready
        self._records = records or {}

    def ready_doc(self, role):
        return self._ready.get(role)

    def version_records(self):
        return self._records, []

    def procs_of_role(self, role):
        return [{"pid": pid} for pid, rec in self._records.items() if rec["role"] == role]


def _ctx(args):
    return types.SimpleNamespace(roles={
        "spb": {"role": "spb", "module": C.SPB_MODULE, "args": args, "asyncio": False,
                "uses_grpc": False, "probe_tier": None, "loop": None, "expected_procs": 1},
        "other": {"role": "other", "module": "capgrpc.load", "args": ["--sfe-module", "x"],
                  "asyncio": False, "uses_grpc": False, "probe_tier": None, "loop": None,
                  "expected_procs": 1}})


OVERLAY = {"stub_modules": [{"key": "sfe", "sha256": SHA}]}
FRONT = ["--prebaked", "/p", "--sfe-module", "/snap/x/fake_openengine.py"]


def _ready(control):
    return {"spb": {"info": {"control": control}}}


def test_matching_sha_is_checked():
    out = C.stub_module_problems(_ctx(FRONT), LD(OVERLAY, _ready({"sha256": SHA})))
    assert out == {"expected": 1, "checked": 1, "problems": []}  # only the S-PB role counts


def test_generate_only_mode_reports_no_control():
    out = C.stub_module_problems(_ctx(["--prebaked", "/p"]), LD({}, _ready(None)))
    assert out == {"expected": 1, "checked": 1, "problems": []}
    out = C.stub_module_problems(_ctx(["--prebaked", "/p"]), LD({}, _ready({"sha256": SHA})))
    assert out["checked"] == 0 and "without --sfe-module" in out["problems"][0]


def test_mismatch_missing_manifest_entry_and_missing_ready_file():
    out = C.stub_module_problems(_ctx(FRONT), LD(OVERLAY, _ready({"sha256": "b" * 64})))
    assert out["checked"] == 0 and "!= overlay stub module sfe" in out["problems"][0]
    out = C.stub_module_problems(_ctx(FRONT), LD({"stub_modules": []},
                                                 _ready({"sha256": SHA})))
    assert "lists no stub module sfe" in out["problems"][0]
    out = C.stub_module_problems(_ctx(FRONT), LD(None, _ready({"sha256": SHA})))
    assert "lists no stub module sfe" in out["problems"][0]
    out = C.stub_module_problems(_ctx(FRONT), LD(OVERLAY, {}))
    assert out["problems"] == ["role spb: no ready file"]
    out = C.stub_module_problems(_ctx(["--sfe-module=/x.py"]), LD(OVERLAY, _ready(None)))
    assert "!= overlay" in out["problems"][0]  # the = form is frontend mode too


def _records():
    from capcommon.versions import version_record  # noqa: PLC0415
    rec = version_record(include_grpc=False)
    rec["capkit_code_sha"] = "c" * 40
    return {1: dict(rec, role="spb", pid=1), 2: dict(rec, role="other", pid=2)}


def test_c10_is_not_run_on_a_stub_problem_and_passes_otherwise():
    ok = C.check_c10(_ctx(FRONT), LD(OVERLAY, _ready({"sha256": SHA}), _records()))
    assert ok.status == "PASS" and ok.expected == ok.checked == 3
    bad = C.check_c10(_ctx(FRONT), LD(OVERLAY, _ready({"sha256": "b" * 64}), _records()))
    assert bad.status == "NOT_RUN" and bad.expected == 3 and bad.checked == 2
    assert any("!= overlay stub module sfe" in p for p in bad.not_run)
