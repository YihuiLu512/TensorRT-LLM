# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capjudge.admission: admission rejections (CANCELLED, no message, milliseconds, never reached
rank 0) count as overload-type; real cancels, timeouts and slow or streamed failures do not."""

import json
import os
import types

from conftest import judge_mod

A = judge_mod("admission")
S = judge_mod("symptoms")
MS = 1_000_000


class _Tier:
    def __init__(self, seq):
        self.seq = seq


class _Td:
    """The parts of a TierData failures_by_class and Admission read."""

    def __init__(self, seq, role, worker, failures, codes, rows, sent):
        self.tier = _Tier(seq)
        self.summaries = {role: {worker: {"counts": {"sent": sent}, "failures": failures}}}
        ident = {"role": role, "worker": worker}
        self.capbins = {A.REQUESTS_KIND: [(ident, {"meta": {"status_codes": codes}}, rows)]}


def _rows(spec):
    """spec: [(status id, frames, tokens, duration ns)] -> requests columns."""
    cols = {k: [] for k in ("status", "frames", "tokens", "sent_rel_ns", "done_rel_ns")}
    for st, fr, tk, d in spec:
        cols["status"].append(st)
        cols["frames"].append(fr)
        cols["tokens"].append(tk)
        cols["sent_rel_ns"].append(10 * MS)
        cols["done_rel_ns"].append(10 * MS + d)
    return cols


def _raw(tmp_path, taken=None, fatal=None, fe_lines=None):
    raw = tmp_path / "raw"
    (raw / "snx").mkdir(parents=True)
    (raw / "logs").mkdir()
    if taken is not None:
        doc = {"schema": "capkit.snx_summary/1", "counts": {"taken": taken},
               "fatal_error": fatal}
        (raw / "snx" / "snx.11.summary.json").write_text(json.dumps(doc))
    if fe_lines is not None:
        (raw / "logs" / "frontend.w0.7.out").write_text("".join(x + "\n" for x in fe_lines))
    return str(raw)


FE_CANC = ("[W] [serve] OpenEngine engine 'tensorrt_llm' failed a request: OpenEngine Generate on "
           "127.0.0.1:5 failed with CANCELLED: CANCELLED")
FE_UNAV = ("[W] [serve] OpenEngine engine 'tensorrt_llm' failed a request: OpenEngine Generate on "
           "127.0.0.1:5 failed with UNAVAILABLE: Socket closed")


def _grpc_td(n_fail, spec, sent=100, reason="grpc_CANCELLED", hint="unclassified"):
    codes = {"1": reason}
    return _Td(3, "reader", 0, [{"reason": reason, "count": n_fail, "class_hint": hint}],
               codes, _rows(spec), sent)


def _classify(td, raw):
    adm = A.Admission(raw, {td.tier.seq: td})
    return S.failures_by_class(td, {"reader": {"measure_class": "under_test"},
                                    "capload": {"measure_class": "load_source"}}, adm), adm


# ---- per request ------------------------------------------------------------------------------

def test_row_signature():
    assert A.row_is_candidate("grpc_CANCELLED", 0, -1, 0, 2 * MS)
    assert not A.row_is_candidate("grpc_CANCELLED", 1, -1, 0, 2 * MS)  # had a message
    assert not A.row_is_candidate("grpc_CANCELLED", 0, -1, 0, 30_200 * MS)  # 30 s limit path
    assert A.row_is_candidate("protocol_error", 1, 0, 0, 300 * MS)
    assert not A.row_is_candidate("protocol_error", 1, 4, 0, 300 * MS)  # streamed tokens
    assert not A.row_is_candidate("timeout_registered", 0, -1, 0, 2 * MS)
    assert not A.row_is_candidate("client_cancelled", 0, -1, 0, 2 * MS)
    assert not A.row_is_candidate("grpc_CANCELLED", 0, -1, -1, 2 * MS)  # never sent


def test_capbin_status_codes_are_per_file():
    hdr = {"meta": {"status_codes": {"2": "grpc_CANCELLED", "1": "timeout_registered"}}}
    cols = _rows([(2, 0, -1, MS), (1, 0, -1, MS), (2, 3, -1, MS), (0, 33, -1, MS)])
    assert A.candidates_in_capbin(hdr, cols) == {"grpc_CANCELLED": 1}


# ---- known admission rejection (positive control) ---------------------------------------------

def test_e8_admission_rejection_is_overload(tmp_path):
    """E8 J6021612 shape: CANCELLED, 0 messages, p99 1 ms; rank 0 took only the ok requests."""
    td = _grpc_td(5, [(1, 0, -1, MS)] * 5 + [(0, 33, -1, 50 * MS)] * 95)
    f, adm = _classify(td, _raw(tmp_path, taken=95))
    assert f["overload"] == 5 and f["non_overload"] == 0, f
    assert f["by_category"] == {"admission_rejected": 5}
    assert adm.detail["admitted"] is True
    assert f["by_reason"] == {"reader:grpc_CANCELLED": 5}


def test_e9_http_admission_needs_frontend_log_evidence(tmp_path):
    """E9 N-8f0 shape: capload protocol_error, 1 frame, 0 tokens, sub-second; the frontend log
    has one CANCELLED/"CANCELLED" OpenEngine failure per request."""
    def td():
        return _Td(3, "capload", 0, [{"reason": "protocol_error", "count": 3,
                                      "class_hint": "non_overload"}],
                   {"5": "protocol_error", "0": "ok"},
                   _rows([(5, 1, 0, 3 * MS)] * 3 + [(0, 34, 33, 200 * MS)] * 7), 10)
    ok, _ = _classify(td(), _raw(tmp_path / "a", taken=7, fe_lines=[FE_CANC] * 3 + [FE_UNAV]))
    assert ok["overload"] == 3 and ok["non_overload"] == 0, ok
    # negative: the frontend says UNAVAILABLE (a broken connection, not admission)
    bad, adm = _classify(td(), _raw(tmp_path / "b", taken=7, fe_lines=[FE_CANC, FE_UNAV] * 2))
    assert bad["overload"] == 0 and bad["non_overload"] == 3, bad
    assert "frontend-log" in adm.detail["why"]
    # negative: old architecture, no OpenEngine lines at all
    none_, _ = _classify(td(), _raw(tmp_path / "c", taken=7, fe_lines=["INFO: 200 OK"]))
    assert none_["overload"] == 0 and none_["non_overload"] == 3


# ---- known non-overload failures stay non-overload (negative controls) ------------------------

def test_true_cancel_and_streamed_failure_stay_non_overload(tmp_path):
    # the reader's own cancel (new name) and a CANCELLED after messages were received
    td = _Td(3, "reader", 0, [{"reason": "client_cancelled", "count": 2,
                               "class_hint": "non_overload"},
                              {"reason": "grpc_CANCELLED", "count": 2,
                               "class_hint": "unclassified"}],
             {"1": "client_cancelled", "2": "grpc_CANCELLED"},
             _rows([(1, 0, -1, MS), (1, 0, -1, MS), (2, 5, -1, MS), (2, 9, -1, MS)]), 10)
    f, adm = _classify(td, _raw(tmp_path, taken=6))
    assert f["overload"] == 0 and f["non_overload"] == 4, f
    assert adm.detail["why"] == "no admission candidates"


def test_timeout_keeps_registered_overload_and_slow_cancel_is_not_admission(tmp_path):
    td = _Td(3, "reader", 0, [{"reason": "timeout_registered", "count": 1,
                               "class_hint": "overload"},
                              {"reason": "grpc_CANCELLED", "count": 1,
                               "class_hint": "unclassified"}],
             {"1": "timeout_registered", "2": "grpc_CANCELLED"},
             _rows([(1, 0, -1, 30_000 * MS), (2, 0, -1, 30_200 * MS)]), 10)
    f, _ = _classify(td, _raw(tmp_path, taken=8))
    assert f["overload"] == 1 and f["non_overload"] == 1, f
    assert f["by_category"] == {"registered_overload": 1, "unclassified": 1}


def test_launch_gate_negatives(tmp_path):
    spec = [(1, 0, -1, MS)] * 4
    # rank 0 took more than sent minus candidates: some CANCELLED reached rank 0
    f, adm = _classify(_grpc_td(4, spec, sent=10), _raw(tmp_path / "a", taken=7))
    assert f["overload"] == 0 and "rank 0 taken" in adm.detail["why"]
    # no rank 0 self-report / a fatal one
    f, _ = _classify(_grpc_td(4, spec, sent=10), _raw(tmp_path / "b"))
    assert f["overload"] == 0 and f["non_overload"] == 4
    f, _ = _classify(_grpc_td(4, spec, sent=10), _raw(tmp_path / "c", taken=6, fatal="Boom()"))
    assert f["overload"] == 0


def test_partial_entry_is_split(tmp_path):
    td = _grpc_td(5, [(1, 0, -1, MS)] * 3 + [(1, 2, -1, MS)] * 2)
    f, _ = _classify(td, _raw(tmp_path, taken=95))
    assert f["overload"] == 3 and f["non_overload"] == 2, f
    assert f["by_category"] == {"admission_rejected": 3, "unclassified": 2}


def test_without_admission_object_old_rule(tmp_path):
    td = _grpc_td(5, [(1, 0, -1, MS)] * 5)
    f = S.failures_by_class(td, {"reader": {"measure_class": "under_test"}})
    assert f["overload"] == 0 and f["non_overload"] == 5


def test_no_candidate_reason_never_reads_evidence(tmp_path):
    td = _Td(3, "reader", 0, [{"reason": "timeout_registered", "count": 2,
                               "class_hint": "overload"}], {"1": "timeout_registered"},
             _rows([(1, 0, -1, MS)] * 2), 4)
    adm = A.Admission(str(tmp_path / "missing"), {3: td})
    f = S.failures_by_class(td, None, adm)
    assert f["overload"] == 2 and adm.detail == {} and not adm._done


def test_frontend_log_count_across_chunk_boundaries(tmp_path, monkeypatch):
    p = tmp_path / "frontend.w0.1.out"
    p.write_text("".join(x + "\n" for x in ["x" * 37, FE_CANC, FE_UNAV, FE_CANC] * 40))
    monkeypatch.setattr(A, "_CHUNK", 97)
    assert A.count_fe_cancelled([str(p)]) == (80, 40)
    monkeypatch.setattr(A, "_CHUNK", 1 << 20)
    assert A.count_fe_cancelled([str(p)]) == (80, 40)
    q = tmp_path / "frontend.w1.2.out"
    q.write_text(FE_CANC)  # no trailing newline
    assert A.count_fe_cancelled([str(p), str(q)]) == (81, 40)


def test_for_judge_uses_tree_raw(tmp_path):
    ctx = types.SimpleNamespace(tree=types.SimpleNamespace(raw=os.fspath(tmp_path)),
                                note_file=lambda p: None)
    assert A.for_judge(ctx, {}).raw == os.fspath(tmp_path)
