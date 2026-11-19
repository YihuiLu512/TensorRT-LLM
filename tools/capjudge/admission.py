# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission rejections: calls the engine host's gRPC core refuses before the application takes
them, counted as overload-type failures (category "admission_rejected").

When the OpenEngine host's event loop falls behind, grpc's server-side queue of calls the
application has not taken yet grows past grpc.server.max_pending_requests (soft limit 1000, hard
limit 3000 in grpcio 1.84) and the core refuses new calls. The client sees status CANCELLED,
details "CANCELLED", within milliseconds and without a message, and the request never reaches
rank 0. That is the overload outcome of admission. Everything else keeps the design 2.6 rule
(the class comes from the reason; unknown -> non-overload).

Per request, read from the role worker's requests capbin of the same tier (status ids mapped by
that file's own meta.status_codes, which differ between gRPC reader workers):
  * gRPC reader (GRPC_REASON): frames == 0;
  * HTTP loader behind a new-architecture frontend (HTTP_REASON): frames <= 1 (the frontend sends
    the 200 header and one error event) and tokens <= 0;
  * both: done - sent <= ADMISSION_MAX_NS (far below grpc's 30 s unrequested-time limit and the
    registered request timeouts, far above the 1.2 s worst case measured on E9 N-8f0).
Per launch, all of these must hold, otherwise no failure of the launch is reclassified:
  * never reached rank 0: S-NX rank 0 self-reports (raw/snx/snx.<pid>.summary.json) exist, none
    has a fatal error, and candidates <= sent - taken - unconnected (sent: the load generators'
    sent requests over every tier; taken: rank 0's taken requests; unconnected: the load
    generators' conn_refused / conn_fail failures, which never left the client either);
  * HTTP candidates only: the frontend logs (raw/logs/frontend*.out) report at least as many
    OpenEngine Generate failures with status CANCELLED and details "CANCELLED" as there are HTTP
    candidates, because the HTTP loader never sees the gRPC status.
A failure entry is split: at most its candidate count is admission_rejected, the rest keeps the
class the reason gives.
"""

import glob
import json
import os

NS = 1_000_000_000
ADMISSION_MAX_NS = 5 * NS
GRPC_REASON = "grpc_CANCELLED"
HTTP_REASON = "protocol_error"
CATEGORY = "admission_rejected"
REQUESTS_KIND = "loadgen.requests"
UNCONNECTED_REASONS = ("conn_refused", "conn_fail")
FE_LOG_GLOB = "frontend*.out"
FE_LOG_MARK = b"failed a request: OpenEngine Generate on "
FE_LOG_CANCELLED = b"failed with CANCELLED: CANCELLED"
_CHUNK = 64 << 20


def row_is_candidate(reason, frames, tokens, sent, done):
    """True when one failed request of `reason` has the admission signature."""
    if sent < 0 or done < 0 or done - sent > ADMISSION_MAX_NS:
        return False
    if reason == GRPC_REASON:
        return frames == 0
    if reason == HTTP_REASON:
        return frames <= 1 and tokens <= 0
    return False


def candidates_in_capbin(header, cols):
    """{reason: number of requests with the admission signature} of one requests capbin."""
    codes = (header.get("meta") or {}).get("status_codes") or {}
    want = {int(k): v for k, v in codes.items() if v in (GRPC_REASON, HTTP_REASON)}
    out = {}
    if not want:
        return out
    st, fr = cols.get("status", []), cols.get("frames", [])
    tk, se, dn = cols.get("tokens", []), cols.get("sent_rel_ns", []), cols.get("done_rel_ns", [])
    for i, s in enumerate(st):
        reason = want.get(s)
        if reason is not None and row_is_candidate(reason, fr[i], tk[i] if tk else -1, se[i],
                                                   dn[i]):
            out[reason] = out.get(reason, 0) + 1
    return out


def count_fe_cancelled(paths):
    """(CANCELLED/"CANCELLED" OpenEngine Generate failures, other OpenEngine Generate failures)
    reported in the frontend logs `paths`; one warning line per failed request."""
    cancelled = other = 0
    tail = b""
    for p in paths:
        with open(p, "rb") as f:
            while True:
                chunk = f.read(_CHUNK)
                if not chunk:
                    break
                buf = tail + chunk
                cut = buf.rfind(b"\n")
                if cut < 0:
                    tail = buf
                    continue
                body, tail = buf[:cut], buf[cut + 1:]
                c, o = _count_lines(body)
                cancelled += c
                other += o
        if tail:
            c, o = _count_lines(tail)
            cancelled, other, tail = cancelled + c, other + o, b""
    return cancelled, other


def _count_lines(body):
    c = o = 0
    i = body.find(FE_LOG_MARK)
    while i >= 0:
        end = body.find(b"\n", i)
        line = body[i:] if end < 0 else body[i:end]
        if line.rstrip(b"\r").endswith(FE_LOG_CANCELLED):
            c += 1
        else:
            o += 1
        i = body.find(FE_LOG_MARK, i + len(FE_LOG_MARK))
    return c, o


def launch_gate(cand, sent, taken, rank0_ok, fe_cancelled, unconnected=0):
    """(admitted?, reason text) for a launch: cand = {reason: launch total}; taken None = no rank 0
    self-report; fe_cancelled None = no frontend log."""
    n = sum(cand.values())
    if n == 0:
        return False, "no admission candidates"
    if taken is None or not rank0_ok:
        return False, "rank 0 self-report missing or fatal: cannot show the candidates never " \
                      "reached rank 0"
    if n > sent - taken - unconnected:
        return False, "candidates %d > sent %d - rank 0 taken %d - unconnected %d" % (
            n, sent, taken, unconnected)
    h = cand.get(HTTP_REASON, 0)
    if h and (fe_cancelled is None or fe_cancelled < h):
        return False, "HTTP candidates %d > frontend-log CANCELLED failures %s" % (h, fe_cancelled)
    return True, "candidates %d <= sent %d - taken %d - unconnected %d%s" % (
        n, sent, taken, unconnected, "; frontend-log CANCELLED %d" % fe_cancelled if h else "")


class Admission:
    """Launch-level admission evidence, computed on first use (trees without candidate reasons
    never read the rank 0 reports or the frontend logs)."""

    def __init__(self, raw_dir, tds, note_file=None):
        self.raw = raw_dir
        self.tds = tds
        self._note = note_file or (lambda p: None)
        self._done = False
        self.per = {}  # (tier seq, role, worker) -> {reason: candidates}
        self.admitted = False
        self.detail = {}

    def _compute(self):
        self._done = True
        cand, sent, unconnected, loaders = {}, 0, 0, set()
        for seq, td in self.tds.items():
            for ident, header, cols in td.capbins.get(REQUESTS_KIND, []):
                role, worker = ident.get("role"), ident.get("worker")
                loaders.add(role)
                got = candidates_in_capbin(header, cols)
                if got:
                    self.per[(seq, role, worker)] = got
                    for r, n in got.items():
                        cand[r] = cand.get(r, 0) + n
        for td in self.tds.values():
            for role in loaders:
                for doc in td.summaries.get(role, {}).values():
                    sent += int((doc.get("counts") or {}).get("sent", 0))
                    unconnected += sum(int(f.get("count", 0)) for f in doc.get("failures") or []
                                       if f.get("reason") in UNCONNECTED_REASONS)
        taken, rank0_ok = self._rank0()
        fe = None
        if cand.get(HTTP_REASON):
            paths = sorted(glob.glob(os.path.join(self.raw, "logs", FE_LOG_GLOB)))
            if paths:
                fe, other = count_fe_cancelled(paths)
                for p in paths:
                    self._note(p)
                self.detail["fe_log_other_failures"] = other
        self.admitted, why = launch_gate(cand, sent, taken, rank0_ok, fe, unconnected)
        self.detail.update({"candidates": cand, "sent": sent, "unconnected": unconnected,
                            "rank0_taken": taken,
                            "fe_log_cancelled": fe, "admitted": self.admitted, "why": why})

    def _rank0(self):
        paths = sorted(glob.glob(os.path.join(self.raw, "snx", "snx.*.summary.json")))
        if not paths:
            return None, False
        taken, ok = 0, True
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                doc = json.load(f)
            self._note(p)
            doc = doc.get("payload", doc)
            if doc.get("fatal_error"):
                ok = False
            taken += int((doc.get("counts") or {}).get("taken", 0))
        return taken, ok

    def admitted_count(self, seq, role, worker, failure):
        """How many of `failure` (a summary entry of that tier / role / worker) are admission
        rejections."""
        reason = failure.get("reason")
        if reason not in (GRPC_REASON, HTTP_REASON):
            return 0
        if not self._done:
            self._compute()
        if not self.admitted:
            return 0
        got = self.per.get((seq, role, worker), {}).get(reason, 0)
        return min(got, int(failure.get("count", 0)))


def for_judge(ctx, tds):
    """The Admission of a judge context (ctx.tree.raw) over its tier data."""
    return Admission(ctx.tree.raw, tds, getattr(ctx, "note_file", None))
