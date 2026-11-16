# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Version self-report record (D11; INTERFACES.md section 2.5).

Never imports grpc/protobuf unless they are already importable; missing packages report null.
"""

import gc
import os
import platform
import sys

ENV_KEYS = ("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "PYTHONPATH", "PYTHONHASHSEED",
            "TRTLLM_SERVER_DISABLE_GC", "TRTLLM_WORKER_DISABLE_GC",
            "TLLM_NUMA_AWARE_WORKER_AFFINITY", "CAPKIT_CODE_SHA", "CAPPROBE_TIER",
            "CAPPROBE_GROUPS", "CAPPROBE_PATCH_SHA", "CAPPROBE_INJECT")
# C10 (D11). Which fields must be non-null depends on what the role declares in the scenario
# (INTERFACES.md 2.5 / 11): every process reports BASE_REQUIRED; asyncio processes also the loop;
# processes that use grpc/protobuf also those; processes running the probe overlay (P1/P2) also
# the probe patch sha. A field is never required just because some other role needs it.
BASE_REQUIRED = ("python", "implementation", "gc_enabled", "switchinterval_s", "capkit_code_sha")
ASYNCIO_REQUIRED = ("loop_impl",)
GRPC_REQUIRED = ("grpc", "protobuf", "protobuf_impl")
PROBED_REQUIRED = ("probe_tier", "probe_patch_sha")
# Must be identical across every process of a launch where present (null is "absent" here and is
# caught by the required-field rule instead). loop_impl is per role: it must equal the role's
# declared loop, but different roles may legitimately differ.
CONSISTENT_FIELDS = ("python", "implementation", "grpc", "protobuf", "protobuf_impl",
                     "capkit_code_sha", "probe_tier", "probe_patch_sha")
LOOP_IMPL_BY_DECL = {"asyncio": ("asyncio.",), "uvloop": ("uvloop.",)}


def _grpc_version():
    try:
        import grpc  # noqa: PLC0415
    except ImportError:
        return None
    return getattr(grpc, "__version__", None)


def _protobuf():
    try:
        from google.protobuf import __version__ as v  # noqa: PLC0415
    except ImportError:
        return None, None
    try:
        from google.protobuf.internal import api_implementation  # noqa: PLC0415
        impl = api_implementation.Type()
    except ImportError:
        impl = None
    return v, impl


def loop_impl_name(loop):
    if loop is None:
        return None
    t = type(loop)
    return "%s.%s" % (t.__module__, t.__qualname__)


def version_record(loop=None, extra=None, include_grpc=True):
    """Return the D11 record. loop: the running event loop (or None for non-asyncio processes)."""
    pb_v, pb_impl = _protobuf() if include_grpc else (None, None)
    groups = os.environ.get("CAPPROBE_GROUPS")
    rec = {
        "python": sys.version,
        "implementation": platform.python_implementation().lower(),
        "executable": sys.executable,
        "pid": os.getpid(),
        "grpc": _grpc_version() if include_grpc else None,
        "protobuf": pb_v,
        "protobuf_impl": pb_impl,
        "loop_impl": loop_impl_name(loop),
        "gc_enabled": gc.isenabled(),
        "switchinterval_s": sys.getswitchinterval(),
        "capkit_code_sha": os.environ.get("CAPKIT_CODE_SHA") or None,
        "probe_tier": os.environ.get("CAPPROBE_TIER") or None,
        "probe_patch_sha": os.environ.get("CAPPROBE_PATCH_SHA") or None,
        "probe_groups": [g for g in groups.split(",") if g] if groups else [],
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
    }
    if extra:
        for k, v in extra.items():
            if k in rec:
                raise ValueError("extra key %r would overwrite a standard field" % k)
            rec[k] = v
    return rec


def required_fields(decl):
    """Required non-null fields for one role declaration: {"asyncio": bool, "uses_grpc": bool,
    "probe_tier": "P0"|"P1"|"P2"|None} (probe_tier None = not a product process)."""
    req = list(BASE_REQUIRED)
    if decl["asyncio"]:
        req += ASYNCIO_REQUIRED
    if decl["uses_grpc"]:
        req += GRPC_REQUIRED
    if decl.get("probe_tier") in ("P1", "P2"):
        req += PROBED_REQUIRED
    elif decl.get("probe_tier") == "P0":
        req += ("probe_tier",)
    return tuple(req)


def missing_fields(rec, required):
    """Names of required fields that are null/absent (C10 'incomplete')."""
    return [k for k in required if rec.get(k) is None]


def check_versions(records, decls):
    """C10 over one launch.

    records: list of (role, pid, version record or None) for every process the launch EXPECTS
    (from launch.json expected_procs, not from what happened to report). decls: {role: {"asyncio",
    "uses_grpc", "probe_tier", "loop"}} from launch.json. Returns a dict with expected_count,
    checked_count, problems (list of strings) and status PASS / NOT_RUN. Per D11 an incomplete or
    inconsistent report makes the launch NOT_RUN (it is not RED).
    """
    problems = []
    checked = 0
    seen = {}
    for role, pid, rec in records:
        if rec is None:
            problems.append("%s pid %s: no version record" % (role, pid))
            continue
        checked += 1
        d = decls[role]
        miss = missing_fields(rec, required_fields(d))
        if miss:
            problems.append("%s pid %s: missing %s" % (role, pid, miss))
        if d["asyncio"]:
            want = LOOP_IMPL_BY_DECL.get(d.get("loop"))
            got = rec.get("loop_impl") or ""
            if want is None or not got.startswith(want):
                problems.append("%s pid %s: loop_impl %r does not match declared loop %r"
                                % (role, pid, got, d.get("loop")))
        for k in CONSISTENT_FIELDS:
            v = rec.get(k)
            if v is None:
                continue
            if k in seen and seen[k][0] != v:
                problems.append("%s differs: %r (%s) vs %r (%s pid %s)"
                                % (k, seen[k][0], seen[k][1], v, role, pid))
            seen.setdefault(k, (v, "%s pid %s" % (role, pid)))
    expected = len(records)
    if expected == 0 or checked == 0:
        status = "NOT_RUN"
    else:
        status = "NOT_RUN" if problems else "PASS"
    return {"expected_count": expected, "checked_count": checked, "problems": problems,
            "status": status}
