# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the modules and rules added in the INTERFACES.md r2 revision."""

import copy
import json
import os
import subprocess
import sys

import pytest

from capcommon import (expect, hwprofile, policy, runtree, scratchpath, snapshot, versions,
                       wire)
from capcommon.tiers import TierTable, TierTableError

from conftest import TOOLS, good_table_doc


# ---- expect -----------------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 7, 20, 21, 33, 40, 41])
@pytest.mark.parametrize("k,s", [(1, 1), (2, 1), (20, 20), (40, 20), (3, 1), (5, 5)])
@pytest.mark.parametrize("gate", expect.GATES)
def test_msgs_closed_form_matches_enumeration(n, k, s, gate):
    its = expect.msg_iterations(n, k, s, gate)
    assert expect.msgs_per_request(n, k, s, gate) == len(its)
    assert sum(expect.msg_tokens(n, k, s, gate)) == n
    assert its == sorted(its) and its[0] == 1 and its[-1] == expect.iterations(n, k, s)


def test_expect_known_answers():
    # design 2.7 C6 pass sample: s larger than the output length -> exactly 2 (first, last)
    assert expect.msgs_per_request(33, 40, 40, "product") == 2
    # one token: the finished iteration 1 is the only product message; S-FE sends 2 (design 2.1)
    assert expect.msgs_per_request(1, 1, 1, "product") == 1
    assert expect.msgs_per_request(1, 1, 1, "every_step_plus_final") == 2
    # 33 tokens, one token per step, s = 20: iterations 1, 20, 33
    assert expect.msg_iterations(33, 20, 20, "product") == [1, 20, 33]
    assert expect.msg_step_offsets(33, 20, 20, "product") == [0, 19, 32]
    assert expect.sse_frames_per_request(3, True) == 4
    assert expect.sse_frames_per_request(3, False) == 3
    with pytest.raises(expect.ExpectError):
        expect.msgs_per_request(10, 3, 2, "product")  # k % s != 0


def test_reconcile_validation_and_expected_counts():
    doc = {"schema": "capkit.reconcile/1", "chains": [
        {"name": "req", "kind": "requests", "gate": None, "include_usage": None,
         "hops": [{"role": "capload", "source": "summary.counts.ok", "scope": "tier"},
                  {"role": "nullsse", "source": "summary.counts.requests", "scope": "tier"}],
         "corrections": []},
        {"name": "sse", "kind": "chunks", "gate": "product", "include_usage": True,
         "hops": [{"role": "capload", "source": "summary.frames.frames", "scope": "tier"},
                  {"role": "nullsse", "source": "summary.counts.frames_sent", "scope": "tier"}],
         "corrections": [{"name": "usage frame is written by nullsse too", "per_request": 0,
                          "hop": "nullsse"}]}]}
    expect.validate_reconcile(doc, {"capload", "nullsse"})
    sse = doc["chains"][1]
    assert expect.expected_hop_count(sse, 10, 33, 20, 20, "capload") == 10 * 4
    bad = copy.deepcopy(doc)
    bad["chains"][0]["hops"][0]["role"] = "ghost"
    with pytest.raises(expect.ExpectError):
        expect.validate_reconcile(bad, {"capload", "nullsse"})
    bad = copy.deepcopy(doc)
    bad["chains"][1]["hops"][0]["source"] = "summary."
    with pytest.raises(expect.ExpectError):
        expect.validate_reconcile(bad, {"capload", "nullsse"})


# ---- wire -------------------------------------------------------------------------------------

def _stream(req_id, n, frame_bytes, usage=True):
    body = b"".join(wire.padded_content_frame(req_id, frame_bytes) for _ in range(n))
    if usage:
        body += wire.usage_frame(req_id, 5, n)
    return body + wire.DONE_FRAME


@pytest.mark.parametrize("split", [1, 2, 3, 7, 64, 10 ** 6])
def test_sse_counter_any_split(split):
    data = _stream(42, 5, 160)
    c = wire.SseCounter()
    total = 0
    for i in range(0, len(data), split):
        total += c.feed(data[i:i + split])
    assert total == c.frames == expect.sse_frames_per_request(5, True)
    assert c.done_seen and c.completion_tokens == 5 and c.leftover == 0


def test_sse_frames_and_grpc_header():
    assert len(wire.padded_content_frame(7, 200)) == 200
    with pytest.raises(wire.WireError):
        wire.padded_content_frame(7, 10)
    c = wire.SseCounter()
    c.feed(_stream(1, 3, 120, usage=False)[:-3])  # truncated: no [DONE]
    assert c.frames == 3 and not c.done_seen and c.completion_tokens is None
    buf = wire.pack_grpc_request(9, 33, 20, 20, 4, b"prompt")
    assert wire.unpack_grpc_request(buf) == (9, 33, 20, 20, 4, b"prompt")
    assert wire.http_headers(9, 1, 1, 3)[wire.HDR_TIER] == "3"


# ---- versions (C10) ---------------------------------------------------------------------------

def _rec(**kw):
    r = {"python": "3.12.3", "implementation": "cpython", "gc_enabled": True,
         "switchinterval_s": 0.005, "capkit_code_sha": "c" * 40, "loop_impl": None,
         "grpc": None, "protobuf": None, "protobuf_impl": None, "probe_tier": None,
         "probe_patch_sha": None}
    r.update(kw)
    return r


def test_check_versions_per_role():
    decls = {"capload": {"asyncio": True, "uses_grpc": False, "probe_tier": None,
                         "loop": "asyncio"},
             "sidecar": {"asyncio": False, "uses_grpc": False, "probe_tier": None, "loop": None},
             "relay1": {"asyncio": True, "uses_grpc": True, "probe_tier": None, "loop": "uvloop"}}
    grpc = dict(grpc="1.66.0", protobuf="5.27.0", protobuf_impl="upb")
    recs = [("capload", 1, _rec(loop_impl="asyncio.unix_events._UnixSelectorEventLoop")),
            ("sidecar", 2, _rec()),
            ("relay1", 3, _rec(loop_impl="uvloop.Loop", **grpc))]
    ok = versions.check_versions(recs, decls)
    assert ok["status"] == "PASS" and ok["expected_count"] == 3 == ok["checked_count"]
    # D11 positive control: one process forced to the pure-python protobuf -> NOT_RUN (not RED)
    bad = recs[:2] + [("relay1", 3, _rec(loop_impl="uvloop.Loop",
                                         **dict(grpc, protobuf_impl="python")))]
    bad.append(("relay1", 4, _rec(loop_impl="uvloop.Loop", **grpc)))
    decls_r = versions.check_versions(bad, decls)
    assert decls_r["status"] == "NOT_RUN" and any("protobuf_impl" in p
                                                   for p in decls_r["problems"])
    missing = versions.check_versions(recs + [("capload", 5, None)], decls)
    assert missing["status"] == "NOT_RUN" and missing["checked_count"] == 3
    wrong_loop = versions.check_versions([("capload", 1, _rec(loop_impl="uvloop.Loop"))], decls)
    assert wrong_loop["status"] == "NOT_RUN"
    assert versions.check_versions([], decls)["status"] == "NOT_RUN"


# ---- hwprofile --------------------------------------------------------------------------------

def _profile(**kw):
    p = {k: "v-" + k for k in hwprofile.IDENTITY_FIELDS}
    p.update({k: "o-" + k for k in hwprofile.OBSERVED_FIELDS})
    p.update(kw)
    return p


def test_hw_profile_sha_ignores_observed_fields():
    a = _profile()
    b = _profile(node_name="other", collected_at="later", cpu_freq_khz=[1], partition="p2")
    assert hwprofile.profile_sha256(a) == hwprofile.profile_sha256(b)
    c = _profile(microcode=hwprofile.unreadable("permission denied"))
    assert hwprofile.profile_sha256(a) != hwprofile.profile_sha256(c)
    assert hwprofile.diff_identity(a, c) == ["microcode"]
    with pytest.raises(hwprofile.HwProfileError):
        hwprofile.profile_sha256(dict(a, extra_key=1))
    d = dict(a)
    del d["partition"]
    with pytest.raises(hwprofile.HwProfileError):
        hwprofile.validate_profile(d)


# ---- policy -----------------------------------------------------------------------------------

def test_no_model_policy(tmp_path):
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}")
    (tok / "config.json").write_text("{}")
    ok, sha = policy.check_no_model_submission("E0", 0, ["/home/scratch.x/w:/w"], str(tok),
                                               "STATUS.md line 3")
    assert set(ok) == set(policy.ASSERTION_KEYS) and len(sha) == 64
    assert policy.tokenizer_dir_sha256(None) == "none"
    for exp, gpus, mounts, basis in [("E11", 0, [], "b"), ("E0", 2, [], "b"),
                                     ("E0", 0, ["/home:/home"], "b"),
                                     ("E0", 0, ["/home/scratch.trt_llm_data_ci/llm-models:/m"],
                                      "b"),
                                     ("E0", 0, [{"src": "/", "dst": "/host"}], "b"),
                                     ("E0", 0, [], " ")]:
        with pytest.raises(policy.PolicyError):
            policy.check_no_model_submission(exp, gpus, mounts, str(tok), basis)
    (tok / "model.safetensors").write_text("x")
    with pytest.raises(policy.PolicyError):
        policy.tokenizer_dir_sha256(str(tok))
    os.remove(tok / "model.safetensors")
    os.symlink("/etc/hostname", tok / "link.json")
    with pytest.raises(policy.PolicyError):
        policy.tokenizer_dir_sha256(str(tok))
    policy.check_test_only_argv(["ssh", "front", "sbatch", "--test-only", "x.sbatch"])
    with pytest.raises(policy.PolicyError):
        policy.check_test_only_argv(["sbatch", "--parsable", "x.sbatch"])


# ---- scratchpath ------------------------------------------------------------------------------

def test_scratch_path_guard(tmp_path):
    base = str(tmp_path)
    good = os.path.join(base, "u-123")
    os.mkdir(good)
    assert scratchpath.assert_job_scratch_path(good, base, "u", "123") == good
    for p in [base, os.path.join(base, "u-1234"), os.path.join(base, "x", "u-123"),
              os.path.join(base, "u-123", ".."), good + "/", "u-123"]:
        with pytest.raises(scratchpath.ScratchPathError):
            scratchpath.assert_job_scratch_path(p, base, "u", "123")
    with pytest.raises(scratchpath.ScratchPathError):
        scratchpath.assert_job_scratch_path(good, base, "u/..", "123")
    ver = tmp_path / "verify.json"
    ver.write_text(json.dumps({"passed": False}))
    env = dict(os.environ, PYTHONPATH=TOOLS)
    cmd = [sys.executable, "-m", "capcommon.scratchpath", "--path", good, "--base", base,
           "--user", "u", "--job-id", "123", "--verified-file", str(ver)]
    assert subprocess.run(cmd, env=env, capture_output=True).returncode == 3
    ver.write_text(json.dumps({"passed": True}))
    assert subprocess.run(cmd, env=env, capture_output=True).returncode == 0
    link = os.path.join(base, "v-1")
    os.symlink(good, link)
    with pytest.raises(scratchpath.ScratchPathError):
        scratchpath.assert_job_scratch_path(link, base, "v", "1")


# ---- snapshot slot declarations ----------------------------------------------------------------

def test_register_decls_eager():
    decls = {"count.F2.msgs": {"kind": "counter", "site": "F2", "group": "count",
                               "per": "message"},
             "lag.loop": {"kind": "hist", "site": None, "group": "lag", "per": "sample"},
             "backlog.inflight": {"kind": "gauge", "site": None, "group": "backlog",
                                  "per": "none"}}
    reg = snapshot.SlotRegistry()
    got = reg.register_decls(decls, groups={"count", "backlog"})
    assert set(got) == {"count.F2.msgs", "backlog.inflight"}
    assert reg.get("count.F2.msgs") == 0  # present before the first pass
    with pytest.raises(snapshot.SnapshotError):
        snapshot.validate_slot_decls({"count.F2.msgs": dict(decls["count.F2.msgs"],
                                                            group="lag")})
    with pytest.raises(snapshot.SnapshotError):
        snapshot.validate_slot_decls({"count.F2.msgs": {"kind": "counter"}})


# ---- tier table axis / load_value / request timeout --------------------------------------------

def test_axis_and_load_value(table_path):
    tt = TierTable.load(table_path())
    assert tt.axis["load_key"] == "rate_rps" and tt.request_timeout_ns == 30_000_000_000
    assert tt.load_order_key(2) == 200.0 and tt.load_order_key(1) is None
    for mutate in [lambda d: d["tiers"][2].__setitem__("load_value", 199.0),
                   lambda d: d["tiers"][1].__setitem__("load_value", 1.0),
                   lambda d: d["tiers"][2].__setitem__("load_value", None),
                   lambda d: d.__setitem__("request_timeout_ns", 0),
                   lambda d: d["axis"].__setitem__("load_key", "nominal_msgs_per_s"),
                   lambda d: d["axis"].__setitem__("order", "up"),
                   lambda d: d.pop("axis")]:
        doc = good_table_doc()
        mutate(doc)
        with pytest.raises(TierTableError):
            TierTable.load(table_path(doc, "bad.json"))
    doc = good_table_doc()
    doc["axis"] = {"name": "R", "load_key": "beat_ns", "unit": "ns", "order": "desc"}
    for t in doc["tiers"]:
        t["load_value"] = None if t["kind"] == "drain" else t["beat_ns"]
    tt = TierTable.load(table_path(doc, "beat.json"))
    assert tt.load_order_key(2) == -10_000_000


def test_runtree_job_and_shm_helpers(tmp_path):
    assert runtree.job_dir_name("5601234", 1) == "J5601234-r1"
    assert runtree.launch_shm_subdir("/dev/shm/u-1/snap", 0, 3) == "/dev/shm/u-1/snap/r0-L3"
    with pytest.raises(Exception):
        runtree.launch_shm_subdir("/x", -1, 0)
