# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import pytest

from capcommon import inject
from capcommon.snapshot import SlotRegistry


def test_parse_and_validate():
    specs = inject.parse_inject_specs("site=F2,mode=busy,delta_ns=1000;site=G3,mode=drop,every_n=4,phase=1")
    assert [s.site for s in specs] == ["F2", "G3"] and specs[1].every_n == 4
    assert inject.parse_inject_spec(specs[0].to_text()) == specs[0]
    assert inject.parse_inject_specs("") == [] and inject.parse_inject_specs(None) == []
    bad = ["site=F2,mode=busy", "site=F2,mode=busy,delta_ns=0", "site=F2,mode=nope",
           "site=F2,mode=drop,delta_ns=5", "site=F2,mode=miscount,miscount_delta=0",
           "site=F2,mode=sleep,delta_ns=1,where=aux_thread", "site=F2,mode=drop,every_n=2,phase=2",
           "site=F2,mode=drop,role=capload", "site=F 2,mode=drop", "site=F2,mode=busy,delta_ns=x",
           "site=F2,mode=drop,site=F3", "site=F2,mode=drop;site=F2,mode=drop", "mode=drop"]
    for text in bad:
        with pytest.raises(inject.InjectSpecError):
            inject.parse_inject_specs(text)


def test_selection_and_heartbeat():
    reg = SlotRegistry()
    inj = inject.Injector("recv.chunk", reg)
    for _ in range(4):
        assert inj.hit() == 0  # inactive: hits counted, nothing applied
    inj.set_spec(inject.InjectSpec("recv.chunk", "drop", every_n=3, phase=1))
    drops = [inj.should_drop() for _ in range(9)]
    # passes since activation h = 1..9; selected when (h-1) % 3 == 1 -> h = 2, 5, 8.
    # (Counting from process start, cumulative hits 5..13, would have selected h = 5, 8, 11,
    # i.e. positions 0, 3, 6 -- the activation-relative rule must not depend on earlier passes.)
    assert drops == [False, True, False, False, True, False, False, True, False]
    st = inj.stats()
    assert st["hits"] == 13 and st["applied"] == 3 and st["hits_since_activation"] == 9
    assert reg.get("inject.recv.chunk.hits") == 13
    with pytest.raises(inject.InjectSpecError):
        inj.hit()  # wrong call for mode
    assert inject.expected_applied(st["hits_since_activation"], 3, 1) == st["applied"]
    assert inject.expected_applied(0, 1, 0) == 0 and inject.expected_applied(1, 2, 1) == 0
    assert inject.expected_applied_delta(2, 9, 3, 1) == 2


def test_busy_sleep_miscount_short():
    inj = inject.Injector("x")
    inj.set_spec(inject.InjectSpec("x", "busy", delta_ns=2_000_000))
    t0 = time.monotonic_ns()
    spent = inj.hit()
    assert spent >= 2_000_000 and time.monotonic_ns() - t0 >= 2_000_000
    inj.set_spec(inject.InjectSpec("x", "sleep", delta_ns=2_000_000))
    assert inj.hit() >= 2_000_000
    st = inj.stats()
    assert st["applied"] == 2 and st["applied_ns"] >= 4_000_000
    inj.set_spec(inject.InjectSpec("x", "miscount", miscount_delta=-1, every_n=2))
    # hits continue from 2: h = 3, 4, 5, 6; selected when (h - 1) % 2 == 0
    assert [inj.count_delta() for _ in range(4)] == [0, 1, 0, 1]
    inj2 = inject.Injector("y")
    inj2.set_spec(inject.InjectSpec("y", "short", short_tokens=3))
    assert inj2.short_by() == 3
    inj2.set_spec(None)
    assert inj2.short_by() == 0


def test_aux_thread_busy():
    inj = inject.Injector("relay.msg")
    spec = inject.InjectSpec("relay.msg", "busy", delta_ns=1_000_000, where="aux_thread")
    with pytest.raises(inject.InjectSpecError):  # helper thread must be pre-started, not lazy
        inj.set_spec(spec)
    inj.prepare_aux()
    assert inj.aux_thread_name == "capkit-inject-aux-relay.msg"
    inj.set_spec(spec)
    t0 = time.monotonic_ns()
    for _ in range(5):
        assert inj.hit() == 0
    assert time.monotonic_ns() - t0 < 5_000_000  # caller did not spin
    deadline = time.monotonic() + 5
    while inj.stats()["applied"] < 5 and time.monotonic() < deadline:
        time.sleep(0.005)
    st = inj.stats()
    assert st["applied"] == 5 and st["applied_ns"] >= 5_000_000 and st["aux_pending"] == 0
    inj.close()


def test_aux_pending_counts_unfinished_work():
    inj = inject.Injector("relay.msg")
    inj.prepare_aux()
    inj.set_spec(inject.InjectSpec("relay.msg", "busy", delta_ns=200_000_000, where="aux_thread"))
    for _ in range(3):
        inj.hit()
    c = inj.counters()
    assert c["aux_pending"] >= 2  # 3 x 200 ms queued; at most one can be done this soon
    assert c["applied"] + c["aux_pending"] == 3
    inj.close()


def test_tier_report_deltas():
    reg = SlotRegistry()
    s = inject.InjectorSet(["recv.chunk"], reg)
    for _ in range(7):
        s["recv.chunk"].should_drop()  # tier A: no spec
    m0 = s.mark()
    s.apply_tier_specs([inject.parse_inject_spec("site=recv.chunk,mode=drop,every_n=2,phase=1")])
    m1 = s.mark()
    for _ in range(5):
        s["recv.chunk"].should_drop()
    m2 = s.mark()
    rep_a = inject.tier_report(m0, m1)[0]
    assert rep_a["activation_unchanged"] is False and rep_a["expected_applied_delta"] is None
    rep_b = inject.tier_report(m1, m2)[0]
    assert rep_b["hits_delta"] == 5 and rep_b["hits_since_activation_at_start"] == 0
    assert rep_b["expected_applied_delta"] == 2 == rep_b["applied_delta"]  # passes 2 and 4
    for _ in range(3):
        s["recv.chunk"].should_drop()
    rep_c = inject.tier_report(m2, s.mark())[0]  # same activation: passes 6, 7, 8 -> 6, 8 hit
    assert rep_c["expected_applied_delta"] == 2 == rep_c["applied_delta"]


def test_injector_set():
    reg = SlotRegistry()
    s = inject.InjectorSet(["send.req", "recv.chunk"], reg)
    s.set_launch_specs([inject.parse_inject_spec("site=send.req,mode=drop")])
    with pytest.raises(inject.InjectSpecError):
        s.set_launch_specs([inject.parse_inject_spec("site=nope,mode=drop")])
    s.set_launch_specs([inject.parse_inject_spec("site=send.req,mode=drop")])
    s.apply_tier_specs([inject.parse_inject_spec("site=recv.chunk,mode=drop")])
    assert s["recv.chunk"].active and s["send.req"].active
    s.apply_tier_specs([])
    assert not s["recv.chunk"].active and s["send.req"].active
    with pytest.raises(inject.InjectSpecError):
        s.apply_tier_specs([inject.parse_inject_spec("site=send.req,mode=drop")])
    assert {d["site"] for d in s.stats()} == {"send.req", "recv.chunk"}
    names = s.prestart_aux([inject.parse_inject_spec(
        "site=recv.chunk,mode=busy,delta_ns=1000,where=aux_thread")])
    assert names == ["capkit-inject-aux-recv.chunk"]
    s.close()
