# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capprobe: in-process probes for product processes (INTERFACES.md sections 6, 7; design 2.4).

Imported by the probe patch in the TRT-LLM probe tree. It does not depend on tensorrt_llm.
Switches are read ONCE at import from CAPPROBE_* environment variables (capprobe.config) into
module-level constants. Every code location of a patched site is one guarded call:

    if _capprobe.ON_F2:
        _capprobe.F2()

so when a site is off the cost is one module attribute load and one branch. When anything is on,
counters live in one preallocated int64 array (capcommon.snapshot.SlotRegistry; each slot has one
writer thread, no locks on the hot path) and a background thread copies the whole array to
<CAPPROBE_SHM_DIR>/<pid>.bin every CAPPROBE_INTERVAL_MS. Nothing on a measured path does I/O.

Event-loop lag: `install_loop_lag(site)` is called from inside the running loop (guarded by
`if _capprobe.LAG:`); it starts a capcommon.aio.LoopLagProbe writing the hist slot lag.<site>
and refreshes the D11 version self-report in the snapshot meta with the loop class.

Self-check injection (CAPPROBE_INJECT, capcommon.inject) is fixed for the process lifetime and
supports busy / sleep / miscount at the count sites; heartbeat counters inject.<site>.* appear
in the snapshot.
"""

import array
import asyncio
import atexit
import os
import sys

from capcommon import inject as _inject
from capcommon import snapshot as _snapshot
from capcommon import versions as _versions

from . import sites as _sites
from .config import CapprobeConfigError, parse_config

__all__ = ["CONFIG", "TIER", "GROUPS", "COUNT", "LAG", "ACTIVE", "SITES", "SLOTS",
           "install_loop_lag", "flush", "snapshot_path", "CapprobeConfigError"]

SITES = _sites.SITES
SLOTS = _sites.SLOTS

# `python -m capprobe.decode|overlay|patchcheck` imports this package first. Those tool processes
# must never activate probes (they would leave an unattributed <pid>.bin in the snapshot dir), so
# the exact module named after `-m` in the original command line selects tool mode: no env parse,
# everything off. Product processes always parse the switches (and fail loudly when they are bad).
TOOL_MODULES = ("capprobe.decode", "capprobe.overlay", "capprobe.patchcheck")


def _tool_mode(orig_argv):
    for i in range(1, len(orig_argv) - 1):
        if orig_argv[i] == "-m":
            return orig_argv[i + 1] in TOOL_MODULES
    return False


TOOL_MODE = _tool_mode(getattr(sys, "orig_argv", ()))
CONFIG = None if TOOL_MODE else parse_config()
TIER = None if TOOL_MODE else CONFIG.tier
GROUPS = () if TOOL_MODE else CONFIG.groups
COUNT = False if TOOL_MODE else CONFIG.count
LAG = False if TOOL_MODE else CONFIG.lag
ACTIVE = False if TOOL_MODE else CONFIG.active


def _noop():
    return None


# Site switches and callables (rebound below when the site is on). Listed explicitly so a
# reader (and capprobe.patchcheck) can see every name the patch may use.
ON_H1 = ON_H3 = ON_F2 = ON_F4 = False
H1 = H3 = F2 = F4 = _noop


def _loop_versions(loops):
    """D11 record. loops: [(site, loop)] installed so far; the first one is the loop reported in
    loop_impl (a process runs one event loop at each install point)."""
    loop = loops[0][1] if loops else None
    extra = {"probe_loops": [{"site": s, "loop_impl": _versions.loop_impl_name(lp)}
                             for s, lp in loops]}
    return _versions.version_record(loop=loop, extra=extra,
                                    include_grpc="grpc" in sys.modules)


class _ProbeWriter(_snapshot.SnapshotWriter):
    """SnapshotWriter whose meta `versions` can be refreshed (after a loop is installed, or once
    grpc has been imported by the product after capprobe)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loops = []
        self._grpc_seen = "grpc" in sys.modules

    def refresh_versions(self):
        rec = _loop_versions(self.loops)
        reg = self.registry
        with reg._lock:  # the meta is rebuilt under the same lock in write_now
            self._extra["versions"] = rec
            self._meta_gen = -1
        self._grpc_seen = "grpc" in sys.modules

    def write_now(self):
        if not self._grpc_seen and "grpc" in sys.modules:
            self.refresh_versions()
        super().write_now()


class _State:
    __slots__ = ("pid", "registry", "writer", "injectors", "lag_probes", "installs_off")


_state = None


def _make_site(site, reg, injectors):
    off = reg.counter(_sites.COUNT_SITES[site]["slot"]) if COUNT else None
    inj = injectors[site] if injectors is not None else None
    vals = reg.values
    if inj is None:
        if off is None:
            return None

        def site_count():
            vals[off] += 1
        return site_count
    spec = inj.spec
    if spec is not None and spec.mode == "miscount":
        count_delta = inj.count_delta  # config guarantees the count group is on

        def site_miscount():
            vals[off] += count_delta()
        return site_miscount
    hit = inj.hit  # busy / sleep, or no spec at this site (only counts the pass)
    if off is None:
        def site_hit():
            hit()
        return site_hit

    def site_count_hit():
        vals[off] += 1
        hit()
    return site_count_hit


def _setup():
    """Build the registry, injectors, site callables and the snapshot writer for this pid."""
    global _state
    g = globals()
    for site in _sites.COUNT_SITES:
        g["ON_" + site] = False
        g[site] = _noop
    if not ACTIVE:
        _state = None
        return
    st = _State()
    st.pid = os.getpid()
    reg = _snapshot.SlotRegistry()
    groups = set(GROUPS)
    if CONFIG.inject_specs:
        groups.add("inject")
    reg.register_decls(SLOTS, groups=groups)
    st.registry = reg
    st.injectors = None
    if CONFIG.inject_specs:
        st.injectors = _inject.InjectorSet(_sites.INJECT_SITES, reg)
        st.injectors.prestart_aux(CONFIG.inject_specs)  # helper threads before activation
        st.injectors.set_launch_specs(CONFIG.inject_specs)
    for site in _sites.COUNT_SITES:
        fn = _make_site(site, reg, st.injectors)
        if fn is not None:
            g[site] = fn
            g["ON_" + site] = True
    st.lag_probes = {}
    st.installs_off = ({s: reg.counter("lag.%s.installs" % s) for s in _sites.LAG_SITES}
                       if LAG else {})
    identity = dict(CONFIG.identity)
    identity["pid"] = st.pid
    meta = {"identity": identity, "role_hint": CONFIG.role_hint,
            "versions": _loop_versions([]), "probe": {
                "tier": TIER, "groups": list(GROUPS),
                "inject": [s.to_dict() for s in CONFIG.inject_specs],
                "lag_period_ns": CONFIG.lag_period_ns,
                "sites_on": sorted(s for s in _sites.COUNT_SITES if g["ON_" + s])}}
    st.writer = _ProbeWriter(reg, CONFIG.shm_dir, CONFIG.interval_ns, meta_extra=meta,
                             pid=st.pid)
    _state = st
    st.writer.start()


def _after_fork_in_child():
    # The writer thread, aux threads and event loops do not survive fork; rebuild everything with
    # zeroed counters under the child's pid (the parent's counts stay the parent's).
    old = _state
    if old is not None and old.injectors is not None:
        for inj in old.injectors.injectors.values():
            inj._aux = None  # dead thread object; never stop() it from the child
    _setup()


def _at_exit():
    st = _state
    if st is None or st.pid != os.getpid():
        return
    try:
        st.writer.stop()
    except OSError:
        pass


def install_loop_lag(site):
    """Start the event-loop lag probe on the RUNNING loop at install point `site` (Lfe, Lmfl,
    Lmfa). Must be called from a coroutine on that loop. Installing the same site twice in one
    process is an error (it would double-count)."""
    if not LAG:
        raise CapprobeConfigError("install_loop_lag called but the lag group is off "
                                  "(the call must be guarded by `if _capprobe.LAG:`)")
    if site not in _sites.LAG_SITES:
        raise CapprobeConfigError("unknown lag site %r; known: %s" % (site, list(_sites.LAG_SITES)))
    st = _state
    if site in st.lag_probes:
        raise CapprobeConfigError("lag probe %s installed twice in pid %d" % (site, st.pid))
    loop = asyncio.get_running_loop()
    from capcommon import aio as _aio  # noqa: PLC0415  (only lag-on processes need it)

    probe = _aio.LoopLagProbe(loop, st.registry, site, CONFIG.lag_period_ns).start()
    st.lag_probes[site] = probe
    st.registry.values[st.installs_off[site]] += 1
    st.writer.loops.append((site, loop))
    st.writer.refresh_versions()
    return probe


def flush():
    """Write one snapshot now (tests and orderly shutdown). No-op when nothing is on."""
    st = _state
    if st is not None and st.pid == os.getpid():
        st.writer.write_now()


def snapshot_path():
    st = _state
    return None if st is None else st.writer.path


def injector_marks():
    """InjectorSet.mark() of this process, or None without injection (tests, diagnostics)."""
    st = _state
    return None if st is None or st.injectors is None else st.injectors.mark()


def _registry_values():
    st = _state
    return None if st is None else array.array("q", st.registry.values)


_setup()
if ACTIVE:
    os.register_at_fork(after_in_child=_after_fork_in_child)
    atexit.register(_at_exit)
