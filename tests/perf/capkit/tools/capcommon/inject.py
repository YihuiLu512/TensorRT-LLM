# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Injection / self-check switches shared by every tool and capprobe (INTERFACES.md section 7).

Each site owns one Injector. Heartbeat counters inject.<site>.hits / .applied / .applied_ns live in
the process's slot registry when one is given (so they appear in snapshots), else in a private
array. Hits are counted even when no spec is active.

Selection is relative to activation: the h-th pass since set_spec() (h from 1) is selected iff
(h - 1) % every_n == phase, so a spec that becomes active mid-run selects the same passes no
matter how many passes happened before it. Per-tier reports are deltas between two mark()s
(INTERFACES.md 7.3). Helper threads for where=aux_thread are created up front by
InjectorSet.prestart_aux(), never lazily, so the thread inventory taken before the first load tier
already contains them (C13).
"""

import array
import queue
import re
import threading
import time

from .exitcodes import ConfigError

MODES = ("busy", "sleep", "miscount", "drop", "short")
WHERES = ("inline", "aux_thread")
SITE_RE = re.compile(r"^[A-Za-z0-9_.]{1,64}$")
ROLE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_INT_KEYS = ("delta_ns", "every_n", "phase", "miscount_delta", "short_tokens")
_ALL_KEYS = ("site", "mode", "where", "role") + _INT_KEYS
ENV_VAR = "CAPPROBE_INJECT"


class InjectSpecError(ConfigError):
    pass


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


class InjectSpec:
    __slots__ = _ALL_KEYS

    def __init__(self, site, mode, delta_ns=None, every_n=1, phase=0, miscount_delta=None,
                 short_tokens=None, where="inline", role=None):
        self.site = site
        self.mode = mode
        self.delta_ns = delta_ns
        self.every_n = every_n
        self.phase = phase
        self.miscount_delta = miscount_delta
        self.short_tokens = short_tokens
        self.where = where
        self.role = role
        self.validate()

    def validate(self, require_role=False):
        if not isinstance(self.site, str) or not SITE_RE.match(self.site):
            raise InjectSpecError("inject site invalid: %r" % (self.site,))
        if self.mode not in MODES:
            raise InjectSpecError("inject mode must be one of %s, got %r" % (MODES, self.mode))
        if self.where not in WHERES:
            raise InjectSpecError("inject where must be one of %s" % (WHERES,))
        if self.where == "aux_thread" and self.mode != "busy":
            raise InjectSpecError("where=aux_thread is only valid with mode=busy")
        if not _is_int(self.every_n) or self.every_n < 1:
            raise InjectSpecError("every_n must be an int >= 1")
        if not _is_int(self.phase) or not 0 <= self.phase < self.every_n:
            raise InjectSpecError("phase must be an int in [0, every_n)")
        need = {"busy": "delta_ns", "sleep": "delta_ns", "miscount": "miscount_delta",
                "short": "short_tokens"}.get(self.mode)
        for k in ("delta_ns", "miscount_delta", "short_tokens"):
            v = getattr(self, k)
            if k == need:
                if not _is_int(v):
                    raise InjectSpecError("mode=%s requires int %s" % (self.mode, k))
            elif v is not None:
                raise InjectSpecError("%s is not valid with mode=%s" % (k, self.mode))
        if self.mode in ("busy", "sleep") and self.delta_ns <= 0:
            raise InjectSpecError("delta_ns must be > 0")
        if self.mode == "miscount" and self.miscount_delta == 0:
            raise InjectSpecError("miscount_delta must be non-zero")
        if self.mode == "short" and self.short_tokens < 1:
            raise InjectSpecError("short_tokens must be >= 1")
        if self.role is not None and (not isinstance(self.role, str)
                                      or not ROLE_RE.match(self.role)):
            raise InjectSpecError("role invalid: %r" % (self.role,))
        if require_role and self.role is None:
            raise InjectSpecError("inject specs in a tier table need a role")

    @classmethod
    def from_dict(cls, d, require_role=False):
        if not isinstance(d, dict):
            raise InjectSpecError("inject spec must be an object")
        extra = set(d) - set(_ALL_KEYS)
        if extra:
            raise InjectSpecError("unknown inject keys %s" % sorted(extra))
        if "site" not in d or "mode" not in d:
            raise InjectSpecError("inject spec needs site and mode")
        spec = cls(**d)
        spec.validate(require_role=require_role)
        return spec

    def to_dict(self):
        out = {"site": self.site, "mode": self.mode, "every_n": self.every_n,
               "phase": self.phase, "where": self.where}
        for k in ("delta_ns", "miscount_delta", "short_tokens", "role"):
            v = getattr(self, k)
            if v is not None:
                out[k] = v
        return out

    def to_text(self):
        return ",".join("%s=%s" % (k, v) for k, v in self.to_dict().items() if k != "role")

    def __eq__(self, other):
        return isinstance(other, InjectSpec) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return "InjectSpec(%s)" % self.to_text()


def parse_inject_spec(text):
    """Parse 'site=F2,mode=busy,delta_ns=20000' into an InjectSpec (no role allowed)."""
    d = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise InjectSpecError("inject spec item %r is not key=value" % part)
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if k in d:
            raise InjectSpecError("duplicate key %r in inject spec" % k)
        if k == "role":
            raise InjectSpecError("role is only allowed in tier tables")
        if k in _INT_KEYS:
            try:
                d[k] = int(v, 10)
            except ValueError:
                raise InjectSpecError("%s must be an integer, got %r" % (k, v))
        else:
            d[k] = v
    return InjectSpec.from_dict(d)


def parse_inject_specs(text):
    """Parse ';'-separated specs (env var form). Empty / None -> []. Duplicate sites rejected."""
    if not text:
        return []
    specs = [parse_inject_spec(p) for p in text.split(";") if p.strip()]
    check_unique_sites(specs)
    return specs


def check_unique_sites(specs):
    sites = [s.site for s in specs]
    dup = sorted({s for s in sites if sites.count(s) > 1})
    if dup:
        raise InjectSpecError("more than one inject spec for site(s) %s" % dup)


class _PrivateSlots:
    """Minimal stand-in for snapshot.SlotRegistry when an Injector is used without one."""

    def __init__(self):
        self.values = array.array("q")
        self._names = {}

    def counter(self, name):
        if name not in self._names:
            self._names[name] = len(self.values)
            self.values.append(0)
        return self._names[name]


class _AuxSpinner:
    """A helper Python thread that performs busy-waits handed to it (where=aux_thread)."""

    def __init__(self, injector):
        self._inj = injector
        self.done = 0  # busy-waits completed (written only by the helper thread)
        self._q = queue.SimpleQueue()
        self._t = threading.Thread(target=self._run, name="capkit-inject-aux-%s" % injector.site,
                                   daemon=True)
        self._t.start()

    def kick(self, delta_ns):
        self._q.put(delta_ns)

    def _run(self):
        vals = self._inj._vals
        ia, ins = self._inj._i_applied, self._inj._i_applied_ns
        mono = time.monotonic_ns
        while True:
            delta = self._q.get()
            if delta is None:
                return
            t0 = mono()
            end = t0 + delta
            t = t0
            while t < end:
                t = mono()
            vals[ia] += 1
            vals[ins] += t - t0
            self.done += 1

    def stop(self):
        self._q.put(None)


class Injector:
    """One injection site. Call the method matching the spec's mode at the site.

    Not thread-safe across writers: one site is passed by one thread (single-writer slots).
    """

    def __init__(self, site, registry=None):
        if not isinstance(site, str) or not SITE_RE.match(site):
            raise InjectSpecError("site invalid: %r" % (site,))
        self.site = site
        reg = registry if registry is not None else _PrivateSlots()
        self._reg = reg
        self._i_hits = reg.counter("inject.%s.hits" % site)
        self._i_applied = reg.counter("inject.%s.applied" % site)
        self._i_applied_ns = reg.counter("inject.%s.applied_ns" % site)
        self._vals = reg.values
        self.spec = None
        self.active = False
        self._aux = None
        self._aux_kicked = 0
        self._base = 0  # cumulative hits at the last set_spec()
        self.activation_id = 0  # incremented by every set_spec()

    def prepare_aux(self):
        """Start the helper thread for where=aux_thread specs (call once, at process start)."""
        if self._aux is None:
            self._aux = _AuxSpinner(self)

    @property
    def aux_thread_name(self):
        return None if self._aux is None else self._aux._t.name

    def set_spec(self, spec):
        """Activate `spec` (an InjectSpec for this site) or deactivate with None.

        Selection restarts from the next pass. An aux_thread spec requires prepare_aux() first.
        """
        if spec is not None:
            if spec.site != self.site:
                raise InjectSpecError("spec site %r != injector site %r" % (spec.site, self.site))
            if spec.where == "aux_thread" and self._aux is None:
                raise InjectSpecError("site %s: aux_thread spec but the helper thread was not "
                                      "pre-started (InjectorSet.prestart_aux)" % self.site)
        self._base = self._vals[self._i_hits]
        self.activation_id += 1
        self.spec = spec
        self.active = spec is not None

    def _selected(self, modes):
        """Count one hit; return True if this hit is selected by the active spec.

        A call whose kind does not match the active spec's mode is a wiring bug and raises on
        every hit (not only on selected ones), so it cannot go unnoticed.
        """
        vals = self._vals
        vals[self._i_hits] += 1
        if not self.active:
            return False
        sp = self.spec
        if sp.mode not in modes:
            raise InjectSpecError("site %s: called as %s but spec mode is %s"
                                  % (self.site, "/".join(modes), sp.mode))
        return (vals[self._i_hits] - self._base - 1) % sp.every_n == sp.phase

    def hit(self):
        """busy / sleep sites. Returns the ns spent inline (0 when not selected / aux)."""
        if not self._selected(("busy", "sleep")):
            return 0
        sp = self.spec
        if sp.mode == "busy":
            if sp.where == "aux_thread":
                self._aux_kicked += 1
                self._aux.kick(sp.delta_ns)
                return 0
            t0 = time.monotonic_ns()
            end = t0 + sp.delta_ns
            t = t0
            while t < end:
                t = time.monotonic_ns()
        else:  # sleep
            t0 = time.monotonic_ns()
            time.sleep(sp.delta_ns / 1e9)
            t = time.monotonic_ns()
        self._vals[self._i_applied] += 1
        self._vals[self._i_applied_ns] += t - t0
        return t - t0

    def count_delta(self):
        """miscount sites: the amount to add to the site's counter for this event."""
        if not self._selected(("miscount",)):
            return 1
        self._vals[self._i_applied] += 1
        return 1 + self.spec.miscount_delta

    def should_drop(self):
        """drop sites: True when this item must be dropped."""
        if not self._selected(("drop",)):
            return False
        self._vals[self._i_applied] += 1
        return True

    def short_by(self):
        """short sites (once per request): tokens to omit from this request."""
        if not self._selected(("short",)):
            return 0
        self._vals[self._i_applied] += 1
        return self.spec.short_tokens

    def counters(self):
        """Cumulative counters plus the activation-relative pass count (one mark)."""
        v = self._vals
        done = self._aux.done if self._aux is not None else 0
        return {"hits": v[self._i_hits], "applied": v[self._i_applied],
                "applied_ns": v[self._i_applied_ns],
                "hits_since_activation": v[self._i_hits] - self._base,
                "activation_id": self.activation_id,
                "spec": self.spec.to_dict() if self.spec else None,
                "aux_pending": self._aux_kicked - done}

    def stats(self):
        c = self.counters()
        c["site"] = self.site
        c["mode"] = self.spec.mode if self.spec else None
        return c

    def close(self):
        if self._aux is not None:
            self._aux.stop()
            self._aux = None


def expected_applied(hits, every_n, phase):
    """Number of selected passes among the first `hits` passes since activation (INTERFACES.md
    section 7.3). `hits` must be counted since activation, not since process start."""
    if hits <= phase:
        return 0
    return (hits - phase - 1) // every_n + 1


def expected_applied_delta(h_start, h_end, every_n, phase):
    """Selected passes between two marks of the same activation (hits since activation)."""
    if h_end < h_start:
        raise ValueError("h_end < h_start")
    return expected_applied(h_end, every_n, phase) - expected_applied(h_start, every_n, phase)


def tier_report(start_mark, end_mark):
    """Per-site per-tier inject entries (INTERFACES.md 5.4.1) from two InjectorSet.mark()s.

    expected_applied_delta is null when the activation changed inside the interval or no spec
    was active; the judge then treats the heartbeat check for that site as NOT_RUN.
    """
    out = []
    for site in sorted(end_mark):
        a, b = start_mark[site], end_mark[site]
        same = a["activation_id"] == b["activation_id"]
        spec = b["spec"]
        exp = None
        if same and spec is not None:
            exp = expected_applied_delta(a["hits_since_activation"], b["hits_since_activation"],
                                         spec["every_n"], spec["phase"])
        out.append({"site": site, "spec": spec, "activation_unchanged": same,
                    "hits_delta": b["hits"] - a["hits"],
                    "applied_delta": b["applied"] - a["applied"],
                    "applied_ns_delta": b["applied_ns"] - a["applied_ns"],
                    "hits_since_activation_at_start": a["hits_since_activation"],
                    "hits_since_activation_at_end": b["hits_since_activation"],
                    "expected_applied_delta": exp,
                    "aux_pending_at_end": b["aux_pending"]})
    return out


class InjectorSet:
    """All injection sites of one process. Unknown sites in a spec are a config error."""

    def __init__(self, sites, registry=None):
        self.injectors = {s: Injector(s, registry) for s in sites}
        self.launch_specs = {}

    def __getitem__(self, site):
        return self.injectors[site]

    def _check_known(self, specs):
        unknown = sorted({s.site for s in specs} - set(self.injectors))
        if unknown:
            raise InjectSpecError("unknown inject site(s) %s; this process supports %s"
                                  % (unknown, sorted(self.injectors)))

    def set_launch_specs(self, specs):
        check_unique_sites(specs)
        self._check_known(specs)
        self.launch_specs = {s.site: s for s in specs}
        for site, inj in self.injectors.items():
            inj.set_spec(self.launch_specs.get(site))

    def apply_tier_specs(self, specs):
        """Switch to a tier's specs (launch-level specs stay; clashes are a config error)."""
        check_unique_sites(specs)
        self._check_known(specs)
        clash = sorted({s.site for s in specs} & set(self.launch_specs))
        if clash:
            raise InjectSpecError("tier spec clashes with launch spec on %s" % clash)
        tier = {s.site: s for s in specs}
        for site, inj in self.injectors.items():
            want = self.launch_specs.get(site) or tier.get(site)
            if want is not inj.spec:
                inj.set_spec(want)

    def prestart_aux(self, specs):
        """Pre-start helper threads for every site that any of `specs` (all specs this process
        may ever activate: launch-level + every tier's) runs with where=aux_thread."""
        self._check_known(specs)
        for s in specs:
            if s.where == "aux_thread":
                self.injectors[s.site].prepare_aux()
        return sorted(i.aux_thread_name for i in self.injectors.values()
                      if i.aux_thread_name is not None)

    def mark(self):
        return {site: inj.counters() for site, inj in self.injectors.items()}

    def stats(self):
        return [inj.stats() for inj in self.injectors.values()]

    def close(self):
        for inj in self.injectors.values():
            inj.close()
