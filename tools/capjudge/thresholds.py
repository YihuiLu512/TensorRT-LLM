# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thresholds come only from registered calibration files (DATA-AND-ANALYSIS 4.6, INTERFACES 9.2).

There is no command-line threshold. A band key is "<T code>:<quantity>:<object>" (INTERFACES 9.4),
e.g. "T1:msg_lag_p50_ns:capload". Every noise band belongs to one regime: "subsat" (measured on
sub-saturation tiers) or "near_boundary" (measured on tiers near the boundary); a band without a
`regime` field is "subsat". The same (key, regime) defined by two sources is an input error (no
silent precedence); the same key may come from two sources when each owns a different regime.
Only near_boundary_noise files may be registered more than once (MULTI_KINDS).
A judged tier reads only the bands of its own regime (register.json judge.near_boundary_tiers
marks the near-boundary tiers; every other tier is subsat): a near-boundary tier never falls
back to a sub-saturation band (DATA-AND-ANALYSIS 4 item 6). A key that no source defines for
the tier's regime -> the check that needs it is NOT_RUN.
"""

from . import stats

KEY_CODES = ("T1", "T2", "T3", "T6")
REGIMES = ("subsat", "near_boundary")
# Calibration kinds a launch may register more than once (e.g. one near_boundary_noise file per
# regime); every other kind at most once. Either way a (key, regime) has one source.
MULTI_KINDS = ("near_boundary_noise",)


class BandError(Exception):
    pass


def band_key(code, quantity, obj):
    if code not in KEY_CODES:
        raise ValueError("unknown statistic code %r" % code)
    for part in (quantity, obj):
        if not isinstance(part, str) or not part or ":" in part:
            raise ValueError("bad band key part %r" % (part,))
    return "%s:%s:%s" % (code, quantity, obj)


def check_noise_bands(nb, where):
    """Validate a `noise_bands` object; returns {key: (band, regime)}. Raises BandError."""
    if not isinstance(nb, dict):
        raise BandError("%s: noise_bands must be an object" % where)
    out = {}
    for key, band in nb.items():
        parts = key.split(":") if isinstance(key, str) else []
        if len(parts) != 3 or parts[0] not in KEY_CODES or not parts[1] or not parts[2]:
            raise BandError("%s: bad band key %r" % (where, key))
        if not stats.band_is_valid(band):
            raise BandError("%s: band %r must have numeric min <= max and int n >= 1"
                            % (where, key))
        regime = band.get("regime", "subsat")
        if regime not in REGIMES:
            raise BandError("%s: band %r regime %r not one of %s" % (where, key, regime,
                                                                     REGIMES))
        out[key] = (band, regime)
    return out


class Bands:
    """Merged view of the noise bands of every calibration source of one launch."""

    def __init__(self):
        self._bands = {}  # (key, regime) -> (band, source label)
        self.sources = []  # [{"kind", "relpath", "sha256", "n_keys"}]
        # source kind -> the whole document (pilot: d14_softirq etc.); a kind in MULTI_KINDS ->
        # {relpath: document}
        self.extras = {}
        self.lookups = {}  # "<regime>|<key>" -> "found" / "missing" (written to inputs.json)

    def add_source(self, kind, relpath, sha256, doc):
        nb = doc.get("noise_bands") if isinstance(doc, dict) else None
        checked = check_noise_bands({} if nb is None else nb, relpath)
        for key, (band, regime) in checked.items():
            if (key, regime) in self._bands:
                raise BandError("band %r (regime %s) defined by both %s and %s"
                                % (key, regime, self._bands[(key, regime)][1], relpath))
        if kind in MULTI_KINDS:
            if relpath in self.extras.get(kind, {}):
                raise BandError("calibration file %s registered twice" % relpath)
        elif kind in self.extras:
            raise BandError("two calibration files of kind %s" % kind)
        for key, (band, regime) in checked.items():
            self._bands[(key, regime)] = (band, relpath)
        self.sources.append({"kind": kind, "relpath": relpath, "sha256": sha256,
                             "n_keys": len(checked)})
        if kind in MULTI_KINDS:
            self.extras.setdefault(kind, {})[relpath] = doc
        else:
            self.extras[kind] = doc

    def get(self, key, regime):
        if regime not in REGIMES:
            raise ValueError("unknown regime %r" % (regime,))
        hit = self._bands.get((key, regime))
        self.lookups["%s|%s" % (regime, key)] = "found" if hit else "missing"
        return None if hit is None else hit[0]

    def view(self, regime):
        """An object with get(key) bound to one regime (what a judged tier reads)."""
        return _RegimeView(self, regime)

    def source_of(self, key, regime):
        hit = self._bands.get((key, regime))
        return None if hit is None else hit[1]

    def keys(self):
        return sorted(self._bands)


class _RegimeView:
    def __init__(self, bands, regime):
        if regime not in REGIMES:
            raise ValueError("unknown regime %r" % (regime,))
        self._b, self.regime = bands, regime

    def get(self, key):
        return self._b.get(key, self.regime)
