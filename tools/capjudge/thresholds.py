# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thresholds come only from registered calibration files (DATA-AND-ANALYSIS 4.6, INTERFACES 9.2).

There is no command-line threshold. A band key is "<T code>:<quantity>:<object>" (INTERFACES 9.4),
e.g. "T1:msg_lag_p50_ns:capload". The same key defined by two sources is an input error (no
silent precedence). A key that no source defines -> the check that needs it is NOT_RUN.
"""

from . import stats

KEY_CODES = ("T1", "T2", "T3", "T6")


class BandError(Exception):
    pass


def band_key(code, quantity, obj):
    if code not in KEY_CODES:
        raise ValueError("unknown statistic code %r" % code)
    for part in (quantity, obj):
        if not isinstance(part, str) or not part or ":" in part:
            raise ValueError("bad band key part %r" % (part,))
    return "%s:%s:%s" % (code, quantity, obj)


class Bands:
    """Merged view of the noise bands of every calibration source of one launch."""

    def __init__(self):
        self._bands = {}  # key -> (band, source label)
        self.sources = []  # [{"kind", "relpath", "sha256", "n_keys"}]
        self.extras = {}  # source kind -> the whole document (pilot: d14_softirq etc.)
        self.lookups = {}  # key -> "found" / "missing" (written to inputs.json)

    def add_source(self, kind, relpath, sha256, doc):
        nb = doc.get("noise_bands") if isinstance(doc, dict) else None
        if nb is None:
            nb = {}
        if not isinstance(nb, dict):
            raise BandError("%s: noise_bands must be an object" % relpath)
        for key, band in nb.items():
            parts = key.split(":")
            if len(parts) != 3 or parts[0] not in KEY_CODES or not parts[1] or not parts[2]:
                raise BandError("%s: bad band key %r" % (relpath, key))
            if not stats.band_is_valid(band):
                raise BandError("%s: band %r must have numeric min <= max and int n >= 1"
                                % (relpath, key))
            if key in self._bands:
                raise BandError("band %r defined by both %s and %s"
                                % (key, self._bands[key][1], relpath))
            self._bands[key] = (band, relpath)
        self.sources.append({"kind": kind, "relpath": relpath, "sha256": sha256,
                             "n_keys": len(nb)})
        if kind in self.extras:
            raise BandError("two calibration files of kind %s" % kind)
        self.extras[kind] = doc

    def get(self, key):
        hit = self._bands.get(key)
        self.lookups[key] = "found" if hit else "missing"
        return None if hit is None else hit[0]

    def source_of(self, key):
        hit = self._bands.get(key)
        return None if hit is None else hit[1]

    def keys(self):
        return sorted(self._bands)
