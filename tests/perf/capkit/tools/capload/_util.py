# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the load generator and the null sink (no hot-path state here)."""

from capcommon import hist as _hist
from capcommon.exitcodes import ConfigError
from capcommon.inject import InjectorSet, InjectSpecError, check_unique_sites


def build_injectors(site_modes, registry, cli_specs, tier_table, role):
    """InjectorSet for `site_modes` ({site: allowed modes}); launch specs = command line + the
    tier table's launch_inject for `role`. Every spec this process may ever activate is checked
    against the site's allowed modes now (exit 3 instead of a silently unwired switch), and
    aux_thread helpers are pre-started (INTERFACES.md 5.3, 7)."""
    injs = InjectorSet(sorted(site_modes), registry)
    launch = list(cli_specs) + [s for s in tier_table.launch_inject if s.role == role]
    every = launch + [s for t in tier_table.tiers for s in t.inject if s.role == role]
    try:
        check_unique_sites(launch)
        for s in every:
            if s.site not in site_modes:
                raise InjectSpecError("unknown inject site %r; %s supports %s"
                                      % (s.site, role, sorted(site_modes)))
            if s.mode not in site_modes[s.site]:
                raise InjectSpecError("site %s does not support mode=%s (supported: %s)"
                                      % (s.site, s.mode, list(site_modes[s.site])))
        for t in tier_table.tiers:
            tier_specs = [s for s in t.inject if s.role == role]
            check_unique_sites(tier_specs)
            clash = {s.site for s in tier_specs} & {s.site for s in launch}
            if clash:
                raise InjectSpecError("tier %d inject clashes with a launch spec on %s"
                                      % (t.seq, sorted(clash)))
        aux_names = injs.prestart_aux(every)
        injs.set_launch_specs(launch)
    except InjectSpecError as e:
        raise ConfigError(str(e))
    return injs, aux_names


def tier_specs_for(tier_table, role, seq):
    return [s for s in tier_table.tiers[seq].inject if s.role == role]


def hist_summary(values):
    """{"hist": {"bins": "loghist/1", "counts": [...]}, "n", "max", "min"} of an iterable of ints."""
    counts = _hist.new_counts()
    n = 0
    mx = None
    mn = None
    for v in values:
        _hist.add(counts, v)
        n += 1
        if mx is None or v > mx:
            mx = v
        if mn is None or v < mn:
            mn = v
    return {"hist": {"bins": _hist.NAME, "counts": counts}, "n": n, "max": mx, "min": mn,
            "p50": _hist.quantile(counts, 0.5) if n else None,
            "p99": _hist.quantile(counts, 0.99) if n else None}
