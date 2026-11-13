# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse and validate capprobe's environment switches (INTERFACES.md section 6.5, erratum R1b).

Pure function of a mapping and a command line, so tests can exercise every rule without
re-importing capprobe. Activation:

* no CAPPROBE_* variable at all: probes off, nothing is read further;
* CAPPROBE_* present: the values that are present are validated whatever the process (unknown
  variable, bad tier, groups that contradict the tier, bad injection, bad entry rule ...);
  probes stay off unless the process command line matches the registered entry
  (CAPPROBE_ENTRY, capcommon.probe_entry). Spawn pool children, MPI workers and other importers
  inherit the variables but not the command line, so they stay off silently;
* the matched entry process: the full rule set (tier required and not P0, shm dir, interval,
  identity file, lag period).

Every violation raises CapprobeConfigError (a RuntimeError): the product process then fails to
start, which is louder than silently measuring nothing.
"""

import json
import os

from capcommon import inject as _inject
from capcommon import probe_entry as _entry

from . import sites as _sites

TIERS = ("P0", "P1", "P2")
ENV_GROUPS = "CAPPROBE_GROUPS"
ENV_TIER = "CAPPROBE_TIER"
ENV_SHM_DIR = "CAPPROBE_SHM_DIR"
ENV_INTERVAL_MS = "CAPPROBE_INTERVAL_MS"
ENV_LAG_PERIOD_US = "CAPPROBE_LAG_PERIOD_US"
ENV_INJECT = "CAPPROBE_INJECT"
ENV_PATCH_SHA = "CAPPROBE_PATCH_SHA"
ENV_IDENTITY_FILE = "CAPPROBE_IDENTITY_FILE"
ENV_ROLE_HINT = "CAPPROBE_ROLE_HINT"
ENV_ENTRY = _entry.ENV_VAR
KNOWN_VARS = (ENV_GROUPS, ENV_TIER, ENV_SHM_DIR, ENV_INTERVAL_MS, ENV_LAG_PERIOD_US, ENV_INJECT,
              ENV_PATCH_SHA, ENV_IDENTITY_FILE, ENV_ROLE_HINT, ENV_ENTRY)
IDENTITY_SCHEMA = "capkit.identity/1"
# Why a process left probes off (ProbeConfig.reason); "entry" = the registered entry process.
REASON_NO_ENV = "no_capprobe_env"
REASON_NO_ENTRY = "no_entry_rule"
REASON_NOT_ENTRY = "not_the_registered_entry"
REASON_ENTRY = "entry"


class CapprobeConfigError(RuntimeError):
    pass


class ProbeConfig:
    """Validated switches. `active` is True when anything writes snapshots."""

    __slots__ = ("tier", "groups", "count", "lag", "inject_specs", "shm_dir", "interval_ns",
                 "lag_period_ns", "identity", "role_hint", "patch_sha", "active", "enabled",
                 "reason", "entry")

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__ if k != "inject_specs"} | {
            "inject_specs": [s.to_dict() for s in self.inject_specs]}


def _positive_int(env, name):
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        raise CapprobeConfigError("%s is required here" % name)
    try:
        v = int(raw, 10)
    except ValueError:
        raise CapprobeConfigError("%s must be an integer, got %r" % (name, raw)) from None
    if v <= 0:
        raise CapprobeConfigError("%s must be > 0, got %d" % (name, v))
    return v


def _parse_groups(raw):
    if raw is None or raw.strip() == "":
        return ()
    groups = [g.strip() for g in raw.split(",")]
    if any(not g for g in groups):
        raise CapprobeConfigError("%s has an empty item: %r" % (ENV_GROUPS, raw))
    unknown = sorted(set(groups) - set(_sites.ALL_GROUPS))
    if unknown:
        raise CapprobeConfigError("%s names unknown group(s) %s; known: %s"
                                  % (ENV_GROUPS, unknown, list(_sites.ALL_GROUPS)))
    reserved = sorted(set(groups) & set(_sites.RESERVED_GROUPS))
    if reserved:
        raise CapprobeConfigError("%s names group(s) %s that this capprobe build does not "
                                  "implement (implemented: %s)"
                                  % (ENV_GROUPS, reserved, list(_sites.IMPLEMENTED_GROUPS)))
    if len(set(groups)) != len(groups):
        raise CapprobeConfigError("%s repeats a group: %r" % (ENV_GROUPS, raw))
    return tuple(g for g in _sites.ALL_GROUPS if g in groups)


def _load_identity(path):
    # capcommon.identity validates the field set and values; imported only when needed.
    from capcommon import identity as _identity  # noqa: PLC0415

    try:
        with open(path, "rb") as f:
            doc = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError) as e:
        raise CapprobeConfigError("%s=%r unreadable: %s" % (ENV_IDENTITY_FILE, path, e)) from None
    if not isinstance(doc, dict) or doc.get("schema") != IDENTITY_SCHEMA:
        raise CapprobeConfigError("%s: schema must be %s" % (ENV_IDENTITY_FILE, IDENTITY_SCHEMA))
    ident = doc.get("identity")
    if not isinstance(ident, dict) or set(ident) != set(_identity.ALL_FIELDS):
        raise CapprobeConfigError("%s: identity must have exactly the fields %s"
                                  % (ENV_IDENTITY_FILE, list(_identity.ALL_FIELDS)))
    try:
        obj = _identity.Identity(**ident)
    except _identity.ConfigError as e:
        raise CapprobeConfigError("%s: %s" % (ENV_IDENTITY_FILE, e)) from None
    return obj.to_dict()


def _off(reason, entry=None):
    cfg = ProbeConfig()
    cfg.tier = cfg.shm_dir = cfg.interval_ns = cfg.lag_period_ns = cfg.identity = None
    cfg.role_hint = cfg.patch_sha = None
    cfg.groups, cfg.inject_specs = (), ()
    cfg.count = cfg.lag = cfg.active = cfg.enabled = False
    cfg.reason, cfg.entry = reason, entry
    return cfg


def _static_checks(env):
    """Rules that hold for every process carrying CAPPROBE_*. Returns (tier or None, groups,
    inject specs, entry rule or None)."""
    unknown = sorted(k for k in env if k.startswith("CAPPROBE_") and k not in KNOWN_VARS)
    if unknown:
        raise CapprobeConfigError("unknown capprobe variable(s) %s; known: %s"
                                  % (unknown, list(KNOWN_VARS)))
    tier = env.get(ENV_TIER)
    if tier is not None and tier not in TIERS:
        raise CapprobeConfigError("%s must be one of %s, got %r" % (ENV_TIER, list(TIERS), tier))
    groups = _parse_groups(env.get(ENV_GROUPS))
    if tier is None and groups:
        raise CapprobeConfigError("%s=%s without %s" % (ENV_GROUPS, list(groups), ENV_TIER))
    if tier in ("P0", "P1") and groups:
        raise CapprobeConfigError("%s=%s requires %s to be empty, got %s"
                                  % (ENV_TIER, tier, ENV_GROUPS, list(groups)))
    if tier == "P2" and not groups:
        raise CapprobeConfigError("%s=P2 requires a non-empty %s" % (ENV_TIER, ENV_GROUPS))
    try:
        specs = _inject.parse_inject_specs(env.get(ENV_INJECT))
    except _inject.InjectSpecError as e:
        raise CapprobeConfigError("%s: %s" % (ENV_INJECT, e)) from None
    for s in specs:
        if s.site not in _sites.INJECT_SITES:
            raise CapprobeConfigError("%s: unknown site %r; capprobe supports %s"
                                      % (ENV_INJECT, s.site, list(_sites.INJECT_SITES)))
        if s.mode not in _sites.PROBE_INJECT_MODES:
            raise CapprobeConfigError("%s: mode %r is not supported inside product code "
                                      "(supported: %s)"
                                      % (ENV_INJECT, s.mode, list(_sites.PROBE_INJECT_MODES)))
        if s.mode == "miscount" and "count" not in groups:
            raise CapprobeConfigError("%s: miscount at %s needs the count group on (otherwise "
                                      "the miscount is invisible)" % (ENV_INJECT, s.site))
    if specs and tier is None:
        raise CapprobeConfigError("%s without %s" % (ENV_INJECT, ENV_TIER))
    for name in (ENV_INTERVAL_MS, ENV_LAG_PERIOD_US):
        if name in env:
            _positive_int(env, name)
    shm = env.get(ENV_SHM_DIR)
    if shm is not None and not os.path.isabs(shm):
        raise CapprobeConfigError("%s must be an absolute path, got %r" % (ENV_SHM_DIR, shm))
    entry = None
    if ENV_ENTRY in env:
        try:
            entry = _entry.parse(env[ENV_ENTRY])
        except _entry.EntryRuleError as e:
            raise CapprobeConfigError(str(e)) from None
    return tier, groups, specs, entry


def parse_config(env=None, argv=None):
    """Return a ProbeConfig from `env` (default os.environ) for a process whose command line is
    `argv` (default capcommon.probe_entry.process_argv()). Raises CapprobeConfigError."""
    env = os.environ if env is None else env
    if not any(k.startswith("CAPPROBE_") for k in env):
        return _off(REASON_NO_ENV)
    tier, groups, specs, entry = _static_checks(env)
    if entry is None:
        return _off(REASON_NO_ENTRY)
    argv = _entry.process_argv() if argv is None else argv
    if _entry.match(entry, argv, path=env.get("PATH")) is None:
        return _off(REASON_NOT_ENTRY, entry)
    if tier is None:
        raise CapprobeConfigError("this process is the registered entry (%s) but %s is missing"
                                  % (ENV_ENTRY, ENV_TIER))
    if tier == "P0":
        raise CapprobeConfigError("capprobe was imported under %s=P0: a P0 overlay must not "
                                  "contain the probe patch" % ENV_TIER)
    cfg = _off(REASON_ENTRY, entry)
    cfg.enabled = True
    cfg.tier = tier
    cfg.groups = groups
    cfg.count = "count" in groups
    cfg.lag = "lag" in groups
    cfg.inject_specs = specs
    cfg.patch_sha = env.get(ENV_PATCH_SHA) or None
    cfg.role_hint = env.get(ENV_ROLE_HINT) or None
    cfg.active = bool(groups) or bool(specs)
    if cfg.active:
        shm = env.get(ENV_SHM_DIR)
        if not shm:
            raise CapprobeConfigError("%s must be an absolute path when a group or an injection "
                                      "is on, got %r" % (ENV_SHM_DIR, shm))
        cfg.shm_dir = shm
        cfg.interval_ns = _positive_int(env, ENV_INTERVAL_MS) * 1_000_000
        ipath = env.get(ENV_IDENTITY_FILE)
        if not ipath:
            raise CapprobeConfigError("%s is required when a group or an injection is on"
                                      % ENV_IDENTITY_FILE)
        cfg.identity = _load_identity(ipath)
    if cfg.lag:
        cfg.lag_period_ns = _positive_int(env, ENV_LAG_PERIOD_US) * 1_000
    return cfg
