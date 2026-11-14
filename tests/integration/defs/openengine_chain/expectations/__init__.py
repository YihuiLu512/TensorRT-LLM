# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What existing tests are expected to do on the chain, one table per work item.

Each ``WT-xx.py``, ``WT-xx-fixN.py``, ``WT-20-R2.py`` or ``WT-20-R3.py`` in
this directory belongs to the work item it is named after and defines
``EXPECTATIONS``, a list of :class:`Expectation`. The harness applies them; the
tests themselves are not edited.

``match`` is ``<test file name>::<test name>``, where ``*`` matches any text
and every other character, including ``[`` and ``]``, matches itself. For
example ``_test_openai_chat.py::test_beam_search*``.

A ``leg_expect`` entry says how the chain proof of a test run against a
disaggregated server is judged (see ``proof.judge_disagg``); without one, both
legs must carry traffic.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .. import conventions as cv

# The test must fail with an HTTP error carrying ``status``; that counts as a pass.
HTTP_STATUS = "http_status"
# The server must fail to start, and the engine's or the front end's log must contain
# ``log_contains``.
STARTUP_REJECTED = "startup_rejected"
# The test sends no Generate call by design, so zero engine calls is not a failure.
NO_ENGINE_TRAFFIC = "no_engine_traffic"
# The test cancels Generate calls, and a call cancelled before it reaches the engine
# is counted by the front end only: the chain proof accepts a front-end count at or
# above the engine's instead of an equal one (see proof.check_counts). List each such
# test by name; every other test keeps the exact count check.
CANCELS_CALLS = "cancels_calls"
# The test does not apply to the chain.
SKIP = "skip"
# How the chain proof of a disaggregated test is judged: ``leg`` is one of LEG_VALUES.
LEG_EXPECT = "leg_expect"

KINDS = (HTTP_STATUS, STARTUP_REJECTED, NO_ENGINE_TRAFFIC, CANCELS_CALLS, SKIP, LEG_EXPECT)
# Kinds that decide a test's outcome; at most one may apply to a test.
OUTCOME_KINDS = (HTTP_STATUS, STARTUP_REJECTED, SKIP)

# The values of a leg_expect entry:
# - gen_zero_ok: the context leg may end the request, so the generation role may get no call.
# - ctx_zero_ok: the disaggregated server may prefill locally, so the context role may get
#   no call and the generation workers may get aggregated calls.
# - single_side: only one side runs: the context side alone, or the generation side alone
#   in the generation-only benchmark mode.
# - direct_worker: the test sends to one worker front end itself, not through the
#   disaggregated server; ``direct_requests`` is the number of generation requests it sends,
#   if the test knows it.
# - direct_two_leg: the test sends both legs to the two worker front ends itself.
# - direct_ctx_leg: the test sends context legs to a context worker front end itself and no
#   generation leg; ``orphans`` is the number of hand-offs left unused (default: one per
#   context request the worker front ends received).
# - killed: the test kills the workers of the roles in ``killed``.
# - cross_node: the context and generation workers run on two nodes, each in a pytest of its
#   own; the runner judges what needs both nodes' readings.
GEN_ZERO_OK = "gen_zero_ok"
CTX_ZERO_OK = "ctx_zero_ok"
SINGLE_SIDE = "single_side"
DIRECT_WORKER = "direct_worker"
DIRECT_TWO_LEG = "direct_two_leg"
DIRECT_CTX_LEG = "direct_ctx_leg"
KILLED = "killed"
CROSS_NODE = "cross_node"
LEG_VALUES = (
    GEN_ZERO_OK,
    CTX_ZERO_OK,
    SINGLE_SIDE,
    DIRECT_WORKER,
    DIRECT_TWO_LEG,
    DIRECT_CTX_LEG,
    KILLED,
    CROSS_NODE,
)
ROLES = ("ctx", "gen")

_WT_FILE = re.compile(r"^(?:WT-\d{2}(?:-fix\d+)?|WT-20-R[23])\.py$")


@dataclass(frozen=True)
class Expectation:
    wt: str
    match: str
    kind: str
    reason: str
    status: int | None = None
    log_contains: str | None = None
    chains: tuple[str, ...] = cv.CHAINS
    leg: str | None = None
    killed: tuple[str, ...] = ()
    orphans: int | None = None
    direct_requests: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.wt}: unknown kind {self.kind!r} for {self.match!r}")
        if "::" not in self.match:
            raise ValueError(f"{self.wt}: match {self.match!r} must be '<file>::<test>'")
        if not self.reason.strip():
            raise ValueError(f"{self.wt}: {self.match!r} needs a reason")
        if (self.kind == HTTP_STATUS) != (self.status is not None):
            raise ValueError(
                f"{self.wt}: {self.match!r}: status is required for, and only for, {HTTP_STATUS}"
            )
        if (self.kind == STARTUP_REJECTED) != bool(self.log_contains):
            raise ValueError(
                f"{self.wt}: {self.match!r}: log_contains is required for, and only "
                f"for, {STARTUP_REJECTED}"
            )
        unknown = set(self.chains) - set(cv.CHAINS)
        if unknown or not self.chains:
            raise ValueError(f"{self.wt}: {self.match!r}: bad chains {self.chains!r}")
        self._check_leg()

    def _check_leg(self) -> None:
        where = f"{self.wt}: {self.match!r}"
        if (self.kind == LEG_EXPECT) != (self.leg is not None):
            raise ValueError(f"{where}: leg is required for, and only for, {LEG_EXPECT}")
        if self.leg is not None and self.leg not in LEG_VALUES:
            raise ValueError(f"{where}: unknown leg {self.leg!r}; one of {LEG_VALUES}")
        if (self.leg == KILLED) != bool(self.killed):
            raise ValueError(f"{where}: killed is required for, and only for, leg {KILLED}")
        if (
            not isinstance(self.killed, tuple)
            or not set(self.killed) <= set(ROLES)
            or len(set(self.killed)) != len(self.killed)
        ):
            raise ValueError(
                f"{where}: killed {self.killed!r} must be a tuple of distinct roles {ROLES}"
            )
        for name, value, leg in (
            ("orphans", self.orphans, DIRECT_CTX_LEG),
            ("direct_requests", self.direct_requests, DIRECT_WORKER),
        ):
            if value is None:
                continue
            if self.leg != leg:
                raise ValueError(f"{where}: {name} is only for leg {leg}")
            if type(value) is not int or value < 0:
                raise ValueError(f"{where}: {name} must be a non-negative int, not {value!r}")

    def matches(self, short_test_id: str, chain: str) -> bool:
        return chain in self.chains and _pattern(self.match).fullmatch(short_test_id) is not None


@functools.lru_cache(maxsize=None)
def _pattern(match: str) -> re.Pattern[str]:
    return re.compile(".*".join(re.escape(part) for part in match.split("*")))


def short_id(nodeid: str) -> str:
    """``dir/sub/_test_x.py::Cls::test_y[p]`` -> ``_test_x.py::Cls::test_y[p]``."""
    path, sep, rest = nodeid.partition("::")
    return os.path.basename(path) + sep + rest


def load(directory: str | None = None) -> list[Expectation]:
    """Load every work item's table in ``directory``, checking that each entry names its own file.

    Any other ``.py`` file there except ``__init__.py`` is an error, so that a
    misnamed table is not silently ignored.
    """
    directory = directory or os.path.dirname(os.path.abspath(__file__))
    loaded: list[Expectation] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".py") or name == "__init__.py":
            continue
        if not _WT_FILE.match(name):
            raise ValueError(
                f"{name}: expectation tables are named WT-<nn>.py, WT-<nn>-fix<n>.py, "
                "WT-20-R2.py or WT-20-R3.py"
            )
        wt = name[:-3]
        spec = importlib.util.spec_from_file_location(
            f"openengine_chain_expectations_{wt}", os.path.join(directory, name)
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        table = getattr(module, "EXPECTATIONS", None)
        if not isinstance(table, (list, tuple)):
            raise ValueError(f"{name} must define EXPECTATIONS as a list")
        for entry in table:
            if not isinstance(entry, Expectation):
                raise ValueError(f"{name}: {entry!r} is not an Expectation")
            if entry.wt != wt:
                raise ValueError(f"{name}: entry {entry.match!r} says it belongs to {entry.wt}")
            loaded.append(entry)
    return loaded


def applicable(expectations: Iterable[Expectation], nodeid: str, chain: str) -> list[Expectation]:
    """The expectations for one test; more than one outcome-deciding entry is an error."""
    tid = short_id(nodeid)
    found = [e for e in expectations if e.matches(tid, chain)]
    for kinds in (OUTCOME_KINDS, (LEG_EXPECT,)):
        clashing = [e for e in found if e.kind in kinds]
        if len(clashing) > 1:
            raise ValueError(
                f"{tid}: conflicting expectations "
                + ", ".join(f"{e.wt}:{e.kind}" for e in clashing)
            )
    return found


def outcome_expectation(found: Sequence[Expectation]) -> Expectation | None:
    return next((e for e in found if e.kind in OUTCOME_KINDS), None)


def leg_expectation(found: Sequence[Expectation]) -> Expectation | None:
    """The test's ``leg_expect`` entry; :func:`applicable` allows at most one."""
    return next((e for e in found if e.kind == LEG_EXPECT), None)


def http_status_of(error: BaseException) -> int | None:
    """The HTTP status an exception from openai, requests or httpx carries, if any."""
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None
