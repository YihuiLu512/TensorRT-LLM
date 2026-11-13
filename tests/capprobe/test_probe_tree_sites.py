# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the real probe patch places each site (read from the probe tree's working tree, the
content an overlay copies): F4 on both paths that drain the request queue, every lag install
point inside the try whose finally cleans up and after the role's start-up has completed, and
H1 counting /v1/ requests only. capprobe.patchcheck checks the form of each call; these tests
check the position and the behaviour that the form alone does not show."""

import ast
import asyncio
import os
import typing

import pytest
from probe_testutil import require_probe_tree

from capprobe import sites as capsites


def parse(tree, rel):
    with open(os.path.join(tree, rel), encoding="utf-8") as f:
        src = f.read()
    return src, ast.parse(src, filename=rel)


def parents(mod):
    out = {}
    for node in ast.walk(mod):
        for child in ast.iter_child_nodes(node):
            out[child] = node
    return out


def guarded_ifs(mod, lines):
    found = {n.lineno: n for n in ast.walk(mod) if isinstance(n, ast.If) and n.lineno in lines}
    assert sorted(found) == sorted(lines), "no guarded if at some of %s" % lines
    return [found[ln] for ln in lines]


def enclosing(node, par, kinds):
    while node in par:
        node = par[node]
        if isinstance(node, kinds):
            return node
    return None


def function_of(node, par):
    return enclosing(node, par, (ast.FunctionDef, ast.AsyncFunctionDef))


def calls_in(stmt):
    """Names of every function called in `stmt` (`a.b()` -> 'a.b', `f()` -> 'f')."""
    out = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Call):
            out.add(ast.unparse(n.func))
    return out


def f4_functions(tree, rel, lines):
    _, mod = parse(tree, rel)
    par = parents(mod)
    return sorted(function_of(n, par).name for n in guarded_ifs(mod, lines))


# Lag install point: function, the call that completes start-up before it, the call that
# starts serving after it (all direct statements of the same try body).
LAG_POSITION = {
    "Lfe": ("run", "facade.start", "server"),
    "Lmfl": ("_run_launcher", "asyncio.shield", "_serve"),
    "Lmfa": ("_run_attached", "_wait_for_launcher", "_serve"),
}


def lag_position_errors(tree, rel, lines, site):
    fn_name, started, serving = LAG_POSITION[site]
    _, mod = parse(tree, rel)
    par = parents(mod)
    errors = []
    for node in guarded_ifs(mod, lines):
        fn = function_of(node, par)
        if fn is None or fn.name != fn_name:
            errors.append("%s:%d not in %s" % (site, node.lineno, fn_name))
            continue
        try_node = par.get(node)
        if not isinstance(try_node, ast.Try) or node not in try_node.body:
            errors.append("%s:%d is not a direct statement of a try body" % (site, node.lineno))
            continue
        if not try_node.finalbody:
            errors.append("%s:%d: its try has no finally" % (site, node.lineno))
        body = try_node.body
        i = body.index(node)
        start_at = [j for j, s in enumerate(body) if started in calls_in(s)]
        serve_at = [j for j, s in enumerate(body) if serving in calls_in(s)]
        if not start_at or start_at[0] >= i:
            errors.append("%s:%d is not after %s()" % (site, node.lineno, started))
        if not serve_at or serve_at[0] <= i:
            errors.append("%s:%d is not before %s()" % (site, node.lineno, serving))
    return errors


def h1_counts(tree, rel, line, paths, on=True):
    """Run the patched ServerArrivalTimeMiddleware.__call__ (extracted from the probe tree) with
    a stub capprobe; return how many of `paths` (and a websocket and a lifespan scope) count."""
    src, mod = parse(tree, rel)
    cls = [n for n in mod.body if isinstance(n, ast.ClassDef)
           and n.name == "ServerArrivalTimeMiddleware"]
    assert len(cls) == 1 and cls[0].lineno <= line <= cls[0].end_lineno
    hits = []

    class Probe:
        ON_H1 = on

        @staticmethod
        def H1():
            hits.append(1)

    class Clock:
        def now(self):
            return 0

    ns = {"_capprobe": Probe, "Optional": typing.Optional, "AdjustedSteadyClock": Clock}
    exec(compile(ast.get_source_segment(src, cls[0]), rel, "exec"), ns)  # noqa: S102

    async def app(scope, receive, send):
        return None

    async def drive():
        mw = ns["ServerArrivalTimeMiddleware"](app, adjusted_clock=Clock())
        for p in paths:
            await mw({"type": "http", "path": p}, None, None)
        await mw({"type": "websocket", "path": "/v1/x"}, None, None)
        await mw({"type": "lifespan"}, None, None)

    asyncio.run(drive())
    return len(hits)


def test_f4_counts_on_both_queue_paths():
    tree = require_probe_tree()
    d = capsites.SITES["F4"]
    assert f4_functions(tree, d["file"], d["lines"]) == ["__anext__", "_feed"]


@pytest.mark.parametrize("site", sorted(LAG_POSITION))
def test_lag_installed_after_start_inside_try(site):
    assert set(LAG_POSITION) == set(capsites.LAG_SITES)
    tree = require_probe_tree()
    d = capsites.SITES[site]
    assert lag_position_errors(tree, d["file"], d["lines"], site) == []


def test_h1_counts_v1_requests_only():
    tree = require_probe_tree()
    d = capsites.SITES["H1"]
    paths = ["/health", "/metrics", "/v1/models", "/v1/completions", "/v1/chat/completions",
             "/version", "/v1"]
    assert h1_counts(tree, d["file"], d["lines"][0], paths) == 3
    assert h1_counts(tree, d["file"], d["lines"][0], paths, on=False) == 0
