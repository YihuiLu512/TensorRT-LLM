# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static check of the probe patch in a probe tree (B3).

    python -m capprobe.patchcheck --probe-tree PATH --base-commit SHA [--rev REV] [--out FILE]

Checks, against the diff <base-commit> .. <rev> (default: the working tree, i.e. what an overlay
would copy):
  1. only tensorrt_llm/**.py files change, no file is added or removed, and no line is deleted
     (the patch is additions only);
  2. every changed file compiles;
  3. every reference to `_capprobe` is one of: the module-level `import capprobe as _capprobe`
     (exactly one per changed file), the guard of an `if` (`_capprobe.ON_<site>` or
     `_capprobe.LAG`, alone or as the first operand of `and`), or the single statement in such
     an `if` body calling the matching site (`_capprobe.<site>()` /
     `_capprobe.install_loop_lag("<site>")`) with no `else`;
  4. the guarded sites found equal capprobe.SITES: each site in SITES[site]["file"] only, at
     exactly the lines SITES[site]["lines"] (one guarded call per listed line, none elsewhere);
     the expected counts come from SITES, and 0 expected sites is NOT_RUN;
  5. every added line belongs to an import or a guarded site (nothing else was slipped in).

Prints a JSON report; exit 0 = PASS, 3 = FAIL or NOT_RUN.
"""

import argparse
import ast
import json
import re
import subprocess
import sys

from capcommon import exitcodes

from . import sites as _sites

ALIAS = "_capprobe"
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _git(tree, *args):
    return subprocess.run(["git", "-C", tree, *args], check=True, capture_output=True,
                          text=True).stdout


def _diff_args(base, rev):
    return [base] if rev is None else [base, rev]


def _read(tree, rel, rev):
    if rev is None:
        with open("%s/%s" % (tree, rel), "rb") as f:
            return f.read().decode("utf-8")
    return _git(tree, "show", "%s:%s" % (rev, rel))


def _added_lines(tree, base, rev, rel):
    out = _git(tree, "diff", "-U0", "--no-color", *_diff_args(base, rev), "--", rel)
    lines = set()
    for ln in out.splitlines():
        m = _HUNK_RE.match(ln)
        if m:
            start, count = int(m.group(1)), int(m.group(2) or "1")
            lines.update(range(start, start + count))
    return lines


def _attr_of_alias(node):
    """'X' if node is `_capprobe.X`, else None."""
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == ALIAS):
        return node.attr
    return None


def _guard_name(test):
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
        return _attr_of_alias(test.values[0])
    return _attr_of_alias(test)


def _site_of_call(stmt):
    """(site, kind) for `_capprobe.<site>()` or `_capprobe.install_loop_lag("<site>")`."""
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return None
    call = stmt.value
    fn = _attr_of_alias(call.func)
    if fn is None or call.keywords:
        return None
    if fn == "install_loop_lag":
        if (len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                and isinstance(call.args[0].value, str)):
            return call.args[0].value, "lag"
        return None
    if call.args:
        return None
    return fn, "count"


def check_file(rel, source):
    """Return (found_sites {site: [lines]}, accounted_lines set, violations list) for one file."""
    violations = []
    found = {}
    accounted = set()
    try:
        tree = ast.parse(source, filename=rel)
        compile(tree, rel, "exec")
    except SyntaxError as e:
        return found, accounted, ["%s: does not compile: %s" % (rel, e)]
    ok_nodes = set()  # ids of Name nodes whose use is allowed
    imports = 0
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for a in stmt.names:
                if a.asname == ALIAS:
                    if a.name != "capprobe" or len(stmt.names) != 1:
                        violations.append("%s:%d: alias %s must be a lone `import capprobe as "
                                          "%s`" % (rel, stmt.lineno, ALIAS, ALIAS))
                    imports += 1
                    accounted.add(stmt.lineno)
    if imports != 1:
        violations.append("%s: expected exactly 1 module-level `import capprobe as %s`, found %d"
                          % (rel, ALIAS, imports))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.lineno not in accounted:
            for a in node.names:
                if a.name.split(".")[0] == "capprobe" or a.asname == ALIAS:
                    violations.append("%s:%d: capprobe imported other than at module level"
                                      % (rel, node.lineno))
        if not isinstance(node, ast.If):
            continue
        guard = _guard_name(node.test)
        if guard is None:
            continue
        test_name = node.test.values[0] if isinstance(node.test, ast.BoolOp) else node.test
        ok_nodes.add(id(test_name.value))
        if node.orelse:
            violations.append("%s:%d: guarded probe if must not have an else"
                              % (rel, node.lineno))
        if len(node.body) != 1 or _site_of_call(node.body[0]) is None:
            violations.append("%s:%d: guarded probe if body must be one probe call"
                              % (rel, node.lineno))
            continue
        site, kind = _site_of_call(node.body[0])
        want_guard = "LAG" if kind == "lag" else "ON_" + site
        if guard != want_guard:
            violations.append("%s:%d: call for site %s is guarded by %s, expected %s"
                              % (rel, node.lineno, site, guard, want_guard))
        found.setdefault(site, []).append(node.lineno)
        call = node.body[0].value
        ok_nodes.add(id(call.func.value))
        accounted.update(range(node.lineno, node.body[0].end_lineno + 1))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == ALIAS and id(node) not in ok_nodes:
            violations.append("%s:%d: unguarded or unexpected use of %s"
                              % (rel, node.lineno, ALIAS))
    return {k: sorted(v) for k, v in found.items()}, accounted, violations


def run_check(tree, base, rev=None, expected_sites=None):
    """Return the report dict (status PASS / FAIL / NOT_RUN)."""
    expected = dict(_sites.SITES if expected_sites is None else expected_sites)
    violations = []
    status_out = _git(tree, "diff", "--name-status", "--no-renames", *_diff_args(base, rev))
    changed = []
    for ln in status_out.splitlines():
        st, rel = ln.split("\t", 1)
        if st != "M":
            violations.append("%s: change type %s (only modifications are allowed)" % (rel, st))
        if not (rel.startswith("tensorrt_llm/") and rel.endswith(".py")):
            violations.append("%s: outside tensorrt_llm/**.py" % rel)
        if st == "M":
            changed.append(rel)
    if rev is None:
        untracked = _git(tree, "ls-files", "--others", "--exclude-standard", "--", "tensorrt_llm")
        for rel in untracked.splitlines():
            violations.append("%s: untracked file in the probe tree" % rel)
    numstat = _git(tree, "diff", "--numstat", "--no-renames", *_diff_args(base, rev))
    for ln in numstat.splitlines():
        add, dele, rel = ln.split("\t", 2)
        if dele != "0":
            violations.append("%s: %s line(s) deleted (the patch must be additions only)"
                              % (rel, dele))
    found = {}
    files = {}
    for rel in changed:
        if not rel.endswith(".py"):
            continue
        src = _read(tree, rel, rev)
        f_sites, accounted, v = check_file(rel, src)
        violations += v
        added = _added_lines(tree, base, rev, rel)
        stray = sorted(added - accounted)
        if stray:
            violations.append("%s: added line(s) %s are not part of a probe import or guarded "
                              "site" % (rel, stray))
        files[rel] = {"sites": f_sites, "added_lines": len(added)}
        for site, lines in f_sites.items():
            if site in found:
                violations.append("site %s found in %s and %s" % (site, found[site][0], rel))
                continue
            found[site] = (rel, lines)
    for site, d in expected.items():
        want = (d["file"], list(d["lines"]))
        got = found.get(site)
        if not want[1] or want[1] != sorted(set(want[1])):
            violations.append("site %s: SITES lines %s must be non-empty and strictly ascending"
                              % (site, want[1]))
        if got is None:
            violations.append("site %s (%s:%s) not found in the patch"
                              % (site, d["file"], d["lines"]))
        elif got != want:
            violations.append("site %s found at %s:%s but SITES says %s:%s"
                              % (site, got[0], got[1], want[0], want[1]))
    for site in sorted(set(found) - set(expected)):
        violations.append("site %s at %s:%s is not declared in SITES" % (site, *found[site]))
    if not expected:
        status = "NOT_RUN"
    else:
        status = "FAIL" if violations else "PASS"
    return {"schema": "capkit.patchcheck/1", "probe_tree": tree, "base_commit": base,
            "rev": rev or "WORKTREE", "expected_site_count": len(expected),
            "found_site_count": len(found),
            "expected_location_count": sum(len(d["lines"]) for d in expected.values()),
            "found_location_count": sum(len(v[1]) for v in found.values()),
            "changed_files": files,
            "violations": violations, "status": status}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capprobe.patchcheck",
                                 description="static check of the capprobe patch")
    ap.add_argument("--probe-tree", required=True)
    ap.add_argument("--base-commit", required=True)
    ap.add_argument("--rev", default=None, help="commit to check (default: working tree)")
    ap.add_argument("--out", default=None, help="also write the JSON report here")
    args = ap.parse_args(argv)
    try:
        rep = run_check(args.probe_tree, args.base_commit, args.rev)
    except subprocess.CalledProcessError as e:
        print("capprobe.patchcheck: git failed: %s\n%s" % (e, e.stderr), file=sys.stderr)
        return exitcodes.CONFIG
    text = json.dumps(rep, indent=1, sort_keys=True)
    print(text)
    if args.out:
        from capcommon import jsonio  # noqa: PLC0415
        jsonio.atomic_write_bytes(args.out, (text + "\n").encode("utf-8"))
    return exitcodes.OK if rep["status"] == "PASS" else exitcodes.CONFIG


if __name__ == "__main__":
    sys.exit(main())
