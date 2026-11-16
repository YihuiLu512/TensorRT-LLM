# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The probed-entry rule (CAPPROBE_ENTRY): which process may switch capprobe on.

capprobe is imported by every process that imports a patched product module: the product
process the orchestrator started, but also spawn pool children, MPI workers and any other
importer, which inherit the product's environment. Only the registered entry may activate
probes. The orchestrator writes the rule for each product role it starts; capprobe matches it
against the importing process's own command line at import time.

Rule (`capkit.probe_entry/1`, one JSON object in the CAPPROBE_ENTRY environment variable):

    {"schema": "capkit.probe_entry/1", "program": "<absolute realpath>",
     "args": ["...", ...], "cwd": "<absolute realpath>"}

`cwd` is the realpath of the directory the role was started in; `program` is the realpath of
argv[0] (a name with a slash is taken relative to `cwd`; a bare name is looked up on the role's
PATH, whose relative entries are also taken relative to `cwd`, as exec does after the chdir);
`args` is the exact rest of that argv.

A process command line (sys.orig_argv, or /proc/self/cmdline before Python 3.10) matches in
either of two forms. In both, "is `program`" means: the element resolves to `program`, resolved
as above against the rule's `cwd` (never the process's current directory, which the product may
have changed before importing capprobe). Only an element with a slash, or a bare name at
position 0 (looked up on PATH), is resolved; a bare name later in the line never is.

1. Direct: the line ends with `args` and the element just before them is `program`. Position 0
   is a native, `python script.py` or `python -m mod` start; a later position is a `#!` script,
   whose interpreter the kernel puts first (`python3 /usr/local/bin/trtllm-serve ...`). A re-exec
   `[sys.executable] + sys.argv` of a script start puts the same script path before the same
   args, so it matches.
2. Re-executed module: only when `args` starts with the two elements `-m`, `<mod>`. Python
   replaces them by the module's file in sys.argv, so the multi-frontend launcher's re-exec
   `[sys.executable] + sys.argv` runs `python <file> <rest>`. The line matches when it ends with
   <rest> (the args after `-m <mod>`), the element before <rest> is a path ending in
   `<mod as a/b/c>.py` or `<a/b/c>/__main__.py` (by name only; nothing is imported), and the
   element before that is `program`. Interpreter options before `-m` (`python -u -m mod`) are
   not part of sys.argv either, so such a re-exec does not match.

A spawn child (`python -c 'from multiprocessing.spawn ...' --multiprocessing-fork`), an MPI
worker or another tool does not match.
"""

import json
import os
import shutil
import sys

ENV_VAR = "CAPPROBE_ENTRY"
SCHEMA = "capkit.probe_entry/1"
MATCH_RULE = ("the process command line ends with exactly `args` and the element just before "
              "them resolves to `program`; or, when `args` starts with `-m <mod>`, it ends with "
              "the args after `-m <mod>`, the element before them is a path ending in "
              "<mod>.py or <mod>/__main__.py, and the element before that resolves to "
              "`program`. Resolution: realpath against the rule's `cwd`; PATH lookup only for a "
              "bare name at position 0")


class EntryRuleError(ValueError):
    pass


def _abs_path_list(path, cwd):
    """PATH string (None: this process's PATH) with every relative (or empty) entry made
    absolute against `cwd`, the way exec resolves them after the chdir to `cwd`."""
    path = os.environ.get("PATH", os.defpath) if path is None else path
    return os.pathsep.join(e if os.path.isabs(e) else os.path.join(cwd, e)
                           for e in path.split(os.pathsep))


def resolve_program(name, cwd, path):
    """realpath of an executable named in an argv of a process started in `cwd` with PATH
    `path`: a name with a slash is taken relative to `cwd`; a bare name is looked up in `path`
    (relative entries against `cwd`). None when it cannot be found."""
    if "/" in name:
        return os.path.realpath(os.path.join(cwd, name))
    found = shutil.which(name, path=_abs_path_list(path, cwd))
    return None if found is None else os.path.realpath(found)


def build_rule(argv, cwd, path):
    """Rule for a process started with `argv` in `cwd` with PATH `path` (the orchestrator)."""
    if not isinstance(argv, (list, tuple)) or not argv or \
            any(not isinstance(a, str) for a in argv):
        raise EntryRuleError("argv must be a non-empty list of strings")
    if not isinstance(cwd, str) or not cwd:
        raise EntryRuleError("cwd must be a non-empty string")
    cwd = os.path.realpath(cwd)
    prog = resolve_program(argv[0], cwd, path)
    if prog is None:
        raise EntryRuleError("cannot resolve the entry program %r on PATH" % argv[0])
    return {"schema": SCHEMA, "program": prog, "args": list(argv[1:]), "cwd": cwd}


def encode(rule):
    return json.dumps(rule, sort_keys=True, separators=(",", ":"))


def _normalised_abs(value):
    return isinstance(value, str) and os.path.isabs(value) and os.path.normpath(value) == value


def parse(raw):
    """Decode and validate a CAPPROBE_ENTRY value; raises EntryRuleError."""
    try:
        doc = json.loads(raw)
    except (TypeError, ValueError) as e:
        raise EntryRuleError("%s is not JSON: %s" % (ENV_VAR, e)) from None
    if not isinstance(doc, dict) or set(doc) != {"schema", "program", "args", "cwd"} \
            or doc["schema"] != SCHEMA:
        raise EntryRuleError("%s must be exactly {schema: %s, program, args, cwd}"
                             % (ENV_VAR, SCHEMA))
    for key in ("program", "cwd"):
        if not _normalised_abs(doc[key]):
            raise EntryRuleError("%s %s must be a normalised absolute path, got %r"
                                 % (ENV_VAR, key, doc[key]))
    args = doc["args"]
    if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
        raise EntryRuleError("%s args must be a list of strings" % ENV_VAR)
    return doc


def process_argv():
    """This process's command line as the kernel started it (not sys.argv, which a spawn child
    replaces with its parent's)."""
    orig = getattr(sys, "orig_argv", None)
    if orig:
        return list(orig)
    try:
        with open("/proc/self/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return []
    return [a.decode("utf-8", "surrogateescape") for a in raw.split(b"\0")[:-1]]


def _tail_index(argv, tail):
    """Index of the element just before `tail` when `argv` ends with exactly `tail`, else None."""
    i = len(argv) - 1 - len(tail)
    if i < 0 or list(argv[i + 1:]) != list(tail):
        return None
    return i


def _is_program(argv, i, rule, path):
    elem = argv[i]
    if "/" not in elem and i != 0:
        return False
    return resolve_program(elem, rule["cwd"], path) == rule["program"]


def _module_form(args):
    """(module path a/b/c, rest) when `args` is `-m <dotted module> rest...`, else None."""
    if len(args) < 2 or args[0] != "-m":
        return None
    parts = args[1].split(".")
    if not all(p.isidentifier() for p in parts):
        return None
    return "/".join(parts), args[2:]


def _is_module_file(elem, mod_path):
    p = os.path.normpath(elem)
    return any(p == s or p.endswith("/" + s)
               for s in (mod_path + ".py", mod_path + "/__main__.py"))


def match(rule, argv, path=None):
    """Index in `argv` of the element that matched `program` (see module doc), or None. `path`
    is the PATH for a bare name at position 0 (default: this process's PATH)."""
    i = _tail_index(argv, rule["args"])
    if i is not None and _is_program(argv, i, rule, path):
        return i
    form = _module_form(rule["args"])
    if form is not None:
        mod_path, rest = form
        j = _tail_index(argv, rest)
        if j is not None and j >= 1 and _is_module_file(argv[j], mod_path) \
                and _is_program(argv, j - 1, rule, path):
            return j - 1
    return None
