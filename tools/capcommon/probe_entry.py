# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The probed-entry rule (CAPPROBE_ENTRY): which process may switch capprobe on.

capprobe is imported by every process that imports a patched product module: the product
process the orchestrator started, but also spawn pool children, MPI workers and any other
importer, which inherit the product's environment. Only the registered entry may activate
probes. The orchestrator writes the rule for each product role it starts; capprobe matches it
against the importing process's own command line at import time.

Rule (`capkit.probe_entry/1`, one JSON object in the CAPPROBE_ENTRY environment variable):

    {"schema": "capkit.probe_entry/1", "program": "<absolute realpath>", "args": ["...", ...]}

`program` is the realpath of the executable the orchestrator started (argv[0] resolved against
the role's PATH, or against the role's cwd when it contains a slash); `args` is the exact rest
of that argv. A process matches when its command line (sys.orig_argv, or /proc/self/cmdline
before Python 3.10) ends with `args` and the element just before them resolves to `program`.
Position 0 covers a native or `python script.py` start; a later position covers
a `#!` script, whose interpreter the kernel puts first (`python3 /usr/local/bin/trtllm-serve
...`). A re-exec of the same command line (the multi-frontend launcher re-runs
`[sys.executable] + sys.argv`) therefore matches; a spawn child (`python -c 'from
multiprocessing.spawn ...' --multiprocessing-fork`), an MPI worker or another tool does not.
Only elements that contain a slash (or position 0, resolved through PATH) are resolved.
"""

import json
import os
import shutil
import sys

ENV_VAR = "CAPPROBE_ENTRY"
SCHEMA = "capkit.probe_entry/1"
MATCH_RULE = ("the process command line ends with exactly `args`, and the element just before "
              "them resolves (realpath; PATH lookup only for a bare name at position 0) to "
              "`program`")


class EntryRuleError(ValueError):
    pass


def resolve_program(name, cwd, path):
    """realpath of an executable named in an argv: a name with a slash is taken relative to
    `cwd`; a bare name is looked up in `path` (a PATH string). None when it cannot be found."""
    if "/" in name:
        return os.path.realpath(name if os.path.isabs(name) else os.path.join(cwd, name))
    found = shutil.which(name, path=path)
    return None if found is None else os.path.realpath(found)


def build_rule(argv, cwd, path):
    """Rule for a process started with `argv` in `cwd` with PATH `path` (the orchestrator)."""
    if not isinstance(argv, (list, tuple)) or not argv or \
            any(not isinstance(a, str) for a in argv):
        raise EntryRuleError("argv must be a non-empty list of strings")
    prog = resolve_program(argv[0], cwd, path)
    if prog is None:
        raise EntryRuleError("cannot resolve the entry program %r on PATH" % argv[0])
    return {"schema": SCHEMA, "program": prog, "args": list(argv[1:])}


def encode(rule):
    return json.dumps(rule, sort_keys=True, separators=(",", ":"))


def parse(raw):
    """Decode and validate a CAPPROBE_ENTRY value; raises EntryRuleError."""
    try:
        doc = json.loads(raw)
    except (TypeError, ValueError) as e:
        raise EntryRuleError("%s is not JSON: %s" % (ENV_VAR, e)) from None
    if not isinstance(doc, dict) or set(doc) != {"schema", "program", "args"} \
            or doc["schema"] != SCHEMA:
        raise EntryRuleError("%s must be exactly {schema: %s, program, args}" % (ENV_VAR, SCHEMA))
    prog, args = doc["program"], doc["args"]
    if not isinstance(prog, str) or not os.path.isabs(prog) or os.path.normpath(prog) != prog:
        raise EntryRuleError("%s program must be a normalised absolute path, got %r"
                             % (ENV_VAR, prog))
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


def match(rule, argv, path=None):
    """Index i at which `argv` matches `rule` (see module doc), or None."""
    i = len(argv) - 1 - len(rule["args"])  # `args` is the exact tail, so i is fixed
    if i < 0 or list(argv[i + 1:]) != rule["args"]:
        return None
    elem = argv[i]
    if "/" in elem:
        got = os.path.realpath(elem)
    elif i == 0:
        found = shutil.which(elem, path=os.environ.get("PATH", os.defpath) if path is None
                             else path)
        got = None if found is None else os.path.realpath(found)
    else:
        got = None
    return i if got == rule["program"] else None
