# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recompute record of an analysis product (DATA-AND-ANALYSIS 4 rule 8; INTERFACES 14).

Every analysis product `<name>.json` gets `<name>.cmd.txt` next to it: the module, the code
sha, the command as invoked (`invoked`: the interpreter and sys.argv verbatim, e.g. the
analysis/*.py wrapper) and its `python -m` equivalent with the parsed arguments (`command`, run
with `pythonpath`), every input file with its sha256, and the exit code. Both files are new
(never overwritten); the record is written only after the product, so a product without its
record is an interrupted run.
"""

import os
import shlex
import sys

from capcommon.clock import wall_iso
from capcommon.hashing import sha256_file
from capcommon.jsonio import atomic_write_bytes


def cmd_path(out):
    """`<dir>/<name>.json` -> `<dir>/<name>.cmd.txt` (any other suffix: `<out>.cmd.txt`)."""
    base = out[:-len(".json")] if out.endswith(".json") else out
    return base + ".cmd.txt"


def refuse_existing(out):
    """Error text when the product or its record already exists, else None."""
    for p in (out, cmd_path(out)):
        if os.path.lexists(p):
            return "%s exists (never overwritten)" % p
    return None


def write(out, module, argv, inputs, code=0):
    """Write the record of `out`. inputs: list of paths whose sha256 the product depends on."""
    from .judge import code_sha  # noqa: PLC0415  (judge imports the whole package)
    sha, note = code_sha()
    lines = ["product\t%s" % os.path.abspath(out),
             "product_sha256\t%s" % sha256_file(out),
             "module\t%s" % module,
             "code_sha\t%s" % (sha if sha else "null (%s)" % note),
             "cwd\t%s" % os.getcwd(),
             "pythonpath\t%s" % os.environ.get("PYTHONPATH", ""),
             "invoked\t%s" % " ".join(shlex.quote(x) for x in [sys.executable] + sys.argv),
             "command\t%s" % " ".join(shlex.quote(x) for x in
                                      [sys.executable, "-m", module] + list(argv)),
             "exit_code\t%d" % code,
             "written_at\t%s" % wall_iso()]
    for p in inputs:
        lines.append("input\t%s\t%s" % (os.path.abspath(p), sha256_file(p)))
    atomic_write_bytes(cmd_path(out), ("\n".join(lines) + "\n").encode("utf-8"))
