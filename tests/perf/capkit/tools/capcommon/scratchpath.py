# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard for deleting a job's node-local scratch directory (INTERFACES.md 5.9; plan CONSTRAINTS 6).

Only the job's inner script deletes, only after copy-back verification passed, and only a path
that this check accepts. Usage from bash (exit 0 = may delete; anything else = do not delete):

    python3 -m capcommon.scratchpath --path "${LOCAL_ROOT:?}" --base "${LOCAL_BASE:?}" \\
        --user "${USER:?}" --job-id "${SLURM_JOB_ID:?}" --verified-file "${VERIFY_JSON:?}"
"""

import argparse
import json
import os
import sys

from .exitcodes import CONFIG, OK, ConfigError


class ScratchPathError(ConfigError):
    pass


def assert_job_scratch_path(path, base, user, job_id):
    """Accept only <base>/<user>-<job_id> exactly (normalised, absolute, not a symlink)."""
    for name, v in (("path", path), ("base", base), ("user", user), ("job_id", job_id)):
        if not isinstance(v, str) or not v:
            raise ScratchPathError("%s must be a non-empty string" % name)
    if "/" in user or "/" in job_id or user in (".", "..") or job_id in (".", ".."):
        raise ScratchPathError("user and job_id must be plain names")
    if not os.path.isabs(path) or not os.path.isabs(base):
        raise ScratchPathError("path and base must be absolute")
    want = os.path.join(os.path.normpath(base), "%s-%s" % (user, job_id))
    if path != want:  # exact string: no trailing slash, no '..', no doubled separators
        raise ScratchPathError("path %r is not exactly %r" % (path, want))
    if os.path.islink(path):
        raise ScratchPathError("path %r is a symlink" % path)
    return want


def check_verified(verified_file):
    """The copy-back verification record must exist and say passed: true."""
    try:
        with open(verified_file, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise ScratchPathError("cannot read copy-back verification %s: %s" % (verified_file, e))
    if not isinstance(doc, dict) or doc.get("passed") is not True:
        raise ScratchPathError("copy-back verification did not pass: %s" % verified_file)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capcommon.scratchpath")
    ap.add_argument("--path", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--verified-file", required=True)
    a = ap.parse_args(argv)
    try:
        assert_job_scratch_path(a.path, a.base, a.user, a.job_id)
        check_verified(a.verified_file)
    except ConfigError as e:
        print("scratchpath: refuse: %s" % e, file=sys.stderr)
        return CONFIG
    print("scratchpath: ok %s" % a.path)
    return OK


if __name__ == "__main__":
    sys.exit(main())
