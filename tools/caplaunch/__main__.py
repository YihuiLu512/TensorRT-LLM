# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m caplaunch: one launch (INTERFACES.md 11, caplaunch row; README.md for extras)."""

import argparse
import sys

from capcommon.cli import add_identity_args
from capcommon.exitcodes import CapkitError


def build_parser():
    p = argparse.ArgumentParser(prog="python -m caplaunch", description=__doc__)
    add_identity_args(p)
    g = p.add_argument_group("launch (all required; provenance values may be 'none')")
    g.add_argument("--run-dir", required=True, help="launch tree <T> on the node-local disk")
    g.add_argument("--workdir", required=True, help="<W>; every relpath is relative to it")
    g.add_argument("--register", required=True, help="register.json (relpath to <W> or abs)")
    g.add_argument("--scenario", required=True, help="scenario config (inside <W>)")
    g.add_argument("--pin-table", required=True, help="pin_table.json from capenv.pins")
    g.add_argument("--shm-dir", required=True,
                   help="job-level snapshot root /dev/shm/<user>-<job>/snap; the launch "
                        "creates r<restart>-L<seq>/ under it")
    g.add_argument("--snapshot-interval-ms", type=int, required=True)
    g.add_argument("--origin-lead-ms", type=int, required=True,
                   help="origin = now + lead, chosen before the first process starts; every "
                        "role must be ready before it")
    g.add_argument("--housekeeping-ms", type=int, required=True,
                   help="period of readiness / discovery / liveness checks")
    g.add_argument("--kill-after-s", type=float, required=True,
                   help="after stop grace, wait this long before SIGKILL")
    g.add_argument("--capkit-sha", required=True)
    g.add_argument("--probe-tree-sha", required=True)
    g.add_argument("--baseline-sha", required=True)
    g.add_argument("--snapshot-dir", required=True, help="script/code snapshot path of the job")
    g.add_argument("--image-path", required=True)
    g.add_argument("--image-sha256", required=True)
    g.add_argument("--hw-profile-sha", required=True)
    g.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="scenario parameter; the set of keys must equal params_required")
    g.add_argument("--nvidia-smi", required=True,
                   help="nvidia-smi executable (absolute path); fills the {nvidia_smi} "
                        "placeholder of scenario arguments (the sidecar's GPU queries)")
    g.add_argument("--overlay-manifest", required=True,
                   help="job-level overlay manifest (<snap>/overlay_manifest.json), copied into "
                        "raw/env/; when it does not exist and a role has a probe tier, the "
                        "launch stops as NOT_RUN before starting any process")
    g.add_argument("--job-env-dir", required=True,
                   help="job-level env/ directory (gate results, gpu_idle.json, ...); its files "
                        "are copied into raw/env/. Not a directory -> the launch stops as NOT_RUN "
                        "before starting any process")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .orchestrator import Launch  # noqa: PLC0415
    try:
        launch = Launch(args)
        return launch.run()
    except CapkitError as e:
        print("caplaunch: %s" % e, file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
