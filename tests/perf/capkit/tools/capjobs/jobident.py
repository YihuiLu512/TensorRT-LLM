# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identity for the job-level files the container-side modules write (INTERFACES.md 2.1, 5.5).

Job-level files belong to no single launch; they carry the plan's job_arm and launch_seq 0, both
passed explicitly by the inner script (see jobs/README.md, open issue on job-level identity).
"""

from capcommon.cli import add_identity_args
from capcommon.identity import Identity


def add_args(parser):
    add_identity_args(parser)


def from_args(a):
    return Identity(experiment=a.experiment, arm=a.arm, launch_seq=a.launch_seq,
                    job_id=a.job_id, restart_count=a.restart_count,
                    register_sha=a.register_sha, role=a.role, worker=a.worker).with_pid()
