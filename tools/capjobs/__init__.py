# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capjobs: job scaffolding for capkit (INTERFACES.md 5.5, 5.7, 5.9, 9.5, 11).

VM-side modules (stdlib only, python3.11): submit, remote, resources, snapshot, cdbexclude,
jobstate, slurmview. Container-side modules (run inside the job's container): overlay, envfiles,
launches, copyback. Every module depends only on the standard library and capcommon.
"""
