# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware profile keys and checksum (INTERFACES.md 5.5.3; plan DATA-AND-ANALYSIS 8.1).

The profile has exactly IDENTITY_FIELDS + OBSERVED_FIELDS. Only identity fields enter
profile_sha256: they describe the hardware combination ("same model" = equal sha). Observed
fields change between jobs on the same node (time, free memory, current frequency, partition,
link counters) and are recorded but never hashed, otherwise the gate could never pass.

Every value is either the collected value or {"unreadable": "<reason>"}; an unreadable marker of
an identity field is part of the sha (two nodes that both cannot read a field still compare
equal on it; a node that can read it does not).
"""

from .exitcodes import ConfigError
from .hashing import canonical_json_sha256

SCHEMA = "capkit.hw_profile/1"

# key -> how it is collected (documentation for capenv.hwprofile; values are normalised as noted)
IDENTITY_FIELDS = {
    "chassis_model": "slurm_view.json node SystemProductName / system_product_name (string)",
    "cpu_arch": "lscpu Architecture (string)",
    "cpu_model": "lscpu Model name (string)",
    "sockets": "lscpu Socket(s) (int)",
    "cores_per_socket": "lscpu Core(s) per socket (int)",
    "threads_per_core": "lscpu Thread(s) per core (int)",
    "microcode": "/proc/cpuinfo microcode of cpu0 (string)",
    "numa_layout": "list of {node:int, cpus:'a-b,c'} from /sys/devices/system/node",
    "cpu_governor": "sorted unique scaling_governor values (list of strings)",
    "mem_total_gib": "MemTotal rounded to the nearest GiB (int)",
    "mem_per_numa_gib": "list of per-node MemTotal rounded to GiB, by node (list of int)",
    "gpu_model": "nvidia-smi name; single string if all equal, else the sorted list",
    "gpu_count": "nvidia-smi line count (int)",
    "gpu_mem_mib": "nvidia-smi memory.total in MiB, sorted list of int",
    "gpu_driver": "nvidia-smi driver_version (string)",
    "nics": "sorted list of {model, speed_mbps} for physical NICs (no interface names)",
    "local_disk_type": "lsblk ROTA/TRAN/MODEL of the disk holding the local run dir (string)",
    "kernel": "uname -r (string)",
    "container_runtime": "enroot/pyxis version string as the job script recorded it",
    "image_sha256": "sha256 of the container image file",
    "code_baseline": "TRT-LLM baseline commit (40 hex)",
}
OBSERVED_FIELDS = {
    "node_name": "hostname / SLURMD_NODENAME",
    "partition": "slurm_view.json job Partition",
    "collected_at": "ISO8601 UTC (clock.wall_iso)",
    "collected_mono_ns": "time.monotonic_ns()",
    "cpu_freq_khz": "per-cpu scaling_cur_freq at collection time",
    "mem_available_kib": "MemAvailable",
    "local_disk_free_bytes": "statvfs of the local run dir",
    "nic_link": "per-interface operstate and rx/tx byte counters",
    "sources": "per-key: where the value came from (container / slurm_view / unreadable)",
}


class HwProfileError(ConfigError):
    pass


def unreadable(reason):
    return {"unreadable": str(reason)}


def is_unreadable(v):
    return isinstance(v, dict) and set(v) == {"unreadable"}


def validate_profile(profile):
    """The profile dict must have exactly the identity + observed keys."""
    if not isinstance(profile, dict):
        raise HwProfileError("profile must be an object")
    want = set(IDENTITY_FIELDS) | set(OBSERVED_FIELDS)
    missing, extra = sorted(want - set(profile)), sorted(set(profile) - want)
    if missing or extra:
        raise HwProfileError("profile keys missing=%s extra=%s" % (missing, extra))
    return profile


def identity_part(profile):
    validate_profile(profile)
    return {k: profile[k] for k in IDENTITY_FIELDS}


def profile_sha256(profile):
    """canonical_json_sha256 over the identity fields only."""
    return canonical_json_sha256(identity_part(profile))


def diff_identity(a, b):
    """Identity keys whose values differ between two profiles (for the comparability record)."""
    ia, ib = identity_part(a), identity_part(b)
    return sorted(k for k in IDENTITY_FIELDS if ia[k] != ib[k])
