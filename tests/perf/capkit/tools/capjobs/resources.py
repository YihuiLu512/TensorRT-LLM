# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource arguments for sbatch (plan RESOURCES section 4; CONSTRAINTS section 2 item 2).

Written new: nothing here is inherited from the WT-21 submit.py resource block. Every value is a
required parameter without a default; the builder only validates and formats. --exclusive,
--nodes=1 and --ntasks=1 are always set (whole-node, single-task jobs).
"""

import re

from capcommon.exitcodes import ConfigError
from capcommon.policy import MAX_GPUS

ACCOUNT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
QOS_RE = ACCOUNT_RE
PARTITION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9@+/._-]{0,127}$")
NODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
TIME_RE = re.compile(r"^(?:(\d{1,2})-)?(\d{1,2}):(\d{2}):(\d{2})$")
MEM_RE = re.compile(r"^[1-9][0-9]{0,8}[KMGT]?$")
GPU_TYPE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# The literal that means "no node is excluded" (explicit, so --exclude has no default).
EXCLUDE_NONE = "none"


class ResourceError(ConfigError):
    pass


def parse_time_s(text):
    """Slurm time limit [D-]HH:MM:SS -> seconds."""
    m = TIME_RE.match(text or "")
    if not m:
        raise ResourceError("time must be [D-]HH:MM:SS, got %r" % (text,))
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    if mi > 59 or s > 59:
        raise ResourceError("bad minutes/seconds in %r" % text)
    total = ((d * 24 + h) * 60 + mi) * 60 + s
    if total <= 0:
        raise ResourceError("time limit must be > 0")
    return total


def split_list(text, rx, what):
    items = [x for x in (text or "").split(",")]
    if not items or any(not rx.match(x) for x in items):
        raise ResourceError("bad %s list %r" % (what, text))
    if len(set(items)) != len(items):
        raise ResourceError("duplicate entries in %s list %r" % (what, text))
    return items


def normalize_exclude(text):
    """'none' -> [] (explicitly nothing excluded); otherwise a validated node list."""
    if text == EXCLUDE_NONE:
        return []
    return split_list(text, NODE_RE, "exclude")


def validate(res):
    """res: dict with account, partition, qos, exclude (list), time, mem, cpus_per_task, gpus,
    gpu_type (str or None), job_name. Returns a normalised copy or raises ResourceError."""
    need = ("account", "partition", "qos", "exclude", "time", "mem", "cpus_per_task", "gpus",
            "gpu_type", "job_name")
    missing = [k for k in need if k not in res]
    if missing:
        raise ResourceError("missing resource fields %s" % missing)
    out = dict(res)
    if not ACCOUNT_RE.match(res["account"] or ""):
        raise ResourceError("bad account %r" % (res["account"],))
    if not QOS_RE.match(res["qos"] or ""):
        raise ResourceError("bad qos %r" % (res["qos"],))
    out["partition"] = ",".join(split_list(res["partition"], PARTITION_RE, "partition"))
    if not isinstance(res["exclude"], list) or any(not NODE_RE.match(n) for n in res["exclude"]):
        raise ResourceError("exclude must be a validated node list")
    out["time_s"] = parse_time_s(res["time"])
    if not MEM_RE.match(res["mem"] or ""):
        raise ResourceError("mem must be like 1900G, got %r" % (res["mem"],))
    c = res["cpus_per_task"]
    if not isinstance(c, int) or isinstance(c, bool) or c <= 0:
        raise ResourceError("cpus_per_task must be an int > 0")
    g = res["gpus"]
    if not isinstance(g, int) or isinstance(g, bool) or not 0 <= g <= MAX_GPUS:
        raise ResourceError("gpus must be an int in [0, %d]" % MAX_GPUS)
    if res["gpu_type"] is not None and not GPU_TYPE_RE.match(res["gpu_type"]):
        raise ResourceError("bad gpu type %r" % (res["gpu_type"],))
    if res["gpu_type"] is not None and g == 0:
        raise ResourceError("--gpu-type given with --gpus 0")
    if not JOB_NAME_RE.match(res["job_name"] or ""):
        raise ResourceError("bad job name %r" % (res["job_name"],))
    return out


def resource_flags(res):
    """sbatch flags for validated resources (no script, no --parsable/--test-only)."""
    r = validate(res)
    flags = ["--job-name=" + r["job_name"], "--account=" + r["account"],
             "--partition=" + r["partition"], "--qos=" + r["qos"], "--time=" + r["time"],
             "--mem=" + r["mem"], "--cpus-per-task=%d" % r["cpus_per_task"],
             "--nodes=1", "--ntasks=1", "--exclusive"]
    if r["exclude"]:
        flags.append("--exclude=" + ",".join(r["exclude"]))
    if r["gpus"] > 0:
        if r["gpu_type"]:
            flags.append("--gres=gpu:%s:%d" % (r["gpu_type"], r["gpus"]))
        else:
            flags.append("--gres=gpu:%d" % r["gpus"])
    return flags


def prep_tokenizer_flags(res):
    """sbatch flags of the tokenizer-copy job: the smallest request that works. No GPU (no
    --gres), no --exclusive, one task on one node; --partition only when given (None = let the
    scheduler pick, the global partition rule) and --exclude only when non-empty.
    res: account, qos, time, mem, cpus_per_task, job_name, partition (str or None), exclude."""
    need = ("account", "qos", "time", "mem", "cpus_per_task", "job_name", "partition",
            "exclude")
    missing = [k for k in need if k not in res]
    if missing:
        raise ResourceError("missing resource fields %s" % missing)
    if not ACCOUNT_RE.match(res["account"] or "") or not QOS_RE.match(res["qos"] or ""):
        raise ResourceError("bad account %r or qos %r" % (res["account"], res["qos"]))
    parse_time_s(res["time"])
    if not MEM_RE.match(res["mem"] or ""):
        raise ResourceError("mem must be like 4G, got %r" % (res["mem"],))
    c = res["cpus_per_task"]
    if not isinstance(c, int) or isinstance(c, bool) or c <= 0:
        raise ResourceError("cpus_per_task must be an int > 0")
    if not JOB_NAME_RE.match(res["job_name"] or ""):
        raise ResourceError("bad job name %r" % (res["job_name"],))
    if not isinstance(res["exclude"], list) or any(not NODE_RE.match(n) for n in res["exclude"]):
        raise ResourceError("exclude must be a validated node list")
    flags = ["--job-name=" + res["job_name"], "--account=" + res["account"],
             "--qos=" + res["qos"], "--time=" + res["time"], "--mem=" + res["mem"],
             "--cpus-per-task=%d" % c, "--nodes=1", "--ntasks=1"]
    if res["partition"] is not None:
        flags.append("--partition=" + ",".join(split_list(res["partition"], PARTITION_RE,
                                                          "partition")))
    if res["exclude"]:
        flags.append("--exclude=" + ",".join(res["exclude"]))
    bad = [f for f in flags if f.startswith(("--gres", "--gpus", "--exclusive"))]
    if bad:
        raise ResourceError("the tokenizer copy must not request %s" % bad)
    return flags


def submit_argv(res, log_path_pattern, script, script_args):
    """Remote argv for the real submission. Never contains --test-only."""
    argv = ["sbatch", "--parsable"] + resource_flags(res) + [
        "--output=" + log_path_pattern, "--open-mode=append", "--export=ALL", script] \
        + list(script_args)
    if "--test-only" in argv:
        raise ResourceError("submit argv must not contain --test-only")
    return argv


def probe_argv(res):
    """Remote argv for the probe: same resource flags, --test-only, a trivial wrapped script."""
    return ["sbatch", "--test-only"] + resource_flags(res) + ["--wrap=true"]
