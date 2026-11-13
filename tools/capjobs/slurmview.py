# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Front-end fallback for RESOURCES section 3 items 1-3 (INTERFACES.md 5.5.2):
<job dir>/env/slurm_view.json, schema capkit.slurm_view/1.

Item 1 = scontrol show job J; item 2 = scontrol show node <NodeList of item 1>;
item 3 = squeue -h -w <node> -o '%i %u %T'. Raw stdout is stored verbatim with its sha256; the
reader (capenv.gate) does the judging. When item 1 yields no single node, items 2-3 are recorded
with rc null and a reason (not silently skipped).
"""

import os
import re

from capcommon.clock import wall_iso
from capcommon.hashing import sha256_bytes
from capcommon.jsonio import atomic_write_bytes, dumps

from .remote import run_remote

SCHEMA = "capkit.slurm_view/1"
NODELIST_RE = re.compile(r"(?:^|\s)NodeList=(\S+)")
SINGLE_NODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def node_of(scontrol_job_text):
    """The single node name in 'NodeList=' of `scontrol show job`, or None."""
    found = NODELIST_RE.findall(scontrol_job_text or "")
    if len(found) != 1:
        return None
    n = found[0]
    return n if SINGLE_NODE_RE.match(n) and n != "(null)" else None


def _item(i, res):
    out = res["stdout"] if res["stdout"] is not None else ""
    return {"id": i, "cmd": res["remote_argv"], "rc": res["rc"], "stdout": out,
            "stdout_sha256": sha256_bytes(out.encode("utf-8")), "stderr": res["stderr"],
            "taken_wall": wall_iso()}


def collect(runner, login_host, job_id):
    items = []
    r1 = run_remote(runner, login_host, ["scontrol", "show", "job", str(job_id)])
    items.append(_item(1, r1))
    node = node_of(r1["stdout"]) if r1["rc"] == 0 else None
    if node is None:
        for i, cmd in ((2, ["scontrol", "show", "node", "<unknown>"]),
                       (3, ["squeue", "-h", "-w", "<unknown>", "-o", "%i %u %T"])):
            items.append({"id": i, "cmd": cmd, "rc": None, "stdout": "", "stdout_sha256": None,
                          "stderr": "not run: item 1 gave no single NodeList", "taken_wall":
                          wall_iso()})
    else:
        items.append(_item(2, run_remote(runner, login_host, ["scontrol", "show", "node", node])))
        items.append(_item(3, run_remote(runner, login_host,
                                         ["squeue", "-h", "-w", node, "-o", "%i %u %T"])))
    return {"schema": SCHEMA, "job_id": str(job_id), "node": node, "items": items}


def write(job_dir, doc):
    if not os.path.isdir(job_dir):
        raise FileNotFoundError("job dir %s does not exist (the job creates it)" % job_dir)
    env = os.path.join(job_dir, "env")
    os.makedirs(env, exist_ok=True)
    path = os.path.join(env, "slurm_view.json")
    atomic_write_bytes(path, (dumps(doc) + "\n").encode("utf-8"))
    return path
