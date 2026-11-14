# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Front-end fallback for RESOURCES section 3 items 1-3 (INTERFACES.md 5.5.2):
<job dir>/env/slurm_view.json, schema capkit.slurm_view/1.

Item 1 = scontrol show job J; item 2 = scontrol show node <NodeList of item 1>;
item 3 = squeue -h -w <node> -o '%i %u %T'. Raw stdout is stored verbatim with its sha256; the
reader (capenv.gate) does the judging. When item 1 yields no single node, items 2-3 are recorded
with rc null, attempts 0 and a reason (not silently skipped).

Each query is retried up to `attempts` times while it fails (rc != 0, or null = ssh could not start
or timed out), waiting retry_wait_s * k before try k + 1; the item keeps the last try and lists the
rcs of the earlier ones. A query that still fails after the last try leaves the view incomplete:
failed_items() names it and the caller does not write the file, so the next call can try again
(the job waits up to slurm_view_wait_s for the file).
"""

import os
import re
import time

from capcommon.clock import wall_iso
from capcommon.exitcodes import ConfigError
from capcommon.hashing import sha256_bytes
from capcommon.jsonio import atomic_write_bytes, dumps

from .remote import run_remote

SCHEMA = "capkit.slurm_view/1"
NODELIST_RE = re.compile(r"(?:^|\s)NodeList=(\S+)")
SINGLE_NODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_ATTEMPTS = 3
DEFAULT_RETRY_WAIT_S = 2.0
MAX_ATTEMPTS = 10
MAX_RETRY_WAIT_S = 60.0


def node_of(scontrol_job_text):
    """The single node name in 'NodeList=' of `scontrol show job`, or None."""
    found = NODELIST_RE.findall(scontrol_job_text or "")
    if len(found) != 1:
        return None
    n = found[0]
    return n if SINGLE_NODE_RE.match(n) and n != "(null)" else None


def check_retry(attempts, retry_wait_s):
    if isinstance(attempts, bool) or not isinstance(attempts, int) \
            or not 1 <= attempts <= MAX_ATTEMPTS:
        raise ConfigError("attempts must be an integer in [1, %d]" % MAX_ATTEMPTS)
    if isinstance(retry_wait_s, bool) or not isinstance(retry_wait_s, (int, float)) \
            or not 0 <= retry_wait_s <= MAX_RETRY_WAIT_S:
        raise ConfigError("retry wait must be in [0, %g] s" % MAX_RETRY_WAIT_S)


def _query(runner, login_host, remote_argv, attempts, retry_wait_s, sleep):
    """run_remote with bounded retries; returns (last result, rcs of the earlier failed tries)."""
    earlier = []
    for k in range(1, attempts + 1):
        res = run_remote(runner, login_host, remote_argv)
        if res["rc"] == 0 or k == attempts:
            return res, earlier
        earlier.append(res["rc"])
        sleep(retry_wait_s * k)
    raise AssertionError("unreachable")


def _item(i, res, earlier):
    out = res["stdout"] if res["stdout"] is not None else ""
    return {"id": i, "cmd": res["remote_argv"], "rc": res["rc"], "stdout": out,
            "stdout_sha256": sha256_bytes(out.encode("utf-8")), "stderr": res["stderr"],
            "taken_wall": wall_iso(), "attempts": len(earlier) + 1, "retried_rcs": earlier}


def failed_items(doc):
    """Items whose query ran and still failed after its last try (not the 'not run' ones)."""
    return [it for it in doc["items"] if it["attempts"] > 0 and it["rc"] != 0]


def collect(runner, login_host, job_id, attempts, retry_wait_s, sleep=time.sleep):
    check_retry(attempts, retry_wait_s)

    def query(i, remote_argv):
        res, earlier = _query(runner, login_host, remote_argv, attempts, retry_wait_s, sleep)
        return _item(i, res, earlier), res

    items = []
    it1, r1 = query(1, ["scontrol", "show", "job", str(job_id)])
    items.append(it1)
    node = node_of(r1["stdout"]) if r1["rc"] == 0 else None
    if node is None:
        for i, cmd in ((2, ["scontrol", "show", "node", "<unknown>"]),
                       (3, ["squeue", "-h", "-w", "<unknown>", "-o", "%i %u %T"])):
            items.append({"id": i, "cmd": cmd, "rc": None, "stdout": "", "stdout_sha256": None,
                          "stderr": "not run: item 1 gave no single NodeList", "taken_wall":
                          wall_iso(), "attempts": 0, "retried_rcs": []})
    else:
        items.append(query(2, ["scontrol", "show", "node", node])[0])
        items.append(query(3, ["squeue", "-h", "-w", node, "-o", "%i %u %T"])[0])
    return {"schema": SCHEMA, "job_id": str(job_id), "node": node, "items": items}


def write(job_dir, doc):
    if not os.path.isdir(job_dir):
        raise FileNotFoundError("job dir %s does not exist (the job creates it)" % job_dir)
    env = os.path.join(job_dir, "env")
    os.makedirs(env, exist_ok=True)
    path = os.path.join(env, "slurm_view.json")
    atomic_write_bytes(path, (dumps(doc) + "\n").encode("utf-8"))
    return path
