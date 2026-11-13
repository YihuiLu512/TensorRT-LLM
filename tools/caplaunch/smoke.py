# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m caplaunch.smoke: E0(c) feasibility smokes that batch-1 code can do.

Items (config `capkit.smoke_config/1`, configs/e0/e0c-smoke.json): env premises (compat library
path, NVIDIA_VISIBLE_DEVICES, PMIX/PMI/SLURM variables; import probe with and without the compat
path as its control), switchinterval per interpreter (D13), D14 loopback via capenv.loopback,
readability (taskset, numactl, schedstat, cpufreq, cpuidle, /proc/stat softirq, memory guard),
the core budget of E4-E9 topologies from the pin table and sysfs topology, and two checks that
read a finished launch tree (uvicorn access-log lines; httpprobe empty-delta frames). The
frontend + S-FE requests themselves run as a launch (scenario e0c-frontend-http.scenario.json).
Statuses: PASS / FAIL / RECORDED (a reading, no verdict) / NOT_RUN (precondition missing).
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

from capcommon.cli import add_identity_args
from capcommon.jsonio import atomic_write_bytes, dumps

SCHEMA = "capkit.smoke/1"
CONFIG_SCHEMA = "capkit.smoke_config/1"


def _run(argv, timeout_s, env=None):
    t0 = time.monotonic_ns()
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
        return {"argv": argv, "rc": r.returncode, "stdout": r.stdout[-4000:],
                "stderr": r.stderr[-4000:], "elapsed_ns": time.monotonic_ns() - t0}
    except FileNotFoundError as e:
        return {"argv": argv, "rc": None, "error": "not found: %s" % e}
    except subprocess.TimeoutExpired:
        return {"argv": argv, "rc": None, "error": "timeout after %ss" % timeout_s}


def _readable(path):
    try:
        with open(path) as f:
            return True, f.read(200)
    except OSError as e:
        return False, str(e)


def item(id_, status, **kw):
    kw.update({"id": id_, "status": status})
    return kw


def env_premises(cfg, P):
    out = []
    compat = P.get("cuda_compat_dir")
    ld = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if compat is None:
        out.append(item("env.compat_path", "NOT_RUN", reason="param cuda_compat_dir not given"))
    else:
        ok = os.path.isdir(compat) and ld[:1] == [compat]
        out.append(item("env.compat_path", "PASS" if ok else "FAIL", compat_dir=compat,
                        exists=os.path.isdir(compat), ld_library_path=ld))
    out.append(item("env.nvidia_visible_devices", "RECORDED",
                    value=os.environ.get("NVIDIA_VISIBLE_DEVICES")))
    keys = sorted(k for k in os.environ if k.startswith(("PMIX_", "PMI_", "SLURM_")))
    out.append(item("env.mpi_slurm_vars", "RECORDED", names=keys, count=len(keys)))
    py, imp = P.get("python"), cfg.get("import_probe")
    if py is None or not imp or compat is None:
        out.append(item("env.import_probe", "NOT_RUN",
                        reason="needs params python, cuda_compat_dir and config import_probe"))
    else:
        with_env = dict(os.environ)
        without = dict(os.environ)
        without["LD_LIBRARY_PATH"] = ":".join(p for p in ld if p and p != compat)
        a = _run([py, "-c", imp["code"]], imp["timeout_s"], with_env)
        b = _run([py, "-c", imp["code"]], imp["timeout_s"], without)
        st = "PASS" if a["rc"] == 0 else "FAIL"
        out.append(item("env.import_probe", st, with_compat=a, without_compat_control=b,
                        control_informative=b["rc"] != 0))
    out.append(item("env.mpi_launch_probe", "NOT_RUN",
                    reason="needs S-NX / MpiPoolSession (batch 2)"))
    return out


def switchinterval(cfg, P):
    pys = [P[k] for k in cfg.get("interpreters", []) if k in P]
    if not pys:
        return [item("d13.switchinterval", "NOT_RUN", reason="no interpreter params given")]
    res = [_run([p, "-c", "import sys; print(repr(sys.getswitchinterval()))"], 60) for p in pys]
    st = "RECORDED" if all(r["rc"] == 0 for r in res) else "FAIL"
    return [item("d13.switchinterval", st, readings=res,
                 this_process=sys.getswitchinterval())]


def d14_loopback(cfg, P, outdir, ident_argv):
    lb = cfg.get("loopback")
    keys = ("loopback_sender_cpu", "loopback_receiver_cpu")
    if not lb or any(k not in P for k in keys):
        return [item("d14.loopback", "NOT_RUN", reason="needs config loopback and params %s"
                     % (keys,))]
    out = os.path.join(outdir, "d14_loopback.json")
    r = _run([sys.executable, "-m", "capenv.loopback"] + ident_argv + ["--sender-cpu",
              str(P["loopback_sender_cpu"]), "--receiver-cpu", str(P["loopback_receiver_cpu"]),
              "--duration-s", str(lb["duration_s"]), "--msg-bytes", str(lb["msg_bytes"]),
              "--out", out], lb["duration_s"] * 4 + 60)
    if r.get("rc") != 0 or not os.path.exists(out):
        return [item("d14.loopback", "FAIL", run=r)]
    return [item("d14.loopback", "RECORDED", run=r, out=out,
                 note="a reading for the pin table, not a verdict")]


def _own_cgroup_memory():
    found = {}
    try:
        with open("/proc/self/cgroup") as f:
            lines = f.read().splitlines()
    except OSError as e:
        return {"error": str(e)}
    for ln in lines:
        _, ctrl, path = ln.split(":", 2)
        if ctrl == "":  # cgroup v2
            ok, val = _readable("/sys/fs/cgroup%s/memory.max" % path)
            found["v2_memory.max"] = val.strip() if ok else None
        elif "memory" in ctrl.split(","):
            ok, val = _readable("/sys/fs/cgroup/memory%s/memory.limit_in_bytes" % path)
            found["v1_memory.limit_in_bytes"] = val.strip() if ok else None
    return found


def readability(cfg, P):
    out = []
    for tool, argv in (("taskset", ["taskset", "-p", str(os.getpid())]),
                       ("numactl", ["numactl", "--hardware"])):
        if shutil.which(tool) is None:
            out.append(item("read.%s" % tool, "FAIL", reason="not on PATH"))
            continue
        r = _run(argv, 30)
        out.append(item("read.%s" % tool, "PASS" if r.get("rc") == 0 else "FAIL", run=r))
    files = {"schedstat": "/proc/self/schedstat", "softirq": "/proc/softirqs",
             "proc_stat": "/proc/stat",
             "cpufreq": "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq",
             "cpuidle": "/sys/devices/system/cpu/cpu%d/cpuidle/state0/name"}
    cpu = sorted(os.sched_getaffinity(0))[0]
    for name, path in files.items():
        path = path % cpu if "%d" in path else path
        ok, val = _readable(path)
        # cpufreq/cpuidle may be absent on VMs; that is a reading, not a failure of the tool.
        st = "PASS" if ok else ("RECORDED" if name in ("cpufreq", "cpuidle") else "FAIL")
        out.append(item("read.%s" % name, st, path=path, sample=val[:120]))
    mem = _own_cgroup_memory()
    out.append(item("read.memory_guard", "RECORDED", cgroup=mem,
                    note="whether systemd-run MemoryMax is in effect must be read here, "
                         "not assumed"))
    return out


def _node_cpus():
    nodes = {}
    for d in glob.glob("/sys/devices/system/node/node[0-9]*"):
        ok, val = _readable(os.path.join(d, "cpulist"))
        if ok:
            nodes[int(d.rsplit("node", 1)[1])] = _parse_cpulist(val.strip())
    return nodes


def _parse_cpulist(s):
    out = []
    for part in filter(None, s.split(",")):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def core_budget(cfg, P):
    avail = sorted(os.sched_getaffinity(0))
    nodes = {n: sorted(set(c) & set(avail)) for n, c in _node_cpus().items()}
    res = []
    for topo in cfg.get("topologies", []):
        need, missing = {}, []
        for role, v in topo["roles"].items():
            if isinstance(v, dict):
                if v["param"] not in P:
                    missing.append(v["param"])
                    continue
                v = int(P[v["param"]])
            need[role] = v
        if missing:
            res.append(item("budget.%s" % topo["name"], "NOT_RUN", missing_params=missing))
            continue
        total = sum(need.values())
        res.append(item("budget.%s" % topo["name"], "PASS" if total <= len(avail) else "FAIL",
                        need=need, total=total, available=len(avail),
                        per_numa_available={str(k): len(v) for k, v in sorted(nodes.items())}))
    return res or [item("budget", "NOT_RUN", reason="config has no topologies")]


ACCESS_RE = re.compile(rb'"(GET|POST) [^ "]+ HTTP/1\.[01]" \d{3}')


def tree_checks(cfg, tree):
    if tree is None:
        return [item("tree.access_log", "NOT_RUN", reason="no --tree given"),
                item("tree.empty_delta", "NOT_RUN", reason="no --tree given")]
    per = {}
    for p in sorted(glob.glob(os.path.join(tree, "raw", "logs", "*"))):
        with open(p, "rb") as f:
            n = len(ACCESS_RE.findall(f.read()))
        if n:
            per[os.path.basename(p)] = n
    logs = glob.glob(os.path.join(tree, "raw", "logs", "*"))
    out = [item("tree.access_log", "RECORDED" if logs else "NOT_RUN", files_scanned=len(logs),
                access_lines=per, total=sum(per.values()),
                note="nonzero means request log lines go to the log sink (§7 trap)")]
    hp = os.path.join(tree, cfg.get("httpprobe_out", "raw/httpprobe.json"))
    if not os.path.exists(hp):
        out.append(item("tree.empty_delta", "NOT_RUN", reason="%s missing" % hp))
        return out
    with open(hp) as f:
        doc = json.load(f)
    rs = doc.get("results", [])
    bad = [r.get("name") for r in rs if r.get("error") or r.get("status") != 200
           or not r.get("done_seen")
           or r["ssecounter"]["frames"] != r["n_frames_excl_done"]]  # both exclude [DONE]
    summary = [{k: r.get(k) for k in ("name", "endpoint", "status", "n_frames_excl_done",
                                      "n_empty_text_frames", "crlf_seen", "done_seen")}
               for r in rs]
    st = "NOT_RUN" if not rs else ("FAIL" if bad else "PASS")
    out.append(item("tree.empty_delta", st, expected_requests=cfg.get("httpprobe_requests"),
                    seen_requests=len(rs), bad=bad, results=summary))
    if rs and cfg.get("httpprobe_requests") not in (None, len(rs)):
        out[-1]["status"] = "FAIL"
    return out


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m caplaunch.smoke", description=__doc__)
    add_identity_args(p)  # passed on to capenv.loopback (INTERFACES 2.1)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True, help="new directory for smoke.json and readings")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--tree", default=None, help="a finished e0c-frontend-http launch tree")
    a = p.parse_args(argv)
    with open(a.config) as f:
        cfg = json.load(f)
    if cfg.get("schema") != CONFIG_SCHEMA:
        print("smoke: config schema must be %s" % CONFIG_SCHEMA, file=sys.stderr)
        return 3
    P = {}
    for kv in a.param:
        k, sep, v = kv.partition("=")
        if not sep:
            print("smoke: --param needs KEY=VALUE: %s" % kv, file=sys.stderr)
            return 2
        P[k] = v
    os.makedirs(a.out)  # must be new
    ident = ["--experiment", a.experiment, "--arm", a.arm, "--launch-seq", str(a.launch_seq),
             "--job-id", a.job_id, "--restart-count", str(a.restart_count),
             "--register-sha", a.register_sha, "--role", a.role]
    items = (env_premises(cfg, P) + switchinterval(cfg, P) + d14_loopback(cfg, P, a.out, ident)
             + readability(cfg, P) + core_budget(cfg, P) + tree_checks(cfg, a.tree))
    counts = {s: sum(1 for i in items if i["status"] == s)
              for s in ("PASS", "FAIL", "RECORDED", "NOT_RUN")}
    doc = {"schema": SCHEMA, "identity": ident, "config": os.path.abspath(a.config), "params": P,
           "tree": a.tree, "host": os.uname().nodename, "counts": counts, "items": items}
    atomic_write_bytes(os.path.join(a.out, "smoke.json"), (dumps(doc) + "\n").encode())
    for i in items:
        print("%-9s %s" % (i["status"], i["id"]))
    return 4 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
