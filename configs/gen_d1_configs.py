# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generator of configs/d1: scenario TEMPLATES for the C++ engine host (D1, TRTLLM-18068) arms of
design card E16 and the E9 supplementary arms (D1 plan 04_perf-plan.md sections 2, 3, 4).

Every arm is derived from a configs/snx template (read, not regenerated) with only the D1 delta, so
each D1 arm stays comparable with the S-NX arm it comes from:

  E16 (experiment E16; from e8-k4-<shape>: capgrpc.load K = 4 -> engine host, S-NX rank 0)
    e16-py-k4-<s>        E8-py: the Python host, single-frontend mode (E8 unchanged, own id)
    e16-pynd-k4-<s>      E8-py-nd: E8-py with every request's extra.detokenize = false
                         (capgrpc.reqbytes --extra-detokenize false), the C++ host's default
    e16-2l-k4-<s>        E8-2l: the launcher of the C++ host mode (2 lanes, multi-frontend mode)
                         and the load on its loopback port: the Python Generate on lane 0
                         (serve_nx --report-target private); the C++ host runs idle on lane 1
    e16-nx-k4-<s>        E16-nx: the C++ host (lane 1) in front of S-NX, --grpc-threads, self-report
                         (--diag-out {raw}/d1)
    e16-nx-nodiag-k4-<s> E16-nx without --diag-out (P0: cost of the self-report)
  E9 supplementary (experiment E9; from e9-n<n>f0-<shape>: capload -> n new-architecture frontends
  -> engine host)
    e9-nc<n>f0-<s>       NC: the frontends connect to the C++ host (public port)
    e9-n2l8f0-<s>        N2l: the 8 frontends connect to the launcher's loopback port (Python host in
                         the two-lane multi-frontend mode)

The C++ host is started by the launcher (trtllm-serve --openengine-host-impl cpp) as a child of
the host role's process, so it runs on the host role's own cores (4, as the Python host; the
rank 0 worker and procmon keep their cores). E16-zr (replay stub S-ZR instead of rank 0) is not
generated here: its response source is capstubs.szr_launch (branch szr-d1, W13); see
configs/d1/README.md.

Usage: python configs/gen_d1_configs.py <new output dir>   (writes <dir>/d1; reads configs/snx)
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SNX = os.path.join(HERE, "snx")
SUB = "d1"
REG = {"E16": "09_d1-host/perf/E16/register", "E9": "09_d1-host/perf/E9s/register"}
CPP_IMPL = ["--openengine-host-impl", "cpp"]
D1_THREADS = ["--cpp-host-arg=--grpc-threads", "--cpp-host-arg", "{param:d1_grpc_threads}"]
D1_DIAG = ["--cpp-host-arg=--diag-out", "--cpp-host-arg", "{raw}/d1"]
D1_BIN = ["--cpp-host-bin", "{param:d1_bin}"]
NOTE = ("D1 arm (configs/gen_d1_configs.py, TRTLLM-18068 04_perf-plan.md): derived from "
        "configs/snx/%s; %s")


def load(name):
    with open(os.path.join(SNX, name)) as f:
        return json.load(f)


def w(out, name, doc):
    d = os.path.join(out, SUB)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    assert not os.path.exists(p), p
    with open(p, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=False)
        f.write("\n")


def params(obj, out):
    if isinstance(obj, str):
        rest = obj
        while "{param:" in rest:
            rest = rest[rest.index("{param:") + 7:]
            out.add(rest[:rest.index("}")])
    elif isinstance(obj, dict):
        [params(v, out) for v in obj.values()]
    elif isinstance(obj, list):
        [params(v, out) for v in obj]
    return out


def host_delta(role, own=(), serve=(), note=""):
    """Insert own serve_nx options before `--` and serve arguments at the end of the host role."""
    a = role["args"]
    i = a.index("--")
    role["args"] = a[:i] + list(own) + a[i:] + list(serve)
    role["note"] = role.get("note", "") + "; " + note


def derive(exp, src, name, edit, what):
    """(scenario, tiers template) of arm `name` from configs/snx/<src>.*."""
    sc = load(src + ".scenario.json")
    tp = load(src + ".tiers.template.json")
    sc, tp = copy.deepcopy(sc), copy.deepcopy(tp)
    roles = {r["role"]: r for r in sc["roles"]}
    edit(sc, roles)
    reg = REG[exp]
    sc.update({"experiment": exp, "name": name,
               "tier_table": "%s/%s.tiers.json" % (reg, name),
               "schedule": "%s/%s.schedule.capbin" % (reg, name),
               "prompts": "%s/%s.prompts.json" % (reg, name)})
    sc["params_required"] = sorted(params([sc["roles"], sc.get("job_inputs") or {}], set()))
    sc["note"] = NOTE % (src, what) + ". " + sc["note"]
    tp.update({"experiment": exp, "name": name})
    tp["note"] = NOTE % (src, what) + "; " + tp["note"]
    return sc, tp


def cpp_host(diag=True, private=False):
    def edit(sc, roles):
        own = D1_BIN + D1_THREADS + (D1_DIAG if diag else [])
        if private:
            own = own + ["--report-target", "private"]
        host_delta(roles["host"], own, CPP_IMPL,
                   "C++ host mode: the launcher starts trtllm-openengine-host (param d1_bin) "
                   "as attached frontend 1 on the public port, on this role's own cores, "
                   "--grpc-threads (param d1_grpc_threads)%s%s" % (
                       ", self-report in {raw}/d1" if diag else ", no self-report",
                       "; --report-target private: report_addr is the launcher's loopback "
                       "OpenEngine port (Python Generate, lane 0)" if private else ""))
    return edit


def no_detokenize(sc, roles):
    sc["job_inputs"]["request_bytes"]["args"] += ["--extra-detokenize", "false"]


def unchanged(sc, roles):
    pass


def main(out):
    arms = []
    for shape in ("mc", "r"):
        src = "e8-k4-%s" % shape
        arms += [
            ("E16", src, "e16-py-k4-%s" % shape, unchanged,
             "E8-py: the Python host, single-frontend mode, E8 unchanged"),
            ("E16", src, "e16-pynd-k4-%s" % shape, no_detokenize,
             "E8-py-nd: E8-py with extra.detokenize = false in every request"),
            ("E16", src, "e16-2l-k4-%s" % shape, cpp_host(diag=False, private=True),
             "E8-2l: load on the C++ host mode launcher's loopback Generate (Python, lane 0)"),
            ("E16", src, "e16-nx-k4-%s" % shape, cpp_host(diag=True),
             "E16-nx: the C++ host in front of S-NX, self-report on"),
            ("E16", src, "e16-nx-nodiag-k4-%s" % shape, cpp_host(diag=False),
             "E16-nx without the self-report (--diag-out off)")]
        for n in (1, 8):
            src9 = "e9-n%df0-%s" % (n, shape)
            arms.append(("E9", src9, "e9-nc%df0-%s" % (n, shape), cpp_host(diag=True),
                         "NC-%df0: the frontends connect to the C++ host" % n))
            if n == 8:  # the plan's N2l arm is in the 8-frontend group only
                arms.append(("E9", src9, "e9-n2l%df0-%s" % (n, shape),
                             cpp_host(diag=False, private=True),
                             "N2l-%df0: the frontends connect to the C++ host mode launcher's "
                             "loopback port" % n))
    for exp, src, name, edit, what in arms:
        sc, tp = derive(exp, src, name, edit, what)
        w(out, name + ".scenario.json", sc)
        w(out, name + ".tiers.template.json", tp)
        with open(os.path.join(SNX, src + ".values.todo.json")) as f:
            todo = f.read()
        d = os.path.join(out, SUB, name + ".values.todo.json")
        assert not os.path.exists(d), d
        with open(d, "w") as f:
            f.write(todo)


if __name__ == "__main__":
    main(sys.argv[1])
