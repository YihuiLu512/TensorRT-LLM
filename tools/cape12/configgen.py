# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m cape12.configgen: E12-L2a perf-sanity YAMLs for one launch (design 12.6(a)(c)).

The server entry is copied unchanged from the WT-21 U4 source YAML (pc_u4: GLM-5.3-Flash FP8
TP4/EP4 MTP3 13312/33, stream_interval 20; the arm is chosen per step by the inner script with
TRTLLM_PERF_ARM, as WT-21 did). Only client_configs change; every client carries
client_env_var for cape12.reqlog_l2a (E12_MODE, E12_DELTA_MS, E12_SCHEDULE, E12_RUN,
E12_WARMUP_SEED). The warm-up pass of test_perf_sanity uses the measured seed plus one
(tests/integration/defs/perf/warmup_pass.py), so with no --seed in the measured command the
warm-up client's seed is 1: E12_WARMUP_SEED=1.

  mtype --source Y --server NAME --k K --text-first {0,1} --text-dataset D --out O
      M-type launch: K closed con8 runs per input (text and token-id alternating, starting with
      text when --text-first 1), then con1 text and con1 token-id.
  itype --source Y --server NAME --text-dataset D --out O [--doses-ms a,b,..]
        [--schedule F ...]
      I-type launch: front = con8 text, con8 token-id, con1 text, con1 token-id (closed); back =
      one closed_delta con8 text run per dose (A-def launches only: give --doses-ms), then one
      open con8 text run per --schedule file (mode open; a schedule of kind replay -> replay).

Run length is registered: con8 x 20 iterations = 160 requests (12.6(b)); con1 x 30.
"""

import argparse
import copy
import json
import sys

import yaml

ISL, OSL = 13312, 33
WARMUP_SEED = 1


def client(name, con, iters, env, dataset=None):
    c = {"name": name, "concurrency": con, "iterations": iters, "isl": ISL, "osl": OSL,
         "backend": "openai", "client_env_var": " ".join("%s=%s" % kv for kv in env)}
    if dataset:
        c["dataset_file"] = dataset
    return c


def _env(run, mode="closed", **extra):
    return [("E12_RUN", run), ("E12_MODE", mode), ("E12_WARMUP_SEED", WARMUP_SEED)] + \
        sorted(extra.items())


def mtype_clients(k, text_first, text_dataset):
    out = []
    for j in range(k):
        pair = [("text", text_dataset), ("tok", None)]
        if not text_first:
            pair.reverse()
        for kind, ds in pair:
            name = "c8_r%d_%s" % (j + 1, kind)
            out.append(client(name, 8, 20, _env(name), ds))
    out.append(client("c1_text", 1, 30, _env("c1_text"), text_dataset))
    out.append(client("c1_tok", 1, 30, _env("c1_tok")))
    return out


def itype_clients(text_dataset, doses_ms, schedules):
    out = [client("c8_front_text", 8, 20, _env("c8_front_text"), text_dataset),
           client("c8_front_tok", 8, 20, _env("c8_front_tok")),
           client("c1_text", 1, 30, _env("c1_text"), text_dataset),
           client("c1_tok", 1, 30, _env("c1_tok"))]
    for i, d in enumerate(doses_ms):
        name = "c8_dose%02d_text" % (i + 1)
        out.append(client(name, 8, 20, _env(name, "closed_delta", E12_DELTA_MS=d),
                          text_dataset))
    for i, (path, kind) in enumerate(schedules):
        name = "c8_%s%02d_text" % (kind, i + 1)
        out.append(client(name, 8, 20, _env(name, kind, E12_SCHEDULE=path), text_dataset))
    return out


def build(source_doc, server_name, clients):
    hits = [s for s in source_doc["server_configs"] if s["name"] == server_name]
    if len(hits) != 1:
        raise ValueError("source has %d server_configs named %r" % (len(hits), server_name))
    s = copy.deepcopy(hits[0])
    s["client_configs"] = clients
    doc = {k: copy.deepcopy(v) for k, v in source_doc.items() if k != "server_configs"}
    doc["server_configs"] = [s]
    return doc


def schedule_kind(path):
    kind = json.load(open(path)).get("kind")
    if kind not in ("poisson", "replay"):
        raise ValueError("%s: schedule kind %r" % (path, kind))
    return "open" if kind == "poisson" else "replay"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cape12.configgen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("mtype", "itype"):
        p = sub.add_parser(name)
        p.add_argument("--source", required=True)
        p.add_argument("--server", required=True)
        p.add_argument("--text-dataset", required=True)
        p.add_argument("--out", required=True)
        if name == "mtype":
            p.add_argument("--k", type=int, required=True)
            p.add_argument("--text-first", type=int, choices=(0, 1), required=True)
        else:
            p.add_argument("--doses-ms", default="")
            p.add_argument("--schedule", action="append", default=[])
    a = ap.parse_args(argv)
    try:
        src = yaml.safe_load(open(a.source))
        if a.cmd == "mtype":
            if a.k < 2:
                raise ValueError("--k must be >= 2 (run and launch variance components)")
            clients = mtype_clients(a.k, bool(a.text_first), a.text_dataset)
        else:
            doses = [float(x) for x in a.doses_ms.split(",") if x != ""]
            clients = itype_clients(a.text_dataset, doses,
                                    [(p, schedule_kind(p)) for p in a.schedule])
        doc = build(src, a.server, clients)
        with open(a.out, "x") as f:
            f.write("# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
                    "AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"
                    "# E12-L2a %s, generated by cape12.configgen from %s\n" % (a.cmd, a.source))
            yaml.safe_dump(doc, f, sort_keys=False)
    except (ValueError, OSError, KeyError) as e:
        print("cape12.configgen: %s" % e, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
