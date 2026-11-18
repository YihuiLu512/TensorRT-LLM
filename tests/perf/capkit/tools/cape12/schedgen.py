# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m cape12.schedgen: E12-L2a time tables and dose sequences (design 12.6(e)(f)).

  poisson --seed S --n N --rate R --out F
      N Poisson arrivals at R requests/s from a unit-rate sequence (random.Random(S).expovariate
      (1.0)), first offset 0; the unit sequence depends on S and N only, so one registered seed
      gives the same shape at every rate (the run-time scaling of 12.6(f)).
  replay --reqlog F --out G [--n N]
      the call times (t_call, else t0) of a closed-loop run's reqlog, relative to the first one,
      as an absolute time table (12.6(f) trajectory replay). Refuses a warm-up reqlog and a
      reqlog whose request count differs from --n.
  dose --levels a,b,c --sequences K --seed S --out F
      K orderings of the registered dose levels, random.Random(S + k).shuffle; each job's second
      A-def launch uses the reverse of its sequence (12.6(e)).

Every output is JSON with its inputs recorded; schedules use schema cape12.schedule/1 (read by
cape12.reqlog_l2a). The output file must not exist.
"""

import argparse
import json
import random
import sys

from cape12.reqlog_l2a import SCHEDULE_SCHEMA

DOSE_SCHEMA = "cape12.doses/1"


def poisson_offsets(seed, n, rate):
    if n <= 0 or rate <= 0:
        raise ValueError("n and rate must be > 0")
    rng = random.Random(seed)
    unit, t = [0.0], 0.0
    for _ in range(n - 1):
        t += rng.expovariate(1.0)
        unit.append(t)
    return [u / rate for u in unit], unit


def replay_offsets(lines, n=None):
    hdr = [x for x in lines if x.get("hdr") == 1]
    reqs = sorted((x for x in lines if "i" in x), key=lambda x: x["i"])
    if len(hdr) != 1:
        raise ValueError("reqlog must have exactly one header line, got %d" % len(hdr))
    if hdr[0].get("warmup"):
        raise ValueError("reqlog is a warm-up client's")
    if hdr[0].get("mode", "closed") != "closed":
        raise ValueError("replay needs a closed-loop reqlog, got mode %r" % hdr[0].get("mode"))
    if not reqs or (n is not None and len(reqs) != n):
        raise ValueError("reqlog has %d requests, expected %s" % (len(reqs), n))
    ts = [x.get("t_call", x.get("t0")) for x in reqs]
    t0 = min(ts)
    return sorted(t - t0 for t in ts)


def dose_sequences(levels, k, seed):
    if not levels or k <= 0:
        raise ValueError("need levels and k > 0")
    out = []
    for j in range(k):
        seq = list(levels)
        random.Random(seed + j).shuffle(seq)
        out.append({"index": j, "forward": seq, "reverse": seq[::-1]})
    return out


def _write(path, doc):
    with open(path, "x") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cape12.schedgen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("poisson")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--rate", type=float, required=True)
    p.add_argument("--out", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--reqlog", required=True)
    r.add_argument("--n", type=int)
    r.add_argument("--out", required=True)
    d = sub.add_parser("dose")
    d.add_argument("--levels", required=True, help="comma-separated delta values (ms)")
    d.add_argument("--sequences", type=int, required=True)
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "poisson":
            offs, unit = poisson_offsets(a.seed, a.n, a.rate)
            _write(a.out, {"schema": SCHEDULE_SCHEMA, "kind": "poisson", "seed": a.seed,
                           "n": a.n, "rate_rps": a.rate, "unit_offsets": unit,
                           "offsets_s": offs})
        elif a.cmd == "replay":
            lines = [json.loads(x) for x in open(a.reqlog) if x.strip()]
            offs = replay_offsets(lines, a.n)
            _write(a.out, {"schema": SCHEDULE_SCHEMA, "kind": "replay", "source": a.reqlog,
                           "n": len(offs), "offsets_s": offs})
        else:
            levels = [float(x) for x in a.levels.split(",")]
            _write(a.out, {"schema": DOSE_SCHEMA, "levels_ms": levels, "seed": a.seed,
                           "sequences": dose_sequences(levels, a.sequences, a.seed)})
    except (ValueError, OSError) as e:
        print("cape12.schedgen: %s" % e, file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
