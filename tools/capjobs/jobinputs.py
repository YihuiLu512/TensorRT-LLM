# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate one launch's job inputs (capcommon.jobinputs; INTERFACES errata B2B-9).

capjobs.launches calls generate() in the job after the gate, before the launch, with the same
environment the launch gets. The command line is for running a launch outside a job (the same
files and records the job would make):

    python -m capjobs.jobinputs --workdir W --scenario <relpath in W> --python PY \
        --out-dir DIR [--param KEY=VALUE]... [--timeout-s S]

Prints one `--job-input KIND=PATH` pair per kind (the caplaunch arguments). Exit 0, or 3 when a
generator fails, the scenario has no job_inputs key or fails
capcommon.jobinputs.check_scenario.
"""

import argparse
import json
import os
import subprocess
import sys

from capcommon import jobinputs as JI
from capcommon.clock import wall_iso
from capcommon.exitcodes import CONFIG, OK, ConfigError
from capcommon.hashing import sha256_file
from capcommon.jsonio import atomic_write_bytes, dumps
from capcommon.templates import parse_params


class JobInputError(ConfigError):
    """One kind failed; .records holds the records so far, the failed kind's last (a kind that
    failed before its generator ran has rc None, an `error` and no record file)."""

    def __init__(self, msg, records):
        ConfigError.__init__(self, msg)
        self.records = records


def params_from_argv(argv):
    """{KEY: VALUE} of every --param KEY=VALUE / --param=KEY=VALUE in a caplaunch argv, parsed
    as caplaunch parses them (capcommon.templates.parse_params)."""
    pairs = []
    for i, a in enumerate(argv):
        if a == "--param":
            pairs.append(argv[i + 1] if i + 1 < len(argv) else "")
        elif a.startswith("--param="):
            pairs.append(a[len("--param="):])
    return parse_params(pairs)


def generate(doc, workdir, params, python, out_dir, env, timeout_s, run=subprocess.run):
    """Run every generator the scenario declares. Returns ({kind: path}, records), ({}, []) when
    it has no job_inputs key; raises ConfigError for a bad section or scenario key it reads
    (capcommon.jobinputs.check_scenario), JobInputError naming the first kind that failed (its
    log and record stay in out_dir). run: called as subprocess.run (capjobs.launches passes its
    SIGTERM-aware run_helper; whatever else it raises propagates)."""
    if not JI.declares(doc):
        return {}, []
    kinds = JI.check_scenario(doc)
    inputs = {k: os.path.join(workdir, doc[k]) for k in JI.REGISTERED_INPUTS}
    os.makedirs(out_dir, exist_ok=True)
    paths, records = {}, []
    for kind in kinds:
        out = os.path.join(out_dir, "%s.json" % kind)
        log = out + ".log"
        try:
            if os.path.lexists(out):
                raise ConfigError("%s already exists" % out)
            argv = JI.generator_argv(doc, kind, params, inputs, python, out)
            used = sorted({k for a in doc[JI.SECTION][kind]["args"]
                           for k in JI.REGISTERED_INPUTS if "{input:%s}" % k in a})
            used_sha = {k: sha256_file(inputs[k]) for k in used}
        except (ConfigError, OSError) as e:
            records.append({"schema": JI.RECORD_SCHEMA, "kind": kind,
                            "module": JI.KINDS[kind][0], "rc": None, "out_sha256": None,
                            "log": None, "error": str(e)})
            raise JobInputError("job input %s: %s" % (kind, e), records)
        rec = {"schema": JI.RECORD_SCHEMA, "kind": kind, "module": JI.KINDS[kind][0],
               "argv": argv, "inputs": used_sha, "started_wall": wall_iso(), "rc": None,
               "out_sha256": None, "log": log}
        try:
            with open(log, "wb") as lf:
                rec["rc"] = run(argv, env=env, stdout=lf, stderr=lf,
                                stdin=subprocess.DEVNULL, timeout=timeout_s).returncode
        except (OSError, subprocess.TimeoutExpired) as e:
            rec["error"] = str(e)
        rec["ended_wall"] = wall_iso()
        why = None
        if rec["rc"] != 0:
            why = "exit %r%s (log %s)" % (rec["rc"], "; " + rec["error"] if "error" in rec
                                          else "", log)
        elif not os.path.isfile(out):
            why = "exit 0 but no %s" % out
        else:
            try:
                with open(out, "r", encoding="utf-8") as f:
                    schema = json.load(f).get("schema")
            except (OSError, ValueError, AttributeError) as e:
                schema = "unreadable: %s" % e
            if schema != JI.KINDS[kind][1]:
                why = "%s has schema %r, not %s" % (out, schema, JI.KINDS[kind][1])
            else:
                rec["out_sha256"] = sha256_file(out)
        atomic_write_bytes(out + ".record.json", (dumps(rec) + "\n").encode("utf-8"))
        records.append(rec)
        if why is not None:
            raise JobInputError("job input %s: %s" % (kind, why), records)
        paths[kind] = out
    return paths, records


def caplaunch_args(paths):
    out = []
    for kind, path in paths.items():
        out += ["--job-input", "%s=%s" % (kind, path)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m capjobs.jobinputs", allow_abbrev=False)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--scenario", required=True, help="scenario relpath inside --workdir")
    ap.add_argument("--python", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    a = ap.parse_args(argv)
    try:
        params = params_from_argv(["--param=%s" % p for p in a.param])
        with open(os.path.join(a.workdir, a.scenario), "r", encoding="utf-8") as f:
            doc = json.load(f)
        if not JI.declares(doc):
            raise ConfigError("scenario %s declares no job_inputs" % a.scenario)
        paths, _recs = generate(doc, os.path.realpath(a.workdir), params, a.python, a.out_dir,
                                dict(os.environ), a.timeout_s)
    except (ConfigError, OSError, ValueError, KeyError, TypeError) as e:
        print("capjobs.jobinputs: %s" % e, file=sys.stderr)
        return CONFIG
    print(" ".join(caplaunch_args(paths)))
    return OK


if __name__ == "__main__":
    sys.exit(main())
