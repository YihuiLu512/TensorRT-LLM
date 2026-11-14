# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Job inputs: files a launch needs that only the container can build (INTERFACES errata B2B-9).

A scenario lists them in its optional top-level `job_inputs` section:

    "job_inputs": {"prebaked": {"args": ["--pb2-module", "M", "--tier-table", "{input:tier_table}",
                                         "--token-ids", "{param:spb_token_ids}", ...]}}

Each kind has one fixed generator module (KINDS). capjobs.launches runs, in the job after the
gate and before the launch, `<python> -m <module> <args with placeholders resolved> --out <file>`
and writes a record next to the file (<file>.record.json, `capkit.job_input_record/1`); caplaunch
gets `--job-input KIND=<file>`, checks the record (check_record), copies both into the launch tree
and resolves `{input:KIND}` to the copy. Generator args may use `{param:NAME}` (NAME in the
scenario's params_required) and `{input:tier_table|schedule|prompts}` (the registered inputs);
`--out` is appended and must not appear in them.
"""

import re

from .exitcodes import ConfigError

RECORD_SCHEMA = "capkit.job_input_record/1"
# kind -> (generator module, schema of the file it writes)
KINDS = {
    "fingerprint": ("capstubs.fingerprint", "capkit.fingerprint/1"),
    "request_bytes": ("capgrpc.reqbytes", "capkit.reqbytes/1"),
    "prebaked": ("capstubs.prebaked", "capkit.prebaked/1"),
}
REGISTERED_INPUTS = ("tier_table", "schedule", "prompts")
SECTION = "job_inputs"
_PH_RE = re.compile(r"\{([a-z_]+)(?::([A-Za-z0-9_.-]+))?\}")


def _err(msg):
    raise ConfigError("scenario job_inputs: %s" % msg)


def kinds_of(doc):
    """Kinds a scenario document declares, in KINDS order ([] without the section)."""
    sec = doc.get(SECTION) if isinstance(doc, dict) else None
    if not isinstance(sec, dict):
        return []
    return [k for k in KINDS if k in sec]


def validate_section(sec, params_required):
    """Structure check of a scenario's job_inputs section; raises ConfigError."""
    if not isinstance(sec, dict) or not sec:
        _err("must be a non-empty object {kind: {args: [...]}}")
    unknown = sorted(set(sec) - set(KINDS))
    if unknown:
        _err("unknown kinds %s (known: %s)" % (unknown, list(KINDS)))
    for kind, spec in sec.items():
        if not isinstance(spec, dict) or set(spec) != {"args"}:
            _err("%s must be exactly {args: [...]}" % kind)
        args = spec["args"]
        if not isinstance(args, list) or any(not isinstance(a, str) or not a for a in args):
            _err("%s.args must be a list of non-empty strings" % kind)
        for a in args:
            if a.split("=", 1)[0] == "--out":
                _err("%s.args must not contain --out (it is appended)" % kind)
            for name, arg in _placeholders(a):
                if name == "param" and arg in params_required:
                    continue
                if name == "input" and arg in REGISTERED_INPUTS:
                    continue
                _err("%s.args: placeholder {%s%s} is not {param:<params_required>} or "
                     "{input:%s}" % (kind, name, "" if arg is None else ":" + arg,
                                     "|".join(REGISTERED_INPUTS)))
    return sec


def _placeholders(text):
    return [(m.group(1), m.group(2)) for m in _PH_RE.finditer(text)]


def generator_argv(doc, kind, params, input_paths, python, out):
    """The generator command line of one kind: python -m <module> <args resolved> --out <out>.
    params: {name: value} (the launch's --param values); input_paths: {tier_table|schedule|
    prompts: absolute path}."""
    spec = doc[SECTION][kind]

    def sub(m):
        name, arg = m.group(1), m.group(2)
        if name == "param":
            if arg not in params:
                raise ConfigError("job input %s: no --param %s" % (kind, arg))
            return params[arg]
        return input_paths[arg]
    return [python, "-m", KINDS[kind][0]] + [_PH_RE.sub(sub, a) for a in spec["args"]] \
        + ["--out", out]


def check_record(record, kind, file_sha256, registered_sha256):
    """The record capjobs.launches wrote for one generated file: same kind and module, the file
    it describes has file_sha256, and every registered input it read had the sha the launch
    registered (registered_sha256: {tier_table|schedule|prompts: sha256}). Raises ConfigError."""
    if not isinstance(record, dict) or record.get("schema") != RECORD_SCHEMA:
        raise ConfigError("job input %s: record schema must be %s" % (kind, RECORD_SCHEMA))
    if record.get("kind") != kind or record.get("module") != KINDS[kind][0]:
        raise ConfigError("job input %s: record kind/module %r/%r" % (
            kind, record.get("kind"), record.get("module")))
    if record.get("rc") != 0 or record.get("out_sha256") != file_sha256:
        raise ConfigError("job input %s: record rc %r, out_sha256 %r != file sha %s"
                          % (kind, record.get("rc"), record.get("out_sha256"), file_sha256))
    used = record.get("inputs")
    if not isinstance(used, dict) or any(k not in REGISTERED_INPUTS for k in used):
        raise ConfigError("job input %s: record inputs must map %s to sha256"
                          % (kind, list(REGISTERED_INPUTS)))
    for k, sha in sorted(used.items()):
        if sha != registered_sha256.get(k):
            raise ConfigError("job input %s was built from %s sha %s, the launch registers %s"
                              % (kind, k, sha, registered_sha256.get(k)))
