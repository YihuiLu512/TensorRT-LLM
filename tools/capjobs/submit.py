# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The only submission entry for jobs that do not load a real model (INTERFACES.md 9.5, 11;
plan CONSTRAINTS section 2, RESOURCES sections 3-4).

    PYTHONPATH=<C>/tools:<C>/jobs python3.11 -m capjobs.submit <sub-command> ...

Sub-commands: probe (the only path that runs `sbatch --test-only`), submit, local-record,
slurm-view, jobstate, record-end, exclude, prep-tokenizer (draft: dry-run only). See
jobs/README.md. Nothing has a default resource value; every cluster command goes through
capjobs.remote (ssh, allow-listed commands).
"""

import argparse
import json
import os
import re
import secrets
import shlex
import subprocess
import sys

from capcommon import ledger, policy
from capcommon.clock import wall_iso
from capcommon.exitcodes import CONFIG, OK, RUNTIME, CapkitError, ConfigError
from capcommon.hashing import is_git_sha, is_sha256_hex, sha256_file
from capcommon.identity import ARM_RE, JOB_ID_RE

from . import cdbexclude, jobstate, launchplan, remote, resources, slurmview, snapshot

PATH_RE = re.compile(r"^/[A-Za-z0-9._/@+-]*$")
MODULE_RE = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$")
EXE_RE = re.compile(r"^[A-Za-z0-9._/+-]+$")
JOBSTATE_EXIT = {jobstate.ENDED: 0, jobstate.NOT_ENDED: 20, jobstate.UNKNOWN: 21}
SBATCH_TEMPLATE = "jobs/capkit_job.sbatch"
PREP_TOKENIZER_TEMPLATE = "jobs/prep_tokenizer.sbatch.draft"


class SubmitError(ConfigError):
    pass


# ---- small helpers ----------------------------------------------------------------------------

def check_path(p, what):
    if not isinstance(p, str) or not PATH_RE.match(p) or "/../" in p or p.endswith("/..") \
            or (p != "/" and p.endswith("/")):
        raise SubmitError("%s must be an absolute path of [A-Za-z0-9._/@+-], got %r" % (what, p))
    return p


def basis_line_present(workdir, basis):
    """The basis text must be non-empty, one line, and appear verbatim as a whole line of
    <W>/STATUS.md (CONSTRAINTS 2 item 6)."""
    if not isinstance(basis, str) or not basis.strip() or "\n" in basis or "\t" in basis:
        raise SubmitError("--basis must be one non-empty line without tabs")
    status = os.path.join(workdir, "STATUS.md")
    try:
        with open(status, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\r\n") for ln in f]
    except OSError as e:
        raise SubmitError("cannot read %s: %s" % (status, e))
    if basis not in lines:
        raise SubmitError("the --basis text is not a whole line of %s" % status)
    return True


def parse_mount(text):
    """'SRC:DST[:ro|:rw]' -> dict. Paths are absolute, no ':' or ','."""
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise SubmitError("mount must be SRC:DST[:ro|rw], got %r" % text)
    src, dst = check_path(parts[0], "mount src"), check_path(parts[1], "mount dst")
    mode = parts[2] if len(parts) == 3 else "rw"
    if mode not in ("ro", "rw"):
        raise SubmitError("mount mode must be ro or rw: %r" % text)
    return {"src": src, "dst": dst, "mode": mode, "text": text}


def _under(path, top):
    return path == top or top == "/" or path.startswith(top.rstrip("/") + "/")


def check_coverage(mounts, need_rw, need_ro):
    """Every needed path (and its realpath) lies under an identity mount (src == dst); paths the
    job writes need a rw mount. Returns the list of checks made (never empty)."""
    checks = []
    for paths, want_rw in ((need_rw, True), (need_ro, False)):
        for p in paths:
            for q in sorted({p, os.path.realpath(p)}):
                ok = [m for m in mounts if m["src"] == m["dst"] and _under(q, m["src"])
                      and (m["mode"] == "rw" or not want_rw)]
                if not ok:
                    raise SubmitError("path %s is not under an identity %s mount" %
                                      (q, "rw" if want_rw else "ro/rw"))
                checks.append({"path": q, "mount": ok[0]["text"]})
    if not checks:
        raise SubmitError("no mount coverage checks were made")
    return checks


def code_state(code_root, paths):
    """(HEAD sha, porcelain status lines limited to paths) of the code repo; read-only git."""
    head = subprocess.run(["git", "-C", code_root, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False)
    if head.returncode != 0:
        raise SubmitError("git rev-parse failed in %s: %s" % (code_root, head.stderr.strip()))
    st = subprocess.run(["git", "-C", code_root, "status", "--porcelain", "--"] + list(paths),
                        capture_output=True, text=True, check=False)
    if st.returncode != 0:
        raise SubmitError("git status failed in %s: %s" % (code_root, st.stderr.strip()))
    return head.stdout.strip(), [ln for ln in st.stdout.splitlines() if ln.strip()]


def make_plan_id(experiment):
    ts = wall_iso().replace("-", "").replace(":", "").replace(".", "")
    return "P%s-%s-%s" % (ts, experiment, secrets.token_hex(3))


def ledger_row(**kw):
    row = {c: "" for c in ledger.SUBMISSION_COLUMNS}
    unknown = set(kw) - set(row)
    if unknown:
        raise SubmitError("unknown ledger columns %s" % sorted(unknown))
    row.update({k: ("" if v is None else str(v)) for k, v in kw.items()})
    del row["event_at"]
    return row


def job_env_bytes(values):
    """'export KEY=<shell-quoted value>' lines for the sbatch/inner scripts to source."""
    lines = ["# generated by capjobs.submit; sourced by capkit_job.sbatch and capkit_inner.sh"]
    for k in sorted(values):
        if not re.match(r"^CAPJOBS_[A-Z0-9_]+$", k):
            raise SubmitError("bad job.env key %r" % k)
        v = str(values[k])
        if "\n" in v or "\x00" in v:
            raise SubmitError("job.env value for %s contains a newline" % k)
        lines.append("export %s=%s" % (k, shlex.quote(v)))
    return ("\n".join(lines) + "\n").encode("utf-8")


def parse_parsable(stdout):
    """`sbatch --parsable` prints '<jobid>' or '<jobid>;<cluster>' on its last line."""
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    if not lines:
        return None
    jid = lines[-1].split(";")[0]
    return jid if jid.isdigit() else None


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def emit(doc):
    print(json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False))

# ---- SECTION: resources -------------------------------------------------------------------------

def add_resource_args(p):
    """All required, no defaults (RESOURCES 4). --gpu-type is optional: without it the gres is
    the untyped 'gpu:N'."""
    p.add_argument("--account", required=True)
    p.add_argument("--partition", required=True, help="comma list; quote names with + @ /")
    p.add_argument("--qos", required=True)
    ex = p.add_mutually_exclusive_group(required=True)
    ex.add_argument("--exclude", help="comma list of nodes, or the literal 'none'")
    ex.add_argument("--exclude-file", help="JSON written by the 'exclude' sub-command")
    p.add_argument("--time", required=True, help="[D-]HH:MM:SS")
    p.add_argument("--mem", required=True, help="e.g. 1900G")
    p.add_argument("--cpus-per-task", required=True, type=int)
    p.add_argument("--gpus", required=True, type=int)
    p.add_argument("--gpu-type")
    p.add_argument("--job-name", required=True)


def resources_from_args(a):
    """Validated resource dict; with --exclude-file the file's partitions must equal --partition."""
    if a.exclude_file is not None:
        doc = cdbexclude.read_result(a.exclude_file)
        parts = sorted(a.partition.split(","))
        if sorted(doc["partitions"]) != parts:
            raise SubmitError("exclude file partitions %s != --partition %s"
                              % (doc["partitions"], parts))
        exclude = list(doc["exclude"])
        source = {"file": os.path.abspath(a.exclude_file),
                  "sha256": sha256_file(a.exclude_file)}
    else:
        exclude = resources.normalize_exclude(a.exclude)
        source = {"literal": a.exclude}
    res = resources.validate({
        "account": a.account, "partition": a.partition, "qos": a.qos, "exclude": exclude,
        "time": a.time, "mem": a.mem, "cpus_per_task": a.cpus_per_task, "gpus": a.gpus,
        "gpu_type": a.gpu_type, "job_name": a.job_name})
    res["exclude_source"] = source
    return res


def add_remote_args(p):
    p.add_argument("--login-host", required=True)
    p.add_argument("--ssh-timeout-s", required=True, type=float)


def get_runner(a, runner):
    return runner if runner is not None else remote.Runner(a.ssh_timeout_s)

# ---- SECTION: sub-commands ----------------------------------------------------------------------

def cmd_probe(a, runner):
    """sbatch --test-only through ssh; command and full output appended to --probe-log."""
    res = resources_from_args(a)
    remote_argv = resources.probe_argv(res)
    policy.check_test_only_argv(remote_argv)
    argv = remote.ssh_argv(a.login_host, remote_argv)
    if a.dry_run:
        emit({"dry_run": True, "remote_argv": remote_argv, "argv": argv})
        return OK
    log = check_path(os.path.abspath(a.probe_log), "--probe-log")
    rec = {"t_wall": wall_iso(), "phase": "start", "remote_argv": remote_argv, "argv": argv,
           "exclude_source": res["exclude_source"]}
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    rc, out, err = get_runner(a, runner).run(argv)
    rec = {"t_wall": wall_iso(), "phase": "result", "argv": argv, "rc": rc, "stdout": out,
           "stderr": err}
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    emit({"rc": rc, "stdout": out, "stderr": err, "probe_log": log})
    return OK if rc == 0 else RUNTIME


def _load_overlay_manifest(path, head_sha):
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise SubmitError("overlay manifest must be a JSON object")
    if doc.get("non_python_changed") != []:
        raise SubmitError("overlay manifest non_python_changed must be [] (else rebuild image)")
    if doc.get("tier") not in ("P0", "P1", "P2"):
        raise SubmitError("overlay manifest tier must be P0/P1/P2")
    if doc.get("capkit_code_sha") != head_sha:
        raise SubmitError("overlay manifest capkit_code_sha %r != code HEAD %r"
                          % (doc.get("capkit_code_sha"), head_sha))
    snapshot.overlay_entries(doc)
    return doc


def _check_registry(workdir, register_sha):
    row = ledger.registry_lookup(os.path.join(workdir, "REGISTRY.tsv"), register_sha)
    if row is None or ledger.registry_kind(row) != "register":
        raise SubmitError("register sha %s is not in REGISTRY.tsv as kind=register" % register_sha)
    return row


def _plan_identity(a, plan):
    """Explicit --experiment/--register-sha/--arms/--launches must equal the plan's."""
    if a.experiment != plan["experiment"]:
        raise SubmitError("--experiment %r != launch plan %r" % (a.experiment, plan["experiment"]))
    if a.register_sha != plan["register_sha"]:
        raise SubmitError("--register-sha differs from the launch plan")
    arms = a.arms.split(",")
    if any(not ARM_RE.match(x) for x in arms) or arms != launchplan.arms(plan):
        raise SubmitError("--arms %s != launch plan arms %s" % (arms, launchplan.arms(plan)))
    if a.launches != len(plan["launches"]):
        raise SubmitError("--launches %d != launch plan %d" % (a.launches, len(plan["launches"])))
    return arms


def prepare_submit(a, code_state_fn):
    """Every check that precedes any write. Returns a dict describing the submission."""
    workdir = check_path(os.path.abspath(a.workdir), "--workdir")
    code_root = check_path(os.path.abspath(a.code_root), "--code-root")
    plan = launchplan.load(a.launch_plan)
    arms = _plan_identity(a, plan)
    basis_line_present(workdir, a.basis)
    mounts = [parse_mount(m) for m in a.mount]
    tok = None if a.no_tokenizer else check_path(os.path.abspath(a.tokenizer_dir),
                                                 "--tokenizer-dir")
    assertions, tok_sha = policy.check_no_model_submission(
        a.experiment, a.gpus, [m["text"] for m in mounts], tok, a.basis)
    res = resources_from_args(a)
    need = sum(ln["budget_s"] + plan["margin_s"] for ln in plan["launches"])
    if need > res["time_s"]:
        raise SubmitError("launch budgets + margins = %d s exceed --time %s" % (need, a.time))
    paths = {k: check_path(os.path.abspath(getattr(a, k)), "--" + k.replace("_", "-"))
             for k in ("snapshot_root", "run_root", "local_base", "shm_base", "image")}
    coverage = check_coverage(
        mounts, [paths["run_root"], paths["local_base"], paths["shm_base"]],
        [paths["snapshot_root"]] + ([tok] if tok else []))
    _check_registry(workdir, a.register_sha)
    if not is_sha256_hex(a.image_sha256) or a.image_size <= 0:
        raise SubmitError("--image-sha256 must be 64 hex and --image-size > 0")
    if not os.path.isfile(paths["image"]) or os.path.getsize(paths["image"]) != a.image_size:
        raise SubmitError("image %s missing or its size != --image-size" % paths["image"])
    if not MODULE_RE.match(a.gate_module):
        raise SubmitError("bad --gate-module %r" % a.gate_module)
    if not is_sha256_hex(a.registered_hw_sha):
        raise SubmitError("--registered-hw-sha must be 64 hex")
    comparable = "" if a.no_comparable else check_path(os.path.abspath(a.comparable_file),
                                                       "--comparable-file")
    for exe, what in ((a.container_python, "--container-python"), (a.nvidia_smi, "--nvidia-smi")):
        if not EXE_RE.match(exe):
            raise SubmitError("bad %s %r" % (what, exe))
    if a.slurm_view_wait_s <= 0 or a.min_local_free_bytes <= 0:
        raise SubmitError("--slurm-view-wait-s and --min-local-free-bytes must be > 0")
    includes = list(a.include or [])
    head, dirty = code_state_fn(code_root, ["jobs", "tools"] + includes)
    if not is_git_sha(head):
        raise SubmitError("code HEAD %r is not a 40-hex sha" % head)
    overlay = _load_overlay_manifest(a.overlay_manifest, head)
    file_plan = snapshot.plan_files(code_root, includes, overlay)
    for rel, sha, size in snapshot.overlay_entries(overlay):
        src = os.path.join(code_root, rel)
        if os.path.getsize(src) != size or sha256_file(src) != sha:
            raise SubmitError("overlay source %s differs from the overlay manifest" % rel)
    return {"workdir": workdir, "code_root": code_root, "plan": plan, "arms": arms,
            "mounts": mounts, "tokenizer_dir": tok, "tok_sha": tok_sha,
            "assertions": assertions, "res": res, "paths": paths, "coverage": coverage,
            "comparable": comparable, "head": head, "dirty": dirty, "overlay": overlay,
            "includes": includes, "file_plan": sorted(file_plan)}


def _job_env(a, prep, plan_id, snap_dir):
    p = prep["paths"]
    return {
        "CAPJOBS_PLAN_ID": plan_id, "CAPJOBS_SNAP": snap_dir,
        "CAPJOBS_EXPERIMENT": a.experiment, "CAPJOBS_REGISTER_SHA": a.register_sha,
        "CAPJOBS_JOB_ARM": prep["plan"]["job_arm"], "CAPJOBS_RUN_ROOT": p["run_root"],
        "CAPJOBS_LOCAL_BASE": p["local_base"], "CAPJOBS_SHM_BASE": p["shm_base"],
        "CAPJOBS_MIN_LOCAL_FREE_BYTES": a.min_local_free_bytes,
        "CAPJOBS_IMAGE": p["image"], "CAPJOBS_IMAGE_SHA256": a.image_sha256,
        "CAPJOBS_IMAGE_SIZE": a.image_size,
        "CAPJOBS_MOUNTS": ",".join(m["text"] for m in prep["mounts"]),
        "CAPJOBS_SRUN_MPI": a.srun_mpi, "CAPJOBS_PYTHON": a.container_python,
        "CAPJOBS_GATE_MODULE": a.gate_module, "CAPJOBS_REGISTERED_HW_SHA": a.registered_hw_sha,
        "CAPJOBS_COMPARABLE_FILE": prep["comparable"],
        "CAPJOBS_SLURM_VIEW_WAIT_S": a.slurm_view_wait_s, "CAPJOBS_NVIDIA_SMI": a.nvidia_smi,
        "CAPJOBS_TIME_LIMIT_S": prep["res"]["time_s"], "CAPJOBS_EXPECT_GPUS": a.gpus,
        "CAPJOBS_CODE_SHA": prep["head"],
    }


def cmd_submit(a, runner, code_state_fn):
    prep = prepare_submit(a, code_state_fn)
    res = prep["res"]
    p = prep["paths"]
    plan_id = make_plan_id(a.experiment)
    snap_dir = os.path.join(p["snapshot_root"], plan_id)
    log_pattern = os.path.join(p["run_root"], "logs", "%x-%j.log")
    remote_argv = resources.submit_argv(res, log_pattern,
                                        os.path.join(snap_dir, SBATCH_TEMPLATE), [snap_dir])
    argv = remote.ssh_argv(a.login_host, remote_argv)
    summary = {"plan_id": plan_id, "snapshot_dir": snap_dir, "remote_argv": remote_argv,
               "argv": argv, "assertions": prep["assertions"],
               "tokenizer_dir_sha256": prep["tok_sha"], "coverage": prep["coverage"],
               "code_head": prep["head"], "dirty": prep["dirty"],
               "snapshot_files": len(prep["file_plan"]), "exclude": res["exclude"]}
    if a.dry_run:
        summary["dry_run"] = True
        summary["would_refuse_dirty"] = bool(prep["dirty"])
        emit(summary)
        return OK
    if prep["dirty"]:
        raise SubmitError("code tree has uncommitted changes under the snapshot paths: %s"
                          % prep["dirty"])
    generated = {
        "launch_plan.json": _read_bytes(a.launch_plan),
        "mounts.json": (json.dumps([m["text"] for m in prep["mounts"]]) + "\n").encode("utf-8"),
        "job.env": job_env_bytes(_job_env(a, prep, plan_id, snap_dir)),
    }
    snap, snap_sha, _entries = snapshot.create(p["snapshot_root"], plan_id, prep["code_root"],
                                               prep["includes"], a.overlay_manifest, generated)
    os.makedirs(os.path.join(p["run_root"], "logs"), exist_ok=True)
    ledger_path = os.path.join(prep["workdir"], "submissions.tsv")
    common = {"plan_id": plan_id, "experiment": a.experiment}
    ledger.submissions_append(ledger_path, ledger_row(
        event="PLANNED", arms=",".join(prep["arms"]), launches=a.launches, basis=a.basis,
        account=res["account"], partition=res["partition"], qos=res["qos"],
        exclude=",".join(res["exclude"]) or resources.EXCLUDE_NONE, time_limit=res["time"],
        mem=res["mem"], cpus_per_task=res["cpus_per_task"], gpus=res["gpus"], exclusive="true",
        snapshot_dir=snap, snapshot_sha256=snap_sha, sbatch_argv=json.dumps(remote_argv),
        mounts_json=json.dumps([m["text"] for m in prep["mounts"]]),
        tokenizer_dir_sha256=prep["tok_sha"], assertions_json=json.dumps(prep["assertions"]),
        **common))
    rc, out, err = get_runner(a, runner).run(argv)
    job_id = parse_parsable(out) if rc == 0 else None
    if job_id is None:
        reason = "sbatch rc=%r stdout=%r stderr=%r" % (rc, (out or "")[-300:], (err or "")[-300:])
        ledger.submissions_append(ledger_path, ledger_row(
            event="ENDED", status="SUBMIT_FAILED", reason=reason.replace("\t", " ")
            .replace("\n", " ").replace("\r", " "), **common))
        summary.update({"submitted": False, "rc": rc, "stdout": out, "stderr": err})
        emit(summary)
        return RUNTIME
    ledger.submissions_append(ledger_path, ledger_row(
        event="SUBMITTED", job_id=job_id, submitted_at=wall_iso(), **common))
    summary.update({"submitted": True, "job_id": job_id, "snapshot_sha256": snap_sha})
    emit(summary)
    return OK


def cmd_local_record(a):
    """PLANNED + SUBMITTED rows for a local run (job id local-*), same five assertions."""
    workdir = check_path(os.path.abspath(a.workdir), "--workdir")
    if not a.job_id.startswith("local-") or not JOB_ID_RE.match(a.job_id):
        raise SubmitError("local runs use job ids 'local-<x>'")
    arms = a.arms.split(",")
    if any(not ARM_RE.match(x) for x in arms) or a.launches <= 0:
        raise SubmitError("bad --arms or --launches")
    basis_line_present(workdir, a.basis)
    mounts = [parse_mount(m)["text"] for m in (a.mount or [])]
    tok = None if a.no_tokenizer else os.path.abspath(a.tokenizer_dir)
    assertions, tok_sha = policy.check_no_model_submission(a.experiment, 0, mounts, tok, a.basis)
    _check_registry(workdir, a.register_sha)
    ledger_path = os.path.join(workdir, "submissions.tsv")
    if ledger.submission_for_job(ledger_path, a.job_id) is not None:
        raise SubmitError("job id %s is already in the ledger" % a.job_id)
    plan_id = make_plan_id(a.experiment)
    common = {"plan_id": plan_id, "experiment": a.experiment}
    ledger.submissions_append(ledger_path, ledger_row(
        event="PLANNED", arms=",".join(arms), launches=a.launches, basis=a.basis, gpus=0,
        exclusive="false", mounts_json=json.dumps(mounts), tokenizer_dir_sha256=tok_sha,
        assertions_json=json.dumps(assertions), **common))
    ledger.submissions_append(ledger_path, ledger_row(
        event="SUBMITTED", job_id=a.job_id, submitted_at=wall_iso(), **common))
    emit({"plan_id": plan_id, "job_id": a.job_id, "ledger": ledger_path})
    return OK


def cmd_slurm_view(a, runner):
    if not JOB_ID_RE.match(a.job_id) or not a.job_id.isdigit():
        raise SubmitError("--job-id must be a numeric cluster job id")
    doc = slurmview.collect(get_runner(a, runner), a.login_host, a.job_id)
    path = slurmview.write(os.path.abspath(a.job_dir), doc)
    emit({"path": path, "node": doc["node"], "rcs": [i["rc"] for i in doc["items"]]})
    return OK


def cmd_jobstate(a, runner):
    q = jobstate.query(get_runner(a, runner), a.login_host, a.job_id)
    emit({k: q[k] for k in ("job_id", "state", "reason", "sacct_state")})
    return JOBSTATE_EXIT[q["state"]]


def cmd_record_end(a, runner):
    """ENDED row for a plan, only after the end criterion says ENDED (cluster jobs)."""
    workdir = check_path(os.path.abspath(a.workdir), "--workdir")
    ledger_path = os.path.join(workdir, "submissions.tsv")
    cur = ledger.submissions_current(ledger_path).get(a.plan_id)
    if cur is None:
        raise SubmitError("plan id %s is not in the ledger" % a.plan_id)
    if cur["event"] == "ENDED":
        raise SubmitError("plan %s already has an ENDED row" % a.plan_id)
    if a.status not in ("VALID", "VOID", "NOT_RUN", "ENV_INVALID", "CANCELLED"):
        raise SubmitError("bad --status %r" % a.status)
    if not a.reason.strip() or "\t" in a.reason or "\n" in a.reason:
        raise SubmitError("--reason must be one non-empty line")
    job_id = cur["job_id"]
    evidence = "local run"
    if not job_id:
        raise SubmitError("plan %s has no job id (never submitted)" % a.plan_id)
    if not job_id.startswith("local-"):
        q = jobstate.query(get_runner(a, runner), a.login_host, job_id)
        if q["state"] != jobstate.ENDED:
            raise SubmitError("job %s is %s (%s); not recording an end" %
                              (job_id, q["state"], q["reason"]))
        evidence = q["reason"]
    ledger.submissions_append(ledger_path, ledger_row(
        event="ENDED", plan_id=a.plan_id, job_id=job_id, status=a.status,
        reason="%s [end criterion: %s]" % (a.reason, evidence), experiment=cur["experiment"]))
    emit({"plan_id": a.plan_id, "job_id": job_id, "status": a.status, "evidence": evidence})
    return OK


def cmd_exclude(a, runner):
    with open(a.main_profile_spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    parts = a.partition.split(",")
    if a.cdb_file is not None:
        with open(a.cdb_file, "r", encoding="utf-8") as f:
            text = f.read()
        source = {"cdb_file": os.path.abspath(a.cdb_file)}
    else:
        if not a.cdb_query or not a.raw_out or not a.login_host or not a.ssh_timeout_s:
            raise SubmitError("--live needs --cdb-query, --raw-out, --login-host, --ssh-timeout-s")
        cdbexclude.load_spec(spec)
        fields = [cdbexclude.HOST_COL] + sorted(spec["fields"]) + [cdbexclude.PART_COL]
        remote_argv = ["cdb", "find", a.cdb_query, "--fields", ",".join(fields), "-o", "csv"]
        rc, out, err = get_runner(a, runner).run(remote.ssh_argv(a.login_host, remote_argv))
        text = "# cmd: %s\n%s# exit: %s\n" % (shlex.join(remote_argv), out or "", rc)
        with open(a.raw_out, "x", encoding="utf-8") as f:
            f.write(text)
        if rc != 0:
            raise SubmitError("cdb find failed rc=%r: %s" % (rc, (err or "").strip()))
        source = {"live": remote_argv, "raw_out": os.path.abspath(a.raw_out)}
    doc = cdbexclude.compute_exclude(text, spec, parts)
    doc["source"] = source
    doc["computed_at"] = wall_iso()
    with open(a.out, "x", encoding="utf-8") as f:
        f.write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    emit({"out": os.path.abspath(a.out), "exclude": doc["exclude"],
          "nodes_in_partitions": doc["nodes_in_partitions"],
          "nodes_matching": len(doc["nodes_matching"])})
    return OK


def cmd_prep_tokenizer(a):
    """Draft only: prints the would-be sbatch command; never submits (needs user approval and
    a new, approved entry; CONSTRAINTS 2 item 3, RESOURCES 9.1 tokenizer note)."""
    if not a.dry_run:
        raise SubmitError("prep-tokenizer is a draft that needs user approval: only --dry-run "
                          "is implemented, nothing is submitted")
    code_root = check_path(os.path.abspath(a.code_root), "--code-root")
    model_dir = check_path(os.path.abspath(a.model_dir), "--model-dir")
    if not ARM_RE.match(a.name):
        raise SubmitError("bad tokenizer --name %r" % a.name)
    dest = os.path.join(code_root, "configs", "tokenizers", a.name)
    if os.path.exists(dest):
        raise SubmitError("destination %s already exists" % dest)
    res = resources_from_args(a)
    if res["gpus"] != 0:
        raise SubmitError("prep-tokenizer requests no GPU (--gpus 0)")
    script = os.path.join(code_root, PREP_TOKENIZER_TEMPLATE)
    log = os.path.join(code_root, "configs", "tokenizers", "%x-%j.log")
    remote_argv = ["sbatch", "--parsable"] + resources.resource_flags(res) + [
        "--output=" + log, "--export=NONE", script, model_dir, dest]
    emit({"dry_run": True, "draft": True, "remote_argv": remote_argv,
          "argv": remote.ssh_argv(a.login_host, remote_argv),
          "note": "not submitted; the template is jobs/prep_tokenizer.sbatch.draft"})
    return OK

# ---- SECTION: main ------------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(prog="python3.11 -m capjobs.submit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="sbatch --test-only (never creates a job)")
    add_resource_args(p)
    add_remote_args(p)
    p.add_argument("--probe-log", required=True)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("submit", help="snapshot, assertions, ledger, sbatch --parsable")
    add_resource_args(p)
    add_remote_args(p)
    p.add_argument("--workdir", required=True)
    p.add_argument("--code-root", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--register-sha", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--launches", required=True, type=int)
    p.add_argument("--basis", required=True)
    p.add_argument("--launch-plan", required=True)
    p.add_argument("--overlay-manifest", required=True)
    p.add_argument("--include", action="append", help="relpath under --code-root (repeatable)")
    p.add_argument("--image", required=True)
    p.add_argument("--image-sha256", required=True)
    p.add_argument("--image-size", required=True, type=int)
    p.add_argument("--mount", required=True, action="append", help="SRC:DST[:ro|rw]")
    tk = p.add_mutually_exclusive_group(required=True)
    tk.add_argument("--tokenizer-dir")
    tk.add_argument("--no-tokenizer", action="store_true")
    p.add_argument("--snapshot-root", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--local-base", required=True)
    p.add_argument("--shm-base", required=True)
    p.add_argument("--min-local-free-bytes", required=True, type=int)
    p.add_argument("--srun-mpi", required=True, choices=("none", "pmix"))
    p.add_argument("--container-python", required=True)
    p.add_argument("--nvidia-smi", required=True)
    p.add_argument("--gate-module", required=True)
    p.add_argument("--registered-hw-sha", required=True)
    cp = p.add_mutually_exclusive_group(required=True)
    cp.add_argument("--comparable-file")
    cp.add_argument("--no-comparable", action="store_true")
    p.add_argument("--slurm-view-wait-s", required=True, type=float)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("local-record", help="ledger rows for a local run (job id local-*)")
    p.add_argument("--workdir", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--register-sha", required=True)
    p.add_argument("--arms", required=True)
    p.add_argument("--launches", required=True, type=int)
    p.add_argument("--job-id", required=True)
    p.add_argument("--basis", required=True)
    p.add_argument("--mount", action="append")
    tk = p.add_mutually_exclusive_group(required=True)
    tk.add_argument("--tokenizer-dir")
    tk.add_argument("--no-tokenizer", action="store_true")

    p = sub.add_parser("slurm-view", help="write <job dir>/env/slurm_view.json")
    add_remote_args(p)
    p.add_argument("--job-id", required=True)
    p.add_argument("--job-dir", required=True)

    p = sub.add_parser("jobstate", help="exit 0 ENDED / 20 NOT_ENDED / 21 UNKNOWN")
    add_remote_args(p)
    p.add_argument("--job-id", required=True)

    p = sub.add_parser("record-end", help="ENDED ledger row after the end criterion holds")
    add_remote_args(p)
    p.add_argument("--workdir", required=True)
    p.add_argument("--plan-id", required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--reason", required=True)

    p = sub.add_parser("exclude", help="compute --exclude from cdb output")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cdb-file")
    src.add_argument("--live", action="store_true")
    p.add_argument("--cdb-query")
    p.add_argument("--raw-out")
    p.add_argument("--login-host")
    p.add_argument("--ssh-timeout-s", type=float)
    p.add_argument("--main-profile-spec", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("prep-tokenizer", help="DRAFT: dry-run only, needs user approval")
    add_resource_args(p)
    p.add_argument("--login-host", required=True)
    p.add_argument("--code-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--dry-run", action="store_true")
    return ap


def main(argv=None, runner=None, code_state_fn=None):
    """runner / code_state_fn are injectable for tests (a stub runner never executes ssh)."""
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "probe":
            return cmd_probe(a, runner)
        if a.cmd == "submit":
            return cmd_submit(a, runner, code_state_fn or code_state)
        if a.cmd == "local-record":
            return cmd_local_record(a)
        if a.cmd == "slurm-view":
            return cmd_slurm_view(a, runner)
        if a.cmd == "jobstate":
            return cmd_jobstate(a, runner)
        if a.cmd == "record-end":
            return cmd_record_end(a, runner)
        if a.cmd == "exclude":
            return cmd_exclude(a, runner)
        if a.cmd == "prep-tokenizer":
            return cmd_prep_tokenizer(a)
    except CapkitError as e:
        print("capjobs.submit: %s" % e, file=sys.stderr)
        return e.exit_code
    except (OSError, ValueError) as e:
        print("capjobs.submit: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return CONFIG
    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
