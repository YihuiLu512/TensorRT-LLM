# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The only submission entry for jobs that do not load a real model (INTERFACES.md 9.5, 11;
plan CONSTRAINTS section 2, RESOURCES sections 3-4).

    PYTHONPATH=<C>/tools:<C>/jobs python3.11 -m capjobs.submit <sub-command> ...

Sub-commands: probe (the only path that runs `sbatch --test-only`), submit, reconcile-submit,
local-record, slurm-view, jobstate, record-end, exclude, prep-tokenizer (the tokenizer copy, the
only job that may mount a registered model dir read-only). See jobs/README.md. Nothing has a default resource value; every cluster command goes through
capjobs.remote (ssh, allow-listed commands).
"""

import argparse
import contextlib
import glob
import io
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
from capcommon.hashing import is_git_sha, is_sha256_hex, sha256_bytes, sha256_file
from capcommon.identity import ARM_RE, JOB_ID_RE

from . import cdbexclude, jobstate, launchplan, remote, resources, slurmview, snapshot

PATH_RE = re.compile(r"^/[A-Za-z0-9._/@+-]*$")
MODULE_RE = re.compile(r"^[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*$")
EXE_RE = re.compile(r"^[A-Za-z0-9._/+-]+$")
JOBSTATE_EXIT = {jobstate.ENDED: 0, jobstate.NOT_ENDED: 20, jobstate.UNKNOWN: 21}
SBATCH_TEMPLATE = "jobs/capkit_job.sbatch"
PREP_TOKENIZER_TEMPLATE = "jobs/prep_tokenizer.sbatch"
REGISTER_SCHEMA = "capkit.register/1"
# register.json "environment" section: the job-level hard limits the job passes to its checks
# (exactly these keys, all required; see jobs/README.md).
REGISTER_ENV_KEYS = ("gpus", "idle_window_s", "idle_busy_max_frac", "slurm_view_wait_s",
                     "chassis_source")
CHASSIS_SOURCES = ("slurm_view", "dmi")
HW_UNREGISTERED = "UNREGISTERED"
# Purpose marker on every PLANNED row (reason column) and repeated on its NOTE / ENDED rows: a
# hw_profile_capture job (--registered-hw-sha UNREGISTERED) always ends ENV_INVALID by design and
# must not be read as an environment failure (e.g. CONSTRAINTS 7 "two ENV_INVALID in a row").
PURPOSE_MEASUREMENT = "purpose=measurement"
PURPOSE_HW_CAPTURE = "purpose=hw_profile_capture"
# A submission whose sbatch outcome is unknown (ssh failed or timed out, or sbatch printed no job
# id) is a NOTE row with status UNKNOWN_SUBMIT; only reconcile-submit moves it on (never a
# resubmit).
UNKNOWN_SUBMIT = ledger.UNKNOWN_SUBMIT
SSH_FAILED_RC = 255
# Every job's container mounts include an identity mount of each of these: the secondary scratch
# disk, where primary-disk directories may resolve (plan RESOURCES section 6).
REQUIRED_IDENTITY_MOUNTS = ("/home/scratch.yihlu_coreai_1",)
PREP_PLAN_SCHEMA = "capkit.prep_plan/1"
PREP_PLAN_MANIFEST = "MANIFEST.json"


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


def job_mounts(mounts, need_rw, need_ro):
    """The mount checks of every job (submit and prep-tokenizer) on parse_mount results: an
    identity mount of every REQUIRED_IDENTITY_MOUNTS root, then check_coverage of the needed
    paths. Returns the coverage checks."""
    for root in REQUIRED_IDENTITY_MOUNTS:
        if not any(m["src"] == m["dst"] == root for m in mounts):
            raise SubmitError("the mounts must include %s:%s (secondary scratch disk; "
                              "primary-disk directories may be symlinks into it)" % (root, root))
    return check_coverage(mounts, need_rw, need_ro)


def git_head(code_root):
    head = subprocess.run(["git", "-C", code_root, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False)
    if head.returncode != 0:
        raise SubmitError("git rev-parse failed in %s: %s" % (code_root, head.stderr.strip()))
    sha = head.stdout.strip()
    if not is_git_sha(sha):
        raise SubmitError("code HEAD %r is not a 40-hex sha" % sha)
    return sha


def committed_file(code_root, rel):
    """(HEAD sha, bytes of rel at HEAD, problem). problem is None when the working-tree file is
    byte-identical to the committed one, else why it is not; read-only git."""
    head = git_head(code_root)
    blob = subprocess.run(["git", "-C", code_root, "cat-file", "blob", "%s:%s" % (head, rel)],
                          capture_output=True, check=False)
    if blob.returncode != 0:
        return head, None, "%s is not committed at HEAD %s" % (rel, head)
    try:
        with open(os.path.join(code_root, rel), "rb") as f:
            work = f.read()
    except OSError as e:
        return head, blob.stdout, "cannot read the working-tree %s: %s" % (rel, e)
    if work != blob.stdout:
        return head, blob.stdout, "the working-tree %s differs from HEAD %s" % (rel, head)
    return head, blob.stdout, None


def code_state(code_root, paths):
    """(HEAD sha, porcelain status lines limited to paths) of the code repo; read-only git."""
    head = git_head(code_root)
    st = subprocess.run(["git", "-C", code_root, "status", "--porcelain", "--"] + list(paths),
                        capture_output=True, text=True, check=False)
    if st.returncode != 0:
        raise SubmitError("git status failed in %s: %s" % (code_root, st.stderr.strip()))
    return head, [ln for ln in st.stdout.splitlines() if ln.strip()]


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
    return jid if jid.isascii() and jid.isdigit() else None


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


def _num(v, lo, hi, lo_open=True):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:
        return False
    return (v > lo if lo_open else v >= lo) and v <= hi


def register_environment(workdir, row, register_sha, experiment):
    """The register.json the REGISTRY row names (sha recomputed) and its "environment" section:
    exactly REGISTER_ENV_KEYS. These values reach the job only through job.env."""
    rel = row["relpath"]
    if os.path.isabs(rel) or rel.startswith("../") or "/../" in rel:
        raise SubmitError("REGISTRY relpath %r must be relative to --workdir" % rel)
    path = os.path.join(workdir, rel)
    try:
        sha = sha256_file(path)
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError) as e:
        raise SubmitError("cannot read register %s: %s" % (path, e))
    if sha != register_sha:
        raise SubmitError("register %s sha256 %s != --register-sha" % (path, sha))
    if not isinstance(doc, dict) or doc.get("schema") != REGISTER_SCHEMA:
        raise SubmitError("register %s is not %s" % (path, REGISTER_SCHEMA))
    if doc.get("experiment") != experiment:
        raise SubmitError("register experiment %r != --experiment %r"
                          % (doc.get("experiment"), experiment))
    env = doc.get("environment")
    if not isinstance(env, dict) or set(env) != set(REGISTER_ENV_KEYS):
        raise SubmitError("register %s: 'environment' must have exactly %s" %
                          (path, REGISTER_ENV_KEYS))
    g = env["gpus"]
    if not isinstance(g, int) or isinstance(g, bool) or not 0 <= g <= policy.MAX_GPUS:
        raise SubmitError("register environment.gpus must be an int in [0, %d]" % policy.MAX_GPUS)
    if not _num(env["idle_window_s"], 0, 3600):
        raise SubmitError("register environment.idle_window_s must be in (0, 3600]")
    if not _num(env["idle_busy_max_frac"], 0, 1):
        raise SubmitError("register environment.idle_busy_max_frac must be in (0, 1]")
    if not _num(env["slurm_view_wait_s"], 0, 3600):
        raise SubmitError("register environment.slurm_view_wait_s must be in (0, 3600]")
    if env["chassis_source"] not in CHASSIS_SOURCES:
        raise SubmitError("register environment.chassis_source must be one of %s"
                          % (CHASSIS_SOURCES,))
    return {"path": path, "sha256": sha, "environment": dict(env), "doc": doc}


def arm_files(register_doc, arm):
    """register.json files for one arm (files, or files_by_arm[arm]) as caplaunch reads them;
    None when the arm has none."""
    has_f = "files" in register_doc
    if has_f == ("files_by_arm" in register_doc):
        raise SubmitError("register needs exactly one of files / files_by_arm")
    files = register_doc["files"] if has_f else (register_doc["files_by_arm"] or {}).get(arm)
    return files if isinstance(files, dict) else None


def check_plan_scenarios(plan, workdir, register_doc):
    """Each launch's --workdir is --workdir and its --scenario is the file register.json lists
    as files.scenario for that launch's arm (same relpath, same sha256): the check caplaunch
    makes in the job, made before anything is submitted."""
    root = os.path.realpath(workdir)
    for ln in plan["launches"]:
        wd, rel = launchplan.scenario_path(ln)
        if os.path.realpath(wd) != root:
            raise SubmitError("launch %d: --workdir %s is not --workdir %s"
                              % (ln["launch_seq"], wd, workdir))
        files = arm_files(register_doc, ln["arm"])
        regd = files.get("scenario") if files is not None else None
        sha = sha256_file(os.path.join(root, rel))
        if not isinstance(regd, dict) or regd.get("relpath") != rel or regd.get("sha256") != sha:
            raise SubmitError("launch %d: scenario %s (sha %s) is not the register's "
                              "files.scenario %r for arm %s"
                              % (ln["launch_seq"], rel, sha, regd, ln["arm"]))


# Stand-ins for what capjobs.launches fills in on the node; they only have to parse the way the
# real values do (JOB_ID and RESTART_COUNT are integers there too).
EXAMPLE_PLACEHOLDER_VALUES = {"PY": "/example/python", "SNAP": "/example/snap",
                              "JOB_DIR": "/example/job", "SHM_ROOT": "/example/shm",
                              "LOCAL_ROOT": "/example/local", "JOB_ID": "1", "RESTART_COUNT": "0"}


def check_plan_argv(plan):
    """Parse each launch's caplaunch arguments with caplaunch's own parser (build_parser) the way
    the job will run them: the argv capjobs.launches builds (launch_argv: placeholders
    substituted, here with example values; identity, --run-dir, --pin-table and the other
    appended arguments added), from after `-m caplaunch`. A missing required flag, an
    abbreviation, an unknown flag or a value of the wrong type is refused here; in the job it
    would make caplaunch exit 2 before starting any role."""
    from caplaunch.__main__ import build_parser  # noqa: PLC0415

    from .launches import launch_argv  # noqa: PLC0415

    def refuse(message):
        raise SubmitError(message)

    for ln in plan["launches"]:
        argv = launch_argv(plan, ln, EXAMPLE_PLACEHOLDER_VALUES,
                           EXAMPLE_PLACEHOLDER_VALUES["JOB_ID"], 0, "/example/run",
                           "/example/overlay_manifest.json", "/example/nvidia-smi",
                           "/example/env", "/example/pin_table.json")
        start = next((i + 2 for i in range(1, len(argv) - 1)
                      if argv[i:i + 2] == ["-m", "caplaunch"]), None)
        if start is None:
            raise SubmitError("launch %d: argv must run <python> -m caplaunch"
                              % ln["launch_seq"])
        parser = build_parser()
        parser.error = refuse
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                parser.parse_args(argv[start:])
        except SubmitError as e:
            raise SubmitError("launch %d: caplaunch would refuse its argv: %s"
                              % (ln["launch_seq"], e)) from None
        except SystemExit:
            raise SubmitError("launch %d: caplaunch would refuse its argv: it exited while "
                              "parsing (e.g. -h)" % ln["launch_seq"]) from None


def workdir_inputs(workdir, reg, plan):
    """The files in <W> the job reads for this plan: the register and every files.* entry of
    each launch's arm (check_coverage also checks each one's realpath, so a symlink out of <W>
    needs its own mount)."""
    out = [reg["path"]]
    for ln in plan["launches"]:
        for e in (arm_files(reg["doc"], ln["arm"]) or {}).values():
            if isinstance(e, dict) and isinstance(e.get("relpath"), str):
                out.append(os.path.join(workdir, e["relpath"]))
    return sorted(set(out))


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
    paths = {k: check_path(os.path.abspath(getattr(a, k)), "--" + k.replace("_", "-"))
             for k in ("snapshot_root", "run_root", "local_base", "shm_base", "image")}
    # <W> is read in the job (capjobs.launches and caplaunch read the registered scenario,
    # register, tier table, schedule and prompts there): ro coverage is enough
    coverage = job_mounts(
        mounts, [paths["run_root"], paths["local_base"], paths["shm_base"]],
        [paths["snapshot_root"], workdir] + ([tok] if tok else []))
    reg = register_environment(workdir, _check_registry(workdir, a.register_sha),
                               a.register_sha, a.experiment)
    renv = reg["environment"]
    if res["gpus"] != renv["gpus"]:
        raise SubmitError("--gpus %d != register environment.gpus %d" % (res["gpus"],
                                                                        renv["gpus"]))
    if res["gpus"] == 0:
        print("capjobs.submit: WARNING: 0 GPUs requested: no GPU positive control, so the "
              "judge's GPU-IDLE check will be NOT_RUN for every launch", file=sys.stderr)
    budget = time_budget(plan, renv)
    if budget["total_s"] > res["time_s"]:
        raise SubmitError("time budget %s = %d s exceeds --time %s"
                          % (budget["parts"], budget["total_s"], a.time))
    image = check_image(paths["image"], a.image_sha256, a.image_size)
    if not MODULE_RE.match(a.gate_module):
        raise SubmitError("bad --gate-module %r" % a.gate_module)
    if not (is_sha256_hex(a.registered_hw_sha) or a.registered_hw_sha == HW_UNREGISTERED):
        raise SubmitError("--registered-hw-sha must be 64 hex or %s" % HW_UNREGISTERED)
    comparable = "" if a.no_comparable else check_path(os.path.abspath(a.comparable_file),
                                                       "--comparable-file")
    if not EXE_RE.match(a.container_python):
        raise SubmitError("bad --container-python %r" % a.container_python)
    # an absolute path: the job's GPU checks never fall back to whatever PATH finds
    check_path(a.nvidia_smi, "--nvidia-smi")
    if a.min_local_free_bytes <= 0:
        raise SubmitError("--min-local-free-bytes must be > 0")
    includes = list(a.include or [])
    head, dirty = code_state_fn(code_root, ["jobs", "tools"] + includes)
    if not is_git_sha(head):
        raise SubmitError("code HEAD %r is not a 40-hex sha" % head)
    overlay = _load_overlay_manifest(a.overlay_manifest, head)
    file_plan = snapshot.plan_files(code_root, includes, overlay)
    launchplan.check_kill_grace(plan)
    check_plan_scenarios(plan, workdir, reg["doc"])
    coverage += check_coverage(mounts, [], workdir_inputs(workdir, reg, plan))
    for ln in plan["launches"]:
        # caplaunch sets CAPKIT_CODE_SHA = --capkit-sha and refuses a different outer value; in the
        # job the outer value is the overlay manifest's capkit_code_sha, i.e. this HEAD
        if launchplan.flag_value(ln["argv"], "--capkit-sha") != head:
            raise SubmitError("launch %d: --capkit-sha must be the code HEAD %s"
                              % (ln["launch_seq"], head))
    check_plan_argv(plan)
    for rel, sha, size in snapshot.overlay_entries(overlay):
        src = os.path.join(code_root, rel)
        if os.path.getsize(src) != size or sha256_file(src) != sha:
            raise SubmitError("overlay source %s differs from the overlay manifest" % rel)
    return {"workdir": workdir, "code_root": code_root, "plan": plan, "arms": arms,
            "mounts": mounts, "tokenizer_dir": tok, "tok_sha": tok_sha,
            "assertions": assertions, "res": res, "paths": paths, "coverage": coverage,
            "comparable": comparable, "head": head, "dirty": dirty, "overlay": overlay,
            "includes": includes, "file_plan": sorted(file_plan), "register": reg,
            "budget": budget, "image": image}


def time_budget(plan, renv):
    """Every in-allocation wait that has a registered or planned bound: per launch budget_s +
    kill_grace_s + margin_s and the pin-table generation bound, plus the gate's idle window and
    the longest slurm_view wait."""
    parts = {"launches": sum(ln["budget_s"] + plan["kill_grace_s"] + plan["margin_s"]
                             for ln in plan["launches"]),
             "pin_tables_s": launchplan.PINS_TIMEOUT_S * len(plan["launches"]),
             "gate_idle_window_s": renv["idle_window_s"],
             "slurm_view_wait_s": renv["slurm_view_wait_s"]}
    return {"parts": parts, "total_s": sum(parts.values())}


def check_image(path, want_sha, want_size):
    """The image is hashed here, once per submission, outside the allocation; the job only
    re-checks size and mtime_ns against the values recorded now (job.env)."""
    if not is_sha256_hex(want_sha) or want_size <= 0:
        raise SubmitError("--image-sha256 must be 64 hex and --image-size > 0")
    try:
        st = os.stat(path)
    except OSError as e:
        raise SubmitError("image %s: %s" % (path, e))
    if not os.path.isfile(path) or st.st_size != want_size:
        raise SubmitError("image %s missing or its size != --image-size" % path)
    got = sha256_file(path)
    st2 = os.stat(path)
    if (st2.st_size, st2.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
        raise SubmitError("image %s changed while it was being hashed" % path)
    if got != want_sha:
        raise SubmitError("image %s sha256 %s != --image-sha256" % (path, got))
    return {"sha256": got, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _job_env(a, prep, plan_id, snap_dir):
    p = prep["paths"]
    renv = prep["register"]["environment"]
    return {
        "CAPJOBS_PLAN_ID": plan_id, "CAPJOBS_SNAP": snap_dir,
        "CAPJOBS_WORKDIR": prep["workdir"], "CAPJOBS_EXPERIMENT": a.experiment,
        "CAPJOBS_REGISTER_SHA": a.register_sha,
        "CAPJOBS_JOB_ARM": prep["plan"]["job_arm"], "CAPJOBS_RUN_ROOT": p["run_root"],
        "CAPJOBS_LOCAL_BASE": p["local_base"], "CAPJOBS_SHM_BASE": p["shm_base"],
        "CAPJOBS_MIN_LOCAL_FREE_BYTES": a.min_local_free_bytes,
        "CAPJOBS_IMAGE": p["image"], "CAPJOBS_IMAGE_SHA256": prep["image"]["sha256"],
        "CAPJOBS_IMAGE_SIZE": prep["image"]["size"],
        "CAPJOBS_IMAGE_MTIME_NS": prep["image"]["mtime_ns"],
        "CAPJOBS_MOUNTS": ",".join(m["text"] for m in prep["mounts"]),
        "CAPJOBS_SRUN_MPI": a.srun_mpi, "CAPJOBS_PYTHON": a.container_python,
        "CAPJOBS_GATE_MODULE": a.gate_module, "CAPJOBS_REGISTERED_HW_SHA": a.registered_hw_sha,
        "CAPJOBS_COMPARABLE_FILE": prep["comparable"],
        "CAPJOBS_NVIDIA_SMI": a.nvidia_smi, "CAPJOBS_TIME_LIMIT_S": prep["res"]["time_s"],
        "CAPJOBS_CODE_SHA": prep["head"],
        # register.json "environment" (RS8: hard limits come from the register, never defaults)
        "CAPJOBS_MAX_ALLOC_GPUS": renv["gpus"], "CAPJOBS_IDLE_WINDOW_S": renv["idle_window_s"],
        "CAPJOBS_IDLE_BUSY_MAX_FRAC": renv["idle_busy_max_frac"],
        "CAPJOBS_SLURM_VIEW_WAIT_S": renv["slurm_view_wait_s"],
        "CAPJOBS_CHASSIS_SOURCE": renv["chassis_source"],
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
               "snapshot_files": len(prep["file_plan"]), "exclude": res["exclude"],
               "register_environment": prep["register"]["environment"],
               "time_budget": prep["budget"], "image": prep["image"]}
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
    purpose = submission_purpose(a.registered_hw_sha)
    ledger.submissions_append(ledger_path, ledger_row(
        event="PLANNED", reason=purpose, arms=",".join(prep["arms"]), launches=a.launches,
        basis=a.basis,
        account=res["account"], partition=res["partition"], qos=res["qos"],
        exclude=",".join(res["exclude"]) or resources.EXCLUDE_NONE, time_limit=res["time"],
        mem=res["mem"], cpus_per_task=res["cpus_per_task"], gpus=res["gpus"], exclusive="true",
        snapshot_dir=snap, snapshot_sha256=snap_sha, sbatch_argv=json.dumps(remote_argv),
        mounts_json=json.dumps([m["text"] for m in prep["mounts"]]),
        tokenizer_dir_sha256=prep["tok_sha"], assertions_json=json.dumps(prep["assertions"]),
        **common))
    attempted_at = wall_iso()
    rc, out, err = get_runner(a, runner).run(argv)
    job_id = parse_parsable(out) if rc == 0 else None
    if job_id is None:
        reason = _one_line("sbatch rc=%r stdout=%r stderr=%r"
                           % (rc, (out or "")[-300:], (err or "")[-300:]))
        if submit_outcome_unknown(rc):
            # The job may be queued: never resubmit, never close the plan automatically.
            ledger.submissions_append(ledger_path, ledger_row(
                event="NOTE", status=UNKNOWN_SUBMIT, submitted_at=attempted_at,
                reason="%s; %s; reconcile with `reconcile-submit` after checking squeue/sacct"
                % (purpose, reason), **common))
            summary.update({"submitted": None, "unknown_submit": True, "rc": rc,
                            "stdout": out, "stderr": err})
            emit(summary)
            return RUNTIME
        ledger.submissions_append(ledger_path, ledger_row(
            event="ENDED", status="SUBMIT_FAILED", reason="%s; %s" % (purpose, reason),
            **common))
        summary.update({"submitted": False, "rc": rc, "stdout": out, "stderr": err})
        emit(summary)
        return RUNTIME
    ledger.submissions_append(ledger_path, ledger_row(
        event="SUBMITTED", job_id=job_id, submitted_at=wall_iso(), **common))
    summary.update({"submitted": True, "job_id": job_id, "snapshot_sha256": snap_sha})
    emit(summary)
    return OK


def submission_purpose(registered_hw_sha):
    return PURPOSE_HW_CAPTURE if registered_hw_sha == HW_UNREGISTERED else PURPOSE_MEASUREMENT


def plan_purpose(ledger_path, plan_id):
    """The purpose marker of the plan's PLANNED row ('' for rows written before the marker)."""
    for row in ledger.read_tsv(ledger_path, ledger.SUBMISSION_COLUMNS):
        if row["plan_id"] == plan_id and row["event"] == "PLANNED":
            return row["reason"] if row["reason"].startswith("purpose=") else ""
    return ""


def _one_line(text):
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def submit_outcome_unknown(rc):
    """True unless sbatch itself reported a failure: ssh could not run or timed out (None), ssh
    failed (255, the remote command may or may not have run), or sbatch exited 0 without a
    parsable job id."""
    return rc is None or rc == 0 or rc == SSH_FAILED_RC


def _is_unknown_submit(cur):
    return ledger.submission_unknown(cur)


def _snap_run_root(snap_dir):
    """CAPJOBS_RUN_ROOT from the plan's own job.env (the job writes its batch.tsv there)."""
    with open(os.path.join(snap_dir, "job.env"), "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("export CAPJOBS_RUN_ROOT="):
                return shlex.split(line.split("=", 1)[1])[0]
    raise SubmitError("no CAPJOBS_RUN_ROOT in %s/job.env" % snap_dir)


def _batch_tsv_plans(run_root, job_id="*"):
    """{batch.tsv path: plan_id} for runs/J<job>-r<n>/batch.tsv under run_root."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_root, "J%s-r*" % job_id, "batch.tsv"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                kv = dict(ln.rstrip("\n").split("\t", 1) for ln in f if "\t" in ln)
        except OSError:
            continue
        out[p] = kv.get("plan_id")
    return out


def cmd_reconcile_submit(a, runner):
    """Close an UNKNOWN_SUBMIT plan from evidence, never by resubmitting:
    --job-id J  needs `scontrol show job J` naming this plan's batch script, or a
                <run root>/J<J>-r*/batch.tsv written by the job with this plan id -> SUBMITTED;
    --not-submitted needs `squeue` (rc 0) to list no job running this plan's script and no
                batch.tsv with this plan id -> ENDED SUBMIT_FAILED."""
    workdir = check_path(os.path.abspath(a.workdir), "--workdir")
    ledger_path = os.path.join(workdir, "submissions.tsv")
    cur = ledger.submissions_current(ledger_path).get(a.plan_id)
    if cur is None or not _is_unknown_submit(cur):
        raise SubmitError("plan %s is not in state %s" % (a.plan_id, UNKNOWN_SUBMIT))
    if not a.reason.strip() or "\t" in a.reason or "\n" in a.reason:
        raise SubmitError("--reason must be one non-empty line")
    snap = cur["snapshot_dir"]
    prep = plan_purpose(ledger_path, a.plan_id) == policy.PREP_TOKENIZER_PURPOSE
    script = os.path.join(snap, PREP_TOKENIZER_TEMPLATE if prep else SBATCH_TEMPLATE)
    run_root = _snap_run_root(snap)
    mine = [p for p, pid in _batch_tsv_plans(run_root).items() if pid == a.plan_id]
    common = {"plan_id": a.plan_id, "experiment": cur["experiment"]}
    run = get_runner(a, runner)
    if a.job_id is not None:
        if not (a.job_id.isascii() and a.job_id.isdigit()):
            raise SubmitError("--job-id must be a numeric cluster job id")
        sc = remote.run_remote(run, a.login_host, ["scontrol", "show", "job", a.job_id])
        by_scontrol = sc["rc"] == 0 and ("Command=" + script) in (sc["stdout"] or "").split()
        by_tsv = [p for p, pid in _batch_tsv_plans(run_root, a.job_id).items()
                  if pid == a.plan_id]
        if not (by_scontrol or by_tsv):
            raise SubmitError("no evidence that job %s runs plan %s (scontrol rc=%r; no "
                              "batch.tsv with this plan id)" % (a.job_id, a.plan_id, sc["rc"]))
        if ledger.submission_for_job(ledger_path, a.job_id) is not None:
            raise SubmitError("job id %s is already in the ledger" % a.job_id)
        evidence = "scontrol Command" if by_scontrol else "batch.tsv %s" % by_tsv[0]
        ledger.submissions_append(ledger_path, ledger_row(
            event="SUBMITTED", job_id=a.job_id, submitted_at=cur["submitted_at"],
            reason=_one_line("%s; reconciled (%s): %s" % (
                plan_purpose(ledger_path, a.plan_id), evidence, a.reason)), **common))
        emit({"plan_id": a.plan_id, "job_id": a.job_id, "evidence": evidence})
        return OK
    sq = remote.run_remote(run, a.login_host, ["squeue", "-h", "--me", "-o", "%i %o"])
    if sq["rc"] != 0:
        raise SubmitError("squeue rc=%r: cannot show that no job runs this plan" % (sq["rc"],))
    listed = [ln for ln in (sq["stdout"] or "").splitlines() if snap in ln]
    if listed or mine:
        raise SubmitError("plan %s has a job: squeue %s, batch.tsv %s" % (a.plan_id, listed, mine))
    ledger.submissions_append(ledger_path, ledger_row(
        event="ENDED", status="SUBMIT_FAILED", reason=_one_line(
            "%s; reconciled: squeue lists no job for %s and no batch.tsv names the plan: %s"
            % (plan_purpose(ledger_path, a.plan_id), script, a.reason)), **common))
    emit({"plan_id": a.plan_id, "status": "SUBMIT_FAILED"})
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
        event="PLANNED", reason=PURPOSE_MEASUREMENT, arms=",".join(arms), launches=a.launches,
        basis=a.basis, gpus=0,
        exclusive="false", mounts_json=json.dumps(mounts), tokenizer_dir_sha256=tok_sha,
        assertions_json=json.dumps(assertions), **common))
    ledger.submissions_append(ledger_path, ledger_row(
        event="SUBMITTED", job_id=a.job_id, submitted_at=wall_iso(), **common))
    emit({"plan_id": plan_id, "job_id": a.job_id, "ledger": ledger_path})
    return OK


def cmd_slurm_view(a, runner):
    """Write slurm_view.json only when every query that ran succeeded (after its retries);
    otherwise write nothing and exit RUNTIME so the caller's next try can still produce it."""
    if not JOB_ID_RE.match(a.job_id) or not a.job_id.isdigit():
        raise SubmitError("--job-id must be a numeric cluster job id")
    job_dir = os.path.abspath(a.job_dir)
    slurmview.check_job_dir(job_dir)
    doc = slurmview.collect(get_runner(a, runner), a.login_host, a.job_id, a.attempts,
                            a.retry_wait_s)
    out = {"node": doc["node"], "rcs": [i["rc"] for i in doc["items"]],
           "attempts": [i["attempts"] for i in doc["items"]]}
    failed = slurmview.failed_items(doc)
    if failed:
        keys = ("id", "cmd", "rc", "stderr", "retried_rcs", "retried_stderr")
        out.update(path=None, failed=[{k: it[k] for k in keys} for it in failed])
        emit(out)
        print("capjobs.submit: slurm-view: item(s) %s still failed after %d attempt(s); "
              "slurm_view.json not written" % ([it["id"] for it in failed], a.attempts),
              file=sys.stderr)
        return RUNTIME
    out["path"] = slurmview.write(job_dir, doc)
    emit(out)
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
    if _is_unknown_submit(cur):
        raise SubmitError("plan %s is %s: run reconcile-submit first" % (a.plan_id, UNKNOWN_SUBMIT))
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
    purpose = plan_purpose(ledger_path, a.plan_id)
    ledger.submissions_append(ledger_path, ledger_row(
        event="ENDED", plan_id=a.plan_id, job_id=job_id, status=a.status,
        reason="%s%s [end criterion: %s]" % (purpose + "; " if purpose else "", a.reason,
                                             evidence), experiment=cur["experiment"]))
    emit({"plan_id": a.plan_id, "job_id": job_id, "status": a.status, "evidence": evidence,
          "purpose": purpose})
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
        body = out or ""
        if body and not body.endswith("\n"):
            body += "\n"
        marker = rc if rc is not None else "none (ssh not run or timed out)"
        text = "# cmd: %s\n%s# exit: %s\n" % (shlex.join(remote_argv), body, marker)
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


def cmd_prep_tokenizer(a, runner):
    """The tokenizer copy job (purpose=prep_tokenizer; user decision 2026-11-13). The only job
    that may mount (part of) the model library: a registered model dir, read-only; no GPU, no
    --exclusive; the job body is bash + coreutils (jobs/prep_tokenizer.sbatch). Order: checks
    (no file written; the job script must be committed at HEAD and the working tree identical)
    -> plan dir <snapshot root>/<plan id>/ with the script as committed, job.env, mounts.json and
    MANIFEST.json (code HEAD + sha256 and size of each file; all read-only) -> PLANNED row whose
    snapshot_sha256 is MANIFEST.json's -> ssh sbatch --parsable -> SUBMITTED row (unknown outcome
    and refusal as in `submit`)."""
    workdir = check_path(os.path.abspath(a.workdir), "--workdir")
    code_root = check_path(os.path.abspath(a.code_root), "--code-root")
    snap_root = check_path(os.path.abspath(a.snapshot_root), "--snapshot-root")
    model_dir = check_path(a.model_dir, "--model-dir")
    if model_dir not in policy.PREP_TOKENIZER_MODEL_DIRS:
        raise SubmitError("--model-dir %s is not a registered model dir %s"
                          % (model_dir, list(policy.PREP_TOKENIZER_MODEL_DIRS)))
    if not ARM_RE.match(a.name):
        raise SubmitError("bad tokenizer --name %r" % a.name)
    dest_parent = os.path.join(code_root, policy.PREP_TOKENIZER_DEST_REL)
    dest = os.path.join(dest_parent, a.name)
    if os.path.lexists(dest) or os.path.lexists(dest + ".partial"):
        raise SubmitError("destination %s (or .partial) already exists" % dest)
    basis_line_present(workdir, a.basis)
    image = check_path(os.path.abspath(a.image), "--image")
    if not os.path.isfile(image):
        raise SubmitError("--image %s is not a file" % image)
    st = os.stat(image)
    try:
        exclude = resources.normalize_exclude(a.exclude)
        flags = resources.prep_tokenizer_flags({
            "account": a.account, "qos": a.qos, "time": a.time, "mem": a.mem,
            "cpus_per_task": a.cpus_per_task, "job_name": a.job_name,
            "partition": a.partition, "exclude": exclude})
    except resources.ResourceError as e:
        raise SubmitError(str(e))
    plan_id = make_plan_id(policy.PREP_TOKENIZER_EXPERIMENT)
    plan_dir = os.path.join(snap_root, plan_id)
    user_mounts = [parse_mount(m) for m in a.mount]
    mounts = ["%s:%s:ro" % (model_dir, model_dir)] + [m["text"] for m in user_mounts]
    assertions = policy.check_prep_tokenizer_submission(0, mounts, a.basis)
    coverage = job_mounts(user_mounts, [dest_parent, plan_dir], [])
    head, script_body, uncommitted = committed_file(code_root, PREP_TOKENIZER_TEMPLATE)
    script = os.path.join(plan_dir, PREP_TOKENIZER_TEMPLATE)
    remote_argv = ["sbatch", "--parsable"] + flags + [
        "--output=" + os.path.join(plan_dir, "%x-%j.log"), "--open-mode=append",
        "--export=ALL", script, plan_dir]  # ALL as for `submit`: srun must be on PATH
    argv = remote.ssh_argv(a.login_host, remote_argv)
    env_values = {
        "CAPJOBS_PLAN_ID": plan_id, "CAPJOBS_RUN_ROOT": plan_dir, "CAPJOBS_IMAGE": image,
        "CAPJOBS_IMAGE_SIZE": st.st_size, "CAPJOBS_IMAGE_MTIME_NS": st.st_mtime_ns,
        "CAPJOBS_MOUNTS": ",".join(mounts), "CAPJOBS_CODE_SHA": head,
        "CAPJOBS_PREP_SCRIPT_SHA256": sha256_bytes(script_body or b""),
        "CAPJOBS_PREP_MODEL_DIR": model_dir, "CAPJOBS_PREP_DEST": dest,
        "CAPJOBS_PREP_ALLOW_GLOBS": " ".join(policy.PREP_TOKENIZER_ALLOW_GLOBS),
        "CAPJOBS_PREP_VOCAB_GLOBS": " ".join(policy.PREP_TOKENIZER_VOCAB_GLOBS),
        "CAPJOBS_PREP_WEIGHT_SUFFIXES": " ".join(policy.WEIGHT_SUFFIXES),
        "CAPJOBS_PREP_MAX_FILE_BYTES": policy.PREP_TOKENIZER_MAX_FILE_BYTES,
        "CAPJOBS_PREP_LIBRARY_ROOTS": " ".join(policy.library_roots())}
    summary = {"plan_id": plan_id, "plan_dir": plan_dir, "dest": dest, "mounts": mounts,
               "coverage": coverage, "assertions": assertions, "remote_argv": remote_argv,
               "argv": argv, "job_env": env_values, "code_head": head,
               "script_sha256": env_values["CAPJOBS_PREP_SCRIPT_SHA256"]}
    if a.dry_run:
        summary["dry_run"] = True
        summary["would_refuse_uncommitted"] = uncommitted or False
        emit(summary)
        return OK
    if uncommitted:
        raise SubmitError("the tokenizer copy job script must be committed: %s" % uncommitted)
    files = ((PREP_TOKENIZER_TEMPLATE, script_body), ("job.env", job_env_bytes(env_values)),
             ("mounts.json", (json.dumps(mounts) + "\n").encode("utf-8")))
    manifest = (json.dumps({"schema": PREP_PLAN_SCHEMA, "plan_id": plan_id, "code_sha": head,
                            "files": [{"path": rel, "sha256": sha256_bytes(data),
                                       "size": len(data)} for rel, data in files]},
                           sort_keys=True, indent=1) + "\n").encode("utf-8")
    os.makedirs(dest_parent, exist_ok=True)
    os.makedirs(os.path.join(plan_dir, "jobs"))
    for rel, data in files + ((PREP_PLAN_MANIFEST, manifest),):
        path = os.path.join(plan_dir, rel)
        with open(path, "xb") as f:
            f.write(data)
        os.chmod(path, 0o444)
    ledger_path = os.path.join(workdir, "submissions.tsv")
    common = {"plan_id": plan_id, "experiment": policy.PREP_TOKENIZER_EXPERIMENT}
    purpose = policy.PREP_TOKENIZER_PURPOSE
    ledger.submissions_append(ledger_path, ledger_row(
        event="PLANNED", reason=purpose, arms=a.name, launches=0, basis=a.basis,
        account=a.account, partition=a.partition or "", qos=a.qos,
        exclude=",".join(exclude) or resources.EXCLUDE_NONE, time_limit=a.time, mem=a.mem,
        cpus_per_task=a.cpus_per_task, gpus=0, exclusive="false", snapshot_dir=plan_dir,
        snapshot_sha256=sha256_bytes(manifest),
        sbatch_argv=json.dumps(remote_argv), mounts_json=json.dumps(mounts),
        tokenizer_dir_sha256="none", assertions_json=json.dumps(assertions), **common))
    attempted_at = wall_iso()
    rc, out, err = get_runner(a, runner).run(argv)
    job_id = parse_parsable(out) if rc == 0 else None
    if job_id is None:
        reason = _one_line("sbatch rc=%r stdout=%r stderr=%r"
                           % (rc, (out or "")[-300:], (err or "")[-300:]))
        if submit_outcome_unknown(rc):
            ledger.submissions_append(ledger_path, ledger_row(
                event="NOTE", status=UNKNOWN_SUBMIT, submitted_at=attempted_at,
                reason="%s; %s; reconcile with `reconcile-submit` after checking squeue/sacct"
                % (purpose, reason), **common))
            summary.update({"submitted": None, "unknown_submit": True, "rc": rc})
            emit(summary)
            return RUNTIME
        ledger.submissions_append(ledger_path, ledger_row(
            event="ENDED", status="SUBMIT_FAILED", reason="%s; %s" % (purpose, reason),
            **common))
        summary.update({"submitted": False, "rc": rc, "stdout": out, "stderr": err})
        emit(summary)
        return RUNTIME
    ledger.submissions_append(ledger_path, ledger_row(
        event="SUBMITTED", job_id=job_id, submitted_at=wall_iso(), **common))
    summary.update({"submitted": True, "job_id": job_id, "snapshot_sha256": sha256_bytes(manifest)})
    emit(summary)
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
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("reconcile-submit", help="close an UNKNOWN_SUBMIT plan from evidence")
    add_remote_args(p)
    p.add_argument("--workdir", required=True)
    p.add_argument("--plan-id", required=True)
    rs = p.add_mutually_exclusive_group(required=True)
    rs.add_argument("--job-id")
    rs.add_argument("--not-submitted", action="store_true")
    p.add_argument("--reason", required=True)

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

    p = sub.add_parser("slurm-view", help="write <job dir>/env/slurm_view.json (exit 4 and no "
                                          "file when a query still fails after its retries)")
    add_remote_args(p)
    p.add_argument("--job-id", required=True)
    p.add_argument("--job-dir", required=True)
    p.add_argument("--attempts", type=int, default=slurmview.DEFAULT_ATTEMPTS,
                   help="tries per query (1-%d, default %d)" % (slurmview.MAX_ATTEMPTS,
                                                                 slurmview.DEFAULT_ATTEMPTS))
    p.add_argument("--retry-wait-s", type=float, default=slurmview.DEFAULT_RETRY_WAIT_S,
                   help="wait retry-wait-s * k before try k + 1 (default %g)"
                        % slurmview.DEFAULT_RETRY_WAIT_S)

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

    p = sub.add_parser("prep-tokenizer", help="tokenizer copy job (registered model dir, ro)")
    add_remote_args(p)
    p.add_argument("--account", required=True)
    p.add_argument("--qos", required=True)
    p.add_argument("--partition", help="optional comma list; omitted = scheduler's choice")
    p.add_argument("--exclude", required=True, help="comma list of nodes, or the literal 'none'")
    p.add_argument("--time", required=True, help="[D-]HH:MM:SS")
    p.add_argument("--mem", required=True, help="e.g. 4G")
    p.add_argument("--cpus-per-task", required=True, type=int)
    p.add_argument("--job-name", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--code-root", required=True)
    p.add_argument("--snapshot-root", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--image", required=True, help="container image with bash and coreutils")
    p.add_argument("--mount", required=True, action="append",
                   help="SRC:DST[:ro|rw], as for `submit`; the registered model dir is added")
    p.add_argument("--basis", required=True)
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
        if a.cmd == "reconcile-submit":
            return cmd_reconcile_submit(a, runner)
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
            return cmd_prep_tokenizer(a, runner)
    except CapkitError as e:
        print("capjobs.submit: %s" % e, file=sys.stderr)
        return e.exit_code
    except (OSError, ValueError) as e:
        print("capjobs.submit: %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return CONFIG
    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
