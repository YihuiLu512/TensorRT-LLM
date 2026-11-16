# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os

import pytest
from conftest import DATA, SAVED_CDB, external_input

from capcommon import policy
from capcommon.exitcodes import ConfigError
from capjobs import cdbexclude, jobstate, launchplan, remote, resources, slurmview

PARTS = ["b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb",
         "b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb"]


def res(**kw):
    d = {"account": "acct", "partition": ",".join(PARTS), "qos": "batch", "exclude": [],
         "time": "01:00:00", "mem": "1900G", "cpus_per_task": 224, "gpus": 1, "gpu_type": None,
         "job_name": "capkit-E0"}
    d.update(kw)
    return d


# ---- resources ---------------------------------------------------------------------------------

def test_time_parse():
    assert resources.parse_time_s("01:00:00") == 3600
    assert resources.parse_time_s("1-00:00:01") == 86401
    for bad in ("60", "1:60:00", "00:00:00", "", "1h"):
        with pytest.raises(ConfigError):
            resources.parse_time_s(bad)


def test_flags_and_argv():
    flags = resources.resource_flags(res())
    assert "--exclusive" in flags and "--nodes=1" in flags and "--ntasks=1" in flags
    assert "--gres=gpu:1" in flags and not any(f.startswith("--exclude") for f in flags)
    assert "--gres=gpu:b200:1" in resources.resource_flags(res(gpu_type="b200"))
    assert not any(f.startswith("--gres") for f in resources.resource_flags(res(gpus=0)))
    assert "--exclude=n1,n2" in resources.resource_flags(res(exclude=["n1", "n2"]))
    sub = resources.submit_argv(res(), "/x/%x-%j.log", "/s/jobs/capkit_job.sbatch", ["/s"])
    assert sub[:2] == ["sbatch", "--parsable"] and "--test-only" not in sub
    probe = resources.probe_argv(res())
    policy.check_test_only_argv(probe)
    with pytest.raises(ConfigError):
        policy.check_test_only_argv(sub)


@pytest.mark.parametrize("kw", [{"gpus": 2}, {"gpus": -1}, {"gpus": True}, {"account": ""},
                                {"account": "a b"}, {"qos": "q;rm"}, {"mem": "lots"},
                                {"cpus_per_task": 0}, {"partition": "a,,b"},
                                {"partition": "p,p"}, {"exclude": ["bad node"]},
                                {"gpu_type": "B200"}, {"gpu_type": "b200", "gpus": 0},
                                {"job_name": "-x"}, {"time": "99"}])
def test_resource_rejects(kw):
    with pytest.raises(ConfigError):
        resources.resource_flags(res(**kw))


def test_exclude_literal():
    assert resources.normalize_exclude("none") == []
    assert resources.normalize_exclude("a,b") == ["a", "b"]
    for bad in ("", "a,,b", "a b", "a,a"):
        with pytest.raises(ConfigError):
            resources.normalize_exclude(bad)


# ---- remote ------------------------------------------------------------------------------------

def test_remote_allowlist():
    a = remote.ssh_argv("front-end", ["squeue", "-h", "-j", "1"])
    assert a[0] == "ssh" and a[-1] == "squeue -h -j 1" and "BatchMode=yes" in a
    for bad in (["scancel", "1"], ["scontrol", "update", "x"], ["cdb", "set"], ["bash", "-c"],
                ["rm", "-rf", "/"], []):
        with pytest.raises(ConfigError):
            remote.ssh_argv("front-end", bad)
    with pytest.raises(ConfigError):
        remote.ssh_argv("bad host;", ["squeue"])
    quoted = remote.ssh_argv("h", ["sbatch", "--partition=a@b+c/d", "x y"])[-1]
    assert quoted == "sbatch --partition=a@b+c/d 'x y'"


# ---- cdb exclude --------------------------------------------------------------------------------

def spec():
    with open(os.path.join(DATA, "g0_main_profile.json")) as f:
        return json.load(f)


def saved():
    with open(external_input(SAVED_CDB)) as f:
        return f.read()


def test_exclude_on_saved_csv():
    r = cdbexclude.compute_exclude(saved(), spec(), PARTS)
    # Expected numbers come from the independent grouping 00_plan/cluster-probe/20261113/
    # 05_dgxb200_groups.md: G0 has 29 nodes in these two partitions; G1 (6), G2 (3), G3 (2 of 3,
    # umbriel-b200-078 is in another partition) and G7 (1) differ.
    assert r["nodes_in_partitions"] == 41
    assert len(r["nodes_matching"]) == 29
    assert r["exclude"] == sorted([
        "umb-b200-239", "umb-b200-248", "umbriel-b200-037", "umbriel-b200-072",
        "umbriel-b200-079", "umbriel-b200-081", "umbriel-b200-022", "umbriel-b200-048",
        "umbriel-b200-049", "umbriel-b200-027", "umbriel-b200-034", "umbriel-b200-091"])
    assert "umbriel-b200-020" not in r["exclude"]  # other partition: not ours to exclude
    assert r["differences"]["umbriel-b200-091"] == {
        "system.os_kernel_version": {"want": "6.8.0-139-generic", "got": "6.8.0-138-generic"}}


def test_exclude_single_partition():
    r = cdbexclude.compute_exclude(saved(), spec(), PARTS[1:])
    assert r["nodes_in_partitions"] == 14 and r["exclude"] == ["umb-b200-239", "umb-b200-248"]


def test_exclude_failures():
    text = saved()
    with pytest.raises(ConfigError, match="NOT_RUN"):
        cdbexclude.compute_exclude(text, spec(), ["no-such-partition"])
    with pytest.raises(ConfigError, match="no node"):
        cdbexclude.compute_exclude(text, spec(), PARTS + ["no-such-partition"])
    s = spec()
    s["fields"]["system.os_kernel_version"] = "9.9"
    with pytest.raises(ConfigError, match="matches main profile"):
        cdbexclude.compute_exclude(text, s, PARTS)
    s = spec()
    s["fields"]["nonexistent.field"] = "x"
    with pytest.raises(ConfigError, match="not in cdb output"):
        cdbexclude.compute_exclude(text, s, PARTS)
    with pytest.raises(ConfigError, match="exit marker"):
        cdbexclude.compute_exclude(text.replace("# exit: 0", "# exit: 1"), spec(), PARTS)
    with pytest.raises(ConfigError):
        cdbexclude.compute_exclude(text, {"schema": "x", "name": "G0", "fields": {}}, PARTS)
    with pytest.raises(ConfigError):
        cdbexclude.compute_exclude("# exit: 0\n", spec(), PARTS)
    lines = text.splitlines()
    dup = "\n".join(lines + [lines[-2]]) + "\n"
    with pytest.raises(ConfigError, match="duplicate"):
        cdbexclude.compute_exclude(dup, spec(), PARTS)


def test_exclude_multi_partition_row():
    hdr = "system.hostname,system.product_name,slurm.partitions"
    text = "\n".join([hdr, "n1,DGX,['p1', 'p2']", "n2,OTHER,['p2']", "n3,DGX,['p3']"])
    s = {"schema": cdbexclude.SPEC_SCHEMA, "name": "x", "fields": {"system.product_name": "DGX"}}
    r = cdbexclude.compute_exclude(text, s, ["p2"])
    assert r["exclude"] == ["n2"] and r["nodes_matching"] == ["n1"]
    with pytest.raises(ConfigError, match="list literal"):
        cdbexclude.compute_exclude(hdr + "\nn1,DGX,p1\n", s, ["p1"])


# ---- jobstate ----------------------------------------------------------------------------------

@pytest.mark.parametrize("sq_rc,sq_out,sa_rc,sa_out,want", [
    (0, "", 0, "123|COMPLETED|0:0\n", jobstate.ENDED),
    (0, "", 0, "123|CANCELLED by 42|0:0\n", jobstate.ENDED),
    (0, "", 0, "123|TIMEOUT|0:0\n", jobstate.ENDED),
    (0, "123 RUNNING\n", 0, "123|RUNNING|0:0\n", jobstate.NOT_ENDED),
    (0, "", 0, "123|RUNNING|0:0\n", jobstate.NOT_ENDED),
    (0, "", 0, "123|REQUEUED|0:0\n", jobstate.NOT_ENDED),
    (1, "", 0, "123|COMPLETED|0:0\n", jobstate.UNKNOWN),
    (None, "", 0, "123|COMPLETED|0:0\n", jobstate.UNKNOWN),
    (0, "", 1, "", jobstate.UNKNOWN),
    (0, "", 0, "", jobstate.UNKNOWN),
    (0, "", 0, "124|COMPLETED|0:0\n", jobstate.UNKNOWN),
    (0, "", 0, "123|COMPLETED|0:0\n123|COMPLETED|0:0\n", jobstate.UNKNOWN),
    (0, "", 0, "123|WEIRD|0:0\n", jobstate.UNKNOWN),
])
def test_jobstate_decide(sq_rc, sq_out, sa_rc, sa_out, want):
    assert jobstate.decide("123", sq_rc, sq_out, sa_rc, sa_out)[0] == want


PURGED = "slurm_load_jobs error: Invalid job id specified\n"


@pytest.mark.parametrize("sq_err,sa_rc,sa_out,want", [
    # K17: a job purged from slurmctld (after MinJobAge) is decided by sacct alone
    (PURGED, 0, "123|COMPLETED|0:0\n", jobstate.ENDED),
    (PURGED, 0, "123|RUNNING|0:0\n", jobstate.NOT_ENDED),
    (PURGED, 1, "", jobstate.UNKNOWN),
    (PURGED, 0, "", jobstate.UNKNOWN),
    ("slurm_load_jobs error: Unable to contact slurm controller\n", 0, "123|COMPLETED|0:0\n",
     jobstate.UNKNOWN),
])
def test_jobstate_purged_job_falls_back_to_sacct(sq_err, sa_rc, sa_out, want):
    st, reason, _ = jobstate.decide("123", 1, "", sa_rc, sa_out, sq_err)
    assert st == want, reason
    if want == jobstate.ENDED:
        assert "purged" in reason


def test_jobstate_query(stub_runner):
    r = stub_runner({"squeue": (0, "", ""), "sacct": (0, "77|COMPLETED|0:0\n", "")})
    q = jobstate.query(r, "fe", "77")
    assert q["state"] == jobstate.ENDED and len(r.calls) == 2
    r = stub_runner({"squeue": (1, "", PURGED), "sacct": (0, "77|COMPLETED|0:0\n", "")})
    assert jobstate.query(r, "fe", "77")["state"] == jobstate.ENDED
    with pytest.raises(ConfigError):
        jobstate.query(r, "fe", "local-1")


# ---- slurm view --------------------------------------------------------------------------------

def test_slurm_view(tmp_path, stub_runner):
    job = "JobId=77 JobName=x\n   NodeList=umbriel-b200-021 BatchHost=umbriel-b200-021\n"
    r = stub_runner({"scontrol": lambda argv: (0, job if " job " in argv[-1] else "NodeName=n",
                                               ""),
                     "squeue": (0, "77 me RUNNING\n", "")})
    doc = slurmview.collect(r, "fe", "77", 1, 0)
    assert doc["node"] == "umbriel-b200-021" and [i["rc"] for i in doc["items"]] == [0, 0, 0]
    assert r.calls[2][-1] == "squeue -h -w umbriel-b200-021 -o '%i %u %T'"
    with pytest.raises(FileNotFoundError):
        slurmview.write(str(tmp_path / "missing"), doc)
    (tmp_path / "J77-r0").mkdir()
    p = slurmview.write(str(tmp_path / "J77-r0"), doc)
    assert json.load(open(p))["schema"] == "capkit.slurm_view/1"
    r2 = stub_runner({"scontrol": (0, "JobId=77 NodeList=(null)\n", "")})
    doc2 = slurmview.collect(r2, "fe", "77", 1, 0)
    assert doc2["node"] is None and [i["rc"] for i in doc2["items"]] == [0, None, None]
    assert slurmview.node_of("NodeList=a\nNodeList=b") is None
    assert slurmview.node_of("ReqNodeList=(null) NodeList=n1") == "n1"


# ---- launch plan -------------------------------------------------------------------------------

def plan_doc(**kw):
    d = {"schema": launchplan.SCHEMA, "experiment": "E0", "register_sha": "a" * 64,
         "job_arm": "job", "margin_s": 10, "kill_grace_s": 30,
         "launches": [{"launch_seq": 0, "arm": "nullsse-a", "budget_s": 100,
                       "argv": ["{PY}", "-m", "caplaunch", "--x", "{SNAP}/c.json"]},
                      {"launch_seq": 1, "arm": "sfe-a", "budget_s": 100,
                       "argv": ["{PY}", "-m", "caplaunch"]}]}
    d.update(kw)
    return d


def test_plan_ok_and_arms():
    p = launchplan.validate(plan_doc())
    assert launchplan.arms(p) == ["nullsse-a", "sfe-a"]
    assert launchplan.substitute("{SNAP}/c.json", {"SNAP": "/s"}) == "/s/c.json"
    with pytest.raises(ConfigError):
        launchplan.substitute("{JOB_DIR}", {"SNAP": "/s"})


def _grace_plan(tmp_path, kill_grace_s, argv_tail=("--scenario", "c/s.json",
                                                   "--kill-after-s", "2"), scen=None):
    w = tmp_path / "W"
    (w / "c").mkdir(parents=True)
    (w / "c" / "s.json").write_text(json.dumps(
        scen if scen is not None else {"stop_grace_s": 10, "stop_order": ["a", "b", "c", "d"]}))
    d = plan_doc(kill_grace_s=kill_grace_s)
    for ln in d["launches"]:
        ln["argv"] = ["{PY}", "-m", "caplaunch", "--workdir", str(w)] + [
            x.replace("<W>", str(w)) for x in argv_tail]
    return launchplan.validate(d), str(w)


def test_kill_grace_covers_the_orchestrator_worst_stop(tmp_path):
    """4 roles x (10 + 2) + 2 x 2 = 52 s: kill_grace_s 52 passes, 51 is refused."""
    plan, w = _grace_plan(tmp_path, 52)
    assert launchplan.check_kill_grace(plan) == {0: 52.0, 1: 52.0}
    assert launchplan.scenario_path(plan["launches"][0]) == (w, "c/s.json")
    plan51, _w = _grace_plan(tmp_path / "x", 51)
    with pytest.raises(ConfigError, match="worst-case orchestrator stop 52.0"):
        launchplan.check_kill_grace(plan51)


def test_scenario_is_the_registered_file_in_the_work_dir(tmp_path):
    """SM-1: --scenario is relative to --workdir or an absolute path inside it, never a {SNAP}/
    path (caplaunch compares its relpath under --workdir with the register)."""
    plan, w = _grace_plan(tmp_path, 52, ("--scenario", "<W>/c/s.json", "--kill-after-s", "2"))
    assert launchplan.scenario_path(plan["launches"][0]) == (w, "c/s.json")
    assert launchplan.check_kill_grace(plan) == {0: 52.0, 1: 52.0}


@pytest.mark.parametrize("tail,scen", [
    (("--scenario", "c/s.json"), None),  # no --kill-after-s
    (("--kill-after-s", "2"), None),  # no --scenario
    (("--scenario", "/abs/s.json", "--kill-after-s", "2"), None),  # outside --workdir
    (("--scenario", "../s.json", "--kill-after-s", "2"), None),
    (("--scenario", "{SNAP}/c/s.json", "--kill-after-s", "2"), None),
    (("--scenario", "<W>/{JOB_DIR}/s.json", "--kill-after-s", "2"), None),
    (("--scenario=c/s.json", "--kill-after-s", "0"), None),
    (("--scenario", "c/s.json", "--kill-after-s", "2", "--kill-after-s", "1"), None),
    (("--scenario", "c/missing.json", "--kill-after-s", "2"), None),
    (("--scenario", "c/s.json", "--kill-after-s", "2"), {"stop_grace_s": 1}),
    (("--scenario", "c/s.json", "--kill-after-s", "2"),
     {"stop_grace_s": -1, "stop_order": ["a"]}),
    (("--scenario", "c/s.json", "--kill-after-s", "2", "--workdir", "/x"), None),
])
def test_kill_grace_needs_a_computable_bound(tmp_path, tail, scen):
    plan, _w = _grace_plan(tmp_path, 3600, tail, scen)
    with pytest.raises(ConfigError):
        launchplan.check_kill_grace(plan)


def test_workdir_must_be_an_absolute_literal_path(tmp_path):
    for wd in ("rel/W", "{SNAP}"):
        d = plan_doc()
        for ln in d["launches"]:
            ln["argv"] = ["{PY}", "-m", "caplaunch", "--workdir", wd, "--scenario", "s.json",
                          "--kill-after-s", "2"]
        with pytest.raises(ConfigError, match="--workdir"):
            launchplan.check_kill_grace(launchplan.validate(d))


@pytest.mark.parametrize("mut", [
    lambda d: d.update(schema="x"),
    lambda d: d.update(register_sha="A" * 64),
    lambda d: d.update(margin_s=-1),
    lambda d: d.pop("kill_grace_s"),
    lambda d: d.update(kill_grace_s=0),
    lambda d: d["launches"][0]["argv"].extend(["--overlay-manifest", "/x"]),
    lambda d: d["launches"][0]["argv"].extend(["--nvidia-smi", "/x"]),
    lambda d: d["launches"][0]["argv"].append("--job-env-dir=/x"),
    lambda d: d.update(launches=[]),
    lambda d: d.update(extra=1),
    lambda d: d["launches"][1].update(launch_seq=0),
    lambda d: d["launches"][0]["argv"].append("--arm=x"),
    lambda d: d["launches"][0]["argv"].extend(["--run-dir", "/x"]),
    lambda d: d["launches"][0]["argv"].append("{HOME}"),
    lambda d: d["launches"][0].update(budget_s=0),
    lambda d: d["launches"][0].update(arm="bad arm"),
])
def test_plan_rejects(mut):
    d = plan_doc()
    mut(d)
    with pytest.raises(ConfigError):
        launchplan.validate(d)


# ---- policy: mounts are compared by literal path and by realpath (K9) -------------------------

def test_check_mounts_resolves_symlinks(tmp_path):
    link = tmp_path / "models"
    os.symlink("/home/scratch.trt_llm_data_ci/llm-models", str(link))
    with pytest.raises(policy.PolicyError) as e:
        policy.check_mounts(["%s:/data:ro" % link])
    assert "model library" in str(e.value)
    up = tmp_path / "up"
    os.symlink("/home", str(up))  # an ancestor of both libraries, reached through a link
    with pytest.raises(policy.PolicyError):
        policy.check_mounts([{"src": str(up), "dst": "/h"}])
    # positive control: the same call accepts a plain directory
    policy.check_mounts(["%s:/data:ro" % tmp_path])


def _caplaunch_full_argv():
    from caplaunch.__main__ import build_parser  # noqa: PLC0415
    p = build_parser()
    argv = []
    for act in p._actions:
        if act.required:
            argv += [act.option_strings[0], "1"]
    return p, argv


def test_caplaunch_refuses_abbreviated_flags():
    """U7: capjobs checks the plan argv by exact flag spellings (RESERVED_FLAGS, flag_value), so
    caplaunch must not accept an abbreviation of any of its long options (`--scen x`,
    `--scen=x`): it would reach the same option while bypassing those checks."""
    p, argv = _caplaunch_full_argv()
    p.parse_args(argv)  # positive control: the full spellings parse
    longs = [o for act in p._actions for o in act.option_strings if o.startswith("--")
             and o != "--help"]
    inspected = set(launchplan.RESERVED_FLAGS) | {"--workdir", "--scenario", "--kill-after-s",
                                                  "--capkit-sha"}
    assert inspected <= set(longs)
    for opt in longs:
        for abbrev in (opt[:-1], opt[:-1] + "=1"):
            with pytest.raises(SystemExit) as e:
                p.parse_args(argv + [abbrev] + ([] if "=" in abbrev else ["1"]))
            assert e.value.code == 2, (opt, abbrev)
