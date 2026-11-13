# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capprobe.overlay on throw-away repositories (P0 / P1 / P2 and every refusal), plus the change
classification on the real image..baseline range."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
from probe_testutil import TOOLS, clean_env
from gitfix import BASE_A, PATCHED_A, TEST_SITES, commit_all, git, init, write

from capprobe import overlay
from capprobe import sites as capsites

PROBE_TREE = os.path.abspath(os.path.join(TOOLS, "..", "probe-tree"))
IMAGE_COMMIT = "9fd3822a0019d23a2a6b372a219e0e3d51a1a95d"
BASE_COMMIT = "2f22a98d6e8b32ed185bcdfe051ef0e4405ae4e8"


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(capsites, "SITES", TEST_SITES)
    prod = tmp_path / "prod"
    init(prod)
    write(prod, "tensorrt_llm/a.py", "OLD = 1\n")
    write(prod, "tensorrt_llm/b.py", "B = 1\n")
    write(prod, "tensorrt_llm/README.md", "r1\n")
    write(prod, "cpp/x.cc", "int x;\n")
    write(prod, "docs/d.md", "d1\n")
    image = commit_all(prod, "image")
    write(prod, "tensorrt_llm/a.py", BASE_A)
    write(prod, "tensorrt_llm/sub/c.py", "C = 1\n")
    write(prod, "docs/d.md", "d2\n")
    base = commit_all(prod, "base")
    git(prod, "checkout", "-q", "-b", "probe")
    write(prod, "tensorrt_llm/a.py", PATCHED_A)
    probe = commit_all(prod, "probe")
    cap = tmp_path / "capkit"
    init(cap)
    write(cap, "tools/capprobe/__init__.py", "P = 1\n")
    write(cap, "tools/capprobe/sites.py", "S = 1\n")
    write(cap, "tools/capcommon/__init__.py", "Q = 1\n")
    write(cap, "tools/capcommon/README.md", "not deployed\n")
    write(cap, "probe-tree/ignored.txt", "x\n")
    capsha = commit_all(cap, "capkit")
    return {"prod": str(prod), "image": image, "base": base, "probe": probe, "cap": str(cap),
            "capsha": capsha, "tmp": tmp_path}


def build(w, tier, groups=(), packages=("capprobe", "capcommon"), **kw):
    if tier == "P0":
        kw.setdefault("export_dir", str(w["tmp"] / "export"))
    return overlay.build_manifest(tier, w["prod"], w["image"], w["base"], w["cap"], w["capsha"],
                                  list(packages), list(groups), **kw)


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_p0_uses_base_blobs(world):
    m = build(world, "P0")
    assert m["tier"] == "P0" and m["target_commit"] == m["base_commit"] == world["base"]
    assert [f["dst"] for f in m["files"]] == ["tensorrt_llm/a.py", "tensorrt_llm/sub/c.py"]
    a = m["files"][0]
    assert os.path.isabs(a["src"])  # export dir is outside the capkit root here
    with open(a["src"]) as f:
        assert f.read() == BASE_A  # base blob, not the patched working file
    assert a["sha256"] == _sha(a["src"]) and a["size"] == len(BASE_A)
    assert m["groups"] == [] and m["sites"] == {} and m["patchcheck"] is None
    assert m["non_python_changed"] == [] and m["image_is_ancestor_of_target"] is True
    assert [c["dst"] for c in m["capkit_packages"]] == [
        "capprobe/__init__.py", "capprobe/sites.py", "capcommon/__init__.py"]
    assert m["capkit_code_sha"] == world["capsha"]


def test_p1_p2_use_probe_head(world):
    m1 = build(world, "P1")
    m2 = build(world, "P2", groups=["lag", "count"])
    assert m1["target_commit"] == m2["target_commit"] == world["probe"]
    assert m1["files"] == m2["files"]  # P1 and P2 differ only in groups
    a = [f for f in m2["files"] if f["dst"] == "tensorrt_llm/a.py"][0]
    assert a["src"] == os.path.join(world["prod"], "tensorrt_llm/a.py")
    assert a["sha256"] == hashlib.sha256(PATCHED_A.encode()).hexdigest()
    assert m1["sites"] == {} and m2["sites"] == TEST_SITES
    assert m2["groups"] == ["count", "lag"] and m2["patchcheck"]["status"] == "PASS"
    m3 = build(world, "P2", groups=["count"], sites=["F2"])
    assert m3["sites_selection"] == "cli" and list(m3["sites"]) == ["F2"]


@pytest.mark.parametrize("kw,msg", [
    (dict(tier="P1", groups=["count"]), "empty --groups"),
    (dict(tier="P2", groups=[]), "non-empty"),
    (dict(tier="P2", groups=["bytes"]), "not implemented"),
    (dict(tier="P1", packages=["capcommon"]), "needs packages"),
    (dict(tier="P1", sites=["F2"]), "only valid with tier P2"),
    (dict(tier="P2", groups=["count"], sites=["H1"]), "not sites of the open groups"),
    (dict(tier="P0", export_dir=None), "--export-dir"),
])
def test_argument_refusals(world, kw, msg):
    tier = kw.pop("tier")
    with pytest.raises(overlay.OverlayError, match=msg):
        build(world, tier, **kw)


def _amend_base(w, rel, text):
    """Make `rel` differ between image and target by changing it in a NEW base commit (the probe
    branch then merges it, so base..probe is still exactly the probe patch)."""
    git(w["prod"], "checkout", "-q", "main")
    write(w["prod"], rel, text)
    w["base"] = commit_all(w["prod"], "newer base")
    git(w["prod"], "checkout", "-q", "probe")
    git(w["prod"], "merge", "-q", "--no-edit", "main")
    w["probe"] = git(w["prod"], "rev-parse", "HEAD")
    rep = overlay._patchcheck.run_check(w["prod"], w["base"], w["probe"])
    assert rep["status"] == "PASS", rep  # positive control: only the image..target rule fires


def test_cpp_change_refused(world):
    _amend_base(world, "cpp/x.cc", "int y;\n")
    with pytest.raises(overlay.OverlayError, match="rebuild the image.*cpp/x.cc"):
        build(world, "P1")


def test_unshipped_file_needs_explicit_allow(world):
    _amend_base(world, "tensorrt_llm/README.md", "r2\n")
    with pytest.raises(overlay.OverlayError, match="README.md"):
        build(world, "P1")
    m = build(world, "P1", allow_unshipped=["tensorrt_llm/README.md"])
    assert m["unshipped_allowed"] == ["tensorrt_llm/README.md"]


def test_stale_or_outside_allow_refused(world):
    with pytest.raises(overlay.OverlayError, match="did not change"):
        build(world, "P1", allow_unshipped=["tensorrt_llm/README.md"])
    _amend_base(world, "requirements.txt", "x==1\n")
    with pytest.raises(overlay.OverlayError, match="requirements.txt"):
        build(world, "P1")


def test_deleted_python_refused(world):
    git(world["prod"], "checkout", "-q", "main")
    git(world["prod"], "rm", "-q", "tensorrt_llm/b.py")
    world["base"] = commit_all(world["prod"], "delete b in base")
    git(world["prod"], "checkout", "-q", "probe")
    git(world["prod"], "merge", "-q", "--no-edit", "main")
    with pytest.raises(overlay.OverlayError, match="cannot remove them"):
        build(world, "P1")
    with pytest.raises(overlay.OverlayError, match="cannot remove them"):
        build(world, "P0")


def test_dirty_probe_tree_refused(world):
    write(world["prod"], "tensorrt_llm/a.py", PATCHED_A + "# local edit\n")
    with pytest.raises(overlay.OverlayError, match="not clean"):
        build(world, "P1")


def test_src_bytes_must_equal_blob_even_if_git_says_clean(world):
    # assume-unchanged hides the edit from `git status`; the byte comparison still catches it
    git(world["prod"], "update-index", "--assume-unchanged", "tensorrt_llm/a.py")
    write(world["prod"], "tensorrt_llm/a.py", PATCHED_A + "# hidden edit\n")
    assert git(world["prod"], "status", "--porcelain") == ""  # positive control
    with pytest.raises(overlay.OverlayError, match="differs from the committed blob"):
        build(world, "P1")


def test_bad_patch_refused(world):
    write(world["prod"], "tensorrt_llm/a.py", PATCHED_A.replace("ON_F2", "ON_H1"))
    commit_all(world["prod"], "bad guard")
    with pytest.raises(overlay.OverlayError, match="patchcheck FAIL"):
        build(world, "P1")


def test_capkit_sha_and_cleanliness(world):
    old = world["capsha"]
    write(world["cap"], "tools/capprobe/sites.py", "S = 2\n")
    world["capsha"] = commit_all(world["cap"], "newer capkit")
    with pytest.raises(overlay.OverlayError, match="HEAD"):
        build(dict(world, capsha=old), "P1")  # stale --capkit-sha
    write(world["cap"], "tools/capprobe/sites.py", "S = 3\n")
    with pytest.raises(overlay.OverlayError, match="uncommitted"):
        build(world, "P1")


def test_cli_writes_manifest_and_exit_codes(world, tmp_path):
    out = tmp_path / "overlay_manifest.json"
    base_args = [sys.executable, "-m", "capprobe.overlay", "--probe-tree", world["prod"],
                 "--image-commit", world["image"], "--base-commit", world["base"],
                 "--capkit-root", world["cap"], "--capkit-sha", world["capsha"],
                 "--packages", "capprobe,capcommon", "--out", str(out)]
    # the CLI uses the real SITES table, which the synthetic patch does not match: P0 only
    r = subprocess.run(base_args + ["--tier", "P0", "--groups", "",
                                    "--export-dir", str(tmp_path / "exp")],
                       env=clean_env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    m = json.loads(out.read_text())
    assert m["schema"] == "capkit.overlay/1" and len(m["files"]) == 2
    r = subprocess.run(base_args + ["--tier", "P1", "--groups", ""], env=clean_env(),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 3 and "patchcheck" in r.stderr  # real SITES not found in toy patch


def test_classify_changes():
    ns = ("M\ttensorrt_llm/a.py\nA\ttensorrt_llm/x/y.py\nD\ttensorrt_llm/old.py\n"
          "M\ttensorrt_llm/grpc/README.md\nM\tcpp/k.cu\nM\trequirements-dev.txt\n"
          "M\tdocs/a.md\nA\ttests/t.py\nM\tscripts/requirements.txt\n")
    py, nonpy, dele = overlay.classify_changes(ns)
    assert py == ["tensorrt_llm/a.py", "tensorrt_llm/x/y.py"]
    assert nonpy == ["cpp/k.cu", "requirements-dev.txt", "tensorrt_llm/grpc/README.md"]
    assert dele == ["tensorrt_llm/old.py"]


@pytest.mark.skipif(not os.path.isdir(PROBE_TREE), reason="probe tree not present")
def test_real_image_to_baseline_range():
    # RESOURCES.md section 5: cpp/** unchanged and 37 tensorrt_llm/**.py between image and base.
    ns = overlay._git(PROBE_TREE, "diff", "--name-status", "--no-renames", IMAGE_COMMIT,
                      BASE_COMMIT)
    py, nonpy, dele = overlay.classify_changes(ns)
    assert len(py) == 37 and dele == []
    assert nonpy == ["tensorrt_llm/grpc/openengine/README.md"]  # needs --allow-unshipped
    assert not overlay.is_ancestor(PROBE_TREE, IMAGE_COMMIT, BASE_COMMIT)
