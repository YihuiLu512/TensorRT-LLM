# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for the S-PB tests. The launch harness, tier tables and prompts are the gRPC tests'
(tests/grpc/conftest.py, loaded under a private name so the two conftest modules do not clash);
this file adds prebaked / request-bytes writers built with the pb2 test double (oe_double) and
the real generated module from <C>/probe-tree (container-free: it only needs protobuf)."""

import importlib
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.abspath(os.path.join(HERE, "..", ".."))
_spec = importlib.util.spec_from_file_location("_capstubs_grpc_conftest",
                                               os.path.join(C, "tests", "grpc", "conftest.py"))
g = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = g
_spec.loader.exec_module(g)

TOOLS, MS, SHA = g.TOOLS, g.MS, g.SHA
Launch, table_doc, tier, prompts_doc = g.Launch, g.table_doc, g.tier, g.prompts_doc
per_tier, total = g.per_tier, g.total
if HERE not in sys.path:
    sys.path.insert(0, HERE)

GENERATED = os.path.join(C, "probe-tree", "tensorrt_llm", "grpc", "openengine", "_generated")
METHOD = "/openengine.v1.Inference/Generate"


@pytest.fixture
def launch(tmp_path):
    L = Launch(tmp_path, sys.executable)
    try:
        yield L
    finally:
        L.kill_all()


_REAL = {}


def real_pb2():
    """The product's generated generation_pb2 from <C>/probe-tree, loaded as a private package
    (importing it through tensorrt_llm would need the product's native bindings). Fails, not
    skips, without the probe tree: a skipped cross-check would read as a pass."""
    if "gen" not in _REAL:
        assert os.path.isfile(os.path.join(GENERATED, "generation_pb2.py")), \
            "needs the probe-tree worktree: %s missing" % GENERATED
        name = "_capstubs_real_oe_generated"
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(GENERATED, "__init__.py"), submodule_search_locations=[GENERATED])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
        _REAL["gen"] = importlib.import_module(name + ".generation_pb2")
        _REAL["oe"] = importlib.import_module(name + ".openengine_pb2")
    return _REAL["gen"], _REAL["oe"]


def write_json(path, doc):
    with open(str(path), "w") as f:
        json.dump(doc, f)
    return str(path)


def write_prebaked(launch, pb2=None, token_ids=(11, 12, 13), prompt_tokens=4, short_tokens=(),
                   name="prebaked.json"):
    from capstubs.prebaked import build_prebaked
    import oe_double
    doc = build_prebaked(pb2 or oe_double, launch.tt, list(token_ids), prompt_tokens,
                         short_tokens)
    return write_json(launch.tmp / name, doc), doc


def write_reqbytes(launch, pb2=None, name="reqbytes.json"):
    from capgrpc.reqbytes import build_openengine
    import oe_double
    doc = build_openengine(pb2 or oe_double, g.prompts_doc(), launch.tt, "m0", METHOD)
    return write_json(launch.tmp / name, doc)


def spb_args(prebaked_path, *extra):
    return ["--workers", "1", "--prebaked", prebaked_path] + list(extra)
