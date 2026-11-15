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


# A stand-in for fake_openengine.py in S-PB's frontend mode. The names S-PB takes from the real
# module (probe-tree tests/unittest/grpc/openengine_client): _parse_args, _profile_from_args,
# FakeEngine(profile, tokens_per_step=, step_delay=, record=), _Control(engine), _Health(engine),
# openengine_pb2_grpc.add_ControlServicer_to_server and
# health_pb2_grpc.add_HealthServicer_to_server. The real module needs the product package, so this
# stand-in answers Control/GetServerInfo with what S-PB handed it, as raw bytes.
STANDIN = '''
import argparse
import dataclasses
import json
import types

import grpc


@dataclasses.dataclass
class Prof:
    fields: dict


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="trtllm-like")
    p.add_argument("--model")
    p.add_argument("--max-context-length", type=int)
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--eos-token-ids", type=int, nargs="+")
    p.add_argument("--profile-json")
    return p.parse_args(argv)


def _profile_from_args(args):
    return Prof(dict(vars(args)))


class FakeEngine:
    def __init__(self, profile, tokens_per_step=1, step_delay=0.0, record=True):
        self.profile, self.knobs = profile, [tokens_per_step, step_delay, record]


class _Control:
    def __init__(self, engine):
        self.engine = engine


class _Health:
    def __init__(self, engine):
        self.engine = engine


class _Routes(grpc.GenericRpcHandler):
    def __init__(self, routes):
        self.routes = routes

    def service(self, details):
        return self.routes.get(details.method)


def _unary(fn):
    return grpc.unary_unary_rpc_method_handler(fn, request_deserializer=None,
                                               response_serializer=None)


def _add_control(servicer, server):
    async def info(request, context):
        e = servicer.engine
        return json.dumps({"profile": e.profile.fields, "knobs": e.knobs}).encode()
    server.add_generic_rpc_handlers([_Routes({"/openengine.v1.Control/GetServerInfo":
                                              _unary(info)})])


def _add_health(servicer, server):
    async def check(request, context):
        return b"SERVING"
    server.add_generic_rpc_handlers([_Routes({"/grpc.health.v1.Health/Check": _unary(check)})])


openengine_pb2_grpc = types.SimpleNamespace(add_ControlServicer_to_server=_add_control)
health_pb2_grpc = types.SimpleNamespace(add_HealthServicer_to_server=_add_health)
'''
