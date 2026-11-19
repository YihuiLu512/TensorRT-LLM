# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-bytes files: the request_id prefix merges into a GenerateRequest template (checked
with a stand-in message class built from a descriptor, since the product module is
container-only), the import guard exits 3, and raw bodies reach the server unchanged (a capkit
server rejects them, which exercises the loader's failure accounting)."""

import json
import os
import subprocess
import sys
import types

import pytest

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from capgrpc.reqbytes import (ReqBytes, build_openengine, encode_pb_string_field,
                                   load_prompts)

from conftest import TOOLS, per_tier, table_doc


def fake_pb2():
    """GenerateRequest subset with the product's field numbers (generation.proto)."""
    fdp = descriptor_pb2.FileDescriptorProto(name="fake_gen.proto", package="fakeoe",
                                             syntax="proto3")
    tok = fdp.message_type.add(name="TokenIds")
    tok.field.add(name="ids", number=1, type=13, label=3)  # repeated uint32
    stop = fdp.message_type.add(name="StoppingOptions")
    stop.field.add(name="max_tokens", number=1, type=13, label=1, proto3_optional=True,
                   oneof_index=0)
    stop.field.add(name="ignore_eos", number=4, type=8, label=1, proto3_optional=True,
                   oneof_index=1)
    stop.oneof_decl.add(name="_max_tokens")
    stop.oneof_decl.add(name="_ignore_eos")
    req = fdp.message_type.add(name="GenerateRequest")
    req.field.add(name="request_id", number=1, type=9, label=1)
    req.field.add(name="model", number=2, type=9, label=1)
    req.field.add(name="prompt", number=3, type=9, label=1, oneof_index=0)
    req.field.add(name="token_ids", number=4, type=11, label=1, type_name=".fakeoe.TokenIds",
                  oneof_index=0)
    req.field.add(name="stopping", number=6, type=11, label=1,
                  type_name=".fakeoe.StoppingOptions")
    req.oneof_decl.add(name="input")
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("fakeoe.GenerateRequest"))
    return types.SimpleNamespace(GenerateRequest=cls)


def test_openengine_template_merges_request_id(tmp_path):
    from capcommon.tiers import TierTable
    from conftest import prompts_doc
    p = tmp_path / "t.json"
    p.write_text(json.dumps(table_doc([("warmup", 200, 10.0, {"n": 7}), ("drain", 200, 0.0, {}),
                                       ("up", 200, 20.0, {"n": 9, "prompt_set": "t"}),
                                       ("drain", 200, 0.0, {})])))
    tt = TierTable.load(str(p))
    pb2 = fake_pb2()
    doc = build_openengine(pb2, prompts_doc(), tt, "m0")
    rb = ReqBytes(doc)
    assert rb.id_field_number == 1 and len(rb.table) == 2 + 2  # set p x N=7, set t x N=9
    body = encode_pb_string_field(1, "123456") + rb.lookup("p", 1, 7)
    msg = pb2.GenerateRequest.FromString(body)
    assert msg.request_id == "123456" and msg.model == "m0"
    assert list(msg.token_ids.ids) == [4, 5, 6, 7]
    assert msg.stopping.max_tokens == 7 and msg.stopping.ignore_eos is True
    msg2 = pb2.GenerateRequest.FromString(encode_pb_string_field(1, "9") + rb.lookup("t", 0, 9))
    assert msg2.request_id == "9" and msg2.prompt == "hello" and msg2.stopping.max_tokens == 9
    assert rb.lookup("t", 0, 7) is None  # no template for a combination the table never uses


@pytest.mark.parametrize("body,exc", [("raise RuntimeError('boom')\n", "RuntimeError"),
                                      ("def f(:\n", "SyntaxError"),
                                      ("import os\nos.no_such_attr_capkit\n", "AttributeError")])
def test_import_pb2_maps_any_import_time_exception_to_config(tmp_path, monkeypatch, body, exc):
    """RS29: a generated module that fails while importing (not only ImportError) is a
    ConfigError (exit 3) naming the module and the exception type."""
    from capcommon.exitcodes import ConfigError  # noqa: PLC0415
    from capgrpc.reqbytes import import_pb2  # noqa: PLC0415
    name = "capkit_test_bad_pb2_%s" % exc.lower()
    (tmp_path / (name + ".py")).write_text(body)
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ConfigError) as ei:
        import_pb2(name)
    assert name in str(ei.value) and exc in str(ei.value) and ei.value.exit_code == 3
    assert name not in sys.modules


def test_import_guard_and_raw_cli(tmp_path):
    env = dict(os.environ, PYTHONPATH=TOOLS)
    tt = tmp_path / "t.json"
    tt.write_text(json.dumps(table_doc([("warmup", 200, 10.0, {}), ("drain", 200, 0.0, {})])))
    pr = tmp_path / "p.json"
    from conftest import prompts_doc
    pr.write_text(json.dumps(prompts_doc()))
    load_prompts(str(pr))
    r = subprocess.run([sys.executable, "-m", "capgrpc.reqbytes", "openengine",
                        "--pb2-module", "no_such_module_capkit_test", "--prompts", str(pr),
                        "--tier-table", str(tt), "--model", "m", "--out", str(tmp_path / "o")],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "cannot import" in r.stderr
    assert not (tmp_path / "o").exists()
    raw = tmp_path / "body.bin"
    raw.write_bytes(b"\x00not-a-capkit-request")
    out = tmp_path / "raw.json"
    r = subprocess.run([sys.executable, "-m", "capgrpc.reqbytes", "raw", "--in", str(raw),
                        "--out", str(out)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert ReqBytes(json.loads(out.read_text())).wildcard == b"\x00not-a-capkit-request"


def test_raw_body_rejected_by_capkit_sink_is_counted(launch):
    launch.write_table(table_doc([("warmup", 300, 10.0, {}), ("drain", 300, 0.0, {})]))
    raw = launch.tmp / "body.bin"
    raw.write_bytes(b"short")
    rbf = str(launch.tmp / "raw.json")
    env = dict(os.environ, PYTHONPATH=TOOLS)
    subprocess.run([sys.executable, "-m", "capgrpc.reqbytes", "raw", "--in", str(raw),
                    "--out", rbf], env=env, check=True, timeout=60)
    launch.set_origin(4.0)
    launch.server("sink", "capgrpc.nullgrpc", ["--workers", "1", "--in-process", "--msg-bytes", "8",
                                       "--pacing", "asap"])
    launch.wait_file(launch.tree.port_file("sink"), 15, "sink")
    launch.start("load", "capgrpc.load",
                 launch.common("load", cpus="3")
                 + ["--target-port-file", launch.tree.port_file("sink"),
                    "--method", "/x.Y/Z", "--workers", "1", "--in-process",
                    "--channels-per-worker", "1", "--schedule", launch.sched_path,
                    "--request-format", "raw", "--request-bytes", rbf,
                    "--msg-sample-mod", "1", "--max-records-per-worker", "100"])
    assert launch.wait_exit("load", 20) == 0, launch.stderr("load")
    assert launch.stop("sink") == 0
    ld, sk = launch.summaries("load"), launch.summaries("sink")
    planned = per_tier(ld, "counts.planned")
    assert sum(planned.values()) > 0
    assert per_tier(ld, "counts.failed") == planned and per_tier(ld, "counts.ok")[0] == 0
    fails = [f for docs in ld.values() for d in docs for f in d["failures"]]
    assert [{k: f[k] for k in ("reason", "count", "class_hint")} for f in fails] == [
        {"reason": "grpc_INVALID_ARGUMENT", "count": planned[0], "class_hint": "unclassified"}]
    assert fails[0]["zero_msgs"] == planned[0] and fails[0]["max_send_to_end_ns"] > 0
    assert sum(per_tier(sk, "counts.streams_rejected").values()) == sum(planned.values())
    reqs, heads = launch.records("load", "requests")
    assert set(reqs["status"]) == {1}
    assert heads[0]["meta"]["status_codes"] == {"1": "grpc_INVALID_ARGUMENT"}


def fake_pb2_with_extra():
    """fake_pb2's GenerateRequest plus field 12 `extra` (google.protobuf.Struct), as
    generation.proto."""
    from google.protobuf import struct_pb2  # noqa: PLC0415
    fdp = descriptor_pb2.FileDescriptorProto(name="fake_gen_x.proto", package="fakeox",
                                             syntax="proto3",
                                             dependency=["google/protobuf/struct.proto"])
    stop = fdp.message_type.add(name="StoppingOptions")
    stop.field.add(name="max_tokens", number=1, type=13, label=1)
    stop.field.add(name="ignore_eos", number=4, type=8, label=1)
    req = fdp.message_type.add(name="GenerateRequest")
    req.field.add(name="request_id", number=1, type=9, label=1)
    req.field.add(name="model", number=2, type=9, label=1)
    req.field.add(name="prompt", number=3, type=9, label=1)
    req.field.add(name="stopping", number=6, type=11, label=1,
                  type_name=".fakeox.StoppingOptions")
    req.field.add(name="extra", number=12, type=11, label=1,
                  type_name=".google.protobuf.Struct")
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(struct_pb2.DESCRIPTOR.serialized_pb)
    pool.Add(fdp)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("fakeox.GenerateRequest"))
    return types.SimpleNamespace(GenerateRequest=cls)


def _text_table(tmp_path):
    from capcommon.tiers import TierTable
    p = tmp_path / "t.json"
    p.write_text(json.dumps(table_doc([("warmup", 200, 10.0, {"n": 3, "prompt_set": "t"}),
                                       ("drain", 200, 0.0, {})])))
    return TierTable.load(str(p))


def test_openengine_extra_detokenize(tmp_path):
    from conftest import prompts_doc
    tt, pb2 = _text_table(tmp_path), fake_pb2_with_extra()
    plain = ReqBytes(build_openengine(pb2, prompts_doc(), tt, "m0"))
    nd = ReqBytes(build_openengine(pb2, prompts_doc(), tt, "m0", extra={"detokenize": False}))
    m0 = pb2.GenerateRequest.FromString(encode_pb_string_field(1, "5") + plain.lookup("t", 0, 3))
    m1 = pb2.GenerateRequest.FromString(encode_pb_string_field(1, "5") + nd.lookup("t", 0, 3))
    # default unchanged: no extra at all; with the option the only difference is extra
    assert not m0.HasField("extra")
    assert m1.HasField("extra") and dict(m1.extra) == {"detokenize": False}
    assert m1.extra.fields["detokenize"].WhichOneof("kind") == "bool_value"
    m1.ClearField("extra")
    assert m1 == m0 and plain.lookup("t", 0, 3) != nd.lookup("t", 0, 3)


def test_openengine_extra_needs_a_struct_field(tmp_path):
    from capcommon.exitcodes import ConfigError  # noqa: PLC0415
    from conftest import prompts_doc
    with pytest.raises(ConfigError):
        build_openengine(fake_pb2(), prompts_doc(), _text_table(tmp_path), "m0",
                         extra={"detokenize": False})
    # without extra the Struct-less module still works (the default path)
    assert build_openengine(fake_pb2(), prompts_doc(), _text_table(tmp_path), "m0")["entries"]
