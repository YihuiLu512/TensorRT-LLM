# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test double of the product's generation_pb2: the OpenEngine messages S-PB reads or writes,
with the field numbers of openengine/v1/generation.proto, built in a private descriptor pool.

make(overrides) builds a variant whose field numbers differ ({("Usage", "completion_tokens"): 7})
or whose optional output_index fields lose presence ({"no_presence": True}); the module-level
names are the faithful variant (importable as `--pb2-module oe_double`).
"""

import types

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

T_UINT32, T_BOOL, T_STRING, T_MESSAGE, T_ENUM, T_DOUBLE = 13, 8, 9, 11, 14, 1
L_OPT, L_REP = 1, 3


def _field(msg, name, number, ftype, label=L_OPT, type_name=None, optional=False,
           oneof=None):
    f = msg.field.add(name=name, number=number, type=ftype, label=label)
    if type_name:
        f.type_name = type_name
    if optional:
        f.proto3_optional = True
        msg.oneof_decl.add(name="_" + name)
        f.oneof_index = len(msg.oneof_decl) - 1
    elif oneof is not None:
        f.oneof_index = oneof
    return f


def make(overrides=None, presence=True, pkg="fakeoe"):
    ov = dict(overrides or {})

    def num(msg, field, default):
        return ov.get((msg, field), default)

    fdp = descriptor_pb2.FileDescriptorProto(name="%s_gen.proto" % pkg, package=pkg,
                                             syntax="proto3")
    en = fdp.enum_type.add(name="FinishReason")
    for i, n in enumerate(("FINISH_REASON_UNSPECIFIED", "FINISH_REASON_STOP",
                           "FINISH_REASON_LENGTH")):
        en.value.add(name=n, number=i)
    m = fdp.message_type.add(name="TokenIds")
    _field(m, "ids", 1, T_UINT32, L_REP)
    m = fdp.message_type.add(name="StoppingOptions")
    _field(m, "max_tokens", num("StoppingOptions", "max_tokens", 1), T_UINT32, optional=True)
    _field(m, "ignore_eos", 4, T_BOOL, optional=True)
    m = fdp.message_type.add(name="GenerateRequest")
    m.oneof_decl.add(name="input")
    _field(m, "request_id", num("GenerateRequest", "request_id", 1), T_STRING)
    _field(m, "model", 2, T_STRING)
    _field(m, "prompt", 3, T_STRING, oneof=0)
    _field(m, "token_ids", 4, T_MESSAGE, type_name=".%s.TokenIds" % pkg, oneof=0)
    _field(m, "stopping", num("GenerateRequest", "stopping", 6), T_MESSAGE,
           type_name=".%s.StoppingOptions" % pkg)
    m = fdp.message_type.add(name="TokenInfo")
    _field(m, "token_id", 1, T_UINT32)
    _field(m, "token", 2, T_STRING)
    _field(m, "logprob", 3, T_DOUBLE, optional=True)
    m = fdp.message_type.add(name="TokenOutput")
    _field(m, "output_index", num("TokenOutput", "output_index", 1), T_UINT32,
           optional=presence)
    _field(m, "tokens", num("TokenOutput", "tokens", 2), T_MESSAGE, L_REP,
           type_name=".%s.TokenInfo" % pkg)
    _field(m, "text", 3, T_STRING)
    for name in ("PromptOutput", "PrefillReady", "EngineError"):
        fdp.message_type.add(name=name)
    m = fdp.message_type.add(name="GenerationFinished")
    _field(m, "output_index", num("GenerationFinished", "output_index", 1), T_UINT32,
           optional=presence)
    _field(m, "reason", num("GenerationFinished", "reason", 2), T_ENUM,
           type_name=".%s.FinishReason" % pkg)
    m = fdp.message_type.add(name="Usage")
    for i, n in enumerate(("prompt_tokens", "completion_tokens", "total_tokens")):
        _field(m, n, num("Usage", n, i + 1), T_UINT32)
    m = fdp.message_type.add(name="GenerateResponse")
    m.oneof_decl.add(name="event")
    _field(m, "request_id", num("GenerateResponse", "request_id", 1), T_STRING)
    for name, tname, default in (("prompt", "PromptOutput", 2), ("token", "TokenOutput", 3),
                                 ("prefill_ready", "PrefillReady", 4),
                                 ("finished", "GenerationFinished", 5),
                                 ("error", "EngineError", 6)):
        _field(m, name, num("GenerateResponse", name, default), T_MESSAGE,
               type_name=".%s.%s" % (pkg, tname), oneof=0)
    _field(m, "usage", num("GenerateResponse", "usage", 10), T_MESSAGE,
           type_name=".%s.Usage" % pkg)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    ns = types.SimpleNamespace()
    for msg in fdp.message_type:
        setattr(ns, msg.name, message_factory.GetMessageClass(
            pool.FindMessageTypeByName("%s.%s" % (pkg, msg.name))))
    for v in en.value:
        setattr(ns, v.name, v.number)
    return ns


_faithful = make()
globals().update(vars(_faithful))
