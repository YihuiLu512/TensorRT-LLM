# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Client double for capgrpc.oe_sink: a `tensorrt_llm` package that holds only the files the
OpenEngine client path imports, taken unchanged from <C>/probe-tree (the product's own
openengine_client, bindings with its generated modules and manifest, openengine_conventions, and
openengine_poc's engine_client / engine_client_impl), plus two stand-ins: empty package
__init__ files (the real ones import the whole library) and a `tensorrt_llm.logger` that is a
stdlib logger. The probe tree's tests/unittest/grpc/openengine_client/fake_openengine.py (the
real S-FE) imports from the same package, so S-PB's frontend mode can run it here.

build(dest, copy=False) writes the package under dest and returns dest (put it on PYTHONPATH).
Fails, not skips, without the probe tree: a skipped product check would read as a pass."""

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.abspath(os.path.join(HERE, "..", "..", "probe-tree"))
SFE = os.path.join(PROBE, "tests", "unittest", "grpc", "openengine_client", "fake_openengine.py")
# (path under dest, path under the probe tree) of every real file or directory.
REAL = (("tensorrt_llm/grpc/openengine/bindings.py", "tensorrt_llm/grpc/openengine/bindings.py"),
        ("tensorrt_llm/grpc/openengine/_generated", "tensorrt_llm/grpc/openengine/_generated"),
        ("tensorrt_llm/grpc/openengine/proto", "tensorrt_llm/grpc/openengine/proto"),
        ("tensorrt_llm/grpc/openengine_conventions.py",
         "tensorrt_llm/grpc/openengine_conventions.py"),
        ("tensorrt_llm/grpc/openengine_client", "tensorrt_llm/grpc/openengine_client"),
        ("tensorrt_llm/serve/openengine_poc/engine_client.py",
         "tensorrt_llm/serve/openengine_poc/engine_client.py"),
        ("tensorrt_llm/serve/openengine_poc/engine_client_impl.py",
         "tensorrt_llm/serve/openengine_poc/engine_client_impl.py"))
EMPTY_INITS = ("tensorrt_llm", "tensorrt_llm/grpc", "tensorrt_llm/grpc/openengine",
               "tensorrt_llm/serve", "tensorrt_llm/serve/openengine_poc")
LOGGER = 'import logging\n\nlogger = logging.getLogger("tensorrt_llm")\n'


def build(dest, copy=False):
    assert os.path.isfile(os.path.join(PROBE, REAL[0][1])), \
        "needs the probe-tree worktree at %s" % PROBE
    for d in EMPTY_INITS:
        os.makedirs(os.path.join(dest, d), exist_ok=True)
        open(os.path.join(dest, d, "__init__.py"), "w").close()
    with open(os.path.join(dest, "tensorrt_llm", "logger.py"), "w") as f:
        f.write(LOGGER)
    for rel, src in REAL:
        s, t = os.path.join(PROBE, src), os.path.join(dest, rel)
        if not copy:
            os.symlink(s, t)
        elif os.path.isdir(s):
            shutil.copytree(s, t, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copyfile(s, t)
    return dest
