# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry for tests that start ``trtllm-serve`` themselves.

Stateless, so tests may import it as ``defs.openengine_chain.h2``. With
``TRTLLM_TEST_OPENENGINE_CHAIN`` unset it is ``popen``; otherwise it starts the
chain through the harness, imported under its one canonical name.
"""

import os

from ..trt_test_alternative import popen
from .conventions import ENV_CHAIN


def serve_popen(cmd, **kwargs):
    """``popen(cmd, **kwargs)`` for a ``trtllm-serve`` command line, through the chain when selected.

    The chain takes only ``stdout`` (a file or ``None``), ``stderr`` (``None``,
    the same file or ``STDOUT``) and ``env``.
    """
    if not os.environ.get(ENV_CHAIN):
        return popen(cmd, **kwargs)
    from openengine_chain.launch import chain_popen

    return chain_popen(cmd, **kwargs)
