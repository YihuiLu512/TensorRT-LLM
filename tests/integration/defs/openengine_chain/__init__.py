# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test harness that runs serve tests through ``trtllm-serve`` -> OpenEngine -> engine.

Import it as the top-level package ``openengine_chain`` (``runner.py`` puts a
directory holding only this package on ``PYTHONPATH``): the live-server registry
in :mod:`openengine_chain.launch` must exist once per process. Tests under
``tests/integration/defs`` import :mod:`.h2` as ``defs.openengine_chain.h2``
(it imports ``defs.trt_test_alternative``); it and the constants it reads hold
no state, and it reaches the harness through the top-level name. This module
imports nothing so that importing the package has no side effects.
"""
