# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A server module whose OpenEngineServer does NOT build its servicer from this module's
OpenEngineInferenceServicer name (it uses snl_double's): S-NL's wrap check must refuse it."""

from snl_double import OpenEngineInferenceServicer, OpenEngineServer  # noqa: F401
