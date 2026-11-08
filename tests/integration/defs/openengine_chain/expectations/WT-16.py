# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WT-16's multimodal tests that the chain answers differently."""

from openengine_chain.expectations import HTTP_STATUS, Expectation

EXPECTATIONS = [
    Expectation(
        "WT-16",
        "_test_openai_chat_multimodal.py::test_single_chat_session_image_embeds*",
        HTTP_STATUS,
        "precomputed image embeddings have no carrier in a Generate request (OE-MED-5)",
        status=400,
    ),
]
