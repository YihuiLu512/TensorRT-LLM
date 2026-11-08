# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A post_processor_hook that records which process runs it.

``trtllm-serve --post_processor_hook _test_openengine_wt13_hooks.ProcessRecordingHook``
imports it by path in every process that postprocesses, so it is a top-level,
no-argument class in a module of its own.
"""

import multiprocessing
import os

from tensorrt_llm.executor.postprocessor_hook import (
    PostProcessorHookChunk,
    PostProcessorHookVerdict,
    emit,
)

# Each call appends "<process name> <pid> <request id>" to "<this path>.<pid>",
# one file per process, so that lines from different processes never interleave.
LOG_ENV = "TRTLLM_TEST_POSTPROC_HOOK_LOG"


class ProcessRecordingHook:
    """Leave every chunk as it is, and record the process that ran the hook on it."""

    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        path = os.environ.get(LOG_ENV)
        if path:
            name = multiprocessing.current_process().name
            with open(f"{path}.{os.getpid()}", "a") as log:
                log.write(f"{name} {os.getpid()} {chunk.request_id}\n")
        return emit(chunk.text_diff)
