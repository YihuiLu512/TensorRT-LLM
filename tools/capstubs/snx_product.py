# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX product adapter: the only place NullExecutor touches TRT-LLM objects (design 2.3 S-NX,
"响应对象用真件").

- convert: executor_request_to_llm_request(req_id, request, [], False) -- the product function
  PyExecutor's request queue uses when it activates a request (llm_request.py), called on the
  step thread;
- add_tokens: LlmRequest.add_new_token(token, beam 0) per token, state GENERATION_IN_PROGRESS,
  py_decoding_iter = the request's own iteration (what the sampler advances);
- finish: finish_by_reason(LENGTH | CANCELLED);
- create_response: the real LlmRequest.create_response() (C++ create_serialized_result), with the
  fields _handle_responses sets on the result (cached_tokens, per_pos_drafted / accepted).

Imports tensorrt_llm at construction: only inside the product's processes (worker, nx_driver).
"""


class ProductAdapter:
    name = "product"

    def __init__(self):
        from tensorrt_llm._torch.pyexecutor.llm_request import (  # noqa: PLC0415
            LlmRequestState, executor_request_to_llm_request)
        from tensorrt_llm.bindings import executor as tllm  # noqa: PLC0415
        self._to_llm = executor_request_to_llm_request
        self._generating = LlmRequestState.GENERATION_IN_PROGRESS
        self._length = tllm.FinishReason.LENGTH
        self._cancelled = tllm.FinishReason.CANCELLED

    def convert(self, req_id, executor_request):
        return self._to_llm(req_id, executor_request, [], False)

    def info(self, llm, want_prompt):
        prompt_len = llm.prompt_len
        ids = None
        if want_prompt:
            ids = list(llm.get_tokens(0)[:prompt_len])
        return llm.py_client_id, llm.max_new_tokens, prompt_len, ids

    def add_tokens(self, llm, token_ids, decoding_iter):
        for t in token_ids:
            llm.add_new_token(t, 0)
        llm.state = self._generating
        llm.py_decoding_iter = decoding_iter

    def finish(self, llm, cancelled):
        llm.finish_by_reason(self._cancelled if cancelled else self._length)

    def create_response(self, llm):
        resp = llm.create_response()
        if resp is not None:
            res = resp.result
            res.cached_tokens = llm.cached_tokens
            res.per_pos_drafted = llm.py_per_pos_drafted
            res.per_pos_accepted = llm.py_per_pos_accepted
        return resp
