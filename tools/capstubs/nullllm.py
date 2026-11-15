# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NullLlm: the parts of tensorrt_llm.LLM the product OpenEngine servicer reads (design 2.3
"S-NL", D:202), with no executor, worker or GPU behind them. S-NL (capstubs.null_llm_server) runs
the product's OpenEngineServer on top of it.

- generate_async(inputs, sampling_params, streaming=True, ...) returns a NullHandle: an async
  iterator whose __anext__ takes exactly ONE queued step (D:198; the product's GenerationResult
  handles one response per iteration and the servicer cuts the delta from the cumulative
  output.token_ids itself). N = sampling_params.max_tokens; the tokens come from a fixed list
  cycled over the stream, or from the request's prompt token ids cycled (explicit choice).
- The steps are put into each handle's unbounded queue by a "source" (the S-NL metronome, or the
  as-soon-as-possible source of minimal mode) that never waits for the servicer: the queue sits
  upstream of the servicer's yield.
- No request list is kept (the test helper FakeLlm keeps every handle, which grows without bound).

Imports nothing from tensorrt_llm: the tokenizer, when there is one, is handed in by the caller.
"""

import asyncio
import collections
import contextvars

# Set by S-NL's Generate wrapper for the duration of one RPC; generate_async attaches its handle
# to it so the wrapper can read the stream's first step and token count (one task per RPC).
CURRENT_RPC = contextvars.ContextVar("capstubs_null_llm_rpc", default=None)

DEFAULT_LLM_ID = "capkit-null-llm"
TOKEN_SOURCES = ("fixed", "prompt")


class NullLlmError(ValueError):
    """A request NullLlm cannot serve (the servicer maps ValueError to INVALID_ARGUMENT)."""


class NullKvCacheConfig:
    def __init__(self, tokens_per_block):
        self.tokens_per_block = tokens_per_block
        self.kv_events_config = None
        self.use_kv_cache_manager_v2 = False


class NullArgs:
    """The LLM args fields the servicer, Control and capability helpers read. Fields that are
    read with getattr(..., default) elsewhere and left out here take the product's default;
    a field read directly and missing raises AttributeError (a loud failure, not a guess)."""

    def __init__(self, max_seq_len, max_batch_size, max_num_tokens, tokens_per_block=32):
        self.backend = "pytorch"
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.max_beam_width = 1
        self.tensor_parallel_size = 1
        self.pipeline_parallel_size = 1
        self.context_parallel_size = 1
        self.enable_attention_dp = False
        self.kv_cache_config = NullKvCacheConfig(tokens_per_block)
        self.guided_decoding_backend = None
        self.reasoning_parser = None
        self.return_perf_metrics = False
        self.enable_iter_perf_stats = False
        self.lora_config = None
        self.speculative_config = None
        self.scheduler_config = None
        self.cache_transceiver_config = None
        self.disable_mm_encoder = False
        self.num_serve_frontends = 1

    def to_dict(self):
        d = dict(vars(self))
        d["kv_cache_config"] = dict(vars(self.kv_cache_config))
        return d


class NullInputProcessor:
    """Token-id input accepted; no media, no multimodal token expansion."""
    supports_token_ids_input = True
    supports_token_id_mm_expansion = False
    config = None


class NullExecutor:
    """What the server's health monitor and Control look at on llm._executor."""
    _fatal_error = None
    _engine_dead = False

    def __init__(self):
        self.shut = False

    def is_shutdown(self):
        return self.shut


class NullOutput:
    """One CompletionOutput (index 0) as _format_result reads it; token_ids is cumulative."""
    __slots__ = ("index", "token_ids", "text", "logprobs", "prompt_logprobs", "finish_reason",
                 "stop_reason", "disaggregated_params")

    def __init__(self):
        self.index = 0
        self.token_ids = []
        self.text = ""
        self.logprobs = []
        self.prompt_logprobs = []
        self.finish_reason = None
        self.stop_reason = None
        self.disaggregated_params = None


class NullHandle:
    """The result handle of one request: the request's queue of steps and, like the product's
    GenerationResult, the result object __anext__ returns (itself, updated by one step).

    Set by the source in open(): n (tokens to produce, after any short injection), toks (tokens
    of each queued step, in order), first (the stream's first beat step, -1 without a beat).
    Ends: finished with finish_reason "length" after the last step; "cancelled" when aborted;
    error set (the servicer turns it into an error event) when the source truncated the stream
    (the beat schedule ended before its last step)."""

    __slots__ = ("src", "q", "ev", "n", "toks", "first", "produced", "taken", "cycle", "pos",
                 "prompt_token_ids", "outputs", "finished", "error", "cached_tokens", "aborted",
                 "truncated", "closed", "outcome", "offs", "shorted_by")

    def __init__(self, src, prompt_token_ids, cycle):
        self.src = src
        self.q = collections.deque()
        self.ev = asyncio.Event()
        self.n = 0
        self.toks = ()
        self.offs = ()
        self.first = -1
        self.produced = 0
        self.taken = 0
        self.cycle = cycle
        self.pos = 0
        self.prompt_token_ids = prompt_token_ids
        self.outputs = [NullOutput()]
        self.finished = False
        self.error = None
        self.cached_tokens = 0
        self.aborted = False
        self.truncated = False
        self.closed = False
        self.outcome = None
        self.shorted_by = 0

    # -- source side -------------------------------------------------------------------------
    def push(self, j):
        """Queue step j (an index into toks). Called by the source only."""
        self.q.append(j)
        self.ev.set()

    def truncate(self):
        self.truncated = True
        self.ev.set()

    # -- servicer side ------------------------------------------------------------------------
    def abort(self):
        if not self.closed:
            self.aborted = True
            self.src.close(self, "cancelled")
        self.ev.set()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.finished or self.error is not None:
            raise StopAsyncIteration
        q = self.q
        while True:
            if self.aborted:
                return self._finish("cancelled")
            if q:
                break
            if self.truncated:
                self.error = "S-NL: the beat schedule ended before this stream's last step"
                self.src.close(self, "truncated")
                return self
            self.ev.clear()
            await self.ev.wait()
        j = q.popleft()
        self.src.took(self)
        t = self.toks[j]
        out = self.outputs[0]
        ids, cyc, n = out.token_ids, self.cycle, len(self.cycle)
        p = self.pos
        for _ in range(t):
            ids.append(cyc[p % n])
            p += 1
        self.pos = p
        self.produced += t
        self.taken += 1
        if self.produced >= self.n:
            return self._finish("length")
        return self

    def _finish(self, reason):
        self.outputs[0].finish_reason = reason
        self.finished = True
        if not self.closed:
            self.src.close(self, "completed" if reason == "length" else "cancelled")
        return self


class NullLlm:
    """The LLM stand-in. `source` provides open(handle, max_tokens), took(handle) and
    close(handle, outcome) (see capstubs.null_llm_server); `tokenizer` is the product tokenizer
    (or None: text prompts then carry no prompt token ids and the server advertises no text
    input); `hf_model_dir` is the directory the tokenizer was loaded from (the Control servicer
    fingerprints its files)."""

    def __init__(self, source, args, *, token_source="fixed", token_ids=(11,), tokenizer=None,
                 hf_model_dir=None, llm_id=DEFAULT_LLM_ID):
        if token_source not in TOKEN_SOURCES:
            raise ValueError("token_source must be one of %s" % (TOKEN_SOURCES,))
        if token_source == "fixed" and not token_ids:
            raise ValueError("token_source fixed needs a non-empty token_ids")
        self._source = source
        self.args = args
        self.token_source = token_source
        self.token_ids = tuple(token_ids)
        self.tokenizer = tokenizer
        self._hf_model_dir = hf_model_dir
        self._hf_model_config = None
        self.input_processor = NullInputProcessor()
        self._executor = NullExecutor()
        self.llm_id = llm_id

    def _prompt_ids(self, inputs):
        if isinstance(inputs, dict):
            ids = inputs.get("prompt_token_ids")
            if ids is not None:
                return list(ids)
            text = inputs.get("prompt")
        else:
            text = inputs
        if not isinstance(text, str):
            raise NullLlmError("NullLlm takes a text prompt or prompt_token_ids, got %s"
                               % type(inputs).__name__)
        if self.tokenizer is None:
            return []
        return list(self.tokenizer.encode(text))

    def generate_async(self, inputs, sampling_params=None, streaming=False, **kwargs):
        if not streaming:
            raise NullLlmError("NullLlm serves streaming requests only")
        if kwargs.get("disaggregated_params") is not None:
            raise NullLlmError("NullLlm serves aggregated requests only")
        n = getattr(sampling_params, "max_tokens", None)
        if type(n) is not int or n < 1:
            raise NullLlmError("NullLlm needs sampling_params.max_tokens >= 1, got %r" % (n,))
        if getattr(sampling_params, "n", 1) not in (1, None):
            raise NullLlmError("NullLlm produces one output sequence")
        prompt = self._prompt_ids(inputs)
        if self.token_source == "prompt":
            if not prompt:
                raise NullLlmError("token source 'prompt' needs a non-empty prompt")
            cycle = prompt
        else:
            cycle = self.token_ids
        h = NullHandle(self._source, prompt, cycle)
        self._source.open(h, n)
        rec = CURRENT_RPC.get()
        if rec is not None:
            rec.handle = h
        return h

    def preprocess(self, *args, **kwargs):
        raise NullLlmError("NullLlm has no input processor for media requests")

    def _check_health(self):
        return not self._executor.shut

    def get_stats(self, timeout=None):
        return []

    def shutdown(self):
        self._executor.shut = True
