# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Responses request asks for logprobs only when it gives ``top_logprobs``.

Responses output carries no logprobs, so a request that leaves
``top_logprobs`` out must not be refused for an engine that declares no output
logprobs. A request that gives ``top_logprobs``, 0 included, still asks for
them.
"""

import asyncio
import dataclasses
import json
from types import SimpleNamespace

import httpx
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.serve.openai_protocol import ResponsesRequest, ResponsesResponse
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.openengine_poc import OpenEngineLLM, contracts, mapping
from tensorrt_llm.serve.openengine_poc.facade import adapt_server
from tensorrt_llm.tokenizer import TransformersTokenizer

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from fake_openengine import PROFILES, FakeEngine  # noqa: E402
from fakes import trtllm_like  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

PROMPT = [2, 3, 4]
# The request's top_logprobs as sent ("absent" leaves it out), and the output
# logprobs it asks for.
TOP_LOGPROBS = {
    "absent": (..., None),
    "null": (None, None),
    "zero": (0, 0),
    "three": (3, 3),
}


def _request(top_logprobs=...) -> ResponsesRequest:
    body = {"model": "m", "input": "Hi"}
    if top_logprobs is not ...:
        body["top_logprobs"] = top_logprobs
    return ResponsesRequest.model_validate(body)


@pytest.mark.parametrize("case", TOP_LOGPROBS)
def test_logprobs_are_asked_for_only_when_top_logprobs_is_given(case):
    given, asked = TOP_LOGPROBS[case]
    assert _request(given).to_sampling_params().logprobs == asked


@pytest.mark.parametrize("case", TOP_LOGPROBS)
def test_the_response_reports_the_requested_top_logprobs(case):
    given, asked = TOP_LOGPROBS[case]
    request = _request(given)
    response = ResponsesResponse.from_request(
        request,
        request.to_sampling_params(),
        model_name="m",
        created_time=0,
        output=[],
        status="completed",
    )
    assert response.top_logprobs == (asked or 0)


def _no_output_logprobs(caps):
    return dataclasses.replace(caps, output_logprobs_max_top_n=None)


@pytest.mark.parametrize("case", TOP_LOGPROBS)
def test_the_mapping_refuses_only_a_request_asking_logprobs_of_an_engine_without_them(case):
    given, asked = TOP_LOGPROBS[case]
    info, caps = trtllm_like()
    sampling_params = _request(given).to_sampling_params()
    sampling_params.max_tokens = 1

    def run(capabilities):
        return mapping.map_request(
            PROMPT,
            sampling_params,
            capabilities,
            contracts.GenerateOptions(),
            engine_name=info.engine_name,
        )

    assert [r.output_logprobs_top_n for r in run(caps)] == [asked]
    if asked is None:
        assert [r.output_logprobs_top_n for r in run(_no_output_logprobs(caps))] == [None]
    else:
        with pytest.raises(ValueError, match=f"^logprobs={asked} is not supported"):
            run(_no_output_logprobs(caps))


# --- through OpenAIServer, the facade and gRPC to the fake engine ---------------------

MODEL = "stub-model"
WAIT_SECONDS = 30
_VOCAB = {"<unk>": 0, "</s>": 1, "He": 2, "llo": 3, " world": 4, "!": 5}
EOS = 1


def _tokenizer() -> TransformersTokenizer:
    backend = Tokenizer(WordLevel(_VOCAB, unk_token="<unk>"))
    backend.add_special_tokens(["</s>"])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>")
    hf.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return TransformersTokenizer(hf)


def _profile(**changes):
    return PROFILES["trtllm-like"].replace(
        model=MODEL,
        max_context_length=64,
        vocab_size=len(_VOCAB),
        eos_token_ids=(EOS,),
        bos_token_id=None,
        **changes,
    )


def _unused_input_processor(prompt, sampling_params):
    # A text-only Responses request reaches the facade as the token ids its chat
    # template renders, so the facade never tokenizes it.
    raise AssertionError("a Responses request should arrive as token ids")


def _args():
    return SimpleNamespace(
        backend="pytorch",
        reasoning_parser=None,
        gather_generation_logits=False,
        guided_decoding_backend=None,
        num_postprocess_workers=0,
        cache_transceiver_config=None,
        trust_remote_code=False,
        checkpoint_format=None,
        return_perf_metrics=False,
        perf_metrics_output_dir=None,
        speculative_config=None,
        post_processor_hook=None,
        sleep_config=None,
        num_serve_frontends=1,
        enable_iter_perf_stats=False,
        max_beam_width=1,
        custom_tokenizer=None,
        max_batch_size=None,
        kv_cache_config=None,
    )


def _post_responses(profile, tmp_path, **fields):
    """POST one Responses request to OpenAIServer on the facade and a fake ``profile`` engine.

    Returns the HTTP status, the reply body and the Generate requests the engine received.
    """

    async def main():
        async with FakeEngine(profile) as engine:
            facade = OpenEngineLLM(
                args=_args(),
                tokenizer=_tokenizer(),
                input_processor=_unused_input_processor,
                hf_model_dir=str(tmp_path),
                endpoint=engine.target,
                ready_timeout=WAIT_SECONDS,
            )
            await facade.start()
            previous_media_executor = BaseMediaIO._executor
            server = OpenAIServer(
                generator=facade,
                model=MODEL,
                tool_parser=None,
                server_role=None,
                metadata_server_cfg=None,
            )
            adapt_server(server, facade)
            BaseMediaIO.set_executor(previous_media_executor)
            server._media_load_executor.shutdown(wait=False)
            try:
                transport = httpx.ASGITransport(app=server.app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
                    reply = await asyncio.wait_for(
                        http.post(
                            "/v1/responses",
                            json={"model": MODEL, "input": "Hi", "max_output_tokens": 4, **fields},
                        ),
                        WAIT_SECONDS * 2,
                    )
                    return reply.status_code, reply.json(), list(engine.requests)
            finally:
                await facade.aclose()
                server._input_proc_executor.shutdown(wait=False)
                server._metrics_stats_executor.shutdown(wait=False)

    return asyncio.run(main())


def test_over_grpc_a_responses_request_is_served_by_an_engine_without_output_logprobs(tmp_path):
    status, reply, requests = _post_responses(_profile(output_logprobs_max_top_n=None), tmp_path)
    assert status == 200, reply
    assert reply["top_logprobs"] == 0
    assert len(requests) == 1
    assert not requests[0].response.return_output_logprobs
    assert not requests[0].response.HasField("output_candidates")


@pytest.mark.parametrize("top_logprobs", [0, 2])
def test_over_grpc_top_logprobs_asks_the_engine_for_logprobs(tmp_path, top_logprobs):
    status, reply, requests = _post_responses(_profile(), tmp_path, top_logprobs=top_logprobs)
    assert status == 200, reply
    assert reply["top_logprobs"] == top_logprobs
    assert len(requests) == 1
    assert requests[0].response.return_output_logprobs
    # top_n is a oneof member, so 0 counts only when top_n is the one set.
    assert requests[0].response.output_candidates.WhichOneof("selection") == "top_n"
    assert requests[0].response.output_candidates.top_n == top_logprobs


@pytest.mark.parametrize("top_logprobs", [0, 2])
def test_over_grpc_top_logprobs_is_refused_for_an_engine_without_output_logprobs(
    tmp_path, top_logprobs
):
    status, reply, requests = _post_responses(
        _profile(output_logprobs_max_top_n=None), tmp_path, top_logprobs=top_logprobs
    )
    assert status == 400, reply
    assert f"logprobs={top_logprobs} is not supported" in json.dumps(reply), reply
    assert requests == []
