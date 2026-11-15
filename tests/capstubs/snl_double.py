# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test double of the product server module tensorrt_llm.grpc.openengine.server for S-NL
(--server-module snl_double): OpenEngineServer(host, port, llm, model) builds its servicer from
the module-level name OpenEngineInferenceServicer (as the product's server.py does) and registers
its Generate with protobuf (de)serializers of the pb2 double; the servicer follows the product's
Generate on the paths S-NL depends on (probe-tree servicer.py / streaming.py at the capkit
baseline): request_id must be non-empty and unique among active requests, a ValueError from
generate_async becomes INVALID_ARGUMENT, one token response per result with the delta of the
cumulative output.token_ids, a finished response with usage after the final result, an error
response for result.error, abort() of the handle when the stream ends without the engine having
finished; a Control route runs the product's inference probe. It is not the product: what S-NL does with the real servicer is checked in the cluster
smoke (INTERFACES errata FT1)."""

import grpc

import oe_double

METHOD = "/openengine.v1.Inference/Generate"
PROBE = "/openengine.v1.Control/CheckHealth"


class _SP:
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens
        self.n = 1


class OpenEngineInferenceServicer:
    def __init__(self, llm, model, **kwargs):
        self._llm = llm
        self._model = model
        self._guided = llm.args.guided_decoding_backend
        self._active = {}

    async def Generate(self, request, context):  # noqa: N802 - the product's name
        rid = request.request_id
        if not rid:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request_id must be non-empty")
        if rid in self._active:
            await context.abort(grpc.StatusCode.ALREADY_EXISTS, "request_id already active")
        self._active[rid] = True
        handle = None
        terminal = False
        try:
            kind = request.WhichOneof("input")
            inputs = ({"prompt_token_ids": list(request.token_ids.ids)} if kind == "token_ids"
                      else request.prompt)
            try:
                handle = self._llm.generate_async(
                    inputs=inputs, sampling_params=_SP(request.stopping.max_tokens),
                    streaming=True, trace_headers=None, disaggregated_params=None)
            except ValueError as e:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
                return
            sent = 0
            async for result in handle:
                if context.cancelled():
                    return
                if result.error:
                    terminal = True
                    yield oe_double.GenerateResponse(request_id=rid, error=oe_double.EngineError())
                    return
                out = result.outputs[0]
                delta = out.token_ids[sent:]
                sent = len(out.token_ids)
                if delta:
                    yield oe_double.GenerateResponse(request_id=rid, token=oe_double.TokenOutput(
                        output_index=0, tokens=[oe_double.TokenInfo(token_id=t) for t in delta]))
                if result.finished:
                    terminal = True
                    n = len(out.token_ids)
                    p = len(result.prompt_token_ids)
                    yield oe_double.GenerateResponse(
                        request_id=rid,
                        finished=oe_double.GenerationFinished(
                            output_index=0,
                            reason=oe_double.FINISH_REASON_LENGTH
                            if out.finish_reason == "length" else 0),
                        usage=oe_double.Usage(prompt_tokens=p, completion_tokens=n,
                                              total_tokens=p + n))
                    return
        finally:
            if handle is not None and not terminal:
                handle.abort()
            self._active.pop(rid, None)


async def _inference_probe(llm):
    """The product Control's inference probe (control.py _run_inference_probe): one token,
    non-streaming, awaited with aresult()."""
    handle = llm.generate_async([1], sampling_params=_SP(1), streaming=False)
    await handle.aresult()
    return b"READY"


class _Routes(grpc.GenericRpcHandler):
    def __init__(self, servicer, llm):
        async def probe(request, context):
            return await _inference_probe(llm)

        self._h = {METHOD: grpc.unary_stream_rpc_method_handler(
            servicer.Generate, request_deserializer=oe_double.GenerateRequest.FromString,
            response_serializer=oe_double.GenerateResponse.SerializeToString),
            PROBE: grpc.unary_unary_rpc_method_handler(
                probe, request_deserializer=None,
                response_serializer=None)}

    def service(self, details):
        return self._h.get(details.method)


class OpenEngineServer:
    def __init__(self, host, port, llm, model, **kwargs):
        self._server = grpc.aio.server()
        servicer = OpenEngineInferenceServicer(llm, model)  # the module-level name
        self._server.add_generic_rpc_handlers([_Routes(servicer, llm)])
        self.port = self._server.add_insecure_port("%s:%d" % (host, port))
        self.host = host

    async def start(self):
        await self._server.start()

    async def stop(self, grace=5.0):
        await self._server.stop(grace)
