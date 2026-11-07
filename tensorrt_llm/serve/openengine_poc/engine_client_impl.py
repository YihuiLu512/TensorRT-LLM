# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``EngineClient`` over an OpenEngine v1 engine, built on the OpenEngine client library.

:class:`OpenEngineEngineClient` turns each :class:`EngineRequest` into a
``GenerateRequest`` and each ``GenerateResponse`` back into an
:class:`EngineEvent`, and caches the engine's ``ServerInfo`` and ``ModelInfo``
as :class:`EngineInfo` and :class:`ModelCapabilities`. Errors are the client
library's: :class:`OpenEngineRpcError` for a failed call (raised by the first
``__anext__`` when the engine refuses a request), :class:`OpenEngineEngineError`
when the engine ends an accepted request with an error, and
:class:`OpenEngineProtocolError` for a malformed stream.
"""

from __future__ import annotations

import grpc

from tensorrt_llm.grpc.openengine.bindings import (
    generation_pb2,
    lifecycle_pb2,
    lora_pb2,
    model_pb2,
    server_pb2,
)
from tensorrt_llm.grpc.openengine_client import (
    GenerateStream,
    OpenEngineClient,
    OpenEngineProtocolError,
    OpenEngineRpcError,
)
from tensorrt_llm.grpc.openengine_conventions import model_capabilities_from_proto, struct_to_dict

from .engine_client import (
    EngineEvent,
    EngineHealth,
    EngineInfo,
    EngineLoad,
    EngineRequest,
    FinishReason,
    HealthState,
    KvEventSource,
    LoraAdapter,
    ModelCapabilities,
    OutputFinished,
    PromptInfo,
    TokenDelta,
    TokenLogprob,
    Usage,
)

__all__ = ["EngineEventStream", "OpenEngineEngineClient", "to_generate_request"]

_SAMPLING_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "seed",
    }
)
_MODALITIES = {
    "image": generation_pb2.MODALITY_IMAGE,
    "video": generation_pb2.MODALITY_VIDEO,
    "audio": generation_pb2.MODALITY_AUDIO,
}
_FINISH_REASONS: dict[int, FinishReason] = {
    generation_pb2.FINISH_REASON_STOP: "stop",
    generation_pb2.FINISH_REASON_LENGTH: "length",
    generation_pb2.FINISH_REASON_CANCELLED: "cancelled",
}
_HEALTH_STATES: dict[int, HealthState] = {
    lifecycle_pb2.HEALTH_STATE_STARTING: "starting",
    lifecycle_pb2.HEALTH_STATE_READY: "ready",
    lifecycle_pb2.HEALTH_STATE_DEGRADED: "degraded",
    lifecycle_pb2.HEALTH_STATE_NOT_READY: "not_ready",
}


def to_generate_request(request: EngineRequest) -> generation_pb2.GenerateRequest:
    """Build the ``GenerateRequest`` for one :class:`EngineRequest`.

    Raises:
        ValueError: ``request.sampling`` names a field ``SamplingParams`` does not have.
    """
    message = generation_pb2.GenerateRequest(
        request_id=request.request_id,
        model=request.model,
        token_ids=generation_pb2.TokenIds(ids=request.prompt_token_ids),
    )
    unknown = sorted(set(request.sampling) - _SAMPLING_FIELDS)
    if unknown:
        raise ValueError(f"unknown OpenEngine sampling field(s): {', '.join(unknown)}")
    sampling = message.sampling
    for name, value in request.sampling.items():
        setattr(sampling, name, value)
    if request.num_sequences > 1:
        sampling.num_sequences = request.num_sequences

    stopping = message.stopping
    if request.max_tokens is not None:
        stopping.max_tokens = request.max_tokens
    if request.min_tokens is not None:
        stopping.min_tokens = request.min_tokens
    for token_id in request.stop_token_ids:
        stopping.conditions.add(stop_token_id=token_id)
    for text in request.stop_texts:
        stopping.conditions.add(stop_text=text)
    if request.ignore_eos is not None:
        stopping.ignore_eos = request.ignore_eos
    if request.include_stop_in_output is not None:
        stopping.include_stop_in_output = request.include_stop_in_output

    response = message.response
    if request.output_logprobs_top_n is not None:
        response.return_output_logprobs = True
        response.output_candidates.top_n = request.output_logprobs_top_n
    if request.prompt_logprobs_top_n is not None:
        response.return_prompt_logprobs = True
        response.prompt_candidates.top_n = request.prompt_logprobs_top_n

    if request.guided is not None:
        if request.guided.mode == "json_object":
            message.guided.json_object.SetInParent()
        else:
            setattr(message.guided, request.guided.mode, request.guided.value or "")
    if request.cache_salt is not None:
        message.kv.cache_salt = request.cache_salt
    for media in request.media:
        item = message.media.add(modality=_MODALITIES[media.modality])
        if media.url is not None:
            item.url = media.url
        elif media.data_uri is not None:
            item.data_uri = media.data_uri
        else:
            item.raw_bytes = media.raw_bytes or b""
        if media.mime_type:
            item.mime_type = media.mime_type
        if media.uuid:
            item.uuid = media.uuid
    if request.lora_name:
        message.lora_name = request.lora_name
    if request.extra:
        message.extra.update(dict(request.extra))
    return message


def _token_logprob(info: generation_pb2.TokenInfo) -> TokenLogprob:
    return TokenLogprob(
        token_id=info.token_id,
        logprob=info.logprob,
        rank=info.rank if info.HasField("rank") else None,
        top=tuple((candidate.token_id, candidate.logprob) for candidate in info.candidates),
    )


def _usage(message: generation_pb2.Usage) -> Usage:
    return Usage(
        prompt_tokens=message.prompt_tokens,
        completion_tokens=message.completion_tokens,
        cached_prompt_tokens=message.cached_prompt_tokens
        if message.HasField("cached_prompt_tokens")
        else None,
    )


class EngineEventStream:
    """The events of one Generate call; see :meth:`OpenEngineEngineClient.generate`.

    Besides iterating, :meth:`cancel` cancels the call from synchronous code and
    :meth:`aclose` cancels it if it has not finished and waits for the call to end.
    """

    def __init__(self, stream: GenerateStream, *, endpoint: str, logprobs: bool) -> None:
        self._stream = stream
        self._endpoint = endpoint
        self._logprobs = logprobs

    @property
    def request_id(self) -> str:
        return self._stream.request_id

    @property
    def sent(self) -> bool:
        """Whether the call was sent to the engine."""
        return self._stream.sent

    @property
    def cancelled(self) -> bool:
        """Whether the call was cancelled by this side before it finished."""
        return self._stream.cancelled

    def cancel(self) -> None:
        """Cancel the call; the engine treats it as an abort. Has no effect once it finished."""
        self._stream.cancel()

    async def aclose(self) -> None:
        await self._stream.aclose()

    def __aiter__(self) -> EngineEventStream:
        return self

    async def __anext__(self) -> EngineEvent:
        while True:
            response = await self._stream.__anext__()
            event = self._event(response)
            if event is not None:
                return event

    def _event(self, response: generation_pb2.GenerateResponse) -> EngineEvent | None:
        kind = response.WhichOneof("event")
        if kind == "token":
            token = response.token
            return TokenDelta(
                output_index=token.output_index,
                token_ids=tuple(info.token_id for info in token.tokens),
                logprobs=tuple(_token_logprob(info) for info in token.tokens)
                if self._logprobs
                else None,
            )
        if kind == "finished":
            finished = response.finished
            match = finished.stop_match.WhichOneof("match")
            return OutputFinished(
                output_index=finished.output_index,
                reason=_FINISH_REASONS[finished.reason],
                eos_token_id=finished.stop_match.eos_token_id if match == "eos_token_id" else None,
                stop_token_id=finished.stop_match.stop_token_id
                if match == "stop_token_id"
                else None,
                stop_text=finished.stop_match.stop_text if match == "stop_text" else None,
                usage=_usage(response.usage) if response.HasField("usage") else None,
            )
        if kind == "prompt":
            return PromptInfo(
                token_logprobs=tuple(_token_logprob(info) for info in response.prompt.tokens)
            )
        if kind == "prefill_ready":
            # Requests from this client never ask for prefill only.
            raise OpenEngineProtocolError(
                f"OpenEngine engine sent prefill_ready for request {self.request_id}, "
                "which did not ask for prefill only",
                target=self._endpoint,
                request_id=self.request_id,
            )
        return None


def _engine_info(info: server_pb2.ServerInfo) -> EngineInfo:
    capacity = info.capacity
    return EngineInfo(
        engine_name=info.engine_name,
        engine_version=info.engine_version,
        instance_id=info.instance_id,
        schema_revision=info.schema_revision,
        minimum_client_revision=info.minimum_client_revision,
        schema_release=info.schema_release,
        max_running_requests=capacity.max_running_requests
        if capacity.HasField("max_running_requests")
        else None,
        max_loras=capacity.max_loras if capacity.HasField("max_loras") else None,
        extra=struct_to_dict(info.extra) if info.HasField("extra") else {},
    )


class OpenEngineEngineClient:
    """:class:`EngineClient` for one OpenEngine v1 engine.

    Create it with :meth:`connect`. Every method runs on the event loop it was
    connected on. ``server_info()`` and ``model_info()`` return what the last
    :meth:`connect` or :meth:`refresh` read.
    """

    def __init__(self, client: OpenEngineClient) -> None:
        self._client = client
        self._server_info: EngineInfo
        self._model_info: ModelCapabilities
        self._cache(client.server_info, client.model_info)

    @classmethod
    async def connect(
        cls,
        endpoint: str,
        *,
        ready_timeout: float,
        control_timeout: float = 5.0,
        allow_newer_schema: bool = False,
    ) -> OpenEngineEngineClient:
        """Connect, wait until the engine is ready and read its identity and capabilities.

        Raises:
            OpenEngineRpcError: The engine was not ready in time, or a Control RPC failed.
            OpenEngineIncompatibleServerError: The engine's schema revision is not supported.
        """
        client = await OpenEngineClient.connect(
            endpoint,
            ready_timeout=ready_timeout,
            control_timeout=control_timeout,
            allow_newer_schema=allow_newer_schema,
        )
        return cls(client)

    @property
    def endpoint(self) -> str:
        return self._client.target

    def _cache(self, server_info: server_pb2.ServerInfo, model_info: model_pb2.ModelInfo) -> None:
        self._server_info = _engine_info(server_info)
        self._model_info = model_capabilities_from_proto(server_info, model_info)

    async def refresh(self) -> None:
        """Read the engine's identity and capabilities again.

        Raises:
            OpenEngineRpcError: A Control RPC failed.
            OpenEngineIncompatibleServerError: The engine's schema revision is not supported.
        """
        await self._client.refresh()
        self._cache(self._client.server_info, self._client.model_info)

    async def current_instance_id(self) -> str:
        """The engine's instance id now, read without updating the cached identity."""
        return (await self._client.get_server_info()).instance_id

    def generate(self, request: EngineRequest) -> EngineEventStream:
        """Start one Generate call; the request is sent at the first iteration.

        Raises:
            ValueError: The request cannot be expressed in OpenEngine, or its
                metadata is invalid.
        """
        message = to_generate_request(request)
        stream = self._client.generate(message, metadata=dict(request.metadata))
        return EngineEventStream(
            stream, endpoint=self.endpoint, logprobs=request.output_logprobs_top_n is not None
        )

    async def abort(self, request_id: str) -> None:
        await self._client.abort(request_id)

    async def health(self, *, deep: bool = False, timeout: float | None = None) -> EngineHealth:
        response = await self._client.health(include_inference_probe=deep, timeout=timeout)
        return EngineHealth(
            state=_HEALTH_STATES.get(response.state, "unspecified"),
            checks=tuple(
                (check.name, _HEALTH_STATES.get(check.state, "unspecified"), check.message)
                for check in response.checks
            ),
        )

    async def load(self) -> EngineLoad | None:
        info = await self._client.get_load()
        if info is None:
            return None

        def optional(name: str) -> int | None:
            return getattr(info, name) if info.HasField(name) else None

        return EngineLoad(
            instance_id=info.instance_id,
            running_requests=optional("running_requests"),
            queued_requests=optional("queued_requests"),
            used_kv_blocks=optional("used_kv_blocks"),
            total_kv_blocks=optional("total_kv_blocks"),
            attributes=struct_to_dict(info.attributes) if info.HasField("attributes") else {},
        )

    def server_info(self) -> EngineInfo:
        return self._server_info

    def model_info(self) -> ModelCapabilities:
        return self._model_info

    async def load_lora(self, name: str, int_id: int, path: str) -> bool:
        response = await self._client.load_lora(
            lora_pb2.LoraAdapter(lora_id=int_id, lora_name=name, source_path=path)
        )
        return response.already_loaded

    async def list_loras(self) -> list[LoraAdapter]:
        return [
            LoraAdapter(name=adapter.lora_name, int_id=adapter.lora_id, path=adapter.source_path)
            for adapter in await self._client.list_loras()
        ]

    async def kv_event_sources(self) -> list[KvEventSource]:
        try:
            sources = await self._client.get_kv_event_sources()
        except OpenEngineRpcError as error:
            if error.code == grpc.StatusCode.UNIMPLEMENTED:
                return []
            raise
        return [
            KvEventSource(
                transport=source.transport,
                host=source.endpoint_addr.host,
                port=source.endpoint_addr.port,
                protocol=source.endpoint_addr.protocol,
                topic=source.topic,
                replay_endpoint=source.replay_endpoint,
                data_parallel_rank=source.data_parallel_rank
                if source.HasField("data_parallel_rank")
                else None,
                encoding=source.encoding,
            )
            for source in sources
        ]

    async def close(self) -> None:
        """Cancel the calls still running and close the connection. Safe to call again."""
        await self._client.close()
