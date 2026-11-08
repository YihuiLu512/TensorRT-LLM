# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``map_media`` slot: a prompt's media as ``MediaItem`` references for the engine (OE-MED).

The engine fetches, decodes and preprocesses the media and expands their
placeholders; the frontend sends the media as the request gave them, in
placeholder order, with the prompt's placeholders unexpanded (OE-MED-2). So
the serving side does not download or decode anything: for a chat request
``OpenAIServer`` takes each item's reference from the messages
(:func:`chat_media_references`) instead of loading it.

How a reference becomes a ``MediaItem`` (OE-MED-3):

* an ``http``/``https`` URL is sent as ``url``, and the engine fetches it;
* a ``data:`` URI is sent as ``data_uri``;
* a local path or ``file://`` URI, which only the frontend can read, is read
  under the ``TRTLLM_MEDIA_ALLOWED_LOCAL_PATH`` policy of the in-process server
  and sent as a ``data_uri``; so are ``bytes`` given through the Python API.
  ``raw_bytes`` is never sent.

Media the frontend already decoded (a PIL image, a tensor), precomputed
embeddings (OE-MED-5) and ``mm_processor_kwargs`` have no carrier and are
refused, as are modalities the engine does not declare (OE-MED-1).
"""

from __future__ import annotations

import base64
import mimetypes
import os
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any
from urllib.parse import urlparse

from .engine_client import MediaSpec, ModelCapabilities

__all__ = ["MAX_INLINE_MEDIA_BYTES", "chat_media_references", "map_media"]

_MODALITIES = ("image", "video", "audio")
# Characters of data URIs one request may carry. A Generate message is at
# most 64 MiB (OE-TRN-3); the 2 MiB left hold the rest of the request, whose
# token ids take at most 3 bytes each for vocabularies below 2**21.
MAX_INLINE_MEDIA_BYTES = 62 << 20


def _refused(what: str) -> ValueError:
    return ValueError(f"{what} cannot be sent to an OpenEngine engine")


# -- Chat messages ------------------------------------------------------------


def chat_media_references(messages: Iterable[MutableMapping[str, Any]]) -> dict[str, list[str]]:
    """The media references of chat messages, by modality, in the order ``OpenAIServer`` numbers them.

    Each modality's list is ordered as ``parse_chat_messages_coroutines``
    numbers the items of that modality, so ``mm_item_order`` indexes it. A
    message's content parts may be a one-pass iterator; they are read into a
    list and put back into the message, so the messages can be parsed again.

    Raises:
        ValueError: A part holds image embeddings (OE-MED-5), an
            ``input_audio`` part has no data, or a media part's url is not a
            string.
    """
    from tensorrt_llm.serve.chat_utils import MM_PARSER_MAP, VALID_MESSAGE_CONTENT_MM_PART_TYPES

    references: dict[str, list[str]] = {}
    for message in messages:
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        parts = list(content)
        message["content"] = parts
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if not isinstance(part_type, str) or part_type not in MM_PARSER_MAP:
                # Parsing the messages reports unknown part types.
                continue
            value = MM_PARSER_MAP[part_type](part)
            if part_type == "text" or (
                value is None and part_type in VALID_MESSAGE_CONTENT_MM_PART_TYPES
            ):
                # Parsing skips an empty part the same way.
                continue
            if part_type == "image_embeds":
                raise _refused("image embeddings (image_embeds)")
            if part_type == "input_audio":
                data = value.get("data") if isinstance(value, dict) else None
                if not isinstance(data, str) or not data:
                    raise ValueError(
                        "input_audio part is missing a non-empty 'data' field with "
                        "base64-encoded audio content."
                    )
                # As parsing does: the base64 audio becomes a data URI.
                references.setdefault("audio", []).append(
                    f"data:audio/{value.get('format', '')};base64,{data}"
                )
                continue
            if not isinstance(value, str):
                raise ValueError(f"{part_type} must give its url as a string")
            modality = part_type.removesuffix("_url")
            references.setdefault(modality, []).append(value)
    return references


# -- Mapping ------------------------------------------------------------------


def _read_local(reference: str, budget: int) -> bytes:
    """A local media file, read under the server's allowed-directory policy.

    ``budget`` bounds the file's data URI, which base64 makes 4/3 as long.
    """
    from tensorrt_llm.inputs.media_io import open_allowed_local_media

    limit = budget // 4 * 3
    fd = open_allowed_local_media(reference)
    try:
        chunks: list[bytes] = []
        remaining = limit + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1 << 20))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except OSError as error:
        raise ValueError(f"local media file could not be read: {reference!r}") from error
    finally:
        os.close(fd)
    data = b"".join(chunks)
    if len(data) > limit:
        raise _too_large()
    return data


def _too_large() -> ValueError:
    return ValueError(
        f"the inline media of a request (data URIs, local files and bytes) exceed "
        f"{MAX_INLINE_MEDIA_BYTES} bytes as data URIs; send large media as http(s) URLs"
    )


def _data_uri(modality: str, data: bytes, mime_type: str | None, budget: int) -> str:
    if not data:
        raise ValueError(f"empty {modality} data")
    uri = f"data:{mime_type or ''};base64,{base64.b64encode(data).decode('ascii')}"
    if len(uri) > budget:
        raise _too_large()
    return uri


def _mime_type(reference: str, modality: str) -> str | None:
    guessed, _ = mimetypes.guess_type(urlparse(reference).path or reference)
    return guessed if guessed and guessed.startswith(f"{modality}/") else None


def _media_spec(modality: str, item: Any, uuid: str | None, budget: int) -> MediaSpec:
    if isinstance(item, (bytes, bytearray, memoryview)):
        data_uri = _data_uri(modality, bytes(item), None, budget)
        return MediaSpec(modality=modality, data_uri=data_uri, uuid=uuid)
    if not isinstance(item, str):
        raise _refused(
            f"{modality} given as decoded {type(item).__name__} (send a URL, a data URI or bytes)"
        )
    if not item:
        raise ValueError(f"empty {modality} reference")
    parsed = urlparse(item)
    scheme = parsed.scheme.lower()
    if scheme in ("http", "https"):
        if not parsed.hostname:
            raise ValueError(f"{modality} URL has no host: {item!r}")
        return MediaSpec(modality=modality, url=item, uuid=uuid)
    if scheme == "data":
        if len(item) > budget:
            raise _too_large()
        return MediaSpec(modality=modality, data_uri=item, uuid=uuid)
    if scheme in ("", "file"):
        data = _read_local(item, budget)
        data_uri = _data_uri(modality, data, _mime_type(item, modality), budget)
        return MediaSpec(modality=modality, data_uri=data_uri, uuid=uuid)
    raise ValueError(f"unsupported {modality} URL scheme {scheme!r}")


def _ordered(
    data: Mapping[str, Sequence[Any]], item_order: Sequence[Mapping[str, Any]] | None
) -> list[tuple[str, int]]:
    """``(modality, index)`` of every item, in placeholder order."""
    if not item_order:
        return [
            (modality, index) for modality, items in data.items() for index in range(len(items))
        ]
    if (
        not isinstance(item_order, Sequence)
        or isinstance(item_order, str)
        or not all(isinstance(entry, Mapping) for entry in item_order)
    ):
        raise ValueError("mm_item_order must list {'modality', 'index'} entries")
    order = [(entry.get("modality"), entry.get("index")) for entry in item_order]
    expected = sorted((m, i) for m, items in data.items() for i in range(len(items)))
    if sorted(order, key=repr) != sorted(expected, key=repr):
        raise ValueError("mm_item_order does not list each multimodal item exactly once")
    return order


def _uuids(modality: str, given: Any, count: int) -> list[str | None]:
    """One uuid (or None) per item of ``modality``; a single uuid stands for a single item."""
    if given is None:
        return [None] * count
    if isinstance(given, str):
        given = [given]
    if not isinstance(given, Sequence) or len(given) != count:
        raise ValueError(
            f"multi_modal_uuids[{modality!r}] must give one uuid or None for each of its "
            f"{count} items"
        )
    if not all(uuid is None or isinstance(uuid, str) for uuid in given):
        raise ValueError(f"multi_modal_uuids[{modality!r}] must hold strings or None")
    return [uuid or None for uuid in given]


def map_media(
    inputs: Mapping[str, Any], capabilities: ModelCapabilities, *, engine_name: str
) -> list[MediaSpec]:
    """The media of a prompt as ``MediaSpec`` items in placeholder order.

    ``inputs["multi_modal_data"]`` maps each modality to its items: URL or
    data-URI strings, local paths, or bytes. ``mm_item_order`` gives the
    placeholder order (by modality, then index, when absent);
    ``multi_modal_uuids`` gives each item's ``uuid`` (OE-MED-4).

    Raises:
        ValueError: The engine does not take these media, or they cannot be
            sent; reported as HTTP 400.
    """
    if inputs.get("multi_modal_embeddings") is not None:
        raise _refused("multi_modal_embeddings (precomputed embeddings)")
    if inputs.get("mm_processor_kwargs"):
        raise _refused("mm_processor_kwargs")
    data = inputs.get("multi_modal_data")
    if not data:
        return []
    if not isinstance(data, Mapping):
        raise ValueError("multi_modal_data must map each modality to its items")
    for modality, items in data.items():
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            raise ValueError(f"multi_modal_data[{modality!r}] must list its items")
    data = {modality: list(items) for modality, items in data.items() if items}
    for modality in data:
        if modality not in _MODALITIES:
            raise ValueError(f"unsupported multimodal modality {modality!r}")
        if (
            not capabilities.supports_multimodal
            or modality not in capabilities.supported_modalities
        ):
            # OE-MED-1.
            supported = ", ".join(sorted(capabilities.supported_modalities)) or "none"
            raise ValueError(
                f"multimodal {modality} inputs are not supported by the OpenEngine engine "
                f"{engine_name!r} (supported modalities: {supported})"
            )
    uuids = inputs.get("multi_modal_uuids") or {}
    if not isinstance(uuids, Mapping):
        raise ValueError("multi_modal_uuids must map each modality to its items' uuids")
    item_uuids = {
        modality: _uuids(modality, uuids.get(modality), len(data[modality])) for modality in data
    }
    budget = MAX_INLINE_MEDIA_BYTES
    specs = []
    for modality, index in _ordered(data, inputs.get("mm_item_order")):
        spec = _media_spec(modality, data[modality][index], item_uuids[modality][index], budget)
        budget -= len(spec.data_uri or "")
        specs.append(spec)
    return specs
