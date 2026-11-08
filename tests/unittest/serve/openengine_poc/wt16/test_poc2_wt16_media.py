# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The map_media slot (OE-MED): media references become MediaItems, checked against the gRPC fake."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import pytest

from tensorrt_llm.grpc.openengine.bindings import generation_pb2
from tensorrt_llm.serve.openengine_poc import contracts, media
from tensorrt_llm.serve.openengine_poc.engine_client import EngineRequest, MediaSpec
from tensorrt_llm.serve.openengine_poc.engine_client_impl import OpenEngineEngineClient

__extra_import_path__ = ["~/tests/unittest/grpc/openengine_client"]
from fake_openengine import PROFILES, FakeEngine  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

ALL = frozenset({"image", "video", "audio"})
PNG = b"\x89PNG\r\n\x1a\nfake-image"
DATA_URI = "data:image/png;base64," + base64.b64encode(PNG).decode()


def _caps(modalities=ALL, supports_multimodal=True):
    return SimpleNamespace(supports_multimodal=supports_multimodal, supported_modalities=modalities)


def _map(inputs, caps=None):
    return media.map_media(inputs, caps or _caps(), engine_name="fake")


@pytest.fixture
def media_dir(tmp_path, monkeypatch):
    root = tmp_path / "media"
    root.mkdir()
    (root / "seashore.png").write_bytes(PNG)
    monkeypatch.setenv("TRTLLM_MEDIA_ALLOWED_LOCAL_PATH", str(root))
    return root


def test_the_slot_resolves_to_this_module():
    contracts.reset_slot_cache()
    try:
        assert contracts.resolve_slot("map_media")[1] == f"{media.__name__}:map_media"
    finally:
        contracts.reset_slot_cache()


# -- The three sources --------------------------------------------------------


def test_urls_stay_and_inline_media_become_data_uris(media_dir):
    # OE-MED-3: local files and bytes go as data URIs, never as raw_bytes.
    specs = _map(
        {
            "multi_modal_data": {
                "image": ["https://example.com/a.png", DATA_URI, str(media_dir / "seashore.png")],
                "audio": [b"RIFF-audio"],
            }
        }
    )
    assert specs == [
        MediaSpec("image", url="https://example.com/a.png"),
        MediaSpec("image", data_uri=DATA_URI),
        MediaSpec("image", data_uri=DATA_URI),
        MediaSpec("audio", data_uri="data:;base64," + base64.b64encode(b"RIFF-audio").decode()),
    ]


def test_a_file_uri_is_read_like_a_path(media_dir):
    [spec] = _map({"multi_modal_data": {"image": [f"file://{media_dir / 'seashore.png'}"]}})
    assert spec.data_uri == DATA_URI


def test_an_empty_local_file_is_refused(media_dir):
    (media_dir / "empty.png").write_bytes(b"")
    with pytest.raises(ValueError, match="empty image data"):
        _map({"multi_modal_data": {"image": [str(media_dir / "empty.png")]}})


@pytest.mark.parametrize("allowed", [None, "elsewhere"])
def test_local_files_follow_the_allowed_directory(tmp_path, monkeypatch, allowed):
    path = tmp_path / "secret.png"
    path.write_bytes(PNG)
    if allowed is None:
        monkeypatch.delenv("TRTLLM_MEDIA_ALLOWED_LOCAL_PATH", raising=False)
    else:
        (tmp_path / allowed).mkdir()
        monkeypatch.setenv("TRTLLM_MEDIA_ALLOWED_LOCAL_PATH", str(tmp_path / allowed))
    with pytest.raises(ValueError, match="Local media paths are not accepted"):
        _map({"multi_modal_data": {"image": [str(path)]}})


def test_items_follow_the_placeholder_order_and_carry_their_uuids():
    specs = _map(
        {
            "multi_modal_data": {
                "image": ["http://h/1.png", "http://h/2.png"],
                "video": ["http://h/v"],
            },
            "mm_item_order": [
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 1},
                {"modality": "image", "index": 0},
            ],
            "multi_modal_uuids": {"image": [None, "hash-2"]},
        }
    )
    assert [(s.modality, s.url, s.uuid) for s in specs] == [
        ("video", "http://h/v", None),
        ("image", "http://h/2.png", "hash-2"),
        ("image", "http://h/1.png", None),
    ]


def test_a_single_uuid_names_a_single_item():
    [spec] = _map(
        {"multi_modal_data": {"image": ["http://h/1.png"]}, "multi_modal_uuids": {"image": "u"}}
    )
    assert spec.uuid == "u"


def test_an_item_order_that_skips_an_item_is_refused():
    with pytest.raises(ValueError, match="exactly once"):
        _map(
            {
                "multi_modal_data": {"image": ["http://h/1.png", "http://h/2.png"]},
                "mm_item_order": [{"modality": "image", "index": 0}],
            }
        )


def test_no_media_maps_to_nothing():
    assert _map({"prompt": "x"}) == []
    assert _map({"prompt": "x", "multi_modal_data": {}}) == []


# -- Refusals ------------------------------------------------------------------


@pytest.mark.parametrize(
    "caps",
    [_caps(supports_multimodal=False), _caps(modalities=frozenset({"video"}))],
    ids=["no-multimodal", "no-image"],
)
def test_media_the_engine_does_not_declare_are_refused(caps):
    # OE-MED-1.
    with pytest.raises(ValueError, match="multimodal image inputs are not supported"):
        _map({"multi_modal_data": {"image": ["http://h/1.png"]}}, caps)


@pytest.mark.parametrize(
    "inputs, match",
    [
        ({"multi_modal_embeddings": {"image": [object()]}}, "multi_modal_embeddings"),
        ({"mm_processor_kwargs": {"fps": 2}}, "mm_processor_kwargs"),
        ({"multi_modal_data": {"image": [object()]}}, "decoded object"),
        ({"multi_modal_data": {"image": [b""]}}, "empty image data"),
        ({"multi_modal_data": {"image": ["ftp://h/1.png"]}}, "scheme 'ftp'"),
        ({"multi_modal_data": {"image": ["https:///no-host"]}}, "no host"),
        ({"multi_modal_data": {"text": ["x"]}}, "unsupported multimodal modality"),
        ({"multi_modal_data": {"image": "http://h/1.png"}}, "must list its items"),
        ({"multi_modal_data": {"image": {"http://h/1.png": 1}}}, "must list its items"),
        ({"multi_modal_data": {"image": [""]}}, "empty image reference"),
        (
            {"multi_modal_data": {"image": ["http://h/1.png"]}, "mm_item_order": ["image"]},
            "mm_item_order must list",
        ),
        (
            {"multi_modal_data": {"image": ["http://h/1.png"]}, "mm_item_order": 1},
            "mm_item_order must list",
        ),
        (
            {"multi_modal_data": {"image": ["http://h/1.png"]}, "multi_modal_uuids": ["u"]},
            "multi_modal_uuids must map",
        ),
        (
            {
                "multi_modal_data": {"image": ["http://h/1.png"] * 2},
                "multi_modal_uuids": {"image": ["u"]},
            },
            "one uuid or None for each of its 2 items",
        ),
        (
            {
                "multi_modal_data": {"image": ["http://h/1.png"]},
                "multi_modal_uuids": {"image": [7]},
            },
            "must hold strings or None",
        ),
    ],
)
def test_what_has_no_carrier_is_refused(inputs, match):
    with pytest.raises(ValueError, match=match):
        _map(inputs)


def test_inline_media_beyond_the_message_budget_are_refused(monkeypatch, media_dir):
    # The local file counts as its data URI, as long as DATA_URI.
    monkeypatch.setattr(media, "MAX_INLINE_MEDIA_BYTES", 2 * len(DATA_URI) - 1)
    with pytest.raises(ValueError, match="exceed"):
        _map({"multi_modal_data": {"image": [str(media_dir / "seashore.png"), DATA_URI]}})
    monkeypatch.setattr(media, "MAX_INLINE_MEDIA_BYTES", len(DATA_URI))
    with pytest.raises(ValueError, match="exceed"):
        _map({"multi_modal_data": {"image": [DATA_URI, PNG]}})
    monkeypatch.setattr(media, "MAX_INLINE_MEDIA_BYTES", 2 * len(DATA_URI))
    assert (
        len(_map({"multi_modal_data": {"image": [str(media_dir / "seashore.png"), DATA_URI]}})) == 2
    )
    # URLs do not count: the engine fetches them.
    assert len(_map({"multi_modal_data": {"image": ["http://h/1.png"] * 3, "audio": [PNG]}})) == 4


# -- Chat messages ------------------------------------------------------------


def _messages():
    def parts():
        # A one-pass iterator, as a validated request holds.
        yield {"type": "text", "text": "Describe"}
        yield {"type": "image_url", "image_url": {"url": "http://h/1.png"}}
        yield {"type": "image_url", "image_url": {}}
        yield {"type": "video_url", "video_url": {"url": "/data/v.mp4"}}
        yield {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}

    return [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": parts()},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": DATA_URI}}]},
        {"role": "assistant", "content": None},
    ]


@pytest.fixture
def vlm_type(monkeypatch):
    """A model type with a placeholder per modality, for parsing chat messages."""
    from tensorrt_llm.inputs.registry import (
        MULTIMODAL_PLACEHOLDER_REGISTRY,
        MultimodalPlaceholderMetadata,
    )

    name = "poc2_wt16_vlm"
    monkeypatch.setitem(
        MULTIMODAL_PLACEHOLDER_REGISTRY._multimodal_placeholder_by_model_type,
        name,
        MultimodalPlaceholderMetadata(
            placeholder_map={m: f"<{m}>" for m in ("image", "video", "audio")}
        ),
    )
    return name


def test_chat_references_by_modality_in_parse_order(vlm_type):
    from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines

    messages = _messages()
    references = media.chat_media_references(messages)
    assert references == {
        "image": ["http://h/1.png", DATA_URI],
        "video": ["/data/v.mp4"],
        "audio": ["data:audio/wav;base64,QUJD"],
    }
    # The iterator was read into a list and put back, so the messages parse again
    # and number their items as the references are listed.
    assert isinstance(messages[1]["content"], list)
    _, coroutine, _, item_order = parse_chat_messages_coroutines(
        messages, None, model_type_override=vlm_type, resolve_media=False
    )
    assert asyncio.run(coroutine) == (None, None)
    assert [(e["modality"], e["index"]) for e in item_order] == [
        ("image", 0),
        ("video", 0),
        ("audio", 0),
        ("image", 1),
    ]


def test_chat_image_embeds_are_refused():
    # OE-MED-5.
    messages = [
        {"role": "user", "content": [{"type": "image_embeds", "image_embeds": {"data": "x"}}]}
    ]
    with pytest.raises(ValueError, match="image embeddings"):
        media.chat_media_references(messages)


def test_chat_urls_that_are_not_strings_are_refused():
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": {"x": 1}}}]}
    ]
    with pytest.raises(ValueError, match="image_url must give its url as a string"):
        media.chat_media_references(messages)


def test_chat_text_only_has_no_references():
    assert media.chat_media_references([{"role": "user", "content": "Hi"}]) == {}


# -- The MediaItems the engine receives (gRPC fake) ---------------------------


def test_the_engine_receives_each_source_as_its_media_item(media_dir):
    specs = _map(
        {
            "multi_modal_data": {
                "image": ["https://example.com/a.png", DATA_URI, str(media_dir / "seashore.png")],
                "video": ["http://h/v.mp4"],
            },
            "multi_modal_uuids": {"image": ["u0", None, None]},
        }
    )
    profile = PROFILES["trtllm-like"].replace(modalities=ALL)

    async def main():
        async with FakeEngine(profile) as engine:
            client = await OpenEngineEngineClient.connect(engine.target, ready_timeout=10)
            try:
                request = EngineRequest(
                    request_id="trtllm-media", prompt_token_ids=[5, 6, 7], max_tokens=1, media=specs
                )
                async for _ in client.generate(request):
                    pass
            finally:
                await client.close()
            return engine.received

    [received] = asyncio.run(main())
    assert received.rejection is None
    items = received.request.media
    assert [item.modality for item in items] == [generation_pb2.MODALITY_IMAGE] * 3 + [
        generation_pb2.MODALITY_VIDEO
    ]
    assert [item.WhichOneof("source") for item in items] == ["url", "data_uri", "data_uri", "url"]
    assert items[0].url == "https://example.com/a.png" and items[0].uuid == "u0"
    assert items[1].data_uri == DATA_URI
    assert items[2].data_uri == DATA_URI
    assert items[3].url == "http://h/v.mp4"
