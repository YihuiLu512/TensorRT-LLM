# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The two legs' GenerateRequests (SV-DSG-7) and the request dump (CS-19), on CPU.

NT-021 (DG-85): through the orchestrator, the context and the generation
engine receive the same ``GenerateRequest`` but for the keys each leg alone
carries (EX-DSG-7) and ``request_id``; the context leg's ``max_tokens`` is the
client's. The worker frontends' request dumps (``TRTLLM_OPENENGINE_REQUEST_DUMP_DIR``)
hold what the engines received, the prompt token ids as their count and hash,
and pair up by ``disagg_request_id``. Unset, nothing is written.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest
import yaml
from google.protobuf import json_format

from tensorrt_llm.grpc.openengine.bindings import generation_pb2
from tensorrt_llm.grpc.openengine_conventions import (
    REQ_DISAGG_REQUEST_ID,
    REQ_ORIG_ELAPSED_S,
    REQ_REQUEST_TYPE,
    REQ_SCHEDULE_STYLE,
    SERVE_REQUEST_DUMP_DIR_ENV,
)
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openengine_poc.engine_client_impl import dump_generate_request
from tensorrt_llm.serve.openengine_poc.facade import OpenEngineLLM

__extra_import_path__ = ["..", "~/tests/unittest/grpc/openengine_client"]
from cpu_disagg_chain import (  # noqa: E402
    CpuDisaggChain,
    _chain_profile,
    _fingerprint,
    make_model_dir,
)
from fake_openengine import FakeEngine, disagg_profile  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

PROMPT = "He llo world"
# EX-DSG-7: the keys only one leg carries.
PREFILL_ONLY_EXTRA = (REQ_REQUEST_TYPE, REQ_DISAGG_REQUEST_ID, REQ_SCHEDULE_STYLE)
DECODE_ONLY_EXTRA = (REQ_ORIG_ELAPSED_S,)


@pytest.fixture(scope="module", params=["trtllm", "ds"])
def chain(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"wt52-legs-{request.param}")
    config = directory / "frontend.yaml"
    config.write_text(yaml.safe_dump({"cache_transceiver_config": {"backend": "DEFAULT"}}))
    args = ["--config", str(config)]
    with CpuDisaggChain(
        directory, request.param, frontend_args={"context": args, "generation": args}
    ) as chain:
        yield chain


def _comparable(request: generation_pb2.GenerateRequest, leg_keys: tuple[str, ...]) -> dict:
    """``request`` as a dict without ``request_id`` and the keys only its leg carries."""
    data = json_format.MessageToDict(request, preserving_proto_field_name=True)
    data.pop("request_id")
    extra = data.get("extra", {})
    for key in leg_keys:
        extra.pop(key, None)
    if not extra:
        data.pop("extra", None)
    kv = data.get("kv", {})
    kv.pop("session", None)
    if not kv:
        data.pop("kv", None)
    return data


def _request_bodies(shape: str) -> list[dict]:
    bodies = [
        {"max_tokens": 3},
        {"max_tokens": 5, "temperature": 0.5, "top_p": 0.9, "seed": 7, "stop": ["!"]},
        {"max_tokens": 4, "stop": ["x"], "ignore_eos": True},
    ]
    if shape == "trtllm":
        bodies += [
            {"max_tokens": 4, "logprobs": 2, "top_k": 3, "min_p": 0.1, "min_tokens": 2},
            {"max_tokens": 4, "frequency_penalty": 0.5, "presence_penalty": 0.5},
        ]
    return bodies


def test_ctx_and_gen_generate_requests_match(chain):
    """NT-021: both legs carry the same request; only the single-leg keys and request_id differ."""
    for body in _request_bodies(chain.shape):
        ctx_seen, gen_seen = len(chain.ctx.engine.requests), len(chain.gen.engine.requests)
        response = chain.complete(prompt=PROMPT, **body)
        assert response.status_code == 200, (body, response.text)
        [prefill] = chain.ctx.engine.requests[ctx_seen:]
        [decode] = chain.gen.engine.requests[gen_seen:]
        assert prefill.extra[REQ_REQUEST_TYPE] == "context_only"
        assert decode.kv.HasField("session")
        # The context leg's max_tokens is the client's: the engine enforces one token itself.
        assert prefill.stopping.max_tokens == body["max_tokens"]
        assert _comparable(prefill, PREFILL_ONLY_EXTRA) == _comparable(decode, DECODE_ONLY_EXTRA)
        assert prefill.request_id != decode.request_id


def _dump(worker, request: generation_pb2.GenerateRequest) -> dict:
    return json.loads((worker.dump_dir / f"{request.request_id}.json").read_text())


def test_the_dumps_hold_what_the_engines_received_and_pair_up(chain):
    """CS-19: one file per GenerateRequest, matching the engine's record, paired by disagg_request_id."""
    ctx_seen, gen_seen = len(chain.ctx.engine.requests), len(chain.gen.engine.requests)
    response = chain.complete(prompt=PROMPT, max_tokens=4, seed=3)
    assert response.status_code == 200, response.text
    [prefill] = chain.ctx.engine.requests[ctx_seen:]
    [decode] = chain.gen.engine.requests[gen_seen:]
    ctx_dump, gen_dump = _dump(chain.ctx, prefill), _dump(chain.gen, decode)
    for dump, request in ((ctx_dump, prefill), (gen_dump, decode)):
        ids = list(request.token_ids.ids)
        assert dump.pop("token_ids") == {
            "count": len(ids),
            "length": len(json.dumps(ids, separators=(",", ":"))),
            "sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        }
        expected = json_format.MessageToDict(request, preserving_proto_field_name=True)
        expected.pop("token_ids")
        assert dump == expected
    disagg_request_id = ctx_dump["extra"][REQ_DISAGG_REQUEST_ID]
    assert gen_dump["kv"]["session"]["attributes_struct"]["disagg_request_id"] == disagg_request_id
    # The dumps alone tell that the legs match (CH-7 on GPU reads them this way).
    for dump, keys in ((ctx_dump, PREFILL_ONLY_EXTRA), (gen_dump, DECODE_ONLY_EXTRA)):
        dump.pop("request_id")
        for key in keys:
            dump.get("extra", {}).pop(key, None)
        dump.get("kv", {}).pop("session", None)
        for name in ("extra", "kv"):
            if name in dump and not dump[name]:
                dump.pop(name)
    assert ctx_dump == gen_dump


def _request(request_id: str, media: bytes | None = None) -> generation_pb2.GenerateRequest:
    message = generation_pb2.GenerateRequest(
        request_id=request_id, model="m", token_ids=generation_pb2.TokenIds(ids=[3, 1, 4])
    )
    if media is not None:
        message.media.add(modality=generation_pb2.MODALITY_IMAGE, raw_bytes=media)
    return message


def test_nothing_is_written_without_the_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(SERVE_REQUEST_DUMP_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert dump_generate_request(_request("r1")) is None
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setenv(SERVE_REQUEST_DUMP_DIR_ENV, "")
    assert dump_generate_request(_request("r1")) is None
    assert list(tmp_path.iterdir()) == []


def test_a_dump_hides_the_prompt_and_raw_media(tmp_path, monkeypatch):
    directory = tmp_path / "dumps"
    monkeypatch.setenv(SERVE_REQUEST_DUMP_DIR_ENV, str(directory))
    path = dump_generate_request(_request("r-1.a", media=b"\x89PNG"))
    assert path == directory / "r-1.a.json"
    dump = json.loads(path.read_text())
    assert dump["token_ids"]["count"] == 3
    assert "3, 1, 4" not in path.read_text() and "[3,1,4]" not in path.read_text()
    assert dump["media"][0]["raw_bytes"] == {
        "length": 4,
        "sha256": hashlib.sha256(b"\x89PNG").hexdigest(),
    }
    assert sorted(p.name for p in directory.iterdir()) == ["r-1.a.json"]


@pytest.mark.parametrize("request_id", ["../escape", "a/b", ".hidden", "x" * 200, ""])
def test_a_request_id_that_is_not_a_file_name_is_hashed(tmp_path, monkeypatch, request_id):
    directory = tmp_path / "dumps"
    monkeypatch.setenv(SERVE_REQUEST_DUMP_DIR_ENV, str(directory))
    path = dump_generate_request(_request(request_id))
    assert path == directory / f"{hashlib.sha256(request_id.encode()).hexdigest()}.json"
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == [path]


def test_a_dump_that_cannot_be_written_does_not_stop_the_request(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv(SERVE_REQUEST_DUMP_DIR_ENV, str(blocker / "sub"))
    with pytest.raises(OSError):
        dump_generate_request(_request("r1"))
    model = make_model_dir(tmp_path / "model")
    profile = _chain_profile(disagg_profile("trtllm-like"), _fingerprint(model))

    async def main() -> None:
        async with FakeEngine(profile) as fake:
            facade = OpenEngineLLM.from_llm_args(
                {"model": str(model)}, fake.target, ready_timeout=30
            )
            await facade.start()
            try:
                output = await facade.generate_async(
                    {"prompt_token_ids": [2, 3]}, SamplingParams(max_tokens=2)
                )
            finally:
                await facade.aclose()
        assert output.outputs[0].finish_reason is not None
        assert len(fake.requests) == 1

    asyncio.run(main())
    assert blocker.read_text() == ""
