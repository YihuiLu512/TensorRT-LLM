# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract v10: the disaggregated-serving types, the handoff codec signatures and the counters."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import os
import pickle
import typing

import pytest

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.serve.openengine_poc import contracts
from tensorrt_llm.serve.openengine_poc import engine_client as ec

__extra_import_path__ = [".."]
from fakes import FakeEngineClient, kv_connector_like, prefill_ready_script  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _session() -> ec.KvSessionSpec:
    return ec.KvSessionSpec(
        session_id="7",
        transfer_backend="NIXL",
        dp_rank=1,
        endpoints=(ec.KvEndpoint("10.0.0.1", 5000, "tcp"),),
        attributes={
            "ctx_info_endpoint": "tcp://10.0.0.1:5000",
            "disagg_request_id": str(1 << 41),
            "ctx_usage": {"prompt_tokens": 4.0},
        },
    )


def test_contract_version():
    assert contracts.POC2_CONTRACT_VERSION == 10


def test_new_request_fields_default_to_an_aggregated_call():
    request = ec.EngineRequest(request_id="r", prompt_token_ids=[1], max_tokens=4)
    assert request.request_type is None
    assert request.kv_session is None
    assert typing.get_args(ec.DisaggRequestType) == ("context_only", "context_and_generation")


def test_engine_info_defaults_report_no_kv_connector():
    info = ec.EngineInfo(engine_name="e", engine_version="1", instance_id="i", schema_revision=1)
    assert info.engine_role == "unspecified"
    assert info.kv_connector is None
    assert typing.get_args(ec.EngineRole) == ("unspecified", "aggregated", "prefill", "decode")
    connector = ec.KvConnectorInfo()
    assert not (connector.enabled or connector.supports_remote_prefill)
    assert connector.local_endpoints == () and connector.schema_version is None


def test_prefill_ready_is_an_engine_event():
    assert ec.PrefillReady in typing.get_args(ec.EngineEvent)
    event = ec.PrefillReady(output_index=0, kv_session=_session())
    assert event.usage is None
    for name in ("KvEndpoint", "KvSessionSpec", "KvConnectorInfo", "PrefillReady", "EngineRole"):
        assert name in ec.__all__


def test_a_prefill_ready_event_crosses_into_a_postprocessing_process():
    event = ec.PrefillReady(output_index=0, kv_session=_session(), usage=ec.Usage(4, 1, 0))
    message = contracts.PPEvent(request_id="r", event=event)
    contracts.check_plain_data(message, path="event")
    assert pickle.loads(pickle.dumps(message)) == message


def test_frontend_launch_config_worker_fields():
    config = contracts.FrontendLaunchConfig(
        host="h", port=0, llm_args={}, endpoint="e:1", internal_disagg_auth_key="secret-key"
    )
    assert config.server_role is None and config.disagg_cluster_config is None
    assert config.internal_disagg_auth_key == "secret-key"
    assert "secret-key" not in repr(config)


def test_the_handoff_codec_signatures():
    assert contracts.HANDOFF_MODULE == "tensorrt_llm.serve.openengine_poc.handoff"
    # The codec is not a slot.
    assert "decode_kv_session" not in contracts.slots()
    decode = inspect.signature(contracts.DecodeKvSession.__call__)
    encode = inspect.signature(contracts.EncodeKvSession.__call__)
    assert list(decode.parameters) == ["self", "session"]
    assert list(encode.parameters) == ["self", "params", "handoff_attributes", "transfer_backend"]


@pytest.mark.parametrize("empty", [{}, None])
def test_no_handoff_attributes_dump_to_none(empty):
    assert contracts.dump_handoff_attributes(empty) is None
    assert contracts.load_handoff_attributes(None) == {}


def test_deeply_nested_handoff_attributes_are_refused():
    nested: dict = {}
    for _ in range(100_000):
        nested = {"x": nested}
    with pytest.raises(ValueError):
        contracts.dump_handoff_attributes(nested)


def test_handoff_attributes_round_trip_in_a_canonical_form():
    attributes = {"z": [1, 2.5, None], "a": {"nested": "x"}, "trtllm_decode_preparation": "{}"}
    text = contracts.dump_handoff_attributes(attributes)
    assert text == json.dumps(attributes, sort_keys=True, separators=(",", ":"))
    assert contracts.dump_handoff_attributes(dict(reversed(list(attributes.items())))) == text
    assert contracts.load_handoff_attributes(text) == attributes


@pytest.mark.parametrize(
    "bad",
    [
        {"x": float("nan")},
        {"x": float("inf")},
        {"x": object()},
        {1: "x"},
        {None: "x"},
        {"x": {1: "y"}},
        {"x": [{True: "y"}]},
    ],
)
def test_handoff_attributes_must_be_json_data(bad):
    with pytest.raises(ValueError):
        contracts.dump_handoff_attributes(bad)


@pytest.mark.parametrize(
    "text",
    [
        "[1]",
        '"x"',
        "1",
        "null",
        "not json",
        '{"x":NaN}',
        '{"x":-Infinity}',
        '{"x":1e999}',
        '{"x":[-1e999]}',
        "[" * 100_000,
    ],
)
def test_handoff_attributes_must_load_as_an_object(text):
    with pytest.raises(ValueError):
        contracts.load_handoff_attributes(text)


# -- Serving-side counters ------------------------------------------------------


def test_counter_add_counts_named_counters(tmp_path):
    counter = contracts.GenerateCounter(tmp_path)
    assert counter.counts == {"generate_sent": 0}
    assert counter.add("prefill_ready_received") == 1
    assert counter.add("prefill_ready_received", 2) == 3
    counter.increment()
    assert counter.counts == {"generate_sent": 1, "prefill_ready_received": 3}


@pytest.mark.parametrize(
    "key",
    ["generate_sent", "generate_sent.context_only", "bogus", "trtllm.counters.generate_received"],
)
def test_counter_add_refuses_other_keys(key):
    with pytest.raises(ValueError):
        contracts.GenerateCounter(None).add(key)


@pytest.mark.parametrize("count", [-1, 1.0, True, "1"])
def test_counters_refuse_a_count_that_is_not_a_non_negative_int(count):
    counter = contracts.GenerateCounter(None)
    with pytest.raises(ValueError):
        counter.add("http_rejected", count)
    with pytest.raises(ValueError):
        counter.increment(count)
    assert counter.counts == {"generate_sent": 0}


def test_increment_counts_a_bucket_with_the_total():
    counter = contracts.GenerateCounter(None)
    assert counter.increment(2, bucket="context_only") == 2
    assert counter.increment() == 3
    assert counter.counts == {"generate_sent": 3, "generate_sent.context_only": 2}
    with pytest.raises(ValueError):
        counter.increment(bucket="bogus")
    assert counter.value == 3


def test_read_serve_counters_adds_up_open_and_closed_frontends(tmp_path):
    open_counter = contracts.GenerateCounter(tmp_path)
    open_counter.start()
    try:
        open_counter.increment(2, bucket="context_only")
        # A second frontend that has closed its counter, as another process leaves it.
        other = os.getpid() + 1
        record = {"pid": other, "generate_sent": 5, "http_rejected": 1}
        (tmp_path / f"{other}.json").write_text(json.dumps(record))
        totals = contracts.read_serve_counters(tmp_path)
        assert set(totals) == set(conv.SERVE_COUNTERS)
        assert totals["generate_sent"] == 7
        assert totals["generate_sent.context_only"] == 2
        assert totals["http_rejected"] == 1
        assert totals["prefill_ready_received"] == 0
        # read_generate_sent reads generate_sent alone.
        assert contracts.read_generate_sent(tmp_path) == 7
        by_pid = contracts.read_serve_counters_by_pid(tmp_path)
        assert by_pid[other]["generate_sent"] == 5
    finally:
        open_counter.close()
    # Closed, the counter leaves its final counts in its file.
    assert contracts.read_serve_counters(tmp_path)["generate_sent.context_only"] == 2


def test_a_counter_closed_before_add_rewrites_its_file(tmp_path):
    counter = contracts.GenerateCounter(tmp_path)
    counter.start()
    counter.close()
    counter.add("http_received.aggregated")
    assert contracts.read_serve_counters(tmp_path)["http_received.aggregated"] == 1


def test_a_record_with_a_counter_that_is_not_an_integer_is_refused(tmp_path):
    (tmp_path / "1.json").write_text(
        json.dumps({"pid": 1, "generate_sent": 0, "http_rejected": "1"})
    )
    with pytest.raises(ValueError):
        contracts.read_serve_counters(tmp_path)


# -- The in-process fake ---------------------------------------------------------


def _run(client: FakeEngineClient, request: ec.EngineRequest) -> list[ec.EngineEvent]:
    async def collect() -> list[ec.EngineEvent]:
        return [event async for event in client.generate(request)]

    return asyncio.run(collect())


def test_the_fake_rejects_a_prefill_call_for_several_sequences():
    client = FakeEngineClient(script=prefill_ready_script)
    request = ec.EngineRequest(
        request_id="r",
        prompt_token_ids=[1],
        max_tokens=4,
        num_sequences=2,
        request_type="context_only",
    )
    with pytest.raises(ValueError):
        _run(client, request)


def test_the_fake_ends_a_prefill_call_with_prefill_ready():
    client = FakeEngineClient(script=prefill_ready_script)
    client.info = dataclasses.replace(client.info, kv_connector=kv_connector_like())
    request = ec.EngineRequest(
        request_id="r",
        prompt_token_ids=[1, 2],
        max_tokens=4,
        request_type="context_only",
        extra={"disagg_request_id": str(1 << 41)},
    )
    events = _run(client, request)
    assert isinstance(events[-1], ec.PrefillReady)
    assert events[-1].kv_session.attributes["disagg_request_id"] == str(1 << 41)
    assert client.active == set() and client.cancelled == []
    decode = ec.EngineRequest(
        request_id="g", prompt_token_ids=[1, 2], max_tokens=4, kv_session=events[-1].kv_session
    )
    assert isinstance(_run(client, decode)[-1], ec.OutputFinished)
    attributes = asyncio.run(client.load()).attributes
    assert attributes[conv.COUNTER_PREFILL_READY_SENT] == 1
    buckets = {
        bucket: attributes[key] for bucket, key in conv.COUNTER_GENERATE_ACCEPTED_BUCKETS.items()
    }
    assert buckets == {"context_only": 1, "generation_only": 1, "aggregated": 0}
    assert sum(buckets.values()) == attributes[conv.COUNTER_GENERATE_ACCEPTED]
    assert client.server_info().kv_connector.supports_remote_prefill
