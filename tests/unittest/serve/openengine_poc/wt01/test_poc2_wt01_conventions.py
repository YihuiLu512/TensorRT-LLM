# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the OpenEngine conventions module (capabilities, fingerprint, counters, priority)."""

import json
import subprocess
import sys

import pytest

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.grpc.openengine.bindings import model_pb2, server_pb2

__extra_import_path__ = [".."]
from fakes import TINY_EOS, TINY_VOCAB, tiny_hf_tokenizer, tiny_tokenizer  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def test_version_and_keys():
    assert conv.CONVENTIONS_VERSION == 2
    assert len(set(conv.BOOL_CAPABILITY_KEYS)) == len(conv.BOOL_CAPABILITY_KEYS) == 11
    assert set(conv.REQUEST_KEY_CAPABILITY) == set(conv.REQUEST_KEYS) - {conv.REQ_DETOKENIZE}
    assert set(conv.REQUEST_KEY_CAPABILITY.values()) == set(conv.REQUEST_CAPABILITY_KEYS)
    # Existing keys keep their names; new ones are namespaced and flat.
    for key in conv.REQUEST_KEYS:
        assert key in ("detokenize", "conversation_id") or key.startswith("trtllm.")


@pytest.mark.parametrize(
    "value, expected",
    [(True, True), (False, False), (1, False), ("true", False), (None, False)],
)
def test_capability_keys_must_be_booleans(value, expected):
    caps = conv.parse_extra_capabilities({conv.CAP_MIN_P: value})
    assert caps.supports(conv.CAP_MIN_P) is expected
    assert not caps.builtin


@pytest.mark.parametrize(
    "value, expected",
    [("per_request", "per_request"), ("bogus", "none"), (None, "none"), (3, "none")],
)
def test_seed_scope(value, expected):
    assert conv.parse_extra_capabilities({conv.CAP_SEED_SCOPE: value}).seed_scope == expected


def test_builtin_table_only_for_unconverted_trtllm_server():
    builtin = conv.resolve_extra_capabilities("tensorrt_llm", {}, {})
    assert builtin is conv.BUILTIN_CAPABILITIES
    assert builtin.declared == frozenset(
        conv.SAMPLING_CAPABILITY_KEYS + (conv.CAP_STOP_TEXT, conv.CAP_REQ_CONVERSATION_ID)
    )
    assert builtin.seed_scope == "none"
    # A server that reports a conventions version is read, even with no keys.
    declared = conv.resolve_extra_capabilities(
        "tensorrt_llm", {conv.CONVENTIONS_VERSION_KEY: 2.0}, {}
    )
    assert declared.declared == frozenset() and not declared.builtin
    # No other engine gets a built-in table.
    assert conv.resolve_extra_capabilities("glm5", {}, {}).declared == frozenset()


@pytest.mark.parametrize(
    "value, expected",
    [
        (2, 2),
        (2.0, 2),
        (2.5, None),
        ("12345678901234567890", 12345678901234567890),
        ("-5", -5),
        (True, None),
        # Malformed strings read as missing rather than raising.
        ("--5", None),
        ("-", None),
        ("", None),
        (" 5", None),
        ("1_000", None),
        ("+5", None),
        ("9" * 5000, None),
    ],
)
def test_as_int(value, expected):
    assert conv.as_int(value) == expected


def test_priority_round_trip():
    assert conv.encode_priority(0.0) == 0
    assert conv.encode_priority(0.5) == 500
    assert conv.encode_priority(1.0) == 1000
    assert conv.decode_priority(250) == 0.25
    for bad in (-0.1, 1.1, True):
        with pytest.raises(ValueError):
            conv.encode_priority(bad)
    with pytest.raises(ValueError):
        conv.decode_priority(1001)


def _model_dir(tmp_path, generation_eos=None):
    tiny_hf_tokenizer().save_pretrained(tmp_path)
    if generation_eos is not None:
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": generation_eos})
        )
    return tmp_path


def test_fingerprint_computation(tmp_path):
    model_dir = _model_dir(tmp_path, generation_eos=[TINY_EOS, 5])
    fp = conv.compute_tokenizer_fingerprint(tiny_tokenizer(), model_dir)
    assert fp.vocab_size == len(TINY_VOCAB)
    assert fp.eos_token_ids == (TINY_EOS, 5)
    assert fp.bos_token_id is None
    assert {"tokenizer.json", "tokenizer_config.json"} <= set(fp.files)
    # The wrapper and the bare Hugging Face tokenizer give the same fingerprint.
    assert conv.compute_tokenizer_fingerprint(tiny_hf_tokenizer(), model_dir) == fp
    # A single-number eos_token_id in generation_config.json is read too.
    _model_dir(tmp_path, generation_eos=6)
    assert conv.compute_tokenizer_fingerprint(tiny_tokenizer(), tmp_path).eos_token_ids == (1, 6)


def test_fingerprint_extra_round_trip_and_compare(tmp_path):
    fp = conv.compute_tokenizer_fingerprint(tiny_tokenizer(), _model_dir(tmp_path))
    # Struct numbers come back as floats.
    extra = json.loads(json.dumps(fp.to_extra()))
    extra[conv.FP_VOCAB_SIZE] = float(extra[conv.FP_VOCAB_SIZE])
    assert conv.TokenizerFingerprint.from_extra(extra) == fp
    assert conv.compare_tokenizer_fingerprints(fp, fp) == []

    other_files = dict(fp.files, **{"tokenizer.json": "0" * 64, "only_engine.json": "1" * 64})
    other = conv.TokenizerFingerprint(
        vocab_size=fp.vocab_size + 1,
        eos_token_ids=fp.eos_token_ids,
        bos_token_id=3,
        files_sha256=tuple(sorted(other_files.items())),
    )
    mismatches = conv.compare_tokenizer_fingerprints(fp, other)
    assert [m.item for m in mismatches] == [
        "vocab_size",
        "bos_token_id",
        "files_sha256[tokenizer.json]",
    ]
    assert "serve=" in str(mismatches[0]) and "engine=" in str(mismatches[0])


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {conv.FP_VOCAB_SIZE: 7},
        {conv.FP_VOCAB_SIZE: 7, conv.FP_EOS_TOKEN_IDS: ["x"]},
        {conv.FP_VOCAB_SIZE: 7, conv.FP_EOS_TOKEN_IDS: [1], conv.FP_BOS_TOKEN_ID: "x"},
    ],
)
def test_fingerprint_missing_or_malformed(extra):
    assert conv.TokenizerFingerprint.from_extra(extra) is None


def test_engine_counters():
    assert conv.engine_counters({}) is None
    assert conv.engine_counters(
        {conv.COUNTER_GENERATE_RECEIVED: 3.0, conv.COUNTER_GENERATE_ACCEPTED: 2.0}
    ) == (3, 2)


def test_model_capabilities_from_proto():
    server = server_pb2.ServerInfo(engine_name="glm5", schema_revision=1)
    server.extra.update({conv.CONVENTIONS_VERSION_KEY: 2, conv.SERVER_DP_RANK_TARGETING_KEY: True})
    model = model_pb2.ModelInfo(
        served_model_name="m",
        max_context_length=128,
        supports_token_ids_input=True,
        supports_text_input=False,
        supports_lora=False,
    )
    model.generation.output_logprobs.supported = True
    model.generation.output_logprobs.max_top_n = 20
    model.generation.prompt_logprobs.supported = False
    model.generation.guided_decoding.supported = True
    model.generation.guided_decoding.modes.extend(
        [model_pb2.GUIDED_DECODING_MODE_JSON_SCHEMA, model_pb2.GUIDED_DECODING_MODE_REGEX]
    )
    model.generation.supports_stop_in_output = True
    model.extra.update(
        {
            conv.CAP_MIN_P: True,
            conv.CAP_STOP_TEXT: True,
            conv.CAP_SEED_SCOPE: "batch_dependent",
            conv.MODEL_SUPPORTED_MODALITIES_KEY: ["image"],
            conv.FP_VOCAB_SIZE: 7,
            conv.FP_EOS_TOKEN_IDS: [1],
            conv.FP_BOS_TOKEN_ID: None,
            conv.FP_FILES_SHA256: {},
        }
    )
    caps = conv.model_capabilities_from_proto(server, model)
    assert caps.served_model_name == "m"
    assert caps.max_context_length == 128 and caps.max_output_tokens is None
    assert caps.supports_token_ids_input and not caps.supports_text_input
    assert caps.output_logprobs_max_top_n == 20
    assert caps.prompt_logprobs_max_top_n is None
    assert caps.guided_modes == {"json_schema", "regex"}
    assert caps.max_num_sequences == 1  # unset
    assert caps.supports_stop_in_output and not caps.supports_priority
    assert caps.supported_modalities == {"image"}
    assert caps.supports_dp_rank_targeting
    assert caps.supports(conv.CAP_MIN_P) and caps.supports(conv.CAP_STOP_TEXT)
    assert not caps.supports(conv.CAP_FREQUENCY_PENALTY)
    assert caps.extra.seed_scope == "batch_dependent"
    assert caps.tokenizer_fingerprint == conv.TokenizerFingerprint(7, (1,), None)


def test_unset_messages_mean_unsupported():
    caps = conv.model_capabilities_from_proto(
        server_pb2.ServerInfo(engine_name="other"), model_pb2.ModelInfo(served_model_name="m")
    )
    assert caps == conv.ModelCapabilities(served_model_name="m")


def test_module_does_not_import_the_serving_side():
    # The client library imports this module, so it must not add serving modules
    # to what importing the package already loads.
    code = (
        "import sys, tensorrt_llm; before = set(sys.modules); "
        "import tensorrt_llm.grpc.openengine_conventions; "
        "print(sorted(m for m in set(sys.modules) - before if m.startswith('tensorrt_llm.serve')))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert out.stdout.strip() == "[]"
