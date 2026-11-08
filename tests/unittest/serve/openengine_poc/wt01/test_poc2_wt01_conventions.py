# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of the OpenEngine conventions module (capabilities, fingerprint, counters, priority)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tensorrt_llm.grpc import openengine_conventions as conv
from tensorrt_llm.grpc.openengine.bindings import model_pb2, server_pb2

__extra_import_path__ = [".."]
from fakes import TINY_EOS, TINY_VOCAB, tiny_hf_tokenizer, tiny_tokenizer  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

NAN = float("nan")
INF = float("inf")


def test_version_and_keys():
    assert conv.CONVENTIONS_VERSION == 3
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


@pytest.mark.parametrize(
    "value, expected",
    [("per_request", "per_request"), ("bogus", "none"), (None, "none"), (3, "none")],
)
def test_seed_scope(value, expected):
    assert conv.parse_extra_capabilities({conv.CAP_SEED_SCOPE: value}).seed_scope == expected


def test_a_trtllm_engine_without_the_conventions_declares_nothing():
    """Every engine is read from what it reports; a TRT-LLM server that reports no keys declares nothing."""
    server = server_pb2.ServerInfo(engine_name="tensorrt_llm")
    caps = conv.model_capabilities_from_proto(server, model_pb2.ModelInfo())
    assert caps.extra.declared == frozenset()
    for name in conv.SAMPLING_FIELD_CAPABILITY:
        assert caps.accepted(name).accept == "none"
    # Parameters the version 2 keys do not cover are protocol fields every engine takes.
    for name in set(conv.SAMPLING_PARAM_NAMES) - set(conv.SAMPLING_FIELD_CAPABILITY):
        assert caps.accepted(name).accept == "any"


def test_version_2_keys_read_as_any_or_none():
    caps = conv.parse_extra_capabilities({conv.CAP_MIN_P: True, conv.CAP_REPETITION_PENALTY: False})
    assert caps.params is None
    assert caps.accepted("min_p") == conv.ANY_VALUE
    assert caps.accepted("repetition_penalty") == conv.NO_VALUE
    assert caps.accepted("frequency_penalty") == conv.NO_VALUE
    assert caps.accepted("temperature") == conv.ANY_VALUE


def test_params_are_read_alone_when_reported():
    caps = conv.parse_extra_capabilities(
        {
            conv.PARAMS_KEY: {
                "min_p": {"accept": "range", "min": 0, "max": 1, "default": 0.0},
                "repetition_penalty": {"accept": "values", "values": [1.0]},
            },
            # The version 2 keys are ignored next to trtllm.params.
            conv.CAP_MIN_P: False,
            conv.CAP_FREQUENCY_PENALTY: True,
        }
    )
    assert caps.accepted("min_p").accepts(0.5) and not caps.accepted("min_p").accepts(1.5)
    assert caps.accepted("min_p").default == 0.0
    assert caps.accepted("repetition_penalty").accepts(1) is True
    assert caps.accepted("repetition_penalty").accepts(1.2) is False
    # A parameter missing from trtllm.params accepts nothing.
    assert caps.accepted("frequency_penalty") == conv.NO_VALUE
    assert caps.accepted("temperature") == conv.NO_VALUE


@pytest.mark.parametrize(
    "described, inside, outside",
    [
        ({"accept": "any"}, [0.0, -5.0, 1e30, NAN, INF], []),
        ({"accept": "none"}, [], [0.0, 1.0]),
        ({"accept": "values", "values": [0.0, 2]}, [0.0, -0.0, 2.0], [1.0, 1e-300, NAN]),
        ({"accept": "range", "min": 0, "max": 1}, [0.0, 0.5, 1.0], [-1e-9, 1.0000001, NAN]),
        ({"accept": "range", "min": 1, "max": 1, "default": "generation_config"}, [1.0], [1.1]),
        (
            {"accept": "range", "min": 0, "min_exclusive": True, "max": 1, "max_exclusive": True},
            [1e-300, 0.999],
            [0.0, 1.0],
        ),
        ({"accept": "range", "min": 2}, [2.0, 1e300], [1.999, NAN, INF]),
        ({"accept": "range", "max": 2, "max_exclusive": True}, [-1e300, 1.999], [2.0, -INF]),
    ],
)
def test_accepted_values(described, inside, outside):
    accepted = conv.AcceptedValues.from_extra("p", described)
    assert all(accepted.accepts(value) for value in inside)
    assert not any(accepted.accepts(value) for value in outside)
    # What an engine reports reads back the same.
    assert conv.AcceptedValues.from_extra("p", accepted.to_extra()) == accepted


@pytest.mark.parametrize(
    "described",
    [
        None,
        [],
        {},
        {"accept": "maybe"},
        {"accept": "range"},
        {"accept": "range", "min": "0"},
        {"accept": "range", "min": True},
        {"accept": "range", "max": float("inf")},
        {"accept": "range", "min": 10**400},
        {"accept": "range", "min": 0, "min_exclusive": 1},
        {"accept": "range", "min": 2, "max": 1},
        {"accept": "range", "min": 1, "max": 1, "max_exclusive": True},
        {"accept": "values"},
        {"accept": "values", "values": []},
        {"accept": "values", "values": [1.0, "x"]},
        {"accept": "values", "values": [NAN]},
        {"accept": "values", "values": [1.0, INF]},
        {"accept": "values", "values": 1.0},
        {"accept": "any", "default": {"x": 1}},
        {"accept": "any", "default": [1.0]},
    ],
)
def test_malformed_params_name_the_parameter_and_value(described):
    with pytest.raises(conv.CapabilityFormatError, match=r"trtllm\.params\.top_p"):
        conv.parse_extra_capabilities({conv.PARAMS_KEY: {"top_p": described}})


@pytest.mark.parametrize("declared", [[], "any", 1, None])
def test_params_must_be_an_object(declared):
    with pytest.raises(conv.CapabilityFormatError, match=r"trtllm\.params is not an object"):
        conv.parse_extra_capabilities({conv.PARAMS_KEY: declared})


def test_params_from_proto():
    model = model_pb2.ModelInfo()
    model.extra.update(
        {conv.PARAMS_KEY: {"top_k": {"accept": "range", "min": 0, "default": "generation_config"}}}
    )
    caps = conv.model_capabilities_from_proto(server_pb2.ServerInfo(engine_name="e"), model)
    assert caps.accepted("top_k").accepts(5) and not caps.accepted("top_k").accepts(-1)
    assert caps.accepted("top_k").default == conv.DEFAULT_FROM_GENERATION_CONFIG
    assert caps.accepted("top_p") == conv.NO_VALUE
    # The declaration does not stop the capabilities from comparing or hashing.
    assert caps == conv.model_capabilities_from_proto(server_pb2.ServerInfo(engine_name="e"), model)
    hash(caps)


def test_malformed_params_from_proto_fail():
    model = model_pb2.ModelInfo()
    model.extra.update({conv.PARAMS_KEY: {"min_p": {"accept": "values", "values": []}}})
    with pytest.raises(conv.CapabilityFormatError, match="min_p"):
        conv.model_capabilities_from_proto(server_pb2.ServerInfo(engine_name="e"), model)


@pytest.mark.parametrize("value", [INF, NAN])
def test_a_non_finite_number_in_the_model_extra_is_a_format_error(value):
    model = model_pb2.ModelInfo()
    model.extra.update({conv.PARAMS_KEY: {"top_p": {"accept": "range", "max": value}}})
    with pytest.raises(conv.CapabilityFormatError, match=r"ModelInfo\.extra cannot be read"):
        conv.model_capabilities_from_proto(server_pb2.ServerInfo(engine_name="e"), model)


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
    # A generation_config.json that is not a JSON object adds nothing.
    (tmp_path / "generation_config.json").write_text("[6]")
    assert conv.compute_tokenizer_fingerprint(tiny_tokenizer(), tmp_path).eos_token_ids == (1,)


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


def test_fingerprint_reads_a_list_of_tokenizer_eos_ids():
    class Tokenizer:
        eos_token_id = [5, 2]
        bos_token_id = None

        def __len__(self):
            return 9

    tokenizer = Tokenizer()
    assert conv.compute_tokenizer_fingerprint(tokenizer, None).eos_token_ids == (2, 5)


def test_the_harness_copies_match_the_conventions():
    # The harness keeps its own copies so it imports without TensorRT-LLM.
    path = Path(__file__).parents[4] / "integration/defs/openengine_chain/conventions.py"
    spec = importlib.util.spec_from_file_location("_poc2_wt01_harness_conventions", path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    assert harness.ENV_COUNTER_DIR == conv.SERVE_COUNTER_DIR_ENV
    assert harness.FRONTEND_COUNTER_FIELD == conv.SERVE_COUNTER_GENERATE_SENT
    assert harness.ENGINE_COUNTER_KEY == conv.COUNTER_GENERATE_RECEIVED


def test_module_does_not_import_the_serving_side():
    # Engines import this module too, so it must not add serving modules to what
    # importing the package already loads.
    code = (
        "import sys, tensorrt_llm; before = set(sys.modules); "
        "import tensorrt_llm.grpc.openengine_conventions; "
        "print(sorted(m for m in set(sys.modules) - before if m.startswith('tensorrt_llm.serve')))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    # Importing tensorrt_llm prints a version banner first.
    assert out.stdout.strip().splitlines()[-1] == "[]"
