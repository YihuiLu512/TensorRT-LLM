# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared capability resolution for OpenEngine Control and Generate."""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Collection, Mapping

from tensorrt_llm.inputs.model_config import resolve_top_level_model_type
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_PLACEHOLDER_REGISTRY,
    BaseMultimodalInputProcessor,
)

from .kv_events import events_config

# Guides each grammar backend can build, keyed by the `GuidedDecoding` oneof
# field name. Control advertises from this table and Generate enforces it, so
# the two cannot disagree. llguidance has no structural-tag matcher.
GUIDE_SUPPORT_BY_BACKEND: dict[str, frozenset[str]] = {
    "xgrammar": frozenset(
        {"json_schema", "regex", "ebnf_grammar", "structural_tag", "json_object"}
    ),
    "llguidance": frozenset({"json_schema", "regex", "ebnf_grammar", "json_object"}),
}


@dataclass(frozen=True)
class MediaCapabilities:
    """Media support shared by Control and Generate."""

    placeholder_templates: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "placeholder_templates", MappingProxyType(dict(self.placeholder_templates))
        )

    def __hash__(self) -> int:
        return hash(frozenset(self.placeholder_templates.items()))

    @property
    def modalities(self) -> frozenset[str]:
        """Modalities supported by both the model and request mapper."""
        return frozenset(self.placeholder_templates)

    def placeholder(self, modality: str, item_number: int) -> str:
        """Format the model placeholder for a one-based media item."""
        return self.placeholder_templates[modality].format(item_number)


def supported_guides(backend: str | None) -> frozenset[str]:
    """Guides `backend` can build; empty when none can be.

    With no backend configured the engine never builds a grammar and drops the
    per-request guided params silently, so accepting the request would return
    unconstrained text as a success. Failing closed here also keeps Generate
    aligned with GetModelInfo, which reports guided decoding unsupported for
    that engine.
    """
    if not backend:
        return frozenset()
    return GUIDE_SUPPORT_BY_BACKEND.get(backend.lower(), frozenset())


def _loaded_model_type(llm: Any) -> str:
    """Return the top-level model type for the loaded input processor."""
    input_processor = getattr(llm, "input_processor", None)
    config = getattr(input_processor, "config", None) or getattr(llm, "_hf_model_config", None)
    if config is None:
        return ""
    return resolve_top_level_model_type(config)


def supports_image_routing(llm: Any, modalities: Collection[str]) -> bool:
    """Whether the loaded model exposes the validated image KV-event contract."""
    processor_config = getattr(getattr(llm, "input_processor", None), "config", None)
    if "image" not in modalities or resolve_top_level_model_type(processor_config) != "qwen3_vl":
        return False
    cache = getattr(llm.args, "kv_cache_config", None)
    return (
        getattr(llm.args, "cache_transceiver_config", None) is None
        and getattr(cache, "use_kv_cache_manager_v2", False) is True
        and getattr(cache, "enable_block_reuse", False) is True
        and events_config(llm) is not None
    )


def resolve_media_capabilities(
    llm: Any, implemented_modalities: Collection[str]
) -> MediaCapabilities:
    """Intersect model support with implemented request handlers.

    A live registered processor prevents registry-only false positives.
    """
    input_processor = getattr(llm, "input_processor", None)
    if getattr(getattr(llm, "args", None), "disable_mm_encoder", False):
        return MediaCapabilities({})

    model_type = _loaded_model_type(llm)
    if not model_type:
        return MediaCapabilities({})
    if not isinstance(input_processor, BaseMultimodalInputProcessor):
        registered_model_type = getattr(type(input_processor), "_registered_model_type", None)
        if registered_model_type != model_type:
            return MediaCapabilities({})
        if not callable(input_processor):
            return MediaCapabilities({})

    supported_modalities = getattr(input_processor, "supported_modalities", frozenset())
    placeholders = {
        modality: MULTIMODAL_PLACEHOLDER_REGISTRY.get_placeholder(model_type, modality)
        for modality in implemented_modalities
        if modality in supported_modalities
        and MULTIMODAL_PLACEHOLDER_REGISTRY.is_valid(model_type, modality)
    }
    # Fail startup before Control advertises a malformed template.
    for modality, template in placeholders.items():
        try:
            template.format(1)
        except (IndexError, KeyError, ValueError) as error:
            raise RuntimeError(
                f"invalid multimodal placeholder for model type '{model_type}' "
                f"and modality '{modality}': {error}"
            ) from error
    return MediaCapabilities(placeholders)


# OpenEngine v1 has no fields for some capabilities, for a tokenizer
# fingerprint, or for request counters. The `extra` conventions (version 2)
# carry them as flat `trtllm.*` keys in ServerInfo.extra, ModelInfo.extra and
# LoadInfo.attributes.
# TODO: import these keys and the fingerprint from
# tensorrt_llm/grpc/openengine_conventions.py once it exists.
CONVENTIONS_VERSION_KEY = "trtllm.conventions_version"
CONVENTIONS_VERSION = 2

SAMPLING_MIN_P_KEY = "trtllm.sampling.min_p"
SAMPLING_FREQUENCY_PENALTY_KEY = "trtllm.sampling.frequency_penalty"
SAMPLING_PRESENCE_PENALTY_KEY = "trtllm.sampling.presence_penalty"
SAMPLING_REPETITION_PENALTY_KEY = "trtllm.sampling.repetition_penalty"
SAMPLING_SEED_SCOPE_KEY = "trtllm.sampling.seed_scope"
STOPPING_STOP_TEXT_KEY = "trtllm.stopping.stop_text"

REQUEST_CONVERSATION_ID_KEY = "trtllm.request.conversation_id"
REQUEST_THINKING_TOKEN_BUDGET_KEY = "trtllm.request.thinking_token_budget"
REQUEST_LOGIT_BIAS_KEY = "trtllm.request.logit_bias"
REQUEST_KV_HINTS_KEY = "trtllm.request.kv_hints"
REQUEST_SCHEDULING_KEY = "trtllm.request.scheduling"
REQUEST_BEST_OF_KEY = "trtllm.request.best_of"

TOKENIZER_VOCAB_SIZE_KEY = "trtllm.tokenizer.vocab_size"
TOKENIZER_EOS_TOKEN_IDS_KEY = "trtllm.tokenizer.eos_token_ids"
TOKENIZER_BOS_TOKEN_ID_KEY = "trtllm.tokenizer.bos_token_id"
TOKENIZER_FILES_SHA256_KEY = "trtllm.tokenizer.files_sha256"
# Model-directory files hashed into the fingerprint, when present.
TOKENIZER_FINGERPRINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "chat_template.json",
)

COUNTER_GENERATE_RECEIVED_KEY = "trtllm.counters.generate_received"
COUNTER_GENERATE_ACCEPTED_KEY = "trtllm.counters.generate_accepted"

# Upper bound of the `num_sequences` limit Control reports; a smaller batch
# capacity lowers it. Generate does not reject a larger value.
MAX_NUM_SEQUENCES = 16


def _token_ids(value: Any) -> list[int]:
    """The integer token ids in an unset, scalar or list token-id setting."""
    values = value if isinstance(value, list) else [value]
    return [v for v in values if isinstance(v, int) and not isinstance(v, bool)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_fingerprint(
    tokenizer: Any, model_dir: str | os.PathLike[str] | None
) -> dict[str, Any]:
    """Compute the tokenizer fingerprint keys for ``ModelInfo.extra``.

    - ``vocab_size``: ``len()`` of the Hugging Face tokenizer, which counts added tokens.
    - ``eos_token_ids``: the tokenizer's ``eos_token_id`` together with the
      ``eos_token_id`` (a number or a list) of ``generation_config.json`` in
      ``model_dir``, without duplicates, ascending.
    - ``bos_token_id``: the tokenizer's ``bos_token_id`` when it is an integer, else None.
    - ``files_sha256``: sha256 of each of ``TOKENIZER_FINGERPRINT_FILES`` present in ``model_dir``.

    Args:
        tokenizer: A Hugging Face tokenizer, or a TensorRT-LLM wrapper whose
            ``tokenizer`` attribute holds one.
        model_dir: The local model directory, or None to hash no files and read
            no generation config.

    Raises:
        TypeError: The tokenizer does not report a vocabulary size.
        NotImplementedError: The tokenizer derives from Hugging Face's
            ``PreTrainedTokenizerBase`` without implementing ``__len__``.
        OSError: A file in ``model_dir`` cannot be read.
        ValueError: ``generation_config.json`` is not valid JSON.
    """
    hf_tokenizer = getattr(tokenizer, "tokenizer", None)
    if hf_tokenizer is None:
        hf_tokenizer = tokenizer
    vocab_size = len(hf_tokenizer)
    eos_token_ids = set(_token_ids(getattr(hf_tokenizer, "eos_token_id", None)))
    files_sha256 = {}
    if model_dir is not None:
        directory = Path(model_dir)
        generation_config = directory / "generation_config.json"
        if generation_config.is_file():
            config = json.loads(generation_config.read_text(encoding="utf-8"))
            if isinstance(config, dict):
                eos_token_ids.update(_token_ids(config.get("eos_token_id")))
        for name in TOKENIZER_FINGERPRINT_FILES:
            path = directory / name
            if path.is_file():
                files_sha256[name] = _sha256(path)
    bos_token_id = getattr(hf_tokenizer, "bos_token_id", None)
    if not isinstance(bos_token_id, int) or isinstance(bos_token_id, bool):
        bos_token_id = None
    return {
        TOKENIZER_VOCAB_SIZE_KEY: vocab_size,
        TOKENIZER_EOS_TOKEN_IDS_KEY: sorted(eos_token_ids),
        TOKENIZER_BOS_TOKEN_ID_KEY: bos_token_id,
        TOKENIZER_FILES_SHA256_KEY: files_sha256,
    }


class GenerateCounters:
    """Counts of ``Generate`` calls handled by one inference servicer.

    Both counts start at zero when the servicer is created and only increase.
    They are updated on the gRPC event loop, which serializes the updates. Each
    OpenEngine frontend process has its own servicer, so with several frontends
    the counts are per frontend, not per engine.
    """

    def __init__(self) -> None:
        self.received = 0
        """int: Generate calls that reached the servicer, including ones it rejected."""
        self.accepted = 0
        """int: Generate calls whose request ``generate_async`` accepted."""

    def record_received(self) -> None:
        """Count one Generate call as the servicer starts handling it, before validation."""
        self.received += 1

    def record_accepted(self) -> None:
        """Count one Generate call whose request ``generate_async`` accepted."""
        self.accepted += 1

    def attributes(self) -> dict[str, int]:
        """The counts as ``LoadInfo.attributes`` keys."""
        return {
            COUNTER_GENERATE_RECEIVED_KEY: self.received,
            COUNTER_GENERATE_ACCEPTED_KEY: self.accepted,
        }


__all__ = [
    "COUNTER_GENERATE_ACCEPTED_KEY",
    "COUNTER_GENERATE_RECEIVED_KEY",
    "CONVENTIONS_VERSION",
    "CONVENTIONS_VERSION_KEY",
    "GUIDE_SUPPORT_BY_BACKEND",
    "GenerateCounters",
    "MAX_NUM_SEQUENCES",
    "MediaCapabilities",
    "REQUEST_BEST_OF_KEY",
    "REQUEST_CONVERSATION_ID_KEY",
    "REQUEST_KV_HINTS_KEY",
    "REQUEST_LOGIT_BIAS_KEY",
    "REQUEST_SCHEDULING_KEY",
    "REQUEST_THINKING_TOKEN_BUDGET_KEY",
    "SAMPLING_FREQUENCY_PENALTY_KEY",
    "SAMPLING_MIN_P_KEY",
    "SAMPLING_PRESENCE_PENALTY_KEY",
    "SAMPLING_REPETITION_PENALTY_KEY",
    "SAMPLING_SEED_SCOPE_KEY",
    "STOPPING_STOP_TEXT_KEY",
    "TOKENIZER_BOS_TOKEN_ID_KEY",
    "TOKENIZER_EOS_TOKEN_IDS_KEY",
    "TOKENIZER_FILES_SHA256_KEY",
    "TOKENIZER_FINGERPRINT_FILES",
    "TOKENIZER_VOCAB_SIZE_KEY",
    "resolve_media_capabilities",
    "supported_guides",
    "supports_image_routing",
    "tokenizer_fingerprint",
]
