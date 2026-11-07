# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared capability resolution for OpenEngine Control and Generate."""

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Collection, Mapping

# OpenEngine v1 has no fields for some capabilities, for a tokenizer
# fingerprint, or for request counters. The `extra` conventions (version 2)
# carry them as flat `trtllm.*` keys in ServerInfo.extra, ModelInfo.extra and
# LoadInfo.attributes; this module re-exports them under its own names.
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_FREQUENCY_PENALTY as SAMPLING_FREQUENCY_PENALTY_KEY,
)
from tensorrt_llm.grpc.openengine_conventions import CAP_MIN_P as SAMPLING_MIN_P_KEY
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_PRESENCE_PENALTY as SAMPLING_PRESENCE_PENALTY_KEY,
)
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_REPETITION_PENALTY as SAMPLING_REPETITION_PENALTY_KEY,
)
from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_BEST_OF as REQUEST_BEST_OF_KEY
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_REQ_CONVERSATION_ID as REQUEST_CONVERSATION_ID_KEY,
)
from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_KV_HINTS as REQUEST_KV_HINTS_KEY
from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_LOGIT_BIAS as REQUEST_LOGIT_BIAS_KEY
from tensorrt_llm.grpc.openengine_conventions import CAP_REQ_SCHEDULING as REQUEST_SCHEDULING_KEY
from tensorrt_llm.grpc.openengine_conventions import (
    CAP_REQ_THINKING_TOKEN_BUDGET as REQUEST_THINKING_TOKEN_BUDGET_KEY,
)
from tensorrt_llm.grpc.openengine_conventions import CAP_SEED_SCOPE as SAMPLING_SEED_SCOPE_KEY
from tensorrt_llm.grpc.openengine_conventions import CAP_STOP_TEXT as STOPPING_STOP_TEXT_KEY
from tensorrt_llm.grpc.openengine_conventions import (
    CONVENTIONS_VERSION,
    CONVENTIONS_VERSION_KEY,
    compute_tokenizer_fingerprint,
)
from tensorrt_llm.grpc.openengine_conventions import (
    COUNTER_GENERATE_ACCEPTED as COUNTER_GENERATE_ACCEPTED_KEY,
)
from tensorrt_llm.grpc.openengine_conventions import (
    COUNTER_GENERATE_RECEIVED as COUNTER_GENERATE_RECEIVED_KEY,
)
from tensorrt_llm.grpc.openengine_conventions import (
    FINGERPRINT_FILES as TOKENIZER_FINGERPRINT_FILES,
)
from tensorrt_llm.grpc.openengine_conventions import FP_BOS_TOKEN_ID as TOKENIZER_BOS_TOKEN_ID_KEY
from tensorrt_llm.grpc.openengine_conventions import FP_EOS_TOKEN_IDS as TOKENIZER_EOS_TOKEN_IDS_KEY
from tensorrt_llm.grpc.openengine_conventions import FP_FILES_SHA256 as TOKENIZER_FILES_SHA256_KEY
from tensorrt_llm.grpc.openengine_conventions import FP_VOCAB_SIZE as TOKENIZER_VOCAB_SIZE_KEY
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


# Upper bound of the `num_sequences` limit; a smaller batch capacity lowers it.
# Generate rejects a larger value.
MAX_NUM_SEQUENCES = 16


def tokenizer_fingerprint(
    tokenizer: Any, model_dir: str | os.PathLike[str] | None
) -> dict[str, Any]:
    """The tokenizer fingerprint keys for ``ModelInfo.extra``.

    Computed by :func:`~tensorrt_llm.grpc.openengine_conventions.compute_tokenizer_fingerprint`,
    which documents the algorithm and what it raises.
    """
    return compute_tokenizer_fingerprint(tokenizer, model_dir).to_extra()


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
