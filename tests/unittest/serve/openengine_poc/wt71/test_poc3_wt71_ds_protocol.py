# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dsv4-serve accepts the fields of trtllm-serve's DisaggregatedParams its workers can receive (CD-09).

Needs a ds-engine checkout in ``TRTLLM_TEST_DS_ROOT``; skipped without one. dsv4-serve's
protocol module is imported in a child process, so its package's import-time settings stay
out of this one.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[5]
TRTLLM_PROTOCOL = REPO / "tensorrt_llm" / "serve" / "openai_protocol.py"

PROBE = """
import json, sys
from dsv4_serve.protocol import DisaggregatedParams
from pydantic import ValidationError
fields = list(DisaggregatedParams.model_fields)
sample = json.loads(sys.argv[1])
params = DisaggregatedParams.model_validate(sample)
rejected = []
for bad in json.loads(sys.argv[2]):
    try:
        DisaggregatedParams.model_validate({**sample, **bad})
    except ValidationError:
        rejected.append(bad)
print(json.dumps({"fields": fields, "dump": params.model_dump(exclude_none=True), "rejected": rejected}))
"""

SAMPLE = {
    "request_type": "context_only",
    "ctx_request_id": 7,
    "remote_tail_start": 0,
    "multimodal_positions": [0, 12],
    "multimodal_lengths": [4, 4],
    "orig_elapsed_s": 0.25,
}
INVALID = [
    {"remote_tail_start": -1},
    {"multimodal_positions": [-1, 12]},
    {"multimodal_lengths": [0, 4]},
    # trtllm-serve's cross-field check of the spans.
    {"multimodal_lengths": [4]},
    {"multimodal_lengths": None},
    {"not_a_field": 1},
]


def _trtllm_fields() -> list[str]:
    tree = ast.parse(TRTLLM_PROTOCOL.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "DisaggregatedParams":
            return [
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ]
    raise AssertionError(f"no DisaggregatedParams in {TRTLLM_PROTOCOL}")


@pytest.fixture(scope="module")
def ds_reading():
    root = os.environ.get("TRTLLM_TEST_DS_ROOT")
    if not root or not (Path(root) / "python" / "dsv4_serve" / "protocol.py").is_file():
        pytest.skip("needs a ds-engine checkout in TRTLLM_TEST_DS_ROOT")
    env = {**os.environ, "PYTHONPATH": str(Path(root) / "python")}
    done = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(SAMPLE), json.dumps(INVALID)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1])


# Only a context worker serving through an OpenEngine engine sets it, and trtllm-serve leaves it
# out of the serialized params when it is unset, so it never reaches a dsv4-serve worker.
OPENENGINE_ONLY = {"openengine_handoff_attributes"}


def test_ds_takes_every_trtllm_field_in_its_order(ds_reading):
    trtllm = _trtllm_fields()
    assert {
        "remote_tail_start",
        "multimodal_positions",
        "multimodal_lengths",
        "orig_elapsed_s",
    } <= set(trtllm)
    # dsv4-serve keeps trtllm-serve's field order, so that both serialize alike.
    assert ds_reading["fields"] == [name for name in trtllm if name not in OPENENGINE_ONLY]


def test_ds_validates_them_as_trtllm_does(ds_reading):
    assert ds_reading["dump"] == {**SAMPLE, "return_prompt_token_ids_b64": False}
    assert ds_reading["rejected"] == INVALID


COMPLETION_PROBE = """
import json
from dsv4_serve.protocol import CompletionResponse, CompletionResponseChoice, UsageInfo
def dump(**extra):
    choice = CompletionResponseChoice(index=0, text="")
    usage = UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4)
    return CompletionResponse(model="m", choices=[choice], usage=usage, **extra).model_dump()
print(json.dumps({"ids": dump(prompt_token_ids=[1, 2, 3]), "b64": dump(prompt_token_ids_b64="AQAAAA==")}))
"""


def _trtllm_class_fields(name: str) -> set[str]:
    tree = ast.parse(TRTLLM_PROTOCOL.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    raise AssertionError(f"no {name} in {TRTLLM_PROTOCOL}")


def test_ds_completion_response_leaves_out_an_absent_b64_prompt(ds_reading):
    # trtllm-serve's disaggregated server parses a context worker's completion response with its
    # own CompletionResponse, which refuses keys it does not have (prompt_token_ids_b64 among
    # them), even as null.
    root = os.environ["TRTLLM_TEST_DS_ROOT"]
    done = subprocess.run(
        [sys.executable, "-c", COMPLETION_PROBE],
        env={**os.environ, "PYTHONPATH": str(Path(root) / "python")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    dumps = json.loads(done.stdout.splitlines()[-1])
    assert "prompt_token_ids_b64" not in dumps["ids"]
    assert set(dumps["ids"]) <= _trtllm_class_fields("CompletionResponse")
    assert dumps["b64"]["prompt_token_ids_b64"] == "AQAAAA=="
