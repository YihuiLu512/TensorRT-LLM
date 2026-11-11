# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loading and matching the per-work-item expectation tables."""

import ast
import types
from pathlib import Path

import pytest
from openengine_chain import expectations as ex

pytestmark = pytest.mark.cpu_only

_HEADER = "from openengine_chain.expectations import Expectation\n"


def _table(directory, name, body):
    (directory / name).write_text(_HEADER + body)


def test_repository_tables_load():
    for entry in ex.load():
        assert entry.wt.startswith("WT-")


def test_load_reads_only_work_item_files(tmp_path):
    _table(
        tmp_path,
        "WT-11.py",
        "EXPECTATIONS = [Expectation('WT-11', '_test_openai_chat.py::test_beam*', "
        "'http_status', 'X-a', status=400)]\n",
    )
    _table(tmp_path, "WT-11-fix1.py", "EXPECTATIONS = []\n")
    _table(tmp_path, "__init__.py", "raise RuntimeError('not a table')\n")
    _table(tmp_path, "README.md", "not python\n")
    loaded = ex.load(str(tmp_path))
    assert [(e.wt, e.kind) for e in loaded] == [("WT-11", "http_status")]


@pytest.mark.parametrize("task", ["WT-20-R2", "WT-20-R3"])
def test_load_reads_integration_round_tables(tmp_path, task):
    _table(
        tmp_path,
        f"{task}.py",
        f"EXPECTATIONS = [Expectation('{task}', 'helper.py::test_parse', "
        "'no_engine_traffic', 'pure helper')]\n",
    )
    loaded = ex.load(str(tmp_path))
    assert [(e.wt, e.match, e.kind) for e in loaded] == [
        (task, "helper.py::test_parse", ex.NO_ENGINE_TRAFFIC)
    ]


@pytest.mark.parametrize(
    "name",
    ["helper.py", "WT-1.py", "wt-11.py", "WT-11-fix.py", "WT-20-R4.py", "WT-11-R2.py"],
)
def test_load_rejects_misnamed_python_files(tmp_path, name):
    _table(tmp_path, name, "EXPECTATIONS = []\n")
    with pytest.raises(ValueError, match="named WT-"):
        ex.load(str(tmp_path))


@pytest.mark.parametrize(
    "body, error",
    [
        ("EXPECTATIONS = [Expectation('WT-12', 'a.py::t', 'skip', 'r')]\n", "belongs to WT-12"),
        ("EXPECTATIONS = None\n", "must define EXPECTATIONS"),
        ("EXPECTATIONS = ['x']\n", "not an Expectation"),
    ],
)
def test_load_rejects_bad_tables(tmp_path, body, error):
    _table(tmp_path, "WT-11.py", body)
    with pytest.raises(ValueError, match=error):
        ex.load(str(tmp_path))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind="unknown"),
        dict(match="no_separator"),
        dict(reason=" "),
        dict(kind="http_status"),
        dict(kind="skip", status=400),
        dict(kind="startup_rejected"),
        dict(kind="skip", log_contains="x"),
        dict(chains=("vllm",)),
        dict(chains=()),
    ],
)
def test_invalid_entries(kwargs):
    fields = dict(wt="WT-11", match="a.py::t", kind="skip", reason="r")
    fields.update(kwargs)
    with pytest.raises(ValueError):
        ex.Expectation(**fields)


def test_match_uses_star_only_and_brackets_literally():
    e = ex.Expectation("WT-11", "_test_openai_chat.py::test_beam[*-2]", "skip", "X-a")
    assert e.matches("_test_openai_chat.py::test_beam[pytorch-2]", "trtllm")
    assert not e.matches("_test_openai_chat.py::test_beam[pytorch-3]", "trtllm")
    assert not e.matches("_test_openai_chat.py::test_beamX", "trtllm")
    glm_only = ex.Expectation("WT-06", "a.py::t", "skip", "r", chains=("glm",))
    assert not glm_only.matches("a.py::t", "trtllm")


def test_applicable_uses_the_file_name_and_rejects_conflicts():
    skip = ex.Expectation("WT-11", "a.py::t*", "skip", "r")
    traffic = ex.Expectation("WT-10", "a.py::t1", "no_engine_traffic", "r")
    found = ex.applicable([skip, traffic], "llmapi/apps/a.py::t1", "trtllm")
    assert found == [skip, traffic]
    assert ex.outcome_expectation(found) is skip
    status = ex.Expectation("WT-12", "a.py::t1", "http_status", "r", status=400)
    with pytest.raises(ValueError, match="conflicting"):
        ex.applicable([skip, status], "dir/a.py::t1", "trtllm")


def test_http_status_of():
    assert ex.http_status_of(types.SimpleNamespace(status_code=400)) == 400
    assert (
        ex.http_status_of(types.SimpleNamespace(response=types.SimpleNamespace(status_code=404)))
        == 404
    )
    assert ex.http_status_of(AssertionError("200 != 400")) is None


@pytest.mark.parametrize(
    "test, listed",
    [
        ("_test_openai_misc.py::test_version[pytorch-with_beam_search-8-32768]", True),
        ("_test_openai_misc.py::test_health[pytorch-with_beam_search-8-32768]", True),
        ("_test_openai_misc.py::test_model[pytorch-with_beam_search-8-32768]", True),
        ("_test_openai_misc.py::test_tokenize[pytorch-with_beam_search-8-32768]", True),
        ("_test_openai_misc.py::test_health_generate[pytorch-with_beam_search-8-32768]", False),
        ("_test_openai_misc.py::test_request_cancellation[pytorch-x-True]", False),
        ("_test_openai_completions.py::test_completion_with_invalid_logit_bias[pytorch]", True),
        ("_test_openai_completions.py::test_completion_with_logit_bias_effect[pytorch]", False),
        ("_test_openai_responses_entrypoint.py::test_delete_invalid_response_id", True),
        ("_test_openai_responses_entrypoint.py::test_delete", False),
        ("_test_anthropic_messages.py::test_count_tokens_grows_with_the_prompt", True),
        ("_test_anthropic_messages.py::test_count_tokens_accounts_for_the_whole_prompt", False),
    ],
)
def test_shipped_tables_list_tests_that_do_not_generate(test, listed):
    nodeid = f"llmapi/apps/{test}"
    found = ex.applicable(ex.load(), nodeid, "trtllm")
    kinds = [e.kind for e in found if e.kind == ex.NO_ENGINE_TRAFFIC]
    assert kinds == ([ex.NO_ENGINE_TRAFFIC] if listed else [])


@pytest.mark.parametrize(
    "test, listed",
    [
        (
            "_test_openai_misc.py::test_request_cancellation[pytorch-no_beam_search-8-32768-False]",
            True,
        ),
        (
            "_test_openai_misc.py::test_request_cancellation[pytorch-with_beam_search-8-32768-True]",
            True,
        ),
        (
            "_test_openai_kv_cache_contamination.py::test_no_kv_cache_contamination_under_cancel",
            True,
        ),
        (
            "_test_openai_kv_cache_contamination.py::test_no_kv_cache_contamination_number_set_probe",
            True,
        ),
        ("_test_openai_kv_cache_contamination.py::test_parse_integers_all_same_line", False),
        ("_test_openai_misc.py::test_health_generate[pytorch-no_beam_search-8-32768]", False),
        ("_test_openai_chat.py::test_chat_streaming[pytorch-disable_processpool]", False),
    ],
)
def test_shipped_tables_relax_the_count_check_only_for_tests_that_cancel_calls(test, listed):
    found = ex.applicable(ex.load(), f"llmapi/apps/{test}", "trtllm")
    assert any(e.kind == ex.CANCELS_CALLS for e in found) == listed


_T31 = "_test_openai_kv_cache_contamination.py"
_T31_PROBES = (
    "test_no_kv_cache_contamination_under_cancel",
    "test_no_kv_cache_contamination_number_set_probe",
)


def _t31_tests():
    source = Path(__file__).resolve().parents[5] / "tests/unittest/llmapi/apps" / _T31
    tree = ast.parse(source.read_text(encoding="utf-8"))
    # Exact entries cover module-level tests only; a test class would need its own entries.
    assert not [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    return [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def test_shipped_wt20_r2_table_names_every_t31_test_except_the_probes():
    # The probes send requests while the other tests of the module only parse text.
    tables = ex.load()
    tests = _t31_tests()
    names = [node.name for node in tests]
    assert set(_T31_PROBES) <= set(names)
    helpers = [node.name for node in tests if node.name not in _T31_PROBES]
    # A parametrized or fixture-using helper would get a node id the exact entries miss.
    assert not [
        node.name
        for node in tests
        if node.name in helpers and (node.decorator_list or node.args.args)
    ]
    assert sorted(e.match for e in tables if e.wt == "WT-20-R2") == sorted(
        f"{_T31}::{name}" for name in helpers
    )
    for name in names:
        nodeid = f"llmapi/apps/{_T31}::{name}"
        on_trtllm = [
            e.wt for e in ex.applicable(tables, nodeid, "trtllm") if e.kind == ex.NO_ENGINE_TRAFFIC
        ]
        assert on_trtllm == ([] if name in _T31_PROBES else ["WT-20-R2"]), name
        assert not [e for e in ex.applicable(tables, nodeid, "glm") if e.wt == "WT-20-R2"]


def test_shipped_tables_name_each_test_that_cancels_calls():
    # The relaxed count check must not reach other tests through a broad pattern.
    matches = sorted(e.match for e in ex.load() if e.kind == ex.CANCELS_CALLS)
    assert matches == [
        "_test_openai_kv_cache_contamination.py::test_no_kv_cache_contamination_number_set_probe*",
        "_test_openai_kv_cache_contamination.py::test_no_kv_cache_contamination_under_cancel*",
        "_test_openai_misc.py::test_request_cancellation[*",
    ]


@pytest.mark.parametrize(
    "test",
    [
        "_test_openai_completions.py::test_batch_completions_beam_search[pytorch-prompts0]",
        "_test_openai_chat.py::test_multiple_responses_with_beam_search[pytorch]",
        "_test_openai_reasoning.py::test_reasoning_parser[Qwen3/Qwen3-0.6B-pytorch-with_beam_search-False]",
        "_test_openai_reasoning.py::test_reasoning_parser[Qwen3/Qwen3-0.6B-pytorch-with_beam_search-True]",
        "_test_openai_misc.py::test_health_generate[pytorch-with_beam_search-8-32768]",
        "_test_openai_misc.py::test_request_cancellation[pytorch-with_beam_search-8-32768-True]",
        "_test_openai_misc.py::test_version[pytorch-with_beam_search-8-32768]",
    ],
)
def test_shipped_tables_expect_beam_servers_to_be_refused_at_startup(test):
    found = ex.outcome_expectation(ex.applicable(ex.load(), f"llmapi/apps/{test}", "trtllm"))
    assert found is not None and found.kind == ex.STARTUP_REJECTED
    assert found.log_contains == "max_beam_width > 1 is not supported"


@pytest.mark.parametrize(
    "test",
    [
        "_test_openai_misc.py::test_health_generate[pytorch-no_beam_search-8-32768]",
        "_test_openai_reasoning.py::test_reasoning_parser[Qwen3/Qwen3-0.6B-pytorch-no_beam_search-False]",
    ],
)
def test_shipped_tables_leave_servers_without_beam_alone(test):
    found = ex.applicable(ex.load(), f"llmapi/apps/{test}", "trtllm")
    assert ex.outcome_expectation(found) is None


@pytest.mark.parametrize("chain, skipped", [("trtllm", False), ("glm", True)])
def test_kv_cache_utilization_test_runs_where_the_engine_gets_load_metrics(chain, skipped):
    test = "_test_openai_prometheus.py::test_kv_cache_metrics_available_before_first_request"
    found = ex.outcome_expectation(ex.applicable(ex.load(), f"llmapi/apps/{test}", chain))
    assert (found is not None and found.kind == ex.SKIP) == skipped
