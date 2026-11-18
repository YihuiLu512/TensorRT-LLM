# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-TM: serve_nx --worker real / --llm-api-set, and capstubs.tm_worker.PinnedWorker with a stand-in
product base class (no tensorrt_llm on this VM; the product path runs in the E11 cluster smoke)."""

import importlib
import json
import os
import sys
import types

import pytest

from capstubs import serve_nx
from capstubs.null_executor import ENV_VAR


def test_llm_api_sets_are_written_into_the_yaml(tmp_path):
    extra = serve_nx.parse_llm_api_sets(
        ["load_format=\"dummy\"", "model_kwargs={\"num_hidden_layers\": 1}",
         "max_batch_size=2048", "print_iter_log=true"])
    args, why = serve_nx.with_stream_interval(["m"], 1, str(tmp_path), extra)
    assert why is None
    text = open(args[2]).read()
    assert text == ('stream_interval: 1\nload_format: "dummy"\n'
                    'model_kwargs: {"num_hidden_layers": 1}\nmax_batch_size: 2048\n'
                    'print_iter_log: true\n')


@pytest.mark.parametrize("item", ["x", "A=1", "stream_interval=2", "k=notjson", "=1"])
def test_llm_api_set_refusals(item):
    with pytest.raises(ValueError):
        serve_nx.parse_llm_api_sets([item])
    with pytest.raises(ValueError):
        serve_nx.parse_llm_api_sets(["k=1", "k=2"])


def test_main_refuses_llm_api_set_without_stream_interval(monkeypatch, capsys):
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert serve_nx.main(["--llm-api-set", "k=1", "--", "model"]) == 3
    assert "needs --stream-interval" in capsys.readouterr().err
    assert serve_nx.main(["--llm-api-set", "bad", "--stream-interval", "1", "--", "m"]) == 3
    with pytest.raises(SystemExit):
        serve_nx.build_parser().parse_args(["--worker", "fake"])
    assert serve_nx.build_parser().parse_args([]).worker == "null"
    assert serve_nx.build_parser().parse_args(["--worker", "product"]).worker == "product"


@pytest.fixture
def tm(monkeypatch):
    """capstubs.tm_worker imported over a stand-in tensorrt_llm.executor.worker."""
    calls = []

    class FakeBase:
        def setup_engine(self):
            calls.append(os.environ.get("TLLM_NUMA_AWARE_WORKER_AFFINITY"))

    pkg = {n: types.ModuleType(n) for n in ("tensorrt_llm", "tensorrt_llm.executor",
                                            "tensorrt_llm.executor.worker")}
    pkg["tensorrt_llm.executor.worker"].GenerationExecutorWorker = FakeBase
    for n, m in pkg.items():
        monkeypatch.setitem(sys.modules, n, m)
    monkeypatch.delitem(sys.modules, "capstubs.tm_worker", raising=False)
    mod = importlib.import_module("capstubs.tm_worker")
    yield mod, calls
    sys.modules.pop("capstubs.tm_worker", None)


def test_tm_settings(tm):
    mod, _ = tm
    assert mod.tm_settings({}) == ([], None)
    assert mod.tm_settings({mod.ENV_SNX: json.dumps({"worker_cpus": [3, 4], "out_dir": "/o"})}) \
        == ([3, 4], "/o")
    with pytest.raises(ValueError):
        mod.tm_settings({mod.ENV_SNX: json.dumps({"worker_cpus": [-1]})})


def test_pinned_worker_pins_and_sets_affinity_env(tm, tmp_path, monkeypatch):
    mod, calls = tm
    cpus = sorted(os.sched_getaffinity(0))[:1]
    monkeypatch.setenv(mod.ENV_SNX, json.dumps({"worker_cpus": cpus, "out_dir": str(tmp_path)}))
    monkeypatch.delenv(mod.ENV_AFFINITY, raising=False)
    before = os.sched_getaffinity(0)
    try:
        mod.PinnedWorker().setup_engine()
        assert calls == ["0"]  # set before the product's setup_engine ran
        doc = json.load(open(os.path.join(str(tmp_path), "tm_worker.%d.json" % os.getpid())))
        assert doc["worker_cpus"] == cpus and doc["affinity_kept"] is True
        assert doc["env_affinity_before"] is None and doc["env_affinity"] == "0"
    finally:
        for tid in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(tid), before)
            except OSError:
                pass
