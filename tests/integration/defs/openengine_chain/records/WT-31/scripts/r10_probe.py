# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R10 (WT-31) probes on the control form T: KV transfer paths on one node, and three behaviors.

    python3 r10_probe.py --out <dir> [--hook-dir <dir holding the hook's sitecustomize.py>]

Runs in tests/integration/defs of the checkout under test, inside the image, with 2 GPUs. Every
cluster is started by the test harness itself (``setup_disagg_cluster`` of
disaggregated/test_disaggregated.py, Qwen3-0.6B, no chain switch), so the deployment is the one
the existing disagg tests use. Writes ``<out>/probe.json`` (one record per scenario, rewritten after
each scenario; a scenario that raises keeps what it collected before the exception) and copies the
worker logs of every cluster that started to ``<out>/<cluster>/`` (``kv-<backend>-<shared|split>``,
``behaviors``, ``affinity-both``; a cluster that fails to start leaves its logs where the harness
writes them, under ``<out>``). Each scenario's processes carry their own ``R10_RUN_TAG`` and are
ended after the scenario, so the next one starts on free GPUs. A failed scenario is a measured
result, so the exit code is 0 whenever probe.json is written; PROBE_DONE lists the failed ones.

* KV path (DQ-24, gate G1): cache transceiver backend DEFAULT / NIXL / UCX, with ctx and gen on
  the same GPU (``share_gpu``) and on two GPUs of the node (not run with fewer than 2 visible
  GPUs); three greedy requests each.
* UV-25: ``n=2``, ``best_of=2`` (completions) and ``n=2`` (chat) through the orchestrator.
* UV-33: a request whose first generated token is its single-token stop word.
* UV-29: ``subagent_affinity_scope: both`` with a parent-session header (the orchestrator signs
  what it forwards); the hook records what each worker derives as
  ``SchedulingParams.subagent_affinity_id``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import sys
import time
import traceback

import requests
import torch

DEFS = os.path.dirname(os.path.abspath("disaggregated"))
# What pytest provides for these modules: tests/ (pytest.ini ``pythonpath``, for test_common),
# tests/integration (the ``defs`` package) and the directory of disagg_test_utils.
for path in (
    os.path.dirname(os.path.dirname(DEFS)),
    os.path.dirname(DEFS),
    os.path.join(DEFS, "disaggregated"),
):
    sys.path.insert(0, path)

from defs.disaggregated import test_disaggregated as td  # noqa: E402

CONFIG = os.path.join(DEFS, "disaggregated", "test_configs", "disagg_config.yaml")
PROMPT = "The capital of France is"
AFFINITY_HEADER = "x-r10-parent-session"


def post(port: int, path: str, body: dict, headers: dict | None = None) -> dict:
    started = time.time()
    try:
        r = requests.post(f"http://localhost:{port}{path}", json=body, headers=headers, timeout=300)
        try:
            payload = r.json()
        except ValueError:
            payload = r.text[:2000]
        return {
            "status": r.status_code,
            "body": payload,
            "seconds": round(time.time() - started, 3),
        }
    except requests.RequestException as error:
        return {"status": None, "error": repr(error), "seconds": round(time.time() - started, 3)}


def model_id(port: int) -> str:
    r = requests.get(f"http://localhost:{port}/v1/models", timeout=60)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


def worker_logs(work_dir: str, dest: str) -> dict:
    os.makedirs(dest, exist_ok=True)
    facts = {}
    for path in glob.glob(os.path.join(work_dir, "*.log")) + glob.glob(
        os.path.join(work_dir, "*.yaml")
    ):
        shutil.copy(path, dest)
        if path.endswith(".log"):
            with open(path, errors="replace") as f:
                lines = [
                    ln.strip()[:300]
                    for ln in f
                    if any(
                        k in ln
                        for k in (
                            "NIXL",
                            "Nixl",
                            "nixl",
                            "UCX",
                            "Ucx",
                            "ucx",
                            "cache_transceiver",
                            "CacheTransceiver",
                            "transceiver",
                            "backend",
                            "cuda_ipc",
                            "subagent",
                            "R10_PROBE",
                        )
                    )
                ]
            facts[os.path.basename(path)] = lines[:40]
    return facts


def first_text(response: dict) -> str | None:
    """The first choice's text of a completions response, or None."""
    body = response.get("body")
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("text")


def tagged(tag: str) -> list[int]:
    """Live processes whose environment carries ``R10_RUN_TAG=<tag>`` (not this process)."""
    want = b"R10_RUN_TAG=" + tag.encode()
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as f:
                if want in f.read().split(b"\0"):
                    pids.append(int(entry))
        except OSError:
            continue
    return pids


def reap(tag: str) -> dict:
    """End what a scenario left behind: TERM, KILL after 10 s (listed again), as r10_shard.sh."""
    found = tagged(tag)
    for sig, pids in ((signal.SIGTERM, found), (signal.SIGKILL, None if found else [])):
        if pids is None:
            time.sleep(10)
            pids = tagged(tag)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
    if found:
        time.sleep(2)
    return {"found": len(found), "after_kill": len(tagged(tag))}


class Cluster:
    """One disagg cluster of the test harness; the worker log facts go to ``record["log_facts"]``."""

    def __init__(
        self,
        out: str,
        name: str,
        overrides: dict,
        share_gpu: bool,
        record: dict,
        env: dict | None = None,
        server_extra: dict | None = None,
    ):
        self.out, self.name, self.overrides = out, name, overrides
        self.share_gpu, self.record = share_gpu, record
        self.env, self.server_extra = env, server_extra
        self.parts = None

    def __enter__(self):
        original = td.run_disagg_server

        def run_server(config, work_dir, port=0, **kwargs):
            config.update(self.server_extra or {})
            return original(config, work_dir, port, **kwargs)

        td.run_disagg_server = run_server
        try:
            self.parts = td.setup_disagg_cluster(
                CONFIG,
                env=self.env,
                gen_env=self.env,
                ctx_env=self.env,
                cwd=self.out,
                server_start_timeout=900,
                save_log=True,
                share_gpu=self.share_gpu,
                config_overrides=self.overrides,
            )
        finally:
            td.run_disagg_server = original
        return self

    @property
    def port(self) -> int:
        return self.parts[4]

    def __exit__(self, *exc):
        _, ctx, gen, server, _, work_dir = self.parts
        td.terminate(*ctx, *gen, server)
        self.record["log_facts"] = worker_logs(work_dir, os.path.join(self.out, self.name))
        return False


def kv_scenario(record: dict, out: str, backend: str, share_gpu: bool) -> None:
    name = record["scenario"]
    transceiver = {"cache_transceiver_config": {"backend": backend}}
    record.update(backend=backend, share_gpu=share_gpu, requests=[])
    with Cluster(
        out,
        name,
        {"context_servers": transceiver, "generation_servers": transceiver},
        share_gpu,
        record,
    ) as c:
        model = model_id(c.port)
        for i in range(3):
            record["requests"].append(
                post(
                    c.port,
                    "/v1/completions",
                    {
                        "model": model,
                        "prompt": f"{PROMPT} ({i})",
                        "max_tokens": 16,
                        "temperature": 0,
                    },
                )
            )
    record["ok"] = all(r["status"] == 200 and first_text(r) for r in record["requests"])


def behaviors(record: dict, out: str) -> None:
    records = record.setdefault("items", [])
    with Cluster(out, "behaviors", {}, True, record) as c:
        model = model_id(c.port)
        base = {"model": model, "prompt": PROMPT, "max_tokens": 16, "temperature": 0}
        records.append(
            {
                "scenario": "UV-25 completions n=2",
                "response": post(c.port, "/v1/completions", dict(base, n=2, temperature=0.8)),
            }
        )
        records.append(
            {
                "scenario": "UV-25 completions best_of=2 n=1",
                "response": post(
                    c.port, "/v1/completions", dict(base, best_of=2, n=1, temperature=0.8)
                ),
            }
        )
        records.append(
            {
                "scenario": "UV-25 chat n=2",
                "response": post(
                    c.port,
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": PROMPT}],
                        "max_tokens": 16,
                        "n": 2,
                        "temperature": 0.8,
                    },
                ),
            }
        )
        first = post(c.port, "/v1/completions", dict(base, max_tokens=1, logprobs=1))
        records.append({"scenario": "UV-33 first token", "response": first})
        try:
            token = first["body"]["choices"][0]["logprobs"]["tokens"][0]
        except (KeyError, IndexError, TypeError):
            token = first_text(first)
        if token:
            records.append(
                {
                    "scenario": "UV-33 stop=[first token]",
                    "stop": token,
                    "response": post(c.port, "/v1/completions", dict(base, stop=[token])),
                }
            )
            records.append(
                {
                    "scenario": "UV-33 stop=[first token] stream",
                    "stop": token,
                    "response": post(
                        c.port, "/v1/completions", dict(base, stop=[token], stream=True)
                    ),
                }
            )
        records.append(
            {
                "scenario": "UV-33 reference no stop",
                "response": post(c.port, "/v1/completions", base),
            }
        )


def affinity(record: dict, out: str, hook_dir: str | None) -> None:
    env = dict(os.environ)
    log = os.path.join(out, "affinity-hook.jsonl")
    if hook_dir:
        env["PYTHONPATH"] = os.pathsep.join(p for p in (hook_dir, env.get("PYTHONPATH", "")) if p)
        env["R10_PROBE_LOG"] = log
    router = {"router": {"type": "conversation"}}
    with Cluster(
        out,
        "affinity-both",
        {"context_servers": router, "generation_servers": router},
        True,
        record,
        env=env,
        server_extra={
            "conversation_affinity_header_for_subagents": AFFINITY_HEADER,
            "subagent_affinity_scope": "both",
        },
    ) as c:
        model = model_id(c.port)
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 8}
        record["requests"] = []
        for _ in range(2):
            record["requests"].append(
                post(c.port, "/v1/chat/completions", body, {AFFINITY_HEADER: "parent-A"})
            )
        record["no_header"] = post(c.port, "/v1/chat/completions", body)
    if os.path.exists(log):
        with open(log) as f:
            record["hook"] = [json.loads(line) for line in f]
    else:
        record["hook"] = "no hook log (hook not loaded or never called)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--hook-dir")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = []

    base_tag = os.environ.get("R10_RUN_TAG", "r10-probe")

    def attempt(label, func, *a):
        record = {"scenario": label}
        results.append(record)
        started = time.time()
        # The workers and the server inherit the tag (the harness copies os.environ), so that
        # what this scenario leaves behind, worker children included, ends before the next one.
        tag = f"{base_tag}/{re.sub(r'[^A-Za-z0-9_.-]+', '_', label)}"
        os.environ["R10_RUN_TAG"] = tag
        try:
            func(record, *a)
        except Exception as error:  # noqa: BLE001 - a failed scenario is a probe result
            record.update(exception=repr(error), trace=traceback.format_exc())
        finally:
            os.environ["R10_RUN_TAG"] = base_tag
        record["leftovers"] = reap(tag)
        record["seconds"] = round(time.time() - started)
        print(f"PROBE {label} {record['seconds']}s", flush=True)
        with open(os.path.join(args.out, "probe.json"), "w") as f:
            json.dump(results, f, indent=1, default=str)

    gpus = torch.cuda.device_count()
    for share in (True, False):
        for backend in ("DEFAULT", "NIXL", "UCX"):
            label = f"kv-{backend.lower()}-{'shared' if share else 'split'}"
            if not share and gpus < 2:
                results.append({"scenario": label, "not_run": f"{gpus} visible GPU(s)"})
                continue
            attempt(label, kv_scenario, args.out, backend, share)
    attempt("UV-25/UV-33 behaviors", behaviors, args.out)
    attempt("UV-29 subagent_affinity_scope=both", affinity, args.out, args.hook_dir)
    failed = [
        r["scenario"] for r in results if "exception" in r or "not_run" in r or r.get("ok") is False
    ]
    print("PROBE_DONE failed:", failed, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
