# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-TM PinnedWorker: the product GenerationExecutorWorker, unchanged except that setup_engine
first pins every thread of the worker process to the rank 0 core set (design E11: the real
PyExecutor in place of S-NX, same core layout as E9).

Like capstubs.null_worker it lives in an importable module of its own: the proxy hands the class
to the MPI worker by reference, and the worker reads its settings from the environment it
inherits (MpiPoolSession forwards TRTLLM* / TLLM* variables): worker_cpus and out_dir of the S-NX
config in TRTLLM_CAPKIT_SNX (serve_nx --worker real still writes it; only these two keys are
read here).

Before the product's setup_engine runs, TLLM_NUMA_AWARE_WORKER_AFFINITY is set to "0" when unset
(design D2: unset, the product removes an external affinity restriction in
configure_cpu_affinity). Threads the product starts afterwards (PyExecutor, await_response) inherit
the pinned set. Before / after affinity, the variable and the thread names are written to
<out_dir>/tm_worker.<pid>.json once the engine is set up.
"""

import json
import os
import time

from tensorrt_llm.executor.worker import GenerationExecutorWorker

from .affinity import pin_all_tasks, thread_names

ENV_SNX = "TRTLLM_CAPKIT_SNX"
ENV_AFFINITY = "TLLM_NUMA_AWARE_WORKER_AFFINITY"
SCHEMA = "capkit.tm_worker/1"


def tm_settings(environ=None):
    """(worker_cpus list, out_dir or None) from the S-NX config JSON in the environment."""
    env = os.environ if environ is None else environ
    try:
        doc = json.loads(env.get(ENV_SNX) or "{}")
    except ValueError:
        doc = {}
    cpus = doc.get("worker_cpus") or []
    if not isinstance(cpus, list) or any(not isinstance(c, int) or isinstance(c, bool) or c < 0
                                         for c in cpus):
        raise ValueError("%s worker_cpus must be a list of cpu numbers" % ENV_SNX)
    out = doc.get("out_dir")
    return cpus, out if isinstance(out, str) and out else None


def _write(out_dir, doc):
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tm_worker.%d.json" % os.getpid())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


class PinnedWorker(GenerationExecutorWorker):
    """GenerationExecutorWorker with its threads pinned to the S-NX config's worker_cpus."""

    def setup_engine(self):
        cpus, out_dir = tm_settings()
        aff_env_before = os.environ.get(ENV_AFFINITY)
        if aff_env_before is None:
            os.environ[ENV_AFFINITY] = "0"
        pin = pin_all_tasks(cpus) if cpus else None
        t0 = time.monotonic_ns()
        super().setup_engine()
        t1 = time.monotonic_ns()
        after = sorted(os.sched_getaffinity(0))
        _write(out_dir, {"schema": SCHEMA, "pid": os.getpid(), "worker_cpus": cpus,
                         "pin": pin, "affinity_after_setup": after,
                         "affinity_kept": (not cpus) or after == sorted(cpus),
                         "env_affinity_before": aff_env_before,
                         "env_affinity": os.environ.get(ENV_AFFINITY),
                         "setup_engine_ns": t1 - t0, "threads": thread_names()})
