# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-NX NullWorker: the product GenerationExecutorWorker with setup_engine replaced by a
NullExecutor (design 2.3 S-NX "做法").

It must live in an importable module of its own (not the entry script): the proxy hands the class
to the MPI worker by reference (pickled module + name), and the worker process starts from a
`python -c` bootstrap that never runs the entry script (mpi_session.py). Everything the worker
needs is therefore set up here, from the environment the worker inherits: the S-NX config is the
JSON in TRTLLM_CAPKIT_SNX (MpiPoolSession forwards TRTLLM* variables).

setup_engine does not call _configure_device (no torch.cuda.set_device, no architecture check, no
CPU affinity) and does not build a checkpoint loader or PyExecutor; it sets the attributes the
real implementation sets and the worker reads later (base_worker.py setup_engine): engine,
mapping, max_seq_len, checkpoint_loader = None, _lora_manager = None.
"""

from tensorrt_llm.executor.worker import GenerationExecutorWorker

from capcommon.exitcodes import ConfigError

from .affinity import pin_all_tasks
from .null_executor import NullExecutor, SnxConfig
from .snx_product import ProductAdapter


def build_null_executor(llm_args, cfg=None):
    """A started NullExecutor for this worker: config from TRTLLM_CAPKIT_SNX unless given,
    tier table loaded from it, stream_interval cross-checked against llm_args. With
    cfg.worker_cpus every thread of this process is pinned there first (rank 0's own cores;
    serve_nx / nx_role --worker-cores), and the before / after affinity goes into meta.json."""
    cfg = cfg or SnxConfig.from_env()
    pin = pin_all_tasks(cfg.worker_cpus) if cfg.worker_cpus else None
    tt = None
    if cfg.tier_table:
        from capcommon.tiers import TierTable  # noqa: PLC0415
        tt = TierTable.load(cfg.tier_table)
    llm_s = getattr(llm_args, "stream_interval", None) if llm_args is not None else None
    max_seq_len = getattr(llm_args, "max_seq_len", None) if llm_args is not None else None
    return NullExecutor(cfg, ProductAdapter(), tier_table=tt, llm_stream_interval=llm_s,
                        max_seq_len=max_seq_len or cfg.max_seq_len, pin_info=pin)


class NullWorker(GenerationExecutorWorker):
    """GenerationExecutorWorker whose engine is a NullExecutor."""

    def setup_engine(self):
        if self.llm_args is None:
            raise ValueError("llm_args is required to set up the worker engine")
        try:
            engine = build_null_executor(self.llm_args)
        except ConfigError as e:
            raise ValueError("S-NX: %s" % e) from e
        self.mapping = self.llm_args.parallel_config.to_mapping()
        self.checkpoint_loader = None
        self.max_seq_len = engine.max_seq_len
        self.engine = engine
        self._lora_manager = None
