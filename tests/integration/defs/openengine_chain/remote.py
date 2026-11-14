# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RemoteOpenAIServer``'s entry into the chain."""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from typing import IO, Any

import yaml

from . import conventions as cv
from .launch import (
    DEFAULT_ENGINE_READY_TIMEOUT_S,
    TERMINATE_GRACE_S,
    ChainError,
    ServeHandle,
    _address,
    _results_dir,
    _stop_waits,
    launch_serve_command,
    launch_serve_commands,
    pick_port,
    reserve_ports,
    resolve_chain,
)
from .split import (
    WORKER_ROLES,
    ChainUnrunnableError,
    disagg_server_config,
    disagg_worker_commands,
    is_disagg_worker,
    option_value,
    parse_serve_command,
    worker_gpus,
)

# The address trtllm-serve listens on without --host.
DEFAULT_SERVE_HOST = "localhost"


def resolve_topology() -> str | None:
    """``"disagg"`` or ``None`` from ``TRTLLM_TEST_OPENENGINE_TOPOLOGY``."""
    value = os.environ.get(cv.ENV_TOPOLOGY, "")
    if value == "":
        return None
    if value != cv.TOPOLOGY_DISAGG:
        raise ChainError(f"{cv.ENV_TOPOLOGY}={value!r}; expected unset or {cv.TOPOLOGY_DISAGG!r}")
    return value


def launch_for_remote_server(
    launch_cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    llmapi_launch: bool,
    role: Any,
    sink: IO[Any] | int | None,
) -> ServeHandle | DisaggTopology | None:
    """Start ``launch_cmd`` for ``RemoteOpenAIServer``; the caller waits for ``/health``.

    The caller calls this with a chain or the topology selected. With
    ``TRTLLM_TEST_OPENENGINE_TOPOLOGY=disagg``, an aggregated server (no
    ``role``, and not a disaggregated worker by
    :func:`~openengine_chain.split.is_disagg_worker`) starts as a
    :class:`DisaggTopology`; with the topology set and no chain, a
    disaggregated worker or a ``trtllm-llmapi-launch`` server is left to the
    caller: the function returns ``None``. With a chain, a
    ``trtllm-llmapi-launch`` server is refused, and so is the topology on the
    glm chain; a server that is not moved starts as an engine plus a front end.
    """
    chain = resolve_chain()
    topology = resolve_topology()
    if topology is not None and chain == cv.CHAIN_NONE and (role is not None or llmapi_launch):
        return None
    if llmapi_launch:
        raise ChainError("the OpenEngine chain does not support trtllm-llmapi-launch")
    if topology is not None:
        if chain == cv.CHAIN_GLM:
            raise ChainUnrunnableError(
                f"not runnable: {cv.ENV_TOPOLOGY}={topology} on the glm chain, whose engine "
                "serves no disaggregated worker (it refuses kv.session)"
            )
        strict = chain != cv.CHAIN_NONE
        if role is None and not is_disagg_worker(
            parse_serve_command(launch_cmd, strict).args, strict
        ):
            return launch_disagg_topology(launch_cmd, env=env, sink=sink, chain=chain)
        if chain == cv.CHAIN_NONE:
            return None
    return launch_serve_command(launch_cmd, env=env, sink=sink)


def visible_gpus(env: Mapping[str, str]) -> list[str]:
    """The GPUs a process started with ``env`` sees, as ``CUDA_VISIBLE_DEVICES`` entries."""
    given = env.get("CUDA_VISIBLE_DEVICES")
    if given is not None:
        return [gpu.strip() for gpu in given.split(",") if gpu.strip()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ChainError(f"cannot list the GPUs with nvidia-smi: {error}") from error
    return [line.strip() for line in out.splitlines() if line.strip()]


class DisaggTopology:
    """An aggregated test's server started as two workers and a disaggregated server.

    Stands in for a :class:`~openengine_chain.launch.ServeHandle` in
    ``RemoteOpenAIServer``: the disaggregated server listens on the address
    the test asked for and is the ``frontend``. ``ctx_handle`` and
    ``gen_handle`` are the two workers: each an engine plus a front end when a
    chain is selected, one in-process server when none is.
    """

    def __init__(
        self,
        *,
        chain: str,
        host: str,
        port: int | None,
        ctx_handle: ServeHandle,
        gen_handle: ServeHandle,
        server: subprocess.Popen,
        workdir: str,
    ) -> None:
        self.chain = chain
        self.host = host
        self.port = port
        self.ctx_handle = ctx_handle
        self.gen_handle = gen_handle
        self.server = server
        self.workdir = workdir
        # The disaggregated server outliving SIGKILL. They are also added to the context
        # worker's handle, whose teardown errors the pytest plugin reports when a chain is
        # selected; the workers' handles report their own processes.
        self.teardown_errors: list[str] = []
        self._stopped = False

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes the topology does not have, such as a ServeHandle's
        # engine_endpoint or counter_dir: those are per worker.
        raise AttributeError(
            f"{type(self).__name__} has no {name!r}: a moved test has two workers; read it "
            "from ctx_handle or gen_handle"
        )

    @property
    def frontend(self) -> subprocess.Popen:
        return self.server

    @property
    def handles(self) -> tuple[ServeHandle, ServeHandle]:
        return (self.ctx_handle, self.gen_handle)

    def poll(self) -> int | None:
        """The disaggregated server's exit code; if only a worker has exited, its code, or 1 for 0."""
        rc = self.server.poll()
        if rc is not None:
            return rc
        for handle in self.handles:
            worker_rc = handle.poll()
            if worker_rc is not None:
                return worker_rc or 1
        return None

    def url_for(self, *parts: str) -> str:
        return f"http://{self.host}:{self.port}/" + "/".join(parts)

    def frontend_port_problem(self) -> str | None:
        """Always ``None``: the disaggregated server is not restarted on another port."""
        return None

    def restart_frontend(self, port: int) -> None:
        raise ChainError("the disaggregated server of a moved test is not restarted")

    def terminate(self, grace: float = TERMINATE_GRACE_S) -> None:
        """Stop the disaggregated server, then the generation and the context worker."""
        if self._stopped:
            return
        self._stopped = True
        try:
            _stop_waits(self.server, grace)
            if self.server.poll() is None:
                error = f"disaggregated server pid {self.server.pid} still running after SIGKILL"
                self.teardown_errors.append(error)
                self.ctx_handle.teardown_errors.append(error)
        finally:
            try:
                self.gen_handle.terminate(grace)
            finally:
                self.ctx_handle.terminate(grace)


def launch_disagg_topology(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    sink: IO[Any] | int | None,
    chain: str,
    server_start_timeout: float = DEFAULT_ENGINE_READY_TIMEOUT_S,
) -> DisaggTopology:
    """Start an aggregated ``trtllm-serve`` command line as two workers and a disaggregated server.

    :func:`~openengine_chain.split.disagg_worker_commands` takes the command
    line apart; the workers listen on the command line's ``--host`` at ports
    the harness picks, get GPUs by :func:`~openengine_chain.split.worker_gpus`,
    and load at the same time. Once both have started, the disaggregated
    server (never through the chain) listens on the command line's ``--host``
    and ``--port`` and reaches the workers at the address clients use for
    that host. On a failure everything started is stopped before the error
    propagates.
    """
    strict = chain != cv.CHAIN_NONE
    command = parse_serve_command(cmd, strict)
    host, port = _address(command.args)
    listen_host = option_value(command.args, "--host", strict) or DEFAULT_SERVE_HOST
    workdir = _results_dir()
    auth_key = secrets.token_hex(32)
    # The disaggregated server binds its port only after the workers have started;
    # while they start, a socket bound to the port, if it can be bound, keeps it from
    # other processes. The socket is closed just before the server starts.
    reserve_ports(port)
    holder = _hold_port(port)
    try:
        ports = {"context": pick_port(port)}
        ports["generation"] = pick_port(port, ports["context"])
        workers = disagg_worker_commands(
            cmd, ports=ports, host=listen_host, auth_key=auth_key, out_dir=workdir, strict=strict
        )
        gpus = worker_gpus(visible_gpus(env), workers.gpus_per_worker, workers.share_gpu)
        ctx_handle, gen_handle = launch_serve_commands(
            [workers.commands[role] for role in WORKER_ROLES],
            labels=[f"the {role} worker" for role in WORKER_ROLES],
            envs=[{**env, "CUDA_VISIBLE_DEVICES": gpus[role]} for role in WORKER_ROLES],
            sinks=[sink, sink],
            chain=chain,
            kv_aware_router=False,
            role_indices=[0, 0],
        )
        try:
            config = disagg_server_config(
                host=listen_host,
                port=port,
                workers={role: f"{host}:{ports[role]}" for role in WORKER_ROLES},
                auth_key=auth_key,
            )
            fd, config_path = tempfile.mkstemp(prefix="disagg-server-", suffix=".yaml", dir=workdir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f)
            # The executable part without a "serve" subcommand.
            executable = list(
                command.prefix[:-1] if command.prefix[-1] == "serve" else command.prefix
            )
            server_cmd = executable + [
                "disaggregated",
                "-c",
                config_path,
                "--server_start_timeout",
                str(int(server_start_timeout)),
            ]
            server_env = {k: v for k, v in env.items() if k != "CUDA_VISIBLE_DEVICES"}
            if holder is not None:
                holder.close()
            server = subprocess.Popen(server_cmd, env=server_env, stdout=sink, stderr=sink)
        except BaseException:
            try:
                gen_handle.terminate()
            finally:
                ctx_handle.terminate()
            raise
        return DisaggTopology(
            chain=chain,
            host=host,
            port=port,
            ctx_handle=ctx_handle,
            gen_handle=gen_handle,
            server=server,
            workdir=workdir,
        )
    finally:
        if holder is not None:
            holder.close()


def _hold_port(port: int) -> socket.socket | None:
    """A socket bound to ``port`` on all interfaces, or ``None`` if ``port`` cannot be bound now."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
    except OSError:
        sock.close()
        return None
    return sock
