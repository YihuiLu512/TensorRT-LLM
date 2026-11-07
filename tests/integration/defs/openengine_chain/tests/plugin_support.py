# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A chain server stand-in for the plugin tests: counters and topology are plain attributes."""

from openengine_chain import launch, proof


class FakeHandle:
    def __init__(self, frontend=0, engine=0, topology=proof.OK, frontend_log=None, unwritten=False):
        self.chain = "trtllm"
        # unwritten: the front end counts but has not written its counter file yet.
        self.counters = proof.Counters(
            None if unwritten else frontend, engine, frontend_unwritten=unwritten
        )
        self.topology = proof.Check(topology, "fake topology")
        self.final_counters = None
        self.final_topology = None
        self.proven_topology = None
        self._engine_name = None
        self.teardown_errors = []
        self.teardown_notes = []
        self.engine_log = None
        self.frontend_log = frontend_log
        self.started_seq = launch.next_seq()
        self.ended_seq = None
        launch._HANDLES.append(self)

    @property
    def alive(self):
        return self.ended_seq is None

    def send(self, frontend, engine=None):
        engine = frontend if engine is None else engine
        if self.counters.frontend_unwritten:
            self.counters = proof.Counters(0, self.counters.engine)
        self.counters = proof.Counters(
            None if self.counters.frontend is None else self.counters.frontend + frontend,
            None if self.counters.engine is None else self.counters.engine + engine,
        )

    def end(self, errors=()):
        self.teardown_errors.extend(errors)
        self.final_counters = self.counters
        self.final_topology = self.topology
        self.ended_seq = launch.next_seq()


def install(monkeypatch):
    """Route the plugin's process and RPC readers to FakeHandle attributes."""
    monkeypatch.setattr(launch, "_HANDLES", [])
    monkeypatch.setattr(
        proof, "read_counters", lambda h: h.counters if h.alive else h.final_counters
    )
    monkeypatch.setattr(proof, "check_topology", lambda h: h.topology)
    monkeypatch.setattr(proof, "read_engine_name", lambda h: "fake-engine")
