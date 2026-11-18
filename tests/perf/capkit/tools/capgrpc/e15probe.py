# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python -m capgrpc.e15probe: E15-B admission checks against a running OpenEngine engine.

An aux observer role. It writes its ready file at once, sleeps until origin + --at-s (a drain
tier with no load in flight), connects to the engine's gRPC address with the product's
`tensorrt_llm.grpc.openengine_client.OpenEngineClient`, and runs, in order:

  health_plain   Health(include_inference_probe=False): state READY
  health_probe   Health(include_inference_probe=True): state READY and a READY check named
                 "inference_probe"
  getload        GetLoad answers (not UNIMPLEMENTED) with running_requests set
  cancel         a streamed Generate cancelled by the client after --cancel-after-chunks token
                 events; then GetLoad.running_requests back to its value before the request
                 within --clear-timeout-s
  abort          a streamed Generate aborted with Abort(request_id) after the same number of
                 token events: outcome ABORTED, the stream ends, running_requests back
  greedy_aa      two greedy (temperature 0, top_k 1, seed 0) Generate calls of --greedy-tokens
                 tokens on the same prompt, one at a time: token ids equal (A/A within the arm;
                 the ids and their sha256 are recorded so two arms can be compared offline)

Each check records ok, the wall/monotonic times and its details; one failing check does not
stop the others. The whole sequence has a deadline (--deadline-s). It writes --result, then
stays alive until SIGTERM (the orchestrator stops aux roles at the end of the launch). No timing
here is a measurement result.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time

from capcommon import exitcodes
from capcommon.cli import add_common_args, common_from_args
from capcommon.clock import wall_iso
from capcommon.exitcodes import ConfigError
from capcommon.jsonio import atomic_write_bytes, dumps
from capcommon.lifecycle import StopController, pin_self, write_ready_file
from capcommon.versions import version_record

SCHEMA = "capkit.e15probe/1"
CHECKS = ("health_plain", "health_probe", "getload", "cancel", "abort", "greedy_aa")


class ProductApi:
    """The product's client and message helpers (imported only inside the container)."""

    def __init__(self):
        from tensorrt_llm.grpc.openengine.bindings import (  # noqa: PLC0415
            generation_pb2, lifecycle_pb2)
        from tensorrt_llm.grpc.openengine_client import client as oec  # noqa: PLC0415
        self.gen, self.life, self.oec = generation_pb2, lifecycle_pb2, oec

    async def connect(self, target, ready_timeout_s):
        return await self.oec.OpenEngineClient.connect(target, ready_timeout=ready_timeout_s)

    def health_state(self, resp):
        return self.life.HealthState.Name(resp.state)

    def health_checks(self, resp):
        return [(c.name, self.life.HealthState.Name(c.state), c.message) for c in resp.checks]

    def running(self, load):
        if load is None:
            return None
        return load.running_requests if load.HasField("running_requests") else None

    def queued(self, load):
        return load.queued_requests if load is not None and load.HasField("queued_requests") \
            else None

    def request(self, model, token_ids, max_tokens, greedy):
        r = self.gen.GenerateRequest(model=model)
        r.token_ids.ids.extend(token_ids)
        r.stopping.max_tokens = max_tokens
        r.stopping.ignore_eos = True
        if greedy:
            r.sampling.temperature = 0.0
            r.sampling.top_k = 1
            r.sampling.seed = 0
        return r

    def event(self, resp):
        """('token', [ids]) | ('finished', reason name) | (other event name, None)."""
        kind = resp.WhichOneof("event")
        if kind == "token":
            return "token", [t.token_id for t in resp.token.tokens]
        if kind == "finished":
            return "finished", self.gen.FinishReason.Name(resp.finished.reason)
        return kind, None

    def abort_outcome(self, outcome):
        return outcome.name if hasattr(outcome, "name") else str(outcome)


def _rec(name, t0_mono):
    return {"check": name, "ok": False, "started_wall": wall_iso(),
            "started_rel_s": round((time.monotonic_ns() - t0_mono) / 1e9, 3)}


async def _load_now(api, client):
    load = await client.get_load()
    return api.running(load), api.queued(load)


async def _wait_cleared(api, client, base, timeout_s, poll_s=0.1):
    """Seconds until running_requests <= base, or None on timeout / no reading."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        run, _q = await _load_now(api, client)
        if run is None:
            return None
        if run <= base:
            return round(time.monotonic() - t0, 3)
        await asyncio.sleep(poll_s)
    return None


async def _stream_until(api, stream, n_events):
    """Read token events until n_events of them arrived (or the stream ends)."""
    got, tokens, finished = 0, [], None
    async for resp in stream:
        kind, val = api.event(resp)
        if kind == "token":
            got += 1
            tokens.extend(val)
            if got >= n_events:
                break
        elif kind == "finished":
            finished = val
            break
    return got, tokens, finished


async def _check_health(api, client, rec, probe, timeout_s):
    resp = await client.health(include_inference_probe=probe, timeout=timeout_s)
    rec["state"] = api.health_state(resp)
    rec["checks"] = [{"name": n, "state": s, "message": m} for n, s, m in api.health_checks(resp)]
    ok = rec["state"] == "HEALTH_STATE_READY"
    if probe:
        ok = ok and any(c["name"] == "inference_probe" and c["state"] == "HEALTH_STATE_READY"
                        for c in rec["checks"])
    rec["ok"] = ok


async def _check_getload(api, client, rec):
    load = await client.get_load()
    rec["implemented"] = load is not None
    rec["running_requests"], rec["queued_requests"] = api.running(load), api.queued(load)
    rec["ok"] = load is not None and rec["running_requests"] is not None


async def _check_stop(api, client, rec, cfg, how):
    """how: 'cancel' (client cancels the stream) or 'abort' (Abort(request_id))."""
    base, _q = await _load_now(api, client)
    rec["running_before"] = base
    req = api.request(cfg["model"], cfg["prompt"], cfg["max_tokens"], greedy=False)
    async with client.generate(req, timeout=cfg["stream_timeout_s"]) as stream:
        got, _tok, fin = await _stream_until(api, stream, cfg["after_chunks"])
        rec.update(request_id=stream.request_id, token_events_before_stop=got,
                   finished_before_stop=fin)
        run_mid, _q = await _load_now(api, client)
        rec["running_mid"] = run_mid
        if how == "cancel":
            stream.cancel()
            rec["stream_cancelled"] = stream.cancelled is True  # a property on GenerateStream
        else:
            rec["abort_outcome"] = api.abort_outcome(await client.abort(stream.request_id))
            rest = 0
            try:
                async for resp in stream:
                    kind, val = api.event(resp)
                    rest += kind == "token"
                    if kind == "finished":
                        rec["finished_after_abort"] = val
            except Exception as e:  # noqa: BLE001  the product raises its own error types
                rec["stream_end_error"] = "%s: %s" % (type(e).__name__, e)
            rec["token_events_after_abort"] = rest
    rec["clear_s"] = await _wait_cleared(api, client, base or 0, cfg["clear_timeout_s"])
    ok = got >= cfg["after_chunks"] and fin is None and rec["clear_s"] is not None \
        and (run_mid or 0) > (base or 0)
    if how == "abort":
        ok = ok and rec["abort_outcome"] == "ABORTED"
    rec["ok"] = bool(ok)


async def _check_greedy(api, client, rec, cfg):
    runs = []
    for _ in range(2):
        req = api.request(cfg["model"], cfg["prompt"], cfg["greedy_tokens"], greedy=True)
        tokens, fin = [], None
        async with client.generate(req, timeout=cfg["stream_timeout_s"]) as stream:
            async for resp in stream:
                kind, val = api.event(resp)
                if kind == "token":
                    tokens.extend(val)
                elif kind == "finished":
                    fin = val
        runs.append({"n": len(tokens), "finished": fin, "token_ids": tokens,
                     "sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()})
    rec["runs"] = runs
    rec["ok"] = runs[0]["n"] == cfg["greedy_tokens"] and runs[0]["token_ids"] == \
        runs[1]["token_ids"]


async def run_checks(api, cfg):
    """The selected checks (cfg["checks"], default all) in CHECKS order; returns the list of
    records (never raises for one check)."""
    t0 = time.monotonic_ns()
    out = []
    client = await api.connect(cfg["target"], cfg["ready_timeout_s"])
    try:
        steps = [("health_plain", lambda r: _check_health(api, client, r, False,
                                                          cfg["control_timeout_s"])),
                 ("health_probe", lambda r: _check_health(api, client, r, True,
                                                          cfg["probe_timeout_s"])),
                 ("getload", lambda r: _check_getload(api, client, r)),
                 ("cancel", lambda r: _check_stop(api, client, r, cfg, "cancel")),
                 ("abort", lambda r: _check_stop(api, client, r, cfg, "abort")),
                 ("greedy_aa", lambda r: _check_greedy(api, client, r, cfg))]
        want = cfg.get("checks") or CHECKS
        for name, fn in steps:
            if name not in want:
                continue
            rec = _rec(name, t0)
            try:
                await asyncio.wait_for(fn(rec), cfg["check_timeout_s"])
            except Exception as e:  # noqa: BLE001  record and go on to the next check
                rec["error"] = "%s: %s" % (type(e).__name__, e)
                rec["ok"] = False
            rec["ended_rel_s"] = round((time.monotonic_ns() - t0) / 1e9, 3)
            out.append(rec)
    finally:
        await client.close()
    return out


def build_parser():
    ap = argparse.ArgumentParser(prog="capgrpc.e15probe", description=__doc__.split("\n\n")[0])
    add_common_args(ap, cpus=True, inject=False, stop=True)
    g = ap.add_argument_group("e15probe")
    g.add_argument("--origin-ns", type=int, required=True)
    g.add_argument("--ready-file", required=True)
    g.add_argument("--target", required=True, help="host:port of the engine's gRPC server")
    g.add_argument("--model", required=True, help="the served model name")
    g.add_argument("--prompts", required=True, help="capkit.prompts/1 file (token_ids set)")
    g.add_argument("--prompt-set", required=True)
    g.add_argument("--at-s", type=float, required=True, help="start at origin + this (s)")
    g.add_argument("--result", required=True)
    o = ap.add_argument_group("optional")
    o.add_argument("--max-tokens", type=int, default=1024)
    o.add_argument("--cancel-after-chunks", type=int, default=8)
    o.add_argument("--greedy-tokens", type=int, default=64)
    o.add_argument("--ready-timeout-s", type=float, default=60.0)
    o.add_argument("--control-timeout-s", type=float, default=5.0)
    o.add_argument("--probe-timeout-s", type=float, default=120.0)
    o.add_argument("--clear-timeout-s", type=float, default=30.0)
    o.add_argument("--stream-timeout-s", type=float, default=120.0)
    o.add_argument("--check-timeout-s", type=float, default=150.0)
    o.add_argument("--checks", default=",".join(CHECKS),
                   help="comma-separated subset of %s (run in that order)" % ",".join(CHECKS))
    return ap


def config_from_args(a):
    for k in ("max_tokens", "cancel_after_chunks", "greedy_tokens"):
        if getattr(a, k) <= 0:
            raise ConfigError("--%s must be > 0" % k.replace("_", "-"))
    if a.cancel_after_chunks >= a.max_tokens:
        raise ConfigError("--cancel-after-chunks must be < --max-tokens")
    checks = [c for c in a.checks.split(",") if c]
    bad = [c for c in checks if c not in CHECKS]
    if not checks or bad or len(set(checks)) != len(checks):
        raise ConfigError("--checks must be a non-empty subset of %s, got %r"
                          % (",".join(CHECKS), a.checks))
    doc = json.load(open(a.prompts))
    s = (doc.get("sets") or {}).get(a.prompt_set)
    if doc.get("schema") != "capkit.prompts/1" or not s or s.get("kind") != "token_ids":
        raise ConfigError("%s: need a capkit.prompts/1 token_ids set %r" % (a.prompts,
                                                                            a.prompt_set))
    return {"target": a.target, "model": a.model, "prompt": list(s["items"][0]),
            "max_tokens": a.max_tokens, "after_chunks": a.cancel_after_chunks,
            "greedy_tokens": a.greedy_tokens, "ready_timeout_s": a.ready_timeout_s,
            "control_timeout_s": a.control_timeout_s, "probe_timeout_s": a.probe_timeout_s,
            "clear_timeout_s": a.clear_timeout_s, "stream_timeout_s": a.stream_timeout_s,
            "check_timeout_s": a.check_timeout_s,
            "checks": [c for c in CHECKS if c in checks]}


def write_result(path, identity, cfg, records, error=None):
    doc = {"schema": SCHEMA, "identity": identity.to_dict(), "written_wall": wall_iso(),
           "config": {k: v for k, v in cfg.items() if k != "prompt"},
           "prompt_len": len(cfg["prompt"]), "error": error, "checks": records,
           "all_ok": error is None and len(records) == len(cfg.get("checks") or CHECKS)
           and all(r["ok"] for r in records)}
    atomic_write_bytes(path, (dumps(doc) + "\n").encode())
    return doc


def main(argv=None, api=None):
    a = build_parser().parse_args(argv)
    try:
        common = common_from_args(a)
        cfg = config_from_args(a)
        affinity = pin_self(common.cpus)
    except (ConfigError, OSError, ValueError) as e:
        print("capgrpc.e15probe: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    stop = StopController()
    write_ready_file(a.ready_file, common.identity, version_record(include_grpc=True),
                     affinity=affinity, info={"threads": []})
    target = a.origin_ns + int(a.at_s * 1e9)
    while not stop.requested and time.monotonic_ns() < target:
        stop.wait(min(1.0, (target - time.monotonic_ns()) / 1e9))
    records, error = [], None
    if not stop.requested:
        try:
            records = asyncio.run(run_checks(api or ProductApi(), cfg))
        except Exception as e:  # noqa: BLE001  connect/import failure: recorded, not raised
            error = "%s: %s" % (type(e).__name__, e)
    doc = write_result(a.result, common.identity, cfg, records, error)
    print("capgrpc.e15probe: all_ok=%s %s" % (doc["all_ok"], " ".join(
        "%s=%s" % (r["check"], r["ok"]) for r in records) or error), file=sys.stderr)
    while not stop.requested:
        stop.wait(1.0)
    return exitcodes.OK


if __name__ == "__main__":
    sys.exit(main())
