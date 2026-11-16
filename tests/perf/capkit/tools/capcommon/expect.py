# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expected per-request message / frame counts and the reconcile spec (INTERFACES.md 5.8).

Every stub, null sink, beat source and the judge compute "how many messages / SSE frames a request
should produce, at which step" with these functions, so the counts that C4, C6, C7 and X2 compare
against come from one implementation (design 2.3 fidelity row F1; 2.7 C6, C7).

Terms (per request): N = max_tokens, k = tokens_per_msg, s = stream_interval, tokens_per_step
tps = k / s (an integer; tier tables enforce k % s == 0). A stream runs T = ceil(N / tps) decoding
iterations, numbered 1..T. Iteration i ends at token min(i * tps, N).

Gates (which iterations emit a message):
  product                -- the real engine (py_executor stream gate): iteration 1, every
                            iteration with i % s == 0, and the last iteration T.
  every_step_plus_final  -- S-FE: one message per iteration plus one final message (T + 1).
capkit's own null sinks and beat sources (nullsse, nullgrpc) use the product gate.
"""

from .exitcodes import ConfigError

GATES = ("product", "every_step_plus_final")
RECONCILE_SCHEMA = "capkit.reconcile/1"
CHAINS = ("requests", "messages", "chunks")
SCOPES = ("tier", "launch")


class ExpectError(ConfigError):
    pass


def _check(max_tokens, tokens_per_msg, stream_interval):
    for name, v in (("max_tokens", max_tokens), ("tokens_per_msg", tokens_per_msg),
                    ("stream_interval", stream_interval)):
        if not isinstance(v, int) or isinstance(v, bool):
            raise ExpectError("%s must be an int" % name)
    if max_tokens < 1 or tokens_per_msg < 1 or stream_interval < 1:
        raise ExpectError("max_tokens, tokens_per_msg, stream_interval must be >= 1")
    if tokens_per_msg % stream_interval:
        raise ExpectError("tokens_per_msg must be a multiple of stream_interval")
    return tokens_per_msg // stream_interval


def iterations(max_tokens, tokens_per_msg, stream_interval):
    """T = ceil(N / tokens_per_step): decoding iterations of one stream."""
    tps = _check(max_tokens, tokens_per_msg, stream_interval)
    return -(-max_tokens // tps)


def msg_iterations(max_tokens, tokens_per_msg, stream_interval, gate):
    """1-based iteration index of each message, in order (the final message of
    every_step_plus_final carries the index T, after the per-step message of T)."""
    if gate not in GATES:
        raise ExpectError("gate must be one of %s" % (GATES,))
    t = iterations(max_tokens, tokens_per_msg, stream_interval)
    if gate == "every_step_plus_final":
        return list(range(1, t + 1)) + [t]
    s = stream_interval
    return [i for i in range(1, t + 1) if i == 1 or i == t or i % s == 0]


def msgs_per_request(max_tokens, tokens_per_msg, stream_interval, gate):
    """Number of response messages one request yields (closed form of msg_iterations)."""
    if gate not in GATES:
        raise ExpectError("gate must be one of %s" % (GATES,))
    t = iterations(max_tokens, tokens_per_msg, stream_interval)
    if gate == "every_step_plus_final":
        return t + 1
    s = stream_interval
    n = t // s
    if s != 1:
        n += 1  # iteration 1 (not a multiple of s)
    if t % s != 0 and t != 1:
        n += 1  # the last iteration (not a multiple of s, not iteration 1)
    return n


def msg_tokens(max_tokens, tokens_per_msg, stream_interval, gate):
    """Tokens carried by each message (sums to max_tokens; S-FE's final message carries 0)."""
    tps = _check(max_tokens, tokens_per_msg, stream_interval)
    out, prev = [], 0
    its = msg_iterations(max_tokens, tokens_per_msg, stream_interval, gate)
    for i in its:
        end = min(i * tps, max_tokens)
        out.append(end - prev)
        prev = end
    return out


def msg_step_offsets(max_tokens, tokens_per_msg, stream_interval, gate):
    """Step offset of each message relative to the stream's first step: iteration i is emitted
    on beat step first_step + (i - 1). X2 for message j uses
    BeatSchedule.planned_rel_ns(first_step + offsets[j])."""
    return [i - 1 for i in msg_iterations(max_tokens, tokens_per_msg, stream_interval, gate)]


def sse_frames_per_request(n_msgs, include_usage):
    """SSE frames a client counts for one request whose server emits n_msgs content frames.

    Counting rule (INTERFACES.md 4.5): `frames` counts every SSE event except the `[DONE]`
    sentinel; the usage frame (sent when the request asked include_usage) counts as one frame.
    """
    if not isinstance(n_msgs, int) or n_msgs < 0:
        raise ExpectError("n_msgs must be an int >= 0")
    return n_msgs + (1 if include_usage else 0)


def validate_reconcile(doc, roles):
    """Validate a scenario's `reconcile` section (INTERFACES.md 5.8) against its role names.

    doc: {"schema": "capkit.reconcile/1", "chains": [{"name", "kind": requests|messages|chunks,
    "gate": product|every_step_plus_final|null, "include_usage": bool|null,
    "hops": [{"role", "source", "scope"}], "corrections": [{"name", "per_request": int,
    "hop": role}]}]}. source is "summary.<dotted path>" (into the role's per-tier summary, e.g.
    summary.counts.ok, summary.frames.frames) or "slot:<slot name>" (window delta of a slot).
    """
    if not isinstance(doc, dict) or doc.get("schema") != RECONCILE_SCHEMA:
        raise ExpectError("reconcile.schema must be %s" % RECONCILE_SCHEMA)
    chains = doc.get("chains")
    if not isinstance(chains, list) or not chains:
        raise ExpectError("reconcile.chains must be a non-empty list")
    names = set()
    for c in chains:
        if not isinstance(c, dict) or set(c) != {"name", "kind", "gate", "include_usage", "hops",
                                                  "corrections"}:
            raise ExpectError("each chain needs exactly name/kind/gate/include_usage/hops/"
                              "corrections")
        if c["name"] in names or not isinstance(c["name"], str) or not c["name"]:
            raise ExpectError("chain names must be unique non-empty strings")
        names.add(c["name"])
        if c["kind"] not in CHAINS:
            raise ExpectError("chain kind must be one of %s" % (CHAINS,))
        if c["kind"] == "requests":
            if c["gate"] is not None or c["include_usage"] is not None:
                raise ExpectError("a requests chain has gate and include_usage null")
        else:
            if c["gate"] not in GATES:
                raise ExpectError("chain %s: gate must be one of %s" % (c["name"], GATES))
            if c["kind"] == "chunks" and not isinstance(c["include_usage"], bool):
                raise ExpectError("chain %s: chunks need include_usage true/false" % c["name"])
            if c["kind"] == "messages" and c["include_usage"] is not None:
                raise ExpectError("chain %s: messages chains have include_usage null" % c["name"])
        hops = c["hops"]
        if not isinstance(hops, list) or len(hops) < 1:
            raise ExpectError("chain %s needs at least one hop" % c["name"])
        for h in hops:
            if not isinstance(h, dict) or set(h) != {"role", "source", "scope"}:
                raise ExpectError("chain %s: hop needs exactly role/source/scope" % c["name"])
            if h["role"] not in roles:
                raise ExpectError("chain %s: unknown role %r" % (c["name"], h["role"]))
            src = h["source"]
            if not isinstance(src, str) or not (
                    (src.startswith("summary.") and len(src) > len("summary."))
                    or (src.startswith("slot:") and len(src) > len("slot:"))):
                raise ExpectError("chain %s: source must be summary.<path> or slot:<name>"
                                  % c["name"])
            if h["scope"] not in SCOPES:
                raise ExpectError("chain %s: scope must be one of %s" % (c["name"], SCOPES))
        corr = c["corrections"]
        if not isinstance(corr, list):
            raise ExpectError("chain %s: corrections must be a list" % c["name"])
        for k in corr:
            if not isinstance(k, dict) or set(k) != {"name", "per_request", "hop"}:
                raise ExpectError("chain %s: correction needs exactly name/per_request/hop"
                                  % c["name"])
            if not isinstance(k["per_request"], int) or isinstance(k["per_request"], bool):
                raise ExpectError("chain %s: correction per_request must be an int" % c["name"])
            if k["hop"] not in {h["role"] for h in hops}:
                raise ExpectError("chain %s: correction hop must be one of the chain's roles"
                                  % c["name"])
    return doc


def expected_hop_count(chain, n_requests, max_tokens, tokens_per_msg, stream_interval, hop_role):
    """Expected count at one hop for n_requests requests of one tier's parameters."""
    corr = sum(k["per_request"] for k in chain["corrections"] if k["hop"] == hop_role)
    if chain["kind"] == "requests":
        per = 1
    else:
        per = msgs_per_request(max_tokens, tokens_per_msg, stream_interval, chain["gate"])
        if chain["kind"] == "chunks":
            per = sse_frames_per_request(per, chain["include_usage"])
    return n_requests * (per + corr)
