# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readiness criteria (scenario `readiness` list; every criterion must hold).

Each checker returns (ok, detail). They never block for long: the orchestrator polls them
until the launch origin, which is the readiness deadline (INTERFACES.md 0).
"""

import http.client
import os

from capcommon.exitcodes import ConfigError
from capcommon.jsonio import read_json_stamped
from capcommon.lifecycle import PORTFILE_SCHEMA, READY_SCHEMA


class ReadyError(ConfigError):
    """A readiness file exists but is wrong (identity of another launch, bad content)."""


def read_plain_port(path):
    """S-FE --port-file: a single integer line. Returns port or None if absent/partial."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None
    if not text.endswith("\n"):
        return None
    try:
        port = int(text.strip())
    except ValueError:
        raise ReadyError("%s: not a port number: %r" % (path, text))
    if not 0 < port < 65536:
        raise ReadyError("%s: port out of range: %d" % (path, port))
    return port


def read_report_addr(path):
    """trtllm-serve --report_addr: 'host:port\\n' (IPv6 hosts bracketed). (host, port) or None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None
    if not text.endswith("\n"):
        return None
    host, sep, port = text.strip().rpartition(":")
    if not sep or not host:
        raise ReadyError("%s: not host:port: %r" % (path, text))
    try:
        return host, int(port)
    except ValueError:
        raise ReadyError("%s: bad port in %r" % (path, text))


def read_capkit_port(path, launch_key):
    if not os.path.exists(path):
        return None
    ident, doc = read_json_stamped(path, PORTFILE_SCHEMA)
    if list(ident.launch_key()) != list(launch_key):
        raise ReadyError("%s: port file identity is not this launch's" % path)
    return doc["host"], doc["port"]


def read_capkit_ready(path, launch_key, role):
    """(identity, doc) of a capkit ready file, or None if it does not exist yet."""
    if not os.path.exists(path):
        return None
    ident, doc = read_json_stamped(path, READY_SCHEMA)
    if list(ident.launch_key()) != list(launch_key):
        raise ReadyError("%s: ready file identity is not this launch's" % path)
    if ident.role != role:
        raise ReadyError("%s: ready file role %r != %r" % (path, ident.role, role))
    for c in doc.get("children") or []:
        if not isinstance(c.get("pid"), int) or not isinstance(c.get("worker"), int):
            raise ReadyError("%s: bad children entry %r" % (path, c))
        if c.get("role") != role:
            raise ReadyError("%s: child %r reports another role" % (path, c))
    return ident, doc


def http_get_status(host, port, path, timeout_s):
    """Status code of GET http://host:port/path, or None on connection errors."""
    h = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    conn = http.client.HTTPConnection(h, port, timeout=timeout_s)
    try:
        conn.request("GET", path, headers={"Connection": "close"})
        resp = conn.getresponse()
        resp.read()
        return resp.status
    except OSError:
        return None
    finally:
        conn.close()
