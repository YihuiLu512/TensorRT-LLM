# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R10 probe hook (WT-31, UV-29): log what a worker derives as subagent_affinity_id.

Active only when R10_PROBE_LOG is set (r10_probe.py sets it, with this directory on PYTHONPATH,
for the workers of the affinity scenario). After tensorrt_llm.serve.openai_server is imported it
wraps OpenAIServer._get_scheduling_params and appends one JSON line per call. Measurement only;
the wrapped method's result is returned and its exception re-raised unchanged. A sitecustomize shadowed by this file is
loaded afterwards.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys

_LOG = os.environ.get("R10_PROBE_LOG")
_TARGET = "tensorrt_llm.serve.openai_server"


def _patch(module):
    cls = module.OpenAIServer
    original = cls._get_scheduling_params

    def wrapped(self, request, raw_request):
        disagg = getattr(request, "disaggregated_params", None)
        record = {
            "pid": os.getpid(),
            "server_role": str(getattr(self, "server_role", None)),
            "request_type": getattr(disagg, "request_type", None),
            "header_present": raw_request is not None
            and any(k.lower().startswith("x-trtllm-subagent") for k in raw_request.headers.keys()),
        }
        try:
            params = original(self, request, raw_request)
        except Exception as error:
            # A rejected derivation is a result too; the request fails as it would without the hook.
            record["raised"] = repr(error)[:300]
            raise
        else:
            record["subagent_affinity_id"] = getattr(params, "subagent_affinity_id", None)
            return params
        finally:
            try:
                with open(_LOG, "a") as f:
                    f.write(json.dumps(record, default=str) + "\n")
            except OSError:
                pass  # a lost log line must not change the request's outcome

    cls._get_scheduling_params = wrapped


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        loader.exec_module = exec_and_patch
        return spec


if _LOG:
    sys.meta_path.insert(0, _Finder())

_here = os.path.dirname(os.path.abspath(__file__))
for _entry in sys.path:
    if os.path.abspath(_entry or ".") == _here:
        continue
    _candidate = os.path.join(_entry, "sitecustomize.py")
    if os.path.isfile(_candidate):
        _spec = importlib.util.spec_from_file_location("_r10_original_sitecustomize", _candidate)
        _spec.loader.exec_module(importlib.util.module_from_spec(_spec))
        break
