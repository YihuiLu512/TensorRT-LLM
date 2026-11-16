# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capgrpc.oe_sink: open-loop driver that reads Generate streams through the product's OpenEngine
client classes and discards the events (design E3 card 3b; INTERFACES errata FT1-8..FT1-12).

- One process, one channel (the product's one channel per frontend), no HTTP.
- --tier generate-stream: tensorrt_llm.grpc.openengine_client.OpenEngineClient.generate ->
  GenerateStream (read + stream-contract validation). --tier engine-event-stream: additionally
  tensorrt_llm.serve.openengine_poc.engine_client_impl.OpenEngineEngineClient.generate on an
  EngineRequest (EngineRequest -> GenerateRequest per request, protobuf -> dataclass per event).
- --connect product: the client's own connect() (Control: Health, GetServerInfo, GetModelInfo;
  needs a peer that answers Control). --connect channel: the client object is built on a channel
  with the options connect() sets and no Control call (OpenEngineClient constructor;
  OpenEngineEngineClient without its capability cache); the Generate path is the product's own.
- Schedule, arrival, tiers, send lag, records and the summary are capgrpc.load's (LoadWorker /
  RequestPlan, same reader arguments), so the judge reads it as a load role and a scenario swaps
  it with the weak reader (capgrpc.load --in-process --workers 1 --channels-per-worker 1) by the
  module name and the flags only this tool has.
- Reader-side switches: the recv.msg site (busy / sleep, blocking the loop thread) and
  --slow-read-us (an asyncio pause after every message of a stream, the known-answer sample
  "reader slow").
The product is imported at start-up; any failure to import it exits 3 (container only), and what
was imported is in the ready file and every tier summary.
"""

import argparse
import asyncio
import importlib
import sys

import grpc

from capcommon.aio import call_at_ns, run as aio_run
from capcommon.cli import add_common_args, common_from_args
from capcommon.clock import now_ns
from capcommon.exitcodes import CapkitError, ConfigError

from . import SITES, SLOTS as LOAD_SLOTS
from .common import ToolRuntime, product_versions, wait_channels_ready
from .loader import LoadWorker, RequestPlan, add_reader_args, validate_args as validate_reader

TOOL = "capgrpc.oe_sink"
TIERS = ("generate-stream", "engine-event-stream")
CONNECTS = ("product", "channel")
# capgrpc.load's slots except bytes.recv: product objects carry no wire size.
SLOTS = {k: v for k, v in LOAD_SLOTS.items() if k != "bytes.recv"}

CLIENT_MODULE = "tensorrt_llm.grpc.openengine_client"
BINDINGS_MODULE = "tensorrt_llm.grpc.openengine.bindings"
ENGINE_MODULE = "tensorrt_llm.serve.openengine_poc.engine_client_impl"
REQUEST_MODULE = "tensorrt_llm.serve.openengine_poc.engine_client"
CLIENT_NAMES = ("OpenEngineClient", "GenerateStream", "OpenEngineClientError",
                "OpenEngineRpcError", "OpenEngineEngineError", "OpenEngineProtocolError",
                "MAX_MESSAGE_BYTES", "new_request_id")
ENGINE_NAMES = ("OpenEngineEngineClient", "EngineEventStream", "to_generate_request")


# ---- product ------------------------------------------------------------------------------------

def _import(name):
    try:
        return importlib.import_module(name)
    except (Exception, SystemExit) as e:  # noqa: BLE001 - whatever the import raises, exit 3
        raise ConfigError("oe_sink needs the product module %s (container only): %s: %s"
                          % (name, type(e).__name__, e))


def _names(mod, names):
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        raise ConfigError("%s has no %s" % (mod.__name__, ", ".join(missing)))
    return [getattr(mod, n) for n in names]


class _PathRecorder:
    """A stand-in channel that records the method paths a generated stub registers."""

    def __init__(self):
        self.paths = []

    def _record(self, path, *args, **kwargs):
        self.paths.append(path)

    unary_unary = unary_stream = stream_unary = stream_stream = _record


def generate_method(grpc_module):
    """The Generate method path the product's InferenceStub calls."""
    rec = _PathRecorder()
    grpc_module.InferenceStub(rec)
    gen = [p for p in rec.paths if p.rsplit("/", 1)[-1] == "Generate"]
    if len(gen) != 1:
        raise ConfigError("the product InferenceStub registers %d Generate paths: %s"
                          % (len(gen), rec.paths))
    return gen[0]


class Product:
    """The product client classes one run drives, and the report of what was imported."""

    def __init__(self, tier):
        self.tier = tier
        oc = _import(CLIENT_MODULE)
        bindings = _import(BINDINGS_MODULE)
        (self.Client, self.Stream, self.ClientError, self.RpcError, self.EngineError,
         self.ProtocolError, self.max_message_bytes, self.new_request_id) = _names(oc,
                                                                                  CLIENT_NAMES)
        self.pb2, self.pb2_grpc = _names(bindings, ("generation_pb2", "openengine_pb2_grpc"))
        self.method = generate_method(self.pb2_grpc)
        self.EngineClient = self.EventStream = self.to_generate_request = None
        self.EngineRequest = None
        mods = [oc, bindings]
        if tier == "engine-event-stream":
            eci = _import(ENGINE_MODULE)
            ecm = _import(REQUEST_MODULE)
            self.EngineClient, self.EventStream, self.to_generate_request = _names(eci,
                                                                                  ENGINE_NAMES)
            (self.EngineRequest,) = _names(ecm, ("EngineRequest",))
            mods += [eci, ecm]
        self.report = {"tier": tier, "generate_method": self.method,
                       "modules": {m.__name__: getattr(m, "__file__", None) for m in mods},
                       "stream_class": _qualname(self.Stream),
                       "event_stream_class": _qualname(self.EventStream)
                       if self.EventStream else None}
        self.report.update(product_versions())


def _qualname(cls):
    return "%s.%s" % (cls.__module__, cls.__qualname__)


# ---- request templates --------------------------------------------------------------------------

def engine_request_kwargs(product, msg):
    """EngineRequest fields (all but request_id) that rebuild the GenerateRequest template `msg`
    through the product's own to_generate_request; exit 3 when they cannot."""
    if not msg.HasField("token_ids"):
        raise ConfigError("--tier engine-event-stream needs token-id prompts (EngineRequest has "
                          "no text prompt)")
    kw = {"model": msg.model, "prompt_token_ids": list(msg.token_ids.ids),
          "max_tokens": msg.stopping.max_tokens if msg.stopping.HasField("max_tokens") else None}
    if msg.stopping.HasField("ignore_eos"):
        kw["ignore_eos"] = msg.stopping.ignore_eos
    sampling = {f.name: v for f, v in msg.sampling.ListFields()}
    if sampling:
        kw["sampling"] = sampling
    try:
        back = product.to_generate_request(product.EngineRequest(request_id="", **kw))
    except (TypeError, ValueError) as e:
        raise ConfigError("request template cannot be an EngineRequest: %s" % e)
    if back.SerializeToString(deterministic=True) != msg.SerializeToString(deterministic=True):
        raise ConfigError("request template uses GenerateRequest fields the EngineRequest path "
                          "does not rebuild: template %s, rebuilt %s" % (msg, back))
    return kw


class Templates:
    """The request-bytes templates parsed once (before the origin) with the product's pb2."""

    def __init__(self, product, plan):
        self.product = product
        self.plan = plan
        rb = plan.reqbytes
        self.raw = plan.format == "raw"
        raws = {None: rb.wildcard} if self.raw else dict(rb.table)
        self.msgs, self.kw = {}, {}
        for key, raw in raws.items():
            try:
                msg = product.pb2.GenerateRequest.FromString(raw)
            except Exception as e:  # noqa: BLE001 - any decode failure of a template is exit 3
                raise ConfigError("request template %s does not parse as the product's "
                                  "GenerateRequest: %s" % (key, e))
            if msg.request_id:
                raise ConfigError("request template %s carries a request_id" % (key,))
            self.msgs[key] = msg
            if product.tier == "engine-event-stream":
                self.kw[key] = engine_request_kwargs(product, msg)

    def key(self, rid, seq):
        return None if self.raw else self.plan.template_key(rid, seq)

    def request_id(self, rid):
        """openengine: the decimal req_id (S-PB / S-NL map it to the schedule); raw: a fresh
        product id."""
        return self.product.new_request_id() if self.raw else str(rid)


# ---- worker -------------------------------------------------------------------------------------

class OeSinkWorker(LoadWorker):
    """capgrpc.load's in-process worker with the product client in place of raw channels."""

    COUNT_BYTES = False

    def __init__(self, args, rt, plan, product, templates):
        super().__init__(args, rt, plan, None)
        self.product = product
        self.templates = templates
        self.tier2 = product.tier == "engine-event-stream"
        self.connect_mode = args.connect
        self.slow_s = args.slow_read_us / 1e6 if args.slow_read_us else None
        self.pauses_by_tier = [0] * len(rt.tt)
        self.client = self.engine = None
        self.loop = None

    # -- connection --------------------------------------------------------------------------------
    async def connect(self, target):
        P = self.product
        self.loop = asyncio.get_running_loop()
        if self.connect_mode == "channel":
            opts = [("grpc.max_send_message_length", P.max_message_bytes),
                    ("grpc.max_receive_message_length", P.max_message_bytes)]
            channel = grpc.aio.insecure_channel(target, options=opts)
            self.client = P.Client(target, channel, control_timeout=self.args.control_timeout_s,
                                   allow_newer_schema=False)
            await wait_channels_ready([channel], self.origin)
            if self.tier2:
                self.engine = P.EngineClient.__new__(P.EngineClient)
                self.engine._client = self.client
        else:
            ready = (self.origin - now_ns()) / 1e9
            if ready <= 0:
                raise CapkitError("origin already passed before connecting")
            cls = P.EngineClient if self.tier2 else P.Client
            try:
                c = await cls.connect(target, ready_timeout=ready,
                                      control_timeout=self.args.control_timeout_s)
            except P.ClientError as e:
                raise CapkitError("product %s.connect(%s) failed before the origin: %s: %s"
                                  % (cls.__name__, target, type(e).__name__, e))
            if self.tier2:
                self.engine = c
            else:
                self.client = c
        self.self_check()

    def self_check(self):
        """Build one stream from the first planned request and cancel it before its first
        iteration (the product never sends it): the request path is wired before the origin."""
        if not self.plan.n:
            return
        try:
            st = self.open_stream(self.plan.req_ids[0], self.plan.tseq[0], None)
            st.cancel()
            ok = st.cancelled and not st.sent
        except Exception as e:  # noqa: BLE001 - any failure here is a wiring problem, exit 3
            raise ConfigError("oe_sink self-check: building a %s stream failed: %s: %s"
                              % (self.product.tier, type(e).__name__, e))
        if not ok:
            raise ConfigError("oe_sink self-check: a stream cancelled before its first "
                              "iteration reports cancelled=%r sent=%r" % (st.cancelled, st.sent))

    async def disconnect(self):
        # OpenEngineEngineClient.close closes its client; with --connect channel the engine
        # client is the hollow one built in connect(), so close the client directly
        if self.engine is not None and self.connect_mode == "product":
            await self.engine.close()
        elif self.client is not None:
            await self.client.close()

    def connection_info(self):
        return {"local_subchannel_pool": None, "connect": self.connect_mode,
                "imported": self.product.report, "slow_read_ns": self._slow_ns()}

    def _slow_ns(self):
        return self.args.slow_read_us * 1000 if self.args.slow_read_us else None

    # -- one request -------------------------------------------------------------------------------
    def open_stream(self, rid, seq, timeout):
        """The product stream of one request (the RPC is sent on its first iteration)."""
        T = self.templates
        key = T.key(rid, seq)
        if self.tier2:
            req = self.product.EngineRequest(request_id=T.request_id(rid), **T.kw[key])
            return self.engine.generate(req)
        msg = T.msgs[key]
        msg.request_id = T.request_id(rid)  # generate() sends a copy taken now
        return self.client.generate(msg, timeout=timeout)

    @staticmethod
    def _expire(stream, expired):
        expired.append(True)
        stream.cancel()

    async def one(self, i):
        plan, v, P = self.plan, self.v, self.product
        rid, seq, prel = plan.req_ids[i], plan.tseq[i], plan.planned[i]
        t0, timeout = self.begin_request(seq, prel)
        active = True
        nmsg = 0
        first = -1
        reason = None
        stream = timer = None
        expired = []
        sampled = rid % self.mod == 0
        msgplan = self.msgplan
        mst = msgplan.stream(prel, seq) if msgplan is not None else None
        inj_recv, inj_drop, slow = self.inj_recv, self.inj_drop, self.slow_s
        try:
            stream = self.open_stream(rid, seq, timeout)
            if self.tier2 and timeout is not None:
                # EngineEventStream has no deadline: cancel at the same instant instead
                timer = call_at_ns(self.loop, t0 + int(timeout * 1e9), self._expire, stream,
                                   expired)
            anext = stream.__anext__
            while True:
                try:
                    item = await anext()
                except StopAsyncIteration:
                    break
                t = now_ns()
                inj_recv.hit()
                if inj_drop.should_drop():
                    continue
                if nmsg == 0:
                    rt_ = self.recv_types[seq]
                    tn = type(item).__name__
                    rt_[tn] = rt_.get(tn, 0) + 1
                    first = t
                v[self.i_read] += 1
                if msgplan is not None:
                    self.planned_msg(mst, nmsg, t, rid, seq, sampled)
                nmsg += 1
                if slow is not None:
                    self.pauses_by_tier[seq] += 1
                    await asyncio.sleep(slow)
            v[self.i_active] -= 1
            active = False
            if stream.cancelled:
                reason = "timeout_registered" if expired else "grpc_CANCELLED"
        except P.RpcError as e:
            reason = self._reason(e.code)
        except P.EngineError:
            reason = "engine_error"
        except P.ProtocolError:
            reason = "protocol_error"
        except asyncio.CancelledError:
            reason = "grpc_CANCELLED"
        except Exception as e:  # noqa: BLE001 - a client-side fault still ends the request once
            reason = "client_error"
            self.log_client_error(rid, e)
        finally:
            if timer is not None:
                timer.cancel()
            if stream is not None:
                stream.cancel()  # no-op once the stream has ended
            if active:
                v[self.i_active] -= 1
        self.end_request(rid, seq, prel, t0, first, nmsg, -1, reason, 0)

    def extra_summary(self, seq):
        return {"local_subchannel_pool": None,
                "oe_sink": {"tier": self.product.tier, "connect": self.connect_mode,
                            "timeout_mechanism": "client_cancel" if self.tier2
                            else "grpc_deadline",
                            "slow_read_ns": self._slow_ns(),
                            "slow_read_pauses": self.pauses_by_tier[seq]},
                "imported": self.product.report}


# ---- entry ------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="python -m capgrpc.oe_sink",
                                description=__doc__.split("\n")[0], allow_abbrev=False)
    add_common_args(p, loop=True, tiers=True, snapshot=True, ready=True, port=False)
    g = p.add_argument_group("capgrpc.oe_sink (plus capgrpc.load's reader arguments)")
    add_reader_args(g)
    g.add_argument("--tier", required=True, choices=TIERS)
    g.add_argument("--connect", required=True, choices=CONNECTS)
    g.add_argument("--control-timeout-s", type=float, default=5.0,
                   help="the client's control_timeout (product default 5.0)")
    g.add_argument("--slow-read-us", type=int,
                   help="pause this long (asyncio) after every message of a stream")
    p.set_defaults(stamped=False, shared_subchannel_pool=False)
    return p


def validate_args(args, common):
    if args.workers != 1:
        raise ConfigError("oe_sink is one process: --workers must be 1")
    if args.channels_per_worker != 1:
        raise ConfigError("oe_sink drives one channel: --channels-per-worker must be 1")
    if args.request_format == "capkit":
        raise ConfigError("oe_sink sends OpenEngine requests: --request-format openengine or raw")
    if args.slow_read_us is not None and args.slow_read_us < 1:
        raise ConfigError("--slow-read-us must be >= 1")
    if not args.control_timeout_s > 0:
        raise ConfigError("--control-timeout-s must be > 0")
    validate_reader(args, common)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        common = common_from_args(args)
        validate_args(args, common)
        args.in_process = True  # always one process (--in-process is accepted for argv parity)
        product = Product(args.tier)
        if args.method != product.method:
            raise ConfigError("--method %s is not the product's Generate path %s"
                              % (args.method, product.method))
        rt = ToolRuntime(common, TOOL, SLOTS, SITES, 0, 1, [common.cpus[0]],
                         args.lag_period_us, args.lag_records_capacity)
        plan = RequestPlan(args, rt.tt, 0)
        templates = Templates(product, plan)
        w = OeSinkWorker(args, rt, plan, product, templates)
        return aio_run(w.run, common.loop)
    except CapkitError as e:
        print("%s: %s" % (TOOL, e), file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
