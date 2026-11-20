# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S-ZR launcher: inputs and process start for the C++ replay stub `trtllm-oehost-szr` (D1 E16-zr;
TRT-LLM `cpp/tensorrt_llm/openengine_host/tools/szr/`).

S-ZR stands in for rank 0 on the two sockets of one attached frontend (lane 1): it binds the request
entry and pushes S-NX-shaped result frames (pickle.dumps(list of LlmResponse) || HMAC-SHA256) to the
frontend's result lane on the S-NX step grid. Subcommands (all take the S-NX config options of
capstubs.serve_nx: --config-json, --set KEY=JSON, --tier-table, --origin-ns):

- prepare     (container) response templates, capkit.szr_templates/1: for every shape (n new tokens,
              not final / final LENGTH / final CANCELLED) a real LlmResponse from S-NX's product path
              (capstubs.snx_product.ProductAdapter: executor_request_to_llm_request, add_new_token,
              finish_by_reason, create_response) on an executor request built by the product worker's
              own BaseWorker._enqueue_request from a GenerationRequest (the D1 template package's
              request_template.pkl with --request-template, else a synthetic one), pickled as a list
              and cut into a head element (first list item) and a tail element (later items) with the
              request_id / client_id opcodes and the token offsets located by markers; then a Python
              self-check assembles random frames from the templates and compares pickle.loads of them
              with pickle.dumps of freshly made product responses, byte for byte per element.
- timetable   capkit.szr_timetable/1 from the S-NX config's tier table and origin (the BeatSchedule
              segments, outside beats, k per tier, stream interval): the grid S-ZR steps on.
- frame-sizes capkit.szr_frame_sizes/1 from an S-NX self-report directory (responses per
              await_responses take, optionally one tier): the per-frame response count distribution.
- run         standalone: makes an HMAC key and the attach info of a two-lane classic executor
              (request entry = S-ZR, lane 1 = the frontend under test), writes attach.json, d1.env
              (TLLM_EXECUTOR_ATTACH_INFO, TLLM_EXECUTOR_FRONTEND_ID=1) and hmac_key.hex (0600) into
              --out-dir, starts S-ZR and supervises it (SIGTERM / SIGINT forwarded).
- make-wrapper / wrap
              E16-zr with the real launcher: make-wrapper writes <out-dir>/szr-d1-wrap, an executable to
              set as TLLM_OPENENGINE_CPP_HOST_BIN. The launcher (serve_nx ... --openengine-host-impl cpp)
              starts it with the D1 command line and its attach info; `wrap` prepares the templates from
              the launcher's template package (--template <dir> of the D1 command line), starts S-ZR with
              the launcher's HMAC key, bound to a new request entry and connected to lane 1, rewrites
              request_addr in TLLM_EXECUTOR_ATTACH_INFO to S-ZR's and execs the real D1 binary. rank 0
              (S-NX) stays up and idle: control plane, template package and the shutdown sentinel on
              lane 1 still come from the launcher; only the data path goes to S-ZR.
              capstubs.serve_nx --cpp-host-szr-bin writes the wrapper itself at start (write_wrapper
              with snx_env): the wrap step then reads the launch's S-NX config (tier table, origin,
              rank 0 worker cpus) from TRTLLM_CAPKIT_SNX, which the C++ host inherits.

Not replayed (refused in the S-NX config): k_mix, token_source "prompt", first_response_delay_steps,
prefill_token_budget, step_time_replay, max_active, busy_ns_per_step.
"""

import argparse
import copy
import json
import os
import pickle
import pickletools
import random
import secrets
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time

SCHEMA_TEMPLATES = "capkit.szr_templates/1"
SCHEMA_TIMETABLE = "capkit.szr_timetable/1"
SCHEMA_FRAME_SIZES = "capkit.szr_frame_sizes/1"
SCHEMA_RECORD = "capkit.szr_launch/1"
STATES = ("gen", "length", "cancelled")
RID_MARKERS = (0x3F000001, 0x3F000002)
CID_MARKERS = ((1 << 57) | 0xA11CE5, (1 << 57) | 0xB0B0B5)
TOKEN_MARKER_BASE = 0x31A00000
FRONTEND_SHIFT = 57
UNSUPPORTED_SNX = {"k_mix": [], "token_source": "fixed", "first_response_delay_steps": 0,
                   "prefill_token_budget": 0, "step_time_replay": "", "max_active": 0,
                   "busy_ns_per_step": 0}


class LaunchError(Exception):
    pass


# ---- pickle cutting -----------------------------------------------------------------------------
def strip_frames(data):
    """The pickle stream without FRAME opcodes (frames are an I/O hint; loads does not need them)."""
    ops = list(pickletools.genops(data))
    out = bytearray()
    for i, (op, _arg, pos) in enumerate(ops):
        end = ops[i + 1][2] if i + 1 < len(ops) else len(data)
        if op.name != "FRAME":
            out += data[pos:end]
    return bytes(out)


def _ops(data):
    ops = list(pickletools.genops(data))
    return [(op.name, arg, pos, (ops[i + 1][2] if i + 1 < len(ops) else len(data)))
            for i, (op, arg, pos) in enumerate(ops)]


def cut_elements(a, b, rids, cids, token_markers, result_bytes):
    """Head bytes of a and tail bytes of b (pickled in the list [a, b]), each with its request_id /
    client_id slots [pos, len, kind] and token offsets. result_bytes: the two `_result` bytes."""
    one = strip_frames(pickle.dumps([a]))
    two = strip_frames(pickle.dumps([a, b]))
    if not (one.startswith(b"\x80\x04]\x94") and one.endswith(b"a.")):
        raise LaunchError("unexpected single-item list pickle")
    head = one[4:-2]
    pre = b"\x80\x04]\x94(" + head
    if not (two.startswith(pre) and two.endswith(b"e.")):
        raise LaunchError("the head element pickles differently inside a two-item list")
    tail = two[len(pre):-2]
    out = []
    for idx, (base, body) in enumerate(((5, head), (len(pre), tail))):
        slots = []
        for kind, marker in ((0, rids[idx]), (1, cids[idx])):
            hits = [(pos - base, end - pos) for name, arg, pos, end in _ops(two)
                    if base <= pos < base + len(body) and arg == marker
                    and name in ("BININT", "BININT1", "BININT2", "LONG1", "LONG4")]
            if len(hits) != 1:
                raise LaunchError("marker %#x found %d times" % (marker, len(hits)))
            slots.append([hits[0][0], hits[0][1], kind])
        rb = result_bytes[idx]
        if body.count(rb) != 1:
            raise LaunchError("the _result bytes are not found exactly once in the element")
        toks = []
        for t in token_markers:
            enc = struct.pack("<i", t)
            if body.count(enc) != 1 or rb.count(enc) != 1:
                raise LaunchError("token marker %#x is not unique" % t)
            toks.append(body.index(enc))
        out.append({"hex": body.hex(), "slots": slots, "tokens": toks})
    return out[0], out[1]


def memo_profile(element_bytes):
    """(MEMOIZE count, max BINGET / LONG_BINGET index) of one element."""
    n, hi = 0, -1
    for name, arg, _pos, _end in _ops(b"\x80\x04" + element_bytes + b"."):
        if name == "MEMOIZE":
            n += 1
        elif name in ("BINGET", "LONG_BINGET"):
            hi = max(hi, arg)
    return n, hi


# ---- Python model of the C++ frame assembly (szrFrame.cpp) -----------------------------------
def pickle_int(v):
    if v < 0:
        raise ValueError("negative")
    if v <= 0xFF:
        return b"K" + bytes([v])
    if v <= 0xFFFF:
        return b"M" + struct.pack("<H", v)
    if v <= 0x7FFFFFFF:
        return b"J" + struct.pack("<i", v)
    n = v.bit_length() // 8 + 1
    return b"\x8a" + bytes([n]) + v.to_bytes(n, "little")


def emit_element(t, rid, cid, tokens):
    body = bytearray(bytes.fromhex(t["hex"]))
    for i, off in enumerate(t["tokens"]):
        body[off:off + 4] = struct.pack("<i", tokens[i])
    out, src = bytearray(), 0
    for pos, ln, kind in sorted(t["slots"]):
        out += body[src:pos] + pickle_int(rid if kind == 0 else cid)
        src = pos + ln
    return bytes(out + body[src:])


def assemble_py(templates, specs, token_ids):
    """specs: [(rid, cid, n, state, first_token_index)] -> the frame payload (no HMAC)."""
    shapes = {(s["n"], s["state"]): s for s in templates["shapes"]}
    body = bytearray(b"]\x94")
    count = len(specs)
    for i, (rid, cid, n, state, first) in enumerate(specs):
        if count > 1 and i % 1000 == 0:
            body += b"("
        t = shapes[(n, state)]["head" if i == 0 else "tail"]
        toks = [token_ids[(first + j) % len(token_ids)] for j in range(n)]
        body += emit_element(t, rid, cid, toks)
        if count == 1:
            body += b"a"
        elif i % 1000 == 999 or i + 1 == count:
            body += b"e"
    body += b"."
    return b"\x80\x04\x95" + struct.pack("<Q", len(body)) + bytes(body)

# ---- response factories ---------------------------------------------------------------------------
class ProductFactory:
    """Real LlmResponse objects through S-NX's ProductAdapter on executor requests made by the
    product worker's BaseWorker._enqueue_request (worker attributes stubbed: no LoRA, an engine
    that records the executor request, max_seq_len)."""
    name = "product"

    def __init__(self, request_template=None, max_seq_len=131072):
        import dataclasses  # noqa: PLC0415

        from tensorrt_llm.executor.base_worker import BaseWorker  # noqa: PLC0415
        from tensorrt_llm.executor.request import GenerationRequest  # noqa: PLC0415
        from tensorrt_llm.executor.utils import namespace_client_id  # noqa: PLC0415
        from tensorrt_llm.sampling_params import SamplingParams  # noqa: PLC0415

        from capstubs.snx_product import ProductAdapter  # noqa: PLC0415
        self.ad = ProductAdapter()
        self._enqueue = BaseWorker._enqueue_request
        self.max_seq_len = max_seq_len
        names = [f.name for f in dataclasses.fields(SamplingParams)]
        self.max_tokens_index = names.index("max_tokens")
        probe = SamplingParams(max_tokens=12345)
        red = probe.__reduce_ex__(4)
        if red[1][0][self.max_tokens_index] != 12345:
            raise LaunchError("SamplingParams.__reduce_ex__ does not carry max_tokens at index %d"
                              % self.max_tokens_index)
        if request_template:
            with open(request_template, "rb") as f:
                data = f.read()
            self.request = pickle.loads(data)
            self.request_source = {"kind": "template", "path": os.path.abspath(request_template),
                                   "sha256": _sha256(data)}
        else:
            self.request = GenerationRequest([1, 2, 3, 4], SamplingParams(max_tokens=16,
                                                                          detokenize=False),
                                             streaming=True)
            self.request.set_id(namespace_client_id(1, 1))
            self.request_source = {"kind": "synthetic"}
        self.versions = _versions()

    def executor_request(self, gen_request, client_id=None):
        class _Engine:
            kv_cache_transceiver = None
            last = None

            def enqueue_request(self, r, result_wait_queue=None):
                self.last = r
                return 1

        class _Worker:
            pass

        w = _Worker()
        w._lora_manager = None
        w.engine = _Engine()
        w.max_seq_len = self.max_seq_len
        gr = gen_request
        if client_id is not None:
            gr = copy.deepcopy(gen_request)
            gr.id = client_id
        self._enqueue(w, gr)
        return w.engine.last

    def response(self, rid, cid, tokens, state):
        llm = self.ad.convert(rid, self.executor_request(self.request, cid))
        if tokens:
            self.ad.add_tokens(llm, list(tokens), 1)
        if state != "gen":
            self.ad.finish(llm, state == "cancelled")
        resp = self.ad.create_response(llm)
        if resp is None:
            raise LaunchError("create_response() returned None for %d tokens, %s" % (len(tokens),
                                                                                      state))
        return resp

    @staticmethod
    def result_bytes(resp):
        return resp.result._result


class FakeResult:
    def __init__(self, result, is_final):
        self._result = result
        self.is_final = is_final


class FakeResponse:
    def __init__(self, request_id, result, client_id):
        self.request_id = request_id
        self.error_msg = None
        self.result = result
        self.client_id = client_id


class FakeFactory:
    """Product-free stand-in with the same shape (tests on the VM)."""
    name = "fake"
    max_tokens_index = 3
    request_source = {"kind": "fake"}
    versions = {}

    def response(self, rid, cid, tokens, state):
        final = state != "gen"
        reason = {"gen": 0, "length": 3, "cancelled": 5}[state]
        res = (bytes([final]) + struct.pack("<QQ", 1, len(tokens))
               + b"".join(struct.pack("<i", t) for t in tokens) + struct.pack("<Qi", 1, reason)
               + b"\x00" * 15)
        return FakeResponse(rid, FakeResult(res, final), cid)

    @staticmethod
    def result_bytes(resp):
        return resp.result._result


def _sha256(data):
    import hashlib  # noqa: PLC0415
    return hashlib.sha256(data).hexdigest()


def _versions():
    out = {"python": sys.version.split()[0], "pickle_default_protocol": pickle.DEFAULT_PROTOCOL}
    try:
        import tensorrt_llm  # noqa: PLC0415
        out["tensorrt_llm"] = getattr(tensorrt_llm, "__version__", None)
        out["tensorrt_llm_file"] = tensorrt_llm.__file__
    except ImportError:
        pass
    return out


def make_templates(factory, nmax, self_check_frames=64, seed=0):
    shapes = []
    head_memo = None
    for state in STATES:
        for n in range(0 if state == "cancelled" else 1, nmax + 1):
            marks = [TOKEN_MARKER_BASE + i for i in range(n)]
            a = factory.response(RID_MARKERS[0], CID_MARKERS[0], marks, state)
            b = factory.response(RID_MARKERS[1], CID_MARKERS[1], marks, state)
            head, tail = cut_elements(a, b, RID_MARKERS, CID_MARKERS, marks,
                                      (factory.result_bytes(a), factory.result_bytes(b)))
            hm, _ = memo_profile(bytes.fromhex(head["hex"]))
            _, tail_hi = memo_profile(bytes.fromhex(tail["hex"]))
            if head_memo is None:
                head_memo = hm
            if hm != head_memo or tail_hi > head_memo:
                raise LaunchError("memo layout differs between shapes (head %d vs %d, tail ref %d)"
                                  % (hm, head_memo, tail_hi))
            shapes.append({"n": n, "state": state, "head": head, "tail": tail})
    doc = {"schema": SCHEMA_TEMPLATES, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "factory": factory.name, "request_source": factory.request_source,
           "versions": factory.versions, "max_tokens_index": factory.max_tokens_index,
           "nmax": nmax, "head_memo_count": head_memo + 1, "shapes": shapes}
    doc["self_check"] = self_check(doc, factory, self_check_frames, seed)
    return doc


def self_check(templates, factory, frames, seed):
    """Random frames assembled by the Python model must load to objects that pickle exactly like
    freshly made factory responses (element by element)."""
    rng = random.Random(seed)
    nmax = templates["nmax"]
    token_ids = [1000, 2000, 3000, 4000, 7, 0x7FFFFFFF]
    checked = 0
    for f in range(frames):
        count = rng.choice([1, 2, 3, 17, 1001] if f % 16 == 0 else [1, 2, 5, 33])
        specs = []
        for _ in range(count):
            state = rng.choice(STATES)
            n = rng.randint(0 if state == "cancelled" else 1, nmax)
            rid = rng.choice([1, 255, 256, 65535, 65536, 2**31 - 1, 2**31, 2**40, 2**63 + 5])
            cid = (1 << FRONTEND_SHIFT) | rng.randrange(1 << FRONTEND_SHIFT)
            specs.append((rid, cid, n, state, rng.randrange(100)))
        got = pickle.loads(assemble_py(templates, specs, token_ids))
        if len(got) != count:
            raise LaunchError("self-check: %d elements loaded, %d assembled" % (len(got), count))
        for obj, (rid, cid, n, state, first) in zip(got, specs):
            toks = [token_ids[(first + j) % len(token_ids)] for j in range(n)]
            want = factory.response(rid, cid, toks, state)
            if pickle.dumps(obj) != pickle.dumps(want):
                raise LaunchError("self-check: element (rid %d, n %d, %s) differs" % (rid, n, state))
            checked += 1
    return {"frames": frames, "elements": checked, "seed": seed}

# ---- S-NX config -> S-ZR plan -------------------------------------------------------------------
def check_replayable(cfg):
    bad = [k for k, ok in UNSUPPORTED_SNX.items() if getattr(cfg, k) != ok]
    if bad:
        raise LaunchError("S-ZR does not replay these S-NX settings: %s" % ", ".join(sorted(bad)))


def timetable_doc(cfg):
    from capcommon.schedule import BeatSchedule  # noqa: PLC0415
    from capcommon.tiers import TierTable  # noqa: PLC0415

    from capstubs.null_executor import tier_shape  # noqa: PLC0415
    tt = TierTable.load(cfg.tier_table)
    tiers = list(tt.tiers)
    if not tiers or any(t.beat_ns <= 0 for t in tiers):
        raise LaunchError("every tier needs beat_ns > 0 (the step period)")
    k_of, s = tier_shape(tt, cfg)
    bs = BeatSchedule(tt)
    return {"schema": SCHEMA_TIMETABLE, "tier_table": os.path.abspath(cfg.tier_table),
            "tier_table_sha256": getattr(tt, "sha256", None), "origin_ns": cfg.origin_ns,
            "end_rel_ns": tt.end_ns, "pre_beat_ns": cfg.outside_beat_ns or tiers[0].beat_ns,
            "post_beat_ns": cfg.outside_beat_ns or tiers[-1].beat_ns, "stream_interval": s,
            "segments": [list(x) for x in bs.segments()],
            "beat_of_tier": {str(t.seq): t.beat_ns for t in tiers},
            "k_of_tier": {str(k): v for k, v in sorted(k_of.items())}}


def required_tokens(cfg, timetable):
    ks = [cfg.tokens_per_step] + (list(timetable["k_of_tier"].values()) if timetable else [])
    s = (timetable["stream_interval"] if timetable else cfg.stream_interval) or 1
    return max(max(cfg.first_step_tokens or k, s * k) for k in ks)


def frame_sizes_doc(snx_dir, tier_seq=None, cfg=None):
    from capstubs.null_executor import read_records  # noqa: PLC0415
    lo = hi = None
    if tier_seq is not None:
        from capcommon.tiers import TierTable  # noqa: PLC0415
        tt = TierTable.load(cfg.tier_table)
        t = [x for x in tt.tiers if x.seq == tier_seq]
        if not t:
            raise LaunchError("tier %d is not in the tier table" % tier_seq)
        lo, hi = cfg.origin_ns + t[0].start_ns, cfg.origin_ns + t[0].start_ns + t[0].duration_ns
    hist = {}
    files = sorted(f for f in os.listdir(snx_dir) if f.endswith(".takes.rec"))
    if not files:
        raise LaunchError("no snx.*.takes.rec under %s" % snx_dir)
    for f in files:
        for r in read_records(os.path.join(snx_dir, f), "takes"):
            if lo is not None and not lo <= r["call_ns"] < hi:
                continue
            if r["responses"] > 0:
                hist[r["responses"]] = hist.get(r["responses"], 0) + 1
    if not hist:
        raise LaunchError("no takes in the selected window")
    return {"schema": SCHEMA_FRAME_SIZES, "source": os.path.abspath(snx_dir), "tier_seq": tier_seq,
            "takes": sum(hist.values()), "sizes": [[k, v] for k, v in sorted(hist.items())]}


# ---- process start -------------------------------------------------------------------------------
def write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w") as f:
        f.write(text)


def szr_argv(a, cfg, out_dir, templates, timetable_path, request_addr, result_addr):
    argv = [a.szr_bin, "--request-addr", request_addr, "--result-addr", result_addr,
            "--templates", templates, "--token-ids", ",".join(str(t) for t in cfg.token_ids),
            "--tokens-per-step", str(cfg.tokens_per_step),
            "--first-step-tokens", str(cfg.first_step_tokens), "--max-seq-len", str(a.max_seq_len),
            "--idle-restart", cfg.idle_restart, "--report", os.path.join(out_dir, "szr.report.jsonl"),
            "--report-interval-ms", str(a.report_interval_ms),
            "--summary", os.path.join(out_dir, "szr.summary.json"),
            "--ready-file", os.path.join(out_dir, "szr.ready.json"), "--seed", str(a.seed),
            "--max-per-frame", str(a.max_per_frame), "--sndhwm", str(a.sndhwm)]
    if timetable_path:
        argv += ["--timetable", timetable_path]
    else:
        argv += ["--beat-ns", str(cfg.beat_ns), "--stream-interval", str(cfg.stream_interval or 1)]
    if a.frame_sizes:
        argv += ["--frame-sizes", a.frame_sizes]
    if a.dump:
        os.makedirs(os.path.join(out_dir, "dump"), exist_ok=True)
        argv += ["--dump-dir", os.path.join(out_dir, "dump"), "--dump-every", str(a.dump_every),
                 "--dump-reservoir", str(a.dump_reservoir)]
    if a.debug_corrupt_client_id:
        argv.append("--debug-corrupt-client-id")
    for flag, v in (("--busy-ns-per-step", a.busy_ns_per_step),
                    ("--busy-ns-per-frame", a.busy_ns_per_frame), ("--duration-s", a.duration_s)):
        if v:
            argv += [flag, str(v)]
    if a.cpus:
        argv += ["--cpus", a.cpus]
    return argv


def prepare_inputs(a, cfg, out_dir, request_template=None):
    """(templates path, timetable path or None): writes them into out_dir unless --templates."""
    check_replayable(cfg)
    tt = timetable_doc(cfg) if cfg.tier_table else None
    tt_path = None
    if tt:
        tt_path = os.path.join(out_dir, "szr.timetable.json")
        _write_json(tt_path, tt)
    need = required_tokens(cfg, tt)
    if a.templates:
        with open(a.templates, "r", encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("nmax", -1) < need:
            raise LaunchError("templates nmax %s < %d needed by this plan" % (doc.get("nmax"), need))
        return a.templates, tt_path
    factory = ProductFactory(request_template, a.max_seq_len)
    doc = make_templates(factory, max(need, a.nmax), a.self_check_frames)
    path = os.path.join(out_dir, "szr.templates.json")
    _write_json(path, doc)
    return path, tt_path


def _write_json(path, doc):
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def wait_ready(proc, path, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return
        if proc.poll() is not None:
            raise LaunchError("S-ZR exited with %s before it was ready" % proc.returncode)
        time.sleep(0.05)
    proc.kill()
    raise LaunchError("S-ZR not ready after %.0f s" % timeout_s)


def start_szr(argv, key_hex, ready_path, timeout_s, log_path):
    env = dict(os.environ)
    env["SZR_HMAC_KEY_HEX"] = key_hex
    log = open(log_path, "ab")  # noqa: SIM115 -- owned by the child for its lifetime
    proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
    wait_ready(proc, ready_path, timeout_s)
    return proc

def cmd_run(a, cfg):
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    templates, tt_path = prepare_inputs(a, cfg, out_dir, a.request_template)
    ipc_dir = a.ipc_dir or tempfile.mkdtemp(prefix="szr-")
    key = secrets.token_hex(32)
    request_addr = "ipc://%s/request.sock" % ipc_dir
    lanes = ["ipc://%s/result_0.sock" % ipc_dir, "ipc://%s/result_1.sock" % ipc_dir]
    attach = {"mode": "classic", "request_addr": request_addr, "result_addrs": lanes,
              "hmac_key": key, "model_world_size": 1, "rpc_addr": None, "rpc_hmac_key": None}
    write_private(os.path.join(out_dir, "hmac_key.hex"), key + "\n")
    write_private(os.path.join(out_dir, "attach.json"), json.dumps(attach) + "\n")
    write_private(os.path.join(out_dir, "d1.env"),
                  "export TLLM_EXECUTOR_ATTACH_INFO='%s'\nexport TLLM_EXECUTOR_FRONTEND_ID=1\n"
                  % json.dumps(attach))
    argv = szr_argv(a, cfg, out_dir, templates, tt_path, request_addr, lanes[1])
    _write_json(os.path.join(out_dir, "szr_launch.json"),
                {"schema": SCHEMA_RECORD, "mode": "run", "argv": argv, "snx_config": cfg.to_doc(),
                 "templates": templates, "timetable": tt_path, "ipc_dir": ipc_dir,
                 "attach_file": os.path.join(out_dir, "attach.json")})
    proc = start_szr(argv, key, os.path.join(out_dir, "szr.ready.json"), a.ready_timeout_s,
                     os.path.join(out_dir, "szr.log"))

    def forward(signum, _frame):
        proc.send_signal(signum)

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    print("szr_launch: S-ZR pid %d ready; attach info in %s" % (proc.pid,
                                                               os.path.join(out_dir, "d1.env")),
          flush=True)
    return proc.wait()


def write_wrapper(out_dir, d1_bin, argv, snx_env=False):
    """Write <out_dir>/wrap.json and the executable <out_dir>/szr-d1-wrap; returns its path.
    snx_env: the wrap step takes the S-NX config from TRTLLM_CAPKIT_SNX (set by capstubs.serve_nx
    for the launcher and inherited by the C++ host it starts) instead of argv's config options."""
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    build_parser().parse_args(["run"] + list(argv))  # refuse bad S-ZR options before the launch
    tools = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _write_json(os.path.join(out_dir, "wrap.json"),
                {"schema": SCHEMA_RECORD, "mode": "wrap", "argv": list(argv), "d1_bin": d1_bin,
                 "snx_env": bool(snx_env)})
    path = os.path.join(out_dir, "szr-d1-wrap")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\n# S-ZR wrapper for TLLM_OPENENGINE_CPP_HOST_BIN (capstubs.szr_launch)\n"
                "PYTHONPATH=%s${PYTHONPATH:+:$PYTHONPATH} exec %s -m capstubs.szr_launch wrap "
                "--wrap-config %s -- \"$@\"\n" % (tools, sys.executable,
                                                  os.path.join(out_dir, "wrap.json")))
    os.chmod(path, 0o755)
    return path


def cmd_make_wrapper(a, argv_rest):
    print(write_wrapper(a.out_dir, a.d1_bin, argv_rest))
    return 0


def config_from_env():
    """The S-NX config capstubs.serve_nx put into TRTLLM_CAPKIT_SNX (tier table, origin, out_dir,
    worker cpus of the launch), checked like resolve()."""
    from capstubs.null_executor import ENV_VAR, SnxConfig, tier_shape  # noqa: PLC0415
    text = os.environ.get(ENV_VAR)
    if not text:
        raise LaunchError("%s is not set (the wrapper must run under capstubs.serve_nx)" % ENV_VAR)
    cfg = SnxConfig.from_doc(json.loads(text))
    if cfg.tier_table:
        from capcommon.tiers import TierTable  # noqa: PLC0415
        tier_shape(TierTable.load(cfg.tier_table), cfg)
    return cfg


def d1_flag(args, flag):
    for i, x in enumerate(args):
        if x == flag and i + 1 < len(args):
            return args[i + 1]
        if x.startswith(flag + "="):
            return x[len(flag) + 1:]
    return None


def cmd_wrap(wrap_config, d1_args):
    with open(wrap_config, "r", encoding="utf-8") as f:
        rec = json.load(f)
    a = build_parser().parse_args(["run"] + rec["argv"])
    cfg = config_from_env() if rec.get("snx_env") else resolve(a)
    info = json.loads(os.environ["TLLM_EXECUTOR_ATTACH_INFO"])
    fid = int(os.environ["TLLM_EXECUTOR_FRONTEND_ID"])
    out_dir = os.path.abspath(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    template_dir = d1_flag(d1_args, "--template")
    req_tmpl = os.path.join(template_dir, "request_template.pkl") if template_dir else None
    templates, tt_path = prepare_inputs(a, cfg, out_dir, req_tmpl)
    if template_dir:
        with open(os.path.join(template_dir, "fields.json"), "r", encoding="utf-8") as f:
            fields = json.load(f)
        idx = [x["index"] for x in fields["sampling_params"]["fields"] if x["name"] == "max_tokens"]
        with open(templates, "r", encoding="utf-8") as f:
            have = json.load(f)["max_tokens_index"]
        if idx != [have]:
            raise LaunchError("max_tokens index: template package %s, templates %d" % (idx, have))
    ipc_dir = a.ipc_dir or tempfile.mkdtemp(prefix="szr-")
    request_addr = "ipc://%s/request.sock" % ipc_dir
    lane = info["result_addrs"][fid]
    argv = szr_argv(a, cfg, out_dir, templates, tt_path, request_addr, lane) + ["--exit-with-parent"]
    _write_json(os.path.join(out_dir, "szr_launch.json"),
                {"schema": SCHEMA_RECORD, "mode": "wrap", "argv": argv, "snx_config": cfg.to_doc(),
                 "snx_env": bool(rec.get("snx_env")),
                 "templates": templates, "timetable": tt_path, "ipc_dir": ipc_dir,
                 "rank0_request_addr": info["request_addr"], "lane": lane, "frontend_id": fid,
                 "d1_bin": rec["d1_bin"], "d1_args": d1_args})
    write_private(os.path.join(out_dir, "hmac_key.hex"), info["hmac_key"] + "\n")
    start_szr(argv, info["hmac_key"], os.path.join(out_dir, "szr.ready.json"), a.ready_timeout_s,
              os.path.join(out_dir, "szr.log"))
    info["request_addr"] = request_addr
    os.environ["TLLM_EXECUTOR_ATTACH_INFO"] = json.dumps(info)
    if a.d1_cpus:
        from capstubs.affinity import pin_all_tasks  # noqa: PLC0415
        pin_all_tasks([int(c) for c in _cpu_list(a.d1_cpus)])
    os.execv(rec["d1_bin"], [rec["d1_bin"]] + d1_args)


def _cpu_list(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


# ---- command line ---------------------------------------------------------------------------------
def _common(p):
    p.add_argument("--config-json", help="base capkit.snx_config/1 file (as serve_nx)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    p.add_argument("--tier-table")
    p.add_argument("--origin-ns", type=int)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-seq-len", type=int, default=131072,
                   help="the worker's max_seq_len (max_tokens clamp; S-NX default 131072)")


def _szr_opts(p):
    p.add_argument("--szr-bin", help="trtllm-oehost-szr executable")
    p.add_argument("--templates", help="prepared capkit.szr_templates/1 (else prepared here)")
    p.add_argument("--request-template", help="request_template.pkl (run mode)")
    p.add_argument("--nmax", type=int, default=0, help="prepare at least this many tokens/response")
    p.add_argument("--self-check-frames", type=int, default=64)
    p.add_argument("--ipc-dir", help="directory for the ipc sockets (default: a new /tmp dir)")
    p.add_argument("--cpus", help="S-ZR cpu list, e.g. 8-10")
    p.add_argument("--d1-cpus", help="wrap: pin D1 to these cpus before exec")
    p.add_argument("--max-per-frame", type=int, default=0)
    p.add_argument("--frame-sizes", help="capkit.szr_frame_sizes/1 file")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--busy-ns-per-step", type=int, default=0)
    p.add_argument("--busy-ns-per-frame", type=int, default=0)
    p.add_argument("--report-interval-ms", type=int, default=1000)
    p.add_argument("--dump", action="store_true", help="write V-25 samples to <out-dir>/dump")
    p.add_argument("--dump-every", type=int, default=1000)
    p.add_argument("--dump-reservoir", type=int, default=200)
    p.add_argument("--debug-corrupt-client-id", action="store_true")
    p.add_argument("--duration-s", type=float, default=0)
    p.add_argument("--sndhwm", type=int, default=1000)
    p.add_argument("--ready-timeout-s", type=float, default=60.0)


def build_parser():
    p = argparse.ArgumentParser(prog="python -m capstubs.szr_launch", allow_abbrev=False,
                                description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    _common(q)
    q.add_argument("--request-template")
    q.add_argument("--nmax", type=int, default=0)
    q.add_argument("--self-check-frames", type=int, default=64)
    q.add_argument("--factory", choices=("product", "fake"), default="product")
    q = sub.add_parser("timetable")
    _common(q)
    q = sub.add_parser("frame-sizes")
    _common(q)
    q.add_argument("--snx-dir", required=True)
    q.add_argument("--tier-seq", type=int)
    for name in ("run", "make-wrapper"):
        q = sub.add_parser(name)
        _common(q)
        _szr_opts(q)
        if name == "make-wrapper":
            q.add_argument("--d1-bin", required=True)
    return p


def resolve(a):
    from capstubs.serve_nx import resolve_config  # noqa: PLC0415
    ns = argparse.Namespace(config_json=a.config_json, set=a.set, tier_table=a.tier_table,
                            origin_ns=a.origin_ns, out_dir=None)
    return resolve_config(ns)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["wrap"]:
        if len(argv) < 4 or argv[1] != "--wrap-config" or argv[3] != "--":
            print("usage: szr_launch wrap --wrap-config FILE -- <D1 args>", file=sys.stderr)
            return 2
        cmd_wrap(argv[2], argv[4:])
        return 0
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "make-wrapper":
            raw = argv[1:]
            i = raw.index("--d1-bin")
            return cmd_make_wrapper(a, raw[:i] + raw[i + 2:])
        cfg = resolve(a)
        os.makedirs(a.out_dir, exist_ok=True)
        if a.cmd == "prepare":
            check_replayable(cfg)
            tt = timetable_doc(cfg) if cfg.tier_table else None
            need = max(required_tokens(cfg, tt), a.nmax)
            factory = (FakeFactory() if a.factory == "fake"
                       else ProductFactory(a.request_template, a.max_seq_len))
            doc = make_templates(factory, need, a.self_check_frames)
            _write_json(os.path.join(a.out_dir, "szr.templates.json"), doc)
            print("templates: nmax %d, %d shapes, self-check %s" % (need, len(doc["shapes"]),
                                                                     doc["self_check"]))
            return 0
        if a.cmd == "timetable":
            _write_json(os.path.join(a.out_dir, "szr.timetable.json"), timetable_doc(cfg))
            return 0
        if a.cmd == "frame-sizes":
            _write_json(os.path.join(a.out_dir, "szr.frame_sizes.json"),
                        frame_sizes_doc(a.snx_dir, a.tier_seq, cfg))
            return 0
        if not a.szr_bin:
            raise LaunchError("--szr-bin is required")
        return cmd_run(a, cfg)
    except LaunchError as e:
        print("szr_launch: %s" % e, file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
