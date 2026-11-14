# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hardware profile collector (DATA-AND-ANALYSIS.md 8.1; INTERFACES.md 5.5.3).

Produces exactly the keys of capcommon.hwprofile (IDENTITY_FIELDS + OBSERVED_FIELDS). A value
that cannot be read is {"unreadable": "<reason>"}; a missing command is never an error.
profile_sha256 = capcommon.hwprofile.profile_sha256 (identity fields only; node name, partition,
time, free memory, current frequency, disk free space and link counters are observed fields and
never hashed).

Raw evidence that has no profile key (full lscpu / lscpu -e / numactl -H / free -b / lsblk / df
text, iommu group count, DMI product name, sibling table, scontrol node text) goes into a
separate hw_raw.json (capkit.hw_raw/1) whose sha256 is recorded in hw_profile.json.

    python -m capenv.hwprofile <identity> --out env/hw_profile.json --raw-out env/hw_raw.json
        --local-dir DIR --chassis-source {slurm_view,dmi} --nvidia-smi PATH
        [--slurm-view-file PATH] [--image-sha256 HEX] [--code-baseline SHA40]
        [--container-runtime TEXT] [--timeout-s S]
"""

import argparse
import json
import os
import socket
import sys
import time

from capcommon import exitcodes
from capcommon.cli import add_identity_args
from capcommon.clock import wall_iso
from capcommon.exitcodes import ConfigError
from capcommon.hashing import is_git_sha, is_sha256_hex, sha256_file
from capcommon.hwprofile import (IDENTITY_FIELDS, OBSERVED_FIELDS, SCHEMA, profile_sha256,
                                 unreadable, validate_profile)
from capcommon.identity import Identity
from capcommon.jsonio import read_json, write_json_stamped

from . import gpu as gpu_mod, procfs, runcmd, topology

RAW_SCHEMA = "capkit.hw_raw/1"
GIB = 1 << 30


def _c_env():
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    return env


def parse_lscpu_kv(text):
    out = {}
    for line in (text or "").splitlines():
        k, sep, v = line.partition(":")
        if sep:
            out[k.strip()] = v.strip()
    return out


def parse_kv_tokens(text):
    """scontrol 'Key=Value Key2=Value2' text -> dict (case kept; later keys win)."""
    out = {}
    for tok in (text or "").split():
        k, sep, v = tok.partition("=")
        if sep and k:
            out[k] = v
    return out


def _slurm_items(path):
    """{id: item} from slurm_view.json, or {} when no file; raises ConfigError on bad schema."""
    if not path:
        return {}
    doc = read_json(path)
    if not isinstance(doc, dict) or doc.get("schema") != "capkit.slurm_view/1":
        raise ConfigError("%s is not a capkit.slurm_view/1 file" % path)
    return {it.get("id"): it for it in doc.get("items", [])}


class Collector:
    def __init__(self, local_dir, chassis_source, slurm_view_file=None, image_sha256=None,
                 code_baseline=None, container_runtime=None, timeout_s=30.0,
                 proc_root=procfs.PROC, sys_root=procfs.SYS, runner=runcmd.run,
                 nvidia_smi=None):
        self.local_dir = local_dir
        self.chassis_source = chassis_source
        self.slurm = _slurm_items(slurm_view_file)
        self.image_sha256 = image_sha256
        self.code_baseline = code_baseline
        self.container_runtime = container_runtime
        self.timeout_s = timeout_s
        self.proc_root, self.sys_root = proc_root, sys_root
        self.runner = runner
        self.nvidia_smi = nvidia_smi  # None -> GPU keys unreadable ("nvidia-smi path not given")
        self.prof = {}
        self.src = {}
        self.raw = {}

    def run(self, argv):
        return self.runner(argv, self.timeout_s, env=_c_env())

    def put(self, key, value, source):
        self.prof[key] = value
        self.src[key] = source if not (isinstance(value, dict) and "unreadable" in value) \
            else "unreadable"

    def _raw_cmd(self, name, argv):
        rec = self.run(argv)
        self.raw[name] = {k: rec.get(k) for k in ("cmd", "rc", "stdout", "stdout_sha256",
                                                  "reason")}
        return rec

    # -- CPU ---------------------------------------------------------------------------------
    def cpu(self):
        rec = self._raw_cmd("lscpu", ["lscpu"])
        self._raw_cmd("lscpu_e", ["lscpu", "-e"])
        prec = self._raw_cmd("lscpu_p", list(topology.LSCPU_ARGV))
        kv = parse_lscpu_kv(rec["stdout"]) if runcmd.ok(rec) else {}
        why = runcmd.why(rec)

        def from_kv(key, field, conv=str):
            if field in kv:
                try:
                    self.put(key, conv(kv[field]), "container:lscpu")
                    return
                except ValueError:
                    pass
            self.put(key, unreadable("lscpu %s: %s" % (field, why if not kv else "missing")),
                     None)

        from_kv("cpu_arch", "Architecture")
        from_kv("cpu_model", "Model name")
        from_kv("sockets", "Socket(s)", int)
        from_kv("cores_per_socket", "Core(s) per socket", int)
        from_kv("threads_per_core", "Thread(s) per core", int)
        if runcmd.ok(prec):
            try:
                self.raw["topology"] = topology.parse_lscpu_p(prec["stdout"]).summary()
            except ConfigError as e:
                self.raw["topology"] = unreadable(str(e))
        txt, err = procfs.try_read_text(os.path.join(self.proc_root, "cpuinfo"))
        mc = None
        if txt is not None:
            for line in txt.splitlines():
                k, _, v = line.partition(":")
                if k.strip() == "microcode":
                    mc = v.strip()
                    break
        self.put("microcode", mc if mc else unreadable(err or "no microcode line in cpuinfo"),
                 "container:/proc/cpuinfo")
        nodes = self._numa_nodes()
        if nodes:
            self.put("numa_layout", [{"node": n, "cpus": c} for n, c, _m in nodes],
                     "container:/sys/devices/system/node")
            mems = [m for _n, _c, m in nodes]
            self.put("mem_per_numa_gib",
                     [int(round(m / GIB)) if m is not None else None for m in mems]
                     if all(m is not None for m in mems)
                     else unreadable("node meminfo unreadable"), "container:node meminfo")
        else:
            self.put("numa_layout", unreadable("no /sys/devices/system/node/node*"), None)
            self.put("mem_per_numa_gib", unreadable("no NUMA nodes"), None)
        self._raw_cmd("numactl_H", ["numactl", "-H"])
        govs, freqs = set(), {}
        try:
            cpus = procfs.online_cpus(self.sys_root)
        except OSError:
            cpus = []
        for c in cpus:
            base = os.path.join(self.sys_root, "devices/system/cpu/cpu%d/cpufreq" % c)
            g, _ = procfs.try_read_text(os.path.join(base, "scaling_governor"))
            if g is not None:
                govs.add(g.strip())
            f, _ = procfs.try_read_text(os.path.join(base, "scaling_cur_freq"))
            if f is not None and f.strip().isdigit():
                freqs[str(c)] = int(f.strip())
        self.put("cpu_governor", sorted(govs) if govs else
                 unreadable("no cpufreq/scaling_governor readable"), "container:sysfs")
        self.put("cpu_freq_khz", freqs if freqs else
                 unreadable("no cpufreq/scaling_cur_freq readable"), "container:sysfs")

    def _numa_nodes(self):
        base = os.path.join(self.sys_root, "devices/system/node")
        try:
            names = sorted((n for n in os.listdir(base) if n.startswith("node")
                            and n[4:].isdigit()), key=lambda n: int(n[4:]))
        except OSError:
            return []
        out = []
        for n in names:
            cpus, _ = procfs.try_read_text(os.path.join(base, n, "cpulist"))
            mi, _ = procfs.try_read_text(os.path.join(base, n, "meminfo"))
            mem = None
            for line in (mi or "").splitlines():
                if "MemTotal:" in line:
                    mem = int(line.split("MemTotal:")[1].split()[0]) * 1024
            out.append((int(n[4:]), (cpus or "").strip(), mem))
        return out

    # -- memory, GPU, NICs, disk -------------------------------------------------------------
    def memory(self):
        txt, err = procfs.try_read_text(os.path.join(self.proc_root, "meminfo"))
        kv = {}
        for line in (txt or "").splitlines():
            k, _, v = line.partition(":")
            if v.strip().split():
                kv[k.strip()] = int(v.split()[0])
        self.put("mem_total_gib", int(round(kv["MemTotal"] * 1024 / GIB)) if "MemTotal" in kv
                 else unreadable(err or "no MemTotal"), "container:/proc/meminfo")
        self.put("mem_available_kib", kv.get("MemAvailable", unreadable(err or "no MemAvailable")),
                 "container:/proc/meminfo")
        self._raw_cmd("free_b", ["free", "-b"])

    def gpus(self):
        rec, gpus = gpu_mod.query_gpus(self.timeout_s, runner=lambda a, t: self.run(a),
                                       nvsmi=self.nvidia_smi)
        self.raw["nvidia_smi_query_gpu"] = {k: rec.get(k) for k in ("cmd", "rc", "stdout",
                                                                     "reason")}
        if runcmd.ok(rec) and rec.get("reason") is None and gpus:
            names = sorted({g["name"] for g in gpus})
            self.put("gpu_model", names[0] if len(names) == 1 else names, "container:nvidia-smi")
            self.put("gpu_count", len(gpus), "container:nvidia-smi")
            mems = [g["mem_total_mib"] for g in gpus]
            self.put("gpu_mem_mib", sorted(mems) if all(isinstance(m, (int, float)) for m in mems)
                     else unreadable("nvidia-smi memory.total unreadable: %r" % (mems,)),
                     "container:nvidia-smi")
            drivers = sorted({g["driver"] for g in gpus})
            self.put("gpu_driver", drivers[0] if len(drivers) == 1 else drivers,
                     "container:nvidia-smi")
        else:
            why = "nvidia-smi: %s" % (runcmd.why(rec) if not (runcmd.ok(rec) and not gpus
                                                           and rec.get("reason") is None)
                                      else "listed no GPU")
            for k in ("gpu_model", "gpu_count", "gpu_mem_mib", "gpu_driver"):
                self.put(k, unreadable(why), None)

    def nics(self):
        self.pci_nics()
        base = os.path.join(self.sys_root, "class/net")
        nics, link = [], {}
        try:
            names = sorted(os.listdir(base))
        except OSError as e:
            self.put("netdevs", unreadable("cannot list %s: %s" % (base, e)), None)
            self.put("nic_link", unreadable("cannot list %s" % base), None)
            return
        for n in names:
            d = os.path.join(base, n)
            ent = {}
            for f in ("operstate", "statistics/rx_bytes", "statistics/tx_bytes"):
                v, _ = procfs.try_read_text(os.path.join(d, f))
                ent[f.split("/")[-1]] = (int(v) if v is not None and v.strip().isdigit()
                                         else (v.strip() if v is not None else None))
            link[n] = ent
            if not os.path.exists(os.path.join(d, "device")):
                continue  # virtual interface (lo, bridges, veth): not a NIC of the chassis
            ven, _ = procfs.try_read_text(os.path.join(d, "device/vendor"))
            dev, _ = procfs.try_read_text(os.path.join(d, "device/device"))
            spd, serr = procfs.try_read_text(os.path.join(d, "speed"))
            try:
                speed = int(spd.strip()) if spd is not None else None
            except ValueError:
                speed = None
            nics.append({"model": "%s:%s" % ((ven or "?").strip(), (dev or "?").strip()),
                         "speed_mbps": speed if speed is not None and speed >= 0 else
                         unreadable(serr or "speed %r (link down?)" % (spd or "").strip())})
        nics.sort(key=lambda x: json.dumps(x, sort_keys=True))
        self.put("netdevs", nics, "container:/sys/class/net")
        self.put("nic_link", link, "container:/sys/class/net")

    def pci_nics(self):
        """Identity "nics": network controllers (PCI class 0x02xxxx) counted by vendor:device,
        SR-IOV virtual functions (a physfn link) left out. Which network interfaces a container
        sees and their link state are not hardware identity (they go to netdevs / nic_link)."""
        base = os.path.join(self.sys_root, "bus/pci/devices")
        try:
            addrs = sorted(os.listdir(base))
        except OSError as e:
            self.put("nics", unreadable("cannot list %s: %s" % (base, e)), None)
            return
        counts = {}
        for a in addrs:
            d = os.path.join(base, a)
            cls, cerr = procfs.try_read_text(os.path.join(d, "class"))
            if cls is None:
                self.put("nics", unreadable("%s/class: %s" % (a, cerr)), None)
                return
            if not cls.strip().lower().startswith("0x02") or \
                    os.path.lexists(os.path.join(d, "physfn")):
                continue
            ids = []
            for name in ("vendor", "device"):
                v, err = procfs.try_read_text(os.path.join(d, name))
                if v is None or not v.strip():
                    # a guessed model would be a different identity value, not an unreadable one
                    self.put("nics", unreadable("%s/%s: %s" % (a, name, err or "empty")), None)
                    return
                ids.append(v.strip())
            model = "%s:%s" % tuple(ids)
            counts[model] = counts.get(model, 0) + 1
        self.put("nics", [{"model": m, "count": n} for m, n in sorted(counts.items())],
                 "container:/sys/bus/pci")

    def disk(self):
        self._raw_cmd("lsblk", ["lsblk", "-o", "NAME,MAJ:MIN,TYPE,ROTA,TRAN,MODEL,SIZE,MOUNTPOINT"])
        self._raw_cmd("df", ["df", "-P", self.local_dir])
        try:
            sv = os.statvfs(self.local_dir)
            self.put("local_disk_free_bytes", sv.f_bavail * sv.f_frsize, "container:statvfs")
        except OSError as e:
            self.put("local_disk_free_bytes", unreadable(str(e)), None)
        self.put("local_disk_type", self._disk_type(), "container:mountinfo+sysfs")

    def _disk_type(self):
        try:
            real = os.path.realpath(self.local_dir)
            st = os.stat(real)
        except OSError as e:
            return unreadable("local dir: %s" % e)
        txt, err = procfs.try_read_text(os.path.join(self.proc_root, "self/mountinfo"))
        if txt is None:
            return unreadable("mountinfo: %s" % err)
        best = None
        for line in txt.splitlines():
            left, _, right = line.partition(" - ")
            f = left.split()
            if len(f) < 5:
                continue
            mp = f[4].replace("\\040", " ")
            if real == mp or real.startswith(mp.rstrip("/") + "/") or mp == "/":
                if best is None or len(mp) > len(best[0]):
                    best = (mp, right.split()[0] if right.split() else "?")
        fstype = best[1] if best else "?"
        maj, mnr = os.major(st.st_dev), os.minor(st.st_dev)
        blk = os.path.join(self.sys_root, "dev/block/%d:%d" % (maj, mnr))
        if not os.path.exists(blk):
            return "%s;no-block-device" % fstype
        dev = os.path.realpath(blk)
        if os.path.exists(os.path.join(dev, "partition")):
            dev = os.path.dirname(dev)
        rota, _ = procfs.try_read_text(os.path.join(dev, "queue/rotational"))
        model, _ = procfs.try_read_text(os.path.join(dev, "device/model"))
        return "%s;rota=%s;model=%s" % (fstype, (rota or "?").strip(), (model or "?").strip())

    # -- software, chassis, node -------------------------------------------------------------
    def software(self):
        self.put("kernel", os.uname().release, "container:uname")
        self.raw["uname"] = list(os.uname())
        self.put("container_runtime", self.container_runtime if self.container_runtime else
                 unreadable("not passed (--container-runtime)"), "caller")
        self.put("image_sha256", self.image_sha256 if self.image_sha256 else
                 unreadable("not passed (--image-sha256)"), "caller")
        self.put("code_baseline", self.code_baseline if self.code_baseline else
                 unreadable("not passed (--code-baseline)"), "caller")
        try:
            self.raw["iommu_groups_count"] = len(os.listdir(
                os.path.join(self.sys_root, "kernel/iommu_groups")))
        except OSError as e:
            self.raw["iommu_groups_count"] = unreadable(str(e))

    def chassis_and_node(self):
        dmi, derr = procfs.try_read_text(os.path.join(self.sys_root,
                                                      "class/dmi/id/product_name"))
        self.raw["dmi_product_name"] = dmi.strip() if dmi is not None else unreadable(derr)
        node_item, job_item = self.slurm.get(2), self.slurm.get(1)
        self.raw["slurm_view_items"] = sorted(k for k in self.slurm if k is not None)
        if self.chassis_source == "dmi":
            self.put("chassis_model", dmi.strip() if dmi and dmi.strip() else
                     unreadable("dmi product_name: %s" % (derr or "empty")), "container:dmi")
        else:
            kv = parse_kv_tokens(node_item.get("stdout")) if node_item else {}
            val = None
            for k, v in kv.items():
                if k.lower() in ("systemproductname", "system_product_name"):
                    val = v
            if node_item is None:
                why = "no slurm_view item 2 (scontrol show node)"
            elif node_item.get("rc") != 0:
                why = "slurm_view item 2 rc=%s" % node_item.get("rc")
            else:
                why = "no SystemProductName in scontrol show node output"
            self.put("chassis_model", val if val else unreadable(why), "slurm_view")
        jkv = parse_kv_tokens(job_item.get("stdout")) if job_item and job_item.get("rc") == 0 \
            else {}
        self.put("partition", jkv.get("Partition") or unreadable(
            "no slurm_view item 1 with Partition="), "slurm_view")
        node = os.environ.get("SLURMD_NODENAME") or socket.gethostname()
        self.put("node_name", node, "env:SLURMD_NODENAME" if os.environ.get("SLURMD_NODENAME")
                 else "socket.gethostname")

    def collect(self):
        self.cpu()
        self.memory()
        self.gpus()
        self.nics()
        self.disk()
        self.software()
        self.chassis_and_node()
        self.prof["collected_at"] = wall_iso()
        self.prof["collected_mono_ns"] = time.monotonic_ns()
        self.src["collected_at"] = self.src["collected_mono_ns"] = "clock"
        self.prof["sources"] = {k: self.src.get(k) for k in
                                sorted(set(IDENTITY_FIELDS) | set(OBSERVED_FIELDS))
                                if k != "sources"}
        validate_profile(self.prof)
        return self.prof


def main(argv=None):
    ap = argparse.ArgumentParser(prog="capenv.hwprofile", description=__doc__.split("\n\n")[0])
    add_identity_args(ap)
    ap.add_argument("--out", required=True, help="env/hw_profile.json")
    ap.add_argument("--raw-out", required=True, help="env/hw_raw.json")
    ap.add_argument("--local-dir", required=True, help="the node-local run dir (disk type/free)")
    ap.add_argument("--chassis-source", required=True, choices=("slurm_view", "dmi"))
    ap.add_argument("--nvidia-smi", required=True, help="nvidia-smi executable (GPU keys)")
    ap.add_argument("--slurm-view-file")
    ap.add_argument("--image-sha256")
    ap.add_argument("--code-baseline")
    ap.add_argument("--container-runtime")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    args = ap.parse_args(argv)
    try:
        if args.image_sha256 and not is_sha256_hex(args.image_sha256):
            raise ConfigError("--image-sha256 must be 64 lowercase hex")
        if args.code_baseline and not is_git_sha(args.code_baseline):
            raise ConfigError("--code-baseline must be a 40-hex commit")
        ident = Identity(experiment=args.experiment, arm=args.arm, launch_seq=args.launch_seq,
                         job_id=args.job_id, restart_count=args.restart_count,
                         register_sha=args.register_sha, role=args.role, worker=args.worker,
                         pid=os.getpid())
        col = Collector(args.local_dir, args.chassis_source, args.slurm_view_file,
                        args.image_sha256, args.code_baseline, args.container_runtime,
                        args.timeout_s, nvidia_smi=args.nvidia_smi)
        prof = col.collect()
        write_json_stamped(args.raw_out, RAW_SCHEMA, ident, {"raw": col.raw})
        sha = profile_sha256(prof)
        write_json_stamped(args.out, SCHEMA, ident, {
            "profile": prof, "profile_sha256": sha, "sha_fields": sorted(IDENTITY_FIELDS),
            "raw_file_sha256": sha256_file(args.raw_out),
            "unreadable_identity_fields": sorted(
                k for k in IDENTITY_FIELDS if isinstance(prof[k], dict)
                and "unreadable" in prof[k])})
    except ConfigError as e:
        print("capenv.hwprofile: %s" % e, file=sys.stderr)
        return exitcodes.CONFIG
    print(json.dumps({"profile_sha256": sha, "out": args.out}))
    return exitcodes.OK


def load_profile(path):
    """(profile dict, recorded sha) from hw_profile.json; recomputes and checks the sha."""
    from capcommon.jsonio import read_json_stamped  # noqa: PLC0415
    _ident, doc = read_json_stamped(path, SCHEMA)
    prof = validate_profile(doc["profile"])
    sha = profile_sha256(prof)
    if sha != doc.get("profile_sha256"):
        raise ConfigError("%s: recorded profile_sha256 does not match its profile" % path)
    return prof, sha


if __name__ == "__main__":
    sys.exit(main())
