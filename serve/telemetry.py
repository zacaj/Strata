"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
import platform
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        return out

    # -------------------------------------------------------------------- per process (Strata vs the rest)
    class ProcInfo(ctypes.Structure):          # nvmlProcessInfo_t, the _v2 / _v3 functions' entry
        _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                    ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]

    class ProcInfoV1(ctypes.Structure):        # nvmlProcessInfo_v1_t, the unversioned function's entry
        _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]

    class UtilSample(ctypes.Structure):        # nvmlProcessUtilizationSample_t
        _fields_ = [("pid", ctypes.c_uint), ("timeStamp", ctypes.c_ulonglong), ("smUtil", ctypes.c_uint),
                    ("memUtil", ctypes.c_uint), ("encUtil", ctypes.c_uint), ("decUtil", ctypes.c_uint)]

    NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF          # NVML_VALUE_NOT_AVAILABLE: what WDDM (Windows) reports per process
    _util_ts = 0

    def _running(self, kind):
        """[(pid, bytes or None)] of the compute or graphics processes, or None when NVML cannot list them."""
        for suffix, st in (("_v3", self.ProcInfo), ("_v2", self.ProcInfo), ("", self.ProcInfoV1)):
            fn = getattr(self.lib, f"nvmlDeviceGet{kind}RunningProcesses{suffix}", None)
            if fn is None:
                continue
            try:
                n = ctypes.c_uint(0)
                r = fn(self.dev, ctypes.byref(n), None)
                if r not in (0, 7):            # 7: NVML_ERROR_INSUFFICIENT_SIZE, n now holds the count
                    continue
                n = ctypes.c_uint(n.value + 8)  # room for processes that start between the two calls
                buf = (st * n.value)()
                if fn(self.dev, ctypes.byref(n), buf) != 0:
                    continue
                return [(int(e.pid), None if e.usedGpuMemory == self.NOT_AVAILABLE else int(e.usedGpuMemory))
                        for e in buf[:n.value]]
            except (AttributeError, OSError):
                continue
        return None

    def proc_mem(self, pids):
        """Bytes of this card's memory the processes `pids` hold, or None when the driver does not say (WDDM)."""
        if not self.ok():
            return None
        rows = self._running("Compute")
        if rows is None:
            return None
        mine = [b for p, b in rows if p in pids]
        if any(b is None for b in mine):
            return None
        return sum(mine)

    def proc_util(self, pids):
        """Percent of the card's time the processes `pids` ran a kernel since the last call (NVML's per-process
        samples; like the card's own load figure it counts a kernel that spins as busy).  None: not supported."""
        if not self.ok():
            return None
        fn = getattr(self.lib, "nvmlDeviceGetProcessUtilization", None)
        if fn is None:
            return None
        try:
            n = ctypes.c_uint(0)
            ts = ctypes.c_ulonglong(self._util_ts)
            r = fn(self.dev, None, ctypes.byref(n), ts)
            if r == 6:                          # NVML_ERROR_NOT_FOUND: no samples since then, nothing ran
                return 0.0
            if r not in (0, 7) or n.value == 0:
                return None if r not in (0, 7) else 0.0
            buf = (self.UtilSample * n.value)()
            r = fn(self.dev, buf, ctypes.byref(n), ts)
            if r == 6:
                return 0.0
            if r != 0:
                return None
        except (AttributeError, OSError):
            return None
        rows = buf[:n.value]
        if rows:
            self._util_ts = max(int(e.timeStamp) for e in rows)
        per = collections.defaultdict(list)
        for e in rows:
            if int(e.pid) in pids:
                per[int(e.pid)].append(int(e.smUtil))
        return float(min(100.0, sum(sum(v) / len(v) for v in per.values())))


# ------------------------------------------------------------------------------------------------ AMD (Linux sysfs)
SYSFS = "/sys"


def amd_device_dir(index, sysfs=None):
    """The amdgpu sysfs folder (/sys/class/drm/renderD<N>/device) of the AMD GPU that HIP numbers `index`: the KFD
    topology's GPU nodes in order, the CPU nodes skipped, linked to their render node by drm_render_minor - the
    numbering setup's amd_gpus() and HIP_VISIBLE_DEVICES use.  None when there is no such card (or no amdgpu)."""
    base = os.path.join(sysfs or SYSFS, "class", "kfd", "kfd", "topology", "nodes")
    try:
        nodes = sorted((n for n in os.listdir(base) if n.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    for n in nodes:
        try:
            with open(os.path.join(base, n, "properties"), encoding="utf-8") as f:
                props = dict(line.strip().partition(" ")[::2] for line in f if line.strip())
            if int(props.get("gfx_target_version") or 0) == 0 or int(props.get("simd_count") or 0) == 0:
                continue
            gpus.append(props)
        except (OSError, ValueError):
            continue
    if not 0 <= index < len(gpus) or not gpus[index].get("drm_render_minor"):
        return None
    dev = os.path.join(sysfs or SYSFS, "class", "drm", "renderD" + gpus[index]["drm_render_minor"].strip(), "device")
    return dev if os.path.isdir(dev) else None


class _Amd:
    """#301: an AMD card's readings from the amdgpu driver's sysfs files (Linux; no ROCm library needed), with _Nvml's
    interface: load (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature
    (temp1_input, the edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap)."""

    def __init__(self, index=0, sysfs=None):
        self.dev = amd_device_dir(index, sysfs)
        self.hwmon = None
        if self.dev:
            try:
                hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
                self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
            except OSError:
                pass

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _int(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        try:
            with open(os.path.join(self.dev, "product_name"), encoding="utf-8") as f:
                return f.read().strip() or "AMD Radeon"
        except (OSError, TypeError):
            return "AMD Radeon"

    @staticmethod
    def _gen(text):
        """The PCIe generation of a link-speed file's contents ("16.0 GT/s PCIe" -> 4), or None."""
        try:
            return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}[float(str(text).split()[0])]
        except (ValueError, TypeError, IndexError, KeyError):
            return None

    @staticmethod
    def _bdf(s):
        """True for a sysfs pci device name ("0000:03:00.0"); nothing is imported for it."""
        return (len(s) == 12 and s[4] == ":" and s[7] == ":" and s[10] == "." and s[11] in "01234567"
                and all(c in "0123456789abcdef" for c in s[0:4] + s[5:7] + s[8:10]))

    @staticmethod
    def _read(d, name):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, TypeError):
            return None

    def _hops(self):
        """The PCIe devices between the root port and this card, the card last: the sysfs path names every
        hop (a card behind a bridge chain has more than one; a directly attached card has one)."""
        out, prefix = [], []
        for part in os.path.realpath(self.dev or "").split("/"):
            prefix.append(part)
            if self._bdf(part):
                out.append("/".join(prefix))
        return out

    def link(self):
        """The PCIe link the card actually gets: the **narrowest/slowest hop** between the root port and the card,
        from each hop's `max_link_speed` / `max_link_width` (the capability, so a power-saving downgrade or a Gen3
        slot cannot make it read low).  A card that is Gen4 on its own hop but sits behind a Gen3 root port really
        runs at Gen3, and that is the number a PCIe bandwidth budget needs.
        `pcie_own_gen` keeps the card's own hop aside: the gap between the two is what a user has to see.

        The bottleneck is only claimed when **every** hop of the path could be read; `pcie_path` reports how many
        of them were ("4/4").  A kernel or a container that hides part of /sys/devices would otherwise drop the
        unreadable hops in silence and report the card's own Gen4 hop as the whole path - i.e. be optimistic
        exactly where it matters.  When the walk is incomplete the reading falls back to the card's own negotiated
        link and pcie_path says so."""
        hops = self._hops()
        gens, widths, read = [], [], 0
        for d in hops:
            g = self._gen(_Amd._read(d, "max_link_speed"))
            w = self._int(os.path.join(d, "max_link_width"))
            read += 1 if (g is not None or w is not None) else 0
            if g:
                gens.append(g)
            if w:
                widths.append(w)
        own = self.dev or ""
        own_gen = self._gen(_Amd._read(own, "current_link_speed"))
        whole = bool(hops) and read == len(hops)
        gen = min(gens) if (whole and gens) else own_gen
        width = min(widths) if (whole and widths) else self._int(os.path.join(own, "current_link_width"))
        return {"pcie_gen": gen, "pcie_gen_max": gen, "pcie_own_gen": own_gen, "pcie_width": width,
                "pcie_path": "%d/%d" % (read, len(hops))}


    def read(self):
        out = {"util": self._int(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": self._int(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": self._int(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = self._int(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            p = self._int(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = self._int(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = self._int(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        out.update(self.link())
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), or the amdgpu sysfs files with the AMD backend (#301)."""
    return _Amd(index) if amd else _Nvml(index)


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def windows_display_adapters(vendor=None) -> list:
    """#1587: the display adapters Windows has drivers for, from the registry's display class (no subprocess, no vendor
    library): [{"name", "provider", "driver", "vram_bytes"}] in the class's order.  `vendor`: "amd", "nvidia" or
    "intel" keeps that vendor's cards.  [] off Windows or when nothing can be read."""
    if os.name != "nt":
        return []
    try:
        import winreg
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return []
    out = []
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            k = winreg.OpenKey(root, sub)
        except OSError:
            continue

        def val(name, k=k):
            try:
                return winreg.QueryValueEx(k, name)[0]
            except OSError:
                return None
        name = val("DriverDesc")
        if not isinstance(name, str) or not name.strip():
            continue
        mem = val("HardwareInformation.qwMemorySize")
        if mem is None:
            mem = val("HardwareInformation.MemorySize")
        if isinstance(mem, (bytes, bytearray)):
            mem = int.from_bytes(bytes(mem[:8]), "little") if mem else None
        out.append({"name": name.strip(), "provider": str(val("ProviderName") or ""), "driver": val("DriverVersion"),
                    "vram_bytes": mem if isinstance(mem, int) and mem > 0 else None})
    if vendor:
        keys = {"amd": ("advanced micro devices", "amd", "radeon"), "nvidia": ("nvidia",), "intel": ("intel",)}[vendor]
        out = [a for a in out if any(s in (a["provider"] + " " + a["name"]).lower() for s in keys)]
    return out


def windows_amd_cards(adapters=None) -> list:
    """The AMD cards HIP runs on, as windows_display_adapters lists them.  A Ryzen's integrated Radeon is listed too
    (512 MiB on a Ryzen 5 7600): beside a card with 2 GiB or more it is left out (an APU alone keeps it)."""
    cards = windows_display_adapters("amd") if adapters is None else adapters
    return [c for c in cards if (c["vram_bytes"] or 0) >= 2 << 30] or cards


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read."""
    g = gpu_reader(index, amd)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ Strata vs the rest
def pdh_instance_pid(name):
    """`pid_1234_luid_0x00000000_0x0000C8D4_phys_0[_eng_3_engtype_Compute]` -> (1234, luid, engine or None)."""
    parts = name.split("_")
    try:
        pid = int(parts[parts.index("pid") + 1])
        i = parts.index("luid")
        luid = "_".join(parts[i + 1:i + 3])
        eng = parts[parts.index("eng") + 1] if "eng" in parts else None
    except (ValueError, IndexError):
        return None, None, None
    return pid, luid, eng


def pdh_gpu_split(mem_items, eng_items, pids, min_bytes=64 << 20):
    """Windows' per-process GPU counters -> (bytes, percent) of the processes `pids`.  `mem_items`: (instance, bytes)
    of `\\GPU Process Memory(*)\\Dedicated Usage`; `eng_items`: (instance, percent) of `\\GPU Engine(*)\\Utilization
    Percentage`.  Only adapters on which `pids` hold `min_bytes` or more count (the NVIDIA card, not an iGPU that
    draws the desktop).  The percent is Task Manager's: per adapter the busiest engine, the mean over the adapters."""
    held = collections.Counter()
    for name, v in mem_items:
        pid, luid, _ = pdh_instance_pid(name)
        if pid in pids and v:
            held[luid] += v
    luids = {l for l, b in held.items() if b >= min_bytes}
    if not luids:
        return None, None
    engines = collections.defaultdict(float)
    for name, v in eng_items:
        pid, luid, eng = pdh_instance_pid(name)
        if pid in pids and luid in luids and eng is not None:
            engines[(luid, eng)] += v or 0.0
    util = [max([v for (l, _), v in engines.items() if l == luid] or [0.0]) for luid in luids]
    return sum(held[l] for l in luids), min(100.0, sum(util) / len(util))


class _PdhGpu:
    """Per-process VRAM and GPU load on Windows from the performance counters Task Manager reads (NVML reports no
    per-process memory under WDDM).  Anything that fails leaves the readings None."""
    PDH_FMT_DOUBLE, PDH_FMT_NOCAP100, PDH_MORE_DATA = 0x200, 0x8000, 0x800007D2

    class Value(ctypes.Structure):
        _fields_ = [("CStatus", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]

    class Item(ctypes.Structure):
        _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", ctypes.c_byte * 16)]

    def __init__(self):
        self.pdh = self.q = None
        if os.name != "nt":
            return
        try:
            self.pdh = ctypes.WinDLL("pdh")
            q = ctypes.c_void_p()
            if self.pdh.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
                self.pdh = None
                return
            self.q = q
            self.mem, self.eng = ctypes.c_void_p(), ctypes.c_void_p()
            if (self.pdh.PdhAddEnglishCounterW(q, "\\GPU Process Memory(*)\\Dedicated Usage", None,
                                               ctypes.byref(self.mem)) != 0 or
                    self.pdh.PdhAddEnglishCounterW(q, "\\GPU Engine(*)\\Utilization Percentage", None,
                                                   ctypes.byref(self.eng)) != 0):
                self.pdh = None
                return
            self.pdh.PdhCollectQueryData(q)     # the engine counter is a rate: the next collect gives the first value
        except (AttributeError, OSError):
            self.pdh = None

    def ok(self):
        return self.pdh is not None

    def _array(self, counter):
        size, count = ctypes.c_ulong(0), ctypes.c_ulong(0)
        fmt = self.PDH_FMT_DOUBLE | self.PDH_FMT_NOCAP100
        r = self.pdh.PdhGetFormattedCounterArrayW(counter, fmt, ctypes.byref(size), ctypes.byref(count), None)
        if (r & 0xFFFFFFFF) != self.PDH_MORE_DATA or size.value == 0:
            return []
        buf = ctypes.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(counter, fmt, ctypes.byref(size), ctypes.byref(count), buf) != 0:
            return []
        items = ctypes.cast(buf, ctypes.POINTER(self.Item))
        out = []
        for i in range(count.value):
            v = self.Value.from_buffer_copy(bytes(items[i].FmtValue))
            if v.CStatus in (0, 1):             # PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA
                out.append((items[i].szName or "", v.doubleValue))
        return out

    def read(self, pids):
        if not self.ok():
            return None, None
        try:
            if self.pdh.PdhCollectQueryData(self.q) != 0:
                return None, None
            return pdh_gpu_split(self._array(self.mem), self._array(self.eng), pids)
        except (AttributeError, OSError, ValueError):
            return None, None


class _ProcTree:
    """CPU, RAM and disk reads of one process and everything it started (the server: the engine, the vision helper,
    replicas, MCP servers it launched).  Needs psutil."""

    def __init__(self, ps, root_pid):
        self.ps, self.root_pid, self.procs, self._io_prev = ps, root_pid, {}, None
        self.ncpu = ps.cpu_count() or os.cpu_count() or 1

    def pids(self):
        try:
            root = self.ps.Process(self.root_pid)
            tree = [root] + root.children(recursive=True)
        except (self.ps.Error, OSError):
            return set()
        live = {}
        for p in tree:
            old = self.procs.get(p.pid)
            try:
                same = old is not None and old.create_time() == p.create_time()   # not a new process with an old pid
            except (self.ps.Error, OSError):
                same = False
            live[p.pid] = old if same else p
        self.procs = live
        return set(live)

    def read(self):
        """{"cpu": percent of ALL logical CPUs, "ram": resident private bytes, "disk_read_mb": MB/s}."""
        self.pids()
        cpu = ram = rd = 0.0
        for p in list(self.procs.values()):
            try:
                with p.oneshot():
                    cpu += p.cpu_percent(interval=None)      # the first call per process reads 0: a baseline
                    m = p.memory_info()
                    # Linux: rss counts mapped file pages (an mmapped experts.bin) that the OS also counts as cache;
                    # the private part is what "used" counts.  Windows: the working set.
                    ram += m.rss - getattr(m, "shared", 0) if sys.platform.startswith("linux") else m.rss
                    try:
                        rd += p.io_counters().read_bytes
                    except (AttributeError, self.ps.Error, OSError):
                        pass
            except (self.ps.Error, OSError):
                continue
        t = time.time()
        prev, self._io_prev = self._io_prev, (t, rd)
        disk = (max(0.0, rd - prev[1]) / (t - prev[0]) / 2**20) if prev and t > prev[0] else None
        return {"cpu": min(100.0, cpu / self.ncpu), "ram": ram, "disk_read_mb": disk}


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False, root_pid=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own.  `amd`: the AMD backend's
        cards, numbered as HIP numbers them, read from sysfs (#301).  `root_pid`: the process whose tree is Strata (the
        server, which starts the engine): its share of the CPU, RAM, disk reads, VRAM and GPU load is recorded beside
        the system's (strata_* keys), so the Monitor can show Strata + the rest = the total."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.gpus = [(i, gpu_reader(i, amd)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.tree = _ProcTree(self.ps, root_pid) if self.ps is not None and root_pid else None
        self.pdh = _PdhGpu() if self.tree is not None and os.name == "nt" and not amd else None
        gpu_name = " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None
        win_amd = None
        if gpu_name is None and amd and os.name == "nt":
            # #1587: no readings for an AMD card on Windows yet, but its name (and VRAM size) are in the registry, so the
            # dashboard and strata_bench do not call the run "CPU only".  The class order is taken as HIP's order.
            cards = windows_amd_cards()
            win_amd = [cards[i] for i in idx if 0 <= i < len(cards)] or cards[:1]
            gpu_name = " + ".join(c["name"] for c in win_amd) if win_amd else None
        self.static = {
            "gpu_name": gpu_name,
            "gpu_vram_total_mib": (sum(c["vram_bytes"] for c in win_amd) >> 20
                                   if win_amd and all(c["vram_bytes"] for c in win_amd) else None),
            "gpu_count": len(self.gpus),
            # #1380: the AMD readings are the amdgpu driver's Linux sysfs files; a Windows AMD card has none yet, and the
            # dashboard said "not readable (NVML)" or showed empty tiles with no word why
            "gpu_note": ("no GPU load or VRAM readings for AMD cards on Windows yet (Linux reads them from the amdgpu "
                         "driver); the engine's own VRAM figures are in its log" if amd and not self.gpu.ok() else None),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
            "strata_split": self.tree is not None,   # the strata_* readings are recorded
        }
        self._disk_prev = None
        self._stop = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        if c is None:   # psutil found no disk (a gVisor container, Windows with its disk counters off)
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self):
        s = {}
        if self.gpu.ok():
            reads = [(i, g.read()) for i, g in self.gpus]
            g = dict(reads[0][1])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                s["gpus"] = [{"index": i, "util": r.get("util"), "mem_used": r.get("mem_used"),
                              "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power")}
                             for i, r in reads]
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.tree is not None:
            try:
                s.update(self._strata())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def _strata(self):
        """Strata's own share of each reading (the process tree of `root_pid`); None where it cannot be read."""
        t = self.tree.read()
        out = {"strata_cpu": t["cpu"], "strata_ram": t["ram"], "strata_disk_read_mb": t["disk_read_mb"]}
        pids = set(self.tree.procs)
        mem = [getattr(g, "proc_mem", lambda _p: None)(pids) for _, g in self.gpus]
        util = [getattr(g, "proc_util", lambda _p: None)(pids) for _, g in self.gpus]
        out["gpus_strata"] = [{"index": i, "mem": m, "util": u} for (i, _), m, u in zip(self.gpus, mem, util)]
        out["strata_gpu_mem"] = sum(mem) if mem and all(m is not None for m in mem) else None
        out["strata_gpu_util"] = sum(util) / len(util) if util and all(u is not None for u in util) else None
        if self.pdh is not None and (out["strata_gpu_mem"] is None or out["strata_gpu_util"] is None):
            pm, pu = self.pdh.read(pids)        # WDDM: NVML has no per-process memory there
            if out["strata_gpu_mem"] is None:
                out["strata_gpu_mem"] = pm
            if out["strata_gpu_util"] is None:
                out["strata_gpu_util"] = pu
        return out

    def close(self):
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the process."""
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            s = self.sample()
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean", "strata_cpu", "strata_ram",
                          "strata_disk_read_mb", "strata_gpu_mem", "strata_gpu_util"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            self._stop.wait(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
