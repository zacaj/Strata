"""Strata vs the rest of the PC in the Monitor's readings (serve/telemetry.py); no GPU, NVML or Windows needed."""
import ctypes
import os
import subprocess
import sys
import time
import unittest

from serve import telemetry
from serve.telemetry import _Nvml, pdh_gpu_split, pdh_instance_pid

LUID_NV, LUID_IGPU = "0x00000000_0x0000C8D4", "0x00000000_0x0000AB12"


def mem(pid, luid, v):
    return f"pid_{pid}_luid_{luid}_phys_0", v


def eng(pid, luid, n, kind, v):
    return f"pid_{pid}_luid_{luid}_phys_0_eng_{n}_engtype_{kind}", v


class PdhSplit(unittest.TestCase):
    def test_instance_names(self):
        self.assertEqual(pdh_instance_pid(f"pid_42_luid_{LUID_NV}_phys_0_eng_3_engtype_Compute"), (42, LUID_NV, "3"))
        self.assertEqual(pdh_instance_pid(f"pid_42_luid_{LUID_NV}_phys_0"), (42, LUID_NV, None))
        self.assertEqual(pdh_instance_pid("_Total"), (None, None, None))

    def test_only_strata_on_its_card(self):
        mems = [mem(10, LUID_NV, 14 << 30), mem(11, LUID_NV, 200 << 20), mem(99, LUID_NV, 900 << 20),
                mem(10, LUID_IGPU, 1 << 20)]              # the iGPU that draws the desktop: below the floor
        engs = [eng(10, LUID_NV, 0, "3D", 5.0), eng(10, LUID_NV, 1, "Compute_0", 60.0),
                eng(11, LUID_NV, 1, "Compute_0", 30.0),    # the same engine as pid 10: they add
                eng(99, LUID_NV, 1, "Compute_0", 9.0),     # not Strata
                eng(10, LUID_IGPU, 0, "3D", 50.0)]         # not the card Strata holds memory on
        b, u = pdh_gpu_split(mems, engs, {10, 11})
        self.assertEqual(b, (14 << 30) + (200 << 20))
        self.assertAlmostEqual(u, 90.0)

    def test_nothing_held(self):
        self.assertEqual(pdh_gpu_split([mem(99, LUID_NV, 1 << 30)], [], {10}), (None, None))


class FakeNvml:
    """Fills the caller's ctypes buffers the way libnvidia-ml does: a count query, then the entries."""

    def __init__(self, procs, samples=None, versions=("_v3",)):
        self.procs, self.samples = procs, samples
        for v in versions:
            setattr(self, f"nvmlDeviceGetComputeRunningProcesses{v}", self._running)

    def _running(self, dev, n, buf):
        n = n._obj
        if buf is None:
            n.value = len(self.procs)
            return 7
        for i, (pid, b) in enumerate(self.procs):
            buf[i].pid, buf[i].usedGpuMemory = pid, b
        n.value = len(self.procs)
        return 0

    def nvmlDeviceGetProcessUtilization(self, dev, buf, n, ts):
        if self.samples is None:
            return 3                                     # NVML_ERROR_NOT_SUPPORTED
        rows = [s for s in self.samples if s[1] > ts.value]
        n = n._obj
        if not rows:
            return 6
        if buf is None:
            n.value = len(rows)
            return 7
        for i, (pid, t, sm) in enumerate(rows):
            buf[i].pid, buf[i].timeStamp, buf[i].smUtil = pid, t, sm
        n.value = len(rows)
        return 0


def nvml(fake):
    g = _Nvml.__new__(_Nvml)
    g.lib, g.dev = fake, ctypes.c_void_p(1)
    return g


class NvmlPerProcess(unittest.TestCase):
    def test_memory(self):
        g = nvml(FakeNvml([(10, 14 << 30), (11, 1 << 20), (99, 2 << 30)]))
        self.assertEqual(g.proc_mem({10, 11}), (14 << 30) + (1 << 20))
        self.assertEqual(g.proc_mem({5}), 0)

    def test_memory_wddm_is_unknown(self):
        g = nvml(FakeNvml([(10, _Nvml.NOT_AVAILABLE), (99, _Nvml.NOT_AVAILABLE)]))
        self.assertIsNone(g.proc_mem({10}))

    def test_older_driver_v2(self):
        g = nvml(FakeNvml([(10, 3 << 30)], versions=("_v2",)))
        self.assertEqual(g.proc_mem({10}), 3 << 30)

    def test_util_mean_per_process_since_last_read(self):
        f = FakeNvml([], samples=[(10, 100, 80), (10, 200, 100), (11, 200, 10), (99, 200, 50)])
        g = nvml(f)
        self.assertAlmostEqual(g.proc_util({10, 11}), 100.0)   # (80 + 100) / 2 + 10, capped at 100
        f.samples.append((10, 300, 40))
        self.assertAlmostEqual(g.proc_util({10, 11}), 40.0)    # only the samples after the last one read
        self.assertEqual(g.proc_util({10, 11}), 0.0)           # none since: nothing ran

    def test_util_not_supported(self):
        self.assertIsNone(nvml(FakeNvml([], samples=None)).proc_util({10}))


@unittest.skipUnless(telemetry.Telemetry and __import__("importlib").util.find_spec("psutil"), "needs psutil")
class ProcessTree(unittest.TestCase):
    def test_child_cpu_and_ram_count_as_strata(self):
        child = subprocess.Popen([sys.executable, "-c",
                                  "import time\nx = bytearray(150 << 20)\nt = time.time()\nwhile time.time() - t < 5: pass"])
        try:
            t = telemetry.Telemetry(root_pid=os.getpid())
            try:
                deadline = time.time() + 6
                now = {}
                while time.time() < deadline:
                    time.sleep(0.5)
                    now = t.snapshot()["now"]
                    if (now.get("strata_cpu") or 0) > 0 and (now.get("strata_ram") or 0) > (150 << 20):
                        break
                self.assertTrue(t.snapshot()["static"]["strata_split"])
                ncpu = os.cpu_count() or 1
                self.assertGreater(now["strata_cpu"], 50.0 / ncpu)     # one busy child: about 100 / ncpu percent
                self.assertLessEqual(now["strata_cpu"], 100.0)
                self.assertGreater(now["strata_ram"], 150 << 20)
                self.assertIsNone(now["strata_gpu_mem"])               # no NVML here
            finally:
                t.close()
        finally:
            child.kill()
            child.wait()

    def test_off_without_root(self):
        t = telemetry.Telemetry()
        try:
            self.assertFalse(t.snapshot()["static"]["strata_split"])
        finally:
            t.close()


if __name__ == "__main__":
    unittest.main()
