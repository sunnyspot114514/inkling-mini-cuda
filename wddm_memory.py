"""Read Windows GPU Process Memory counters for one process, across its adapters."""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time


class CounterValue(ctypes.Structure):
    _fields_ = [("status", wintypes.DWORD), ("value", ctypes.c_longlong)]


class CounterItem(ctypes.Structure):
    _fields_ = [("name", ctypes.c_wchar_p), ("counter", CounterValue)]


class ProcessGpuMemory:
    def __init__(self, pid):
        if os.name != "nt":
            raise OSError("GPU Process Memory counters require WDDM on Windows")
        self.prefix = f"pid_{pid}_luid_"
        self.pdh = ctypes.WinDLL("pdh")
        self.query = ctypes.c_void_p()
        self.check(self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)))
        self.counters = {}
        try:
            for kind in ("Dedicated", "Shared"):
                path = rf"\GPU Process Memory(*)\{kind} Usage"
                handle = ctypes.c_void_p()
                self.check(self.pdh.PdhAddEnglishCounterW(self.query, path, 0, ctypes.byref(handle)))
                self.counters[kind.lower() + "_bytes"] = handle
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status):
        if status:
            raise OSError(f"Performance counter error 0x{status & 0xffffffff:08x}")

    def read(self):
        self.check(self.pdh.PdhCollectQueryData(self.query))
        sample = {}
        for key, counter in self.counters.items():
            size, count = wintypes.DWORD(), wintypes.DWORD()
            args = (counter, 0x400, ctypes.byref(size), ctypes.byref(count))  # PDH_FMT_LARGE
            status = self.pdh.PdhGetFormattedCounterArrayW(*args, None) & 0xffffffff
            if status != 0x800007D2:  # PDH_MORE_DATA
                self.check(status)
            if not size.value:
                return None
            buffer = ctypes.create_string_buffer(size.value)
            status = self.pdh.PdhGetFormattedCounterArrayW(*args, buffer) & 0xffffffff
            if status == 0x800007D2:  # instance appeared between the two calls
                return None
            self.check(status)
            entries = ctypes.cast(buffer, ctypes.POINTER(CounterItem))
            values = [entries[i].counter.value for i in range(count.value)
                      if entries[i].name.startswith(self.prefix) and entries[i].counter.status in (0, 1)]
            if not values:
                return None
            sample[key] = sum(values)
        return sample

    def close(self):
        if self.query.value:
            self.pdh.PdhCloseQuery(self.query)
            self.query = ctypes.c_void_p()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=float, default=1.)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.pid < 1 or args.interval <= 0 or args.samples < 1:
        parser.error("PID, sample count and interval must be positive")
    meter = ProcessGpuMemory(args.pid)
    observations = []
    start = time.perf_counter()
    try:
        for i in range(args.samples):
            delay = start + i * args.interval - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            reading = meter.read()
            if reading is not None:
                observations.append({"elapsed_s": time.perf_counter() - start, **reading})
    finally:
        meter.close()
    if not observations:
        raise RuntimeError("No valid WDDM samples for this process")
    peaks = {k: max(s[k] for s in observations) for k in ("dedicated_bytes", "shared_bytes")}
    report = {"source": "Windows GPU Process Memory across the requested process's adapters",
              "requested_samples": args.samples, "interval_s": args.interval,
              "valid_samples": len(observations), "sampled_peak": peaks, "samples": observations}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "samples"}))


if __name__ == "__main__":
    main()
