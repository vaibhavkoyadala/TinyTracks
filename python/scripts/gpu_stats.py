"""GPU and memory readings for the Monitor tab.

On macOS, the Metal driver publishes its counters under IOAccelerator, and `ioreg` reads them
in about 20 ms. Other platforms return available=False, and the tab says so.
"""
from __future__ import annotations

import platform
import re
import subprocess
import threading
import time

_KEYS = {
    "device_pct": r'"Device Utilization %"=(\d+)',
    "renderer_pct": r'"Renderer Utilization %"=(\d+)',
    "tiler_pct": r'"Tiler Utilization %"=(\d+)',
    "memory_in_use": r'"In use system memory"=(\d+)',
    "memory_allocated": r'"Alloc system memory"=(\d+)',
}
CACHE_SECONDS = 0.5
_lock = threading.Lock()
_cache: tuple[float, dict] | None = None


def _total_memory() -> int | None:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


_TOTAL_MEMORY = _total_memory() if platform.system() == "Darwin" else None


def read() -> dict:
    global _cache
    with _lock:
        now = time.time()
        if _cache and now - _cache[0] < CACHE_SECONDS:
            return _cache[1]
        result = _read_uncached()
        _cache = (now, result)
        return result


def _read_uncached() -> dict:
    base = {"available": False, "source": None, "memory_total": _TOTAL_MEMORY, "reason": None}
    if platform.system() != "Darwin":
        return {**base, "reason": "GPU counters are only implemented for macOS (Metal) yet"}
    try:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                             capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {**base, "reason": f"ioreg failed: {exc}"}
    match = re.search(r'"PerformanceStatistics" = \{([^}]*)\}', out)
    if not match:
        return {**base, "reason": "no Metal performance counters found"}
    stats = {}
    for name, pattern in _KEYS.items():
        m = re.search(pattern, match.group(1))
        stats[name] = int(m.group(1)) if m else None
    return {**base, "available": True, "source": "IOAccelerator", **stats}
