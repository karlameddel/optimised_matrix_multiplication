#!/usr/bin/env python3
"""Write results/environment.json: hardware, OS, versions and power state at the time of the run.
Run it once at the start of the benchmark session (and again if anything changes)."""
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy
import psutil

ROOT = Path(__file__).resolve().parent.parent


def cmd(*a):
    try:
        r = subprocess.run(a, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"unavailable ({e})"


def cpu_name():
    if platform.system() == "Windows":
        out = cmd("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name")
        if out and not out.startswith("unavailable"):
            return out
    return platform.processor() or "unknown"


b = psutil.sensors_battery()
env = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "cpu": cpu_name(),
    "physical_cores": psutil.cpu_count(logical=False),
    "logical_processors": psutil.cpu_count(logical=True),
    "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
    "os": platform.platform(),
    "java": cmd("java", "-version"),
    "javac": cmd("javac", "-version"),
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "psutil": psutil.__version__,
    "power_plugged": None if b is None else b.power_plugged,
    "battery_percent": None if b is None else b.percent,
    "note": "Set the Windows power mode manually and record it in docs/environment.md; "
            "verify the logical-processor -> core mapping (e.g. Sysinternals Coreinfo: coreinfo -c).",
}
out = ROOT / "results" / "environment.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(env, indent=2))
print(json.dumps(env, indent=2))
