#!/usr/bin/env python3
"""Benchmark harness for Assignment 2 (Java: dense optimisations and CSR sparse multiplication).

Same protocol as Assignment 1 (see docs/environment.md for what changed):
  * every configuration runs in its own JVM process, one compute thread, pinned to --cpus;
  * probe run (1 repetition, no warm-up) -> time of one multiplication and peak memory
    (psutil, polled every 10 ms; Windows peak working set summed over the process tree);
  * timing run: warm-up (>= N repetitions AND >= warmup_min_ms, excluded) + measured repetitions,
    all inside one process. Short kernels are batched automatically (time reported PER operation);
  * if the probe kernel takes > costly_s seconds: 3 repetitions and no warm-up (reported);
  * limits: time_s per run and mem_bytes of peak memory. A violation is recorded as a result
    (timeout / memory_exceeded) without any invented time; larger n of the same series is skipped;
  * checksums of all methods on the same input are compared and mismatches are reported.

The list of configurations comes from config/experiment.json (dense sizes, block sweep, JVM
controls, sparse density sweep and size sweep with the dense reference methods).

Outputs (append mode, resumable; one dir, default results/):
  raw_measurements.csv   one row per measured sample (phase = kernel | convert)
  run_status.csv         one row per configuration: status, probe time, memory metrics, power state

Usage (repo root):
  python scripts/run_benchmarks.py --dry-run
  python scripts/run_benchmarks.py --out-dir results/test --sizes 100 200 --methods ijk ikj csr
  python scripts/run_benchmarks.py
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
DATA, BUILD, CONFIG = ROOT / "data", ROOT / "build", ROOT / "config" / "experiment.json"

RAW_HEADER = ["timestamp", "method", "block", "n", "density", "numeric_type", "threads", "jvm_flags",
              "timer", "phase", "rep", "batch", "seconds_per_op", "checksum", "nnz_a", "nnz_b", "nnz_c",
              "warmup_iterations", "reps_planned", "affinity"]
STATUS_HEADER = ["timestamp", "method", "block", "n", "density", "jvm_flags", "status", "probe_seconds",
                 "peak_memory_bytes", "memory_metric", "live_heap_bytes_after_gc", "array_estimate_bytes",
                 "dense_inputs_bytes", "reps", "warmup_planned", "warmup_iterations", "time_limit_s",
                 "mem_limit_bytes", "affinity", "power_plugged", "battery_percent", "java_version", "gc"]


# ----------------------------------------------------------------------------- configurations
def build_configs(cfg):
    d, s = cfg["dense"], cfg.get("sparse", {})
    out = {}

    def add(method, n, density=None, block=0, flags_name="default", flags=()):
        key = (method, n, density, block if method == "blocked" else 0, flags_name)
        out.setdefault(key, {"method": method, "n": n, "density": density,
                             "block": block if method == "blocked" else 0,
                             "flags_name": flags_name, "flags": list(flags)})

    for n in d["sizes"]:
        for m in d["methods"]:
            add(m, n, None, d["block"])
    bs = d.get("block_sweep")
    if bs:
        for b in bs["sizes"]:
            add("blocked", bs["n"], None, b)
    for c in d.get("controls", []):
        for n in c["sizes"]:
            for m in c["methods"]:
                add(m, n, None, d["block"], c["name"], c["jvm_flags"])
    refs = s.get("dense_reference_methods", [])
    ds, ss = s.get("density_sweep"), s.get("size_sweep")
    pairs = set()
    if ds:
        pairs |= {(ds["n"], x) for x in ds["densities"]}
    if ss:
        pairs |= {(n, ss["density"]) for n in ss["sizes"]}
    for n, dens in sorted(pairs):
        add("csr", n, dens)
        for m in refs:
            add(m, n, dens, d["block"])
    return sorted(out.values(), key=lambda c: (c["n"], c["density"] or 0, c["method"], c["block"], c["flags_name"]))


def series(c):
    return (c["method"], c["density"], c["block"], c["flags_name"])


def key_of(c):
    return "|".join(str(x) for x in (c["method"], c["block"], c["n"], c["density"], c["flags_name"]))


def input_paths(c):
    n, d = c["n"], c["density"]
    if d is None:
        return DATA / "dense" / f"A_n{n}.bin", DATA / "dense" / f"B_n{n}.bin"
    return DATA / "sparse" / f"A_n{n}_d{d:g}.bin", DATA / "sparse" / f"B_n{n}_d{d:g}.bin"


# ----------------------------------------------------------------------------- process handling
def sample_memory(ps):
    total = 0
    try:
        procs = [ps] + ps.children(recursive=True)
    except psutil.Error:
        return 0
    for p in procs:
        try:
            mi = p.memory_info()
            total += getattr(mi, "peak_wset", mi.rss)
        except psutil.Error:
            pass
    return total


def kill_tree(proc):
    try:
        for ch in psutil.Process(proc.pid).children(recursive=True):
            ch.kill()
    except psutil.Error:
        pass
    proc.kill()


def run_monitored(cmd, timeout_s, cpus, mem_limit):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ps = psutil.Process(proc.pid)
    try:
        ps.cpu_affinity(cpus)
    except (psutil.Error, AttributeError, OSError) as e:
        print(f"  WARNING: could not set affinity: {e}")
    start, peak, status = time.perf_counter(), 0, "ok"
    while proc.poll() is None:
        peak = max(peak, sample_memory(ps))
        if peak > mem_limit:
            status = "memory_exceeded"
            break
        if time.perf_counter() - start > timeout_s:
            status = "timeout"
            break
        time.sleep(0.01)
    if status != "ok":
        kill_tree(proc)
    out, err = proc.communicate()
    if status == "ok" and proc.returncode != 0:
        status = "error"
    return status, out, err, peak


def parse(out):
    rows, meta = [], {}
    for line in out.strip().splitlines():
        if line.startswith("#"):
            k, _, v = line[1:].partition("=")
            meta[k] = v
        elif line:
            p = line.split(",")
            rows.append({"phase": p[0], "rep": int(p[1]), "batch": int(p[2]), "seconds": float(p[3]),
                         "checksum": float(p[4]), "nnz_a": int(p[5]), "nnz_b": int(p[6]), "nnz_c": int(p[7])})
    return rows, meta


def append(path, header, row):
    new = not path.exists() or path.stat().st_size == 0     # an empty file still needs its header
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def load_status(path):
    done, blocked = set(), set()
    if path.exists():
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                c = {"method": r["method"], "density": float(r["density"]) if r["density"] else None,
                     "block": int(r["block"]), "flags_name": r["jvm_flags"] or "default", "n": int(r["n"])}
                done.add(key_of(c))
                if r["status"] != "ok":
                    blocked.add(series(c))
    return done, blocked


def power_state():
    try:
        b = psutil.sensors_battery()
    except Exception:
        b = None
    if b is None:
        return "unknown", ""
    return str(b.power_plugged), f"{b.percent:.0f}"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--sizes", type=int, nargs="+", help="restrict to these n")
    ap.add_argument("--methods", nargs="+", help="restrict to these methods (ijk ikj blocked csr)")
    ap.add_argument("--reps", type=int, help="override measured repetitions")
    ap.add_argument("--cpus", type=int, nargs="+", help="logical processors for the benchmark process")
    ap.add_argument("--harness-cpus", type=int, nargs="*", default=None,
                    help="pin this harness process to these logical processors (disjoint from --cpus)")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    ap.add_argument("--dry-run", action="store_true", help="list the configurations and exit")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    cpus = args.cpus or cfg["cpus"]
    lim, jv = cfg["limits"], cfg["java"]
    reps_cfg = args.reps or jv["reps"]
    time_lim, mem_lim, costly = lim["time_s"], lim["mem_bytes"], lim["costly_s"]

    configs = build_configs(cfg)
    if args.sizes:
        configs = [c for c in configs if c["n"] in args.sizes]
    if args.methods:
        configs = [c for c in configs if c["method"] in args.methods]
    if args.dry_run:
        for c in configs:
            print(key_of(c), c["flags"])
        print(f"{len(configs)} configurations")
        return
    if args.harness_cpus:
        try:
            psutil.Process().cpu_affinity(args.harness_cpus)
        except (psutil.Error, AttributeError, OSError) as e:
            print(f"WARNING: could not pin the harness: {e}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path, status_path = out_dir / "raw_measurements.csv", out_dir / "run_status.csv"
    done, blocked = load_status(status_path)
    aff = "+".join(map(str, cpus))
    checks = {}

    for c in configs:
        k = key_of(c)
        ts = datetime.now().isoformat(timespec="seconds")
        dens = "" if c["density"] is None else c["density"]
        if k in done:
            print(f"[skip] {k} already done")
            continue
        plugged, batt = power_state()

        def status_row(status, probe_s="", peak="", meta=None, reps="", wu="", wu_it=""):
            meta = meta or {}
            return [ts, c["method"], c["block"], c["n"], dens, c["flags_name"] if c["flags"] else "default",
                    status, probe_s, peak, "peak_working_set", meta.get("live_heap_bytes_after_gc", ""),
                    meta.get("array_estimate_bytes", ""), meta.get("dense_inputs_bytes", ""), reps, wu, wu_it,
                    time_lim, mem_lim, aff, plugged, batt, meta.get("java_version", ""), meta.get("gc", "")]

        if series(c) in blocked:
            append(status_path, STATUS_HEADER, status_row("skipped_after_failure"))
            print(f"[skip] {k}: a smaller size of this series already failed")
            continue
        a, b = input_paths(c)
        if not (a.exists() and b.exists()):
            sys.exit(f"missing input {a.name}/{b.name}: run scripts/generate_inputs.py")

        base = ["java", f"-Xmx{jv['xmx']}"] + c["flags"] + ["-cp", str(BUILD), "bd.Main", "--method", c["method"],
                "--a", str(a), "--b", str(b), "--n", str(c["n"]), "--block", str(c["block"] or 64)]

        print(f"[probe] {k} ...", flush=True)
        st, out, err, peak = run_monitored(base + ["--reps", "1", "--warmup", "0", "--warmup-min-ms", "0",
                                                  "--batch", "1"], time_lim + 60, cpus, mem_lim)
        if st == "error":
            sys.exit(f"{k} failed: {err.strip()[:500]}")
        rows, meta = parse(out) if st == "ok" else ([], {})
        kern = [r for r in rows if r["phase"] == "kernel"]
        probe_s = kern[0]["seconds"] if kern else ""
        if st == "ok" and probe_s > time_lim:
            st = "timeout"
        if st != "ok":
            append(status_path, STATUS_HEADER, status_row(st, probe_s, peak, meta, 0, 0))
            blocked.add(series(c))
            print(f"  -> {st} (peak {peak / 1e6:.0f} MB)")
            continue

        if probe_s > costly:
            reps, warmup, wmin = 3, 0, 0
        else:
            reps, warmup, wmin = reps_cfg, jv["warmup"], jv["warmup_min_ms"]
        print(f"  probe {probe_s:.4f}s, peak {peak / 1e6:.0f} MB -> {reps} reps + >= {warmup} warm-up", flush=True)
        budget = time_lim * (reps + warmup + 1) + 60 + wmin / 1000
        st2, out, err, _ = run_monitored(base + ["--reps", str(reps), "--warmup", str(warmup),
                                                  "--warmup-min-ms", str(wmin), "--target-ms", str(jv["target_ms"])],
                                         budget, cpus, mem_lim)
        if st2 != "ok":
            append(status_path, STATUS_HEADER, status_row(st2 + "_timing_run", probe_s, peak, meta, reps, warmup))
            blocked.add(series(c))
            print(f"  -> {st2} in timing run {err.strip()[:200] if st2 == 'error' else ''}")
            continue
        rows, meta2 = parse(out)
        timer = "phase_separated" if c["method"] == "csr" else "kernel_only"
        for r in rows:
            append(raw_path, RAW_HEADER, [ts, c["method"], c["block"], c["n"], dens, "float64", 1,
                   c["flags_name"] if c["flags"] else "default", timer, r["phase"], r["rep"], r["batch"],
                   f"{r['seconds']:.9e}", f"{r['checksum']:.12e}", r["nnz_a"], r["nnz_b"], r["nnz_c"],
                   meta2.get("warmup_iterations", ""), reps, aff])
        ks = [r for r in rows if r["phase"] == "kernel"]
        if len({r["checksum"] for r in ks}) != 1:
            print(f"  WARNING: checksum changes between repetitions for {k}")
        checks.setdefault((c["n"], c["density"]), {})[k] = ks[0]["checksum"]
        vals = checks[(c["n"], c["density"])]
        ref = next(iter(vals.values()))
        if any(abs(v - ref) > 1e-9 * abs(ref) for v in vals.values()):
            print(f"  WARNING: checksums differ across methods on the same input: {vals}")
        final = "ok" if max(r["seconds"] for r in ks) <= time_lim else "rep_over_time_limit"
        append(status_path, STATUS_HEADER, status_row(final, probe_s, peak, meta2 or meta, reps, warmup,
                                                      meta2.get("warmup_iterations", "")))
        t = sorted(r["seconds"] for r in ks)
        print(f"  median kernel {t[len(t) // 2]:.6g} s (batch {ks[0]['batch']})")


if __name__ == "__main__":
    main()
