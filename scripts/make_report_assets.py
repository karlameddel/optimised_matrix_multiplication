#!/usr/bin/env python3
"""Tables and figures for Assignment 2 from results/raw_measurements.csv and run_status.csv.

Every table/figure uses the MEDIAN of the measured samples (per-operation seconds) and shows the
interquartile range. Nothing is averaged across phases: conversion and kernel times are kept apart;
'including conversion' is the sum of the two medians and is labelled as such.

Usage: python scripts/make_report_assets.py [--in-dir results] [--config config/experiment.json]
Writes <in-dir>/tables/*.csv and <in-dir>/figures/*.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def summarise(df):
    keys = ["method", "block", "n", "density", "jvm_flags", "phase"]
    g = df.groupby(keys)["seconds_per_op"]
    s = g.agg(reps="count", median="median", q1=lambda x: x.quantile(.25), q3=lambda x: x.quantile(.75),
              min="min", max="max").reset_index()
    s["iqr"] = s["q3"] - s["q1"]
    s["iqr_pct_of_median"] = 100 * s["iqr"] / s["median"]
    s["batch"] = g.first().reset_index()["seconds_per_op"].index.map(lambda _: np.nan)
    b = df.groupby(keys)["batch"].first().reset_index()["batch"]
    s["batch"] = b.values
    return s


def style(ax, xlabel, ylabel, title, loglog=True):
    if loglog:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, which="both", alpha=.3); ax.legend()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(ROOT / "results"))
    ap.add_argument("--config", default=str(ROOT / "config" / "experiment.json"))
    args = ap.parse_args()
    rd = Path(args.in_dir)
    tdir, fdir = rd / "tables", rd / "figures"
    tdir.mkdir(exist_ok=True, parents=True); fdir.mkdir(exist_ok=True, parents=True)
    cfg = json.loads(Path(args.config).read_text())

    df = pd.read_csv(rd / "raw_measurements.csv")
    st = pd.read_csv(rd / "run_status.csv")
    for d in (df, st):
        d["density"] = d["density"].fillna(1.0)           # dense random inputs have density 1
        d["block"] = d["block"].fillna(0).astype(int)
        d["jvm_flags"] = d["jvm_flags"].fillna("default")
    S = summarise(df)
    S.to_csv(tdir / "summary_time.csv", index=False)
    K = S[S.phase == "kernel"]

    # ------------------------------------------------------------------ Part A: dense
    D = K[(K.density == 1.0) & (K.jvm_flags == "default")]
    blk = D[D.method == "blocked"]
    main_block = int(blk.groupby("block")["n"].nunique().idxmax()) if len(blk) else 64
    Dm = D[(D.method != "blocked") | (D.block == main_block)].copy()
    Dm["label"] = Dm.apply(lambda r: f"blocked (bs={r.block})" if r.method == "blocked" else r.method, axis=1)
    base = Dm[Dm.method == "ijk"].set_index("n")["median"]
    Dm["speedup_vs_ijk"] = Dm.apply(lambda r: base.get(r.n, np.nan) / r["median"], axis=1)
    Dm["ns_per_inner_iter"] = Dm["median"] / Dm["n"] ** 3 * 1e9
    Dm.to_csv(tdir / "dense_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lab, g in Dm.groupby("label"):
        ax.errorbar(g.n, g["median"], yerr=[g["median"] - g.q1, g.q3 - g["median"]], marker="o", capsize=2, label=lab)
    if len(Dm):
        ref = Dm.iloc[(Dm.n - Dm.n.max()).abs().argsort()[:1]]
        n0, t0 = float(ref.n.iloc[0]), float(ref["median"].iloc[0])
        xs = np.array(sorted(Dm.n.unique()), dtype=float)
        ax.plot(xs, t0 * (xs / n0) ** 3, "k--", lw=.8, label=r"$\propto n^3$ (reference)")
    style(ax, "Matrix size n (n x n, float64)", "Kernel time per multiplication (s)",
          "Dense methods: runtime vs n (median, bars = IQR)")
    fig.tight_layout(); fig.savefig(fdir / "dense_time_vs_n.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lab, g in Dm.groupby("label"):
        ax.plot(g.n, g.ns_per_inner_iter, marker="o", label=lab)
    style(ax, "Matrix size n", "Median time / n^3 (ns per inner iteration)",
          "Cost per multiply-add (flat = cubic)", loglog=False)
    ax.set_xscale("log"); fig.tight_layout(); fig.savefig(fdir / "dense_ns_per_iter.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sp = Dm[Dm.method != "ijk"]
    for lab, g in sp.groupby("label"):
        ax.plot(g.n, g.speedup_vs_ijk, marker="o", label=lab)
    ax.axhline(1, color="k", lw=.8)
    style(ax, "Matrix size n", "Speedup over ijk baseline (same session)", "Dense speedup vs baseline", loglog=False)
    ax.set_xscale("log"); fig.tight_layout(); fig.savefig(fdir / "dense_speedup.png", dpi=150); plt.close(fig)

    # block-size sweep
    bs_cfg = cfg["dense"].get("block_sweep")
    if bs_cfg:
        B = D[(D.method == "blocked") & (D.n == bs_cfg["n"])].sort_values("block")
        B.to_csv(tdir / "block_sweep.csv", index=False)
        if len(B) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(B.block, B["median"], yerr=[B["median"] - B.q1, B.q3 - B["median"]], marker="o", capsize=2)
            for m in ("ikj", "ijk"):
                r = D[(D.method == m) & (D.n == bs_cfg["n"])]
                if len(r):
                    ax.axhline(float(r["median"].iloc[0]), ls="--", lw=.8, label=m)
            ax.set_xscale("log", base=2)
            style(ax, "Block size bs", "Kernel time (s)", f"Block-size sweep, n = {bs_cfg['n']}", loglog=False)
            ax.set_xscale("log", base=2); fig.tight_layout(); fig.savefig(fdir / "block_sweep.png", dpi=150); plt.close(fig)

    # JVM control: no auto-vectorisation
    C = K[(K.density == 1.0) & (K.jvm_flags != "default")]
    if len(C):
        M = D.merge(C, on=["method", "block", "n"], suffixes=("_default", "_control"))
        M["control_over_default"] = M["median_control"] / M["median_default"]
        M[["method", "block", "n", "jvm_flags_control", "median_default", "median_control", "control_over_default"]] \
            .to_csv(tdir / "jvm_control.csv", index=False)

    # ------------------------------------------------------------------ Part B: sparse
    SP = K[(K.method == "csr")]
    Cv = S[(S.method == "csr") & (S.phase == "convert")]
    refs = K[(K.method != "csr") & (K.density < 1.0) & (K.jvm_flags == "default")]
    rows = []
    for _, r in SP.iterrows():
        cv = Cv[(Cv.n == r.n) & (Cv.density == r.density)]
        t_conv = float(cv["median"].iloc[0]) if len(cv) else np.nan
        for _, d in refs[(refs.n == r.n) & (refs.density == r.density)].iterrows():
            rows.append({"n": r.n, "density": r.density, "sparsity_pct": 100 * (1 - r.density), "dense_method": d.method,
                         "t_dense": d["median"], "t_csr_kernel": r["median"], "t_csr_convert": t_conv,
                         "t_csr_kernel_plus_convert": r["median"] + t_conv,
                         "speedup_kernel_only": d["median"] / r["median"],
                         "speedup_incl_conversion": d["median"] / (r["median"] + t_conv),
                         "iqr_pct_csr_kernel": r.iqr_pct_of_median, "iqr_pct_dense": d.iqr_pct_of_median})
    V = pd.DataFrame(rows)
    V.to_csv(tdir / "sparse_vs_dense.csv", index=False)

    nz = df[(df.method == "csr") & (df.phase == "kernel") & (df.rep == 0)][["n", "density", "nnz_a", "nnz_b", "nnz_c"]].copy()
    nz["density_C_observed"] = nz.nnz_c / nz.n ** 2
    nz["density_C_expected_uniform"] = 1 - (1 - nz.density ** 2) ** nz.n
    nz.sort_values(["n", "density"]).to_csv(tdir / "sparse_nnz.csv", index=False)

    ds, ss = cfg["sparse"].get("density_sweep"), cfg["sparse"].get("size_sweep")
    if ds and len(V):
        v = V[V.n == ds["n"]].sort_values("density")
        if len(v):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(100 * v.density.unique(), v.groupby("density")["t_csr_kernel"].first(), "o-", label="CSR kernel")
            ax.plot(100 * v.density.unique(), v.groupby("density")["t_csr_kernel_plus_convert"].first(), "s--",
                    label="CSR kernel + conversion (sum of medians)")
            for m, g in v.groupby("dense_method"):
                ax.plot(100 * g.density, g.t_dense, "^-", label=f"dense {m}")
            style(ax, "Density of A and B (% nonzeros)", "Time per multiplication (s)",
                  f"Sparse vs dense, n = {ds['n']}")
            fig.tight_layout(); fig.savefig(fdir / "sparse_density_sweep.png", dpi=150); plt.close(fig)
    if ss and len(V):
        v = V[np.isclose(V.density, ss["density"])].sort_values("n")
        if len(v):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            u = v.groupby("n").first()
            ax.plot(u.index, u.t_csr_kernel, "o-", label="CSR kernel")
            ax.plot(u.index, u.t_csr_kernel_plus_convert, "s--", label="CSR kernel + conversion")
            for m, g in v.groupby("dense_method"):
                ax.plot(g.n, g.t_dense, "^-", label=f"dense {m}")
            style(ax, "Matrix size n", "Time per multiplication (s)", f"Sparse vs dense, density = {100 * ss['density']:g}%")
            fig.tight_layout(); fig.savefig(fdir / "sparse_size_sweep.png", dpi=150); plt.close(fig)
    if len(nz):
        fig, ax = plt.subplots(figsize=(6, 4))
        for n, g in nz.groupby("n"):
            g = g.sort_values("density")
            ax.plot(100 * g.density, 100 * g.density_C_observed, "o", label=f"observed n={n}")
            ax.plot(100 * g.density, 100 * g.density_C_expected_uniform, "-", lw=.8, label=f"expected n={n}")
        style(ax, "Density of A and B (%)", "Density of C (%)", "Fill-in of the product")
        fig.tight_layout(); fig.savefig(fdir / "sparse_fill_in.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ memory and limits
    ok = st[st.status.isin(["ok", "rep_over_time_limit"])].copy()
    mem = ok[["method", "block", "n", "density", "jvm_flags", "peak_memory_bytes", "live_heap_bytes_after_gc",
              "array_estimate_bytes", "dense_inputs_bytes"]].copy()
    for c in ("peak_memory_bytes", "live_heap_bytes_after_gc", "array_estimate_bytes", "dense_inputs_bytes"):
        mem[c.replace("_bytes", "_mb")] = mem[c] / 1e6
    mem.drop(columns=[c for c in mem.columns if c.endswith("_bytes")]).to_csv(tdir / "memory.csv", index=False)

    lim = []
    for (m, d, b, f), g in st.groupby(["method", "density", "block", "jvm_flags"]):
        g = g.sort_values("n")
        good = g[g.status.isin(["ok", "rep_over_time_limit"])]
        bad = g[~g.status.isin(["ok", "rep_over_time_limit", "skipped_after_failure"])]
        lim.append({"method": m, "density": d, "block": b, "jvm_flags": f,
                    "largest_tested_ok_n": int(good.n.max()) if len(good) else None,
                    "largest_attempted_n": int(g[g.status != "skipped_after_failure"].n.max()),
                    "first_failure": (f"n={int(bad.n.min())}: {bad.sort_values('n').status.iloc[0]}" if len(bad) else "")})
    pd.DataFrame(lim).to_csv(tdir / "limits.csv", index=False)

    print(f"tables -> {tdir}\nfigures -> {fdir}")


if __name__ == "__main__":
    main()
