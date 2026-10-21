#!/usr/bin/env python3
"""Correctness validation for Assignment 2 (run BEFORE benchmarking).

Every method (ijk, ikj, blocked with several block sizes, csr) is checked against NumPy (A @ B),
which is used ONLY as a reference and never in a measurement:
  1. hand-computed 2x2 product;
  2. A*I = A, I*B = B, A*0 = 0 (n in --sizes);
  3. random dense matrices (the Assignment 1 inputs: same seeds, regenerated here);
  4. random sparse matrices at several densities (csr, and the dense methods on the same logical
     matrices), including an all-zero row and an all-zero column case;
  5. block sizes that do not divide n, and larger than n (blocked only).
Tolerance (same as Assignment 1): |x - ref| <= ATOL + RTOL*|ref|, RTOL = 1e-9, ATOL = 1e-12.
All entries are >= 0 (no cancellation), so the relative error of a sum of n terms is about n*2^-53.
The nonzero count of the output (csr) must equal NumPy's exactly: with non-negative entries a
product entry is zero only if no pair of nonzeros meets.
Additionally reports whether each result is bit-identical to the ijk baseline. This is expected
(same additions in the same order, no FMA in Java) but is NOT a pass criterion.

Usage: python scripts/verify_correctness.py [--sizes 10 50 100 200]
Writes results/tables/correctness.csv; exit code 1 on any failure.
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from generate_inputs import dense_matrix, sparse_matrix  # noqa: E402

BUILD = ROOT / "build"
OUT_CSV = ROOT / "results" / "tables" / "correctness.csv"
RTOL, ATOL = 1e-9, 1e-12


def run(method, a, b, n, out, block=64):
    cmd = ["java", "-cp", str(BUILD), "bd.Main", "--method", method, "--a", str(a), "--b", str(b),
           "--n", str(n), "--block", str(block), "--reps", "1", "--warmup", "0",
           "--warmup-min-ms", "0", "--batch", "1", "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    line = [l for l in r.stdout.splitlines() if l.startswith("kernel,")][-1].split(",")
    c = np.fromfile(out, dtype="<f8")
    if c.size != n * n:
        raise RuntimeError(f"output has {c.size} values, expected {n * n}")
    return c.reshape(n, n), float(line[4]), int(line[7])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 100, 200])
    args = ap.parse_args()
    if not (BUILD / "bd" / "Main.class").exists():
        sys.exit("missing build/bd/Main.class: compile first (see README)")

    cases = []  # (case, n, A, B, density_label)
    cases.append(("known_2x2", 2, np.array([[1., 2.], [3., 4.]]), np.array([[5., 6.], [7., 8.]]), ""))
    for n in args.sizes:
        A, B = dense_matrix(n, 0), dense_matrix(n, 1)
        I, Z = np.eye(n), np.zeros((n, n))
        cases += [("A_times_identity", n, A, I, ""), ("identity_times_B", n, I, B, ""),
                  ("A_times_zero", n, A, Z, ""), ("random_dense", n, A, B, "1")]
        for d in (0.01, 0.05, 0.1):
            if int(round(d * n * n)) >= 1:
                cases.append(("random_sparse", n, sparse_matrix(n, 0, d), sparse_matrix(n, 1, d), str(d)))
        As = sparse_matrix(n, 0, 0.05)
        As[0, :] = 0.0                       # all-zero row
        Bs = sparse_matrix(n, 1, 0.05)
        Bs[:, 0] = 0.0                       # all-zero column
        cases.append(("zero_row_and_col", n, As, Bs, "0.05"))

    methods = [("ijk", 0), ("ikj", 0), ("csr", 0)] + [("blocked", bs) for bs in (7, 32, 64, 128, 300)]
    rows, all_ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for case, n, A, B, dens in cases:
            a_p, b_p, c_p = tmp / "A.bin", tmp / "B.bin", tmp / "C.bin"
            A.astype("<f8").tofile(a_p)
            B.astype("<f8").tofile(b_p)
            ref = A @ B
            ref_nnz, ref_sum = int(np.count_nonzero(ref)), float(ref.sum())
            base = None
            for method, bs in methods:
                try:
                    c, checksum, nnz_c = run(method, a_p, b_p, n, c_p, bs or 64)
                    err = np.abs(c - ref)
                    nz = ref != 0
                    rel = float((err[nz] / np.abs(ref[nz])).max()) if nz.any() else 0.0
                    ok = bool(np.all(err <= ATOL + RTOL * np.abs(ref))) and \
                        abs(checksum - ref_sum) <= ATOL + RTOL * abs(ref_sum)
                    if method == "csr":
                        ok &= (nnz_c == ref_nnz)
                    if method == "ijk":
                        base = c
                    same = "" if base is None else str(bool(np.array_equal(c, base)))
                    abs_e = float(err.max())
                except Exception as e:
                    abs_e = rel = float("nan"); ok = False; same = ""
                    print(f"  ERROR {method}/{case}/n={n}: {e}")
                all_ok &= ok
                rows.append([case, n, method, bs or "", dens, f"{abs_e:.3e}", f"{rel:.3e}", same,
                             RTOL, ATOL, "PASS" if ok else "FAIL"])
                print(f"[{'PASS' if ok else 'FAIL'}] {case:17s} n={n:<4d} {method:8s} bs={bs or '-':<4} "
                      f"d={dens or '-':<5} max_abs={abs_e:.2e} max_rel={rel:.2e} bit_identical_to_ijk={same or '-'}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "n", "method", "block", "density", "max_abs_error", "max_rel_error",
                    "bit_identical_to_ijk", "rtol", "atol", "result"])
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'} ({len(rows)} tests)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()