# Optimised Dense and Sparse Matrix Multiplication in Java

**Big Data · Assignment 2 · Academic year 2026-2027**

## Research questions

> **A.** How much do loop reordering (ikj) and cache blocking reduce the runtime of dense
> `n × n` float64 multiplication in Java, compared with the basic i-j-k triple loop of
> Assignment 1, on one machine and one thread?
>
> **B.** For which tested sparsity levels and sizes does a CSR × CSR (Gustavson) multiplication
> beat a dense method on the same logical matrices, and how much do conversion and the
> density of the product reduce that advantage?

Conclusions are limited to the tested implementations, inputs and hardware.

## Reuse from Assignment 1

| Reused | Where | Note |
|---|---|---|
| Baseline kernel (i-j-k) | `src/main/java/bd/dense/BasicIJK.java` | Same loop and indexing as `MatMul.matmul` of Assignment 1 (repo `basic_matrix_multiplication`, commit `<FILL IN: git rev-parse --short HEAD>`) |
| Dense input matrices | `scripts/generate_inputs.py` | Same seeds `[42, n, tag]` -> same bytes; `--check-a1` compares SHA-256 with the Assignment 1 manifest |
| Harness protocol | `scripts/run_benchmarks.py` | One process per configuration, probe + timing run, limits 300 s / 4 GB, median and IQR |
| Correctness criterion | `scripts/verify_correctness.py` | `rtol = 1e-9`, `atol = 1e-12`, NumPy as reference only |

Measurements of Assignment 1 are **not** reused: the baseline is measured again in this session so that
all methods share the same conditions.

## Layout

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── config/experiment.json        # sizes, block sizes, densities, JVM controls, limits
├── data/
│   ├── manifest.json             # seeds, nnz and SHA-256 of every input
│   ├── dense/                    # A_n<n>.bin, B_n<n>.bin   (same files as Assignment 1; not versioned)
│   └── sparse/                   # A_n<n>_d<density>.bin ... (not versioned)
├── src/main/java/bd/
│   ├── Main.java                 # CLI + timing loop (one configuration per process)
│   ├── dense/    BasicIJK.java, LoopReorderIKJ.java, Blocked.java
│   ├── sparse/   CSRMatrix.java, Converter.java (dense -> CSR), CSRMultiply.java (Gustavson)
│   ├── io/       MatrixIO.java
│   └── util/     Checksum.java
├── scripts/
│   ├── generate_inputs.py
│   ├── verify_correctness.py     # run BEFORE benchmarking
│   ├── run_benchmarks.py
│   └── make_report_assets.py     # tables and figures from the raw data
├── results/                      # raw_measurements.csv, run_status.csv, tables/, figures/
└── docs/environment.md
```

## Requirements

Windows 11 (the harness reads the Windows peak working set; on Linux it falls back to RSS),
JDK 21 (`javac` and `java` on the PATH), Python 3.11 with `requirements.txt` in a virtual environment.
Plug the laptop in, select the best-performance power mode, close other applications and do not
use the machine during the benchmark. Keep the same power state for the whole session.

## How to reproduce (PowerShell, repository root)

```powershell
# 0. Virtual environment (once) and dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 1. Build (no special flags; the JVM options are part of the experiment, see docs/environment.md)
mkdir build
javac -d build (Get-ChildItem src/main/java -Recurse -Filter *.java).FullName

# 2. Inputs (about 1 GB in total) and check against Assignment 1
python scripts/generate_inputs.py --check-a1 path\to\assignment1\data\manifest.json

# 3. Correctness (before any measurement; must end with ALL TESTS PASSED)
python scripts/verify_correctness.py

# 4. Benchmarks, plugged in (resumable; --dry-run lists the configurations).
#    Record the environment in docs/environment.md first.
python scripts/run_benchmarks.py --dry-run
python scripts/run_benchmarks.py --harness-cpus 0 1

# 5. Tables and figures
python scripts/make_report_assets.py
```

Run one configuration by hand:

```powershell
java -Xmx4g -cp build bd.Main --method blocked --a data\dense\A_n1000.bin --b data\dense\B_n1000.bin --n 1000 --block 64 --reps 5 --warmup 3
java -Xmx4g -cp build bd.Main --method csr --a data\sparse\A_n2000_d0.01.bin --b data\sparse\B_n2000_d0.01.bin --n 2000 --reps 5 --warmup 3
```

Output: `phase,rep,batch,seconds_per_op,checksum,nnz_a,nnz_b,nnz_c`, then `#key=value` metadata.
Methods: `ijk`, `ikj`, `blocked` (uses `--block`), `csr`.

## Experimental design in brief

| Aspect | Setting |
|---|---|
| Dense methods | `ijk` (baseline), `ikj` (loop reordering), `blocked` (ikj inside bs × bs × bs tiles; bs from a sweep at one n, main comparison with one fixed bs) |
| Sparse method | CSR × CSR, Gustavson row-by-row, dense accumulator of length n; output columns not sorted |
| Dense reference for Part B | `dense_reference_methods` in the config, run on the SAME sparse input files |
| Numeric type | float64; Java does not fuse multiply-add, so results are expected bit-identical across methods (checked, not required) |
| Dense inputs | Assignment 1 files, uniform [0, 1) |
| Sparse inputs | Dense files containing zeros: exactly `round(d·n²)` uniformly random positions without replacement, values uniform (0, 1], seed `[42, n, tag, round(d·1e6)]` |
| Timed region, dense | Kernel only; C allocated outside; `ikj`/`blocked` zero C inside (O(n²)) |
| Timed region, sparse | Two separate phases: `convert` (dense A and B -> CSR, allocation included) and `kernel` (CSR × CSR, allocation of C included). Never added by the program; the analysis script reports "kernel" and "kernel + conversion" separately |
| Warm-up / batching | ≥ 3 repetitions and ≥ 300 ms; samples of short kernels are batched to ≈ 20 ms and reported as seconds per operation |
| Repetitions | 5 (3 and no warm-up if the probe kernel exceeds 30 s) |
| Memory | Peak working set of the process (psutil, probe run) + live heap after GC (in process) + array-storage estimate. Three different quantities |
| Budgets | 300 s and 4 GB per run; violations recorded as results |
| Control | Same dense kernels with `-XX:-UseSuperWord` (JIT auto-vectorisation off) to separate the memory-access effect from SIMD |

Results, analysis and limitations are in the PDF report (`report/`).

## Output files

`results/raw_measurements.csv`: one row per measured sample.
`timestamp, method, block, n, density, numeric_type, threads, jvm_flags, timer, phase, rep, batch,
seconds_per_op, checksum, nnz_a, nnz_b, nnz_c, warmup_iterations, reps_planned, affinity`
(`density` empty = fully dense random input; `phase` = `kernel` or `convert`; `nnz_c = -1` on conversion rows).

`results/run_status.csv`: one row per configuration with status, probe time, peak memory, live heap,
estimates, limits, affinity, power state and JVM/GC.

The environment (CPU, RAM, OS, versions, power state, affinity) is documented in `docs/environment.md`.

`results/tables/`: `summary_time.csv`, `dense_summary.csv`, `block_sweep.csv`, `jvm_control.csv`,
`sparse_vs_dense.csv`, `sparse_nnz.csv`, `memory.csv`, `limits.csv`, `correctness.csv`.

## Use of generative AI

Generative AI (Claude) was used for planning, drafting and debugging code and documentation, in line
with the course policy. See the report for details.