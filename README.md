# Optimised Dense and Sparse Matrix Multiplication in Java

**Big Data · Assignment 2 · Academic year 2026-2027**

## Research questions

> **A.** How much do loop reordering (ikj) and cache blocking reduce the runtime of dense
> `n × n` float64 multiplication in Java, compared with the basic i-j-k triple loop of
> Assignment 1, on one machine and one thread?
>
> **B.** For which tested sparsity levels and sizes does a CSR × CSR (Gustavson) multiplication
> beat the best dense method on the same logical matrices, and how much do conversion and the
> density of the product reduce that advantage?

Conclusions are limited to the tested implementations, inputs and hardware.

## Reuse from Assignment 1

| Reused | Where | Note |
|---|---|---|
| Baseline kernel `ijk` | `DenseKernels.ijk` | Same loop and indexing as `MatMul.matmul` of Assignment 1 (repo `basic_matrix_multiplication`, commit `<FILL IN: git rev-parse --short HEAD>`) |
| Dense input matrices | `scripts/generate_inputs.py` | Same seeds `[42, n, tag]` -> same bytes; `--check-a1` compares SHA-256 with the Assignment 1 manifest |
| Harness protocol | `scripts/run_benchmarks.py` | One process per configuration, probe + timing run, limits 300 s / 4 GB, median and IQR |
| Correctness criterion | `scripts/verify_correctness.py` | `rtol = 1e-9`, `atol = 1e-12`, NumPy as reference only |

## Layout

```
optimized_sparse_matrix_multiplication/
├── README.md
├── requirements.txt              # numpy, pandas, matplotlib, psutil, scipy (solo verificación)
├── .gitignore                    # data/*.bin, out/, __pycache__
├── config/
│   └── experiment.json           # tamaños, bloques, densidades, semillas, repeticiones, límites
├── data/
│   ├── manifest.json             # SHA-256 de cada fichero de entrada
│   ├── dense/                    # A_n.bin, B_n.bin (float64, little-endian, row-major)
│   └── sparse/                   # A_n_dXX.bin, B_n_dXX.bin (formato definido abajo)
├── src/
│   └── main/java/bd/
│       ├── Main.java             # CLI: método, n, densidad, nº de repeticiones, warm-up
│       ├── io/MatrixIO.java      # lectura de binarios dense y sparse
│       ├── dense/
│       │   ├── BasicIJK.java     # baseline reutilizado del Assignment 1
│       │   ├── LoopReorderIKJ.java
│       │   └── Blocked.java      # tamaño de bloque como parámetro
│       ├── sparse/
│       │   ├── CSRMatrix.java    # values[], colIdx[], rowPtr[]
│       │   ├── CSRMultiply.java  # CSR x CSR (Gustavson), código propio
│       │   └── Converter.java    # dense -> CSR (medido aparte)
│       └── util/Checksum.java
├── scripts/
│   ├── generate_inputs.py        # dense + sparse con PCG64, escribe manifest
│   ├── verify_correctness.py     # identidad, ceros, 2x2, aleatorias vs NumPy/SciPy
│   ├── run_benchmarks.py         # harness: un proceso por configuración, límites 300 s / 4 GB
│   └── make_report_assets.py     # tablas y figuras desde el CSV
├── results/
│   ├── raw_measurements.csv
│   ├── environment.json          # CPU, RAM, JDK, flags, estado de energía
│   ├── tables/
│   │   ├── correctness.csv
│   │   ├── dense_summary.csv
│   │   └── sparse_summary.csv
│   └── figures/
└── report/
    ├── Assignment02_Surname_Report.pdf
    └── main.tex (y .bib)
```

## Requirements

Windows 11 (the harness reads the Windows peak working set; on Linux it falls back to RSS),
JDK 21 (`javac`, `java` on the PATH), Python 3.11 with `requirements.txt`.
Plug the laptop in, select the best-performance power mode, close other applications and do not
use the machine during the benchmark.

## How to reproduce (PowerShell, repository root)

```powershell
python -m pip install -r requirements.txt

# 1. Build (no special flags; the JVM options are part of the experiment, see docs/environment.md)
mkdir build
javac -d build (Get-ChildItem src/java -Recurse -Filter *.java).FullName

# 2. Inputs (about 1 GB for the default configuration) and check against Assignment 1
python scripts/generate_inputs.py --check-a1 path\to\assignment1\data\manifest.json

# 3. Correctness (before any measurement)
python scripts/verify_correctness.py

# 4. Environment, then benchmarks (resumable; --dry-run lists the configurations)
python scripts/capture_environment.py
python scripts/run_benchmarks.py --dry-run
python scripts/run_benchmarks.py --harness-cpus 0 1

# 5. Tables and figures
python scripts/analyze.py
```

Run one configuration by hand:

```powershell
java -Xmx4g -cp build bd.Main --method blocked --a data\A_n1000.bin --b data\B_n1000.bin --n 1000 --block 64 --reps 5 --warmup 3
java -Xmx4g -cp build bd.Main --method csr     --a data\A_n2000_d0.01.bin --b data\B_n2000_d0.01.bin --n 2000 --reps 5 --warmup 3
```

Output: `phase,rep,batch,seconds_per_op,checksum,nnz_a,nnz_b,nnz_c`, then `#key=value` metadata.

## Experimental design in brief

| Aspect | Setting |
|---|---|
| Dense methods | `ijk` (baseline), `ikj` (loop reordering), `blocked` (ikj inside bs × bs × bs tiles; bs from a sweep at one n, main comparison with one fixed bs) |
| Sparse method | CSR × CSR, Gustavson row-by-row, dense accumulator of length n; output columns not sorted |
| Dense reference for Part B | `dense_reference_methods` in the config, run on the SAME sparse input files |
| Numeric type | float64; no FMA, no `-ffast-math` equivalent; results are expected bit-identical across methods (checked, not required) |
| Dense inputs | Assignment 1 files, uniform [0, 1) |
| Sparse inputs | Dense files containing zeros: exactly `round(d·n²)` uniformly random positions without replacement, values uniform (0, 1], seed `[42, n, tag, round(d·1e6)]` |
| Timed region, dense | Kernel only; C allocated outside; `ikj`/`blocked` zero C inside (O(n²)) |
| Timed region, sparse | Two separate phases: `convert` (dense A and B -> CSR, allocation included) and `kernel` (CSR × CSR, allocation of C included). Never added by the program; `analyze.py` reports "kernel" and "kernel + conversion" separately |
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

## Use of generative AI

Generative AI (Claude) was used for planning, drafting and debugging code and documentation, in line
with the course policy. See the report for details.
