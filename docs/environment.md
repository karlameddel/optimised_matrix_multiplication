# Experimental environment (fill in at measurement time)

Run `python scripts/capture_environment.py` and copy the relevant values here.

| Component | Value |
|---|---|
| CPU | Intel Core Ultra 9 185H (hybrid) |
| Physical cores / logical processors | 16 / 22 |
| RAM | 31.7 GB |
| OS | Windows 11 Home 10.0.26200 |
| Java | OpenJDK 21.0.6 (Temurin) -- CONFIRM, default JIT, default GC (record `#gc` from the output) |
| Python / NumPy / pandas / matplotlib / psutil | 3.11.9 / 2.2.6 / 3.0.0 / 3.10.6 / 7.0.0 |
| **Power state** | **<plugged in or battery?> / <power mode> -- RECORD BEFORE RUNNING** |
| Battery level and temperature | <record> |

## Settings
- JVM: `java -Xmx4g -cp build bd.Main ...`, default JIT and GC. Control runs add `-XX:-UseSuperWord`.
- Affinity: benchmark process on logical processors `cpus` of `config/experiment.json`; harness pinned with
  `--harness-cpus`. Mapping to physical/P cores: <verified with Coreinfo? yes/no>.
- Warm-up: >= 3 repetitions and >= 300 ms; batching to ~20 ms; 5 measured repetitions.
- Budgets: 300 s and 4 GB per run.

## Changes with respect to Assignment 1
- Harness is pinned (`--harness-cpus`) and the power state is recorded for every configuration.
- Warm-up has a minimum duration and short kernels are batched (A1 reported microsecond kernels unbatched).
- Memory: live heap after GC added next to the peak working set.

## Known limitations
<fill in after the runs>
