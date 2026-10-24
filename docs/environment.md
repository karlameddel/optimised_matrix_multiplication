# Experimental environment
| Component | Value |
|---|---|
| CPU | Intel(R) Core(TM) Ultra 9 185H |
| Physical cores / logical processors | 16 / 22 |
| RAM | 31.7 GB |
| OS | Windows 11 Home 10.0.26200 |
| Java | OpenJDK 21.0.6 (Temurin-21.0.6+7), default JIT, G1 GC |
| Python / NumPy / pandas / matplotlib / psutil | 3.11.9 / 2.2.6 / 3.0.0 / 3.10.6 / 7.0.0 |
| **Power state** | **Plugged in** |

## Settings
- JVM: `java -Xmx4g -cp build bd.Main ...`, default JIT and GC. Control runs add `-XX:-UseSuperWord`.
- Affinity: benchmark process on logical processors `cpus` of `config/experiment.json`.
- Warm-up: >= 3 repetitions and >= 300 ms; batching to ~20 ms; 5 measured repetitions.
- Budgets: 300 s and 4 GB per run.