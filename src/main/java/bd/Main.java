package bd;

import bd.dense.BasicIJK;
import bd.dense.Blocked;
import bd.dense.LoopReorderIKJ;
import bd.io.MatrixIO;
import bd.sparse.CSRMatrix;
import bd.sparse.CSRMultiply;
import bd.sparse.Converter;
import bd.util.Checksum;

import java.io.IOException;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Assignment 2 benchmark driver (one configuration per process).
 *
 * Usage:
 *   java -Xmx4g -cp build bd.Main --method ijk|ikj|blocked|csr --a A.bin --b B.bin --n N
 *        [--block 64] [--reps 5] [--warmup 3] [--warmup-min-ms 300]
 *        [--batch 0] [--target-ms 20] [--out C.bin]
 *
 * Timed regions (the timer never includes input loading, checksums or nonzero counting):
 *   ijk | ikj | blocked  phase "kernel": the multiplication (ikj/blocked include zeroing C).
 *                        C is allocated outside the timer.
 *   csr                  phase "convert": dense A,B -> CSR (both matrices, allocation included)
 *                        phase "kernel" : CSR x CSR (Gustavson), allocation of C included
 *                        The two phases are timed separately and never added by this program.
 *
 * Warm-up: at least --warmup repetitions AND at least --warmup-min-ms of warm-up time, so short
 * kernels also reach JIT steady state. Excluded from the reported samples.
 * Batching: --batch 0 (auto) sizes each sample so that it lasts about --target-ms; the reported
 * time is always seconds PER multiplication (total / batch). --batch N forces N.
 *
 * stdout: one CSV line per measured sample
 *   phase,rep,batch,seconds_per_op,checksum,nnz_a,nnz_b,nnz_c
 * followed by "#key=value" metadata lines (live heap, estimates, JVM information).
 */
public final class Main {
    private static double[] A, B, C;          // dense
    private static CSRMatrix cA, cB, cC;            // sparse
    private static int n, block;

    interface Op { void run(); }

    public static void main(String[] args) throws IOException {
        Map<String, String> o = parse(args);
        String method = req(o, "method");
        n = Integer.parseInt(req(o, "n"));
        block = Integer.parseInt(o.getOrDefault("block", "64"));
        int reps = Integer.parseInt(o.getOrDefault("reps", "5"));
        int warmup = Integer.parseInt(o.getOrDefault("warmup", "3"));
        long warmupMinMs = Long.parseLong(o.getOrDefault("warmup-min-ms", "300"));
        int batchOpt = Integer.parseInt(o.getOrDefault("batch", "0"));
        double targetMs = Double.parseDouble(o.getOrDefault("target-ms", "20"));
        boolean sparse = method.equals("csr");
        if (!sparse && !method.equals("ijk") && !method.equals("ikj") && !method.equals("blocked"))
            throw new IllegalArgumentException("unknown method: " + method);
        if (block < 1) throw new IllegalArgumentException("block must be >= 1");

        A = MatrixIO.load(req(o, "a"), n);
        B = MatrixIO.load(req(o, "b"), n);
        long nnzA = Checksum.countNonZero(A), nnzB = Checksum.countNonZero(B);
        String[] phases;
        Op[] ops;
        if (sparse) {
            phases = new String[] {"convert", "kernel"};
            ops = new Op[] {
                () -> { cA = Converter.fromDense(A, n); cB = Converter.fromDense(B, n); },
                () -> { cC = CSRMultiply.multiply(cA, cB); }
            };
        } else {
            C = new double[n * n];
            phases = new String[] {"kernel"};
            ops = new Op[] {
                switch (method) {
                    case "ijk" -> () -> BasicIJK.multiply(A, B, C, n);
                    case "ikj" -> () -> LoopReorderIKJ.multiply(A, B, C, n);
                    default -> () -> Blocked.multiply(A, B, C, n, block);
                }
            };
        }

        // ---- warm-up (excluded) ----
        long wStart = System.nanoTime();
        int w = 0;
        double[] lastPhaseNs = new double[phases.length];
        while (w < 100000 && (w < warmup || (System.nanoTime() - wStart) / 1e6 < warmupMinMs)) {
            for (int p = 0; p < ops.length; p++) {
                long t0 = System.nanoTime();
                ops[p].run();
                lastPhaseNs[p] = System.nanoTime() - t0;
            }
            w++;
        }

        // ---- batch size per phase ----
        int[] batch = new int[phases.length];
        for (int p = 0; p < phases.length; p++) {
            if (batchOpt > 0) batch[p] = batchOpt;
            else if (w == 0) batch[p] = 1;      // no warm-up: no estimate
            else batch[p] = (int) Math.max(1, Math.min(100000, Math.ceil(targetMs * 1e6 / Math.max(1.0, lastPhaseNs[p]))));
        }

        // ---- measured samples ----
        for (int r = 0; r < reps; r++) {
            for (int p = 0; p < ops.length; p++) {
                long t0 = System.nanoTime();
                for (int b = 0; b < batch[p]; b++) ops[p].run();
                long t1 = System.nanoTime();
                double secPerOp = (t1 - t0) / 1e9 / batch[p];
                // checksum and counts: outside the timed region, consumes the result
                double checksum;
                long nnzC;
                if (sparse) {
                    if (phases[p].equals("convert")) { checksum = Checksum.sum(cA.val) + Checksum.sum(cB.val); nnzC = -1; }
                    else { checksum = Checksum.sum(cC.val); nnzC = cC.nnz(); }
                } else { checksum = Checksum.sum(C); nnzC = Checksum.countNonZero(C); }
                System.out.printf(Locale.ROOT, "%s,%d,%d,%.9e,%.12e,%d,%d,%d%n",
                        phases[p], r, batch[p], secPerOp, checksum, nnzA, nnzB, nnzC);
                System.out.flush();
            }
        }

        // ---- metadata: live heap after a full GC (arrays still referenced) ----
        System.gc();
        Runtime rt = Runtime.getRuntime();
        long live = rt.totalMemory() - rt.freeMemory();
        long estimate, inputsDense;
        if (sparse) { estimate = cA.storageBytes() + cB.storageBytes() + cC.storageBytes(); inputsDense = 16L * n * n; }
        else { estimate = 24L * n * n; inputsDense = 0; }
        StringBuilder gcs = new StringBuilder();
        for (GarbageCollectorMXBean g : ManagementFactory.getGarbageCollectorMXBeans()) gcs.append(g.getName()).append('/');
        System.out.println("#live_heap_bytes_after_gc=" + live);
        System.out.println("#array_estimate_bytes=" + estimate);
        System.out.println("#dense_inputs_bytes=" + inputsDense);
        System.out.println("#warmup_iterations=" + w);
        System.out.println("#java_version=" + System.getProperty("java.version") + " " + System.getProperty("java.vm.name"));
        System.out.println("#available_processors=" + rt.availableProcessors());
        System.out.println("#max_heap_bytes=" + rt.maxMemory());
        System.out.println("#gc=" + gcs);

        if (o.containsKey("out")) {
            double[] dense = sparse ? cC.toDense() : C;
            MatrixIO.save(o.get("out"), dense, n);
        }
        // keep everything reachable until after System.gc()
        java.lang.ref.Reference.reachabilityFence(A);
        java.lang.ref.Reference.reachabilityFence(B);
        if (sparse) { java.lang.ref.Reference.reachabilityFence(cA); java.lang.ref.Reference.reachabilityFence(cB); java.lang.ref.Reference.reachabilityFence(cC); }
        else java.lang.ref.Reference.reachabilityFence(C);
    }



    private static String req(Map<String, String> o, String k) {
        String v = o.get(k);
        if (v == null) throw new IllegalArgumentException("missing --" + k);
        return v;
    }

    private static Map<String, String> parse(String[] args) {
        Map<String, String> m = new HashMap<>();
        for (int i = 0; i < args.length; i += 2) {
            if (!args[i].startsWith("--") || i + 1 >= args.length)
                throw new IllegalArgumentException("bad argument near '" + args[i] + "'");
            m.put(args[i].substring(2), args[i + 1]);
        }
        return m;
    }
}
