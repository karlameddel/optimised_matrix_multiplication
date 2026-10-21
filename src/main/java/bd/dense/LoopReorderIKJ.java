package bd.dense;

import java.util.Arrays;

/**
 * Loop reordering i-k-j: B and C are traversed along rows (stride 1).
 * Accumulates into C, so C is zeroed inside the kernel (O(n^2), part of the timed region).
 * For each C[i][j] the products are added in ascending k, like the baseline.
 */
public final class LoopReorderIKJ {
    private LoopReorderIKJ() {}

    public static void multiply(double[] A, double[] B, double[] C, int n) {
        Arrays.fill(C, 0.0);
        for (int i = 0; i < n; i++) {
            int co = i * n;
            for (int k = 0; k < n; k++) {
                double a = A[i * n + k];
                int bo = k * n;
                for (int j = 0; j < n; j++)
                    C[co + j] += a * B[bo + j];
            }
        }
    }
}
