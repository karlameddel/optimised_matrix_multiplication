package bd.dense;

/**
 * Baseline of Assignment 1 (unchanged): triple loop i-j-k, scalar accumulator.
 * B is read down a column (stride n). Overwrites every entry of C, so C needs no zeroing.
 */
public final class BasicIJK {
    private BasicIJK() {}

    public static void multiply(double[] A, double[] B, double[] C, int n) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++)
                    s += A[i * n + k] * B[k * n + j];
                C[i * n + j] = s;
            }
    }
}
