package bd.sparse;

import java.util.Arrays;

/**
 * C = A * B for CSR matrices: Gustavson's row-by-row algorithm with a dense accumulator of
 * length n. For row i of C, every stored A[i][k] scales row k of B and is merged into acc[].
 * marker[j] holds the last row that touched column j, so acc[] never needs clearing.
 * The result arrays grow by doubling and are trimmed to the exact size at the end (allocation is
 * inside the timed region because the output size depends on the data).
 * The columns of each row of C are NOT sorted.
 */
public final class CSRMultiply {
    private CSRMultiply() {}

    public static CSRMatrix multiply(CSRMatrix A, CSRMatrix B) {
        final int n = A.n;
        final double[] acc = new double[n];
        final int[] marker = new int[n];
        Arrays.fill(marker, -1);
        final int[] cols = new int[n];
        final int[] cPtr = new int[n + 1];
        int cap = Math.max(16, A.nnz() + B.nnz());
        int[] cIdx = new int[cap];
        double[] cVal = new double[cap];
        int nnz = 0;

        for (int i = 0; i < n; i++) {
            int cnt = 0;
            for (int p = A.rowPtr[i]; p < A.rowPtr[i + 1]; p++) {
                final int k = A.colIdx[p];
                final double a = A.val[p];
                for (int q = B.rowPtr[k]; q < B.rowPtr[k + 1]; q++) {
                    final int j = B.colIdx[q];
                    if (marker[j] != i) { marker[j] = i; acc[j] = a * B.val[q]; cols[cnt++] = j; }
                    else acc[j] += a * B.val[q];
                }
            }
            if (nnz + cnt > cIdx.length) {
                long want = Math.max(2L * cIdx.length, (long) nnz + cnt);
                if (want > Integer.MAX_VALUE - 8) want = Integer.MAX_VALUE - 8;
                if (want < (long) nnz + cnt) throw new IllegalStateException("result has more than 2^31 nonzeros");
                cIdx = Arrays.copyOf(cIdx, (int) want);
                cVal = Arrays.copyOf(cVal, (int) want);
            }
            for (int t = 0; t < cnt; t++) { int j = cols[t]; cIdx[nnz] = j; cVal[nnz] = acc[j]; nnz++; }
            cPtr[i + 1] = nnz;
        }
        return new CSRMatrix(n, cPtr, Arrays.copyOf(cIdx, nnz), Arrays.copyOf(cVal, nnz));
    }
}
