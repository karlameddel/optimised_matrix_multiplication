package bd.sparse;

/**
 * Compressed Sparse Row matrix (own implementation, no library).
 *   rowPtr[i]..rowPtr[i+1]-1 index the stored entries of row i
 *   colIdx[p] column of entry p        val[p] value of entry p
 * Storage: 8 B (double) + 4 B (int) per nonzero + 4 B per row pointer.
 */
public final class CSRMatrix {
    public final int n;
    public final int[] rowPtr;
    public final int[] colIdx;
    public final double[] val;

    public CSRMatrix(int n, int[] rowPtr, int[] colIdx, double[] val) {
        this.n = n;
        this.rowPtr = rowPtr;
        this.colIdx = colIdx;
        this.val = val;
    }

    public int nnz() { return rowPtr[n]; }

    /** Payload of the three arrays in bytes (not JVM object overhead). */
    public long storageBytes() { return 12L * nnz() + 4L * (n + 1); }

    /** CSR -> dense. Validation and output file only; never timed. */
    public double[] toDense() {
        double[] m = new double[n * n];
        for (int i = 0; i < n; i++)
            for (int p = rowPtr[i]; p < rowPtr[i + 1]; p++)
                m[i * n + colIdx[p]] = val[p];
        return m;
    }
}
