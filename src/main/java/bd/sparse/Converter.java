package bd.sparse;

/** Dense -> CSR conversion (timed separately from the multiplication). */
public final class Converter {
    private Converter() {}

    /** Two passes (count, then fill) with exact allocation. Entries equal to 0.0 are dropped. */
    public static CSRMatrix fromDense(double[] m, int n) {
        int[] rowPtr = new int[n + 1];
        int nnz = 0;
        for (int i = 0; i < n; i++) {
            int base = i * n;
            for (int j = 0; j < n; j++) if (m[base + j] != 0.0) nnz++;
            rowPtr[i + 1] = nnz;
        }
        int[] colIdx = new int[nnz];
        double[] val = new double[nnz];
        int p = 0;
        for (int i = 0; i < n; i++) {
            int base = i * n;
            for (int j = 0; j < n; j++) {
                double v = m[base + j];
                if (v != 0.0) { colIdx[p] = j; val[p] = v; p++; }
            }
        }
        return new CSRMatrix(n, rowPtr, colIdx, val);
    }
}
