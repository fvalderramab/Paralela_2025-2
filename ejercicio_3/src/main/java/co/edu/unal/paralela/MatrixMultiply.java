package co.edu.unal.paralela;

import static edu.rice.pcdp.PCDP.forseq2d;
import static edu.rice.pcdp.PCDP.forall2dChunked;

/**
 * Clase envolvente pata implementar de forma eficiente la multiplicación dde matrices en paralelo.
 */
public final class MatrixMultiply {
    /**
     * Constructor por omisión.
     */
    private MatrixMultiply() {
    }

    /**
     * Realiza una multiplicación de matrices bidimensionales (A x B = C) de forma secuencial.
     *
     * @param A Una matriz de entrada con dimensiones NxN
     * @param B Una matriz de entrada con dimensiones NxN
     * @param C Matriz de salida
     * @param N Tamaño de las matrices de entrada
     */
    public static void seqMatrixMultiply(final double[][] A, final double[][] B,
            final double[][] C, final int N) {
        forseq2d(0, N - 1, 0, N - 1, (i, j) -> {
            C[i][j] = 0.0;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        });
    }

    /**
     * Método auxiliar para transponer una matriz.
     * Convierte el acceso por columnas (ineficiente) en acceso por filas (eficiente).
     * * @param matrix Matriz original
     * @param N Tamaño de la matriz
     * @return Nueva matriz transpuesta
     */
    private static double[][] transpose(final double[][] matrix, final int N) {
        final double[][] transposed = new double[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    /**
     * Realiza una multiplicación de matrices bidimensionales (A x B = C) de forma paralela.
     * Incluye optimización de localidad de datos mediante transposición.
     */
    public static void parMatrixMultiply(final double[][] A, final double[][] B,
            final double[][] C, final int N) {
        
        /* * OPTIMIZACIÓN DE DATOS (Data Locality):
         * Transponemos la matriz B antes del cálculo paralelo.
         * Costo: O(N^2) (despreciable comparado con O(N^3) de la multiplicación).
         * Beneficio: Convertimos el acceso a B en secuencial, reduciendo drásticamente
         * los cache misses y aprovechando el ancho de banda de memoria.
         */
        final double[][] B_trans = transpose(B, N);
        
        /*
         * OPTIMIZACIÓN DE CICLOS (Loop Parallelism):
         * Chunk size de 16 seleccionado empíricamente para balancear carga vs overhead.
         */
        final int CHUNK_SIZE = 16;
        
        forall2dChunked(0, N - 1, 0, N - 1, CHUNK_SIZE, (i, j) -> {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                // Usamos B_trans[j][k] en lugar de B[k][j]
                // Esto permite acceso contiguo en memoria para ambas matrices.
                sum += A[i][k] * B_trans[j][k];
            }
            C[i][j] = sum;
        });
    }
}
