#include <stdio.h>
#include <stdlib.h>
#include <omp.h>    // OpenMP header for thread control
#include "mkl.h" // Includes CBLAS headers for cblas_dgemm

float *A, *B, *C;

void run(int N)
{
    int m = N; // Number of rows in A and C
    int n = N; // Number of columns in B and C
    int k = N; // Number of columns in A and rows in B

    // Allocate memory for matrices (row-major order)
    A = (float *)malloc(m * k * sizeof(float));
    B = (float *)malloc(k * n * sizeof(float));
    C = (float *)malloc(m * n * sizeof(float));

    srand48(0); // seed

    // Initialize A and B with some values
    for (int i = 0; i < m * k; i++) A[i] = drand48();
    for (int i = 0; i < k * n; i++) B[i] = drand48();

    // Define alpha and beta
    float alpha = 1.0;
    float beta = 0.0;

    double start_time = omp_get_wtime();
    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cblas_sgemm(CblasRowMajor,                             // Row-major
                CblasNoTrans, CblasNoTrans,                // no transpose
                m, n, k,                                   // Dimensions: m x n = m x k * k x n
                alpha,                                     // Scaling factor for A * B
                A, k,                                      // Matrix A and leading dimension
                B, n,                                      // Matrix B and leading dimension
                beta,                                      // Scaling factor for C
                C, n);                                     // Matrix C and leading dimension

    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;
    // printf("Matrix multiplication runtime: %0.6f seconds\n", elapsed_time);
    printf("%d,%0.6f\n", N, elapsed_time*1000); // miliseconds
    fflush(stdout);

    // Print the result
    // printf("Matrix C (result):\n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%8.2f ", C[i * n + j]); // Access in row-major order
    //     }
    //     printf("\n");
    // }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Correct format: ./a.out proc_count\n");
        return 0;
    }
    // Set the number of threads
    int num_threads = atoi(argv[1]); // Change this to your desired number of threads
    omp_set_num_threads(num_threads);

    // Print the number of threads
    printf("Number of threads set: %d\n", omp_get_max_threads());

    int n = 1024;
    run(n);
    return 0;
}
