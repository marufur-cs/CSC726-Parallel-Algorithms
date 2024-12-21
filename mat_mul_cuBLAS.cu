#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <set>
using namespace std;

void run(int N)
{
    // int N = atoi(argv[1]);

    // Matrix dimensions
    int m = N; // Number of rows in A and C
    int n = N; // Number of columns in B and C
    int k = N; // Number of columns in A and rows in B

    // Allocate memory for matrices (row-major order)
    double *h_A = (double *)malloc(m * k * sizeof(double));
    double *h_B = (double *)malloc(k * n * sizeof(double));
    // double *h_C = (double *)malloc(m * n * sizeof(double));

    srand48(0); // seed

    // Initialize A and B with some values
    for (int i = 0; i < m * k; i++) h_A[i] = drand48();
    for (int i = 0; i < k * n; i++) h_B[i] = drand48();

    // Device matrices
    double *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, m * k * sizeof(double));
    cudaMalloc((void **)&d_B, k * n * sizeof(double));
    cudaMalloc((void **)&d_C, m * n * sizeof(double));

    // Copy matrices A, B, and C to the device
    cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_C, h_C, m * n * sizeof(double), cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set alpha and beta
    double alpha = 1.0;
    double beta = 0.0;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record the start event
    cudaEventRecord(start, 0);

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                d_A, m,
                d_B, k,
                &beta,
                d_C, m);


    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Wait for the GPU to finish before exiting
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Runtime Error after kernel execution: %s\n", cudaGetErrorString(err));
    }

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    // printf("Matrix multiplication completed in %.6f seconds\n", elapsed_time_ms/1000);

    // Copy result back to host
    // cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Print result
    // printf("Matrix C (result):\n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%8.2f ", h_C[i * n + j]);
    //     }
    //     printf("\n");
    // }
    printf("%d,%0.6f\n", N, elapsed_time_ms);
    fflush(stdout);

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
}

int main(int argc, char *argv[]) {
    int n = 1024;
    run(n);
    return 0;
}
