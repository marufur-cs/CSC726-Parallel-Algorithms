#include <iostream>
#include <cstdio>
#include <cassert>
#include <chrono>
#include <set>
using namespace std;

__host__ __device__ inline int get_idx(const int& row, const int& col, const int& n) { return row + col*n;}

__global__ void mat_mult_gpu(float *ad, float *bd, float* cd, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= n || col >= n) return;

    float sum = 0;
    for (int k = 0; k < n; k++) {
        sum += ad[get_idx(row, k, n)] * bd[get_idx(k, col, n)];
    }

    cd[get_idx(row, col, n)] = sum;
}

void run(int N)
{
    // declaration
    float* a = (float *)malloc(N * N * sizeof(float));
    float* b = (float *)malloc(N * N * sizeof(float));
    float* c = (float *)malloc(N * N * sizeof(float));
    float* resultFromGpu = (float *)malloc(N * N * sizeof(float));

    // value population
    srand48(42); // seed
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[get_idx(i, j, N)] = drand48();
            b[get_idx(i, j, N)] = drand48();
        }
    }

    // GPU call section starts
    float *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, N * N * sizeof(float));
    cudaMalloc((void**)&dev_b, N * N * sizeof(float));
    cudaMalloc((void**)&dev_c, N * N * sizeof(float));

    cudaMemcpy(dev_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice);

    int blockDimSize = 32;
    int gridDimSize = N/blockDimSize;
    if (N % blockDimSize) gridDimSize++;

    dim3 dimGrid(gridDimSize, gridDimSize);
    dim3 dimBlock(blockDimSize, blockDimSize);

    // Time CUDA kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mat_mult_gpu<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);

    // Wait for the GPU to finish before exiting
    cudaDeviceSynchronize();

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cudaTime;
    cudaEventElapsedTime(&cudaTime, start, stop);

    printf("CUDA Execution Time: %0.6f ms\n", cudaTime);

    // Copy the result from gpu
    cudaMemcpy(resultFromGpu, dev_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free Memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
    free(resultFromGpu);
}

int main()
{
    int n = 1024;
    run(n);
    return 0;
}