#include <iostream>
#include <cstdio>
#include <cassert>
#include <chrono>
#include <set>
using namespace std;

#define BLOCK_DIM 32
#define DATA_BLOCK_DIM 64 // must be <= n and >= BLOCK_DIM. Related to The amount of data a thread block'll load before a set of multiplication operation

__host__ __device__ inline int get_idx(const int& row, const int& col, const int& n) { return row + col*n;} // n=row_dimension

__global__ void mat_mult_gpu(float *ad, float *bd, float* cd, int n)
{
    // column-major ordering
    int thread_row = threadIdx.x % BLOCK_DIM, thread_col = threadIdx.x / BLOCK_DIM;
    int row = blockIdx.y * BLOCK_DIM + thread_row;
    int col = blockIdx.x * BLOCK_DIM + thread_col;

    const int data_block_count = n / DATA_BLOCK_DIM + (n % DATA_BLOCK_DIM != 0);
    __shared__ float a_shared[BLOCK_DIM * DATA_BLOCK_DIM], b_shared[DATA_BLOCK_DIM * BLOCK_DIM];

    // regarding data loading
    int a_col_start = thread_col, b_row_start = thread_row;

    // initialization
    float sum = 0;

    // load data blocks and perform computations on them
    for (int b = 0; b < data_block_count; b++) {
        // load a's block*data_block data into shared memory
        for (int i = a_col_start, j = thread_col; j < DATA_BLOCK_DIM; i += BLOCK_DIM, j += BLOCK_DIM) {
            a_shared[get_idx(thread_row, j, BLOCK_DIM)] = (i<n && row<n)? ad[get_idx(row, i, n)] : 0;
        }
        a_col_start += DATA_BLOCK_DIM;

        // load b's data_block*block data into shared memory
        for (int i = b_row_start, j = thread_row; j < DATA_BLOCK_DIM; i += BLOCK_DIM, j += BLOCK_DIM) {
            b_shared[get_idx(j, thread_col, DATA_BLOCK_DIM)] = (i<n && col<n)? bd[get_idx(i, col, n)] : 0;
        }
        b_row_start += DATA_BLOCK_DIM;

        __syncthreads(); // sync for loading completion

        for (int k = 0; k < DATA_BLOCK_DIM; k++) {
            // printf(" row=%d, col=%d,,,, %0.6f, %0.6f\n", row, col,a_shared[get_idx(thread_row, k, BLOCK_DIM)], b_shared[get_idx(k, thread_col, DATA_BLOCK_DIM)]);
            sum += a_shared[get_idx(thread_row, k, BLOCK_DIM)] * b_shared[get_idx(k, thread_col, DATA_BLOCK_DIM)];
        }

        __syncthreads(); // sync threads before loading new data in the next iteration
    }

    if (row < n && col < n) cd[get_idx(row, col, n)] = sum;
}

void run(int N)
{
    // declaration
    float* a = (float *)malloc(N * N * sizeof(float));
    float* b = (float *)malloc(N * N * sizeof(float));
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

    int blockDimSize = BLOCK_DIM;
    int gridDimSize = N/blockDimSize;
    if (N % blockDimSize) gridDimSize++;

    dim3 dimGrid(gridDimSize, gridDimSize);
    dim3 dimBlock(blockDimSize * blockDimSize); // blockDim 1-dimensional for global memory coalescing

    // Time CUDA kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mat_mult_gpu<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Wait for the GPU to finish before exiting
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Runtime Error after kernel execution: %s\n", cudaGetErrorString(err));
    }

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
    free(resultFromGpu);
}

int main()
{
    int n = 1024;
    run(n);
    return 0;
}