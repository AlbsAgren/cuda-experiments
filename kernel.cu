#include "macro.h"

#include<stdio.h>

__global__
void vec_add_kernel(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < n) {
        C[i] = A[i] + B[i];
    }
}

void vec_add(float* A_h, float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d;
    float *B_d;
    float *C_d;

    // Allocate memory on device
    CUDA_CHECK(cudaMalloc((void**)&A_d, size));
    CUDA_CHECK(cudaMalloc((void**)&B_d, size));
    CUDA_CHECK(cudaMalloc((void**)&C_d, size));

    // Copy memory from host to device
    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // Launch kernel with ceil(n/256.0) blocks of 256 threads
    vec_add_kernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

void fill_array_with_random_floats(float* array, int size) {
    srand(0);
    for(int i = 0; i < size; i++) {
        array[i] = (float)rand() / (float)rand();
    }
}

void print_array(float* array, int size) {
    printf("\n");
    for(int i = 0; i < size; i++) {
        printf("\nIdx: %d, value: %.2f", i, array[i]);
    }
    printf("\n");
}

int main() {
    int size = 10;
    float A_h[size];
    float B_h[size];
    float C_h[size];

    fill_array_with_random_floats(A_h, size);
    fill_array_with_random_floats(B_h, size);

    vec_add(A_h, B_h, C_h, size);

    print_array(C_h, size);
}
