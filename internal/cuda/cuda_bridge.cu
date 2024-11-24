// Codi  per a la interacció Go <-> C.
// CreatedAt: 2024/11/24 dg. JIQ

#include "cuda_bridge.h"
#include "fixed_point.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

// Declaració del kernel
extern __global__ void process_kernel(fixed_point_t* input, fixed_point_t* output, int size);

extern "C" {

int cuda_init(void) {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

void cuda_cleanup(void) {
    cudaDeviceReset();
}

int cuda_process_fixed_point(int32_t* input, int32_t* output, int size) {
    fixed_point_t *d_input, *d_output;
    cudaError_t err;
    
    // Aloca memòria a la GPU
    err = cudaMalloc(&d_input, size * sizeof(fixed_point_t));
    if (err != cudaSuccess) {
        printf("CUDA malloc input error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    err = cudaMalloc(&d_output, size * sizeof(fixed_point_t));
    if (err != cudaSuccess) {
        cudaFree(d_input);
        printf("CUDA malloc output error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copia input a la GPU
    err = cudaMemcpy(d_input, input, size * sizeof(fixed_point_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        printf("CUDA memcpy input error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Executa el kernel
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    process_kernel<<<numBlocks, blockSize>>>(d_input, d_output, size);
    
    // Comprova errors del kernel
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Copia resultat de tornada
    err = cudaMemcpy(output, d_output, size * sizeof(fixed_point_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_output);
        printf("CUDA memcpy output error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Neteja
    cudaFree(d_input);
    cudaFree(d_output);
    
    return 0;
}

}