// Tipus FixedPoint dins l'entorn CUDA.
// CreatedAt: 2024/11/24 dg. JIQ

#include "fixed_point.cuh"
#include <cuda_runtime.h>

// Definim les funcions amb __device__ i __host__
__device__ __host__ fixed_point_t fp_from_float(float f) {
    if (f >= 1.0f) return INT32_MAX;
    if (f <= -1.0f) return INT32_MIN;
    return (fixed_point_t)(f * (float)INT32_MAX);
}

__device__ __host__ float fp_to_float(fixed_point_t fp) {
    return (float)fp / (float)INT32_MAX;
}

__device__ __host__ fixed_point_t fp_add(fixed_point_t a, fixed_point_t b) {
    int64_t sum = (int64_t)a + (int64_t)b;
    if (sum > INT32_MAX) return INT32_MAX;
    if (sum < INT32_MIN) return INT32_MIN;
    return (fixed_point_t)sum;
}

__device__ __host__ fixed_point_t fp_sub(fixed_point_t a, fixed_point_t b) {
    int64_t diff = (int64_t)a - (int64_t)b;
    if (diff > INT32_MAX) return INT32_MAX;
    if (diff < INT32_MIN) return INT32_MIN;
    return (fixed_point_t)diff;
}

__device__ __host__ fixed_point_t fp_mul(fixed_point_t a, fixed_point_t b) {
    int64_t prod = ((int64_t)a * (int64_t)b) / INT32_MAX;
    if (prod > INT32_MAX) return INT32_MAX;
    if (prod < INT32_MIN) return INT32_MIN;
    return (fixed_point_t)prod;
}

__device__ __host__ bool fp_less_than(fixed_point_t a, fixed_point_t b) {
    return a < b;
}

__device__ __host__ bool fp_greater_than(fixed_point_t a, fixed_point_t b) {
    return a > b;
}

// També podríem afegir:
__device__ __host__ bool fp_less_or_equal(fixed_point_t a, fixed_point_t b) {
    return a <= b;
}

__device__ __host__ bool fp_greater_or_equal(fixed_point_t a, fixed_point_t b) {
    return a >= b;
}

__device__ __host__ bool fp_equal(fixed_point_t a, fixed_point_t b) {
    return a == b;
}

// Kernel d'exemple
__global__ void process_kernel(fixed_point_t* input, fixed_point_t* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = fp_to_float(input[idx]);
        output[idx] = fp_from_float(val * 0.5f);
    }
}