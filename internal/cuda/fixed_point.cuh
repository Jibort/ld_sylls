// Tipus FixedPoint dins l'entorn CUDA.
// CreatedAt: 2024/11/24 dg. JIQ

#ifndef FIXED_POINT_CUH
#define FIXED_POINT_CUH

#include <cuda_runtime.h>
#include <stdint.h>

typedef int32_t fixed_point_t;

// Declarem les funcions amb __device__ i __host__
__device__ __host__ fixed_point_t fp_from_float(float f);
__device__ __host__ float fp_to_float(fixed_point_t fp);
__device__ __host__ fixed_point_t fp_add(fixed_point_t a, fixed_point_t b);
__device__ __host__ fixed_point_t fp_sub(fixed_point_t a, fixed_point_t b);
__device__ __host__ fixed_point_t fp_mul(fixed_point_t a, fixed_point_t b);
__device__ __host__ bool fp_less_than(fixed_point_t a, fixed_point_t b);
__device__ __host__ bool fp_greater_than(fixed_point_t a, fixed_point_t b);
__device__ __host__ bool fp_less_or_equal(fixed_point_t a, fixed_point_t b);
__device__ __host__ bool fp_greater_or_equal(fixed_point_t a, fixed_point_t b);
__device__ __host__ bool fp_equal(fixed_point_t a, fixed_point_t b);

#endif // FIXED_POINT_CUH