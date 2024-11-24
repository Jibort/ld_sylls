// Capçalera per a la interacció Go <-> C.
// CreatedAt: 2024/11/24 dg. JIQ

#ifndef CUDA_BRIDGE_H
#define CUDA_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

int cuda_init(void);
void cuda_cleanup(void);
int cuda_process_fixed_point(int32_t* input, int32_t* output, int size);

#ifdef __cplusplus
}
#endif

#endif // CUDA_BRIDGE_H