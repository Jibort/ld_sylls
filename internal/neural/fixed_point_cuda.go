//go:build cuda

// Tipus FixedPoint per a treballar amb CUDA.
// CreatedAt: 2024/11/24 dg. JIQ

package neural

// // #cgo LDFLAGS: -L${SRCDIR}/../../lib -lcudabridge
// // #include "cuda_bridge.h"
// import "C"
import (
	"fmt"
	"unsafe"
)

type CUDAFixedPoint struct {
	value int32
}

func InitCUDA() error {
	if res := C.cuda_init(); res != 0 {
		return fmt.Errorf("failed to initialize CUDA")
	}
	return nil
}

func CleanupCUDA() {
	C.cuda_cleanup()
}

func ProcessFixedPoints(input []CUDAFixedPoint) ([]CUDAFixedPoint, error) {
	if len(input) == 0 {
		return nil, nil
	}

	output := make([]CUDAFixedPoint, len(input))

	// Converteix a array de int32
	inputRaw := make([]int32, len(input))
	for i, v := range input {
		inputRaw[i] = v.value
	}

	// Crida a CUDA
	res := C.cuda_process_fixed_point(
		(*C.int32_t)(unsafe.Pointer(&inputRaw[0])),
		(*C.int32_t)(unsafe.Pointer(&output[0].value)),
		C.int(len(input)))

	if res != 0 {
		return nil, fmt.Errorf("CUDA processing failed")
	}

	return output, nil
}
