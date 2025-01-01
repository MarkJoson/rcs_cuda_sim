#ifndef __CUDA_HELPER_H__
#define __CUDA_HELPER_H__

#include <cuda_runtime.h>
#include <cstdio>

#ifdef __DRIVER_TYPES_H__
static inline const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
#endif

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    throw std::exception();
    // exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

#define GET_CUDA_ID(id, maxID) 	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= maxID) return

#endif