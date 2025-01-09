#include "core/EnvGroupManager.cuh"

using namespace cuda_simulator::core::env_group_impl;

namespace cuda_simulator {
namespace core {
namespace env_group_impl {

__constant__ __device__ int constant_mem_pool[CONST_MEM_WORD_SIZE];
__constant__ __device__ uint32_t d_active_group_count;
__constant__ __device__ int d_const_mem_alloc_words;

} // env_group_impl
} // core
} // cuda_simulator
