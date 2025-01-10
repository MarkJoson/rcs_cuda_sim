#include "core/EnvGroupManager.cuh"

using namespace cuda_simulator::core::env_group_impl;

namespace cuda_simulator {
namespace core {
namespace env_group_impl {

__constant__ int constant_mem_pool[CONST_MEM_WORD_SIZE];
__constant__ uint32_t d_num_active_group;

} // env_group_impl
} // core
} // cuda_simulator
