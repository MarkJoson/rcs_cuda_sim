#include "core/EnvGroupManager.cuh"

using namespace cuda_simulator::core::env_group_impl;

namespace cuda_simulator {
namespace core {
namespace env_group_impl {

__constant__ ActiveGroupMapperStorage d_agm_storage_;
ActiveGroupMapperStorage EGActiveGroupMapper::h_agm_storage_;

extern __constant__ int constant_mem_pool[CONST_MEM_WORD_SIZE];
extern __constant__ ConstMemPoolConfig d_cmp_config_;
ConstMemPoolConfig EGConstantMemoryPool::h_cmp_config_;


} // env_group_impl
} // core
} // cuda_simulator
