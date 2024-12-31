#ifndef CUDASIM_ENVIRON_GROUP_CONFIG_ITEM_HH
#define CUDASIM_ENVIRON_GROUP_CONFIG_ITEM_HH

#include <cstddef>
#include <cstdint>

namespace cuda_simulator
{
namespace core
{

class EnvironGroupConfigItemBase
{
public:



protected:
    void *d_arr_;
    int item_memsize_;
};


template<typename T>
class EnvironGroupConfigItem : public EnvironGroupConfigItemBase
{
public:
    __device__ T* getConfig(int env_group_idx) {
        return static_cast<const T*>(d_arr_) + env_group_idx;
    }
private:
};

} // namespace core
} // namespace cuda_simulator



#endif //CUDASIM_ENVIRON_GROUP_ENTRY_HH