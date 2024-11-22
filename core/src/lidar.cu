// #include "component.h"
// #include "storage.h"
#include <thrust/device_vector.h>

// #define M_PI    (3.1415926)

// // constexpr int LIDAR_LINES = VConfig::AgentRayNum;
// // constexpr float LIDAR_RESOLU_RAD = 2.0 * M_PI / VConfig::AgentRayNum;
// // constexpr float LIDAR_RESOLU_INV = 1.0 / LIDAR_RESOLU_RAD;
// // constexpr float LIDAR_MAX_RANGE = 6.0f;

// namespace RSG_SIM
// {

//     struct ComponentSpec {
//     public:
//         ComponentSpec(const std::string &type) : type_(type) {}
//         std::string type_;
//     };

//     struct CLidarSpec : public ComponentSpec {
//         CLidarSpec() : ComponentSpec("lidar") {}
//         int total_count_;       // 环境组中的激光雷达总数
//         int ray_count_;         // 单个激光雷达的射线数
//         float resolu_rad_;      // 分辨率
//         float max_range_;       // 最大距离
//     };

//     class Lidar : public Component
//     {
//         void onNewEnvironGroupCreated(EnvironGroup *env_group, ComponentSpec *spec) override
//         {

//         }

//         void initialize(ExecutionCtx &ctx, ) override
//         {
//         }

//         void reset(ExecutionCtx &ctx) override
//         {
//         }

//         void compute(ExecutionCtx &ctx) override
//         {
//             // 激光雷达<==>位置坐标
//             // 
//             auto pos = ctx.group->getTensor<float>("/spacial/pos")->to_thrust_ptr();
//             auto rot_rad = ctx.group->getTensor<float>("/spacial/rot_rad")->to_thrust_ptr();

//             auto enabled = ctx.group->getTensor<uint8_t>("/lidar/enabled")->to_thrust_ptr();
//             auto perc_data = ctx.group->getTensor<float>("/lidar/perc_data");
            
//             CLidarSpec spec;

//             int total_ray_count = perc_data->elemCount();
//             thrust::device_ptr<float> perc_data_arr = thrust::device_ptr<float>(perc_data->ptr());
//             thrust::fill(perc_data_arr, perc_data_arr+total_ray_count, spec.max_range_ );



//             // int lidar_count = ctx.xxxx->getConfig(lidar);
//             // int lidar_count
            
//             // thrust::fill(perc_data, perc_data+spec, LIDAR_MAX_RANGE);


            

//         }
//     };

// } // namespace RSG_SIM

using Array = thrust::device_vector<float>;

void compute_lidar(
    const Array &x,
    const Array &y,
    const Array &angle,
    const Array &enabled, 
    const Array &lines_)
{
    thrust::device_vector<float> lidar_data();

}