#ifndef __GTENSOR_BASE_H__
#define __GTENSOR_BASE_H__

#include "ITensor.h"

namespace RSG_SIM {

namespace internal {
    class TorchTensorImpl;
}

class GTensorBase : public ITensor {
public:
    explicit GTensorBase(const TensorMeta& meta);
    ~GTensorBase() override;
    
    // 元信息接口实现
    const TensorMeta& meta() const override { return meta_; }
    bool isTypeMatch(const std::type_index& type) const override {
        return meta_.type_info == type;
    }
    
    // 基础信息实现
    const std::string& name() const override { return meta_.name; }
    const std::vector<int64_t>& shape() const override { return meta_.shape; }
    size_t elemCount() const override;
    size_t elemSize() const override { return meta_.type_size; }
    size_t dim() const override { return meta_.shape.size(); }
    
    // 数据访问实现
    void* ptr() override;
    const void* ptr() const override;
    
    // 设备操作实现
    bool isOnCPU() const override;
    bool isOnGPU() const override;
    void toCPU() override;
    void toGPU() override;
    
    // 数据操作实现
    void zero() override;
    void fill(const void* value) override;
    
    // 形状操作实现
    void reshape(const std::vector<int64_t>& shape) override;
    void resize(const std::vector<int64_t>& shape) override;
    void squeeze(int dim = -1) override;
    void unsqueeze(int dim) override;
    void flatten(int start_dim = 0, int end_dim = -1) override;
    void transpose(int dim0, int dim1) override;
    void permute(const std::vector<int64_t>& dims) override;
    
    // 切片和索引实现
    std::unique_ptr<ITensor> slice(
        int dim, 
        int64_t start, 
        int64_t end, 
        int64_t step = 1) const override;
    void select(int dim, int64_t index) override;
    void index(const std::vector<ITensor*>& indices) override;
    
    // 数学操作实现
    void abs() override;
    void clip(const void* min_val, const void* max_val) override;
    void sqrt() override;
    void pow(double exponent) override;
    void exp() override;
    void log() override;
    
    // 统计操作实现
    std::unique_ptr<ITensor> sum(int dim = -1, bool keepdim = false) const override;
    std::unique_ptr<ITensor> mean(int dim = -1, bool keepdim = false) const override;
    std::unique_ptr<ITensor> std(int dim = -1, bool keepdim = false) const override;
    std::unique_ptr<ITensor> var(int dim = -1, bool keepdim = false) const override;
    std::unique_ptr<ITensor> max(int dim = -1, bool keepdim = false) const override;
    std::unique_ptr<ITensor> min(int dim = -1, bool keepdim = false) const override;
    std::unique_ptr<ITensor> argmax(int dim = -1, bool keepdim = false) const override;
    std::unique_ptr<ITensor> argmin(int dim = -1, bool keepdim = false) const override;
    
    // 比较操作实现
    void eq(const ITensor& other) override;
    void ne(const ITensor& other) override;
    void gt(const ITensor& other) override;
    void lt(const ITensor& other) override;
    void ge(const ITensor& other) override;
    void le(const ITensor& other) override;

protected:
    TensorMeta meta_;
    std::unique_ptr<internal::TorchTensorImpl> impl_;
    
    // 辅助函数
    GTensorBase* createTensorFromImpl(const std::string& name, internal::TorchTensorImpl* impl) const;
    const internal::TorchTensorImpl* getImpl(const ITensor& tensor) const;
};

} // namespace RSG_SIM

#endif