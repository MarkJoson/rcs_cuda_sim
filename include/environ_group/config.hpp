#pragma once
#include <core/types.hpp>

namespace environment {

class EnvironConfig {
public:
    explicit EnvironConfig(int numEnvs) : numEnvs_(numEnvs) {}

    int getNumEnvs() const { return numEnvs_; }

private:
    int numEnvs_;
};

} // namespace environment