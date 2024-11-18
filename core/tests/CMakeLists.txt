file (
    GLOB_RECURSE
        tensor_storage_test_src
    CONFIGURE_DEPENDS
        storage/*.cpp
)

# 创建测试可执行文件
add_executable(test_storage
    ${tensor_storage_test_src}
)

# 设置目标属性
set_target_properties(test_storage PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接依赖
target_link_libraries(test_storage
    PRIVATE
        # tensor_storage
        ${TORCH_LIBRARIES}
        tensor_storage
        googletest_dep
)

# 添加测试
gtest_discover_tests(test_storage
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES
        LABELS "storage"
)
