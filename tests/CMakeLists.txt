file (
    GLOB_RECURSE
        test_storage_src
    CONFIGURE_DEPENDS
        storage/*.cpp
)

# 创建测试可执行文件
# add_executable(test_tree test_tree.cpp)
# set_target_properties(test_tree PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(test_tree PRIVATE ${TORCH_LIBRARIES} corelib googletest_dep)
# gtest_discover_tests(test_tree WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} PROPERTIES LABELS "tree")

# add_executable(test_storage ${test_storage_src})
# set_target_properties(test_storage PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(test_storage PRIVATE ${TORCH_LIBRARIES} corelib googletest_dep)
# gtest_discover_tests(test_storage WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} PROPERTIES LABELS "storage")
