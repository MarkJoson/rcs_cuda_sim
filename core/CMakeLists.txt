

file (
    GLOB_RECURSE
        tensor_storage_src
    CONFIGURE_DEPENDS
        src/storage/*.cpp
)


add_library(tensor_storage
    ${tensor_storage_src}
)

# 设置目标属性
set_target_properties(tensor_storage PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(tensor_storage
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # 构建时使用源代码中的 include 目录
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/roboSim/core>  # 安装后使用安装的 include 目录
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
)

# 链接依赖
target_link_libraries(tensor_storage
    PUBLIC
        ${TORCH_LIBRARIES}
)

add_executable(test_app
    src/main.cpp
)

target_link_libraries(test_app
    tensor_storage
)


if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()


# 创建组件导出
install(TARGETS tensor_storage
    EXPORT RoboSimCoreTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 安装公共头文件
install(FILES ${STORAGE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/roboSim/core
)

# 安装生成的导出头文件
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/roboSim_core_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/roboSim/core
)

# 安装导出目标
install(EXPORT RoboSimCoreTargets
    FILE roboSimCoreTargets.cmake
    NAMESPACE roboSim::core::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/roboSim/core
)