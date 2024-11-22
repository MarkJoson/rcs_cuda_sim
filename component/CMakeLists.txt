
file ( GLOB_RECURSE MAPGEN_SRC "map_generator/*.cpp"  )

add_executable(map_generator
    ${MAPGEN_SRC}
)

# 设置目标属性
set_target_properties(map_generator PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# target_include_directories(map_generator
#     PUBLIC
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # 构建时使用源代码中的 include 目录
#         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/roboSim/core>  # 安装后使用安装的 include 目录
#     PRIVATE
#         ${TORCH_INCLUDE_DIRS}
# )

# 链接依赖
target_link_libraries(map_generator
    PUBLIC
    sfml-graphics
    sfml-window
    sfml-system
)


if(BUILD_TESTING)
    enable_testing()
    # add_subdirectory(tests)
endif()


# 创建组件导出
install(TARGETS map_generator
    EXPORT RoboSimComponentTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 安装公共头文件
install(FILES ${STORAGE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/roboSim/component
)

# 安装生成的导出头文件
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/roboSim_core_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/roboSim/component
)

# 安装导出目标
install(EXPORT RoboSimCoreTargets
    FILE roboSimCoreTargets.cmake
    NAMESPACE roboSim::component::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/roboSim/component
)