find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)


file ( GLOB_RECURSE MAPGEN_SRC "mapgen/*.cpp"  )

add_executable(map_generator ${MAPGEN_SRC} mapgen_postproc_test.cpp )
set_target_properties(map_generator PROPERTIES POSITION_INDEPENDENT_CODE ON )
target_include_directories(map_generator PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(map_generator PUBLIC
    sfml-graphics
    sfml-window
    sfml-system
    ${OpenCV_LIBS}
    glog::glog
)


add_executable(rasterizer ${MAPGEN_SRC} rasterizer.cu)
set_target_properties(rasterizer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_options(rasterizer PUBLIC ${rcsim_link_options})
target_include_directories(rasterizer PUBLIC ${OpenCV_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(rasterizer PUBLIC
    sfml-graphics
    sfml-window
    sfml-system
    ${OpenCV_LIBS}
    glog::glog

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
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudaSim/component
)

# 安装生成的导出头文件
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/roboSim_core_export.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudaSim/component
)

# 安装导出目标
install(EXPORT RoboSimCoreTargets
    FILE roboSimCoreTargets.cmake
    NAMESPACE cudaSim::component::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudaSim/component
)