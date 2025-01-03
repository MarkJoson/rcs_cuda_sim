cmake_minimum_required(VERSION 3.27)

# By default we build for the native cuda architecture. Customize by passing
# '-DCMAKE_CUDA_ARCHITECTURES=89;75;72' to cmake.
if (DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES_SET_EXTERNALLY TRUE)
else()
  set(CMAKE_CUDA_ARCHITECTURES_SET_EXTERNALLY FALSE)
endif()

# 启用CUDA编译
project(cudaSim VERSION 0.0.1 LANGUAGES CXX CUDA)

########################
#        编译选项       #
########################
# Build options
set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_STATIC_LIBRARY_PREFIX "")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BUILD_TESTING "Build tests" OFF)

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/pyenv/)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

################
#    依赖库     #
################

# Include package deps
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(CUDAToolkit REQUIRED)

include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# Setup options for nvcc and gcc
include(cmake/setup_compilers.cmake)

# torch
include(thirdparty/torch/torch.cmake)
find_package(Torch REQUIRED)

# google test
include(thirdparty/googletest/googletest.cmake)
include(GoogleTest)

# boost
find_package(Boost REQUIRED COMPONENTS graph)

############
# Targets #
############

add_executable(test_boost_graph
    src/test_boost_graph.cpp
)

set_target_properties(test_boost_graph PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(test_boost_graph
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>    # 构建时使用源代码中的 include 目录
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>          # 安装后使用安装的 include 目录
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

target_link_libraries(test_boost_graph
    PUBLIC
    ${Boost_LIBRARIES}
    ${TORCH_LIBRARIES}
)


add_executable(test_core
    src/test_core.cu
    src/core/storage/GTensorTorchWrapper.cpp
    src/core/storage/TensorRegistryManager.cpp
    src/core/ReducerComponent.cc
)

# 设置目标属性
set_target_properties(test_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(test_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>    # 构建时使用源代码中的 include 目录
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>          # 安装后使用安装的 include 目录
    PRIVATE
        ${TORCH_INCLUDE_DIRS}
)

# 链接依赖
target_link_libraries(test_core
    PUBLIC
        ${TORCH_LIBRARIES}
)


############
#  Tests   #
############

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

############
#  Tests   #
############


#########
# TESTS #
#########


##########
# EXPORT #
##########
# include(GNUInstallDirs)

# include(CMakePackageConfigHelpers)

# install(TARGETS ${PROJECT_EXPORT_TARGET}
#     EXPORT ${PROJECT_EXPORT_NAME}Targets
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#     INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

# # 安装公共头文件
# install(
#   DIRECTORY ${CMAKE_SOURCE_DIR}/include
#   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
# )

# # 安装生成的导出头文件
# install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cudaSim_core_export.h
#     DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudaSim
# )


# install(
#   EXPORT ${PROJECT_EXPORT_NAME}Targets
#   NAMESPACE ${PROJECT_EXPORT_NAME}
#   FILE ${PROJECT_EXPORT_NAME}Targets.cmake
#   DESTINATION ${PROJECT_INSTALL_CMAKEDIR}
# )


# # generate the config file that is includes the exports
# configure_package_config_file(cmake/Config.cmake.in
#     "${CMAKE_CURRENT_BINARY_DIR}/cudaSimConfig.cmake"
#     INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
#     NO_SET_AND_CHECK_MACRO
#     NO_CHECK_REQUIRED_COMPONENTS_MACRO
#     PATH_VARS
#         CMAKE_INSTALL_INCLUDEDIR
#         CMAKE_INSTALL_LIBDIR
# )

# # generate the version file for the config file
# write_basic_package_version_file(
#     ${PROJECT_EXPORT_NAME}ConfigVersion.cmake
#     VERSION "${cudaSim_VERSION_MAJOR}.${cudaSim_VERSION_MINOR}"
#     COMPATIBILITY AnyNewerVersion
# )

# # install the configuration file
# install(FILES
#     ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_EXPORT_NAME}Config.cmake
#     ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_EXPORT_NAME}ConfigVersion.cmake
#     DESTINATION ${PROJECT_INSTALL_CMAKEDIR})