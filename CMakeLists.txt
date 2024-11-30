cmake_minimum_required(VERSION 3.27)

# By default we build for the native cuda architecture. Customize by passing
# '-DCMAKE_CUDA_ARCHITECTURES=89;75;72' to cmake.
if (DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES_SET_EXTERNALLY TRUE)
else()
  set(CMAKE_CUDA_ARCHITECTURES_SET_EXTERNALLY FALSE)
endif()

# 启用CUDA编译
project(rcsim VERSION 0.0.1 LANGUAGES CXX CUDA)

########################
#        编译选项       #
########################
# Build options
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_STATIC_LIBRARY_PREFIX "")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BUILD_TESTING "Build tests" ON)

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

############
# Sub Dirs #
############

add_subdirectory(core)
add_subdirectory(component)

#########
# TESTS #
#########


##########
# EXPORT #
##########
# 设置安装目录
include(GNUInstallDirs)

include(CMakePackageConfigHelpers)

# generate the config file that is includes the exports
configure_package_config_file(cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/roboCudaSimConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake"
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
    PATH_VARS
        CMAKE_INSTALL_INCLUDEDIR
        CMAKE_INSTALL_LIBDIR
)

# generate the version file for the config file
write_basic_package_version_file(
    "roboCudaSimConfigVersion.cmake"
    VERSION "${roboCudaSim_VERSION_MAJOR}.${roboCudaSim_VERSION_MINOR}"
    COMPATIBILITY AnyNewerVersion
)

# install the configuration file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/roboCudaSimConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/roboCudaSimConfigVersion.cmake
    DESTINATION share/roboCudaSim/cmake)

# install(
#     EXPORT roboCudaSimTargets
#     NAMESPACE roboCudaSim::
#     DESTINATION share/roboCudaSim/cmake
# )
