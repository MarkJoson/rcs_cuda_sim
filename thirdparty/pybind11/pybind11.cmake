include(ExternalProject)

ExternalProject_Add(
    ext_pybind11
    PREFIX pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.zip
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)

find_package(Python3 3.10 COMPONENTS Development REQUIRED)

add_library(cuda_sim_pybind11 INTERFACE)

target_include_directories(cuda_sim_pybind11 SYSTEM INTERFACE
    ${Python3_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/pybind11/include>
    $<INSTALL_INTERFACE:include>
)
add_dependencies(cuda_sim_pybind11
    ext_pybind11
    ${Python3_LIBRARIES}
)
