include(ExternalProject)

ExternalProject_Add(
    ext_eigen
    PREFIX eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
)



add_library(cuda_sim_eigen INTERFACE)

target_include_directories(cuda_sim_eigen SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/eigen/include/eigen3/>
    $<INSTALL_INTERFACE:include/eigen3>
)
add_dependencies(cuda_sim_eigen ext_eigen)
