include(ExternalProject)

ExternalProject_Add(
    ext_nanoflann
    PREFIX nanoflann
    URL https://github.com/jlblancoc/nanoflann/archive/refs/tags/v1.6.3.zip
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_CXX_FLAGS="-I${CMAKE_CURRENT_BINARY_DIR}/eigen/include/eigen3"
)
add_dependencies(ext_nanoflann ext_eigen)

add_library(cuda_sim_nanoflann INTERFACE)

target_include_directories(cuda_sim_nanoflann SYSTEM INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/nanoflann/include>
    $<INSTALL_INTERFACE:include>
)
add_dependencies(cuda_sim_nanoflann ext_nanoflann)
