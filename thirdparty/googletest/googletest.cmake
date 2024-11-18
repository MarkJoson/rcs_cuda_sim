include(ExternalProject)

ExternalProject_Add(
    ext_googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    PREFIX ${CMAKE_BINARY_DIR}/googletest
    
    # 传递 CMake 构建选项
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_CXX_STANDARD=14
        -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
)

add_library(googletest_dep INTERFACE)

target_include_directories(googletest_dep INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/googletest/include>
    $<INSTALL_INTERFACE:googletest/include>
)


target_link_directories(googletest_dep
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/googletest/lib>
    $<INSTALL_INTERFACE:googletest/lib>)

target_link_libraries(googletest_dep
    INTERFACE
        gtest
        gtest_main
        gmock
        gmock_main
)


add_dependencies(googletest_dep ext_googletest)