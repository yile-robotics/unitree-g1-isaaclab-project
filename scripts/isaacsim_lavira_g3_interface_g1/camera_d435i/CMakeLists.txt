cmake_minimum_required(VERSION 3.10)
project(g1_d435i_probe LANGUAGES CXX)
find_package(realsense2 2.54.1 EXACT REQUIRED)
find_package(Threads REQUIRED)
add_executable(g1_d435i_probe capture.cpp)
target_compile_features(g1_d435i_probe PRIVATE cxx_std_17)
target_compile_options(g1_d435i_probe PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(g1_d435i_probe PRIVATE realsense2::realsense2)

# 实时服务与文件取图共用已安装的 SDK，不需要重新编译 SDK。
add_executable(g1_d435i_stream stream.cpp)
target_compile_features(g1_d435i_stream PRIVATE cxx_std_17)
target_compile_options(g1_d435i_stream PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(g1_d435i_stream PRIVATE realsense2::realsense2 Threads::Threads)
