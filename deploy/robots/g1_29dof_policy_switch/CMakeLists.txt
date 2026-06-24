cmake_minimum_required(VERSION 3.12)
project(g1_29dof_policy_switch_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx
  /usr/local/include/iceoryx/v2.0.2
  include
  ${PROJECT_SOURCE_DIR}/../g1_29dof/include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)

add_executable(
  g1_policy_switch_ctrl
  main.cpp
  src/State_PolicySwitch.cpp
)

