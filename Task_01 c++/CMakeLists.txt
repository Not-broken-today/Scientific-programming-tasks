cmake_minimum_required(VERSION 3.16)

set(target_suffix vlad)
set(target_name "softmax_cpu_${target_suffix}")

add_executable(${target_name} main.cpp)

target_compile_features(${target_name} PRIVATE cxx_std_17)

find_package(OpenMP REQUIRED)
target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)