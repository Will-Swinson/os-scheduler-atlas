cmake_minimum_required(VERSION 3.20)
project(scheduler_core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
# find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(scheduler)

add_executable(test_scheduler test_main.cpp)
target_link_libraries(test_scheduler PRIVATE scheduler_core)
target_compile_features(test_scheduler PUBLIC cxx_std_20)