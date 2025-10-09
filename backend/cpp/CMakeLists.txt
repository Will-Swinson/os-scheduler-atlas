cmake_minimum_required(VERSION 3.20)
project(scheduler_core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
# find_package(pybind11 CONFIG REQUIRED)
enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50.zip
)

FetchContent_MakeAvailable(googletest)

add_subdirectory(scheduler)
add_subdirectory(tests)



