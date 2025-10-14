cmake_minimum_required(VERSION 3.20)
project(scheduler_core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50.zip
  )
  
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/v2.11.1.zip
  )
    
FetchContent_MakeAvailable(pybind11)
FetchContent_MakeAvailable(googletest)
    
add_subdirectory(scheduler)
add_subdirectory(tests)



