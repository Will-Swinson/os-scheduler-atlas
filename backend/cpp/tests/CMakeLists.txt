add_executable(
  scheduler_unit_tests
  ./scheduler/unit_tests.cpp
)

add_executable(
  scheduler_integration_tests 
./scheduler/integration_tests.cpp
)

add_executable(
  scheduler_system_tests 
./scheduler/system_tests.cpp
)

target_link_libraries(
  scheduler_unit_tests
  scheduler_core
  gtest_main
)

target_link_libraries(
  scheduler_integration_tests
  scheduler_core
  gtest_main
)

target_link_libraries(
  scheduler_system_tests
  scheduler_core
  gtest_main
)


target_compile_features(scheduler_unit_tests PUBLIC cxx_std_20)
target_compile_features(scheduler_integration_tests PUBLIC cxx_std_20)
target_compile_features(scheduler_system_tests PUBLIC cxx_std_20)

include(GoogleTest)
gtest_discover_tests(scheduler_unit_tests)
gtest_discover_tests(scheduler_integration_tests)
gtest_discover_tests(scheduler_system_tests)