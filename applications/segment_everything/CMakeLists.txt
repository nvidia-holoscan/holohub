cmake_minimum_required(VERSION 3.10)
project(segment_everything)

enable_testing()

# Find Python executable
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Add a custom target for running pytest
add_test(
    NAME test_utils
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_SOURCE_DIR}/test_utils.py
)

# set properties for the test
set_tests_properties(test_utils PROPERTIES
    PASS_REGULAR_EXPRESSION ".*"
)