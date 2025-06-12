#
# Place the license header here
#

cmake_minimum_required(VERSION 3.20)

project(slang_shader)

find_package(holoscan 3.3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Fetch Slang
include(FetchContent)
FetchContent_Declare(
  Slang
  URL https://github.com/shader-slang/slang/releases/download/v2025.10.2/slang-2025.10.2-linux-${CMAKE_SYSTEM_PROCESSOR}.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(Slang)

include(${slang_SOURCE_DIR}/cmake/slangConfig.cmake)

# Fetch nlohmann_json
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# Add slang_shader operator
add_library(slang_shader SHARED
  command.cpp
  slang_shader.cpp
  slang_utils.cpp
  slang_shader_op.cpp
)

add_library(holoscan::ops::slang_shader ALIAS slang_shader)

# generate header from Holoscan slang shader
include(GenHeaderFromFile)
gen_header_from_file(TARGET slang_shader FILE_PATH "holoscan.slang" TYPE text)

target_include_directories(slang_shader
  PRIVATE
    ${slang_SOURCE_DIR}/include
    # for the generated header
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

target_link_libraries(slang_shader
  PUBLIC
    holoscan::core
  PRIVATE
    CUDA::cudart
    slang::slang
    nlohmann_json::nlohmann_json
)

target_include_directories(slang_shader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(slang_shader PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR})
target_compile_definitions(slang_shader PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})

install(TARGETS slang_shader
  COMPONENT holoscan-ops
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

