# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)
find_package(Threads REQUIRED)

set(HOLOHUB_FOXGLOVE_SDK_VERSION "0.23.1" CACHE STRING "Foxglove SDK version")
set(HOLOHUB_FOXGLOVE_SDK_TAG "sdk/v${HOLOHUB_FOXGLOVE_SDK_VERSION}")

if(NOT DEFINED HOLOHUB_FOXGLOVE_SDK_URL)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(HOLOHUB_FOXGLOVE_SDK_URL
      "https://github.com/foxglove/foxglove-sdk/releases/download/${HOLOHUB_FOXGLOVE_SDK_TAG}/foxglove-v${HOLOHUB_FOXGLOVE_SDK_VERSION}-cpp-aarch64-unknown-linux-gnu.zip")
    set(HOLOHUB_FOXGLOVE_SDK_SHA256
      "ec20bcf1aa769fc9b76d84ff3e15c0df3f0a2b3ee56eb7ee45e3f1e4306620bc")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
    set(HOLOHUB_FOXGLOVE_SDK_URL
      "https://github.com/foxglove/foxglove-sdk/releases/download/${HOLOHUB_FOXGLOVE_SDK_TAG}/foxglove-v${HOLOHUB_FOXGLOVE_SDK_VERSION}-cpp-x86_64-unknown-linux-gnu.zip")
    set(HOLOHUB_FOXGLOVE_SDK_SHA256
      "b949295e80eb1a9bb356e657b9c2579c886717fac290c4e48a5e9846063bf2e8")
  else()
    message(FATAL_ERROR
      "Unsupported Foxglove SDK architecture: ${CMAKE_SYSTEM_PROCESSOR}. "
      "Set HOLOHUB_FOXGLOVE_SDK_URL and HOLOHUB_FOXGLOVE_SDK_SHA256 explicitly.")
  endif()
endif()

if(DEFINED HOLOHUB_FOXGLOVE_SDK_URL AND
   (NOT DEFINED HOLOHUB_FOXGLOVE_SDK_SHA256 OR HOLOHUB_FOXGLOVE_SDK_SHA256 STREQUAL ""))
  message(FATAL_ERROR
    "HOLOHUB_FOXGLOVE_SDK_URL is set for ${CMAKE_SYSTEM_PROCESSOR}, but "
    "HOLOHUB_FOXGLOVE_SDK_SHA256 is not set. Set both HOLOHUB_FOXGLOVE_SDK_URL "
    "and HOLOHUB_FOXGLOVE_SDK_SHA256 before FetchContent_Declare.")
endif()

FetchContent_Declare(
  foxglove_sdk
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL "${HOLOHUB_FOXGLOVE_SDK_URL}"
  URL_HASH SHA256=${HOLOHUB_FOXGLOVE_SDK_SHA256}
)
FetchContent_MakeAvailable(foxglove_sdk)

set(_foxglove_root "${foxglove_sdk_SOURCE_DIR}")
if(EXISTS "${_foxglove_root}/foxglove")
  set(_foxglove_root "${_foxglove_root}/foxglove")
endif()

file(GLOB _foxglove_sources CONFIGURE_DEPENDS "${_foxglove_root}/src/*.cpp")

add_library(foxglove_sdk_cpp STATIC ${_foxglove_sources})
add_library(foxglove::sdk ALIAS foxglove_sdk_cpp)

target_include_directories(foxglove_sdk_cpp
  PUBLIC
    "${_foxglove_root}/include"
    "${_foxglove_root}/include/foxglove")

target_link_libraries(foxglove_sdk_cpp
  PUBLIC
    "${_foxglove_root}/lib/libfoxglove.a"
    Threads::Threads
    ${CMAKE_DL_LIBS})

set_target_properties(foxglove_sdk_cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
