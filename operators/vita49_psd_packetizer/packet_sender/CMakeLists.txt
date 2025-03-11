# SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build)
    set(TARGET_DIR "debug")
else()
    set(CARGO_CMD cargo build --release)
    set(TARGET_DIR "release")
endif()

set(PACKET_SENDER_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/libpacket_sender.a")
set(PACKET_SENDER_CXX "${CMAKE_CURRENT_BINARY_DIR}/packet_sender.cpp")
set(PACKET_SENDER_H "${CMAKE_CURRENT_BINARY_DIR}/packet_sender.h")
add_library(packet_sender STATIC ${PACKET_SENDER_CXX} ${PACKET_SENDER_H})
add_custom_command(
    OUTPUT ${PACKET_SENDER_CXX} ${PACKET_SENDER_H}
    COMMAND CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR} RUSTFLAGS="${RUST_FLAGS}" ${CARGO_CMD}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/packet_sender/src/packet_sender.rs.cc ${PACKET_SENDER_CXX}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/packet_sender/src/packet_sender.rs.h ${PACKET_SENDER_H}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/packet_sender.rs
    BYPRODUCTS "${PACKET_SENDER_LIB}"
)
set_target_properties(packet_sender PROPERTIES
    IMPORTED_LOCATION "${PACKET_SENDER_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(packet_sender pthread dl ${PACKET_SENDER_LIB})
target_include_directories(packet_sender INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
