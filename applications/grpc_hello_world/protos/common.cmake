# Copyright 2018 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cmake build file for C++ route_guide example.
# Assumes protobuf and gRPC have been installed using cmake.
# See cmake_externalproject/CMakeLists.txt for all-in-one cmake build
# that automatically builds all the dependencies before building route_guide.

cmake_minimum_required(VERSION 3.8)
find_package(Threads REQUIRED)


function(grpc_generate_cpp SRCS HDRS INCLUDE_DIRS)
    # Expect:
    # - PROTOC_EXECUTABLE: path to protoc
    # - GRPC_CPP_EXECUTABLE: path to grpc_cpp_plugin
    if(NOT ARGN)
        message(SEND_ERROR "Error: grpc_generate_cpp() called without any .proto files")
        return()
    endif()

    foreach(PROTO_FILE ${ARGN})
        message(STATUS "Build proto file ${PROTO_FILE}")
        # Get the full path to the proto file
        get_filename_component(_abs_proto_file "${PROTO_FILE}" ABSOLUTE)
        # Get the name of the proto file without extension
        get_filename_component(_proto_name_we ${PROTO_FILE} NAME_WE)
        # Get the parent directory of the proto file
        get_filename_component(_proto_parent_dir ${_abs_proto_file} DIRECTORY)
        # Get the parent directory of the parent directory
        get_filename_component(_parent_dir ${_proto_parent_dir} DIRECTORY)
        # Append 'generated' to the parent directory
        set(_generated_dir "${_parent_dir}/generated")
        file(MAKE_DIRECTORY ${_generated_dir})


        set(_protobuf_include_path -I ${_proto_parent_dir})

        set(_proto_srcs "${_generated_dir}/${_proto_name_we}.pb.cc")
        set(_proto_hdrs "${_generated_dir}/${_proto_name_we}.pb.h")
        set(_grpc_srcs "${_generated_dir}/${_proto_name_we}.grpc.pb.cc")
        set(_grpc_hdrs "${_generated_dir}/${_proto_name_we}.grpc.pb.h")

        file(REMOVE "${_proto_srcs}" "${_proto_hdrs}" "${_grpc_srcs}" "${_grpc_hdrs}")
        add_custom_command(
            OUTPUT "${_proto_srcs}" "${_proto_hdrs}" "${_grpc_srcs}" "${_grpc_hdrs}"
            COMMAND ${PROTOC_EXECUTABLE}
            ARGS --grpc_out=${_generated_dir}
            --cpp_out=${_generated_dir}
            --plugin=protoc-gen-grpc=${GRPC_CPP_EXECUTABLE}
            ${_protobuf_include_path} ${_abs_proto_file}
            DEPENDS ${_abs_proto_file}
            COMMENT "Running gRPC C++ protocol buffer compiler on ${PROTO_FILE}"
            VERBATIM
        )

        list(APPEND ${SRCS} "${_proto_srcs}")
        list(APPEND ${HDRS} "${_proto_hdrs}")
        list(APPEND ${SRCS} "${_grpc_srcs}")
        list(APPEND ${HDRS} "${_grpc_hdrs}")
        list(APPEND ${INCLUDE_DIRS} "${_generated_dir}")
    endforeach()

    set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
    set(${SRCS} "${${SRCS}}" PARENT_SCOPE)
    set(${HDRS} "${${HDRS}}" PARENT_SCOPE)
    set(${INCLUDE_DIRS} "${${INCLUDE_DIRS}}" PARENT_SCOPE)
endfunction()

# This branch assumes that gRPC and all its dependencies are already installed
# on this system, so they can be located by find_package().

# Find Protobuf installation
# Looks for protobuf-config.cmake file installed by Protobuf's cmake installation.
option(protobuf_MODULE_COMPATIBLE TRUE)
include(FindProtobuf)
find_package(Protobuf CONFIG REQUIRED PATHS "/opt/nvidia/holoscan/3rdparty/grpc/1.54.2/lib/cmake/protobuf/")
message(STATUS "Using protobuf ${Protobuf_VERSION}")

set(PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(GRPCPP_REFLECTION gRPC::grpc++_reflection)
if(CMAKE_CROSSCOMPILING)
  find_program(PROTOC_EXECUTABLE protoc)
  message(STATUS "A Using protoc ${PROTOC_EXECUTABLE}")
else()
  set(PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
  # get_target_property(PROTOC_EXECUTABLE protobuf::protoc LOCATION)
  message(STATUS "B Using protoc ${PROTOC_EXECUTABLE}")
  endif()

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(absl CONFIG REQUIRED PATHS "/opt/nvidia/holoscan/3rdparty/grpc/1.54.2/lib/cmake/absl/")
find_package(gRPC CONFIG REQUIRED PATHS "/opt/nvidia/holoscan/3rdparty/grpc/1.54.2/lib/cmake/grpc/")
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(GRPC_GRPCPP gRPC::grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(GRPC_CPP_EXECUTABLE grpc_cpp_plugin)
else()
  set(GRPC_CPP_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()

# Expose variables with PARENT_SCOPE so that
# root project can use it for including headers and using executables
set(PROTOC_EXECUTABLE ${PROTOC_EXECUTABLE} PARENT_SCOPE)
set(GRPC_CPP_EXECUTABLE ${GRPC_CPP_EXECUTABLE} PARENT_SCOPE)
set(PROTOBUF_LIBPROTOBUF ${PROTOBUF_LIBPROTOBUF} PARENT_SCOPE)
set(GRPCPP_REFLECTION ${GRPCPP_REFLECTION} PARENT_SCOPE)
set(GRPC_GRPCPP ${GRPC_GRPCPP} PARENT_SCOPE)

message(STATUS "=========================================")
get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
  message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
message(STATUS "=========================================")
