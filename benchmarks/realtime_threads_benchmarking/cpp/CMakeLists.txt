# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(realtime_thread_benchmarking CXX)

find_package(holoscan 3.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Create the benchmark executable
add_executable(realtime_thread_benchmark
  realtime_thread_benchmark.cpp
)

# Link against Holoscan
target_link_libraries(realtime_thread_benchmark
  holoscan::core
)

# Set C++ standard
target_compile_features(realtime_thread_benchmark PRIVATE cxx_std_17)

# Install the executable
install(TARGETS realtime_thread_benchmark
  RUNTIME DESTINATION bin
)
