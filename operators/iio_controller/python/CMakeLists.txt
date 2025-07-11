# Copyright 2025 Analog Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

cmake_minimum_required(VERSION 3.20)

find_package(
  holoscan
  2.6
  REQUIRED
  CONFIG
  PATHS
  "/opt/nvidia/holoscan"
  "/workspace/holoscan-sdk/install")

find_library(
  IIO_LIBRARIES
  NAMES iio libiio
  HINTS /usr/lib
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/lib
        /opt/lib64
        REQUIRED)

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
  CPP_CMAKE_TARGET
  iio_controller
  CLASS_NAME
  "IIOAttributeRead, IIOAttributeWrite, IIOBufferInfo, IIOBufferRead, IIOBufferWrite, IIOConfigurator, IIOChannelInfo, IIODataFormat"
  SOURCES
  iio_controller_py.cpp)
