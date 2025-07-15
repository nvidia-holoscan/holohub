# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_holohub_operator(dds_video_publisher)
add_holohub_operator(dds_video_subscriber)

# Build the DDS generated type library
if(OP_dds_video_publisher OR OP_dds_video_subscriber)
  include(RTIConnextDDS)
  add_rti_type_library(dds_video_frame ${CMAKE_CURRENT_SOURCE_DIR}/VideoFrame.idl UNBOUNDED)
endif()

