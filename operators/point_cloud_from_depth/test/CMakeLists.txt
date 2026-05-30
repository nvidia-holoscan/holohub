# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Standalone golden-reference unit test for the deproject kernel. It depends only on
# the CUDA runtime (no Holoscan SDK) and returns a non-zero exit code on failure.
add_executable(point_cloud_from_depth_test
  test_deproject.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../deproject.cu
)

set_target_properties(point_cloud_from_depth_test
  PROPERTIES
    CUDA_ARCHITECTURES "native"
)

add_test(NAME point_cloud_from_depth_test COMMAND point_cloud_from_depth_test)
