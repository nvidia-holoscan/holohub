# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy configuration file into the build tree
add_custom_target(identity_model_onnx ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/identity_model.onnx" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "identity_model.onnx"
  BYPRODUCTS "identity_model.onnx"
)

# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/identity_model.onnx"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)
