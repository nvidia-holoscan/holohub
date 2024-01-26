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

add_custom_target(python_benchmark_model ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_model.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "benchmark_model.py"
  BYPRODUCTS "benchmark_model.py"
)

add_custom_target(python_benchmark_model_yaml ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_model.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "benchmark_model.yaml"
  BYPRODUCTS "benchmark_model.yaml"
)

add_custom_target(python_benchmark_image_model ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_image_model.py" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/onnx_helper.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "benchmark_image_model.py" "onnx_helper.py"
  BYPRODUCTS "benchmark_image_model.py" "onnx_helper.py"
)

add_dependencies(python_benchmark_model python_benchmark_model_yaml)
