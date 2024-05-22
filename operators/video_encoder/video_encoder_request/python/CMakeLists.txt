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

include(pybind11_add_holohub_module)

pybind11_add_holohub_module(
    CPP_CMAKE_TARGET video_encoder_request
    CLASS_NAME "VideoEncoderRequestOp"
    SOURCES video_encoder_request.cpp
)

set(module_target video_encoder_request_python)
set(module_init_file ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py)
    set(module_init_target ${module_target}_init)
    add_custom_target(${module_init_target}
        COMMAND ${CMAKE_COMMAND} -E copy "${module_init_file}" "${HOLOHUB_PYTHON_MODULE_OUT_DIR}/video_encoder_request/"
        DEPENDS "${module_init_file}"
        BYPRODUCTS "${module_binary_dir}/__init__.py"
    )
add_dependencies(${module_target} ${module_init_target})

