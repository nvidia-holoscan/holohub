# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Files to include in our module
list(APPEND basic_network_python_files
    __init__.py
    basic_network_common.py
    basic_network_rx.py
    basic_network_tx.py
)

# Define paths to copy in the holohub python module
list(TRANSFORM basic_network_python_files
    PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/
    OUTPUT_VARIABLE basic_network_python_src_files
)
set(basic_network_python_out_dir ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/basic_network)
file(MAKE_DIRECTORY ${basic_network_python_out_dir})
list(TRANSFORM basic_network_python_files
    PREPEND ${basic_network_python_out_dir}/
    OUTPUT_VARIABLE basic_network_python_out_files
)

# Target to copy
add_custom_target(basic_network_python
    COMMAND ${CMAKE_COMMAND} -E copy ${basic_network_python_src_files} ${basic_network_python_out_dir}
    DEPENDS ${basic_network_python_src_files}
    BYPRODUCTS ${basic_network_python_out_files}
)
