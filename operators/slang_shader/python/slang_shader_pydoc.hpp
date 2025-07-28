/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYHOLOHUB_OPERATORS_SLANG_SHADER_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_SLANG_SHADER_PYDOC_HPP

#include "macros.hpp"

namespace holoscan::doc {

namespace SlangShaderOp {

PYDOC(SlangShaderOp, R"doc(
The `slang_shader` operator runs a Slang shader.)doc")

// PySlangShaderOp Constructor
PYDOC(SlangShaderOp_python, R"doc(
Operator class to run a Slang shader.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
name : str, optional
    The name of the operator.
shader_source : str, optional
    Slang shader source code
shader_source_file : str, optional
    Slang shader source file
preprocessor_macros: dict, optional
    Preprocessor macros to be used in the shader
allocator: ``holoscan.resources.Allocator``
    Allocator used to allocate the data
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

}  // namespace SlangShaderOp


}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_VOLUME_LOADER_PYDOC_HPP
