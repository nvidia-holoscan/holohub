/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYTHON_TAPNEXT_INFERENCE_PYDOC_HPP
#define PYTHON_TAPNEXT_INFERENCE_PYDOC_HPP

#include <string>

namespace holoscan::doc {

namespace TapNextInferenceOp {

// Constructor
static const char* doc_TapNextInferenceOp = R"doc(
    Operator class to perform TapNext inference.
)doc";

static const char* doc_TapNextInferenceOp_python = R"doc(
    Operator class to perform TapNext inference.
)doc";

static const char* doc_gxf_typename = R"doc(
    The GXF type name of the operator.
)doc";

static const char* doc_initialize = R"doc(
    Initialize the operator.
)doc";

static const char* doc_setup = R"doc(
    Setup the operator.
)doc";

}  // namespace TapNextInferenceOp

}  // namespace holoscan::doc

#endif  // PYTHON_TAPNEXT_INFERENCE_PYDOC_HPP

