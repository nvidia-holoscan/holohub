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

#ifndef PYHOLOHUB_OPERATORS_PROHAWK_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_PROHAWK_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace ProhawkOp {

PYDOC(ProhawkOp, R"doc(
Operator class to use ProHawk filters.
)doc")

// PyProhawkOp Constructor
PYDOC(ProhawkOp_python, R"doc(
Operator class to use ProHawk filters.
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

}  // namespace ProhawkOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_PROHAWK_PYDOC_HPP
