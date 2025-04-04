/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef PYHOLOHUB_OPERATORS_OPERATOR_UTIL_HPP
#define PYHOLOHUB_OPERATORS_OPERATOR_UTIL_HPP

#include <pybind11/pybind11.h>

#include <memory>

#include "holoscan/core/condition.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resource.hpp"

namespace py = pybind11;

namespace holoscan {

inline void add_positional_condition_and_resource_args(Operator* op, const py::args& args) {
  for (auto it = args.begin(); it != args.end(); ++it) {
    if (py::isinstance<Condition>(*it)) {
      op->add_arg(it->cast<std::shared_ptr<Condition>>());
    } else if (py::isinstance<Resource>(*it)) {
      op->add_arg(it->cast<std::shared_ptr<Resource>>());
    } else {
      HOLOSCAN_LOG_WARN(
          "Unhandled positional argument detected (only Condition and Resource objects can be "
          "parsed positionally)");
    }
  }
}

}  // namespace holoscan

#endif /* PYHOLOHUB_OPERATORS_OPERATOR_UTIL_HPP */
