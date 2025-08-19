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

#include "include/gamma_correction/gamma_correction.hpp"

#include <algorithm>

#include "gamma_correction_slang.hpp"

namespace holoscan::ops {

void GammaCorrectionOp::setup(OperatorSpec& spec) {
  // Get the data type and component count from the arguments and
  // remove it from the argument list
  auto data_type_arg = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return arg.name() == "data_type"; });
  if (data_type_arg == args().end()) {
    throw std::runtime_error("data_type argument is required");
  }
  std::string data_type_str = std::any_cast<std::string>(data_type_arg->value());
  args().erase(data_type_arg);

  auto component_count_arg = std::find_if(args().begin(), args().end(), [](const auto& arg) {
    return arg.name() == "component_count";
  });
  uint32_t component_count = 1;
  if (component_count_arg != args().end()) {
    component_count = std::any_cast<int32_t>(component_count_arg->value());
    args().erase(component_count_arg);
  }

  // Create the preprocessor macros
  std::map<std::string, std::string> preprocessor_macros;
  preprocessor_macros["DATA_TYPE"] = data_type_str;
  preprocessor_macros["COMPONENT_COUNT"] = std::to_string(component_count);

  bool normalize = true;
  if ((data_type_str.find("float") == 0) || (data_type_str.find("double") == 0)) {
    normalize = false;
  }
  preprocessor_macros["NORMALIZE"] = normalize ? "1" : "0";

  add_arg(Arg("preprocessor_macros", preprocessor_macros));
  add_arg(Arg("shader_source", std::string(gamma_correction_slang)));

  SlangShaderOp::setup(spec);
}

}  // namespace holoscan::ops
