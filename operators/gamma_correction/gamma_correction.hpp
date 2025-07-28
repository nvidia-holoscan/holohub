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

#ifndef GAMMA_CORRECTION_OP_HPP
#define GAMMA_CORRECTION_OP_HPP

#include <algorithm>

#include "slang_shader_op.hpp"

namespace holoscan::ops {

/**
 * @brief Gamma correction operator.
 */
class GammaCorrectionOp : public SlangShaderOp {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  explicit GammaCorrectionOp(ArgT&& arg, ArgsT&&... more_args)
      : ::holoscan::ops::SlangShaderOp(std::forward<ArgT>(arg), std::forward<ArgsT>(more_args)...) {
    // Get the data type from the arguments and remove it from the argument list
    auto data_type_arg = std::find_if(
        args().begin(), args().end(), [](const auto& arg) { return arg.name() == "data_type"; });
    if (data_type_arg == args().end()) {
      throw std::runtime_error("data_type argument is required");
    }
    std::string data_type_str = std::any_cast<std::string>(data_type_arg->value());
    args().erase(data_type_arg);

    // Create the preprocessor macros
    std::map<std::string, std::string> preprocessor_macros;
    preprocessor_macros["DATA_TYPE"] = data_type_str;

    uint32_t element_count;
    if (data_type_str.back() == '4') {
      element_count = 4;
    } else if (data_type_str.back() == '3') {
      element_count = 3;
    } else if (data_type_str.back() == '2') {
      element_count = 2;
    } else if (data_type_str.back() == '1') {
      element_count = 1;
    }
    preprocessor_macros["ELEMENT_COUNT"] = std::to_string(element_count);

    bool normalize = true;
    if ((data_type_str.find("float") == 0) || (data_type_str.find("double") == 0)) {
      normalize = false;
    }
    preprocessor_macros["NORMALIZE"] = normalize ? "1" : "0";

    add_arg(Arg("preprocessor_macros", preprocessor_macros));
    add_arg(Arg("shader_source", std::string(g_shader_source)));
  }

  GammaCorrectionOp() : SlangShaderOp(Arg("shader_source", std::string(g_shader_source))) {}

 private:
  static constexpr std::string_view g_shader_source = R"(
import holoscan;

[holoscan::input("input")]
[holoscan::output("output")]
RWStructuredBuffer<DATA_TYPE> buffer;

[holoscan::parameter("gamma")]
float gamma;

[holoscan::size_of("input")]
uint3 size;

[holoscan::strides_of("input")]
uint64_t3 strides;

#ifdef NORMALIZE
static const float range = (1 << (sizeof(DATA_TYPE) * 8)) - 1;
DATA_TYPE apply(DATA_TYPE value) {
    float fvalue = (float)value / range;
    fvalue = pow(fvalue, 1.f / gamma);
    fvalue = fvalue * range + 0.5f;
    return DATA_TYPE(fvalue);
}
#else
float apply(float value) {
    return pow(value, 1.f / gamma);
}
#endif

[holoscan::invocations::size_of("input")]
[shader("compute")]
void gamma_correction(uint3 gid : SV_DispatchThreadID)
{
    if ((gid.x >= size.x) || (gid.y >= size.y)) {
        return;
    }

    const uint64_t offset = gid.x + gid.y * strides.x;

    // apply gamma correction to each component except alpha
    DATA_TYPE value = buffer[offset];
    value.x = apply(value.x);
#if ELEMENT_COUNT > 1
    value.y = apply(value.y);
#endif
#if ELEMENT_COUNT > 2
    value.z = apply(value.z);
#endif
    buffer[offset] = value;
})";
};

}  // namespace holoscan::ops

#endif /* GAMMA_CORRECTION_OP_HPP */
