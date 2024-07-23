/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef VTK_OPERATOR__HPP
#define VTK_OPERATOR__HPP

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/operator.hpp"

namespace holoscan::ops {

/**
 * @brief This operator is used to render the video stream and annotations on the screen.
 */
class VtkRendererOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(VtkRendererOp)

  void setup(OperatorSpec& spec) override;
  void start() override;
  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override;

 private:
  Parameter<std::string> window_name;
  Parameter<std::vector<std::string>> labels;
  Parameter<uint32_t> height;
  Parameter<uint32_t> width;

  struct Internals;
  std::shared_ptr<Internals> internals;
};
}  // namespace holoscan::ops

#endif /* HOLOSCAN_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_TOOL_TRACKING_POSTPROCESSOR_HPP */
