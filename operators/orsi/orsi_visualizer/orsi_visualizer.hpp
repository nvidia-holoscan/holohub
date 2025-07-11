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

#pragma once

// clang-format off
#define GLFW_INCLUDE_NONE 1
#include <GL/glew.h>
#include <GLFW/glfw3.h>  // NOLINT(build/include_order)
// clang-format on

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

#include "vis_intf.hpp"

namespace holoscan::ops::orsi {

/**
 * @brief Operator class to visualize the tool tracking results.
 *
 */

class OrsiVisualizationOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OrsiVisualizationOp)

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

  // GLFW window related callbacks
  void onFramebufferSizeCallback(int width, int height);
  void onWindowFocusCallback(int focused);
  void onChar(unsigned int codepoint);
  void onEnter(int entered);
  void onMouseMove(double x, double y);
  void onMouseButtonCallback(int button, int action, int mods);
  void onScrollCallback(double x, double y);
  void onKeyCallback(int key, int scancode, int action, int mods);

 protected:
  // pointer to vis implementation
  std::unique_ptr<holoscan::orsi::vis::VisIntf> pimpl_;

 private:
  // GLFW members and callback
  GLFWwindow* window_ = nullptr;
  // GL viewport
  int vp_width_ = 0;
  int vp_height_ = 0;

  Parameter<std::vector<holoscan::IOSpec*>> receivers_;
  Parameter<std::shared_ptr<BooleanCondition>> window_close_scheduling_term_;

  CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops::orsi
