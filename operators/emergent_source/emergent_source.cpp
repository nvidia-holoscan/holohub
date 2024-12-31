/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "emergent_source.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

namespace holoscan::ops {

void EmergentSourceOp::setup(OperatorSpec& spec) {
  auto& signal = spec.output<gxf::Entity>("signal");

  constexpr uint32_t kDefaultWidth = 4200;
  constexpr uint32_t kDefaultHeight = 2160;
  constexpr uint32_t kDefaultFramerate = 240;
  constexpr bool kDefaultRDMA = false;
  constexpr uint32_t kDefaultExposure = 3072;
  constexpr uint32_t kDefaultGain = 4095;

  spec.param(signal_, "signal", "Output", "Output channel", &signal);
  spec.param(width_, "width", "Width", "Width of the stream.", kDefaultWidth);
  spec.param(height_, "height", "Height", "Height of the stream.", kDefaultHeight);
  spec.param(framerate_, "framerate", "Framerate", "Framerate of the stream.", kDefaultFramerate);
  spec.param(use_rdma_, "rdma", "RDMA", "Enable RDMA.", kDefaultRDMA);
  spec.param(exposure_, "exposure", "Exposure", "Exposure time", kDefaultExposure);
  spec.param(gain_, "gain", "Gain", "Analog Gain", kDefaultWidth);
}

void EmergentSourceOp::initialize() {
  holoscan::ops::GXFOperator::initialize();
}

}  // namespace holoscan::ops
