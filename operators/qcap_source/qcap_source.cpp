/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights reserved.
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

#include "qcap_source.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

namespace holoscan::ops {

void QCAPSourceOp::setup(OperatorSpec& spec) {
  auto& video_buffer_output = spec.output<gxf::Entity>("video_buffer_output");

  constexpr char kDefaultDevice[] = "SC0710 PCI";
  constexpr uint32_t kDefaultChannel = 0;
  constexpr uint32_t kDefaultWidth = 3840;
  constexpr uint32_t kDefaultHeight = 2160;
  constexpr uint32_t kDefaultFramerate = 60;
  constexpr bool kDefaultRDMA = false;
  constexpr char kDefaultPixelFormat[] = "bgr24";
  constexpr char kDefaultInputType[] = "auto";
  constexpr uint32_t kDefaultMSTMode = 0;
  constexpr uint32_t kDefaultSDI12GMode = 0;

  spec.param(video_buffer_output_,
             "video_buffer_output",
             "VideoBufferOutput",
             "Output for the video buffer.",
             &video_buffer_output);
  spec.param(
      device_specifier_, "device", "Device", "Device specifier.", std::string(kDefaultDevice));
  spec.param(channel_, "channel", "Channel", "Channel to use.", kDefaultChannel);
  spec.param(width_, "width", "Width", "Width of the stream.", kDefaultWidth);
  spec.param(height_, "height", "Height", "Height of the stream.", kDefaultHeight);
  spec.param(framerate_, "framerate", "Framerate", "Framerate of the stream.", kDefaultFramerate);
  spec.param(use_rdma_, "rdma", "RDMA", "Enable RDMA.", kDefaultRDMA);
  spec.param(
      pixel_format_, "pixel_format", "PixelFormat", "Pixel Format.", std::string(kDefaultPixelFormat));
  spec.param(
      input_type_, "input_type", "InputType", "Input Type.", std::string(kDefaultInputType));
  spec.param(
      mst_mode_, "mst_mode", "MSTMode", "MST Mode.", kDefaultMSTMode);
  spec.param(
      mst_mode_, "sdi12g_mode", "SDI12GMode", "SDI 12G Mode.", kDefaultSDI12GMode);
}

void QCAPSourceOp::initialize() {
  holoscan::ops::GXFOperator::initialize();
}

}  // namespace holoscan::ops
