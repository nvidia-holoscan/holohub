/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "v4l2_video_capture.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

void V4L2VideoCaptureOp::setup(OperatorSpec& spec) {
  auto& signal = spec.output<gxf::Entity>("signal");

  spec.param(signal_, "signal", "Output", "Output channel", &signal);

  static constexpr char kDefaultDevice[] = "/dev/video0";
  static constexpr char kDefaultPixelFormat[] = "RGBA32";
  static constexpr uint32_t kDefaultWidth = 1920;
  static constexpr uint32_t kDefaultHeight = 1080;
  static constexpr uint32_t kDefaultNumBuffers = 2;

  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");

  spec.param(
      device_, "device", "VideoDevice", "Path to the V4L2 device", std::string(kDefaultDevice));
  spec.param(width_, "width", "Width", "Width of the V4L2 image", kDefaultWidth);
  spec.param(height_, "height", "Height", "Height of the V4L2 image", kDefaultHeight);
  spec.param(num_buffers_,
             "numBuffers",
             "NumBuffers",
             "Number of V4L2 buffers to use",
             kDefaultNumBuffers);
  spec.param(pixel_format_,
             "pixel_format",
             "Pixel Format",
             "Pixel format of capture stream (RGBA32 or YUYV)",
             std::string(kDefaultPixelFormat));
}

void V4L2VideoCaptureOp::initialize() {
  holoscan::ops::GXFOperator::initialize();
}

}  // namespace holoscan::ops
