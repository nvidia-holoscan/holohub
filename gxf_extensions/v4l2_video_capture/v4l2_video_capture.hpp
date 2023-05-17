/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_V4L2_VIDEO_CAPTURE_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_V4L2_VIDEO_CAPTURE_HPP_

#include <linux/videodev2.h>
#include <string>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace holoscan {

/// @brief Input codelet for common V4L2 camera sources.
///
/// Provides a codelet for a realtime V4L2 source supporting various media inputs on Linux.
/// The output is a VideoBuffer object.
class V4L2VideoCapture : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> signal_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
  gxf::Parameter<std::string> device_;
  gxf::Parameter<uint32_t> width_;
  gxf::Parameter<uint32_t> height_;
  gxf::Parameter<uint32_t> num_buffers_;
  gxf::Parameter<std::string> pixel_format_;

  gxf_result_t v4l2_initialize();
  gxf_result_t v4l2_requestbuffers();
  gxf_result_t v4l2_set_mode();
  gxf_result_t v4l2_start();
  gxf_result_t v4l2_read_buffer(v4l2_buffer& buf);

  void YUYVToRGBA(const void* yuyv, void* rgba, size_t width, size_t height);

  struct Buffer {
    void* ptr;
    size_t length;
  };
  Buffer* buffers_;
  int fd_ = -1;
};

}  // namespace holoscan
}  // namespace nvidia
#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_V4L2_VIDEO_CAPTURE_HPP_
