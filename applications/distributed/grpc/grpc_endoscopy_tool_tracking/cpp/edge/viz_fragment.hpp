/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_VIZ_FRAGMENT_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_VIZ_FRAGMENT_HPP

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

namespace holohub::grpc_h264_endoscopy_tool_tracking {

using namespace holoscan;
using namespace holoscan::ops;

/**
 * @class VizFragment
 * @brief A fragment class for visualizing endoscopy tool tracking using Holoviz.
 *
 * This class inherits from holoscan::Fragment and is used to set up a visualization
 * operator with specified width and height.
 */
class VizFragment : public holoscan::Fragment {
 public:
  VizFragment(const uint32_t width, const uint32_t height)
      : width_(width), height_(height) {}

  void compose() override {
    int64_t source_block_size = width_ * height_ * 3 * 4;
    int64_t source_num_blocks = 2;

    auto visualizer_op = make_operator<ops::HolovizOp>(
        "visualizer_op",
        from_config("holoviz"),
        Arg("width") = width_,
        Arg("height") = height_,
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));
    add_operator(visualizer_op);
  }

 private:
  uint32_t width_;
  uint32_t height_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking
#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_VIZ_FRAGMENT_HPP */
