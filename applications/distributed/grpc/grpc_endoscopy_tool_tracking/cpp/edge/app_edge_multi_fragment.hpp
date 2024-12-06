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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_EDGE_APP_EDGE_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_EDGE_APP_EDGE_HPP

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "video_input_fragment.hpp"
#include "viz_fragment.hpp"

namespace holohub::grpc_h264_endoscopy_tool_tracking {

using namespace holoscan;

/**
 * @class AppEdgeMultiFragment
 * @brief A two-fragment application for the H.264 endoscopy tool tracking application.
 *
 * This class inherits from the holoscan::Application and is a client application offloads the
 * inference and process to a remote gRPC server. It is composed with two fragments, a video input
 * fragment and a visualization fragment using Holoviz. This enables running the edge application on
 * two systems, separating the input from the visualization. For example, a video surveillance
 * camera capturing and streaming input to another system displaying the footage.
 */
class AppEdgeMultiFragment : public holoscan::Application {
 public:
  explicit AppEdgeMultiFragment(const std::vector<std::string>& argv = {}) : Application(argv) {}
  void set_datapath(const std::string& path) { datapath_ = path; }

  void compose() {
    uint32_t width = 854;
    uint32_t height = 480;

    auto video_in = make_fragment<VideoInputFragment>("video_in", datapath_, width, height);
    auto viz = make_fragment<VizFragment>("viz", width, height);

    // Connect the video input fragment to the visualization fragment.
    // - Connect the decoded video frames to the visualizer.
    // - Connect the inference & post-process results to the visualizer.
    add_flow(video_in,
             viz,
             {{"replayer.output", "visualizer_op.receivers"},
              {"incoming_responses.output", "visualizer_op.receivers"}});
  }

 private:
  std::string datapath_ = "data/endoscopy";
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking
#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_EDGE_APP_EDGE_HPP */
