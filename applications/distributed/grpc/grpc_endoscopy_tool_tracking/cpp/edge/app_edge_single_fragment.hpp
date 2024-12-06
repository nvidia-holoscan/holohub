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

#ifndef GRPC_GRPC_ENDOSCOPY_TOOL_TRACKING_CPP_EDGE_APP_EDGE_SINGLE_FRAGMENT_HPP
#define GRPC_GRPC_ENDOSCOPY_TOOL_TRACKING_CPP_EDGE_APP_EDGE_SINGLE_FRAGMENT_HPP

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include "video_input_fragment.hpp"
#include "viz_fragment.hpp"

namespace holohub::grpc_h264_endoscopy_tool_tracking {

using namespace holoscan;

/**
 * @class AppEdgeSingleFragment
 * @brief A two-fragment application for the H.264 endoscopy tool tracking application.
 *
 * This class inherits from the holoscan::Application and is a client application offloads the
 * inference and process to a remote gRPC server. It is composed with two fragments, a video input
 * fragment and a visualization fragment using Holoviz. This enables running the edge application
 * on two systems, separating the input from the visualization. For example, a video surveillance
 * camera capturing and streaming input to another system displaying the footage.
 */
class AppEdgeSingleFragment : public holoscan::Application {
 public:
  explicit AppEdgeSingleFragment(const std::vector<std::string>& argv = {}) : Application(argv) {}
  ~AppEdgeSingleFragment() {
    entity_client_service_->stop_entity_stream();
  }

  void set_datapath(const std::string& path) { datapath_ = path; }

  void compose() {
    uint32_t width = 854;
    uint32_t height = 480;
    int64_t source_block_size = width * height * 3 * 4;
    int64_t source_num_blocks = 2;

    condition_ = make_condition<AsynchronousCondition>("response_available_condition");
    request_queue_ =
        make_resource<ConditionVariableQueue<std::shared_ptr<EntityRequest>>>("request_queue");
    response_queue_ =
        make_resource<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>>(
            "response_queue", condition_);

    auto replayer = make_operator<ops::VideoStreamReplayerOp>(
        "replayer",
        from_config("replayer"),
        Arg("directory", datapath_),
        Arg("allocator", make_resource<RMMAllocator>("video_replayer_allocator")));

    // The GrpcClientRequestOp is responsible for sending data to the gRPC server.
    auto outgoing_requests = make_operator<GrpcClientRequestOp>(
        "outgoing_requests",
        Arg("request_queue") = request_queue_,
        Arg("allocator") = make_resource<RMMAllocator>(
            "pool", Arg("device_memory_max_size") = std::string("256MB")));

    auto visualizer_op = make_operator<ops::HolovizOp>(
        "visualizer_op",
        from_config("holoviz"),
        Arg("width") = width,
        Arg("height") = height,
        Arg("allocator") =
            make_resource<BlockMemoryPool>("allocator", 1, source_block_size, source_num_blocks),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));

    // The GrpcClientResponseOp is responsible for handling incoming responses from the gRPC
    // server.
    auto incoming_responses =
        make_operator<GrpcClientResponseOp>("incoming_responses",
                                            Arg("condition") = condition_,
                                            Arg("response_queue") = response_queue_);

    // Send the frames to the gRPC server for processing.
    add_flow(replayer, outgoing_requests, {{"output", "input"}});

    // Here we add the operator to process the response queue with data received from the gRPC
    // server. The operator will convert the data to a GXF Entity and send it to the Holoviz.
    add_operator(incoming_responses);

    add_flow(replayer, visualizer_op, {{"output", "receivers"}});
    add_flow(incoming_responses, visualizer_op, {{"output", "receivers"}});

    entity_client_service_ = std::make_shared<EntityClientService>(
        from_config("grpc_client.server_address").as<std::string>(),
        from_config("grpc_client.rpc_timeout").as<uint32_t>(),
        from_config("grpc_client.interrupt").as<bool>(),
        request_queue_,
        response_queue_,
        outgoing_requests);
    entity_client_service_->start_entity_stream();
  }

 private:
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue_;
  std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>> response_queue_;
  std::shared_ptr<AsynchronousCondition> condition_;
  std::shared_ptr<EntityClientService> entity_client_service_;
  std::string datapath_ = "data/endoscopy";
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_GRPC_ENDOSCOPY_TOOL_TRACKING_CPP_EDGE_APP_EDGE_SINGLE_FRAGMENT_HPP */
