/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:  // www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <holoscan/holoscan.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

#include "streaming_server_resource.hpp"
#include <holoscan/core/domain/tensor.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

namespace holoscan::ops {
/**
 * @brief Operator that handles downstream (sending) video streaming to clients
 *
 * This operator receives holoscan::Tensor objects, processes them, and sends
 * the processed frames to connected streaming clients using StreamingServerResource.
 * It handles the server-side encoding and transmission of outgoing video streams.
 */
class StreamingServerDownstreamOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StreamingServerDownstreamOp)

  StreamingServerDownstreamOp() = default;
  ~StreamingServerDownstreamOp();

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void start() override;
  void stop() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override;

 private:
  // Configuration parameters (can override StreamingServerResource defaults)
  Parameter<uint32_t> width_;  // /< Frame width
  Parameter<uint32_t> height_;  // /< Frame height
  Parameter<uint32_t> fps_;  // /< Frames per second
  Parameter<bool> enable_processing_;  // /< Enable frame processing (mirroring, etc.)
  Parameter<std::string> processing_type_;  // /< Type of processing to apply
  Parameter<std::shared_ptr<Allocator>> allocator_;  // /< Memory allocator

  // StreamingServerResource reference
  Parameter<std::shared_ptr<StreamingServerResource>> streaming_server_resource_;

  // Internal state
  std::atomic<bool> is_shutting_down_{false};
  std::atomic<bool> downstream_connected_{false};

  // Performance tracking
  std::atomic<uint64_t> frames_processed_{0};
  std::atomic<uint64_t> frames_sent_{0};
  std::atomic<std::chrono::steady_clock::time_point::rep> start_time_ticks_{0};

  // Frame processing methods
  holoscan::Tensor process_frame(const holoscan::Tensor& input_tensor);
  Frame convert_tensor_to_frame(const holoscan::Tensor& tensor);

  // Mirror horizontally (example processing)
  holoscan::Tensor mirror_horizontally(const holoscan::Tensor& input_tensor);

  // Event callback handler (for connection status)
  void on_streaming_server_event(const StreamingServerResource::Event& event);
};
}  // namespace holoscan::ops
